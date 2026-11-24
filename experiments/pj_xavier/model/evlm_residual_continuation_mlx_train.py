#!/usr/bin/env python3
"""Bounded EEG completion adaptation; preserves the audited frozen VLM base.

This is a new protocol. Historical pilots and the installed MLX package are not
modified. A warm start reloads adapter weights with a fresh Adam optimizer; it is
not exact optimizer resume. Numerical EEG reconstruction is self-supervision,
not an inferred action/intent label or evidence of visual/computer-use ability.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import fcntl
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

import direct_qwen_mlx_progressive as audited
import evlm_residual_protocol as residual
import evlm_residual_continuation as continuation

SCHEMA = "evlm-mlx-train-v1"
DATA_SCHEMA = "evlm-data-manifest-v1"
CORE_SHA256 = "035f465baf9de29c28a9a6924b678b335e0c312eb9c7a6448df119fef548faf2"
TASKS = {"action_classification", "quantized_eeg_reconstruction"}
require = audited.require
sha256 = audited.sha256
digest = audited.canonical_digest
atomic_json = audited.atomic_json
utc_now = audited.utc_now


def checked_file(entry, *, allowed_names=None):
    require(isinstance(entry, dict), "missing_file_registration")
    path = Path(entry["path"]).expanduser().resolve()
    require(path.is_file() and re.fullmatch(r"[a-f0-9]{64}", entry.get("sha256", "")),
            "unregistered_or_missing_file")
    if allowed_names is not None:
        require(path.name in allowed_names, "unregistered_split_filename")
    require(sha256(path) == entry["sha256"], "registered_file_hash_mismatch")
    return path


def base_identity(cfg):
    return digest({"model_path": str(Path(cfg["model_path"]).expanduser().resolve()),
                   "config_sha256": cfg["expected_model_config_sha256"],
                   "receipt_sha256": cfg["base_receipt"]["sha256"]})


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    require(cfg.get("schema") == SCHEMA, "unsupported_training_schema")
    require(sha256(Path(audited.__file__)) == CORE_SHA256, "audited_helper_source_changed")
    require(cfg.get("mode") in {"engineering_probe", "train_stage"}, "unsupported_mode")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", cfg.get("run_id", "")), "unsafe_run_id")
    require(cfg.get("num_layers") == 64 and cfg.get("rank") == 8,
            "this_protocol_requires_explicit_all64_rank8")
    require(cfg.get("max_seq_length") in (801, 1024), "registered_context_must_be_801_or_1024")
    require(cfg.get("memory_limit_gb") == 54, "registered_memory_policy_is_54GB")
    require(isinstance(cfg.get("seed"), int), "missing_seed")
    require(0 < cfg.get("learning_rate", 0) <= 2e-5, "invalid_learning_rate")
    require(isinstance(cfg.get("max_updates"), int) and 1 <= cfg["max_updates"] <= 400,
            "stage_budget_must_be_1_to_400_updates")
    require(60 <= cfg.get("max_wall_seconds", 0) <= 172800, "invalid_wall_budget")
    require(isinstance(cfg.get("order_offset", 0), int) and cfg.get("order_offset", 0) >= 0,
            "invalid_order_offset")
    require(isinstance(cfg.get("allow_repeat_epochs", False), bool), "invalid_repeat_permission")
    require(isinstance(cfg.get("eval_max_records", 36), int) and 1 <= cfg.get("eval_max_records", 36) <= 36,
            "invalid_development_evaluation_budget")
    require(cfg.get("max_sentry_drop", 1) in (0, 1), "invalid_fixed_reference_sentry_gate")
    points = cfg.get("checkpoint_steps", [])
    require(points == sorted(set(points)) and points and points[-1] == cfg["max_updates"]
            and all(isinstance(v, int) and 1 <= v <= cfg["max_updates"] for v in points),
            "invalid_checkpoint_steps")
    if cfg["mode"] == "engineering_probe":
        require(cfg["max_updates"] == 2 and points == [2], "probe_is_exactly_two_updates")
        require(cfg["max_wall_seconds"] <= 3600, "probe_wall_budget_exceeds_one_hour")
        require(cfg.get("warm_start_adapter") is None, "engineering_probe_starts_from_base")
    for key in ("model_path", "run_dir", "lock_path"):
        require(Path(cfg[key]).expanduser().is_absolute(), "registered_paths_must_be_absolute")
    model_path = Path(cfg["model_path"]).expanduser().resolve()
    require(Path(cfg["lock_path"]).expanduser().resolve() == model_path.parents[1] / "train_gpu.lock",
            "gpu_lock_not_in_registered_node_root")
    require(sha256(model_path / "config.json") == cfg["expected_model_config_sha256"],
            "base_model_configuration_changed")
    checked_file(cfg["base_receipt"])
    checked_file(cfg["data_manifest"])
    if cfg.get("warm_start_adapter") is not None:
        warm_start_metadata(cfg)
    require(cfg.get("residual_protocol") == "rounded_linear_residual_v1", "wrong_residual_protocol")
    if cfg.get("warm_start_adapter") is not None:
        continuation.admit(cfg)
    else:
        require(not any(k.startswith("residual_parent_") for k in cfg), "unexpected_parent_receipts")
    checked_file(cfg["residual_candidate"])
    return cfg


def warm_start_metadata(cfg):
    registered = cfg["warm_start_adapter"]
    directory = Path(registered["checkpoint_dir"]).expanduser().resolve()
    require(directory.is_dir(), "warm_start_checkpoint_missing")
    require(sha256(directory / "adapters.safetensors") == registered["adapter_sha256"],
            "warm_start_adapter_hash_mismatch")
    require(sha256(directory / "checkpoint.json") == registered["checkpoint_sha256"],
            "warm_start_metadata_hash_mismatch")
    meta = json.loads((directory / "checkpoint.json").read_text())
    require(meta.get("schema") == SCHEMA and meta.get("base_identity") == base_identity(cfg),
            "warm_start_base_or_protocol_mismatch")
    require(meta.get("num_layers") == cfg["num_layers"] and meta.get("rank") == cfg["rank"],
            "warm_start_adapter_scope_mismatch")
    require(meta.get("adapter_sha256") == registered["adapter_sha256"], "warm_start_lineage_mismatch")
    evaluation = json.loads((directory / "evaluation.json").read_text())
    require(evaluation.get("quality_gates_passed") is True and evaluation.get("status") == "completed",
            "cannot_warm_start_failed_or_unreviewed_checkpoint")
    require(meta.get("evaluation_sha256") == sha256(directory / "evaluation.json"),
            "warm_start_evaluation_changed")
    require(meta.get("reference_sentry") == {"n": 6, "passed": 6},
            "warm_start_initial_base_sentry_reference_missing")
    return meta


def reconstruction_prompt(observed, masked_indices, quantization_schema, geometry=None):
    geometry_line = ("Geometry: " + json.dumps(geometry, sort_keys=True, separators=(",", ":")) + "\n"
                     if geometry is not None else "")
    return ("Reconstruct the masked quantized EEG values.\n"
            f"Quantization schema: {quantization_schema}\n"
            + geometry_line +
            "Observed values (null=masked): " + json.dumps(observed, separators=(",", ":")) + "\n"
            "Masked indices: " + json.dumps(masked_indices, separators=(",", ":")) + "\n"
            "Return only a JSON integer array in masked-index order.")


def validate_reconstruction(row, manifest):
    rec = row.get("reconstruction", {})
    observed, indices = rec.get("observed"), rec.get("masked_indices")
    require(isinstance(observed, list) and 2 <= len(observed) <= 512, "invalid_observed_integer_vector")
    require(all(v is None or type(v) is int for v in observed), "noninteger_observation")
    require(isinstance(indices, list) and 1 <= len(indices) <= 16
            and all(type(i) is int and 0 <= i < len(observed) for i in indices)
            and indices == sorted(set(indices)), "invalid_masked_indices")
    require([i for i, v in enumerate(observed) if v is None] == indices, "mask_observation_mismatch")
    require(any(v is not None for v in observed), "no_visible_observation")
    geometry = rec.get("geometry")
    require(isinstance(geometry, dict), "missing_eeg_geometry")
    shape = geometry.get("shape", [])
    require(isinstance(shape, list) and len(shape) == 2 and all(type(x) is int and x > 0 for x in shape)
            and shape[0] * shape[1] == len(observed), "geometry_observation_shape_mismatch")
    channel_cap = manifest.get("max_channels_per_training_block", 8)
    require(type(channel_cap) is int and 1 <= channel_cap <= 8 and shape[0] <= channel_cap,
            "channel_count_exceeds_registered_training_block")
    channels = geometry.get("channel_names")
    require(isinstance(channels, list) and len(channels) == shape[0]
            and all(isinstance(x, str) and x for x in channels), "missing_channel_names")
    sample_rate = geometry.get("sample_rate_hz")
    require(type(sample_rate) in (int, float) and math.isfinite(sample_rate) and sample_rate > 0,
            "invalid_sample_rate")
    require(geometry.get("quantum_uv") == 1 and type(geometry.get("crop_start_sample")) is int
            and geometry["crop_start_sample"] >= 0, "invalid_numeric_eeg_geometry")
    time_count = shape[1]
    require(indices == list(range(indices[0], indices[-1] + 1))
            and indices[0] // time_count == indices[-1] // time_count
            and indices[0] % time_count > 0 and indices[-1] % time_count < time_count - 1,
            "masked_span_must_have_visible_endpoints_in_one_channel")
    target = json.loads(row["messages"][1]["content"])
    require(isinstance(target, list) and len(target) == len(indices)
            and all(type(v) is int for v in target), "invalid_reconstruction_target")
    low, high = manifest["quantization_bounds"]
    require(type(low) is int and type(high) is int and low < high, "invalid_quantization_bounds")
    require(all(low <= v <= high for v in target + [v for v in observed if v is not None]),
            "integer_outside_quantization_bounds")
    require(row["messages"][0]["content"] == reconstruction_prompt(
        observed, indices, manifest["quantization_schema"], geometry=geometry), "reconstruction_prompt_contract_mismatch")
    return target


def read_manifest(cfg):
    manifest_path = checked_file(cfg["data_manifest"])
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema") == DATA_SCHEMA and manifest.get("task_kind") in TASKS,
            "unsupported_data_manifest")
    checked_file(manifest["qc_receipt"])
    source_ids = manifest.get("source_ids", [])
    require(source_ids and len(set(source_ids)) == len(source_ids), "missing_or_duplicate_admitted_sources")
    if manifest["task_kind"] == "quantized_eeg_reconstruction":
        require(isinstance(manifest.get("quantization_schema"), str)
                and manifest["quantization_schema"].strip(), "missing_quantization_schema")
        require(isinstance(manifest.get("quantization_bounds"), list)
                and len(manifest["quantization_bounds"]) == 2, "missing_quantization_bounds")
        channel_cap = manifest.get("max_channels_per_training_block", 8)
        require(type(channel_cap) is int and 1 <= channel_cap <= 8, "invalid_training_channel_block")
    inputs = manifest.get("inputs", {})
    require(set(inputs) <= {"train", "val"} and "train" in inputs, "only_train_and_val_inputs_allowed")
    require(cfg["mode"] == "engineering_probe" or "val" in inputs, "stage_needs_development_validation")
    for item in inputs.values():
        require(type(item.get("count")) is int and item["count"] > 0, "invalid_registered_record_count")
    if "max_seq_length" in manifest:
        require(manifest["max_seq_length"] <= cfg["max_seq_length"], "manifest_context_exceeds_training_context")
    if cfg["mode"] == "train_stage" and not cfg.get("allow_repeat_epochs", False):
        require(cfg.get("order_offset", 0) + cfg["max_updates"] <= inputs["train"]["count"],
                "stage_would_silently_repeat_training_epoch")
    return manifest


def scan_inputs(cfg, tokenizer=None):
    """Validate every row, retaining only this stage and fixed development rows.

    Identity sets scale with source rows; JSON records and token arrays do not.
    Even rows after the selected stage are checked. No early break or silent skip
    is permitted. A real preflight always supplies the frozen tokenizer.
    """
    manifest = read_manifest(cfg)
    source_ids = manifest["source_ids"]
    inputs = manifest["inputs"]
    protected = set(manifest.get("protected_subject_keys", []))
    rows, processed, ids, subjects, recordings, summaries = {}, {}, {}, {}, {}, {}
    full_count = inputs["train"]["count"]
    require(manifest["source_ids"] == ["zenodo:806023"] and full_count == 1977 and manifest["quantization_bounds"] == [-100,100], "residual_diagnostic_scope")
    candidate_path = checked_file(cfg["residual_candidate"])
    candidates = [json.loads(line) for line in candidate_path.open() if line.strip()]
    require(len(candidates) == full_count, "candidate_count_mismatch")
    source_order = [] if cfg["mode"] == "engineering_probe" else selected_order(cfg, full_count)
    chosen_train = set(source_order)
    validation_count = inputs.get("val", {}).get("count", 0)
    val_indices = list(range(validation_count))
    random.Random(cfg["seed"] + 2).shuffle(val_indices)
    val_indices = sorted(val_indices[:cfg.get("eval_max_records", 36)])
    if cfg["mode"] == "engineering_probe":
        val_indices = []
    selected_val = set(val_indices)
    retained_indices = {}
    for split in ("train", "val"):
        if split not in inputs:
            continue
        allowed = {"train.jsonl"} if split == "train" else {"val.jsonl", "valid.jsonl"}
        path = checked_file(inputs[split], allowed_names=allowed)
        ids[split], subjects[split], recordings[split] = set(), set(), set()
        rows[split], processed[split], retained_indices[split] = [], [], []
        summary = {"count": 0, "max_tokens": 0, "min_tokens": None, "max_completion_tokens": 0}
        stream_hash = hashlib.sha256()
        longest_length = -1
        with path.open("rb") as stream:
            for raw_line in stream:
                stream_hash.update(raw_line)
                if not raw_line.strip():
                    continue
                index = summary["count"]
                require(index < inputs[split]["count"], "registered_record_count_mismatch")
                row = json.loads(raw_line)
                require(row.get("split") == split, "unexpected_split_membership")
                require(row.get("task_kind") == manifest["task_kind"], "mixed_unregistered_tasks")
                require(isinstance(row.get("id"), str) and row["id"], "missing_record_id")
                require(row["id"] not in ids[split], "duplicate_record_id")
                ids[split].add(row["id"])
                require(row.get("source_id") in source_ids, "source_not_admitted")
                require(row.get("canonical_subject_id") is not None, "missing_subject_id")
                subject_key = digest([row["source_id"], str(row["canonical_subject_id"])])
                require(subject_key not in protected, "protected_subject_in_development_data")
                subjects[split].add(subject_key)
                if row.get("recording_id") is not None:
                    recordings[split].add(digest([row["source_id"], str(row["recording_id"])]))
                msgs = row.get("messages", [])
                require(len(msgs) == 2 and [m.get("role") for m in msgs] == ["user", "assistant"]
                        and all(isinstance(m.get("content"), str) and m["content"].strip() for m in msgs),
                        "invalid_completion_messages")
                if manifest["task_kind"] == "action_classification":
                    require(msgs[1]["content"] in audited.LABELS, "unexpected_action_label")
                else:
                    validate_reconstruction(row, manifest)
                    row = residual.residual_row(row)
                    if split == "train":
                        require(row == candidates[index], "residual_candidate_source_mismatch")
                item = render(tokenizer, row, cfg["max_seq_length"]) if tokenizer is not None else None
                length = len(item[0]) if item is not None else 0
                if item is not None:
                    summary["max_tokens"] = max(summary["max_tokens"], length)
                    summary["min_tokens"] = length if summary["min_tokens"] is None else min(summary["min_tokens"], length)
                    summary["max_completion_tokens"] = max(summary["max_completion_tokens"], length - item[1])
                if split == "train" and cfg["mode"] == "engineering_probe":
                    # Strictly greater preserves the first source row for ties.
                    if length > longest_length:
                        rows[split][:] = [row]
                        processed[split][:] = [item] if item is not None else []
                        retained_indices[split][:] = [index]
                        longest_length = length
                elif index in (chosen_train if split == "train" else selected_val):
                    rows[split].append(row)
                    if item is not None:
                        processed[split].append(item)
                    retained_indices[split].append(index)
                summary["count"] += 1
        require(summary["count"] == inputs[split]["count"], "registered_record_count_mismatch")
        require(stream_hash.hexdigest() == inputs[split]["sha256"], "input_changed_during_stream_validation")
        summary.update(unique_ids=len(ids[split]), subjects=len(subjects[split]),
                       retained_records=len(rows[split]))
        summaries[split] = summary
    if "val" in rows:
        require(not ids["train"] & ids["val"], "train_val_id_overlap")
        require(not subjects["train"] & subjects["val"], "train_val_subject_overlap")
        require(not recordings["train"] & recordings["val"], "train_val_recording_overlap")
    if cfg["mode"] == "engineering_probe":
        source_order = retained_indices["train"] * 2
    index_to_retained = {index: i for i, index in enumerate(retained_indices["train"])}
    local_order = [index_to_retained[index] for index in source_order]
    require(len(local_order) == cfg["max_updates"], "retained_stage_order_incomplete")
    require(len(rows["train"]) <= 400 and len(rows.get("val", [])) <= 36, "retained_record_budget_exceeded")
    # ids/subjects/recordings and unselected tokens die before the GPU model loads.
    require(sha256(candidate_path) == cfg["residual_candidate"]["sha256"], "candidate_changed_during_scan")
    return manifest, rows, processed, local_order, source_order, retained_indices, summaries


def read_rows(cfg):
    """Metadata-only bounded inspection; GPU training uses full tokenizer preflight."""
    manifest, rows, *_ = scan_inputs(cfg)
    return manifest, rows


def render(tokenizer, row, max_length=1024):
    messages = row["messages"]
    common = dict(tokenize=True, return_dict=False, enable_thinking=False)
    prefix = list(tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True, **common))
    tokens = list(tokenizer.apply_chat_template(messages, add_generation_prompt=False, **common))
    require(tokens[:len(prefix)] == prefix, "train_eval_prefix_mismatch")
    require(0 < len(prefix) < len(tokens) <= max_length, "completion_truncated_or_empty")
    require(tokenizer.decode(tokens[len(prefix):], skip_special_tokens=True).strip()
            == messages[1]["content"].strip(), "completion_tokens_do_not_match_target")
    require(1 <= len(tokens) - len(prefix) <= 128, "completion_budget_exceeded")
    return tokens, len(prefix)


def selected_order(cfg, processed):
    count = processed if type(processed) is int else len(processed)
    if cfg["mode"] == "engineering_probe":
        index = max(range(count), key=lambda i: (len(processed[i][0]), -i))
        return [index, index]
    offset = cfg.get("order_offset", 0)
    return audited.training_order(count, cfg["seed"], offset + cfg["max_updates"])[offset:]


def padded_length(cfg, token_count):
    require(0 < token_count <= cfg["max_seq_length"], "refusing_sequence_truncation")
    return cfg["max_seq_length"] if cfg["mode"] == "engineering_probe" else min(cfg["max_seq_length"], 1 + 32 * ((token_count + 31) // 32))


def preflight(cfg, tokenizer=None):
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], local_files_only=True,
                                                  trust_remote_code=False)
    manifest, rows, processed, order, source_order, retained, summaries = scan_inputs(cfg, tokenizer)
    report = {"schema": SCHEMA, "status": "passed", "created_at": utc_now(),
              "task_kind": manifest["task_kind"], "mode": cfg["mode"],
              "base_identity": base_identity(cfg), "data_manifest_sha256": cfg["data_manifest"]["sha256"],
              "test_opened": False, "enable_thinking": False, "all_prefixes_and_loss_masks_exact": True,
              "max_seq_length": cfg["max_seq_length"], "streaming_validation": True,
              "order_sha256": digest(source_order), "order_offset": cfg.get("order_offset", 0),
              "selected_source_indices": source_order,
              "retained_training_source_indices": retained["train"],
              "retained_validation_source_indices": retained.get("val", []),
              "selected_tokenization_sha256": digest([processed["train"][i] for i in order]),
              "selected_record_hashes": [digest(rows["train"][i]["id"]) for i in order],
              "selected_actual_token_lengths": [len(processed["train"][i][0]) for i in order],
              "selected_padded_batch_lengths": [padded_length(cfg, len(processed["train"][i][0])) for i in order],
              "splits": summaries,
              "probe_repeats_longest_example_for_engineering_only": cfg["mode"] == "engineering_probe"}
    return report, manifest, rows, processed, order


def interpolation_baseline(row):
    rec = row["reconstruction"]
    observed, indices = rec["observed"], rec["masked_indices"]
    time_count = rec["geometry"]["shape"][1]
    require(indices[0] // time_count == indices[-1] // time_count
            and indices[0] % time_count > 0 and indices[-1] % time_count < time_count - 1,
            "interpolation_would_cross_channel_boundary")
    left, right = indices[0] - 1, indices[-1] + 1
    require(observed[left] is not None and observed[right] is not None, "missing_interpolation_endpoint")
    return [observed[left] + (observed[right] - observed[left]) * (i - left) / (right - left) for i in indices]


def reconstruction_metrics(outputs, records, bounds=None):
    require(len(outputs) == len(records) and records, "unpaired_reconstruction_evaluation")
    errors, valid = [], 0
    all_baselines = {"zero": [], "visible_mean": [], "within_channel_interpolation": []}
    valid_baselines = {k: [] for k in all_baselines}
    for output, row in zip(outputs, records):
        target = json.loads(row["messages"][1]["content"])
        visible = [x for x in row["reconstruction"]["observed"] if x is not None]
        baseline = sum(visible) / len(visible)
        baseline_errors = {"zero": [float(x * x) for x in target],
            "visible_mean": [(x - baseline) ** 2 for x in target],
            "within_channel_interpolation": [(a - b) ** 2 for a, b in zip(target, interpolation_baseline(row))]}
        for name, values in baseline_errors.items():
            all_baselines[name].extend(values)
        try:
            pred = json.loads(output.strip())
        except (ValueError, TypeError):
            continue
        if not isinstance(pred, list) or len(pred) != len(target) or not all(type(x) is int for x in pred):
            continue
        if bounds is not None and not all(bounds[0] <= x <= bounds[1] for x in pred):
            continue
        valid += 1
        errors.extend((a - b) ** 2 for a, b in zip(pred, target))
        for name, values in baseline_errors.items():
            valid_baselines[name].extend(values)
    return {"n": len(records), "valid_count": valid, "valid_rate": valid / len(records),
            "invalid_rate": 1 - valid / len(records),
            "mse_on_valid_predictions": sum(errors) / len(errors) if errors else None,
            **{name + "_baseline_mse_all_targets": sum(values) / len(values) for name, values in all_baselines.items()},
            **{name + "_baseline_mse_valid_targets": sum(values) / len(values) if values else None
               for name, values in valid_baselines.items()},
            "interpretation": "masked quantized integer reconstruction; not action/intent/vision accuracy"}


def task_metrics(outputs, records, manifest):
    if manifest["task_kind"] == "quantized_eeg_reconstruction":
        if records[0].get("task_kind") == "quantized_eeg_residual_candidate_v1":
            absolute_outputs, absolute_records = residual.absolute_inputs(outputs, records)
            return {**reconstruction_metrics(absolute_outputs, absolute_records, manifest["quantization_bounds"]), **residual.controls(records), "residual_protocol": "rounded_linear_residual_v1", "metric_units": "uV^2"}
        return reconstruction_metrics(outputs, records, manifest["quantization_bounds"])
    require(len(outputs) == len(records) and records, "unpaired_action_evaluation")
    parsed = [{"gold": row["messages"][1]["content"], "pred": audited.parse_prediction(output)}
              for row, output in zip(records, outputs)]
    return {"n": len(parsed), "accuracy": sum(x["gold"] == x["pred"] for x in parsed) / len(parsed),
            "valid_rate": sum(x["pred"] != "invalid" for x in parsed) / len(parsed),
            "prediction_counts": dict(collections.Counter(x["pred"] for x in parsed))}


def compare_initial_task(initial_outputs, current_outputs, records, manifest):
    """Compare the same development rows, with matched valid targets for MSE."""
    require(len(initial_outputs) == len(current_outputs) == len(records) and records,
            "unpaired_initial_model_comparison")
    if records[0].get("task_kind") == "quantized_eeg_residual_candidate_v1":
        initial_outputs, absolute_records = residual.absolute_inputs(initial_outputs, records)
        current_outputs, _ = residual.absolute_inputs(current_outputs, records)
        records = absolute_records
    initial = task_metrics(initial_outputs, records, manifest)
    current = task_metrics(current_outputs, records, manifest)
    report = {"n": len(records), "initial_valid_rate": initial["valid_rate"],
              "current_valid_rate": current["valid_rate"],
              "valid_rate_delta": current["valid_rate"] - initial["valid_rate"],
              "development_only": True}
    if manifest["task_kind"] == "action_classification":
        report.update(initial_accuracy=initial["accuracy"], current_accuracy=current["accuracy"],
                      accuracy_delta=current["accuracy"] - initial["accuracy"])
        return report
    lo, hi = manifest["quantization_bounds"]
    initial_errors, current_errors, paired = [], [], 0
    for before, after, row in zip(initial_outputs, current_outputs, records):
        target = json.loads(row["messages"][1]["content"])
        try:
            a, b = json.loads(before.strip()), json.loads(after.strip())
        except (ValueError, TypeError):
            continue
        if not all(isinstance(x, list) and len(x) == len(target)
                   and all(type(v) is int and lo <= v <= hi for v in x) for x in (a, b)):
            continue
        paired += 1
        initial_errors.extend((x - y) ** 2 for x, y in zip(a, target))
        current_errors.extend((x - y) ** 2 for x, y in zip(b, target))
    before_mse = sum(initial_errors) / len(initial_errors) if initial_errors else None
    after_mse = sum(current_errors) / len(current_errors) if current_errors else None
    report.update(paired_valid_examples=paired, paired_valid_target_values=len(initial_errors),
                  initial_mse_paired_valid=before_mse, current_mse_paired_valid=after_mse,
                  mse_delta_current_minus_initial=(after_mse - before_mse) if paired else None,
                  interpretation="descriptive paired development comparison; no meaningful-improvement claim or automatic efficacy gate")
    return report


class SupervisorSignal(RuntimeError):
    def __init__(self, signum):
        self.signum = signum
        super().__init__("supervisor_received_signal_" + str(signum))


def supervisor_signal_handler(signum, frame):
    # Default SIGTERM bypasses finally; raising forces child-process-group cleanup.
    raise SupervisorSignal(signum)


def run_supervisor(config_path, cfg):
    run_dir = Path(cfg["run_dir"]).expanduser().resolve()
    model_dir = Path(cfg["model_path"]).expanduser().resolve()
    require(run_dir != model_dir and model_dir not in run_dir.parents and run_dir not in model_dir.parents,
            "run_directory_overlaps_base_model")
    lock = Path(cfg["lock_path"]).expanduser()
    require(lock.is_file() and not lock.is_symlink(), "registered_gpu_lock_missing")
    require(not run_dir.exists(), "refusing_existing_run_directory")
    run_dir.mkdir(parents=True)
    atomic_json(run_dir / "config.json", cfg)
    fd, child, result, status = os.open(lock, os.O_RDWR), None, 2, "failed"
    previous_handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous_handlers:
        signal.signal(sig, supervisor_signal_handler)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            status, result = "blocked_gpu_lock", 11
            return result
        commands = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True,
                                  timeout=10, check=True).stdout.splitlines()
        require(not any("mlx_lm lora" in c or "eeg_llm.py" in c for c in commands), "other_unlocked_trainer_running")
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                   TOKENIZERS_PARALLELISM="false")
        with (run_dir / "worker.log").open("x") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_train",
                                      "--config", str(run_dir / "config.json"), "--lock-fd", str(fd)],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     pass_fds=(fd,), start_new_session=True, env=env)
            atomic_json(run_dir / "supervisor.json", {"status": "running", "pid": os.getpid(),
                "child_pid": child.pid, "started_at": utc_now(), "max_wall_seconds": cfg["max_wall_seconds"]})
            try:
                result = child.wait(timeout=cfg["max_wall_seconds"])
                status = "completed" if result == 0 else "failed"
            except subprocess.TimeoutExpired:
                status, result = "stopped_wall_budget", 124
        return result
    except SupervisorSignal as exc:
        status, result = "stopped_signal", 128 + exc.signum
        return result
    finally:
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=15)
            if child.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=15)
        atomic_json(run_dir / "supervisor.json", {"schema": SCHEMA, "status": status, "exit_code": result,
            "pid": os.getpid(), "child_pid": child.pid if child else None, "ended_at": utc_now(),
            "test_opened": False})
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)


def train_worker(cfg, lock_fd):
    run_dir = Path(cfg["run_dir"])
    require((os.fstat(lock_fd).st_dev, os.fstat(lock_fd).st_ino) ==
            (os.stat(cfg["lock_path"]).st_dev, os.stat(cfg["lock_path"]).st_ino), "worker_missing_gpu_lock")
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    state = {"schema": SCHEMA, "run_id": cfg["run_id"], "mode": cfg["mode"], "status": "preflight",
             "completed_updates": 0, "completed_optimizer_steps": 0, "test_opened": False, "started_at": utc_now(),
             "exact_optimizer_resume": False, "auto_scope_change_allowed": False}

    def update(**changes):
        state.update(changes, updated_at=utc_now())
        atomic_json(run_dir / "state.json", state)

    try:
        update()
        report, manifest, rows, processed, order = preflight(cfg)
        atomic_json(run_dir / "preflight.json", report)
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        import numpy as np
        from mlx.utils import tree_flatten
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.tuner.trainer import grad_checkpoint
        from mlx_lm.tuner.utils import linear_to_lora_layers

        require(mx.distributed.init().size() == 1, "single_gpu_protocol_only")
        mx.set_memory_limit(int(cfg["memory_limit_gb"] * 1e9))
        mx.set_cache_limit(0)
        mx.clear_cache()
        mx.random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        update(status="loading_model")
        model, tokenizer = load(cfg["model_path"], lazy=False)
        actual, _, _, _, _ = preflight(cfg, tokenizer)
        require(actual["splits"] == report["splits"] and actual["order_sha256"] == report["order_sha256"]
                and actual["selected_tokenization_sha256"] == report["selected_tokenization_sha256"],
                "runtime_tokenizer_preflight_mismatch")
        source_count = report["splits"]["train"]["count"]
        source_order = report["selected_source_indices"]
        require(type(model).__module__ == "mlx_lm.models.qwen3_5" and len(model.layers) == 64
                and hasattr(model.language_model, "lm_head")
                and not model.language_model.args.tie_word_embeddings
                and model.language_model.sharding_group is None, "unaudited_model_forward_path")
        model.freeze()
        targets, keys = [], set()
        for i, layer in enumerate(model.layers):
            for name, module in layer.named_modules():
                if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                    targets.append({"layer": i, "name": name, "type": type(module).__name__})
                    keys.add(name)
        require({t["layer"] for t in targets} == set(range(64)), "missing_decoder_layer_targets")
        require(not any(any(x in t["name"] for x in ("embed_tokens", "lm_head", "vision")) for t in targets),
                "unexpected_head_or_vision_training_target")
        parameters = {"rank": cfg["rank"], "scale": 20.0, "dropout": 0.0, "keys": sorted(keys)}
        linear_to_lora_layers(model, 64, parameters)
        trainable = dict(tree_flatten(model.trainable_parameters()))
        expected_names = {f"language_model.model.layers.{t['layer']}.{t['name']}.{suffix}"
                          for t in targets for suffix in ("lora_a", "lora_b")}
        require(set(trainable) == expected_names, "incomplete_or_extra_lora_trainable_targets")
        module_audit = {"num_layers": 64, "rank": cfg["rank"], "targets": targets,
            "target_count": len(targets), "trainable_parameters": sum(x.size for x in trainable.values()),
            "trainable_names": sorted(trainable), "target_names_sha256": digest(sorted(trainable)),
            "base_frozen": True, "head_embedding_vision_excluded": True,
            "memory_policy": {"allocator_limit_is_soft": True, "configured_gb": 54,
                "lifetime_peak_guard_gb": 54, "cache_limit_bytes": 0, "peak_reset": False},
            "grad_checkpoint": "all instances of the shared DecoderLayer class"}
        atomic_json(run_dir / "module_audit.json", module_audit)
        parent = None
        if cfg.get("warm_start_adapter"):
            parent = warm_start_metadata(cfg)
            loaded = mx.load(str(Path(cfg["warm_start_adapter"]["checkpoint_dir"]) / "adapters.safetensors"))
            require(set(loaded) == set(trainable)
                    and all(loaded[k].shape == trainable[k].shape for k in trainable),
                    "warm_start_adapter_keys_or_shapes_mismatch")
            model.load_weights(list(loaded.items()), strict=False)
            del loaded
        require(set(dict(tree_flatten(model.trainable_parameters()))) == expected_names,
                "base_unfrozen_after_adapter_load")
        mx.eval(model.parameters())
        del trainable  # Do not retain initial A/B arrays after optimizer replacements.
        optimizer = optim.Adam(learning_rate=cfg["learning_rate"])
        reference_sentry = {"n": 6, "passed": 6}
        sampler = make_sampler(temp=0.0)
        completion_width = max(s["max_completion_tokens"] for s in report["splits"].values())
        eval_indices = list(range(len(rows.get("val", []))))
        chosen_development = [rows["val"][i] for i in eval_indices]
        baseline_outputs, baseline_task = None, None
        atomic_json(run_dir / "selection.json", {
            "training_indices": source_order,
            "retained_training_source_indices": report["retained_training_source_indices"],
            "validation_indices": report["retained_validation_source_indices"],
            "test_opened": False, "probe_validation_generation": False,
            "source_train_count": source_count,
            "selection_purpose": "engineering" if cfg["mode"] == "engineering_probe" else "development_only"})

        def guard(loss=None):
            elapsed = time.monotonic() - started
            observation = audited.memory_snapshot(mx)
            state["memory_observation"] = {"phase": state["status"], "step": state["completed_updates"], **observation}
            state["peak_memory_gb"] = observation["peak_gb"]
            state["elapsed_seconds"] = elapsed
            loss_ok = loss is None or (math.isfinite(loss) and 0 <= loss <= 100)
            if elapsed >= cfg["max_wall_seconds"] or observation["peak_gb"] > 54 or not loss_ok:
                update()  # Persist completed update, seconds, loss and memory before raising.
            require(elapsed < cfg["max_wall_seconds"], "wall_budget_exceeded")
            audited.check_memory_snapshot(observation, 54, state["completed_updates"])
            require(loss_ok, "nonfinite_or_excessive_loss")
            return observation

        def ask(prompt, max_tokens=24):
            guard()
            ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                    return_dict=False, add_generation_prompt=True, enable_thinking=False)
            answer = generate(model, tokenizer, ids, verbose=False, max_tokens=max_tokens, sampler=sampler)
            mx.clear_cache()
            guard()
            return answer

        def sentry():
            prompts = [
                ("What is 7 + 5? Reply with only the number.", "12"),
                ("Sort these numbers ascending: 9, 2, 5. Reply only: a,b,c with no spaces.", "2,5,9"),
                ("Return only the lowercase English translation of 猫.", "cat"),
                ('Output exactly this JSON and nothing else: {"ok":true}', '{"ok":true}'),
                ("Reply with exactly the word READY and nothing else.", "READY"),
                ("日本語で「了解しました」とだけ答えてください．", "了解しました"),
            ]
            results = [ask(prompt).strip() == expected for prompt, expected in prompts]
            return {"n": 6, "passed": sum(results), "checks": results,
                    "scope": "basic_format_and_instruction_smoke_only"}

        def evaluate_task():
            require(cfg["mode"] == "train_stage", "engineering_probe_has_no_task_generation")
            outputs = []
            update(eval_done=0, eval_target=len(chosen_development))
            for n, row in enumerate(chosen_development, 1):
                outputs.append(ask(row["messages"][0]["content"], max_tokens=min(136, completion_width + 8)))
                update(eval_done=n)
            return task_metrics(outputs, chosen_development, manifest), outputs

        model.eval()
        update(status="baseline_sentry")
        initial_sentry = sentry()
        atomic_json(run_dir / "baseline_sentry.json", {"observed": initial_sentry, "fixed_reference": reference_sentry})
        require(initial_sentry["passed"] >= 6 - cfg.get("max_sentry_drop", 1), "initial_basic_instruction_gate_failed")
        if cfg["mode"] == "train_stage":
            update(status="baseline_task_evaluation")
            baseline_task, baseline_outputs = evaluate_task()
            if parent is not None:
                continuation.verify_baseline(continuation.admit(cfg), baseline_task)
            atomic_json(run_dir / "baseline_task_predictions.private.json", {
                "indices": report["retained_validation_source_indices"], "outputs": baseline_outputs,
                "test_opened": False})
            atomic_json(run_dir / "baseline_task_evaluation.json", {
                "schema": SCHEMA, "task_kind": manifest["task_kind"], "completed_optimizer_steps": 0,
                "model_state": "parent_adapter_before_stage" if parent else "frozen_base_before_stage",
                "metrics": baseline_task, "quality_gate_applied": False,
                "selected_validation_indices": report["retained_validation_source_indices"],
                "data_manifest_sha256": cfg["data_manifest"]["sha256"], "development_only": True,
                "test_opened": False})
        model.train()
        # Every decoder block has this class; apply checkpointing once per process.
        require(len({type(x) for x in model.layers}) == 1, "unexpected_mixed_decoder_layer_classes")
        grad_checkpoint(model.layers[0])

        def loss_fn(current, batch, lengths):
            return audited.completion_head_loss(mx, nn, current, batch, lengths, completion_width)

        value_and_grad = nn.value_and_grad(model, loss_fn)
        compiled_state = [model.state, optimizer.state, mx.random.state]

        @partial(mx.compile, inputs=compiled_state, outputs=compiled_state)
        def step(batch, lengths):
            (loss, tokens), gradients = value_and_grad(model, batch, lengths)
            optimizer.update(model, gradients)
            return loss, tokens

        adapter_config = {"fine_tune_type": "lora", "num_layers": 64, "lora_parameters": parameters,
                          "model": cfg["model_path"], "enable_thinking": False, "mask_prompt": True,
                          "loss_mask": "prefix_length <= target_position < token_length"}
        if manifest["task_kind"] == "quantized_eeg_reconstruction":
            adapter_config["eeg_serialization"] = {
                "task_kind": manifest["task_kind"], "quantization_schema": manifest["quantization_schema"],
                "quantization_bounds": manifest["quantization_bounds"], "geometry_required": True,
                "quantum_uv": 1, "clipping": False,
                "max_channels_per_training_block": manifest.get("max_channels_per_training_block", 8),
                "prompt_function": "evlm_residual_protocol.residual_row",
                "masked_target": "JSON signed residual array; reconstruct absolute uV by adding rounded linear baseline",
                "residual_protocol": "rounded_linear_residual_v1",
            }
        else:
            adapter_config["eeg_serialization"] = {
                "task_kind": manifest["task_kind"],
                "serialization_schema": manifest.get("serialization_schema", "manifest-registered-action-prompt"),
                "data_manifest_sha256": cfg["data_manifest"]["sha256"],
            }

        def save_checkpoint(update_number):
            directory = run_dir / "checkpoints" / f"{update_number:07d}"
            directory.mkdir(parents=True, exist_ok=False)
            update(status="checkpointing", checkpoint_status="writing", attempted_checkpoint=str(directory))
            model.eval()
            try:
                before = ask("Reply with exactly the word READY and nothing else.")
                weights = dict(tree_flatten(model.trainable_parameters()))
                mx.save_safetensors(str(directory / "adapters.safetensors"), weights)
                atomic_json(directory / "adapter_config.json", adapter_config)
                loaded = mx.load(str(directory / "adapters.safetensors"))
                require(set(loaded) == expected_names and all(loaded[k].shape == weights[k].shape for k in weights),
                        "saved_adapter_reload_key_or_shape_mismatch")
                equal = [mx.array_equal(loaded[k], weights[k]) for k in sorted(weights)]
                mx.eval(equal)
                require(all(bool(x.item()) for x in equal), "saved_adapter_tensor_roundtrip_mismatch")
                model.load_weights(list(loaded.items()), strict=False)
                del loaded, weights, equal
                after = ask("Reply with exactly the word READY and nothing else.")
                require(after == before, "adapter_reload_changed_generation")
                sentry_result = sentry()
                evaluated_metrics, private_outputs = None, []
                if cfg["mode"] == "train_stage":
                    update(status="evaluating", eval_done=0, eval_target=len(eval_indices))
                    evaluated_metrics, private_outputs = evaluate_task()
                    evaluated_metrics["initial_model_comparison"] = compare_initial_task(
                        baseline_outputs, private_outputs, chosen_development, manifest)
                    atomic_json(directory / "predictions.private.json", {
                        "indices": report["retained_validation_source_indices"], "outputs": private_outputs, "test_opened": False})
                gates = {"basic_instruction_fixed_base": sentry_result["passed"] >= 6 - cfg.get("max_sentry_drop", 1),
                         "adapter_reload": True,
                         "output_format": evaluated_metrics is None or evaluated_metrics["valid_rate"] >= 0.9}
                passed = all(gates.values())
                evaluation = {"schema": SCHEMA, "status": "completed" if passed else "failed_quality_gates",
                    "mode": cfg["mode"], "task_kind": manifest["task_kind"], "completed_updates": update_number,
                    "completed_optimizer_steps": update_number, "num_layers": 64, "rank": cfg["rank"],
                    "max_seq_length": cfg["max_seq_length"],
                    "expected_model_config_sha256": cfg["expected_model_config_sha256"],
                    "base_identity": base_identity(cfg),
                    "base_receipt_sha256": cfg["base_receipt"]["sha256"],
                    "model_path": str(Path(cfg["model_path"]).expanduser().resolve()),
                    "adapter_sha256": sha256(directory / "adapters.safetensors"),
                    "reference_sentry": reference_sentry, "sentry": sentry_result,
                    "quality_gates": gates, "quality_gates_passed": passed, "metrics": evaluated_metrics,
                    "baseline_task_evaluation_sha256": sha256(run_dir / "baseline_task_evaluation.json") if baseline_task is not None else None,
                    "adapter_reload": {"tensors_exact": True, "ready_generation_equal": True,
                        "scope": "adapter reloaded into same frozen base instance; fresh-process VLM test separate"},
                    "validation_generation_count": len(private_outputs), "test_opened": False,
                    "development_only": True, "memory_observation": guard()}
                atomic_json(directory / "evaluation.json", evaluation)
                meta = {"schema": SCHEMA, "run_id": cfg["run_id"], "mode": cfg["mode"],
                    "status": "completed" if passed else "failed_quality_gates", "quality_gates_passed": passed,
                    "task_kind": manifest["task_kind"], "completed_updates": update_number,
                    "completed_optimizer_steps": update_number,
                    "base_identity": base_identity(cfg), "num_layers": 64, "rank": cfg["rank"],
                    "max_seq_length": cfg["max_seq_length"],
                    "adapter_sha256": sha256(directory / "adapters.safetensors"),
                    "adapter_config_sha256": sha256(directory / "adapter_config.json"),
                    "evaluation_sha256": sha256(directory / "evaluation.json"),
                    "data_manifest_sha256": cfg["data_manifest"]["sha256"],
                    "parent_checkpoint": cfg.get("warm_start_adapter"), "reference_sentry": reference_sentry,
                    "optimizer_initialization": "fresh_Adam", "same_live_optimizer_within_stage": True,
                    "exact_optimizer_resume": False, "order_offset": cfg.get("order_offset", 0),
                    "order_sha256": digest(source_order[:update_number]),
                    "train_presentations": update_number, "unique_train_examples": len(set(order[:update_number])),
                    "source_train_count": source_count, "epoch_equivalent": update_number / source_count,
                    "cumulative_train_presentations": (parent or {}).get("cumulative_train_presentations", 0) + update_number,
                    "test_opened": False}
                atomic_json(directory / "checkpoint.json", meta)
                require(passed, "checkpoint_quality_gate_failed")
                update(checkpoint_status="verified", last_checkpoint=str(directory),
                       last_checkpoint_sha256=sha256(directory / "checkpoint.json"))
            finally:
                model.train()
                mx.clear_cache()

        update(status="training", task_kind=manifest["task_kind"], optimizer="Adam",
               optimizer_initialization="fresh_Adam", initialization="adapter_warm_start" if parent else "fresh_base",
               parent_checkpoint=cfg.get("warm_start_adapter"), max_updates=cfg["max_updates"],
               source_train_count=source_count, retained_train_records=len(rows["train"]),
               retained_validation_records=len(chosen_development), max_seq_length=cfg["max_seq_length"],
               data_manifest_sha256=cfg["data_manifest"]["sha256"])
        for update_number, i in enumerate(order, 1):
            guard()
            tokens, prefix = processed["train"][i]
            padding = padded_length(cfg, len(tokens))
            batch = np.zeros((1, padding), dtype=np.int32)
            batch[0, :len(tokens)] = tokens
            tick = time.monotonic()
            loss, ntokens = step(mx.array(batch), mx.array([[prefix, len(tokens)]]))
            mx.eval(compiled_state, loss, ntokens)
            mx.clear_cache()
            loss_value = float(loss.item())
            state.update(completed_updates=update_number, completed_optimizer_steps=update_number,
                         train_presentations=update_number,
                         unique_train_examples=len(set(order[:update_number])),
                         loss=loss_value if math.isfinite(loss_value) else str(loss_value),
                         supervised_tokens=int(ntokens.item()), update_seconds=time.monotonic() - tick,
                         token_length=len(tokens), padded_batch_tokens=padding, model_input_tokens=padding - 1,
                         epoch_equivalent=update_number / source_count)
            measured = guard(loss_value)
            update(status="training", memory_observation=measured)
            print(json.dumps({k: state[k] for k in ("completed_updates", "loss", "update_seconds", "token_length", "memory_observation")}), flush=True)
            if update_number in cfg["checkpoint_steps"]:
                save_checkpoint(update_number)
        require(sha256(Path(cfg["model_path"]) / "config.json") == cfg["expected_model_config_sha256"],
                "base_config_changed_during_run")
        checked_file(cfg["base_receipt"])
        update(status="completed", ended_at=utc_now(), next_order_offset=cfg.get("order_offset", 0) + len(order),
               final_checkpoint=state["last_checkpoint"], quality_gates_passed=True)
        atomic_json(run_dir / "next_action.json", {"schema": SCHEMA, "action": "review_completed_stage_before_next_source",
            "final_checkpoint": state["last_checkpoint"], "auto_scope_change_allowed": False,
            "test_opened": False, "exact_optimizer_resume": False})
        return 0
    except Exception as exc:
        update(status="failed", ended_at=utc_now(), error_type=type(exc).__name__, error=str(exc)[:1000])
        atomic_json(run_dir / "next_action.json", {"schema": SCHEMA, "action": "review_failure_before_restart",
            "reason": type(exc).__name__, "auto_scope_change_allowed": False, "test_opened": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run", "_train"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "preflight":
            p.add_argument("--receipt", required=True)
        if name == "_train":
            p.add_argument("--lock-fd", type=int, required=True)
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    if args.command == "preflight":
        report, _, _, _, _ = preflight(cfg)
        atomic_json(args.receipt, report)
        print(json.dumps(report))
        return 0
    if args.command == "run":
        return run_supervisor(args.config, cfg)
    return train_worker(cfg, args.lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
