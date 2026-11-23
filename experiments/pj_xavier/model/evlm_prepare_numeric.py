#!/usr/bin/env python3
"""Prepare measured numeric EEG reconstruction from an immutable certified corpus.

CPU only.  No learned EEG encoder, fabricated action/image labels, resplitting,
amplitude correction, clipping, or silent token truncation is permitted.  Raw
and prepared signals stay on their S Mac.  A source window is not the same as
the shorter crop actually exposed to the model; both coverages are recorded.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

PREP_SCHEMA = "evlm-certified-numeric-preparation-v1"
QUANTIZATION_SCHEMA = "eeg-raw32-uv1-channel-major-cycle19-v1"
DATA_SCHEMA = "evlm-data-manifest-v1"
TASK_KIND = "quantized_eeg_reconstruction"
FS = 200
WINDOW_SAMPLES = 600
CROP_SAMPLES = 32
CHANNEL_BLOCK = 8
MASK_POINTS = 4
CROP_SLOTS = math.ceil(WINDOW_SAMPLES / CROP_SAMPLES)
BOUNDS = [-100, 100]
SOURCE_ORDER = ("zenodo:806023", "figshare:2061654", "mendeley:rjx76wd5v6",
                "openneuro:ds005028", "figshare:29606906", "zenodo:8089820")
IDENTITY_FILES = {"manifest_sha256": "manifest.jsonl", "protocol_sha256": "protocol.json",
                  "source_admission_sha256": "source_admission.json",
                  "certificates_sha256": "source_certificates.json",
                  "split_manifest_sha256": "split_manifest.json",
                  "parent_provenance_sha256": "parent_provenance.json",
                  "artifacts_sha256": "artifacts.json"}


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with tmp.open("x") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "corpus_path_escape")
    return path


def source_id(row):
    parts = row["source_recording_id"].split(":")
    require(len(parts) >= 3 and all(parts[:2]), "missing_canonical_source_namespace")
    return ":".join(parts[:2])


def source_slug(value):
    require(re.fullmatch(r"[A-Za-z0-9_.:-]+", value), "unsafe_source_namespace")
    return value.replace(":", "_")


def validate_corpus(corpus, expected_completion_sha256):
    """Read metadata only; positive certificates and inherited denies are binding."""
    corpus = Path(corpus).resolve()
    require(re.fullmatch(r"[0-9a-f]{64}", expected_completion_sha256 or ""),
            "missing_fixed_completion_digest")
    require(sha(corpus / "completion.json") == expected_completion_sha256,
            "certified_completion_changed")
    completion = json.loads((corpus / "completion.json").read_text())
    require(completion.get("status") == "prepared_verified_new_offline_SSL_corpus",
            "corpus_is_not_certified_prepared_SSL")
    require(set(completion["windows"]) <= {"train", "val"}, "unexpected_heldout_partition")
    hashes = {"completion.json": expected_completion_sha256}
    for key, name in IDENTITY_FILES.items():
        actual = sha(corpus / name)
        require(actual == completion.get(key), "certified_metadata_changed_" + name)
        hashes[name] = actual
    admission = json.loads((corpus / "source_admission.json").read_text())
    artifacts = json.loads((corpus / "artifacts.json").read_text())
    rows = [json.loads(line) for line in (corpus / "manifest.jsonl").read_text().splitlines()
            if line.strip()]
    require(dict(collections.Counter(r["split"] for r in rows)) == completion["windows"],
            "certified_partition_count_mismatch")
    require(len({r["id"] for r in rows}) == len(rows), "duplicate_source_window_id")
    denied_sources = set(admission["excluded_source_sha256"])
    denied_payloads = set(admission["excluded_payload_sha256"])
    denied_groups = set(admission["excluded_source_groups"])
    groups = collections.defaultdict(set)
    payloads = collections.defaultdict(set)
    for row in rows:
        split = row["split"]
        require(split in ("train", "val"), "heldout_window_forbidden")
        require(row["unit"] == "V_x_1e4" and row["sample_rate_hz"] == FS
                and row.get("physical_unit_verified") is True, "unverified_units_or_rate")
        require(abs((row["window_end_s"] - row["window_start_s"]) * FS - WINDOW_SAMPLES) < 1e-6,
                "unexpected_source_window_duration")
        channels = row["channels"]
        require(channels and len(set(channels)) == len(channels)
                and all(isinstance(c, str) and c.strip() for c in channels),
                "missing_or_duplicate_channel_mapping")
        require(row.get("subject_group") and row.get("montage_key") and row.get("reference"),
                "missing_source_identity_or_montage")
        require(row["source_sha256"] not in denied_sources
                and row["eeg_payload_sha256"] not in denied_payloads,
                "denied_source_or_payload")
        require(split != "train" or row["subject_group"] not in denied_groups,
                "protected_group_in_training")
        cert = admission["sources"].get(row["source_sha256"])
        require(cert is not None and cert.get("split") == split
                and cert.get("subject_group") == row["subject_group"]
                and cert.get("montage_key") == row["montage_key"]
                and cert.get("eeg_payload_sha256") == row["eeg_payload_sha256"]
                and cert.get("physical_unit_verified") is True,
                "positive_source_certificate_mismatch")
        require(cert.get("fit_allowed") is (split == "train"), "source_fit_role_mismatch")
        if split == "val":
            require(cert.get("validation_allowed") is True, "source_not_allowed_for_development")
        artifact = artifacts.get(row["eeg_path"])
        require(artifact is not None and artifact["dtype"] == "float32"
                and artifact["shape"][1:] == [len(channels), 3, FS]
                and type(row["eeg_index"]) is int
                and 0 <= row["eeg_index"] < artifact["shape"][0], "artifact_geometry_or_slot_mismatch")
        contained(corpus, row["eeg_path"])
        source_id(row)
        groups[split].add(row["subject_group"])
        payloads[split].add(row["eeg_payload_sha256"])
    require(not groups["train"] & groups["val"], "source_train_development_group_overlap")
    require(not payloads["train"] & payloads["val"], "source_train_development_payload_overlap")
    return {"root": corpus, "rows": rows, "artifacts": artifacts, "admission": admission,
            "completion": completion, "metadata_sha256": hashes}


def crop_bounds(window_id, seed, cycle):
    require(type(cycle) is int and 0 <= cycle < CROP_SLOTS, "cycle_must_be_0_through_18")
    offset = int(canonical_sha([QUANTIZATION_SCHEMA, "crop", seed, window_id])[:16], 16)
    slot = (offset + cycle) % CROP_SLOTS
    first = slot * CROP_SAMPLES
    return first, min(first + CROP_SAMPLES, WINDOW_SAMPLES), slot


def mask_indices(window_id, seed, cycle, block_index, channels, times):
    require(1 <= channels <= CHANNEL_BLOCK and times >= MASK_POINTS + 2, "invalid_mask_geometry")
    value = int(canonical_sha([QUANTIZATION_SCHEMA, "mask", seed, window_id,
                               cycle, block_index])[:16], 16)
    channel = value % channels
    # Four contiguous missing samples with a visible endpoint on both sides.
    first = 1 + (value // channels) % (times - MASK_POINTS - 1)
    return [channel * times + first + i for i in range(MASK_POINTS)]


def interpolation_baseline(observed, masked, shape):
    channels, times = shape
    require(len(observed) == channels * times and masked == sorted(set(masked)),
            "invalid_baseline_geometry")
    require(len(masked) == MASK_POINTS
            and masked == list(range(masked[0], masked[0] + MASK_POINTS)), "mask_not_contiguous")
    channel, start = divmod(masked[0], times)
    require(start > 0 and start + MASK_POINTS < times
            and masked[-1] // times == channel, "mask_crosses_channel_or_has_no_endpoints")
    require(all(observed[i] is None for i in masked), "baseline_received_masked_truth")
    left = observed[channel * times + start - 1]
    right = observed[channel * times + start + MASK_POINTS]
    require(type(left) is int and type(right) is int, "missing_baseline_endpoint")
    return [left + (right - left) * (i + 1) / (MASK_POINTS + 1) for i in range(MASK_POINTS)]


def physical_microvolts(window):
    require(window.dtype == np.float32 and window.ndim == 3 and window.shape[1:] == (3, FS),
            "stored_array_not_certified_float32_C_3_200")
    # The stored value is V * 1e4, so microvolts = stored * 100, exactly once.
    values = window.astype(np.float64).reshape(window.shape[0], WINDOW_SAMPLES) * 100.0
    require(np.isfinite(values).all(), "nonfinite_certified_window")
    require(float(np.abs(values).max()) <= 100.0, "stored_window_exceeds_fixed_100uV_QC")
    require(float(values.std(axis=1, ddof=0).min()) >= 0.01,
            "stored_window_below_fixed_0_01uV_QC")
    return values


def make_examples(row, window, *, seed, cycle, prompt_builder, block_channels=CHANNEL_BLOCK):
    require(type(block_channels) is int and 1 <= block_channels <= CHANNEL_BLOCK,
            "block_channels_must_be_1_through_8")
    values = physical_microvolts(window)
    require(values.shape[0] == len(row["channels"]), "physical_channel_mapping_mismatch")
    first, last, slot = crop_bounds(row["id"], seed, cycle)
    quantized = np.rint(values[:, first:last]).astype(np.int16)
    require(int(quantized.min()) >= BOUNDS[0] and int(quantized.max()) <= BOUNDS[1],
            "quantized_value_out_of_declared_range_no_clipping_permitted")
    output = []
    for block, c0 in enumerate(range(0, len(row["channels"]), block_channels)):
        channels = row["channels"][c0:c0 + block_channels]
        matrix = quantized[c0:c0 + len(channels)]
        observed = matrix.reshape(-1).astype(int).tolist()
        indices = mask_indices(row["id"], seed, cycle, block, len(channels), last - first)
        target = [observed[i] for i in indices]
        for index in indices:
            observed[index] = None
        geometry = {"shape": [len(channels), last - first], "channel_names": channels,
                    "sample_rate_hz": FS, "quantum_uv": 1, "crop_start_sample": first}
        prompt = prompt_builder(observed, indices, QUANTIZATION_SCHEMA, geometry=geometry)
        interpolation = interpolation_baseline(observed, indices, geometry["shape"])
        block_tag = "" if block_channels == CHANNEL_BLOCK else f"-ch{block_channels}"
        example = {"id": f"{row['id']}:raw32-cycle{cycle}{block_tag}-block{block}",
                   "source_id": source_id(row), "canonical_subject_id": row["subject_group"],
                   "recording_id": row["source_recording_id"], "split": row["split"],
                   "task_kind": TASK_KIND,
                   "messages": [{"role": "user", "content": prompt},
                                {"role": "assistant", "content": json.dumps(target, separators=(",", ":"))}],
                   "reconstruction": {"observed": observed, "masked_indices": indices,
                                      "geometry": geometry},
                   "baselines": {"zero": [0] * MASK_POINTS, "within_channel_linear": interpolation},
                   "provenance": {"parent_window_id": row["id"], "source_sha256": row["source_sha256"],
                       "eeg_payload_sha256": row["eeg_payload_sha256"], "montage_key": row["montage_key"],
                       "reference": row["reference"], "source_window_samples": WINDOW_SAMPLES,
                       "source_window_seconds": WINDOW_SAMPLES / FS,
                       "exposed_time_samples": last - first, "exposed_time_seconds": (last - first) / FS,
                       "channel_indices": list(range(c0, c0 + len(channels))),
                       "max_channels_per_training_block": block_channels,
                       "crop_cycle": cycle, "crop_slot": slot, "cycle_period": CROP_SLOTS,
                       "mask_seed": seed, "quantization": "nearest integer uV; ties to even; no clipping",
                       "source_unit": "V_x_1e4", "unit_multiplier_to_uV": 100,
                       "target_kind": "measured_quantized_EEG_values", "online_harness_allowed": False}}
        output.append(example)
    return output


def check_tokens(tokenizer, row, max_length=1024):
    common = {"tokenize": True, "return_dict": False, "enable_thinking": False}
    prefix = list(tokenizer.apply_chat_template(row["messages"][:1], add_generation_prompt=True, **common))
    full = list(tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, **common))
    require(full[:len(prefix)] == prefix, "tokenizer_train_eval_prefix_mismatch")
    require(0 < len(prefix) < len(full) <= max_length,
            f"example_exceeds_{max_length}_no_truncation")
    require(tokenizer.decode(full[len(prefix):], skip_special_tokens=True).strip()
            == row["messages"][1]["content"], "tokenizer_completion_changed")
    return len(full), len(full) - len(prefix)


def load_tokenizer(path):
    from transformers import AutoTokenizer
    path = Path(path).resolve()
    require((path / "tokenizer.json").is_file() and (path / "tokenizer_config.json").is_file(),
            "missing_local_frozen_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    identity = {p.name: sha(p) for p in path.iterdir() if p.is_file()
                and p.suffix in (".json", ".jinja", ".model", ".txt")}
    return tokenizer, identity


def read_artifact(corpus, relative, verified):
    artifact = corpus["artifacts"][relative]
    path = contained(corpus["root"], relative)
    require(path.is_file() and path.stat().st_size == artifact["bytes"], "artifact_size_changed")
    if relative not in verified:
        require(sha(path) == artifact["sha256"], "artifact_sha_changed")
        verified.add(relative)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    require(list(array.shape) == artifact["shape"] and array.dtype == np.float32, "artifact_header_changed")
    return array


def metric_accumulator():
    return {"examples": 0, "min_tokens": None, "max_tokens": 0, "max_completion_tokens": 0,
            "zero_squared_error_sum": 0.0, "interpolation_squared_error_sum": 0.0,
            "target_values": 0, "visible_values": 0, "visible_zero_values": 0,
            "all_zero_visible_examples": 0}


def accumulate(metrics, row, lengths):
    total, completion = lengths
    metrics["examples"] += 1
    metrics["min_tokens"] = total if metrics["min_tokens"] is None else min(metrics["min_tokens"], total)
    metrics["max_tokens"] = max(metrics["max_tokens"], total)
    metrics["max_completion_tokens"] = max(metrics["max_completion_tokens"], completion)
    target = json.loads(row["messages"][1]["content"])
    metrics["target_values"] += len(target)
    metrics["zero_squared_error_sum"] += sum(x * x for x in target)
    metrics["interpolation_squared_error_sum"] += sum((a - b) ** 2 for a, b in zip(
        target, row["baselines"]["within_channel_linear"]))
    visible = [x for x in row["reconstruction"]["observed"] if x is not None]
    metrics["visible_values"] += len(visible)
    metrics["visible_zero_values"] += visible.count(0)
    metrics["all_zero_visible_examples"] += not any(visible)


def final_metrics(metrics):
    result = dict(metrics)
    if metrics["target_values"]:
        result["zero_mse_uV2"] = metrics["zero_squared_error_sum"] / metrics["target_values"]
        result["within_channel_linear_mse_uV2"] = metrics["interpolation_squared_error_sum"] / metrics["target_values"]
    result["metric_scope"] = "descriptive data baseline; not trained-model performance"
    return result


def coverage(rows, seed, cycle, block_channels=CHANNEL_BLOCK):
    # Sum each source window once, regardless of how many channel blocks it yields.
    exposed = sum(crop_bounds(r["id"], seed, cycle)[1] - crop_bounds(r["id"], seed, cycle)[0]
                  for r in rows) / FS
    return {"source_windows": len(rows), "source_window_seconds": len(rows) * WINDOW_SAMPLES / FS,
            "unique_exposed_time_seconds": exposed,
            "derived_examples": sum(math.ceil(len(r["channels"]) / block_channels) for r in rows),
            "max_channels_per_training_block": block_channels,
            "source_groups": len({r["subject_group"] for r in rows}),
            "time_coverage_note": "Channel blocks do not multiply seconds. A single cycle is a short crop, not all three seconds.",
            "cycle": cycle, "cycles_for_full_time_coverage": CROP_SLOTS}


def publish_source(output, name, outputs, metrics, corpus, tokenizer_identity, validation_sources,
                   seed, cycle, preparation_identity):
    """Close a source before subsequent source preparation starts."""
    directory = output / "data" / source_slug(name)
    receipt = {"schema": PREP_SCHEMA, "status": "passed", "source_id": name, "created_at": now(),
               "input_corpus": str(corpus["root"]), "metadata_sha256": corpus["metadata_sha256"],
               "tokenizer_files_sha256": tokenizer_identity,
               "preparation_identity": preparation_identity,
               "source_metrics": metrics[name], "development_metrics": metrics["development"],
               "source_QC_preserved": True, "amplitude_clipping": False,
               "unit_conversion": "V_x_1e4 * 100 = microvolts", "gradient_steps": 0,
               "test_review_signals_read": 0, "source_preparation_complete": True,
               "all_source_preparation_complete": False, "model_training_complete": False}
    receipt_path = directory / "qc_receipt.json"
    atomic_json(receipt_path, receipt)
    manifest = {"schema": DATA_SCHEMA, "task_kind": TASK_KIND,
                "inputs": {"train": outputs[name], "val": outputs["development"]},
                "qc_receipt": {"path": str(receipt_path), "sha256": sha(receipt_path)},
                "source_ids": sorted(set([name] + validation_sources)), "protected_subject_keys": [],
                "protection_scope": "Positive source fit/validation certificates and all inherited source/payload/group denies validated before signal reads",
                "quantization_schema": QUANTIZATION_SCHEMA, "quantization_bounds": BOUNDS,
                "max_channels_per_training_block": preparation_identity["max_channels_per_training_block"],
                "max_seq_length": preparation_identity["max_seq_length"],
                "preparation_identity": preparation_identity,
                "mask_repartition_note": "Crop offsets and mask seed/hash function are fixed; changing block geometry can change masked channel and time positions",
                "source_coverage": metrics[name]["coverage"],
                "validation_scope": "Inherited development groups from original corpus; not new-source generalization",
                "seed": seed, "crop_cycle": cycle, "preparation_code_sha256": sha(Path(__file__))}
    path = directory / "manifest.json"
    atomic_json(path, manifest)
    return {"source_id": name, "manifest_path": str(path), "sha256": sha(path),
            "prepared_examples": outputs[name]["count"], "training_completed": False}


def run(args):
    import evlm_mlx_train as trainer
    reconstruction_prompt, validate_reconstruction = trainer.reconstruction_prompt, trainer.validate_reconstruction
    block_channels = getattr(args, "block_channels", CHANNEL_BLOCK)
    max_tokens = getattr(args, "max_tokens", 1024)
    require(type(block_channels) is int and 1 <= block_channels <= CHANNEL_BLOCK,
            "block_channels_must_be_1_through_8")
    require(type(max_tokens) is int and 1 <= max_tokens <= 1024,
            "max_tokens_must_be_1_through_1024")
    output = Path(args.output).resolve()
    input_root = Path(args.corpus).resolve()
    require(output != input_root and not output.is_relative_to(input_root)
            and not input_root.is_relative_to(output), "output_overlaps_certified_input")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "prepare.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        phase = args.command
        state_path = output / (phase + "_state.json")
        require(not state_path.exists(), "immutable_preparation_attempt_already_exists")
        state = {"schema": PREP_SCHEMA, "status": "validating", "phase": phase,
                 "started_at": now(), "pid": os.getpid(), "gradient_steps": 0,
                 "test_review_signals_read": 0, "seed": args.seed, "cycle": args.cycle}
        state.update(max_channels_per_training_block=block_channels, max_seq_length=max_tokens)
        atomic_json(state_path, state)
        try:
            corpus = validate_corpus(input_root, args.expected_completion_sha256)
            tokenizer, tokenizer_identity = load_tokenizer(args.tokenizer)
            preparation_identity = {"seed": args.seed, "crop_cycle": args.cycle,
                                    "quantization_schema": QUANTIZATION_SCHEMA,
                                    "preparation_code_sha256": sha(Path(__file__)),
                                    "trainer_source_sha256": sha(Path(trainer.__file__)),
                                    "max_channels_per_training_block": block_channels,
                                    "max_seq_length": max_tokens}
            train = [r for r in corpus["rows"] if r["split"] == "train"]
            val = [r for r in corpus["rows"] if r["split"] == "val"]
            names = sorted({source_id(r) for r in train},
                           key=lambda s: (SOURCE_ORDER.index(s) if s in SOURCE_ORDER else len(SOURCE_ORDER), s))
            validation_sources = sorted({source_id(r) for r in val})
            validate_manifest = {"quantization_bounds": BOUNDS, "quantization_schema": QUANTIZATION_SCHEMA}
            verified = set()
            metrics = {}
            if phase == "probe":
                for name in names:
                    row = min((r for r in train if source_id(r) == name), key=lambda r: r["id"])
                    arr = read_artifact(corpus, row["eeg_path"], verified)
                    try:
                        examples = make_examples(row, arr[row["eeg_index"]], seed=args.seed,
                                                 cycle=args.cycle, prompt_builder=reconstruction_prompt,
                                                 block_channels=block_channels)
                        metric = metric_accumulator()
                        for example in examples:
                            validate_reconstruction(example, validate_manifest)
                            accumulate(metric, example, check_tokens(tokenizer, example, max_tokens))
                        metrics[name] = final_metrics(metric)
                    finally:
                        arr._mmap.close()
                receipt = {"schema": PREP_SCHEMA, "status": "passed", "representative_train_windows": len(names),
                           "sources": metrics, "metadata_sha256": corpus["metadata_sha256"],
                           "tokenizer_files_sha256": tokenizer_identity,
                           "preparation_identity": preparation_identity,
                           "artifacts_sha_verified": len(verified), "test_review_signals_read": 0,
                           "validation_signals_read": 0, "gradient_steps": 0,
                           "full_preparation_or_training_complete": False, "created_at": now()}
                atomic_json(output / "probe_receipt.json", receipt)
            else:
                require((output / "probe_receipt.json").is_file(), "representative_probe_required_first")
                probe = json.loads((output / "probe_receipt.json").read_text())
                require(probe.get("status") == "passed" and probe["metadata_sha256"] == corpus["metadata_sha256"]
                        and probe["tokenizer_files_sha256"] == tokenizer_identity
                        and probe.get("preparation_identity") == preparation_identity,
                        "probe_identity_mismatch")
                data_root = output / "data"
                require(not data_root.exists(), "refusing_existing_prepared_output")
                data_root.mkdir()
                outputs = {}
                queue = []
                all_sets = [("development", val)] + [(name, [r for r in train if source_id(r) == name]) for name in names]
                for name, selected in all_sets:
                    require(selected, "empty_fixed_source_or_development_split")
                    directory = data_root / ("development" if name == "development" else source_slug(name))
                    directory.mkdir()
                    target = directory / ("val.jsonl" if name == "development" else "train.jsonl")
                    metric = metric_accumulator()
                    state.update(status="preparing", active_source=name, completed_source_windows=0,
                                 generated_examples=0, updated_at=now())
                    atomic_json(state_path, state)
                    grouped = collections.defaultdict(list)
                    for row in selected:
                        grouped[row["eeg_path"]].append(row)
                    count = 0
                    with target.open("x") as f:
                        for rel in sorted(grouped):
                            arr = read_artifact(corpus, rel, verified)
                            try:
                                for row in sorted(grouped[rel], key=lambda r: r["eeg_index"]):
                                    for example in make_examples(row, arr[row["eeg_index"]], seed=args.seed,
                                                                 cycle=args.cycle, prompt_builder=reconstruction_prompt,
                                                                 block_channels=block_channels):
                                        validate_reconstruction(example, validate_manifest)
                                        accumulate(metric, example, check_tokens(tokenizer, example, max_tokens))
                                        f.write(json.dumps(example, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                                    count += 1
                                    if count % 1000 == 0:
                                        f.flush()
                                        state.update(completed_source_windows=count, generated_examples=metric["examples"], updated_at=now())
                                        atomic_json(state_path, state)
                            finally:
                                arr._mmap.close()
                        f.flush()
                        os.fsync(f.fileno())
                    require(count == len(selected), "source_window_coverage_incomplete")
                    outputs[name] = {"path": str(target), "sha256": sha(target), "count": metric["examples"]}
                    metrics[name] = {**final_metrics(metric), "coverage": coverage(selected, args.seed, args.cycle, block_channels)}
                    state.update(completed_source_windows=count, generated_examples=metric["examples"], updated_at=now())
                    atomic_json(state_path, state)
                    if name != "development":
                        queue.append(publish_source(output, name, outputs, metrics, corpus,
                                                    tokenizer_identity, validation_sources, args.seed, args.cycle,
                                                    preparation_identity))
                        atomic_json(output / "source_queue.json", {"schema": PREP_SCHEMA,
                            "sources": queue, "planned_source_order": names,
                            "all_source_preparation_complete": len(queue) == len(names),
                            "total_prepared_train_examples": sum(item["prepared_examples"] for item in queue),
                            "model_training_complete": False})
                receipt = {"schema": PREP_SCHEMA, "status": "passed", "created_at": now(),
                           "input_corpus": str(input_root), "metadata_sha256": corpus["metadata_sha256"],
                           "tokenizer_files_sha256": tokenizer_identity, "artifacts_sha_verified": len(verified),
                           "preparation_identity": preparation_identity,
                           "train_coverage": coverage(train, args.seed, args.cycle, block_channels),
                           "validation_coverage": coverage(val, args.seed, args.cycle, block_channels), "sources": metrics,
                           "source_QC_preserved": True, "amplitude_clipping": False,
                           "train_signal_unit_conversion": "V_x_1e4 * 100 = microvolts",
                           "quantization_schema": QUANTIZATION_SCHEMA, "quantization_bounds": BOUNDS,
                           "test_review_signals_read": 0, "gradient_steps": 0,
                           "data_preparation_complete": True, "model_training_complete": False}
                receipt_path = output / "qc_receipt.json"
                atomic_json(receipt_path, receipt)
                atomic_json(output / "source_queue.json", {"schema": PREP_SCHEMA, "sources": queue,
                            "all_source_preparation_complete": True,
                            "total_prepared_train_examples": sum(outputs[n]["count"] for n in names),
                            "source_windows": len(train), "model_training_complete": False})
            state.update(status="completed", ended_at=now(), artifacts_sha_verified=len(verified))
            atomic_json(state_path, state)
            print(json.dumps({"status": "completed", "phase": phase, "sources": len(names),
                              "output": str(output), "test_review_signals_read": 0, "gradient_steps": 0}))
            return 0
        except BaseException as exc:
            state.update(status="failed", ended_at=now(), error_type=type(exc).__name__, error=str(exc)[:400],
                         resume_allowed=False)
            atomic_json(state_path, state)
            raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("probe", "prepare"))
    p.add_argument("--corpus", required=True)
    p.add_argument("--expected-completion-sha256", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20261124)
    p.add_argument("--cycle", type=int, default=0)
    p.add_argument("--block-channels", type=int, default=CHANNEL_BLOCK)
    p.add_argument("--max-tokens", type=int, default=1024)
    args = p.parse_args(argv)
    require(0 <= args.cycle < CROP_SLOTS, "cycle_must_be_0_through_18")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
