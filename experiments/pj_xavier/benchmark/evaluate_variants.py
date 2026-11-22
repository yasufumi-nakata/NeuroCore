#!/usr/bin/env python3
"""Evaluate five input ablations of one frozen EEG+context checkpoint.

No optimizer, model selection, label generation, or OS action is performed.
Raw time shifts are new physical windows, never renamed existing test records.
Torch/numpy/MNE are imported lazily; importing this file does not touch a GPU.
"""
from __future__ import annotations

import argparse
import configparser
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.parse


VERSION = "pj-xavier-variant-evaluation-v1"
VARIANTS = ("fused", "context_only", "eeg_only", "eeg_shuffled", "eeg_time_shifted")
SHIFT_METHOD = "raw_recording_time_shift"
SHA = re.compile(r"^[0-9a-f]{64}$")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def write_jsonl(path, rows):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    os.replace(temp, path)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def inside(root, value):
    root = Path(root).resolve()
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("evidence path escapes the supplied artifact root")
    return path


def relative(root, path):
    return str(inside(root, path).relative_to(Path(root).resolve()))


def flat_identity(row):
    return (row.get("canonical_dataset_id", row["dataset"]),
            row.get("canonical_subject_id", row["subject"]), row["session"], row["recording_id"])


def nested_identity(row):
    return (row["source_dataset_id"], row.get("canonical_subject_id", row["participant_id"]), row["session_id"], row["recording_id"])


def integral_sample(seconds, sample_rate):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        raise ValueError("finite recording-relative time required")
    value = seconds * sample_rate
    index = round(value)
    if abs(value - index) > 1e-6:
        raise ValueError("timestamp does not identify an exact source sample")
    return index


class EvidenceCache:
    """Cache only immutable file identities and CPU readers, not inference state."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files, self.json, self.manifests, self.checkpoints, self.raws = {}, {}, {}, {}, {}

    def checked(self, path, expected=None):
        path = inside(self.root, path)
        if path not in self.files:
            self.files[path] = file_hash(path)
        actual = self.files[path]
        if expected is not None and (not SHA.fullmatch(str(expected)) or actual != expected):
            raise ValueError(f"evidence file hash mismatch: {path.name}")
        return path, actual

    def read_json(self, path, expected):
        path, _ = self.checked(path, expected)
        if path not in self.json:
            self.json[path] = json.loads(path.read_text())
        return self.json[path]

    def manifest(self, path, expected):
        path, _ = self.checked(path, expected)
        if path not in self.manifests:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            index = {row["id"]: row for row in rows}
            if len(index) != len(rows):
                raise ValueError("duplicate original manifest IDs")
            self.manifests[path] = index
        return path, self.manifests[path]

    def checkpoint(self, path, expected):
        path, _ = self.checked(path, expected)
        if path not in self.checkpoints:
            import torch
            self.checkpoints[path] = torch.load(path, map_location="cpu", weights_only=True)
        return self.checkpoints[path]

    def unchanged(self):
        for path, expected in self.files.items():
            if file_hash(path) != expected:
                raise ValueError(f"protected artifact changed during evaluation: {path.name}")


def brainvision_reader(cache, header_path, binary_path, marker_path):
    header = inside(cache.root, header_path)
    binary, marker = inside(cache.root, binary_path), inside(cache.root, marker_path)
    config = configparser.ConfigParser(interpolation=None, strict=False)
    text = header.read_text(encoding="utf-8-sig")
    config.read_string(text[text.index("["):])
    common = config["Common Infos"]
    actual_binary = inside(cache.root, header.parent / common["DataFile"].replace("\\", "/"))
    actual_marker = inside(cache.root, header.parent / common["MarkerFile"].replace("\\", "/"))
    if actual_binary != binary or actual_marker != marker:
        raise ValueError("BrainVision header does not reference the declared original binary/marker")
    if header not in cache.raws:
        import mne
        cache.raws[header] = mne.io.read_raw_brainvision(header, preload=False, verbose="ERROR")
    return cache.raws[header]


def raw_paths_for(row, raw_root, cache):
    fragment = urllib.parse.unquote(urllib.parse.urlsplit(row["eeg_source_uri"]).fragment)
    if not fragment or Path(fragment).is_absolute():
        raise ValueError("original raw EEG URI must identify a relative archive member")
    binary = inside(raw_root, fragment)
    header, marker = binary.with_suffix(".vhdr"), binary.with_suffix(".vmrk")
    for path in (binary, header, marker):
        cache.checked(path)
    cache.checked(binary, row["eeg_source_sha256"])
    return header, binary, marker


def checked_prepared_array(row, manifest_path, state, cache):
    import numpy as np
    key = row["eeg_path"]
    sealed = (state.get("eeg_artifacts") or {}).get(key)
    if not isinstance(sealed, dict) or not sealed.get("sha256"):
        raise ValueError("checkpoint does not seal this prepared EEG artifact")
    path = inside(cache.root, Path(manifest_path).parent / key)
    cache.checked(path, sealed["sha256"])
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    index = row.get("eeg_index")
    if index is not None:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("nonnegative integer prepared EEG index required")
        array = array[index]
    if array.ndim != 2 or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("finite float32 channels-by-time prepared EEG required")
    return array


def source_id(source):
    identity = {key: source[key] for key in ("raw_eeg_sha256", "sample_rate_hz", "start_sample", "stop_sample", "channels")}
    return "raw-window:" + canonical_hash(identity)


def verify_raw_shift_certificate(detail, original, artifact_root, checkpoint_sha, source_manifest_sha=None, cache=None):
    """Independently decode the cited raw window and match the saved array.

    The original prepared array is checked against hashes inside the actual
    checkpoint, and the original raw crop must reproduce it exactly. This is
    a physical-source control check, not verification of online preprocessing.
    """
    import numpy as np
    if not artifact_root:
        raise ValueError("artifact root required to verify actual raw extraction evidence")
    cache = cache or EvidenceCache(artifact_root)
    if not isinstance(detail, dict) or detail.get("method") != SHIFT_METHOD or detail.get("original_eeg_id") != original["record_id"]:
        raise ValueError("raw-shift method and original target identity required")
    source = detail.get("source", {})
    manifest_path, manifest = cache.manifest(source["manifest_path"], source["manifest_sha256"])
    if source_manifest_sha is not None and source["manifest_sha256"] != source_manifest_sha:
        raise ValueError("raw-shift source manifest differs from training provenance")
    row = manifest.get(original["record_id"])
    if not row or row.get("split") != "test" or flat_identity(row) != nested_identity(original):
        raise ValueError("raw-shift original record is not the same frozen held-out participant/session/recording")
    if list(flat_identity(row)) != source.get("recording_identity"):
        raise ValueError("raw-shift source participant/session/recording mismatch")
    if row["eeg_source_sha256"] != original["eeg"]["sha256"] or source["raw_eeg_sha256"] != row["eeg_source_sha256"]:
        raise ValueError("raw-shift original raw EEG identity mismatch")
    for name, time_key in (("window_start_s", "start_s"), ("window_end_s", "end_s")):
        if abs(row[name] - original["eeg"][time_key]) > 1e-9:
            raise ValueError("original window timestamps differ from frozen manifest")
    if source.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("raw-shift evaluated checkpoint mismatch")
    state = cache.checkpoint(source["checkpoint_path"], checkpoint_sha)
    if state.get("manifest_sha256") != source["manifest_sha256"]:
        raise ValueError("checkpoint was trained against a different source manifest")
    verification = cache.read_json(source["source_verification_path"], source["source_verification_sha256"])
    required = {f"bids_eeg_{row['site']}.tar.gz", f"bids_behavioral_{row['site']}.tar.gz"}
    complete = {item.get("filename") for item in verification.get("sources", [])
                if item.get("ledger_status") == "complete" and item.get("reverified_before_extraction") is True}
    if verification.get("verified") is not True or not required <= complete:
        raise ValueError("complete verified EEG/behavioral source archive evidence required")
    for field in ("raw_eeg", "raw_vhdr", "raw_vmrk"):
        cache.checked(source[field + "_path"], source[field + "_sha256"])
    raw = brainvision_reader(cache, source["raw_vhdr_path"], source["raw_eeg_path"], source["raw_vmrk_path"])
    rate = float(raw.info["sfreq"])
    if rate <= 0 or rate != row["sample_rate_hz"] or rate != source["sample_rate_hz"]:
        raise ValueError("raw-shift sample rate mismatch")
    channels = source.get("channels")
    if channels != row.get("channels") or not channels or any(channel not in raw.ch_names for channel in channels):
        raise ValueError("raw-shift physical channel montage mismatch")
    original_first, original_last = integral_sample(row["window_start_s"], rate), integral_sample(row["window_end_s"], rate)
    first, last = source.get("start_sample"), source.get("stop_sample")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (first, last)):
        raise ValueError("integer raw-shift sample interval required")
    if not 0 <= first < last <= raw.n_times or source.get("raw_n_samples") != raw.n_times:
        raise ValueError("raw-shift interval is outside actual recording bounds")
    shift = detail.get("shift_seconds")
    if not isinstance(shift, (int, float)) or isinstance(shift, bool) or not math.isfinite(shift) or abs(shift + 10.) > 1e-9:
        raise ValueError("the preregistered raw control requires exactly -10 seconds")
    shift_samples = integral_sample(shift, rate)
    if source.get("shift_samples") != shift_samples or first != original_first + shift_samples or last != original_last + shift_samples:
        raise ValueError("raw-shift sample interval does not implement the declared actual time shift")
    if abs(source.get("start_s", math.inf) - first / rate) > 1e-9 or abs(source.get("end_s", math.inf) - last / rate) > 1e-9:
        raise ValueError("raw-shift seconds differ from actual sample interval")
    if max(first, original_first) < min(last, original_last):
        raise ValueError("time-shift source overlaps its original EEG window")
    if detail.get("eeg_source_id") != source_id(source) or detail["eeg_source_id"] in manifest:
        raise ValueError("derived raw source must have its own content-derived ID, never an existing record ID")
    original_array = checked_prepared_array(row, manifest_path, state, cache)
    original_raw = raw.get_data(picks=channels, start=original_first, stop=original_last).astype(np.float32)
    if not np.array_equal(original_array, original_raw):
        raise ValueError("raw/header/channel interpretation does not reproduce the checkpoint-sealed original window")
    path, _ = cache.checked(source["extracted_array_path"], source["extracted_array_sha256"])
    extracted = np.load(path, allow_pickle=False, mmap_mode="r")
    decoded = raw.get_data(picks=channels, start=first, stop=last).astype(np.float32)
    if extracted.shape != original_array.shape or list(extracted.shape) != source.get("shape") or extracted.dtype != np.float32:
        raise ValueError("raw-shift array shape/dtype differs from original model input")
    if not np.isfinite(extracted).all() or not np.array_equal(extracted, decoded):
        raise ValueError("saved raw-shift array does not equal independently extracted real raw samples")
    return {"status": "passed", "source_id": detail["eeg_source_id"], "sample_interval": [first, last],
            "evidence_scope": "Actual raw/header/marker, frozen manifest and checkpoint-sealed input hashes checked; shifted raw crop decoded and compared exactly. Online causality is not established."}


def prepare_raw_shifts(rows, manifest_path, checkpoint_path, state, raw_root, output_dir, artifact_root, shift_seconds=-10.):
    import numpy as np
    cache = EvidenceCache(artifact_root)
    manifest_path, manifest_sha = cache.checked(manifest_path)
    checkpoint_path, checkpoint_sha = cache.checked(checkpoint_path)
    cache.checkpoints[checkpoint_path] = state
    certificates, blocked = {}, []
    destination = Path(output_dir) / "raw_shift_arrays"
    destination.mkdir(parents=True, exist_ok=True)
    try:
        verification_path, verification_sha = cache.checked(Path(raw_root) / "source_verification.json")
    except (OSError, ValueError) as exc:
        reason = f"raw source verification unavailable: {type(exc).__name__}: {exc}"
        report = {"status": "blocked", "expected_records": len(rows), "verified_records": 0,
                  "blocked_records": [{"record_id": row["id"], "reason": reason} for row in rows],
                  "shift_seconds": shift_seconds, "reason": reason}
        write_jsonl(Path(output_dir) / "raw_shift_sources.jsonl", [])
        write_json(Path(output_dir) / "raw_shift_preparation.json", report)
        return {}, report, cache
    for row in rows:
        try:
            header, binary, marker = raw_paths_for(row, raw_root, cache)
            raw = brainvision_reader(cache, header, binary, marker)
            rate = float(raw.info["sfreq"])
            first = integral_sample(row["window_start_s"], rate) + integral_sample(shift_seconds, rate)
            last = integral_sample(row["window_end_s"], rate) + integral_sample(shift_seconds, rate)
            if not 0 <= first < last <= raw.n_times:
                raise ValueError("requested -10-second source interval is outside the actual recording")
            channels = row["channels"]
            shifted = raw.get_data(picks=channels, start=first, stop=last).astype(np.float32)
            target = destination / (hashlib.sha256(row["id"].encode()).hexdigest() + ".npy")
            np.save(target, shifted, allow_pickle=False)
            source = {"manifest_path": relative(artifact_root, manifest_path), "manifest_sha256": manifest_sha,
                      "checkpoint_path": relative(artifact_root, checkpoint_path), "checkpoint_sha256": checkpoint_sha,
                      "source_verification_path": relative(artifact_root, verification_path), "source_verification_sha256": verification_sha,
                      "recording_identity": list(flat_identity(row)), "sample_rate_hz": rate,
                      "start_sample": first, "stop_sample": last, "start_s": first / rate, "end_s": last / rate,
                      "shift_samples": integral_sample(shift_seconds, rate), "raw_n_samples": int(raw.n_times), "channels": channels,
                      "shape": list(shifted.shape), "dtype": "float32",
                      "extracted_array_path": relative(artifact_root, target), "extracted_array_sha256": file_hash(target),
                      "source_preprocessing_unchanged": True, "online_causality_verified": False,
                      "assumed_filter_lookahead_s": row.get("assumed_filter_lookahead_s"),
                      "feature_available_time_s_under_source_assumption": last / rate + row.get("assumed_filter_lookahead_s", 0),
                      "preprocessing_note": "The released BrainVision data were already processed offline; these are original distributed source samples, not unfiltered XDF."}
            for name, path in (("raw_eeg", binary), ("raw_vhdr", header), ("raw_vmrk", marker)):
                source[name + "_path"], source[name + "_sha256"] = relative(artifact_root, path), cache.checked(path)[1]
            detail = {"method": SHIFT_METHOD, "original_eeg_id": row["id"], "shift_seconds": shift_seconds,
                      "source": source, "eeg_source_id": source_id(source)}
            nested = {"record_id": row["id"], "source_dataset_id": flat_identity(row)[0], "participant_id": flat_identity(row)[1],
                      "session_id": row["session"], "recording_id": row["recording_id"],
                      "eeg": {"sha256": row["eeg_source_sha256"], "start_s": row["window_start_s"], "end_s": row["window_end_s"]}}
            verify_raw_shift_certificate(detail, nested, artifact_root, checkpoint_sha, manifest_sha, cache)
            certificates[row["id"]] = detail
        except (OSError, ValueError, KeyError, IndexError, ImportError) as exc:
            blocked.append({"record_id": row["id"], "reason": f"{type(exc).__name__}: {exc}"})
    write_jsonl(Path(output_dir) / "raw_shift_sources.jsonl", [{"record_id": rid, "perturbation": detail} for rid, detail in certificates.items()])
    report = {"status": "ready" if len(certificates) == len(rows) and not blocked else "blocked", "expected_records": len(rows),
              "verified_records": len(certificates), "blocked_records": blocked, "shift_seconds": shift_seconds,
              "reason": None if not blocked else "The complete frozen test set cannot be supplied; no partial-set time-shift score will be reported."}
    write_json(Path(output_dir) / "raw_shift_preparation.json", report)
    return certificates, report, cache


def verify_membership(rows, membership, manifest_sha, state, expected_count):
    if state.get("manifest_sha256") != manifest_sha or membership.get("manifest_sha256") != manifest_sha:
        raise ValueError("checkpoint/membership does not match original manifest bytes")
    if state.get("config", {}).get("mode") != "fused":
        raise ValueError("this runner evaluates input ablations of a fused checkpoint")
    if membership.get("model_selection_partition") != "validation" or membership.get("test_used_for_tuning") is not False:
        raise ValueError("validation-only checkpoint selection provenance required")
    for field, part in (("training_ids", "train"), ("validation_ids", "val"), ("test_ids", "test")):
        expected = {row["id"] for row in rows if row["split"] == part}
        actual = membership.get(field, [])
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f"frozen {field} differs from manifest")
    if membership.get("training_ids_sha256") != hashlib.sha256("\n".join(sorted(membership["training_ids"])).encode()).hexdigest():
        raise ValueError("training membership checksum mismatch")
    test = [row for row in rows if row["split"] == "test"]
    if len(test) != expected_count or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("frozen test record count or unique identities differ from requested contract")
    return test


def restore_loaded_state(api, state, model):
    import torch
    if state.get("schema") != api.SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    model.connector.load_state_dict(state["connector"])
    params = dict(model.base.named_parameters())
    with torch.no_grad():
        for key, value in state["trainable_base"].items():
            if key not in params:
                raise ValueError(f"missing trained adapter parameter: {key}")
            params[key].copy_(value)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


class ConditionStore:
    def __init__(self, base, variant, total, output, certificates=None, root=None):
        self.base, self.variant, self.total, self.output = base, variant, total, output
        self.certificates, self.root, self.count = certificates, root, 0

    def get(self, row):
        self.count += 1
        if self.count % 100 == 0 or self.count == 1:
            progress = {"condition": self.variant, "processed_input_records": self.count, "total_records": self.total, "updated_at": utc_now(), "status": "running"}
            write_json(Path(self.output) / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        if self.certificates is None:
            return self.base.get(row)
        detail = self.certificates[row["id"]]
        source = dict(row)
        source["eeg_path"] = str(inside(self.root, detail["source"]["extracted_array_path"]))
        source.pop("eeg_index", None)
        source.pop("eeg_key", None)
        value = self.base.get(source)
        # EEGStore copies the array before normalization. Do not retain one mmap
        # descriptor per derived window (1956 files can exceed the OS soft limit).
        cache = getattr(self.base, "cache", None)
        if isinstance(cache, dict):
            cache.pop(source["eeg_path"], None)
        return value


def evaluate_conditions(api, model, tokenizer, test_rows, classes, base_store, config, device, checkpoint_sha,
                        output, certificates, shift_preparation, artifact_root):
    conditions, all_predictions = {}, []
    expected_ids = [row["id"] for row in test_rows]
    for variant in VARIANTS:
        if variant == "eeg_time_shifted" and shift_preparation["status"] != "ready":
            conditions[variant] = {"status": "blocked", "reason": shift_preparation["reason"], "preparation": shift_preparation}
            continue
        variant_config = dict(config)
        variant_config["mode"] = variant if variant in ("context_only", "eeg_only") else "fused"
        condition = "shuffled_eeg" if variant == "eeg_shuffled" else "normal"
        store = ConditionStore(base_store, variant, len(test_rows), output,
                               certificates if variant == "eeg_time_shifted" else None, artifact_root)
        print(json.dumps({"variant": variant, "status": "started", "records": len(test_rows)}), flush=True)
        try:
            metrics, predictions = api.evaluate(model, tokenizer, test_rows, classes, store, variant_config, device,
                                                checkpoint_sha, condition=condition, include_truth=True)
            if metrics.get("status") != "measured":
                conditions[variant] = {"status": "blocked", "reason": metrics.get("reason", "model evaluator did not produce a measured result")}
                continue
            if [row["id"] for row in predictions] != expected_ids:
                raise ValueError("condition predictions do not cover the identical ordered frozen test IDs")
            for prediction in predictions:
                prediction.update(variant=variant, checkpoint_sha256=checkpoint_sha, action_executed=False,
                                  ablation_scope="same frozen fused checkpoint; input removal/correspondence control, not a separately trained baseline")
                if variant == "eeg_time_shifted":
                    detail = certificates[prediction["id"]]
                    prediction["perturbation"] = detail
                    prediction["eeg_source_id"] = detail["eeg_source_id"]
            path = Path(output) / f"{variant}_predictions.jsonl"
            write_jsonl(path, predictions)
            conditions[variant] = {**metrics, "variant": variant, "prediction_path": str(path.resolve()), "prediction_sha256": file_hash(path),
                                   "checkpoint_sha256": checkpoint_sha, "frozen_test_ids_sha256": canonical_hash(expected_ids),
                                   "ablation_scope": "same frozen fused checkpoint; no condition-specific optimization or model selection"}
            all_predictions.extend(predictions)
        except (RuntimeError, ValueError, OSError) as exc:
            conditions[variant] = {"status": "blocked", "reason": f"{type(exc).__name__}: {exc}", "partial_predictions_used": False}
        write_json(Path(output) / "condition_progress.json", {"checkpoint_sha256": checkpoint_sha, "conditions": conditions, "status": "running"})
        print(json.dumps({"variant": variant, "status": conditions[variant]["status"]}), flush=True)
    write_jsonl(Path(output) / "predictions.jsonl", all_predictions)
    return conditions


def finish_report(report, path, receipt_path):
    names = report.get("variant_names")
    conditions = report.get("conditions", {})
    complete = names == list(VARIANTS) and set(conditions) == set(VARIANTS) and all(
        row.get("status") == "measured" or (row.get("status") == "blocked" and row.get("reason")) for row in conditions.values())
    if not complete or not SHA.fullmatch(str(report.get("checkpoint_sha256", ""))):
        raise ValueError("all five conditions must have measurements or explicit blocking reasons before completion")
    for result in conditions.values():
        if result["status"] == "measured":
            if result.get("checkpoint_sha256") != report["checkpoint_sha256"]:
                raise ValueError("measured condition checkpoint differs from completion report")
            if result.get("n") != report.get("test_record_count") or result.get("frozen_test_ids_sha256") != report.get("frozen_test_ids_sha256"):
                raise ValueError("measured condition does not cover the frozen final test set")
            if not result.get("prediction_path") or file_hash(result["prediction_path"]) != result.get("prediction_sha256"):
                raise ValueError("measured condition prediction bytes are missing or changed")
    report.update(status="completed", autonomous_agent_validated=False, held_out_replay_validated=False, completed_at=utc_now())
    write_json(path, report)
    receipt = {"status": "completed", "exit_code": 0, "checkpoint_sha256": report["checkpoint_sha256"],
               "variant_names": list(VARIANTS), "report_path": str(Path(path).resolve()), "report_sha256": file_hash(path),
               "completed_at": utc_now(), "interpretation": "Evaluation execution reached a final measurement/blocked outcome for every condition; this is not model or autonomous-goal acceptance."}
    write_json(receipt_path, receipt)
    return receipt


def load_api(path):
    spec = importlib.util.spec_from_file_location("xavier_eeg_model_for_variants", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "checkpoint", "raw-bids-root", "artifact-root", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--model-module", default=str(Path(__file__).resolve().parent.parent / "model/eeg_llm.py"))
    parser.add_argument("--training-membership")
    parser.add_argument("--completion-receipt")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--expected-test-records", type=int, default=1956)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--memory-budget-gib", type=float)
    parser.add_argument("--candidate-batch-size", type=int)
    args = parser.parse_args(argv)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory is nonempty; preserve the frozen evaluation and use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = Path(args.completion_receipt) if args.completion_receipt else Path(args.checkpoint).parent / "variant_evaluation_completion.json"
    if receipt_path.exists():
        parser.error("completion receipt already exists; do not silently repeat the frozen final test")
    write_json(output / "started.json", {"started_at": utc_now(), "configuration": vars(args)})
    started = time.monotonic()
    checkpoint_sha = None
    try:
        import torch
        model_module_sha = file_hash(args.model_module)
        if args.expected_model_sha256 and model_module_sha != args.expected_model_sha256:
            raise ValueError("model module differs from the explicitly expected frozen implementation")
        api = load_api(args.model_module)
        checkpoint = inside(args.artifact_root, args.checkpoint)
        manifest = inside(args.artifact_root, args.manifest)
        checkpoint_sha, manifest_sha = file_hash(checkpoint), file_hash(manifest)
        if args.expected_manifest_sha256 and args.expected_manifest_sha256 != manifest_sha:
            raise ValueError("manifest differs from the explicitly expected frozen source")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        rows = api.read_rows(manifest)
        membership_path = Path(args.training_membership) if args.training_membership else checkpoint.parent / "training_membership.json"
        membership_sha = file_hash(membership_path)
        membership = json.loads(membership_path.read_text())
        test_rows = verify_membership(rows, membership, manifest_sha, state, args.expected_test_records)
        config = dict(state["config"])
        if not re.fullmatch(r"[0-9a-f]{40}", str(config.get("revision", ""))):
            raise ValueError("frozen immutable backbone revision required")
        api.validate_rows(rows, config["split_policy"], "fused")
        if api.eeg_artifact_manifest(rows) != state.get("eeg_artifacts"):
            raise ValueError("prepared EEG bytes differ from checkpoint-sealed artifacts")
        # Raw helpers use the original portable path keys, not internal absolute paths.
        portable_test = [{**row, "eeg_path": row.get("_manifest_eeg_path", row["eeg_path"])} for row in test_rows]
        certificates, preparation, evidence_cache = prepare_raw_shifts(portable_test, manifest, checkpoint, state,
                                                                     args.raw_bids_root, output, args.artifact_root)
        if args.memory_budget_gib is not None:
            config["memory_budget_gib"] = args.memory_budget_gib
        if args.candidate_batch_size is not None:
            if args.candidate_batch_size < 1:
                raise ValueError("candidate batch size must be positive")
            config["candidate_batch_size"] = args.candidate_batch_size
        config["device"] = args.device
        device = api.select_device(args.device)
        allocator = api.apply_allocator_limit(device, config["memory_budget_gib"])
        # One backbone/model load; no optimizer or weight-changing path is used.
        base, tokenizer, backbone = api.load_backbone(config, device)
        model = api.EEGCausalModel(base, config["channels"], config["prefix_tokens"], config["encoder_width"]).to(device)
        api.attach_lora(model, config["lora_rank"])
        restore_loaded_state(api, state, model)
        conditions = evaluate_conditions(api, model, tokenizer, test_rows, state["classes"], api.EEGStore(config["samples"]), config,
                                         device, checkpoint_sha, output, certificates, preparation, args.artifact_root)
        evidence_cache.unchanged()
        if file_hash(membership_path) != membership_sha:
            raise ValueError("training membership changed during final evaluation")
        if file_hash(args.model_module) != model_module_sha:
            raise ValueError("model module changed during final evaluation")
        report = {"schema_version": VERSION, "checkpoint_sha256": checkpoint_sha, "source_manifest_sha256": manifest_sha,
                  "variant_names": list(VARIANTS), "conditions": conditions, "test_record_count": len(test_rows),
                  "frozen_test_ids_sha256": canonical_hash([row["id"] for row in test_rows]), "classes": state["classes"],
                  "elapsed_seconds": time.monotonic() - started, "memory": api.memory_sample(device), "hardware": {**api.hardware(device), **backbone},
                  "allocator_limit": allocator, "runtime_configuration": config, "model_module_sha256": model_module_sha,
                  "evaluator_sha256": file_hash(__file__),
                  "raw_shift_preparation": preparation, "ablation_scope": "same selected frozen fused checkpoint; no independently trained unimodal baselines",
                  "model_selection_changed": False, "weights_updated": False, "test_used_for_tuning": False, "os_actions_executed": 0,
                  "rare_action_limit": {"label": "ai_prompt_submit", "train": sum(r["split"] == "train" and r["label"] == "ai_prompt_submit" for r in rows),
                                        "validation": sum(r["split"] == "val" and r["label"] == "ai_prompt_submit" for r in rows),
                                        "test": sum(r["label"] == "ai_prompt_submit" for r in test_rows),
                                        "agent_request_capability_validated": False,
                                        "interpretation": "Very sparse observed submit events cannot establish either presence or absence of general agent-request ability."},
                  "goal_validation_blockers": ["online preprocessing causality remains unverified", "no independent context-conditioned harness task completion",
                                               "no independently annotated idle exposure", "candidate latency excludes preprocessing and harness execution",
                                               "rare observed submit event is not high-level intent gold"]}
        receipt = finish_report(report, output / "variant_evaluation_report.json", receipt_path)
        print(json.dumps(receipt), flush=True)
        return 0
    except Exception as exc:
        report = {"schema_version": VERSION, "status": "failed", "checkpoint_sha256": checkpoint_sha,
                  "reason": f"{type(exc).__name__}: {exc}", "autonomous_agent_validated": False, "held_out_replay_validated": False}
        write_json(output / "variant_evaluation_report.json", report)
        write_json(receipt_path, {"status": "failed", "exit_code": 1, "checkpoint_sha256": checkpoint_sha,
                                 "report_path": str(output / "variant_evaluation_report.json"), "report_sha256": file_hash(output / "variant_evaluation_report.json"),
                                 "reason": report["reason"]})
        print(json.dumps(report), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
