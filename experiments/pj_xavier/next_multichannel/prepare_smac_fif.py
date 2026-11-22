#!/usr/bin/env python3
"""Prepare a NEW smac-only, offline SSL corpus from verified 16-channel FIF.

The first supported release is Zenodo 806023. Existing metadata-only splits are
preserved; test/review records are never opened. A source is admitted only after
its file and calibrated EEG payload hashes agree with the retained audit and
after the fixed QC below. Source preprocessing history is not certified causal:
only the additional filter/resampling here is causal. No action labels are read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy import signal

FS_OUT = 200
PATCHES = 3
WARMUP_S = 2
PEAK_V = 100e-6
MIN_STD_V = .01e-6
SOURCE_ID = "806023"
HASH_KEYS = ("source_sha256", "source_file_sha256", "source_eeg_sha256",
             "waveform_sha256")
PAYLOAD_KEYS = ("eeg_payload_sha256", "EEG_payload_sha256",
                "decoded_payload_sha256", "decoded_EEG_V_sha256")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n")


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def get_hash(row, names):
    return next((row[k] for k in names if isinstance(row.get(k), str)
                 and len(row[k]) == 64), None)


def read_denies(paths):
    """Read known prior exclusions; do not infer eligibility from absent files."""
    sources, payloads, groups = set(), set(), set()
    inputs = {}

    def collect(obj, inherited=False):
        if isinstance(obj, list):
            for item in obj:
                collect(item, inherited)
            return
        if not isinstance(obj, dict):
            return
        split = obj.get("split", obj.get("ssl_split", obj.get("original_partition")))
        hold = inherited or split in {"val", "test", "review", "review_hold"}
        status = str(obj.get("status", "")).lower()
        hold = hold or "review" in status or "excluded" in status
        if hold:
            for key in HASH_KEYS:
                if isinstance(obj.get(key), str):
                    sources.add(obj[key])
            for key in PAYLOAD_KEYS:
                if isinstance(obj.get(key), str):
                    payloads.add(obj[key])
            for key in ("subject_group", "participant_group"):
                if isinstance(obj.get(key), str):
                    groups.add(obj[key])
        for key, value in obj.items():
            if key in {"excluded_source_sha256", "denied_source_sha256"}:
                require(isinstance(value, list), "source deny list must be an array")
                sources.update(value)
            elif key in {"excluded_payload_sha256", "denied_payload_sha256"}:
                require(isinstance(value, list), "payload deny list must be an array")
                payloads.update(value)
            elif key == "sources" and isinstance(value, dict):
                for source_hash, item in value.items():
                    if isinstance(item, dict):
                        collect({**item, "source_sha256": source_hash}, hold)
            elif isinstance(value, (dict, list)):
                collect(value, hold or "exclusion" in key or key.startswith("denied"))

    for p in paths:
        p = Path(p)
        inputs[str(p)] = sha(p)
        if p.suffix == ".jsonl":
            for row in rows(p):
                collect(row)
        else:
            collect(json.loads(p.read_text()))
    require(sources or payloads, "No prior heldout identities were loaded")
    return sources, payloads, groups, inputs


def register_sources(ledger, metadata, deny_sources, deny_payloads, deny_groups):
    """Preserve the prior split and reject any known cross-fold exact identity."""
    chosen = [r for r in ledger if r.get("dataset_key") == "zenodo-806023-806023"]
    require(len(chosen) == 8, "806023 source ledger must contain all eight recordings")
    prior = {r.get("source_file_sha256"): r for r in metadata
             if str(r.get("source_group", "")).startswith("compact14_16|zenodo:806023:")}
    require(len(prior) == 8, "All eight frozen metadata split records are required")
    planned = []
    seen_payloads = {}
    for source in sorted(chosen, key=lambda r: r["relative_path"]):
        source_hash = get_hash(source, HASH_KEYS)
        payload_hash = get_hash(source, PAYLOAD_KEYS)
        require(source_hash and payload_hash, "Retained source/payload hashes required")
        m = prior.get(source_hash)
        require(m is not None and m.get("eeg_payload_sha256") == payload_hash,
                "Source ledger and prior admission disagree")
        require(m["source_member_or_path"] == source["relative_path"], "Source path mismatch")
        split = m["metadata_only_proposed_split"]
        require(split in {"train", "val", "test"}, "Unknown frozen split")
        require(source.get("eeg_channel_count") == 16, "Only the audited 16 EEG channels are supported")
        require(source.get("unit") == "V_as_decoded_using_FIF_header_calibration",
                "Retained physical-V calibration is not established")
        require(payload_hash not in seen_payloads or seen_payloads[payload_hash] == split,
                "Same calibrated EEG payload crosses frozen folds")
        seen_payloads[payload_hash] = split
        group = source["subject_group"]
        require(m["source_group"] == "compact14_16|" + group, "Group identity mismatch")
        if split == "test":
            state, reason = "excluded", "frozen_test_not_opened"
        elif source_hash in deny_sources or payload_hash in deny_payloads or group in deny_groups:
            state, reason = "excluded", "prior_heldout_or_review_identity"
        elif any("author QC flag" in str(t) for t in m.get("held_reasons", [])):
            state, reason = "excluded", "author_QC_review_hold"
        else:
            state, reason = "candidate", "requires_current_file_payload_and_fixed_QC"
        planned.append({"source": source, "metadata": m, "source_sha256": source_hash,
                        "eeg_payload_sha256": payload_hash, "split": split,
                        "subject_group": group, "status": state, "reason": reason})
    require(Counter(x["split"] for x in planned) == {"train": 4, "val": 2, "test": 2},
            "Frozen 806023 split changed")
    return planned


def causal_transform(x, native_fs):
    """Additional causal SOS and one-native-sample delayed interpolation.

    This function makes no assertion about operations applied by the publisher.
    Each output sample depends only on source samples at or before its cutoff.
    """
    x = np.asarray(x, dtype=np.float64)
    require(x.ndim == 2 and x.shape[0] == 16, "Expected [16,N] calibrated volts")
    require(np.isfinite(x).all(), "Nonfinite source; gaps require separate records")
    require(float(np.max(np.abs(x))) <= .1, "Source magnitude inconsistent with this V policy")
    require(native_fs == 512, "806023 must have its audited native 512 Hz")
    sos = signal.butter(4, (3, 40), btype="bandpass", fs=native_fs, output="sos")
    y = signal.sosfilt(sos, x, axis=1)
    anti = signal.butter(6, 80, btype="lowpass", fs=native_fs, output="sos")
    y = signal.sosfilt(anti, y, axis=1)
    first = math.ceil(WARMUP_S * FS_OUT)
    last = ((x.shape[1] - 1) * FS_OUT) // native_fs
    js = np.arange(first, last + 1, dtype=np.int64)
    n = (js * native_fs) // FS_OUT
    a = ((js * native_fs) % FS_OUT).astype(np.float64) / FS_OUT
    out = y[:, n - 1] * (1 - a) + y[:, n] * a
    return out.astype(np.float32), js, n


def make_windows(x, native_fs):
    y, js, cutoffs = causal_transform(x, native_fs)
    accepted, provenance = [], []
    rejected = Counter()
    width = PATCHES * FS_OUT
    for start in range(0, len(js) - width + 1, width):
        window = y[:, start:start + width]
        if not np.isfinite(window).all():
            rejected["nonfinite"] += 1
        elif np.max(np.abs(window)) > PEAK_V:
            rejected["peak_above_fixed_100uV"] += 1
        elif np.any(window.astype(np.float64).std(axis=1, ddof=0) < MIN_STD_V):
            rejected["flat_below_fixed_0_01uV"] += 1
        else:
            accepted.append((window * 1e4).reshape(16, PATCHES, FS_OUT).astype(np.float32))
            provenance.append({"window_start_s": int(js[start]) / FS_OUT,
                               "window_end_s": (int(js[start + width - 1]) + 1) / FS_OUT,
                               "latest_native_source_index": int(cutoffs[start + width - 1]),
                               "feature_available_native_s": int(cutoffs[start + width - 1]) / native_fs})
    arrays = np.stack(accepted) if accepted else np.empty((0, 16, PATCHES, FS_OUT), np.float32)
    return arrays, provenance, dict(rejected)


def load_verified_fif(path, retained):
    import mne
    require(sha(path) == retained["source_sha256"], "FIF source SHA256 mismatch")
    raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    try:
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        names = [raw.ch_names[i] for i in picks]
        require(names == retained["eeg_channels"], "EEG channel order differs from audit")
        require(len(picks) == 16 and raw.info["sfreq"] == 512, "FIF shape/rate differs from audit")
        require(raw.n_times == retained["sample_count_per_EEG_channel"], "FIF sample count differs")
        calibration = []
        for i in picks:
            ch = raw.info["chs"][i]
            require(int(ch["unit"]) == 107 and int(ch["unit_mul"]) == 0,
                    "EEG channels must be declared in base physical volts")
            require(np.isfinite(ch["cal"]) and ch["cal"] > 0
                    and np.isfinite(ch["range"]) and ch["range"] > 0,
                    "Invalid FIF physical calibration")
            calibration.append({k: ch[k] for k in ("cal", "range", "unit", "unit_mul")})
        x = raw.get_data(picks=picks)
        # Match the retained independent audit: sample-major little-endian f64 V.
        payload = hashlib.sha256(np.asarray(x.T, dtype="<f8", order="C").tobytes()).hexdigest()
        require(payload == retained["decoded_payload_sha256"], "FIF calibrated EEG payload mismatch")
        require(sha(path) == retained["source_sha256"], "FIF source changed during read")
        evidence = {"header_unit": "V", "physical_unit_verified": True,
                    "verification_scope": "FIF-declared physical V and retained full calibrated payload; not independent amplifier calibration",
                    "publisher_proc_history_entries": len(raw.info.get("proc_history", [])),
                    "source_causality_verified": False, "online_harness_allowed": False,
                    "calibration_sha256": hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()}
        return x, names, evidence
    finally:
        raw.close()


def prepare(args):
    out = Path(args.output).expanduser().absolute()
    require(not out.exists(), "Use a new immutable output directory for each corpus")
    input_root = Path(args.raw_root).expanduser().resolve()
    ledger, metadata = rows(args.source_ledger), rows(args.metadata_admission)
    ds, dp, dg, deny_inputs = read_denies(args.deny_json)
    planned = register_sources(ledger, metadata, ds, dp, dg)
    # Add new frozen tests before any waveform reads. Validation is not a test.
    ds.update(p["source_sha256"] for p in planned if p["status"] == "excluded")
    dp.update(p["eeg_payload_sha256"] for p in planned if p["status"] == "excluded")
    # Native full-record decoding is bounded to one <128 MiB record at a time.
    require(shutil.disk_usage(out.parent if out.parent.exists() else input_root).free
            >= int(args.reserve_gib * 1024**3) + 512 * 1024**2,
            "Insufficient smac reserve for this bounded preparation")
    protocol = {"schema": "xavier-smac-fif-offline-ssl-protocol-v1", "source_id": "zenodo:806023",
                "created_epoch": time.time(), "code_sha256": sha(__file__),
                "source_ledger_sha256": sha(args.source_ledger),
                "metadata_admission_sha256": sha(args.metadata_admission),
                "prior_deny_input_hashes": deny_inputs, "split": "existing_metadata_train4_val2_test2_unchanged",
                "QC": {"maximum_absolute_V": PEAK_V, "minimum_channel_std_V": MIN_STD_V,
                       "all_finite": True, "threshold_selection": "fixed_before_current_signal_read; not tuned on validation/test",
                       "comparison_space": "after causal preprocessing in physical V", "clipping_or_rescaling": False},
                "preprocessing": {"bandpass": "Butterworth4 SOS 3-40Hz causal zero-initial-state",
                                  "anti_alias": "Butterworth6 SOS 80Hz causal zero-initial-state",
                                  "resampling": "one-native-sample delayed linear interpolation to 200Hz",
                                  "warmup_seconds": WARMUP_S, "window_seconds": PATCHES, "nonoverlapping": True,
                                  "input": "header-calibrated physical V", "output": "float32 V_x_1e4 [N,16,3,200]"},
                "scope": "offline masked EEG reconstruction only; source preprocessing causality not certified",
                "online_harness_allowed": False, "action_labels_read": False, "test_signals_opened": 0,
                "native_record_working_memory_upper_bound_bytes": 128 * 1024**2}
    out.mkdir(parents=True)
    dump(out / "protocol.json", protocol)
    (out / "waveforms").mkdir()
    manifest, certificates, source_map, artifact_map = [], [], {}, {}
    payload_seen = set()
    totals = Counter()
    for p in planned:
        retained = p["source"]
        short = {k: v for k, v in p.items() if k not in {"source", "metadata"}}
        if p["status"] == "excluded":
            certificates.append({**short, "signal_opened": False})
            continue
        require(p["eeg_payload_sha256"] not in payload_seen,
                "Duplicate EEG record detected; must-link review required")
        payload_seen.add(p["eeg_payload_sha256"])
        relative = Path(retained["relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe source path")
        file = (input_root / relative).resolve()
        require(input_root in file.parents and not file.is_symlink(), "Source escaped raw directory")
        x, channels, proof = load_verified_fif(file, retained)
        arrays, windows, rejected = make_windows(x, 512)
        del x
        require(len(arrays), "No accepted windows in source; keep empty source excluded")
        reference = "publisher_FIF_reference_unspecified_no_rereference"
        montage_key = hashlib.sha256(json.dumps([channels, reference, 512, "zenodo:806023", "physical_V"], sort_keys=True).encode()).hexdigest()
        rel = "waveforms/" + p["source_sha256"] + ".npy"
        np.save(out / rel, arrays, allow_pickle=False)
        artifact_map[rel] = {"shape": list(arrays.shape), "dtype": "float32", "sha256": sha(out / rel), "bytes": (out / rel).stat().st_size}
        entry = {"split": p["split"], "subject_group": p["subject_group"],
                 "eeg_payload_sha256": p["eeg_payload_sha256"], "montage_key": montage_key,
                 "physical_unit_verified": True, "fit_allowed": p["split"] == "train",
                 "validation_allowed": p["split"] == "val", "online_harness_allowed": False,
                 "source_recording_id": "zenodo:806023:" + relative.name,
                 "unit_verification_scope": proof["verification_scope"]}
        source_map[p["source_sha256"]] = entry
        for index, window in enumerate(windows):
            row = {"id": p["source_sha256"] + ":" + str(index), "split": p["split"],
                   "subject_group": p["subject_group"], "source_sha256": p["source_sha256"],
                   "eeg_payload_sha256": p["eeg_payload_sha256"],
                   "source_recording_id": entry["source_recording_id"], **window,
                   "channels": channels, "sample_rate_hz": FS_OUT, "unit": "V_x_1e4",
                   "eeg_path": rel, "eeg_index": index, "montage_key": montage_key,
                   "reference": reference, "native_sample_rate_hz": 512,
                   "physical_unit_verified": True, "source_causality": "unverified_offline_preprocessing",
                   "online_harness_allowed": False}
            manifest.append(row)
        totals[p["split"]] += len(windows)
        certificates.append({**short, **proof, "signal_opened": True, "status": "admitted_offline_SSL",
                             "accepted_windows": len(windows), "rejected_windows": rejected,
                             "prepared_seconds": len(windows) * PATCHES,
                             "output_sha256": artifact_map[rel]["sha256"]})
    require(totals["train"] > 0 and totals["val"] > 0, "Both train and validation are required")
    require(not ({r["subject_group"] for r in manifest if r["split"] == "train"}
                 & {r["subject_group"] for r in manifest if r["split"] == "val"}), "Subject leakage")
    (out / "manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in manifest))
    dump(out / "artifacts.json", artifact_map)
    dump(out / "source_admission.json", {"schema": "xavier-smac-verified-offline-SSL-admission-v1",
         "sources": source_map, "excluded_source_sha256": sorted(ds), "excluded_payload_sha256": sorted(dp),
         "protocol_sha256": sha(out / "protocol.json"), "online_harness_allowed": False,
         "deny_scope": "exact retained file/calibrated-payload identities; near duplicates and unknown human overlap not certified"})
    dump(out / "source_certificates.json", certificates)
    summary = {"status": "prepared_verified_new_offline_SSL_corpus", "windows": dict(totals),
               "nonoverlapping_seconds": {k: v * PATCHES for k, v in totals.items()},
               "train_unique_hours": totals["train"] * PATCHES / 3600,
               "records_admitted": len(source_map), "test_records_not_opened": 2,
               "raw_download_bytes_counted": False, "gradient_exposure_hours": 0,
               "protocol_sha256": sha(out / "protocol.json"), "manifest_sha256": sha(out / "manifest.jsonl"),
               "artifacts_sha256": sha(out / "artifacts.json"),
               "source_admission_sha256": sha(out / "source_admission.json"),
               "online_harness_allowed": False, "completed_epoch": time.time()}
    dump(out / "completion.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", required=True, help="The 806023 dataset folder inside this node's raw directory")
    p.add_argument("--source-ledger", required=True)
    p.add_argument("--metadata-admission", required=True)
    p.add_argument("--deny-json", action="append", required=True, help="Prior source allowlist, exclusions, or recording metadata")
    p.add_argument("--output", required=True)
    p.add_argument("--reserve-gib", type=float, default=30)
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "acquisition"))
    from smac_storage_guard import require_smac_raw_root
    evidence = require_smac_raw_root(args.raw_root)
    node_root = (Path.home() / "pj-xavier" / evidence["node_id"]).resolve()
    output = Path(args.output).expanduser().resolve()
    require(node_root in output.parents and output != node_root, "Prepared output must remain in this smac node's workspace")
    require(args.reserve_gib >= 1, "A positive disk reserve is required")
    print(json.dumps(prepare(args), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
