#!/usr/bin/env python3
"""Freeze and prepare raw-only SSL windows without action labels or held-out EEG.

The registration stage fixes the 75-source allowlist, 42-source denylist,
participant split, offline preprocessing, QC and resource limits before window
selection. The preparation stage consumes only that registration and raw data.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import resource
import shutil
import tempfile
import time
import zipfile

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
from threadpoolctl import threadpool_limits

INVENTORY_SHA = {
    "summary.json": "1a178d95582b09a830be3f53836fa29747b915e43c976e522b27d0d27c063a10",
    "portable_records.jsonl": "9ba6dd425a0ed46628618de5a1898c5077fb21d5ad4c003799a3e8ae69623090",
    "uniba_train_records.jsonl": "7a39fdacc148c0328f02b3224749e4f33aaf8728d5383f5afa3dacd593ffb6bc",
    "exclusions.json": "88a15eb1f3ef16d805eed741fdc573b9ff33c7ff7ff05dfcf2ff35743eb13c13",
}
ITU_CLOCK_SHA = "b7d2bd664c661c2328f5394a7f3a2c24d8177e154974b21ef86cf4788c7ce916"
UNIBA_CLOCK_SHA = "3365c63f4310adf9d5f8c53b6d4a25a1dc110cf25767fd0288688f0a032771a5"
UNIBA_EXTRACTED_SHA = "103e75ec8541d9cb0d061b9ad8411011d0f2433155093ce77f2d07c4da55aa62"
DEADLINE = 1795324658.879168
RESERVE = 35 * 1024**3
WINDOW = 30
TARGET_FS = 200
EDGE = 5
SHARD_WINDOWS = 16


def require(value, why):
    if not value:
        raise ValueError(why)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(data)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def publish(path, data, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def put_json(path, value, *, replace=False):
    publish(path, json_bytes(value), replace=replace)


def contained(root, relative):
    item = PurePosixPath(relative)
    require(not item.is_absolute() and ".." not in item.parts and "\\" not in relative, "source path escapes root")
    path = Path(root).joinpath(*item.parts)
    require(path.resolve().is_relative_to(Path(root).resolve()) and not path.is_symlink(), "source path is escaped or a symlink")
    return path


def participant_split(ids):
    require(len(ids) == len(set(ids)) == 19, "portable participant set must contain 19 groups")
    ordered = sorted(ids, key=lambda item: (hashlib.sha256(("20261122|" + item).encode()).hexdigest(), item))
    return {item: "train" if index < 15 else "val" for index, item in enumerate(ordered)}


def validate_source_declarations(rows, sources, denied):
    """Require original channel order and source/fold identity, not just counts."""
    require(len(rows) == len(sources) and {row["waveform_sha256"] for row in rows} == set(sources), "source coverage or identity changed")
    require(not set(sources).intersection(denied), "excluded source in allowlist")
    groups = {}
    devices = defaultdict(set)
    for row in rows:
        digest = row["waveform_sha256"]
        declaration = sources[digest]
        require(declaration["split"] in ("train", "val"), "only train and SSL validation allowed")
        require(declaration["subject_group"] == row["participant_group"] and declaration["source_recording_id"] == row["recording_id"] and declaration["dataset"] == row["dataset_id"], "source identity declaration changed")
        require(row["ssl_split"] == declaration["split"], "source fold declaration changed")
        group = declaration["subject_group"]
        require(group not in groups or groups[group] == declaration["split"], "same participant occurs in different folds")
        groups[group] = declaration["split"]
        channels = row["eeg_channels"]
        expected_count = 32 if row["dataset_id"] == "zenodo:15698441" else 8
        require(len(channels) == expected_count and len({c["name"] for c in channels}) == expected_count, "EEG channel count/name ambiguity")
        indices = [c["index_in_file"] for c in channels]
        require(indices == sorted(set(indices)) and all(0 <= i < row["channels_in_file"] for i in indices), "EEG channel order differs from original header")
        require(all(math.isfinite(c["gain_to_volts"]) and c["gain_to_volts"] > 0 and math.isfinite(c["offset_volts"]) for c in channels), "invalid native physical calibration")
        if row["dataset_id"] == "zenodo:15698441":
            require(row["device_session"] not in devices[group], "duplicate portable device in participant group")
            devices[group].add(row["device_session"])
        else:
            require(row["dataset_id"] == "zenodo:20801101" and row.get("source_supervised_partition") == "train" and declaration["split"] == "train", "nontraining UniBa source forbidden")
    require(all(value == {"BA", "EM", "LA"} for value in devices.values()), "portable participant must retain all three devices in one fold")


def window_grid(duration, native_fs):
    require(math.isfinite(duration) and duration > 2 * EDGE and int(native_fs) == native_fs and native_fs > 80, "invalid source duration or rate")
    count = max(0, math.floor((duration - 2 * EDGE + 1e-9) / WINDOW))
    return [(EDGE + index * WINDOW, EDGE + (index + 1) * WINDOW) for index in range(count)]


def qc_volts(data):
    x = np.asarray(data)
    require(x.ndim == 2 and x.shape[0] in (8, 32) and x.shape[1] == 6000, "SSL target shape must be channels by 6000")
    if not np.isfinite(x).all():
        return ["nonfinite"], {"max_abs_volts": None, "min_channel_std_volts": None}
    maximum, minimum_std = float(np.max(np.abs(x))), float(np.min(np.std(x, axis=1)))
    reasons = []
    if maximum > 100e-6:
        reasons.append("max_abs_exceeds_100uV")
    if minimum_std < .01e-6:
        reasons.append("a_channel_std_below_0p01uV")
    return reasons, {"max_abs_volts": maximum, "min_channel_std_volts": minimum_std}


def preprocess_window(volts, native_fs):
    require(volts.ndim == 2 and volts.shape[0] in (8, 32) and volts.shape[1] == WINDOW * native_fs, "exact native 30-second window required")
    divisor = math.gcd(int(native_fs), TARGET_FS)
    result = resample_poly(volts, TARGET_FS // divisor, int(native_fs) // divisor, axis=-1,
                           window=("kaiser", 5.0), padtype="constant", cval=0.)
    require(result.shape == (volts.shape[0], 6000), "polyphase target shape changed")
    reasons, statistics = qc_volts(result)
    return (np.asarray(result * 1e4, dtype=np.float32).reshape(volts.shape[0], 30, 200) if not reasons else None), reasons, statistics


def register(args):
    repo, out = Path(args.repo).resolve(), Path(args.output).resolve()
    require(not out.exists(), "registration must be new and immutable")
    inventory = repo / "private/xavier/ssl_source_inventory_v1"
    for name, expected in INVENTORY_SHA.items():
        require(sha(inventory / name) == expected, "fixed source inventory changed:" + name)
    portable, uniba = read_rows(inventory / "portable_records.jsonl"), read_rows(inventory / "uniba_train_records.jsonl")
    require(len(portable) == 57 and len(uniba) == 18, "fixed source counts changed")
    folds = participant_split(sorted({row["participant_group"] for row in portable}))
    sources, raw_records = {}, []
    for row in portable + uniba:
        digest = row["waveform_sha256"]
        require(digest not in sources, "duplicate raw source content")
        split = folds[row["participant_group"]] if row["dataset_id"] == "zenodo:15698441" else "train"
        require(row.get("target_labels_supplied") is False and row.get("selection_role") == "unlabeled_ssl_candidate", "label-containing inventory forbidden")
        source = {"split": split, "subject_group": row["participant_group"], "source_recording_id": row["recording_id"], "dataset": row["dataset_id"]}
        sources[digest] = source
        raw_records.append({**row, "ssl_split": split})
    denied = []
    for row in read_json(inventory / "exclusions.json")["uniba_excluded_recordings"]:
        denied.append({"source_sha256": row["waveform_sha256"], "reason": "original_uniba_" + row["original_partition"],
                       "site": "uniba", "source_recording_id": row["recording_id"]})
    clock_path = repo / "private/xavier/ai_coding_itu_external_clock_v1/alignment_audit.json"
    require(sha(clock_path) == ITU_CLOCK_SHA, "ITU exclusion audit changed")
    itu = read_json(clock_path)["recordings"]
    require(len(itu) == 26, "all ITU sources must be denied, including failed clock")
    for row in itu:
        denied.append({"source_sha256": row["binary_sha256"], "reason": "whole_itu_external_site_excluded", "site": "itu", "source_recording_id": row["header_member"]})
    failed_path = repo / "private/xavier/ai_coding_uniba_audit_v1/alignment_audit.json"
    extracted_path = repo / "private/xavier/verified_ai_coding_uniba_for_m1_v3/extracted_members.json"
    require(sha(failed_path) == UNIBA_CLOCK_SHA and sha(extracted_path) == UNIBA_EXTRACTED_SHA, "UniBa failed-clock source evidence changed")
    extraction = {row["path"]: row for row in read_json(extracted_path)["members"]}
    failed = [row for row in read_json(failed_path)["recordings"] if row["status"] == "blocked"]
    require({row["subject"] for row in failed} == {"sub-P003", "sub-P015"}, "UniBa failed-clock exclusion set changed")
    for row in failed:
        member = "data/01_raw/uniba/" + row["eeg_header"].split("data/01_raw/uniba/", 1)[1]
        member = str(PurePosixPath(member).with_suffix(".eeg"))
        evidence = extraction[member]
        path = contained(repo / "private/xavier/verified_ai_coding_uniba_for_m1_v3", member)
        require(path.stat().st_size == evidence["bytes"] and sha(path) == evidence["sha256"], "failed-clock raw identity changed")
        denied.append({"source_sha256": evidence["sha256"], "reason": "original_uniba_clock_failed_never_used", "site": "uniba", "source_recording_id": member})
    require(len(denied) == len({row["source_sha256"] for row in denied}) == 42 and not set(sources).intersection(row["source_sha256"] for row in denied), "allow/deny identity overlap or incomplete exclusion")
    validate_source_declarations(raw_records, sources, {row["source_sha256"] for row in denied})
    allowlist = {"schema": "xavier-ssl-source-allowlist-v1", "sources": sources, "excluded_source_sha256": sorted(row["source_sha256"] for row in denied),
                 "exclusion_reasons": denied, "all_non_allowlisted_sources_forbidden": True,
                 "portable_split": folds, "participant_split_method": "sha256(UTF8('20261122|' + canonical participant ID)) ascending, first 15 train, last 4 SSL val; all 3 devices same fold",
                 "identity_scope": "Publisher participant groups; cross-dataset human identity overlap is unknown", "source_inventory_hashes": INVENTORY_SHA,
                 "itu_clock_audit_sha256": ITU_CLOCK_SHA, "uniba_clock_audit_sha256": UNIBA_CLOCK_SHA, "uniba_extracted_members_sha256": UNIBA_EXTRACTED_SHA}
    allowlist_bytes = json_bytes(allowlist)
    output_bytes, windows, source_seconds = 0, Counter(), defaultdict(float)
    for row in raw_records:
        count = len(window_grid(row["stored_signal_seconds"], row["sample_rate_hz"]))
        windows[row["ssl_split"]] += count
        source_seconds[row["ssl_split"]] += row["stored_signal_seconds"]
        output_bytes += count * len(row["eeg_channels"]) * 6000 * 4
    largest_member = max(row["waveform_member_bytes"] for row in portable)
    estimate = {"candidate_windows_before_QC": dict(windows), "original_signal_seconds_by_fold": dict(source_seconds),
                "maximum_array_payload_bytes_before_QC": output_bytes, "largest_single_raw_temp_bytes": largest_member,
                "reserve_bytes": RESERVE, "free_bytes_observed": shutil.disk_usage(repo).free,
                "additional_budget_bytes_required": output_bytes + largest_member + 1024**3,
                "new_archive_copy_planned": False, "all_records_concurrent_extraction_planned": False,
                "peak_native_array_budget_bytes": 4 * 1024**3, "max_shard_windows": SHARD_WINDOWS}
    require(estimate["free_bytes_observed"] > RESERVE + estimate["additional_budget_bytes_required"], "insufficient disk reserve for fixed whole-corpus upper bound")
    protocol = {"schema": "xavier-ssl-preprocessing-v1", "registration_epoch": time.time(), "source_code_sha256": sha(__file__),
                "source_allowlist_sha256": hashlib.sha256(allowlist_bytes).hexdigest(), "source_inventory_hashes": INVENTORY_SHA,
                "window_seconds": WINDOW, "stride_seconds": WINDOW, "grid_origin_seconds": EDGE, "exclude_end_seconds": EDGE,
                "sample_rate_hz": TARGET_FS, "channel_policy": "all and only original BIDS EEG channels, original header order; 32 portable or 8 UniBa; no fabricated/interpolated missing channels",
                "units": {"decode": "MNE physical volts from native binary header calibration; preserve BIDS/header disagreement metadata", "output": "V_x_1e4", "output_dtype": "float32"},
                "portable_filter": {"type": "Butterworth", "order": 4, "band_hz": [3, 40], "representation": "SOS", "application": "sosfiltfilt full recording, padtype odd, scipy default padlen formula explicit"},
                "uniba_filter": "none added; distributed 3-40 Hz source retained",
                "resampling": "After filtering and exact native 30 s extraction: scipy resample_poly, rational ratio to 200 Hz, Kaiser beta5.0, constant zero padding",
                "QC": {"scope": "post-resample physical volts per 30-second window", "finite": True, "minimum_std_every_channel_volts": .01e-6, "maximum_absolute_volts": 100e-6, "population_std_ddof": 0},
                "storage": {"array_shape": "[nwindows, channels, 30, 200]", "max_windows_per_shard": SHARD_WINDOWS, "reserve_bytes": RESERVE},
                "cpu_native_thread_limit": 2, "absolute_deadline_epoch": DEADLINE, "maximum_native_working_estimate_bytes": 4 * 1024**3,
                "action_label_columns_read_or_used": False, "format_marker_note": "MNE may parse format annotations when opening raw files; no annotation or action label is selected, extracted or passed to SSL.",
                "fit_or_model_selection_performed": False, "training_started": False,
                "portable_validation_role": "SSL masked reconstruction validation only; not a final test", "online_causal_preprocessing_claimed": False,
                "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "mne", "threadpoolctl")}}
    out.mkdir(parents=True, exist_ok=False)
    publish(out / "source_allowlist.json", allowlist_bytes)
    put_json(out / "protocol.json", protocol)
    put_json(out / "capacity_plan.json", estimate)
    publish(out / "source_records.jsonl", ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in raw_records)).encode())
    receipt = {"status": "registered_before_preprocessing", "protocol_sha256": sha(out / "protocol.json"), "source_allowlist_sha256": sha(out / "source_allowlist.json"),
               "source_records_sha256": sha(out / "source_records.jsonl"), "capacity_plan_sha256": sha(out / "capacity_plan.json"), "source_code_sha256": sha(__file__),
               "allowed_sources": 75, "denied_sources": 42, "portable_train_groups": 15, "portable_ssl_val_groups": 4, "uniba_train_groups": 18, "capacity": estimate}
    put_json(out / "registration.json", receipt)
    print(json.dumps(receipt, allow_nan=False), flush=True)


def capacity_guard(path, needed=0):
    require(time.time() < DEADLINE, "original absolute deadline reached; no restart/continuation authorized")
    require(shutil.disk_usage(path).free > RESERVE + needed, "storage reserve would be exceeded")


def extract_member(z, name, destination, expected_size, expected_sha=None):
    target = contained(destination, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    info = z.getinfo(name)
    require(info.file_size == expected_size, "source member size changed")
    digest, total = hashlib.sha256(), 0
    with z.open(info) as source, target.open("xb") as out:
        for block in iter(lambda: source.read(1024**2), b""):
            capacity_guard(destination, len(block))
            out.write(block)
            total += len(block)
            digest.update(block)
    require(total == expected_size and (expected_sha is None or digest.hexdigest() == expected_sha), "extracted source SHA/size mismatch")
    return target


def direct_unit_check(raw, row, data_path, header_path):
    """Independently compare header-scaled samples with fresh MNE physical volts."""
    channels = row["eeg_channels"]
    names = [item["name"] for item in channels]
    n = row["channels_in_file"]
    largest_difference = 0.
    checked = 0
    if row["source_format"] == "BrainVision":
        require("BinaryFormat=IEEE_FLOAT_32" in header_path.read_text(encoding="utf-8-sig") and "DataOrientation=MULTIPLEXED" in header_path.read_text(encoding="utf-8-sig"), "unsupported direct BrainVision check")
        for start in (0, int(raw.n_times // 2), int(raw.n_times - 4)):
            digital = np.fromfile(data_path, dtype="<f4", count=n * 4, offset=start * n * 4).reshape(4, n).T.astype(np.float64)
            volts = np.stack([digital[item["index_in_file"]] * item["gain_to_volts"] + item["offset_volts"] for item in channels])
            actual = raw.get_data(picks=names, start=start, stop=start + 4)
            require(np.allclose(volts, actual, atol=1e-12, rtol=1e-9, equal_nan=False), "MNE/header voltage calibration mismatch")
            largest_difference = max(largest_difference, float(np.max(np.abs(volts - actual))))
            checked += volts.size
    else:
        with data_path.open("rb") as stream:
            fixed = stream.read(256)
            header_bytes = int(fixed[184:192])
            full = fixed + stream.read(header_bytes - 256)
            samples = [int(full[256 + 216 * n + 8 * i:256 + 216 * n + 8 * (i + 1)]) for i in range(n)]
            record_bytes = 2 * sum(samples)
            n_records = (data_path.stat().st_size - header_bytes) // record_bytes
            for record in (0, n_records // 2, n_records - 1):
                first = record * samples[channels[0]["index_in_file"]]
                expected = []
                for item in channels:
                    index = item["index_in_file"]
                    stream.seek(header_bytes + record * record_bytes + 2 * sum(samples[:index]))
                    digital = np.frombuffer(stream.read(8), dtype="<i2").astype(np.float64)
                    expected.append(digital * item["gain_to_volts"] + item["offset_volts"])
                expected = np.stack(expected)
                actual = raw.get_data(picks=names, start=first, stop=first + 4)
                require(np.allclose(expected, actual, atol=1e-12, rtol=1e-9, equal_nan=False), "MNE/EDF physical-range calibration mismatch")
                largest_difference = max(largest_difference, float(np.max(np.abs(expected - actual))))
                checked += expected.size
    return {"status": "passed", "sample_values_compared": checked, "maximum_absolute_difference_volts": largest_difference,
            "independent_amplifier_calibration": False, "method": "native header digital-to-physical voltage versus MNE at start/middle/end"}


def save_shard(out, source_sha, shard_index, buffer):
    relative = f"waveforms/{source_sha}_{shard_index:04d}.npy"
    path = contained(out, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "refuse an unsealed existing shard")
    array = np.stack(buffer).astype(np.float32, copy=False)
    capacity_guard(out, array.nbytes + 1024**2)
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return relative, {"sha256": sha(path), "bytes": path.stat().st_size, "shape": list(array.shape), "dtype": str(array.dtype)}


def process_source(row, out, allowlist, protocol_sha):
    import mne
    digest = row["waveform_sha256"]
    declaration = allowlist["sources"][digest]
    require(declaration["split"] == row["ssl_split"] and declaration["subject_group"] == row["participant_group"], "source fold/group changed")
    capacity_guard(out, row["waveform_member_bytes"] + 128 * 1024**2)
    channel_names = [item["name"] for item in row["eeg_channels"]]
    estimate = row["stored_signal_seconds"] * row["sample_rate_hz"] * len(channel_names) * 8 * 5 + 128 * 1024**2
    require(estimate <= 4 * 1024**3, "single-record conservative working-memory estimate exceeds fixed budget")
    with tempfile.TemporaryDirectory(prefix="raw-", dir=out / "temporary") as temporary:
        if row["dataset_id"] == "zenodo:15698441":
            with zipfile.ZipFile(row["archive_path"]) as z:
                data_path = extract_member(z, row["waveform_member"], temporary, row["waveform_member_bytes"], digest)
                if row["source_format"] == "BrainVision":
                    header_path = extract_member(z, row["header_member"], temporary, z.getinfo(row["header_member"]).file_size, row["header_sha256"])
                    extract_member(z, row["marker_member"], temporary, z.getinfo(row["marker_member"]).file_size)
                else:
                    header_path = data_path
        else:
            data_path, header_path = Path(row["waveform_path"]), Path(row["header_path"])
            require(sha(data_path) == digest and sha(header_path) == row["header_sha256"], "UniBa allowed original source changed")
        raw = (mne.io.read_raw_brainvision(header_path, preload=False, verbose="ERROR") if row["source_format"] == "BrainVision"
               else mne.io.read_raw_edf(header_path, preload=False, verbose="ERROR"))
        try:
            require(raw.info["sfreq"] == row["sample_rate_hz"] and abs(raw.n_times / raw.info["sfreq"] - row["stored_signal_seconds"]) < 1 / raw.info["sfreq"] + 1e-9,
                    "decoded source duration/rate differs from quantity audit")
            unit_check = direct_unit_check(raw, row, data_path, header_path)
            volts = raw.get_data(picks=channel_names)
        finally:
            raw.close()
        require(volts.shape[0] in (8, 32), "EEG-only channel count changed")
        rate = int(row["sample_rate_hz"])
        if row["dataset_id"] == "zenodo:15698441":
            sos = butter(4, [3, 40], btype="bandpass", fs=rate, output="sos")
            padlen = int(3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())))
            volts = sosfiltfilt(sos, volts, axis=-1, padtype="odd", padlen=padlen)
        buffer, pending, records, artifacts, rejected = [], [], [], {}, []
        grid = window_grid(row["stored_signal_seconds"], rate)
        shard_index = 0
        def flush():
            nonlocal shard_index
            if not buffer:
                return
            path, artifact = save_shard(out, digest, shard_index, buffer)
            artifacts[path] = artifact
            for index, item in enumerate(pending):
                records.append({**item, "eeg_path": path, "eeg_index": index})
            buffer.clear()
            pending.clear()
            shard_index += 1
        for start, end in grid:
            capacity_guard(out)
            value, reasons, statistics = preprocess_window(volts[:, int(start * rate):int(end * rate)], rate)
            if reasons:
                rejected.append({"window_start_s": start, "window_end_s": end, "reasons": reasons, **statistics})
                continue
            require(np.isfinite(value).all() and np.max(np.abs(value)) <= 1.00001, "scaled stored values invalid")
            buffer.append(value)
            pending.append({"id": f"ssl-{digest}-{int(start * TARGET_FS):010d}", "split": declaration["split"], "subject_group": declaration["subject_group"],
                            "source_sha256": digest, "source_recording_id": declaration["source_recording_id"], "dataset": declaration["dataset"],
                            "window_start_s": start, "window_end_s": end, "channels": channel_names, "sample_rate_hz": TARGET_FS, "unit": "V_x_1e4",
                            "data_origin": "real_recording", "preprocessing_protocol_sha256": protocol_sha, "source_native_sample_rate_hz": rate})
            if len(buffer) == SHARD_WINDOWS:
                flush()
        flush()
        del volts
    certificate = {"status": "completed", "source_sha256": digest, "source_recording_id": declaration["source_recording_id"],
                   "split": declaration["split"], "subject_group": declaration["subject_group"], "unit_check": unit_check,
                   "original_signal_seconds": row["stored_signal_seconds"], "candidate_windows": len(grid), "accepted_windows": len(records),
                   "accepted_unique_source_seconds": len(records) * WINDOW, "rejected_windows": rejected,
                   "rejected_reason_counts": dict(Counter(reason for item in rejected for reason in item["reasons"])),
                   "records": records, "artifacts": artifacts, "protocol_sha256": protocol_sha,
                   "bids_header_unit_disagreement": row.get("bids_header_unit_disagreement", False)}
    put_json(out / "source_certificates" / (digest + ".json"), certificate)
    return certificate


def prepare(args):
    started = time.monotonic()
    registration = Path(args.registration).resolve()
    protocol, receipt = read_json(registration / "protocol.json"), read_json(registration / "registration.json")
    protocol_sha = sha(registration / "protocol.json")
    require(protocol_sha == args.expected_protocol_sha256 == receipt["protocol_sha256"] and sha(__file__) == protocol["source_code_sha256"] == receipt["source_code_sha256"], "registered source/protocol changed")
    for name, key in (("source_allowlist.json", "source_allowlist_sha256"), ("source_records.jsonl", "source_records_sha256"), ("capacity_plan.json", "capacity_plan_sha256")):
        require(sha(registration / name) == receipt[key], "registered input changed:" + name)
    allowlist = read_json(registration / "source_allowlist.json")
    require(sha(registration / "source_allowlist.json") == protocol["source_allowlist_sha256"], "allowlist is not sealed by protocol")
    rows = read_rows(registration / "source_records.jsonl")
    require({row["waveform_sha256"] for row in rows} == set(allowlist["sources"]) and len(rows) == 75, "allowlisted source coverage changed")
    require(not set(allowlist["sources"]) & set(allowlist["excluded_source_sha256"]), "excluded source in allowed corpus")
    validate_source_declarations(rows, allowlist["sources"], set(allowlist["excluded_source_sha256"]))
    out = Path(args.output).resolve()
    if args.resume:
        require(out.is_dir() and not (out / "completion.json").exists(), "resume requires an incomplete existing corpus")
        require(sha(out / "protocol.json") == protocol_sha and sha(out / "source_allowlist.json") == protocol["source_allowlist_sha256"], "existing corpus registration differs")
    else:
        out.mkdir(parents=True, exist_ok=False)
        for name in ("protocol.json", "source_allowlist.json", "registration.json", "capacity_plan.json"):
            publish(out / name, (registration / name).read_bytes())
    (out / "temporary").mkdir(exist_ok=True)
    lock = (out / ".preparation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    capacity_guard(out, read_json(registration / "capacity_plan.json")["additional_budget_bytes_required"])
    seen_archives = set()
    for row in rows:
        if row.get("archive_path") and row["archive_path"] not in seen_archives:
            require(sha(row["archive_path"]) == row["archive_sha256"], "original portable archive changed after registration")
            seen_archives.add(row["archive_path"])
    certificates = []
    try:
        for index, row in enumerate(rows):
            capacity_guard(out)
            cert_path = out / "source_certificates" / (row["waveform_sha256"] + ".json")
            if cert_path.exists():
                require(args.resume, "source receipt unexpectedly exists in a new attempt")
                cert = read_json(cert_path)
                require(cert["status"] == "completed" and cert["source_sha256"] == row["waveform_sha256"] and cert["protocol_sha256"] == protocol_sha, "resume receipt source mismatch")
                for name, artifact in cert["artifacts"].items():
                    path = contained(out, name)
                    require(path.stat().st_size == artifact["bytes"] and sha(path) == artifact["sha256"], "completed shard changed; cannot resume")
            else:
                # Interrupted, unsealed outputs stay preserved for investigation;
                # they cannot silently be adopted or overwritten on resume.
                require(not list((out / "waveforms").glob(row["waveform_sha256"] + "_*.npy")), "unsealed source shards require manual recovery before resume")
                cert = process_source(row, out, allowlist, protocol_sha)
            certificates.append(cert)
            put_json(out / "progress.json", {"status": "running", "pid": os.getpid(), "completed_sources": index + 1, "total_sources": 75,
                     "accepted_windows": sum(item["accepted_windows"] for item in certificates), "elapsed_s": time.monotonic() - started,
                     "protocol_sha256": protocol_sha, "free_bytes": shutil.disk_usage(out).free}, replace=True)
            print(json.dumps({"source_number": index + 1, "accepted": cert["accepted_windows"], "candidate": cert["candidate_windows"], "elapsed_s": time.monotonic() - started}), flush=True)
        records = [row for cert in certificates for row in cert["records"]]
        artifacts = {name: value for cert in certificates for name, value in cert["artifacts"].items()}
        require(len({row["id"] for row in records}) == len(records), "duplicate SSL window IDs")
        for row in records:
            require("label" not in row and row["source_sha256"] not in allowlist["excluded_source_sha256"], "action label or forbidden source entered corpus")
            declared = allowlist["sources"][row["source_sha256"]]
            require(all(row[key] == declared[key] for key in ("split", "subject_group", "source_recording_id", "dataset")), "window differs from immutable allowlist")
        publish(out / "manifest.jsonl", ("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records)).encode())
        put_json(out / "artifacts.json", artifacts)
        totals = {split: {"accepted_windows": sum(row["split"] == split for row in records),
                          "accepted_unique_source_seconds": sum(row["split"] == split for row in records) * WINDOW,
                          "one_second_patch_presentations_once": sum(row["split"] == split for row in records) * 30,
                          "source_recordings_with_windows": len({row["source_sha256"] for row in records if row["split"] == split}),
                          "participant_groups_with_windows": len({row["subject_group"] for row in records if row["split"] == split})} for split in ("train", "val")}
        summary = {"schema": "xavier-ssl-prepared-corpus-v1", "status": "completed", "manifest_sha256": sha(out / "manifest.jsonl"),
                   "artifacts_sha256": sha(out / "artifacts.json"), "protocol_sha256": protocol_sha, "source_allowlist_sha256": sha(out / "source_allowlist.json"),
                   "source_inventory_hashes": INVENTORY_SHA, "source_code_sha256": sha(__file__), "allowed_sources": 75, "denied_sources": 42,
                   "total_original_source_seconds": sum(item["original_signal_seconds"] for item in certificates),
                   "candidate_windows_before_QC": sum(item["candidate_windows"] for item in certificates), "accepted_windows": len(records),
                   "rejected_windows": sum(len(item["rejected_windows"]) for item in certificates),
                   "rejected_reason_counts": dict(sum((Counter(item["rejected_reason_counts"]) for item in certificates), Counter())),
                   "folds": totals, "artifact_files": len(artifacts), "artifact_bytes": sum(item["bytes"] for item in artifacts.values()),
                   "original_sources_modified": False, "action_label_columns_read_or_used": False, "model_training_started": False,
                   "all_non_allowlisted_sources_forbidden": True, "online_causal_preprocessing_claimed": False,
                   "unit_check_sources_passed": sum(item["unit_check"]["status"] == "passed" for item in certificates),
                   "portable_ssl_val_is_final_test": False, "latent_intent_validated": False,
                   "elapsed_s": time.monotonic() - started, "process_max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "rss_platform": os.uname().sysname, "native_cpu_threads_max": 2,
                   "quantity_scope": "Source durations, unique accepted intervals, 1s temporal patches and repeated epochs are separate; source/archive/derived bytes are not independent new EEG."}
        require(sha(__file__) == protocol["source_code_sha256"] and sha(registration / "protocol.json") == protocol_sha, "code/protocol changed during preparation")
        put_json(out / "corpus_summary.json", summary)
        put_json(out / "completion.json", {"status": "completed", "exit_code": 0, "manifest_sha256": summary["manifest_sha256"],
                 "artifacts_sha256": summary["artifacts_sha256"], "protocol_sha256": protocol_sha,
                 "source_allowlist_sha256": summary["source_allowlist_sha256"], "summary_sha256": sha(out / "corpus_summary.json"), "completed_epoch": time.time()})
        put_json(out / "progress.json", {"status": "completed", "completed_sources": 75, "accepted_windows": len(records), "summary_sha256": sha(out / "corpus_summary.json")}, replace=True)
        print(json.dumps(summary, allow_nan=False), flush=True)
    except BaseException as error:
        put_json(out / "failure.json", {"status": "failed_or_interrupted", "reason": type(error).__name__ + ": " + str(error),
                 "completed_sources": len(certificates), "protocol_sha256": protocol_sha, "automatic_unsealed_shard_reuse_allowed": False}, replace=True)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    r = commands.add_parser("register")
    r.add_argument("--repo", required=True)
    r.add_argument("--output", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--registration", required=True)
    p.add_argument("--expected-protocol-sha256", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        register(args) if args.command == "register" else prepare(args)


if __name__ == "__main__":
    main()
