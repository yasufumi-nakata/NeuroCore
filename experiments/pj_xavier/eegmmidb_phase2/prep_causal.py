"""Immutable, causal EEGMMIDB phase2 preparation. No fitting or model inference.

The raw EDF calibration is applied once. A zero-state SOS stream and an
unshifted upfirdn stream use only samples available at each output timestamp.
Annotation data choose supervised targets but never become waveform features.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_CEILING
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
import numpy as np
import scipy
from scipy.signal import butter, firwin, sosfilt, upfirdn
from threadpoolctl import threadpool_limits

DATASET = "physionet:eegmmidb:1.0.0"
HEADER_HELPER_SHA = "b4e3979968c427a18509f9b2df9a2cdba452e66827300d212b48b49d11bd5c98"
TASK_RUNS = {4: "left_right", 8: "left_right", 12: "left_right", 6: "fists_feet", 10: "fists_feet", 14: "fists_feet"}
LABELS = {"left_right": {"T1": "left_fist", "T2": "right_fist"}, "fists_feet": {"T1": "both_fists", "T2": "both_feet"}}
FS, LENGTH, CHANNELS = 200, 600, 64
MAX_ABS_V, MIN_STD_V = 1e-4, 1e-8


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def stable_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def save_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    require(not path.exists() and not temp.exists(), "Immutable output exists: " + str(path))
    with temp.open("xb") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.link(temp, path)
    temp.unlink()


def save_json(path, value):
    save_bytes(path, stable_bytes(value))


def save_jsonl(path, values):
    save_bytes(path, b"".join((json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode() for x in values))


def versions():
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__}


def split_assignments():
    ids = [f"{DATASET}:S{i:03d}" for i in range(1, 110)]
    ids.sort(key=lambda x: hashlib.sha256(("20261122|" + x).encode()).hexdigest())
    return {value: "train" if index < 76 else "val" if index < 92 else "test" for index, value in enumerate(ids)}


def guard(meta, phase_root, required_bytes=0):
    require(time.time() < meta["CPU_absolute_deadline_epoch"], "Original CPU preparation deadline reached")
    usage = sum(p.stat().st_size for p in Path(phase_root).rglob("*") if p.is_file())
    require(usage + required_bytes <= meta["incremental_output_budget_bytes"], "Incremental phase2 budget exceeded")
    free = shutil.disk_usage(phase_root).free
    require(free - required_bytes >= meta["disk_reserve_bytes"] + meta["reserved_future_bulk_growth_bytes"],
            "Disk reserve including pending bulk growth would be crossed")


def load_contract(phase_root):
    phase_root = Path(phase_root)
    mroot = phase_root / "metadata"
    meta_path = mroot / "metadata_protocol.json"
    meta = json.loads(meta_path.read_text())
    require(meta["dataset"] == DATASET and meta["status"] == "frozen_metadata_before_signal_diagnostics", "Wrong metadata contract")
    for name, key in (("subject_split.json", "subject_split_sha256"), ("recording_metadata.jsonl", "recording_metadata_sha256")):
        require(sha_file(mroot / name) == meta[key], "Metadata bytes changed: " + name)
    assignments = json.loads((mroot / "subject_split.json").read_text())["assignments"]
    require(assignments == split_assignments(), "Frozen participant hash split differs")
    records = [json.loads(s) for s in (mroot / "recording_metadata.jsonl").read_text().splitlines()]
    require(len(records) == 1526 and len({r["source_recording_id"] for r in records}) == 1526, "Incomplete or repeated source metadata")
    by_source, by_payload = defaultdict(set), defaultdict(set)
    for r in records:
        require(r["split"] == assignments[r["subject_group"]], "Subject split changed")
        by_source[r["source_eeg_sha256"]].add(r["split"])
        by_payload[r["eeg_payload_sha256"]].add(r["split"])
        if r["status"] == "MI_source_candidate":
            require(TASK_RUNS.get(r["run"]) == r["task_family"], "Run-specific MI mapping differs")
    require(all(len(v) == 1 for v in list(by_source.values()) + list(by_payload.values())), "Identical original EEG crosses split")
    require(Counter(r["status"] for r in records) == meta["metadata_record_counts"], "Metadata statuses changed")
    guard(meta, phase_root)
    return meta, records, sha_file(meta_path)


def header_helper():
    path = Path(__file__).resolve().parent.parent / "acquisition" / "audit_eegmmidb_zip.py"
    require(sha_file(path) == HEADER_HELPER_SHA, "Pinned original EDF header parser changed")
    spec = importlib.util.spec_from_file_location("phase2_pinned_edf_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_member(z, record, helper):
    """Decode one bounded member; verify all EDF and ordinary EEG bytes first."""
    name = "files/" + record["member"]
    info = z.getinfo(name)
    require(info.file_size <= 8 * 2**20 and not info.flag_bits & 1, "Unexpected member size/encryption")
    data = z.read(info)
    require(hashlib.sha256(data).hexdigest() == record["source_eeg_sha256"], "Official EDF member SHA differs")
    stream = io.BytesIO(data)
    header, raw_header = helper.edf_header(stream, len(data))
    eeg = [c for c in header["channels"] if not c["is_annotation"]]
    require(header["header_sha256"] == record["header_sha256"] and len(eeg) == 64, "Source EEG header changed")
    require([c["label"] for c in eeg] == record["channels"], "Native channel order changed")
    rate = record["native_sample_rate_hz"]
    require(rate in (128, 160) and all(c["sample_rate_hz"] == rate for c in eeg), "Native sample rate differs")
    require(all(c["unit"] == "uV" for c in eeg), "Unsupported calibrated physical unit")
    require(float(header["signal_duration_seconds_decimal"]) == record["source_seconds"], "Native duration changed")
    n_per = eeg[0]["samples_per_record"]
    require(all(c["samples_per_record"] == n_per for c in eeg), "Mixed native sample counts")
    output = np.empty((64, n_per * header["data_records"]), dtype=np.float64)
    digest = hashlib.sha256()
    for nr in range(header["data_records"]):
        block = stream.read(header["data_record_bytes"])
        require(len(block) == header["data_record_bytes"], "Truncated original data record")
        offset, ci = 0, 0
        for channel in header["channels"]:
            size = channel["samples_per_record"] * 2
            raw = block[offset:offset + size]
            if not channel["is_annotation"]:
                digest.update(raw)
                digital = np.frombuffer(raw, dtype="<i2").astype(np.float64)
                slope = (channel["physical_max"] - channel["physical_min"]) / (channel["digital_max"] - channel["digital_min"])
                physical_uv = (digital - channel["digital_min"]) * slope + channel["physical_min"]
                output[ci, nr*n_per:(nr+1)*n_per] = physical_uv * 1e-6
                ci += 1
            offset += size
    require(digest.hexdigest() == record["eeg_payload_sha256"] and not stream.read(1), "Ordinary EEG payload differs")
    return output, int(rate), {"member": record["member"], "source_recording_id": record["source_recording_id"],
                              "source_sha256": record["source_eeg_sha256"], "eeg_payload_sha256": digest.hexdigest(),
                              "member_bytes": len(data), "EEG_sample_bytes": header["eeg_sample_bytes"],
                              "native_samples": output.shape[1], "native_sample_rate_hz": rate}


def causal_stream(values, native_rate):
    require(native_rate in (128, 160), "Unsupported native sample rate")
    require(values.ndim == 2 and np.isfinite(values).all(), "Nonfinite or invalid original waveform")
    sos = butter(4, [3.0, 40.0], btype="bandpass", fs=native_rate, output="sos")
    filtered = sosfilt(sos, np.asarray(values, dtype=np.float64), axis=-1)
    g = math.gcd(FS, int(native_rate))
    up, down = FS // g, int(native_rate) // g
    taps = firwin(20 * max(up, down) + 1, 1.0 / max(up, down), window=("kaiser", 5.0)) * up
    result = upfirdn(taps, filtered, up=up, down=down, axis=-1)
    # Unshifted causal convolution: output m cannot use native n > floor(m*down/up).
    count = (values.shape[-1] * up + down - 1) // down
    return result[:, :count]


def interval(onset, duration, recording_seconds, *, shifted=False):
    start = Decimal(str(onset)) + Decimal(1) - (Decimal(10) if shifted else Decimal(0))
    sample_start = int((start * FS).to_integral_value(rounding=ROUND_CEILING))
    sample_stop = sample_start + LENGTH
    a, b = Decimal(sample_start)/FS, Decimal(sample_stop)/FS
    reasons = []
    if a < 5:
        reasons.append("insufficient_past_filter_history")
    if b > Decimal(str(recording_seconds)):
        reasons.append("window_exceeds_record")
    if not shifted and (a < Decimal(str(onset)) or b > Decimal(str(onset)) + Decimal(str(duration))):
        reasons.append("window_exceeds_annotation")
    return {"sample_start": sample_start, "sample_stop": sample_stop,
            "window_start_s": float(a), "window_end_s": float(b),
            "grid_shift_s": float(a-start), "reasons": reasons}


def qc(values):
    require(values.shape == (64, 600), "QC requires exactly 64 by 600")
    finite = bool(np.isfinite(values).all())
    maximum = float(np.max(np.abs(values))) if finite else None
    minimum_std = float(np.min(np.std(values, axis=-1, ddof=0))) if finite else None
    reasons = []
    if not finite:
        reasons.append("nonfinite")
    else:
        if maximum > MAX_ABS_V:
            reasons.append("absolute_amplitude_above_100uV")
        if minimum_std < MIN_STD_V:
            reasons.append("channel_population_std_below_0p01uV")
    return {"finite": finite, "maximum_absolute_V": maximum, "minimum_channel_population_std_V": minimum_std,
            "reasons": reasons, "passed": not reasons}


def evaluate_window(stream, span):
    if span["reasons"]:
        return None, {"passed": False, "reasons": span["reasons"]}
    require(0 <= span["sample_start"] < span["sample_stop"] <= stream.shape[-1], "Processed window outside causal stream")
    x = stream[:, span["sample_start"]:span["sample_stop"]]
    check = qc(x)
    return (np.asarray(x, dtype=np.float32) if check["passed"] else None), check


def timing(span, native_rate, source):
    last_native = ((span["sample_stop"] - 1) * native_rate) // FS
    start_native = math.ceil(span["sample_start"] * native_rate / FS)
    stop_native = math.ceil(span["sample_stop"] * native_rate / FS)
    return {"window_start_s": span["window_start_s"], "window_end_s": span["window_end_s"],
            "sample_start": span["sample_start"], "sample_stop": span["sample_stop"],
            "decision_time_s": span["window_end_s"], "feature_available_time_s": (span["sample_stop"]-1)/FS,
            "raw_interval": {"source_recording_id": source, "native_sample_start": start_native,
                             "native_sample_stop": stop_native, "start_s": span["window_start_s"],
                             "end_s": span["window_end_s"], "scope": "target interval, not total causal filter history"},
            "causal_history": {"source_recording_id": source, "native_sample_start": 0,
                               "native_sample_stop": last_native+1, "start_s": 0.0,
                               "last_sample_time_s": last_native/native_rate,
                               "end_s_exclusive": (last_native+1)/native_rate,
                               "scope": "full past recording state; no future samples; unshifted FIR group delay retained"}}


def row_for(record, event, ei, span, protocol_sha, meta_sha, check):
    rid = record["source_recording_id"]
    return {"id": rid + ":event" + str(ei), "dataset": DATASET, "subject": record["subject_group"],
            "subject_group": record["subject_group"], "session": "published_session", "recording_id": rid,
            "source_recording_id": rid, "split": record["split"], "task_family": record["task_family"],
            "label": "rest" if event["code"] == "T0" else LABELS[record["task_family"]][event["code"]],
            "label_source": "recorded_event", "data_origin": "real_recording",
            "source_eeg_sha256": record["source_eeg_sha256"], "source_sha256": record["source_eeg_sha256"],
            "eeg_payload_sha256": record["eeg_payload_sha256"], "source_member": record["member"],
            "channels": record["channels"], "channel_order": record["channels"], "sample_rate_hz": FS,
            "native_sample_rate_hz": record["native_sample_rate_hz"], "unit": "V",
            "event_onset_s": event["onset_seconds"], "event_offset_s": event["onset_seconds"]+event["duration_seconds"],
            "gold_annotation_code": event["code"], "annotation_review_required": False,
            "preprocessing": {"online_causality_verified": True, "protocol_sha256": protocol_sha,
                              "metadata_protocol_sha256": meta_sha, "filter": "3-40Hz causal SOS; unshifted causal FIR resampling",
                              "normalization": "none", "future_samples_used": False},
            "QC": check, **timing(span, int(record["native_sample_rate_hz"]), rid)}


def percentiles(values):
    values = [float(x) for x in values if x is not None]
    return dict(zip(("min", "p05", "median", "p95", "max"), map(float, np.percentile(values, [0, 5, 50, 95, 100])))) if values else None


def verify_archive(archive, meta):
    require(Path(archive).is_file() and not Path(archive).is_symlink(), "Original archive missing or symlink")
    require(Path(archive).stat().st_size == meta["source_archive_bytes"] and sha_file(archive) == meta["source_archive_sha256"], "Original archive identity changed")


def diagnose_train(archive, phase_root):
    meta, records, meta_sha = load_contract(phase_root)
    out = Path(phase_root) / "train_diagnostics"
    require(not out.exists(), "Train diagnostic attempt already exists")
    out.mkdir()
    verify_archive(archive, meta)
    helper = header_helper()
    values, used, reasons = [], [], Counter()
    started = time.time()
    with zipfile.ZipFile(archive) as z, threadpool_limits(limits=2):
        for record in records:
            if record["status"] != "MI_source_candidate" or record["split"] != "train":
                continue
            guard(meta, phase_root)
            raw, fs, source = decode_member(z, record, helper)
            stream = causal_stream(raw, fs)
            used.append(source)
            for ei, event in enumerate(record["events"]):
                if event["code"] not in ("T1", "T2"):
                    continue
                span = interval(event["onset_seconds"], event["duration_seconds"], record["source_seconds"])
                _, check = evaluate_window(stream, span)
                reasons.update(check["reasons"])
                values.append({"id": record["source_recording_id"]+":event"+str(ei), "split": "train", **check})
            if len(used) % 50 == 0:
                print(json.dumps({"stage": "train_diagnostics", "records": len(used), "elapsed_seconds": time.time()-started}), flush=True)
    verify_archive(archive, meta)
    save_jsonl(out / "windows.jsonl", values)
    save_jsonl(out / "used_sources.jsonl", used)
    result = {"schema": "pj-xavier-phase2-train-only-signal-QC-v1", "status": "completed", "code_sha256": sha_file(__file__),
              "metadata_protocol_sha256": meta_sha, "original_zip_before_after_sha256": meta["source_archive_sha256"],
              "versions": versions(), "records": len(used), "candidate_windows": len(values),
              "accepted_candidate_windows": sum(v["passed"] for v in values), "rejection_reason_counts": dict(reasons),
              "maximum_abs_V_distribution": percentiles(v.get("maximum_absolute_V") for v in values),
              "minimum_channel_std_V_distribution": percentiles(v.get("minimum_channel_population_std_V") for v in values),
              "candidate_thresholds": {"maximum_absolute_V": MAX_ABS_V, "minimum_channel_population_std_V": MIN_STD_V},
              "val_waveform_reads": 0, "test_waveform_reads": 0, "model_forward_calls": 0,
              "windows_sha256": sha_file(out/"windows.jsonl"), "sources_sha256": sha_file(out/"used_sources.jsonl"),
              "elapsed_seconds": time.time()-started}
    save_json(out/"summary.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def seal_preprocessing(phase_root, tests_receipt):
    meta, _, meta_sha = load_contract(phase_root)
    diag_path = Path(phase_root)/"train_diagnostics/summary.json"
    diag = json.loads(diag_path.read_text())
    tests = json.loads(Path(tests_receipt).read_text())
    code_sha = sha_file(__file__)
    require(diag["code_sha256"] == code_sha and diag["metadata_protocol_sha256"] == meta_sha, "Train diagnostic source changed")
    require(diag["status"] == "completed" and diag["val_waveform_reads"] == diag["test_waveform_reads"] == 0, "Diagnostic split restriction violated")
    require(tests["status"] == "passed" and tests["prep_source_sha256"] == code_sha, "Meaningful preprocessing tests not sealed")
    protocol = {"schema": "pj-xavier-eegmmidb-phase2-preprocessing-v1", "status": "frozen_before_val_test_signal_processing",
                "metadata_protocol_sha256": meta_sha, "subject_split_sha256": meta["subject_split_sha256"],
                "recording_metadata_sha256": meta["recording_metadata_sha256"], "source_archive_sha256": meta["source_archive_sha256"],
                "source_code_sha256": code_sha, "pinned_header_helper_sha256": HEADER_HELPER_SHA, "versions": versions(),
                "train_only_diagnostics_sha256": sha_file(diag_path), "tests_receipt_sha256": sha_file(tests_receipt),
                "recipe": meta["preprocessing_candidate"], "QC": {"measurement": "after causal filter and causal resampling, physical volts, before float32 save",
                "finite_required": True, "maximum_absolute_V": MAX_ABS_V, "minimum_channel_population_std_V": MIN_STD_V,
                "standard_deviation_ddof": 0, "absolute_amplitude_not_peak_to_peak": True,
                "threshold_decision": "retain initial candidate thresholds; no held-out signal or retention used"},
                "fixed_feature_window": meta["feature_window"], "static_task_context_only": True,
                "minimum_history_s": 5.0, "FIR_group_delay_s_by_native_fs": {"160": .0625, "128": .078125},
                "latency_note": "FIR delay retained; no acausal delay compensation. IIR delay is frequency-dependent.",
                "original_CPU_deadline_epoch": meta["CPU_absolute_deadline_epoch"], "sealed_epoch": time.time()}
    save_json(Path(phase_root)/"metadata/preprocessing_protocol.json", protocol)
    print(json.dumps({"preprocessing_protocol_sha256": sha_file(Path(phase_root)/"metadata/preprocessing_protocol.json")}), flush=True)


def save_array(output, relative, arrays, meta, phase_root):
    require(arrays, "Cannot write an empty array")
    a = np.stack(arrays).astype(np.float32, copy=False)
    require(a.shape[1:] == (64, 600), "Unexpected waveform shape")
    guard(meta, phase_root, a.nbytes+4096)
    target = Path(output)/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".part")
    require(not target.exists() and not temp.exists(), "Existing prepared array")
    with temp.open("xb") as f:
        np.save(f, a, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.link(temp, target)
    temp.unlink()
    return {"sha256": sha_file(target), "bytes": target.stat().st_size, "shape": list(a.shape), "dtype": "float32"}


def union_seconds(rows):
    source_intervals = defaultdict(list)
    for r in rows:
        source_intervals[r["source_eeg_sha256"]].append((r["window_start_s"], r["window_end_s"]))
    total = 0.0
    for spans in source_intervals.values():
        end = None
        for a, b in sorted(spans):
            total += b-a if end is None or a >= end else max(0, b-end)
            end = max(end or b, b)
    return total


def group_summary(rows):
    return {"windows": len(rows), "recordings": len({r["source_recording_id"] for r in rows}),
            "participant_groups": len({r["subject_group"] for r in rows}),
            "label_counts": dict(Counter(r["label"] for r in rows)),
            "task_family_counts": dict(Counter(r["task_family"] for r in rows)),
            "nominal_window_seconds": len(rows)*3.0, "unique_target_interval_seconds": union_seconds(rows)}


def prepare(archive, phase_root):
    started = time.time()
    meta, records, meta_sha = load_contract(phase_root)
    protopath = Path(phase_root)/"metadata/preprocessing_protocol.json"
    proto = json.loads(protopath.read_text())
    proto_sha = sha_file(protopath)
    require(proto["source_code_sha256"] == sha_file(__file__) and proto["versions"] == versions(), "Sealed code or dependencies changed")
    require(proto["metadata_protocol_sha256"] == meta_sha and proto["QC"]["maximum_absolute_V"] == MAX_ABS_V and proto["QC"]["minimum_channel_population_std_V"] == MIN_STD_V, "Recipe changed")
    output = Path(phase_root)/"prepared"
    require(not output.exists(), "Preparation output already exists; previous attempt must be retained")
    output.mkdir()
    verify_archive(archive, meta)
    helper = header_helper()
    main, idle, rejects, sources, held = [], [], [], [], []
    main_art, ctrl_art, idle_art = {}, {}, {}
    with zipfile.ZipFile(archive) as z, threadpool_limits(limits=2):
        for record in records:
            if record["status"] != "MI_source_candidate":
                held.append({k:record[k] for k in ("source_recording_id", "source_eeg_sha256", "subject_group", "split", "status")})
                continue
            guard(meta, phase_root)
            raw, fs, source = decode_member(z, record, helper)
            stream = causal_stream(raw, fs)
            sources.append({**source, "split": record["split"], "subject_group": record["subject_group"], "source_seconds": record["source_seconds"]})
            arrays, controls, idle_arrays, rows, idle_rows = [], [], [], [], []
            basename = Path(record["member"]).stem
            main_path = "waveforms/"+basename+".npy"
            control_path = "controls/"+basename+".npy"
            idle_path = "idle/"+basename+".npy"
            for ei, event in enumerate(record["events"]):
                is_idle = event["code"] == "T0"
                if is_idle and record["split"] != "test":
                    continue
                require(event["code"] in ("T0", "T1", "T2"), "Unknown annotation; no automatic label repair")
                span = interval(event["onset_seconds"], event["duration_seconds"], record["source_seconds"])
                array, check = evaluate_window(stream, span)
                if array is None:
                    rejects.append({"id": record["source_recording_id"]+":event"+str(ei), "role": "idle" if is_idle else "main", "split": record["split"], "source_recording_id": record["source_recording_id"], **check})
                    continue
                row = row_for(record, event, ei, span, proto_sha, meta_sha, check)
                if is_idle:
                    row.update(eeg_path=idle_path, eeg_index=len(idle_arrays), diagnostic_only=True, use_for_fit_or_selection=False)
                    idle_arrays.append(array)
                    idle_rows.append(row)
                else:
                    row.update(eeg_path=main_path, eeg_index=len(arrays))
                    arrays.append(array)
                    shifted = interval(event["onset_seconds"], event["duration_seconds"], record["source_seconds"], shifted=True)
                    control, cqc = evaluate_window(stream, shifted)
                    if control is None:
                        row["time_shift_control"] = {"status": "blocked", "shift_seconds": -10.0, "reason": ";".join(cqc["reasons"]), "QC": cqc}
                    else:
                        row["time_shift_control"] = {"status": "available", "shift_seconds": -10.0,
                            "eeg_path": control_path, "eeg_index": len(controls), "source_eeg_sha256": record["source_eeg_sha256"],
                            "eeg_payload_sha256": record["eeg_payload_sha256"], "source_recording_id": record["source_recording_id"],
                            "QC": cqc, **timing(shifted, fs, record["source_recording_id"])}
                        controls.append(control)
                    rows.append(row)
            if arrays:
                main_art[main_path] = save_array(output, main_path, arrays, meta, phase_root)
            if controls:
                ctrl_art[control_path] = save_array(output, control_path, controls, meta, phase_root)
            if idle_arrays:
                idle_art[idle_path] = save_array(output, idle_path, idle_arrays, meta, phase_root)
            for row in rows:
                row["array_sha256"] = main_art[main_path]["sha256"]
                if row["time_shift_control"]["status"] == "available":
                    row["time_shift_control"]["array_sha256"] = ctrl_art[control_path]["sha256"]
            for row in idle_rows:
                row["array_sha256"] = idle_art[idle_path]["sha256"]
            main.extend(rows)
            idle.extend(idle_rows)
            if len(sources) % 50 == 0:
                print(json.dumps({"stage": "prepare", "records": len(sources), "main_windows": len(main), "elapsed_seconds": time.time()-started}), flush=True)
        # A second independent streaming hash of every actually used original member.
        for source in sources:
            guard(meta, phase_root)
            h = hashlib.sha256()
            with z.open("files/"+source["member"]) as f:
                for chunk in iter(lambda:f.read(2**20), b""):
                    h.update(chunk)
            require(h.hexdigest() == source["source_sha256"], "Used original member changed at final verification")
    verify_archive(archive, meta)
    save_jsonl(output/"manifest.jsonl", main)
    save_jsonl(output/"idle_manifest.jsonl", idle)
    save_jsonl(output/"excluded_windows.jsonl", rejects)
    save_jsonl(output/"excluded_recordings.jsonl", held)
    save_jsonl(output/"source_records.jsonl", sources)
    paired = [{"id": r["id"], "split": r["split"], "task_family": r["task_family"]} for r in main if r["time_shift_control"]["status"] == "available"]
    save_jsonl(output/"paired_control_subset.jsonl", paired)
    for name, value in (("artifacts.json", main_art), ("control_artifacts.json", ctrl_art), ("idle_artifacts.json", idle_art)):
        save_json(output/name, value)
    for name in ("metadata_protocol.json", "preprocessing_protocol.json", "idle_metadata_protocol.json", "subject_split.json"):
        save_bytes(output/name, (Path(phase_root)/"metadata"/name).read_bytes())
    artifacts = {p.name:{"sha256":sha_file(p), "bytes":p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}
    report = {"schema": "pj-xavier-eegmmidb-phase2-corpus-v1", "status": "completed", "preprocessing_protocol_sha256": proto_sha,
        "metadata_protocol_sha256": meta_sha, "prep_source_sha256": sha_file(__file__), "versions": versions(),
        "original_archive_SHA_before_after_verified": meta["source_archive_sha256"], "all_used_EDF_member_SHA_verified_before_and_after": True,
        "source_recordings_decoded": len(sources), "source_recording_seconds": sum(s["source_seconds"] for s in sources),
        "source_member_bytes": sum(s["member_bytes"] for s in sources), "source_EEG_sample_bytes": sum(s["EEG_sample_bytes"] for s in sources),
        "split_assignments_original_counts": meta["split"]["counts"], "main": group_summary(main),
        "by_split": {s:group_summary([r for r in main if r["split"]==s]) for s in ("train", "val", "test")},
        "idle_test_only_diagnostic": group_summary(idle), "idle_not_training_or_selection": True,
        "controls_available": len(paired), "controls_blocked": len(main)-len(paired), "main_rows_dropped_for_control_availability": 0,
        "rejection_counts": dict(Counter(reason for r in rejects for reason in r["reasons"])),
        "rejection_by_split_role": dict(Counter(r["split"]+":"+r["role"]+":"+reason for r in rejects for reason in r["reasons"])),
        "held_recording_count": sum(r["status"]=="held_annotation_review" for r in held), "automatic_annotation_repairs": 0,
        "array_bytes": {"main":sum(x["bytes"] for x in main_art.values()), "control":sum(x["bytes"] for x in ctrl_art.values()), "idle":sum(x["bytes"] for x in idle_art.values())},
        "file_seals": artifacts, "raw_3s_intervals_distinct_from_causal_history": True,
        "model_forward_calls": 0, "test_performance_used_for_QC_or_protocol": False,
        "elapsed_seconds": time.time()-started, "completed_epoch": time.time(), "original_deadline_epoch": meta["CPU_absolute_deadline_epoch"]}
    guard(meta, phase_root, len(stable_bytes(report))+4096)
    save_json(output/"corpus_summary.json", report)
    save_json(output/"completion.json", {"status": "completed", "manifest_sha256": sha_file(output/"manifest.jsonl"),
              "corpus_summary_sha256": sha_file(output/"corpus_summary.json"), "main_artifacts_sha256": sha_file(output/"artifacts.json"),
              "completed_epoch": time.time(), "deadline_epoch": meta["CPU_absolute_deadline_epoch"]})
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("diagnose-train", "seal", "prepare"))
    p.add_argument("--phase-root", required=True)
    p.add_argument("--archive")
    p.add_argument("--tests-receipt")
    args = p.parse_args()
    if args.command == "seal":
        require(args.tests_receipt is not None, "Tests receipt required")
        seal_preprocessing(args.phase_root, args.tests_receipt)
    else:
        require(args.archive is not None, "Immutable archive required")
        (diagnose_train if args.command == "diagnose-train" else prepare)(args.archive, args.phase_root)


if __name__ == "__main__":
    main()
