#!/usr/bin/env python3
"""Prepare an immutable, SMAC-only *offline* CBraMod SSL corpus from ds005028 original DAT.

All 105 original BCI2000 DAT records are required in the frozen audit ledger.
Only author-eligible train groups are decoded. Original states, oracle labels,
decoder outputs, future text, and paired EDF exports never enter SSL tensors.
Plain BCI2000 state timing is *not* certified; this is not an online harness.
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

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE.parent / "acquisition"))
sys.path.insert(0, str(HERE.parent / "model"))
import audit_p300_bci2000 as native
import cbramod_ssl_smac as trainer
import prepare_smac_fif as common
from prepare_smac_fif import FS_OUT, PATCHES, PEAK_V, MIN_STD_V, require, rows, dump, sha, read_denies

RELEASE = "openneuro-ds005028-current-public-S3-e61cb6abf724"
CATALOG = "000233"
CHANNELS = list(native.SCALP)
NATIVE_HZ = 256
REFERENCE = "BCI2000_declared_acquisition_reference_no_rereference"
HOLD = {"openneuro:ds005028:sub-01": "test",
        "openneuro:ds005028:sub-10": "test",
        "openneuro:ds005028:sub-11": "val"}
AUTHOR_EXCLUDED = "openneuro:ds005028:sub-05"
SCIENTIFIC_SCOPE = ("Offline masked waveform SSL with software-header-calibrated V; "
                    "no online source causality, stimulus/event timing, state labels, "
                    "free-form intent or action success certified")


def plan_records(ledger, metadata, deny_sources, deny_payloads, deny_groups):
    require(len(ledger) == 105 and len({r["source_path"] for r in ledger}) == 105,
            "Frozen original 105 DAT records required")
    m = [r for r in metadata if r.get("source") == "p30032"]
    require(len(m) == 105 and len({r["source_member_or_path"] for r in m}) == 105,
            "Frozen 105 metadata-only P300 assignments required")
    by_path = {r["source_member_or_path"]: r for r in m}
    observed = Counter()
    plan = []
    identities = {}
    for rec in sorted(ledger, key=lambda r: r["source_path"]):
        name, group = rec["source_path"], rec["subject_group"]
        assert group.startswith("openneuro:ds005028:sub-")
        meta = by_path.get(name)
        require(meta is not None and meta["source_group"] == "p30032|" + group,
                "Frozen source path/group differs")
        require(meta["source_file_sha256"] == rec["source_file_sha256"]
                and meta["eeg_payload_sha256"] == rec["EEG_payload_sha256"]
                and meta["calibration_sha256"] == rec["calibration_sha256"]
                and meta["channel_count"] == 32 and meta["native_hz"] == NATIVE_HZ,
                "Frozen payload/calibration metadata differs")
        split = HOLD.get(group, "train")
        require(meta["metadata_only_proposed_split"] == split,
                "Frozen metadata-only subject split differs")
        require(rec["source_channels"] == 32 and rec["channel_names"] == CHANNELS
                and rec["sample_rate_hz"] == NATIVE_HZ and rec["scalar_dtype"] == "<f4"
                and rec["scalar_bytes"] == 4 and rec["waveform_filters_or_rescaling_written"] is False,
                "Unexpected original DAT signal layout")
        require(rec["calibration_formula"] == "physical_microV=(raw_AD-SourceChOffset)*SourceChGain"
                and len(rec["gain_AD_to_microV"]) == 32 and len(rec["offset_AD_units"]) == 32,
                "Unreviewed BCI2000 calibration")
        pair = (rec["EEG_payload_sha256"], rec["calibration_sha256"])
        require(pair not in identities or identities[pair] == group,
                "Duplicate EEG across frozen P300 subject groups")
        identities[pair] = group
        observed[split] += 1
        if split != "train":
            status, reason = "excluded_frozen_holdout", "val_test_DAT_never_opened"
        elif group == AUTHOR_EXCLUDED:
            status, reason = "excluded_author_analysis_flag", "subject5_excluded_by_publisher"
        elif (rec["source_file_sha256"] in deny_sources
              or rec["EEG_payload_sha256"] in deny_payloads
              or group in deny_groups or "p30032|" + group in deny_groups):
            status, reason = "excluded_prior_identity", "prior_heldout_or_review_deny"
        else:
            status, reason = "candidate", "must_reverify_source_physicalV_and_fixed_train_QC"
        plan.append({"record": rec, "split": split, "status": status,
                     "reason": reason, "component_id_sha256": meta["component_id_sha256"]})
    require(observed == Counter({"train": 77, "test": 19, "val": 9}),
            "Original 105 participant-level split counts changed")
    return plan


def validate_base(path, expected):
    path = Path(path).expanduser().resolve()
    require(sha(path / "completion.json") == expected,
            "Base corpus completion SHA differs")
    completion = json.loads((path / "completion.json").read_text())
    require(completion["status"] == "prepared_verified_new_offline_SSL_corpus",
            "Base is not immutable verified offline SSL")
    for name, field in (("manifest.jsonl", "manifest_sha256"),
                        ("artifacts.json", "artifacts_sha256"),
                        ("source_admission.json", "source_admission_sha256"),
                        ("protocol.json", "protocol_sha256")):
        require(sha(path / name) == completion[field], "Base "+name+" seal differs")
    trainer.load_corpus(path / "manifest.jsonl", path / "artifacts.json",
                        path / "source_admission.json")
    manifest = rows(path / "manifest.jsonl")
    require(all(r.get("online_harness_allowed") is False for r in manifest),
            "Base must remain offline-only")
    return path, manifest, json.loads((path / "source_admission.json").read_text()), json.loads((path / "artifacts.json").read_text())


def read_original_eeg_volts(path, rec):
    """Read only DAT signal bytes; state bytes are skipped without interpretation."""
    path = Path(path)
    before = path.stat()
    require(before.st_size == rec["source_file_bytes"], "Original DAT byte size differs")
    h, head = native.dat_header(path)
    for key in ("header_sha256", "header_bytes", "frame_bytes", "sample_count",
                "sample_rate_hz", "source_channels", "scalar_dtype", "scalar_bytes",
                "channel_names", "gain_AD_to_microV", "offset_AD_units",
                "state_bytes_per_sample", "sample_block_size"):
        require(h[key] == rec[key], "Original BCI2000 header differs: "+key)
    gains = np.asarray(h["gain_AD_to_microV"], np.float64)
    offsets = np.asarray(h["offset_AD_units"], np.float64)
    require(np.isfinite(gains).all() and np.isfinite(offsets).all()
            and np.all(gains > 0), "Invalid original header calibration")
    whole, eeg = hashlib.sha256(head), hashlib.sha256()
    physical = hashlib.sha256()
    values = []
    with path.open("rb") as stream:
        stream.seek(h["header_bytes"])
        total = 0
        while total < h["sample_count"]:
            n = min(8192, h["sample_count"] - total)
            blob = stream.read(n * h["frame_bytes"])
            require(len(blob) == n * h["frame_bytes"], "Truncated DAT frames")
            whole.update(blob)
            frames = np.frombuffer(blob, dtype=np.uint8).reshape(n, h["frame_bytes"])
            raw_bytes = np.ascontiguousarray(frames[:, :32 * 4]).tobytes()
            eeg.update(raw_bytes)
            raw = np.frombuffer(raw_bytes, dtype="<f4").reshape(n, 32)
            require(np.isfinite(raw).all(), "Nonfinite original DAT EEG")
            volts = (raw.astype(np.float64) - offsets[None, :]) * gains[None, :] * 1e-6
            require(np.isfinite(volts).all(), "Nonfinite calibrated DAT V")
            physical.update(np.ascontiguousarray(volts, dtype="<f8").tobytes())
            values.append(volts.T.copy())
            total += n
        require(stream.read(1) == b"", "Unexpected DAT tail")
    require(whole.hexdigest() == rec["source_file_sha256"]
            and eeg.hexdigest() == rec["EEG_payload_sha256"],
            "Retained original DAT or original EEG payload SHA mismatch")
    calibration = {"channels": h["channel_names"], "gain": h["gain_AD_to_microV"],
                   "offset": h["offset_AD_units"]}
    require(hashlib.sha256(json.dumps(calibration, sort_keys=True,
                      separators=(",", ":")).encode()).hexdigest() == rec["calibration_sha256"],
            "Original header-calibration fingerprint mismatch")
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns),
            "DAT changed during decoding")
    x = np.concatenate(values, axis=1)
    require(x.shape == (32, rec["sample_count"]), "Decoded original sample matrix differs")
    return x, {"original_EEG_payload_sha256": eeg.hexdigest(),
               "decoded_calibrated_V_sha256": physical.hexdigest(),
               "source_sha256": whole.hexdigest(),
               "calibration_sha256": rec["calibration_sha256"],
               "unit_scope": "BCI2000 SourceChGain(uV) and SourceChOffset software-header conversion to V; amplifier calibration unverified",
               "stimulus_or_action_state_bytes_decoded": False,
               "source_causality_certified": False, "online_harness_allowed": False,
               "original_peak_V": float(np.max(np.abs(x)))}


def windows32(x):
    x = np.asarray(x, np.float64)
    require(x.ndim == 2 and x.shape[0] == 32, "Expected 32 EEG channels")
    require(np.isfinite(x).all(), "Original source must be finite")
    require(float(np.max(np.abs(x))) <= 0.1, "native_peak_above_fixed_0_1V")
    sos = signal.butter(4, (3, 40), btype="bandpass", fs=NATIVE_HZ, output="sos")
    anti = signal.butter(6, 80, btype="lowpass", fs=NATIVE_HZ, output="sos")
    y = signal.sosfilt(anti, signal.sosfilt(sos, x, axis=1), axis=1)
    js = np.arange(2 * FS_OUT, ((x.shape[1] - 1) * FS_OUT) // NATIVE_HZ + 1, dtype=np.int64)
    n = (js * NATIVE_HZ) // FS_OUT
    frac = ((js * NATIVE_HZ) % FS_OUT).astype(np.float64) / FS_OUT
    z = y[:, n-1] * (1-frac) + y[:, n] * frac
    arrays, intervals, rejected = [], [], Counter()
    width = PATCHES * FS_OUT
    for start in range(0, len(js) - width + 1, width):
        w = z[:, start:start+width]
        if not np.isfinite(w).all():
            rejected["nonfinite"] += 1
        elif np.max(np.abs(w)) > PEAK_V:
            rejected["peak_above_fixed_100uV"] += 1
        elif np.any(w.std(axis=1, ddof=0) < MIN_STD_V):
            rejected["flat_below_fixed_0_01uV"] += 1
        else:
            arrays.append((w * 1e4).reshape(32, PATCHES, FS_OUT).astype(np.float32))
            intervals.append({"window_start_s": int(js[start]) / FS_OUT,
                              "window_end_s": (int(js[start+width-1]) + 1) / FS_OUT,
                              "latest_native_source_index": int(n[start+width-1]),
                              "feature_available_native_s": int(n[start+width-1]) / NATIVE_HZ})
    a = np.stack(arrays) if arrays else np.empty((0, 32, PATCHES, FS_OUT), np.float32)
    return a, intervals, dict(rejected)


def prepare(args):
    out = Path(args.output).expanduser().absolute()
    raw = Path(args.raw_root).expanduser().resolve()
    require(not out.exists(), "Immutable new output directory required")
    for file, expected in ((args.source_ledger, args.expected_ledger_sha256),
                           (args.metadata_admission, args.expected_metadata_sha256),
                           (args.audit_protocol, args.expected_protocol_sha256)):
        require(sha(file) == expected, "Frozen audit or metadata seal differs: "+str(file))
    audit = json.loads(Path(args.audit_protocol).read_text())
    require(audit["schema"] == "xavier_P300_original_DAT_audit_v1"
            and audit["canonical_format"] == "BCI2000 DAT"
            and audit["source_release_key"] == RELEASE
            and sha(native.__file__) == audit["source_sha256"],
            "Original BCI2000 parser/protocol seal differs")
    base, bm, ba, bart = validate_base(args.base_corpus, args.expected_base_completion_sha256)
    ds, dp, dg, input_hashes = read_denies(args.deny_json)
    ds.update(ba["excluded_source_sha256"]); dp.update(ba["excluded_payload_sha256"])
    plan = plan_records(rows(args.source_ledger), rows(args.metadata_admission), ds, dp, dg)
    for item in plan:
        if item["status"] != "candidate":
            ds.add(item["record"]["source_file_sha256"])
            dp.add(item["record"]["EEG_payload_sha256"])
    prior_sources = set(ba["sources"])
    prior_payloads = {x["eeg_payload_sha256"] for x in ba["sources"].values()}
    require(not prior_sources.intersection(ds) and not prior_payloads.intersection(dp),
            "Prior denied identity overlaps sealed base")
    needed = 512 * 2**20 + sum(x["bytes"] for x in bart.values()) + sum(
        int(p["record"]["sample_count"] / NATIVE_HZ * FS_OUT * 32 * 4)
        for p in plan if p["status"] == "candidate")
    require(args.reserve_gib >= 1 and shutil.disk_usage(raw).free >= args.reserve_gib * 2**30 + needed,
            "Insufficient reserve for bounded P300 preparation and exact base copy")
    out.mkdir(parents=True); (out/"waveforms").mkdir()
    protocol = {"schema": "xavier-smac-p30032-offline-SSL-v1",
       "code_sha256": sha(__file__), "frozen_original_DAT_ledger_sha256": args.expected_ledger_sha256,
       "metadata_admission_sha256": args.expected_metadata_sha256,
       "audit_protocol_sha256": args.expected_protocol_sha256,
       "parser_source_sha256": sha(native.__file__),
       "trainer_sha256": sha(trainer.__file__), "base_completion_sha256": args.expected_base_completion_sha256,
       "prior_denies_sha256": input_hashes, "source": RELEASE, "catalog": CATALOG,
       "original_DAT_records": 105, "subject_level_split": {"train": 77, "val": 9, "test": 19},
       "author_analysis_excluded_group": AUTHOR_EXCLUDED,
       "new_val_or_test_DAT_opened": 0, "new_author_excluded_DAT_opened": 0,
       "only_original_DAT_not_derived_EDF": True,
       "calibration": "BCI2000 header raw_AD-offset then gain_uV then x1e-6 V; no independent amplifier verification",
       "QC": {"native_peak_V": .1, "postprocess_peak_V": PEAK_V,
              "minimum_channel_std_V": MIN_STD_V,
              "fixed_before_new_signal_read": True, "no_clipping_or_unit_guess": True},
       "preprocessing": {"native_hz": NATIVE_HZ, "output_hz": FS_OUT, "channels": CHANNELS,
              "bandpass": "causal Butterworth4 SOS 3-40Hz zero-state",
              "anti_alias": "causal Butterworth6 SOS 80Hz zero-state",
              "resampling": "one-native-sample-delayed convex linear interpolation",
              "warmup_seconds": 2, "nonoverlap_window_seconds": PATCHES, "dtype": "float32",
              "output_unit": "V_x_1e4", "reference": REFERENCE},
       "source_original_causality_certified": False, "online_harness_allowed": False,
       "labels_used": False, "states_decoded": False,
       "validation_domain": "retained 806023 only; P300 new val/test remain untouched",
       "scope": SCIENTIFIC_SCOPE, "created_epoch": time.time()}
    dump(out/"protocol.json", protocol)
    montage = hashlib.sha256(json.dumps([CHANNELS, REFERENCE, NATIVE_HZ, RELEASE, "physical_V"],
                            sort_keys=True).encode()).hexdigest()
    manifest, new_sources, artifacts, certs, new_payloads = [], {}, {}, [], set()
    base_groups = {r["subject_group"] for r in bm if r["split"] != "train"}
    for p in plan:
        rec = p["record"]
        cert = {"source_path": rec["source_path"], "source_sha256": rec["source_file_sha256"],
                "subject_group": rec["subject_group"], "split": p["split"],
                "initial_state": p["status"], "initial_reason": p["reason"],
                "signal_opened": False}
        if p["status"] != "candidate":
            certs.append(cert); continue
        rel = Path(rec["source_path"])
        require(not rel.is_absolute() and ".." not in rel.parts and rel.suffix == ".dat",
                "Unsafe original DAT relative path")
        path = (raw/rel).resolve()
        require(raw in path.parents and not path.is_symlink(), "Source DAT escaped raw root")
        x, proof = read_original_eeg_volts(path, rec)
        cert.update(proof, signal_opened=True)
        if (proof["decoded_calibrated_V_sha256"] in dp
            or proof["decoded_calibrated_V_sha256"] in prior_payloads
            or proof["decoded_calibrated_V_sha256"] in new_payloads
            or rec["EEG_payload_sha256"] in dp
            or rec["source_file_sha256"] in prior_sources
            or rec["subject_group"] in base_groups):
            cert.update(status="excluded_identity", reason="prior_or_new_calibrated_EEG_or_group_identity")
            certs.append(cert); del x; continue
        try:
            array, intervals, rejected = windows32(x)
        except ValueError as e:
            cert.update(status="held_QC", reason=str(e)); certs.append(cert); del x; continue
        finally:
            del x
        cert.update(accepted_windows=len(intervals), rejected_windows=rejected)
        if not intervals:
            cert.update(status="held_QC", reason="no_window_passed_fixed_train_QC")
            certs.append(cert); continue
        relarray = "waveforms/" + rec["source_file_sha256"] + ".npy"
        np.save(out/relarray, array, allow_pickle=False)
        artifacts[relarray] = {"shape": list(array.shape), "dtype": "float32",
                               "bytes": (out/relarray).stat().st_size, "sha256": sha(out/relarray)}
        physical_sha = proof["decoded_calibrated_V_sha256"]
        new_payloads.add(physical_sha)
        new_sources[rec["source_file_sha256"]] = {
            "split": "train", "subject_group": rec["subject_group"],
            "eeg_payload_sha256": physical_sha, "montage_key": montage,
            "physical_unit_verified": True, "fit_allowed": True,
            "validation_allowed": False, "online_harness_allowed": False,
            "source_recording_id": "openneuro:ds005028:"+rec["source_path"],
            "unit_verification_scope": proof["unit_scope"],
            "original_integer_EEG_payload_sha256": rec["EEG_payload_sha256"]}
        for idx, interval in enumerate(intervals):
            manifest.append({"id": rec["source_file_sha256"]+":"+str(idx),
               "split": "train", "subject_group": rec["subject_group"],
               "source_sha256": rec["source_file_sha256"],
               "eeg_payload_sha256": physical_sha,
               "source_recording_id": "openneuro:ds005028:"+rec["source_path"],
               **interval, "channels": CHANNELS, "sample_rate_hz": FS_OUT,
               "unit": "V_x_1e4", "eeg_path": relarray, "eeg_index": idx,
               "montage_key": montage, "reference": REFERENCE,
               "native_sample_rate_hz": NATIVE_HZ, "physical_unit_verified": True,
               "source_causality": "original_DAT_hardware_and_source_processing_uncertified",
               "online_harness_allowed": False})
        cert.update(status="admitted_offline_SSL",
                    prepared_seconds=len(intervals)*PATCHES)
        certs.append(cert)
    dump(out/"source_certificates.json", certs)
    if not manifest:
        result = {"status": "blocked_no_eligible_new_source_windows",
             "new_source_windows": 0, "new_source_records": 0,
             "new_source_unique_hours": 0, "new_gradient_exposure_hours": 0,
             "new_val_or_test_DAT_opened": 0,
             "reasons": dict(Counter(c.get("reason") for c in certs
                      if c.get("status") in ("held_QC","excluded_identity"))),
             "protocol_sha256": sha(out/"protocol.json"),
             "certificates_sha256": sha(out/"source_certificates.json")}
        dump(out/"preparation_result.json", result)
        return result
    for rel, spec in bart.items():
        dest = out/"base"/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base/rel, dest)
        require(sha(dest) == spec["sha256"], "Base array copy SHA mismatch")
        artifacts["base/"+rel] = spec
    manifest = [{**r, "eeg_path": "base/"+r["eeg_path"]} for r in bm] + manifest
    (out/"manifest.jsonl").write_text("".join(
        json.dumps(r, sort_keys=True, allow_nan=False)+"\n" for r in manifest))
    dump(out/"artifacts.json", artifacts)
    dump(out/"source_admission.json", {
      "schema": "xavier-smac-p30032-offline-SSL-admission-v1",
      "sources": {**ba["sources"], **new_sources},
      "excluded_source_sha256": sorted(ds),
      "excluded_payload_sha256": sorted(dp),
      "protocol_sha256": sha(out/"protocol.json"),
      "online_harness_allowed": False,
      "validation_domain": "retained 806023 only; P300 generalization not established",
      "raw_EEG_signal_only": True})
    trainer.load_corpus(out/"manifest.jsonl",out/"artifacts.json",out/"source_admission.json")
    new_n = sum(v["shape"][0] for k,v in artifacts.items() if not k.startswith("base/"))
    result = {"status": "prepared_verified_new_offline_SSL_corpus",
       "windows": dict(Counter(r["split"] for r in manifest)),
       "new_source_windows": new_n, "new_source_records": len(new_sources),
       "new_source_unique_hours": new_n*PATCHES/3600,
       "retained_base_windows": len(bm), "new_val_or_test_DAT_opened": 0,
       "new_author_excluded_DAT_opened": 0,
       "P300_generalization_evaluated": False, "online_harness_allowed": False,
       "gradient_exposure_hours": 0,
       "protocol_sha256": sha(out/"protocol.json"),
       "manifest_sha256": sha(out/"manifest.jsonl"),
       "artifacts_sha256": sha(out/"artifacts.json"),
       "source_admission_sha256": sha(out/"source_admission.json"),
       "certificates_sha256": sha(out/"source_certificates.json"),
       "base_completion_sha256": args.expected_base_completion_sha256}
    dump(out/"completion.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("raw-root", "source-ledger", "metadata-admission", "audit-protocol",
              "expected-ledger-sha256", "expected-metadata-sha256",
              "expected-protocol-sha256", "base-corpus",
              "expected-base-completion-sha256", "output"):
        p.add_argument("--"+k, required=True)
    p.add_argument("--deny-json", action="append", required=True)
    p.add_argument("--reserve-gib", type=float, default=100)
    args = p.parse_args()
    from smac_storage_guard import require_smac_raw_root
    evidence = require_smac_raw_root(args.raw_root)
    node_root = (Path.home()/"pj-xavier"/evidence["node_id"]).resolve()
    require(node_root in Path(args.output).expanduser().absolute().parents,
            "Prepared output must be inside assigned SMAC node workspace")
    result = prepare(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False),flush=True)
    if result["status"].startswith("blocked"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
