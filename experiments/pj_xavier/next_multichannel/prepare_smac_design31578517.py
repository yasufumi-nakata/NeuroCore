#!/usr/bin/env python3
"""Prepare all frozen train EDFs from Figshare 31578517 v2 after the BCI corpus.

The 17/5/3 participant split was sealed before any signal decoding. Only train
EDFs are opened. Native rates use the exact header sample-count/duration ratio;
EDF digital calibration is independently checked against MNE at three intervals.
Questionnaires, event labels, val/test signals and publisher scripts are not used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import mne
import numpy as np
import scipy
from scipy import signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "acquisition"))
sys.path.insert(0, str(HERE.parent / "model"))
import cbramod_ssl_smac as trainer
import freeze_smac_design31578517_metadata as frozen
import prepare_smac_fif as common
from prepare_smac_fif import FS_OUT, PATCHES, PEAK_V, MIN_STD_V, dump, read_denies, require, rows, sha

PLAN_SHA256 = "42e0c2470d1415661a2aa6a0a493375788349388095cacd6b79c3ddc3e5d73d7"
FREEZE_CODE_SHA256 = "d4d1b03daa54878d4064df263d2107d445401a16c69d3d52497a33deae4d5f6a"
SOURCE = frozen.SOURCE
CHANNELS = frozen.CHANNELS
REFERENCE = "publisher_REST_or_average_reference_not_independently_verified_no_rereference"
N_EEG = 32
FORMULA = "V=((digital-digital_min)*(physical_max_uV-physical_min_uV)/(digital_max-digital_min)+physical_min_uV)*1e-6"


PAYLOAD_KEYS = ("eeg_payload_sha256", "EEG_payload_sha256", "decoded_EEG_V_sha256",
                "original_integer_EEG_payload_sha256", "original_EEG_scalar_payload_sha256")
GROUP_DENY_KEYS = ("excluded_source_groups", "excluded_subject_groups", "denied_source_groups",
                   "denied_subject_groups")


def identities(record):
    return {record[k] for k in PAYLOAD_KEYS if isinstance(record.get(k), str) and len(record[k]) == 64}


def explicit_group_denies(value, inherited=False):
    """Retain explicit lists and original val/test/review groups in ancestor proof."""
    found = set()
    if isinstance(value, dict):
        fold = value.get("split", value.get("metadata_only_proposed_split",
                                           value.get("original_partition", value.get("ssl_split"))))
        status = str(value.get("status", "")).lower()
        # A rejected train alias/QC record does not turn its otherwise eligible
        # train participant into a heldout group. Explicit group lists still win.
        held = fold in {"val", "test", "review", "review_hold"}
        if fold != "train":
            held = held or inherited or "review" in status or status.startswith(("held", "excluded"))
        if held:
            for key in ("subject_group", "participant_group", "source_group"):
                if isinstance(value.get(key), str):
                    found.add(value[key])
        for key, item in value.items():
            if key in GROUP_DENY_KEYS:
                require(isinstance(item, list) and all(isinstance(v, str) for v in item),
                        "Invalid historical group deny list")
                found.update(item)
            elif isinstance(item, (dict, list)):
                found.update(explicit_group_denies(item, held))
    elif isinstance(value, list):
        for item in value:
            found.update(explicit_group_denies(item, inherited))
    return found


def validate_base(path, expected_completion_sha):
    base = Path(path).expanduser().resolve()
    require(sha(base/"completion.json") == expected_completion_sha, "Base completion SHA changed")
    completion = json.loads((base/"completion.json").read_text())
    require(completion["status"] == "prepared_verified_new_offline_SSL_corpus",
            "Base is not a verified SSL corpus")
    for name, key in (("manifest.jsonl", "manifest_sha256"), ("artifacts.json", "artifacts_sha256"),
                      ("source_admission.json", "source_admission_sha256"), ("protocol.json", "protocol_sha256")):
        require(sha(base/name) == completion[key], "Base metadata seal changed: " + name)
    optional = {}
    for name, key in (("source_certificates.json", "certificates_sha256"),
                      ("split_manifest.json", "split_manifest_sha256"),
                      ("parent_provenance.json", "parent_provenance_sha256")):
        if key in completion:
            require(sha(base/name) == completion[key], "Base optional metadata seal changed: " + name)
            optional[name] = json.loads((base/name).read_text())
        else:
            require(not (base/name).exists(), "Unsealed historical metadata: " + name)
    trainer.load_corpus(base/"manifest.jsonl", base/"artifacts.json", base/"source_admission.json")
    manifest = rows(base/"manifest.jsonl")
    require(all(r.get("online_harness_allowed") is False for r in manifest), "Base must remain offline-only")
    require(all(r["source_recording_id"].startswith("zenodo:806023:") for r in manifest if r["split"] == "val"),
            "Preserve only the existing 806023 validation domain")
    admission = json.loads((base/"source_admission.json").read_text())
    provenance = {"completion": completion, "completion_sha256": expected_completion_sha,
                  "optional_metadata": optional, "source_admission": admission,
                  "protocol": json.loads((base/"protocol.json").read_text())}
    return base, manifest, admission, json.loads((base/"artifacts.json").read_text()), provenance


def load_plan(path):
    require(sha(path) == PLAN_SHA256, "Frozen metadata plan seal differs")
    require(sha(frozen.__file__) == FREEZE_CODE_SHA256, "Frozen split implementation changed")
    plan = json.loads(Path(path).read_text())
    require(plan["schema"] == frozen.SCHEMA and plan["source"] == SOURCE,
            "Wrong frozen source plan")
    require(plan["provider_metadata_sha256"] == frozen.METADATA_SHA256
            and plan["header_evidence_sha256"] == frozen.HEADER_EVIDENCE_SHA256,
            "Original provider/header metadata bindings changed")
    when = datetime.fromisoformat(plan["frozen_at_utc"])
    require(when.tzinfo is not None and when <= datetime.now(timezone.utc),
            "Metadata must be frozen before current signal reads")
    require(plan["metadata_signal_reads"] == 0
            and plan["questionnaire_or_label_tables_opened"] == 0
            and plan["newsource_val_and_test_signals_must_remain_unopened"] is True,
            "Invalid pre-signal split contract")
    records = plan["records"]
    require(len(records) == 25 and len({r["subject_group"] for r in records}) == 25
            and len({r["source_sha256"] for r in records}) == 25,
            "All original participant identities required")
    require(dict(Counter(r["split"] for r in records)) == frozen.SPLIT_COUNTS,
            "Frozen participant counts changed")
    ordered = sorted(records, key=lambda r: r["group_id_sha256"])
    for rank, r in enumerate(ordered):
        split = "train" if rank < 17 else ("val" if rank < 22 else "test")
        require(r["hash_rank"] == rank and r["split"] == split
                and r["signal_opened_before_freeze"] is False,
                "Frozen pre-signal group assignment changed")
    return plan


def read_header(path):
    with Path(path).open("rb") as f:
        fixed = f.read(256)
        require(len(fixed) == 256 and fixed[:8] == b"0       ", "Not original EDF")
        count = int(fixed[252:256].decode("ascii").strip())
        hsize = int(fixed[184:192].decode("ascii").strip())
        require(count == N_EEG and hsize == (count + 1) * 256,
                "Unreviewed original EDF channel geometry")
        data = f.read(hsize - 256)
        require(len(data) == hsize - 256, "Truncated original EDF header")
    c, offset = {}, 0
    for key, width in (("label", 16), ("transducer", 80), ("unit", 8),
                       ("pmin", 8), ("pmax", 8), ("dmin", 8), ("dmax", 8),
                       ("prefilter", 80), ("samples", 8), ("reserved", 32)):
        c[key] = [data[offset + i * width:offset + (i+1) * width].decode("latin1").strip()
                  for i in range(count)]
        offset += width * count
    nrecords = int(fixed[236:244].decode("ascii").strip())
    duration = float(fixed[244:252].decode("ascii").strip())
    return {"n_channels": count, "header_bytes": hsize, "n_records": nrecords,
            "record_duration_s": duration, "duration_s": nrecords * duration,
            "channels": c}, fixed + data


def decode_train_edf(path, record):
    """Replay source identities and calibrate only a frozen train record."""
    require(record["split"] == "train", "Only frozen train EDF may be opened")
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Expected a regular original EDF")
    before = path.stat()
    require(before.st_size == record["source_bytes"], "Original EDF size changed")
    header, prefix = read_header(path)
    require(hashlib.sha256(prefix).hexdigest() == record["header_sha256"],
            "Original EDF header SHA changed")
    require(header == record["header"], "Original EDF calibration/geometry changed")
    c = header["channels"]
    require(c["label"] == CHANNELS and c["unit"] == ["uV"] * N_EEG,
            "Physical EEG units or channel identity changed")
    ns = [int(v) for v in c["samples"]]
    require(len(set(ns)) == 1 and header["n_records"] == 1,
            "Expected one uniform-channel record")
    rate = Fraction(ns[0], 1) / Fraction(str(header["record_duration_s"]))
    require(rate.numerator == record["native_rate_numerator"]
            and rate.denominator == record["native_rate_denominator"]
            and ns[0] == record["native_sample_count"], "Actual original sample rate changed")
    require(before.st_size == len(prefix) + sum(ns) * 2, "Unexpected EDF payload/tail size")
    whole, md5, digital_hash = hashlib.sha256(prefix), hashlib.md5(prefix), hashlib.sha256()
    with path.open("rb") as f:
        require(f.read(len(prefix)) == prefix, "Original header changed during read")
        blob = f.read(sum(ns) * 2)
        require(len(blob) == sum(ns) * 2 and f.read(1) == b"", "Truncated or trailing EDF bytes")
    whole.update(blob); md5.update(blob); digital_hash.update(blob)
    require(whole.hexdigest() == record["source_sha256"], "Original EDF SHA changed")
    require(md5.hexdigest() == record["published_md5"], "Official EDF checksum differs")
    digital = np.frombuffer(blob, dtype="<i2").reshape(N_EEG, ns[0])
    dmin = np.array([int(v) for v in c["dmin"]], np.float64)[:, None]
    dmax = np.array([int(v) for v in c["dmax"]], np.float64)[:, None]
    pmin = np.array([float(v) for v in c["pmin"]], np.float64)[:, None]
    pmax = np.array([float(v) for v in c["pmax"]], np.float64)[:, None]
    require(np.all(np.isfinite(pmin)) and np.all(np.isfinite(pmax))
            and np.all(pmax > pmin) and np.all(dmax > dmin),
            "Invalid physical calibration")
    x = ((digital.astype(np.float64) - dmin) * (pmax - pmin) / (dmax - dmin) + pmin) * 1e-6
    require(np.isfinite(x).all(), "Nonfinite decoded physical EEG volts")
    physical_hash = hashlib.sha256(np.ascontiguousarray(x.T, dtype="<f8").tobytes()).hexdigest()
    del blob, digital
    raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    try:
        require(raw.ch_names == CHANNELS and raw.n_times == x.shape[1]
                and abs(raw.info["sfreq"] - float(rate)) <= 1e-9,
                "Independent MNE channel/time geometry differs")
        starts = [0, x.shape[1] // 2, x.shape[1] - 7]
        differences = []
        for start in starts:
            observed = raw.get_data(picks=list(range(N_EEG)), start=start, stop=start+7,
                                    verbose="ERROR").astype(np.float64)
            differences.append(float(np.max(np.abs(observed - x[:, start:start+7]))))
        require(max(differences) <= 1e-11, "Independent MNE three-interval physical-V check failed")
    finally:
        raw.close()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino),
            "Original EDF changed during decode")
    calibration = {k: c[k] for k in ("label", "unit", "pmin", "pmax", "dmin", "dmax")}
    proof = {
        "physical_unit_verified": True,
        "verification_scope": "Declared original EDF uV calibration converted to V, full source/payload identity and MNE three-interval match; amplifier calibration unverified",
        "calibration_formula": FORMULA,
        "calibration_sha256": hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest(),
        "original_integer_EEG_payload_sha256": digital_hash.hexdigest(),
        "eeg_payload_sha256": physical_hash,
        "MNE_three_interval_difference_V": differences,
        "MNE_three_interval_physical_V_passed": True,
        "observed_native_peak_V": float(np.max(np.abs(x))),
        "native_sample_rate_hz": float(rate),
        "native_rate_numerator": rate.numerator, "native_rate_denominator": rate.denominator,
        "native_seconds": x.shape[1] / float(rate),
        "label_or_questionnaire_used": False, "source_causality_verified": False,
        "online_harness_allowed": False,
    }
    return x, proof


def filters(native_hz):
    return (signal.butter(4, (3, 40), btype="bandpass", fs=native_hz, output="sos"),
            signal.butter(6, 80, btype="lowpass", fs=native_hz, output="sos"))


def causal_transform(x, rate_numerator, rate_denominator):
    x = np.asarray(x, dtype=np.float64)
    require(x.ndim == 2 and x.shape[0] == N_EEG and x.shape[1] > 0,
            "Expected 32-channel physical EEG volts")
    require(np.isfinite(x).all(), "Nonfinite native physical EEG")
    require(float(np.max(np.abs(x))) <= .1, "native_peak_above_fixed_0_1V_source_bound")
    rate = Fraction(rate_numerator, rate_denominator)
    require(1023.99 <= float(rate) <= 1024.01, "Unreviewed original sample rate")
    bandpass, antialias = filters(float(rate))
    y = signal.sosfilt(antialias, signal.sosfilt(bandpass, x, axis=1), axis=1)
    last = ((x.shape[1] - 1) * FS_OUT * rate.denominator) // rate.numerator
    js = np.arange(2 * FS_OUT, last + 1, dtype=np.int64)
    # Exact rational cutoffs, one native sample delayed; no future native sample.
    numerator = js * rate.numerator
    denominator = FS_OUT * rate.denominator
    cutoff = numerator // denominator
    alpha = (numerator % denominator).astype(np.float64) / denominator
    z = y[:, cutoff-1] * (1-alpha) + y[:, cutoff] * alpha
    return z, js, cutoff


def causal_windows(x, rate_numerator, rate_denominator):
    z, js, cutoff = causal_transform(x, rate_numerator, rate_denominator)
    rate = rate_numerator / rate_denominator
    arrays, windows, rejects = [], [], Counter()
    width = PATCHES * FS_OUT
    for start in range(0, z.shape[1] - width + 1, width):
        w = z[:, start:start+width]
        if not np.isfinite(w).all():
            rejects["nonfinite"] += 1
        elif float(np.max(np.abs(w))) > PEAK_V:
            rejects["peak_above_fixed_100uV"] += 1
        elif np.any(w.std(axis=1, ddof=0) < MIN_STD_V):
            rejects["flat_below_fixed_0_01uV"] += 1
        else:
            arrays.append((w * 1e4).reshape(N_EEG, PATCHES, FS_OUT).astype(np.float32))
            windows.append({"window_start_s": int(js[start]) / FS_OUT,
                            "window_end_s": (int(js[start+width-1]) + 1) / FS_OUT,
                            "latest_native_source_index": int(cutoff[start+width-1]),
                            "feature_available_native_s": int(cutoff[start+width-1]) / rate})
    array = np.stack(arrays) if arrays else np.empty((0, N_EEG, PATCHES, FS_OUT), np.float32)
    return array, windows, dict(rejects)


def prepare(args):
    started = datetime.now(timezone.utc).isoformat()
    plan = load_plan(args.frozen_plan)
    raw_root = Path(args.raw_root).expanduser().resolve()
    out = Path(args.output).expanduser().absolute()
    require(not out.exists(), "Use a new immutable Design EDF corpus directory")
    base, base_rows, admission, base_artifacts, base_provenance = validate_base(
        args.base_corpus, args.expected_base_completion_sha256)
    require(any(r["source_recording_id"].startswith("zenodo:8089820:")
                and r["split"] == "train" for r in base_rows),
            "The cumulative post-BCI corpus is required; do not branch from an earlier parent")
    require(not any(r["source_recording_id"].startswith(SOURCE + ":") for r in base_rows),
            "This source is already present in the base corpus")
    ds, dp, dg, deny_inputs = read_denies(args.deny_json)
    ds.update(admission["excluded_source_sha256"])
    dp.update(admission["excluded_payload_sha256"])
    dg.update(explicit_group_denies(base_provenance))
    for deny_path in args.deny_json:
        value = rows(deny_path) if Path(deny_path).suffix == ".jsonl" else json.loads(Path(deny_path).read_text())
        dg.update(explicit_group_denies(value))
    base_sources = set(admission["sources"])
    base_payloads = set().union(*(identities(r) for r in admission["sources"].values()))
    base_held_groups = {r["subject_group"] for r in base_rows if r["split"] != "train"}
    require(not base_sources & ds and not base_payloads & dp,
            "Prior exclusion overlaps retained corpus")
    holds = [r for r in plan["records"] if r["split"] != "train"]
    ds.update(r["source_sha256"] for r in holds)
    dg.update(r["subject_group"] for r in holds)
    require(not base_sources & ds, "New heldout source identity overlaps retained corpus")
    max_new_bytes = int(sum(r["native_seconds"] for r in plan["records"]
                            if r["split"] == "train") * FS_OUT * N_EEG * 4)
    reserve = int(args.reserve_gib * 1024**3)
    require(args.reserve_gib >= 1 and shutil.disk_usage(raw_root).free >= reserve
            + max_new_bytes + sum(v["bytes"] for v in base_artifacts.values()) + 512*1024**2,
            "Insufficient fixed storage reserve")
    out.mkdir(parents=True); (out/"waveforms").mkdir()
    shutil.copyfile(args.frozen_plan, out/"split_manifest.json")
    require(sha(out/"split_manifest.json") == PLAN_SHA256, "Copied freeze plan changed")
    dump(out/"parent_provenance.json", base_provenance)
    protocol = {
        "schema": "xavier-smac-design31578517-offline-source-addition-v1",
        "started_at_utc": started, "metadata_frozen_at_utc": plan["frozen_at_utc"],
        "code_sha256": sha(__file__), "freeze_code_sha256": FREEZE_CODE_SHA256,
        "frozen_plan_sha256": PLAN_SHA256, "source": SOURCE, "catalog_id": frozen.CATALOG,
        "base_completion_sha256": args.expected_base_completion_sha256,
        "parent_provenance_sha256": sha(out/"parent_provenance.json"),
        "common_helpers_sha256": sha(common.__file__), "trainer_contract_sha256": sha(trainer.__file__),
        "prior_deny_input_hashes": deny_inputs, "split_counts": frozen.SPLIT_COUNTS,
        "source_causality_verified": False, "online_harness_allowed": False,
        "validation_domain": "retained 806023 only; no Design source generalization claim",
        "group_evidence": plan["group_evidence"],
        "questionnaire_events_labels_publisher_scripts_used": False,
        "QC": {"native_max_abs_V": .1, "postprocess_max_abs_V": PEAK_V,
               "minimum_channel_std_V": MIN_STD_V, "unit_correction_or_clipping": False,
               "fixed_before_current_signal_reads": True},
        "preprocessing": {
            "native_hz": "exact frozen samples_per_record / record_duration_s for each original EDF",
            "output_hz": FS_OUT, "EEG_channels": CHANNELS,
            "scipy_version": scipy.__version__, "bandpass_design_N": 4,
            "bandpass_Hz": [3, 40], "anti_alias_design_N": 6, "anti_alias_Hz": 80,
            "method": "forward-only SOS, zero initial state per original participant record",
            "rate_conversion": "one-native-sample delayed linear interpolation at exact rational native cutoffs",
            "warmup_seconds": 2, "window_seconds": PATCHES, "nonoverlapping": True,
            "output_shape": "[N,32,3,200]", "output_unit": "V_x_1e4",
            "reference": REFERENCE},
        "source_header_and_README_correct_abstract_channel_rate_discrepancy": True,
        "physical_calibration_formula": FORMULA,
        "physical_identity": "SHA256(sample-major little-endian float64 calibrated volts)",
        "cross_fold_payload_comparison_limit": "Val/test signals remain unopened. Their complete-file SHA identities and metadata groups are excluded; undisclosed aliases beyond metadata cannot be ruled out.",
    }
    dump(out/"protocol.json", protocol)
    new_manifest, new_sources, artifacts, certificates = [], {}, {}, []
    seen_payloads = set()
    montage = hashlib.sha256(json.dumps([CHANNELS, REFERENCE, SOURCE, "actual_header_rate_to_200Hz"]).encode()).hexdigest()
    for record in plan["records"]:
        cert = {k: record[k] for k in ("relative_path", "source_sha256", "subject_group", "split", "provider_file_id")}
        cert.update(signal_opened=False)
        if record["split"] != "train":
            cert.update(status="excluded_frozen_holdout", reason="new_source_val_test_signal_not_opened")
        elif record["source_sha256"] in ds or record["subject_group"] in dg | base_held_groups:
            cert.update(status="excluded_prior_identity", reason="prior_source_or_group_deny")
        elif record["source_sha256"] in base_sources:
            cert.update(status="excluded_base_identity", reason="original_source_already_represented")
        else:
            rel = Path(record["relative_path"])
            require(not rel.is_absolute() and ".." not in rel.parts and rel.name == str(rel)
                    and rel.suffix.lower() == ".edf", "Unsafe original EDF path")
            original = raw_root/rel
            require(not original.is_symlink() and raw_root in original.resolve().parents,
                    "Original EDF escaped source root")
            x, proof = decode_train_edf(original, record)
            cert.update(proof, signal_opened=True)
            new_identities = identities(proof)
            if new_identities & dp:
                cert.update(status="excluded_prior_identity", reason="prior_decoded_payload_deny")
                ds.add(record["source_sha256"]); dp.update(new_identities); del x
            elif new_identities & base_payloads:
                cert.update(status="excluded_base_identity", reason="decoded_payload_already_represented")
                del x
            else:
                require(not new_identities & seen_payloads,
                        "Duplicate new EEG payload requires metadata must-link review")
                seen_payloads.update(new_identities)
                if proof["observed_native_peak_V"] > .1:
                    cert.update(status="held_QC", reason="native_peak_above_fixed_0_1V_source_bound")
                    del x
                else:
                    array, windows, rejected = causal_windows(
                        x, record["native_rate_numerator"], record["native_rate_denominator"])
                    del x
                    cert.update(accepted_windows=len(windows), rejected_windows=rejected)
                    if not windows:
                        cert.update(status="held_QC", reason="no_window_meets_fixed_V_QC")
                    else:
                        relarray = "waveforms/" + record["source_sha256"] + ".npy"
                        require(shutil.disk_usage(out).free >= reserve + array.nbytes,
                                "Fixed storage reserve reached")
                        np.save(out/relarray, array, allow_pickle=False)
                        artifacts[relarray] = {"shape": list(array.shape), "dtype": "float32",
                                              "sha256": sha(out/relarray), "bytes": (out/relarray).stat().st_size}
                        entry = {"split": "train", "subject_group": record["subject_group"],
                                 "eeg_payload_sha256": proof["eeg_payload_sha256"],
                                 "original_integer_EEG_payload_sha256": proof["original_integer_EEG_payload_sha256"],
                                 "montage_key": montage, "physical_unit_verified": True,
                                 "fit_allowed": True, "validation_allowed": False, "online_harness_allowed": False,
                                 "source_recording_id": SOURCE + ":" + record["provider_file_id"],
                                 "component_id_sha256": record["group_id_sha256"]}
                        new_sources[record["source_sha256"]] = entry
                        for index, window in enumerate(windows):
                            new_manifest.append({
                                "id": record["source_sha256"] + ":" + str(index), "split": "train",
                                "subject_group": record["subject_group"],
                                "source_sha256": record["source_sha256"],
                                "eeg_payload_sha256": proof["eeg_payload_sha256"],
                                "source_recording_id": entry["source_recording_id"], **window,
                                "channels": CHANNELS, "sample_rate_hz": FS_OUT, "unit": "V_x_1e4",
                                "eeg_path": relarray, "eeg_index": index, "montage_key": montage,
                                "reference": REFERENCE, "native_sample_rate_hz": record["native_sample_rate_hz"],
                                "native_rate_numerator": record["native_rate_numerator"],
                                "native_rate_denominator": record["native_rate_denominator"],
                                "physical_unit_verified": True, "source_causality": "unverified_publisher_EDF_preprocessing",
                                "online_harness_allowed": False})
                        cert.update(status="admitted_offline_SSL", prepared_seconds=len(windows)*PATCHES)
                    del array
        certificates.append(cert)
        dump(out/"source_certificates.json", certificates)
        dump(out/"progress.json", {"phase": "preparing", "records_reviewed": len(certificates),
                                  "new_windows": len(new_manifest), "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    common_result = {
        "new_source_records": len(new_sources), "new_source_windows": len(new_manifest),
        "new_source_unique_hours": len(new_manifest)*PATCHES/3600, "new_gradient_exposure_hours": 0,
        "val_records_not_opened": 5, "test_records_not_opened": 3,
        "newsource_val_or_test_signals_opened": 0, "questionnaire_or_label_tables_opened": 0,
        "online_harness_allowed": False, "source_causality_verified": False,
        "validation_domain": "806023 only; no Design source generalization estimate",
        "base_completion_sha256": args.expected_base_completion_sha256,
        "frozen_plan_sha256": PLAN_SHA256, "split_manifest_sha256": sha(out/"split_manifest.json"),
        "parent_provenance_sha256": sha(out/"parent_provenance.json"), "metadata_frozen_at_utc": plan["frozen_at_utc"],
        "protocol_sha256": sha(out/"protocol.json"), "certificates_sha256": sha(out/"source_certificates.json"),
    }
    if not new_manifest:
        result = {**common_result, "status": "blocked_no_eligible_new_source_windows",
                  "base_corpus_not_copied": True,
                  "reasons": dict(Counter(c["reason"] for c in certificates))}
        dump(out/"preparation_result.json", result); return result
    for rel, spec in base_artifacts.items():
        target = out/"base"/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        require(shutil.disk_usage(out).free >= reserve + spec["bytes"], "Reserve reached before base copy")
        shutil.copyfile(base/rel, target)
        require(sha(target) == spec["sha256"], "Retained base array copy differs")
        artifacts["base/"+rel] = spec
    manifest = [{**r, "eeg_path": "base/"+r["eeg_path"]} for r in base_rows] + new_manifest
    (out/"manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False)+"\n" for r in manifest))
    dump(out/"artifacts.json", artifacts)
    dump(out/"source_admission.json", {
        "schema": "xavier-smac-design31578517-offline-admission-v1",
        "sources": {**admission["sources"], **new_sources},
        "excluded_source_sha256": sorted(ds), "excluded_payload_sha256": sorted(dp),
        "excluded_subject_groups": sorted(dg), "excluded_source_groups": sorted(dg),
        "parent_provenance_sha256": sha(out/"parent_provenance.json"),
        "split_manifest_sha256": sha(out/"split_manifest.json"), "protocol_sha256": sha(out/"protocol.json"),
        "online_harness_allowed": False, "validation_domain": "retained 806023 only"})
    trainer.load_corpus(out/"manifest.jsonl", out/"artifacts.json", out/"source_admission.json")
    result = {**common_result, "status": "prepared_verified_new_offline_SSL_corpus",
              "windows": dict(Counter(r["split"] for r in manifest)),
              "retained_base_windows": len(base_rows),
              "combined_train_unique_hours": sum(r["window_end_s"]-r["window_start_s"]
                                                 for r in manifest if r["split"] == "train")/3600,
              "manifest_sha256": sha(out/"manifest.jsonl"), "artifacts_sha256": sha(out/"artifacts.json"),
              "source_admission_sha256": sha(out/"source_admission.json")}
    dump(out/"completion.json", result)
    dump(out/"progress.json", {"phase": "complete", **result})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-root", "frozen-plan", "base-corpus", "expected-base-completion-sha256", "output"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--deny-json", action="append", required=True)
    p.add_argument("--reserve-gib", type=float, default=100)
    args = p.parse_args()
    from smac_storage_guard import require_smac_raw_root
    evidence = require_smac_raw_root(args.raw_root)
    node_root = (Path.home()/"pj-xavier"/evidence["node_id"]).resolve()
    require(node_root in Path(args.output).expanduser().resolve().parents
            and node_root in Path(args.base_corpus).expanduser().resolve().parents,
            "Prepared and retained corpora must be on the current S Mac")
    result = prepare(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "prepared_verified_new_offline_SSL_corpus" else 2


if __name__ == "__main__":
    raise SystemExit(main())
