#!/usr/bin/env python3
"""SMAC-only offline masked SSL corpus addition for Figshare 29606906 original BDF.

All 20 archive members enter metadata-only split registration BEFORE signal
decoding. Only 14 train-member signals are extracted. Existing fixed
100uV/0.01uV QC, native 0.1V bound, and prior base admission remain unchanged.
EEG physical calibration is cross-checked against raw BDF digital fields.
BDF recording provenance is not certified for online causality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import mne
import numpy as np
from scipy import signal

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE.parent / "acquisition"))
sys.path.insert(0, str(HERE.parent / "model"))
import cbramod_ssl_smac as trainer
from prepare_smac_fif import FS_OUT, PATCHES, PEAK_V, MIN_STD_V, require, dump, rows, sha

ARCHIVE_SHA = "d2b2953170e82228c608a9937b180b9dcb8c413b8b0fb93a8d160cd210f15102"
ARCHIVE_BYTES = 598663902
NATIVE_FS = 500
N_EEG = 16
SOURCE = "figshare:29606906"
CATALOG = "000073"
GROUP_PREFIX = SOURCE + ":sub"
REFERENCE = "publisher_BioSemi_BDF_reference_unspecified_no_rereference"
FORMULA = "physical_V=(digital-digital_min)*(physical_max_uV-physical_min_uV)/(digital_max-digital_min)/1e6+physical_min_uV/1e6"


def split_for_subject(subject):
    require(re.fullmatch(r"sub(?:[1-9]|1[0-9]|20)", subject) is not None, "Unregistered BDF subject")
    k = int(hashlib.sha256((SOURCE + ":" + subject).encode()).hexdigest()[:8], 16) % 10
    return "train" if k < 7 else ("val" if k < 9 else "test")


def freeze_members(archive):
    cmd = ["/usr/bin/bsdtar", "-tf", str(archive)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90, check=True)
    members = [s for s in result.stdout.splitlines() if s.lower().endswith(".bdf")]
    require(len(members) == 20, "Exactly 20 original BDF archive members are required")
    index = {}
    for member in members:
        parts = Path(member).parts
        require(len(parts) == 3 and parts[1].startswith("sub")
                and re.fullmatch(r"(?:[A-Za-z0-9_]+)\.bdf", parts[2]),
                "Unexpected archive BDF member hierarchy")
        subject = parts[1]
        require(subject not in index, "One BDF per subject group required")
        index[subject] = {"group": GROUP_PREFIX + subject,
                          "split": split_for_subject(subject),
                          "member": member, "source_signal_opened": False}
    require(set(index) == {"sub" + str(n) for n in range(1, 21)}, "BDF group roster differs")
    require(Counter(r["split"] for r in index.values()) ==
            {"train": 14, "val": 5, "test": 1}, "Pre-signal split counts changed")
    return index


def read_bdf_physical_v(file):
    with Path(file).open("rb") as f:
        hdr = f.read(256)
        require(hdr[:1] == b"\xff" and b"BIOSEMI" in hdr[:15], "Not the expected BioSemi BDF original")
        hsize = int(hdr[184:192].decode().strip())
        count = int(hdr[252:256].decode().strip())
        duration = float(hdr[244:252].decode().strip())
        nrecords = int(hdr[236:244].decode().strip())
        require(count == 36 and hsize == (count+1)*256 and duration == 1 and nrecords > 100,
                "Unexpected BDF header geometry")
        data = f.read(256*count)
        offset = 0
        props = {}
        for name, width in (("label", 16), ("transducer", 80), ("dim", 8), ("pmin", 8),
                            ("pmax", 8), ("dmin", 8), ("dmax", 8), ("prefilter", 80),
                            ("samples", 8), ("reserved", 32)):
            props[name] = [data[offset + width*i:offset + width*(i+1)].decode("latin1").strip()
                           for i in range(count)]
            offset += width*count
        require(all(props["dim"][i] == "uV" and props["samples"][i] == "500" for i in range(N_EEG))
                and all(props["label"][i] == "BDF Annotations" for i in range(N_EEG, count)),
                "BDF physical microvolts or non-EEG channel boundaries are unverified")
        require(all(props["samples"][i] == "38" for i in range(N_EEG, count)),
                "Unexpected BDF annotation sample rate")
        expected = hsize + nrecords * sum(int(x) for x in props["samples"]) * 3
        require(Path(file).stat().st_size == expected, "Original BDF size differs from physical header")
        f.seek(hsize)
        b = np.frombuffer(f.read(3*3), dtype=np.uint8).reshape(-1, 3)
        digital = (b[:, 0].astype(np.int32) |
                   (b[:, 1].astype(np.int32) << 8) |
                   (b[:, 2].astype(np.int32) << 16))
        digital = (digital ^ 0x800000) - 0x800000
        lo, hi = [float(props[k][0]) for k in ("pmin", "pmax")]
        dlo, dhi = [float(props[k][0]) for k in ("dmin", "dmax")]
        manual = ((digital-dlo)*(hi-lo)/(dhi-dlo)+lo)*1e-6
    mne_raw = mne.io.read_raw_bdf(str(file), preload=False, verbose="ERROR")
    idx = mne.pick_types(mne_raw.info, eeg=True, exclude=[])
    require(len(idx) == N_EEG and mne_raw.info["sfreq"] == NATIVE_FS and
            mne_raw.n_times == nrecords * NATIVE_FS, "Independent BDF reader shape differs")
    x = mne_raw.get_data(picks=idx, verbose="ERROR").astype(np.float64)
    require(np.isfinite(x).all(), "Nonfinite calibrated original BDF EEG")
    require(float(np.max(np.abs(x))) <= .1, "native_peak_above_fixed_0_1V_source_bound")
    require(float(np.max(np.abs(x[0, :3]-manual))) <= 1e-11,
            "Independent BDF digital-to-physical calibration failed")
    return x, [mne_raw.ch_names[i] for i in idx], {
        "native_seconds": float(mne_raw.n_times/NATIVE_FS),
        "digital_header_and_MNE_V_probes_max_difference": float(np.max(np.abs(x[0, :3]-manual))),
        "physical_unit_source": "BDF header uV channel calibration and independent MNE match",
        "channels": [mne_raw.ch_names[i] for i in idx],
    }


def causal_windows(x):
    sos = signal.butter(4, [3, 40], btype="bandpass", fs=NATIVE_FS, output="sos")
    y = signal.sosfilt(sos, x, axis=1)
    # Delayed linear interpolation at t=j/200 uses only original indices n-1,n.
    js = np.arange(2*FS_OUT, ((x.shape[1]-1)*FS_OUT)//NATIVE_FS, dtype=np.int64)
    n = (js*NATIVE_FS)//FS_OUT
    alpha = ((js*NATIVE_FS) % FS_OUT)/FS_OUT
    z = y[:, n-1]*(1-alpha) + y[:, n]*alpha
    arrays, windows, rejects = [], [], Counter()
    width = FS_OUT * PATCHES
    for start in range(0, z.shape[1]-width+1, width):
        w = z[:, start:start+width]
        if not np.isfinite(w).all():
            rejects["nonfinite"] += 1
        elif float(np.max(np.abs(w))) > PEAK_V:
            rejects["peak_above_fixed_100uV"] += 1
        elif float(np.min(w.std(axis=1))) < MIN_STD_V:
            rejects["flat_below_fixed_0_01uV"] += 1
        else:
            arrays.append((w*1e4).reshape(N_EEG, PATCHES, FS_OUT).astype(np.float32))
            windows.append({
                "window_start_s": float(js[start]/FS_OUT),
                "window_end_s": float((js[start+width-1]+1)/FS_OUT),
                "latest_native_source_index": int(n[start+width-1]),
                "feature_available_native_s": float(n[start+width-1]/NATIVE_FS),
            })
    arr = np.stack(arrays) if arrays else np.empty((0, N_EEG, PATCHES, FS_OUT), np.float32)
    return arr, windows, dict(rejects)


def prepare(args):
    from smac_storage_guard import require_smac_raw_root
    guard = require_smac_raw_root(args.raw_root)
    node_root = (Path.home()/"pj-xavier"/guard["node_id"]).absolute()
    archive = Path(args.archive).expanduser().absolute()
    base = Path(args.base_corpus).expanduser().absolute()
    out = Path(args.output).expanduser().absolute()
    require(node_root in archive.parents and node_root in base.parents and node_root in out.parents,
            "Source, base, and output must reside on same S Mac node")
    require(not out.exists(), "Do not overwrite an immutable BDF preparation attempt")
    require(archive.is_file() and archive.stat().st_size == ARCHIVE_BYTES and sha(archive) == ARCHIVE_SHA,
            "Original Figshare RAR source SHA or size mismatch")
    require(sha(base/"completion.json") == args.expected_base_completion_sha256,
            "Parent corpus completion SHA mismatch")
    completion = json.loads((base/"completion.json").read_text())
    require(completion["status"] == "prepared_verified_new_offline_SSL_corpus",
            "Parent corpus not sealed")
    for name, key in (("manifest.jsonl", "manifest_sha256"),
                      ("artifacts.json", "artifacts_sha256"),
                      ("source_admission.json", "source_admission_sha256"),
                      ("protocol.json", "protocol_sha256")):
        require(sha(base/name) == completion[key], "Parent corpus source seal mismatch: "+name)
    trainer.load_corpus(base/"manifest.jsonl", base/"artifacts.json", base/"source_admission.json")
    bm = rows(base/"manifest.jsonl")
    bart = json.loads((base/"artifacts.json").read_text())
    ba = json.loads((base/"source_admission.json").read_text())
    members = freeze_members(archive)
    possible = sum(13_000_000 for k,v in members.items() if v["split"] == "train")
    require(shutil.disk_usage(out.parent).free >= args.reserve_gib*1024**3 + possible +
            sum(v["bytes"] for v in bart.values()) + 2*1024**3, "Storage reserve or recovery capacity insufficient")
    out.mkdir(parents=True)
    (out/"original_train_bdf").mkdir()
    (out/"waveforms").mkdir()
    # Sealed before any BDF signal is extracted.
    dump(out/"split_manifest.json", {"schema": "xavier-bdf29606906-metadata-split-v1",
         "split_protocol": "sha256('figshare:29606906:subN')[:8] % 10; 0..6 train,7..8 val,9 test",
         "member_roster": members, "original_archive_sha256": ARCHIVE_SHA,
         "test_or_val_signal_opened": 0, "frozen_before_signal_extraction": True})
    archive_split_sha = sha(out/"split_manifest.json")
    deny_s = set(ba["excluded_source_sha256"])
    deny_p = set(ba["excluded_payload_sha256"])
    old_sources = set(ba["sources"])
    old_payload = {r["eeg_payload_sha256"] for r in ba["sources"].values()}
    old_val_groups = {r["subject_group"] for r in bm if r["split"] != "train"}
    protocol = {
        "schema": "xavier-smac-bdf29606906-offline-SSL-v1",
        "scientific_source": "Figshare 29606906 v2 original 20 BDF in dataset.rar",
        "catalog_id": CATALOG, "archive_sha256": ARCHIVE_SHA,
        "split_manifest_sha256": archive_split_sha,
        "trainer_sha256": sha(trainer.__file__), "preparation_sha256": sha(__file__),
        "base_completion_sha256": args.expected_base_completion_sha256,
        "source_groups": {"train":14, "val":5, "test":1},
        "new_source_validation_or_test_BDF_decoded":0,
        "source_units": FORMULA, "reference":REFERENCE,
        "native_sample_rate_hz": NATIVE_FS, "output_hz":FS_OUT,
        "EEG_channels":16, "annotations_excluded":20,
        "preprocessing": {
            "causal_SOS_order":4, "bandpass_hz":[3,40],
            "sos_coefficients":signal.butter(4,[3,40],btype="bandpass",fs=NATIVE_FS,output="sos").tolist(),
            "startup_discard_seconds":2, "native_sample_delay":1/NATIVE_FS,
            "interpolation":"linear(n-1,n) with n=floor(native_time*500), no future sample",
            "window_seconds":PATCHES, "output_unit":"V_x_1e4",
            "QC_native_peak_V":0.1, "QC_processed_peak_V":PEAK_V,
            "QC_min_channel_std_V": MIN_STD_V,
            "QC_frozen_before_raw_signal":True, "no_clipping_or_unit_guessing":True,
        },
        "offline_only":True, "source_acquisition_processing_causality_unverified":True,
        "labels_or_IMU_used": False,
        "online_harness_allowed": False,
        "published_source_subsets_not_independent_EEG":True,
        "created_epoch":time.time(),
    }
    dump(out/"protocol.json", protocol)
    added, certs, source_admitted, artifact_added = [], [], {}, {}
    seen_sources, seen_payloads = set(), set()
    for subject, member in sorted(members.items(), key=lambda kv:int(kv[0][3:])):
        cert = {"subject_group":member["group"],"split":member["split"],
                "archive_member":member["member"],"signal_opened":False}
        if member["split"] != "train":
            cert.update(status="frozen_heldout_signal_not_opened");certs.append(cert);continue
        dest = out/"original_train_bdf"/(subject+".bdf")
        with dest.open("wb") as stream:
            result = subprocess.run(["/usr/bin/bsdtar","-xOf",str(archive),member["member"]],
                                    stdout=stream,stderr=subprocess.PIPE,timeout=90)
        require(result.returncode == 0 and dest.stat().st_size > 1000,
                "Archive extraction failed for frozen train BDF")
        orig_sha = sha(dest)
        require(orig_sha not in seen_sources, "Cross-ID duplicate original BDF")
        seen_sources.add(orig_sha)
        cert.update(source_sha256=orig_sha, source_file_bytes=dest.stat().st_size, signal_opened=True)
        if orig_sha in old_sources or orig_sha in deny_s or member["group"] in old_val_groups:
            cert.update(status="excluded_prior_source_or_group");certs.append(cert);continue
        x, channels, raw_proof = read_bdf_physical_v(dest)
        native_hash = hashlib.sha256(np.asarray(x.T, dtype="<f8",order="C").tobytes()).hexdigest()
        cert.update(raw_proof, eeg_payload_sha256=native_hash)
        require(native_hash not in seen_payloads and native_hash not in old_payload,
                "Original BDF EEG duplicate native physical payload")
        seen_payloads.add(native_hash)
        if native_hash in deny_p:
            cert.update(status="excluded_prior_payload_identity");certs.append(cert);continue
        arrays, windows, rejected = causal_windows(x)
        del x
        cert.update(qc_rejects=rejected, accepted_windows=len(windows))
        if not windows:
            cert.update(status="held_no_QC_eligible_windows");certs.append(cert);continue
        rel = "waveforms/"+orig_sha+".npy"
        np.save(out/rel,arrays,allow_pickle=False)
        artifact_added[rel] = {
            "shape": list(arrays.shape), "dtype": "float32", "sha256":sha(out/rel),
            "bytes":(out/rel).stat().st_size}
        montage = hashlib.sha256(json.dumps([channels,REFERENCE,NATIVE_FS,SOURCE],
                                             sort_keys=True).encode()).hexdigest()
        source_admitted[orig_sha] = {
            "split":"train", "subject_group":member["group"],
            "eeg_payload_sha256":native_hash,"montage_key":montage,
            "physical_unit_verified":True, "fit_allowed":True,
            "validation_allowed":False, "online_harness_allowed":False,
            "source_recording_id":SOURCE+":"+member["member"],
            "calibration_proof":FORMULA,
        }
        for i,w in enumerate(windows):
            added.append({
                "id": orig_sha+":"+str(i), "split":"train",
                "subject_group":member["group"], "source_sha256":orig_sha,
                "eeg_payload_sha256":native_hash,
                "source_recording_id":SOURCE+":"+member["member"],
                **w, "channels":channels, "sample_rate_hz":FS_OUT,
                "unit":"V_x_1e4", "eeg_path":rel, "eeg_index":i,
                "montage_key":montage, "reference":REFERENCE,
                "native_sample_rate_hz":NATIVE_FS, "physical_unit_verified":True,
                "source_causality":"BDF original causal hardware processing unverified",
                "online_harness_allowed":False,
            })
        cert.update(status="admitted_offline_ssl",seconds=len(windows)*PATCHES)
        certs.append(cert)
    dump(out/"source_certificates.json",certs)
    if not added:
        result={"status":"blocked_no_eligible_new_source_windows",
                "source_records_opened":14,"new_source_windows":0,"actual_gradient_steps":0,
                "frozen_heldout_signals_opened":0,"protocol_sha256":sha(out/"protocol.json"),
                "certificates_sha256":sha(out/"source_certificates.json")}
        dump(out/"preparation_result.json",result)
        return result
    artifacts = {}
    for rel,spec in bart.items():
        destination = out/"base"/rel
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(base/rel,destination)
        require(sha(destination) == spec["sha256"],"Parent array copy SHA mismatch")
        artifacts["base/"+rel]=spec
    artifacts.update(artifact_added)
    manifest=[{**r,"eeg_path":"base/"+r["eeg_path"]} for r in bm]+added
    (out/"manifest.jsonl").write_text("".join(json.dumps(r,sort_keys=True,allow_nan=False)+"\n" for r in manifest))
    dump(out/"artifacts.json",artifacts)
    dump(out/"source_admission.json",{
        "schema":"xavier-smac-BDF29606906-offline-SSL-admission-v1",
        "sources":{**ba["sources"],**source_admitted},
        "excluded_source_sha256":sorted(deny_s),
        "excluded_payload_sha256":sorted(deny_p),
        "protocol_sha256":sha(out/"protocol.json"),
        "validation_domain":"retained 806023 only; not new BDF generalization",
        "online_harness_allowed":False,"raw_EEG_signal_only":True,
    })
    trainer.load_corpus(out/"manifest.jsonl",out/"artifacts.json",out/"source_admission.json")
    result={
        "status":"prepared_verified_new_offline_SSL_corpus",
        "windows":dict(Counter(r["split"] for r in manifest)),
        "new_source_records":len(source_admitted),
        "new_source_windows":len(added),
        "new_source_unique_hours":len(added)*PATCHES/3600,
        "new_gradient_exposure_hours":0,
        "retained_base_windows":len(bm),
        "frozen_heldout_signals_opened":0,
        "parent_completion_sha256":args.expected_base_completion_sha256,
        "manifest_sha256":sha(out/"manifest.jsonl"),
        "artifacts_sha256":sha(out/"artifacts.json"),
        "source_admission_sha256":sha(out/"source_admission.json"),
        "protocol_sha256":sha(out/"protocol.json"),
        "certificates_sha256":sha(out/"source_certificates.json"),
        "split_manifest_sha256":archive_split_sha,
        "online_harness_allowed":False,
        "new_source_generalization_evaluated":False,
    }
    dump(out/"completion.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("raw-root","archive","base-corpus","expected-base-completion-sha256","output"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--reserve-gib",type=float,default=100)
    args=p.parse_args()
    result=prepare(args)
    print(json.dumps(result,sort_keys=True),flush=True)
    return 0 if result["status"]=="prepared_verified_new_offline_SSL_corpus" else 2


if __name__=="__main__":
    raise SystemExit(main())
