#!/usr/bin/env python3
"""Verify ITU sources, extract them, and audit clocks as an external held-out site.

CPU only. This creates no training split, windows, classifier, or predictions.
The fixed UniBa alignment rules are imported unchanged. No phase scale, clock
threshold, model parameter, or action-label rule is selected using ITU results.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import time

import numpy as np

from audit_brainvision_archive import file_checksums, normalized_member, read_ledger, write_json
from prepare_ai_coding import COMMON_CHANNELS, digest, find_beh, fit_clock, phase_anchors, phase_transitions, read_tsv


SOURCE = "https://zenodo.org/records/20801101"
EXPECTED = {
    "bids_eeg_itu.tar.gz": (4650970671, "af98ad619d3e6f5576c164e601f0c598", "1e192b5e65931b9a44858acd987d540d3436de3fddf99454170b93434c2c99b5"),
    "bids_behavioral_itu.tar.gz": (14991088, "8e0a0ee9827caf06310dff88880458f9", "fb501e02f74120f665ce860e004bf214bd17381068e497e4986c2a56923167c9"),
}
FROZEN_LOADER = "38e3cca302b456f5c918c03d93d5f71d77ffe636efbc3f90840cf89ac2d8badc"


def extract_verified_site(archive, root):
    records, seen = [], set()
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = normalized_member(member.name)
            if not name.startswith("data/01_raw/itu/") or member.isdir():
                continue
            if not member.isfile() or name in seen:
                raise ValueError("Nonregular or repeated source member")
            seen.add(name)
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Only the named temporary file belongs to this extraction. Existing
            # completed files are compared and never overwritten silently.
            temporary = destination.with_name(destination.name + ".extracting")
            sha, count = hashlib.sha256(), 0
            with temporary.open("wb") as output:
                source = tar.extractfile(member)
                while block := source.read(2 * 1024 * 1024):
                    output.write(block); sha.update(block); count += len(block)
            if count != member.size:
                raise ValueError("Truncated source member")
            value = sha.hexdigest()
            if destination.exists():
                if digest(destination) != value:
                    raise ValueError("Existing extracted file differs from the verified archive")
                temporary.unlink()
            else:
                temporary.replace(destination)
            records.append({"path": name, "bytes": count, "sha256": value})
    if not records:
        raise ValueError("No original ITU members found")
    return records


def main(args):
    import mne
    started = time.monotonic()
    out, root, raw = Path(args.output), Path(args.extract_root), Path(args.archive_dir)
    out.mkdir(parents=True, exist_ok=True); root.mkdir(parents=True, exist_ok=True)
    if (out / "alignment_audit.json").exists():
        raise ValueError("Audit already completed; preserve it")
    loader = Path(__file__).with_name("prepare_ai_coding.py")
    if digest(loader) != FROZEN_LOADER:
        raise ValueError("Frozen alignment implementation changed")
    quantity = json.loads(Path(args.quantity_audit).read_text())
    if quantity["aggregate"]["archive_verification"]["sha256"] != EXPECTED["bids_eeg_itu.tar.gz"][2]:
        raise ValueError("Quantity audit does not identify the expected EEG archive")
    if quantity["aggregate"]["headers_blocked"] or quantity["aggregate"]["headers_total"] != 26:
        raise ValueError("A complete 26-record quantity audit is required")
    if shutil.disk_usage(root).free < quantity["aggregate"]["total_archive_member_bytes"] + 2 * 1024 ** 3:
        raise ValueError("Insufficient disk for preserved verified extraction plus margin")
    recipe_path = Path(__file__).with_name("itu_recipe_evidence.json")
    recipe = json.loads(recipe_path.read_text())
    verification, members = [], []
    for name, (size, md5, sha256) in EXPECTED.items():
        ledger = read_ledger(args.ledger, name)
        published = json.loads(ledger["checksum_json"])
        if ledger["status"] != "complete" or ledger["expected_bytes"] != size or ledger["sha256"] != sha256 or published != {"algorithm": "md5", "value": md5}:
            raise ValueError("Source ledger/registered checksum is not complete and consistent")
        print("Verifying", name, flush=True)
        measured = file_checksums(raw / name)
        if any(measured[k] != expected for k, expected in (("bytes", size), ("md5", md5), ("sha256", sha256))):
            raise ValueError("Source archive re-verification failed")
        print("Extracting verified", name, flush=True)
        current = extract_verified_site(raw / name, root)
        members.extend(current)
        verification.append({"filename": name, **measured, "ledger_status": "complete", "reverified_before_extraction": True})
    source_verification = {"verified": True, "sources": verification, "source": SOURCE}
    write_json(root / "source_verification.json", source_verification)
    write_json(root / "extracted_members.json", {"members": members})
    member_lookup = {r["path"]: r for r in members}
    for record in quantity["records"]:
        if member_lookup[record["binary_member"]]["sha256"] != record["binary_sha256"]:
            raise ValueError("Extracted EEG differs from independent quantity-audit binary digest")
        if member_lookup[record["header_member"]]["sha256"] != record["header_sha256"]:
            raise ValueError("Extracted header differs from independent quantity audit")
    audits = []
    for quantity_record in quantity["records"]:
        path = root / quantity_record["header_member"]
        record = {"header_member": quantity_record["header_member"], "site": "itu", "partition": "external_test",
                  "binary_sha256": quantity_record["binary_sha256"], "header_sha256": quantity_record["header_sha256"],
                  "recording_duration_s": quantity_record["duration_seconds"], "status": "blocked"}
        match = re.search(r"(sub-[^_]+)_(ses-[^_]+)_", path.name)
        if match: record.update(subject="itu:" + match.group(1), session=match.group(2))
        try:
            signal = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
            fs = float(signal.info["sfreq"])
            if fs != quantity_record["sampling_frequency_hz"] or signal.n_times != quantity_record["samples_per_channel"]:
                raise ValueError("MNE differs from independent binary/header quantity audit")
            record.update(sample_rate_hz=fs, channels=list(signal.ch_names),
                          common_channels_missing=[c for c in COMMON_CHANNELS if c not in signal.ch_names],
                          otree_phase_present="otree_phase" in signal.ch_names)
            if record["common_channels_missing"] or not record["otree_phase_present"]:
                raise ValueError("Required common EEG or synchronization channels missing")
            phase_file = find_beh(path.parent.parent / "beh", "otree")
            phase = signal.get_data(picks=["otree_phase"])[0]
            if not np.isfinite(phase).all(): raise ValueError("Nonfinite phase samples")
            record["phase_unique_values_rounded_6dp"] = np.unique(np.round(phase, 6)).tolist()
            if len(record["phase_unique_values_rounded_6dp"]) > 32:
                raise ValueError("Phase channel does not contain a small discrete event code set")
            record["phase_sha256_mne_float64"] = hashlib.sha256(np.asarray(phase, dtype="<f8").tobytes()).hexdigest()
            record["behavior_phase_member"] = str(phase_file.relative_to(root))
            record["behavior_phase_sha256"] = digest(phase_file)
            clock = fit_clock(phase_transitions(phase, fs), phase_anchors(read_tsv(phase_file)),
                              max_error_ms=100, max_drift_ppm=5000, min_anchors=4, min_span_s=120)
            clock["explicit_phase_channel_scale"] = 1
            record["clock_sync"] = clock
            if clock["verified"]: record["status"] = "alignment_passed_external_candidate"
            else: record["error"] = {"type": "ClockAlignmentFailed", "message": clock["reason"]}
            signal.close()
        except Exception as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
        audits.append(record)
        print(json.dumps({k: record.get(k) for k in ("subject", "status", "error")}), flush=True)
    passed = [r for r in audits if r["status"] == "alignment_passed_external_candidate"]
    aggregate = {
        "source": SOURCE, "site": "itu", "status": "clock_audit_complete",
        "recording_count": len(audits), "source_recording_hours": sum(r["recording_duration_s"] for r in audits) / 3600,
        "alignment_passed": len(passed), "alignment_blocked": len(audits) - len(passed),
        "aligned_source_hours": sum(r["recording_duration_s"] for r in passed) / 3600,
        "blocked_reasons": dict(Counter(r.get("error", {}).get("message", "unknown") for r in audits if r not in passed)),
        "common8_present_recordings": sum(not r.get("common_channels_missing", ["unknown"]) for r in audits),
        "phase_present_recordings": sum(r.get("otree_phase_present", False) for r in audits),
        "phase_scale": 1, "clock_error_threshold_ms": 100, "clock_drift_threshold_ppm": 5000,
        "passed_max_error_ms_range": [min(r["clock_sync"]["max_error_ms"] for r in passed), max(r["clock_sync"]["max_error_ms"] for r in passed)] if passed else None,
        "frozen_alignment_module_sha256": FROZEN_LOADER, "recipe_evidence_sha256": digest(recipe_path),
        "recipe": recipe, "source_verification": source_verification,
        "all_participants_partition": "external_test", "used_for_training_or_model_selection": False,
        "model_loaded": False, "gpu_execution": False, "predictions_created": False,
        "online_causality_verified": False, "original_xdf_clock_verified": False,
        "quality_or_external_performance_established": False,
        "mne_version": mne.__version__, "elapsed_seconds": time.monotonic() - started,
    }
    write_json(out / "alignment_audit.json", {"aggregate": aggregate, "recordings": audits})
    aggregate["private_alignment_audit_sha256"] = digest(out / "alignment_audit.json")
    write_json(out / "aggregate.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive-dir", "ledger", "quantity-audit", "extract-root", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        write_json(Path(args.output) / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
