#!/usr/bin/env python3
"""Prepare frozen ITU external_test EEG windows without training or inference."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import sys
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"

SOURCE = "https://zenodo.org/records/20801101"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as f:
        for value in iter(lambda: f.read(1024**2), b""):
            result.update(value)
    return result.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def require_hash(path, expected):
    if digest(path) != expected:
        raise ValueError("Frozen input hash mismatch:" + Path(path).name)


def load_helpers(protocol):
    directory = Path(__file__).resolve().parent
    for name, expected in protocol["helper_source_sha256"].items():
        require_hash(directory / name, expected)
    # No model module or torch is imported. Helpers are identical to UniBa v3.
    sys.path.insert(0, str(directory))
    natural = importlib.import_module("prepare_natural_cohort")
    coding = importlib.import_module("prepare_ai_coding")
    archive = importlib.import_module("audit_brainvision_archive")
    for module in (natural, coding, archive):
        path = Path(module.__file__).resolve()
        if path.parent != directory:
            raise ValueError("Helper import resolved outside the frozen code directory")
        require_hash(path, protocol["helper_source_sha256"][path.name])
    return natural, coding, archive


def canonical_subject(site, label):
    if site != "itu" or not isinstance(label, str):
        raise ValueError("Only ITU observation IDs are allowed")
    raw = label.removeprefix("itu:")
    if not re.fullmatch(r"sub-[A-Za-z0-9]+", raw):
        raise ValueError("Invalid site-qualified participant label")
    return "itu:" + raw


def clock_eligible(record, protocol):
    if record.get("site") != "itu" or record.get("partition") != "external_test":
        raise ValueError("External clock audit contains another site or partition")
    canonical_subject(record["site"], record["subject"])
    passed = record.get("status") == "alignment_passed_external_candidate"
    clock = record.get("clock_sync", {})
    if passed != (clock.get("verified") is True):
        raise ValueError("Clock status and verification disagree")
    if passed:
        inclusion = protocol["inclusion"]
        if (not 0 <= clock["max_error_ms"] <= inclusion["clock_max_error_ms"] or
                not 0 <= clock["drift_ppm"] <= inclusion["clock_max_drift_ppm"] or
                clock.get("explicit_phase_channel_scale") != inclusion["clock_phase_scale"] or
                clock.get("anchor_count", 0) < 4 or clock.get("anchor_span_s", 0) < 120):
            raise ValueError("Frozen clock acceptance thresholds violated")
    return passed


def compare_cross_site(uniba_rows, itu_records):
    if not uniba_rows or any(row.get("site") != "uniba" for row in uniba_rows):
        raise ValueError("Cross-site reference must be the frozen UniBa manifest")
    uniba = {r["eeg_source_sha256"] for r in uniba_rows}
    itu = {r["binary_sha256"] for r in itu_records}
    if uniba & itu:
        raise ValueError("Exact raw waveform SHA256 overlaps between ITU and UniBa")
    uniba_labels = {r["subject"].split(":", 1)[-1] for r in uniba_rows}
    itu_labels = {canonical_subject(r["site"], r["subject"]).split(":", 1)[-1] for r in itu_records}
    return {"uniba_raw_waveform_hashes": len(uniba), "itu_raw_waveform_hashes": len(itu),
            "exact_binary_hash_matches": 0, "shared_unqualified_participant_labels": len(uniba_labels & itu_labels),
            "canonical_observation_id_overlap": 0, "human_identity_disjointness_verified": False,
            "scope": "All source hashes referenced by frozen UniBa v3 versus all audited ITU raw binaries; byte distinction does not establish different people."}


def member_path(root, member):
    relative = PurePosixPath(member.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not str(relative).startswith("data/01_raw/itu/"):
        raise ValueError("Invalid or non-ITU source member")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Source member is absent, a symlink, or outside verified ITU root")
    return path


def checked_member(root, member, inventory, verified):
    path = member_path(root, member)
    expected = inventory.get(member)
    if expected is None:
        raise ValueError("Source member is absent from pinned extraction inventory")
    before = path.stat()
    value = digest(path)
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after) or value != expected["sha256"] or after.st_size != expected["bytes"]:
        raise ValueError("Verified extracted source member changed")
    verified[member] = {"sha256": value, "bytes": after.st_size, "identity": list(identity(after))}
    return path


def validate_timing(row):
    guard = row["clock_uncertainty_guard_s"]
    if (row["split"] != "external_test" or row["site"] != "itu" or
            row["window_end_s"] - row["window_start_s"] != 1.75 or
            row["window_end_s"] > row["event_time_s"] - 4.5 - guard + 1e-8 or
            row["decision_time_s"] != row["window_end_s"] + 4.125 or
            row["environment_time_s"] > row["window_start_s"] - guard + 1e-8):
        raise ValueError("External partition or predecision signal/context timing changed")
    for key, expected in (("history_pre_eeg", row["window_start_s"] - guard),
                          ("history_pre_decision_guarded", row["decision_time_s"] - guard)):
        history = row[key]
        latest = history["max_source_onset_used_s"]
        if (history["cutoff_s"] != expected or expected >= row["event_time_s"] or
                (latest is not None and not latest < expected) or history["target_or_future_events_included"] is not False):
            raise ValueError("Future or target event entered history")


def prepare(args):
    import numpy as np
    import mne
    started = time.time()
    root, output = Path(args.bids_root).resolve(), Path(args.output).resolve()
    if output.exists():
        raise ValueError("External cohort output already exists; no overwrite or reselection")
    protocol_path, parent_path = Path(args.protocol).resolve(), Path(args.parent_protocol).resolve()
    require_hash(protocol_path, args.expected_protocol_sha256)
    protocol = json.loads(protocol_path.read_text())
    if protocol["partition"] != "external_test" or protocol["site"] != "itu":
        raise ValueError("Only the frozen external_test protocol is accepted")
    fixed = protocol["frozen_inputs"]
    paths = {"uniba_v3_protocol_sha256": parent_path, "uniba_v3_manifest_sha256": Path(args.uniba_manifest),
             "itu_alignment_audit_sha256": Path(args.alignment_audit),
             "itu_source_verification_sha256": root / "source_verification.json",
             "itu_extracted_members_sha256": root / "extracted_members.json",
             "itu_recipe_evidence_sha256": Path(__file__).with_name("itu_recipe_evidence.json")}
    for key, path in paths.items(): require_hash(path, fixed[key])
    source_sha = digest(__file__)
    natural, coding, archive = load_helpers(protocol)
    parent = json.loads(parent_path.read_text())
    for key, expected in protocol["signal_confirmation"].items():
        if key != "new_filtering_resampling_rereferencing" and parent["signal"].get(key) != expected:
            raise ValueError("Frozen parent signal setting differs:" + key)
    if tuple(parent["signal"]["channels"]) != coding.COMMON_CHANNELS:
        raise ValueError("Common eight-channel order changed")
    source_verification = json.loads((root / "source_verification.json").read_text())
    if source_verification.get("verified") is not True:
        raise ValueError("Source archive verification is incomplete")
    for name, key in (("bids_eeg_itu.tar.gz", "itu_eeg_archive_sha256"), ("bids_behavioral_itu.tar.gz", "itu_behavior_archive_sha256")):
        matches = [r for r in source_verification["sources"] if r["filename"] == name]
        if len(matches) != 1 or matches[0]["sha256"] != fixed[key] or matches[0]["ledger_status"] != "complete":
            raise ValueError("ITU source archive verification mismatch")
    clock_audit = json.loads(Path(args.alignment_audit).read_text())
    old_records = clock_audit["recordings"]
    if len(old_records) != protocol["inclusion"]["audit_recordings"]:
        raise ValueError("Frozen ITU recording inventory changed")
    eligibility = {r["header_member"]: clock_eligible(r, protocol) for r in old_records}
    if len(eligibility) != len(old_records) or sum(eligibility.values()) != protocol["inclusion"]["clock_passed_recordings_expected"]:
        raise ValueError("Frozen ITU clock pass inventory changed")
    uniba_rows = [json.loads(line) for line in Path(args.uniba_manifest).read_text().splitlines() if line.strip()]
    comparison = compare_cross_site(uniba_rows, old_records)
    members = json.loads((root / "extracted_members.json").read_text())["members"]
    inventory = {r["path"]: r for r in members}
    if len(inventory) != len(members):
        raise ValueError("Repeated extracted member identity")
    if shutil.disk_usage(root).free < args.min_free_gib * 1024**3:
        raise ValueError("Disk reserve prevents external cohort preparation")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preparation_started.json", {"pid": os.getpid(), "run_token": getattr(args, "run_token", None),
               "protocol_sha256": args.expected_protocol_sha256, "started_epoch": started})
    (output / "protocol.json").write_bytes(protocol_path.read_bytes())
    (output / "uniba_v3_protocol.json").write_bytes(parent_path.read_bytes())
    write_json(output / "frozen_inputs.json", {"protocol_sha256": args.expected_protocol_sha256, "source_sha256": source_sha,
               "inputs": fixed, "helper_source_sha256": protocol["helper_source_sha256"], "started_epoch": started})
    rows, features, labels, audits, verified, artifacts = [], [], [], [], {}, {}
    for old in sorted(old_records, key=lambda r: r["header_member"]):
        subject = canonical_subject(old["site"], old["subject"])
        report = {"subject": subject, "session": old["session"], "site": "itu", "split": "external_test",
                  "header_member": old["header_member"], "clock_passed": eligibility[old["header_member"]],
                  "recording_duration_s": old["recording_duration_s"], "status": "blocked"}
        header = checked_member(root, old["header_member"], inventory, verified)
        if digest(header) != old["header_sha256"]:
            raise ValueError("Header differs from frozen clock audit")
        header_info = archive.parse_header(header.read_bytes())
        waveform_member = str(PurePosixPath(old["header_member"]).parent / header_info["datafile"])
        waveform = checked_member(root, waveform_member, inventory, verified)
        waveform_sha = verified[waveform_member]["sha256"]
        if waveform_sha != old["binary_sha256"]:
            raise ValueError("Waveform differs from frozen clock audit")
        marker_names = re.findall(r"^MarkerFile\s*=\s*(.+)$", header.read_text(encoding="utf-8-sig"), re.MULTILINE)
        if len(marker_names) != 1 or "/" in marker_names[0] or "\\" in marker_names[0] or marker_names[0].strip() in {"", ".", ".."}:
            raise ValueError("BrainVision marker reference is ambiguous or unsafe")
        checked_member(root, str(PurePosixPath(old["header_member"]).parent / marker_names[0].strip()), inventory, verified)
        phase_file = checked_member(root, old["behavior_phase_member"], inventory, verified)
        phase_sha = natural.require_parent_phase_hash(phase_file, {old["behavior_phase_sha256"]})
        report.update(raw_waveform_sha256=waveform_sha, header_sha256=old["header_sha256"],
                      behavior_phase_sha256=phase_sha, source_members_verified=True)
        if not eligibility[old["header_member"]]:
            report.update(reason="frozen_clock_audit_rejected_recording", clock_reason=old["clock_sync"].get("reason"),
                          clock_max_error_ms=old["clock_sync"].get("max_error_ms"), selected_windows=0)
            audits.append(report)
            continue
        beh = header.parent.parent / "beh"
        if coding.find_beh(beh, "otree").resolve() != phase_file.resolve():
            raise ValueError("oTree lookup differs from pinned phase member")
        # Hash each behavioral source before the unchanged helper consumes it.
        for acquisition in ("keypress", "mouseclick", "mousemove", "gpt"):
            for file in sorted(beh.glob(f"*_acq-{acquisition}_*_events.tsv")):
                checked_member(root, str(file.relative_to(root)), inventory, verified)
        clock = old["clock_sync"]
        events, history, available, history_files, onset_rejects = natural.read_recorded_events(beh, clock)
        cases = [event for event in events if event["label"] == "ai_prompt_submit"]
        raw = mne.io.read_raw_brainvision(header, preload=False, verbose="ERROR")
        try:
            fs = float(raw.info["sfreq"])
            if (fs != 256 or raw.n_times / fs != old["recording_duration_s"] or
                    not set(coding.COMMON_CHANNELS) <= set(raw.ch_names)):
                raise ValueError("Current waveform dimensions differ from audited common-eight source")
            phase_rows = coding.read_tsv(phase_file)
            candidates, candidate_rejects = natural.make_candidates(events, phase_rows, clock, raw.n_times, fs, parent)
            selected, selection = natural.select_case_control(candidates, [e["time_s"] for e in cases], header.stem, parent)
            report.update(raw_recorded_request_count=len(cases), eligible_action_windows=len(candidates), selection=selection,
                          candidate_rejections=candidate_rejects, onset_rejections=onset_rejects,
                          selected_classes=dict(Counter(e["label"] for e in selected)), selected_windows=len(selected),
                          clock_max_error_ms=clock["max_error_ms"], selection_recording_id=header.stem)
            if not selected:
                report.update(status="clock_passed_no_selected_windows", reason="unchanged_protocol_selected_zero_windows")
                audits.append(report)
                continue
            needed = len(selected) * 8 * 448 * 4 + 4096
            if shutil.disk_usage(output).free - needed < args.min_free_gib * 1024**3:
                raise ValueError("Disk reserve prevents external waveform write")
            target = output / "waveforms" / ("itu_" + hashlib.sha256(old["header_member"].encode()).hexdigest()[:20] + ".npy")
            target.parent.mkdir(exist_ok=True)
            array = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(len(selected), 8, 448))
            for index, event in enumerate(selected):
                first, last = event["first"], event["last"]
                value = raw.get_data(picks=list(coding.COMMON_CHANNELS), start=first, stop=last).astype(np.float32)
                if value.shape != (8, 448) or not np.isfinite(value).all():
                    raise ValueError("Nonfinite or wrong-shaped external EEG window")
                array[index] = value
                guard = event["clock_guard_s"]
                horizons = tuple(parent["history_features"]["horizons_seconds"])
                pre = natural.history_features(history, available, first / fs - guard, horizons)
                decision = natural.history_features(history, available, event["timing"]["decision_time_s"] - guard, horizons)
                evidence = Path(event["evidence_path"])
                uid = f"aicoding-itu-external-v3-{header.stem}-{index:06d}"
                row = {"id": uid, "dataset": "zenodo:20801101", "canonical_dataset_id": "zenodo:20120211",
                       "subject": subject, "session": old["session"], "site": "itu", "split": "external_test",
                       "recording_id": "itu:" + header.stem, "selection_recording_id": header.stem,
                       "eeg_path": str(target.relative_to(output)), "eeg_index": index, "data_origin": "real_recording",
                       "sample_rate_hz": fs, "channels": list(coding.COMMON_CHANNELS), "window_start_s": first / fs,
                       "window_end_s": last / fs, "event_time_s": event["time_s"], **event["timing"],
                       "clock_uncertainty_guard_s": guard, "label": event["label"], "label_source": "recorded_event",
                       "target_kind": "observed_action", "label_evidence_uri": SOURCE + "/files/bids_behavioral_itu.tar.gz#" + str(evidence.relative_to(root)),
                       "label_evidence_sha256": event["evidence_sha256"], "label_evidence_row": event["evidence_row"],
                       "label_evidence_event_type": event["event_type"], "eeg_source_sha256": waveform_sha,
                       "eeg_header_sha256": old["header_sha256"], "eeg_source_uri": SOURCE + "/files/bids_eeg_itu.tar.gz#" + waveform_member,
                       "environment_evidence_uri": SOURCE + "/files/bids_behavioral_itu.tar.gz#" + old["behavior_phase_member"],
                       "environment_sha256": phase_sha, "clock_sync": clock, **event["context"],
                       "environment_ai_availability": event["ai_availability"],
                       "preprocessing": {"source_preprocessed": True, "online_causality_verified": False,
                                         "assumed_filter_lookahead_s": 4.125, "feature_availability_verified": False,
                                         "evidence_uri": SOURCE + "/files/scripts.tar.gz#scripts/01-b_load.ipynb"},
                       "cohort_protocol_sha256": args.expected_protocol_sha256, "parent_cohort_protocol_sha256": fixed["uniba_v3_protocol_sha256"],
                       "sampling_design": "case_control_enriched", "case_control_role": event["case_control_role"],
                       "control_availability_stratum": event.get("control_availability_stratum"), "quiet_window_filter_applied": False,
                       "human_identity_disjointness_verified": False,
                       "history_pre_eeg": pre, "history_pre_decision_guarded": decision,
                       "history_evidence": [{"acquisition": acq, "uri": SOURCE + "/files/bids_behavioral_itu.tar.gz#" + str(Path(v["path"]).relative_to(root)), "sha256": v["sha256"]}
                                            for acq, v in history_files.items() if acq != "gpt"]}
                validate_timing(row)
                rows.append(row)
                features.append({k: row[k] for k in ("id", "subject", "session", "recording_id", "site", "split", "label", "case_control_role", "sampling_design", "cohort_protocol_sha256", "history_pre_eeg", "history_pre_decision_guarded")})
                labels.append({k: row[k] for k in ("id", "site", "split", "label", "label_source", "target_kind", "label_evidence_uri", "label_evidence_sha256", "label_evidence_row", "label_evidence_event_type", "event_time_s")})
            array.flush(); del array
            artifacts[str(target.relative_to(output))] = {"sha256": digest(target), "bytes": target.stat().st_size}
            report.update(status="prepared", prepared_array_sha256=digest(target), prepared_array_bytes=target.stat().st_size)
            audits.append(report)
            print(json.dumps({k: report[k] for k in ("subject", "status", "selected_windows", "selected_classes")}), flush=True)
        finally:
            raw.close()
    if len(audits) != len(old_records) or any(r["split"] != "external_test" for r in rows):
        raise ValueError("Incomplete recording audit or unexpected partition")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Repeated external observation ID")
    for name, value in verified.items():
        stat = member_path(root, name).stat()
        if list((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)) != value["identity"]:
            raise ValueError("A verified source member changed during cohort construction")
    for key, path in paths.items(): require_hash(path, fixed[key])
    require_hash(protocol_path, args.expected_protocol_sha256)
    require_hash(__file__, source_sha)
    load_helpers(protocol)
    for filename, records in (("manifest.jsonl", rows), ("history_features.jsonl", features), ("labels.jsonl", labels)):
        write_jsonl(output / filename, records)
    write_json(output / "eeg_artifacts.json", artifacts)
    write_json(output / "source_members_verified.json", verified)
    membership = {"partition": "external_test", "training_ids": [], "validation_ids": [],
                  "external_test_ids": sorted(r["id"] for r in rows), "external_subject_ids": sorted({r["subject"] for r in rows}),
                  "all_clock_passed_recordings_processed": sorted(k for k, v in eligibility.items() if v),
                  "model_selection_allowed": False, "human_identity_disjointness_verified": False,
                  "manifest_sha256": digest(output / "manifest.jsonl")}
    write_json(output / "external_membership.json", membership)
    counts = dict(Counter(r["label"] for r in rows))
    aggregate = {"schema": protocol["schema"], "status": "prepared", "site": "itu", "partition": "external_test",
                 "source_release": "zenodo:20801101", "protocol_sha256": args.expected_protocol_sha256,
                 "preparer_sha256": source_sha, "helper_source_sha256": protocol["helper_source_sha256"],
                 "frozen_inputs": fixed, "recordings_audited": len(audits), "clock_pass_recordings_processed": sum(eligibility.values()),
                 "clock_blocked_recordings": sum(not v for v in eligibility.values()), "status_counts": dict(Counter(r["status"] for r in audits)),
                 "selected_windows": len(rows), "counts": {"train": {}, "val": {}, "external_test": counts},
                 "external_observation_subjects": len({r["subject"] for r in rows}), "eeg_array_files": len(artifacts),
                 "prepared_eeg_bytes": sum(v["bytes"] for v in artifacts.values()),
                 "selected_input_hours": len(rows) * 1.75 / 3600,
                 "source_recording_hours_all": sum(r["recording_duration_s"] for r in old_records) / 3600,
                 "clock_passed_recording_hours": sum(r["recording_duration_s"] for r in old_records if eligibility[r["header_member"]]) / 3600,
                 "source_hours_and_window_hours_are_not_additive": True,
                 "raw_request_events_in_clock_passed_records": sum(r.get("raw_recorded_request_count", 0) for r in audits),
                 "cross_site_comparison": comparison, "source_members_rehashed": len(verified),
                 "manifest_sha256": digest(output / "manifest.jsonl"), "history_features_sha256": digest(output / "history_features.jsonl"),
                 "labels_sha256": digest(output / "labels.jsonl"), "membership_sha256": digest(output / "external_membership.json"),
                 "eeg_artifacts_sha256": digest(output / "eeg_artifacts.json"),
                 "online_causality_verified": False, "latent_intent_labels_available": False,
                 "natural_population_precision_estimated": False, "training_started": False, "model_loaded": False,
                 "model_selection_on_itu": False, "predictions_created": False, "gpu_execution": False,
                 "preparation_device": "cpu", "cpu_threads_maximum": 2, "mne_version": mne.__version__,
                 "numpy_version": np.__version__, "elapsed_s": time.time() - started}
    write_json(output / "cohort_audit.json", {"aggregate": aggregate, "recordings": audits, "configuration": vars(args)})
    write_json(output / "cohort.aggregate.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)
    return aggregate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bids-root", "alignment-audit", "uniba-manifest", "parent-protocol", "protocol", "expected-protocol-sha256", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--min-free-gib", type=float, default=35)
    args = p.parse_args()
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        p.error("min-free-gib must be nonnegative")
    args.run_token = secrets.token_hex(16)
    try:
        prepare(args)
    except BaseException as error:
        output = Path(args.output)
        marker = output / "preparation_started.json"
        owned = (marker.is_file() and json.loads(marker.read_text()).get("run_token") == args.run_token
                 and json.loads(marker.read_text()).get("pid") == os.getpid())
        if owned and not (output / "failure.json").exists() and not (output / "cohort.aggregate.json").exists():
            write_json(output / "failure.json", {"status": "failed_or_interrupted", "error_type": type(error).__name__,
                       "reason": str(error), "automatic_reselection_allowed": False})
        raise


if __name__ == "__main__":
    main()
