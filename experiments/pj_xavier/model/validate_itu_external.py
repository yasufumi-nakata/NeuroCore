#!/usr/bin/env python3
"""Recheck every fixed external window, source label and history without inference."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

if not __debug__:
    raise RuntimeError("Independent validation requires Python assertions enabled")

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""): digest.update(block)
    return digest.hexdigest()


def json_file(path):
    return json.loads(Path(path).read_text())


def json_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate(args):
    import numpy as np
    import mne
    started = time.time()
    prepared, root, code = Path(args.prepared).resolve(), Path(args.bids_root).resolve(), Path(args.code_dir).resolve()
    destination = Path(args.output)
    if destination.exists(): raise ValueError("Existing independent validation receipt is preserved")
    summary = json_file(prepared / "cohort.aggregate.json")
    assert summary["status"] == "prepared" and summary["partition"] == "external_test"
    assert summary["manifest_sha256"] == args.expected_manifest_sha256 == sha(prepared / "manifest.jsonl")
    assert sha(code / "prepare_itu_external.py") == summary["preparer_sha256"]
    sys.path.insert(0, str(code))
    external = importlib.import_module("prepare_itu_external")
    assert Path(external.__file__).resolve() == code / "prepare_itu_external.py"
    protocol = json_file(prepared / "protocol.json")
    assert sha(prepared / "protocol.json") == summary["protocol_sha256"]
    natural, coding, archive = external.load_helpers(protocol)
    parent = json_file(prepared / "uniba_v3_protocol.json")
    assert sha(prepared / "uniba_v3_protocol.json") == protocol["frozen_inputs"]["uniba_v3_protocol_sha256"]
    assert sha(args.alignment_audit) == protocol["frozen_inputs"]["itu_alignment_audit_sha256"]
    assert sha(args.uniba_manifest) == protocol["frozen_inputs"]["uniba_v3_manifest_sha256"]
    for file, key in (("history_features.jsonl", "history_features_sha256"), ("labels.jsonl", "labels_sha256"),
                      ("external_membership.json", "membership_sha256"), ("eeg_artifacts.json", "eeg_artifacts_sha256")):
        assert sha(prepared / file) == summary[key]
    rows, labels, histories = (json_rows(prepared / name) for name in ("manifest.jsonl", "labels.jsonl", "history_features.jsonl"))
    assert len(rows) == len(labels) == len(histories) == summary["selected_windows"]
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(row["site"] == "itu" and row["split"] == "external_test" for row in rows)
    membership = json_file(prepared / "external_membership.json")
    assert membership["training_ids"] == membership["validation_ids"] == []
    assert membership["external_test_ids"] == sorted(r["id"] for r in rows)
    for row, label, history in zip(rows, labels, histories):
        assert all(row[key] == value for key, value in label.items())
        assert all(row[key] == value for key, value in history.items())
        external.validate_timing(row)
    artifacts = json_file(prepared / "eeg_artifacts.json")
    assert set(artifacts) == {row["eeg_path"] for row in rows}
    assert set(artifacts) == {str(path.relative_to(prepared)) for path in (prepared / "waveforms").glob("*.npy")}
    arrays = {}
    for name, evidence in artifacts.items():
        path = prepared / name
        assert not path.is_symlink() and path.resolve().is_relative_to(prepared)
        assert sha(path) == evidence["sha256"] and path.stat().st_size == evidence["bytes"]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        selected = [r for r in rows if r["eeg_path"] == name]
        assert array.dtype == np.dtype("float32") and array.shape == (len(selected), 8, 448)
        assert np.isfinite(array).all() and sorted(r["eeg_index"] for r in selected) == list(range(len(selected)))
        arrays[name] = array
    # Freshly hash every source member used by the preparer, including all 26 binaries.
    source_members = json_file(prepared / "source_members_verified.json")
    for name, evidence in source_members.items():
        path = external.member_path(root, name)
        assert sha(path) == evidence["sha256"] and path.stat().st_size == evidence["bytes"]
    clock_audit = json_file(args.alignment_audit)
    comparison = external.compare_cross_site(json_rows(args.uniba_manifest), clock_audit["recordings"])
    assert comparison == summary["cross_site_comparison"]
    groups = defaultdict(list)
    for row in rows: groups[row["recording_id"]].append(row)
    actual_audit = json_file(prepared / "cohort_audit.json")
    assert len(actual_audit["recordings"]) == len(clock_audit["recordings"]) == 26
    eligible_headers = []
    waveform_checks = label_checks = history_checks = 0
    for old in clock_audit["recordings"]:
        header = external.member_path(root, old["header_member"])
        selected_rows = groups.get("itu:" + header.stem, [])
        if not external.clock_eligible(old, protocol):
            assert not selected_rows
            continue
        eligible_headers.append(old["header_member"])
        phase = external.member_path(root, old["behavior_phase_member"])
        natural.require_parent_phase_hash(phase, {old["behavior_phase_sha256"]})
        events, history, available, _, _ = natural.read_recorded_events(header.parent.parent / "beh", old["clock_sync"])
        candidates, _ = natural.make_candidates(events, coding.read_tsv(phase), old["clock_sync"],
                                                  int(old["recording_duration_s"] * 256), 256., parent)
        selected, _ = natural.select_case_control(candidates, [e["time_s"] for e in events if e["label"] == "ai_prompt_submit"], header.stem, parent)
        selected_rows.sort(key=lambda r: r["eeg_index"])
        assert len(selected_rows) == len(selected)
        prior_last = None
        raw = mne.io.read_raw_brainvision(header, preload=False, verbose="ERROR")
        try:
            for row, event in zip(selected_rows, selected):
                assert row["label"] == event["label"] and row["label_evidence_row"] == event["evidence_row"]
                assert row["label_evidence_sha256"] == event["evidence_sha256"] and row["case_control_role"] == event["case_control_role"]
                assert row["label_evidence_event_type"] == event["event_type"] and row["event_time_s"] == event["time_s"]
                assert row["window_start_s"] == event["first"] / 256 and row["window_end_s"] == event["last"] / 256
                assert all(row[key] == value for key, value in event["context"].items())
                assert prior_last is None or event["first"] >= prior_last
                prior_last = event["last"]
                values = raw.get_data(picks=list(coding.COMMON_CHANNELS), start=event["first"], stop=event["last"]).astype(np.float32)
                assert np.array_equal(values, arrays[row["eeg_path"]][row["eeg_index"]])
                waveform_checks += 1; label_checks += 1
                for key in ("history_pre_eeg", "history_pre_decision_guarded"):
                    recomputed = natural.history_features(history, available, row[key]["cutoff_s"], tuple(parent["history_features"]["horizons_seconds"]))
                    assert recomputed == row[key]
                    history_checks += 1
        finally:
            raw.close()
    assert sorted(eligible_headers) == membership["all_clock_passed_recordings_processed"]
    assert waveform_checks == label_checks == len(rows)
    assert history_checks == 2 * len(rows)
    assert summary["counts"] == {"train": {}, "val": {}, "external_test": dict(Counter(r["label"] for r in rows))}
    result = {"schema": "xavier-itu-external-independent-validation-v1", "status": "passed", "validator_sha256": sha(__file__),
              "manifest_sha256": sha(prepared / "manifest.jsonl"), "protocol_sha256": summary["protocol_sha256"],
              "arrays_checked": len(arrays), "all_prepared_arrays_hash_finite_shape_passed": True,
              "source_members_freshly_rehashed": len(source_members), "actual_waveform_windows_compared_sample_for_sample": waveform_checks,
              "recorded_event_labels_rederived": label_checks, "past_history_feature_sets_rederived": history_checks,
              "selected_case_control_ids_match_unchanged_helper": True, "input_windows_nonoverlapping": True,
              "clock_passed_recordings_checked": len(eligible_headers), "failed_clock_records_used": 0,
              "train_or_validation_rows": 0, "cross_site_binary_hash_matches": comparison["exact_binary_hash_matches"],
              "human_identity_disjointness_verified": False, "original_manifest_changed": False,
              "model_loaded": False, "predictions_created": False, "gpu_execution": False, "elapsed_s": time.time() - started}
    external.write_json(destination, result)
    print(json.dumps(result))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "bids-root", "alignment-audit", "uniba-manifest", "code-dir", "expected-manifest-sha256", "output"):
        p.add_argument("--" + name, required=True)
    validate(p.parse_args())


if __name__ == "__main__":
    main()
