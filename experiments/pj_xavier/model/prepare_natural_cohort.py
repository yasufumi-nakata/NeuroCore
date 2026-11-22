#!/usr/bin/env python3
"""Prepare preregistered v3 enriched action windows with honest causal history.

CPU-only preparation. The parent v2 manifest, subject split, loader and running
model are read-only. No model predictions, synthetic labels or EEG-based sample
selection are used. Case-control precision is not population precision.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from prepare_ai_coding import (COMMON_CHANNELS, SOURCE, availability_timing,
                               digest, environment_before, find_beh, onset, read_tsv)


EVENT_KINDS = ("key_down", "key_up", "mouse_click", "mouse_move")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def availability_state(rows, clock, context):
    scale, offset = clock["scale_eeg_per_behavior_second"], clock["offset_eeg_seconds"]
    for row in rows:
        t = onset(row)
        if t is not None and t >= 0 and t * scale + offset == context["environment_time_s"]:
            value = str(row.get("ai", "")).strip().lower()
            return "true" if value in {"true", "1"} else ("false" if value in {"false", "0"} else "unknown")
    return "unknown"


def read_recorded_events(beh, clock):
    """Read onsets only; duration/key content/coordinates are never features."""
    scale, offset = clock["scale_eeg_per_behavior_second"], clock["offset_eeg_seconds"]
    events = []
    history = {kind: [] for kind in EVENT_KINDS}
    availability = {kind: False for kind in EVENT_KINDS}
    files = {}
    rejected = Counter()
    for acquisition in ("keypress", "mouseclick", "mousemove", "gpt"):
        paths = sorted(Path(beh).glob(f"*_acq-{acquisition}_*_events.tsv"))
        if len(paths) > 1:
            raise ValueError(f"Ambiguous {acquisition} file")
        if not paths:
            continue
        path = paths[0]
        files[acquisition] = {"path": str(path), "sha256": digest(path)}
        for kind in {"keypress": ("key_down", "key_up"), "mouseclick": ("mouse_click",),
                     "mousemove": ("mouse_move",), "gpt": ()}[acquisition]:
            availability[kind] = True
        for index, row in enumerate(read_tsv(path)):
            t = onset(row)
            if t is None or t < 0:
                rejected[f"{acquisition}_negative_or_invalid_onset"] += 1
                continue
            teeg = t * scale + offset
            label = None
            event_type = None
            if acquisition == "keypress":
                action = row.get("action", "").strip().lower()
                if action in ("down", "press", "pressed", "keydown"):
                    history["key_down"].append(teeg)
                    label, event_type = "keyboard", "key_down"
                elif action in ("up", "release", "released", "keyup"):
                    history["key_up"].append(teeg)
                else:
                    rejected["unrecognized_key_action"] += 1
            elif acquisition == "mouseclick":
                history["mouse_click"].append(teeg)
                label, event_type = "mouse_click", "mouseclick"
            elif acquisition == "mousemove":
                history["mouse_move"].append(teeg)
            elif row.get("trial_type") == "GPTSentPrompt":
                label, event_type = "ai_prompt_submit", "GPTSentPrompt"
            if label:
                events.append({"time_s": teeg, "behavior_time_s": t, "label": label,
                               "event_type": event_type, "acquisition": acquisition,
                               "evidence_path": str(path), "evidence_row": index + 2,
                               "evidence_sha256": files[acquisition]["sha256"]})
    return events, {k: np.array(sorted(v), dtype=np.float64) for k, v in history.items()}, availability, files, dict(rejected)


def history_features(history, availability, cutoff_s, horizons=(1, 5, 15, 60)):
    if not math.isfinite(cutoff_s):
        raise ValueError("Finite history cutoff required")
    features = {}
    latest = []
    for kind in EVENT_KINDS:
        values = history[kind]
        right = int(np.searchsorted(values, cutoff_s, side="left"))
        features[f"{kind}_source_available"] = bool(availability[kind])
        for horizon in horizons:
            left = int(np.searchsorted(values, cutoff_s - horizon, side="left"))
            features[f"{kind}_count_{horizon}s"] = right - left
        if right:
            last = float(values[right - 1])
            latest.append(last)
            features[f"{kind}_seconds_since_last_capped_60"] = min(60., cutoff_s - last)
        else:
            features[f"{kind}_seconds_since_last_capped_60"] = None
    max_used = max(latest) if latest else None
    if max_used is not None and not max_used < cutoff_s:
        raise AssertionError("Future history event entered features")
    return {"cutoff_s": cutoff_s, "max_source_onset_used_s": max_used,
            "source_fields": ["onset", "action only for key-down/up classification"],
            "target_or_future_events_included": False, "features": features}


def make_candidates(events, otree_rows, clock, n_samples, fs, protocol):
    settings = protocol["signal"]
    width = int(settings["window_samples"])
    guard = clock["max_error_ms"] / 1000
    candidates = []
    rejected = Counter()
    for event in events:
        last = math.floor((event["time_s"] - settings["eeg_window_minimum_lead_s"] - guard) * fs)
        first = last - width
        if first < 0 or last > n_samples:
            rejected[(event["label"], "outside_eeg")] += 1
            continue
        context = environment_before(otree_rows, clock, first / fs - guard)
        if context is None:
            rejected[(event["label"], "no_guarded_pre_eeg_context")] += 1
            continue
        timing = availability_timing(last / fs, settings["eeg_window_minimum_lead_s"],
                                     settings["assumed_filter_lookahead_s"])
        if timing["decision_time_s"] + timing["minimum_lead_s"] > event["time_s"] - guard + 1e-8:
            raise AssertionError("Observed action guard was violated")
        candidates.append({**event, "first": first, "last": last, "context": context,
                           "ai_availability": availability_state(otree_rows, clock, context),
                           "timing": timing, "clock_guard_s": guard})
    return candidates, {f"{label}:{reason}": n for (label, reason), n in rejected.items()}


def overlaps(candidate, selected):
    return any(candidate["first"] < other["last"] and other["first"] < candidate["last"] for other in selected)


def select_case_control(candidates, all_case_times, recording_id, protocol):
    settings = protocol["selection"]
    selected = []
    rejected = Counter()
    cases = sorted((r for r in candidates if r["label"] == "ai_prompt_submit"),
                   key=lambda r: (r["time_s"], r["acquisition"], r["evidence_row"]))
    for event in cases:
        if overlaps(event, selected):
            rejected["overlapping_case_window"] += 1
        else:
            selected.append({**event, "case_control_role": "case"})
    kept_cases = list(selected)
    strata = Counter(e["ai_availability"] for e in kept_cases)
    quotas = {(state, label): count * n for state, count in strata.items()
              for label, n in settings["control_label_quotas_per_case"].items()}
    if not kept_cases:
        quotas = {("any", label): n for label, n in settings["zero_case_recording_controls"].items()}
    excluded_radius = protocol["control_definition"]["exclude_within_seconds_of_any_case_event"]
    times = np.array(sorted(all_case_times), dtype=np.float64)
    controls = []
    for event in candidates:
        if event["label"] == "ai_prompt_submit":
            continue
        pos = int(np.searchsorted(times, event["time_s"]))
        neighbors = times[max(0, pos - 1):pos + 1]
        if any(abs(float(t) - event["time_s"]) <= excluded_radius for t in neighbors):
            rejected["control_matches_request_microcluster"] += 1
            continue
        controls.append(event)
    counts = Counter()
    for (state, label), quota in sorted(quotas.items()):
        pool = [e for e in controls if e["label"] == label and (state == "any" or e["ai_availability"] == state)]
        def rank(event):
            return hashlib.sha256(f"{settings['seed']}:{recording_id}:{event['acquisition']}:{event['evidence_row']}".encode()).hexdigest()
        for event in sorted(pool, key=rank):
            if counts[(state, label)] >= quota:
                break
            if overlaps(event, selected):
                rejected["control_overlaps_priority_case_or_retained_control"] += 1
                continue
            selected.append({**event, "case_control_role": "control", "control_availability_stratum": state})
            counts[(state, label)] += 1
    result = sorted(selected, key=lambda e: (e["first"], e["last"], e["evidence_row"]))
    if any(a["last"] > b["first"] for a, b in zip(result, result[1:])):
        raise AssertionError("Selected EEG windows overlap")
    return result, {"eligible_case_windows": len(cases), "retained_cases": len(kept_cases),
                    "retained_controls": len(result) - len(kept_cases), "rejected": dict(rejected),
                    "control_quotas": [{"ai_availability": state, "label": label, "requested": quota,
                                        "retained": counts[(state, label)], "deficit": quota - counts[(state, label)]}
                                       for (state, label), quota in sorted(quotas.items())]}


def load_frozen_inputs(protocol_path, parent, root, expected_protocol_sha):
    protocol_bytes = Path(protocol_path).read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    if protocol_sha != expected_protocol_sha:
        raise ValueError("Protocol bytes do not match the registered SHA-256")
    protocol = json.loads(protocol_bytes)
    expected = protocol["frozen_inputs"]
    for filename, key in [("manifest.jsonl", "parent_manifest_sha256"),
                          ("subject_split.json", "parent_subject_split_sha256"),
                          ("alignment_audit.json", "parent_alignment_audit_sha256")]:
        if digest(parent / filename) != expected[key]:
            raise ValueError(f"Frozen parent file changed: {filename}")
    if digest(Path(__file__).with_name("prepare_ai_coding.py")) != expected["frozen_loader_sha256"]:
        raise ValueError("Frozen parent loader changed")
    verification = json.loads((root / "source_verification.json").read_text())
    if verification.get("verified") is not True:
        raise ValueError("Source archive verification is incomplete")
    for name, key in [("bids_eeg_uniba.tar.gz", "source_eeg_archive_sha256"),
                      ("bids_behavioral_uniba.tar.gz", "source_behavior_archive_sha256")]:
        entries = [e for e in verification["sources"] if e["filename"] == name]
        if len(entries) != 1 or entries[0]["sha256"] != expected[key] or entries[0]["ledger_status"] != "complete":
            raise ValueError(f"Source verification mismatch: {name}")
    with (parent / "manifest.jsonl").open() as f:
        parent_rows = [json.loads(line) for line in f if line.strip()]
    lookup = defaultdict(lambda: {"eeg": set(), "environment": set()})
    for row in parent_rows:
        key = (row["subject"], row["session"], row["recording_id"])
        lookup[key]["eeg"].add(row["eeg_source_sha256"])
        lookup[key]["environment"].add(row["environment_sha256"])
    return (protocol, protocol_bytes, protocol_sha, json.loads((parent / "subject_split.json").read_text()),
            json.loads((parent / "alignment_audit.json").read_text()), lookup)


def require_parent_phase_hash(path, expected):
    actual = digest(path)
    if expected != {actual}:
        raise ValueError("oTree phase member differs from frozen parent; old clock cannot transform changed phase events")
    return actual


def prepare(args):
    import mne
    parent, root, output = Path(args.parent_dir).resolve(), Path(args.bids_root).resolve(), Path(args.output).resolve()
    protocol, protocol_bytes, protocol_sha, split, previous, hashes = load_frozen_inputs(
        Path(args.protocol), parent, root, args.expected_protocol_sha256)
    if output.exists():
        raise ValueError("New cohort output already exists; preserve it and inspect its audit")
    output.mkdir(parents=True)
    (output / "protocol.json").write_bytes(protocol_bytes)
    (output / "subject_split.json").write_bytes((parent / "subject_split.json").read_bytes())
    parent_root = Path(previous["configuration"]["bids_root"])
    all_rows = []
    all_features = []
    audits = []
    for old in previous["recordings"]:
        report = {"subject": old["subject"], "session": old["session"], "site": old["site"],
                  "recording_duration_s": old["recording_duration_s"], "status": "blocked"}
        if old["status"] != "prepared" or old.get("clock_sync", {}).get("verified") is not True:
            report["reason"] = "parent_clock_audit_rejected_recording"
            report["parent_clock_reason"] = old.get("clock_sync", {}).get("reason")
            audits.append(report)
            continue
        header = root / Path(old["eeg_header"]).relative_to(parent_root)
        waveform = header.with_suffix(".eeg")
        subject = f"{old['site']}:{old['subject']}"
        key = (subject, old["session"], header.stem)
        waveform_sha = digest(waveform)
        if hashes.get(key, {}).get("eeg") != {waveform_sha}:
            raise ValueError(f"Raw waveform differs from frozen parent: {header.name}")
        raw = mne.io.read_raw_brainvision(header, preload=False, verbose="ERROR")
        fs = float(raw.info["sfreq"])
        if fs != protocol["signal"]["sample_rate_hz"] or tuple(protocol["signal"]["channels"]) != COMMON_CHANNELS:
            raise ValueError("Unexpected montage or sampling rate")
        if not set(COMMON_CHANNELS) <= set(raw.ch_names):
            raise ValueError("Required physical channels missing")
        beh = header.parent.parent / "beh"
        clock = old["clock_sync"]
        phase_file = find_beh(beh, "otree")
        phase_sha = require_parent_phase_hash(phase_file, hashes[key]["environment"])
        phase_rows = read_tsv(phase_file)
        events, history, source_available, history_files, onset_rejects = read_recorded_events(beh, clock)
        cases = [e for e in events if e["label"] == "ai_prompt_submit"]
        candidates, candidate_rejects = make_candidates(events, phase_rows, clock, raw.n_times, fs, protocol)
        selected, selection_report = select_case_control(candidates, [e["time_s"] for e in cases], header.stem, protocol)
        report.update(raw_recorded_request_count=len(cases), eligible_action_windows=len(candidates),
                      candidate_rejections=candidate_rejects, onset_rejections=onset_rejects,
                      selection=selection_report, selected_classes=dict(Counter(e["label"] for e in selected)),
                      clock_max_error_ms=clock["max_error_ms"], split=split[subject])
        if not selected:
            report["reason"] = "no_selected_windows"
            audits.append(report)
            continue
        name = f"uniba_{old['subject']}_{old['session']}_{hashlib.sha256(str(header.relative_to(root)).encode()).hexdigest()[:10]}.npy"
        target = output / "waveforms" / name
        target.parent.mkdir(exist_ok=True)
        array = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32,
                                          shape=(len(selected), len(COMMON_CHANNELS), protocol["signal"]["window_samples"]))
        for index, event in enumerate(selected):
            first, last = event["first"], event["last"]
            eeg = raw.get_data(picks=list(COMMON_CHANNELS), start=first, stop=last).astype(np.float32)
            if eeg.shape != array.shape[1:] or not np.isfinite(eeg).all():
                raise ValueError("Invalid actual EEG window; preserve partial output for audit")
            array[index] = eeg
            guard = event["clock_guard_s"]
            pre_eeg = history_features(history, source_available, first / fs - guard,
                                       tuple(protocol["history_features"]["horizons_seconds"]))
            pre_decision = history_features(history, source_available, event["timing"]["decision_time_s"] - guard,
                                            tuple(protocol["history_features"]["horizons_seconds"]))
            if any(h["cutoff_s"] >= event["time_s"] for h in (pre_eeg, pre_decision)):
                raise AssertionError("History cutoff reaches target time")
            evidence = Path(event["evidence_path"])
            uid = f"aicoding-v3-natural-{old['site']}-{old['subject']}-{old['session']}-{header.stem}-{index:06d}"
            row = {"id": uid, "dataset": "zenodo:20801101", "canonical_dataset_id": "zenodo:20120211",
                   "subject": subject, "session": old["session"], "recording_id": header.stem,
                   "site": old["site"], "split": split[subject], "eeg_path": str(target.relative_to(output)), "eeg_index": index,
                   "data_origin": "real_recording", "sample_rate_hz": fs, "channels": list(COMMON_CHANNELS),
                   "window_start_s": first / fs, "window_end_s": last / fs, "event_time_s": event["time_s"],
                   **event["timing"], "clock_uncertainty_guard_s": guard,
                   "label": event["label"], "label_source": "recorded_event", "target_kind": "observed_action",
                   "label_evidence_uri": SOURCE + "/files/bids_behavioral_uniba.tar.gz#" + str(evidence.relative_to(root)),
                   "label_evidence_sha256": event["evidence_sha256"], "label_evidence_row": event["evidence_row"],
                   "label_evidence_event_type": event["event_type"],
                   "eeg_source_uri": SOURCE + "/files/bids_eeg_uniba.tar.gz#" + str(waveform.relative_to(root)),
                   "eeg_source_sha256": waveform_sha, "eeg_header_sha256": digest(header),
                   "environment_evidence_uri": SOURCE + "/files/bids_behavioral_uniba.tar.gz#" + str(phase_file.relative_to(root)),
                   "environment_sha256": phase_sha, "clock_sync": clock, **event["context"],
                   "environment_ai_availability": event["ai_availability"],
                   "preprocessing": {"source_preprocessed": True, "online_causality_verified": False,
                                     "assumed_filter_lookahead_s": protocol["signal"]["assumed_filter_lookahead_s"],
                                     "feature_availability_verified": False,
                                     "evidence_uri": SOURCE + "/files/scripts.tar.gz#scripts/01-b_load.ipynb"},
                   "cohort_protocol_sha256": protocol_sha, "sampling_design": "case_control_enriched",
                   "case_control_role": event["case_control_role"],
                   "control_availability_stratum": event.get("control_availability_stratum"),
                   "quiet_window_filter_applied": False,
                   "movement_artifact_control": "natural prior key/mouse activity retained; motor/EMG confounding requires history baselines",
                   "history_pre_eeg": pre_eeg, "history_pre_decision_guarded": pre_decision,
                   "history_evidence": [{"acquisition": acq, "uri": SOURCE + "/files/bids_behavioral_uniba.tar.gz#" + str(Path(f["path"]).relative_to(root)),
                                         "sha256": f["sha256"]} for acq, f in history_files.items() if acq != "gpt"]}
            all_rows.append(row)
            all_features.append({k: row[k] for k in ["id", "subject", "session", "recording_id", "split", "label",
                                                      "case_control_role", "sampling_design", "cohort_protocol_sha256",
                                                      "history_pre_eeg", "history_pre_decision_guarded"]})
        array.flush()
        del array
        report.update(status="prepared", selected_windows=len(selected), waveform_sha256=waveform_sha,
                      prepared_array_sha256=digest(target), prepared_array_bytes=target.stat().st_size)
        audits.append(report)
        print(json.dumps({k: report[k] for k in ["subject", "split", "status", "raw_recorded_request_count", "selected_classes"]}), flush=True)
    if not all_rows or set(r["split"] for r in all_rows) != {"train", "val", "test"}:
        raise ValueError("All fixed subject partitions must remain nonempty")
    for name, records in [("manifest.jsonl", all_rows), ("history_features.jsonl", all_features)]:
        (output / name).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records))
    counts = {s: dict(Counter(r["label"] for r in all_rows if r["split"] == s)) for s in ("train", "val", "test")}
    aggregate = {"schema": protocol["schema"], "status": "prepared", "source": SOURCE,
                 "protocol_sha256": protocol_sha, "preparer_sha256": digest(Path(__file__)),
                 "frozen_inputs": protocol["frozen_inputs"], "subject_split_sha256": digest(output / "subject_split.json"),
                 "manifest_sha256": digest(output / "manifest.jsonl"), "history_features_sha256": digest(output / "history_features.jsonl"),
                 "sampling_design": "case_control_enriched", "counts": counts, "selected_windows": len(all_rows),
                 "subjects_per_split": {s: len({r["subject"] for r in all_rows if r["split"] == s}) for s in counts},
                 "raw_request_events_in_clock_passed_records": sum(r.get("raw_recorded_request_count", 0) for r in audits),
                 "selected_request_windows": sum(r["label"] == "ai_prompt_submit" for r in all_rows),
                 "parent_selected_request_windows": 5,
                 "source_recording_hours_all": previous["source_recording_hours"],
                 "clock_passed_recording_hours": sum(r["recording_duration_s"] for r in previous["recordings"]
                                                       if r["status"] == "prepared" and r.get("clock_sync", {}).get("verified") is True) / 3600,
                 "retained_recording_hours": sum(r["recording_duration_s"] for r in audits if r["status"] == "prepared") / 3600,
                 "selected_input_hours": sum(r["window_end_s"] - r["window_start_s"] for r in all_rows) / 3600,
                 "prepared_eeg_bytes": sum(r.get("prepared_array_bytes", 0) for r in audits),
                 "status_counts": dict(Counter(r["status"] for r in audits)),
                 "online_causality_verified": False, "latent_intent_labels_available": False,
                 "natural_population_precision_estimated": False, "independent_of_v2_source_hours": False,
                 "goal_benchmark_status": "blocked_pending_causal_preprocessing_real_harness_goals_and_intent_ground_truth",
                 "training_started": False, "preparation_device": "cpu"}
    write_json(output / "cohort_audit.json", {**aggregate, "configuration": vars(args), "recordings": audits})
    write_json(output / "cohort.aggregate.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)
    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", required=True)
    parser.add_argument("--parent-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        prepare(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        raise
