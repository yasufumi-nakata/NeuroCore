#!/usr/bin/env python3
"""Audit AI-assisted-coding clocks and prepare subject-held-out EEG action windows.

The release is already filtered offline. Even aligned windows do not establish
causal online preprocessing or latent mental intentions. Only recorded actions
are targets. The oTree/stimulus channels are synchronization evidence, never EEG
inputs. Public source XDF recordings are not included in the release.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


COMMON_CHANNELS = ("CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4")
PHASE_CODES = {
    ("eyeCalibrationPageBegin", None): 10,
    ("taskDescription", False): 30, ("taskBegin", False): 40, ("nasatlxBegin", False): 50,
    ("taskDescription", True): 70, ("taskBegin", True): 80, ("nasatlxBegin", True): 90,
}
SOURCE = "https://zenodo.org/records/20801101"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def onset(row):
    try:
        value = float(row["onset"])
        return value if math.isfinite(value) else None
    except (ValueError, TypeError, KeyError):
        return None


def phase_anchors(otree_rows):
    points = []
    for row in otree_rows:
        t = onset(row)
        kind = row.get("trial_type", "")
        text = str(row.get("ai", "")).strip().lower()
        ai = True if text in {"true", "1"} else (False if text in {"false", "0"} else None)
        code = PHASE_CODES.get((kind, None if kind == "eyeCalibrationPageBegin" else ai))
        if code and t is not None and t >= 0:
            points.append((t, code))
    return points


def phase_transitions(values: np.ndarray, sample_rate: float):
    rounded = np.rint(values).astype(np.int64)
    # Ignore values that are not close to integer codes rather than interpreting
    # corrupted continuous signals as a synchronization channel.
    valid = np.isfinite(values) & (np.abs(values - rounded) < .05)
    indices = np.flatnonzero(np.r_[True, rounded[1:] != rounded[:-1]])
    allowed = set(PHASE_CODES.values())
    return [(int(i) / sample_rate, int(rounded[i])) for i in indices if valid[i] and rounded[i] in allowed]


def fit_clock(eeg_points, behavioral_points, max_error_ms=100, max_drift_ppm=5000,
              min_anchors=4, min_span_s=120):
    """Fit EEG seconds = scale * behavioral seconds + offset.

    Only codes occurring once on BOTH sides are eligible. There is no manual
    offset, forced matching or post-hoc outlier deletion to manufacture a pass.
    All usable anchors must pass residual and leave-one-anchor-out checks.
    """
    ec, bc = defaultdict(list), defaultdict(list)
    for t, code in eeg_points:
        ec[int(code)].append(float(t))
    for t, code in behavioral_points:
        bc[int(code)].append(float(t))
    codes = sorted(c for c in ec.keys() & bc.keys() if len(ec[c]) == len(bc[c]) == 1)
    result: dict[str, Any] = {
        "verified": False, "method": "affine_phase_anchor_alignment_with_leave_one_anchor_out",
        "eligible_codes": codes, "anchor_count": len(codes),
        "excluded_duplicate_codes": sorted(c for c in ec.keys() | bc.keys() if len(ec[c]) > 1 or len(bc[c]) > 1),
        "max_allowed_error_ms": max_error_ms, "max_allowed_drift_ppm": max_drift_ppm,
        "claim_scope": "consistency of published EEG phase channel with published behavioral phase timestamps; no original-XDF verification",
    }
    if len(codes) < min_anchors:
        result["reason"] = "insufficient_unique_phase_anchors"
        return result
    x = np.array([bc[c][0] for c in codes])
    y = np.array([ec[c][0] for c in codes])
    if np.ptp(x) < min_span_s:
        result["reason"] = "insufficient_anchor_time_span"
        return result
    order = np.argsort(x)
    if np.any(np.diff(y[order]) <= 0):
        result["reason"] = "nonmonotonic_phase_correspondence"
        return result
    design = np.stack((x, np.ones_like(x)), axis=1)
    scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - (x * scale + offset)
    loo = []
    for index in range(len(x)):
        keep = np.arange(len(x)) != index
        a, b = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        loo.append(float(y[index] - (a * x[index] + b)))
    error_ms = max(float(np.max(np.abs(residual))), max(abs(v) for v in loo)) * 1000
    drift_ppm = abs(float(scale) - 1) * 1e6
    result.update({"scale_eeg_per_behavior_second": float(scale), "offset_eeg_seconds": float(offset),
                   "max_error_ms": error_ms, "max_fit_residual_ms": float(np.max(np.abs(residual))) * 1000,
                   "drift_ppm": drift_ppm, "anchor_span_s": float(np.ptp(x)),
                   "anchors": [{"phase_code": c, "behavior_s": float(xx), "eeg_s": float(yy),
                                "residual_ms": float(rr) * 1000, "leave_one_out_error_ms": ll * 1000}
                               for c, xx, yy, rr, ll in zip(codes, x, y, residual, loo)]})
    if not math.isfinite(error_ms) or scale <= 0 or drift_ppm > max_drift_ppm:
        result["reason"] = "implausible_clock_scale"
    elif error_ms > max_error_ms:
        result["reason"] = "phase_alignment_error_above_threshold"
    else:
        result["verified"] = True
        result["reason"] = "all_unique_anchor_and_leave_one_out_checks_passed"
    return result


def split_subjects(subjects, seed=20261122):
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required for held-out subject splits")
    order = sorted(subjects, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    n_train = max(1, int(len(order) * .6))
    n_val = max(1, int(len(order) * .2))
    return {subject: "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            for i, subject in enumerate(order)}


def find_beh(beh: Path, acquisition: str):
    matches = sorted(beh.glob(f"*_acq-{acquisition}_*_events.tsv"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {acquisition} event file; found {len(matches)}")
    return matches[0]


def action_events(beh: Path, clock: dict, include_ai_submit=True):
    scale, offset = clock["scale_eeg_per_behavior_second"], clock["offset_eeg_seconds"]
    events, guard_times = [], []
    for acquisition in ("keypress", "mouseclick", "gpt"):
        paths = sorted(beh.glob(f"*_acq-{acquisition}_*_events.tsv"))
        if len(paths) > 1:
            raise ValueError(f"Ambiguous {acquisition} recording")
        if not paths:
            continue
        path = paths[0]
        for index, row in enumerate(read_tsv(path)):
            t = onset(row)
            if t is None or t < 0:
                continue
            teeg = t * scale + offset
            label = None
            if acquisition == "keypress":
                guard_times.append(teeg)
                # The verified v2 UniBa TSV uses `down` / `up`. Legacy spellings
                # remain explicit aliases; release events are guards, not targets.
                if row.get("action", "").strip().lower() in ("down", "press", "pressed", "keydown"):
                    label = "keyboard"
            elif acquisition == "mouseclick":
                guard_times.append(teeg)
                label = "mouse_click"
            elif include_ai_submit and row.get("trial_type") == "GPTSentPrompt":
                guard_times.append(teeg)
                label = "ai_prompt_submit"
            if label:
                events.append({"time_s": teeg, "behavior_time_s": t, "label": label,
                               "evidence_path": str(path), "evidence_row": index + 2})
    # Near-simultaneous UI events can describe the same submitted action. Resolve
    # them deterministically with a declared priority rather than duplicate EEG.
    priority = {"ai_prompt_submit": 3, "mouse_click": 2, "keyboard": 1}
    clustered = []
    for event in sorted(events, key=lambda x: x["time_s"]):
        if clustered and event["time_s"] - clustered[-1]["time_s"] <= .15:
            if priority[event["label"]] > priority[clustered[-1]["label"]]:
                clustered[-1] = event
        else:
            clustered.append(event)
    return clustered, np.array(sorted(guard_times))


def environment_before(otree_rows, clock, cutoff_s):
    scale, offset = clock["scale_eeg_per_behavior_second"], clock["offset_eeg_seconds"]
    prior = [(onset(r) * scale + offset, r) for r in otree_rows if onset(r) is not None
             and onset(r) >= 0 and onset(r) * scale + offset <= cutoff_s]
    if not prior:
        return None
    t, row = max(prior, key=lambda x: x[0])
    kind = str(row.get("trial_type", "unknown"))
    ai = str(row.get("ai", "unknown"))
    # No future GPT prompt, target key, click coordinates or action label appears.
    return {"environment_text": f"Java programming experiment. Last recorded experiment page or event: {kind}. AI availability recorded at that event: {ai}.",
            "environment_time_s": t, "environment_source": "predecision_harness",
            "environment_contains_target_label": False,
            "environment_scope": "coarse recorded experiment-state log; not a screen reconstruction"}


def availability_timing(window_end_s, window_lead_s, filter_lookahead_s):
    if filter_lookahead_s < 0 or window_lead_s <= filter_lookahead_s:
        raise ValueError("EEG window-to-event gap must exceed assumed offline-filter lookahead")
    return {"decision_time_s": window_end_s + filter_lookahead_s,
            "feature_available_time_s": window_end_s + filter_lookahead_s,
            "minimum_lead_s": window_lead_s - filter_lookahead_s,
            "eeg_window_minimum_lead_s": window_lead_s,
            "assumed_filter_lookahead_s": filter_lookahead_s,
            "feature_availability_verified": False}


def main(args):
    import mne
    root, output = Path(args.bids_root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    verification_path = root / "source_verification.json"
    verification = json.loads(verification_path.read_text()) if verification_path.exists() else None
    if args.prepare:
        if not verification or verification.get("verified") is not True:
            raise ValueError("Preparation requires completed archive checksum verification; diagnostic partial archives are forbidden")
        required_archives = {f"bids_eeg_{args.site}.tar.gz", f"bids_behavioral_{args.site}.tar.gz"}
        verified_archives = {r.get("filename") for r in verification.get("sources", [])
                             if r.get("ledger_status") == "complete" and r.get("reverified_before_extraction") is True}
        if not required_archives <= verified_archives:
            raise ValueError("EEG and behavioral archives must both have completed pre-extraction verification")
        availability_timing(0, args.lead_s, args.filter_lookahead_s)
    files = [p for p in sorted(root.rglob("*_eeg.vhdr")) if "derivatives" not in p.parts]
    records = []
    for path in files:
        match = re.search(r"(sub-[^_]+)_(ses-[^_]+)_", path.name)
        if not match:
            continue
        records.append((path, match.group(1), match.group(2)))
    subjects = sorted({f"{args.site}:{s}" for _, s, _ in records})
    assignments = split_subjects(subjects, args.seed) if args.prepare else {}
    if args.prepare and (output / "manifest.jsonl").exists():
        raise ValueError("Prepared output already exists; preserve its split assignment and use a new output")
    audits, rows_all, class_counts = [], [], Counter()
    for path, subject, session in records:
        record = {"subject": subject, "session": session, "site": args.site, "eeg_header": str(path), "status": "blocked"}
        try:
            raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
            fs = float(raw.info["sfreq"])
            duration = raw.n_times / fs
            record.update(sample_rate_hz=fs, recording_duration_s=duration, channels=list(raw.ch_names))
            if "otree_phase" not in raw.ch_names:
                raise ValueError("otree_phase synchronization channel is absent")
            missing = [c for c in COMMON_CHANNELS if c not in raw.ch_names]
            if missing:
                raise ValueError(f"Common physical EEG channels absent: {missing}")
            beh = path.parent.parent / "beh"
            phase_file = find_beh(beh, "otree")
            otree_rows = read_tsv(phase_file)
            eeg_phase = raw.get_data(picks=["otree_phase"])[0] * args.phase_scale
            clock = fit_clock(phase_transitions(eeg_phase, fs), phase_anchors(otree_rows),
                              max_error_ms=args.max_clock_error_ms, max_drift_ppm=args.max_clock_drift_ppm)
            clock["explicit_phase_channel_scale"] = args.phase_scale
            record["clock_sync"] = clock
            if not clock["verified"]:
                raise ValueError(f"Clock synchronization failed: {clock['reason']}")
            events, guard_times = action_events(beh, clock, include_ai_submit=not args.no_ai_submit)
            if args.exclude_mouse_movement:
                for movement in beh.glob("*_acq-mousemove_*_events.tsv"):
                    ts = [onset(r) for r in read_tsv(movement)]
                    mapped = [t * clock["scale_eeg_per_behavior_second"] + clock["offset_eeg_seconds"]
                              for t in ts if t is not None and t >= 0]
                    guard_times = np.sort(np.r_[guard_times, mapped])
            samples = round(args.window_s * fs)
            clock_guard_s = clock["max_error_ms"] / 1000
            candidates, discarded = [], Counter()
            previous_end = -math.inf
            for event in events:
                target = event["time_s"]
                last = math.floor((target - args.lead_s - clock_guard_s) * fs)
                first = last - samples
                start, end = first / fs, last / fs
                if first < 0 or last > raw.n_times:
                    discarded["outside_eeg"] += 1
                    continue
                if start < previous_end:
                    discarded["overlapping_input_window"] += 1
                    continue
                if np.searchsorted(guard_times, end, side="left") > np.searchsorted(guard_times, start, side="left"):
                    discarded["other_recorded_action_in_input"] += 1
                    continue
                context = environment_before(otree_rows, clock, start)
                if context is None:
                    discarded["no_causal_environment_log"] += 1
                    continue
                if args.max_windows_per_recording and len(candidates) >= args.max_windows_per_recording:
                    discarded["explicit_per_recording_limit"] += 1
                    continue
                candidates.append((event, first, last, context))
                previous_end = end
            record.update(status="aligned", candidate_windows=len(candidates), candidate_classes=dict(Counter(e[0]["label"] for e in candidates)),
                          discarded=dict(discarded), samples_per_window=samples, selected_channels=list(COMMON_CHANNELS),
                          excluded_input_channels=[c for c in raw.ch_names if c not in COMMON_CHANNELS])
            if args.prepare and candidates:
                waveform = path.with_suffix(".eeg")
                if not waveform.exists():
                    raise ValueError("Cannot find the BrainVision binary for source hashing")
                waveform_sha = digest(waveform)
                phase_sha = digest(phase_file)
                evidence_hashes = {}
                out_name = f"{args.site}_{subject}_{session}_{hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest()[:10]}.npy"
                out_path = output / "waveforms" / out_name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                                shape=(len(candidates), len(COMMON_CHANNELS), samples))
                for index, (event, first, last, context) in enumerate(candidates):
                    eeg = raw.get_data(picks=list(COMMON_CHANNELS), start=first, stop=last).astype(np.float32)
                    if eeg.shape != (len(COMMON_CHANNELS), samples) or not np.isfinite(eeg).all():
                        discarded["invalid_waveform"] += 1
                        continue
                    arr[index] = eeg
                    evidence = Path(event["evidence_path"])
                    if evidence not in evidence_hashes:
                        evidence_hashes[evidence] = digest(evidence)
                    archive_eeg = SOURCE + f"/files/bids_eeg_{args.site}.tar.gz#" + str(waveform.relative_to(root))
                    archive_beh = SOURCE + f"/files/bids_behavioral_{args.site}.tar.gz#" + str(evidence.relative_to(root))
                    archive_phase = SOURCE + f"/files/bids_behavioral_{args.site}.tar.gz#" + str(phase_file.relative_to(root))
                    row = {
                        "id": f"aicoding20801101-{args.site}-{subject}-{session}-{path.stem}-{index:06d}",
                        "dataset": "zenodo:20801101", "canonical_dataset_id": "zenodo:20120211",
                        "subject": f"{args.site}:{subject}", "session": session, "recording_id": path.stem,
                        "site": args.site, "split": assignments[f"{args.site}:{subject}"],
                        "eeg_path": str(out_path.relative_to(output)), "eeg_index": index,
                        "data_origin": "real_recording", "sample_rate_hz": fs, "channels": list(COMMON_CHANNELS),
                        "window_start_s": first / fs, "window_end_s": last / fs,
                        "event_time_s": event["time_s"],
                        **availability_timing(last / fs, args.lead_s, args.filter_lookahead_s),
                        "clock_uncertainty_guard_s": clock_guard_s,
                        "label": event["label"], "label_source": "recorded_event", "target_kind": "observed_action",
                        "eeg_source_uri": archive_eeg, "eeg_source_sha256": waveform_sha,
                        "label_evidence_uri": archive_beh, "label_evidence_sha256": evidence_hashes[evidence],
                        "label_evidence_row": event["evidence_row"],
                        "environment_evidence_uri": archive_phase, "environment_sha256": phase_sha,
                        "clock_sync": clock, **context,
                        "preprocessing": {"source_preprocessed": True, "online_causality_verified": False,
                                          "assumed_filter_lookahead_s": args.filter_lookahead_s,
                                          "feature_availability_verified": False,
                                          "availability_assumption": "Published MNE 1.10.2 export recipe at 256 Hz has combined FIR future support 4.125 s; binary-to-recipe equivalence has not been verified",
                                          "evidence_uri": SOURCE + "/files/scripts.tar.gz#scripts/01-b_load.ipynb",
                                          "description": "Published export applies offline 3–40 Hz and notch filtering before BrainVision; original XDF not distributed"},
                        "movement_artifact_control": "excludes recorded mouse movements" if args.exclude_mouse_movement else "mouse movement not excluded; movement/EMG confounding remains",
                    }
                    rows_all.append(row)
                    class_counts[(row["split"], row["label"])] += 1
                arr.flush()
                del arr
                record["status"] = "prepared"
                record["prepared_windows"] = sum(1 for r in rows_all if r["subject"] == f"{args.site}:{subject}" and r["recording_id"] == path.stem)
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        audits.append(record)
        print(json.dumps({k: record[k] for k in ("subject", "site", "status", "candidate_windows", "error") if k in record}), flush=True)
    if args.prepare:
        (output / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows_all))
        (output / "subject_split.json").write_text(json.dumps(assignments, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "source": SOURCE, "site": args.site, "recording_count": len(records), "subject_count": len(subjects),
        "status_counts": dict(Counter(r["status"] for r in audits)), "prepared_windows": len(rows_all),
        "counts": {split: {label: n for (s, label), n in class_counts.items() if s == split} for split in ("train", "val", "test")},
        "manifest_sha256": digest(output / "manifest.jsonl") if args.prepare else None,
        "source_recording_hours": sum(r.get("recording_duration_s", 0) for r in audits) / 3600,
        "selected_unique_input_hours": sum((r["window_end_s"] - r["window_start_s"]) for r in rows_all) / 3600,
        "per_split_unique_input_hours": {s: sum(r["window_end_s"] - r["window_start_s"] for r in rows_all if r["split"] == s) / 3600
                                          for s in ("train", "val", "test")},
        "prepared_eeg_bytes": sum((output / name).stat().st_size for name in {r["eeg_path"] for r in rows_all}),
        "online_causality_verified": False, "latent_intent_labels_available": False,
        "source_verification": verification,
        "goal_benchmark_status": "blocked_pending_causal_preprocessing_real_harness_goals_and_intent_ground_truth",
        "configuration": vars(args), "recordings": audits,
    }
    (output / "alignment_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("recordings", "configuration")}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bids-root", required=True)
    p.add_argument("--site", choices=("uniba", "itu"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prepare", action="store_true", help="Write actual EEG windows after each recording passes clock checks")
    p.add_argument("--seed", type=int, default=20261122)
    p.add_argument("--window-s", type=float, default=1.75)
    p.add_argument("--lead-s", type=float, default=4.5, help="Minimum EEG window-end to observed-action gap, including assumed offline-filter lookahead")
    p.add_argument("--filter-lookahead-s", type=float, default=4.125,
                   help="Declared conditional feature availability delay; not proof that released binaries reproduce the published recipe")
    p.add_argument("--max-clock-error-ms", type=float, default=100)
    p.add_argument("--max-clock-drift-ppm", type=float, default=5000)
    p.add_argument("--phase-scale", type=float, default=1, help="Explicit reader-unit correction for integer oTree codes, only after checking actual file units")
    p.add_argument("--exclude-mouse-movement", action="store_true")
    p.add_argument("--no-ai-submit", action="store_true")
    p.add_argument("--max-windows-per-recording", type=int, default=0)
    args = p.parse_args()
    if args.window_s <= 0 or args.lead_s <= 0 or args.max_clock_error_ms <= 0:
        p.error("Positive window, lead and clock error thresholds are required")
    main(args)
