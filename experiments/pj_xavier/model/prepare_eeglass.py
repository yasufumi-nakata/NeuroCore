#!/usr/bin/env python3
"""Prepare real EEGlass pre-action windows; no fabricated environment/intention.

Chronological blocks are a one-participant pilot, not cross-subject evidence.
All observed key/button/wheel events in the pre-action interval are excluded.
This event filter does not prove an artifact-free EEG signal.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def press_class(description: str):
    if not description.endswith(" pressed"):
        return None
    key = description[:-len(" pressed")]
    if key.startswith("MouseButton"):
        return "mouse_click"
    if key.startswith("MouseWheel"):
        return "mouse_scroll"
    # EEGlass keyboard key labels are recorded as individual nonempty tokens.
    # Record exact original label in provenance; do not infer mental intent.
    if key and "Mouse" not in key:
        return "keyboard"
    return None


def prepare(args):
    import mne
    source = Path(args.set_file).resolve()
    raw = mne.io.read_raw_eeglab(source, preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    if not math.isclose(fs, args.sample_rate):
        # Actual anti-aliased resampling occurs once before selecting windows.
        raw.resample(args.sample_rate, npad="auto", verbose="ERROR")
        fs = float(raw.info["sfreq"])
    duration = raw.n_times / fs
    all_events = sorted((float(a["onset"]), str(a["description"])) for a in raw.annotations)
    event_times = np.asarray([t for t, _ in all_events])
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.jsonl").exists():
        raise ValueError("Output already contains a manifest; choose a new directory to preserve split identity")
    set_sha = digest(source)
    fdt = source.with_suffix(".fdt")
    fdt_sha = digest(fdt) if fdt.exists() else None
    source_sha = fdt_sha or set_sha
    source_url = "https://zenodo.org/records/4453396"
    source_uri = source_url + "/files/" + (fdt.name if fdt.exists() else source.name)
    usable = []
    discarded = Counter()
    # Boundaries refer to the complete recording, independent of label distribution.
    boundaries = (duration * .6, duration * .8)
    n_samples = round((args.lookback_s - args.lead_s) * fs)
    previous_end = -math.inf
    for event_index, (t, description) in enumerate(all_events):
        label = press_class(description)
        if label is None:
            continue
        start, end = t - args.lookback_s, t - args.lead_s
        if start < 0 or end > duration:
            discarded["outside_recording"] += 1
            continue
        if any(start < b + args.embargo_s and end > b - args.embargo_s for b in boundaries):
            discarded["split_boundary_embargo"] += 1
            continue
        count = np.searchsorted(event_times, end, side="left") - np.searchsorted(event_times, start, side="left")
        if count:
            discarded["other_recorded_event_in_input"] += 1
            continue
        if start < previous_end:
            discarded["overlaps_previous_input"] += 1
            continue
        first = round(start * fs)
        last = first + n_samples
        # Rounding must not put samples after the specified pre-action guard.
        if last > math.floor(end * fs + 1e-8):
            first -= 1
            last -= 1
        actual_start, actual_end = first / fs, last / fs
        if first < 0 or last > raw.n_times:
            discarded["rounded_outside_recording"] += 1
            continue
        actual_count = (np.searchsorted(event_times, actual_end, side="left")
                        - np.searchsorted(event_times, actual_start, side="left"))
        if actual_count or actual_start < previous_end:
            discarded["rounded_event_or_window_overlap"] += 1
            continue
        x = raw.get_data(start=first, stop=last).astype(np.float32)
        if not np.isfinite(x).all() or x.shape[1] != n_samples:
            discarded["invalid_signal"] += 1
            continue
        split = "train" if end < boundaries[0] else ("val" if end < boundaries[1] else "test")
        previous_end = actual_end
        row = {
            "id": f"eeglass-4453396-{event_index:06d}",
            "dataset": "zenodo:4453396", "canonical_dataset_id": "zenodo:4453396",
            "subject": "release-participant-1", "session": "pctasks", "recording_id": source.stem,
            "eeg_path": "eeg_windows.npy", "eeg_index": len(usable),
            "window_start_s": actual_start, "window_end_s": actual_end,
            "event_time_s": t, "decision_time_s": actual_end,
            "minimum_lead_s": args.lead_s, "label": label,
            "label_source": "recorded_event", "target_kind": "observed_action",
            "label_evidence_uri": source_url + "/files/" + source.name,
            "original_event_description": description,
            "split": split, "data_origin": "real_recording", "sample_rate_hz": fs,
            "eeg_source_uri": source_uri, "eeg_source_sha256": source_sha,
            "eeg_set_sha256": set_sha,
            "channels": list(raw.ch_names),
            "environment_status": "not_recorded_in_this_release",
            "filter_note": "No annotated input event in selected window; motion/EMG artifacts and cognitive intent remain unverified",
        }
        usable.append((row, x))
    if not usable:
        raise ValueError("No eligible windows")
    array = np.stack([x for _, x in usable])
    np.save(out / "eeg_windows.npy", array, allow_pickle=False)
    manifest = out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r, _ in usable))
    summary = {
        "source": source_url, "recording_duration_s": duration, "sample_rate_hz": fs,
        "channels_in_file": list(raw.ch_names), "n_samples_per_window": n_samples,
        "eligible_windows": len(usable), "unique_input_seconds": len(usable) * n_samples / fs,
        "split_counts": {s: dict(Counter(r["label"] for r, _ in usable if r["split"] == s)) for s in ("train", "val", "test")},
        "discarded": dict(discarded), "boundary_times_s": boundaries, "boundary_embargo_s": args.embargo_s,
        "eeg_sha256": digest(out / "eeg_windows.npy"), "manifest_sha256": digest(manifest),
        "interpretation": "One-participant chronological observed-action pilot; neither intention labels nor agentic benchmark success",
    }
    (out / "preparation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sample-rate", type=float, default=250)
    p.add_argument("--lookback-s", type=float, default=2)
    p.add_argument("--lead-s", type=float, default=.25)
    p.add_argument("--embargo-s", type=float, default=10)
    a = p.parse_args()
    if not 0 < a.lead_s < a.lookback_s or a.embargo_s < 0:
        p.error("Require 0 < lead_s < lookback_s and nonnegative embargo")
    prepare(a)
