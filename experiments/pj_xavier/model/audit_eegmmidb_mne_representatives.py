#!/usr/bin/env python3
"""Independent, bounded MNE cross-check of a pinned EEGMMIDB native ledger.

This reads nine deterministically selected EDF headers and annotations. It
does not preload waveforms, train models, change labels, repair time intervals,
or replace the exhaustive native quantity audit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
import warnings
import zipfile

PINS = {
    "archive": "0ff5f7f38d1d14b8e485581ad4cf5b327e6494bc2ac7c7d7d4f7e9518a950cbe",
    "recordings": "51b801d07876ae185bd6f6d0bc465769b68d6265cf5ab9356e9ab4ff896ebea2",
    "members": "c291d869f76749f0e8882e940aafaa5caf7b18358497728d4323f60ebba7d76c",
    "summary": "015cc62aaa5f206086cd7c2ba360a18ff5745f6a7b7f4fbaf546bd8214809097",
    "review": "bc6105f2781d1967d9ad0843aea8569c74f4d1ae52c4933cba898da0dafb7d4d",
}
TOLERANCE_SECONDS = 1e-9


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def json_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def eeg_channels(row):
    return [channel for channel in row["channels"] if not channel["is_annotation"]]


def sample_rate(row):
    rates = {float(channel["sample_rate_hz"]) for channel in eeg_channels(row)}
    if len(rates) != 1:
        raise ValueError("Representative selection requires a single EEG rate")
    return next(iter(rates))


def nonrest_baseline(row):
    return row["task"].startswith("baseline_") and any(
        code != "T0" for code in row["annotation_counts"]
    )


def choose_representatives(rows):
    """Choose from prior metadata before any MNE EDF read."""
    selected = defaultdict(list)

    def first(candidates, reason):
        candidates = sorted(candidates, key=lambda row: row["member"])
        if not candidates:
            raise ValueError("Missing prespecified representative category: " + reason)
        selected[candidates[0]["member"]].append(reason)

    clean = [
        row for row in rows
        if sample_rate(row) == 160
        and not row["annotation_errors"]
        and not nonrest_baseline(row)
    ]
    first(
        [row for row in clean if row["task"] == "baseline_eyes_open"],
        "lexicographically first clean 160 Hz eyes-open baseline",
    )
    first(
        [row for row in clean if row["task"].startswith("motor_imagery_")],
        "lexicographically first clean 160 Hz motor-imagery record",
    )
    groups = defaultdict(list)
    for row in rows:
        if sample_rate(row) == 128:
            groups[row["subject_id"]].append(row)
    for subject, group in sorted(groups.items()):
        first(group, "first 128 Hz record in prior affected group " + subject)
    for row in sorted(rows, key=lambda item: item["member"]):
        if nonrest_baseline(row):
            selected[row["member"]].append("all baseline records with non-T0 code")
    overlap_groups = defaultdict(list)
    bounds_groups = defaultdict(list)
    for row in rows:
        errors = row["annotation_errors"]
        if sample_rate(row) == 160 and errors.get("overlapping_annotation_intervals", 0):
            overlap_groups[row["subject_id"]].append(row)
        if errors.get("annotation_outside_signal_duration", 0):
            bounds_groups[row["subject_id"]].append(row)
    for subject, group in sorted(overlap_groups.items()):
        first(group, "first 160 Hz overlapping annotation in group " + subject)
    for subject, group in sorted(bounds_groups.items()):
        first(group, "first out-of-range annotation in group " + subject)
    selected[max(clean, key=lambda row: row["member"])["member"]].append(
        "lexicographically last clean 160 Hz record"
    )
    if not 8 <= len(selected) <= 12:
        raise ValueError("Representative bound of 8 to 12 records violated")
    return dict(sorted(selected.items()))


def annotation_events(annotations):
    return [
        {
            "onset_seconds": float(onset),
            "duration_seconds": float(duration),
            "code": str(description),
        }
        for onset, duration, description in zip(
            annotations.onset, annotations.duration, annotations.description, strict=True
        )
    ]


def compare_events(expected, observed):
    differences = []
    if len(expected) != len(observed):
        differences.append(
            {"kind": "count", "expected": len(expected), "observed": len(observed)}
        )
    max_onset_error = max_duration_error = 0.0
    for index, (left, right) in enumerate(zip(expected, observed)):
        onset_error = abs(left["onset_seconds"] - right["onset_seconds"])
        duration_error = abs(left["duration_seconds"] - right["duration_seconds"])
        max_onset_error = max(max_onset_error, onset_error)
        max_duration_error = max(max_duration_error, duration_error)
        if (
            left["code"] != right["code"]
            or onset_error > TOLERANCE_SECONDS
            or duration_error > TOLERANCE_SECONDS
        ):
            differences.append({"index": index, "expected": left, "observed": right})
    return {
        "agrees_within_tolerance": not differences,
        "expected_count": len(expected),
        "observed_count": len(observed),
        "max_onset_error_seconds": max_onset_error,
        "max_duration_error_seconds": max_duration_error,
        "differences": differences,
    }


def interval_diagnostics(events, duration):
    overlaps, outside = [], []
    previous_end = -math.inf
    for index, event in sorted(
        enumerate(events), key=lambda pair: pair[1]["onset_seconds"]
    ):
        start = event["onset_seconds"]
        end = start + event["duration_seconds"]
        if start < previous_end - TOLERANCE_SECONDS:
            overlaps.append(
                {"index": index, "overlap_seconds": min(end, previous_end) - start}
            )
        if start < -TOLERANCE_SECONDS or end > duration + TOLERANCE_SECONDS:
            outside.append(
                {
                    "index": index,
                    "before_start_seconds": max(0.0, -start),
                    "past_end_seconds": max(0.0, end - duration),
                }
            )
        previous_end = max(previous_end, end)
    return {"overlaps": overlaps, "outside_recording": outside}


def read_with_warnings(function, *args, **kwargs):
    output = io.StringIO()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with redirect_stdout(output), redirect_stderr(output):
            value = function(*args, **kwargs)
    return value, [
        {"category": warning.category.__name__, "message": str(warning.message)}
        for warning in captured
    ], output.getvalue()


def clipping_explanation(expected, observed, recording_seconds):
    """Recognize observed Raw boundary clipping without modifying either list."""
    if len(expected) != len(observed):
        return {
            "all_differences_are_recording_end_duration_clips": False,
            "reason": "Counts differ; no automatic dropped-event explanation",
        }
    changes = []
    for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
        if (
            left["code"] != right["code"]
            or abs(left["onset_seconds"] - right["onset_seconds"]) > TOLERANCE_SECONDS
        ):
            return {
                "all_differences_are_recording_end_duration_clips": False,
                "reason": "Description or onset changed",
            }
        delta = right["duration_seconds"] - left["duration_seconds"]
        if abs(delta) <= TOLERANCE_SECONDS:
            continue
        expected_end = left["onset_seconds"] + left["duration_seconds"]
        observed_end = right["onset_seconds"] + right["duration_seconds"]
        clipped = (
            delta < 0
            and expected_end > recording_seconds + TOLERANCE_SECONDS
            and abs(observed_end - recording_seconds) <= TOLERANCE_SECONDS
        )
        changes.append(
            {
                "index": index,
                "observed_duration_change_seconds": delta,
                "native_end_seconds": expected_end,
                "raw_attached_end_seconds": observed_end,
                "matches_recording_bound": clipped,
            }
        )
    return {
        "all_differences_are_recording_end_duration_clips": bool(changes)
        and all(change["matches_recording_bound"] for change in changes),
        "changes": changes,
        "events_rewritten_by_this_audit": 0,
    }


def audit_one(row, member_seal, archive, temporary_parent, mne):
    binary = archive.read("files/" + row["member"])
    member_hash = hashlib.sha256(binary).hexdigest()
    if (
        member_hash != row["member_sha256"]
        or member_hash != member_seal["sha256"]
        or len(binary) != row["member_bytes"]
        or len(binary) != member_seal["bytes"]
        or not member_seal["official_checksum_verified"]
    ):
        raise ValueError("Selected EDF bytes differ from the verified source ledger")
    temporary_path = None
    with tempfile.TemporaryDirectory(prefix="owned-edf-", dir=temporary_parent) as folder:
        temporary_path = Path(folder)
        path = temporary_path / Path(row["member"]).name
        path.write_bytes(binary)
        standalone, standalone_warnings, standalone_log = read_with_warnings(
            mne.read_annotations, path, encoding="utf8"
        )
        raw, raw_warnings, raw_log = read_with_warnings(
            mne.io.read_raw_edf, path, preload=False, encoding="utf8", verbose="WARNING"
        )
        try:
            if raw.preload:
                raise ValueError("Waveform preloading was not authorized for this audit")
            observed_samples = int(raw.n_times)
            observed_rate = float(raw.info["sfreq"])
            observed_channels = list(raw.ch_names)
            attached = annotation_events(raw.annotations)
        finally:
            raw.close()
        standalone_events = annotation_events(standalone)
    channels = eeg_channels(row)
    native_samples = {int(channel["total_samples"]) for channel in channels}
    if len(native_samples) != 1:
        raise ValueError("Mixed native channel sample counts")
    expected_samples = next(iter(native_samples))
    expected_rate = sample_rate(row)
    native_events = [
        {key: event[key] for key in ("onset_seconds", "duration_seconds", "code")}
        for event in row["events"]
    ]
    duration = float(row["signal_duration_seconds_decimal"])
    standalone_comparison = compare_events(native_events, standalone_events)
    raw_comparison = compare_events(native_events, attached)
    clip = clipping_explanation(native_events, attached, duration)
    result = {
        "member": row["member"],
        "member_sha256": member_hash,
        "member_bytes": len(binary),
        "native": {
            "n_times": expected_samples,
            "sample_rate_hz": expected_rate,
            "duration_seconds": duration,
            "channel_count": len(channels),
            "events": native_events,
            "prior_annotation_errors": row["annotation_errors"],
            "baseline_with_nonrest_code": nonrest_baseline(row),
        },
        "MNE_raw_header": {
            "n_times": observed_samples,
            "sample_rate_hz": observed_rate,
            "duration_seconds": observed_samples / observed_rate,
            "channel_count": len(observed_channels),
            "preload": False,
        },
        "header_agreement": (
            observed_samples == expected_samples
            and observed_rate == expected_rate
            and observed_channels == [channel["label"] for channel in channels]
            and abs(observed_samples / observed_rate - duration) <= TOLERANCE_SECONDS
        ),
        "standalone_annotations": standalone_events,
        "raw_attached_annotations": attached,
        "native_vs_standalone": standalone_comparison,
        "native_vs_raw_attached": raw_comparison,
        "raw_attachment_boundary_explanation": clip,
        "interval_checks": {
            "native": interval_diagnostics(native_events, duration),
            "standalone_MNE": interval_diagnostics(standalone_events, duration),
            "raw_attached_MNE": interval_diagnostics(attached, duration),
        },
        "warnings": {
            "read_annotations": standalone_warnings,
            "read_raw_edf": raw_warnings,
        },
        "reader_logs": {"read_annotations": standalone_log, "read_raw_edf": raw_log},
        "temporary_directory_removed": not temporary_path.exists(),
    }
    result["unexplained_reader_difference"] = not (
        result["header_agreement"]
        and standalone_comparison["agrees_within_tolerance"]
        and (
            raw_comparison["agrees_within_tolerance"]
            or clip["all_differences_are_recording_end_duration_clips"]
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "recordings", "members", "summary", "review"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--public-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "2"
    import mne

    output = args.output_dir.resolve()
    public = args.public_dir.resolve()
    if (output.exists() and any(output.iterdir())) or (
        public.exists() and any(public.iterdir())
    ):
        raise ValueError("Preserve prior audit: output directory is nonempty")
    output.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    actual_hashes = {}
    for name, expected in PINS.items():
        actual_hashes[name] = sha256(getattr(args, name))
        if actual_hashes[name] != expected:
            raise ValueError("Source bytes differ from fixed " + name + " identity")
    rows = json_rows(args.recordings)
    member_seals = {row["member"]: row for row in json_rows(args.members)}
    selected = choose_representatives(rows)
    by_name = {row["member"]: row for row in rows}
    edf_source = Path(importlib.import_module("mne.io.edf.edf").__file__)
    annotation_source = Path(importlib.import_module("mne.annotations").__file__)
    protocol = {
        "schema": "xavier-eegmmidb-MNE-representative-protocol-v1",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_sha256": sha256(__file__),
        "source_sha256": actual_hashes,
        "MNE_version": mne.__version__,
        "python_version": platform.python_version(),
        "MNE_reader_source_sha256": sha256(edf_source),
        "MNE_annotations_source_sha256": sha256(annotation_source),
        "selected_records_and_reasons": selected,
        "selection_input": "Prior pinned native header/TAL ledger only; no MNE EDF read or waveform inspection before registration",
        "maximum_records": 12,
        "seconds_absolute_tolerance": TOLERANCE_SECONDS,
        "methods": [
            "mne.read_annotations with UTF-8, retaining standalone TAL durations",
            "mne.io.read_raw_edf with preload=False and UTF-8; compare n_times, sfreq, channel order and attached annotations",
            "Capture Python warnings and distinguish observed recording-end duration clipping from standalone parsing",
            "Extract only each selected SHA/CRC-verified member into an owned temporary directory, then close and remove it",
        ],
        "scope": "Independent representative parser check, not exhaustive waveform quality, annotation semantics or training eligibility",
        "label_corrections": 0,
        "model_forward_calls": 0,
    }
    write_json(output / "protocol.json", protocol)
    write_json(public / "protocol.json", protocol)
    results = []
    with zipfile.ZipFile(args.archive) as archive:
        for name in selected:
            result = audit_one(
                by_name[name], member_seals[name], archive, output, mne
            )
            result["selection_reasons"] = selected[name]
            results.append(result)
            with (output / "records.jsonl").open("a") as handle:
                handle.write(json.dumps(result, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        "member": name,
                        "header_agreement": result["header_agreement"],
                        "standalone_agreement": result["native_vs_standalone"]["agrees_within_tolerance"],
                        "raw_attachment_exact": result["native_vs_raw_attached"]["agrees_within_tolerance"],
                        "raw_boundary_clip": result["raw_attachment_boundary_explanation"]["all_differences_are_recording_end_duration_clips"],
                    }
                ),
                flush=True,
            )
    for name, expected in actual_hashes.items():
        if sha256(getattr(args, name)) != expected:
            raise ValueError("Original source mutated during representative audit")
    summaries = []
    for result in results:
        summary = {
            key: result[key]
            for key in (
                "member", "member_sha256", "member_bytes", "selection_reasons",
                "MNE_raw_header", "header_agreement", "native_vs_standalone",
                "native_vs_raw_attached", "raw_attachment_boundary_explanation",
                "warnings", "temporary_directory_removed", "unexplained_reader_difference",
            )
        }
        summary["native"] = {
            key: value for key, value in result["native"].items() if key != "events"
        }
        summary["interval_counts"] = {
            name: {key: len(value) for key, value in diagnosis.items()}
            for name, diagnosis in result["interval_checks"].items()
        }
        summaries.append(summary)
    unexplained = sum(result["unexplained_reader_difference"] for result in results)
    clips = sum(
        result["raw_attachment_boundary_explanation"]["all_differences_are_recording_end_duration_clips"]
        for result in results
    )
    summary = {
        "schema": "xavier-eegmmidb-MNE-representative-result-v1",
        "status": "differences_require_review" if unexplained else "representative_parsers_agree_with_documented_raw_attachment_behavior",
        "protocol_sha256": sha256(output / "protocol.json"),
        "private_detail_sha256": sha256(output / "records.jsonl"),
        "source_sha256": actual_hashes,
        "source_hashes_rechecked_after_audit": True,
        "MNE_version": mne.__version__,
        "records_checked": len(results),
        "full_source_recordings": len(rows),
        "header_agreements": sum(result["header_agreement"] for result in results),
        "standalone_annotation_agreements": sum(result["native_vs_standalone"]["agrees_within_tolerance"] for result in results),
        "raw_attachment_exact_agreements": sum(result["native_vs_raw_attached"]["agrees_within_tolerance"] for result in results),
        "raw_attachment_duration_clip_records": clips,
        "unexplained_reader_difference_records": unexplained,
        "waveform_preloads": 0,
        "model_forward_calls": 0,
        "label_corrections": 0,
        "new_training_eligibility_granted": False,
        "existing_quantity_or_model_reports_modified": False,
        "all_owned_temporary_directories_removed": all(result["temporary_directory_removed"] for result in results),
        "elapsed_seconds": time.monotonic() - started,
        "records": summaries,
        "limitations": [
            "Only deterministically selected representatives were checked by MNE; the native audit remains the separate whole-dataset quantity record.",
            "Reader agreement does not establish biological signal quality, label correctness, channel wiring or online control capability.",
            "128 Hz is an observed header value. Every selected 128 Hz source belongs to a prior overlap-flagged group; it is not certified as normal.",
            "MNE Raw attachment can shorten a boundary-crossing annotation. The original standalone TAL interval and bytes are retained rather than silently corrected.",
            "Baseline T1 remains an observed non-rest code under an unmapped semantic case; a parser match does not resolve its task meaning.",
        ],
    }
    write_json(output / "summary.json", summary)
    write_json(public / "summary.json", summary)
    text = "# EEGMMIDB: independent representative MNE parser check\n\n"
    text += (
        f"MNE {mne.__version__} independently checked {len(results)} deterministically selected "
        f"EDF records from the pinned 1,526-record native audit. Headers agreed in "
        f"{summary['header_agreements']}/{len(results)}, and standalone annotation parsing "
        f"agreed in {summary['standalone_annotation_agreements']}/{len(results)}. "
        f"Raw attachment was exact in {summary['raw_attachment_exact_agreements']}/{len(results)}; "
        f"{clips} record(s) showed a separately documented recording-end duration clip. "
        f"Unexplained reader differences: {unexplained}.\n\n"
    )
    text += "| EDF record | Hz | Samples | Header | Standalone TAL | Raw attachment |\n| --- | ---: | ---: | --- | --- | --- |\n"
    for result in results:
        raw_match = result["native_vs_raw_attached"]["agrees_within_tolerance"]
        clipped = result["raw_attachment_boundary_explanation"]["all_differences_are_recording_end_duration_clips"]
        raw_status = "exact" if raw_match else ("end-duration clip" if clipped else "review")
        text += (
            f"| {Path(result['member']).stem} | {result['MNE_raw_header']['sample_rate_hz']:g} | "
            f"{result['MNE_raw_header']['n_times']} | {result['header_agreement']} | "
            f"{result['native_vs_standalone']['agrees_within_tolerance']} | {raw_status} |\n"
        )
    text += (
        "\nSelection was registered from the prior native ledger before any MNE EDF read. "
        "It includes clean 160 Hz baseline and motor-imagery controls, the first 128 Hz record "
        "in each affected group, baseline non-T0 cases, the first 160 Hz overlap case, "
        "out-of-range representatives, and the final clean record. Overlapping categories "
        "are deduplicated. All original ZIP and ledger hashes were verified before and after.\n\n"
        "Each selected member was verified against both the recording and officially checked "
        "member ledgers, temporarily extracted, read with preload=False, closed and removed "
        "from its owned temporary directory. No original was changed. Standalone read_annotations "
        "and Raw.annotations are recorded separately, including every warning and interval "
        "difference. The absolute float-comparison tolerance is 1 nanosecond.\n\n"
        "The actual observed sample rates and annotation durations remain unchanged. "
        "Small overlaps and baseline T1 labels are confirmed as source annotations when both "
        "parsers agree; their cause and semantic correctness remain unresolved. Any Raw "
        "boundary clipping is reader attachment behavior, not a correction performed by this audit.\n\n"
        "See [protocol.json](protocol.json) for the deterministic selection and pinned reader "
        "source identities, and [summary.json](summary.json) for per-record differences and "
        "warnings. Full annotation lists remain in the private detailed ledger. This is "
        "a representative parser check, not whole-dataset waveform validation or authorization "
        "to include flagged labels in training. Existing quantity, SSL, and model evaluations "
        "were not changed; model forward calls and label corrections were zero.\n"
    )
    (public / "README.md").write_text(text)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "records": len(results),
                "unexplained": unexplained,
                "raw_duration_clip_records": clips,
                "summary_sha256": sha256(public / "summary.json"),
                "protocol_sha256": sha256(public / "protocol.json"),
                "README_sha256": sha256(public / "README.md"),
            }
        ),
        flush=True,
    )
    return 2 if unexplained else 0


if __name__ == "__main__":
    raise SystemExit(main())
