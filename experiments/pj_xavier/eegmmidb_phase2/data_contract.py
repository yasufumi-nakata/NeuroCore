"""Strict phase2 data identity and training-only array access."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

TASKS = {
    "left_right": ("left_fist", "right_fist"),
    "fists_feet": ("both_fists", "both_feet"),
}
PRIMARY_TASK = "left_right"
SHAPE = (64, 600)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def object_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def is_sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def contained(root, name):
    path = Path(name)
    require(
        isinstance(name, str) and not path.is_absolute()
        and ".." not in path.parts and "\\" not in name,
        "Array paths must be relative and contained",
    )
    target = Path(root) / path
    require(
        target.resolve().is_relative_to(Path(root).resolve()) and not target.is_symlink(),
        "Array path escapes the prepared corpus",
    )
    return target


def load_corpus(manifest, artifacts, *, verify_bytes=True):
    """Read metadata; checksums/array headers may include held-out files, no fit."""
    manifest, artifacts = Path(manifest).resolve(), Path(artifacts).resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    seals = json.loads(artifacts.read_text())
    require(bool(rows) and isinstance(seals, dict) and bool(seals), "Empty corpus")
    ids, slots = set(), defaultdict(set)
    group_splits = defaultdict(set)
    source_splits, payload_splits, file_splits = (
        defaultdict(set), defaultdict(set), defaultdict(set)
    )
    source_intervals = defaultdict(list)
    channel_order = rows[0].get("channels")
    require(
        isinstance(channel_order, list) and len(channel_order) == 64
        and len(set(channel_order)) == 64,
        "Exactly 64 uniquely named EEG channels are required",
    )
    for row in rows:
        identity = row.get("id")
        require(isinstance(identity, str) and identity and identity not in ids, "Repeated or missing ID")
        ids.add(identity)
        split = row.get("split")
        require(split in ("train", "val", "test"), "Unsupported partition")
        task = row.get("task_family")
        require(task in TASKS and row.get("label") in TASKS[task], "Invalid task-specific label")
        require(row.get("label_source") == "recorded_event", "Only recorded cue-condition labels admitted")
        require(row.get("unit") == "V" and row.get("sample_rate_hz") == 200, "Expected volts at 200 Hz")
        require(row.get("channels") == channel_order, "Channel order changed within corpus")
        require(row.get("annotation_review_required") is False, "Unresolved annotation metadata")
        require(
            row.get("preprocessing", {}).get("online_causality_verified") is True,
            "Causal preprocessing verification is required",
        )
        group = row.get("subject_group")
        require(isinstance(group, str) and group.startswith("physionet:eegmmidb:1.0.0:"), "Canonical subject group missing")
        source = row.get("source_eeg_sha256", row.get("source_sha256"))
        payload = row.get("eeg_payload_sha256")
        require(is_sha(source) and is_sha(payload), "Original EDF and ordinary-EEG hashes required")
        for alias in ("source_sha256", "source_eeg_sha256", "eeg_source_sha256"):
            if alias in row:
                require(row[alias] == source, "Original source hash aliases disagree")
        group_splits[group].add(split)
        source_splits[source].add(split)
        payload_splits[payload].add(split)
        path, index = row.get("eeg_path"), row.get("eeg_index")
        require(isinstance(index, int) and not isinstance(index, bool) and index >= 0, "EEG index must be a nonnegative integer")
        require(path in seals and index not in slots[path], "Missing or duplicated array slot")
        seal = seals[path]
        require(
            seal.get("dtype") == "float32" and len(seal.get("shape", [])) == 3
            and tuple(seal["shape"][1:]) == SHAPE and seal["shape"][0] > 0,
            "Array seal must describe float32 N x 64 x 600",
        )
        require(index < seal["shape"][0], "Array index exceeds sealed shape")
        require(is_sha(seal.get("sha256")) and isinstance(seal.get("bytes"), int), "Array identity missing")
        slots[path].add(index)
        file_splits[path].add(split)
        start, end = float(row["window_start_s"]), float(row["window_end_s"])
        decision, available = float(row["decision_time_s"]), float(row["feature_available_time_s"])
        require(all(math.isfinite(t) for t in (start, end, decision, available)), "Nonfinite time")
        require(
            start >= 0 and abs(end - start - 3) < 1e-7
            and end <= decision + 1e-8 and available <= decision + 1e-8,
            "Window is not available at the recorded decision",
        )
        source_intervals[source].append((start, end, identity))
        row["_path"] = str(contained(manifest.parent, path).resolve())
    for mapping in (group_splits, source_splits, payload_splits, file_splits):
        require(all(len(splits) == 1 for splits in mapping.values()), "Source, payload, participant or array crosses partitions")
    for spans in source_intervals.values():
        spans.sort()
        require(
            all(left[1] <= right[0] + 1e-8 for left, right in zip(spans, spans[1:])),
            "Overlapping original intervals in the main corpus",
        )
    require(set(slots) == set(seals), "Extra or missing main array artifacts")
    for path, seal in seals.items():
        require(slots[path] == set(range(seal["shape"][0])), "Each array slot needs exactly one provenance row")
        target = contained(manifest.parent, path)
        if verify_bytes:
            require(target.stat().st_size == seal["bytes"] and sha256(target) == seal["sha256"], "Prepared waveform bytes changed")
        array = np.load(target, mmap_mode="r", allow_pickle=False)
        require(array.dtype == np.float32 and list(array.shape) == seal["shape"], "Prepared array header differs")
    parts = {part: [row for row in rows if row["split"] == part] for part in ("train", "val", "test")}
    for part in parts:
        require(bool(parts[part]), "Empty partition: " + part)
    for task, labels in TASKS.items():
        for part in ("train", "val"):
            require(
                {row["label"] for row in parts[part] if row["task_family"] == task} == set(labels),
                "Both classes of each task are required in train and validation",
            )
    audit = {
        "schema": "xavier-phase2-main-corpus-identity-v1",
        "manifest_sha256": sha256(manifest),
        "artifacts_sha256": sha256(artifacts),
        "array_count": len(seals),
        "array_file_bytes": sum(value["bytes"] for value in seals.values()),
        "row_count": len(rows),
        "partition_counts": {part: len(value) for part, value in parts.items()},
        "participant_groups": {part: len({row["subject_group"] for row in value}) for part, value in parts.items()},
        "all_array_slots_unique_and_covered": True,
        "raw_and_payload_partition_disjoint": True,
        "channel_order": channel_order,
        "test_forward_calls_in_validation": 0,
    }
    return rows, seals, audit


class EEGStore:
    def __init__(self):
        self.arrays = {}

    def get(self, row):
        name = row["_path"]
        if name not in self.arrays:
            self.arrays[name] = np.load(name, mmap_mode="r", allow_pickle=False)
        values = np.array(self.arrays[name][row["eeg_index"]], dtype=np.float32, copy=True)
        require(values.shape == SHAPE and np.isfinite(values).all(), "Invalid stored EEG")
        return values


def source_interval_union(rows):
    groups = defaultdict(list)
    for row in rows:
        source = row.get("source_eeg_sha256", row.get("source_sha256"))
        groups[source].append((float(row["window_start_s"]), float(row["window_end_s"])))
    total = 0.0
    for intervals in groups.values():
        left = right = None
        for start, end in sorted(intervals):
            if left is None:
                left, right = start, end
            elif start > right:
                total += right - left
                left, right = start, end
            else:
                right = max(right, end)
        if left is not None:
            total += right - left
    return total


def binary_metrics(targets, probabilities):
    y = np.asarray(targets, dtype=int)
    p = np.asarray(probabilities, dtype=np.float64)
    require(y.size > 0 and p.shape == (len(y), 2), "Wrong probability shape")
    require(np.isfinite(p).all() and np.all(p >= 0) and np.allclose(p.sum(1), 1, atol=1e-5), "Invalid probabilities")
    prediction = p.argmax(1)
    confusion = np.bincount(2 * y + prediction, minlength=4).reshape(2, 2)
    support = confusion.sum(1)
    recalls = np.divide(confusion.diagonal(), support, out=np.zeros(2), where=support > 0)
    return {
        "n": len(y), "support": support.tolist(), "confusion": confusion.tolist(),
        "accuracy": float((prediction == y).mean()),
        "balanced_accuracy": float(recalls.mean()) if np.all(support > 0) else None,
        "negative_log_likelihood": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
    }


def partition_metrics(rows, probabilities):
    probabilities = np.asarray(probabilities)
    tasks = {}
    for task, labels in TASKS.items():
        indices = [i for i, row in enumerate(rows) if row["task_family"] == task]
        require(bool(indices), "Missing task during validation")
        chosen = [rows[i] for i in indices]
        y = [labels.index(row["label"]) for row in chosen]
        p = probabilities[indices]
        metric = binary_metrics(y, p)
        subjects = defaultdict(list)
        for i, row in enumerate(chosen):
            subjects[row["subject_group"]].append(i)
        per_subject = {
            subject: binary_metrics(np.asarray(y)[idx], p[idx])
            for subject, idx in sorted(subjects.items())
        }
        valid = [value["balanced_accuracy"] for value in per_subject.values() if value["balanced_accuracy"] is not None]
        metric.update(
            mean_subject_balanced_accuracy=float(np.mean(valid)) if valid else None,
            subject_groups=len(subjects),
            subjects_missing_either_class=len(subjects) - len(valid),
            per_subject=per_subject,
        )
        tasks[task] = metric
    return {
        "tasks": tasks,
        "selection_metric": "left_right mean_subject_balanced_accuracy",
        "selection_score": tasks[PRIMARY_TASK]["mean_subject_balanced_accuracy"],
        "selection_partition": "val",
        "secondary_task_used_for_selection": False,
    }

