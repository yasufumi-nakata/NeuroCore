#!/usr/bin/env python3
"""Leakage-aware PJ-Xavier evaluation. Python >=3.9; no ML dependency.

The program evaluates supplied predictions. It never fabricates predictions,
EEG, intent labels, screen context, or harness execution outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "xavier-benchmark-v1"
VARIANTS = ("context_only", "eeg_only", "fused", "eeg_shuffled", "eeg_time_shifted")
GOLD_SOURCES = {"source_annotation", "event_derived", "human_verified"}
LABEL_SOURCES = GOLD_SOURCES | {"ai_pseudo"}
SPLITS = ("train", "validation", "test")
SHA = re.compile(r"^[0-9a-f]{64}$")


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: JSON object required")
            rows.append(value)
    return rows


def canonical_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def record_digest(records):
    return canonical_hash(sorted(records, key=lambda row: row["record_id"]))


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def gold_label(record):
    label = record.get("label", {})
    if label.get("provenance") not in GOLD_SOURCES:
        return False
    if not label.get("evidence_uri"):
        return False
    if label.get("provenance") == "human_verified":
        return bool(label.get("independent_review") and label.get("reviewer_id"))
    return True


def group_key(record, unit):
    source = record["source_dataset_id"]
    subject = record.get("subject_group_id")
    participant = ("global", subject) if subject else (source, record.get("canonical_subject_id", record["participant_id"]))
    if unit == "dataset":
        return (source,)
    if unit == "participant":
        return participant
    if unit == "session":
        return (*participant, record["session_id"])
    raise ValueError(f"Unsupported group unit: {unit}")


def validate_records(records):
    errors, warnings, identifiers = [], [], set()
    for index, record in enumerate(records):
        rid = record.get("record_id", f"row:{index}")
        if rid in identifiers:
            errors.append(f"{rid}: duplicate record_id")
        identifiers.add(rid)
        for field in ("record_id", "dataset_id", "source_dataset_id", "participant_id", "session_id", "recording_id", "target_kind"):
            if not isinstance(record.get(field), str) or not record[field]:
                errors.append(f"{rid}: missing nonempty {field}")
        if record.get("data_origin") != "real_recording":
            errors.append(f"{rid}: real_recording origin required for empirical evaluation")
        eeg = record.get("eeg", {})
        if not SHA.fullmatch(str(eeg.get("sha256", ""))):
            errors.append(f"{rid}: original EEG asset sha256 required")
        if not eeg.get("uri"):
            errors.append(f"{rid}: EEG asset URI required")
        start, end = eeg.get("start_s"), eeg.get("end_s")
        if not (finite_number(start) and finite_number(end) and 0 <= start < end):
            errors.append(f"{rid}: invalid EEG interval")
        if not (finite_number(eeg.get("sample_rate_hz")) and eeg["sample_rate_hz"] > 0):
            errors.append(f"{rid}: sample_rate_hz required")
        label = record.get("label", {})
        if label.get("provenance") not in LABEL_SOURCES or not isinstance(label.get("value"), str):
            errors.append(f"{rid}: invalid label/provenance")
        if label.get("provenance") == "ai_pseudo" and not all(label.get(k) for k in ("teacher_id", "teacher_revision", "prompt_sha256", "evidence_uri")):
            errors.append(f"{rid}: AI pseudo label provenance incomplete")
        decision = record.get("decision_time_s")
        if not finite_number(decision) or (finite_number(end) and end > decision):
            errors.append(f"{rid}: EEG extends beyond decision cutoff")
        environment = record.get("environment")
        if environment:
            cutoff = environment.get("available_until_s")
            if not finite_number(cutoff) or (finite_number(decision) and cutoff > decision):
                errors.append(f"{rid}: environment contains unavailable future information")
            if not environment.get("uri") or not SHA.fullmatch(str(environment.get("sha256", ""))):
                errors.append(f"{rid}: environment artifact URI/hash required")
            if environment.get("contains_target_label") is not False:
                warnings.append(f"{rid}: context target-label leakage review not recorded")
            sync = record.get("clock_sync", {})
            if sync.get("verified") is not True or not finite_number(sync.get("max_error_ms")) or sync.get("max_error_ms", -1) < 0 or not sync.get("method"):
                warnings.append(f"{rid}: EEG/environment synchronization is unverified")
        else:
            warnings.append(f"{rid}: environment unavailable")
        if record.get("target_kind") in ("observed_action", "explicit_intent"):
            event_time, lead = label.get("event_time_s"), record.get("minimum_lead_s")
            if not finite_number(event_time) or not finite_number(lead) or lead <= 0:
                errors.append(f"{rid}: positive pre-action lead and source event timestamp required")
            elif finite_number(decision) and decision + lead > event_time + 1e-9:
                errors.append(f"{rid}: post-action/concurrent EEG cannot establish pre-action prediction")
            if label.get("provenance") == "event_derived" and record.get("target_kind") != "observed_action":
                errors.append(f"{rid}: observed action cannot be promoted to unobserved intent")
    return {"errors": errors, "warnings": warnings, "record_count": len(records)}


def create_split(records, unit="participant", seed="pj-xavier-v1", embargo_s=10.0):
    checked = validate_records(records)
    if not finite_number(embargo_s) or embargo_s < 0:
        return {"status": "blocked", "reason": "finite nonnegative embargo required"}
    if checked["errors"]:
        return {"status": "blocked", "validation": checked}
    assignments, excluded = {}, {}
    if unit == "temporal_block":
        by_recording = defaultdict(list)
        for record in records:
            by_recording[record["eeg"]["sha256"]].append(record)
        for rows in by_recording.values():
            lo = min(row["eeg"]["start_s"] for row in rows)
            hi = max(row["eeg"]["end_s"] for row in rows)
            cut1, cut2 = lo + .6 * (hi - lo), lo + .8 * (hi - lo)
            for row in rows:
                start, end = row["eeg"]["start_s"], row["eeg"]["end_s"]
                if end <= cut1 - embargo_s / 2:
                    split = "train"
                elif start >= cut1 + embargo_s / 2 and end <= cut2 - embargo_s / 2:
                    split = "validation"
                elif start >= cut2 + embargo_s / 2:
                    split = "test"
                else:
                    excluded[row["record_id"]] = "temporal boundary/embargo"
                    continue
                assignments[row["record_id"]] = split
        generalization = "within-recording temporal transfer only; no new-person or new-dataset claim"
    else:
        grouped = defaultdict(list)
        for row in records:
            grouped[group_key(row, unit)].append(row)
        groups = sorted(grouped, key=lambda key: canonical_hash([seed, key]))
        if len(groups) < 3:
            return {"status": "blocked", "reason": f"{unit} split requires at least 3 independent groups", "groups": len(groups), "validation": checked}
        ntrain = max(1, min(len(groups) - 2, math.floor(.6 * len(groups))))
        nval = max(1, min(len(groups) - ntrain - 1, math.floor(.2 * len(groups))))
        for index, key in enumerate(groups):
            split = "train" if index < ntrain else "validation" if index < ntrain + nval else "test"
            for row in grouped[key]:
                assignments[row["record_id"]] = split
        generalization = {"participant": "held-out participant", "session": "held-out session; a participant may be seen in training", "dataset": "held-out canonical source dataset"}[unit]
    # Splitting is independent of labels; non-gold evaluation rows are explicitly excluded.
    for row in records:
        rid = row["record_id"]
        if assignments.get(rid) in ("validation", "test") and not gold_label(row):
            del assignments[rid]
            excluded[rid] = "evaluation requires independent source/human gold; pseudo label excluded"
    counts = Counter(assignments.values())
    result = {"schema_version": VERSION, "status": "ready" if all(counts[name] for name in SPLITS) else "blocked", "unit": unit, "seed": seed, "embargo_s": embargo_s, "records_sha256": record_digest(records), "generalization_scope": generalization, "assignments": assignments, "excluded": excluded, "counts": dict(counts), "validation": checked}
    if result["status"] == "blocked":
        result["reason"] = "an evaluation/training partition is empty after eligibility/purge"
    errors = validate_split(records, result)
    if errors:
        result["status"], result["split_errors"] = "blocked", errors
    return result


def validate_split(records, split):
    errors = []
    ids = {row["record_id"] for row in records}
    assignments, excluded = split.get("assignments", {}), split.get("excluded", {})
    if split.get("records_sha256") != record_digest(records):
        errors.append("records differ from frozen split manifest")
    if set(assignments) & set(excluded) or set(assignments) | set(excluded) != ids:
        errors.append("split assignments/exclusions must partition all record ids exactly")
    if any(name not in SPLITS for name in assignments.values()):
        return errors + ["unknown split assignment"]
    unit, group_splits, asset_rows = split.get("unit"), defaultdict(set), defaultdict(list)
    if unit not in ("participant", "session", "dataset", "temporal_block"):
        return errors + ["unsupported split grouping unit"]
    if any(not any(name == required for name in assignments.values()) for required in SPLITS):
        errors.append("train, validation and test partitions must all be nonempty")
    for row in records:
        name = assignments.get(row["record_id"])
        if name is None:
            continue
        if name != "train" and not gold_label(row):
            errors.append(f"{row['record_id']}: non-gold label in evaluation partition")
        if unit != "temporal_block":
            group_splits[group_key(row, unit)].add(name)
        eeg = row["eeg"]
        asset_rows[eeg["sha256"]].append((eeg["start_s"], eeg["end_s"], name, row["record_id"]))
    if any(len(names) > 1 for names in group_splits.values()):
        errors.append(f"{unit} group crosses split boundary")
    embargo = split.get("embargo_s", 0)
    if not finite_number(embargo) or embargo < 0:
        return errors + ["finite nonnegative embargo required"]
    for rows in asset_rows.values():
        ordered = sorted(rows)
        if unit == "temporal_block":
            rank = {"train": 0, "validation": 1, "test": 2}
            if any(rank[a[2]] > rank[b[2]] for a, b in zip(ordered, ordered[1:])):
                errors.append("temporal_block must preserve chronological train -> validation -> test order")
        # Keep active intervals rather than comparing only neighboring starts.
        active = []
        for start, end, name, rid in ordered:
            active = [(old_end, old_name, old_rid) for old_end, old_name, old_rid in active if old_end + embargo > start]
            for old_end, old_name, old_rid in active:
                if old_name != name:
                    errors.append(f"{old_rid}/{rid}: shared EEG interval or embargo crosses split boundary")
            active.append((end, name, rid))
    return errors


def preprocessing_checks(records):
    """A pre-event crop is not causal if prior filtering used future samples."""
    blocked = []
    for row in records:
        reason = None
        prep = row.get("preprocessing") or {}
        if not isinstance(prep, dict):
            prep = {}
        future = prep.get("max_future_support_s")
        if prep.get("online_causality_verified") is not True:
            reason = "online preprocessing causality is absent or explicitly unverified"
        elif not prep.get("evidence_uri") or not finite_number(future) or future < 0:
            reason = "causal availability proof and quantified future sample support required"
        elif row["eeg"]["end_s"] + future > row["decision_time_s"] + 1e-9:
            reason = "preprocessed EEG depends on samples unavailable at decision time"
        if reason:
            blocked.append({"record_id": row["record_id"], "reason": reason})
    return {"status": "passed" if records and not blocked else "blocked", "records_checked": len(records),
            "blocked_record_count": len(blocked), "blocked_records": blocked,
            "interpretation": "Metadata availability contract; successful retrospective classification alone does not establish prospective causal preprocessing."}


def negative_control_checks(records, predictions, artifact_root=None, source_manifest_sha=None):
    """Check logged source assignments; never infer a missing perturbation."""
    lookup = {row["record_id"]: row for row in records}
    expected = set(lookup)
    results, raw_cache = {}, None
    for variant in ("eeg_shuffled", "eeg_time_shifted"):
        rows = {rid: pred for (name, rid), pred in predictions.items() if name == variant}
        issues, source_ids = [], []
        if set(rows) != expected:
            results[variant] = {"status": "unavailable", "reason": "identical complete held-out record set not supplied"}
            continue
        for rid, pred in rows.items():
            detail = pred.get("perturbation")
            if not isinstance(detail, dict):
                issues.append(f"{rid}: perturbation metadata unavailable")
                continue
            if variant == "eeg_time_shifted" and detail.get("method") == "raw_recording_time_shift":
                try:
                    from evaluate_variants import EvidenceCache, verify_raw_shift_certificate
                    if not artifact_root:
                        raise ValueError("artifact root required for independently verified raw extraction")
                    if detail.get("eeg_source_id") in lookup:
                        raise ValueError("newly extracted raw source cannot impersonate an existing held-out record")
                    if raw_cache is None:
                        raw_cache = EvidenceCache(artifact_root)
                    verify_raw_shift_certificate(detail, lookup[rid], artifact_root,
                                                 pred.get("checkpoint_sha256"), source_manifest_sha, raw_cache)
                except (OSError, ValueError, TypeError, KeyError, IndexError, ImportError, RuntimeError) as exc:
                    issues.append(f"{rid}: raw extraction verification failed: {type(exc).__name__}: {exc}")
                continue
            sid = detail.get("eeg_source_id")
            if detail.get("original_eeg_id") != rid or not isinstance(sid, str) or sid not in lookup or sid == rid:
                issues.append(f"{rid}: perturbation must cite a different real held-out EEG source and its original target ID")
                continue
            original, source = lookup[rid], lookup[sid]
            source_ids.append(sid)
            if group_key(original, "session") != group_key(source, "session"):
                issues.append(f"{rid}: EEG control crosses person/session; this does not isolate within-session EEG correspondence")
            if variant == "eeg_shuffled":
                if detail.get("method") not in ("nonzero_cyclic_shift", "seeded_permutation", "within_group_permutation"):
                    issues.append(f"{rid}: unknown shuffle construction")
            else:
                if detail.get("method") != "recording_time_shift":
                    issues.append(f"{rid}: recording_time_shift evidence required")
                if original["eeg"]["sha256"] != source["eeg"]["sha256"]:
                    issues.append(f"{rid}: time shift must stay in the same original recording")
                shift = detail.get("shift_seconds")
                actual = source["eeg"]["start_s"] - original["eeg"]["start_s"]
                if not finite_number(shift) or abs(shift - actual) > 1e-7:
                    issues.append(f"{rid}: shift_seconds differs from the actual source-window time difference")
                if max(original["eeg"]["start_s"], source["eeg"]["start_s"]) < min(original["eeg"]["end_s"], source["eeg"]["end_s"]):
                    issues.append(f"{rid}: time-shift control still overlaps the original EEG window")
        if variant == "eeg_shuffled" and (len(source_ids) != len(expected) or set(source_ids) != expected):
            issues.append("shuffle source mapping is not a complete held-out permutation")
        if variant == "eeg_time_shifted" and raw_cache is not None:
            try:
                raw_cache.unchanged()
            except (OSError, ValueError) as exc:
                issues.append(str(exc))
        results[variant] = {"status": "passed" if not issues else "failed", "records_checked": len(rows), "issues": issues,
                            "evidence_scope": "Actual raw extraction certificates were independently decoded and compared; original inputs were matched to checkpoint-sealed bytes. This is not prospective causal preprocessing." if variant == "eeg_time_shifted" and raw_cache is not None else "Consistency of logged source IDs and original recording times; not an independent rerun of model inputs."}
    return results


def classification_metrics(rows, classes, ece_bins=10):
    counts = Counter()
    confusion = {truth: {pred: 0 for pred in [*classes, "__abstain__"]} for truth in classes}
    bins = [[0, 0.0, 0] for _ in range(ece_bins)]
    nll, brier, accepted, accepted_correct, forced_correct = 0., 0., 0, 0, 0
    for truth, probabilities, abstain in rows:
        pred = max(classes, key=lambda name: probabilities[name])
        confidence = probabilities[pred]
        correct = int(pred == truth)
        forced_correct += correct
        counts[truth] += 1
        confusion[truth]["__abstain__" if abstain else pred] += 1
        if not abstain:
            accepted += 1
            accepted_correct += correct
        nll -= math.log(max(probabilities[truth], 1e-15))
        brier += sum((probabilities[name] - int(name == truth)) ** 2 for name in classes)
        bucket = min(ece_bins - 1, int(confidence * ece_bins))
        bins[bucket][0] += 1
        bins[bucket][1] += confidence
        bins[bucket][2] += correct
    n = len(rows)
    if not n:
        return {"status": "unavailable", "reason": "no labeled predictions"}
    present = [name for name in classes if counts[name]]
    recall, f1 = [], []
    for name in classes:
        tp = confusion[name][name]
        fn = counts[name] - tp
        fp = sum(confusion[other][name] for other in classes if other != name)
        if counts[name]:
            recall.append(tp / counts[name])
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    ece = sum(count / n * abs(conf_sum / count - correct_sum / count) for count, conf_sum, correct_sum in bins if count)
    return {"status": "measured", "n": n, "class_counts": dict(counts), "missing_classes": sorted(set(classes) - set(present)), "forced_accuracy": forced_correct / n, "accepted_correct_over_all": accepted_correct / n, "balanced_accuracy": statistics.mean(recall), "macro_f1": statistics.mean(f1), "coverage": accepted / n, "selective_accuracy": accepted_correct / accepted if accepted else None, "negative_log_likelihood": nll / n, "brier_multiclass_sum": brier / n, "expected_calibration_error": ece, "ece_bins": ece_bins, "confusion": confusion, "class_averaging": "macro_f1 includes all declared classes (zero when unsupported); balanced_accuracy averages recalls of present classes; missing classes block goal validation", "abstention_policy": "Abstentions count as false negatives in balanced_accuracy and macro_f1; forced_accuracy ignores abstention."}


def bootstrap_increment(records, predictions, left, right, classes, seed=1731, repeats=500):
    grouped = defaultdict(list)
    for record in records:
        grouped[group_key(record, "participant")].append(record)
    keys = list(grouped)
    if len(keys) < 5:
        return {"status": "unavailable", "reason": "fewer than 5 independent participant groups; no reliable participant-cluster interval", "groups": len(keys)}
    rng, deltas = random.Random(seed), []
    for _ in range(repeats):
        sample = [row for key in rng.choices(keys, k=len(keys)) for row in grouped[key]]
        score = {}
        for variant in (left, right):
            rows = [(row["label"]["value"], predictions[(variant, row["record_id"])]["probabilities"], predictions[(variant, row["record_id"])].get("abstain", False)) for row in sample]
            score[variant] = classification_metrics(rows, classes)["macro_f1"]
        deltas.append(score[left] - score[right])
    deltas.sort()
    return {"status": "measured", "method": "paired participant-cluster percentile bootstrap", "groups": len(keys), "replicates": repeats, "seed": seed, "metric": "macro_f1", "lower_95": deltas[int(.025 * repeats)], "upper_95": deltas[min(repeats - 1, math.ceil(.975 * repeats) - 1)]}


def training_checks(training, split, checkpoint_sha):
    errors = []
    if not training:
        return ["training provenance manifest unavailable"]
    assignments = split.get("assignments", {})
    for rid in training.get("training_record_ids", []):
        if assignments.get(rid) != "train":
            errors.append(f"{rid}: checkpoint trained on record outside training partition")
    if not training.get("training_record_ids"):
        errors.append("training record provenance empty")
    if training.get("split_manifest_sha256") != canonical_hash(split):
        errors.append("checkpoint split hash differs from frozen manifest")
    if training.get("checkpoint_sha256") != checkpoint_sha or not SHA.fullmatch(str(checkpoint_sha)):
        errors.append("prediction/checkpoint artifact checksum missing or inconsistent")
    if training.get("model_selection_partition") != "validation" or training.get("test_used_for_tuning") is not False:
        errors.append("test-set tuning exclusion not established")
    for rid in training.get("teacher_training_record_ids", []):
        if assignments.get(rid) in ("validation", "test"):
            errors.append(f"{rid}: pseudo-label teacher trained on held-out evaluation data")
    return errors


def resource_checks(resource, checkpoint_sha, max_latency_ms=None):
    reasons = []
    if not resource:
        return {"status": "blocked", "reasons": ["actual resource measurement unavailable"]}
    for field in ("machine_id", "runtime", "accelerator", "measurement_method", "memory_measurement_scope"):
        if not resource.get(field):
            reasons.append(f"resource {field} required")
    if resource.get("measurement_mode") != "actual" or resource.get("checkpoint_sha256") != checkpoint_sha:
        reasons.append("resource measurement must use the evaluated checkpoint")
    if resource.get("hardware_compatible") is not True or resource.get("inference_succeeded") is not True:
        reasons.append("inference success on an available user environment unverified")
    peak, available, p95 = resource.get("peak_memory_bytes"), resource.get("memory_budget_bytes"), resource.get("p95_latency_ms")
    if not all(finite_number(value) and value > 0 for value in (peak, available, p95)):
        reasons.append("measured memory/budget/latency missing")
    elif peak > available:
        reasons.append("measured memory exceeds deployment memory budget")
    capacity = resource.get("memory_capacity_bytes", resource.get("device_physical_memory_bytes"))
    if capacity is not None and (not finite_number(capacity) or capacity <= 0 or (finite_number(available) and available > capacity)):
        reasons.append("configured budget exceeds measured memory capacity or capacity is invalid")
    host_peak, host_capacity = resource.get("host_peak_rss_bytes"), resource.get("host_ram_bytes")
    if host_peak is not None or host_capacity is not None:
        if not all(finite_number(value) and value > 0 for value in (host_peak, host_capacity)) or host_peak > host_capacity:
            reasons.append("separate host RSS exceeds host RAM or host measurement is incomplete")
    if finite_number(p95) and max_latency_ms is not None and p95 > max_latency_ms:
        reasons.append("measured p95 latency exceeds configured acceptance limit")
    evidence_ok = (resource.get("target_environment_verified") is True and bool(resource.get("target_environment_id"))
                   and bool(SHA.fullmatch(str(resource.get("hardware_evidence_sha256", "")))))
    return {"status": "failed" if reasons else "passed", "reasons": reasons, "measurement": resource,
            "target_environment_evidence": "recorded" if evidence_ok else "unverified",
            "limits_are": "project configuration; not research-established universal thresholds",
            "scope": "Reported measured configuration only; accelerator and host memory scopes must not be conflated, and component latency is not end-to-end task latency."}


def acceptance_checks(report, criteria):
    """Project-configured targets, never universal or literature-derived claims."""
    if not criteria:
        return {"status": "not_configured", "reason": "No project acceptance criteria supplied."}
    fused = report.get("metrics", {}).get("fused", {})
    replay = report.get("harness_replay", {})
    resource = report.get("resource_feasibility", {}).get("measurement", {})
    values = {
        "min_macro_f1": fused.get("macro_f1"),
        "min_coverage": fused.get("coverage"),
        "min_selective_accuracy": fused.get("selective_accuracy"),
        "min_task_completion_rate": replay.get("task_completion_rate"),
        "min_replay_episodes": replay.get("valid_episodes"),
        "min_idle_minutes": (replay.get("idle_seconds_observed", 0) / 60),
        "min_test_participants": report.get("test_participants"),
        "max_false_actions_per_minute": replay.get("false_actions_per_minute"),
        "max_p95_latency_ms": resource.get("p95_latency_ms"),
    }
    checks = {}
    for key, limit in criteria.get("limits", {}).items():
        observed = values.get(key)
        if key not in values or not finite_number(limit):
            checks[key] = {"status": "blocked", "reason": "unknown/non-numeric criterion"}
        elif not finite_number(observed):
            checks[key] = {"status": "blocked", "limit": limit, "observed": observed, "reason": "required measurement unavailable"}
        else:
            passed = observed >= limit if key.startswith("min_") else observed <= limit
            checks[key] = {"status": "passed" if passed else "failed", "limit": limit, "observed": observed}
    if not checks:
        return {"status": "not_configured", "reason": "No numeric limits supplied."}
    status = "passed" if all(check["status"] == "passed" for check in checks.values()) else "blocked" if any(check["status"] == "blocked" for check in checks.values()) else "failed"
    return {"status": status, "criteria_id": criteria.get("id"), "criteria_basis": criteria.get("basis"), "checks": checks, "interpretation": "Engineering targets chosen for this project; not established performance estimates."}


def replay_metrics(episodes, artifact_root, test_ids, checkpoint_sha):
    if not episodes:
        return {"status": "unavailable", "reason": "no independent held-out harness replay traces supplied", "task_completion_rate": None, "false_actions_per_minute": None}
    root = Path(artifact_root).resolve() if artifact_root else None
    valid, rejected, idle, scopes = [], [], defaultdict(list), set()
    seen_episode_ids, seen_trace_hashes = set(), set()
    false_events = set()
    for episode in episodes:
        reason = None
        if episode.get("episode_id") in seen_episode_ids or episode.get("trace_sha256") in seen_trace_hashes:
            reason = "duplicate episode or replay trace"
        elif not root:
            reason = "artifact root required to verify trace bytes"
        elif episode.get("checkpoint_sha256") != checkpoint_sha or episode.get("variant") != "fused":
            reason = "checkpoint/variant mismatch"
        elif not episode.get("record_ids") or not set(episode["record_ids"]).issubset(test_ids):
            reason = "episode does not exclusively use held-out EEG records"
        elif episode.get("ground_truth_source") != "independent_harness_assertion" or episode.get("environment_goal_seen_in_training") is not False:
            reason = "independent goal/assertion provenance missing"
        if not reason:
            path = (root / episode.get("trace_path", "")).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                reason = "trace path missing or outside artifact root"
            elif hashlib.sha256(path.read_bytes()).hexdigest() != episode.get("trace_sha256"):
                reason = "trace checksum mismatch"
            else:
                trace = read_jsonl(path)
                terminal = [row for row in trace if row.get("kind") == "terminal_assertion" and row.get("goal_id") == episode.get("goal_id") and isinstance(row.get("passed"), bool)]
                if len(terminal) != 1:
                    reason = "exactly one matching terminal assertion required"
                else:
                    seen_episode_ids.add(episode.get("episode_id"))
                    seen_trace_hashes.add(episode.get("trace_sha256"))
                    scopes.add(episode.get("task_scope", "unspecified"))
                    valid.append(terminal[0]["passed"])
                    # Only non-overlapping, explicitly observed idle time is a denominator.
                    for event in trace:
                        if episode.get("action_logging_complete") is True and event.get("kind") == "observed_idle_interval" and event.get("independent_annotation") is True:
                            start, end = event.get("start_s"), event.get("end_s")
                            if finite_number(start) and finite_number(end) and end > start:
                                idle[event["stream_id"]].append((start, end))
                    for event in trace:
                        if event.get("kind") == "executed_action" and event.get("is_false_activation") is True and event.get("independent_annotation") is True:
                            t = event.get("time_s")
                            if finite_number(t) and any(start <= t < end for start, end in idle.get(event.get("stream_id"), [])):
                                false_events.add((event["stream_id"], event["event_id"]))
        if reason:
            rejected.append({"episode_id": episode.get("episode_id"), "reason": reason})
    idle_seconds = 0.
    for intervals in idle.values():
        union = []
        for start, end in sorted(intervals):
            if union and start <= union[-1][1]:
                union[-1][1] = max(union[-1][1], end)
            else:
                union.append([start, end])
        idle_seconds += sum(end - start for start, end in union)
    return {"status": "measured" if valid else "blocked", "valid_episodes": len(valid), "task_scopes": sorted(scopes), "rejected_episodes": rejected, "task_completion_rate": sum(valid) / len(valid) if valid else None, "idle_seconds_observed": idle_seconds, "false_activation_events": len(false_events) if idle_seconds else None, "false_actions_per_minute": len(false_events) / (idle_seconds / 60) if idle_seconds else None, "idle_metric_status": "measured" if idle_seconds else "unavailable: no independently annotated idle exposure"}


def evaluate(records, split, predictions, training=None, resource=None, episodes=None, artifact_root=None, max_latency_ms=None, acceptance=None):
    validation = validate_records(records)
    if validation["errors"]:
        return {"schema_version": VERSION, "status": "blocked", "data_validation": validation, "autonomous_agent_validated": False}
    errors = validation["errors"] + validate_split(records, split)
    test_records = [row for row in records if split.get("assignments", {}).get(row["record_id"]) == "test"]
    test_ids, classes = {row["record_id"] for row in test_records}, sorted({row["label"]["value"] for row in records if gold_label(row)})
    indexed, shas, checkpoint_ids = {}, set(), set()
    for pred in predictions:
        key = (pred.get("variant"), pred.get("record_id"))
        if key in indexed:
            errors.append(f"duplicate prediction: {key}")
        indexed[key] = pred
        shas.add(pred.get("checkpoint_sha256"))
        checkpoint_ids.add(pred.get("checkpoint_id"))
        probabilities = pred.get("probabilities", {})
        if set(probabilities) != set(classes) or not all(finite_number(p) and 0 <= p <= 1 for p in probabilities.values()) or abs(sum(probabilities.values()) - 1) > 1e-5:
            errors.append(f"invalid class probabilities: {key}")
        if pred.get("record_id") not in test_ids or pred.get("variant") not in VARIANTS:
            errors.append(f"prediction outside declared held-out benchmark: {key}")
        if not isinstance(pred.get("abstain", False), bool):
            errors.append(f"abstain must be boolean: {key}")
        if pred.get("variant") in ("eeg_shuffled", "eeg_time_shifted") and not pred.get("perturbation"):
            errors.append(f"negative control transformation not recorded: {key}")
    if len(shas) != 1 or len(checkpoint_ids) != 1:
        errors.append("one checkpoint per report required")
    checkpoint_sha = next(iter(shas)) if len(shas) == 1 else None
    errors.extend(training_checks(training, split, checkpoint_sha))
    report = {"schema_version": VERSION, "checkpoint_id": next(iter(checkpoint_ids)) if len(checkpoint_ids) == 1 else None, "checkpoint_sha256": checkpoint_sha, "status": "blocked", "data_validation": validation, "integrity_errors": errors, "test_records": len(test_records), "test_participants": len({group_key(row, "participant") for row in test_records}), "classes": classes, "generalization_scope": split.get("generalization_scope"), "metrics": {}, "eeg_increment": {}, "resource_feasibility": resource_checks(resource, checkpoint_sha, max_latency_ms), "harness_replay": replay_metrics(episodes or [], artifact_root, test_ids, checkpoint_sha), "preprocessing_causality": preprocessing_checks(test_records), "negative_controls": negative_control_checks(test_records, indexed, artifact_root, (training or {}).get("source_flat_manifest_sha256")), "autonomous_agent_validated": False, "interpretation": "Observed action prediction is a proxy. EEG-based autonomous behavior requires independently scored harness replay and an EEG contribution beyond context."}
    if errors or not test_records:
        return report
    complete_variants = []
    for variant in VARIANTS:
        available_ids = {rid for name, rid in indexed if name == variant}
        if available_ids != test_ids:
            report["metrics"][variant] = {"status": "blocked", "reason": "all variants must predict the identical frozen test records", "missing": len(test_ids - available_ids)}
            continue
        rows = [(row["label"]["value"], indexed[(variant, row["record_id"])]["probabilities"], indexed[(variant, row["record_id"])].get("abstain", False)) for row in test_records]
        report["metrics"][variant] = classification_metrics(rows, classes)
        complete_variants.append(variant)
    if "fused" in complete_variants:
        for other in ("context_only", "eeg_shuffled", "eeg_time_shifted"):
            if other in complete_variants:
                report["eeg_increment"][other] = {"macro_f1_difference": report["metrics"]["fused"]["macro_f1"] - report["metrics"][other]["macro_f1"], "interval": bootstrap_increment(test_records, indexed, "fused", other, classes)}
    goal_blockers = []
    if len(complete_variants) != len(VARIANTS):
        goal_blockers.append("ablation/negative control predictions incomplete")
    if any(metrics.get("missing_classes") for metrics in report["metrics"].values()):
        goal_blockers.append("held-out test set does not cover all declared classes")
    if any(not row.get("environment") or row.get("clock_sync", {}).get("verified") is not True or row.get("environment", {}).get("contains_target_label") is not False for row in test_records):
        goal_blockers.append("synchronized pre-decision context with leakage review missing")
    if any(row.get("environment") and row["environment"].get("available_until_s", math.inf) > row["eeg"]["start_s"] for row in test_records):
        goal_blockers.append("environment context was not available before the EEG input window")
    if report["preprocessing_causality"]["status"] != "passed":
        goal_blockers.append("online_preprocessing_unverified: future filter support/availability has not passed")
    if any(check.get("status") != "passed" for check in report["negative_controls"].values()):
        goal_blockers.append("shuffle/time-shift source evidence has not passed")
    if any(row.get("target_kind") not in ("observed_action", "explicit_intent") for row in test_records):
        goal_blockers.append("test labels do not measure pre-action intent/action prediction")
    if report["resource_feasibility"]["status"] != "passed":
        goal_blockers.append("deployment resource gate has not passed")
    if report["resource_feasibility"].get("target_environment_evidence") != "recorded":
        goal_blockers.append("measured checkpoint execution is not linked to a verified available user environment")
    if resource and resource.get("latency_scope") != "end_to_end":
        goal_blockers.append("measured latency does not cover end-to-end preprocessing, inference and harness execution")
    if report["harness_replay"]["status"] != "measured":
        goal_blockers.append("held-out harness task completion has not been measured")
    elif report["harness_replay"].get("task_scopes") != ["context_conditioned_agent_operation"]:
        goal_blockers.append("replay tasks do not establish the requested environment-conditioned agent operation")
    for baseline in ("context_only", "eeg_shuffled", "eeg_time_shifted"):
        ci = report["eeg_increment"].get(baseline, {}).get("interval", {})
        if ci.get("status") != "measured" or ci.get("lower_95", -1) <= 0:
            goal_blockers.append(f"positive EEG contribution over {baseline} not established")
    report["acceptance"] = acceptance_checks(report, acceptance)
    if report["acceptance"]["status"] != "passed":
        goal_blockers.append("configured project acceptance criteria have not passed")
    report["goal_validation_blockers"] = goal_blockers
    report["status"] = "evaluated_goal_unproven" if goal_blockers else "held_out_replay_acceptance_passed"
    report["held_out_replay_validated"] = not goal_blockers
    # Retrospective replay alone does not establish prospective operation.
    report["autonomous_agent_validated"] = False
    report["deployment_readiness"] = "requires user-goal acceptance thresholds and prospective evaluation"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    split_parser = sub.add_parser("split")
    split_parser.add_argument("--records", required=True)
    split_parser.add_argument("--unit", choices=("participant", "session", "dataset", "temporal_block"), default="participant")
    split_parser.add_argument("--seed", default="pj-xavier-v1")
    split_parser.add_argument("--embargo-s", type=float, default=10.)
    split_parser.add_argument("--out", required=True)
    evaluator = sub.add_parser("evaluate")
    for name in ("records", "split", "predictions", "out"):
        evaluator.add_argument(f"--{name}", required=True)
    for name in ("training", "resource", "episodes", "artifact-root", "acceptance"):
        evaluator.add_argument(f"--{name}")
    evaluator.add_argument("--max-latency-ms", type=float)
    args = parser.parse_args()
    records = read_jsonl(args.records)
    if args.command == "split":
        if args.embargo_s < 0:
            parser.error("--embargo-s must be nonnegative")
        result = create_split(records, args.unit, args.seed, args.embargo_s)
    else:
        load = lambda path: json.loads(Path(path).read_text()) if path else None
        result = evaluate(records, load(args.split), read_jsonl(args.predictions), load(args.training), load(args.resource), read_jsonl(args.episodes) if args.episodes else None, args.artifact_root, args.max_latency_ms, load(args.acceptance))
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
