#!/usr/bin/env python3
"""Adapt the flat training schema to the strict benchmark schema.

Missing provenance/context is left missing, never inferred as validated.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path

from xavier_benchmark import canonical_hash, read_jsonl, record_digest, validate_records, validate_split


def normalize_legacy_decisions(rows, source_manifest_sha256):
    """Explicit compatibility for the old EEGlass decision-at-event metadata bug.

    The original JSONL is never written. Waveform references, window boundaries,
    labels, memberships, and all other input columns must remain identical.
    """
    if not rows or not re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256):
        raise ValueError("Legacy migration requires records and the original manifest bytes SHA256")
    normalized, changes = [], []
    original_rows_hash = canonical_hash(rows)
    for row in rows:
        if "record_id" in row or not isinstance(row.get("id"), str):
            raise ValueError("Legacy decision migration accepts the original flat training schema only")
        decision, event, end, lead = (row.get(key) for key in ("decision_time_s", "event_time_s", "window_end_s", "minimum_lead_s"))
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in (decision, event, end, lead)):
            raise ValueError(f"{row['id']}: finite decision/event/end/lead timestamps required for migration")
        if decision != event or lead <= 0 or end > event - lead + 1e-9:
            raise ValueError(f"{row['id']}: legacy migration requires decision_time_s == event_time_s and window_end_s <= event_time_s - minimum_lead_s")
        if row.get("label_source") != "recorded_event" or row.get("target_kind") != "observed_action":
            raise ValueError(f"{row['id']}: migration is restricted to observed-action recorded-event rows")
        new = copy.deepcopy(row)
        new["decision_time_s"] = end
        immutable_before = {key: value for key, value in row.items() if key != "decision_time_s"}
        immutable_after = {key: value for key, value in new.items() if key != "decision_time_s"}
        if canonical_hash(immutable_before) != canonical_hash(immutable_after):
            raise AssertionError("Migration changed fields other than decision_time_s")
        change = {"record_id": row["id"], "old_decision_time_s": decision, "new_decision_time_s": end,
                  "event_time_s": event, "minimum_lead_s": lead, "window_start_s": row.get("window_start_s"),
                  "window_end_s": end, "all_other_original_fields_sha256": canonical_hash(immutable_before),
                  "waveform_references_and_window_boundaries_unchanged": True, "labels_and_split_unchanged": True}
        changes.append(change)
        normalized.append(new)
    if canonical_hash(rows) != original_rows_hash:
        raise AssertionError("Original rows mutated during compatibility conversion")
    migration = {"schema_version": "xavier-explicit-time-metadata-migration-v1",
                 "requested_by_flag": "--legacy-decision-at-event", "silent_conversion": False,
                 "reason": "Legacy preparation incorrectly named the action-event time as decision_time_s; waveform already stopped at the earlier EEG cutoff.",
                 "source_manifest_sha256": source_manifest_sha256, "source_manifest_bytes_modified": False,
                 "waveform_bytes_modified_by_adapter": False, "labels_and_split_unchanged": True,
                 "record_count": len(changes), "record_changes": changes}
    migration["migration_id"] = canonical_hash(migration)
    for row, change in zip(normalized, changes):
        row["_timestamp_metadata_migration"] = {**change, "migration_id": migration["migration_id"], "source_manifest_sha256": source_manifest_sha256}
    return normalized, migration


def migration_summary(migration):
    return {key: migration[key] for key in ("migration_id", "schema_version", "requested_by_flag", "reason", "source_manifest_sha256", "source_manifest_bytes_modified", "waveform_bytes_modified_by_adapter", "labels_and_split_unchanged", "record_count")}


def adapt_records(rows, base_dir=None):
    result = []
    for row in rows:
        if "record_id" in row and isinstance(row.get("eeg"), dict):
            result.append(row)
            continue
        label_source = {"recorded_event": "event_derived", "human_verified": "human_verified", "ai_pseudo": "ai_pseudo", "source_annotation": "source_annotation", "dataset_annotation": "source_annotation"}.get(row.get("label_source"), row.get("label_source"))
        label = {"value": row.get("label"), "provenance": label_source, "evidence_uri": row.get("label_evidence_uri"), "event_time_s": row.get("event_time_s")}
        for key in ("teacher_id", "teacher_revision", "prompt_sha256", "reviewer_id", "independent_review"):
            if key in row:
                label[key] = row[key]
        record = {
            "record_id": row.get("id"),
            "dataset_id": row.get("dataset"),
            "dataset_version": row.get("dataset_version"),
            "source_dataset_id": row.get("source_dataset_id", row.get("canonical_dataset_id", row.get("dataset"))),
            "participant_id": str(row.get("subject", "")),
            "session_id": str(row.get("session", "")),
            "recording_id": row.get("recording_id"),
            "data_origin": row.get("data_origin", "synthetic" if row.get("label_source") == "synthetic" else None),
            "target_kind": row.get("target_kind", "observed_action" if label_source == "event_derived" else "unspecified"),
            "decision_time_s": row.get("decision_time_s"),
            "minimum_lead_s": row.get("minimum_lead_s"),
            "eeg": {"uri": row.get("eeg_source_uri"), "sha256": row.get("eeg_source_sha256"), "start_s": row.get("window_start_s"), "end_s": row.get("window_end_s"), "sample_rate_hz": row.get("sample_rate_hz")},
            "label": label,
            "environment": None,
            "adapter_note": "No unrecorded provenance, label, timestamp, or environment is inferred by this adapter.",
        }
        if row.get("subject_group_id"):
            record["subject_group_id"] = row["subject_group_id"]
        for key in ("preprocessing", "movement_artifact_control", "clock_uncertainty_guard_s", "canonical_subject_id", "feature_available_time_s", "assumed_filter_lookahead_s", "eeg_window_minimum_lead_s"):
            if key in row:
                record[key] = copy.deepcopy(row[key])
        if row.get("_timestamp_metadata_migration"):
            record["timestamp_metadata_migration"] = copy.deepcopy(row["_timestamp_metadata_migration"])
            record["adapter_note"] = "Explicit --legacy-decision-at-event compatibility: old event-time decision metadata corrected to existing EEG cutoff; waveform, label and split fields unchanged. See timestamp_metadata_migration."
        if row.get("environment_text") is not None or row.get("environment_path"):
            record["environment"] = {"uri": row.get("environment_evidence_uri", row.get("environment_path")), "sha256": row.get("environment_sha256"), "available_until_s": row.get("environment_time_s"), "contains_target_label": row.get("environment_contains_target_label")}
            if row.get("environment_scope"):
                record["environment"]["scope"] = row["environment_scope"]
            record["clock_sync"] = row.get("clock_sync", {})
        result.append(record)
    return result


def adapt_split(rows, records, unit, embargo_s, migration=None):
    mapping = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
    assignments, excluded = {}, {}
    for row in rows:
        rid = row.get("id", row.get("record_id"))
        if row.get("split") in mapping:
            assignments[rid] = mapping[row["split"]]
        else:
            excluded[rid] = row.get("excluded_reason", "not assigned to a training/evaluation partition")
    manifest = {"schema_version": "xavier-benchmark-v1", "status": "ready", "unit": unit, "seed": "imported-training-split", "embargo_s": embargo_s, "records_sha256": record_digest(records), "assignments": assignments, "excluded": excluded, "generalization_scope": "within-recording temporal pilot only" if unit == "temporal_block" else unit, "note": "Imported assignments are checked, not silently repaired."}
    if migration:
        manifest["data_migration"] = migration_summary(migration)
        manifest["generalization_scope"] += "; explicit legacy timestamp metadata correction: decision at event normalized to existing EEG cutoff; migration_id=" + migration["migration_id"] + "; original_manifest_sha256=" + migration["source_manifest_sha256"]
    errors = validate_records(records)["errors"]
    if not errors:
        errors += validate_split(records, manifest)
    if errors:
        manifest["status"], manifest["errors"] = "blocked", errors
    return manifest


def adapt_predictions(rows, checkpoint_sha=None, class_names=None):
    condition_map = {"eeg_only": "eeg_only", "context_only": "context_only", "fused": "fused", "eeg_shuffled": "eeg_shuffled", "shuffled_eeg": "eeg_shuffled", "eeg_time_shifted": "eeg_time_shifted", "time_shifted_eeg": "eeg_time_shifted"}
    result = []
    for row in rows:
        condition = row.get("variant", row.get("condition"))
        if condition == "normal":
            condition = row.get("mode")
        probabilities = row.get("probabilities")
        if isinstance(probabilities, list):
            if not class_names or len(class_names) != len(probabilities):
                raise ValueError("class_names matching vector length required for vector probabilities")
            probabilities = dict(zip(class_names, probabilities))
        result.append({"record_id": row.get("record_id", row.get("id")), "checkpoint_id": row.get("checkpoint_id", row.get("checkpoint")), "checkpoint_sha256": row.get("checkpoint_sha256", checkpoint_sha), "variant": condition_map.get(condition, condition), "probabilities": probabilities, "abstain": row.get("abstain", False), "latency_ms": row.get("latency_ms"), "perturbation": row.get("perturbation")})
    return result


def adapt_training(membership, source_manifest_path, split, checkpoint_metadata):
    source_hash = hashlib.sha256(Path(source_manifest_path).read_bytes()).hexdigest()
    if membership.get("manifest_sha256") != source_hash or checkpoint_metadata.get("manifest_sha256") != source_hash:
        raise ValueError("Training/checkpoint manifest hash differs from input flat JSONL bytes")
    training_ids = membership.get("training_ids", [])
    training_hash = hashlib.sha256("\n".join(sorted(training_ids)).encode()).hexdigest()
    if membership.get("training_ids_sha256") != training_hash:
        raise ValueError("Training membership hash mismatch")
    for field, expected in (("training_ids", "train"), ("validation_ids", "validation"), ("test_ids", "test")):
        actual = {rid for rid, partition in split["assignments"].items() if partition == expected}
        if set(membership.get(field, [])) != actual:
            raise ValueError(f"{field} differs from imported split assignments")
    return {
        "training_record_ids": training_ids,
        "teacher_training_record_ids": membership.get("teacher_training_record_ids", []),
        "split_manifest_sha256": canonical_hash(split),
        "source_flat_manifest_sha256": source_hash,
        "checkpoint_sha256": checkpoint_metadata.get("checkpoint_sha256"),
        "model_selection_partition": membership.get("model_selection_partition", checkpoint_metadata.get("model_selection_partition")),
        "test_used_for_tuning": membership.get("test_used_for_tuning", checkpoint_metadata.get("test_used_for_tuning")),
        "foundation_pretraining_overlap": membership.get("foundation_pretraining_overlap", "unknown"),
        "adapter_note": "All memberships checked against the flat manifest bytes and the imported split; selection statements are inherited, not inferred.",
    }


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def annotate_evaluation_report(report_path, migration_path):
    """Add explicit migration provenance to a report without changing any metric."""
    report_path, migration_path = Path(report_path), Path(migration_path)
    report = json.loads(report_path.read_text())
    migration = json.loads(migration_path.read_text())
    ident = migration.get("migration_id")
    identity_payload = {key: value for key, value in migration.items() if key not in ("migration_id", "normalized_records_sha256", "normalized_split_manifest_sha256")}
    if canonical_hash(identity_payload) != ident:
        raise ValueError("Timestamp migration provenance hash mismatch")
    if not ident or ident not in str(report.get("generalization_scope", "")):
        raise ValueError("Evaluation report does not identify this explicit timestamp migration")
    existing = report.get("data_migrations", [])
    if any(item.get("migration_id") == ident for item in existing):
        return report
    report["evaluation_sha256_before_migration_annotation"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    report["data_migrations"] = [*existing, migration]
    report["legacy_timestamp_metadata_error_corrected"] = True
    report.setdefault("data_quality_notes", []).append("The source manifest contained a decision_time_s naming error. Compatibility conversion explicitly moved only decision metadata from event time to the already-recorded EEG cutoff. Original source bytes, waveform/window references, labels and split memberships were preserved; this does not establish online causal preprocessing or autonomous-agent success.")
    temp = report_path.with_suffix(report_path.suffix + ".migration.tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, report_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records")
    parser.add_argument("--out-dir")
    parser.add_argument("--unit", choices=("participant", "session", "dataset", "temporal_block"))
    parser.add_argument("--embargo-s", type=float, default=10.)
    parser.add_argument("--predictions", nargs="+")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--training-membership")
    parser.add_argument("--checkpoint-metadata")
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--legacy-decision-at-event", action="store_true", help="Explicitly correct legacy observed-action decision timestamps while preserving original input bytes and memberships")
    parser.add_argument("--annotate-evaluation-report", help="After evaluation, attach the explicit migration's full provenance without changing metrics")
    parser.add_argument("--migration", help="migration.json for --annotate-evaluation-report")
    args = parser.parse_args()
    if args.annotate_evaluation_report:
        if not args.migration:
            parser.error("--annotate-evaluation-report requires --migration")
        report = annotate_evaluation_report(args.annotate_evaluation_report, args.migration)
        print(json.dumps({"status": report["status"], "migration_annotation": "preserved", "output": args.annotate_evaluation_report}))
        return
    if not (args.records and args.out_dir and args.unit):
        parser.error("conversion requires --records, --out-dir and --unit")
    source = read_jsonl(args.records)
    migration = None
    if args.legacy_decision_at_event:
        normalized, migration = normalize_legacy_decisions(source, hashlib.sha256(Path(args.records).read_bytes()).hexdigest())
    else:
        normalized = source
    records = adapt_records(normalized)
    split = adapt_split(source, records, args.unit, args.embargo_s, migration)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "records.jsonl", records)
    (out / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n")
    if migration:
        migration["normalized_records_sha256"] = record_digest(records)
        migration["normalized_split_manifest_sha256"] = canonical_hash(split)
        (out / "migration.json").write_text(json.dumps(migration, ensure_ascii=False, indent=2) + "\n")
    if args.predictions:
        prediction_rows = [row for path in args.predictions for row in read_jsonl(path)]
        write_jsonl(out / "predictions.jsonl", adapt_predictions(prediction_rows, args.checkpoint_sha256, args.classes))
    if args.training_membership or args.checkpoint_metadata:
        if not (args.training_membership and args.checkpoint_metadata):
            parser.error("--training-membership and --checkpoint-metadata must be supplied together")
        membership = json.loads(Path(args.training_membership).read_text())
        checkpoint_metadata = json.loads(Path(args.checkpoint_metadata).read_text())
        training = adapt_training(membership, args.records, split, checkpoint_metadata)
        (out / "training.json").write_text(json.dumps(training, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": split["status"], "records": len(records), "split_manifest_sha256": canonical_hash(split), "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
