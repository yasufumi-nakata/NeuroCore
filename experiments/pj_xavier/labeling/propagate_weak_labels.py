#!/usr/bin/env python3
"""Reuse a verified train-only event dictionary without creating new gold.

No model or network calls. Exact event descriptions, original asset identity,
ontology, membership and teacher-result provenance must agree before reuse.
"""
import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import ollama_label_worker as worker


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def unique_index(items, key):
    result = {}
    for item in items:
        ident = item.get(key)
        if not isinstance(ident, str) or not ident or ident in result:
            raise ValueError(f"Unique nonempty {key} required")
        result[ident] = item
    return result


def event_key(job):
    evidence = job.get("evidence", [])
    if len(evidence) != 1 or evidence[0].get("kind") != "recorded_event":
        raise ValueError("Exact dictionary propagation only supports one recorded_event per job")
    return worker.digest({"dataset_id": job["dataset_id"], "kind": "recorded_event", "text": evidence[0]["text"],
                          "source_assets": sorted(job["source_assets"], key=lambda asset: (asset["uri"], asset["sha256"])),
                          "allowed_values": job["allowed_values"]})


def check_event_job(job, source_hash, source_index):
    errors = worker.validate_job(job) + worker.check_job_membership(job, source_hash, source_index)
    if errors:
        raise ValueError(f"{job.get('job_id')}: " + "; ".join(errors))
    source = source_index[job["record_id"]]
    if source.get("label_source") != "recorded_event" or source.get("target_kind") != "observed_action":
        raise ValueError("Dictionary propagation requires source observed-action event labels")
    event_key(job)
    evidence = job["evidence"][0]
    if evidence["text"] != source.get("original_event_description"):
        raise ValueError("Event text differs from frozen source row")
    if not worker.finite_number(evidence.get("timestamp_s")) or not worker.finite_number(source.get("event_time_s")) or not math.isclose(evidence["timestamp_s"], source["event_time_s"], rel_tol=0, abs_tol=1e-9):
        raise ValueError("Event timestamp differs from frozen source row")
    expected_source_hash = source.get("label_evidence_sha256", source.get("eeg_set_sha256"))
    if not expected_source_hash or not any(asset["sha256"] == expected_source_hash and asset["uri"] == source.get("label_evidence_uri") for asset in job["source_assets"]):
        raise ValueError("Event source asset does not match the original frozen annotation artifact")
    original = job.get("original_gold", [])
    if len(original) != 1 or original[0].get("value") != source.get("label") or original[0].get("origin") != "recorded_event":
        raise ValueError("Original gold snapshot differs from frozen event label")


def validate_teacher_result(result, job):
    if result.get("status") not in ("labeled", "unknown", "abstain") or result.get("partition") != "train" or result.get("ground_truth") is not False or result.get("label_origin") != "ai_pseudo":
        raise ValueError("Teacher result is not a completed train-only weak label")
    if result.get("record_id") != job["record_id"] or result.get("source_manifest_sha256") != job["source_manifest_sha256"] or result.get("source_sha256") != worker.digest(worker.source_payload(job)) or result.get("source_assets") != job["source_assets"]:
        raise ValueError("Teacher result source identity/hash mismatch")
    if result.get("prompt_sha256") != worker.digest(worker.messages_for(job)) or result.get("output_schema_sha256") != worker.digest(worker.OUTPUT_SCHEMA) or result.get("system_prompt_sha256") != worker.digest(worker.SYSTEM_PROMPT):
        raise ValueError("Teacher prompt/schema provenance differs from the reviewed dictionary contract")
    if result.get("teacher_called") is not True or not isinstance(result.get("created_at"), str) or not result["created_at"]:
        raise ValueError("Original teacher call and timestamp provenance required")
    if result.get("evidence_sha256") != {item["id"]: worker.digest(item) for item in job["evidence"]}:
        raise ValueError("Teacher evidence hashes differ from original supplied evidence")
    if result.get("original_gold_snapshot") != job.get("original_gold", []) or result.get("original_gold_sha256") != worker.digest(job.get("original_gold", [])) or result.get("gold_labels_modified") is not False:
        raise ValueError("Teacher output changed or misidentified original gold")
    if not worker.SHA_RE.fullmatch(str(result.get("teacher", {}).get("model_digest_sha256", ""))):
        raise ValueError("Teacher model digest required")
    labels = result.get("pseudo_labels", [])
    if any(label.get("label_origin") != "ai_pseudo" or label.get("independent_review") is not False for label in labels):
        raise ValueError("Teacher labels must remain unverified pseudo labels")
    parsed = {"status": result["status"], "abstain_reason": result.get("abstain_reason"),
              "labels": [{key: label.get(key) for key in ("namespace", "value", "evidence")} for label in labels]}
    errors = worker.validate_output(parsed, job)
    if errors:
        raise ValueError("Teacher output no longer passes evidence validation: " + "; ".join(errors))
    return parsed


def build_propagation(source_records, membership_path, all_jobs_path, dictionary_jobs_path, teacher_results_path, results_manifest_path):
    paths = {"source_records": source_records, "training_membership": membership_path, "all_train_jobs": all_jobs_path,
             "dictionary_jobs": dictionary_jobs_path, "teacher_results": teacher_results_path, "teacher_results_manifest": results_manifest_path}
    input_hashes = {key: worker.file_digest(path) for key, path in paths.items()}
    source_hash, source_index = worker.load_source_records(source_records)
    membership = json.loads(Path(membership_path).read_text())
    if membership.get("manifest_sha256") != source_hash:
        raise ValueError("Training membership does not refer to the original frozen manifest bytes")
    for key, partitions in (("training_ids", {"train"}), ("validation_ids", {"val", "validation"}), ("test_ids", {"test", "holdout"})):
        declared = membership.get(key, [])
        actual = {rid for rid, row in source_index.items() if row["split"] in partitions}
        if len(declared) != len(set(declared)) or set(declared) != actual:
            raise ValueError(f"{key} changed or does not match frozen source split membership")
    training_hash = hashlib.sha256("\n".join(sorted(membership["training_ids"])).encode()).hexdigest()
    if membership.get("training_ids_sha256") != training_hash:
        raise ValueError("Training ID checksum mismatch")
    all_jobs = unique_index(rows(all_jobs_path), "job_id")
    dictionary = unique_index(rows(dictionary_jobs_path), "job_id")
    results = unique_index(rows(teacher_results_path), "job_id")
    results_manifest = json.loads(Path(results_manifest_path).read_text())
    if results_manifest.get("results_file_sha256") != input_hashes["teacher_results"] or results_manifest.get("jobs_file_sha256") != input_hashes["dictionary_jobs"]:
        raise ValueError("Teacher result shard manifest hash mismatch")
    if set(results) != set(dictionary) or not set(dictionary).issubset(all_jobs):
        raise ValueError("Exactly one completed teacher result per dictionary representative is required")
    if {job["record_id"] for job in all_jobs.values()} != set(membership["training_ids"]) or len(all_jobs) != len(membership["training_ids"]):
        raise ValueError("The propagation target set must exactly equal the frozen training records")
    for job in all_jobs.values():
        check_event_job(job, source_hash, source_index)
    by_key, inverse = {}, defaultdict(list)
    for ident, representative in dictionary.items():
        if worker.digest(representative) != worker.digest(all_jobs[ident]):
            raise ValueError("Dictionary representative differs from the original full train job")
        validate_teacher_result(results[ident], representative)
        key = event_key(representative)
        if key in by_key:
            raise ValueError("Ambiguous duplicate dictionary entry for an exact event pattern")
        by_key[key] = ident
    propagated, agreements, unknown = [], 0, 0
    for ident, job in sorted(all_jobs.items()):
        key = event_key(job)
        if key not in by_key:
            raise ValueError("No exact verified representative for a target event; inference is not invented")
        representative_id = by_key[key]
        teacher = results[representative_id]
        inverse[representative_id].append(job["record_id"])
        original = source_index[job["record_id"]]
        labels = copy.deepcopy(teacher["pseudo_labels"])
        predicted = [label["value"] for label in labels if label["namespace"] == "observed_action"]
        if len(predicted) == 1:
            agreements += int(predicted[0] == original["label"])
        else:
            unknown += 1
        propagated.append({"record_id": job["record_id"], "dataset_id": job["dataset_id"], "split": "train",
                           "source_manifest_sha256": source_hash, "source_job_sha256": worker.digest(job),
                           "label_origin": "ai_pseudo", "ground_truth": False, "independent_review": False,
                           "status": teacher["status"], "pseudo_labels": labels,
                           "original_gold_snapshot": copy.deepcopy(job["original_gold"]), "gold_labels_modified": False,
                           "target_event_evidence": copy.deepcopy(job["evidence"]),
                           "propagation": {"rule": "exact_recorded_event_text_same_source_assets_and_same_ontology",
                                           "event_pattern_sha256": key, "representative_job_id": representative_id,
                                           "representative_result_sha256": worker.digest(teacher),
                                           "teacher_inference_performed_for_this_record": ident == representative_id,
                                           "teacher_evidence_ids_refer_to": "representative job; target evidence is separately retained, never substituted silently"},
                           "teacher": copy.deepcopy(teacher["teacher"]), "teacher_prompt_sha256": teacher["prompt_sha256"],
                           "teacher_created_at": teacher.get("created_at"),
                           "quality_note": "Dictionary normalization of an existing recorded event. No new waveform, latent-intent gold, or independent gold label was created."})
    mapping = [{"representative_job_id": ident, "event_pattern_sha256": event_key(dictionary[ident]),
                "source_record_ids": sorted(ids), "record_count": len(ids),
                "source_manifest_sha256": source_hash, "teacher_result_sha256": worker.digest(results[ident])}
               for ident, ids in sorted(inverse.items())]
    for key, path in paths.items():
        if worker.file_digest(path) != input_hashes[key]:
            raise ValueError("An input artifact changed during propagation; no output may be committed")
    summary = {"schema_version": "xavier-exact-event-weak-label-propagation-v1", "status": "verified_train_only_dictionary_reuse",
               "input_sha256": input_hashes, "source_manifest_sha256": source_hash,
               "independent_teacher_jobs": len(dictionary), "training_record_references": len(propagated),
               "held_out_records_touched": 0, "original_gold_labels_modified": 0,
               "new_waveform_records_created": 0, "new_latent_intent_gold_labels_created": 0,
               "teacher_calls_performed_by_propagation": 0,
               "dictionary_results_with_usable_observed_action": sum(sum(label["namespace"] == "observed_action" for label in result["pseudo_labels"]) == 1 for result in results.values()),
               "references_without_single_action_label": unknown,
               "agreement_with_original_event_labels": {"count": agreements, "denominator": len(propagated),
                                                        "fraction": agreements / len(propagated) if propagated else None,
                                                        "interpretation": "Event-name normalization consistency, not a new intent-decoding accuracy estimate."},
               "legacy_result_membership_verified_posthoc": True,
               "note": "The original source files, original train/validation/test assignments, teacher outputs, and gold labels were preserved."}
    return summary, propagated, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-records", "training-membership", "all-jobs", "dictionary-jobs", "teacher-results", "results-manifest", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    summary, propagated, mapping = build_propagation(args.source_records, args.training_membership, args.all_jobs, args.dictionary_jobs, args.teacher_results, args.results_manifest)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is nonempty; preserve existing evidence and choose a new one")
    out.mkdir(parents=True, exist_ok=True)
    for name, values in (("propagated_weak_labels.jsonl", propagated), ("dictionary_membership.jsonl", mapping)):
        path = out / name
        with path.open("w") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        summary[name + "_sha256"] = worker.file_digest(path)
    worker.atomic_json(out / "propagation_audit.json", summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
