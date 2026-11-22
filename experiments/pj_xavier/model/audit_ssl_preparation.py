#!/usr/bin/env python3
"""Read-only accounting of a sealed raw EEG preprocessing result.

This audit checks every original fixed-grid window against the disjoint
accepted/rejected certificate sets. It does not rederive the filter output;
the trainer's independent array/QC preflight has that separate role.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value, why):
    if not value:
        raise ValueError(why)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audit_certificate(row, cert):
    """Independent grid definition and complete, mutually exclusive coverage."""
    digest = row["waveform_sha256"]
    require(cert["status"] == "completed" and cert["source_sha256"] == digest, "source certificate identity mismatch")
    require(cert["source_recording_id"] == row["recording_id"] and cert["split"] == row["ssl_split"] and cert["subject_group"] == row["participant_group"], "source provenance mismatch")
    duration = row["stored_signal_seconds"]
    require(cert["original_signal_seconds"] == duration, "source duration changed")
    expected = set()
    start = 5
    while start + 30 <= duration - 5 + 1e-9:
        expected.add((start, start + 30))
        start += 30
    accepted = [(item["window_start_s"], item["window_end_s"]) for item in cert["records"]]
    rejected = [(item["window_start_s"], item["window_end_s"]) for item in cert["rejected_windows"]]
    require(len(accepted) == len(set(accepted)) and len(rejected) == len(set(rejected)), "a source interval was duplicated")
    require(not set(accepted).intersection(rejected) and set(accepted).union(rejected) == expected, "candidate grid has a gap, overlap or extra interval")
    require(cert["accepted_windows"] == len(accepted) and cert["candidate_windows"] == len(expected), "certificate window totals disagree")
    require(cert["accepted_unique_source_seconds"] == 30 * len(accepted), "accepted time differs from the disjoint interval sum")
    require(cert["unit_check"]["status"] == "passed", "native voltage conversion was not validated")
    reasons = Counter()
    for item in cert["rejected_windows"]:
        require(item["reasons"] and len(item["reasons"]) == len(set(item["reasons"])), "rejection reasons missing or duplicate")
        known = {"nonfinite", "max_abs_exceeds_100uV", "a_channel_std_below_0p01uV"}
        require(set(item["reasons"]).issubset(known), "unregistered QC rejection")
        reasons.update(item["reasons"])
    require(dict(reasons) == cert["rejected_reason_counts"], "rejection reason totals disagree")
    expected_channels = [item["name"] for item in row["eeg_channels"]]
    slots = defaultdict(set)
    for item in cert["records"]:
        require(item["source_sha256"] == digest and item["split"] == row["ssl_split"] and item["subject_group"] == row["participant_group"], "accepted window identity mismatch")
        require(item["channels"] == expected_channels and item["sample_rate_hz"] == 200 and item["unit"] == "V_x_1e4", "accepted channel, rate or unit changed")
        require(not any(key in item for key in ("label", "action", "target", "action_label")), "supervised label appeared in the SSL row")
        index = item["eeg_index"]
        require(isinstance(index, int) and not isinstance(index, bool) and index >= 0 and index not in slots[item["eeg_path"]], "array slot reused or invalid")
        slots[item["eeg_path"]].add(index)
    require(set(slots) == set(cert["artifacts"]), "unreferenced or missing source shard")
    for name, artifact in cert["artifacts"].items():
        shape = artifact["shape"]
        require(len(shape) == 4 and shape[1:] == [len(expected_channels), 30, 200] and 1 <= shape[0] <= 16, "source shard shape invalid")
        require(slots[name] == set(range(shape[0])) and artifact["dtype"] == "float32", "array slots do not completely match their manifest")
    return {"candidate_windows": len(expected), "accepted_windows": len(accepted), "rejected_windows": len(rejected),
            "original_source_seconds": duration, "accepted_unique_source_seconds": len(accepted) * 30,
            "original_waveform_bytes": row["waveform_member_bytes"], "original_eeg_sample_bytes": row["eeg_sample_bytes"],
            "rejected_reason_counts": dict(reasons), "bids_header_unit_disagreement": bool(row.get("bids_header_unit_disagreement", False)),
            "array_bytes": sum(a["bytes"] for a in cert["artifacts"].values()), "array_files": len(cert["artifacts"])}


def audit(args):
    corpus, registration = Path(args.corpus).resolve(), Path(args.registration).resolve()
    completed = read_json(corpus / "completion.json")
    require(completed["status"] == "completed" and completed["exit_code"] == 0, "preparation incomplete")
    summary = read_json(corpus / "corpus_summary.json")
    for name, key in (("manifest.jsonl", "manifest_sha256"), ("artifacts.json", "artifacts_sha256"), ("protocol.json", "protocol_sha256"), ("source_allowlist.json", "source_allowlist_sha256"), ("corpus_summary.json", "summary_sha256")):
        require(sha(corpus / name) == completed[key], "completed corpus seal changed: " + name)
    receipt = read_json(registration / "registration.json")
    require(sha(registration / "source_records.jsonl") == receipt["source_records_sha256"] and receipt["protocol_sha256"] == completed["protocol_sha256"], "source registration differs")
    rows, declared = read_rows(registration / "source_records.jsonl"), read_json(corpus / "source_allowlist.json")
    require(len(rows) == len(declared["sources"]) == 75 and len(set(declared["excluded_source_sha256"])) == 42, "75/42 source boundary changed")
    require({row["waveform_sha256"] for row in rows} == set(declared["sources"]), "source allowlist coverage changed")
    require(not set(declared["sources"]).intersection(declared["excluded_source_sha256"]), "allowed and excluded waveforms overlap")
    records, artifacts, certificates, result_rows = [], {}, [], []
    groups = {}
    for row in rows:
        digest = row["waveform_sha256"]
        cert_path = corpus / "source_certificates" / (digest + ".json")
        cert = read_json(cert_path)
        require(cert["protocol_sha256"] == completed["protocol_sha256"], "source used a different protocol")
        require(not set(artifacts).intersection(cert["artifacts"]), "shard shared between distinct raw sources")
        values = audit_certificate(row, cert)
        declaration = declared["sources"][digest]
        require(all(cert[key] == declaration[key] for key in ("split", "subject_group", "source_recording_id")), "allowlist provenance mismatch")
        group, split = row["participant_group"], row["ssl_split"]
        require(group not in groups or groups[group] == split, "publisher participant crossed SSL folds")
        groups[group] = split
        result_rows.append({"dataset": row["dataset_id"], "device": row.get("device_session", "UniBa"), "split": split, "group": group, **values})
        records.extend(cert["records"])
        artifacts.update(cert["artifacts"])
        certificates.append({"sha256": sha(cert_path), "source_sha256": digest})
    require(records == read_rows(corpus / "manifest.jsonl"), "final manifest differs from source certificates")
    require(artifacts == read_json(corpus / "artifacts.json"), "final artifacts differ from source certificates")
    require(len({row["id"] for row in records}) == len(records), "duplicate final row IDs")
    extra_certificates = {p.stem for p in (corpus / "source_certificates").glob("*.json")} - set(declared["sources"])
    require(not extra_certificates, "extra source certificate outside the allowlist")
    def aggregate(items):
        total = {key: sum(item[key] for item in items) for key in ("candidate_windows", "accepted_windows", "rejected_windows", "original_source_seconds", "accepted_unique_source_seconds", "original_waveform_bytes", "original_eeg_sample_bytes", "array_bytes", "array_files")}
        total.update({"recordings": len(items), "publisher_participant_groups": len({item["group"] for item in items}),
                      "recordings_with_windows": sum(item["accepted_windows"] > 0 for item in items),
                      "recordings_with_zero_windows": sum(item["accepted_windows"] == 0 for item in items),
                      "participant_groups_with_windows": len({item["group"] for item in items if item["accepted_windows"]}),
                      "bids_header_unit_disagreement_recordings": sum(item["bids_header_unit_disagreement"] for item in items),
                      "rejected_reason_counts": dict(sum((Counter(item["rejected_reason_counts"]) for item in items), Counter()))})
        total["accepted_fraction_of_candidate_windows"] = total["accepted_windows"] / total["candidate_windows"] if total["candidate_windows"] else None
        total["original_recording_hours"] = total["original_source_seconds"] / 3600
        total["accepted_recording_hours"] = total["accepted_unique_source_seconds"] / 3600
        total["one_second_temporal_patches_once"] = total["accepted_windows"] * 30
        return total
    total = aggregate(result_rows)
    require(total["accepted_windows"] == summary["accepted_windows"] and total["candidate_windows"] == summary["candidate_windows_before_QC"] and total["rejected_windows"] == summary["rejected_windows"], "public quantities disagree with the source receipts")
    require(total["array_bytes"] == summary["artifact_bytes"] and total["array_files"] == summary["artifact_files"] and abs(total["original_source_seconds"] - summary["total_original_source_seconds"]) < 1e-7, "source bytes/time totals disagree")
    report = {"schema": "xavier-ssl-preparation-accounting-v1", "status": "passed", "source_code_sha256": sha(__file__),
              "preparer_source_sha256": summary["source_code_sha256"], "manifest_sha256": completed["manifest_sha256"], "artifact_manifest_sha256": completed["artifacts_sha256"],
              "preprocessing_protocol_sha256": completed["protocol_sha256"], "source_allowlist_sha256": completed["source_allowlist_sha256"],
              "completion_sha256": sha(corpus / "completion.json"), "corpus_summary_sha256": sha(corpus / "corpus_summary.json"),
              "source_records_sha256": receipt["source_records_sha256"], "registration_sha256": sha(registration / "registration.json"),
              "certificates_ledger_sha256": hashlib.sha256(json.dumps(certificates, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
              "total": total, "folds": {split: aggregate([item for item in result_rows if item["split"] == split]) for split in ("train", "val")},
              "dataset_and_device": {key: aggregate([item for item in result_rows if item["dataset"] + ":" + item["device"] == key]) for key in sorted({item["dataset"] + ":" + item["device"] for item in result_rows})},
              "allowlisted_sources": 75, "excluded_source_hashes": 42,
              "scope": {"all_grid_windows_accounted_for": True, "accepted_rejected_overlap": False, "duplicate_raw_hashes": False,
                        "raw_interval_overlap_within_source": False, "participant_cross_split": False, "supervised_action_labels_in_manifest": False,
                        "original_waveforms_modified": False, "numerical_filter_rederivation_performed_by_this_audit": False,
                        "array_hash_and_QC_independent_trainer_preflight_required": True, "offline_causal_inference_proven": False,
                        "unique_cross_dataset_human_count_verified": False, "unique_cross_device_wall_clock_time_verified": False,
                        "latent_intent_or_autonomous_agent_validated": False}}
    out = Path(args.output)
    require(not out.exists(), "audit output already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix="." + out.name, dir=out.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.link(temporary, out)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(json.dumps({"status": "passed", "audit_sha256": sha(out), "total": total, "folds": report["folds"]}, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--output", required=True)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
