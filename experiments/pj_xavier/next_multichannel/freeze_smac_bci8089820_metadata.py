#!/usr/bin/env python3
"""Freeze source-group splits for Zenodo8089820 using previously audited metadata only.

No signals, ZIP members, event times, labels, or subject-specific model inputs
are read. Metadata is private. Freeze is a prerequisite, not fit approval.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

ARCHIVE_SHA = "7c611b08995f10c2d313d194e3e75c88f7792403455994d77ca37e16104fcacf"
RECORDINGS_SHA = "ba13f499b92d498842b41dff906423afc77d5fdfde720c6da0da173c86413912"
CONSTRAINTS_SHA = "3adca0c48cc582846eb3f0a0dfa7b52a6f3723f840932acaa430576a12f2ee65"
SALT = "pj-xavier-z8089820-source-group-folds-v1-20261123"
EXPECTED_GROUPS = {"A": 60, "B": 21, "C": 6}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")


def make_plan(records: list[dict], must_links: list[dict], flagged_groups: dict) -> dict:
    require(len(records) == 694, "Expected exactly 694 previously audited GDF records")
    names = [r["source_member"] for r in records]
    require(len(set(names)) == 694 and all(re.fullmatch(
        r"BCI Database/Signals/DATA [ABC]/[ABC][0-9]+/[ABC][0-9]+_[A-Za-z0-9_]+\.gdf", n
    ) for n in names), "Invalid or duplicate original GDF member names")
    groups = sorted({r["subject_group"] for r in records})
    require(len(groups) == 87 and len(flagged_groups) == 7, "Participant roster or flag count changed")
    counts = Counter(g.split(":")[2] for g in groups)
    require(dict(counts) == EXPECTED_GROUPS, "Unexpected A/B/C source-group roster")
    require(set(flagged_groups).issubset(groups), "Flag unknown group")
    parent = {g: g for g in groups}

    def root(g: str) -> str:
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    def union(a: str, b: str) -> None:
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    require(len(must_links) == 1, "Cross-group duplicate constraints changed")
    for link in must_links:
        members = link["subject_groups"]
        require(len(members) == 2 and len(set(members)) == 2 and all(g in parent for g in members),
                "Unknown/malformed must-link groups")
        require(link["cross_person_identity_claim"] is False, "Do not infer participant identity")
        union(members[0], members[1])

    components: dict[str, list[str]] = defaultdict(list)
    for g in groups:
        components[root(g)].append(g)
    comps_by_subdataset = defaultdict(list)
    for members in components.values():
        sub = {m.split(":")[2] for m in members}
        require(len(sub) == 1, "Cross-dataset must-link needs independent review")
        comps_by_subdataset[next(iter(sub))].append(sorted(members))
    assignments: dict[str, str] = {}
    fold_component_counts = Counter()
    for ds in sorted(comps_by_subdataset):
        comps = comps_by_subdataset[ds]
        review = [c for c in comps if set(c) & set(flagged_groups)]
        eligible = [c for c in comps if c not in review]
        eligible.sort(key=lambda c: (sha((SALT + "|" + "|".join(c)).encode()), c))
        ntrain = round(len(eligible) * 0.70)
        nval = round((len(eligible) - ntrain) / 2)
        for idx, component in enumerate(eligible):
            fold = "train" if idx < ntrain else ("val" if idx < ntrain + nval else "test")
            for g in component:
                assignments[g] = fold
            fold_component_counts[ds + ":" + fold] += 1
        for component in review:
            for g in component:
                assignments[g] = "review"
            fold_component_counts[ds + ":review"] += 1

    require(set(assignments) == set(groups), "Not all groups assigned")
    for link in must_links:
        require(len({assignments[g] for g in link["subject_groups"]}) == 1, "Must-link split leakage")
    require(all(assignments[g] == "review" for g in flagged_groups), "Author flagged group admitted")
    safe_rows = []
    for record in sorted(records, key=lambda r: r["source_member"]):
        require(record["original_event_sample_rate_hz"] == 0 and
                record["annotation_explicit_event_timebase_verified"] is False,
                "Incorrectly certified cue event timebase")
        require(re.fullmatch(r"[a-f0-9]{64}", record["source_file_sha256"]) is not None,
                "Missing source-file SHA")
        group = record["subject_group"]
        require(group in assignments, "Unknown record group")
        safe_rows.append({
            "source_member": record["source_member"],
            "source_file_sha256": record["source_file_sha256"],
            "eeg_payload_sha256": record["eeg_payload_sha256"],
            "calibration_sha256": record["EEG_calibration_sha256"],
            "subject_group": group,
            "split": assignments[group],
            "source_event_timebase_verified": False,
            "offline_ssl_possible_after_physical_QC": assignments[group] == "train",
            "supervised_cue_training_allowed": False,
            "online_causal_claim_allowed": False,
            "individual_record_QC_passed": False
        })
    summary = {
        "schema": "xavier-z8089820-frozen-metadata-only-v1",
        "source_archive_sha256": ARCHIVE_SHA,
        "prior_record_ledger_sha256": RECORDINGS_SHA,
        "prior_constraints_sha256": CONSTRAINTS_SHA,
        "seed": SALT,
        "groups_by_split": dict(sorted(Counter(assignments.values()).items())),
        "records_by_split": dict(sorted(Counter(r["split"] for r in safe_rows).items())),
        "components_by_dataset_and_split": dict(sorted(fold_component_counts.items())),
        "must_link_components": len(must_links),
        "review_flagged_groups": len(flagged_groups),
        "source_files_opened": 0,
        "new_gradient_steps": 0,
        "individual_record_QC_passed": False,
        "event_timebase_verified": False,
        "offline_ssl_training_allowed_now": False,
        "supervised_training_allowed_now": False,
        "cross_release_duplicate_check": "required_before_fit",
        "require_pre_fit": [
            "S Mac zip complete and matches pinned whole SHA256",
            "zip member source SHA and original EEG payload/calibration match audited metadata",
            "train-only physical voltage calibration and fixed QC",
            "review/val/test/previous heldout and cross-release deny",
            "immutable protocol and genuine MPS backward/capacity probe",
        ],
    }
    return {"summary": summary, "members": safe_rows, "group_assignments": assignments}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--constraints", type=Path, required=True)
    p.add_argument("--recordings", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    raw = a.constraints.read_bytes()
    require(sha(raw) == CONSTRAINTS_SHA, "Source constraints file hash differs")
    constraint = json.loads(raw)
    require(constraint["source_archive_sha256"] == ARCHIVE_SHA and
            constraint["source_record_ledger_sha256"] == RECORDINGS_SHA and
            constraint["training_authorized_by_this_audit"] is False,
            "Frozen source provenance or training permission differs")
    require(sha(a.recordings.read_bytes()) == RECORDINGS_SHA, "Recorded audit ledger differs")
    plan = make_plan(constraint["record_metadata"],
                     constraint["must_link_groups"], constraint["author_reported_QC_flags"])
    blob = canonical_json(plan)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        require(a.out.read_bytes() == blob, "Refuse to overwrite a different frozen split")
    else:
        a.out.write_bytes(blob)
    print(json.dumps({"manifest_sha256": sha(blob), **plan["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
