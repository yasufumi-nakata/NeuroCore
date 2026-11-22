"""Metadata-only admission for new PJ-Xavier EEG sources.

No signal is read or trained here. All row-level output is private. A proposed
split is *not* authorization to use waveforms; QC, units and provenance gates
are separately required. Previously sealed phase2/3 splits are immutable.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).absolute().parent
REPO = HERE.parents[2]
LEDGERS = {
    "compact14_16": ("private/xavier/expansion_candidates_v1/header_audit_v1/records.jsonl", None),
    "bci27": ("private/xavier/expansion_candidates_v1/bci_header_audit_v2/recordings.jsonl", 27),
    "inmid14": ("private/xavier/expansion_candidates_v1/InMID_header_audit_v1/records.jsonl", 14),
    "p30032": ("private/xavier/expansion_candidates_v1/P300_header_audit_v1/records.jsonl", 32),
    "cnt14": ("private/xavier/expansion_candidates_v1/MI_threeclass_header_audit_v1/records.jsonl", 14),
    "ssvep64_processed": ("private/xavier/expansion_candidates_v1/SSVEP_header_audit_v2/records.jsonl", 64),
    "brainstorm": ("private/xavier/expansion_candidates_v1/Brainstorm_header_audit_v1/records.jsonl", 62),
    "small_bci_mixed": ("private/xavier/expansion_candidates_v1/small_BCI_batch_header_audit_v1/records.jsonl", None),
}
EXISTING_SPLIT = "private/xavier/eegmmidb_phase2_v1/metadata/subject_split.json"
BCI_FLAGS = "private/xavier/expansion_candidates_v1/bci_header_audit_v2/future_training_constraints_v1.json"
FIF_MUST_LINK = "private/xavier/expansion_candidates_v1/SSVEP2392979_duplicate_constraints_v1/future_split_constraints.private.json"
WHITELIST_CHANNELS = frozenset((14, 16, 27, 32, 64))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf8")).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def nonempty_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, k: str) -> str:
        self.parent.setdefault(k, k)
        if self.parent[k] != k:
            self.parent[k] = self.find(self.parent[k])
        return self.parent[k]

    def union(self, a: str, b: str) -> None:
        x, y = self.find(a), self.find(b)
        if x != y:
            self.parent[max(x, y)] = min(x, y)


def source_group(source: str, row: dict) -> str | None:
    g = row.get("subject_group", row.get("source_group"))
    return source + "|" + g if nonempty_str(g) else None


def channel_count(source: str, row: dict, declared: int | None) -> int | None:
    if source == "compact14_16":
        return row.get("eeg_channel_count")
    if source == "small_bci_mixed":
        ch = row.get("channels")
        return len(ch) if isinstance(ch, list) else ch if isinstance(ch, int) else None
    if source == "brainstorm":
        ch = row.get("EEG_candidate_channels")
        return int(ch) if isinstance(ch, int) else declared
    if source == "inmid14":
        ch = row.get("EEG_channels")
        return int(ch) if isinstance(ch, int) else len(ch) if isinstance(ch, list) else None
    if source == "cnt14":
        return len(row.get("EEG_channels") or [])
    if source == "ssvep64_processed":
        ch = row.get("EEG_channels")
        return int(ch) if isinstance(ch, int) else len(ch) if isinstance(ch, list) else None
    return declared


def describe_label(source: str, row: dict) -> dict:
    # These keys are separate from prediction context: never auto-promote.
    if source == "compact14_16":
        return {"original": "public stimulus/task fields, not inferred intent", "observed_event": "unit/event checks differ by catalog 000413/001035",
                "original_supervised_gold": False, "oracle": None, "decoder_output": None}
    if source == "bci27":
        return {"original": "GDF event codes retained", "observed_event": "event-rate field 0; native fallback not time gold",
                "original_supervised_gold": False, "oracle": None, "decoder_output": None}
    if source == "p30032":
        return {"original": "BCI2000 raw state codes", "observed_event": "plain-state block-lag unresolved",
                "original_supervised_gold": False, "oracle": ["TextToSpell", "StimulusType"],
                "decoder_output": ["SelectedTarget", "SelectedRow", "SelectedColumn", "TextResult (end-of-run)"]}
    if source == "cnt14":
        return {"original": "source numeric event codes", "observed_event": "MNE-fixed reader mapping for codes 1/2/3 only",
                "original_supervised_gold": False, "oracle": None, "decoder_output": None}
    if source == "ssvep64_processed":
        return {"original": "SET event array empty", "observed_event": "fractional Status not rounded",
                "original_supervised_gold": False, "oracle": None, "decoder_output": None}
    return {"original": "source code/marker retained, ontology not promoted",
            "observed_event": "independent onset gold not confirmed",
            "original_supervised_gold": False, "oracle": None, "decoder_output": None}


def readiness(source: str, count: int | None) -> tuple[str, list[str]]:
    reasons = ["train-only QC not sealed", "metadata split only; no training authorization",
               "cross-source payload/heldout deny confirmation pending"]
    if count not in WHITELIST_CHANNELS:
        reasons.append("channel count outside 14/16/27/32/64 or unclear")
    if source == "compact14_16":
        reasons.append("14ch TXT unit unresolved; 16ch FIF requires fixed group QC and verified stim timing")
    elif source == "bci27":
        reasons.append("original GDF event timebase 0; no supervised gold")
    elif source == "p30032":
        reasons.extend(["BCI2000 timing lag unresolved", "actual-person group identity unresolved"])
    elif source == "cnt14":
        reasons.append("code4 ontology and original sample count unresolved")
    elif source == "inmid14":
        reasons.append("marker ontology and video latency unverified")
    elif source == "ssvep64_processed":
        reasons.extend(["amplitude unit unknown", "offline preprocessing causality unknown"])
    elif source == "brainstorm":
        reasons.append("electrode role 19/20 uncertain")
    elif source == "small_bci_mixed":
        reasons.append("dataset-specific units and interval overlap not established for mixed releases")
    return "held_pending_gates", reasons


def propose_split(component_key: str) -> str:
    # Fixed metadata-only split proposal; no signal/labels looked at.
    bucket = int(digest("PJ-Xavier-next-multichannel-v1|" + component_key)[:16], 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def build(repo: Path, output: Path) -> dict:
    repo = repo.absolute()
    old = json.loads((repo / EXISTING_SPLIT).read_text())
    old_assignments = old["assignments"]
    assert isinstance(old_assignments, dict)
    assert len(old_assignments) == 109
    assert set(old_assignments.values()) == {"train", "val", "test"}

    flags = json.loads((repo / BCI_FLAGS).read_text())["author_reported_QC_flags"]
    fif = json.loads((repo / FIF_MUST_LINK).read_text())
    uf = UnionFind()
    records: list[dict] = []
    inputs = {EXISTING_SPLIT: file_digest(repo / EXISTING_SPLIT),
              BCI_FLAGS: file_digest(repo / BCI_FLAGS),
              FIF_MUST_LINK: file_digest(repo / FIF_MUST_LINK)}
    duplicate_keys: dict[tuple[str, str, str], str] = {}
    all_source_counts: Counter[str] = Counter()
    missing_group: Counter[str] = Counter()

    for source, (rel, declared) in LEDGERS.items():
        path = repo / rel
        inputs[rel] = file_digest(path)
        with path.open() as fh:
            for line_index, text in enumerate(fh):
                row = json.loads(text)
                all_source_counts[source] += 1
                group = source_group(source, row)
                if group is None:
                    group = f"{source}|unknown-row-{line_index}"
                    missing_group[source] += 1
                uf.find(group)
                payload = nonempty_str(row.get("EEG_payload_sha256", row.get("eeg_payload_sha256",row.get("decoded_payload_sha256"))))
                calibration = nonempty_str(row.get("EEG_calibration_sha256",row.get("calibration_sha256")))
                # Only link when both calibrated payload and calibration are verified.
                if payload and calibration:
                    key = (source, payload, calibration)
                    if key in duplicate_keys:
                        uf.union(group, duplicate_keys[key])
                    duplicate_keys[key] = group
                n = channel_count(source, row, declared)
                status, reasons = readiness(source, n)
                if group.removeprefix(source + "|") in flags and source == "bci27":
                    reasons.append("author QC flag: review hold")
                if group in old_assignments:
                    reasons.append("existing heldout group; must not reassign")
                records.append({
                    "source": source, "ledger_row": line_index, "source_group": group,
                    "source_file_sha256": row.get("source_file_sha256",row.get("source_sha256")),
                    "eeg_payload_sha256": payload, "calibration_sha256": calibration,
                    "channel_count": n, "native_hz": row.get("native_sample_rate_hz",row.get("sample_rate_hz")),
                    "unit_provenance": row.get("unit_scope",row.get("physical_unit_scope",
                                     row.get("explicit_amplitude_unit",row.get("unit",row.get("calibration_scope"))))),
                    "raw_label": describe_label(source,row),
                    "ai_weak_label": {"value": None, "provenance": None, "quality": "not_generated"},
                    "status": status, "held_reasons": reasons,
                    "source_member_or_path": row.get("source_member",row.get("source_path",
                                              row.get("source_SET_path",row.get("relative_path")))),
                })

    for pair in fif["pairs"]:
        uf.union("small_bci_mixed|" + pair["left_group"],"small_bci_mixed|" + pair["right_group"])
    # Cross-person source groups from the sealed BCI constraint ledger.
    bc = json.loads((repo / BCI_FLAGS).read_text())
    for entry in bc["must_link_groups"]:
        ids = (entry.get("source_participant_ids") or entry.get("source_groups")
               or entry.get("subject_groups") or [])
        for other in ids[1:]:
            uf.union("bci27|" + ids[0], "bci27|" + other)

    comp_members: dict[str, set[str]] = defaultdict(set)
    for r in records:
        comp_members[uf.find(r["source_group"])].add(r["source_group"])
    # Must-link unit is smallest canonical key in each union, not a record index.
    for r in records:
        component = uf.find(r["source_group"])
        groups = sorted(comp_members[component])
        fixed = set(old_assignments[g] for g in groups if g in old_assignments)
        if len(fixed) > 1:
            raise ValueError("existing fixed subject split crosses must-link component")
        split = next(iter(fixed)) if fixed else propose_split(component)
        r["metadata_only_proposed_split"] = split
        r["component_id_sha256"] = digest(component)
        r["must_link_group_count"] = len(groups)
        r["is_existing_split_fixed"] = bool(fixed)
        if fixed and split != "train":
            r["held_reasons"].append("existing fixed validation/test deny")
        # Fail closed. "proposed_split" is never ready for gradient input.
        r["fit_allowed"] = False
        r["validation_gold_allowed"] = False
        r["test_gold_allowed"] = False
        r["offline_noncausal_input_allowed_for_online_harness"] = False

    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec,sort_keys=True,ensure_ascii=False)+"\n")

    counts = Counter((r["source"],r["metadata_only_proposed_split"]) for r in records)
    report = {
        "schema": "pj_xavier_next_multichannel_cpu_admission_v1",
        "status": "metadata_only_all_new_source_gradient_exposure_zero",
        "records_checked": len(records),
        "source_counts": dict(sorted(all_source_counts.items())),
        "proposed_metadata_split_counts": {s:{fold:counts[(s,fold)] for fold in ("train","val","test")}
                                           for s in sorted(all_source_counts)},
        "missing_source_group_by_source": dict(sorted(missing_group.items())),
        "known_old_split_count_preserved": {k:list(old_assignments.values()).count(k)
                                           for k in ("train","val","test")},
        "must_link_component_count": len(comp_members),
        "source_input_sha256": inputs,
        "private_manifest_sha256": file_digest(output),
        "new_gradient_input_hours": 0,
        "new_gold_labels_from_AI": 0,
        "online_causal_samples_admitted": 0,
        "all_records_held_for_signal_QC_and_gates": True,
        "blocked_supervised_timing": ["bci27","p30032","cnt14","ssvep64_processed"],
        "warning": "Proposed metadata splits are not approved subject identities or independent heldout claims. Reconcile cross-source waveform hashes and group identity before fit. Never use old val/test in SSL."
    }
    return report


if __name__ == "__main__":
    output = REPO / "private/xavier/phase4_automation_1615/source_admission.private.jsonl"
    aggregate = REPO / "experiments/pj_xavier/reports/next_multichannel_admission_20261122_1615.aggregate.json"
    rep = build(REPO, output)
    aggregate.write_text(json.dumps(rep,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    print(json.dumps({"status":rep["status"],"record_count":rep["records_checked"],
                      "must_link_components":rep["must_link_component_count"],
                      "source_counts":rep["source_counts"],"manifest_sha":rep["private_manifest_sha256"]},
                     ensure_ascii=False))
