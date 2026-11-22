#!/usr/bin/env python3
"""Freeze Figshare 31578517 v2 participant groups using metadata only.

No EDF payload, questionnaire, event or outcome table is opened. The publisher's
25 complete participant files are distinct metadata groups, not independently
verified biological identities. The exact EDF headers and release hashes are
retained privately; public code contains no participant identifiers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

SOURCE = "figshare:31578517:v2"
CATALOG = "000336"
METADATA_SHA256 = "5e68ee6e2b712b037282989dba7723f80ef30b2c1990a78fd521f312555a0822"
HEADER_EVIDENCE_SHA256 = "70290db281b300fadb9cf790d510c69482b330adede35d78fe8cc18ac7ad1185"
CHANNELS = ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
            "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5",
            "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2"]
SPLIT_COUNTS = {"train": 17, "val": 5, "test": 3}
NAME_PATTERN = re.compile(r"sub-([A-Za-z0-9]+)_task-art_eeg\.edf")
SCHEMA = "xavier-smac-design31578517-metadata-freeze-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def sha(path):
    return digest_bytes(Path(path).read_bytes())


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      indent=2).encode() + b"\n"


def validate_header(row):
    c = row["channels"]
    require(row["n_channels"] == 32 and row["header_bytes"] == 8448,
            "Expected exactly 32 delivered EEG channels")
    require(c["label"] == CHANNELS and len(set(c["label"])) == 32,
            "Original EEG channel order differs")
    require(all(len(values) == 32 for values in c.values()),
            "Incomplete original channel header fields")
    require(c["unit"] == ["uV"] * 32, "Unverified original physical unit")
    require(all(int(v) == -32768 for v in c["dmin"])
            and all(int(v) == 32767 for v in c["dmax"]),
            "Unexpected EDF digital calibration")
    for lo, hi in zip(c["pmin"], c["pmax"]):
        require(math.isfinite(float(lo)) and math.isfinite(float(hi))
                and float(lo) < float(hi), "Invalid EDF physical calibration")
    ns = [int(x) for x in c["samples"]]
    require(len(set(ns)) == 1 and ns[0] > 0, "Different channel sample counts")
    duration = Fraction(str(row["record_duration_s"]))
    require(duration > 0 and row["n_records"] == 1,
            "Expected one original record covering each participant")
    fs = Fraction(ns[0], 1) / duration
    require(1023.99 <= float(fs) <= 1024.01, "Unexpected actual header sample rate")
    require(row["actual_bytes"] == row["expected_bytes"]
            == row["header_bytes"] + row["n_records"] * sum(ns) * 2,
            "Header geometry and verified download size differ")
    require(abs(row["duration_s"] - float(duration * row["n_records"])) < 1e-9,
            "Original record duration mismatch")
    require(row["payload_read"] is False, "Freeze requires header-only evidence")
    for key in ("file_sha256", "header_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", row[key]) is not None,
                "Source/header identity missing")
    return fs, ns[0] * row["n_records"]


def make_plan(metadata, evidence):
    require(metadata["id"] == 31578517 and metadata["version"] == 2,
            "Use the pinned official Figshare release")
    require(metadata["is_public"] is True and metadata["download_disabled"] is False
            and metadata["is_confidential"] is False and metadata["is_embargoed"] is False,
            "Public release access conditions changed")
    require(metadata["license"]["name"] == "CC BY 4.0", "Source license changed")
    require(evidence["source"] == SOURCE and evidence["signal_payload_read"] is False
            and evidence["raw_signal_bytes_to_workstation"] == 0,
            "Header evidence must precede signal decoding")
    readme = evidence["author_code"]["README.txt"]
    require("25 subjects have complete EEG" in readme
            and "Recorded EEG Channels: 32 channels" in readme
            and "Approximately 1024 Hz" in readme,
            "Required delivered README corrections/group provenance are missing")
    files = [r for r in metadata["files"] if r["name"].lower().endswith(".edf")]
    rows = evidence["rows"]
    require(len(files) == len(rows) == 25, "All 25 original EDF metadata rows required")
    by_name = {r["name"]: r for r in files}
    require(len(by_name) == 25, "Duplicate original EDF filename")
    require(len({str(r["id"]) for r in files}) == 25, "Duplicate provider file identity")
    planned = []
    for r in rows:
        name = r["relative_path"]
        m = NAME_PATTERN.fullmatch(name)
        require(m is not None and Path(name).name == name, "Unreviewed participant filename")
        f = by_name.get(name)
        require(f is not None and f["is_link_only"] is False
                and f["size"] == r["expected_bytes"], "Official file/header identity differs")
        require(re.fullmatch(r"[0-9a-f]{32}", f["computed_md5"]) is not None,
                "Published file MD5 missing")
        require(not f.get("supplied_md5") or f["supplied_md5"] == f["computed_md5"],
                "Published checksum conflict")
        fs, samples = validate_header(r)
        group_hash = digest_bytes((SOURCE + ":participant:" + m.group(1)).encode())
        planned.append({
            "relative_path": name, "provider_file_id": str(f["id"]),
            "source_sha256": r["file_sha256"], "source_bytes": r["actual_bytes"],
            "published_md5": f["computed_md5"], "header_sha256": r["header_sha256"],
            "subject_group": SOURCE + ":group-sha256:" + group_hash,
            "group_id_sha256": group_hash, "native_sample_count": samples,
            "native_rate_numerator": fs.numerator, "native_rate_denominator": fs.denominator,
            "native_sample_rate_hz": float(fs), "native_seconds": r["duration_s"],
            "header": {key: r[key] for key in ("n_channels", "header_bytes", "n_records",
                                               "record_duration_s", "duration_s", "channels")},
            "signal_opened_before_freeze": False,
        })
    for field in ("relative_path", "provider_file_id", "source_sha256",
                  "published_md5", "subject_group", "group_id_sha256"):
        require(len({r[field] for r in planned}) == 25,
                "Duplicate source/group identity requires must-link review: " + field)
    planned.sort(key=lambda r: r["group_id_sha256"])
    for rank, row in enumerate(planned):
        row["hash_rank"] = rank
        row["split"] = "train" if rank < 17 else ("val" if rank < 22 else "test")
        row["fit_candidate"] = row["split"] == "train"
    require(dict(Counter(r["split"] for r in planned)) == SPLIT_COUNTS,
            "Fixed group split counts differ")
    return {
        "schema": SCHEMA, "source": SOURCE, "catalog_id": CATALOG,
        "provider_metadata_sha256": METADATA_SHA256,
        "header_evidence_sha256": HEADER_EVIDENCE_SHA256,
        "readme_sha256": digest_bytes(readme.encode()),
        "publisher_preprocessing_code_sha256": digest_bytes(evidence["author_code"]["pre.py"].encode()),
        "split_method": "lexicographic SHA256(source_version + ':participant:' + original_subject_token); first17 train,next5 val,last3 test",
        "split_counts": SPLIT_COUNTS, "group_count": 25,
        "group_evidence": "Publisher describes 25 complete participants; exactly one unique EDF and source SHA/MD5 per filename subject. Biological independence and identities beyond supplied metadata are not independently verified.",
        "unresolved_cross_dataset_participant_linkage": True,
        "metadata_signal_reads": 0, "questionnaire_or_label_tables_opened": 0,
        "source_causality_verified": False, "online_harness_allowed": False,
        "newsource_val_and_test_signals_must_remain_unopened": True,
        "native_total_hours_metadata": sum(r["native_seconds"] for r in planned) / 3600,
        "records": planned,
    }


def freeze(metadata_path, evidence_path, output):
    require(sha(metadata_path) == METADATA_SHA256, "Pinned provider metadata seal differs")
    require(sha(evidence_path) == HEADER_EVIDENCE_SHA256, "Header evidence seal differs")
    target = Path(output).expanduser().absolute()
    require(not target.exists(), "Use a new immutable metadata-freeze directory")
    body = make_plan(json.loads(Path(metadata_path).read_text()),
                     json.loads(Path(evidence_path).read_text()))
    body["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    body["freeze_code_sha256"] = sha(__file__)
    blob = canonical(body)
    target.mkdir(parents=True)
    (target / "frozen_plan.json").write_bytes(blob)
    receipt = {
        "status": "metadata_only_group_split_frozen",
        "frozen_at_utc": body["frozen_at_utc"], "plan_sha256": digest_bytes(blob),
        "source": SOURCE, "split_counts": SPLIT_COUNTS,
        "metadata_signal_reads": 0, "questionnaire_or_label_tables_opened": 0,
        "metadata_sha256": METADATA_SHA256, "header_evidence_sha256": HEADER_EVIDENCE_SHA256,
        "freeze_code_sha256": body["freeze_code_sha256"],
    }
    (target / "freeze_receipt.json").write_bytes(canonical(receipt))
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", required=True)
    p.add_argument("--headers", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print(json.dumps(freeze(args.metadata, args.headers, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
