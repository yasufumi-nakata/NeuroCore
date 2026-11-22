#!/usr/bin/env python3
"""Read-only, exhaustive queued-file path audit. No downloads or waveform claims."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
import time


RULES_PATH = Path(__file__).with_name("structure_audit_rules_v1.json")
FIELDS = ("id", "catalog_id", "dataset_key", "relative_path", "original_relative_path", "expected_bytes")
LABELS = {
    "archive_uninspected": ("未展開 archive／分割 archive", "archive_uninspected"),
    "video_payload": ("動画拡張子", "other_modality_payload_candidate"),
    "audio_payload": ("音声拡張子", "other_modality_payload_candidate"),
    "image_payload": ("静止画像／図の拡張子", "other_modality_payload_candidate"),
    "neuroimage_payload": ("MRI 等の neuroimage／DICOM 拡張子", "other_modality_payload_candidate"),
    "software_or_documentation": ("ソフトウェア／文書", "ancillary_or_documentation"),
    "other_physio_signal_or_array": ("ECG／EOG／EMG／physio 等の信号・表候補", "other_physio_candidate"),
    "other_physio_ancillary": ("ECG／EOG／EMG／physio 等の付随ファイル候補", "ancillary_or_documentation"),
    "modality_path_conflict": ("EEG／iEEG／MEG のパス根拠が競合", "ambiguous_or_unknown"),
    "generic_eeg_format_ambiguous": ("汎用 EEG 形式，モダリティ未確認", "generic_signal_format_unverified"),
    "generic_meeg_format_ambiguous": ("汎用 MEG／EEG 形式，モダリティ未確認", "generic_signal_format_unverified"),
    "generic_neurophysiology_container": ("NWB／XDF 等の汎用神経生理容器", "generic_signal_format_unverified"),
    "numeric_container_without_modality": ("数値・表・binary 容器，モダリティ未確認", "ambiguous_or_unknown"),
    "ancillary_without_modality": ("sidecar／注釈候補，モダリティ未指定", "ancillary_or_documentation"),
    "other_compressed_uninspected": ("その他の単一圧縮形式，中身未確認", "ambiguous_or_unknown"),
    "other_unknown": ("その他・判定材料不足", "ambiguous_or_unknown"),
    "path_unusable": ("利用できる相対パスなし", "ambiguous_or_unknown"),
}
for _modality, _name in (("eeg", "EEG"), ("ieeg", "iEEG"), ("meg", "MEG"), ("neuroimage", "MRI 等")):
    for _kind, _description in (("signal", "信号形式候補"), ("numeric", "数値・表容器候補"),
                                ("ancillary", "sidecar／注釈候補"), ("other", "その他の未確認ファイル")):
        _group = ("explicit_eeg_signal_candidate" if _modality == "eeg" and _kind == "signal" else
                  "explicit_eeg_numeric_candidate" if _modality == "eeg" and _kind == "numeric" else
                  "ieeg_content_candidate" if _modality == "ieeg" and _kind in {"signal", "numeric"} else
                  "meg_container_eeg_unverified" if _modality == "meg" and _kind in {"signal", "numeric"} else
                  "ancillary_or_documentation" if _kind == "ancillary" else "ambiguous_or_unknown")
        LABELS[_modality + "_path_" + _kind] = (_name + " パス／命名の" + _description, _group)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")


class Classifier:
    def __init__(self, rules):
        if rules.get("schema_version") != "pj-xavier-path-structure-rules-v1":
            raise ValueError("unsupported frozen classification rules")
        self.rules = rules
        self.sets = {key: set(value) for key, value in rules.items() if isinstance(value, list)}
        self.volume = re.compile(rules["archive_volume_regex"])

    def classify(self, path):
        if not isinstance(path, str) or not path or "\x00" in path or "://" in path:
            return "path_unusable", ["invalid_or_missing_relative_path"], "", []
        text = path.replace("\\", "/").casefold()
        parts = PurePosixPath(text).parts
        if not parts or text.startswith("/") or ".." in parts:
            return "path_unusable", ["absolute_or_traversal_path"], "", []
        name = parts[-1]
        flags = ["derivative_path"] if "derivatives" in parts else []
        for suffix in self.rules["archive_suffixes"]:
            if name.endswith(suffix):
                return "archive_uninspected", ["archive_suffix:" + suffix], suffix, flags
        if self.volume.search(name):
            return "archive_uninspected", ["split_archive_suffix"], "archive-volume", flags
        compression = next((x for x in self.rules["single_compression_suffixes"] if name.endswith(x)), "")
        inner = name[:-len(compression)] if compression else name
        extension = PurePosixPath(inner).suffix
        stem = inner[:-len(extension)] if extension else inner
        tokens = set(re.split(r"[_\-.\s]+", stem))
        evidence = []
        modalities = set()
        for modality, values in self.rules["modality_directory_tokens"].items():
            if set(values).intersection(parts[:-1]):
                modalities.add(modality); evidence.append(modality + "_directory")
        for modality, values in self.rules["modality_filename_tokens"].items():
            if set(values).intersection(tokens):
                modalities.add(modality); evidence.append(modality + "_filename_token")
        if any(x.endswith(".ds") for x in parts[:-1]):
            modalities.add("meg"); evidence.append("ds_package")
        if any(x.endswith(".mefd") for x in parts[:-1]):
            modalities.add("ieeg"); evidence.append("mefd_package")
        mff_signal = any(x.endswith(".mff") for x in parts[:-1]) and re.fullmatch(r"signal\d+\.bin", inner)
        if modalities:
            flags.extend("path_hint_" + x for x in sorted(modalities))
        if compression:
            flags.append("single_compressed_payload_unread")
        suffix = extension + compression
        for field, category in (("video_extensions", "video_payload"), ("audio_extensions", "audio_payload"),
                                ("image_extensions", "image_payload"), ("neuroimage_extensions", "neuroimage_payload")):
            if extension in self.sets[field]:
                return category, evidence + ["payload_extension:" + extension], suffix, flags
        if extension in self.sets["documentation_extensions"] | self.sets["software_extensions"]:
            return "software_or_documentation", evidence + ["code_or_document_extension"], suffix, flags
        ancillary = (extension in self.sets["sidecar_extensions"] or stem in self.sets["sidecar_exact_names"] or
                     (extension in {".tsv", ".csv", ".txt"} and
                      bool(tokens.intersection(self.sets["sidecar_stems"]))))
        physiology = tokens.intersection(self.sets["physiology_filename_tokens"])
        if physiology:
            category = "other_physio_ancillary" if ancillary else "other_physio_signal_or_array"
            return category, evidence + ["physiology_filename_token:" + x for x in sorted(physiology)], suffix, flags
        neural = modalities.intersection({"eeg", "ieeg", "meg"})
        if len(neural) > 1:
            return "modality_path_conflict", evidence, suffix, flags
        if neural:
            modality = next(iter(neural))
        elif "neuroimage" in modalities:
            modality = "neuroimage"
        else:
            modality = None
        signal = extension in self.sets["signal_extensions"] or (modality == "meg" and inner.startswith("c,rf"))
        numeric = extension in self.sets["numeric_container_extensions"]
        if modality:
            kind = "ancillary" if ancillary else "signal" if signal else "numeric" if numeric else "other"
            return modality + "_path_" + kind, evidence + ["suffix_role:" + kind], suffix, flags
        if ancillary:
            return "ancillary_without_modality", ["sidecar_or_annotation_suffix"], suffix, flags
        if extension in self.sets["generic_eeg_format_extensions"] or mff_signal:
            return "generic_eeg_format_ambiguous", ["signal_extension_or_mff_without_explicit_modality"], suffix, flags
        if extension in self.sets["generic_meeg_format_extensions"]:
            return "generic_meeg_format_ambiguous", ["shared_meg_eeg_format"], suffix, flags
        if extension in self.sets["generic_neurophysiology_extensions"]:
            return "generic_neurophysiology_container", ["generic_neural_container"], suffix, flags
        if numeric:
            return "numeric_container_without_modality", ["numeric_table_or_binary_extension"], suffix, flags
        if compression:
            return "other_compressed_uninspected", ["unrecognized_single_compressed_type"], suffix, flags
        return "other_unknown", ["insufficient_path_evidence"], suffix, flags


def counter():
    return {"files": 0, "known_bytes": 0, "unknown_size_files": 0, "zero_size_files": 0, "releases": set()}


def add(value, size, release):
    value["files"] += 1
    if size is None:
        value["unknown_size_files"] += 1
    else:
        if type(size) is not int or size < 0:
            raise ValueError("expected_bytes is not a nonnegative integer or NULL")
        value["known_bytes"] += size
        value["zero_size_files"] += int(size == 0)
    value["releases"].add(release)


def public_count(value, total=None):
    result = {key: (len(data) if key == "releases" else data) for key, data in value.items()}
    if total:
        result["known_bytes_fraction"] = result["known_bytes"] / total["known_bytes"] if total["known_bytes"] else 0.
        result["files_fraction"] = result["files"] / total["files"] if total["files"] else 0.
    return result


def release_review(categories):
    if categories.get("eeg_path_signal"):
        return "explicit_eeg_signal_path_candidate", "P1_read_signal_header_and_resolve_companions"
    if categories.get("eeg_path_numeric"):
        return "explicit_eeg_numeric_candidate_only", "P2_inspect_numeric_schema_and_units"
    if any(categories.get(x) for x in ("meg_path_signal", "meg_path_numeric", "generic_meeg_format_ambiguous")):
        return "meg_container_requires_eeg_channel_check", "P2_read_channel_types_and_EEGChannelCount"
    if categories.get("generic_eeg_format_ambiguous") or categories.get("generic_neurophysiology_container"):
        return "generic_signal_container_requires_header", "P2_identify_channels_sampling_units_and_scope"
    if categories.get("ieeg_path_signal") or categories.get("ieeg_path_numeric"):
        return "intracranial_candidate_without_explicit_scalp_eeg", "P3_review_invasive_to_scalp_transfer_scope"
    if categories.get("eeg_path_ancillary"):
        return "eeg_companion_without_explicit_signal_candidate", "P2_resolve_companion_to_waveform_reference"
    if categories.get("archive_uninspected"):
        return "archive_requires_inventory", "P2_inspect_archive_inventory_before_volume_commitment"
    return "no_explicit_eeg_signal_evidence_from_paths", "P3_review_official_release_scope_and_unrecognized_formats"


def audit(database, output, rules_path=RULES_PATH, private_detail=None):
    output, database, rules_path = Path(output), Path(database).resolve(), Path(rules_path).resolve()
    if output.exists():
        raise ValueError("audit output already exists; refusing to overwrite")
    if not database.is_file():
        raise ValueError("source SQLite database is absent")
    rules_bytes = rules_path.read_bytes()
    rules = json.loads(rules_bytes)
    classify = Classifier(rules)
    source_hash, detail_hash = hashlib.sha256(), hashlib.sha256()
    rules_hash, code_hash = hashlib.sha256(rules_bytes).hexdigest(), sha256(__file__)
    started = time.time()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "frozen_protocol.json", {"started_epoch": started, "rules_sha256": rules_hash,
               "code_sha256": code_hash, "rules": rules, "rules_frozen_before_full_queue_scan": True})
    total = counter()
    categories, groups, extensions, flags, providers = (defaultdict(counter) for _ in range(5))
    releases = {}
    details = None
    if private_detail:
        detail_path = Path(private_detail)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        details = gzip.open(detail_path, "xb", compresslevel=3)
    uri = database.as_uri() + "?mode=ro"
    db = sqlite3.connect(uri, uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA cache_size=-8192")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("BEGIN")
        query_only = db.execute("PRAGMA query_only").fetchone()[0]
        base = db.execute("SELECT COUNT(*), COALESCE(SUM(expected_bytes),0), SUM(expected_bytes IS NULL), COUNT(DISTINCT dataset_key) FROM downloads").fetchone()
        source_table = {"files": base[0], "known_bytes": base[1], "unknown_size_files": base[2] or 0, "releases": base[3]}
        rows = db.execute("SELECT " + ",".join(FIELDS) + " FROM downloads ORDER BY id")
        for row in rows:
            record = {key: row[key] for key in FIELDS}
            source_hash.update(encoded(record))
            path = row["original_relative_path"] or row["relative_path"]
            category, evidence, suffix, hints = classify.classify(path)
            if category not in LABELS:
                raise ValueError("unregistered classifier category")
            dataset, size = row["dataset_key"], row["expected_bytes"]
            provider = dataset.partition("-")[0]
            for bucket in (total, categories[category], groups[LABELS[category][1]], extensions[suffix or "[no extension]"], providers[provider]):
                add(bucket, size, dataset)
            for flag in hints:
                add(flags[flag], size, dataset)
            release = releases.setdefault(dataset, {"totals": counter(), "categories": Counter(), "category_bytes": Counter(),
                "catalog_ids": set(), "provider": provider})
            add(release["totals"], size, dataset)
            release["categories"][category] += 1
            release["category_bytes"][category] += size or 0
            release["catalog_ids"].add(str(row["catalog_id"]))
            detail = {"job_id": row["id"], "release_key": dataset, "catalog_id": row["catalog_id"],
                      "relative_path_evidence": path, "path_field": "original_relative_path" if row["original_relative_path"] else "relative_path",
                      "known_bytes": size, "category": category, "reason": evidence, "flags": hints,
                      "header_validated": False, "training_approved": False}
            payload = encoded(detail)
            detail_hash.update(payload)
            if details:
                details.write(payload)
        db.rollback()
    finally:
        db.close()
        if details:
            details.close()
    actual = public_count(total)
    if sha256(rules_path) != rules_hash or sha256(__file__) != code_hash:
        raise ValueError("frozen audit rules or source changed during execution")
    assert query_only == 1
    for key in ("files", "known_bytes", "unknown_size_files", "releases"):
        if actual[key] != source_table[key]:
            raise ValueError("source total mismatch:" + key)
    for bins in (categories, groups, extensions, providers):
        for key in ("files", "known_bytes", "unknown_size_files", "zero_size_files"):
            if sum(value[key] for value in bins.values()) != total[key]:
                raise ValueError("exclusive classification sum mismatch:" + key)
    review_groups = defaultdict(counter)
    release_rows = []
    for dataset, item in sorted(releases.items()):
        review, priority = release_review(item["categories"])
        counts = public_count(item["totals"])
        group = review_groups[review]
        for key in ("files", "known_bytes", "unknown_size_files", "zero_size_files"):
            group[key] += counts[key]
        group["releases"].add(dataset)
        release_rows.append({"release_key": dataset, "catalog_ids": ",".join(sorted(item["catalog_ids"])),
                             "provider": item["provider"], **{k: v for k, v in counts.items() if k != "releases"},
                             "path_review": review, "proposed_next_step": priority,
                             "explicit_eeg_signal_files": item["categories"]["eeg_path_signal"],
                             "explicit_eeg_signal_known_bytes": item["category_bytes"]["eeg_path_signal"],
                             "explicit_eeg_numeric_files": item["categories"]["eeg_path_numeric"],
                             "archive_files": item["categories"]["archive_uninspected"],
                             "meg_candidate_files": sum(item["categories"][x] for x in ("meg_path_signal", "meg_path_numeric", "generic_meeg_format_ambiguous")),
                             "ieeg_candidate_files": sum(item["categories"][x] for x in ("ieeg_path_signal", "ieeg_path_numeric")),
                             "neuroimage_payload_files": item["categories"]["neuroimage_payload"],
                             "video_payload_files": item["categories"]["video_payload"],
                             "header_validated_by_this_audit": False, "queue_changed": False})
    for key in ("files", "known_bytes", "unknown_size_files", "zero_size_files"):
        if sum(v[key] for v in review_groups.values()) != total[key]:
            raise ValueError("exclusive release classification sum mismatch")
    if sum(len(v["releases"]) for v in review_groups.values()) != len(releases):
        raise ValueError("release assignment not exclusive")
    with (output / "release_review.csv").open("x", newline="", encoding="utf-8") as f:
        if release_rows:
            writer = csv.DictWriter(f, fieldnames=list(release_rows[0]))
            writer.writeheader(); writer.writerows(release_rows)
    result = {"schema_version": "pj-xavier-queue-structure-audit-v1", "status": "completed",
              "started_epoch": started, "completed_epoch": time.time(),
              "source": {"database_role": "bulk downloads SQLite", "query_mode": "read_only_query_only_single_transaction",
                         "hashed_columns": list(FIELDS), "ordered_metadata_sha256": source_hash.hexdigest(),
                         "status_columns_excluded": True, "urls_read_or_published": False},
              "protocol_sha256": rules_hash, "source_code_sha256": code_hash,
              "classification_stream_sha256": detail_hash.hexdigest(), "totals": actual,
              "source_sql_totals": source_table,
              "category_counts": {key: {"label": LABELS[key][0], "group": LABELS[key][1], **public_count(value, total)} for key, value in sorted(categories.items())},
              "exclusive_groups": {key: public_count(value, total) for key, value in sorted(groups.items())},
              "release_review_groups": {key: public_count(value, total) for key, value in sorted(review_groups.items())},
              "provider_counts": {key: public_count(value, total) for key, value in sorted(providers.items())},
              "extension_counts": {key: public_count(value, total) for key, value in sorted(extensions.items())},
              "nonadditive_path_flags": {key: public_count(value, total) for key, value in sorted(flags.items())},
              "integrity_checks": {"all_job_counts_equal": True, "all_known_bytes_equal": True, "all_unknown_size_counts_equal": True,
                                   "category_assignment_exclusive": True, "release_review_assignment_exclusive": True},
              "header_validated_by_this_audit": 0, "archives_inspected": 0, "unique_eeg_bytes_estimated": False,
              "recording_hours_estimated": False, "queue_modified": False, "downloads_started": 0,
              "training_approval_changed": False, "limitations": rules["critical_limitations"]}
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rules", default=RULES_PATH, type=Path)
    parser.add_argument("--private-detail", type=Path, help="Optional gzip JSONL of per-job path evidence; never publish")
    args = parser.parse_args()
    result = audit(args.db, args.output_dir, args.rules, args.private_detail)
    print(json.dumps({key: result[key] for key in ("status", "totals", "source", "integrity_checks")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
