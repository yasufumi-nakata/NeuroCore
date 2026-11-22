#!/usr/bin/env python3
"""Snapshot acquired files and combine locations without counting replicas twice.

This module reports release payload bytes, not pure EEG bytes or recording hours.
It does not download, change a ledger, or infer signal duration from file size.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_destination(root: Path, dataset_key: str, relative_path: str) -> Path:
    parts = Path(dataset_key) / relative_path
    if Path(dataset_key).is_absolute() or Path(relative_path).is_absolute() or ".." in parts.parts:
        raise ValueError("unsafe ledger path")
    current = root
    for piece in parts.parts:
        current = current / piece
        if current.is_symlink():
            raise ValueError("symlink in ledger path")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError("destination outside root")
    return current


def snapshot(ledger: Path, root: Path, location: str, rehash: bool = False) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    def ledger_rows():
        with closing(sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            for row in db.execute(
                "SELECT id,catalog_id,dataset_key,relative_path,status,bytes_present,expected_bytes,sha256 "
                "FROM downloads ORDER BY id"):
                yield dict(row)
    # Preserve the original canonical JSON-array hash without materializing a
    # potentially half-million-row ledger or doing filesystem stats for every
    # queued path. Only actual partial files and completed entries need a stat.
    actual_partials = {str(p.relative_to(root))[:-5] for p in root.rglob("*.part")}
    ledger_digest = hashlib.sha256(b"[")
    row_count, statuses = 0, Counter()
    complete, partial, held = [], [], []
    for row in ledger_rows():
        if row_count:
            ledger_digest.update(b",")
        ledger_digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        row_count += 1
        statuses[row["status"]] += 1
        relative_key = str(Path(row["dataset_key"]) / row["relative_path"])
        if row["status"] != "complete" and relative_key not in actual_partials:
            continue
        identity = {key: row[key] for key in ("id", "catalog_id", "dataset_key", "relative_path")}
        try:
            target = safe_destination(root, row["dataset_key"], row["relative_path"])
            if row["status"] == "complete":
                if not target.is_file():
                    raise ValueError("complete file missing")
                before = target.stat()
                if before.st_size != row["bytes_present"]:
                    raise ValueError("actual size differs from completion ledger")
                if row["expected_bytes"] is not None and before.st_size != row["expected_bytes"]:
                    raise ValueError("actual size differs from pinned expected size")
                if not re.fullmatch(r"[a-f0-9]{64}", row["sha256"] or ""):
                    raise ValueError("completion ledger has no valid SHA-256")
                if rehash and sha256_file(target) != row["sha256"]:
                    raise ValueError("actual SHA-256 differs from completion ledger")
                after = target.stat()
                if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                    raise ValueError("file changed during snapshot")
                complete.append({**identity, "bytes": before.st_size, "sha256": row["sha256"],
                                 "integrity": "sha256_reverified" if rehash else "completion_sha256_and_current_size"})
            else:
                part = target.with_name(target.name + ".part")
                if part.is_symlink():
                    raise ValueError("partial path is a symlink")
                if part.is_file():
                    partial.append({**identity, "bytes": part.stat().st_size,
                                    "expected_bytes": row["expected_bytes"], "status": row["status"]})
        except (OSError, ValueError) as exc:
            held.append({**identity, "reason": str(exc)})
    ledger_digest.update(b"]")
    return {"schema": "pj-xavier-acquired-snapshot-v1", "location": location,
            "observed_started_utc": started, "observed_finished_utc": datetime.now(timezone.utc).isoformat(),
            "ledger_rows": row_count, "ledger_snapshot_sha256": ledger_digest.hexdigest(),
            "ledger_hash_format": "SHA-256 of canonical JSON array ordered by ledger ID; computed as a stream",
            "ledger_status_counts": dict(statuses),
            "complete_files": complete, "partial_files": partial, "held_files": held,
            "rehash_requested": rehash, "byte_scope": "release payloads including non-EEG modalities and sidecars",
            "partial_note": "Partial files may keep growing during and after this observation; no integrity claim."}


def summarize(snapshots: list[dict]) -> dict:
    locations = [item["location"] for item in snapshots]
    if len(set(locations)) != len(locations):
        raise ValueError("duplicate location snapshot would double count physical copies")
    unique, all_files, location_summaries = {}, [], []
    for snap in snapshots:
        if snap["schema"] != "pj-xavier-acquired-snapshot-v1":
            raise ValueError("unsupported snapshot schema")
        seen_ids = set()
        for entry in snap["complete_files"]:
            if entry["id"] in seen_ids:
                raise ValueError("duplicate completed ledger row")
            seen_ids.add(entry["id"])
            digest, size = entry["sha256"], entry["bytes"]
            if not re.fullmatch(r"[a-f0-9]{64}", digest) or not isinstance(size, int) or size < 0:
                raise ValueError("invalid completion identity")
            if digest in unique and unique[digest] != size:
                raise ValueError("same SHA-256 associated with different sizes")
            unique[digest] = size
            all_files.append(entry)
        location_summaries.append({"location": snap["location"],
            "observed_started_utc": snap["observed_started_utc"], "observed_finished_utc": snap["observed_finished_utc"],
            "ledger_rows": snap["ledger_rows"], "ledger_snapshot_sha256": snap["ledger_snapshot_sha256"],
            "ledger_status_counts": snap["ledger_status_counts"], "complete_file_copies": len(snap["complete_files"]),
            "complete_file_copy_bytes": sum(entry["bytes"] for entry in snap["complete_files"]),
            "partial_file_copies": len(snap["partial_files"]),
            "partial_observed_bytes": sum(entry["bytes"] for entry in snap["partial_files"]),
            "held_file_count": len(snap["held_files"]), "sha256_reverified": snap["rehash_requested"]})
    return {"schema": "pj-xavier-acquired-progress-v1", "locations": location_summaries,
            "completed_file_copies": len(all_files), "completed_file_copy_bytes": sum(e["bytes"] for e in all_files),
            "completed_unique_sha256_contents": len(unique), "completed_unique_sha256_bytes": sum(unique.values()),
            "deduplication": "Identical complete file content SHA-256 is counted once across all supplied ledgers.",
            "interpretation": "Acquired release payload bytes, including multimodal sidecars. This is not pure EEG bytes, unique recordings, or all catalog data.",
            "known_deduplication_limits": ["Different raw/filtered exports can have different hashes for the same recording.",
                                           "Archives are not compared with their extracted members.",
                                           "Replica archives outside the selected acquisition ledgers are omitted."],
            "recording_hours": None, "hours_reason": "Requires independent header and recording-identity audits; never inferred from bytes.",
            "all_catalog_files_downloaded": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    observe = commands.add_parser("snapshot")
    observe.add_argument("--ledger", type=Path, required=True)
    observe.add_argument("--root", type=Path, required=True)
    observe.add_argument("--location", required=True)
    observe.add_argument("--rehash", action="store_true")
    observe.add_argument("--output", type=Path, required=True)
    combine = commands.add_parser("summarize")
    combine.add_argument("--snapshots", type=Path, nargs="+", required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output already exists; choose a new observation filename")
    if args.command == "snapshot":
        result = snapshot(args.ledger, args.root, args.location, args.rehash)
    else:
        result = summarize([json.loads(path.read_text()) for path in args.snapshots])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), "schema": result["schema"]}))


if __name__ == "__main__":
    main()
