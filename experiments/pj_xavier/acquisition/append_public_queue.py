#!/usr/bin/env python3
"""Freeze a refreshed catalog and append public EEG candidates without edits.

Only INSERTs are permitted on the live downloads table. No worker is launched,
stopped, or given a new deadline/budget. Unresolved scopes become review records.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
import urllib.parse

from acquire import atomic_json, checksum_spec, dataset_storage_key, hash_file, migrate_manifest_records, now, safe_relative, validate_url


SCOPES = {"include_provisional", "component_only_provisional"}
SEED_IDS = {"001430", "010914"}
SEED_RELEASES = {("zenodo", "4453396"), ("zenodo", "20801101")}
META_FIELDS = ("id", "dataset_key", "relative_path", "url", "expected_bytes", "checksum_json", "provider_file_identity")


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def metadata(row):
    value = {key: row[key] for key in META_FIELDS}
    value["checksum_json"] = checksum_spec(json.loads(value["checksum_json"] or "null"))
    return value


def catalog_ids(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        ids = [str(r.get("ID") or r.get("catalog_id") or "").zfill(6) for r in csv.DictReader(f)]
    if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch(r"\d{6}", value) for value in ids):
        raise ValueError("catalog IDs must be unique six-digit values")
    return set(ids)


def freeze_snapshot(source, directory, reserve_bytes):
    source, directory = Path(source), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    required = source.stat().st_size * 2 + 128 * 1024**2
    if shutil.disk_usage(directory).free < reserve_bytes + required:
        raise ValueError("metadata snapshot would cross disk reserve")
    temporary = directory / ("capture_" + str(time.time_ns()) + ".tmp")
    digest = hashlib.sha256()
    with source.open("rb") as src, temporary.open("xb") as dst:
        before = os.fstat(src.fileno())
        for block in iter(lambda: src.read(1024**2), b""):
            if shutil.disk_usage(directory).free <= reserve_bytes + len(block):
                raise ValueError("disk reserve reached during metadata snapshot")
            digest.update(block)
            dst.write(block)
        after = os.fstat(src.fileno())
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("source was modified in place during capture")
        dst.flush()
        os.fsync(dst.fileno())
    sha = digest.hexdigest()
    target = directory / ("catalog_" + sha + ".jsonl")
    if target.exists():
        if hash_file(target) != sha:
            raise ValueError("immutable snapshot path content changed")
        temporary.unlink()
    else:
        os.replace(temporary, target)
        target.chmod(0o444)
    return target, sha


def source_review(row):
    """Evidence-ranked proposal, not a scope rewrite or training permission."""
    description = html.unescape(re.sub(r"<[^>]+>", " ", str(row.get("source_description", ""))))
    title = str(row.get("title", ""))
    evidence_url = row.get("manifest_evidence_url") or row.get("verification_url")
    official_response = bool(evidence_url and re.fullmatch(r"[0-9a-f]{64}", str(row.get("metadata_body_sha256", ""))))
    eeg = re.compile(r"\b(?:EEG|electroencephalograph(?:y|ic|ical))\b", re.I)
    dataset = re.compile(r"\b(?:data(?:set)?|recordings?|signals?|recorded)\b", re.I)
    negative = re.search(r"\b(?:no EEG|without EEG|does not contain EEG|synthetic|simulated|intracranial|ECoG|rodent|macaque)\b", description, re.I)
    if official_response and eeg.search(description) and dataset.search(description) and not negative:
        priority, reason = "official_description_EEG_data_candidate", "official retained source description explicitly mentions EEG data; waveform body not yet verified"
        proposed = "component_only_provisional" if re.search(r"\b(?:fMRI|MEG|ECG|EDA|gaze|screen recordings)\b", description, re.I) else "include_provisional"
    elif eeg.search(title) or re.search(r"\b(?:SSVEP|P300|motor imagery)\b", title, re.I):
        priority, reason, proposed = "title_signal_needs_official_metadata_review", "title suggests EEG/BCI; title/file extension alone is insufficient", None
    else:
        priority, reason, proposed = "unresolved_metadata_review", "source evidence insufficient or mixed/non-scalp/synthetic terms need review", None
    if negative:
        priority, reason, proposed = "conflicting_scope_metadata_review", "description contains potentially excluded or mixed scope terms", None
    return {"catalog_id": row["catalog_id"], "title": title, "current_scope": row.get("scope_decision"),
            "current_status": row.get("status"), "priority": priority, "reason": reason, "proposed_acquisition_scope": proposed,
            "source_url": evidence_url, "metadata_body_sha256": row.get("metadata_body_sha256"),
            "description_excerpt": description[:900], "resolution_basis": row.get("resolution_basis"),
            "file_entries": len(row.get("files", [])), "waveform_content_verified": False,
            "training_approval": "not_granted", "automatically_ingested": False,
            "license": row.get("license"), "action": "review official metadata and waveform companions before extending acquisition scope"}


def prepare_release(db, row):
    normalized, migration = migrate_manifest_records([row])
    row = normalized[0]
    if row.get("acquisition_quarantine"):
        return None, row["acquisition_quarantine"]
    dataset = dataset_storage_key(row)
    if len(dataset.encode()) > 240:
        return None, [{"issue": "dataset_storage_key_exceeds_portable_filesystem_component_limit"}]
    old = db.execute("SELECT * FROM downloads WHERE dataset_key=?", (dataset,)).fetchall()
    by_path = {r["relative_path"]: r for r in old}
    by_identity = {}
    for r in old:
        if r["provider_file_identity"]:
            by_identity.setdefault(r["provider_file_identity"], []).append(r)
    new_jobs, guards, skipped, reasons, seen = [], {}, 0, [], {}
    for file in row.get("files", []):
        if file.get("declared_public") is not True or not file.get("manifest_evidence_url"):
            skipped += 1
            continue
        try:
            path = safe_relative(file["path"])
            if any(len(part.encode()) > 255 for part in Path(path).parts):
                raise ValueError("published_path_component_exceeds_filesystem_limit")
            url = validate_url(file["url"])
            checksum = checksum_spec(file.get("published_checksum"))
            size = file.get("size_bytes")
            if isinstance(size, int) and size > 2**63 - 1:
                raise ValueError("published_file_size_exceeds_sqlite_integer_range")
            known = isinstance(size, int) and not isinstance(size, bool) and size >= 0
            same_identity = by_identity.get(file.get("provider_file_identity"), [])
            if len(same_identity) > 1:
                exact = [r for r in same_identity if r["relative_path"] == path]
                if len(exact) != 1:
                    raise ValueError("ambiguous_existing_storage_paths_for_same_remote_file")
                existing = exact[0]
            else:
                existing = same_identity[0] if same_identity else by_path.get(path)
            if existing:
                old_meta = metadata(existing)
                if (old_meta["expected_bytes"] != (size if known else None) or old_meta["checksum_json"] != checksum
                        or old_meta["url"] != url or old_meta["provider_file_identity"] != file.get("provider_file_identity")):
                    raise ValueError("same_pinned_file_or_storage_path_metadata_conflict")
                # Preserve the already assigned storage path, including a legacy
                # path currently being transferred. Do not rename or alias rows.
                guards[existing["id"]] = old_meta
                continue
            ident = hashlib.sha256((dataset + "/" + path).encode()).hexdigest()
            job = {"id": ident, "catalog_id": row["catalog_id"], "dataset_key": dataset, "relative_path": path, "url": url,
                   "expected_bytes": size if known else None, "checksum_json": json.dumps(checksum, sort_keys=True),
                   "original_relative_path": file.get("original_path", path), "provider_file_id": file.get("provider_file_id"),
                   "provider_file_identity": file.get("provider_file_identity"), "status": "queued" if known else "metadata_blocked",
                   "error": None if known else "published_file_size_unverified; measure_before_bulk_transfer"}
            if ident in seen and metadata(seen[ident]) != metadata(job):
                raise ValueError("fresh_manifest_has_conflicting_duplicate_storage_path")
            if ident not in seen:
                new_jobs.append(job)
                seen[ident] = job
        except (ValueError, KeyError, TypeError) as exc:
            reasons.append({"path": file.get("path"), "issue": str(exc)})
    if reasons:
        return None, reasons
    return {"catalog_id": row["catalog_id"], "dataset_key": dataset, "source_version": row.get("source_version"),
            "scope_decision": row.get("scope_decision"), "training_use_status": row.get("training_use_status", "review_pending"),
            "training_approved": False, "companion_resolution_required": bool(row.get("companion_resolution_required")),
            "source_manifest_complete": row.get("manifest_complete") is True, "existing_guards": list(guards.values()),
            "new_jobs": new_jobs, "source_file_entries": len(row.get("files", [])), "unverified_public_files_skipped": skipped,
            "namespace_changes_in_fresh_manifest": migration["renamed_file_entries"]}, []


def make_plan(source, snapshots, catalog, db_path, output, reserve_bytes):
    output = Path(output)
    if output.exists():
        raise ValueError("use a new plan directory; immutable plan must not be overwritten")
    output.mkdir(parents=True)
    snapshot, snapshot_sha = freeze_snapshot(source, snapshots, reserve_bytes)
    expected_ids = catalog_ids(catalog)
    seen, counts, review_counts = set(), Counter(), Counter()
    known_bytes, unknown_sizes, proposed_jobs, selected_entries = 0, 0, 0, 0
    jobs_path, audit_path, review_path = (output / name for name in ("releases.jsonl", "catalog_audit.jsonl", "scope_review.jsonl"))
    with sqlite3.connect("file:" + str(Path(db_path).resolve()) + "?mode=ro", uri=True, timeout=30) as db, jobs_path.open("x") as jobs, audit_path.open("x") as audit, review_path.open("x") as reviews:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        schema = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='downloads'").fetchone()
        if not schema:
            raise ValueError("downloads ledger is missing")
        for row in read_rows(snapshot):
            cid = row["catalog_id"]
            if cid in seen:
                raise ValueError("duplicate catalog row in fresh snapshot")
            seen.add(cid)
            status, detail = "not_resolved_public", {}
            if row.get("scope_decision") == "unresolved":
                review = source_review(row)
                reviews.write(json.dumps(review, ensure_ascii=False) + "\n")
                review_counts[review["priority"]] += 1
            if cid in SEED_IDS or (row.get("provider"), str(row.get("source_id"))) in SEED_RELEASES:
                status = "seed_managed_separately"
            elif row.get("status") in {"excluded", "duplicate", "restricted"}:
                status = row["status"]
            elif row.get("scope_decision") not in SCOPES:
                status = "scope_review_pending"
            elif row.get("status") == "resolved":
                counts["resolved_EEG_provisional_catalog_rows"] += 1
                release, problems = prepare_release(db, row)
                if problems:
                    status, detail = "metadata_conflict_held", {"reasons": problems}
                else:
                    selected_entries += release["source_file_entries"]
                    n = len(release["new_jobs"])
                    status = "proposed_additions" if n else "unchanged_existing_release"
                    jobs.write(json.dumps(release, ensure_ascii=False) + "\n")
                    proposed_jobs += n
                    known_bytes += sum(j["expected_bytes"] or 0 for j in release["new_jobs"])
                    unknown_sizes += sum(j["expected_bytes"] is None for j in release["new_jobs"])
                    detail = {"dataset_key": release["dataset_key"], "new_jobs": n, "existing_files": len(release["existing_guards"]),
                              "companion_resolution_required": release["companion_resolution_required"]}
            counts[status] += 1
            audit.write(json.dumps({"catalog_id": cid, "scope_decision": row.get("scope_decision"), "source_status": row.get("status"),
                                    "plan_status": status, **detail}, ensure_ascii=False) + "\n")
        db.rollback()
    if seen != expected_ids:
        raise ValueError("fresh snapshot does not exactly cover current catalog IDs")
    files = {p.name: {"sha256": hash_file(p), "bytes": p.stat().st_size} for p in (jobs_path, audit_path, review_path)}
    plan_id = hashlib.sha256((snapshot_sha + "\n" + files["releases.jsonl"]["sha256"]).encode()).hexdigest()
    result = {"schema_version": "xavier-append-public-queue-v1", "plan_id": plan_id, "created_at": now(),
              "source_snapshot": str(snapshot), "source_snapshot_sha256": snapshot_sha, "source_snapshot_bytes": snapshot.stat().st_size,
              "catalog_sha256": hash_file(catalog), "catalog_rows": len(seen), "coverage_missing": 0, "coverage_extra": 0,
              "database_path": str(Path(db_path).resolve()), "files": files, "counts": dict(counts), "scope_review_counts": dict(review_counts),
              "selected_source_file_entries": selected_entries, "new_job_candidates": proposed_jobs,
              "new_known_release_file_bytes": known_bytes, "new_unknown_size_files": unknown_sizes,
              "byte_scope": "candidate whole-release files across versions and modalities; not unique EEG bytes and not downloaded bytes",
              "unique_EEG_hours": None, "training_approved": False, "worker_operations": 0,
              "source_access_holds_respected_at_apply": True}
    atomic_json(output / "plan.json", result)
    return result


def apply_plan(plan_dir, db_path, expected_plan_sha256, receipt_path, reserve_bytes):
    plan_dir, db_path = Path(plan_dir), Path(db_path).resolve()
    if hash_file(plan_dir / "plan.json") != expected_plan_sha256:
        raise ValueError("reviewed plan hash changed")
    plan = json.loads((plan_dir / "plan.json").read_text())
    if str(db_path) != plan["database_path"] or not db_path.is_file():
        raise ValueError("apply target differs from planned existing ledger")
    if hash_file(plan["source_snapshot"]) != plan["source_snapshot_sha256"]:
        raise ValueError("immutable source snapshot changed")
    for name, record in plan["files"].items():
        if hash_file(plan_dir / name) != record["sha256"]:
            raise ValueError("plan artifact changed:" + name)
    minimum = max(64 * 1024**2, int(plan["new_job_candidates"] * 4096))
    if shutil.disk_usage(db_path.parent).free < reserve_bytes + minimum:
        raise ValueError("metadata append would cross disk reserve")
    summary = {"plan_id": plan["plan_id"], "plan_sha256": expected_plan_sha256, "snapshot_sha256": plan["source_snapshot_sha256"],
               "inserted_jobs": 0, "inserted_known_bytes": 0, "inserted_unknown_size_jobs": 0,
               "access_requirement_held_new_jobs": 0, "already_present_jobs": 0, "held_releases": [],
               "existing_download_rows_updated_or_deleted": 0, "raw_files_touched": 0, "worker_start_stop_operations": 0,
               "deadline_or_budget_changes": 0, "training_approved": False, "byte_scope": plan["byte_scope"]}
    with db_path.with_suffix(".append.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with sqlite3.connect(str(db_path), timeout=30) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout=30000")
            db.execute("CREATE TABLE IF NOT EXISTS append_queue_imports (plan_id TEXT PRIMARY KEY, snapshot_sha256 TEXT, summary_json TEXT, applied_at TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS append_queue_sources (plan_id TEXT, dataset_key TEXT, catalog_id TEXT, scope_decision TEXT, source_record_json TEXT, PRIMARY KEY(plan_id,dataset_key,catalog_id))")
            db.execute("CREATE TABLE IF NOT EXISTS append_queue_holds (plan_id TEXT, dataset_key TEXT, reason TEXT, observed_at TEXT, PRIMARY KEY(plan_id,dataset_key))")
            db.commit()
            # Defense in depth: this importer cannot UPDATE/DELETE a download row.
            db.set_authorizer(lambda action, arg1, arg2, database, trigger:
                              sqlite3.SQLITE_DENY if arg1 == "downloads" and action in {sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE, sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_ALTER_TABLE}
                              else sqlite3.SQLITE_OK)
            for release in read_rows(plan_dir / "releases.jsonl"):
                if shutil.disk_usage(db_path.parent).free <= reserve_bytes + 64 * 1024**2:
                    summary["capacity_pause"] = True
                    break
                added, present, known, unknown, action_held = 0, 0, 0, 0, 0
                try:
                    db.execute("BEGIN IMMEDIATE")
                    for guard in release["existing_guards"]:
                        current = db.execute("SELECT * FROM downloads WHERE id=?", (guard["id"],)).fetchone()
                        if not current or metadata(current) != guard:
                            raise ValueError("existing_pinned_metadata_changed_since_plan")
                    hold_table = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='dataset_access_holds'").fetchone()
                    access_hold = db.execute("SELECT reason FROM dataset_access_holds WHERE dataset_key=?", (release["dataset_key"],)).fetchone() if hold_table else None
                    for job in release["new_jobs"]:
                        existing = db.execute("SELECT * FROM downloads WHERE id=? OR (dataset_key=? AND relative_path=?)", (job["id"], job["dataset_key"], job["relative_path"])).fetchone()
                        if existing:
                            if metadata(existing) != metadata(job):
                                raise ValueError("new_path_now_claimed_by_conflicting_pinned_metadata")
                            present += 1
                            continue
                        status = "access_action_required" if access_hold else job["status"]
                        error = access_hold["reason"] if access_hold else job["error"]
                        db.execute("""INSERT INTO downloads (id,catalog_id,dataset_key,relative_path,url,expected_bytes,checksum_json,
                          original_relative_path,provider_file_id,provider_file_identity,status,error,updated_at)
                          VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                   tuple(job[k] for k in ("id", "catalog_id", "dataset_key", "relative_path", "url", "expected_bytes", "checksum_json", "original_relative_path", "provider_file_id", "provider_file_identity"))
                                   + (status, error, now()))
                        added += 1
                        known += job["expected_bytes"] or 0
                        unknown += int(job["expected_bytes"] is None)
                        action_held += int(bool(access_hold))
                    source_record = {k: v for k, v in release.items() if k not in {"new_jobs", "existing_guards"}}
                    db.execute("INSERT OR IGNORE INTO append_queue_sources VALUES(?,?,?,?,?)", (plan["plan_id"], release["dataset_key"], release["catalog_id"], release["scope_decision"], json.dumps(source_record, ensure_ascii=False)))
                    db.commit()
                    summary["inserted_jobs"] += added
                    summary["already_present_jobs"] += present
                    summary["inserted_known_bytes"] += known
                    summary["inserted_unknown_size_jobs"] += unknown
                    summary["access_requirement_held_new_jobs"] += action_held
                except (sqlite3.Error, ValueError, KeyError) as exc:
                    db.rollback()
                    reason = type(exc).__name__ + ":" + str(exc)[:300]
                    summary["held_releases"].append({"dataset_key": release["dataset_key"], "catalog_id": release["catalog_id"], "reason": reason})
                    db.execute("INSERT OR REPLACE INTO append_queue_holds VALUES(?,?,?,?)", (plan["plan_id"], release["dataset_key"], reason, now()))
                    db.commit()
            summary["applied_at"] = now()
            db.execute("INSERT OR REPLACE INTO append_queue_imports VALUES(?,?,?,?)", (plan["plan_id"], plan["source_snapshot_sha256"], json.dumps(summary), summary["applied_at"]))
            db.commit()
    atomic_json(receipt_path, summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("plan")
    for name in ("source", "snapshots", "catalog", "db", "output"):
        prepare.add_argument("--" + name, required=True)
    prepare.add_argument("--reserve-gib", type=float, default=35)
    apply = sub.add_parser("apply")
    for name in ("plan-dir", "db", "expected-plan-sha256", "receipt"):
        apply.add_argument("--" + name, required=True)
    apply.add_argument("--reserve-gib", type=float, default=35)
    args = p.parse_args()
    if args.command == "plan":
        result = make_plan(args.source, args.snapshots, args.catalog, args.db, args.output, int(args.reserve_gib * 1024**3))
    else:
        result = apply_plan(args.plan_dir, args.db, args.expected_plan_sha256, args.receipt, int(args.reserve_gib * 1024**3))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
