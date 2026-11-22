#!/usr/bin/env python3
"""Hold only a Dataverse release whose guestbook requirement has live evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import urllib.parse

from acquire import atomic_json, dataset_storage_key, now


def hold(db_path, evidence_path, *, catalog_id="000037"):
    evidence_path = Path(evidence_path)
    body = evidence_path.read_bytes()
    evidence = json.loads(body)
    message = evidence.get("probe_response", {}).get("message", "")
    matched = re.fullmatch(r"You may not download (?:this file|these files) without the required Guestbook response for guestbookID (\d{1,12})\.?", message, re.I)
    source = urllib.parse.urlsplit(evidence["dataset_source_url"])
    if (evidence.get("probe_http_status") != 400 or evidence.get("probe_response", {}).get("status") != "ERROR"
            or not matched or int(matched.group(1)) != evidence.get("guestbook_id")
            or evidence.get("submission_performed") is not False or source.scheme != "https" or source.username or source.password
            or not source.path.startswith("/api/datasets/") or not evidence.get("published_file_ids")):
        raise ValueError("missing or inconsistent official guestbook evidence")
    dataset = dataset_storage_key({"provider": "dataverse", "source_id": evidence["dataset_pid"], "source_version": evidence["source_version"]})
    known_ids = {str(value) for value in evidence["published_file_ids"]}
    evidence_sha = hashlib.sha256(body).hexdigest()
    error = json.dumps({"code": "guestbook_required", "guestbook_id": evidence["guestbook_id"], "http_status": 400,
                        "evidence_sha256": evidence_sha, "submission_performed": False}, sort_keys=True)
    with sqlite3.connect(str(db_path), timeout=30) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN IMMEDIATE")
        rows = db.execute("SELECT * FROM downloads WHERE dataset_key=? AND catalog_id=?", (dataset, catalog_id)).fetchall()
        if not rows:
            raise ValueError("verified release is not present in this ledger")
        for row in rows:
            url = urllib.parse.urlsplit(row["url"])
            match = re.fullmatch(r"/api/access/datafile/(\d+)", url.path)
            if url.scheme != "https" or url.hostname != source.hostname or not match or match.group(1) not in known_ids:
                raise ValueError("ledger file is not in the officially verified release")
        preserved = {row["id"]: tuple(row) for row in rows if row["status"] not in {"retry", "queued"}}
        affected = [row["id"] for row in rows if row["status"] in {"retry", "queued"}]
        db.execute("""CREATE TABLE IF NOT EXISTS dataset_access_holds (
          dataset_key TEXT PRIMARY KEY, reason TEXT, evidence_sha256 TEXT, evidence_path TEXT, verified_at TEXT)""")
        db.execute("INSERT OR REPLACE INTO dataset_access_holds VALUES(?,?,?,?,?)", (dataset, error, evidence_sha, str(evidence_path), now()))
        count = db.execute("UPDATE downloads SET status='access_action_required',error=?,retry_at=0,lease_until=0,updated_at=? WHERE dataset_key=? AND catalog_id=? AND status IN ('retry','queued')",
                           (error, now(), dataset, catalog_id)).rowcount
        after = db.execute("SELECT * FROM downloads WHERE dataset_key=? AND catalog_id=?", (dataset, catalog_id)).fetchall()
        for row in after:
            if row["id"] in preserved and tuple(row) != preserved[row["id"]]:
                raise AssertionError("non-pending row changed; transaction will roll back")
        receipt = {"catalog_id": catalog_id, "dataset_key": dataset, "reason": "guestbook_required", "guestbook_id": evidence["guestbook_id"],
                   "official_release_file_count": len(known_ids), "ledger_release_file_count": len(rows), "rows_changed": count,
                   "affected_ids_sha256": hashlib.sha256("\n".join(sorted(affected)).encode()).hexdigest(),
                   "preserved_non_pending_rows": len(preserved), "evidence_sha256": evidence_sha, "evidence_path": str(evidence_path),
                   "verified_at": now(), "raw_files_touched": 0, "worker_start_or_stop_operations": 0,
                   "complete_and_running_unchanged": True, "submission_performed": False}
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--catalog-id", default="000037")
    p.add_argument("--receipt", required=True)
    args = p.parse_args()
    result = hold(args.db, args.evidence, catalog_id=args.catalog_id)
    atomic_json(args.receipt, result)
    print(json.dumps(result, ensure_ascii=False))
