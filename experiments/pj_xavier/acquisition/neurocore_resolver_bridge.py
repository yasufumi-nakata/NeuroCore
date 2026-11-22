#!/usr/bin/env python3
"""Reuse NeuroCore's maintained provider resolvers, retain restartable queue.

Run with NeuroCore's Python environment. This bridge performs metadata requests;
bulk bytes are transferred by acquire.py, never by the in-memory materializer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.parse

from acquire import make_file, now, read_jsonl, write_jsonl


def conversion(row, resolution):
    result = dict(row)
    result["resolver_attempted_at"] = now()
    result["resolver_status"] = resolution.status
    result["resolver_errors"] = list(resolution.errors)
    result["resolver_implementation"] = "neurocore.materialization.resolve_remote_files"
    files, rejected, seen = [], [], set()
    for remote in resolution.files:
        # The maintained resolver skips authenticated candidates. Actual payload
        # 401/403 stays restricted in the download ledger; no credentials injected.
        try:
            parsed = urllib.parse.urlsplit(remote.url)
            if parsed.scheme != "https":
                rejected.append({"name": remote.name, "reason": "non_HTTPS_not_automatically_transferred"})
                continue
            key = (remote.url, remote.name)
            if key in seen:
                continue
            seen.add(key)
            source_url = remote.source_url or row["source_url"]
            namespace = "source_" + hashlib.sha256(source_url.encode()).hexdigest()[:12]
            item = make_file(namespace + "/" + remote.name, remote.url, remote.size_bytes,
                             remote.checksum or None, evidence_url=source_url)
            item["original_provider_path"] = remote.name
            item["provider"] = remote.provider
            item["resolver_action"] = remote.materialization_action
            item["public_evidence_basis"] = "maintained_resolver_only_unauthenticated_candidates"
            files.append(item)
        except ValueError as e:
            rejected.append({"name": remote.name, "reason": str(e)})
    result["files"] = files
    result["file_rejections"] = rejected
    result["source_version"] = row.get("source_version") or "observed-unpinned"
    result["manifest_complete"] = False  # provider pagination limits can truncate
    result["manifest_completeness_reason"] = "legacy_resolver_does_not_report_exhaustive_pagination_proof"
    result["resolved_at"] = now()
    result["status"] = "resolved" if files else ("metadata_only" if resolution.status == "empty" else "unknown")
    result["reason"] = "maintained_provider_resolver_files" if files else "no_public_files_resolved; see resolver_errors"
    result["release_file_bytes"] = sum(f["size_bytes"] for f in files) if files and all(f["size_bytes"] is not None for f in files) else None
    result["byte_scope"] = "resolved_release_files_all_modalities; possibly_partial_or_mirrored; not_unique_EEG_bytes"
    result["eeg_signal_bytes"] = None
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--catalog-ids", nargs="*")
    p.add_argument("--timeout", type=float, default=20)
    p.add_argument("--max-pages", type=int, default=100)
    p.add_argument("--retry", action="store_true")
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--request-delay", type=float, default=1.0)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo / "backend"))
    from neurocore.datasets import load_eeg_dataset_inventory
    from neurocore.acquisition import plan_inventory_acquisition
    from neurocore.materialization import resolve_remote_files
    plans = {str(p.record_id).zfill(6): p for p in plan_inventory_acquisition(load_eeg_dataset_inventory(args.inventory))}
    rows = read_jsonl(args.queue)
    selected = set(args.catalog_ids or [])
    end, attempted = time.monotonic() + args.seconds, 0
    for i, row in enumerate(rows):
        if attempted >= args.limit or time.monotonic() >= end:
            break
        if selected and row["catalog_id"] not in selected:
            continue
        if row["status"] in {"excluded", "duplicate", "restricted", "resolved"}:
            continue
        if row.get("resolver_attempted_at") and not args.retry:
            continue
        plan = plans.get(row["catalog_id"])
        if plan is None:
            rows[i] = {**row, "reason": "record_missing_in_neurocore_inventory", "resolver_attempted_at": now()}
        else:
            resolution = resolve_remote_files(plan, timeout=args.timeout, max_pages=args.max_pages)
            rows[i] = conversion(row, resolution)
        attempted += 1
        write_jsonl(args.output, rows)
        print(json.dumps({"catalog_id": row["catalog_id"], "status": rows[i]["status"], "files": len(rows[i]["files"])}), flush=True)
        if args.request_delay:
            time.sleep(args.request_delay)
    write_jsonl(args.output, rows)
    print(json.dumps({"attempted_this_run": attempted, "records": len(rows), "statuses": dict(Counter(r["status"] for r in rows))}))


if __name__ == "__main__":
    main()
