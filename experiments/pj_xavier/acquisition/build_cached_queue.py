#!/usr/bin/env python3
"""Convert preserved official API bodies into a provenance-marked download queue.

No network requests. SHA-256 verification establishes the body is the same one
recorded in the audit. Public availability is historical until download succeeds.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from acquire import normalize_catalog, parse_zenodo, parse_figshare, parse_dataverse, write_jsonl


def build(catalog, live_records, cache_root):
    rows = normalize_catalog(catalog)
    evidence = {r["catalog_id"]: r for r in map(json.loads, Path(live_records).open())}
    parsers = {"zenodo": parse_zenodo, "figshare": parse_figshare, "dataverse": parse_dataverse}
    errors = []
    for row in rows:
        if row["status"] in {"excluded", "duplicate", "restricted"}:
            continue
        source = evidence.get(row["catalog_id"], {})
        parser = parsers.get(source.get("source_adapter"))
        http = source.get("http_evidence", {})
        if not parser or not http.get("evidence_file") or http.get("body_truncated") or http.get("status") != 200:
            continue
        path = Path(cache_root) / http["evidence_file"]
        if not path.exists():
            continue
        try:
            body = gzip.decompress(path.read_bytes())
            if hashlib.sha256(body).hexdigest() != http.get("body_sha256"):
                raise ValueError("retained HTTP body checksum mismatch")
            parsed = parser(json.loads(body), http["final_url"])
            row.update(parsed)
            row["resolved_at"] = http["retrieved_at"]
            row["resolution_basis"] = "retained_official_API_response_not_live_recheck"
            row["metadata_body_sha256"] = http["body_sha256"]
            row["release_file_bytes"] = sum(f["size_bytes"] for f in row["files"]) if all(f["size_bytes"] is not None for f in row["files"]) else None
            row["eeg_signal_bytes"] = None
            row["byte_scope"] = "whole_release_all_modalities_not_unique_EEG_payload"
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({"catalog_id": row["catalog_id"], "error": str(exc)})
    # Concepts and versions may appear under different catalog URL keys. Collapse
    # only exact observed provider/version, not partly overlapping derived corpora.
    seen = {}
    for row in rows:
        if row["status"] != "resolved":
            continue
        key = (row["provider"], str(row.get("source_version")), row["source_id"])
        if row["provider"] == "zenodo":
            key = ("zenodo", str(row.get("source_version")))
        if key in seen:
            row.update(status="duplicate", duplicate_of=seen[key], reason="same_observed_provider_release", files=[])
        else:
            seen[key] = row["catalog_id"]
    summary = {"catalog_rows": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
               "providers_resolved": dict(Counter(r["provider"] for r in rows if r["status"] == "resolved")),
               "file_entries": sum(len(r["files"]) for r in rows if r["status"] == "resolved"),
               "whole_release_bytes_not_deduplicated_across_corpora": sum(f["size_bytes"] or 0 for r in rows if r["status"] == "resolved" for f in r["files"]),
               "unique_EEG_hours": None, "errors": errors,
               "note": "Historical metadata queue only. No payload downloads performed. Includes unknown-scope candidates for acquisition, not approved training examples."}
    return rows, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", required=True)
    p.add_argument("--live-records", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", required=True)
    args = p.parse_args()
    rows, summary = build(args.catalog, args.live_records, args.cache_root)
    write_jsonl(args.output, rows)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
