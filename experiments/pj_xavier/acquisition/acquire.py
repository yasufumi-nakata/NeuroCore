#!/usr/bin/env python3
"""PJ-Xavier public EEG acquisition. Standard library only; never executes data.

Commands: normalize -> resolve -> download -> report. The JSONL catalog ledger
retains every catalog row. SQLite tracks bytes, retries, checksums and leases.
Public availability is not a decision about training or redistribution rights.
"""
from __future__ import annotations

import argparse
import csv
import copy
import datetime as dt
import hashlib
from http.client import IncompleteRead
import html.parser
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter

VERSION = "xavier-acquisition-v1"
USER_AGENT = "PJ-Xavier-public-EEG-acquisition/1.0 (bounded single-worker downloader)"
SIGNALS = {".edf", ".bdf", ".gdf", ".set", ".fdt", ".vhdr", ".eeg", ".cnt", ".mff", ".nwb", ".xdf"}
ARCHIVES = (".zip", ".tar.gz", ".tgz", ".tar", ".rar", ".7z")
RESTRICTED = {"restricted_by_source", "gated_by_source", "unauthenticated_access_restricted", "embargoed_by_source"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def safe_relative(name):
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise ValueError("invalid file path")
    p = PurePosixPath(name)
    if p.is_absolute() or any(x in {"..", ".", ""} for x in name.split("/")) or not p.parts:
        raise ValueError("unsafe relative path")
    return str(p)


def validate_url(url):
    p = urllib.parse.urlsplit(url)
    if p.scheme != "https" or not p.hostname or p.username or p.password:
        raise ValueError("only unauthenticated HTTPS URLs are permitted")
    return url


def checksum_spec(value):
    if not value:
        return None
    if isinstance(value, dict):
        algorithm = value.get("algorithm") or value.get("type")
        digest = value.get("value") or value.get("digest")
    elif ":" in value:
        algorithm, digest = value.split(":", 1)
    else:
        return {"algorithm": "unidentified", "value": str(value)}
    algorithm = str(algorithm).lower().replace("-", "")
    if algorithm in {"md5", "sha1", "sha256", "sha512"}:
        n = hashlib.new(algorithm).digest_size * 2
        if not re.fullmatch("[0-9a-fA-F]{%d}" % n, str(digest)):
            raise ValueError("invalid published checksum")
    return {"algorithm": algorithm, "value": str(digest).lower()}


def file_role(name):
    low = name.lower()
    if PurePosixPath(low).suffix in SIGNALS:
        return "signal_container_candidate"
    if low.endswith(ARCHIVES):
        return "eeg_archive_candidate" if "eeg" in low else "archive_contents_unverified"
    if any(t in low for t in ("screen", "video", "gaze", "eye", "behavior", "behaviour", "event", "stimuli", "stimulus")):
        return "paired_modality_or_event_candidate"
    return "metadata_or_other_release_file"


def make_file(name, url, size=None, checksum=None, *, evidence_url, restricted=False, provider_file_id=None):
    name = safe_relative(name)
    validate_url(url)
    if size is not None:
        size = int(size)
        if size < 0:
            raise ValueError("negative published size")
    return {"path": name, "url": url, "size_bytes": size,
            "published_checksum": checksum_spec(checksum), "role": file_role(name),
            "declared_public": not restricted, "manifest_evidence_url": evidence_url,
            "provider_file_id": str(provider_file_id) if provider_file_id is not None else None,
            "downloaded": False, "waveform_duration_seconds": None}


def provider_identity(row, file):
    """Stable remote-file identity, independent of the provider's display name."""
    provider = file.get("provider") or row.get("provider") or "unknown"
    url = validate_url(file["url"])
    parts = urllib.parse.urlsplit(url)
    fid = file.get("provider_file_id") or file.get("file_id")
    if fid is None and provider == "figshare" and (parts.hostname or "").endswith("figshare.com"):
        m = re.search(r"/(?:files|file/download)/(\d+)(?:/|$)", parts.path)
        fid = m.group(1) if m else None
    if fid is None and provider == "dataverse":
        m = re.search(r"/api/access/datafile/(\d+)(?:/|$)", parts.path)
        fid = m.group(1) if m else None
    canonical_url = urllib.parse.urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))
    scope = provider + (":" + (parts.hostname or "") if provider == "dataverse" else "")
    identity = scope + ":file:" + str(fid) if fid is not None else scope + ":url:" + canonical_url
    return {"provider": provider, "provider_file_id": str(fid) if fid is not None else None,
            "identity": identity, "canonical_url": canonical_url}


def migrate_manifest_records(records):
    """Preserve every entry; namespace same-name different remote files.

    Name collisions do not imply the provider changed one pinned file. A change
    to size/checksum for the same provider file identity remains quarantined.
    Ambiguous original directory/companion relationships are never inferred.
    """
    rows = copy.deepcopy(records)
    changes, issues, enriched = [], [], 0
    count_before = sum(len(r.get("files", [])) for r in records)
    for row in rows:
        identities, groups, row_issues = {}, {}, []
        for i, file in enumerate(row.get("files", [])):
            try:
                original = safe_relative(file.get("original_path") or file["path"])
                descriptor = provider_identity(row, file)
                file["original_path"] = original
                file["provider_file_id"] = descriptor["provider_file_id"]
                file["provider_file_identity"] = descriptor["identity"]
                file["original_url"] = file.get("original_url") or file["url"]
                enriched += 1
                identity = descriptor["identity"]
                signature = (file.get("size_bytes"), json.dumps(file.get("published_checksum"), sort_keys=True))
                previous = identities.get(identity)
                if previous and previous != signature:
                    row_issues.append({"issue": "same_pinned_file_identity_has_conflicting_size_or_checksum",
                                       "provider_file_identity": identity, "path": original,
                                       "first": previous, "next": signature})
                else:
                    identities[identity] = signature
                groups.setdefault(unicodedata.normalize("NFC", original).casefold(), []).append(i)
            except (KeyError, ValueError, TypeError) as e:
                row_issues.append({"issue": "invalid_file_metadata", "file_index": i, "error": str(e)})
        for indices in groups.values():
            # Equal names and equal provider identity are duplicate declarations
            # of one remote file. Their entries are retained in the manifest.
            variants = {(row["files"][i]["provider_file_identity"], row["files"][i]["original_path"]) for i in indices}
            if len(variants) <= 1:
                continue
            row["companion_resolution_required"] = True
            row["load_training_status"] = "hold_until_original_directory_and_companion_relationships_verified"
            for i in indices:
                file = row["files"][i]
                old_path = file["path"]
                identity = file["provider_file_identity"]
                token = re.sub(r"[^A-Za-z0-9._-]", "_", file.get("provider_file_id") or "url")[:64]
                digest = hashlib.sha256((identity + "\n" + file["original_path"]).encode()).hexdigest()[:16]
                prefix = f"provider_files/{row.get('provider', 'source')}_{token}_{digest}"
                file["path"] = safe_relative(prefix + "/" + file["original_path"])
                file["companion_resolution_required"] = True
                file["storage_namespace_reason"] = "same_display_path_different_remote_file_or_case_variant"
                if file["path"] != old_path:
                    changes.append({"catalog_id": row["catalog_id"], "source_version": row.get("source_version"),
                                    "original_path": file["original_path"], "old_path": old_path,
                                    "new_path": file["path"], "provider_file_id": file.get("provider_file_id"),
                                    "provider_file_identity": identity, "url": file["url"],
                                    "size_bytes": file.get("size_bytes"), "published_checksum": file.get("published_checksum")})
        if row_issues:
            row["acquisition_quarantine"] = row_issues
            row["pre_quarantine_status"] = row.get("pre_quarantine_status", row.get("status"))
            row["status"] = "quarantined"
            issues.append({"catalog_id": row["catalog_id"], "issues": row_issues})
        row["manifest_storage_schema"] = "provider_file_identity_v2"
    count_after = sum(len(r.get("files", [])) for r in rows)
    if count_after != count_before:
        raise AssertionError("manifest migration must preserve every file entry")
    return rows, {"migration_version": "provider_file_identity_v2", "catalog_rows": len(rows),
                  "file_entries_before": count_before, "file_entries_after": count_after,
                  "metadata_enriched_file_entries": enriched, "renamed_file_entries": len(changes),
                  "releases_with_companion_review": sum(bool(r.get("companion_resolution_required")) for r in rows),
                  "quarantined_releases": issues, "path_changes": changes,
                  "raw_files_moved_or_deleted": 0}


def normalize_catalog(path):
    """Accept the English audit CSV or Japanese catalog_audit_snapshot.csv."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out, seen = [], {}
    for row in rows:
        ev = row.get("live_evidence", "")
        ev = json.loads(ev) if ev else {}
        ident = str(row.get("catalog_id") or row.get("ID") or "").zfill(6)
        url = row.get("primary_url") or row.get("元の一次資料URL") or row.get("公開URL") or ""
        family = row.get("source_family_key") or row.get("出典系列キー") or url
        pool = row.get("pool_decision") or row.get("対象判定（暫定）") or "unknown"
        access = row.get("live_access_status") or row.get("取得条件の証拠") or "unknown"
        source_id = str(ev.get("source_id") or "")
        provider = ev.get("source_adapter") or ""
        if not provider or provider == "landing":
            text = family + " " + url
            match = re.search(r"(?:zenodo(?:_record)?[.:/]|zenodo-concept:)(\d+)", text)
            if not match:
                match = re.search(r"zenodo\.org/(?:records?/|doi/10\.5281/zenodo\.)(\d+)", text)
            if match:
                provider, source_id = "zenodo", match.group(1)
            elif "figshare_article:" in family:
                provider, source_id = "figshare", family.split(":", 1)[1]
            elif re.search(r"ds\d{6}", text) and "openneuro" in text:
                provider, source_id = "openneuro", re.search(r"ds\d{6}", text).group()
            elif "physionet.org" in url:
                provider = "physionet"
            elif family.startswith("osf:"):
                provider, source_id = "osf", family.split(":", 1)[1]
            else:
                provider = provider or "unsupported"
        excluded = pool.startswith("exclude_") or pool.startswith("対象外")
        status, reason = "unknown", "metadata_resolution_pending"
        if excluded:
            status, reason = "excluded", "catalog_scope_exclusion_requires_explicit_reclassification_to_reverse"
        elif access in RESTRICTED:
            status, reason = "restricted", access
        elif access == "metadata_only":
            status, reason = "metadata_only", "public_metadata_does_not_establish_waveform_access"
        key = family + "@" + str(ev.get("version") or row.get("source_release") or row.get("観測した版") or "")
        duplicate_of = seen.get(key) if family else None
        if duplicate_of and not excluded:
            status, reason = "duplicate", "same_provider_release_key_not_cross_corpus_recording_deduplication"
        else:
            seen[key] = ident
        out.append({"schema_version": VERSION, "catalog_id": ident,
                    "title": row.get("catalog_title") or row.get("データセット名"),
                    "provider": provider, "source_id": source_id, "source_url": url,
                    "verification_url": ev.get("verification_url") or row.get("今回の検証先URL"),
                    "source_family_key": family, "source_version": ev.get("version") or row.get("観測した版"),
                    "source_concept_key": ev.get("source_family_id"),
                    "scope_decision": pool, "access_evidence": access,
                    "status": status, "reason": reason, "duplicate_of": duplicate_of,
                    "license": ev.get("source_license") or row.get("ライセンス原記載"),
                    "training_use_status": "license_and_source_terms_review_pending",
                    "files": [], "resolved_at": None,
                    "source_audit_at": ev.get("audited_at") or row.get("照合日時（UTC）"),
                    "hours": None, "hours_status": "not_sum_of_catalog_claims"})
    return out


class Http:
    def __init__(self, timeout=30, delay=1.0, max_metadata_bytes=64 * 1024**2):
        self.timeout, self.delay = timeout, delay
        self.last = {}
        self.max_metadata_bytes = max_metadata_bytes

    def request(self, url, headers=None):
        validate_url(url)
        host = urllib.parse.urlsplit(url).hostname
        pause = self.delay - (time.monotonic() - self.last.get(host, 0))
        if pause > 0:
            time.sleep(pause)
        self.last[host] = time.monotonic()
        return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})}), timeout=self.timeout)

    def text(self, url):
        with self.request(url) as r:
            data = r.read(self.max_metadata_bytes + 1)
            if len(data) > self.max_metadata_bytes:
                raise ValueError("metadata exceeds bounded response size")
            return data.decode("utf-8")

    def json(self, url):
        return json.loads(self.text(url))


def parse_zenodo(data, evidence_url):
    metadata = data.get("metadata", {})
    rid = str(data["id"])
    access = metadata.get("access_right") or data.get("access", {}).get("files")
    if access not in {"open", "public"}:
        return {"status": "restricted", "reason": "source_does_not_declare_files_public", "files": [], "source_version": rid}
    raw = data.get("files", [])
    if isinstance(raw, dict):
        raw = list(raw.get("entries", {}).values())
    files = []
    for f in raw:
        name = f.get("key") or f.get("filename")
        url = f.get("links", {}).get("content") or f.get("links", {}).get("self")
        # This is the provider's documented record/files endpoint, derived only
        # from names and version IDs present in the official manifest.
        url = url or f"https://zenodo.org/records/{rid}/files/{urllib.parse.quote(name)}?download=1"
        files.append(make_file(name, url, f.get("size"), f.get("checksum"), evidence_url=evidence_url, provider_file_id=f.get("id")))
    return {"status": "resolved" if files else "metadata_only", "reason": "public_version_file_manifest",
            "source_id": rid, "source_version": rid, "source_concept_key": "zenodo-concept:" + str(data.get("conceptrecid", rid)),
            "license": metadata.get("license"), "files": files, "manifest_complete": True,
            "source_description": metadata.get("description"), "manifest_evidence_url": evidence_url}


def parse_figshare(data, evidence_url):
    if data.get("is_embargoed"):
        return {"status": "restricted", "reason": "source_embargo", "files": []}
    files = []
    skipped = 0
    for f in data.get("files", []):
        if f.get("is_link_only") or not f.get("download_url"):
            skipped += 1
            continue
        digest = f.get("computed_md5") or f.get("supplied_md5")
        files.append(make_file(f["name"], f["download_url"], f.get("size"),
                               "md5:" + digest if digest else None, evidence_url=evidence_url, provider_file_id=f.get("id")))
    return {"status": "resolved" if files else "metadata_only", "reason": "public_article_files_link_only_entries_not_downloaded",
            "source_version": data.get("version"), "license": data.get("license"), "files": files,
            "manifest_complete": skipped == 0, "link_only_entries": skipped, "manifest_evidence_url": evidence_url}


def parse_dataverse(data, evidence_url):
    version = data.get("data", {}).get("latestVersion", {})
    origin = urllib.parse.urlsplit(evidence_url)
    base = f"{origin.scheme}://{origin.netloc}"
    files, restricted = [], 0
    for entry in version.get("files", []):
        f = entry.get("dataFile", {})
        if entry.get("restricted"):
            restricted += 1
            continue
        name = (entry.get("directoryLabel", "").strip("/") + "/" + entry.get("label", f.get("filename", ""))).lstrip("/")
        files.append(make_file(name, f"{base}/api/access/datafile/{f['id']}", f.get("filesize"), f.get("checksum"), evidence_url=evidence_url, provider_file_id=f["id"]))
    return {"status": "resolved" if files else ("restricted" if restricted else "metadata_only"),
            "reason": "public_files_only_restricted_files_skipped", "files": files,
            "restricted_file_count": restricted, "manifest_complete": True,
            "source_version": f"{version.get('versionNumber')}.{version.get('versionMinorNumber')}",
            "license": version.get("license"), "source_terms": version.get("termsOfUse"),
            "manifest_evidence_url": evidence_url}


class LinkParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(v for k, v in attrs if k == "href")


def resolve_physionet(row, http, max_files=200000):
    m = re.search(r"physionet\.org/(?:content|files)/([^/]+)/([^/]+)", row["source_url"])
    if not m:
        raise ValueError("PhysioNet requires explicit observed project and version")
    project, version = m.groups()
    base = f"https://physionet.org/files/{project}/{version}/"
    pending, visited, files = [base], set(), []
    while pending:
        url = pending.pop(0)
        if url in visited:
            continue
        visited.add(url)
        if len(visited) > 20000:
            raise ValueError("directory traversal limit reached")
        parser = LinkParser()
        parser.feed(http.text(url))
        for href in parser.links:
            child = urllib.parse.urljoin(url, href)
            if not child.startswith(base) or "?" in child or "#" in child or child == base:
                continue
            if child.endswith("/"):
                if child not in visited:
                    pending.append(child)
            else:
                name = urllib.parse.unquote(child[len(base):])
                files.append(make_file(name, child, evidence_url=url))
                if len(files) > max_files:
                    raise ValueError("file manifest limit reached")
    files = list({f["url"]: f for f in files}.values())
    return {"status": "resolved" if files else "metadata_only", "reason": "public_unauthenticated_file_tree",
            "source_version": version, "files": files, "manifest_complete": True,
            "manifest_evidence_url": base, "size_status": "not_published_in_parser_manifest_sizes_verified_during_transfer"}


def resolve_openneuro(row, http, max_files=200000):
    ds = row["source_id"]
    if not re.fullmatch(r"ds\d{6}", ds):
        raise ValueError("invalid OpenNeuro dataset ID")
    description_url = f"https://s3.amazonaws.com/openneuro.org/{ds}/dataset_description.json"
    description = http.json(description_url)
    base = "https://s3.amazonaws.com/openneuro.org/"
    token, files, pages = None, [], 0
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        query = {"list-type": "2", "prefix": ds + "/", "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        url = base + "?" + urllib.parse.urlencode(query)
        root = ET.fromstring(http.text(url))
        pages += 1
        for obj in root.findall("s:Contents", ns):
            key = obj.findtext("s:Key", namespaces=ns)
            if not key.startswith(ds + "/") or key.endswith("/"):
                continue
            name = key[len(ds) + 1:]
            f = make_file(name, base + urllib.parse.quote(key), int(obj.findtext("s:Size", namespaces=ns)), evidence_url=url)
            f["source_etag"] = obj.findtext("s:ETag", namespaces=ns)
            # S3 ETag is not uniformly a content MD5 (multipart and encryption).
            files.append(f)
            if len(files) > max_files:
                raise ValueError("OpenNeuro manifest file limit reached")
        if root.findtext("s:IsTruncated", namespaces=ns) != "true":
            break
        token = root.findtext("s:NextContinuationToken", namespaces=ns)
        if not token or pages > 10000:
            raise ValueError("invalid S3 pagination")
    return {"status": "resolved" if files else "metadata_only", "reason": "public_S3_snapshot_version_not_immutable",
            "files": files, "license": description.get("License"), "manifest_complete": True,
            "source_version": "current-public-S3", "source_reported_doi": description.get("DatasetDOI"),
            "manifest_evidence_url": description_url, "version_pin_status": "hash_every_download_before_freezing_training_manifest"}


def resolve_osf(row, http, max_files=200000):
    ident = row["source_id"]
    if not re.fullmatch(r"[a-z0-9]{5,}", ident):
        raise ValueError("invalid OSF node ID")
    base = f"https://api.osf.io/v2/nodes/{ident}/"
    node = http.json(base)
    if not node.get("data", {}).get("attributes", {}).get("public"):
        return {"status": "restricted", "reason": "OSF_node_not_public", "files": []}
    pending, seen, files = [base + "files/"], set(), []
    while pending:
        url = pending.pop(0)
        if url in seen:
            continue
        if urllib.parse.urlsplit(url).hostname != "api.osf.io":
            raise ValueError("OSF metadata link leaves API origin")
        seen.add(url)
        data = http.json(url)
        for item in data.get("data", []):
            a, links = item.get("attributes", {}), item.get("links", {})
            if a.get("kind") == "file":
                if not links.get("download"):
                    continue
                path = a.get("materialized_path", a.get("name", "")).lstrip("/")
                files.append(make_file(path, links["download"], a.get("size"), evidence_url=url, provider_file_id=item.get("id")))
                if len(files) > max_files:
                    raise ValueError("OSF manifest file limit reached")
            else:
                child = item.get("relationships", {}).get("files", {}).get("links", {}).get("related", {})
                if isinstance(child, str):
                    pending.append(child)
                elif child.get("href"):
                    pending.append(child["href"])
        if data.get("links", {}).get("next"):
            pending.append(data["links"]["next"])
        if len(seen) > 20000:
            raise ValueError("OSF traversal limit reached")
    return {"status": "resolved" if files else "metadata_only", "reason": "OSF_public_file_tree",
            "files": files, "manifest_complete": True, "manifest_evidence_url": base,
            "source_version": "public-tree-observed-" + now(), "license": row.get("license")}


def resolve_record(row, http):
    if row["status"] in {"excluded", "duplicate", "restricted"}:
        return row
    result = dict(row)
    result["resolver_attempts"] = int(row.get("resolver_attempts", 0)) + 1
    try:
        provider, ident = row["provider"], row["source_id"]
        if provider == "zenodo":
            url = f"https://zenodo.org/api/records/{ident}"
            update = parse_zenodo(http.json(url), url)
        elif provider == "figshare":
            # Explicit versions are honored when numeric. Resolving a concept
            # without version pins the actual returned version in this ledger.
            url = f"https://api.figshare.com/v2/articles/{ident}"
            version = row.get("source_version")
            if str(version or "").isdigit():
                url += f"/versions/{version}"
            update = parse_figshare(http.json(url), url)
        elif provider == "dataverse":
            url = row.get("verification_url")
            if not url:
                p = urllib.parse.urlsplit(row["source_url"])
                query = urllib.parse.parse_qs(p.query)
                pid = query.get("persistentId", [None])[0]
                if not pid:
                    raise ValueError("Dataverse persistentId API resolution pending")
                url = f"{p.scheme}://{p.netloc}/api/datasets/:persistentId/?" + urllib.parse.urlencode({"persistentId": pid})
            update = parse_dataverse(http.json(url), url)
        elif provider == "physionet":
            update = resolve_physionet(row, http)
        elif provider == "openneuro":
            update = resolve_openneuro(row, http)
        elif provider == "osf":
            update = resolve_osf(row, http)
        else:
            return {**result, "reason": "provider_adapter_not_implemented; maintain_queue_for_follow_up",
                    "resolver_attempted_at": now(), "resolver_retry_at": time.time() + 86400}
        result.update(update)
        result["resolved_at"] = now()
        result["resolver_attempted_at"] = result["resolved_at"]
        result["resolver_retry_at"] = 0
        result["release_file_bytes"] = sum(f["size_bytes"] for f in result["files"]) if all(f["size_bytes"] is not None for f in result["files"]) else None
        result["eeg_signal_bytes"] = None
        result["byte_scope"] = "whole_release_all_modalities_not_unique_EEG_payload"
    except urllib.error.HTTPError as e:
        pause = min(86400, 60 * 2 ** min(result["resolver_attempts"], 10))
        retry = e.headers.get("Retry-After", "") if e.headers else ""
        if retry.isdigit():
            pause = max(pause, int(retry))
        result.update(status="restricted" if e.code in {401, 403} else "unknown", reason=f"HTTP_{e.code}",
                      resolved_at=now(), resolver_attempted_at=now(), resolver_retry_at=time.time() + pause)
    except (ValueError, OSError, TimeoutError, KeyError, ET.ParseError) as e:
        result.update(status="unknown", reason=f"resolution_failed:{type(e).__name__}:{str(e)[:300]}", resolved_at=now(),
                      resolver_attempted_at=now(), resolver_retry_at=time.time() + 3600)
    return result


def init_db(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS downloads (
        id TEXT PRIMARY KEY, catalog_id TEXT NOT NULL, dataset_key TEXT NOT NULL,
        relative_path TEXT NOT NULL, url TEXT NOT NULL, expected_bytes INTEGER,
        checksum_json TEXT, status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
        retry_at REAL NOT NULL DEFAULT 0, lease_until REAL NOT NULL DEFAULT 0,
        bytes_present INTEGER NOT NULL DEFAULT 0, sha256 TEXT, error TEXT, updated_at TEXT,
        UNIQUE(dataset_key, relative_path))""")
    present = {r[1] for r in db.execute("PRAGMA table_info(downloads)")}
    for column in ("original_relative_path", "provider_file_id", "provider_file_identity"):
        if column not in present:
            db.execute("ALTER TABLE downloads ADD COLUMN " + column + " TEXT")
    db.execute("""CREATE TABLE IF NOT EXISTS quarantined_releases (
        dataset_key TEXT PRIMARY KEY, catalog_id TEXT, retained_file_entries INTEGER,
        reasons_json TEXT, updated_at TEXT)""")
    db.execute("""CREATE TABLE IF NOT EXISTS preserved_manifest_aliases (
        old_id TEXT, new_id TEXT, original_path TEXT, new_path TEXT,
        previous_status TEXT, reason TEXT, updated_at TEXT, PRIMARY KEY(old_id,new_id))""")
    db.commit()
    return db


def dataset_storage_key(row):
    dataset = f"{row['provider']}-{row['source_id']}-{row.get('source_version', 'observed')}"
    dataset = re.sub(r"[^A-Za-z0-9._-]", "_", dataset)
    if str(row.get("source_version", "")) in {"current-public-S3", "observed-unpinned"}:
        manifest_identity = sorted((f["path"], f.get("size_bytes"), f.get("source_etag"),
                                    json.dumps(f.get("published_checksum"), sort_keys=True)) for f in row["files"])
        dataset += "-" + hashlib.sha256(json.dumps(manifest_identity).encode()).hexdigest()[:12]
    return dataset


def _quarantine_release(db, row, dataset, reasons):
    db.execute("INSERT OR REPLACE INTO quarantined_releases VALUES(?,?,?,?,?)",
               (dataset, row["catalog_id"], len(row.get("files", [])), json.dumps(reasons, ensure_ascii=False), now()))
    # Existing completed bytes are retained, and a running transfer is not
    # interrupted by a manifest import. New/paused jobs are held for review.
    db.execute("UPDATE downloads SET status='manifest_quarantined',error=?,updated_at=? WHERE dataset_key=? AND status NOT IN ('complete','running','legacy_preserved','superseded_manifest_path')",
               (json.dumps(reasons, ensure_ascii=False)[:1200], now(), dataset))


def ingest(db, records, *, quarantine_conflicts=True):
    """Import releases independently; preserve distinct provider file IDs.

    Default is per-release quarantine, so one corrupt release cannot roll back
    the entire corpus queue. Strict mode is available for unit validation.
    """
    records, _migration = migrate_manifest_records(records)
    count = 0
    for row in records:
        if row.get("acquisition_quarantine"):
            dataset = dataset_storage_key(row)
            if not quarantine_conflicts:
                raise ValueError("manifest conflict: " + json.dumps(row["acquisition_quarantine"]))
            _quarantine_release(db, row, dataset, row["acquisition_quarantine"])
            db.commit()
            continue
        if row["status"] != "resolved":
            continue
        dataset, added = dataset_storage_key(row), 0
        db.execute("SAVEPOINT import_release")
        try:
            for f in row["files"]:
                if not f.get("declared_public") or not f.get("manifest_evidence_url"):
                    continue
                name = safe_relative(f["path"])
                validate_url(f["url"])
                ident = hashlib.sha256((dataset + "/" + name).encode()).hexdigest()
                if f.get("original_path", name) != name:
                    legacy_id = hashlib.sha256((dataset + "/" + f["original_path"]).encode()).hexdigest()
                    legacy = db.execute("SELECT url,status FROM downloads WHERE id=?", (legacy_id,)).fetchone()
                    if legacy and legacy[0] == f["url"]:
                        if legacy[1] == "running":
                            raise ValueError("manifest migration waits for existing legacy-path transfer to stop")
                        db.execute("INSERT OR IGNORE INTO preserved_manifest_aliases VALUES(?,?,?,?,?,?,?)",
                                   (legacy_id, ident, f["original_path"], name, legacy[1], "namespace_collision_migration_no_raw_files_deleted", now()))
                        db.execute("UPDATE downloads SET status=?,updated_at=? WHERE id=?",
                                   ("legacy_preserved" if legacy[1] in {"complete", "legacy_preserved"} else "superseded_manifest_path", now(), legacy_id))
                old = db.execute("SELECT expected_bytes,checksum_json,url,provider_file_identity FROM downloads WHERE id=?", (ident,)).fetchone()
                if old:
                    old_size, old_sum, old_url, old_identity = old
                    if old_size is not None and f.get("size_bytes") is not None and old_size != f["size_bytes"]:
                        raise ValueError("manifest conflict: published size changed for an existing pinned file")
                    if json.loads(old_sum) and f.get("published_checksum") and json.loads(old_sum) != f["published_checksum"]:
                        raise ValueError("manifest conflict: checksum changed for an existing pinned file")
                    if old_identity and old_identity != f.get("provider_file_identity") and old_url != f["url"]:
                        raise ValueError("manifest conflict: a different provider file now claims an existing pinned storage path")
                size = f.get("size_bytes")
                size_known = isinstance(size, int) and not isinstance(size, bool) and size >= 0
                status = "queued" if size_known else "metadata_blocked"
                reason = None if size_known else "published_file_size_unverified; measure_before_bulk_transfer"
                values = (ident, row["catalog_id"], dataset, name, f["url"], size if size_known else None,
                          json.dumps(f.get("published_checksum")), now(), f.get("original_path", name),
                          f.get("provider_file_id"), f.get("provider_file_identity"), status, reason)
                cursor = db.execute("""INSERT OR IGNORE INTO downloads
                   (id,catalog_id,dataset_key,relative_path,url,expected_bytes,checksum_json,updated_at,
                    original_relative_path,provider_file_id,provider_file_identity,status,error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)
                added += cursor.rowcount
                if old:
                    db.execute("UPDATE downloads SET original_relative_path=?,provider_file_id=?,provider_file_identity=? WHERE id=?",
                               (f.get("original_path", name), f.get("provider_file_id"), f.get("provider_file_identity"), ident))
                    if old[0] is None and size_known:
                        db.execute("UPDATE downloads SET expected_bytes=?,status=CASE WHEN status='metadata_blocked' THEN 'queued' ELSE status END,error=NULL WHERE id=?", (size, ident))
            db.execute("DELETE FROM quarantined_releases WHERE dataset_key=?", (dataset,))
            db.execute("RELEASE SAVEPOINT import_release")
            db.commit()
            count += added
        except (ValueError, KeyError, TypeError, sqlite3.IntegrityError) as e:
            db.execute("ROLLBACK TO SAVEPOINT import_release")
            db.execute("RELEASE SAVEPOINT import_release")
            if not quarantine_conflicts:
                raise
            _quarantine_release(db, row, dataset, [{"issue": "ingest_release_failure", "error": str(e)}])
            db.commit()
    return count


def hash_file(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024**2):
            sha.update(block)
    return sha.hexdigest()


def verify_file(path, expected_size, checksum):
    if expected_size is not None and Path(path).stat().st_size != expected_size:
        raise ValueError("download size does not match published manifest")
    sha, published = hashlib.sha256(), None
    if checksum and checksum["algorithm"] in {"md5", "sha1", "sha256", "sha512"}:
        published = hashlib.new(checksum["algorithm"])
    with Path(path).open("rb") as f:
        while block := f.read(1024**2):
            sha.update(block)
            if published:
                published.update(block)
    if published and published.hexdigest() != checksum["value"]:
        raise ValueError("published checksum mismatch")
    return sha.hexdigest()


def destination(root, dataset, name):
    base = Path(root).resolve()
    path = base / safe_relative(dataset) / safe_relative(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.resolve().is_relative_to(base):
        raise ValueError("destination escapes data root")
    return path


def transfer(job, root, http, reserve_bytes, deadline=None, transfer_limit=None):
    path = destination(root, job["dataset_key"], job["relative_path"])
    part = path.with_name(path.name + ".part")
    partial_metadata = path.with_name(path.name + ".part.meta.json")
    if part.is_symlink() or partial_metadata.is_symlink():
        raise ValueError("partial file must not be a symlink")
    checksum = json.loads(job["checksum_json"])
    expected = job["expected_bytes"]
    if path.exists():
        return {"status": "complete", "bytes_present": path.stat().st_size,
                "sha256": verify_file(path, expected, checksum), "transferred_bytes": 0}
    offset = part.stat().st_size if part.exists() else 0
    validator = None
    validator_field = None
    meta = {}
    if offset and partial_metadata.exists():
        meta = json.loads(partial_metadata.read_text())
        if meta.get("url_sha256") != hashlib.sha256(job["url"].encode()).hexdigest():
            raise ValueError("partial file belongs to a different URL")
        validator = meta.get("etag")
        validator_field = "ETag"
        if not validator or validator.startswith("W/"):
            validator = meta.get("last_modified")
            validator_field = "Last-Modified"
    if expected is not None and offset == expected:
        sha = verify_file(part, expected, checksum)
        os.replace(part, path)
        partial_metadata.unlink(missing_ok=True)
        return {"status": "complete", "bytes_present": expected, "sha256": sha, "transferred_bytes": 0}
    if expected is not None and offset > expected:
        raise ValueError("partial file larger than manifest; quarantine before retry")
    remaining = (expected - offset) if expected is not None else 64 * 1024**2
    if shutil.disk_usage(path.parent).free < reserve_bytes + min(remaining, 64 * 1024**2):
        return {"status": "capacity_blocked", "bytes_present": offset, "transferred_bytes": 0}
    headers = {"Accept-Encoding": "identity"}
    if offset and not validator and not (checksum and checksum["algorithm"] in {"md5", "sha1", "sha256", "sha512"}):
        # Do not join bytes from two remote generations when identity is unknown.
        offset = 0
    if offset:
        headers["Range"] = f"bytes={offset}-"
        if validator:
            headers["If-Range"] = validator
    transferred = 0
    with http.request(job["url"], headers) as response:
        status = response.status
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
            if not m or int(m[1]) != offset:
                raise ValueError("resumed Content-Range start mismatch")
            if offset and validator:
                actual_validator = response.headers.get(validator_field)
                if actual_validator is not None and actual_validator != validator:
                    raise ValueError("resumed response source validator changed; refusing mixed-version bytes")
                has_checksum = checksum and checksum["algorithm"] in {"md5", "sha1", "sha256", "sha512"}
                if actual_validator is None and not has_checksum:
                    raise ValueError("resumed response missing source validator without published checksum")
                for field, old_value in (("ETag", meta.get("etag")), ("Last-Modified", meta.get("last_modified"))):
                    value = response.headers.get(field)
                    if old_value and value and old_value != value:
                        raise ValueError("resumed response source validator changed; refusing mixed-version bytes")
            if m[3] != "*":
                total = int(m[3])
                if expected is not None and total != expected:
                    raise ValueError("remote file changed since published manifest")
                expected = expected if expected is not None else total
            mode = "ab" if offset else "wb"
        elif status == 200:
            # Server ignored Range: restart, never append a second whole file.
            mode, offset = "wb", 0
            declared = response.headers.get("Content-Length")
            if declared and expected is not None and int(declared) != expected:
                raise ValueError("response size differs from manifest")
            if declared and expected is None:
                expected = int(declared)
        else:
            raise ValueError(f"unexpected transfer HTTP status {status}")
        # Login pages can return 200. For non-HTML payloads reject this before save.
        ctype = response.headers.get("Content-Type", "").lower()
        if "text/html" in ctype and path.suffix.lower() not in {".html", ".htm"}:
            raise ValueError("HTML response where data file expected")
        atomic_json(partial_metadata, {"url_sha256": hashlib.sha256(job["url"].encode()).hexdigest(),
                    "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified"),
                    "expected_bytes": expected, "started_at": now()})
        with part.open(mode) as f:
            while True:
                if deadline and time.monotonic() >= deadline:
                    return {"status": "paused", "bytes_present": f.tell(), "transferred_bytes": transferred}
                if transfer_limit is not None and transferred >= transfer_limit:
                    return {"status": "paused", "bytes_present": f.tell(), "transferred_bytes": transferred}
                chunk_size = min(1024**2, transfer_limit - transferred) if transfer_limit is not None else 1024**2
                if shutil.disk_usage(path.parent).free < reserve_bytes + chunk_size:
                    return {"status": "capacity_blocked", "bytes_present": f.tell(), "transferred_bytes": transferred}
                try:
                    block = response.read(chunk_size)
                except (OSError, urllib.error.URLError, IncompleteRead) as exc:
                    partial = getattr(exc, "partial", b"")
                    if isinstance(partial, bytes) and partial:
                        if len(partial) > chunk_size or (expected is not None and f.tell() + len(partial) > expected):
                            return {"status": "integrity_failed", "bytes_present": f.tell(),
                                    "transferred_bytes": transferred + len(partial), "error": "interrupted_response_exceeds_declared_size_or_transfer_limit"}
                        f.write(partial)
                        transferred += len(partial)
                    return {"status": "retry", "bytes_present": f.tell(), "transferred_bytes": transferred,
                            "retry_after_seconds": 60, "error": "interrupted_transfer:" + type(exc).__name__}
                if not block:
                    break
                if expected is not None and f.tell() + len(block) > expected:
                    return {"status": "integrity_failed", "bytes_present": f.tell(),
                            "transferred_bytes": transferred + len(block), "error": "server_sent_more_bytes_than_declared"}
                f.write(block)
                transferred += len(block)
    size = part.stat().st_size
    if expected is not None and size < expected:
        # A provider/proxy may terminate a large HTTP response cleanly before its
        # declared total. Keep the validated partial prefix and resume it later.
        return {"status": "retry", "bytes_present": size, "transferred_bytes": transferred,
                "retry_after_seconds": 60, "error": f"incomplete_transfer_eof:{size}/{expected}"}
    try:
        sha = verify_file(part, expected, checksum)
    except ValueError as e:
        return {"status": "integrity_failed", "bytes_present": size, "transferred_bytes": transferred,
                "error": str(e)}
    os.replace(part, path)
    partial_metadata.unlink(missing_ok=True)
    return {"status": "complete", "bytes_present": size, "sha256": sha, "transferred_bytes": transferred}


def access_requirement_from_http_error(error):
    """Recognize an explicit access workflow; never store request tokens/body text."""
    if not isinstance(error, urllib.error.HTTPError) or error.code != 400:
        return None
    try:
        path = urllib.parse.urlsplit(error.url).path
        if not path.startswith(("/api/access/datafile/", "/api/access/datafiles/", "/api/access/dataset/")):
            return None
        body = error.read(8193)
        if len(body) > 8192:
            return None
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict) or payload.get("status") != "ERROR":
            return None
        match = re.fullmatch(r"You may not download (?:this file|these files) without the required Guestbook response for guestbookID (\d{1,12})\.?",
                             str(payload.get("message", "")).strip(), re.IGNORECASE)
        if not match:
            return None
        return {"code": "guestbook_required", "guestbook_id": int(match.group(1)), "http_status": 400,
                "response_body_sha256": hashlib.sha256(body).hexdigest(), "submission_performed": False}
    except (OSError, ValueError, UnicodeDecodeError, TypeError):
        return None


def worker(db, root, *, reserve_gib=20, seconds=300, max_bytes=None, max_files=None, http=None):
    """One worker process per node; fixed 1 active transfer keeps host fanout bounded.

    SQLite leases prevent concurrent workers sharing a filesystem from claiming
    the same job. Every retry persists next-attempt time; no busy polling.
    """
    http = http or Http()
    db.row_factory = sqlite3.Row
    deadline = time.monotonic() + seconds
    transferred, processed = 0, 0
    Path(root).mkdir(parents=True, exist_ok=True)
    while time.monotonic() < deadline and (max_files is None or processed < max_files):
        if max_bytes is not None and transferred >= max_bytes:
            break
        db.execute("BEGIN IMMEDIATE")
        timestamp = time.time()
        row = db.execute("""SELECT * FROM downloads WHERE
            (status IN ('queued','paused','retry','capacity_blocked') OR
            (status='running' AND lease_until < ?)) AND retry_at <= ?
            ORDER BY attempts, catalog_id, expected_bytes IS NULL, expected_bytes, relative_path LIMIT 1""", (timestamp, timestamp)).fetchone()
        if not row:
            db.commit()
            break
        db.execute("UPDATE downloads SET status='running', lease_until=?, attempts=attempts+1, updated_at=? WHERE id=?",
                   (timestamp + max(600, seconds + 120), now(), row["id"]))
        db.commit()
        job = dict(row)
        try:
            result = transfer(job, root, http, int(reserve_gib * 1024**3), deadline,
                              None if max_bytes is None else max_bytes - transferred)
            retry_at = time.time() + 300 if result["status"] == "capacity_blocked" else 0
            if result["status"] == "retry":
                retry_at = time.time() + max(result.get("retry_after_seconds", 60), min(3600, 60 * 2 ** min(job["attempts"], 6)))
            db.execute("UPDATE downloads SET status=?,bytes_present=?,sha256=?,error=?,retry_at=?,lease_until=0,updated_at=? WHERE id=?",
                       (result["status"], result["bytes_present"], result.get("sha256"), result.get("error"), retry_at, now(), job["id"]))
            transferred += result["transferred_bytes"]
            processed += 1
            db.commit()
            if result["status"] == "capacity_blocked":
                break
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError, socket.timeout) as e:
            code = getattr(e, "code", None)
            requirement = access_requirement_from_http_error(e)
            status = "access_action_required" if requirement else ("restricted" if code in {401, 403} else ("integrity_failed" if isinstance(e, ValueError) else "retry"))
            pause = min(86400, 60 * 2 ** min(job["attempts"], 10))
            if code == 429 and e.headers.get("Retry-After", "").isdigit():
                pause = max(pause, int(e.headers["Retry-After"]))
            partial = destination(root, job["dataset_key"], job["relative_path"])
            partial = partial.with_name(partial.name + ".part")
            actual_bytes = partial.stat().st_size if partial.exists() else 0
            error_text = json.dumps(requirement, sort_keys=True) if requirement else (
                "HTTPError:status=" + str(code) if isinstance(e, urllib.error.HTTPError) else f"{type(e).__name__}:{str(e)[:400]}")
            db.execute("UPDATE downloads SET status=?,error=?,retry_at=?,lease_until=0,bytes_present=?,updated_at=? WHERE id=?",
                       (status, error_text, 0 if requirement else time.time() + pause, actual_bytes, now(), job["id"]))
            db.commit()
            processed += 1
            if code == 429:
                break
    return {"files_processed": processed, "bytes_transferred_this_run": transferred, "report": db_report(db)}


def db_report(db):
    counts = {r[0]: {"files": r[1], "bytes_present": r[2], "expected_bytes_known": r[3], "unknown_size_files": r[4]}
              for r in db.execute("SELECT status,count(*),sum(bytes_present),sum(coalesce(expected_bytes,0)),sum(expected_bytes IS NULL) FROM downloads GROUP BY status")}
    quarantined = db.execute("SELECT count(*) FROM quarantined_releases").fetchone()[0]
    aliases = db.execute("SELECT count(*) FROM preserved_manifest_aliases").fetchone()[0]
    active = {k: v for k, v in counts.items() if k not in {"legacy_preserved", "superseded_manifest_path"}}
    return {"updated_at": now(), "statuses": counts, "quarantined_release_count": quarantined,
            "preserved_legacy_alias_count": aliases, "all_manifest_jobs_complete": bool(active) and set(active) == {"complete"} and quarantined == 0,
            "byte_scope": "downloaded_whole_files_all_modalities_including_metadata_not_unique_EEG_hours",
            "verified_waveform_hours": None, "hours_unavailable_reason": "header/channel/source-recording dedup audit required after download"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    n = sub.add_parser("normalize")
    n.add_argument("--catalog", required=True)
    n.add_argument("--output", required=True)
    migration = sub.add_parser("migrate-manifest")
    migration.add_argument("--input", required=True)
    migration.add_argument("--output", required=True)
    migration.add_argument("--ledger", required=True)
    z = sub.add_parser("seed")
    z.add_argument("--zenodo", action="append", default=[], metavar="CATALOG_ID=RECORD_ID", help="Fetch the official public API for a pinned version")
    z.add_argument("--zenodo-json", action="append", default=[], metavar="CATALOG_ID=JSON_PATH", help="Read an already retrieved official Zenodo API body")
    z.add_argument("--output", required=True)
    r = sub.add_parser("resolve")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--catalog-ids", nargs="*")
    r.add_argument("--limit", type=int, default=25)
    r.add_argument("--delay", type=float, default=1.0)
    d = sub.add_parser("download")
    d.add_argument("--manifest", required=True)
    d.add_argument("--db", required=True)
    d.add_argument("--root", required=True)
    d.add_argument("--reserve-gib", type=float, default=20)
    d.add_argument("--seconds", type=int, default=300)
    d.add_argument("--max-bytes", type=int)
    d.add_argument("--max-files", type=int)
    d.add_argument("--catalog-ids", nargs="*")
    s = sub.add_parser("report")
    s.add_argument("--db", required=True)
    args = p.parse_args(argv)
    if args.command == "normalize":
        rows = normalize_catalog(args.catalog)
        write_jsonl(args.output, rows)
        print(json.dumps({"rows": len(rows), "statuses": dict(Counter(r["status"] for r in rows))}))
    elif args.command == "migrate-manifest":
        if Path(args.input).resolve() == Path(args.output).resolve():
            raise SystemExit("migration output must differ from input; preserve the original manifest")
        original_hash = hash_file(args.input)
        rows, ledger = migrate_manifest_records(read_jsonl(args.input))
        write_jsonl(args.output, rows)
        ledger.update(input_path=str(Path(args.input).resolve()), output_path=str(Path(args.output).resolve()),
                      input_sha256=original_hash, output_sha256=hash_file(args.output), migrated_at=now())
        atomic_json(args.ledger, ledger)
        print(json.dumps({k: v for k, v in ledger.items() if k not in {"path_changes", "quarantined_releases"}}, ensure_ascii=False))
    elif args.command == "seed":
        rows = []
        http = Http()
        sources = []
        for value in args.zenodo:
            ident, rid = value.split("=", 1)
            if not rid.isdigit():
                raise SystemExit("Zenodo record ID must be numeric")
            url = f"https://zenodo.org/api/records/{rid}"
            sources.append((ident.zfill(6), http.json(url), url, "live_official_API_response"))
        for value in args.zenodo_json:
            ident, path = value.split("=", 1)
            metadata = json.loads(Path(path).read_text())
            sources.append((ident.zfill(6), metadata, f"https://zenodo.org/api/records/{metadata['id']}", "caller_supplied_official_API_body"))
        for ident, metadata, url, basis in sources:
            row = {"schema_version": VERSION, "catalog_id": ident, "provider": "zenodo", "source_id": str(metadata["id"]),
                   "source_url": f"https://zenodo.org/records/{metadata['id']}", "title": metadata.get("metadata", {}).get("title"),
                   "scope_decision": "explicitly_selected_EEG_release_candidate", "training_use_status": "license_and_source_terms_review_pending",
                   "resolved_at": now(), "resolution_basis": basis, "hours": None, "eeg_signal_bytes": None,
                   "byte_scope": "whole_release_all_modalities_not_unique_EEG_payload"}
            row.update(parse_zenodo(metadata, url))
            row["release_file_bytes"] = sum(f["size_bytes"] for f in row["files"]) if all(f["size_bytes"] is not None for f in row["files"]) else None
            rows.append(row)
        write_jsonl(args.output, rows)
        print(json.dumps({"rows": len(rows), "statuses": dict(Counter(r["status"] for r in rows)), "files": sum(len(r["files"]) for r in rows)}))
    elif args.command == "resolve":
        rows, count = read_jsonl(args.input), 0
        http = Http(delay=args.delay)
        selected = set(args.catalog_ids or [])
        # New unresolved records take priority over repeated unsupported/error
        # rows, so repeated continuation runs can make progress through all rows.
        order = sorted(range(len(rows)), key=lambda i: (bool(rows[i].get("resolver_attempted_at")), rows[i].get("resolver_attempts", 0)))
        for i in order:
            row = rows[i]
            if selected and row["catalog_id"] not in selected:
                continue
            if row["status"] in {"excluded", "duplicate", "restricted", "resolved"}:
                continue
            if row.get("resolver_retry_at", 0) > time.time():
                continue
            if count >= args.limit:
                break
            rows[i] = resolve_record(row, http)
            count += 1
            write_jsonl(args.output, rows)
        write_jsonl(args.output, rows)
        print(json.dumps({"attempted": count, "rows": len(rows), "statuses": dict(Counter(r["status"] for r in rows))}))
    elif args.command == "download":
        if args.reserve_gib < 1 or args.seconds < 1:
            raise SystemExit("reserve must be >=1 GiB and seconds must be positive")
        rows = read_jsonl(args.manifest)
        if args.catalog_ids:
            rows = [r for r in rows if r["catalog_id"] in args.catalog_ids]
        db = init_db(args.db)
        added = ingest(db, rows)
        result = worker(db, args.root, reserve_gib=args.reserve_gib, seconds=args.seconds,
                        max_bytes=args.max_bytes, max_files=args.max_files)
        result["new_file_jobs_ingested"] = added
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(db_report(init_db(args.db)), ensure_ascii=False))


if __name__ == "__main__":
    main()
