#!/usr/bin/env python3
"""Verify Kaggle's single-file ZIP transport without weakening source size checks.

Kaggle's public file listing describes uncompressed files. Large single-file
downloads are wrapped in a ZIP. This metadata inspector reads bounded byte
ranges only, pins the dataset version, and matches the archive member to the
official file listing before proposing a transport metadata correction.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from pathlib import PurePosixPath
import re
import urllib.parse
import urllib.request
import zipfile

MAX_METADATA_BYTES = 1024 * 1024
MAX_RANGE_REQUESTS = 20
MAX_SINGLE_RANGE = 131072
OFFICIAL_DESCRIPTION = "https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_descriptor(url, version):
    parts = urllib.parse.urlsplit(url)
    require(parts.scheme == "https" and parts.hostname == "www.kaggle.com" and not parts.username
            and not parts.password and parts.port in (None, 443), "Expected official Kaggle HTTPS URL")
    match = re.fullmatch(r"/api/v1/datasets/download/([^/]+)/([^/]+)", parts.path)
    require(match is not None, "Unreviewed Kaggle dataset download path")
    query = urllib.parse.parse_qs(parts.query, strict_parsing=True)
    require(set(query) <= {"file_name", "datasetVersionNumber"} and len(query.get("file_name", [])) == 1,
            "Expected one explicit source file")
    require(isinstance(version, int) and not isinstance(version, bool) and version > 0, "Pin a positive source version")
    if "datasetVersionNumber" in query:
        require(query["datasetVersionNumber"] == [str(version)], "Dataset version differs")
    member = query["file_name"][0]
    path = PurePosixPath(member)
    require(member and not path.is_absolute() and "\\" not in member and "\x00" not in member
            and all(p not in ("", ".", "..") for p in member.split("/")), "Unsafe source member")
    pinned = urllib.parse.urlunsplit(("https", "www.kaggle.com", parts.path,
            urllib.parse.urlencode({"file_name": member, "datasetVersionNumber": version}), ""))
    return {"owner": match[1], "dataset": match[2], "version": version, "member": member, "url": pinned}


def google_md5(headers):
    values = []
    hash_headers = headers.get_all("x-goog-hash", []) if hasattr(headers, "get_all") else [headers.get("x-goog-hash") or ""]
    for item in ",".join(hash_headers).split(","):
        key, separator, value = item.strip().partition("=")
        if separator and key == "md5":
            raw = base64.b64decode(value, validate=True)
            require(len(raw) == 16, "Invalid Google object MD5")
            values.append(raw.hex())
    require(len(set(values)) <= 1, "Conflicting Google object MD5 values")
    return {"algorithm": "md5", "value": values[0]} if values else None


class MetadataRangeReader(io.RawIOBase):
    """Seekable bounded metadata view; never persists a remote ZIP or waveform."""

    def __init__(self, descriptor, opener=None):
        self.descriptor = descriptor
        self.opener = opener or urllib.request.urlopen
        self.position = 0
        self.total = None
        self.etag = None
        self.object_path = None
        self.published_checksum = None
        self.requests = 0
        self.bytes_read = 0
        self.first_four = self._span(0, 3)
        require(self.first_four == b"PK\x03\x04", "Single-file response is not a ZIP")
        require(self.total > 22, "ZIP too small")
        self.position = 0

    def _span(self, start, end):
        require(0 <= start <= end and end - start + 1 <= MAX_SINGLE_RANGE, "Unbounded metadata range")
        require(self.requests < MAX_RANGE_REQUESTS
                and self.bytes_read + end - start + 1 <= MAX_METADATA_BYTES, "Metadata byte/request budget exceeded")
        headers = {"User-Agent": "NeuroCore-Xavier-metadata-inspection/1",
                   "Accept-Encoding": "identity", "Range": f"bytes={start}-{end}"}
        if self.etag:
            headers["If-Match"] = self.etag
        request = urllib.request.Request(self.descriptor["url"], headers=headers)
        with self.opener(request, timeout=20) as response:
            require(response.status == 206, "Server must honor metadata Range")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            require(match is not None and int(match[1]) == start and int(match[2]) == end,
                    "Metadata Content-Range differs")
            total = int(match[3])
            require(end < total and (self.total is None or total == self.total), "ZIP size changed between ranges")
            target = urllib.parse.urlsplit(response.url)
            require(target.scheme == "https" and target.hostname == "storage.googleapis.com"
                    and target.port in (None, 443) and not target.username and not target.password,
                    "Unreviewed Kaggle object origin")
            object_path = urllib.parse.unquote(target.path)
            suffix = "/compressed/" + self.descriptor["member"] + ".zip"
            require(re.fullmatch(r"/kaggle-data-sets/\d+/\d+/compressed/.+", object_path) is not None
                    and object_path.endswith(suffix), "Kaggle object path does not bind the requested file")
            etag = response.headers.get("ETag")
            require(etag and not etag.startswith("W/"), "A strong transport validator is required")
            checksum = google_md5(response.headers)
            require(checksum is not None, "Google object MD5 is required for corrected ZIP bytes")
            if self.total is not None:
                require(etag == self.etag and object_path == self.object_path
                        and checksum == self.published_checksum, "Transport generation changed between ranges")
            value = response.read(end - start + 2)
            require(len(value) == end - start + 1, "Metadata response length differs")
        self.total, self.etag, self.object_path = total, etag, object_path
        self.published_checksum = checksum
        self.requests += 1
        self.bytes_read += len(value)
        return value

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        origin = (0, self.position, self.total)[whence] if whence in (0, 1, 2) else None
        require(origin is not None and origin + offset >= 0, "Invalid ZIP metadata seek")
        self.position = origin + offset
        return self.position

    def read(self, size=-1):
        if self.position >= self.total or size == 0:
            return b""
        size = self.total - self.position if size is None or size < 0 else min(size, self.total - self.position)
        value = self._span(self.position, self.position + size - 1)
        self.position += len(value)
        return value


def inspect_transport(url, version, official_file, opener=None):
    descriptor = source_descriptor(url, version)
    require(official_file.get("name") == descriptor["member"], "Official file name does not match the request")
    expected = official_file.get("totalBytes")
    require(isinstance(expected, int) and not isinstance(expected, bool) and expected > 0,
            "Official uncompressed file size must be positive")
    reader = MetadataRangeReader(descriptor, opener=opener)
    with zipfile.ZipFile(reader) as archive:
        entries = archive.infolist()
        require(1 <= len(entries) <= 4, "Not a bounded single-file ZIP")
        for item in entries:
            name = item.filename.rstrip("/")
            require(name and not name.startswith("/") and "\\" not in name
                    and all(p not in ("", ".", "..") for p in name.split("/")),
                    "Unsafe archive member name")
            require((item.external_attr >> 16) & 0o170000 != 0o120000, "ZIP symlink is not a source file")
            require(not item.flag_bits & 1, "Encrypted ZIP is not admitted")
        files = [item for item in entries if not item.is_dir()]
        require(len(files) == 1, "The API response contains multiple source files")
        item = files[0]
        require(item.filename in (descriptor["member"], PurePosixPath(descriptor["member"]).name),
                "ZIP member does not match the official source file")
        require(item.file_size == expected, "ZIP uncompressed size differs from official metadata")
        require(item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), "Unreviewed ZIP compression")
        require(0 < item.compress_size <= reader.total and item.header_offset >= 0,
                "Invalid ZIP compressed extent")
        member = {"name": item.filename, "uncompressed_bytes": item.file_size, "compressed_bytes": item.compress_size,
                  "crc32_hex": f"{item.CRC:08x}", "compression_method": item.compress_type,
                  "header_offset": item.header_offset}
    result = {"schema": "xavier-kaggle-single-file-zip-transport-v1", "source": descriptor,
              "official_file_metadata_sha256": canonical_sha(official_file), "source_uncompressed_bytes": expected,
              "archive_transport_bytes": reader.total, "archive_member": member,
              "published_archive_checksum": reader.published_checksum, "transport_etag": reader.etag,
              "object_origin_path": "https://storage.googleapis.com" + reader.object_path,
              "metadata_range_requests": reader.requests, "metadata_range_bytes": reader.bytes_read,
              "source_signal_decoded": False, "archive_downloaded": False,
              "full_member_crc_verified": False, "official_transport_description": OFFICIAL_DESCRIPTION}
    result["descriptor_sha256"] = canonical_sha(result)
    return result


def corrected_file_metadata(file, descriptor):
    """Return a proposed entry. Applying it to an existing ledger is separate."""
    require(descriptor.get("schema") == "xavier-kaggle-single-file-zip-transport-v1", "Unknown transport proof")
    unsealed = {k: v for k, v in descriptor.items() if k != "descriptor_sha256"}
    require(canonical_sha(unsealed) == descriptor.get("descriptor_sha256"), "Transport proof seal differs")
    original = source_descriptor(file["url"], descriptor["source"]["version"])
    require(original == descriptor["source"], "Correction is bound to another source")
    require(file.get("size_bytes") == descriptor["source_uncompressed_bytes"], "Original manifest size differs")
    require(file["path"].endswith(descriptor["source"]["member"] + ".zip"), "Original path is not the expected ZIP wrapper")
    require(not file.get("published_checksum"), "An existing published checksum must not be replaced")
    output = copy.deepcopy(file)
    output.update(url=descriptor["source"]["url"], size_bytes=descriptor["archive_transport_bytes"],
                  published_checksum=descriptor["published_archive_checksum"],
                  provider_file_identity="kaggle:url:" + descriptor["source"]["url"])
    output["transport_metadata_correction"] = {
        "reason": "published_uncompressed_source_size_was_assigned_to_provider_zip_wrapper",
        "original_url": file["url"], "original_size_bytes": file["size_bytes"],
        "original_published_checksum": file.get("published_checksum"),
        "source_version": descriptor["source"]["version"], "source_member": descriptor["source"]["member"],
        "source_uncompressed_bytes": descriptor["source_uncompressed_bytes"],
        "transport_descriptor_sha256": descriptor["descriptor_sha256"],
        "archive_member_crc32_hex": descriptor["archive_member"]["crc32_hex"],
        "full_archive_verification_required_at_download": True,
        "full_member_crc_and_source_checks_required_before_training": True}
    return output
