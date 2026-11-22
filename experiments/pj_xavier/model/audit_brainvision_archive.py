#!/usr/bin/env python3
"""Count verified BrainVision recording time without loading an EEG model.

Read-only against the download ledger/archive. Full compressed MD5 and SHA-256
are checked before interpreting any header. Each binary member is streamed to
SHA-256 for duplicate detection; samples are not decoded or tested for quality.
No extraction, GPU use, clock alignment, labeling, or training is performed.

Format reference (Brain Products, BVCDF 1.0, document 2.5, 2019-12-12):
https://www.brainproducts.com/download/specification-of-brainvision-core-data-format-1-0/
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile
import time


FORMAT_SOURCE = "https://www.brainproducts.com/download/specification-of-brainvision-core-data-format-1-0/"
BYTES_PER_VALUE = {"IEEE_FLOAT_32": 4, "INT_16": 2}
AUXILIARY_NAMES = {"otree_phase", "STI_artificial"}
CHUNK = 2 * 1024 * 1024


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def normalized_member(name):
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Unsafe archive member path")
    return str(path)


def parse_header(payload):
    text = payload.decode("utf-8-sig")
    lines = text.splitlines()
    if not lines or lines[0].strip() not in {
        "BrainVision Data Exchange Header File Version 1.0",
        "Brain Vision Data Exchange Header File Version 1.0",
    }:
        raise ValueError("Unsupported BrainVision identification line")
    sections, section = {}, None
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            if section in sections:
                raise ValueError("Duplicate header section")
            sections[section] = {}
        elif section == "Comment":
            continue
        elif section is not None and "=" in line:
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key in sections[section]:
                raise ValueError("Duplicate header key")
            sections[section][key] = value
        else:
            raise ValueError("Malformed header line")
    common = sections["Common Infos"]
    if common.get("DataFormat") != "BINARY" or common.get("DataOrientation") != "MULTIPLEXED":
        raise ValueError("Only multiplexed binary recordings are audited")
    if common.get("DataType", "TIMEDOMAIN") != "TIMEDOMAIN":
        raise ValueError("Frequency-domain data is not recording time")
    if common.get("Averaged", "NO") != "NO" or common.get("SegmentationType", "NOTSEGMENTED") != "NOTSEGMENTED":
        raise ValueError("Averaged or segmented data needs a separate duration audit")
    channels = int(common["NumberOfChannels"])
    interval = float(common["SamplingInterval"])
    if channels <= 0 or not math.isfinite(interval) or interval <= 0:
        raise ValueError("Invalid channel count or sampling interval")
    format_name = sections["Binary Infos"]["BinaryFormat"]
    if format_name not in BYTES_PER_VALUE:
        raise ValueError("Unsupported binary sample format")
    channel_section = sections["Channel Infos"]
    if set(channel_section) != {f"Ch{i}" for i in range(1, channels + 1)}:
        raise ValueError("Header channel indices do not match NumberOfChannels")
    names = [channel_section[f"Ch{i}"].split(",", 1)[0].replace("\\1", ",")
             for i in range(1, channels + 1)]
    if any(not n for n in names) or len(set(names)) != channels:
        raise ValueError("Missing or repeated channel name")
    datafile = common["DataFile"].replace("\\", "/")
    if "/" in datafile or datafile in {".", ".."} or not datafile:
        raise ValueError("DataFile must name a file in the same archive directory")
    return {"datafile": datafile, "channel_names": names, "binary_channels": channels,
            "sampling_frequency_hz": 1e6 / interval, "binary_format": format_name,
            "bytes_per_value": BYTES_PER_VALUE[format_name]}


def read_ledger(path, relative_path):
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "select expected_bytes,checksum_json,status,sha256,updated_at "
            "from downloads where relative_path=?", (relative_path,)).fetchall()
    if len(rows) != 1:
        raise ValueError("Exactly one ledger row is required")
    return dict(rows[0])


def file_checksums(path):
    before = path.stat()
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            md5.update(block)
            sha256.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Archive changed while hashing")
    return {"bytes": after.st_size, "md5": md5.hexdigest(), "sha256": sha256.hexdigest(),
            "mtime_ns": after.st_mtime_ns}


def archive_inventory(path):
    inventory, contents = {}, {}
    with tarfile.open(path, "r|gz") as archive:
        for item in archive:
            name = normalized_member(item.name)
            if item.isdir():
                continue
            if name in inventory:
                raise ValueError("Repeated archive file path")
            inventory[name] = {"bytes": item.size, "regular_file": item.isfile()}
            if not item.isfile():
                continue
            metadata = name.endswith((".vhdr", "_channels.tsv", "_eeg.json"))
            binary = name.endswith((".eeg", ".seg", ".avg"))
            if not (metadata or binary):
                continue
            if metadata and item.size > 2 * 1024 * 1024:
                raise ValueError("Unexpectedly large recording metadata member")
            stream = archive.extractfile(item)
            sha, total, blocks = hashlib.sha256(), 0, []
            while block := stream.read(CHUNK):
                total += len(block)
                sha.update(block)
                if metadata:
                    blocks.append(block)
            if total != item.size:
                raise ValueError("Truncated tar member")
            inventory[name]["sha256"] = sha.hexdigest()
            if metadata:
                contents[name] = b"".join(blocks)
    return inventory, contents


def recording_from_header(name, inventory, contents):
    header = parse_header(contents[name])
    binary_name = str(PurePosixPath(name).parent / header.pop("datafile"))
    binary = inventory.get(binary_name)
    if not binary or not binary["regular_file"] or not binary.get("sha256"):
        raise ValueError("Referenced binary is missing or not a regular EEG member")
    frame_size = header["binary_channels"] * header["bytes_per_value"]
    samples, remainder = divmod(binary["bytes"], frame_size)
    if remainder or samples <= 0:
        raise ValueError("Binary byte count is not a positive integral number of sample frames")
    seconds = samples / header["sampling_frequency_hz"]
    row = {"header_member": name, "header_sha256": inventory[name]["sha256"],
           "binary_member": binary_name, "binary_sha256": binary["sha256"],
           "binary_bytes": binary["bytes"], **header, "samples_per_channel": samples,
           "duration_seconds": seconds, "duration_hours": seconds / 3600,
           "physical_eeg_channels": None, "auxiliary_names": [], "warnings": []}
    if name.endswith("_eeg.vhdr"):
        channel_path = name[:-len("_eeg.vhdr")] + "_channels.tsv"
        json_path = name[:-len(".vhdr")] + ".json"
    else:
        channel_path, json_path = "", ""
    if channel_path in contents:
        channels = list(csv.DictReader(io.StringIO(contents[channel_path].decode("utf-8-sig")), delimiter="\t"))
        if [r.get("name") for r in channels] != row["channel_names"]:
            raise ValueError("BIDS channel ordering differs from the binary header")
        row["channel_metadata_sha256"] = inventory[channel_path]["sha256"]
        row["channel_types"] = dict(Counter(r.get("type", "").upper() for r in channels))
        row["auxiliary_names"] = [r["name"] for r in channels if r["name"] in AUXILIARY_NAMES]
        row["physical_eeg_channels"] = sum(r.get("type", "").upper() == "EEG" and r["name"] not in AUXILIARY_NAMES for r in channels)
    else:
        row["warnings"].append("BIDS channel types unavailable; physical EEG count is unknown")
    if json_path in contents:
        metadata = json.loads(contents[json_path])
        row["json_metadata_sha256"] = inventory[json_path]["sha256"]
        row["declared_recording_duration_seconds"] = metadata.get("RecordingDuration")
        row["declared_eeg_channels"] = metadata.get("EEGChannelCount")
        declared = metadata.get("RecordingDuration")
        if isinstance(declared, (int, float)) and abs(declared - seconds) > 1 / header["sampling_frequency_hz"] + 1e-6:
            row["warnings"].append("BIDS duration differs from binary-derived duration by more than one sample")
        declared_channels = metadata.get("EEGChannelCount")
        if row["physical_eeg_channels"] is not None and declared_channels is not None and declared_channels != row["physical_eeg_channels"]:
            row["warnings"].append("BIDS EEGChannelCount differs from physical EEG channel types")
    return row


def audit(args):
    started = time.monotonic()
    archive, out = Path(args.archive), Path(args.output_dir)
    if (out / "recording_audit.json").exists():
        raise ValueError("Completed audit already exists; choose a separate output directory")
    row = read_ledger(args.ledger, args.relative_path)
    checksum = json.loads(row["checksum_json"])
    if row["expected_bytes"] != args.expected_bytes or checksum != {"algorithm": "md5", "value": args.expected_md5}:
        raise ValueError("Ledger differs from the separately specified published size/MD5")
    if row["status"] != "complete" or not archive.is_file():
        part = Path(str(archive) + ".part")
        result = {"status": "pending_verified_archive", "ledger_status": row["status"],
                  "expected_bytes": args.expected_bytes, "part_bytes": part.stat().st_size if part.exists() else None,
                  "recording_hours": None, "reason": "No partial archive enters a full-recording duration total"}
        write_json(out / "pending.json", result)
        return result, 3
    print("Checking complete archive bytes and both checksums", flush=True)
    verification = file_checksums(archive)
    if verification["bytes"] != args.expected_bytes or verification["md5"] != args.expected_md5:
        raise ValueError("Archive does not match published size/MD5")
    if not row["sha256"] or verification["sha256"] != row["sha256"]:
        raise ValueError("Archive SHA-256 does not match the completed download ledger")
    print("Streaming every binary member and recording header", flush=True)
    inventory, contents = archive_inventory(archive)
    after = archive.stat()
    if (after.st_size, after.st_mtime_ns) != (verification["bytes"], verification["mtime_ns"]):
        raise ValueError("Verified archive changed while parsing")
    records, blocked = [], []
    prefix = normalized_member(args.recording_prefix).rstrip("/") + "/"
    excluded_headers = []
    for name in sorted(contents):
        if not name.endswith(".vhdr"):
            continue
        if not name.startswith(prefix):
            excluded_headers.append(name)
            continue
        try:
            records.append(recording_from_header(name, inventory, contents))
        except (ValueError, KeyError, UnicodeError, json.JSONDecodeError) as error:
            blocked.append({"header_member": name, "reason": str(error)})
    if not records and not blocked:
        raise ValueError("No BrainVision header was found")
    unique_payloads, duplicate_refs = {}, []
    for record in records:
        key = record["binary_sha256"]
        if key in unique_payloads:
            first = unique_payloads[key]
            if any(first[k] != record[k] for k in ("binary_channels", "sampling_frequency_hz", "binary_format")):
                raise ValueError("Identical binary is given conflicting channel/sample-rate interpretation")
            duplicate_refs.append({"first_header": first["header_member"], "duplicate_header": record["header_member"], "binary_sha256": key})
        else:
            unique_payloads[key] = record
    unique = list(unique_payloads.values())
    aggregate = {
        "schema": "xavier-verified-brainvision-quantity-audit-v1",
        "status": "complete" if not blocked else "complete_with_blocked_headers",
        "source": args.source, "source_archive": archive.name,
        "recording_member_prefix": prefix, "headers_outside_prefix_excluded": len(excluded_headers),
        "archive_verification": verification, "source_ledger_status": row["status"],
        "headers_total": len(records) + len(blocked), "headers_parsed": len(records), "headers_blocked": len(blocked),
        "blocked_reasons": dict(Counter(r["reason"] for r in blocked)),
        "unique_binary_payloads": len(unique), "duplicate_binary_references": len(duplicate_refs),
        "summed_header_recording_hours": math.fsum(r["duration_hours"] for r in records),
        "unique_binary_recording_hours": math.fsum(r["duration_hours"] for r in unique),
        "full_dataset_unique_recording_hours": math.fsum(r["duration_hours"] for r in unique) if not blocked else None,
        "unique_binary_bytes": sum(r["binary_bytes"] for r in unique),
        "total_archive_member_bytes": sum(r["bytes"] for r in inventory.values() if r["regular_file"]),
        "binary_channel_counts": dict(Counter(str(r["binary_channels"]) for r in records)),
        "physical_eeg_channel_counts": dict(Counter(str(r["physical_eeg_channels"]) for r in records)),
        "sample_rates_hz": dict(Counter(str(r["sampling_frequency_hz"]) for r in records)),
        "warnings": dict(Counter(w for r in records for w in r["warnings"])),
        "method": "samples = binary bytes / (header channels * bytes per scalar); duration = samples / header sampling frequency",
        "format_reference": FORMAT_SOURCE,
        "samples_decoded_or_quality_checked": False, "behavior_clock_alignment_checked": False,
        "online_causality_verified": False, "training_eligibility_established": False,
        "quantity_only": True, "elapsed_seconds": time.monotonic() - started,
    }
    full = {"aggregate": aggregate, "records": records, "blocked": blocked,
            "duplicate_references": duplicate_refs, "excluded_headers": excluded_headers}
    write_json(out / "recording_audit.json", full)
    aggregate["private_record_audit_sha256"] = hashlib.sha256((out / "recording_audit.json").read_bytes()).hexdigest()
    write_json(out / "aggregate.json", aggregate)
    return aggregate, 0 if not blocked else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--relative-path", required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--recording-prefix", required=True,
                        help="Archive path containing original recordings; derived output is excluded")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        result, exit_code = audit(args)
    except Exception as error:
        result = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "recording_hours": None}
        write_json(Path(args.output_dir) / "audit_failure.json", result)
        exit_code = 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
