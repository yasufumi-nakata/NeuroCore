#!/usr/bin/env python3
"""Read-only, bounded EEGMMIDB 1.0.0 ZIP integrity/header/annotation audit.

Never extracts or transforms waveform files. Official SHA256SUMS and RECORDS
must cover the complete release. Counts recording seconds, not channel-hours.
Physical calibration remains the publisher's declaration, not hardware proof.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
import zipfile

VERSION = "pj-xavier-eegmmidb-archive-audit-v1"
SOURCES = {
    "dataset": "https://physionet.org/content/eegmmidb/1.0.0/",
    "checksums": "https://physionet.org/files/eegmmidb/1.0.0/SHA256SUMS.txt",
    "records": "https://physionet.org/files/eegmmidb/1.0.0/RECORDS",
    "run_mapping": "https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html",
    "edf": "https://www.edfplus.info/specs/edf.html",
    "edf_plus": "https://www.edfplus.info/specs/edfplus.html",
}
MAX_RECORD_BYTES = 16 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_name(name):
    require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name,
            "invalid archive path")
    parts = name.split("/")
    require(not PurePosixPath(name).is_absolute() and not any(p in {"", ".", ".."} for p in parts),
            "unsafe archive path")
    return name


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def checksums(text):
    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
        require(match is not None, "malformed checksum row")
        name = safe_name(match.group(2))
        require(name not in result, "duplicate checksum member")
        result[name] = match.group(1).lower()
    require(bool(result), "empty official checksum manifest")
    return result


def run_task(run):
    mapping = {1: "baseline_eyes_open", 2: "baseline_eyes_closed"}
    for runs, task in [((3, 7, 11), "motor_execution_left_right_fist"),
                       ((4, 8, 12), "motor_imagery_left_right_fist"),
                       ((5, 9, 13), "motor_execution_both_fists_feet"),
                       ((6, 10, 14), "motor_imagery_both_fists_feet")]:
        mapping.update({r: task for r in runs})
    require(run in mapping, "run is outside the published 1-14 mapping")
    return mapping[run]


def semantic_label(run, code):
    task = run_task(run)
    if code == "T0":
        return task if run in (1, 2) else "rest"
    if code not in {"T1", "T2"} or run in (1, 2):
        return "unmapped_annotation"
    action = ("left_fist" if code == "T1" else "right_fist") if "left_right" in task else (
        "both_fists" if code == "T1" else "both_feet")
    return ("imagined_" if "imagery" in task else "executed_") + action


def read_exact(stream, count):
    value = stream.read(count)
    require(len(value) == count, "truncated EDF header or data record")
    return value


def edf_header(stream, file_size):
    first = read_exact(stream, 256)
    require(first[:8].strip() == b"0", "unsupported EDF/BDF version")
    nsignals = int(first[252:256].strip())
    header_bytes = int(first[184:192].strip())
    require(1 <= nsignals <= 1024 and header_bytes == 256 * (1 + nsignals),
            "invalid EDF header length or signal count")
    require(header_bytes <= file_size, "EDF header exceeds member length")
    extra = read_exact(stream, header_bytes - 256)
    arrays, offset = {}, 0
    fields = [("label", 16), ("transducer", 80), ("unit", 8), ("physical_min", 8),
              ("physical_max", 8), ("digital_min", 8), ("digital_max", 8),
              ("prefilter", 80), ("samples_per_record", 8), ("reserved", 32)]
    for key, width in fields:
        arrays[key] = [extra[offset + i * width:offset + (i + 1) * width].decode("latin1").strip()
                       for i in range(nsignals)]
        offset += width * nsignals
    require(offset == len(extra), "EDF signal table not fully consumed")
    channels = [{key: value[i] for key, value in arrays.items()} for i in range(nsignals)]
    for channel in channels:
        channel["samples_per_record"] = int(channel["samples_per_record"])
        require(channel["samples_per_record"] > 0, "nonpositive channel samples per record")
        channel["is_annotation"] = channel["label"].lower() == "edf annotations"
        for key in ("physical_min", "physical_max", "digital_min", "digital_max"):
            channel[key] = float(channel[key])
            require(math.isfinite(channel[key]), "nonfinite header calibration")
        require(channel["digital_max"] > channel["digital_min"], "invalid digital calibration")
    duration = Decimal(first[244:252].decode("ascii").strip())
    require(duration.is_finite() and duration > 0, "nonpositive/nonfinite data record duration")
    record_bytes = 2 * sum(c["samples_per_record"] for c in channels)
    require(0 < record_bytes <= MAX_RECORD_BYTES, "data record exceeds bounded buffer")
    declared = int(first[236:244].strip())
    payload = file_size - header_bytes
    require(payload % record_bytes == 0, "partial EDF data record")
    nrecords = payload // record_bytes
    require(nrecords > 0 and declared in {-1, nrecords}, "EDF declared record count differs from payload")
    eeg = [c for c in channels if not c["is_annotation"]]
    require(eeg, "EDF has no ordinary signal channels")
    for channel in channels:
        channel["sample_rate_hz"] = float(Decimal(channel["samples_per_record"]) / duration)
        channel["total_samples"] = channel["samples_per_record"] * nrecords
    return {"header_bytes": header_bytes, "header_sha256": hashlib.sha256(first + extra).hexdigest(),
            "data_records": nrecords, "declared_data_records": declared,
            "record_count_inferred_from_size": declared == -1,
            "data_record_duration_seconds_decimal": str(duration),
            "signal_duration_seconds_decimal": str(duration * nrecords),
            "continuity_marker": first[192:236].decode("ascii").strip(),
            "channels": channels, "data_record_bytes": record_bytes,
            "eeg_channel_count": len(eeg), "annotation_channel_count": nsignals - len(eeg),
            "eeg_sample_values": sum(c["total_samples"] for c in eeg),
            "eeg_sample_bytes": 2 * sum(c["total_samples"] for c in eeg),
            "annotation_payload_bytes": 2 * sum(c["total_samples"] for c in channels if c["is_annotation"])}, first + extra


def tal_annotations(block):
    entries = []
    for tal in block.split(b"\x00"):
        if not tal:
            continue
        parts = tal.split(b"\x14")
        require(len(parts) >= 2 and parts[-1] == b"", "malformed EDF+ TAL separator")
        timing = parts[0].split(b"\x15")
        require(len(timing) in {1, 2} and timing[0].startswith((b"+", b"-")), "invalid TAL onset")
        onset = Decimal(timing[0].decode("ascii"))
        duration = Decimal(timing[1].decode("ascii")) if len(timing) == 2 else None
        require(onset.is_finite() and (duration is None or duration.is_finite() and duration >= 0), "invalid TAL time")
        for label in parts[1:-1]:
            if label:
                entries.append({"onset_seconds": float(onset),
                                "duration_seconds": float(duration) if duration is not None else None,
                                "code": label.decode("utf-8")})
    return entries


def audit_edf(stream, file_size, name):
    match = re.fullmatch(r"S(\d{3})/S\1R(\d{2})\.edf", name)
    require(match is not None, "EDF member is outside official subject/run naming")
    header, raw_header = edf_header(stream, file_size)
    digest, signal_digest = hashlib.sha256(raw_header), hashlib.sha256()
    events, errors = [], []
    for _ in range(header["data_records"]):
        record = read_exact(stream, header["data_record_bytes"])
        digest.update(record)
        offset = 0
        for channel in header["channels"]:
            size = channel["samples_per_record"] * 2
            values = record[offset:offset + size]
            if channel["is_annotation"]:
                try:
                    events.extend(tal_annotations(values))
                except (ValueError, UnicodeDecodeError, ArithmeticError) as error:
                    errors.append("TAL_decode:" + type(error).__name__)
            else:
                signal_digest.update(values)
            offset += size
        require(offset == len(record), "record bytes not fully assigned to channels")
    require(stream.read(1) == b"", "extra bytes after EDF records")
    run = int(match.group(2))
    duration = float(Decimal(header["signal_duration_seconds_decimal"]))
    intervals = []
    for event in events:
        event["semantic_label"] = semantic_label(run, event["code"])
        d = event["duration_seconds"]
        if event["code"] not in {"T0", "T1", "T2"}:
            errors.append("unmapped_annotation_code")
        if d is not None:
            a, b = event["onset_seconds"], event["onset_seconds"] + d
            if a < -1e-6 or b > duration + 1e-6:
                errors.append("annotation_outside_signal_duration")
            intervals.append((a, b))
    intervals.sort()
    for left, right in zip(intervals, intervals[1:]):
        if right[0] < left[1] - 1e-6:
            errors.append("overlapping_annotation_intervals")
    header.update(member=name, member_sha256=digest.hexdigest(), eeg_payload_sha256=signal_digest.hexdigest(),
                  subject_id="S" + match.group(1), run=run, task=run_task(run), member_bytes=file_size,
                  annotation_counts=dict(Counter(e["code"] for e in events)),
                  semantic_annotation_counts=dict(Counter(e["semantic_label"] for e in events)),
                  annotation_interval_sum_seconds=sum(b - a for a, b in intervals),
                  annotation_errors=dict(Counter(errors)), events=events)
    return header


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def audit(archive, checksum_file, records_file, output, *, prefix="", expected_archive_sha256=None,
          deadline=None, max_uncompressed_bytes=5_000_000_000):
    started = time.time()
    archive, checksum_file, records_file, output = map(Path, (archive, checksum_file, records_file, output))
    require(not archive.is_symlink() and archive.is_file(), "archive missing or symlink")
    require(not output.exists(), "existing audit output must not be overwritten")
    before = archive.stat()
    output.mkdir(parents=True)
    checksum_sha, records_sha = sha_file(checksum_file), sha_file(records_file)
    expected = checksums(checksum_file.read_text())
    require("RECORDS" not in expected or expected["RECORDS"] == records_sha,
            "external RECORDS does not match official checksum manifest")
    records = [safe_name(s.strip()) for s in records_file.read_text().splitlines() if s.strip()]
    require(len(set(records)) == len(records) and records, "duplicate or missing RECORDS")
    require(all(s.endswith(".edf") and s in expected for s in records), "RECORDS lacks official checksums")
    require(not prefix or prefix.endswith("/") and safe_name(prefix[:-1]), "invalid explicit ZIP prefix")
    archive_sha = sha_file(archive)
    require(expected_archive_sha256 is None or archive_sha == expected_archive_sha256, "archive ledger SHA mismatch")
    members, edfs = [], []
    total_member_bytes = 0
    with zipfile.ZipFile(archive) as z, (output / "members.jsonl").open("x") as ledger, (output / "recordings.jsonl").open("x") as record_ledger:
        infos, named = z.infolist(), {}
        for item in infos:
            if item.is_dir():
                safe_name(item.filename.rstrip("/"))
                continue
            safe_name(item.filename)
            require(not stat.S_ISLNK(item.external_attr >> 16) and not item.flag_bits & 1, "symlink/encrypted ZIP member")
            require(item.filename.startswith(prefix), "ZIP member outside explicit prefix")
            name = safe_name(item.filename[len(prefix):])
            require(name not in named, "duplicate canonical archive member")
            named[name] = item
            total_member_bytes += item.file_size
        require(total_member_bytes <= max_uncompressed_bytes, "archive expanded-byte budget exceeded")
        require(set(expected) <= set(named), "official checksum members absent from ZIP")
        require(set(records) == {s for s in named if s.endswith(".edf")}, "ZIP EEG files and official RECORDS differ")
        for number, (name, item) in enumerate(sorted(named.items())):
            require(deadline is None or time.time() < deadline, "original deadline reached")
            with z.open(item) as stream:
                if name.endswith(".edf"):
                    row = audit_edf(stream, item.file_size, name)
                    actual_sha = row["member_sha256"]
                    record_ledger.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    edfs.append({k: v for k, v in row.items() if k != "events"})
                else:
                    digest, counted = hashlib.sha256(), 0
                    for block in iter(lambda: stream.read(1024**2), b""):
                        counted += len(block)
                        digest.update(block)
                    require(counted == item.file_size, "ZIP member stream size mismatch")
                    actual_sha = digest.hexdigest()
            known = expected.get(name)
            if name == "SHA256SUMS.txt":
                known = checksum_sha
            require(known is None or actual_sha == known, "official member SHA mismatch: " + name)
            member = {"member": name, "bytes": item.file_size, "compressed_member_bytes": item.compress_size,
                      "sha256": actual_sha, "official_checksum_verified": known is not None}
            members.append(member)
            ledger.write(json.dumps(member) + "\n")
            if number % 100 == 0:
                print(json.dumps({"members_checked": number + 1, "total_members": len(named),
                                  "EEG_records_audited": len(edfs)}), flush=True)
    after = archive.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino),
            "source archive changed during audit")
    unique = {}
    def signal_interpretation(row):
        return [(c["label"], c["unit"], c["physical_min"], c["physical_max"],
                 c["digital_min"], c["digital_max"], c["sample_rate_hz"], c["total_samples"])
                for c in row["channels"] if not c["is_annotation"]]
    for row in edfs:
        identity = row["eeg_payload_sha256"]
        old = unique.get(identity)
        if old:
            require((old["eeg_channel_count"], old["signal_duration_seconds_decimal"], old["eeg_sample_bytes"], signal_interpretation(old))
                    == (row["eeg_channel_count"], row["signal_duration_seconds_decimal"], row["eeg_sample_bytes"], signal_interpretation(row)),
                    "same raw EEG payload has conflicting time/channel/calibration interpretation")
        else:
            unique[identity] = row
    durations = sum((Decimal(r["signal_duration_seconds_decimal"]) for r in edfs), Decimal(0))
    unique_durations = sum((Decimal(r["signal_duration_seconds_decimal"]) for r in unique.values()), Decimal(0))
    tasks = {}
    for task in sorted({r["task"] for r in edfs}):
        selected = [r for r in edfs if r["task"] == task]
        seconds = sum((Decimal(r["signal_duration_seconds_decimal"]) for r in selected), Decimal(0))
        codes, semantic = Counter(), Counter()
        for row in selected:
            codes.update(row["annotation_counts"])
            semantic.update(row["semantic_annotation_counts"])
        tasks[task] = {"recordings": len(selected), "subjects": len({r["subject_id"] for r in selected}),
                       "signal_seconds": float(seconds), "signal_hours": float(seconds / Decimal(3600)),
                       "eeg_sample_bytes": sum(r["eeg_sample_bytes"] for r in selected),
                       "annotation_counts": dict(codes), "semantic_annotation_counts": dict(semantic)}
    bytes_by_type = Counter()
    for member in members:
        name = member["member"]
        category = "EDF_files" if name.endswith(".edf") else "WFDB_event_files" if name.endswith(".edf.event") else "metadata_or_documentation"
        bytes_by_type[category] += member["bytes"]
    result = {"schema": VERSION, "status": "completed", "source_id": "physionet:eegmmidb:1.0.0",
        "catalog_id": "009861", "started_epoch": started, "ended_epoch": time.time(),
        "source_code_sha256": sha_file(__file__), "source_urls": SOURCES,
        "archive_sha256": archive_sha, "archive_distribution_bytes": before.st_size,
        "official_checksum_manifest_sha256": checksum_sha, "official_RECORDS_sha256": records_sha,
        "all_published_member_hashes_verified": True, "official_checksum_members": len(expected),
        "archive_regular_members": len(members), "archive_directory_entries": sum(i.is_dir() for i in infos),
        "archive_uncompressed_member_bytes": total_member_bytes, "uncompressed_bytes_by_type": dict(bytes_by_type),
        "unlisted_members": [m["member"] for m in members if not m["official_checksum_verified"]],
        "official_RECORDS_coverage_complete": True, "eeg_recordings": len(edfs),
        "publisher_subject_groups": len({r["subject_id"] for r in edfs}),
        "unique_EEG_payload_recordings": len(unique), "duplicate_EEG_payload_recordings": len(edfs) - len(unique),
        "all_recording_signal_seconds_decimal": str(durations), "all_recording_signal_hours": float(durations / Decimal(3600)),
        "content_unique_signal_seconds_decimal": str(unique_durations),
        "content_unique_signal_hours": float(unique_durations / Decimal(3600)),
        "eeg_physical_sample_storage_bytes": sum(r["eeg_sample_bytes"] for r in edfs),
        "content_unique_eeg_sample_storage_bytes": sum(r["eeg_sample_bytes"] for r in unique.values()),
        "edf_annotation_payload_bytes": sum(r["annotation_payload_bytes"] for r in edfs),
        "edf_header_bytes": sum(r["header_bytes"] for r in edfs),
        "eeg_sample_values": sum(r["eeg_sample_values"] for r in edfs),
        "eeg_channel_count_records": dict(Counter(str(r["eeg_channel_count"]) for r in edfs)),
        "sample_rate_set_records": dict(Counter(
            json.dumps(sorted({c["sample_rate_hz"] for c in r["channels"] if not c["is_annotation"]})) for r in edfs)),
        "unit_set_records": dict(Counter(
            json.dumps(sorted({c["unit"] for c in r["channels"] if not c["is_annotation"]})) for r in edfs)),
        "continuity_marker_records": dict(Counter(r["continuity_marker"] for r in edfs)),
        "record_count_inferred_files": sum(r["record_count_inferred_from_size"] for r in edfs),
        "annotation_error_records": sum(bool(r["annotation_errors"]) for r in edfs),
        "annotation_errors": dict(sum((Counter(r["annotation_errors"]) for r in edfs), Counter())),
        "record_duration_seconds_min": float(min(Decimal(r["signal_duration_seconds_decimal"]) for r in edfs)),
        "record_duration_seconds_max": float(max(Decimal(r["signal_duration_seconds_decimal"]) for r in edfs)),
        "tasks": tasks, "members_ledger_sha256": sha_file(output / "members.jsonl"),
        "recordings_ledger_sha256": sha_file(output / "recordings.jsonl"),
        "training_or_model_operations": 0, "raw_files_modified": 0, "archive_extracted_to_disk": False,
        "interpretation": {
            "duration": "header/sample-derived recording seconds; not channel-hours, unique-person time, or autonomous-control labels",
            "storage": "physical sample storage bytes means original two-byte EEG values; hardware calibration not independently established",
            "deduplication": "within-release exact EEG payload SHA; aliases not acquired or added; cross-dataset shared participants remain unknown",
            "labels": "publisher task annotations interpreted by run; not inferred thoughts, freeform intent, or agent actions",
            "event_companion": "WFDB .event checksum verified; its semantic equality to EDF+ TAL is stated by publisher, not independently decoded here",
            "training": "not added to existing frozen SSL, alignment, or benchmark cohorts; new protocol required"}}
    require(sum(bytes_by_type.values()) == total_member_bytes, "uncompressed byte totals differ")
    require(result["eeg_physical_sample_storage_bytes"] + result["edf_annotation_payload_bytes"] + result["edf_header_bytes"] == bytes_by_type["EDF_files"],
            "EDF byte partitions overlap or omit bytes")
    save_json(output / "summary.json", result)
    save_json(output / "completion.json", {"status": "completed", "exit_code": 0,
        "summary_sha256": sha_file(output / "summary.json"), "archive_sha256": archive_sha,
        "finished_epoch": time.time(), "raw_files_modified": 0})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True)
    p.add_argument("--checksums", required=True)
    p.add_argument("--records", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--zip-prefix", default="")
    p.add_argument("--expected-archive-sha256", required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    p.add_argument("--max-uncompressed-bytes", type=int, default=5_000_000_000)
    args = p.parse_args()
    result = audit(args.archive, args.checksums, args.records, args.output, prefix=args.zip_prefix,
                   expected_archive_sha256=args.expected_archive_sha256, deadline=args.deadline_epoch,
                   max_uncompressed_bytes=args.max_uncompressed_bytes)
    print(json.dumps({k: result[k] for k in ("status", "eeg_recordings", "publisher_subject_groups",
          "archive_distribution_bytes", "archive_uncompressed_member_bytes", "eeg_physical_sample_storage_bytes",
          "content_unique_signal_hours", "annotation_error_records")}), flush=True)


if __name__ == "__main__":
    main()
