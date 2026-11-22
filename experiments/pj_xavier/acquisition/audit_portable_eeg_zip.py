#!/usr/bin/env python3
"""Audit original recordings in the fixed portable-EEG release, without extraction.

Counts only raw data/, cross-checks BIDS sidecars and header/body sizes, and reads
every counted signal member to verify ZIP CRC and calculate content SHA-256.
EDF specification: https://www.edfplus.info/specs/edf.html
Release: https://zenodo.org/records/15698441
"""
from __future__ import annotations

import argparse
from collections import Counter
import configparser
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def brainvision_metadata(header: bytes, binary_bytes: int) -> dict:
    text = header.decode("utf-8-sig")
    if not text.startswith("Brain Vision Data Exchange Header File"):
        raise ValueError("unsupported BrainVision header")
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read_string(text[text.index("[Common Infos]"):])
    c = cfg["Common Infos"]
    if c["DataFormat"] != "BINARY" or c["DataOrientation"] not in {"MULTIPLEXED", "VECTORIZED"}:
        raise ValueError("unsupported BrainVision storage")
    width = {"IEEE_FLOAT_32": 4, "INT_16": 2, "UINT_16": 2}.get(cfg["Binary Infos"]["BinaryFormat"])
    channels, interval = int(c["NumberOfChannels"]), float(c["SamplingInterval"])
    if width is None or channels <= 0 or not math.isfinite(interval) or interval <= 0:
        raise ValueError("invalid BrainVision channel count, type, or interval")
    if binary_bytes <= 0 or binary_bytes % (width * channels):
        raise ValueError("BrainVision binary is not whole sample frames")
    samples = binary_bytes // (width * channels)
    if "DataPoints" in c and int(c["DataPoints"]) != samples:
        raise ValueError("BrainVision DataPoints differs from binary")
    labels = [cfg["Channel Infos"][f"Ch{i}"].split(",")[0].replace("\\1", ",") for i in range(1, channels + 1)]
    if len(set(labels)) != channels:
        raise ValueError("duplicate BrainVision channel labels")
    return {"format": "BrainVision", "data_file": c["DataFile"], "channels": channels,
            "labels": labels, "sample_rate_hz": 1e6 / interval, "samples_per_channel": samples,
            "duration_seconds": samples * interval / 1e6, "signal_body_bytes": binary_bytes,
            "bytes_per_channel": samples * width, "continuous": True}


def edf_metadata(header: bytes, total_bytes: int) -> dict:
    if len(header) < 256 or header[:8].strip() != b"0":
        raise ValueError("unsupported EDF version or truncated header")
    header_bytes = int(header[184:192])
    declared_records = int(header[236:244])
    record_seconds, channels = float(header[244:252]), int(header[252:256])
    if channels <= 0 or header_bytes != 256 + channels * 256 or len(header) != header_bytes:
        raise ValueError("EDF header length/channel count mismatch")
    if not math.isfinite(record_seconds) or record_seconds <= 0:
        raise ValueError("invalid EDF record duration")
    labels = [header[256 + 16 * i:256 + 16 * (i + 1)].decode("ascii").strip() for i in range(channels)]
    offset = 256 + 216 * channels
    samples = [int(header[offset + 8 * i:offset + 8 * (i + 1)]) for i in range(channels)]
    if any(n <= 0 for n in samples):
        raise ValueError("invalid EDF samples per data record")
    record_bytes = 2 * sum(samples)
    body = total_bytes - header_bytes
    if body <= 0 or body % record_bytes:
        raise ValueError("EDF body is not whole data records")
    observed_records = body // record_bytes
    if declared_records != -1 and declared_records != observed_records:
        raise ValueError("EDF declared record count differs from body")
    reserved = header[192:236].decode("ascii").strip()
    return {"format": "EDF", "channels": channels, "labels": labels, "data_records": observed_records,
            "record_seconds": record_seconds, "duration_seconds": record_seconds * observed_records,
            "sample_rates_hz": [n / record_seconds for n in samples],
            "sample_counts": [n * observed_records for n in samples],
            "signal_body_bytes": body, "header_bytes": header_bytes,
            "continuous": not reserved.startswith("EDF+D"), "edf_record_padding_possible": True}


def zip_header(z, name):
    with z.open(name) as stream:
        first = stream.read(256)
        if len(first) != 256:
            raise ValueError("truncated EDF fixed header")
        size = int(first[184:192])
        if not 256 <= size <= 1024**2:
            raise ValueError("unreasonable EDF header size")
        return first + stream.read(size - 256)


def audit(archive, expected_sha):
    archive = Path(archive)
    if file_hash(archive) != expected_sha:
        raise ValueError("archive does not match fixed acquisition SHA-256")
    records, held, seen_data, body_hashes = [], [], set(), set()
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ambiguous duplicate ZIP members")
        chosen = sorted(n for n in names if n.startswith("raw data/") and n.endswith(("_eeg.vhdr", "_eeg.edf")))
        for name in chosen:
            try:
                stem = name.rsplit(".", 1)[0]
                sidecar = json.loads(z.read(stem + ".json"))
                channel_rows = list(csv.DictReader(io.StringIO(z.read(stem[:-4] + "_channels.tsv").decode("utf-8-sig")), delimiter="\t"))
                eeg_labels = [r["name"] for r in channel_rows if r["type"].upper() == "EEG"]
                if len(set(eeg_labels)) != len(eeg_labels) or not eeg_labels:
                    raise ValueError("missing or duplicate BIDS EEG channel names")
                if name.endswith(".vhdr"):
                    head = z.read(name)
                    text = head.decode("utf-8-sig")
                    match = re.search(r"^DataFile=(.+)$", text, re.MULTILINE)
                    if not match:
                        raise ValueError("missing BrainVision data file")
                    rel = PurePosixPath(match.group(1).strip())
                    if rel.is_absolute() or ".." in rel.parts or "\\" in str(rel):
                        raise ValueError("unsafe BrainVision data member")
                    data_name = str(PurePosixPath(name).parent / rel)
                    meta = brainvision_metadata(head, z.getinfo(data_name).file_size)
                else:
                    head, data_name = zip_header(z, name), name
                    meta = edf_metadata(head, z.getinfo(name).file_size)
                if data_name in seen_data:
                    raise ValueError("same waveform member referenced twice")
                if int(sidecar["EEGChannelCount"]) != len(eeg_labels):
                    raise ValueError("BIDS EEGChannelCount differs from channels.tsv")
                if any(label not in meta["labels"] for label in eeg_labels):
                    raise ValueError("BIDS EEG labels not present in binary header")
                eeg_indices = [meta["labels"].index(label) for label in eeg_labels]
                rates = ([meta["sample_rate_hz"]] * len(eeg_labels) if meta["format"] == "BrainVision"
                         else [meta["sample_rates_hz"][i] for i in eeg_indices])
                if any(not math.isclose(rate, float(sidecar["SamplingFrequency"]), abs_tol=1e-9) for rate in rates):
                    raise ValueError("header EEG sample rates differ from BIDS sidecar")
                eeg_bytes = (meta["bytes_per_channel"] * len(eeg_labels) if meta["format"] == "BrainVision"
                             else 2 * sum(meta["sample_counts"][i] for i in eeg_indices))
                h = hashlib.sha256()
                read_bytes = 0
                with z.open(data_name) as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
                        read_bytes += len(chunk)
                        h.update(chunk)
                if read_bytes != z.getinfo(data_name).file_size:
                    raise ValueError("ZIP member stream size mismatch")
                if h.hexdigest() in body_hashes:
                    raise ValueError("duplicate waveform content SHA-256")
                declared_duration = float(sidecar["RecordingDuration"])
                if not math.isfinite(declared_duration) or declared_duration <= 0:
                    raise ValueError("invalid sidecar duration")
                difference = meta["duration_seconds"] - declared_duration
                tolerance = max(1 / min(rates), meta.get("record_seconds", 0)) + 1e-7
                if abs(difference) > tolerance:
                    raise ValueError("header and sidecar duration differ beyond one sample/data record")
                seen_data.add(data_name)
                body_hashes.add(h.hexdigest())
                record = {"recording": stem, "format": meta["format"], "channels_in_file": meta["channels"],
                          "eeg_channels": len(eeg_labels), "eeg_sample_rates_hz": sorted(set(rates)),
                          "stored_signal_seconds": meta["duration_seconds"], "source_sidecar_seconds": declared_duration,
                          "stored_minus_sidecar_seconds": difference, "member_bytes": read_bytes,
                          "eeg_sample_bytes": eeg_bytes, "signal_member_sha256": h.hexdigest(),
                          "header_sha256": hashlib.sha256(head).hexdigest(), "zip_crc_verified": True,
                          "continuous": meta["continuous"]}
                records.append(record)
            except (KeyError, ValueError, OSError, configparser.Error, zipfile.BadZipFile) as exc:
                held.append({"recording_header": name, "reason": str(exc)})
    return {"schema": "pj-xavier-portable-eeg-header-audit-v1", "observed_utc": datetime.now(timezone.utc).isoformat(),
            "source_url": "https://zenodo.org/records/15698441", "archive_sha256": expected_sha,
            "archive_bytes": archive.stat().st_size, "selected_prefix": "raw data/", "candidate_recordings": len(chosen),
            "verified_recordings": len(records), "held_recordings": held,
            "subjects": len({re.search(r"/sub-([^/]+)/", r["recording"])[1] for r in records}),
            "formats": dict(Counter(r["format"] for r in records)),
            "stored_signal_seconds": sum(r["stored_signal_seconds"] for r in records),
            "stored_signal_hours": sum(r["stored_signal_seconds"] for r in records) / 3600,
            "source_sidecar_seconds": sum(r["source_sidecar_seconds"] for r in records),
            "source_sidecar_hours": sum(r["source_sidecar_seconds"] for r in records) / 3600,
            "eeg_sample_bytes": sum(r["eeg_sample_bytes"] for r in records), "records": records,
            "scope": "Stored original recording duration, not channel-hours. Processing outputs and statistical variants outside raw data/ are excluded.",
            "limitations": ["EDF fixed-size records can include end padding; sidecar and stored duration are both reported.",
                            "No usable-window, artifact quality, intent-label, training-license, or causal online-preprocessing approval is implied.",
                            "Distinct recordings are based on raw member identities and content hashes; cross-dataset re-exports require a separate audit."],
            "format_source": "https://www.edfplus.info/specs/edf.html"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit("preserve existing audit; choose a new output")
    result = audit(a.archive, a.expected_sha256)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
