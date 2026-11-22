#!/usr/bin/env python3
"""Read-only raw EEG inventory for later SSL: portable 57 plus UniBa train 18.

No extraction, copying, training, labels, or new split assignment occurs here.
All three portable devices keep the same publisher participant grouping. UniBa
validation/test records and the whole ITU site are excluded from SSL candidates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import configparser
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import time
import zipfile

PORTABLE_ARCHIVE_SHA = "e71b328b41ec62490e46984c227d368148b63abbf3a2a39405c2975a72ae14be"
PORTABLE_AUDIT_SHA = "2f621f7d2b7243b923e46e9553c7a5519cac89beef8f2eea9975227037a06125"
HELPER_SHA = "ba48cb23a72a54a243feff520f285117209bb08ccfa4918f0aa9b230f1f35287"
UNIBA_MANIFEST_SHA = "030e8550bd6f4c79005722eb617e9b64f5f80e87b77866b7f17bec4e7e2a1093"
V3_CHANNELS = ("CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def require(value, why):
    if not value:
        raise ValueError(why)


def unit_to_volts(unit):
    canonical = unit.strip().replace("µ", "u").replace("μ", "u")
    factors = {"V": 1., "mV": 1e-3, "uV": 1e-6, "nV": 1e-9}
    require(canonical in factors, "unrecognized physical unit; manual review required")
    return factors[canonical]


def brainvision_scales(header):
    text = header.decode("utf-8-sig")
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read_string(text[text.index("[Common Infos]"):])
    result = []
    for i in range(1, int(cfg["Common Infos"]["NumberOfChannels"]) + 1):
        parts = cfg["Channel Infos"][f"Ch{i}"].split(",")
        require(len(parts) >= 4 and parts[2] != "" and parts[3] != "", "explicit BrainVision resolution/unit required")
        resolution = float(parts[2])
        require(math.isfinite(resolution) and resolution > 0, "invalid BrainVision resolution")
        result.append({"name": parts[0].replace("\\1", ","), "header_unit": parts[3], "resolution_in_header_unit": resolution,
                       "gain_to_volts": resolution * unit_to_volts(parts[3]), "offset_volts": 0.})
    return cfg, result


def edf_scales(header):
    n = int(header[252:256])
    require(n > 0 and len(header) == 256 + 256 * n, "invalid EDF header size")
    def values(offset, width):
        return [header[256 + offset * n + i * width:256 + offset * n + (i + 1) * width].decode("ascii").strip() for i in range(n)]
    labels, units = values(0, 16), values(96, 8)
    pmin, pmax, dmin, dmax = (values(offset, 8) for offset in (104, 112, 120, 128))
    result = []
    for i, label in enumerate(labels):
        if label == "EDF Annotations":
            result.append({"name": label, "header_unit": units[i], "annotation_channel": True})
            continue
        physical_min, physical_max, digital_min, digital_max = map(float, (pmin[i], pmax[i], dmin[i], dmax[i]))
        require(all(math.isfinite(v) for v in (physical_min, physical_max, digital_min, digital_max)) and digital_min < digital_max and physical_min < physical_max,
                "invalid EDF physical/digital range")
        scale = (physical_max - physical_min) / (digital_max - digital_min)
        item = {"name": label, "header_unit": units[i], "physical_min": physical_min, "physical_max": physical_max,
                "digital_min": digital_min, "digital_max": digital_max}
        # Non-EEG status/time channels may be dimensionless. They are retained in
        # header metadata but never selected as EEG or given an invented unit.
        try:
            factor = unit_to_volts(units[i])
            item.update(gain_to_volts=scale * factor, offset_volts=(physical_min - digital_min * scale) * factor)
        except ValueError:
            item.update(physical_unit_not_voltage=True)
        result.append(item)
    return result


def contained_member(parent, value):
    relative = PurePosixPath(value)
    require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in value, "unsafe source member")
    return str(PurePosixPath(parent) / relative)


def eeg_channels(channel_bytes, scales, *, document_unit_disagreement=False):
    declared = list(csv.DictReader(io.StringIO(channel_bytes.decode("utf-8-sig")), delimiter="\t"))
    source = {item["name"]: item for item in scales}
    require(len(source) == len(scales), "duplicate signal header channel")
    result = []
    for row in declared:
        if row["type"].upper() != "EEG":
            continue
        require(row["name"] in source, "EEG channel missing in header")
        item = source[row["name"]]
        require("gain_to_volts" in item, "EEG header needs an explicit physical voltage conversion")
        agrees = unit_to_volts(row["units"]) == unit_to_volts(item["header_unit"])
        require(agrees or document_unit_disagreement, "BIDS/header unit mismatch")
        result.append({**item, "index_in_file": list(source).index(row["name"]), "bids_unit": row["units"],
                       "bids_status": row.get("status"), "bids_and_header_unit_agree": agrees})
    require(bool(result) and len({item["name"] for item in result}) == len(result), "missing/duplicate BIDS EEG channel")
    return result


def verify_header_scale_against_frozen_window(data_path, header_config, channels, row, prepared_root, artifact, mne_raw):
    """Check numeric interpretation, without claiming independent calibration."""
    import numpy as np
    require(header_config["Common Infos"]["DataOrientation"] == "MULTIPLEXED"
            and header_config["Binary Infos"]["BinaryFormat"] == "IEEE_FLOAT_32", "unsupported direct header-scale check")
    n_channels = int(header_config["Common Infos"]["NumberOfChannels"])
    rate = 1e6 / float(header_config["Common Infos"]["SamplingInterval"])
    start = round(row["window_start_s"] * rate)
    samples = round((row["window_end_s"] - row["window_start_s"]) * rate)
    require(start >= 0 and samples == 448 and rate == 256, "fixed training window timing changed")
    numbers = np.fromfile(data_path, dtype="<f4", count=samples * n_channels, offset=start * n_channels * 4)
    require(numbers.size == samples * n_channels, "raw training scale check truncated")
    numbers = numbers.reshape(samples, n_channels).T.astype(np.float64)
    volts = np.stack([numbers[item["index_in_file"]] * item["gain_to_volts"] + item["offset_volts"] for item in channels]).astype(np.float32)
    path = Path(prepared_root) / contained_member("", row["eeg_path"])
    require(path.resolve().is_relative_to(Path(prepared_root).resolve()) and path.stat().st_size == artifact["bytes"] and sha(path) == artifact["sha256"], "sealed training artifact changed")
    expected = np.load(path, mmap_mode="r", allow_pickle=False)[int(row["eeg_index"])]
    require(np.array_equal(volts, expected), "header-based voltage conversion differs from sealed training window")
    decoded = mne_raw.get_data(picks=[item["name"] for item in channels], start=start, stop=start + samples).astype(np.float32)
    require(np.array_equal(decoded, expected), "fresh MNE header decoding differs from sealed training window")
    return {"status": "passed", "prepared_array_sha256": artifact["sha256"], "samples_compared_per_channel": samples,
            "channels_compared": len(channels), "sample_for_sample_equal": True, "fresh_mne_decode_equal": True,
            "physical_amplitude_independently_calibrated": False,
            "scope": "Header-based numeric-to-volts decoding reproduces existing frozen train inputs; conflicting BIDS units retained, no independent amplifier calibration claimed."}


def validate_portable_groups(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["participant_group"]].append(record)
    require(len(records) == 57 and len(groups) == 19, "portable full 57/19 membership changed")
    for items in groups.values():
        require(len(items) == 3 and {item["device_session"] for item in items} == {"BA", "EM", "LA"}, "same portable participant must retain all three device sessions")
    require(len({record["waveform_sha256"] for record in records}) == 57, "portable source content duplicate")
    return groups


def select_uniba_training(rows):
    require(len(rows) == 758 and len({row["id"] for row in rows}) == 758, "frozen UniBa full cohort changed")
    grouped, subjects = defaultdict(list), defaultdict(set)
    for row in rows:
        require(row["subject"].startswith("uniba:") and row["split"] in {"train", "val", "test"}, "ITU/unknown site or partition in UniBa cohort")
        grouped[row["eeg_source_sha256"]].append(row)
        subjects[row["subject"]].add(row["split"])
    require(all(len(parts) == 1 for parts in subjects.values()), "participant crosses partitions")
    require(all(len({row["split"] for row in items}) == 1 for items in grouped.values()), "raw source crosses partitions")
    counts = Counter(items[0]["split"] for items in grouped.values())
    require(counts == {"train": 18, "val": 6, "test": 8}, "expected UniBa 18/6/8 raw source membership changed")
    return [items[0] for items in grouped.values() if items[0]["split"] == "train"], [items[0] for items in grouped.values() if items[0]["split"] != "train"]


def build(args):
    started = time.monotonic()
    repo, out = Path(args.repo).resolve(), Path(args.output).resolve()
    require(not out.exists(), "inventory output must be new")
    helper_path = Path(__file__).with_name("audit_portable_eeg_zip.py")
    require(sha(helper_path) == HELPER_SHA, "header audit helper changed")
    spec = importlib.util.spec_from_file_location("fixed_portable_header_audit", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    archive = repo / "private/xavier/raw/zenodo-15698441-15698441/dataset.zip"
    audit_path = repo / "experiments/pj_xavier/reports/portable_eeg_15698441_audit.json"
    manifest = repo / "private/xavier/prepared/ai_coding_uniba_v3_natural_casecontrol_verified/manifest.jsonl"
    require(sha(audit_path) == PORTABLE_AUDIT_SHA and sha(manifest) == UNIBA_MANIFEST_SHA, "source audit/cohort hash mismatch")
    before = archive.stat()
    require(before.st_size == 3294723748 and sha(archive) == PORTABLE_ARCHIVE_SHA, "portable raw archive integrity changed")
    audit = json.loads(audit_path.read_text())
    require(audit["verified_recordings"] == 57 and audit["held_recordings"] == [] and audit["archive_sha256"] == PORTABLE_ARCHIVE_SHA, "prior raw audit incomplete")
    portable = []
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist()) == len(set(z.namelist())), "ambiguous ZIP member names")
        description_bytes = z.read("raw data/dataset_description.json")
        description = json.loads(description_bytes)
        for original in audit["records"]:
            stem = original["recording"]
            match = re.fullmatch(r"raw data/(sub-[^/]+)/ses-(BA|EM|LA)/eeg/[^/]+_eeg", stem)
            require(match is not None, "unrecognized raw portable BIDS member")
            participant, device = match.groups()
            header_name = stem + (".vhdr" if original["format"] == "BrainVision" else ".edf")
            header = z.read(header_name) if original["format"] == "BrainVision" else helper.zip_header(z, header_name)
            require(hashlib.sha256(header).hexdigest() == original["header_sha256"], "header differs from completed raw audit")
            sidecar_name, channels_name = stem + ".json", stem[:-4] + "_channels.tsv"
            sidecar_bytes, channels_bytes = z.read(sidecar_name), z.read(channels_name)
            sidecar = json.loads(sidecar_bytes)
            marker = None
            if original["format"] == "BrainVision":
                cfg, scales = brainvision_scales(header)
                data_name = contained_member(PurePosixPath(header_name).parent, cfg["Common Infos"]["DataFile"])
                marker = contained_member(PurePosixPath(header_name).parent, cfg["Common Infos"]["MarkerFile"])
                require(marker in z.namelist(), "BrainVision marker companion missing")
                meta = helper.brainvision_metadata(header, z.getinfo(data_name).file_size)
            else:
                scales, data_name = edf_scales(header), header_name
                meta = helper.edf_metadata(header, z.getinfo(data_name).file_size)
            channels = eeg_channels(channels_bytes, scales)
            require(len(channels) == original["eeg_channels"] == 32 and z.getinfo(data_name).file_size == original["member_bytes"], "portable EEG count/member bytes changed")
            require(abs(meta["duration_seconds"] - original["stored_signal_seconds"]) < 1e-8, "portable original duration differs")
            portable.append({"dataset_id": "zenodo:15698441", "recording_id": stem, "participant_group": "zenodo:15698441:" + participant,
                             "publisher_participant_id": participant, "device_session": device, "manufacturer": sidecar.get("Manufacturer"),
                             "source_format": original["format"], "archive_path": str(archive), "archive_sha256": PORTABLE_ARCHIVE_SHA,
                             "header_member": header_name, "waveform_member": data_name, "marker_member": marker,
                             "sidecar_member": sidecar_name, "channels_member": channels_name,
                             "waveform_sha256": original["signal_member_sha256"], "header_sha256": original["header_sha256"],
                             "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(), "channels_sha256": hashlib.sha256(channels_bytes).hexdigest(),
                             "sample_rate_hz": original["eeg_sample_rates_hz"][0], "eeg_channels": channels,
                             "channels_in_file": original["channels_in_file"], "stored_signal_seconds": original["stored_signal_seconds"],
                             "sidecar_signal_seconds": original["source_sidecar_seconds"], "waveform_member_bytes": original["member_bytes"],
                             "eeg_sample_bytes": original["eeg_sample_bytes"], "continuous": original["continuous"],
                             "missing_current_v3_channels": sorted(set(V3_CHANNELS) - {item["name"] for item in channels}),
                             "bids_license_literal": description.get("License"), "license_version_in_bids": "unspecified",
                             "selection_role": "unlabeled_ssl_candidate", "target_labels_supplied": False,
                             "split_grouping_rule": "all devices for a publisher participant stay together; no device-wise train/val split"})
        zip_all_uncompressed = sum(item.file_size for item in z.infolist() if not item.is_dir())
    validate_portable_groups(portable)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    allowed, excluded = select_uniba_training(rows)
    artifact_path = repo / "private/xavier/runs/smollm2_1_7b_uniba_natural_v3_fused_v1/eeg_artifacts.json"
    require(sha(artifact_path) == "0a2989891e46e2aacab785df162a84376f18acde444deb49e341a3b96969192f", "sealed training artifact index changed")
    artifacts = json.loads(artifact_path.read_text())
    import mne
    raw_root = repo / "private/xavier/verified_ai_coding_uniba_for_m1_v3"
    uniba = []
    for original in allowed:
        member = original["eeg_source_uri"].split("#", 1)[1]
        require(member.startswith("data/01_raw/uniba/"), "only UniBa training site may enter SSL inventory")
        data_path = raw_root / contained_member("", member)
        require(data_path.resolve().is_relative_to(raw_root.resolve()) and not data_path.is_symlink(), "raw path escapes fixed root")
        header_path = data_path.with_suffix(".vhdr")
        require(sha(data_path) == original["eeg_source_sha256"] and sha(header_path) == original["eeg_header_sha256"], "UniBa raw bytes differ from fixed train cohort")
        header = header_path.read_bytes()
        cfg, scales = brainvision_scales(header)
        require(cfg["Common Infos"]["DataFile"] == data_path.name, "UniBa header points at another waveform")
        channels_path = header_path.with_name(header_path.stem[:-4] + "_channels.tsv")
        channel_bytes = channels_path.read_bytes()
        channels = eeg_channels(channel_bytes, scales, document_unit_disagreement=True)
        meta = helper.brainvision_metadata(header, data_path.stat().st_size)
        require(len(channels) == 8 and [item["name"] for item in channels] == list(V3_CHANNELS) and meta["sample_rate_hz"] == 256, "UniBa train montage/sample rate changed")
        matching_train = [row for row in rows if row["split"] == "train" and row["eeg_source_sha256"] == original["eeg_source_sha256"]]
        raw = mne.io.read_raw_brainvision(header_path, preload=False, verbose="ERROR")
        try:
            checked = [verify_header_scale_against_frozen_window(data_path, cfg, channels, row, manifest.parent, artifacts[row["eeg_path"]], raw) for row in matching_train]
        finally:
            raw.close()
        scale_check = {"status": "passed", "windows_checked": len(checked), "samples_compared_per_channel": sum(item["samples_compared_per_channel"] for item in checked),
                       "channels_compared": 8, "sample_for_sample_equal": all(item["sample_for_sample_equal"] for item in checked),
                       "fresh_mne_decode_equal": all(item["fresh_mne_decode_equal"] for item in checked), "physical_amplitude_independently_calibrated": False,
                       "scope": "Every existing supervised training window in this raw source matches both fresh MNE header-to-volts and independent direct binary/header conversion; no claim of independent amplifier calibration."}
        uniba.append({"dataset_id": "zenodo:20801101", "canonical_dataset_id": "zenodo:20120211", "site": "uniba",
                      "recording_id": original["recording_id"], "participant_group": "zenodo:20120211:" + original["subject"],
                      "publisher_participant_id": original["subject"], "source_format": "BrainVision",
                      "waveform_path": str(data_path), "header_path": str(header_path), "channels_path": str(channels_path),
                      "waveform_sha256": original["eeg_source_sha256"], "header_sha256": original["eeg_header_sha256"],
                      "channels_sha256": hashlib.sha256(channel_bytes).hexdigest(), "channels_in_file": meta["channels"], "eeg_channels": channels,
                      "sample_rate_hz": meta["sample_rate_hz"], "samples_per_channel": meta["samples_per_channel"], "stored_signal_seconds": meta["duration_seconds"],
                      "waveform_member_bytes": data_path.stat().st_size, "eeg_sample_bytes": meta["bytes_per_channel"] * len(channels),
                      "source_supervised_partition": "train", "selection_role": "unlabeled_ssl_candidate", "target_labels_supplied": False,
                      "source_manifest_sha256": UNIBA_MANIFEST_SHA, "raw_waveform_rehashed_this_run": True,
                      "bids_header_unit_disagreement": any(not item["bids_and_header_unit_agree"] for item in channels),
                      "header_scale_validation": scale_check,
                      "unit_conversion_policy_to_pin_before_training": "Use BrainVision resolution times header physical unit to decode volts; do not reinterpret raw numbers as volts from channels.tsv or apply conversion twice."})
    source_hashes = [row["waveform_sha256"] for row in portable + uniba]
    require(len(set(source_hashes)) == len(source_hashes), "cross-release source content overlap; quarantine before SSL")
    require(len(uniba) == 18 and len({row["participant_group"] for row in uniba}) == 18, "UniBa train raw/participant count differs")
    after = archive.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), "archive changed during metadata inventory")
    out.mkdir(parents=True, exist_ok=False)
    for name, records in (("portable_records.jsonl", portable), ("uniba_train_records.jsonl", uniba)):
        with (out / name).open("x") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    exclusion = {"uniba_excluded_recordings": [{"recording_id": row["recording_id"], "participant_group": "zenodo:20120211:" + row["subject"],
                   "waveform_sha256": row["eeg_source_sha256"], "original_partition": row["split"]} for row in excluded],
                 "itu_all_excluded": True, "itu_reason": "whole external evaluation site is forbidden for SSL fitting, selection or threshold tuning",
                 "source_manifest_sha256": UNIBA_MANIFEST_SHA}
    (out / "exclusions.json").write_text(json.dumps(exclusion, indent=2) + "\n")
    by_device = {}
    for device in ("BA", "EM", "LA"):
        items = [row for row in portable if row["device_session"] == device]
        by_device[device] = {"recordings": len(items), "sample_rates_hz": sorted({row["sample_rate_hz"] for row in items}),
                             "stored_signal_seconds": sum(row["stored_signal_seconds"] for row in items),
                             "waveform_member_bytes": sum(row["waveform_member_bytes"] for row in items),
                             "eeg_sample_bytes": sum(row["eeg_sample_bytes"] for row in items),
                             "missing_current_v3_channels": sorted({name for row in items for name in row["missing_current_v3_channels"]})}
    totals = lambda items: {"recordings": len(items), "participant_groups": len({row["participant_group"] for row in items}),
                            "stored_signal_seconds": sum(row["stored_signal_seconds"] for row in items),
                            "stored_signal_hours": sum(row["stored_signal_seconds"] for row in items) / 3600,
                            "eeg_sample_bytes": sum(row["eeg_sample_bytes"] for row in items),
                            "waveform_member_bytes": sum(row["waveform_member_bytes"] for row in items)}
    summary = {"schema": "xavier-unlabeled-source-inventory-v1", "status": "inventory_verified_only", "source_sha256": sha(__file__),
               "portable": totals(portable), "portable_by_device": by_device, "uniba_train": totals(uniba), "combined_candidate": totals(portable + uniba),
               "portable_archive_bytes": archive.stat().st_size, "portable_archive_sha256": PORTABLE_ARCHIVE_SHA,
               "portable_all_zip_uncompressed_bytes_including_nonraw": zip_all_uncompressed, "portable_original_audit_sha256": PORTABLE_AUDIT_SHA,
               "original_manifest_sha256": UNIBA_MANIFEST_SHA, "portable_archive_rehashed_this_run": True, "portable_signal_crc_previously_verified": True,
               "uniba_train_waveforms_freshly_rehashed": len(uniba), "uniba_validation_sources_excluded": 6, "uniba_test_sources_excluded": 8,
               "uniba_sources_with_bids_header_unit_disagreement": sum(row["bids_header_unit_disagreement"] for row in uniba),
               "uniba_header_scaling_equal_to_frozen_train_inputs": all(row["header_scale_validation"]["sample_for_sample_equal"] for row in uniba),
               "uniba_fresh_mne_training_windows_verified": sum(row["header_scale_validation"]["windows_checked"] for row in uniba),
               "uniba_fresh_mne_training_sample_values_verified": sum(row["header_scale_validation"]["samples_compared_per_channel"] * 8 for row in uniba),
               "itu_all_excluded": True, "new_raw_copies_or_extractions": 0, "model_training_started": False, "target_labels_supplied": False,
               "inputs_not_modified": True, "elapsed_s": time.monotonic() - started,
               "source_identity_limit": "Dataset/site-qualified publisher groups are used. Human identity overlap between datasets is unknown. Three portable device sessions never become three independent participants.",
               "quantity_scope": "Original stored signal durations, not channel-hours, unique human clock-time, accepted training windows, repeated-epoch exposure, or additive copies of archives and samples.",
               "next_gate": "Inventory only. Fix SSL protocol, source/participant grouping, channel mapping/unit conversion and storage/compute budget before extraction or fitting. The portable montage does not contain all frozen v3 channels; do not silently fabricate missing channels.",
               "files": {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size} for name in ("portable_records.jsonl", "uniba_train_records.jsonl", "exclusions.json")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
