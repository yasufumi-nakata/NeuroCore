#!/usr/bin/env python3
"""Read-only, smac-only audit of Figshare 9970619 EEGLAB .set headers.

Only external .set metadata are read. No .fdt signal bytes, epoch labels,
or source unit conversions are ingested or inferred. The audit never grants
training eligibility; original source evidence and a new scientific protocol
are required before an independent training stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import socket


def approved_node(host: str) -> bool:
    short = host.split(".", 1)[0].lower()
    return re.fullmatch(r"smac[0-9]{2}", short) is not None and short not in {"smac19", "smac31", "smac40"}


def checked_labels(eeg) -> list[str]:
    import numpy as np

    raw = np.atleast_1d(eeg.chanlocs).tolist()
    if not isinstance(raw, list):
        raw = [raw]
    labels = [str(getattr(item, "labels", "")) for item in raw]
    if len(labels) != 73 or len(set(labels)) != 73:
        raise ValueError("Missing, duplicated or changed 73 channel labels")
    if labels[64:] != [*(f"EXG{i}" for i in range(1, 9)), "Status"]:
        raise ValueError("AUX/Status boundary changed; cannot treat all 73 as EEG")
    if any(x.startswith("EXG") or x == "Status" for x in labels[:64]):
        raise ValueError("EEG channels cannot contain EXG/Status")
    return labels


def inspect_headers(directory: Path, *, host: str, expected_count: int = 44) -> dict:
    if not approved_node(host):
        raise ValueError("S Mac worker host required; smac19 excluded")
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Expected a real local header directory")
    from scipy.io import loadmat
    import numpy as np

    paths = sorted(directory.glob("*.set"))
    if len(paths) != expected_count:
        raise ValueError(f"Expected {expected_count} EEGLAB .set files, got {len(paths)}")
    rows = []
    for path in paths:
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size < 1_500_000:
            raise ValueError("Unexpected EEGLAB header file or size")
        item = loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=["EEG"],
                       verify_compressed_data_integrity=True)
        eeg = item.get("EEG")
        if eeg is None or float(eeg.srate) != 200 or int(eeg.nbchan) != 73 or int(eeg.trials) != 1:
            raise ValueError("Unexpected recording structure or sampling rate")
        labels = checked_labels(eeg)
        fdt = str(eeg.data)
        if fdt != path.with_suffix(".fdt").name or not fdt.endswith(".fdt"):
            raise ValueError("Unexpected external signal reference")
        units = {str(getattr(item, "unit", "")) for item in np.atleast_1d(eeg.chanlocs)
                 if getattr(item, "unit", None) not in (None, "")}
        ev = getattr(eeg, "event", None)
        has_event = bool(np.asarray(ev).size) if ev is not None else False
        expected_fdt_size = int(eeg.nbchan) * int(eeg.pnts) * 4
        rows.append({
            "set_name": path.name,
            "set_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "set_bytes": path.stat().st_size,
            "samples": int(eeg.pnts),
            "seconds": int(eeg.pnts) / float(eeg.srate),
            "channel_labels_sha256": hashlib.sha256(json.dumps(labels).encode()).hexdigest(),
            "reference_label": str(eeg.ref),
            "event_present": has_event,
            "explicit_unit_fields": sorted(units),
            "paired_fdt_name": fdt,
            "paired_fdt_expected_bytes_float32": expected_fdt_size,
        })
    n_subject_tokens = len({row["set_name"].split("_")[0] for row in rows})
    return {
        "schema": "xavier-ssvep9970619-metadata-only-audit-v1",
        "scope": "44 EEG .set headers on S Mac; paired signal bytes not opened",
        "header_files_verified": len(rows),
        "subject_name_tokens_not_validated_people": n_subject_tokens,
        "first64_eeg_channels": True,
        "auxiliary_last9": [*(f"EXG{i}" for i in range(1, 9)), "Status"],
        "native_header_rate_hz": 200,
        "duration_seconds_all_recordings": sum(row["seconds"] for row in rows),
        "header_bytes": sum(row["set_bytes"] for row in rows),
        "nominal_paired_fdt_bytes": sum(row["paired_fdt_expected_bytes_float32"] for row in rows),
        "channel_montage_variants": len({row["channel_labels_sha256"] for row in rows}),
        "references": sorted({row["reference_label"] for row in rows}),
        "event_headers_present": sum(row["event_present"] for row in rows),
        "explicit_voltage_unit_fields_present": sum(bool(row["explicit_unit_fields"]) for row in rows),
        "fit_allowed": False,
        "heldout_and_train_split_assigned": False,
        "new_training_seconds": 0,
        "status": "hold_missing_unit_evidence_and_frozen_split",
        "scientific_limits": [
            "EEGLAB conventional microvolt assumption is not an observed per-record physical-V calibration",
            "Original offline resampling and referencing cannot certify online causal preprocessing",
            "No observed event timing gold in 44 headers; classification labels cannot be invented",
            "The source 28,817 s is acquired recording time, not eligible train or gradient exposure",
        ],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headers-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_headers(args.headers_dir, host=socket.gethostname())
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Immutable audit output exists")
    if args.output.is_symlink():
        raise ValueError("Output symlink prohibited")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
