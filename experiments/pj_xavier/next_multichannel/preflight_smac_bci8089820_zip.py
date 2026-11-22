#!/usr/bin/env python3
"""Read-only S Mac preflight of the pinned Zenodo8089820 ZIP.

Never open a .part; never decode EEG; never convert missing event timebase
into ground truth. Only a completed, byte-for-byte pinned source can advance
to later train-only physical-QC admission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import socket
import sqlite3
import sys
import zipfile

PINNED_SHA = "7c611b08995f10c2d313d194e3e75c88f7792403455994d77ca37e16104fcacf"
PINNED_BYTES = 27475466178
PINNED_MEMBER_COUNT = 1071
PINNED_DIRECTORY_COUNT = 96
FROZEN_MANIFEST_SHA = "91d9c764dcf27ac7d85be5cfd90c1484d5c7d41e4d5e0c789c49049b04b461bb"
SOURCE_KEY = "zenodo-8089820-8089820"


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory_frozen(manifest):
    raw = manifest.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == FROZEN_MANIFEST_SHA, "Frozen metadata manifest changed")
    d = json.loads(raw)
    members = [x["source_member"] for x in d["members"]]
    require(len(members) == 694 and len(set(members)) == 694, "Frozen member set differs")
    require(len(d["group_assignments"]) == 87, "Frozen source groups differ")
    require(all(x["source_event_timebase_verified"] is False and
                x["supervised_cue_training_allowed"] is False for x in d["members"]),
            "Unsafe event-timebase promotion")
    require(d["summary"]["offline_ssl_training_allowed_now"] is False, "Fit already authorized unexpectedly")
    return members


def inspect_zip_names(zf, members):
    files = zf.infolist()
    require(sum(not x.is_dir() for x in files) == PINNED_MEMBER_COUNT, "Unexpected number of ZIP file entries")
    require(sum(x.is_dir() for x in files) == PINNED_DIRECTORY_COUNT, "Unexpected number of ZIP directory entries")
    names = [x.filename for x in files]
    require(len(set(names)) == len(names), "Duplicate ZIP entry name")
    for name in names:
        pp = PurePosixPath(name)
        require(not pp.is_absolute() and ".." not in pp.parts and
                all(part not in ("", ".") for part in pp.parts),
                "Unsafe ZIP entry path")
    require(set(members).issubset(names), "A frozen original GDF member is absent")
    return {"member_count": len(names), "file_member_count": PINNED_MEMBER_COUNT, "directory_member_count": PINNED_DIRECTORY_COUNT, "frozen_gdf_member_count": len(members)}


def run(node_root, frozen_manifest):
    members = inventory_frozen(frozen_manifest)
    dbpath = node_root / "downloads.sqlite"
    require(dbpath.is_file(), "Missing actual node download ledger")
    with sqlite3.connect("file:" + str(dbpath) + "?mode=ro", uri=True) as db:
        found = db.execute(
            "select dataset_key,relative_path,status,expected_bytes,bytes_present "
            "from downloads where dataset_key=?", (SOURCE_KEY,)).fetchall()
    require(len(found) == 1, "Pinned archive ledger row absent or ambiguous")
    key, relative, status, expected, present = found[0]
    require(relative == "BCI Database.zip" and expected == PINNED_BYTES, "Pinned ZIP ledger identity changed")
    data = node_root / "raw" / key / relative
    part = data.with_name(data.name + ".part")
    if status != "complete":
        return {
            "kind": "preflight", "status": "blocked_source_archive_incomplete",
            "eligible_for_fit": False, "source_eeg_files_read": 0,
            "ledger_status": status, "expected_bytes": expected,
            "ledger_bytes_present": present,
            "physical_part_bytes": part.stat().st_size if part.exists() else 0,
            "frozen_metadata_sha256": FROZEN_MANIFEST_SHA,
        }
    require(data.is_file() and data.stat().st_size == PINNED_BYTES and present == PINNED_BYTES,
            "Complete ledger and actual ZIP file mismatch")
    whole_sha = sha256_file(data)
    require(whole_sha == PINNED_SHA, "Complete ZIP content differs from the pinned release")
    with zipfile.ZipFile(data, "r") as zf:
        index = inspect_zip_names(zf, members)
    return {
        "kind": "preflight", "status": "ready_for_train_only_physical_qc",
        "eligible_for_fit": False, "source_eeg_files_read": 0,
        "source_zip_sha256": whole_sha, "frozen_metadata_sha256": FROZEN_MANIFEST_SHA,
        **index,
        "pending": ["train-only original GDF source SHA/payload/calibration", "unit and fixed QC",
                    "cross-release duplicate/heldout deny", "MPS backward probe",
                    "separate offline SSL and supervised-cue eligibility"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node-root", type=Path, required=True)
    p.add_argument("--frozen-manifest", type=Path, required=True)
    p.add_argument("--allow-test-host", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    host = socket.gethostname().split(".")[0]
    require(host == "smac05" or a.allow_test_host, "Only the authorized S Mac node may inspect source")
    result = run(a.node_root, a.frozen_manifest)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    sys.exit(2 if result["status"] == "blocked_source_archive_incomplete" else 0)


if __name__ == "__main__":
    main()
