import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

spec = importlib.util.spec_from_file_location("report_progress", Path(__file__).with_name("report_progress.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture(tmp_path):
    root = tmp_path / "raw"
    (root / "release").mkdir(parents=True)
    target = root / "release/a.eeg"
    target.write_bytes(b"abcd")
    ledger = tmp_path / "ledger.sqlite"
    with sqlite3.connect(ledger) as db:
        db.execute("CREATE TABLE downloads(id TEXT,catalog_id TEXT,dataset_key TEXT,relative_path TEXT,status TEXT,bytes_present INTEGER,expected_bytes INTEGER,sha256 TEXT)")
        db.execute("INSERT INTO downloads VALUES (?,?,?,?,?,?,?,?)", ("1", "000001", "release", "a.eeg", "complete", 4, 4, hashlib.sha256(b"abcd").hexdigest()))
    return ledger, root, target


def test_replicas_deduplicated_by_actual_content(tmp_path):
    ledger, root, _ = fixture(tmp_path)
    first = report.snapshot(ledger, root, "colab", True)
    second = report.snapshot(ledger, root, "workstation", True)
    result = report.summarize([first, second])
    assert result["completed_file_copy_bytes"] == 8
    assert result["completed_unique_sha256_bytes"] == 4
    assert result["recording_hours"] is None


def test_same_size_tampering_not_counted_complete(tmp_path):
    ledger, root, target = fixture(tmp_path)
    target.write_bytes(b"evil")
    observed = report.snapshot(ledger, root, "workstation", True)
    assert observed["complete_files"] == []
    assert "SHA-256" in observed["held_files"][0]["reason"]


def test_inflight_partial_uses_actual_size_not_stale_ledger(tmp_path):
    ledger, root, _ = fixture(tmp_path)
    (root / "release/b.eeg.part").write_bytes(b"123")
    with sqlite3.connect(ledger) as db:
        db.execute("INSERT INTO downloads VALUES (?,?,?,?,?,?,?,?)", ("2", "000001", "release", "b.eeg", "running", 0, 10, None))
    observed = report.snapshot(ledger, root, "workstation", True)
    result = report.summarize([observed])
    assert result["completed_unique_sha256_bytes"] == 4
    assert result["locations"][0]["partial_observed_bytes"] == 3


def test_duplicate_location_rejected(tmp_path):
    ledger, root, _ = fixture(tmp_path)
    snap = report.snapshot(ledger, root, "workstation", True)
    with pytest.raises(ValueError, match="duplicate location"):
        report.summarize([snap, snap])


def test_outside_or_symlink_path_rejected(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    with pytest.raises(ValueError, match="unsafe"):
        report.safe_destination(root, "release", "../../outside")
    (root / "release").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        report.safe_destination(root, "release", "outside")


def test_streamed_ledger_hash_matches_canonical_array(tmp_path):
    ledger, root, _ = fixture(tmp_path)
    with sqlite3.connect(ledger) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute("SELECT * FROM downloads ORDER BY id")]
    expected = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert report.snapshot(ledger, root, "mac", True)["ledger_snapshot_sha256"] == expected


def test_access_held_partial_not_lost_from_inventory(tmp_path):
    ledger, root, _ = fixture(tmp_path)
    (root / "release/b.eeg.part").write_bytes(b"123")
    with sqlite3.connect(ledger) as db:
        db.execute("INSERT INTO downloads VALUES (?,?,?,?,?,?,?,?)", ("2", "000001", "release", "b.eeg", "access_action_required", 3, 10, None))
    snap = report.snapshot(ledger, root, "mac", True)
    assert snap["partial_files"][0]["bytes"] == 3
    assert snap["partial_files"][0]["status"] == "access_action_required"
