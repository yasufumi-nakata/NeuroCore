"""Safety and fail-closed admission tests for the S Mac Zenodo8089820 ZIP."""
import json
import hashlib
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).absolute().parent))
import preflight_smac_bci8089820_zip as pre

ROOT = Path(__file__).absolute().parents[3]
MANIFEST = ROOT / "private/xavier/smac_execution_20261122/bci8089820_metadata/frozen_split_v1.json"


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "raw" / pre.SOURCE_KEY / "BCI Database.zip"
        self.data.parent.mkdir(parents=True)
        self.db = sqlite3.connect(str(self.root / "downloads.sqlite"))
        self.addCleanup(self.db.close)
        self.db.execute("CREATE TABLE downloads (dataset_key TEXT,relative_path TEXT,status TEXT,expected_bytes INTEGER,bytes_present INTEGER)")
        self.db.execute("INSERT INTO downloads VALUES (?,?,?,?,?)", (pre.SOURCE_KEY, "BCI Database.zip", "running", pre.PINNED_BYTES, 9))
        self.db.commit()

    def test_manifest_provenance_and_deny(self):
        members = pre.inventory_frozen(MANIFEST)
        self.assertEqual(len(members), 694)
        self.assertEqual(len(set(members)), 694)

    def test_incomplete_archive_never_inspected(self):
        self.data.with_name(self.data.name + ".part").write_bytes(b"not a zip")
        r = pre.run(self.root, MANIFEST)
        self.assertEqual(r["status"], "blocked_source_archive_incomplete")
        self.assertFalse(r["eligible_for_fit"])
        self.assertEqual(r["source_eeg_files_read"], 0)
        self.assertEqual(r["physical_part_bytes"], 9)

    def test_wrong_identity_fails(self):
        self.db.execute("UPDATE downloads SET relative_path='unexpected.zip'")
        self.db.commit()
        with self.assertRaisesRegex(ValueError, "identity"):
            pre.run(self.root, MANIFEST)

    def test_mock_complete_zip_index_only(self):
        members = pre.inventory_frozen(MANIFEST)
        with zipfile.ZipFile(self.data, "w", compression=zipfile.ZIP_STORED) as f:
            for name in members:
                f.writestr(name, b"")
            for i in range(1071 - len(members)):
                f.writestr(f"BCI Database/other/{i:03d}.txt", b"")
            for i in range(96):
                f.writestr(f"BCI Database/directory{i:03d}/", b"")
        pre.PINNED_BYTES = self.data.stat().st_size
        pre.PINNED_SHA = hashlib.sha256(self.data.read_bytes()).hexdigest()
        self.db.execute("UPDATE downloads SET status='complete',expected_bytes=?,bytes_present=?",
                        (pre.PINNED_BYTES, pre.PINNED_BYTES))
        self.db.commit()
        result = pre.run(self.root, MANIFEST)
        self.assertEqual(result["status"], "ready_for_train_only_physical_qc")
        self.assertEqual(result["member_count"], 1167)
        self.assertEqual(result["file_member_count"], 1071)
        self.assertFalse(result["eligible_for_fit"])
        self.assertEqual(result["source_eeg_files_read"], 0)


if __name__ == "__main__":
    unittest.main()
