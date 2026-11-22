"""CPU-only regression tests for archive duration/integrity boundaries."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import unittest

from audit_brainvision_archive import audit, archive_inventory, parse_header, recording_from_header


HEADER = b"""BrainVision Data Exchange Header File Version 1.0
[Common Infos]
Codepage=UTF-8
DataFile=sub-P000_eeg.eeg
DataFormat=BINARY
DataOrientation=MULTIPLEXED
NumberOfChannels=2
SamplingInterval=10000
[Binary Infos]
BinaryFormat=IEEE_FLOAT_32
[Channel Infos]
Ch1=C3,,0.1,uV
Ch2=otree_phase,,1,unit
"""


class ArchiveAuditTests(unittest.TestCase):
    def fixture(self, root, payload=None, extra=None, complete=True):
        base = "data/01_raw/site/sub-P000/ses-S001/eeg/"
        members = {base + "sub-P000_eeg.vhdr": HEADER,
                   base + "sub-P000_eeg.eeg": bytes(2 * 100 * 5 * 4) if payload is None else payload,
                   base + "sub-P000_channels.tsv": b"name\ttype\nC3\tEEG\notree_phase\tMISC\n",
                   base + "sub-P000_eeg.json": b'{"RecordingDuration": 5, "EEGChannelCount": 1}'}
        members.update(extra or {})
        archive = root / "dataset.tar.gz"
        with tarfile.open(archive, "w:gz") as f:
            for name, data in members.items():
                item = tarfile.TarInfo(name); item.size = len(data)
                f.addfile(item, io.BytesIO(data))
        contents = archive.read_bytes()
        db = root / "ledger.sqlite"
        md5, sha = hashlib.md5(contents).hexdigest(), hashlib.sha256(contents).hexdigest()
        with sqlite3.connect(db) as c:
            c.execute("create table downloads(relative_path text,expected_bytes int,checksum_json text,status text,sha256 text,updated_at text)")
            c.execute("insert into downloads values(?,?,?,?,?,?)", (archive.name, len(contents), json.dumps({"algorithm": "md5", "value": md5}), "complete" if complete else "running", sha if complete else None, "fixture"))
        return argparse.Namespace(archive=str(archive), ledger=str(db), relative_path=archive.name,
                                  expected_bytes=len(contents), expected_md5=md5,
                                  recording_prefix="data/01_raw/site",
                                  source="synthetic-test-only", output_dir=str(root / "out"))

    def test_exact_duration_and_auxiliary_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, code = audit(self.fixture(Path(tmp)))
            self.assertEqual(code, 0)
            self.assertEqual(result["unique_binary_recording_hours"], 5 / 3600)
            self.assertEqual(result["physical_eeg_channel_counts"], {"1": 1})
            self.assertFalse(result["training_eligibility_established"])

    def test_truncated_frame_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, code = audit(self.fixture(Path(tmp), payload=bytes(3999)))
            self.assertEqual(code, 2)
            self.assertEqual(result["headers_blocked"], 1)
            self.assertEqual(result["unique_binary_recording_hours"], 0)

    def test_same_binary_payload_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            extra = {"data/01_raw/site/another/sub-P000_eeg.vhdr": HEADER,
                     "data/01_raw/site/another/sub-P000_eeg.eeg": bytes(4000)}
            result, code = audit(self.fixture(Path(tmp), extra=extra))
            self.assertEqual(code, 0)
            self.assertEqual(result["headers_parsed"], 2)
            self.assertEqual(result["unique_binary_payloads"], 1)
            self.assertEqual(result["summed_header_recording_hours"], 10 / 3600)
            self.assertEqual(result["unique_binary_recording_hours"], 5 / 3600)

    def test_derived_headers_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            extra = {"data/02_processed/sub-P000_eeg.vhdr": HEADER,
                     "data/02_processed/sub-P000_eeg.eeg": bytes(4000)}
            result, code = audit(self.fixture(Path(tmp), extra=extra))
            self.assertEqual(code, 0)
            self.assertEqual(result["headers_total"], 1)
            self.assertEqual(result["headers_outside_prefix_excluded"], 1)

    def test_partial_ledger_does_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, code = audit(self.fixture(Path(tmp), complete=False))
            self.assertEqual(code, 3)
            self.assertIsNone(result["recording_hours"])
            self.assertFalse((Path(tmp) / "out/recording_audit.json").exists())

    def test_changed_archive_rejected_before_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            path = Path(args.archive)
            data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "published size/MD5"):
                audit(args)

    def test_datafile_escape_and_unknown_format_rejected(self):
        with self.assertRaisesRegex(ValueError, "same archive directory"):
            parse_header(HEADER.replace(b"DataFile=sub", b"DataFile=../sub"))
        with self.assertRaisesRegex(ValueError, "Unsupported binary"):
            parse_header(HEADER.replace(b"IEEE_FLOAT_32", b"UNKNOWN"))

    def test_channel_order_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            inventory, contents = archive_inventory(args.archive)
            name = next(n for n in contents if n.endswith("_channels.tsv"))
            contents[name] = b"name\ttype\notree_phase\tMISC\nC3\tEEG\n"
            with self.assertRaisesRegex(ValueError, "ordering differs"):
                recording_from_header(next(n for n in contents if n.endswith(".vhdr")), inventory, contents)

    def test_tar_member_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp), extra={"../unsafe.vhdr": HEADER})
            with self.assertRaisesRegex(ValueError, "Unsafe archive"):
                audit(args)


if __name__ == "__main__":
    unittest.main()
