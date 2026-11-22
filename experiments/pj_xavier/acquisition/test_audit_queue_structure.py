import gzip
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import audit_queue_structure as audit


class StructureAuditTests(unittest.TestCase):
    def setUp(self):
        self.classifier = audit.Classifier(json.loads(audit.RULES_PATH.read_text()))
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def category(self, path):
        return self.classifier.classify(path)[0]

    def database(self, rows):
        file = self.root / "jobs.sqlite"
        with sqlite3.connect(file) as db:
            db.execute("CREATE TABLE downloads (id TEXT PRIMARY KEY,catalog_id TEXT,dataset_key TEXT,relative_path TEXT,original_relative_path TEXT,expected_bytes INTEGER,url TEXT,status TEXT)")
            for i, (release, path, size, original) in enumerate(rows):
                db.execute("INSERT INTO downloads VALUES (?,?,?,?,?,?,?,?)", (str(i), "001", release, path, original, size,
                           "https://private.example/file?token=DO_NOT_PUBLISH", "queued"))
        return file

    def test_eeg_waveform_and_essential_sidecars_are_separate(self):
        cases = {"sub-01/eeg/sub-01_task-rest_eeg.edf": "eeg_path_signal",
                 "sub-01/eeg/sub-01_task-rest_eeg.eeg": "eeg_path_signal",
                 "sub-01/eeg/sub-01_task-rest_eeg.vhdr": "eeg_path_ancillary",
                 "sub-01/eeg/sub-01_task-rest_eeg.vmrk": "eeg_path_ancillary",
                 "sub-01/eeg/sub-01_task-rest_eeg.json": "eeg_path_ancillary",
                 "sub-01/eeg/sub-01_channels.tsv": "eeg_path_ancillary",
                 "sub-01/eeg/sub-01_timeseries.csv": "eeg_path_numeric"}
        for path, category in cases.items():
            with self.subTest(path=path): self.assertEqual(self.category(path), category)

    def test_generic_signal_formats_never_prove_scalp_eeg(self):
        for path in ("sample.edf", "sample.bdf", "sample.eeg", "recording.mff/signal1.bin"):
            with self.subTest(path=path): self.assertEqual(self.category(path), "generic_eeg_format_ambiguous")
        self.assertEqual(self.category("sample.fif"), "generic_meeg_format_ambiguous")
        self.assertEqual(self.category("recording.nwb"), "generic_neurophysiology_container")

    def test_meg_and_ieeg_are_separate_with_no_claim_that_meg_excludes_eeg(self):
        self.assertEqual(self.category("sub-01/meg/sub-01_meg.fif"), "meg_path_signal")
        self.assertEqual(self.category("sub-01/meg/sub-01_meg.ds/sub-01.meg4"), "meg_path_signal")
        self.assertEqual(self.category("sub-01/ieeg/sub-01_ieeg.edf"), "ieeg_path_signal")
        review, step = audit.release_review({"meg_path_signal": 1})
        self.assertEqual(review, "meg_container_requires_eeg_channel_check")
        self.assertIn("EEGChannelCount", step)

    def test_archives_and_other_payloads_override_eeg_folder(self):
        for path in ("eeg/eeg.zip", "eeg/sub.tar.gz", "eeg/sub.7z.001"):
            with self.subTest(path=path): self.assertEqual(self.category(path), "archive_uninspected")
        self.assertEqual(self.category("eeg/recording.mp4"), "video_payload")
        self.assertEqual(self.category("eeg/sub_T1w.nii.gz"), "neuroimage_payload")
        self.assertEqual(self.category("eeg/electrode_photo.png"), "image_payload")
        self.assertEqual(self.category("eeg/sub_eeg.edf.gz"), "eeg_path_signal")

    def test_physio_table_and_conflicting_modality_are_not_promoted(self):
        self.assertEqual(self.category("eeg/sub-01_physio.tsv.gz"), "other_physio_signal_or_array")
        self.assertEqual(self.category("eeg/sub-01_ecg.edf"), "other_physio_signal_or_array")
        self.assertEqual(self.category("eeg/sub-01_ieeg.edf"), "modality_path_conflict")
        self.assertEqual(self.category("allegedlyeeg/sub-01.bdf"), "generic_eeg_format_ambiguous")

    def test_full_partition_sums_source_unchanged_no_urls_and_private_evidence(self):
        db = self.database([("openneuro-ds1-v1", "sub/eeg/sub_eeg.edf", 100, None),
                            ("openneuro-ds1-v1", "sub/anat/sub_T1w.nii.gz", 900, None),
                            ("figshare-2-v1", "provider-file-5/renamed", None, "EEG.zip"),
                            ("figshare-2-v1", "data.edf", 0, None),
                            ("zenodo-3-v1", "sub/meg/sub_meg.fif", 200, None)])
        original = audit.sha256(db)
        report = self.root / "report"
        detail = self.root / "private/classifications.jsonl.gz"
        result = audit.audit(db, report, private_detail=detail)
        self.assertEqual(audit.sha256(db), original)
        self.assertEqual(result["totals"], {"files": 5, "known_bytes": 1200, "unknown_size_files": 1, "zero_size_files": 1, "releases": 3})
        for partition in ("category_counts", "exclusive_groups", "extension_counts", "provider_counts", "release_review_groups"):
            with self.subTest(partition=partition):
                self.assertEqual(sum(v["files"] for v in result[partition].values()), 5)
                self.assertEqual(sum(v["known_bytes"] for v in result[partition].values()), 1200)
                self.assertEqual(sum(v["unknown_size_files"] for v in result[partition].values()), 1)
        self.assertEqual(sum(v["releases"] for v in result["release_review_groups"].values()), 3)
        self.assertGreater(sum(v["releases"] for v in result["category_counts"].values()), 3)
        self.assertEqual(result["header_validated_by_this_audit"], 0)
        self.assertFalse(result["queue_modified"])
        for path in report.iterdir():
            self.assertNotIn("https://", path.read_text())
            self.assertNotIn("DO_NOT_PUBLISH", path.read_text())
        with gzip.open(detail, "rt") as f: rows = [json.loads(x) for x in f]
        archived = next(r for r in rows if r["category"] == "archive_uninspected")
        self.assertEqual(archived["relative_path_evidence"], "EEG.zip")
        self.assertEqual(archived["path_field"], "original_relative_path")

    def test_invalid_size_fails_without_modifying_queue(self):
        db = self.database([("zenodo-1-v1", "a.edf", -1, None)])
        original = audit.sha256(db)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            audit.audit(db, self.root / "report")
        self.assertEqual(audit.sha256(db), original)

    def test_existing_output_cannot_be_overwritten(self):
        db = self.database([("zenodo-1-v1", "a.edf", 1, None)])
        output = self.root / "report"; output.mkdir()
        marker = output / "existing.txt"; marker.write_text("retain")
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            audit.audit(db, output)
        self.assertEqual(marker.read_text(), "retain")


if __name__ == "__main__":
    unittest.main()
