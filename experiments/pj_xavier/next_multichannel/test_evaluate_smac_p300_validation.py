"""Security and split invariants for smac-only P300 validation diagnostic."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate_smac_p300_validation as ev


class TestP300ValidationGuard(unittest.TestCase):
    def test_workstation_rejected_before_any_files(self):
        with patch.object(ev.platform, "node", return_value="vpnclient231"):
            with self.assertRaisesRegex(ValueError, "Only approved S Mac nodes"):
                ev.frozen_preflight({})

    def test_smac19_rejected(self):
        with patch.object(ev.platform, "node", return_value="smac19.sfc.keio.ac.jp"):
            with self.assertRaisesRegex(ValueError, "Only approved S Mac nodes"):
                ev.frozen_preflight({})

    def test_unrecognized_protocol_rejected(self):
        with patch.object(ev.platform, "node", return_value="smac35"):
            with self.assertRaisesRegex(ValueError, "Unrecognized frozen evaluation protocol"):
                ev.frozen_preflight({"schema": "arbitrary"})

    def test_no_test_or_checkpoint_selection_may_be_disabled(self):
        cfg = {"schema": "xavier-p300-heldout-checkpoint-validation-v1",
               "holdout_group": "openneuro:ds005028:sub-11",
               "excluded_test_groups": [
                   "openneuro:ds005028:sub-01",
                   "openneuro:ds005028:sub-10"],
               "fixed_seed": 20261122, "fixed_mask": True,
               "no_checkpoint_selection": True, "no_test_read": False}
        with patch.object(ev.platform, "node", return_value="smac35"):
            with self.assertRaisesRegex(ValueError, "Diagnostic must not select"):
                ev.frozen_preflight(cfg)

    def _config(self, root, train_group="different", train_source="train_source"):
        manifest = root / "manifest.jsonl"
        manifest.write_text(json.dumps({"split": "train", "subject_group": train_group,
                                        "source_sha256": train_source}) + "\n")
        return {"source_ledger": str(root / "nonexistent-ledger"),
                "metadata_admission": str(root / "nonexistent-metadata"),
                "training_manifest": str(manifest),
                "holdout_group": "openneuro:ds005028:sub-11"}

    @staticmethod
    def _planned(nval=9, ntest=19, group="openneuro:ds005028:sub-11",
                 source="different_source"):
        val = [{"split": "val", "status": "excluded_frozen_holdout",
                "record": {"subject_group": group,
                           "source_file_sha256": source}} for _ in range(nval)]
        test = [{"split": "test", "status": "excluded_frozen_holdout",
                 "record": {"subject_group": "openneuro:ds005028:sub-01"}} for _ in range(ntest)]
        return val + test

    def test_missing_nine_validation_records_rejected_before_file_io(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            with patch.object(ev.p300, "rows", return_value=[]), \
                 patch.object(ev.p300, "plan_records", return_value=self._planned(nval=8)), \
                 patch.object(ev.p300, "read_original_eeg_volts", side_effect=AssertionError("DAT opened")):
                with self.assertRaisesRegex(ValueError, "Unexpected validation split"):
                    ev.prepare_validation(self._config(p), p / "output")
            self.assertFalse((p / "output").exists())

    def test_wrong_validation_subject_rejected_before_file_io(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            with patch.object(ev.p300, "rows", return_value=[]), \
                 patch.object(ev.p300, "plan_records", return_value=self._planned(group="wrong-group")), \
                 patch.object(ev.p300, "read_original_eeg_volts", side_effect=AssertionError("DAT opened")):
                with self.assertRaisesRegex(ValueError, "Unexpected validation split"):
                    ev.prepare_validation(self._config(p), p / "output")
            self.assertFalse((p / "output").exists())

    def test_wrong_test_count_rejected_before_file_io(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            with patch.object(ev.p300, "rows", return_value=[]), \
                 patch.object(ev.p300, "plan_records", return_value=self._planned(ntest=18)), \
                 patch.object(ev.p300, "read_original_eeg_volts", side_effect=AssertionError("DAT opened")):
                with self.assertRaisesRegex(ValueError, "P300 test count changed"):
                    ev.prepare_validation(self._config(p), p / "output")
            self.assertFalse((p / "output").exists())

    def test_train_subject_reappearance_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            with patch.object(ev.p300, "rows", return_value=[]), \
                 patch.object(ev.p300, "plan_records", return_value=self._planned()), \
                 patch.object(ev.p300, "read_original_eeg_volts", side_effect=AssertionError("DAT opened")):
                with self.assertRaisesRegex(ValueError, "P300 validation group contaminated"):
                    ev.prepare_validation(self._config(p,
                        train_group="openneuro:ds005028:sub-11"), p / "output")
            self.assertFalse((p / "output").exists())


if __name__ == "__main__":
    unittest.main()
