import copy
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssl_external_contract as contract


def source_fixture():
    a, b, held, itu = (c*64 for c in "abcd")
    rows = [
        {"id": "tr", "split": "train", "subject_group": "portable:p1", "source_recording_id": "p1:s1",
         "source_sha256": a, "window_start_s": 5., "window_end_s": 35.},
        {"id": "va", "split": "val", "subject_group": "portable:p2", "source_recording_id": "p2:s1",
         "source_sha256": b, "window_start_s": 5., "window_end_s": 35.},
    ]
    membership = {"train_ids": ["tr"], "val_ids": ["va"],
                  "source_sha256_by_partition": {"train": [a], "val": [b]},
                  "subject_groups_by_partition": {"train": ["portable:p1"], "val": ["portable:p2"]},
                  "test_source_used": False, "behavior_labels_used": 0}
    allowlist = {"sources": {r["source_sha256"]: {k: r[k] for k in ("split", "subject_group", "source_recording_id")} for r in rows},
                 "excluded_source_sha256": [held, itu]}
    return rows, membership, allowlist, [{"split": "test", "eeg_source_sha256": held}], [{"eeg_source_sha256": itu}], {"itu/record.eeg": {"sha256": itu}}


class SourceMembershipTests(unittest.TestCase):
    def test_valid_declared_source_exclusions(self):
        result = contract.validate_source_membership(*source_fixture())
        self.assertEqual(result["training_unique_seconds"], 30)
        self.assertFalse(result["original_upstream_pretraining_membership_reaudited"])

    def test_hidden_itu_hash_is_rejected_even_if_renamed_portable(self):
        values = list(source_fixture())
        values[5]["renamed.eeg"] = {"sha256": "a"*64}
        values[4].append({"eeg_source_sha256": "a"*64})
        values[2]["excluded_source_sha256"].append("a"*64)
        with self.assertRaisesRegex(ValueError, "not admitted|Heldout"):
            contract.validate_source_membership(*values)

    def test_uniba_heldout_hash_must_be_explicitly_excluded(self):
        values = list(source_fixture())
        values[2]["excluded_source_sha256"].remove("c"*64)
        with self.assertRaisesRegex(ValueError, "explicitly excluded"):
            contract.validate_source_membership(*values)

    def test_subject_partition_rewrite_is_rejected(self):
        values = list(source_fixture())
        values[0][0]["split"] = "val"
        with self.assertRaises(ValueError):
            contract.validate_source_membership(*values)

    def test_training_membership_rewrite_is_rejected(self):
        values = list(source_fixture())
        values[1]["train_ids"] = ["tr", "va"]
        with self.assertRaisesRegex(ValueError, "membership differs"):
            contract.validate_source_membership(*values)

    def test_overlapping_ssl_source_intervals_are_rejected(self):
        values = list(source_fixture())
        extra = dict(values[0][0], id="tr2", window_start_s=10., window_end_s=40.)
        values[0].append(extra)
        with self.assertRaisesRegex(ValueError, "overlap"):
            contract.validate_source_membership(*values)

    def test_action_or_pseudo_targets_cannot_enter_ssl(self):
        for field in ("label", "target", "environment_text"):
            with self.subTest(field=field):
                values = list(source_fixture()); values[0][0][field] = "invented_intent"
                with self.assertRaisesRegex(ValueError, "labels reached"):
                    contract.validate_source_membership(*values)

    def test_escaping_evidence_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "escapes"):
                contract.inside(root, "../outside")


class SelectedTensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.state = {"schema": "xavier-cbramod-masked-ssl-v1", "model": {"w": torch.tensor([1., 2.])},
                      "seen_training_ids": ["tr"], "global_step": 2, "training_examples_presented": 3}
        self.weights = {"w": torch.tensor([1., 2.])}
        self.membership = {"train_ids": ["tr"], "val_ids": ["va"]}
        self.seal = {"selected_global_step": 2, "unique_training_windows_before_selection": 1,
                     "unique_training_seconds_before_selection": 30, "training_examples_presented_before_selection": 3,
                     "downstream_test_source_used": False, "itu_sources_used": 0, "behavior_labels_used": 0,
                     "updated_tensor_count": 1}
        self.history = [{"global_step": 1, "val": {"masked_mse": .4}},
                        {"global_step": 2, "val": {"masked_mse": .2}},
                        {"global_step": 3, "val": {"masked_mse": .2}}]
        self.summary = {"best_validation_masked_mse": .2}

    def check(self):
        return contract.validate_selected_tensors(self.state, self.weights, self.seal,
                                                   self.membership, self.history, self.summary, self.torch)

    def test_valid_selected_tensor_identity(self):
        self.assertTrue(self.check()["weights_equal_validation_selected_checkpoint"])

    def test_modified_exported_encoder_tensor_rejected(self):
        self.weights["w"][0] = 9.
        with self.assertRaisesRegex(ValueError, "tensors differ"):
            self.check()

    def test_gradient_exposure_to_validation_id_rejected(self):
        self.state["seen_training_ids"].append("va")
        with self.assertRaisesRegex(ValueError, "nontraining"):
            self.check()

    def test_later_tied_checkpoint_not_earliest_selection_rejected(self):
        self.state["global_step"] = self.seal["selected_global_step"] = 3
        with self.assertRaisesRegex(ValueError, "earliest minimum"):
            self.check()

    def test_overstated_unique_hours_rejected(self):
        self.seal["unique_training_seconds_before_selection"] = 90
        with self.assertRaisesRegex(ValueError, "overstated"):
            self.check()


if __name__ == "__main__":
    unittest.main()
