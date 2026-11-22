import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

import cpu_baselines as b


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.thread_limit = threadpool_limits(limits=2)
        self.rows = []
        self.reference = self.root / "reference"
        self.reference.mkdir()
        self.manifest = self.root / "manifest.jsonl"
        artifacts = {}
        rng = np.random.default_rng(17)
        for split_index, split in enumerate(b.SPLITS):
            filename = f"{split}.npy"
            data = rng.normal(size=(6, 8, 448)).astype("float32")
            np.save(self.root / filename, data, allow_pickle=False)
            artifacts[filename] = {"sha256": b.sha256(self.root / filename), "bytes": (self.root / filename).stat().st_size}
            for index in range(6):
                start = 10 + index * 10
                end = start + 1.75
                self.rows.append({"id": f"{split}-{index}", "dataset": "fixture", "subject": f"subject-{split_index}",
                                  "session": "one", "recording_id": filename, "split": split,
                                  "label": ("keyboard", "mouse_click", "ai_prompt_submit")[index % 3],
                                  "label_source": "recorded_event", "target_kind": "observed_action", "data_origin": "real_recording",
                                  "eeg_path": filename, "eeg_index": index, "channels": list(b.CHANNELS), "sample_rate_hz": 256.,
                                  "eeg_source_sha256": hashlib.sha256(filename.encode()).hexdigest(),
                                  "label_evidence_sha256": "a" * 64, "environment_sha256": "b" * 64,
                                  "window_start_s": start, "window_end_s": end, "decision_time_s": end + 4.125,
                                  "event_time_s": end + 4.51, "feature_available_time_s": end + 4.125, "minimum_lead_s": .375,
                                  "environment_source": "predecision_harness", "environment_contains_target_label": False,
                                  "environment_time_s": 0., "clock_sync": {"verified": True},
                                  "environment_text": "Java programming experiment. Last recorded experiment page or event: instructions. AI availability recorded at that event: False."})
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.membership = {key: [row["id"] for row in self.rows if row["split"] == split]
                           for split, key in (("train", "training_ids"), ("val", "validation_ids"), ("test", "test_ids"))}
        self.membership.update(manifest_sha256=b.sha256(self.manifest), test_used_for_tuning=False)
        (self.reference / "training_membership.json").write_text(json.dumps(self.membership))
        (self.reference / "eeg_artifacts.json").write_text(json.dumps(artifacts))
        self.protocol = {"manifest_sha256": b.sha256(self.manifest), "split_counts": {s: 6 for s in b.SPLITS},
                         "reference_file_sha256": {name: b.sha256(self.reference / name) for name in ("training_membership.json", "eeg_artifacts.json")},
                         "C_grid": [.01, .1]}

    def tearDown(self):
        self.thread_limit.restore_original_limits()
        self.tmp.cleanup()

    def test_frozen_llm_membership_and_waveform_hashes_match(self):
        rows, validation = b.verify_inputs(self.manifest, self.reference, self.protocol)
        self.assertEqual(len(rows), 18)
        self.assertEqual(validation["subjects"], {"train": 1, "val": 1, "test": 1})
        self.assertTrue(validation["subject_disjoint"])

    def test_subject_or_physical_source_alias_across_splits_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[6]["subject"] = rows[0]["subject"]
        with self.assertRaisesRegex(ValueError, "crosses"):
            b.validate_rows(rows, self.membership)
        rows = copy.deepcopy(self.rows)
        rows[6]["eeg_source_sha256"] = rows[0]["eeg_source_sha256"]
        with self.assertRaisesRegex(ValueError, "crosses"):
            b.validate_rows(rows, self.membership)

    def test_duplicate_prepared_window_and_changed_frozen_split_are_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["eeg_index"] = 0
        with self.assertRaisesRegex(ValueError, "more than once"):
            b.validate_rows(rows, self.membership)
        changed = copy.deepcopy(self.membership)
        changed["training_ids"][0], changed["test_ids"][0] = changed["test_ids"][0], changed["training_ids"][0]
        with self.assertRaisesRegex(ValueError, "split IDs"):
            b.validate_rows(self.rows, changed)

    def test_future_context_target_marker_and_non_gold_labels_are_rejected(self):
        for key, value in (("environment_time_s", 1e9), ("environment_contains_target_label", True), ("label_source", "ai_generated")):
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.assertRaises(ValueError):
                b.validate_rows(rows, self.membership)

    def test_manifest_and_prepared_byte_tampering_fail_closed(self):
        array = np.load(self.root / "train.npy", allow_pickle=False)
        array[0, 0, 0] += 1
        np.save(self.root / "train.npy", array, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "prepared EEG bytes"):
            b.verify_inputs(self.manifest, self.reference, self.protocol)
        self.manifest.write_text(self.manifest.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "manifest SHA"):
            b.verify_inputs(self.manifest, self.reference, self.protocol)

    def test_context_uses_no_target_identity_or_time_and_vocabulary_is_train_only(self):
        row = copy.deepcopy(self.rows[0])
        before = b.parse_context(row)
        row.update(label="hidden_test_answer", subject="different-subject", eeg_index=99999,
                   event_time_s=99999999, id="different-id", label_evidence_row=8888)
        self.assertEqual(before, b.parse_context(row))
        unseen = copy.deepcopy(row)
        unseen["environment_text"] = unseen["environment_text"].replace("instructions", "validationOnlyCategory")
        vectorizer, train, val = b.fit_context([row], [unseen])
        self.assertFalse(any("validationOnlyCategory" in name for name in vectorizer.get_feature_names_out()))
        self.assertEqual(train.shape, val.shape)

    def test_spectral_features_identify_alpha_and_preserve_relative_power_under_gain(self):
        t = np.arange(448) / 256
        signal = np.tile(np.sin(2 * np.pi * 10 * t) + .05 * np.sin(2 * np.pi * 6 * t)
                         + .02 * np.sin(2 * np.pi * 20 * t) + .01 * np.sin(2 * np.pi * 35 * t), (8, 1))
        original = b.spectrum_features(signal).reshape(8, 9)
        amplified = b.spectrum_features(signal * 10).reshape(8, 9)
        self.assertTrue(np.all(original[:, 1] > original[:, 2]))
        np.testing.assert_allclose(amplified[:, 4:8], original[:, 4:8], atol=1e-9)
        np.testing.assert_allclose(amplified[:, 8] - original[:, 8], np.full(8, 2.), atol=1e-9)

    def test_scaler_statistics_are_fit_on_train_only(self):
        x_train = np.array([[-2., 0], [-1., 0], [1., 0], [2., 0]])
        x_val = np.array([[1000., 1], [2000., 1]])
        scaler, model, selected = b.select_model(x_train, np.array(["a", "a", "b", "b"]), x_val, np.array(["a", "b"]), ["a", "b"], [.1, 1.])
        np.testing.assert_allclose(scaler.mean_, [0, 0])
        self.assertIn(selected["selected_C"], [.1, 1.])
        self.assertEqual(model.class_weight, "balanced")

    def test_test_consumption_marker_cannot_be_reused(self):
        b.claim_test_once(self.root, "a" * 64)
        with self.assertRaises(FileExistsError):
            b.claim_test_once(self.root, "b" * 64)

    def test_once_only_end_to_end_fixture_preserves_upstream_and_saves_portable_models(self):
        protocol = self.root / "protocol.json"
        protocol.write_text(json.dumps(self.protocol))
        output = self.root / "result"
        before_manifest = b.sha256(self.manifest)
        report = b.run(protocol, self.manifest, self.reference, output, b.sha256(Path(b.__file__)))
        self.assertEqual(report["status"], "completed")
        self.assertFalse(report["latent_intent_goal_validated"])
        self.assertEqual(b.sha256(self.manifest), before_manifest)
        for mode in b.MODES:
            self.assertEqual(report["models"][mode]["test_predict_calls"], 1)
            with np.load(output / (mode + ".npz"), allow_pickle=False) as model:
                self.assertIn("coefficients", model.files)
                self.assertEqual(model["coefficients"].shape[0], 3)
        with self.assertRaisesRegex(ValueError, "output already exists"):
            b.run(protocol, self.manifest, self.reference, output, b.sha256(Path(b.__file__)))


if __name__ == "__main__":
    unittest.main()
