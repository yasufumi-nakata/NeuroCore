import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import history_baselines as hb


def fixture():
    features = {}
    for kind in hb.KINDS:
        features[kind + "_source_available"] = True
        for horizon in hb.HORIZONS:
            features[f"{kind}_count_{horizon}s"] = 0
        features[kind + "_seconds_since_last_capped_60"] = None
    pack = {"cutoff_s": 9.9, "max_source_onset_used_s": None,
            "source_fields": ["onset", "action only for key-down/up classification"],
            "target_or_future_events_included": False, "features": features}
    return {"window_start_s": 10., "decision_time_s": 15., "event_time_s": 16., "clock_uncertainty_guard_s": .1,
            "history_pre_eeg": pack, "history_pre_decision_guarded": {**copy.deepcopy(pack), "cutoff_s": 14.9},
            "history_evidence": [{"acquisition": "keypress", "sha256": "a" * 64, "uri": "source#events.tsv"}]}


class HistoryBaselineTests(unittest.TestCase):
    def test_allowlisted_shape_and_missing(self):
        values, names = hb.history_vector(fixture(), "history_pre_eeg")
        self.assertEqual((len(values), len(names)), (28, 28))
        self.assertNotIn("cutoff_s", names)
        self.assertEqual(values[5:7], [60., 1.])

    def test_future_and_boundary_sources_rejected(self):
        for value in (9.9, 10.0):
            with self.subTest(value=value):
                row = fixture()
                row["history_pre_eeg"]["max_source_onset_used_s"] = value
                with self.assertRaises(ValueError):
                    hb.history_vector(row, "history_pre_eeg")

    def test_wrong_cutoff_rejected(self):
        row = fixture()
        row["history_pre_eeg"]["cutoff_s"] = 10.
        with self.assertRaises(ValueError):
            hb.history_vector(row, "history_pre_eeg")

    def test_gpt_or_target_feature_rejected(self):
        row = fixture()
        row["history_pre_eeg"]["features"]["label"] = "ai_prompt_submit"
        with self.assertRaises(ValueError):
            hb.history_vector(row, "history_pre_eeg")
        row = fixture()
        row["history_evidence"][0]["acquisition"] = "gpt"
        with self.assertRaises(ValueError):
            hb.history_vector(row, "history_pre_eeg")

    def test_nonmonotone_and_missing_count_rejected(self):
        row = fixture()
        row["history_pre_eeg"]["features"]["key_down_count_1s"] = 1
        with self.assertRaises(ValueError):
            hb.history_vector(row, "history_pre_eeg")
        row["history_pre_eeg"]["features"]["key_down_count_1s"] = 0
        row["history_pre_eeg"]["features"]["key_down_count_60s"] = 1
        with self.assertRaises(ValueError):
            hb.history_vector(row, "history_pre_eeg")

    def test_binary_metrics_are_correct(self):
        result = hb.binary_metrics([0, 0, 1, 1], [.1, .4, .35, .8])
        self.assertEqual(result["confusion"], {"tp": 1, "fn": 1, "tn": 2, "fp": 0})
        self.assertAlmostEqual(result["balanced_accuracy"], .75)
        self.assertAlmostEqual(result["auroc"], .75)
        self.assertAlmostEqual(result["auprc_average_precision_enriched_sample_only"], 5 / 6)

    def test_single_class_metrics_not_fabricated(self):
        result = hb.binary_metrics([0, 0], [.1, .2])
        self.assertIsNone(result["auroc"])
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["balanced_accuracy"])

    def test_paired_identical_models_have_zero_intervals(self):
        result = hb.clustered_comparison([0, 1, 0, 1], ["s1", "s1", "s2", "s2"],
                                         {"first": [.2, .9, .3, .8], "second": [.2, .9, .3, .8]}, repetitions=20)
        self.assertEqual(result["n_subjects"], 2)
        for interval in result["paired_differences_first_minus_second"][0]["intervals"].values():
            self.assertEqual(interval["lower_95"], 0.)
            self.assertEqual(interval["upper_95"], 0.)

    def test_cluster_draws_keep_whole_subject(self):
        for indices in hb.cluster_indices(["a", "a", "b", "b", "b"], 20, 7):
            self.assertEqual(sum(indices == 0), sum(indices == 1))
            self.assertEqual(sum(indices == 2), sum(indices == 3))
            self.assertEqual(sum(indices == 3), sum(indices == 4))

    def test_scoring_refuses_non_test_or_partial_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"id": "x", "dataset": "d", "subject": "s", "split": "test", "label": "ai_prompt_submit"},
                    {"id": "y", "dataset": "d", "subject": "t", "split": "test", "label": "keyboard"}]
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
            prediction = root / "prediction.jsonl"
            prediction.write_text(json.dumps({"id": "x", "variant": "fused", "true_label": "ai_prompt_submit",
                                               "probabilities": {"ai_prompt_submit": .7, "keyboard": .3}, "checkpoint_sha256": "c" * 64}) + "\n")
            args = SimpleNamespace(manifest=str(manifest), expected_manifest_sha256=hb.sha(manifest), predictions=[str(prediction)],
                                   output=str(root / "report.json"), bootstrap_repetitions=20)
            with self.assertRaisesRegex(ValueError, "exact frozen test"):
                hb.score_predictions(args)


if __name__ == "__main__":
    unittest.main()
