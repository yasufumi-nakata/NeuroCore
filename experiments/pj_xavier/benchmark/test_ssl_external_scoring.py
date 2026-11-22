"""Synthetic shape/identity fixtures only; no empirical EEG performance claims."""
import json
from pathlib import Path
import tempfile
import unittest

import score_external_predictions as score


class TwentyFourConditionScoringTests(unittest.TestCase):
    def fixture(self, directory):
        classes = ["ai_prompt_submit", "keyboard", "mouse_click"]
        labels = [classes[0]]*126 + [classes[1]]*252 + [classes[2]]*252
        truth = [{"id": "synthetic-"+str(i), "label": label} for i, label in enumerate(labels)]
        variants = ["fused", "context_only", "eeg_only", "eeg_shuffled", "eeg_time_shifted"]
        inputs = {}
        for key in ["smollm17b_v3", "qwen8b_v3_30e", "cbramod_smollm_v3", "cbramod_ssl_smollm_v3", "history"]:
            names = ["history_"+str(i) for i in range(4)] if key == "history" else variants
            p = directory/(key+".jsonl")
            predictions = [{"id": r["id"], "true_label": r["label"], "variant": variant,
                            "probabilities": {"ai_prompt_submit": .25, "other_observed_action": .75} if key == "history" else {"ai_prompt_submit": .25, "keyboard": .4, "mouse_click": .35},
                            "checkpoint_sha256": "f"*64, "abstain": True}
                           for variant in names for r in truth]
            p.write_text("".join(json.dumps(r)+"\n" for r in predictions))
            inputs[key] = {"predictions_path": str(p), "predictions_sha256": score.history.sha(p),
                           "expected_conditions": names, "checkpoint_sha256": "f"*64}
        return truth, inputs

    def test_frozen_scorer_accepts_twenty_four_complete_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            truth, inputs = self.fixture(Path(temp))
            values, provenance = score.collect_predictions(inputs, truth)
            self.assertEqual(len(values), 24)
            self.assertTrue(all(len(v) == 630 for v in values.values()))
            probability = {name: [p["probabilities"]["ai_prompt_submit"] for p in rows] for name, rows in values.items()}
            uncertainty = score.history.clustered_comparison(
                [r["label"] == "ai_prompt_submit" for r in truth], ["synthetic-person-"+str(i%25) for i in range(630)],
                probability, repetitions=20, seed=20261122)
            self.assertEqual(len(uncertainty["models"]), 24)
            self.assertEqual(len(provenance), 5)

    def test_one_missing_fourth_model_record_is_not_subset_scored(self):
        with tempfile.TemporaryDirectory() as temp:
            truth, inputs = self.fixture(Path(temp))
            p = Path(inputs["cbramod_ssl_smollm_v3"]["predictions_path"])
            lines = p.read_text().splitlines(); p.write_text("\n".join(lines[:-1])+"\n")
            inputs["cbramod_ssl_smollm_v3"]["predictions_sha256"] = score.history.sha(p)
            with self.assertRaisesRegex(ValueError, "incomplete external condition"):
                score.collect_predictions(inputs, truth)


if __name__ == "__main__":
    unittest.main()
