import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_ai_coding import (fit_clock, phase_anchors, phase_transitions,
                               split_subjects, environment_before, action_events, availability_timing)


class ClockTests(unittest.TestCase):
    def test_recovers_offset_and_drift_from_independent_phase_anchors(self):
        behavioral = [(100, 30), (200, 40), (600, 50), (700, 70), (800, 80), (1200, 90)]
        eeg = [(t * 1.0002 + 12.5, c) for t, c in behavioral]
        result = fit_clock(eeg, behavioral)
        self.assertTrue(result["verified"])
        self.assertAlmostEqual(result["offset_eeg_seconds"], 12.5)
        self.assertAlmostEqual(result["scale_eeg_per_behavior_second"], 1.0002)
        self.assertLess(result["max_error_ms"], .00001)

    def test_inconsistent_anchor_is_not_silently_deleted(self):
        behavioral = [(100, 30), (200, 40), (600, 50), (700, 70), (800, 80)]
        eeg = [(t + 10 + (5 if c == 70 else 0), c) for t, c in behavioral]
        result = fit_clock(eeg, behavioral)
        self.assertFalse(result["verified"])
        self.assertEqual(result["anchor_count"], 5)

    def test_too_few_or_duplicated_anchors_block_alignment(self):
        b = [(100, 30), (200, 40), (600, 50), (700, 70)]
        e = [(t + 12, c) for t, c in b] + [(150, 30)]
        result = fit_clock(e, b)
        self.assertFalse(result["verified"])
        self.assertIn(30, result["excluded_duplicate_codes"])

    def test_nonmonotonic_pairing_is_rejected(self):
        b = [(100, 30), (200, 40), (600, 50), (700, 70)]
        e = [(100, 30), (650, 40), (600, 50), (700, 70)]
        self.assertFalse(fit_clock(e, b)["verified"])

    def test_only_real_phase_transition_codes_are_used(self):
        x = np.array([0, 0, 30, 30, 0, 40, 40, 50, 50, 999])
        self.assertEqual(phase_transitions(x, 2), [(1., 30), (2.5, 40), (3.5, 50)])

    def test_behavior_phase_parser_does_not_treat_artificial_epochs_as_labels(self):
        rows = [{"onset": 1, "trial_type": "epoch"},
                {"onset": 2, "trial_type": "taskBegin", "ai": "True"},
                {"onset": 3, "trial_type": "taskBegin", "ai": "False"}]
        self.assertEqual(phase_anchors(rows), [(2., 80), (3., 40)])


class CausalDataTests(unittest.TestCase):
    def test_offline_filter_delay_reduces_usable_forecast_lead_and_is_unverified(self):
        result = availability_timing(10., 4.5, 4.125)
        self.assertEqual(result["decision_time_s"], 14.125)
        self.assertEqual(result["minimum_lead_s"], .375)
        self.assertFalse(result["feature_availability_verified"])
        with self.assertRaises(ValueError):
            availability_timing(10., .25, 4.125)

    def test_real_anonymized_clock_fixture_and_down_up_events(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/ai_coding_v2_minimal_recorded_schema.json").read_text())
        anchors = fixture["phase_anchors"]
        clock = fit_clock([(r["eeg_s"], r["phase_code"]) for r in anchors],
                          [(r["behavior_s"], r["phase_code"]) for r in anchors])
        self.assertTrue(clock["verified"])
        self.assertAlmostEqual(clock["offset_eeg_seconds"], 236.9256774531438, places=6)
        self.assertLess(clock["max_error_ms"], 2)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            with (p / "sub-anonymous_acq-keypress_run-01_events.tsv").open("w", newline="") as f:
                w = csv.DictWriter(f, ["onset", "duration", "key", "action"], delimiter="\t")
                w.writeheader()
                w.writerows(fixture["keypress_rows"])
            events, guards = action_events(p, clock)
        self.assertGreaterEqual(sum(e["label"] == "keyboard" for e in events), 4)
        self.assertEqual(len(guards), len(fixture["keypress_rows"]))

    def test_future_environment_is_excluded(self):
        rows = [{"onset": 1, "trial_type": "taskBegin", "ai": "True"},
                {"onset": 10, "trial_type": "future", "ai": "True"}]
        clock = {"scale_eeg_per_behavior_second": 1, "offset_eeg_seconds": 2}
        result = environment_before(rows, clock, 7)
        self.assertEqual(result["environment_time_s"], 3)
        self.assertNotIn("future", result["environment_text"])

    def test_subject_split_is_stable_and_disjoint(self):
        subjects = [f"uniba:P{i:03d}" for i in range(20)]
        a = split_subjects(subjects)
        b = split_subjects(list(reversed(subjects)))
        self.assertEqual(a, b)
        self.assertEqual(set(a.values()), {"train", "val", "test"})
        self.assertEqual(len(a), 20)

    def test_simultaneous_submit_and_mouse_click_are_not_duplicate_targets(self):
        clock = {"scale_eeg_per_behavior_second": 1, "offset_eeg_seconds": 0}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            for acq, fields, rows in [
                ("mouseclick", ["onset", "x", "y"], [{"onset": 5, "x": 1, "y": 1}]),
                ("gpt", ["onset", "trial_type"], [{"onset": 5.05, "trial_type": "GPTSentPrompt"}]),
                ("keypress", ["onset", "key", "action"], [{"onset": 9, "key": "X", "action": "press"}])]:
                with (p / f"sub-X_acq-{acq}_run-01_events.tsv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fields, delimiter="\t")
                    writer.writeheader()
                    writer.writerows(rows)
            events, guards = action_events(p, clock)
        self.assertEqual([e["label"] for e in events], ["ai_prompt_submit", "keyboard"])
        self.assertEqual(len(guards), 3)

    def test_verified_release_down_up_schema_yields_only_key_down_targets(self):
        clock = {"scale_eeg_per_behavior_second": 1, "offset_eeg_seconds": 0}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            path = p / "sub-anonymous_acq-keypress_run-01_events.tsv"
            path.write_text("onset\tduration\tkey\taction\n1\t0\tREDACTED\tdown\n1.2\t0\tREDACTED\tup\n")
            events, guards = action_events(p, clock)
        self.assertEqual([(e["time_s"], e["label"]) for e in events], [(1., "keyboard")])
        self.assertEqual(guards.tolist(), [1., 1.2])


if __name__ == "__main__":
    unittest.main()
