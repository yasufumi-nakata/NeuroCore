import copy
import csv
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prepare_natural_cohort import (EVENT_KINDS, history_features, load_frozen_inputs,
                                   make_candidates, read_recorded_events, select_case_control,
                                   require_parent_phase_hash)


PROTOCOL = json.loads((Path(__file__).parent / "natural_cohort_v3_protocol.json").read_text())


def event(label, first, last, time_s, row, state="true"):
    return {"label": label, "first": first, "last": last, "time_s": time_s,
            "acquisition": "gpt" if label == "ai_prompt_submit" else ("keypress" if label == "keyboard" else "mouseclick"),
            "evidence_row": row, "ai_availability": state}


class SelectionTests(unittest.TestCase):
    def test_cases_win_overlap_and_later_duplicate_case_is_rejected(self):
        data = [event("keyboard", 5, 15, 15, 2), event("ai_prompt_submit", 10, 20, 20, 3),
                event("ai_prompt_submit", 12, 22, 22, 4), event("keyboard", 0, 10, 10, 5),
                event("mouse_click", 25, 35, 35, 6)]
        selected, report = select_case_control(data, [20, 22], "r", PROTOCOL)
        self.assertEqual(sum(r["case_control_role"] == "case" for r in selected), 1)
        self.assertIn(3, [r["evidence_row"] for r in selected])
        self.assertNotIn(2, [r["evidence_row"] for r in selected])
        self.assertNotIn(4, [r["evidence_row"] for r in selected])
        self.assertIn(5, [r["evidence_row"] for r in selected])  # touching endpoint is allowed
        self.assertEqual(report["rejected"]["overlapping_case_window"], 1)

    def test_selection_is_deterministic_and_matches_availability(self):
        data = [event("ai_prompt_submit", 0, 10, 10, 2)]
        for i in range(1, 30):
            data.append(event("keyboard" if i % 2 else "mouse_click", i * 20, i * 20 + 10,
                              i * 20 + 10, i + 2, "false" if i > 20 else "true"))
        a, ra = select_case_control(data, [10], "same-record", PROTOCOL)
        b, rb = select_case_control(list(reversed(data)), [10], "same-record", PROTOCOL)
        self.assertEqual(a, b)
        self.assertEqual(ra, rb)
        self.assertEqual(len(a), 5)
        self.assertTrue(all(e["ai_availability"] == "true" for e in a))
        self.assertEqual(sum(e["label"] == "keyboard" for e in a), 2)
        self.assertEqual(sum(e["label"] == "mouse_click" for e in a), 2)

    def test_zero_request_subject_is_retained_with_real_controls(self):
        data = [event("keyboard" if i % 2 else "mouse_click", i * 20, i * 20 + 10,
                      i * 20 + 10, i + 2) for i in range(30)]
        selected, audit = select_case_control(data, [], "r", PROTOCOL)
        self.assertEqual(len(selected), 8)
        self.assertEqual(audit["retained_cases"], 0)
        self.assertEqual(audit["retained_controls"], 8)
        self.assertTrue(all(r["case_control_role"] == "control" for r in selected))

    def test_quota_shortfall_is_recorded_without_duplicates(self):
        data = [event("ai_prompt_submit", 0, 10, 10, 2), event("keyboard", 20, 30, 30, 3)]
        selected, audit = select_case_control(data, [10], "r", PROTOCOL)
        self.assertEqual(len(selected), 2)
        self.assertEqual(sum(q["deficit"] for q in audit["control_quotas"]), 3)


class CausalityTests(unittest.TestCase):
    def test_changed_phase_member_cannot_reuse_frozen_clock(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "phase.tsv"
            path.write_text("onset\ttrial_type\tai\n10\ttaskBegin\tTrue\n")
            original = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(require_parent_phase_hash(path, {original}), original)
            path.write_text("onset\ttrial_type\tai\n110\ttaskBegin\tTrue\n")
            with self.assertRaisesRegex(ValueError, "old clock"):
                require_parent_phase_hash(path, {original})

    def test_features_are_unchanged_by_target_and_future_events(self):
        history = {k: np.array([], dtype=float) for k in EVENT_KINDS}
        history["key_down"] = np.array([1., 2.])
        availability = {k: k == "key_down" for k in EVENT_KINDS}
        before = history_features(history, availability, 2.5)
        history["key_down"] = np.array([1., 2., 2.5, 3., 100.])
        after = history_features(history, availability, 2.5)
        self.assertEqual(before, after)
        self.assertEqual(before["features"]["key_down_count_1s"], 1)
        self.assertLess(before["max_source_onset_used_s"], before["cutoff_s"])

    def test_parser_uses_recorded_request_only_and_ignores_duration_and_content(self):
        with tempfile.TemporaryDirectory() as d:
            beh = Path(d)
            (beh / "sub-X_acq-keypress_run-01_events.tsv").write_text(
                "onset\tduration\tkey\taction\n2\t99999\tPRIVATE_VALUE\tdown\n4\t12345\tOTHER_VALUE\tup\n")
            (beh / "sub-X_acq-gpt_run-01_events.tsv").write_text(
                "onset\tduration\ttrial_type\n3\t1\tGPTStartPrompting\n7\t2\tGPTSentPrompt\n9\t1\tGPTReceiveText\n")
            clock = {"scale_eeg_per_behavior_second": 1, "offset_eeg_seconds": 0}
            events, history, available, _, _ = read_recorded_events(beh, clock)
            features = history_features(history, available, 3.)
        self.assertEqual([(e["time_s"], e["label"]) for e in events], [(2., "keyboard"), (7., "ai_prompt_submit")])
        self.assertNotIn("PRIVATE_VALUE", json.dumps(features))
        self.assertNotIn("99999", json.dumps(features))
        self.assertEqual(features["features"]["key_down_count_5s"], 1)
        self.assertEqual(features["features"]["key_up_count_5s"], 0)

    def test_candidate_keeps_filter_and_clock_guards(self):
        clock = {"scale_eeg_per_behavior_second": 1, "offset_eeg_seconds": 0, "max_error_ms": 50}
        data = [{"label": "ai_prompt_submit", "time_s": 10.}]
        phases = [{"onset": 0, "trial_type": "taskBegin", "ai": "True"}]
        candidates, rejected = make_candidates(data, phases, clock, 10000, 256, PROTOCOL)
        self.assertFalse(rejected)
        c = candidates[0]
        self.assertLessEqual(c["last"] / 256, 10 - 4.5 - .05)
        self.assertEqual(c["last"] - c["first"], 448)
        self.assertLessEqual(c["timing"]["decision_time_s"] + .375, 10 - .05)
        self.assertLessEqual(c["context"]["environment_time_s"], c["first"] / 256 - .05)
        self.assertFalse(c["timing"]["feature_availability_verified"])

    def test_modified_protocol_is_rejected_before_any_data_access(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "protocol.json"
            p.write_text(json.dumps(PROTOCOL))
            with self.assertRaisesRegex(ValueError, "registered SHA"):
                load_frozen_inputs(p, Path(d), Path(d), "0" * 64)


if __name__ == "__main__":
    unittest.main()
