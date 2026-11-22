"""Synthetic, leakage-oriented P300 tests. These never read a real DAT."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
from p300_context_gate import P300ContractError, partition_p300_record, proof_of_input_boundary


def synthetic_record():
    h = "a" * 64
    return {
        "raw_dat_sha256": h,
        "decision_sample": 2000,
        "sampling_rate_hz": 256,
        "display": {
            "candidate_ids": ["X", "Y"],
            "slot_to_id": {"left": "X", "right": "Y"},
            "known_at_sample": 700,
            "origin": "verified_prior_screen_log",
            "evidence_sha256": "b" * 64,
        },
        "presentations": [{
            "candidate_id": "X", "onset_sample": 1000,
            "available_at_sample": 1010,
            "clock_evidence": "externally_calibrated_display",
            "evidence_sha256": "c" * 64,
        }],
        "protected_labels": {
            "TextToSpell": "X", "StimulusType": "target",
            "target_character": "X",
        },
        "decoder_outputs": {
            "SelectedTarget": "Y", "SelectedRow": 2,
            "SelectedColumn": 4, "TextResult": "XYZ",
        },
    }


class TestP300ContextGate(unittest.TestCase):
    def test_model_input_contains_no_label_or_posterior(self):
        out = partition_p300_record(synthetic_record())
        self.assertEqual(out["oracle_only"]["TextToSpell"], "X")
        self.assertEqual(out["decoder_output_only"]["SelectedTarget"], "Y")
        self.assertEqual(out["model_input"]["presentations"][0]["candidate_id"], "X")
        self.assertNotIn("TextToSpell", str(out["model_input"]))
        self.assertNotIn("StimulusType", str(out["model_input"]))
        self.assertNotIn("SelectedTarget", str(out["model_input"]))
        self.assertNotIn("TextResult", str(out["model_input"]))
        self.assertNotIn("raw_dat_sha256", out["model_input"])
        self.assertFalse(out["action_benchmark_authorized"])
        self.assertFalse(out["supervised_gold_authorized"])
        self.assertIsNone(out["free_intent_gold"])

    def test_protected_labels_do_not_change_input_or_digest(self):
        base = synthetic_record()
        first = partition_p300_record(base)
        base["protected_labels"]["TextToSpell"] = "Y"
        base["protected_labels"]["StimulusType"] = "non-target"
        base["decoder_outputs"]["SelectedTarget"] = "X"
        base["decoder_outputs"]["TextResult"] = "future text"
        second = partition_p300_record(base)
        self.assertEqual(first["model_input"], second["model_input"])
        self.assertEqual(proof_of_input_boundary(first), proof_of_input_boundary(second))

    def test_rejects_top_level_or_nested_target_injection(self):
        for kind in ("target_character", "TextToSpell", "TextResult", "StimulusType"):
            with self.subTest(key=kind):
                a = synthetic_record()
                a["display"][kind] = "X"
                with self.assertRaises(P300ContractError):
                    partition_p300_record(a)
                a = synthetic_record()
                a["presentations"][0][kind] = "X"
                with self.assertRaises(P300ContractError):
                    partition_p300_record(a)
        a = synthetic_record()
        a["free_intent_gold"] = "X"
        with self.assertRaises(P300ContractError):
            partition_p300_record(a)

    def test_rejects_future_display_and_event(self):
        a = synthetic_record()
        a["display"]["known_at_sample"] = 2001
        with self.assertRaises(P300ContractError):
            partition_p300_record(a)
        a = synthetic_record()
        a["presentations"][0]["available_at_sample"] = 2001
        with self.assertRaises(P300ContractError):
            partition_p300_record(a)
        a = synthetic_record()
        a["presentations"][0]["onset_sample"] = 1200
        a["presentations"][0]["available_at_sample"] = 1100
        with self.assertRaises(P300ContractError):
            partition_p300_record(a)

    def test_unverified_reader_fallback_is_not_gold(self):
        a = synthetic_record()
        a["presentations"][0]["clock_evidence"] = "reader_fallback"
        result = partition_p300_record(a)
        self.assertEqual(result["model_input"]["presentations"], [])
        self.assertEqual(len(result["audit_only"]["excluded_unverified_presentations"]), 1)
        self.assertFalse(result["onset_timing_ready_for_future_review"])
        self.assertFalse(result["action_benchmark_authorized"])

    def test_rejects_nonindependent_mapping_and_invalid_provenance(self):
        a = synthetic_record()
        a["display"]["origin"] = "TextToSpell_derived"
        with self.assertRaises(P300ContractError):
            partition_p300_record(a)
        a = synthetic_record()
        a["display"]["slot_to_id"] = {"left": "X", "right": "X"}
        with self.assertRaises(P300ContractError):
            partition_p300_record(a)
        a = synthetic_record()
        a["display"]["evidence_sha256"] = "unverified"
        with self.assertRaises(P300ContractError):
            partition_p300_record(a)

    def test_requires_strict_typed_sample_and_declared_rate(self):
        for value in (True, -1, 3.5, "2000"):
            with self.subTest(value=value):
                a = synthetic_record()
                a["decision_sample"] = value
                with self.assertRaises(P300ContractError):
                    partition_p300_record(a)
        for rate in (False, 0, -100, None, "256"):
            with self.subTest(rate=rate):
                a = synthetic_record()
                a["sampling_rate_hz"] = rate
                with self.assertRaises(P300ContractError):
                    partition_p300_record(a)

    def test_input_does_not_alias_mutable_oracle(self):
        r = synthetic_record()
        out = partition_p300_record(r)
        r["display"]["candidate_ids"][0] = "Z"
        r["protected_labels"]["TextToSpell"] = "Z"
        self.assertEqual(out["model_input"]["candidate_ids"], ["X", "Y"])
        self.assertEqual(out["oracle_only"]["TextToSpell"], "X")


if __name__ == "__main__":
    unittest.main()
