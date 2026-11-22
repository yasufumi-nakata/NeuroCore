import copy
import unittest

import audit_ssl_external_completion as audit


class NewExternalCompletionContractTests(unittest.TestCase):
    def protocol(self):
        return {"model_order": [audit.ssl.MODEL_KEY], "models": {audit.ssl.MODEL_KEY: {}},
                "conditions": audit.prior.VARIANTS,
                "recorded_comparators": {"neural": {k: {} for k in ["smollm17b_v3", "qwen8b_v3_30e", "cbramod_smollm_v3"]}},
                "history_models": ["h1", "h2", "h3", "h4"], "history_inference_authorized": False}

    def test_exact_new_and_recorded_condition_coverage(self):
        self.assertEqual(len(audit.required_condition_keys(self.protocol())), 24)

    def test_missing_original_comparator_rejected(self):
        value = self.protocol(); value["recorded_comparators"]["neural"].pop("qwen8b_v3_30e")
        with self.assertRaisesRegex(ValueError, "identities"):
            audit.required_condition_keys(value)

    def test_silent_original_model_rerun_or_history_inference_rejected(self):
        value = self.protocol(); value["models"]["smollm17b_v3"] = {}
        with self.assertRaisesRegex(ValueError, "fourth"):
            audit.required_condition_keys(value)
        value = self.protocol(); value["history_inference_authorized"] = True
        with self.assertRaisesRegex(ValueError, "history"):
            audit.required_condition_keys(value)

    def test_wrong_checkpoint_or_partial_raw_exclusion_receipt_rejected(self):
        value = {"status": "passed", "protocol_sha256": "p", "aligned_checkpoint_sha256": "w", "guard_sha256": "s",
                 "membership": {"all_ITU_raw_sources_explicitly_excluded": 26,
                                "original_UniBa_val_test_raw_sources_explicitly_excluded": 14,
                                "actual_SSL_raw_source_overlap_with_heldout": 0}}
        audit.verify_preflight(value, "p", "w", "s")
        with self.assertRaisesRegex(ValueError, "identity"):
            audit.verify_preflight(value, "p", "other", "s")
        value["membership"]["all_ITU_raw_sources_explicitly_excluded"] = 25
        with self.assertRaisesRegex(ValueError, "full SSL"):
            audit.verify_preflight(value, "p", "w", "s")


if __name__ == "__main__":
    unittest.main()
