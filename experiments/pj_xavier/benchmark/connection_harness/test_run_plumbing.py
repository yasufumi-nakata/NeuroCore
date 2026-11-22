import json
from pathlib import Path
import tempfile
import unittest

import run_plumbing as runner


class PlumbingContractTests(unittest.TestCase):
    def test_missing_or_duplicate_oracle_case_is_rejected(self):
        rows = [{"case": key, "passed": True} for key in sorted(runner.CASE_IDS)]
        self.assertEqual(len(runner.parse_cases("\n".join(map(json.dumps, rows)))), 8)
        for changed in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                runner.parse_cases("\n".join(map(json.dumps, changed)))

    def test_nonboolean_oracle_result_is_rejected(self):
        rows = [{"case": key, "passed": "true"} for key in sorted(runner.CASE_IDS)]
        with self.assertRaises(ValueError):
            runner.parse_cases("\n".join(map(json.dumps, rows)))

    def test_arbitrary_submission_path_is_not_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                runner.run_fixture("../../unreviewed", "/bin/false", "/bin/false", Path(tmp))

    def test_candidate_context_excludes_oracle_and_solutions(self):
        public = runner.ROOT / "public_input"
        self.assertEqual({p.name for p in public.iterdir()}, {"task.json", "ConnectionAPI.java", "DeviceAPI.java"})
        task = json.loads((public / "task.json").read_text())
        self.assertFalse(task["eeg_input"])
        self.assertFalse(task["participant_solution_input"])
        self.assertFalse(task["model_generated_solution"])
        self.assertNotIn("expected", task)

    def test_wrong_failure_case_cannot_satisfy_expected_fixture(self):
        cases = [{"case": key, "passed": key != "mobile_only_success"} for key in runner.CASE_IDS]
        self.assertFalse(runner.matches_expected("wrong_priority", {"status": "test_failure", "cases": cases}))
        cases = [{"case": key, "passed": key != "wifi_priority_despite_list_order"} for key in runner.CASE_IDS]
        self.assertTrue(runner.matches_expected("wrong_priority", {"status": "test_failure", "cases": cases}))

    def test_accidental_crash_must_receive_zero_cases(self):
        cases = [{"case": key, "passed": False} for key in runner.CASE_IDS]
        self.assertTrue(runner.matches_expected("accidental_npe", {"status": "test_failure", "cases": cases}))
        cases[0]["passed"] = True
        self.assertFalse(runner.matches_expected("accidental_npe", {"status": "test_failure", "cases": cases}))


if __name__ == "__main__":
    unittest.main()
