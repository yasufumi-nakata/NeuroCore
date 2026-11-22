"""Read-only checks for the frozen Zenodo8089820 source-group split."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parent))
import freeze_smac_bci8089820_metadata as freeze

ROOT = Path(__file__).absolute().parents[3]
AUDIT = ROOT / "private/xavier/expansion_candidates_v1/bci_header_audit_v2/future_training_constraints_v1.json"
RECORDINGS = ROOT / "private/xavier/expansion_candidates_v1/bci_header_audit_v2/recordings.jsonl"


class FrozenBciSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.constraints = json.loads(AUDIT.read_text())
        cls.plan = freeze.make_plan(cls.constraints["record_metadata"],
                                    cls.constraints["must_link_groups"],
                                    cls.constraints["author_reported_QC_flags"])

    def test_pinned_inputs(self):
        self.assertEqual(freeze.sha(AUDIT.read_bytes()), freeze.CONSTRAINTS_SHA)
        self.assertEqual(freeze.sha(RECORDINGS.read_bytes()), freeze.RECORDINGS_SHA)
        self.assertEqual(self.constraints["source_archive_sha256"], freeze.ARCHIVE_SHA)

    def test_partition_and_deny(self):
        p = self.plan
        self.assertEqual(p["summary"]["groups_by_split"],
                         {"train": 56, "val": 12, "test": 12, "review": 7})
        self.assertEqual(p["summary"]["records_by_split"],
                         {"train": 446, "val": 96, "test": 96, "review": 56})
        self.assertTrue(all(not x["supervised_cue_training_allowed"] for x in p["members"]))
        self.assertTrue(all(not x["online_causal_claim_allowed"] for x in p["members"]))
        self.assertFalse(p["summary"]["offline_ssl_training_allowed_now"])

    def test_must_link_and_review(self):
        assignments = self.plan["group_assignments"]
        for link in self.constraints["must_link_groups"]:
            self.assertEqual(len({assignments[g] for g in link["subject_groups"]}), 1)
        for g in self.constraints["author_reported_QC_flags"]:
            self.assertEqual(assignments[g], "review")

    def test_same_input_stable_bytes(self):
        again = freeze.make_plan(self.constraints["record_metadata"],
                                 self.constraints["must_link_groups"],
                                 self.constraints["author_reported_QC_flags"])
        self.assertEqual(freeze.canonical_json(self.plan), freeze.canonical_json(again))

    def test_unverified_event_timebase_rejected(self):
        rows = copy.deepcopy(self.constraints["record_metadata"])
        rows[0]["annotation_explicit_event_timebase_verified"] = True
        with self.assertRaisesRegex(ValueError, "timebase"):
            freeze.make_plan(rows, self.constraints["must_link_groups"],
                             self.constraints["author_reported_QC_flags"])

    def test_loss_of_must_link_rejected(self):
        with self.assertRaisesRegex(ValueError, "constraints changed"):
            freeze.make_plan(self.constraints["record_metadata"], [],
                             self.constraints["author_reported_QC_flags"])

    def test_dropped_member_rejected(self):
        with self.assertRaisesRegex(ValueError, "694"):
            freeze.make_plan(self.constraints["record_metadata"][:-1],
                             self.constraints["must_link_groups"],
                             self.constraints["author_reported_QC_flags"])


if __name__ == "__main__":
    unittest.main()
