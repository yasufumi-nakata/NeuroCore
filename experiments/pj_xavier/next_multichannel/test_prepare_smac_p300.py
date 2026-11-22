"""CPU contract tests for original-DAT P300 offline SSL admission."""
from __future__ import annotations
import sys, unittest
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path(__file__).absolute().parents[3]
sys.path.insert(0,str(Path(__file__).parent))
import prepare_smac_p300 as mod


class P300SourceTests(unittest.TestCase):
    def setUp(self):
        self.ledger=mod.rows(ROOT/"private/xavier/expansion_candidates_v1/P300_header_audit_v1/records.jsonl")
        self.metadata=mod.rows(ROOT/"private/xavier/phase4_automation_1615/source_admission.private.jsonl")

    def test_frozen_subject_splits_and_author_flag(self):
        plan=mod.plan_records(self.ledger,self.metadata,set(),set(),set())
        self.assertEqual(Counter(x["split"] for x in plan),{"train":77,"val":9,"test":19})
        self.assertEqual(Counter(x["status"] for x in plan),
                         {"candidate":65,"excluded_author_analysis_flag":12,
                          "excluded_frozen_holdout":28})
        self.assertTrue(all(x["record"]["subject_group"] not in mod.HOLD and
                            x["record"]["subject_group"]!=mod.AUTHOR_EXCLUDED
                            for x in plan if x["status"]=="candidate"))

    def test_prior_payload_denial_blocks_train(self):
        first=next(x for x in self.ledger if x["subject_group"]=="openneuro:ds005028:sub-02")
        out=mod.plan_records(self.ledger,self.metadata,set(),
                             {first["EEG_payload_sha256"]},set())
        self.assertEqual(len([x for x in out if x["reason"]=="prior_heldout_or_review_deny"]),1)

    def test_frozen_cross_subject_identity_rejected(self):
        mutable=[dict(x) for x in self.ledger]
        a=next(x for x in mutable if x["subject_group"]=="openneuro:ds005028:sub-01")
        b=next(x for x in mutable if x["subject_group"]=="openneuro:ds005028:sub-02")
        b["EEG_payload_sha256"]=a["EEG_payload_sha256"]
        b["calibration_sha256"]=a["calibration_sha256"]
        with self.assertRaises(ValueError):
            mod.plan_records(mutable,self.metadata,set(),set(),set())

    def test_causal_window_constant_prefix(self):
        fs=mod.NATIVE_HZ
        t=np.arange(2800)/fs
        x=(20e-6*np.sin(2*np.pi*10*t))[None,:]*np.ones((32,1))
        a,ivals,_=mod.windows32(x)
        self.assertEqual(a.shape[1:],(32,3,200))
        self.assertGreater(len(ivals),0)
        extra=np.concatenate([x, np.zeros((32,1400))],axis=1)
        b,ivals2,_=mod.windows32(extra)
        np.testing.assert_array_equal(a,b[:len(a)])
        self.assertEqual(ivals,ivals2[:len(ivals)])

    def test_flatline_rejected_without_amplitude_repair(self):
        a,ivals,rejected=mod.windows32(np.zeros((32,3000)))
        self.assertEqual(a.shape[0],0)
        self.assertEqual(len(ivals),0)
        self.assertGreater(rejected.get("flat_below_fixed_0_01uV",0),0)
        with self.assertRaisesRegex(ValueError,"native_peak"):
            mod.windows32(np.full((32,3000),.11))

    def test_no_label_fields_in_export_contract(self):
        self.assertNotIn("stimulus",mod.SCIENTIFIC_SCOPE.lower().replace("stimulus/event", ""))
        self.assertEqual(mod.RELEASE,"openneuro-ds005028-current-public-S3-e61cb6abf724")


if __name__=="__main__":
    unittest.main()
