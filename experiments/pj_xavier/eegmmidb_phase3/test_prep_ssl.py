import copy
import importlib.util
from pathlib import Path
import tempfile
import time
import unittest

import prep_ssl as p


def record(subject="S001",split="train",source="a",payload="b"):
    return {"subject_group":p.DATASET+":"+subject,"split":split,"status":"continuous_SSL_source_allowed",
            "source_recording_id":p.DATASET+":"+subject+":R01","source_eeg_sha256":source*64,
            "eeg_payload_sha256":payload*64,"run":1,"channels":["EEG"+str(i) for i in range(64)]}


class TestContinuousPreparation(unittest.TestCase):
    def test_grid_is_annotation_independent_nonoverlap_and_history_guarded(self):
        spans=p.continuous_spans(37.975)
        self.assertEqual(len(spans),10)
        self.assertEqual(spans[0]["window_start_s"],5)
        self.assertEqual(spans[-1]["window_end_s"],35)
        for a,b in zip(spans,spans[1:]):
            self.assertEqual(a["sample_stop"],b["sample_start"])
            self.assertEqual(a["window_end_s"],b["window_start_s"])
        self.assertEqual(p.continuous_spans(7.999),[])
        self.assertEqual(len(p.continuous_spans(8)),1)

    def test_test_sources_and_test_alias_payload_are_rejected(self):
        r=record(split="test")
        with self.assertRaisesRegex(ValueError,"Test"):
            p.validate_allowlist([r],[],{r["subject_group"]:"test"})
        r=record();denied=record("S002","test","c","b")
        with self.assertRaisesRegex(ValueError,"Denied raw/payload"):
            p.validate_allowlist([r],[denied],{r["subject_group"]:"train"})

    def test_partition_change_and_cue_column_are_rejected(self):
        r=record()
        with self.assertRaisesRegex(ValueError,"partition"):
            p.validate_allowlist([r],[],{r["subject_group"]:"val"})
        for forbidden in ("events","label","task_family","gold_annotation_code"):
            x={**r,forbidden:"unwanted"}
            with self.subTest(field=forbidden),self.assertRaisesRegex(ValueError,"Supervision"):
                p.validate_allowlist([x],[],{r["subject_group"]:"train"})

    def test_same_payload_crossing_train_validation_rejected(self):
        a=record();b=record("S002","val","c","b")
        with self.assertRaisesRegex(ValueError,"crosses"):
            p.validate_allowlist([a,b],[],{a["subject_group"]:"train",b["subject_group"]:"val"})

    def test_registered_budget_and_original_absolute_deadline_required(self):
        with tempfile.TemporaryDirectory() as d:
            r={"status":"proposal_not_registered","absolute_deadline_epoch":time.time()+30,
               "max_output_bytes":10,"disk_reserve_bytes":0}
            with self.assertRaisesRegex(ValueError,"registered"):p.guard(r,d)
            r["status"]="authorized_registered";r["absolute_deadline_epoch"]=0
            with self.assertRaisesRegex(ValueError,"deadline"):p.guard(r,d)
            r["absolute_deadline_epoch"]=time.time()+30
            with self.assertRaisesRegex(ValueError,"budget"):p.guard(r,d,11)

    def test_pinned_phase2_operator_has_unchanged_500uV_and_units(self):
        op=p.load_operator(Path(__file__).resolve().parent.parent/"eegmmidb_phase2/prep_causal_v2.py")
        self.assertEqual(op.MAX_ABS_V,5e-4)
        self.assertEqual(op.MIN_STD_V,1e-8)
        self.assertEqual(op.FS,200)
        with tempfile.TemporaryDirectory() as d:
            bad=Path(d)/"different.py";bad.write_text("print('not executed')")
            with self.assertRaisesRegex(ValueError,"changed"):p.load_operator(bad)

    def test_no_overwriting_old_contracts(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/"old.json";p.write_json(target,{"frozen":True})
            before=target.read_bytes()
            with self.assertRaisesRegex(ValueError,"Existing immutable"):p.write_json(target,{"frozen":False})
            self.assertEqual(before,target.read_bytes())


if __name__=="__main__":unittest.main()
