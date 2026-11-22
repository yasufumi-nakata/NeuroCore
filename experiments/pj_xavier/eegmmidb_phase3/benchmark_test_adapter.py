"""Admission and five-model identity regression fixtures; no model inference."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest

import benchmark_adapter as adapter
import harness_compare as harness

class AdapterTests(unittest.TestCase):
    def budget(self):
        return {"session":"xavier-eegmmidb-20261122-p2","remote_root":"/content/xavier_phase3","GPU_memory_budget_gib":21,
                "colab_access":"workstation_CLI_only","phase_limits_seconds":{"heldout_evaluation_and_routing":900},"absolute_deadline_epoch":1795331083.750155}
    def test_only_case_alias_added_to_original_budget(self):
        original=self.budget();before=copy.deepcopy(original);new=adapter.normalized_budget(original)
        self.assertEqual(original,before);self.assertNotIn("gpu_memory_budget_gib",original)
        self.assertEqual(new,{**before,"gpu_memory_budget_gib":21})
    def test_wrong_lease_extension_memory_or_route_rejected(self):
        changes=[("session","other"),("absolute_deadline_epoch",1795332083.750155),("GPU_memory_budget_gib",22),("colab_access","direct_API")]
        for field,value in changes:
            with self.subTest(field=field),self.assertRaises(ValueError):adapter.normalized_budget({**self.budget(),field:value})
        bad=self.budget();bad["phase_limits_seconds"]["heldout_evaluation_and_routing"]=1800
        with self.assertRaises(ValueError):adapter.normalized_budget(bad)
    def test_old_and_new_models_cannot_be_missing_duplicated_or_aliased(self):
        entries=[(name,"/fixture/"+name) for name in harness.MODELS]
        self.assertEqual(set(harness.checked_evaluations(entries)),set(harness.MODELS))
        for values in (entries[:-1],entries+[entries[0]],entries[:-1]+[(entries[-1][0],entries[0][1])],entries[:-1]+[("unknown","/fixture/extra")]):
            with self.subTest(values=values),self.assertRaises(ValueError):harness.checked_evaluations(values)
    def make_report(self,root,name,new=False):
        root.mkdir();(root/"predictions.jsonl").write_text("fixture predictions, never empirical EEG\n")
        checkpoint="new-fixed-checkpoint" if new else harness.OLD_CHECKPOINTS[name]
        report={"status":"completed","execution_seal_sha256":"new" if new else "old","manifest_sha256":"manifest",
                "checkpoint_sha256":checkpoint,"predictions_sha256":adapter.sha(root/"predictions.jsonl"),"full_test_records":1489,"metadata_paired_subset_records":1254}
        adapter.save(root/"evaluation_report.json",report)
        completion={"status":"completed","exit_code":0,"report_sha256":adapter.sha(root/"evaluation_report.json"),"checkpoint_sha256":checkpoint}
        adapter.save(root/"completion.json",completion)
        if new:adapter.save(root/"phase3_context.json",{"status":"completed","execution_seal_sha256":"new","report_sha256":completion["report_sha256"],"checkpoint_sha256":checkpoint})
        return report
    def test_old_receipts_remain_under_original_seal_and_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"old";name="phase2_scratch_eeg";self.make_report(path,name)
            before={p.name:p.read_bytes() for p in path.iterdir()}
            harness.check_report_identity(name,path,"old","new","manifest")
            self.assertEqual(before,{p.name:p.read_bytes() for p in path.iterdir()})
            with self.assertRaises(ValueError):harness.check_report_identity(name,path,"new","new","manifest")
    def test_new_models_require_separate_context_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"new";name="phase3_ssl_eeg";self.make_report(path,name,new=True)
            harness.check_report_identity(name,path,"old","new","manifest")
            value=adapter.read(path/"phase3_context.json");value["checkpoint_sha256"]="other"
            (path/"phase3_context.json").write_text(json.dumps(value))
            with self.assertRaises(ValueError):harness.check_report_identity(name,path,"old","new","manifest")
    def test_unexpected_old_checkpoint_or_mutated_prediction_is_rejected(self):
        for mutation in ("checkpoint","prediction"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/"old";name="phase2_csp_lda";report=self.make_report(path,name)
                if mutation=="checkpoint":
                    report["checkpoint_sha256"]="other";(path/"evaluation_report.json").write_text(json.dumps(report))
                    value=adapter.read(path/"completion.json");value["checkpoint_sha256"]="other";value["report_sha256"]=adapter.sha(path/"evaluation_report.json");(path/"completion.json").write_text(json.dumps(value))
                else:(path/"predictions.jsonl").write_text("changed")
                with self.assertRaises(ValueError):harness.check_report_identity(name,path,"old","new","manifest")
    def test_wrong_paired_denominator_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"old";name="phase2_fused_smollm2";report=self.make_report(path,name)
            report["metadata_paired_subset_records"]=1489;(path/"evaluation_report.json").write_text(json.dumps(report))
            completion=adapter.read(path/"completion.json");completion["report_sha256"]=adapter.sha(path/"evaluation_report.json");(path/"completion.json").write_text(json.dumps(completion))
            with self.assertRaises(ValueError):harness.check_report_identity(name,path,"old","new","manifest")

if __name__=="__main__":unittest.main()
