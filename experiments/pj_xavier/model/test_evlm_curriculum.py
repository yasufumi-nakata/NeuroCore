import copy
import json
import hashlib
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from evlm_curriculum import (
    SCHEMA, atomic, checkpoint_steps, make_stage_config, read_plan,
    sha256, stage_specs, verified_checkpoint, verify_vision_receipt,
    numeric_policy, numeric_score, evaluate_numeric_retention, planned_updates, remaining_presentations,
)

class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        runner = self.root / "runner.py"
        runner.write_text("# pinned test-only runner\n")
        probe = self.root / "probe.json"
        base_path = str((self.root / "model").resolve())
        base_identity = hashlib.sha256(json.dumps({
            "model_path": base_path, "config_sha256": "a"*64,
            "receipt_sha256": "b"*64,
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        atomic(probe, {
            "base_identity": base_identity,
            "mode": "engineering_probe", "quality_gates_passed": True,
            "completed_optimizer_steps": 2, "test_opened": False,
            "num_layers": 64, "rank": 8, "max_seq_length": 1024,
            "expected_model_config_sha256": "a"*64,
        })
        data = self.root / "data.json"
        atomic(data, {"schema": "evlm-numeric-data-v1", "inputs": {"train": {"count": 801}}, "source_ids": ["source_a"]})
        self.plan = {
            "schema": SCHEMA, "curriculum_id": "test", "run_root": str(self.root / "run"),
            "max_stage_updates": 400, "max_controller_wall_seconds": 3600,
            "scope_approved_by_probe": {
                "passed": True, "num_layers": 64,
                "receipt": {"path": str(probe), "sha256": sha256(probe)},
            },
            "runner": {"path": str(runner), "sha256": sha256(runner)},
            "python_path": __import__("sys").executable,
            "sources": [{
                "source_id": "source_a", "train_records": 801,
                "admission_status": "admitted_train_only",
                "test_or_review_used": False, "data_schema": "evlm-numeric-data-v1",
                "data_manifest": {"path": str(data), "sha256": sha256(data)},
            }],
            "stage_template": {
                "schema": "evlm-mlx-train-v1", "mode": "train_stage",
                "num_layers": 64, "rank": 8, "max_seq_length": 1024,
                "expected_model_config_sha256": "a"*64, "memory_limit_gb": 54,
                "model_path": base_path, "base_receipt": {"path": "/unused_fixture", "sha256": "b"*64},
            },
            "vision_gate": {"enabled": True, "runtime": {"path": str(runner), "sha256": sha256(runner)},
                            "python_path": __import__("sys").executable,
                            "base_revision": "f"*40, "max_wall_seconds": 900, "memory_gb": 54},
            "final_test_policy": "never_open", "automatic_scope_expansion": False,
        }
        self.plan_path = self.root / "plan.json"

    def tearDown(self):
        self.temp.cleanup()

    def validate(self):
        atomic(self.plan_path, self.plan)
        return read_plan(self.plan_path)

    def test_valid_pinned_plan(self):
        self.assertEqual(self.validate()["curriculum_id"], "test")

    def test_reduced_context_requires_matching_successful_probe(self):
        self.plan["stage_template"]["max_seq_length"] = 801
        with self.assertRaisesRegex(ValueError, "probe_context_mismatch"):
            self.validate()
        p = Path(self.plan["scope_approved_by_probe"]["receipt"]["path"])
        data = json.loads(p.read_text())
        data["max_seq_length"] = 801
        atomic(p, data)
        self.plan["scope_approved_by_probe"]["receipt"]["sha256"] = sha256(p)
        self.assertEqual(self.validate()["stage_template"]["max_seq_length"], 801)

    def test_every_record_once_in_offsets(self):
        specs = list(stage_specs(self.plan))
        self.assertEqual([(x["order_offset"],x["max_updates"]) for x in specs],
                         [(0,400),(400,400),(800,1)])
        covered = [i for x in specs for i in range(x["order_offset"], x["order_offset"]+x["max_updates"])]
        self.assertEqual(covered, list(range(801)))

    def test_initial_small_stage_covers_every_record_without_repetition(self):
        self.plan["initial_stage_updates"] = 20
        self.validate()
        specs = list(stage_specs(self.plan))
        self.assertEqual([(x["order_offset"], x["max_updates"]) for x in specs],
                         [(0, 20), (20, 400), (420, 381)])
        self.assertEqual(sum(x["max_updates"] for x in specs), 801)

    def test_registered_schedule_pauses_at_400_without_shrinking_source(self):
        self.plan.update(stage_update_schedule=[20, 80, 300], max_total_optimizer_updates=400)
        self.validate()
        specs = list(stage_specs(self.plan))
        self.assertEqual([(s["order_offset"], s["max_updates"]) for s in specs], [(0,20),(20,80),(100,300)])
        self.assertEqual(planned_updates(self.plan), 400)
        self.assertEqual(remaining_presentations(self.plan), 801)
        self.assertEqual(self.plan["sources"][0]["train_records"], 801)
        self.plan["max_total_optimizer_updates"] = 75
        self.assertEqual([(s["order_offset"],s["max_updates"]) for s in stage_specs(self.plan)], [(0,20),(20,55)])

    def test_invalid_schedule_and_total_budget_fail_closed(self):
        self.plan["stage_update_schedule"] = [20, 0, 300]
        with self.assertRaisesRegex(ValueError, "stage_update_schedule"):
            self.validate()
        self.plan["stage_update_schedule"] = [20,80,300]
        self.plan["max_total_optimizer_updates"] = True
        with self.assertRaisesRegex(ValueError, "total_update_budget"):
            self.validate()

    def test_explicit_continuation_does_not_repeat_finished_prefix(self):
        self.plan["sources"][0]["start_order_offset"] = 20
        self.validate()
        specs = list(stage_specs(self.plan))
        self.assertEqual([(x["order_offset"], x["max_updates"]) for x in specs],
                         [(20, 400), (420, 381)])
        self.plan["sources"][0]["start_order_offset"] = 801
        with self.assertRaisesRegex(ValueError, "invalid_explicit_continuation_offset"):
            self.validate()

    def test_next_source_restarts_its_own_order(self):
        src = copy.deepcopy(self.plan["sources"][0])
        src.update(source_id="source_b", train_records=3)
        self.plan["sources"].append(src)
        specs = list(stage_specs(self.plan))
        self.assertEqual(specs[-1]["source_id"], "source_b")
        self.assertEqual(specs[-1]["order_offset"], 0)
        self.assertEqual(specs[-1]["max_updates"], 3)

    def test_immutable_runner_hash(self):
        Path(self.plan["runner"]["path"]).write_text("# replaced\n")
        with self.assertRaisesRegex(ValueError, "runner_hash_mismatch"):
            self.validate()

    def test_immutable_data_hash(self):
        Path(self.plan["sources"][0]["data_manifest"]["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "data_manifest_hash_mismatch"):
            self.validate()

    def test_duplicate_source_is_rejected(self):
        self.plan["sources"].append(copy.deepcopy(self.plan["sources"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate_source_id"):
            self.validate()

    def test_same_data_cannot_be_renamed_as_new_source(self):
        src = copy.deepcopy(self.plan["sources"][0])
        src["source_id"] = "renamed"
        self.plan["sources"].append(src)
        with self.assertRaisesRegex(ValueError, "duplicate_data_manifest"):
            self.validate()

    def test_heldout_and_unadmitted_sources_are_rejected(self):
        self.plan["sources"][0]["test_or_review_used"] = True
        with self.assertRaisesRegex(ValueError, "source_heldout_policy_missing"):
            self.validate()
        self.plan["sources"][0]["test_or_review_used"] = False
        self.plan["sources"][0]["admission_status"] = "downloaded_only"
        with self.assertRaisesRegex(ValueError, "source_not_admitted"):
            self.validate()

    def test_partial_count_cannot_be_reported_as_full_source(self):
        self.plan["sources"][0]["train_records"] = 400
        with self.assertRaisesRegex(ValueError, "source_train_count_differs_from_manifest"):
            self.validate()

    def test_source_label_is_bound_to_manifest(self):
        self.plan["sources"][0]["source_id"] = "unrelated_source"
        with self.assertRaisesRegex(ValueError, "source_id_not_in_manifest"):
            self.validate()

    def test_canonical_source_is_distinct_from_path_slug(self):
        source = self.plan["sources"][0]
        data_path = Path(source["data_manifest"]["path"])
        data = json.loads(data_path.read_text())
        data["source_ids"] = ["zenodo:806023"]
        atomic(data_path, data)
        source.update(source_id="zenodo_806023", canonical_source_id="zenodo:806023")
        source["data_manifest"]["sha256"] = sha256(data_path)
        self.validate()
        spec = next(stage_specs(self.plan))
        self.assertEqual(spec["canonical_source_id"], "zenodo:806023")
        self.assertNotIn(":", spec["stage_id"])

    def test_text_only_curriculum_cannot_skip_vision_gate(self):
        self.plan["vision_gate"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "evlm_curriculum_requires_vision_retention_gate"):
            self.validate()

    def test_vision_gate_binds_new_adapter_and_baseline(self):
        bundle = self.root / "bundle"
        (bundle / "adapter").mkdir(parents=True)
        (bundle / "adapter/adapters.safetensors").write_bytes(b"new-adapter")
        atomic(bundle / "manifest.json", {"fixture": True})
        parent = {"adapter_sha256": sha256(bundle / "adapter/adapters.safetensors")}
        receipt = {"status": "passed", "synthetic_only": True, "cases_per_mode": 2,
                   "base_correct": 2, "adapter_correct": 2, "test_opened": False,
                   "executed_actions": 0, "manifest_sha256": sha256(bundle / "manifest.json")}
        atomic(bundle / "smoke.json", receipt)
        atomic(bundle / "candidate_status.json", {"status": "synthetic_smoke_passed",
               "receipt": "smoke.json", "receipt_sha256": sha256(bundle / "smoke.json")})
        self.assertEqual(verify_vision_receipt(bundle, parent)["adapter_correct"], 2)
        receipt["base_correct"] = 0
        atomic(bundle / "smoke.json", receipt)
        atomic(bundle / "candidate_status.json", {"status": "synthetic_smoke_passed",
               "receipt": "smoke.json", "receipt_sha256": sha256(bundle / "smoke.json")})
        with self.assertRaisesRegex(ValueError, "vision_control_or_adapter_failed"):
            verify_vision_receipt(bundle, parent)

    def fixture_process_flow(self, budget=None, numeric=False, regression=False):
        fake = self.root / "fixture_runner.py"
        fake.write_text("""import sys,json,hashlib,shutil,random
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
args=sys.argv[1:]
def arg(flag): return args[args.index(flag)+1]
action=args[0]
if action=='preflight':
 write(arg('--receipt'),{'status':'passed'})
elif action=='run':
 c=json.loads(Path(arg('--config')).read_text());r=Path(c['run_dir']);r.mkdir(parents=True)
 parent=c['warm_start_adapter']
 if parent:
  assert sha(Path(parent['checkpoint_dir'])/'adapters.safetensors')==parent['adapter_sha256']
 cp=r/'checkpoints'/('%07d'%c['max_updates']);cp.mkdir(parents=True)
 a=cp/'adapters.safetensors';a.write_bytes((c['run_id']+str(parent)).encode())
 write(cp/'evaluation.json',{'status':'completed','quality_gates_passed':True,'test_opened':False,
 'completed_optimizer_steps':c['max_updates'],'adapter_sha256':sha(a)})
 if c.get('fixture_numeric'):
  data=json.loads(Path(c['data_manifest']['path']).read_text())
  inds=list(range(data['inputs']['val']['count']));random.Random(c['seed']+2).shuffle(inds);inds=sorted(inds[:36])
  before=10-c['order_offset']*.1;after=(20 if c.get('fixture_regression') else before-c['max_updates']*.1)
  def metrics(m):return {'n':36,'valid_count':36,'valid_rate':1.0,'invalid_rate':0.0,'mse_on_valid_predictions':m}
  write(r/'baseline_task_evaluation.json',{'completed_optimizer_steps':0,'test_opened':False,
   'task_kind':'quantized_eeg_reconstruction','data_manifest_sha256':c['data_manifest']['sha256'],
   'selected_validation_indices':inds,'metrics':metrics(before)})
  for p in [r/'baseline_task_predictions.private.json',cp/'predictions.private.json']:
   write(p,{'indices':inds,'outputs':['[1,2,3,4]']*36})
  ev=json.loads((cp/'evaluation.json').read_text());ev['metrics']=metrics(after)
  ev['metrics']['initial_model_comparison']={'n':36,'paired_valid_examples':36,'paired_valid_target_values':144,
   'initial_valid_rate':1.0,'current_valid_rate':1.0,'initial_mse_paired_valid':before,
   'current_mse_paired_valid':after,'mse_delta_current_minus_initial':after-before}
  ev['baseline_task_evaluation_sha256']=sha(r/'baseline_task_evaluation.json');write(cp/'evaluation.json',ev)
 write(cp/'checkpoint.json',{'adapter_sha256':sha(a),'completed_optimizer_steps':c['max_updates'],
 'evaluation_sha256':sha(cp/'evaluation.json'),'parent':parent})
 write(r/'state.json',{'status':'completed','completed_optimizer_steps':c['max_updates']})
elif action=='prepare':
 out=Path(arg('--output'));(out/'adapter').mkdir(parents=True)
 source=Path(arg('--adapter'))/'adapters.safetensors'
 assert sha(source)==arg('--expected-adapter-sha256')
 shutil.copyfile(source,out/'adapter/adapters.safetensors')
 write(out/'manifest.json',{'fixture':True})
elif action=='smoke':
 out=Path(arg('--bundle'))
 write(out/'smoke.json',{'status':'passed','synthetic_only':True,'cases_per_mode':2,'base_correct':2,
 'adapter_correct':2,'test_opened':False,'executed_actions':0,'manifest_sha256':sha(out/'manifest.json')})
 write(out/'candidate_status.json',{'status':'synthetic_smoke_passed','receipt':'smoke.json',
 'receipt_sha256':sha(out/'smoke.json')})
else: raise RuntimeError('unrecognized fixture command')
""")
        self.plan["runner"] = {"path": str(fake), "sha256": sha256(fake)}
        self.plan["vision_gate"]["runtime"] = self.plan["runner"]
        self.plan["stage_template"]["lock_path"] = str(self.root / "fixture.lock")
        self.plan["max_stage_updates"] = 2
        src = self.plan["sources"][0]
        src["train_records"] = 3
        mp = Path(src["data_manifest"]["path"])
        data = json.loads(mp.read_text())
        data["inputs"]["train"]["count"] = 3
        atomic(mp, data)
        src["data_manifest"]["sha256"] = sha256(mp)
        if budget is not None:
            self.plan["max_total_optimizer_updates"] = budget
        if numeric:
            self.enable_numeric()
            self.plan["stage_template"].update(fixture_numeric=True, fixture_regression=regression)
            if budget is not None:
                self.plan["initial_stage_updates"] = 1
        self.validate()
        runner = Path(__file__).with_name("evlm_curriculum.py")
        result = subprocess.run([sys.executable, str(runner), "run", "--plan", str(self.plan_path)],
                                capture_output=True, text=True, timeout=35)
        self.assertEqual(result.returncode, 2 if regression else 0, result.stdout+result.stderr)
        state = json.loads((Path(self.plan["run_root"])/"curriculum_state.json").read_text())
        return state

    def test_complete_two_stage_process_flow_preserves_warm_start(self):
        state = self.fixture_process_flow()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["completed_stages"], 2)
        self.assertEqual(state["completed_optimizer_updates"], 3)
        cfgs = sorted((Path(self.plan["run_root"])/"configs").glob("*.json"))
        second = json.loads(cfgs[1].read_text())
        self.assertEqual(second["order_offset"], 2)
        self.assertIsNotNone(second["warm_start_adapter"])
        self.assertTrue(Path(state["latest_vlm_bundle"]).is_dir())

    def test_numeric_two_stage_flow_preserves_best_and_pauses_budget(self):
        state = self.fixture_process_flow(budget=2, numeric=True)
        self.assertEqual(state["status"], "paused_update_budget")
        self.assertEqual(state["completed_optimizer_updates"], 2)
        self.assertEqual(state["completed_stages"], 2)
        self.assertEqual(len(list((Path(self.plan["run_root"])/"configs").glob('*.json'))), 2)
        self.assertEqual(state["remaining_unpresented_records"], 1)
        self.assertEqual(state["numeric_retention"]["best_penalized_score"], 9.8)

    def test_numeric_regression_preserves_receipts_and_starts_no_next_stage(self):
        state = self.fixture_process_flow(numeric=True, regression=True)
        self.assertEqual(state["status"], "stopped_numeric_gate")
        self.assertEqual(state["completed_optimizer_updates"], 2)
        self.assertEqual(state["completed_stages"], 0)
        self.assertEqual(len(list((Path(self.plan["run_root"])/"configs").glob('*.json'))), 1)
        self.assertTrue(Path(state["last_numeric_retention_receipt"]).is_file())
        self.assertEqual(state["numeric_retention"]["best_penalized_score"], 10)

    def test_forged_probe_pass_flag_does_not_override_receipt(self):
        p = Path(self.plan["scope_approved_by_probe"]["receipt"]["path"])
        d = json.loads(p.read_text())
        d["quality_gates_passed"] = False
        atomic(p, d)
        self.plan["scope_approved_by_probe"]["receipt"]["sha256"] = sha256(p)
        with self.assertRaisesRegex(ValueError, "probe_quality_failed"):
            self.validate()

    def test_scope_cannot_expand_after_probe(self):
        self.plan["stage_template"]["num_layers"] = 4
        with self.assertRaisesRegex(ValueError, "scope_differs_from_probe"):
            self.validate()

    def test_checkpoint_schedule_has_final_even_for_small_remainder(self):
        self.assertEqual(checkpoint_steps(1), [1])
        self.assertEqual(checkpoint_steps(20), [20])
        self.assertEqual(checkpoint_steps(400), [20,100,200,400])

    def test_warm_start_is_explicit_in_next_stage(self):
        spec = list(stage_specs(self.plan))[1]
        parent = {"checkpoint_dir": "/a", "adapter_sha256": "c"*64, "checkpoint_sha256": "d"*64}
        cfg = make_stage_config(self.plan, spec, parent)
        self.assertEqual(cfg["order_offset"], 400)
        self.assertEqual(cfg["warm_start_adapter"], parent)
        self.assertEqual(cfg["mode"], "train_stage")

    def make_checkpoint(self):
        cfg = {"run_dir": str(self.root / "stage"), "max_updates": 20}
        run = Path(cfg["run_dir"])
        cp = run / "checkpoints" / "0000020"
        cp.mkdir(parents=True)
        adapter = cp / "adapters.safetensors"
        adapter.write_bytes(b"test-fixture-only")
        atomic(run / "state.json", {"status": "completed", "completed_optimizer_steps": 20})
        atomic(cp / "evaluation.json", {"status": "completed", "quality_gates_passed": True, "test_opened": False,
                                       "adapter_sha256": sha256(adapter), "completed_optimizer_steps": 20})
        atomic(cp / "checkpoint.json", {"adapter_sha256": sha256(adapter),
                                       "completed_optimizer_steps": 20,
                                       "evaluation_sha256": sha256(cp / "evaluation.json")})
        return cfg, cp

    def enable_numeric(self):
        val = self.root / "numeric_val.jsonl"
        val.write_text('synthetic validation fixture\n')
        src = self.plan["sources"][0]
        path = Path(src["data_manifest"]["path"])
        data = json.loads(path.read_text())
        data.update(task_kind="quantized_eeg_reconstruction", quantization_schema="eeg-raw32-uv1-channel-major-cycle19-v1",
                    quantization_bounds=[-100,100])
        data["inputs"]["val"] = {"path":str(val),"sha256":sha256(val),"count":80}
        atomic(path,data)
        src["data_manifest"]["sha256"] = sha256(path)
        self.plan["numeric_retention_gate"] = {"enabled":True}
        self.plan["stage_template"].update(seed=123,eval_max_records=36)

    def numeric_fixture(self, before=10, current=9, before_valid=36, current_valid=36):
        import random
        self.enable_numeric()
        cfg,cp = self.make_checkpoint()
        cfg.update(seed=123,data_manifest=self.plan["sources"][0]["data_manifest"],warm_start_adapter=None)
        indices=list(range(80));random.Random(125).shuffle(indices);indices=sorted(indices[:36])
        def metrics(value, valid):
            return {"n":36,"valid_count":valid,"valid_rate":valid/36,"invalid_rate":1-valid/36,
                    "mse_on_valid_predictions":value if valid else None}
        b={"completed_optimizer_steps":0,"test_opened":False,"task_kind":"quantized_eeg_reconstruction",
           "data_manifest_sha256":cfg['data_manifest']['sha256'],"selected_validation_indices":indices,
           "metrics":metrics(before,before_valid)}
        baseline=Path(cfg["run_dir"])/"baseline_task_evaluation.json";atomic(baseline,b)
        for p in (Path(cfg["run_dir"])/"baseline_task_predictions.private.json",cp/"predictions.private.json"):
            atomic(p,{"indices":indices,"outputs":["[1,2,3,4]"]*36})
        ev=json.loads((cp/"evaluation.json").read_text());ev["metrics"]=metrics(current,current_valid)
        n=min(before_valid,current_valid)
        ev["metrics"]["initial_model_comparison"]={"n":36,"paired_valid_examples":n,"paired_valid_target_values":4*n,
            "initial_valid_rate":before_valid/36,"current_valid_rate":current_valid/36,
            "initial_mse_paired_valid":before if n else None,"current_mse_paired_valid":current if n else None,
            "mse_delta_current_minus_initial":current-before if n else None}
        ev["baseline_task_evaluation_sha256"]=sha256(baseline);atomic(cp/"evaluation.json",ev)
        parent={"checkpoint_dir":str(cp),"adapter_sha256":"c"*64,"checkpoint_sha256":"d"*64}
        return cfg,parent,ev

    def test_numeric_gate_is_optional_and_rejects_unknown_thresholds(self):
        self.assertIsNone(numeric_policy(self.plan))
        self.plan["numeric_retention_gate"]={"enabled":True,"relative_tolerance":.2}
        with self.assertRaisesRegex(ValueError,"unregistered_numeric"):
            numeric_policy(self.plan)

    def test_numeric_score_uses_invalid_penalty_not_valid_only_mse(self):
        policy=numeric_policy({"numeric_retention_gate":{"enabled":True}})
        m={"n":36,"valid_count":35,"valid_rate":35/36,"invalid_rate":1/36,"mse_on_valid_predictions":9}
        self.assertAlmostEqual(numeric_score(m,policy),35/36*9+40000/36)
        m["mse_on_valid_predictions"]=float('nan')
        with self.assertRaisesRegex(ValueError,"unknown_numeric_value"):
            numeric_score(m,policy)

    def test_paired_regression_cannot_hide_behind_format_improvement(self):
        cfg,parent,ev=self.numeric_fixture(before=1,current=2,before_valid=33)
        receipt,state=evaluate_numeric_retention(self.plan,cfg,parent,ev)
        self.assertFalse(receipt["passed"])
        self.assertIn("paired_numeric_mse_regressed_from_stage_start",receipt["reasons"])
        self.assertLess(receipt["current_penalized_score"],receipt["initial_model_penalized_score"])
        self.assertEqual(state["best_reference"],"first_stage_initial_model")

    def test_two_flat_stages_stop_and_best_is_not_reset(self):
        cfg,parent,ev=self.numeric_fixture(before=10,current=10)
        first,state=evaluate_numeric_retention(self.plan,cfg,parent,ev)
        self.assertTrue(first["passed"])
        second,state=evaluate_numeric_retention(self.plan,cfg,parent,ev,state)
        self.assertFalse(second["passed"])
        self.assertEqual(state["stagnant_stages"],2)
        self.assertIn("two_stages_without_new_best",second["reasons"])

    def test_insufficient_paired_population_is_explicit_not_fabricated(self):
        cfg,parent,ev=self.numeric_fixture(before_valid=0)
        receipt,state=evaluate_numeric_retention(self.plan,cfg,parent,ev)
        self.assertTrue(receipt["passed"])
        self.assertFalse(receipt["paired_gate_evaluated"])
        self.assertFalse(receipt["paired_numeric_improvement_observed"])
        self.assertEqual(receipt["initial_model_penalized_score"],40000)

    def test_baseline_hash_and_development_indices_are_bound(self):
        cfg,parent,ev=self.numeric_fixture()
        valid_hash=ev["baseline_task_evaluation_sha256"]
        ev["baseline_task_evaluation_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"baseline_hash_mismatch"):
            evaluate_numeric_retention(self.plan,cfg,parent,ev)
        ev["baseline_task_evaluation_sha256"]=valid_hash
        p=Path(parent["checkpoint_dir"])/"predictions.private.json"
        x=json.loads(p.read_text());x['indices'][0]=999;atomic(p,x)
        with self.assertRaisesRegex(ValueError,"development_selection_mismatch"):
            evaluate_numeric_retention(self.plan,cfg,parent,ev)

    def test_different_validation_across_sources_is_rejected(self):
        self.enable_numeric()
        src=copy.deepcopy(self.plan['sources'][0]);src['source_id']='source_b'
        d=json.loads(Path(src['data_manifest']['path']).read_text());d['source_ids']=['source_b']
        other=self.root/'different_val.jsonl';other.write_text('different fixture')
        d['inputs']['val'].update(path=str(other),sha256=sha256(other))
        p=self.root/'second_data.json';atomic(p,d);src['data_manifest']={'path':str(p),'sha256':sha256(p)}
        self.plan['sources'].append(src)
        with self.assertRaisesRegex(ValueError,'validation_differs_across_sources'):
            self.validate()

    def test_failed_stage_cannot_promote_saved_adapter(self):
        cfg, cp = self.make_checkpoint()
        atomic(Path(cfg["run_dir"]) / "state.json",
               {"status": "failed", "completed_optimizer_steps": 20})
        with self.assertRaisesRegex(ValueError, "stage_not_completed"):
            verified_checkpoint(cfg)

    def test_failed_quality_cannot_promote_saved_adapter(self):
        cfg, cp = self.make_checkpoint()
        atomic(cp / "evaluation.json", {"quality_gates_passed": False, "test_opened": False})
        with self.assertRaisesRegex(ValueError, "stage_quality_gates_not_passed"):
            verified_checkpoint(cfg)

    def test_saved_adapter_is_rehashed_before_warm_start(self):
        cfg, cp = self.make_checkpoint()
        (cp / "adapters.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "saved_adapter_hash_mismatch"):
            verified_checkpoint(cfg)

    def test_valid_checkpoint_returns_precise_parent_binding(self):
        cfg, cp = self.make_checkpoint()
        parent, state, evaluation = verified_checkpoint(cfg)
        self.assertEqual(parent["checkpoint_sha256"], sha256(cp / "checkpoint.json"))
        self.assertEqual(parent["adapter_sha256"], sha256(cp / "adapters.safetensors"))

if __name__ == "__main__":
    unittest.main()
