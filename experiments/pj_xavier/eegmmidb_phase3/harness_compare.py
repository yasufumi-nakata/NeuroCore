"""Five explicit model identities, immutable predictions, unchanged synthetic oracle."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import benchmark_adapter as admission

MODELS=("phase2_csp_lda","phase2_scratch_eeg","phase2_fused_smollm2","phase3_ssl_eeg","phase3_ssl_fused_smollm2")
OLD_CHECKPOINTS={
    "phase2_csp_lda":"6e04bd439c765a7ac2332762ffe7305f1fb0a1cf4f3d7542e5155a186ce6915d",
    "phase2_scratch_eeg":"3d041e1a8c07e97ae1bebd873873964241d3856f8b6d7461f9134924f8c3f54d",
    "phase2_fused_smollm2":"13df40e53dab6087bfef2fac1a4e413a2d00178413f05f4759a0aa1ea849d4ee",
}

def checked_evaluations(values):
    result={}
    for name,path in values:
        admission.require(name in MODELS and name not in result,"unknown/duplicate model identity")
        result[name]=Path(path).resolve()
    admission.require(set(result)==set(MODELS),"all five explicit models are required")
    admission.require(len(set(result.values()))==len(MODELS),"one evaluation was aliased as multiple models")
    return result

def check_report_identity(name,folder,old_seal_sha,new_seal_sha,manifest_sha):
    report=admission.read(folder/"evaluation_report.json");completion=admission.read(folder/"completion.json")
    expected_seal=old_seal_sha if name in OLD_CHECKPOINTS else new_seal_sha
    admission.require(report["status"]=="completed" and report["execution_seal_sha256"]==expected_seal and report["manifest_sha256"]==manifest_sha,"model evaluation cohort/seal differs")
    admission.require(completion["status"]=="completed" and completion["exit_code"]==0 and completion["report_sha256"]==admission.sha(folder/"evaluation_report.json") and completion["checkpoint_sha256"]==report["checkpoint_sha256"],"incomplete/changed evaluation receipt")
    admission.require(admission.sha(folder/"predictions.jsonl")==report["predictions_sha256"],"recorded predictions changed")
    admission.require(report["full_test_records"]==1489 and report["metadata_paired_subset_records"]==1254,"unexpected evaluation denominators")
    if name in OLD_CHECKPOINTS:
        admission.require(report["checkpoint_sha256"]==OLD_CHECKPOINTS[name],"old selected model changed")
    else:
        context=admission.read(folder/"phase3_context.json")
        admission.require(context["status"]=="completed" and context["execution_seal_sha256"]==new_seal_sha and context["report_sha256"]==admission.sha(folder/"evaluation_report.json") and context["checkpoint_sha256"]==report["checkpoint_sha256"],"new evaluation lacks its separate phase3 context proof")
    return report

def score(args):
    seal=admission.verify_scope(args.seal,args.expected_seal_sha256,args.phase2_seal,args.phase2_code)
    phase2_code=Path(args.phase2_code).resolve()
    sys.path.insert(0,str(phase2_code))
    common=importlib.import_module("benchmark_common");audit=importlib.import_module("benchmark_audit")
    mapping=importlib.import_module("harness_mapping");frozen=importlib.import_module("harness_evaluate")
    for module in (common,audit,mapping,frozen):
        admission.require(Path(module.__file__).resolve()==phase2_code/Path(module.__file__).name,"unintended old harness import")
        admission.require(admission.sha(module.__file__)==seal["source_sha256"][Path(module.__file__).name],"frozen harness implementation changed")
    root=Path(args.prepared);out=Path(args.output)
    admission.require(admission.sha(root/"manifest.jsonl")==seal["prepared_manifest_sha256"],"test manifest changed")
    admission.require(admission.sha(root/"paired_control_subset.jsonl")==seal["paired_control_subset_sha256"],"paired control cohort changed")
    rows=common.read_rows(root/"manifest.jsonl");test={r["id"] for r in rows if r["split"]=="test"}
    paired={r["id"] for r in common.read_rows(root/"paired_control_subset.jsonl")} & test
    admission.require(len(test)==1489 and len(paired)==1254,"frozen cohort counts differ")
    evaluations=checked_evaluations(args.evaluation)
    verified={n:check_report_identity(n,p,admission.PHASE2_SEAL,args.expected_seal_sha256,seal["prepared_manifest_sha256"]) for n,p in evaluations.items()}
    routing=frozen.verify_cache(args.route_cache,seal)
    new_fused=verified["phase3_ssl_fused_smollm2"]
    admission.require(routing["checkpoint_sha256"]==new_fused["checkpoint_sha256"] and admission.sha(args.route_cache)==new_fused["routing_cache_sha256"],"router does not match the new frozen fused inference")
    admission.require(not out.exists(),"preserve previous harness comparison");out.mkdir(parents=True)
    started=time.monotonic();reports={};source_inputs={};all_episodes=0
    for name in MODELS:
        folder=evaluations[name];er=verified[name];predictions=common.read_rows(folder/"predictions.jsonl")
        proof=audit.audit_predictions(rows,predictions,er["checkpoint_sha256"])
        source_inputs[name]={"evaluation_report_sha256":admission.sha(folder/"evaluation_report.json"),"predictions_sha256":er["predictions_sha256"],"checkpoint_sha256":er["checkpoint_sha256"],"identity_audit":proof}
        source=predictions+mapping.gold_rows(predictions)
        begin=time.perf_counter();traces,summary=mapping.run_recorded(source,routing["cache"]);elapsed=time.perf_counter()-begin
        subset=[r for r in source if r["id"] in paired];paired_traces,paired_summary=mapping.run_recorded(subset,routing["cache"])
        for condition in list(common.CONDITIONS)+["gold_class_upper_bound"]:
            admission.require({r["id"] for r in paired_traces if r["condition"]==condition}==paired,"paired harness denominator changed")
        with (out/(name+".traces.jsonl")).open("x") as stream:
            for trace in traces:stream.write(json.dumps(trace,allow_nan=False)+"\n")
        all_episodes+=len(traces)
        reports[name]={"all_available_condition_episodes":summary,"common_metadata_subset":paired_summary,
                       "metadata_paired_unique_EEG_records":len(paired),"subject_intervals_full":frozen.subject_intervals(traces),
                       "subject_intervals_common_subset":frozen.subject_intervals(paired_traces),"cached_state_transition_wall_seconds":elapsed,
                       "cached_state_transition_mean_ms_per_episode":elapsed*1000/len(traces),"new_LLM_calls_during_replay":0,
                       "source_checkpoint_sha256":er["checkpoint_sha256"]}
    admission.verify_scope(args.seal,args.expected_seal_sha256,args.phase2_seal,args.phase2_code)
    for name,folder in evaluations.items():check_report_identity(name,folder,admission.PHASE2_SEAL,args.expected_seal_sha256,seal["prepared_manifest_sha256"])
    report={"schema":"xavier-phase3-five-model-same-synthetic-environment-v1","status":"completed","execution_seal_sha256":args.expected_seal_sha256,
            "phase2_parent_execution_seal_sha256":admission.PHASE2_SEAL,"analysis_plan_sha256":seal["analysis_plan_sha256"],
            "router_cache_sha256":admission.sha(args.route_cache),"source_inputs":source_inputs,"models":reports,
            "actual_LLM_routing_inference_calls_referenced":16,"new_LLM_calls_during_CPU_replay":0,"old_model_EEG_forward_calls":0,
            "cache_reused_for_all_models_and_recorded_episodes":True,"previous_phase2_harness_report_unchanged":True,
            "total_synthetic_episodes":all_episodes,"uncached_routing_wall_latency_ms_p50":routing["uncached_wall_latency_ms_p50"],
            "uncached_routing_wall_latency_ms_p95":routing["uncached_wall_latency_ms_p95"],"elapsed_CPU_scoring_seconds":time.monotonic()-started,
            "real_OS_actions":0,"additional_training_steps":0,"exploratory_test_reuse":True,"freeform_intention_or_real_world_agent_goal_validated":False,
            "scope":"Identical recorded predictions drive four independently counterbalanced synthetic in-memory mappings per EEG record through one fixed16-prompt routing cache. This separately named five-model replay preserves all original source, prediction and report bytes. The four mappings are not independent EEG samples."}
    admission.save(out/"harness_report.json",report)
    admission.save(out/"completion.json",{"status":"completed","report_sha256":admission.sha(out/"harness_report.json"),"models":list(MODELS),"new_LLM_calls":0,"real_OS_actions":0})
    return {"status":"completed","report_sha256":admission.sha(out/"harness_report.json"),"total_synthetic_episodes":all_episodes}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("prepared","seal","expected-seal-sha256","phase2-seal","phase2-code","route-cache","output"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--evaluation",action="append",required=True,help="Explicit model name=immutable evaluation folder, all five names")
    args=parser.parse_args();pairs=[]
    for value in args.evaluation:
        admission.require("=" in value,"evaluation must be name=path");pairs.append(value.split("=",1))
    args.evaluation=pairs;print(json.dumps(score(args)))

if __name__=="__main__":main()
