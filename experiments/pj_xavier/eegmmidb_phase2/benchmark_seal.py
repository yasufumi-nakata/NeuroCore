"""Seal data, code, fixed analysis plan and fresh budget before any model fit."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import benchmark_common as common

DATA_FILES={'manifest.jsonl':'prepared_manifest_sha256','artifacts.json':'artifacts_sha256',
    'control_artifacts.json':'control_artifacts_sha256','paired_control_subset.jsonl':'paired_control_subset_sha256',
    'subject_split.json':'subject_split_sha256','metadata_protocol.json':'metadata_protocol_sha256',
    'preprocessing_protocol.json':'preprocessing_protocol_sha256','idle_manifest.jsonl':'idle_manifest_sha256',
    'idle_artifacts.json':'idle_artifacts_sha256','idle_metadata_protocol.json':'idle_metadata_protocol_sha256',
    'corpus_summary.json':'corpus_summary_sha256','completion.json':'preparation_completion_sha256',
    'final_QC_decision.json':'final_QC_decision_sha256','preprocessing_tests_v2.json':'preprocessing_tests_sha256'}


def seal(prepared,budget_path,source_dir,output):
    root=Path(prepared);code=Path(source_dir)
    budget=common.read_json(budget_path)
    common.require(budget['absolute_deadline_epoch']>time.time(),'new evaluation/training budget already expired')
    common.require(0<budget['gpu_memory_budget_gib']<=21 and budget['colab_access']=='workstation_CLI_only','resource or access agreement changed')
    value={'schema':'xavier-eegmmidb-phase2-execution-seal-v1','created_epoch':time.time(),
           'analysis_plan_sha256':common.sha256(code/'protocol.benchmark.v1.json'),
           'actual_budget_receipt_sha256':common.sha256(budget_path),'budget':budget,
           'source_sha256':{p.name:common.sha256(p) for p in sorted(code.iterdir()) if p.is_file() and p.suffix=='.py'},
           **{key:common.sha256(root/name) for name,key in DATA_FILES.items()}}
    summary=common.read_json(root/'corpus_summary.json');completion=common.read_json(root/'completion.json')
    common.require(summary['status']=='completed' and completion['status']=='completed','preparation incomplete')
    common.require(summary.get('model_forward_calls')==0 and summary.get('test_performance_used_for_QC_or_protocol') is False,'preparation involved model/test-result selection')
    common.require(completion['manifest_sha256']==value['prepared_manifest_sha256'] and completion['corpus_summary_sha256']==value['corpus_summary_sha256'],'preparation receipt differs')
    rows,_,audit=common.verify_prepared(root,value)
    from benchmark_idle import verify_idle
    idle=verify_idle(root,value,rows)
    paired=common.read_rows(root/'paired_control_subset.jsonl')
    expected={r['id'] for r in rows if r['time_shift_control']['status']=='available'}
    common.require(len(paired)==len(expected) and {r['id'] for r in paired}==expected,'paired metadata cohort differs')
    value['data_audit']=audit;value['idle_diagnostic_rows']=len(idle)
    value['test_predictions_at_seal']=0;value['prior_models_or_cohorts_modified']=False
    common.save_json(output,value)
    return {'execution_seal_sha256':common.sha256(output),'main_records':len(rows),'idle_test_records':len(idle),'source_files':len(value['source_sha256'])}


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['prepared','budget','source-dir','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();print(json.dumps(seal(a.prepared,a.budget,a.source_dir,a.output)))
