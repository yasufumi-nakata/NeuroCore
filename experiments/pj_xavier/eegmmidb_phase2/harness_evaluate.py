"""16 actual frozen-LLM calls; all recorded trials use an explicit prompt cache."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import benchmark_common as common
import benchmark_audit as audit
import harness_mapping as mapping


def collect_routing(runtime,output,checkpoint_sha,seal,deadline):
    """Run the16 distinct greedy routing texts once; this is not EEG inference."""
    common.require(getattr(runtime,'checkpoint_sha256',None)==checkpoint_sha,'routing runtime checkpoint changed')
    token=runtime.tokenizer
    candidates={s:token.encode(s,add_special_tokens=False) for s in ('X','Y',' X',' Y')}
    common.require(all(len(x)==1 for x in candidates.values()) and candidates['X']!=candidates['Y'] and candidates[' X']!=candidates[' Y'],'target candidates must have equal one-token lengths and distinct identities')
    templates=mapping.required_prompts();cache={};before=len(runtime.route_records)
    for digest,value in sorted(templates.items()):
        common.require(time.monotonic()<deadline,'routing time budget expired')
        started=time.perf_counter();text=runtime.route_text(value['prompt'],max_new_tokens=8);elapsed=(time.perf_counter()-started)*1000
        common.require(isinstance(text,str),'router returned nontext')
        record=runtime.route_records[-1]
        common.require(record['EEG_prefix_used'] is False and record['new_parameter_updates']==0 and record['do_sample'] is False,'routing must use the same frozen base without EEG prefix')
        cache[digest]={'prompt':value['prompt'],'text':text,'parsed_target':mapping.parse_action(text),'wall_latency_ms':elapsed,
                       'model_call_evidence':record}
    common.require(len(runtime.route_records)-before==16 and len(cache)==16,'actual routing call count must be16')
    result={'schema':'xavier-phase2-exact-prompt-routing-cache-v1','status':'completed','checkpoint_sha256':checkpoint_sha,
            'analysis_plan_sha256':seal['analysis_plan_sha256'],'source_sha256':seal['source_sha256'],
            'actual_LLM_greedy_inference_calls':16,'token_candidates':candidates,'cache':cache,
            'uncached_wall_latency_ms_p50':float(np.percentile([v['wall_latency_ms'] for v in cache.values()],50)),
            'uncached_wall_latency_ms_p95':float(np.percentile([v['wall_latency_ms'] for v in cache.values()],95)),
            'max_new_tokens':8,'do_sample':False,'scope':'16 distinct template prompts. Later EEG episodes reuse identical cached texts; they are not fresh LLM invocations.'}
    common.save_json(output,result);return result


def verify_cache(path,seal):
    record=common.read_json(path);templates=mapping.required_prompts()
    common.require(record['status']=='completed' and record['actual_LLM_greedy_inference_calls']==16,'routing generation incomplete')
    common.require(record['analysis_plan_sha256']==seal['analysis_plan_sha256'],'routing analysis plan changed')
    common.require(record['source_sha256']==seal['source_sha256'],'routing code differs from sealed source')
    common.require(set(record['cache'])==set(templates),'unexpected/missing routing template')
    for digest,entry in record['cache'].items():
        common.require(entry['prompt']==templates[digest]['prompt'] and hashlib.sha256(entry['prompt'].encode()).hexdigest()==digest,'routing prompt changed')
        common.require(entry['parsed_target']==mapping.parse_action(entry['text']),'cached parser differs')
    return record


def subject_intervals(traces,seed=20261122):
    grouped=defaultdict(lambda:defaultdict(list))
    for r in traces:grouped[(r['condition'],r['task_family'])][(r['dataset'],r['subject'])].append(r)
    result={}
    for key,groups in grouped.items():
        values=np.array([np.mean([r['task_success'] for r in v]) for v in groups.values()])
        if len(values)<2:result['::'.join(key)]={'status':'blocked_insufficient_subject_groups'};continue
        rng=np.random.default_rng(seed);draw=np.mean(values[rng.integers(0,len(values),(1000,len(values)))],axis=1)
        result['::'.join(key)]={'status':'measured','subject_groups':len(values),'mean_subject_success':float(values.mean()),
                              'lower_95':float(np.percentile(draw,2.5)),'upper_95':float(np.percentile(draw,97.5)),'bootstrap_repetitions':1000}
    return result


def score(prepared,seal_path,expected_seal,cache_path,evaluations,output):
    common.require(common.sha256(seal_path)==expected_seal,'execution seal changed')
    seal=common.read_json(seal_path);root=Path(prepared);out=Path(output)
    for path in [Path(__file__),Path(mapping.__file__),Path(common.__file__),Path(audit.__file__)]:
        common.require(common.sha256(path)==seal['source_sha256'][path.name],'harness scientific source changed')
    common.require(common.sha256(root/'manifest.jsonl')==seal['prepared_manifest_sha256'],'frozen main manifest changed')
    rows=common.read_rows(root/'manifest.jsonl');all_ids={r['id'] for r in rows if r['split']=='test'}
    common.require(common.sha256(root/'paired_control_subset.jsonl')==seal['paired_control_subset_sha256'],'paired cohort changed')
    paired={r['id'] for r in common.read_rows(root/'paired_control_subset.jsonl')} & all_ids
    routing=verify_cache(cache_path,seal);cache=routing['cache']
    common.require(not out.exists(),'preserve earlier harness execution');out.mkdir(parents=True)
    reports={};source_inputs={};models=set();fullstarted=time.monotonic()
    for name,evaluation in evaluations:
        common.require(name not in models,'duplicate model label');models.add(name)
        eroot=Path(evaluation);er=common.read_json(eroot/'evaluation_report.json')
        common.require(er['manifest_sha256']==seal['prepared_manifest_sha256'] and er['execution_seal_sha256']==expected_seal,'model evaluation cohort or seal differs')
        common.require(common.sha256(eroot/'predictions.jsonl')==er['predictions_sha256'],'model predictions changed')
        prediction=common.read_rows(eroot/'predictions.jsonl');proof=audit.audit_predictions(rows,prediction,er['checkpoint_sha256'])
        source_inputs[name]={'evaluation_report_sha256':common.sha256(eroot/'evaluation_report.json'),'predictions_sha256':er['predictions_sha256'],
                            'checkpoint_sha256':er['checkpoint_sha256'],'identity_audit':proof}
        all_prediction=prediction+mapping.gold_rows(prediction)
        started=time.perf_counter();traces,summary=mapping.run_recorded(all_prediction,cache);elapsed=time.perf_counter()-started
        selected=[r for r in all_prediction if r['id'] in paired]
        paired_traces,paired_summary=mapping.run_recorded(selected,cache)
        for condition in list(common.CONDITIONS)+['gold_class_upper_bound']:
            common.require({r['id'] for r in paired_traces if r['condition']==condition}==paired,'harness paired denominator differs')
        with (out/(name+'.traces.jsonl')).open('x') as stream:
            for trace in traces:stream.write(json.dumps(trace,allow_nan=False)+'\n')
        reports[name]={'all_available_condition_episodes':summary,'common_metadata_subset':paired_summary,
                       'metadata_paired_unique_EEG_records':len(paired),'subject_intervals_full':subject_intervals(traces),
                       'subject_intervals_common_subset':subject_intervals(paired_traces),'cached_state_transition_wall_seconds':elapsed,
                       'cached_state_transition_mean_ms_per_episode':elapsed*1000/len(traces) if traces else None,
                       'new_LLM_calls_during_replay':0,'source_checkpoint_sha256':er['checkpoint_sha256']}
    common.require(models=={'csp_lda','scratch_eeg','fused_smollm2'},'complete harness requires the three predeclared model variants')
    common.require(source_inputs['fused_smollm2']['checkpoint_sha256']==routing['checkpoint_sha256'],'routing must use same frozen backbone/checkpoint evidence as fused evaluation')
    report={'schema':'xavier-phase2-synthetic-environment-evaluation-v1','status':'completed','analysis_plan_sha256':seal['analysis_plan_sha256'],
            'execution_seal_sha256':expected_seal,'router_cache_sha256':common.sha256(cache_path),'source_inputs':source_inputs,'models':reports,
            'actual_LLM_routing_inference_calls':16,'cache_reused_for_all_recorded_episodes':True,'uncached_routing_wall_latency_ms_p50':routing['uncached_wall_latency_ms_p50'],
            'uncached_routing_wall_latency_ms_p95':routing['uncached_wall_latency_ms_p95'],'elapsed_CPU_scoring_seconds':time.monotonic()-fullstarted,
            'real_OS_actions':0,'additional_training_steps':0,'freeform_intention_or_real_world_agent_goal_validated':False,
            'scope':'Public cued binary MI decoder drives an independently counterbalanced synthetic in-memory object selection. Gold labels are oracle-only; routing uses decoded class. Four mappings per EEG record do not create four independent EEG samples.'}
    common.save_json(out/'harness_report.json',report)
    common.save_json(out/'completion.json',{'status':'completed','report_sha256':common.sha256(out/'harness_report.json'),
        'report_path':str(out/'harness_report.json'),'models':sorted(models),'actual_LLM_routing_inference_calls':16,'real_OS_actions':0})
    return {'status':'completed','report_sha256':common.sha256(out/'harness_report.json')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','seal','expected-seal-sha256','route-cache','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--evaluation',action='append',required=True,help='Predeclared model name=private evaluation folder; repeat for3models')
    a=parser.parse_args();values=[]
    for text in a.evaluation:
        common.require('=' in text,'evaluation needs name=folder');values.append(text.split('=',1))
    print(json.dumps(score(a.prepared,a.seal,a.expected_seal_sha256,a.route_cache,values,a.output)))
