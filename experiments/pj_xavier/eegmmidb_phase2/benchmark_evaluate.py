"""One frozen checkpoint, full test plus metadata-fixed paired controls, once."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
from threadpoolctl import threadpool_limits

import benchmark_common as common
import benchmark_audit as audit
import benchmark_idle as idle
import harness_evaluate as harness
import harness_mapping


def load_module(path):
    path=Path(path).resolve()
    import sys
    if str(path.parent) not in sys.path:sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('phase2_model_plugin_'+common.sha256(path)[:12],path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def prediction_row(source,output,condition,checkpoint,perturbation):
    labels=common.TASKS[source['task_family']]
    p=common.checked_probabilities(output['probabilities'],labels)
    expected=labels[int(p.argmax())]
    common.require(output['predicted_label']==expected,'predicted class differs from probabilities')
    common.require(output.get('task_family',source['task_family'])==source['task_family'],'prediction task changed')
    return {'id':source['id'],'dataset':source['dataset'],'subject':source['subject'],'session':source['session'],
            'recording_id':source['recording_id'],'task_family':source['task_family'],'true_label':source['label'],
            'predicted_label':expected,'probabilities':{k:float(v) for k,v in zip(labels,p)},'condition':condition,
            'checkpoint_sha256':checkpoint,'latency_ms':output.get('latency_ms'),'latency_scope':output.get('latency_scope'),
            'perturbation':perturbation}


def resource_evidence(device,budget):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    value={'cpu_peak_rss_bytes':int(rss if platform.system()=='Darwin' else rss*1024),'device':device,
           'budget_gib':budget,'scope':'Inference process only; latency does not include causal stream preparation or harness'}
    if device.startswith('cuda'):
        import torch
        value.update(cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                     hardware=torch.cuda.get_device_name())
        value['status']='passed' if value['cuda_peak_reserved_bytes'] <= budget*1024**3 else 'failed_budget_exceeded'
    elif device=='mps':
        import torch
        value.update(mps_sample_allocated_bytes=int(torch.mps.current_allocated_memory()),mps_sample_driver_bytes=int(torch.mps.driver_allocated_memory()),
                     status='measured_inference_true_GPU_peak_unavailable')
    else:value['status']='measured_CPU_inference'
    return value


def evaluate(prepared,seal_path,expected_seal,module_path,run_dir,checkpoint_sha256,output,device='cpu',batch_size=16,memory_budget=21,max_seconds=900,collect_routing=False):
    started=time.monotonic(); prepared=Path(prepared); run_dir=Path(run_dir); output=Path(output)
    common.require(common.sha256(seal_path)==expected_seal,'evaluation execution seal changed')
    seal=common.read_json(seal_path)
    plan_path=Path(__file__).with_name('protocol.benchmark.v1.json')
    common.require(common.sha256(plan_path)==seal['analysis_plan_sha256'],'analysis plan changed')
    source_files=[Path(__file__),Path(common.__file__),Path(module_path),Path(audit.__file__),Path(idle.__file__),Path(harness.__file__),Path(harness_mapping.__file__)]
    for path in source_files:
        common.require(common.sha256(path)==seal['source_sha256'][path.name],'source changed: '+path.name)
    common.require(0<memory_budget<=seal['budget']['gpu_memory_budget_gib'] and batch_size>0,'runtime resource request exceeds sealed budget')
    deadline=min(started+max_seconds,started+seal['budget']['absolute_deadline_epoch']-time.time())
    common.require(time.monotonic()<deadline,'new phase budget expired')
    rows,artifacts,data_audit=common.verify_prepared(prepared,seal)
    idle_rows=idle.verify_idle(prepared,seal,rows)
    require_membership(run_dir,rows,seal)
    test=sorted((r for r in rows if r['split']=='test'),key=lambda r:r['id'])
    common.require(test,'empty test cohort')
    subset_path=prepared/'paired_control_subset.jsonl'
    common.require(common.sha256(subset_path)==seal['paired_control_subset_sha256'],'paired subset changed')
    subset=common.read_rows(subset_path); all_subset_ids={r['id'] for r in subset}
    expected_all={r['id'] for r in rows if r.get('time_shift_control',{}).get('status')=='available'}
    common.require(all_subset_ids==expected_all and len(all_subset_ids)==len(subset),'paired cohort does not match available metadata controls')
    paired_ids=all_subset_ids & {r['id'] for r in test}
    shuffle,shuffle_blocked=common.shuffle_mapping(test,common.read_json(plan_path)['seed'])
    common.require(paired_ids <= set(shuffle),'paired subset contains unavailable derangement; explicit new subset required')
    common.require(not output.exists(),'preserve previous evaluation output')
    marker=run_dir/('phase2_test_consumed_'+seal['prepared_manifest_sha256']+'.json')
    common.require(not marker.exists(),'this checkpoint/run already consumed the frozen test cohort')
    output.mkdir(parents=True);common.save_json(output/'execution_seal.json',seal)
    plugin=load_module(module_path)
    runtime=plugin.load_runtime(run_dir,device,memory_budget_gib=memory_budget,batch_size=batch_size)
    actual=getattr(runtime,'checkpoint_sha256',None)
    if actual is None and (run_dir/'best.pt').exists():actual=common.sha256(run_dir/'best.pt')
    common.require(actual==checkpoint_sha256,'loaded checkpoint differs from sealed selection')
    common.save_json(marker,{'checkpoint_sha256':checkpoint_sha256,'manifest_sha256':seal['prepared_manifest_sha256'],
        'execution_seal_sha256':expected_seal,'created_epoch':time.time(),'meaning':'Test evaluation begun once, not a claim of success'})
    store=common.EEGStore(prepared); predictions=[];blocked={};condition_reports={}
    path=output/'predictions.jsonl'
    with path.open('x') as stream,threadpool_limits(limits=2):
        for condition in common.CONDITIONS:
            available=[]; unavailable=[]
            for r in test:
                reason=None
                if condition=='eeg_shuffled' and r['id'] not in shuffle:reason='singleton_recording_no_derangement'
                if condition=='eeg_time_shifted' and r.get('time_shift_control',{}).get('status')!='available':reason=r.get('time_shift_control',{}).get('reason','raw_shift_unavailable')
                if reason:unavailable.append({'id':r['id'],'reason':reason})
                else:available.append(r)
            blocked[condition]=unavailable
            for begin in range(0,len(available),batch_size):
                common.require(time.monotonic()<deadline,'evaluation time budget expired; never silently complete a partial test')
                batch=available[begin:begin+batch_size]; eeg=[]; evidence=[]
                for r in batch:
                    if condition=='context_only':x=np.zeros((64,600),np.float32);proof={'method':'EEG_removed'}
                    elif condition=='eeg_shuffled':
                        other=shuffle[r['id']];x=store.get(other);proof={'method':'within_recording_derangement','eeg_source_id':other['id'],'original_eeg_id':r['id'],'fixed_points':0}
                    elif condition=='eeg_time_shifted':x=store.get(r,True);proof=dict(r['time_shift_control'],method='actual_causal_stream_minus10seconds')
                    else:x=store.get(r);proof={'method':'original_EEG'}
                    eeg.append(x);evidence.append(proof)
                outputs=runtime.predict_batch(np.stack(eeg),[r['task_family'] for r in batch],condition)
                common.require(len(outputs)==len(batch),'model output count mismatch')
                for r,value,proof in zip(batch,outputs,evidence,strict=True):
                    record=prediction_row(r,value,condition,checkpoint_sha256,proof);predictions.append(record)
                    stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                stream.flush()
            condition_reports[condition]={'status':'measured' if available else 'blocked','n':len(available),'blocked_n':len(unavailable),
                'blocked_reasons':dict(Counter(r['reason'] for r in unavailable))}
    common.save_json(output/'blocked_records.json',blocked)
    prediction_audit=audit.audit_predictions(rows,predictions,checkpoint_sha256)
    common.save_json(output/'prediction_identity_audit.json',prediction_audit)
    idle_report=idle.evaluate_idle(runtime,prepared,seal,rows,output,batch_size,checkpoint_sha256,deadline,idle_rows)
    common.save_json(output/'idle_report.json',idle_report)
    routing_evidence=None
    if collect_routing:
        routing_evidence=harness.collect_routing(runtime,output/'router_cache.json',checkpoint_sha256,seal,deadline)
    if (run_dir/'best.pt').exists():common.require(common.sha256(run_dir/'best.pt')==checkpoint_sha256,'checkpoint mutated during evaluation')
    for path0 in source_files:common.require(common.sha256(path0)==seal['source_sha256'][path0.name],'implementation changed during evaluation')
    common.save_json(output/'inference_completion.json',{'status':'completed_inference','checkpoint_sha256':checkpoint_sha256,'predictions_sha256':common.sha256(path),
        'idle_predictions_sha256':common.sha256(output/'idle_predictions.jsonl'),'routing_calls':16 if collect_routing else 0,
        'created_epoch':time.time(),'meaning':'Inference data complete; statistical scoring may still be pending'})
    plan=common.read_json(plan_path)
    all_scores=common.score_rows(predictions,plan['metrics']['acceptance_threshold'])
    paired=[r for r in predictions if r['id'] in paired_ids]
    for condition in common.CONDITIONS:common.require({r['id'] for r in paired if r['condition']==condition}==paired_ids,'paired comparison denominator differs')
    report={'schema':'xavier-phase2-five-condition-evaluation-v1','status':'completed','checkpoint_sha256':checkpoint_sha256,
        'execution_seal_sha256':expected_seal,'manifest_sha256':seal['prepared_manifest_sha256'],'predictions_sha256':common.sha256(path),
        'conditions':condition_reports,'full_test_records':len(test),'full_test_metrics':all_scores,
        'metadata_paired_subset_records':len(paired_ids),'paired_id_hash':common.object_hash(sorted(paired_ids)),
        'paired_subset_metrics':common.score_rows(paired,plan['metrics']['acceptance_threshold']),
        'full_test_subject_uncertainty':common.per_condition_subject_intervals(predictions,plan['metrics']['bootstrap_repetitions'],plan['seed']),
        'paired_subject_uncertainty':common.participant_bootstrap(predictions,plan['metrics']['bootstrap_repetitions'],plan['seed'],paired_ids),
        'resources':resource_evidence(device,memory_budget),'elapsed_seconds':time.monotonic()-started,'data_audit':data_audit,
        'prediction_identity_audit':prediction_audit,'idle_diagnostic':idle_report,
        'routing_inference_calls':16 if collect_routing else 0,'routing_cache_sha256':common.sha256(output/'router_cache.json') if collect_routing else None,
        'condition_interpretations':plan['condition_interpretations'],
        'goal_validated':False,'scope':'Public instructed MI decoding only; actual synthetic routing evaluation is separate. No natural idle/task completion claim.'}
    common.save_json(output/'evaluation_report.json',report)
    common.save_json(output/'completion.json',{'status':'completed','exit_code':0,'checkpoint_sha256':checkpoint_sha256,
        'variant_names':list(common.CONDITIONS),'report_path':str(output/'evaluation_report.json'),'report_sha256':common.sha256(output/'evaluation_report.json'),
        'test_records':len(test),'paired_records':len(paired_ids),'goal_validated':False})
    print(json.dumps({'status':'completed','report_sha256':common.sha256(output/'evaluation_report.json'),'test_records':len(test),'paired_records':len(paired_ids)}))
    return report


def require_membership(run_dir,rows,seal):
    membership=common.read_json(Path(run_dir)/'training_membership.json')
    common.require(membership.get('manifest_sha256')==seal['prepared_manifest_sha256'],'training cohort differs')
    common.require(membership.get('test_used_for_tuning') is False,'test-based tuning not allowed')
    for split,key in [('train','training_ids'),('val','validation_ids'),('test','test_ids')]:
        expected=sorted(r['id'] for r in rows if r['split']==split)
        common.require(expected==sorted(membership[key]) and len(membership[key])==len(set(membership[key])),'model membership differs from frozen '+split)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','seal','expected-seal-sha256','model-module','run-dir','checkpoint-sha256','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--memory-budget-gib',type=float,default=21);parser.add_argument('--max-seconds',type=int,default=900)
    parser.add_argument('--collect-routing',action='store_true',help='Fused run only:16greedy calls on the same frozen base, then exact-prompt cache')
    a=parser.parse_args();evaluate(a.prepared,a.seal,a.expected_seal_sha256,a.model_module,a.run_dir,a.checkpoint_sha256,a.output,a.device,a.batch_size,a.memory_budget_gib,a.max_seconds,a.collect_routing)
