#!/usr/bin/env python3
"""Frozen EEGMMIDB train-three M1 trial;12 fresh routes, no held-out inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PHASE='private/xavier/eegmmidb_phase2_v1'
CODE='experiments/pj_xavier/eegmmidb_phase2'
RUN='runs/eegnet_smollm_fused_v1'
EXPECTED_MANIFEST='d673ffa0bc54506fc5866a7092fa5ff81deffdcfdf0754b10902cb6fda8657a9'
EXPECTED_SEAL='e1847ae39ef284a49cdc34dbe1695593b6d9a3c46b8d4db046842c403b26eb7f'
CONDITIONS={'fused','context_only','eeg_only','eeg_shuffled','eeg_time_shifted'}
SCHEMA='xavier-phase2-train-three-M1-contract-v1'


def require(value,message):
    if not value:raise ValueError(message)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def relative_file(repo,path):
    repo=Path(repo).resolve();path=Path(path).resolve()
    require(path.is_relative_to(repo) and path.is_file(),'trial source must be an existing file inside the repository')
    return str(path.relative_to(repo))


def contained(repo,relative):
    name=Path(relative);require(not name.is_absolute() and '..' not in name.parts and '\\' not in str(relative),'source path must remain relative')
    path=Path(repo)/name
    require(path.resolve().is_relative_to(Path(repo).resolve()) and not path.is_symlink(),'source path escaped repository')
    return path


def file_record(repo,path):
    relative=relative_file(repo,path);p=contained(repo,relative)
    return {'path':relative,'sha256':sha256(p),'bytes':p.stat().st_size}


def selected_train_rows(rows):
    ids=[r['id'] for r in rows];require(len(ids)==len(set(ids)),'repeated source IDs')
    values=[r for r in rows if r['split']=='train']
    require(len(values)>=3,'at least3training records required')
    return sorted(values,key=lambda r:(hashlib.sha256(('20261122|'+r['id']).encode()).hexdigest(),r['id']))[:3]


def verify_member_ledger(repo,recovery_path,required_paths):
    repo=Path(repo).resolve();phase=repo/PHASE;r=read_json(recovery_path)
    require(r.get('status')=='completed' and r.get('all_member_hashes_verified') is True,'verified recovery is required')
    files=r.get('files');require(isinstance(files,dict) and files and r.get('member_count')==len(files),'recovery member count differs')
    mapped={}
    for key,item in files.items():
        p=contained(phase,key)
        require(Path(item['local_path']).resolve()==p.resolve(),'recovery local path/member differs')
        require(p.is_file() and p.stat().st_size==item['bytes'] and sha256(p)==item['sha256'],'recovered member bytes changed')
        mapped[p.resolve()]=item
    require(all(Path(p).resolve() in mapped for p in required_paths),'required selected run/evaluation absent from recovery ledger')
    return r


def verify_sources(repo,contract):
    require(contract.get('schema')==SCHEMA and contract.get('partition')=='train' and contract.get('test_forward_calls')==0,'trial scope changed')
    require(contract.get('selection_rule')=='first3_SHA256_seed20261122_ID_order_no_outcome_use' and len(contract.get('selected_ids',[]))==3,'selection contract changed')
    require(contract.get('memory_budget_gib')==8 and contract.get('acceptance_threshold')==.8 and contract.get('route_calls')==12
            and contract.get('max_new_tokens')==8 and contract.get('device')=='mps' and contract.get('cloud_calls_allowed') is False
            and contract.get('OS_actions_allowed') is False and contract.get('routing_cache_reuse') is False,'trial resources or execution scope changed')
    require(contract.get('execution_seal_sha256')==EXPECTED_SEAL and contract.get('manifest_sha256')==EXPECTED_MANIFEST,'trial belongs to another experiment')
    required={'manifest','artifacts','benchmark_seal','recovery','wrapper','run:best.pt','run:run_summary.json','run:run_config.json','run:training_membership.json',
              'evaluation:completion.json','evaluation:evaluation_report.json','evaluation:predictions.jsonl'}
    proof=contract.get('training_terminal_proof',{})
    require(proof.get('status')=='verified' and proof.get('selected_validation_records',0)>0 and proof.get('test_used_for_selection') is False,'trial lacks verified complete validation selection')
    required.update({'selection:history','selection:selected_epoch_checkpoint','selection:validation_predictions'})
    if proof.get('stop_reason') in ('stage_time_budget','absolute_deadline'):
        required.update({'selection:stop_receipt','selection:stopped_checkpoint','selection:budget','selection:attempt','selection:workflow_launch','selection:workflow_exit'})
    required.update('source:'+name for name in ('__init__.py','data_contract.py','model.py','runtime.py','train_models.py','benchmark_common.py','harness_mapping.py'))
    require(required <= set(contract.get('files',{})),'trial contract omitted required identity evidence')
    for record in contract['files'].values():
        p=contained(repo,record['path']);require(p.is_file() and p.stat().st_size==record['bytes'] and sha256(p)==record['sha256'],'frozen trial source changed: '+record['path'])
    require(contract['files']['manifest']['sha256']==EXPECTED_MANIFEST and contract['files']['benchmark_seal']['sha256']==EXPECTED_SEAL,'frozen original cohort/seal identity omitted')
    rows=[json.loads(line) for line in contained(repo,contract['files']['manifest']['path']).read_text().splitlines() if line.strip()]
    selected=selected_train_rows(rows)
    require([r['id'] for r in selected]==contract['selected_ids'],'preselected train3 IDs changed')
    for r in selected:
        item=contract['files'].get('array:'+r['eeg_path'])
        require(item and item['path']==str(Path(contract['prepared'])/r['eeg_path']) and item['sha256']==r['array_sha256'],'selected actual EEG array identity absent')
    return selected


def verify_training_terminal(phase,run,summary,config,checkpoint,rows):
    """Accept a planned time cutoff only with a complete val-selected best."""
    phase,run=Path(phase),Path(run)
    history_path=run/'history.jsonl';history=[json.loads(x) for x in history_path.read_text().splitlines() if x.strip()]
    require(sha256(history_path)==summary['history_sha256'] and history,'complete validation history missing or changed')
    require(all(math.isfinite(float(r['validation']['selection_score'])) and r['validation']['selection_partition']=='val' for r in history),'selection history uses incomplete or nonvalidation scores')
    winner=max(history,key=lambda r:r['validation']['selection_score'])
    require(winner['checkpoint_sha256']==checkpoint and winner['validation']['selection_score']==summary['best_primary_validation_subject_BA'],'selected best is not earliest maximum of recorded validation history')
    selected_checkpoint=contained(run,winner['checkpoint'])
    require(sha256(selected_checkpoint)==checkpoint,'selected validation checkpoint file differs from best')
    val_path=run/f"validation_epoch_{winner['epoch']:03d}.jsonl"
    require(sha256(val_path)==winner['validation_predictions_sha256'],'selected validation predictions changed')
    val=[json.loads(x) for x in val_path.read_text().splitlines() if x.strip()]
    originals={r['id']:r for r in rows if r['split']=='val'};seen=set();group_counts={}
    labels={'left_right':('left_fist','right_fist'),'fists_feet':('both_fists','both_feet')}
    for p in val:
        require(p['id'] in originals and p['id'] not in seen,'selected validation prediction IDs differ');seen.add(p['id']);r=originals[p['id']]
        require(p['split']=='val' and p['subject_group']==r['subject_group'] and p['task_family']==r['task_family'] and p['true_label']==r['label'] and p['checkpoint_sha256']==checkpoint,'selected validation label/source differs')
        names=labels[r['task_family']];require(set(p['probabilities'])==set(names),'validation class keys changed')
        values=[p['probabilities'][k] for k in names]
        require(all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1 for x in values) and abs(sum(values)-1)<=1e-5,'validation probabilities invalid')
        predicted=names[0] if values[0]>=values[1] else names[1];require(p['predicted_label']==predicted,'validation argmax differs')
        if r['task_family']=='left_right':
            counts=group_counts.setdefault(r['subject_group'],[[0,0],[0,0]]);target=names.index(r['label'])
            counts[0][target]+=int(predicted==r['label']);counts[1][target]+=1
    require(seen==set(originals),'incomplete selected validation partition')
    scores=[sum(a/b for a,b in zip(correct,support))/2 for correct,support in group_counts.values() if all(support)]
    require(scores and abs(sum(scores)/len(scores)-winner['validation']['selection_score'])<1e-12,'recorded selected metric differs from actual complete validation predictions')
    paths={'selection:history':history_path,'selection:selected_epoch_checkpoint':selected_checkpoint,'selection:validation_predictions':val_path}
    reason=summary.get('stop_reason')
    if summary['status']=='completed':
        require(reason is None and summary['completed_epochs']==config['max_epochs'],'completed training contradicts fixed epoch budget')
    else:
        require(summary['status']=='partial' and reason in ('stage_time_budget','absolute_deadline') and summary['completed_epochs']>=1,'unexpected training failure cannot enter trial')
        stops=sorted(run.glob('stop_attempt_*.json'));require(stops,'planned stop receipt missing')
        stop_path=stops[-1];stop=read_json(stop_path)
        require(stop['reason']==reason and stop['checkpoint']==summary['last_checkpoint'] and stop['resumable'] is True and stop['test_forward_calls']==0 and stop['validation_skipped_after_stop'] is True,'stop state is not a bounded consistent stop')
        stopped=contained(run,stop['checkpoint']);require(sha256(stopped)==stop['checkpoint_sha256'],'stopped checkpoint differs')
        paths.update({'selection:stop_receipt':stop_path,'selection:stopped_checkpoint':stopped})
        budget_path=phase/'gpu_budget_v1.json';budget=read_json(budget_path);expected=budget['absolute_deadline_epoch']-2100
        require(budget['absolute_deadline_epoch']==1795325994.596278,'original phase2 budget changed')
        attempts=sorted(run.glob('attempt_*.json'));require(attempts,'attempt admission absent');attempt_path=attempts[-1];attempt=read_json(attempt_path)
        require(abs(attempt['arguments']['absolute_deadline_epoch']-expected)<1e-6,'run used an unplanned absolute cutoff')
        launch_path=phase/'workflow/train_fused.launch.json';exit_path=phase/'workflow/train_fused.exit.json'
        launch,terminal=read_json(launch_path),read_json(exit_path);command=launch['command']
        require(command.count('--absolute-deadline-epoch')==1 and abs(float(command[command.index('--absolute-deadline-epoch')+1])-expected)<1e-6,'workflow deadline disagrees with fixed allocation')
        require(launch['pid']==terminal['pid'] and terminal['command']==command and terminal['exit_code']==2 and terminal['timed_out'] is False,'planned cutoff did not follow the fixed trainer partial-exit-2 contract')
        require(abs(launch['global_deadline_epoch']-(expected+10))<1e-6,'supervisor deadline changed')
        if reason=='stage_time_budget':require(summary['elapsed_stage_seconds']>=config['max_seconds']-.5,'stage stopped before its planned budget')
        else:require(terminal['finished_epoch']>=expected-.5,'absolute cutoff stopped before its planned deadline')
        paths.update({'selection:budget':budget_path,'selection:attempt':attempt_path,'selection:workflow_launch':launch_path,'selection:workflow_exit':exit_path})
    return {'status':'verified','training_status':summary['status'],'stop_reason':reason,'selected_epoch':winner['epoch'],
            'selected_global_step':winner['global_step'],'selected_validation_subject_BA':winner['validation']['selection_score'],
            'selected_validation_records':len(val),'test_used_for_selection':False,'normal_planned_budget_stop':reason in ('stage_time_budget','absolute_deadline')},paths


def prepare(repo,evaluation,recovery,output):
    repo=Path(repo).resolve();phase=repo/PHASE;code=repo/CODE;run=phase/RUN;prepared=phase/'prepared';evaluation=Path(evaluation).resolve()
    require(evaluation.is_relative_to(phase/'evaluations'),'evaluation must be recovered under phase2 private evaluations')
    seal_path=phase/'benchmark_execution_seal_v2.json';require(sha256(seal_path)==EXPECTED_SEAL,'benchmark execution seal changed')
    seal=read_json(seal_path);require(sha256(prepared/'manifest.jsonl')==EXPECTED_MANIFEST==seal['prepared_manifest_sha256'],'frozen main manifest differs')
    summary=read_json(run/'run_summary.json');config=read_json(run/'run_config.json');membership=read_json(run/'training_membership.json')
    checkpoint=sha256(run/'best.pt')
    require(summary['best_checkpoint_sha256']==checkpoint and config['kind']=='fused' and config['frozen_backbone'] is True,'fused selected checkpoint required')
    require(summary['completed_epochs']>=1,'training has no complete validation epoch')
    require(summary['test_forward_calls']==0 and summary['test_used_for_tuning'] is False and summary['model_selection_partition']=='val','trainer used held-out data')
    require(config['manifest_sha256']==EXPECTED_MANIFEST and config['acceptance_threshold']==.8 and config['backbone']['frozen'] is True,'scientific run config changed')
    require(config['backbone']['id']=='HuggingFaceTB/SmolLM2-1.7B-Instruct' and config['backbone']['revision']=='31b70e2e869a7173562077fd711b654946d38674','unpinned frozen backbone')
    require(membership['manifest_sha256']==EXPECTED_MANIFEST and membership['test_used_for_tuning'] is False,'membership cohort differs')
    rows=[json.loads(x) for x in (prepared/'manifest.jsonl').read_text().splitlines() if x.strip()]
    for split,key in [('train','training_ids'),('val','validation_ids'),('test','test_ids')]:
        require(sorted(membership[key])==sorted(r['id'] for r in rows if r['split']==split),'membership IDs differ')
    selected=selected_train_rows(rows)
    terminal_proof,selection_paths=verify_training_terminal(phase,run,summary,config,checkpoint,rows)
    report=read_json(evaluation/'evaluation_report.json');completion=read_json(evaluation/'completion.json')
    require(completion['status']=='completed' and completion['exit_code']==0 and completion['checkpoint_sha256']==checkpoint
            and set(completion['variant_names'])==CONDITIONS and completion['report_sha256']==sha256(evaluation/'evaluation_report.json'),'five-condition evaluation completion absent or changed')
    require(report['status']=='completed' and report['checkpoint_sha256']==checkpoint and report['execution_seal_sha256']==EXPECTED_SEAL
            and report['manifest_sha256']==EXPECTED_MANIFEST and report['prediction_identity_audit']['status']=='passed','evaluation identity differs')
    require(report['resources']['status']=='passed' and report['resources']['device'].startswith('cuda'),'measured completed GPU inference budget evidence missing')
    require(report['predictions_sha256']==sha256(evaluation/'predictions.jsonl'),'recovered predictions changed')
    required=[run/name for name in ['best.pt','run_summary.json','run_config.json','training_membership.json']]+[evaluation/name for name in ['completion.json','evaluation_report.json','predictions.jsonl']]
    required.extend(path for path in selection_paths.values() if path!=phase/'gpu_budget_v1.json')
    recovered=verify_member_ledger(repo,recovery,required)
    require(recovered['checkpoint_sha256']['fused']==checkpoint,'recovery selected checkpoint differs')
    paths={'manifest':prepared/'manifest.jsonl','artifacts':prepared/'artifacts.json','benchmark_seal':seal_path,'recovery':Path(recovery),'wrapper':Path(__file__)}
    paths.update({'run:'+p.name:p for p in required[:4]});paths.update({'evaluation:'+p.name:p for p in required[4:7]});paths.update(selection_paths)
    for name,digest in seal['source_sha256'].items():
        path=code/name;require(sha256(path)==digest,'phase2 scientific source changed: '+name);paths['source:'+name]=path
    artifacts=read_json(prepared/'artifacts.json')
    for r in selected:
        item=artifacts[r['eeg_path']];p=contained(prepared,r['eeg_path'])
        require(r['split']=='train' and r['label_source']=='recorded_event' and r['array_sha256']==item['sha256'] and sha256(p)==item['sha256'],'preselected EEG array differs')
        paths['array:'+r['eeg_path']]=p
    contract={'schema':SCHEMA,'partition':'train','selected_ids':[r['id'] for r in selected],
        'selection_rule':'first3_SHA256_seed20261122_ID_order_no_outcome_use','manifest_sha256':EXPECTED_MANIFEST,'execution_seal_sha256':EXPECTED_SEAL,
        'checkpoint_sha256':checkpoint,'run':str(run.relative_to(repo)),'prepared':str(prepared.relative_to(repo)),
        'evaluation':str(evaluation.relative_to(repo)),'files':{key:file_record(repo,path) for key,path in paths.items()},
        'training_terminal_proof':terminal_proof,
        'wrapper_sha256':sha256(__file__),'device':'mps','memory_budget_gib':8,'acceptance_threshold':.8,'max_new_tokens':8,'route_calls':12,
        'routing_cache_reuse':False,'test_forward_calls':0,'OS_actions_allowed':False,'cloud_calls_allowed':False,'created_epoch':time.time(),
        'scope':'Three metadata-selected training examples verify a local execution path. This is not held-out accuracy or autonomous real-world task performance.'}
    verify_sources(repo,contract);write_new(output,contract)
    return {'status':'prepared','contract_sha256':sha256(output),'selected_records':3,'model_forward_calls':0,'checkpoint_sha256':checkpoint}


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def execute_episodes(runtime,rows,store,mapping,common,deadline):
    import numpy as np
    require(len(rows)==3 and all(r['split']=='train' for r in rows),'only the preselected three training records may be run')
    require(time.time()<deadline,'trial budget expired before inference')
    predictions=runtime.predict_batch(np.stack([store.get(r) for r in rows]),[r['task_family'] for r in rows],'fused')
    require(len(predictions)==3,'decoder count differs');trace=[];before=len(runtime.route_records)
    for row,prediction in zip(rows,predictions,strict=True):
        labels=common.TASKS[row['task_family']];p=common.checked_probabilities(prediction['probabilities'],labels)
        decoded=labels[int(p.argmax())];require(prediction['predicted_label']==decoded,'decoder argmax differs')
        for item in mapping.environments(row['id'],row['task_family'],20261122):
            require(time.time()<deadline,'trial budget expired during routing')
            prompt=mapping.router_prompt(item['environment'],decoded)
            text=runtime.route_text(prompt,max_new_tokens=8)
            evidence=runtime.route_records[-1]
            require(evidence['EEG_prefix_used'] is False and evidence['new_parameter_updates']==0 and evidence['do_sample'] is False,'routing is not frozen greedy base inference')
            env=mapping.Environment(item['environment']);abstain=bool(p.max()<.8);transition=env.step(text,abstain)
            trace.append({'id':row['id'],'episode_id':item['episode_id'],'task_family':row['task_family'],'decoder_correct':decoded==row['label'],
                'decoded_class':decoded,'probabilities':prediction['probabilities'],'abstained':abstain,'router_output':text,
                'router_correct_given_decoded_class':mapping.parse_action(text)==mapping.target_for(item['environment'],decoded),
                'router_evidence':evidence,'transition':transition,'state':env.state,'task_success':env.oracle(row['label']),
                'environment_sha256':item['environment_sha256'],'training_example_only':True,'real_OS_action':False})
    require(len(runtime.route_records)-before==12 and len(trace)==12,'must execute12fresh LLM calls, never reuse a cache')
    return predictions,trace


def worker(repo,contract_path,expected,output,deadline):
    repo=Path(repo).resolve();output=Path(output);require(sha256(contract_path)==expected,'trial contract changed')
    contract=read_json(contract_path);rows=verify_sources(repo,contract)
    require(sha256(__file__)==contract['wrapper_sha256'],'trial implementation changed')
    code=repo/CODE;sys.path.insert(0,str(code))
    import benchmark_common as common
    import harness_mapping as mapping
    require(Path(common.__file__).resolve()==(code/'benchmark_common.py').resolve() and Path(mapping.__file__).resolve()==(code/'harness_mapping.py').resolve(),'wrong helper module imported')
    plugin=load_module(code/'runtime.py','xavier_phase2_trial_runtime')
    started=time.monotonic()
    runtime=plugin.load_runtime(repo/contract['run'],'mps',memory_budget_gib=8,batch_size=3,dtype='bfloat16',expected_checkpoint_sha256=contract['checkpoint_sha256'],local_files_only=True)
    require(runtime.kind=='fused' and runtime.checkpoint_sha256==contract['checkpoint_sha256'],'loaded runtime differs')
    tokens={s:runtime.tokenizer.encode(s,add_special_tokens=False) for s in ('X','Y',' X',' Y')}
    require(all(len(x)==1 for x in tokens.values()) and tokens['X']!=tokens['Y'] and tokens[' X']!=tokens[' Y'],'routing candidates not equal single-token identities')
    predictions,trace=execute_episodes(runtime,rows,common.EEGStore(repo/contract['prepared']),mapping,common,deadline)
    verify_sources(repo,contract);require(sha256(contract_path)==expected,'contract changed during trial')
    with (output/'trace.jsonl').open('x') as stream:
        for row in trace:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    summary={'schema':'xavier-phase2-recorded-train-three-trial-v1','status':'completed','contract_sha256':expected,
        'checkpoint_sha256':contract['checkpoint_sha256'],'train_records':3,'test_forward_calls':0,'decoder_predictions':predictions,
        'actual_fresh_LLM_routing_calls':12,'routing_cache_hits':0,'synthetic_environment_episodes':12,'successful_synthetic_episodes':sum(r['task_success'] for r in trace),
        'abstained_episodes':sum(r['abstained'] for r in trace),'invalid_actions':sum(r['transition']['status']=='invalid_action' for r in trace),
        'incorrect_target_actions':sum(r['transition']['status']=='selected' and not r['task_success'] for r in trace),
        'token_probe':tokens,'resources':runtime.resource_summary(),'elapsed_seconds':time.monotonic()-started,'trace_sha256':sha256(output/'trace.jsonl'),
        'OS_actions':0,'cloud_API_calls':0,'local_files_only':True,'acceptance_threshold':.8,'probabilities_are_calibrated_intent_confidence':False,
        'heldout_performance':False,'autonomous_goal_validated':False,'source_hashes_unchanged_before_after':True,
        'scope':'Training-three execution smoke check on M1 with12fresh frozen-base routing calls and actual bounded in-memory state transitions; no held-out or real OS task-performance claim.'}
    write_new(output/'summary.json',summary);return {'status':'completed','summary_sha256':sha256(output/'summary.json')}


def run(repo,contract_path,expected,output,max_seconds):
    require(type(max_seconds) is int and 180<=max_seconds<=300,'M1 trial budget must remain180–300seconds')
    started=time.time();deadline=started+max_seconds
    require(sha256(contract_path)==expected,'contract SHA differs');contract=read_json(contract_path);verify_sources(repo,contract)
    require(sha256(__file__)==contract['wrapper_sha256'],'wrapper changed after preparation')
    output=Path(output).resolve();require(not output.exists(),'trial output must be new');output.mkdir(parents=True)
    command=[sys.executable,str(Path(__file__).resolve()),'_worker','--repo',str(Path(repo).resolve()),'--contract',str(Path(contract_path).resolve()),'--expected-contract-sha256',expected,'--output',str(output),'--deadline-epoch',str(deadline)]
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (output/'worker.log').open('x') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        write_new(output/'launch.json',{'pid':process.pid,'argv':command,'started_epoch':started,'deadline_epoch':deadline,'wrapper_sha256':sha256(__file__)})
        try:rc=process.wait(timeout=max(1,deadline-time.time()));timed_out=False
        except subprocess.TimeoutExpired:
            timed_out=True;process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            rc=process.returncode
    summary=output/'summary.json';completed=rc==0 and summary.exists() and read_json(summary).get('status')=='completed'
    write_new(output/'completion.json',{'status':'completed' if completed else 'failed','exit_code':rc,'timed_out':timed_out,
        'started_epoch':started,'finished_epoch':time.time(),'contract_sha256':expected,'summary_sha256':sha256(summary) if summary.exists() else None,
        'test_forward_calls':0,'OS_actions':0,'cloud_API_calls':0})
    require(completed,'M1 trial did not complete; preserve worker log and failure receipt')
    return {'status':'completed','completion_sha256':sha256(output/'completion.json'),'summary_sha256':sha256(summary)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--repo',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--recovery',required=True);p.add_argument('--output',required=True)
    for name in ('run','_worker'):
        p=sub.add_parser(name)
        for field in ('repo','contract','expected-contract-sha256','output'):p.add_argument('--'+field,required=True)
        if name=='run':p.add_argument('--max-seconds',type=int,default=300)
        else:p.add_argument('--deadline-epoch',type=float,required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.repo,args.evaluation,args.recovery,args.output)
    elif args.command=='run':result=run(args.repo,args.contract,args.expected_contract_sha256,args.output,args.max_seconds)
    else:result=worker(args.repo,args.contract,args.expected_contract_sha256,args.output,args.deadline_epoch)
    print(json.dumps(result,ensure_ascii=False))
