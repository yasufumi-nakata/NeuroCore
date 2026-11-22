"""Independent saved-loss/selection/exposure audit; no EEG or LLM forward."""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time

PHASE2_MANIFEST='d673ffa0bc54506fc5866a7092fa5ff81deffdcfdf0754b10902cb6fda8657a9'
SUBJECT_SPLIT='e5b38cbe78f1c0c8c06384c91266db40a8801be20b72262786502f6b5ce2e9c9'
SCIENTIFIC_PLAN='984b516475b06f4e7641074596790c43f76e34d33b6f047239a4ddbd2d0d4016'

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def object_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def same(a,b):
    return isinstance(a,(int,float)) and not isinstance(a,bool) and math.isfinite(a) and math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12)

def contained(root,relative):
    p=Path(relative);require(not p.is_absolute() and '..' not in p.parts and '\\' not in str(p),'untrusted artifact path')
    result=Path(root)/p;require(result.resolve().is_relative_to(Path(root).resolve()),'artifact escaped run');return result

def validation_metrics(records,expected,checkpoint_sha256):
    groups=defaultdict(list);seen=set();values=[]
    for p in records:
        identity=p.get('id');require(identity in expected and identity not in seen,'missing, repeated or unexpected SSL validation ID');seen.add(identity)
        require(p.get('subject_group')==expected[identity]['subject_group'] and p.get('checkpoint_sha256')==checkpoint_sha256,'SSL validation group/checkpoint mismatch')
        value=p.get('masked_MSE');require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0,'invalid recorded masked MSE')
        groups[p['subject_group']].append(float(value));values.append(float(value))
    require(seen==set(expected) and values,'SSL validation is incomplete')
    means={k:math.fsum(v)/len(v) for k,v in sorted(groups.items())}
    return {'mean_subject_masked_MSE':math.fsum(means.values())/len(means),'mean_window_masked_MSE':math.fsum(values)/len(values),'per_subject':means,'windows':len(values)}

def check_recorded_metrics(actual,recorded):
    for key in ['mean_subject_masked_MSE','mean_window_masked_MSE']:require(same(recorded.get(key),actual[key]),'recorded validation metric differs from per-record losses: '+key)
    require(recorded.get('windows')==actual['windows'] and set(recorded.get('per_subject',{}))==set(actual['per_subject']),'validation group/window accounting differs')
    for key,value in actual['per_subject'].items():require(same(recorded['per_subject'][key],value),'recorded subject MSE differs')
    require(recorded.get('mask_condition')=='fixed_ID_seed_same_at_initial_and_every_epoch','validation mask contract differs')

def earliest_minimum(candidates):
    require(candidates and candidates[0]['epoch']==0,'initial checkpoint must be a selection candidate')
    require([c['epoch'] for c in candidates]==list(range(len(candidates))),'complete validation epoch sequence has a gap or duplicate')
    winner=candidates[0]
    for value in candidates[1:]:
        improved=value['score']<winner['score']
        require(value['selected_best'] is improved,'checkpoint selection flag violates strict minimum/earliest tie')
        if improved:winner=value
    return winner

def union_length(intervals):
    total=Fraction(0);begin=end=None
    for a,b in sorted(intervals):
        require(b>a,'nonpositive source interval')
        if end is None:begin,end=a,b
        elif a<=end:end=max(end,b)
        else:total+=end-begin;begin,end=a,b
    return total if end is None else total+end-begin

def exposure(rows):
    native=defaultdict(list);target=defaultdict(list);payload=defaultdict(list);rates={}
    for r in rows:
        source=r['source_eeg_sha256'];signal=r['eeg_payload_sha256'];fs=Fraction(str(r['native_sample_rate_hz']))
        raw=r['raw_interval'];a,b=raw['native_sample_start'],raw['native_sample_stop']
        require(type(a) is int and type(b) is int and fs>0 and a>=0 and b>a,'invalid native sample interval')
        for key in (source,signal):require(key not in rates or rates[key]==fs,'same signal identity has incompatible sample rate');rates[key]=fs
        native[source].append((Fraction(a),Fraction(b)));payload[signal].append((Fraction(a),Fraction(b)))
        target[source].append((Fraction(str(r['window_start_s'])),Fraction(str(r['window_end_s']))))
    native_seconds=sum((union_length(v)/rates[k] for k,v in native.items()),Fraction(0))
    signal_seconds=sum((union_length(v)/rates[k] for k,v in payload.items()),Fraction(0))
    target_seconds=sum((union_length(v) for v in target.values()),Fraction(0))
    return {'unique_window_ids':len({r['id'] for r in rows}),'native_source_interval_union_seconds':float(native_seconds),'physical_payload_interval_union_seconds':float(signal_seconds),
            'target_clock_interval_union_seconds':float(target_seconds),'native_source_interval_union_hours':float(native_seconds/3600),'physical_payload_interval_union_hours':float(signal_seconds/3600),
            'raw_recordings':len(native),'unique_signal_payloads':len(payload),'filter_history_is_not_counted':True}

def checked_seen(identities,by_id):
    require(isinstance(identities,list) and all(isinstance(x,str) for x in identities) and len(identities)==len(set(identities)),'duplicate or malformed observed training ID')
    require(all(i in by_id and by_id[i]['split']=='train' for i in identities),'validation/test/unknown EEG in observed training IDs')
    return [by_id[i] for i in identities]

def tensor_hash(tensors):
    import torch
    h=hashlib.sha256()
    for name,value in sorted(tensors.items()):
        require(isinstance(value,torch.Tensor),'non-tensor exported encoder item')
        v=value.detach().cpu().contiguous();h.update(json.dumps([name,str(v.dtype),list(v.shape)]).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()

def audit(ssl_run,phase2_manifest,expected_contract_sha256,supervised_run=None,fused_run=None):
    import torch
    torch.set_num_threads(2)
    root=Path(ssl_run).resolve();code=Path(__file__).resolve().parent;inputs={}
    def pin(path):
        path=Path(path);digest=sha(path);inputs[str(path)]=digest;return digest
    require(pin(phase2_manifest)==PHASE2_MANIFEST,'frozen MI manifest changed')
    mi=lines(phase2_manifest);mi_by={r['id']:r for r in mi};require(len(mi_by)==len(mi),'duplicate MI ID')
    contract_path=root/'execution_contract.json';require(pin(contract_path)==expected_contract_sha256,'SSL execution contract changed')
    contract=read(contract_path);sources=contract['source_sha256'];require(sources['protocol.scientific.v1.json']==SCIENTIFIC_PLAN,'pre-test phase3 plan differs')
    for name,digest in sources.items():
        path=code.parent/'eegmmidb_phase2'/name.removeprefix('phase2/') if name.startswith('phase2/') else contained(code,name)
        require(pin(path)==digest,'pinned scientific source changed: '+name)
    member=read(root/'training_membership.json');member_sha=pin(root/'training_membership.json');config=read(root/'run_config.json');pin(root/'run_config.json')
    require(config['source_sha256']==sources and config['execution_contract_sha256']==expected_contract_sha256 and config['initial_state_is_candidate'] is True and config['selection_metric']=='mean_subject_masked_MSE','SSL config/selection contract differs')
    for filename,key in [('ssl_manifest.jsonl','manifest_sha256'),('ssl_artifacts.json','artifacts_sha256'),('source_allowlist.jsonl','source_allowlist_sha256'),('source_denylist.jsonl','source_denylist_sha256'),('subject_split.json','subject_split_sha256')]:
        require(pin(root/filename)==member[key],'SSL source/membership artifact changed: '+filename)
    require(member['subject_split_sha256']==SUBJECT_SPLIT and member['test_used_for_tuning'] is False and member['action_labels_used'] is False,'original participant split/use restriction changed')
    assignment=read(root/'subject_split.json')['assignments'];test_groups={s for s,v in assignment.items() if v=='test'};require(len(test_groups)==17,'wrong original test group count')
    rows=lines(root/'ssl_manifest.jsonl');by_id={r['id']:r for r in rows};require(len(by_id)==len(rows),'duplicate SSL ID')
    allow_rows=lines(root/'source_allowlist.jsonl');allowed={r['source_eeg_sha256']:r for r in allow_rows};require(len(allowed)==len(allow_rows),'duplicate allowed recording')
    denied=lines(root/'source_denylist.jsonl');forbidden_sources={r['source_eeg_sha256'] for r in denied};forbidden_payload={r['eeg_payload_sha256'] for r in denied}
    test_sources={r['source_eeg_sha256'] for r in mi if r['split']=='test'};test_payload={r['eeg_payload_sha256'] for r in mi if r['split']=='test'}
    group_split=defaultdict(set);source_split=defaultdict(set);payload_split=defaultdict(set)
    for r in rows:
        require(r['split'] in ('train','val') and not {'label','action_label','events','gold_annotation_code','task_family'}&r.keys(),'behavior label or held-out split in SSL')
        original=allowed.get(r['source_eeg_sha256']);require(original is not None,'unadmitted SSL source')
        for field in ('subject_group','source_recording_id','split','eeg_payload_sha256'):require(r[field]==original[field],'source metadata alias differs')
        require(assignment.get(r['subject_group'])==r['split'] and r['subject_group'] not in test_groups,'test participant or changed participant assignment')
        require(r['source_eeg_sha256'] not in forbidden_sources|test_sources and r['eeg_payload_sha256'] not in forbidden_payload|test_payload,'denied/test raw or physical payload entered SSL')
        group_split[r['subject_group']].add(r['split']);source_split[r['source_eeg_sha256']].add(r['split']);payload_split[r['eeg_payload_sha256']].add(r['split'])
    for mapping in (group_split,source_split,payload_split):require(all(len(v)==1 for v in mapping.values()),'person/raw/payload crossed train and validation')
    for split,key in [('train','training_ids'),('val','validation_ids')]:require(member[key]==sorted(r['id'] for r in rows if r['split']==split),'declared SSL member IDs differ from actual manifest')
    expected={r['id']:r for r in rows if r['split']=='val'};require(expected,'empty SSL validation')
    def load_checkpoint(path):
        digest=pin(path);state=torch.load(path,map_location='cpu',weights_only=True)
        require(state['schema']=='xavier-phase3-masked-EEG-checkpoint-v1' and state['source_sha256']==sources and state['membership_sha256']==member_sha and state['config']==config,'checkpoint/source/membership/config identity differs')
        checked_seen(state['seen_train_ids'],by_id)
        require('optimizer' in state and 'rng' in state,'optimizer/RNG recovery state absent')
        return digest,state
    initial_sha,initial=load_checkpoint(root/'initial_epoch_000.pt')
    require(initial['best_epoch']==initial['global_step']==0 and initial['seen_train_ids']==[] and initial['training_examples_presented']==0,'epoch0 contains optimizer exposure')
    initial_encoder={k.removeprefix('encoder.'):v for k,v in initial['model'].items() if k.startswith('encoder.')}
    require(tensor_hash(initial_encoder)==initial['initial_encoder_tensor_sha256'],'initial encoder tensor hash differs')
    sys.path.insert(0,str(code.parent))
    from eegmmidb_phase2.model import EEGEncoder
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed']);fresh=EEGEncoder().state_dict()
    require(set(fresh)==set(initial_encoder) and all(torch.equal(fresh[k],initial_encoder[k]) for k in fresh),'initial SSL encoder differs from the same seeded scratch encoder')
    initial_values=lines(root/'validation_initial.jsonl');pin(root/'validation_initial.jsonl');pin(root/'initial_validation.json')
    initial_metric=validation_metrics(initial_values,expected,initial_sha);check_recorded_metrics(initial_metric,read(root/'initial_validation.json'))
    candidates=[{'epoch':0,'score':initial_metric['mean_subject_masked_MSE'],'checkpoint_sha256':initial_sha,'checkpoint':'initial_epoch_000.pt','global_step':0}]
    history_path=root/'history.jsonl';history=lines(history_path) if history_path.exists() else []
    if history_path.exists():pin(history_path)
    previous_step=0
    for h in history:
        epoch=h['epoch'];require(type(epoch) is int and epoch==len(candidates),'validation epoch sequence changed')
        digest,state=load_checkpoint(contained(root,h['checkpoint']));require(digest==h['checkpoint_sha256'],'candidate checkpoint changed')
        require(state['epoch_completed'] is True and state['epoch_zero_based']==epoch-1 and state['global_step']==h['global_step']>previous_step,'candidate does not describe a complete new epoch')
        path=root/f'validation_epoch_{epoch:03d}.jsonl';pin(path);metrics=validation_metrics(lines(path),expected,digest);check_recorded_metrics(metrics,h['validation'])
        candidate={'epoch':epoch,'score':metrics['mean_subject_masked_MSE'],'checkpoint_sha256':digest,'checkpoint':h['checkpoint'],'global_step':state['global_step'],'selected_best':h['selected_best']}
        candidates.append(candidate);winner=earliest_minimum(candidates)
        require(state['best_epoch']==winner['epoch'] and same(state['best_validation_score'],winner['score']),'checkpoint running-best state differs from saved validation records')
        previous_step=state['global_step']
    winner=earliest_minimum(candidates);best_sha,best=load_checkpoint(root/'best.pt');require(best_sha==winner['checkpoint_sha256'],'selected encoder is not earliest minimum including epoch0')
    seal=read(root/'selected_encoder_seal.json');pin(root/'selected_encoder_seal.json');weights_path=root/'selected_encoder.pt';weights_sha=pin(weights_path);weights=torch.load(weights_path,map_location='cpu',weights_only=True)
    encoder={k.removeprefix('encoder.'):v for k,v in best['model'].items() if k.startswith('encoder.')}
    require(set(weights)==set(encoder) and all(torch.equal(weights[k],encoder[k]) for k in encoder),'exported encoder differs from selected whole checkpoint')
    require(seal['weights_sha256']==weights_sha and seal['tensor_sha256']==tensor_hash(weights) and seal['best_checkpoint_sha256']==best_sha,'selected encoder seal differs from actual bytes/tensors')
    require(seal['selected_epoch']==winner['epoch'] and same(seal['selected_validation_mean_subject_MSE'],winner['score']),'selected encoder metric/epoch misreported')
    require(seal['updated_from_initial']==(tensor_hash(weights)!=tensor_hash(initial_encoder)),'initial versus adapted encoder status differs')
    summary=read(root/'run_summary.json');pin(root/'run_summary.json')
    terminal_receipt_path=root.parent.parent/'workflow/train_ssl.exit.json';terminal_receipt=read(terminal_receipt_path);terminal_receipt_sha=pin(terminal_receipt_path)
    require(terminal_receipt.get('process_returned_normally') is True and terminal_receipt.get('terminated_by_supervisor') is False and terminal_receipt.get('source_sha256')==sources,'SSL process did not return normally with the pinned source')
    command=terminal_receipt['command'];require(len(command)>3 and Path(command[1]).name=='train_ssl.py' and command[2]=='train' and command.count('--run-dir')==1,'wrong terminal SSL command')
    run_argument=command[command.index('--run-dir')+1];require(run_argument==terminal_receipt['run_dir'] and Path(run_argument).name==root.name and Path(run_argument).parent.name=='runs','terminal receipt belongs to another SSL run')
    if summary['status']=='completed':require(summary['stop_reason'] is None and summary['completed_epochs']==config['max_epochs'] and terminal_receipt['exit_code']==0,'completed SSL status/epoch/exit differs')
    else:
        require(summary['status']=='partial' and summary['stop_reason'] in ('stage_time_budget','absolute_deadline') and terminal_receipt['exit_code']==2,'unexpected SSL failure cannot advance')
        require(summary.get('partial_advancement_exit_receipt_sha256')==terminal_receipt_sha,'time-partial encoder lacks exact normal-exit advancement evidence')
    require(summary['completed_epochs']==len(history) and summary['selected_epoch']==winner['epoch'] and same(summary['selected_validation_mean_subject_MSE'],winner['score']),'final summary selection/history differs')
    require(summary['best_checkpoint_sha256']==best_sha and summary['selected_encoder_seal_sha256']==sha(root/'selected_encoder_seal.json'),'final summary selected identity differs')
    all_seen=checked_seen(summary['seen_train_ids'],by_id)
    require(object_hash(sorted(summary['seen_train_ids']))==summary['seen_train_ids_sha256'] and len(all_seen)==summary['unique_seen_training_windows'],'training exposure ID digest/count differs')
    terminal_candidates=[root/candidates[-1]['checkpoint']]+sorted(root.glob('stopped_attempt_*.pt'))
    terminal_states=[load_checkpoint(p)[1] for p in terminal_candidates];terminal=max(terminal_states,key=lambda s:s['global_step'])
    require(terminal['global_step']==summary['optimizer_steps'] and sorted(terminal['seen_train_ids'])==sorted(summary['seen_train_ids']) and terminal['training_examples_presented']==summary['training_examples_presented'],'final completed-step exposure differs from terminal checkpoint')
    complete_exposure=exposure(all_seen);selected_exposure=exposure(checked_seen(best['seen_train_ids'],by_id))
    require(same(summary['unique_source_interval_union_seconds'],complete_exposure['target_clock_interval_union_seconds']),'reported source interval union differs from actual seen IDs')
    require(fused_run is None or supervised_run is not None,'fused selected-path accounting requires its supervised parent')
    path_exposure={};chain_rows=checked_seen(best['seen_train_ids'],by_id);supervised_selected_sha=None
    for role,run in [('MI_supervised',supervised_run),('fused_alignment',fused_run)]:
        if run is None:continue
        run=Path(run).resolve();state_path=run/'best.pt';digest=pin(state_path);state=torch.load(state_path,map_location='cpu',weights_only=True)
        require(state['schema']=='xavier-eegmmidb-phase3-supervised-checkpoint-v1','wrong downstream checkpoint schema')
        cfg=read(run/'run_config.json');pin(run/'run_config.json');membership=read(run/'training_membership.json');pin(run/'training_membership.json')
        require(state['config']==cfg and state['training_membership_sha256']==sha(run/'training_membership.json'),'downstream configuration/membership differs')
        origin=cfg['ssl_encoder_origin'];require(origin['ssl_best_checkpoint_sha256']==best_sha and origin['selected_encoder_sha256']==weights_sha and origin['selected_encoder_seal_sha256']==sha(root/'selected_encoder_seal.json'),'downstream did not initialize from this selected SSL encoder')
        require(membership['manifest_sha256']==PHASE2_MANIFEST and membership['test_used_for_tuning'] is False,'downstream cohort/use differs')
        for split,key in [('train','training_ids'),('val','validation_ids'),('test','test_ids')]:require(membership[key]==sorted(r['id'] for r in mi if r['split']==split),'downstream member list differs')
        if role=='MI_supervised':supervised_selected_sha=digest
        else:require(cfg['scratch_parent_best_sha256']==supervised_selected_sha,'fused checkpoint initialized from a different supervised selection')
        seen=checked_seen(state['seen_train_ids'],mi_by);chain_rows+=seen
        path_exposure[role]={'checkpoint_sha256':digest,'own_selected_checkpoint_exposure':exposure(seen),'selected_path_union':exposure(chain_rows)}
    result={'schema':'xavier-phase3-independent-SSL-selection-audit-v1','status':'passed','created_epoch':time.time(),'audit_source_sha256':sha(__file__),
            'SSL_execution_contract_sha256':expected_contract_sha256,'SSL_best_checkpoint_sha256':best_sha,'selected_encoder_sha256':weights_sha,
            'complete_validation_candidates':len(candidates),'SSL_validation_windows':len(expected),'SSL_validation_subject_groups':len(initial_metric['per_subject']),
            'initial_validation_subject_MSE':initial_metric['mean_subject_masked_MSE'],'selected_validation_subject_MSE':winner['score'],'selected_epoch':winner['epoch'],
            'earliest_minimum_including_epoch0_verified':True,'all_validation_IDs_and_checkpoint_hashes_verified':True,'source_person_and_payload_exclusions_verified':True,'original_test_subject_groups_excluded':17,
            'same_seeded_initial_encoder_verified':True,'exported_encoder_tensors_equal_selected_checkpoint':True,'selected_encoder_updated_from_initial':seal['updated_from_initial'],
            'completed_optimizer_step_exposure':complete_exposure,'selected_encoder_exposure':selected_exposure,'selected_downstream_path_exposure':path_exposure,
            'training_examples_presented':summary['training_examples_presented'],'epoch_repetitions_do_not_increase_distinct_EEG_hours':True,
            'inputs_sha256':inputs,'model_forward_calls':0,'waveform_bytes_rehashed_by_this_audit':False,'raw_reconstruction_losses_recomputed_by_forward':False,
            'scope':'Recomputes subject-balanced scores from complete saved per-window losses; independently verifies source identity, checkpoint selection, encoder tensors and completed-step seen-ID interval unions. Partial failed-forward exposure outside committed optimizer steps is not inferred.'}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['ssl-run','phase2-manifest','expected-contract-sha256','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--supervised-run');p.add_argument('--fused-run');a=p.parse_args()
    value=audit(a.ssl_run,a.phase2_manifest,a.expected_contract_sha256,a.supervised_run,a.fused_run)
    with Path(a.output).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:value[k] for k in ['status','selected_epoch','complete_validation_candidates','selected_encoder_updated_from_initial','completed_optimizer_step_exposure','selected_encoder_exposure']}))

if __name__=='__main__':main()
