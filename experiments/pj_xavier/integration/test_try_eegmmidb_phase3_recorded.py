"""Targeted demo selection, fresh-call accounting, recovery and no-op tests."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np

import try_eegmmidb_phase3_recorded as trial
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'eegmmidb_phase2'))
import benchmark_common as common
import harness_mapping as mapping


def rows():
    return [{'id':f'train-{i}','split':'train','task_family':'left_right','label':'left_fist'} for i in range(3)]


def terminal_fixture(phase):
    phase=Path(phase);run=phase/trial.RUN;run.mkdir(parents=True);(phase/'workflow').mkdir()
    original=[];pred=[]
    (run/'epoch_001.pt').write_bytes(b'earlier');(run/'epoch_002.pt').write_bytes(b'best')
    best=trial.sha256(run/'epoch_002.pt')
    for i in range(4):
        label=('left_fist','right_fist')[i%2];r={'id':f'val-{i}','split':'val','subject_group':f'person{i//2}','task_family':'left_right','label':label};original.append(r)
        pred.append({'id':r['id'],'split':'val','subject_group':r['subject_group'],'task_family':'left_right','true_label':label,
                     'probabilities':{'left_fist':.9 if i%2==0 else .1,'right_fist':.1 if i%2==0 else .9},'predicted_label':label,'checkpoint_sha256':best})
    v=run/'validation_epoch_002.jsonl';v.write_text(''.join(json.dumps(x)+'\n' for x in pred))
    history=[{'epoch':1,'global_step':10,'validation':{'selection_score':.75,'selection_partition':'val'},'checkpoint':'epoch_001.pt','checkpoint_sha256':trial.sha256(run/'epoch_001.pt')},
             {'epoch':2,'global_step':20,'validation':{'selection_score':1.,'selection_partition':'val'},'checkpoint':'epoch_002.pt','checkpoint_sha256':best,'validation_predictions_sha256':trial.sha256(v)}]
    hp=run/'history.jsonl';hp.write_text(''.join(json.dumps(x)+'\n' for x in history))
    stopped=run/'stopped.pt';stopped.write_bytes(b'bounded stop')
    trial.write_new(run/'stop_attempt_001.json',{'reason':'absolute_deadline','checkpoint':'stopped.pt','checkpoint_sha256':trial.sha256(stopped),'resumable':True,'test_forward_calls':0,'validation_skipped_after_stop':True})
    global_end=1795331083.750155;cutoff=global_end-2100
    trial.write_new(phase/'gpu_budget_v1.json',{'absolute_deadline_epoch':global_end})
    trial.write_new(run/'attempt_001.json',{'arguments':{'absolute_deadline_epoch':cutoff}})
    command=['python','train_supervised.py','train','--run-dir','/content/xavier_phase3/'+trial.RUN,'--absolute-deadline-epoch',str(cutoff)]
    trial.write_new(phase/'workflow/train_fused.launch.json',{'pid':123,'command':command,'started_epoch':cutoff-120,'deadline_epoch':cutoff+10})
    trial.write_new(phase/'workflow/train_fused.exit.json',{'pid':123,'command':command,'exit_code':2,'process_returned_normally':True,'terminated_by_supervisor':False,'source_sha256':{'fixture':'science'},'run_dir':'/content/xavier_phase3/'+trial.RUN,'finished_epoch':cutoff+.2})
    summary={'history_sha256':trial.sha256(hp),'best_primary_validation_subject_BA':1.,'completed_epochs':2,'status':'partial','stop_reason':'absolute_deadline','last_checkpoint':'stopped.pt','elapsed_stage_seconds':300.}
    config={'max_epochs':30,'max_seconds':1800,'source_sha256':{'fixture':'science'}}
    return run,summary,config,best,original


def checked_terminal(phase,*args):
    with mock.patch.object(trial,'BUDGET_SEAL',trial.sha256(Path(phase)/'gpu_budget_v1.json')):
        return trial.verify_training_terminal(phase,*args)


class Store:
    def __init__(self):self.reads=[]
    def get(self,r):self.reads.append(r['id']);return np.zeros((64,600),np.float32)


class Runtime:
    def __init__(self,confidence=.9,mode='correct'):
        self.route_records=[];self.calls=0;self.prompts=[];self.confidence=confidence;self.mode=mode;self.predict_calls=0
    def predict_batch(self,eeg,task_families,condition):
        self.predict_calls+=1
        return [{'predicted_label':'left_fist','probabilities':{'left_fist':self.confidence,'right_fist':1-self.confidence}} for _ in task_families]
    def route_text(self,prompt,max_new_tokens=8):
        self.calls+=1;self.prompts.append(prompt)
        self.route_records.append({'EEG_prefix_used':False,'new_parameter_updates':0 if self.mode!='updates' else 1,'do_sample':False})
        if self.mode=='invalid':return 'X or Y, depending on the cue'
        control=prompt.split('Decoded control: ')[1].split('\n')[0]
        return next(line.split(':')[0].split()[-1] for line in prompt.splitlines() if line.startswith('Target ') and line.endswith('assigned control '+control+'.'))


class SelectionTests(unittest.TestCase):
    def test_training_selection_is_label_and_outcome_independent(self):
        values=rows()+[{'id':'heldout','split':'test','label':'left_fist'},{'id':'selection','split':'val','label':'right_fist'}]
        first=trial.selected_train_rows(values)
        altered=copy.deepcopy(values)
        for r in altered:r['label']='unrelated';r['predicted_correct']=False
        self.assertEqual([r['id'] for r in first],[r['id'] for r in trial.selected_train_rows(altered)])
        self.assertTrue(all(r['split']=='train' for r in first))
    def test_duplicate_or_insufficient_training_rejected(self):
        with self.assertRaises(ValueError):trial.selected_train_rows(rows()+[rows()[0]])
        with self.assertRaises(ValueError):trial.selected_train_rows(rows()[:2])


class ExecutionTests(unittest.TestCase):
    def test_three_decoder_examples_and_twelve_fresh_routes(self):
        runtime=Runtime();store=Store();prediction,trace=trial.execute_episodes(runtime,rows(),store,mapping,common,time.time()+20)
        self.assertEqual(len(prediction),3);self.assertEqual(runtime.predict_calls,1);self.assertEqual(runtime.calls,12);self.assertEqual(len(trace),12)
        self.assertEqual(sum(r['task_success'] for r in trace),12);self.assertEqual(store.reads,[r['id'] for r in rows()])
        self.assertLess(len(set(runtime.prompts)),12,'Repeated contexts must still make fresh calls in this trial')
        for prompt in runtime.prompts:
            for r in rows():self.assertNotIn(r['id'],prompt)
            self.assertNotIn('true_label',prompt);self.assertNotIn('gold',prompt)
    def test_rest_of_route_calls_not_skipped_by_abstention(self):
        runtime=Runtime(confidence=.7);_,trace=trial.execute_episodes(runtime,rows(),Store(),mapping,common,time.time()+20)
        self.assertEqual(runtime.calls,12);self.assertTrue(all(r['abstained'] and r['state']['transitions']==0 and not r['task_success'] for r in trace))
    def test_invalid_text_never_receives_success_or_changes_state(self):
        runtime=Runtime(mode='invalid');_,trace=trial.execute_episodes(runtime,rows(),Store(),mapping,common,time.time()+20)
        self.assertTrue(all(r['transition']['status']=='invalid_action' and r['state']['selected_target'] is None and not r['task_success'] for r in trace))
    def test_heldout_record_rejected_before_any_model_or_array_read(self):
        runtime=Runtime();store=Store();r=rows();r[0]['split']='test'
        with self.assertRaises(ValueError):trial.execute_episodes(runtime,r,store,mapping,common,time.time()+20)
        self.assertEqual(runtime.predict_calls,0);self.assertEqual(store.reads,[])
    def test_expired_budget_rejected_before_any_inference(self):
        runtime=Runtime()
        with self.assertRaises(ValueError):trial.execute_episodes(runtime,rows(),Store(),mapping,common,time.time()-1)
        self.assertEqual(runtime.predict_calls,0)
    def test_parameter_updates_fail_frozen_route_contract(self):
        with self.assertRaises(ValueError):trial.execute_episodes(Runtime(mode='updates'),rows(),Store(),mapping,common,time.time()+20)


class RecoveryTests(unittest.TestCase):
    def test_member_hash_path_and_complete_status_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);phase=repo/trial.PHASE;path=phase/'runs/eegnet_smollm_fused_v1/best.pt';path.parent.mkdir(parents=True);path.write_bytes(b'fixture')
            value={'schema':'xavier-phase3-complete-artifact-recovery-v1','status':'completed','all_member_hashes_verified':True,'member_count':1,
                   'files':{str(path.relative_to(phase)):{'local_path':str(path),'sha256':trial.sha256(path),'bytes':7}}}
            receipt=phase/'recovery.json';receipt.write_text(json.dumps(value))
            self.assertEqual(trial.verify_member_ledger(repo,receipt,[path])['member_count'],1)
            altered=copy.deepcopy(value);next(iter(altered['files'].values()))['local_path']=str(repo/'unrelated')
            receipt.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):trial.verify_member_ledger(repo,receipt,[path])
            receipt.write_text(json.dumps(value));path.write_bytes(b'changed')
            with self.assertRaises(ValueError):trial.verify_member_ledger(repo,receipt,[path])
    def test_unlisted_required_member_or_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp)
            for path in ('../secret','/tmp/secret','dir\\secret'):
                with self.subTest(path=path),self.assertRaises(ValueError):trial.contained(repo,path)
    def test_contract_without_identity_fields_rejected(self):
        with self.assertRaises(ValueError):trial.verify_sources('/tmp',{'schema':trial.SCHEMA})


class PlannedStopTests(unittest.TestCase):
    def test_planned_absolute_deadline_with_complete_validation_is_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,summary,config,best,original=terminal_fixture(tmp)
            proof,files=checked_terminal(tmp,run,summary,config,best,original)
            self.assertTrue(proof['normal_planned_budget_stop']);self.assertEqual(proof['selected_epoch'],2)
            self.assertEqual(proof['selected_validation_records'],4);self.assertIn('selection:workflow_exit',files)
    def test_memory_failure_is_never_misread_as_planned_budget_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,summary,config,best,original=terminal_fixture(tmp);summary['stop_reason']='memory_budget'
            with self.assertRaises(ValueError):checked_terminal(tmp,run,summary,config,best,original)
    def test_partial_requires_the_fixed_trainer_exit_code(self):
        for code in (0,1,-9):
            with self.subTest(code=code),tempfile.TemporaryDirectory() as tmp:
                run,summary,config,best,original=terminal_fixture(tmp)
                p=Path(tmp)/'workflow/train_fused.exit.json';value=json.loads(p.read_text());value['exit_code']=code;p.write_text(json.dumps(value))
                with self.assertRaises(ValueError):checked_terminal(tmp,run,summary,config,best,original)
    def test_stage_budget_requires_positive_matching_terminal_evidence(self):
        for code,timed_out,admitted in ((2,False,True),(0,False,False),(1,False,False),(2,True,False)):
            with self.subTest(code=code,timed_out=timed_out),tempfile.TemporaryDirectory() as tmp:
                run,summary,config,best,original=terminal_fixture(tmp)
                summary.update(stop_reason='stage_time_budget',elapsed_stage_seconds=1800.1)
                stop_path=run/'stop_attempt_001.json';stop=json.loads(stop_path.read_text());stop['reason']='stage_time_budget';stop_path.write_text(json.dumps(stop))
                p=Path(tmp)/'workflow/train_fused.exit.json';value=json.loads(p.read_text());value.update(exit_code=code,process_returned_normally=not timed_out,terminated_by_supervisor=timed_out,finished_epoch=1795328983.);p.write_text(json.dumps(value))
                if admitted:
                    proof,paths=checked_terminal(tmp,run,summary,config,best,original)
                    self.assertTrue(proof['normal_planned_budget_stop']);self.assertIn('selection:workflow_exit',paths)
                else:
                    with self.assertRaises(ValueError):checked_terminal(tmp,run,summary,config,best,original)
    def test_earlier_unplanned_absolute_deadline_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,summary,config,best,original=terminal_fixture(tmp);p=run/'attempt_001.json';a=json.loads(p.read_text());a['arguments']['absolute_deadline_epoch']-=10;p.write_text(json.dumps(a))
            with self.assertRaises(ValueError):checked_terminal(tmp,run,summary,config,best,original)
    def test_selected_validation_predictions_must_support_recorded_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,summary,config,best,original=terminal_fixture(tmp)
            hp=run/'history.jsonl';history=[json.loads(x) for x in hp.read_text().splitlines()];history[1]['validation']['selection_score']=.9;summary['best_primary_validation_subject_BA']=.9
            hp.write_text(''.join(json.dumps(x)+'\n' for x in history));summary['history_sha256']=trial.sha256(hp)
            with self.assertRaises(ValueError):checked_terminal(tmp,run,summary,config,best,original)
    def test_equal_validation_score_must_select_earliest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,summary,config,best,original=terminal_fixture(tmp);hp=run/'history.jsonl';history=[json.loads(x) for x in hp.read_text().splitlines()];history[0]['validation']['selection_score']=1.
            hp.write_text(''.join(json.dumps(x)+'\n' for x in history));summary['history_sha256']=trial.sha256(hp)
            with self.assertRaises(ValueError):checked_terminal(tmp,run,summary,config,best,original)


class SSLGuards(unittest.TestCase):
    def audit(self):
        return {'status':'passed','model_forward_calls':0,
                **{k:True for k in trial.SSL_GATES},
                'selected_downstream_path_exposure':{'fused_alignment':{'checkpoint_sha256':'pinned'}}}
    def test_all_five_independent_gates_required(self):
        value=self.audit();trial.verify_audit_gates(value,'pinned')
        for k in trial.SSL_GATES:
            for changed in (False,None,1):
                with self.subTest(gate=k,value=changed):
                    bad=copy.deepcopy(value);bad[k]=changed
                    with self.assertRaises(ValueError):trial.verify_audit_gates(bad,'pinned')
    def test_audit_cannot_change_selected_fused_or_add_forward(self):
        with self.assertRaises(ValueError):trial.verify_audit_gates(self.audit(),'other')
        bad=self.audit();bad['model_forward_calls']=1
        with self.assertRaises(ValueError):trial.verify_audit_gates(bad,'pinned')
    def test_remote_local_origin_mapping_contains_all_source_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp)
            self.assertEqual(trial.local_audited_path(repo,'/content/xavier_phase3/runs/masked_ssl_v1/best.pt'),repo/trial.PHASE/'runs/masked_ssl_v1/best.pt')
            self.assertEqual(trial.local_audited_path(repo,'/content/xavier_phase2/prepared/manifest.jsonl'),repo/'private/xavier/eegmmidb_phase2_v1/prepared/manifest.jsonl')
            self.assertEqual(trial.local_audited_path(repo,'/content/xavier_phase3/code/eegmmidb_phase3/common.py'),repo/'experiments/pj_xavier/eegmmidb_phase3/common.py')
            for bad in ('/content/other/runs/file','/content/xavier_phase3/../secret','/content/xavier_phase2/code/../../secret'):
                with self.subTest(path=bad),self.assertRaises(ValueError):trial.local_audited_path(repo,bad)
    def test_original_wrapper_and_same_routing_logic_preserved(self):
        old=Path(__file__).with_name('try_eegmmidb_recorded.py')
        self.assertEqual(trial.sha256(old),'cf96749bd5878998a9d9a73e400924e5e36778d82d24e44286905d4aef46881f')
        import ast
        a=ast.parse(old.read_text());b=ast.parse(Path(trial.__file__).read_text())
        for name in ('selected_train_rows','execute_episodes'):
            first=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==name)
            second=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(first),ast.dump(second))
    def test_wrong_budget_identity_rejected_before_partial_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,summary,config,best,original=terminal_fixture(tmp)
            with self.assertRaises(ValueError):trial.verify_training_terminal(tmp,run,summary,config,best,original)
    def test_changed_scientific_source_in_child_exit_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,summary,config,best,original=terminal_fixture(tmp)
            p=Path(tmp)/'workflow/train_fused.exit.json';value=json.loads(p.read_text());value['source_sha256']={'different':'code'};p.write_text(json.dumps(value))
            with self.assertRaises(ValueError):checked_terminal(tmp,run,summary,config,best,original)
    def test_actual_cpu_weights_only_roundtrip_and_parent_lineage(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);phase=repo/trial.PHASE;code=repo/trial.CODE;code.mkdir(parents=True)
            (code/'benchmark_ssl_audit.py').write_text('fixture auditor')
            run=phase/trial.RUN;parent=phase/trial.PARENT_RUN;run.mkdir(parents=True);parent.mkdir(parents=True)
            parent_member=parent/'training_membership.json';parent_member.write_text('{}')
            run_member=run/'training_membership.json';run_member.write_text('{}')
            origin={'selected_encoder_seal_sha256':'fixture'};sources={'source':'fixed'}
            cfg_parent={'ssl_encoder_origin':origin};(parent/'run_config.json').write_text(json.dumps(cfg_parent))
            torch.save({'schema':'fixture-checkpoint','config':cfg_parent,'source_sha256':sources,'training_membership_sha256':trial.sha256(parent_member),'tensor':torch.arange(3)},parent/'best.pt')
            cfg={'ssl_encoder_origin':origin,'scratch_parent_best_sha256':trial.sha256(parent/'best.pt')}
            state={'schema':'fixture-checkpoint','config':cfg,'source_sha256':sources,'training_membership_sha256':trial.sha256(run_member),'optimizer':{},'rng':torch.get_rng_state(),'tensor':torch.arange(4)}
            torch.save(state,run/'best.pt');checkpoint=trial.sha256(run/'best.pt')
            audit=self.audit();audit['selected_downstream_path_exposure']['fused_alignment']['checkpoint_sha256']=checkpoint
            audit['audit_source_sha256']=trial.sha256(code/'benchmark_ssl_audit.py');audit['inputs_sha256']={}
            audit_path=phase/'audit.json';audit_path.write_text(json.dumps(audit))
            membership={'subject_groups':{'test':[]},'raw_source_sha256':{'test':[]},'payload_sha256':{'test':[]}}
            plugin=mock.Mock(CHECKPOINT_SCHEMA='fixture-checkpoint');plugin.source_seals.return_value=sources;plugin.verify_ssl_origin.return_value=origin
            with mock.patch.object(trial,'load_module',return_value=plugin):
                actual,_=trial.verify_ssl_lineage(repo,run,cfg,membership,checkpoint,audit_path);self.assertEqual(actual,origin)
                bad=copy.deepcopy(cfg);bad['scratch_parent_best_sha256']='changed'
                with self.assertRaises(ValueError):trial.verify_ssl_lineage(repo,run,bad,membership,checkpoint,audit_path)


if __name__=='__main__':unittest.main()
