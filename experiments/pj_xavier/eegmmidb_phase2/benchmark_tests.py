"""Targeted leakage, covariance, paired-control and synthetic-oracle tests."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

import benchmark_common as c
import benchmark_csp as b
import harness_mapping as h
import harness_evaluate as he
import benchmark_audit as audit
import benchmark_evaluate as evaluator
import benchmark_idle as idle


def row(i=0,split='train',subject=None,label='left_fist'):
    subject=subject or split
    return {'id':f'{split}-{i}','dataset':'fixture','subject':subject,'session':'s1','recording_id':subject+'-run4',
            'split':split,'task_family':'left_right','label':label,'label_source':'recorded_event','channels':[f'C{x}' for x in range(64)],
            'sample_rate_hz':200,'unit':'V','source_eeg_sha256':c.object_hash(subject),'eeg_payload_sha256':c.object_hash(['EEG',subject]),
            'eeg_path':subject+'.npy','eeg_index':i,'window_start_s':20+i*4,'window_end_s':23+i*4,'decision_time_s':23+i*4,
            'feature_available_time_s':23+i*4-.005,'preprocessing':{'online_causality_verified':True},'annotation_review_required':False}


def prepared_fixture(root):
    root=Path(root);(root/'arrays').mkdir();assignments=c.canonical_subject_split()
    c.save_json(root/'subject_split.json',{'assignments':assignments});c.save_json(root/'metadata_protocol.json',{})
    c.save_json(root/'preprocessing_tests_v2.json',{'status':'passed','failures':0,'errors':0,'prep_source_sha256':'f'*64,
        'future_changes_prefix_invariance_native_rates':[128,160],'independent_MNE_calibration_fixture':True})
    c.save_json(root/'final_QC_decision.json',{'val_waveform_reads_at_decision':0,'test_waveform_reads_at_decision':0,'model_performance_used':False})
    c.save_json(root/'preprocessing_protocol.json',{'status':'frozen_before_val_test_signal_processing','subject_split_sha256':c.sha256(root/'subject_split.json'),
        'metadata_protocol_sha256':c.sha256(root/'metadata_protocol.json'),'source_code_sha256':'f'*64,
        'tests_receipt_sha256':c.sha256(root/'preprocessing_tests_v2.json'),'train_only_QC_decision_sha256':c.sha256(root/'final_QC_decision.json'),
        'QC':{'maximum_absolute_V':.0005,'minimum_channel_population_std_V':1e-8,'finite_required':True}})
    main=[];artifacts={};controls={}
    def timing(r,start):
        a=int(round(start*200));b=a+600;last=((b-1)*160)//200
        return {'window_start_s':a/200,'window_end_s':b/200,'sample_start':a,'sample_stop':b,'decision_time_s':b/200,
            'feature_available_time_s':(b-1)/200,'causal_history':{'source_recording_id':r['recording_id'],'native_sample_start':0,'native_sample_stop':last+1,'last_sample_time_s':last/160}}
    for split in ('train','val','test'):
        subject=next(k for k,v in assignments.items() if v==split)
        mainpath=f'arrays/{split}.npy';controlpath=f'arrays/{split}_control.npy'
        for name,ledger in [(mainpath,artifacts),(controlpath,controls)]:
            np.save(root/name,np.ones((2,64,600),np.float32)*1e-5)
            ledger[name]={'sha256':c.sha256(root/name),'bytes':(root/name).stat().st_size,'shape':[2,64,600],'dtype':'float32'}
        for i in range(2):
            r=row(i,split,subject,label=('left_fist','right_fist')[i]);r['subject_group']=subject;r['native_sample_rate_hz']=160
            r.update(timing(r,20+i*4));r.update(event_onset_s=19+i*4,event_offset_s=23.25+i*4,eeg_path=mainpath,array_sha256=artifacts[mainpath]['sha256'])
            r['preprocessing'].update(future_samples_used=False,protocol_sha256=c.sha256(root/'preprocessing_protocol.json'),metadata_protocol_sha256=c.sha256(root/'metadata_protocol.json'))
            r['time_shift_control']={'status':'available','shift_seconds':-10,'eeg_path':controlpath,'eeg_index':i,'source_eeg_sha256':r['source_eeg_sha256'],
                'eeg_payload_sha256':r['eeg_payload_sha256'],'source_recording_id':r['recording_id'],'array_sha256':controls[controlpath]['sha256'],**timing(r,10+i*4)}
            main.append(r)
    c.save_json(root/'artifacts.json',artifacts);c.save_json(root/'control_artifacts.json',controls)
    (root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in main))
    fields={'manifest.jsonl':'prepared_manifest_sha256','artifacts.json':'artifacts_sha256','control_artifacts.json':'control_artifacts_sha256',
        'subject_split.json':'subject_split_sha256','metadata_protocol.json':'metadata_protocol_sha256','preprocessing_protocol.json':'preprocessing_protocol_sha256',
        'preprocessing_tests_v2.json':'preprocessing_tests_sha256','final_QC_decision.json':'final_QC_decision_sha256'}
    seal={v:c.sha256(root/k) for k,v in fields.items()}
    return main,seal


class LeakageTests(unittest.TestCase):
    def records(self):return [row(split=s) for s in ('train','val','test')]
    def test_correct_schema(self):self.assertEqual(c.validate_rows(self.records())['records'],3)
    def test_subject_or_payload_cross_split_rejected(self):
        for field in ('subject','source_eeg_sha256','eeg_payload_sha256'):
            r=self.records();r[1][field]=r[0][field]
            with self.subTest(field=field),self.assertRaises(ValueError):c.validate_rows(r)
    def test_fractional_and_boolean_indices_rejected(self):
        for value in (.3,True,-1):
            r=self.records();r[0]['eeg_index']=value
            with self.subTest(value=value),self.assertRaises(ValueError):c.validate_rows(r)
    def test_weak_gold_future_or_unverified_processing_rejected(self):
        for field,value in [('label_source','ai_pseudo'),('feature_available_time_s',24),('annotation_review_required',True),('preprocessing',{'online_causality_verified':False})]:
            r=self.records();r[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):c.validate_rows(r)
    def test_overlap_is_rejected(self):
        r=self.records();extra=row(1);extra['window_start_s']=21;extra['window_end_s']=24;extra['decision_time_s']=24;r.append(extra)
        with self.assertRaises(ValueError):c.validate_rows(r)
    def test_shuffle_is_within_recording_derangement(self):
        rows=[row(i,split='test',subject=f's{g}') for g in range(2) for i in range(3)]
        for g in range(2):
            for i in range(3):rows[g*3+i]['id']=f's{g}-{i}'
        mapping,blocked=c.shuffle_mapping(rows,17);self.assertFalse(blocked)
        for r in rows:self.assertNotEqual(r['id'],mapping[r['id']]['id']);self.assertEqual(r['recording_id'],mapping[r['id']]['recording_id'])
    def test_shift_cannot_impersonate_original_or_other_raw(self):
        r=row();ref={'status':'available','shift_seconds':-10.,'source_eeg_sha256':r['source_eeg_sha256'],'source_recording_id':r['recording_id'],
            'window_start_s':10.,'window_end_s':13.,'sample_start':2000,'sample_stop':2600,'eeg_path':'shift.npy','eeg_index':0}
        artifacts={'shift.npy':{'shape':[1,64,600]}};c.validate_shift(r,ref,artifacts)
        for field,value in [('source_eeg_sha256','f'*64),('window_start_s',20.),('sample_stop',2601),('eeg_index',True)]:
            z=dict(ref);z[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):c.validate_shift(r,z,artifacts)


class MetricTests(unittest.TestCase):
    def test_known_metrics_and_abstention(self):
        m=c.metrics([0,0,1,1],[[.9,.1],[.6,.4],[.6,.4],[.1,.9]])
        self.assertAlmostEqual(m['balanced_accuracy'],.75);self.assertAlmostEqual(m['coverage'],.5)
        self.assertAlmostEqual(m['strict_balanced_accuracy'],.5);self.assertAlmostEqual(m['auroc'],.875)
    def test_probabilities_and_singleclass(self):
        self.assertIsNone(c.metrics([0],[[.5,.5]])['auroc'])
        for p in ({'a':.1,'b':.1},{'a':float('nan'),'b':.4}):
            with self.assertRaises(ValueError):c.checked_probabilities(p,['a','b'])
    def test_control_missing_fixed_paired_id_is_rejected(self):
        base=[]
        for i in range(4):
            z=row(i,split='test',subject=f'p{i//2}',label=('left_fist','right_fist')[i%2]);z.update(condition='fused',true_label=z['label'],probabilities={'left_fist':.5,'right_fist':.5});base.append(z)
        controls=[dict(r,condition='eeg_time_shifted') for r in base[:-1]]
        with self.assertRaises(ValueError):c.participant_bootstrap(base+controls,20,1,{r['id'] for r in base})
    def test_paired_identical_model_has_zero_width_difference(self):
        values=[]
        for i in range(4):
            r=row(i,split='test',subject=f'p{i//2}',label=('left_fist','right_fist')[i%2]);r.update(condition='fused',true_label=r['label'],probabilities={'left_fist':.6,'right_fist':.4});values.append(r)
        result=c.participant_bootstrap(values+[dict(r,condition='eeg_shuffled') for r in values],20,1,{r['id'] for r in values})
        intervals=result['comparisons']['left_right'][0]['intervals']
        self.assertTrue(all(v['lower_95']==v['upper_95']==0 for v in intervals.values()))


class IndependentAuditTests(unittest.TestCase):
    def test_generic_runner_completes_exact_five_conditions_and_refuses_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,seal=prepared_fixture(root)
            for name in ('idle_artifacts.json','idle_metadata_protocol.json'):c.save_json(root/name,{})
            (root/'idle_manifest.jsonl').write_text('');(root/'paired_control_subset.jsonl').write_text(''.join(json.dumps({'id':r['id']})+'\n' for r in rows))
            for name,key in [('idle_manifest.jsonl','idle_manifest_sha256'),('idle_artifacts.json','idle_artifacts_sha256'),('idle_metadata_protocol.json','idle_metadata_protocol_sha256'),('paired_control_subset.jsonl','paired_control_subset_sha256')]:seal[key]=c.sha256(root/name)
            run=root/'run';run.mkdir();(run/'best.pt').write_bytes(b'synthetic fixture checkpoint identity only')
            checkpoint=c.sha256(run/'best.pt')
            c.save_json(run/'training_membership.json',{'manifest_sha256':seal['prepared_manifest_sha256'],'test_used_for_tuning':False,
                **{key:[r['id'] for r in rows if r['split']==split] for split,key in [('train','training_ids'),('val','validation_ids'),('test','test_ids')]}})
            plugin=root/'fixture_model.py'
            plugin.write_text('import benchmark_common as c\nclass Runtime:\n checkpoint_sha256='+repr(checkpoint)+'\n def predict_batch(self,eeg,task_families,condition):\n  assert eeg.shape[1:]==(64,600)\n  return [dict(task_family=t,predicted_label=c.TASKS[t][0],probabilities=dict(zip(c.TASKS[t],[.6,.4])),latency_ms=1.) for t in task_families]\ndef load_runtime(*a,**k):return Runtime()\n')
            source=Path(__file__).parent
            seal.update(analysis_plan_sha256=c.sha256(source/'protocol.benchmark.v1.json'),budget={'gpu_memory_budget_gib':21,'absolute_deadline_epoch':time.time()+120},
                source_sha256={p.name:c.sha256(p) for p in source.glob('*.py')})
            seal['source_sha256'][plugin.name]=c.sha256(plugin);c.save_json(root/'seal.json',seal)
            result=evaluator.evaluate(root,root/'seal.json',c.sha256(root/'seal.json'),plugin,run,checkpoint,root/'evaluation',max_seconds=60)
            self.assertEqual([result['conditions'][k]['n'] for k in c.CONDITIONS],[2]*5)
            self.assertEqual(result['metadata_paired_subset_records'],2);self.assertFalse(result['goal_validated'])
            self.assertEqual(result['prediction_identity_audit']['status'],'passed')
            self.assertEqual(result['idle_diagnostic']['conditions']['left_right']['status'],'blocked_no_eligible_T0_windows')
            with self.assertRaisesRegex(ValueError,'already consumed'):
                evaluator.evaluate(root,root/'seal.json',c.sha256(root/'seal.json'),plugin,run,checkpoint,root/'second',max_seconds=60)
    def test_prepared_checks_original_subject_split_and_distinct_control_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows,seal=prepared_fixture(tmp);root=Path(tmp)
            checked,_,_=c.verify_prepared(root,seal);self.assertEqual(len(checked),6)
            changed=copy.deepcopy(rows)
            for r in changed:
                if r['split']=='train':r['split']='val'
                elif r['split']=='val':r['split']='train'
            (root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in changed));seal['prepared_manifest_sha256']=c.sha256(root/'manifest.jsonl')
            with self.assertRaisesRegex(ValueError,'frozen split'):c.verify_prepared(root,seal)
            changed=copy.deepcopy(rows);changed[0]['eeg_path']=changed[0]['time_shift_control']['eeg_path']
            (root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in changed));seal['prepared_manifest_sha256']=c.sha256(root/'manifest.jsonl')
            with self.assertRaises(ValueError):c.verify_prepared(root,seal)
    def test_prediction_ids_true_labels_and_shuffle_cannot_be_changed(self):
        rows=[]
        for i in range(2):
            r=row(i,split='test');r['time_shift_control']={'status':'blocked','reason':'prior_record_boundary'};rows.append(r)
        mapping,_=c.shuffle_mapping(rows,20261122);predictions=[]
        for condition in c.CONDITIONS[:-1]:
            for r in rows:
                proof={'method':'original_EEG'}
                if condition=='context_only':proof={'method':'EEG_removed'}
                if condition=='eeg_shuffled':proof={'method':'within_recording_derangement','eeg_source_id':mapping[r['id']]['id'],'original_eeg_id':r['id'],'fixed_points':0}
                predictions.append(evaluator.prediction_row(r,{'predicted_label':'left_fist','probabilities':{'left_fist':.7,'right_fist':.3}},condition,'f'*64,proof))
        self.assertEqual(audit.audit_predictions(rows,predictions,'f'*64)['status'],'passed')
        for field,value in [('true_label','right_fist'),('subject','replacement'),('checkpoint_sha256','e'*64)]:
            changed=copy.deepcopy(predictions);changed[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):audit.audit_predictions(rows,changed,'f'*64)
        changed=copy.deepcopy(predictions);changed[-1]['perturbation']['eeg_source_id']=changed[-1]['id']
        with self.assertRaises(ValueError):audit.audit_predictions(rows,changed,'f'*64)
        with self.assertRaises(ValueError):audit.audit_predictions(rows,predictions[:-1],'f'*64)
    def test_causal_grid_future_native_and_compensated_availability_rejected(self):
        r=row();r.update(sample_start=4000,sample_stop=4600,native_sample_rate_hz=160,event_onset_s=19.,event_offset_s=23.25)
        r['preprocessing']['future_samples_used']=False
        r['causal_history']={'source_recording_id':r['recording_id'],'native_sample_start':0,'native_sample_stop':3680,'last_sample_time_s':3679/160}
        c.validate_causal_grid(r,r)
        changed=copy.deepcopy(r);changed['causal_history']['native_sample_stop']+=1
        with self.assertRaises(ValueError):c.validate_causal_grid(changed,changed)
        changed=copy.deepcopy(r);changed['feature_available_time_s']-=.01
        with self.assertRaises(ValueError):c.validate_causal_grid(changed,changed)
    def test_original_hash_split_is_fixed_not_arbitrary_partition(self):
        s=c.canonical_subject_split();self.assertEqual(len(s),109)
        self.assertEqual([list(s.values()).count(k) for k in ('train','val','test')],[76,16,17])
    def test_idle_is_diagnostic_not_binary_gold(self):
        report=idle.summarize_idle([{'task_family':'left_right','dataset':'f','subject':'s1','accepted_as_action':True},{'task_family':'left_right','dataset':'f','subject':'s2','accepted_as_action':False}])
        self.assertEqual(report['conditions']['left_right']['fraction_cued_rest_accepted_as_binary_action'],.5)
        self.assertIsNone(report['false_actions_per_minute']);self.assertFalse(report['natural_idle_ground_truth'])


class CSPTests(unittest.TestCase):
    def test_csp_spatial_variance_signal_is_learned(self):
        rng=np.random.default_rng(20);train=[];y=[]
        for i in range(40):
            label=i%2;x=rng.normal(size=(4,300));x[label]*=4;train.append(x);y.append(label)
        filters=b.fit_filters(np.stack([b.trial_covariance(x) for x in train]),y)
        x=np.stack([b.features(z,filters[:2]) for z in train]);lda=b.LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(x,y)
        correct=0
        for i in range(20):
            label=i%2;z=rng.normal(size=(4,300));z[label]*=4;correct+=int(lda.predict(b.features(z,filters[:2])[None])[0]==label)
        self.assertGreaterEqual(correct,19)
    def test_constant_input_not_valid_training_covariance(self):
        with self.assertRaises(ValueError):b.trial_covariance(np.ones((4,100)))


class HarnessTests(unittest.TestCase):
    def test_context_does_not_accept_or_contain_gold_or_record_id(self):
        envs=h.environments('private_trial_id','left_right')
        self.assertEqual(len(envs),4);self.assertEqual(len({x['environment_sha256'] for x in envs}),4)
        for e in envs:
            prompt=h.router_prompt(e['environment'],'left_fist')
            self.assertNotIn('private_trial_id',prompt)
            for forbidden in ('gold','true_label','subject','cue_time'):self.assertNotIn(forbidden,prompt)
            altered=copy.deepcopy(e['environment']);altered['gold']='left_fist'
            with self.assertRaises(ValueError):h.router_prompt(altered,'left_fist')
    def test_ids_and_positions_are_counterbalanced(self):
        e=h.environments('a','left_right');targets=[h.target_for(x['environment'],'left_fist') for x in e]
        self.assertEqual(targets.count('X'),2);self.assertEqual(targets.count('Y'),2)
        self.assertEqual(len(h.required_prompts()),16)
    def test_invalid_text_and_abstain_never_change_state(self):
        env=h.Environment(h.environments('a','left_right')[0]['environment'])
        for text in ['X because it is correct','XY','x','X.','{"target":"X"}',None]:
            self.assertEqual(env.step(text)['status'],'invalid_action');self.assertIsNone(env.state['selected_target'])
        self.assertEqual(env.step('X',True)['status'],'abstained');self.assertEqual(env.state['transitions'],0)
    def test_oracle_and_error_decomposition(self):
        z=row(split='test');z.update(condition='fused',true_label='left_fist',probabilities={'left_fist':.1,'right_fist':.9})
        cache={key:{'prompt':v['prompt'],'text':h.target_for(v['environment'],v['decoded_control'])} for key,v in h.required_prompts().items()}
        traces,report=h.run_recorded([z]+h.gold_rows([z]),cache)
        self.assertEqual(report['conditions']['fused::left_right']['task_success_rate'],0)
        self.assertEqual(report['conditions']['fused::left_right']['router_error_count_among_nonabstained'],0)
        self.assertEqual(report['conditions']['gold_class_upper_bound::left_right']['task_success_rate'],1)
        self.assertTrue(all(r['final_state']['transitions']==1 for r in traces))
    def test_actual_generation_cache_is16_calls_and_replay_does_not_call_model(self):
        class Tokenizer:
            def encode(self,x,add_special_tokens=False):return [1 if x.strip()=='X' else 2]
        class Runtime:
            tokenizer=Tokenizer();checkpoint_sha256='f'*64
            def __init__(self):self.route_records=[]
            def route_text(self,prompt,max_new_tokens=8):
                self.route_records.append({'EEG_prefix_used':False,'new_parameter_updates':0,'do_sample':False})
                return 'X'
        runtime=Runtime()
        with tempfile.TemporaryDirectory() as tmp:
            value=he.collect_routing(runtime,Path(tmp)/'cache.json','f'*64,{'analysis_plan_sha256':'a'*64,'source_sha256':{}},time.monotonic()+10)
            self.assertEqual(value['actual_LLM_greedy_inference_calls'],16);self.assertEqual(len(runtime.route_records),16)
            z=row(split='test');z.update(condition='fused',true_label='left_fist',probabilities={'left_fist':.9,'right_fist':.1})
            h.run_recorded([z],value['cache']);self.assertEqual(len(runtime.route_records),16)
    def test_counterbalance_does_not_turn_episodes_into_independent_EEG_records(self):
        z=row(split='test');z.update(condition='fused',true_label='left_fist',probabilities={'left_fist':.9,'right_fist':.1})
        cache={key:{'prompt':v['prompt'],'text':'X'} for key,v in h.required_prompts().items()}
        _,r=h.run_recorded([z],cache);v=r['conditions']['fused::left_right']
        self.assertEqual(v['episodes'],4);self.assertEqual(v['unique_EEG_records'],1);self.assertEqual(v['task_success_rate'],.5)


if __name__=='__main__':unittest.main()
