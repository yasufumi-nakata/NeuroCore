"""Focused CPU tests for phase3 masking, lineage and exact checkpoint recovery."""
from __future__ import annotations
import argparse,copy,hashlib,json,sys,tempfile,time,unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from torch.nn import functional as F
from transformers import LlamaConfig,LlamaForCausalLM
from eegmmidb_phase3 import common,data_ssl,ssl_model,train_ssl,model,runtime,train_supervised
from eegmmidb_phase3.compare_symbol_head import compare

torch.set_num_threads(2)

def fixture(root):
 root=Path(root);rows=[];artifacts={};allow=[];assignment={}
 channels=[f'C{i}' for i in range(64)]
 for part in ('train','val'):
  group='group_'+part;source=hashlib.sha256(part.encode()).hexdigest();payload=hashlib.sha256(('payload'+part).encode()).hexdigest()
  name=part+'.npy';array=np.random.default_rng(1 if part=='train' else 2).normal(0,2e-5,(4,64,600)).astype(np.float32)
  np.save(root/name,array);seal={'sha256':common.sha256(root/name),'bytes':(root/name).stat().st_size,'shape':list(array.shape),'dtype':'float32'};artifacts[name]=seal
  original={'subject_group':group,'source_recording_id':'record_'+part,'split':part,'source_eeg_sha256':source,'eeg_payload_sha256':payload,'channels':channels}
  allow.append(original);assignment[group]=part
  for i in range(4):
   rows.append({**original,'id':part+str(i),'source_sha256':source,'sample_rate_hz':200,'unit':'V','eeg_path':name,'eeg_index':i,'array_sha256':seal['sha256'],
                'window_start_s':5.+i*3,'window_end_s':8.+i*3,'decision_time_s':8.+i*3,'feature_available_time_s':8.+i*3,
                'preprocessing':{'online_causality_verified':True}})
 assignment['group_test']='test'
 write_rows(root/'manifest.jsonl',rows);write_rows(root/'source_allowlist.jsonl',allow)
 write_rows(root/'source_denylist.jsonl',[{'source_eeg_sha256':'f'*64,'eeg_payload_sha256':'e'*64,'subject_group':'group_test'}])
 common.write_json(root/'subject_split.json',{'assignments':assignment});common.write_json(root/'artifacts.json',artifacts)
 for name in ('metadata_protocol.json','runtime_seal.json','source_records.jsonl','corpus_summary.json','completion.json','resource_receipt.json'):
  common.write_json(root/name,{'fixture':True})
 return rows,artifacts

def write_rows(path,rows):Path(path).write_text(''.join(json.dumps(r)+'\n' for r in rows))

class SSLInputTests(unittest.TestCase):
 def test_all_slot_fixture_units_shape_and_interval_union(self):
  with tempfile.TemporaryDirectory() as d:
   fixture(d);rows,_,audit=data_ssl.load_corpus(d)
   self.assertEqual(audit['window_counts'],{'train':4,'val':4})
   self.assertEqual(audit['source_interval_union_seconds'],{'train':12.,'val':12.})
   self.assertEqual(data_ssl.SSLStore().get(rows[0]).shape,(64,600))
 def test_duplicate_fractional_bool_negative_or_out_of_range_index_fails(self):
  for index in (0,1.5,True,-1,4):
   with self.subTest(index=index),tempfile.TemporaryDirectory() as d:
    rows,_=fixture(d);rows[1]['eeg_index']=index;write_rows(Path(d)/'manifest.jsonl',rows)
    with self.assertRaises(ValueError):data_ssl.load_corpus(d)
 def test_missing_slot_and_overlapping_original_time_fail(self):
  for mutation in ('missing','overlap'):
   with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
    rows,_=fixture(d)
    if mutation=='missing':rows.pop(1)
    else:
     for k in ('window_start_s','window_end_s','decision_time_s','feature_available_time_s'):rows[1][k]=rows[0][k]
    write_rows(Path(d)/'manifest.jsonl',rows)
    with self.assertRaises(ValueError):data_ssl.load_corpus(d)
 def test_labels_or_test_people_cannot_enter_ssl(self):
  for mutation in ('label','task_family','testgroup','testsplit','deny'):
   with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
    rows,_=fixture(d)
    if mutation in ('label','task_family'):rows[0][mutation]='leaked_target'
    if mutation=='testgroup':rows[0]['subject_group']='group_test'
    if mutation=='testsplit':rows[0]['split']='test'
    if mutation=='deny':rows[0]['source_eeg_sha256']='f'*64;rows[0]['source_sha256']='f'*64
    write_rows(Path(d)/'manifest.jsonl',rows)
    with self.assertRaises(ValueError):data_ssl.load_corpus(d)
 def test_array_byte_replacement_and_future_availability_fail(self):
  for mutation in ('bytes','future'):
   with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
    rows,_=fixture(d)
    if mutation=='bytes':
     value=np.load(Path(d)/'train.npy');value[0,0,0]+=1e-6;np.save(Path(d)/'train.npy',value)
    else:rows[0]['feature_available_time_s']+=1;write_rows(Path(d)/'manifest.jsonl',rows)
    with self.assertRaises(ValueError):data_ssl.load_corpus(d)
 def test_test_guard_precedes_any_array_access(self):
  with mock.patch.object(data_ssl.EEGStore,'get') as getter:
   with self.assertRaises(ValueError):data_ssl.SSLStore().get({'split':'test'})
   getter.assert_not_called()

class MaskingTests(unittest.TestCase):
 def test_val_masks_fixed_training_masks_epoch_specific(self):
  a=ssl_model.patch_mask('a','val',0);b=ssl_model.patch_mask('a','val',19)
  self.assertTrue(torch.equal(a,b));self.assertEqual(int(a.sum()),19200)
  self.assertFalse(torch.equal(ssl_model.patch_mask('a','train',0),ssl_model.patch_mask('a','train',1)))
  self.assertFalse(torch.equal(a,ssl_model.patch_mask('b','val')))
 def test_masked_values_not_in_input_and_unmasked_output_not_in_loss(self):
  eeg=torch.randn(2,64,600)*1e-5;masks=torch.stack([ssl_model.patch_mask(i,'train') for i in ('a','b')])
  changed=eeg.clone();changed[masks]=99
  self.assertTrue(torch.equal(ssl_model.make_masked_input(eeg,masks),ssl_model.make_masked_input(changed,masks)))
  pred=torch.zeros_like(eeg,requires_grad=True);loss=ssl_model.losses_per_window(pred,eeg,masks).mean();loss.backward()
  self.assertEqual(float(pred.grad[~masks].abs().sum()),0.)
  huge=pred.detach().clone();huge[~masks]=float('inf')
  torch.testing.assert_close(ssl_model.losses_per_window(huge,eeg,masks).mean(),loss)
 def test_encoder_initial_tensor_matches_seeded_original_and_gradient_exists(self):
  common.seed_everything(20261122);reference=model.ScratchDecoder().encoder
  common.seed_everything(20261122);candidate=ssl_model.MaskedEEGModel().train()
  self.assertEqual(common.tensor_sha(reference.state_dict()),common.tensor_sha(candidate.encoder.state_dict()))
  x=torch.randn(2,64,600)*2e-5;masks=torch.stack([ssl_model.patch_mask(k,'train') for k in ('a','b')])
  pred=candidate(ssl_model.make_masked_input(x,masks));self.assertEqual(pred.shape,x.shape)
  ssl_model.losses_per_window(pred,x,masks).mean().backward()
  self.assertGreater(float(candidate.encoder.temporal.convolution.weight.grad.abs().sum()),0)
  self.assertGreater(float(candidate.decoder[-1].weight.grad.abs().sum()),0)
 def test_validation_function_balances_subjects_not_windows(self):
  rows=[{'id':'small','split':'val','subject_group':'small'}]+[{'id':'large'+str(i),'split':'val','subject_group':'large'} for i in range(3)]
  net=mock.Mock();net.return_value=torch.zeros(4,64,600)
  with mock.patch.object(train_ssl,'tensor_batch',return_value=(torch.zeros(4,64,600),torch.ones(4,64,600,dtype=torch.bool))), mock.patch.object(train_ssl,'make_masked_input',return_value=torch.zeros(4,64,600)), mock.patch.object(train_ssl,'losses_per_window',return_value=torch.tensor([2.,10.,10.,10.])):
   metrics,_=train_ssl.validate(net,rows,None,torch.device('cpu'),4,time.monotonic(),60,time.time()+60,21)
  self.assertEqual(metrics['mean_subject_masked_MSE'],6.)
  self.assertEqual(metrics['mean_window_masked_MSE'],8.)

class CheckpointTests(unittest.TestCase):
 def test_weights_only_and_exact_optimizer_rng_roundtrip(self):
  with tempfile.TemporaryDirectory() as d:
   common.seed_everything(7);first=ssl_model.MaskedEEGModel().train();opt=torch.optim.AdamW(first.parameters(),lr=.001)
   x=torch.randn(2,64,600)*2e-5;masks=torch.stack([ssl_model.patch_mask(k,'train') for k in ('a','b')]);inp=ssl_model.make_masked_input(x,masks)
   opt.zero_grad();ssl_model.losses_per_window(first(inp),x,masks).mean().backward();opt.step()
   path=Path(d)/'partial.pt';train_ssl.save(path,first,opt,{'next_position':2,'epoch_completed':False})
   state=torch.load(path,map_location='cpu',weights_only=True);self.assertEqual(state['next_position'],2)
   opt.zero_grad();ssl_model.losses_per_window(first(inp),x,masks).mean().backward();opt.step();expected=common.clone_cpu(first.state_dict())
   other=ssl_model.MaskedEEGModel().train();other_opt=torch.optim.AdamW(other.parameters(),lr=.001)
   other.load_state_dict(state['model']);other_opt.load_state_dict(state['optimizer']);common.restore_rng(state['rng'])
   other_opt.zero_grad();ssl_model.losses_per_window(other(inp),x,masks).mean().backward();other_opt.step()
   for key,value in other.state_dict().items():torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
 def test_initial_candidate_remains_selected_after_worse_epoch_and_seals(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);net=ssl_model.MaskedEEGModel();opt=torch.optim.AdamW(net.parameters());config={'manifest_sha256':'a'*64,'artifacts_sha256':'b'*64}
   common.write_json(root/'run_config.json',config);common.write_json(root/'training_membership.json',{'fixture':True})
   initial=common.tensor_sha(net.encoder.state_dict());state=train_ssl.state_for(config,common.sha256(root/'training_membership.json'),0,0,0,set(),0,0.,initial,.4,0,False)
   train_ssl.save(root/'initial.pt',net,opt,state);score,selected=train_ssl.select_best(root,root/'initial.pt',.4,None);self.assertTrue(selected)
   old=common.sha256(root/'best.pt');(root/'worse.pt').write_bytes(b'not-selected')
   score,selected=train_ssl.select_best(root,root/'worse.pt',.5,score);self.assertFalse(selected);self.assertEqual(common.sha256(root/'best.pt'),old)
   seal=train_ssl.seal_encoder(root);self.assertEqual(seal['selected_epoch'],0);self.assertFalse(seal['updated_from_initial'])
 def test_non_time_partial_cannot_advance_encoder(self):
  for reason in ('memory_budget','Nonfinite SSL loss','arbitrary error'):
   with self.subTest(reason=reason),tempfile.TemporaryDirectory() as d:
    root=Path(d);common.write_json(root/'run_summary.json',{'status':'partial','stop_reason':reason})
    receipt=root/'exit.json';common.write_json(receipt,{})
    with mock.patch.object(train_ssl,'seal_encoder') as export:
     with self.assertRaisesRegex(ValueError,'Only registered time-limited'):train_ssl.finalize_partial_encoder(root,receipt)
     export.assert_not_called()
 def test_time_partial_still_requires_actual_owned_normal_child_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);common.write_json(root/'run_summary.json',{'status':'partial','stop_reason':'absolute_deadline'})
   receipt=root/'exit.json';common.write_json(receipt,{'exit_code':-9,'process_returned_normally':False,'terminated_by_supervisor':True})
   with self.assertRaisesRegex(ValueError,'return normally'):train_ssl.finalize_partial_encoder(root,receipt)
 def test_small_ssl_fit_uses_only_train_val_and_seals_actual_state(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);prepared=root/'prepared';prepared.mkdir();fixture(prepared)
   budget=root/'budget.json';common.write_json(budget,{'absolute_deadline_epoch':time.time()+120,'phase_limits_seconds':{'SSL_training_and_validation':90.,'train_only_capacity':30}})
   contract_path=root/'contract.json';train_ssl.register(argparse.Namespace(prepared=str(prepared),budget=str(budget),out=str(contract_path)))
   receipt=root/'capacity.json';common.write_json(receipt,{'status':'passed','batch_size':32,'completed_optimizer_steps':50,'execution_contract_sha256':common.sha256(contract_path),'source_sha256':common.source_seals()})
   args=argparse.Namespace(prepared=str(prepared),contract=str(contract_path),run_dir=str(root/'masked_ssl_v1'),device='cpu',max_seconds=90.,absolute_deadline_epoch=time.time()+100,capacity_reports=[str(receipt)],resume=None)
   summary=train_ssl.train(args)
   self.assertEqual(summary['status'],'completed');self.assertEqual(summary['completed_epochs'],20)
   self.assertEqual(summary['unique_source_interval_union_seconds'],12.)
   origin=runtime.verify_ssl_origin(root/'masked_ssl_v1',['group_test'],['f'*64],['e'*64])
   self.assertTrue(origin['downstream_test_person_and_signal_exclusion_verified'])
   selected=torch.load(root/'masked_ssl_v1'/'selected_encoder.pt',map_location='cpu',weights_only=True)
   common.seed_everything(20261122);actual,_,_=train_supervised.build_model('scratch',torch.device('cpu'),torch.float32,True,ssl_run=root/'masked_ssl_v1',optimized_symbol_head=False)
   for key,value in actual.encoder.state_dict().items():self.assertTrue(torch.equal(value,selected[key]))
   with self.assertRaises(ValueError):runtime.verify_ssl_origin(root/'masked_ssl_v1',['group_train'],[],[])

class HeadTests(unittest.TestCase):
 def test_cpu_logits_loss_and_all_trainable_gradients_equivalent(self):
  common.seed_everything(20261122)
  cfg=LlamaConfig(vocab_size=512,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=256,pad_token_id=0)
  cfg._attn_implementation='eager';base=LlamaForCausalLM(cfg)
  limits=json.loads((Path(__file__).with_name('protocol.head_equivalence.v1.json')).read_text())['CPU_FP32']
  result=compare(base,torch.randn(3,64,600)*2e-5,[[1,3,4,5],[1,5],[1,4,5]],torch.tensor([0,1,0]),limits)
  self.assertEqual(result['status'],'passed');self.assertTrue(result['frozen_LLM_gradients_absent'])
 def test_symbol_fallback_preserves_reference_class(self):
  base=LlamaForCausalLM(LlamaConfig(vocab_size=512,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128))
  wrapped=model.FusedDecoder(base,optimized_symbol_head=False).eval()
  x=torch.randn(2,64,600)*2e-5
  with torch.no_grad():
   actual=wrapped(x,[[1,4],[1,5,6]])
   from eegmmidb_phase2.model import FusedDecoder
   expected=FusedDecoder.logits_from_ids(wrapped,x,[[1,4],[1,5,6]])
  torch.testing.assert_close(actual,expected,rtol=0,atol=0)

if __name__=='__main__':unittest.main(verbosity=2)
