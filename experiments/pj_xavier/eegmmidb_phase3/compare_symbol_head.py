"""Predeclared numerical identity guard for the optional two-row output head."""
from __future__ import annotations
import argparse,copy,json,sys,time
from pathlib import Path
if not __package__:sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from torch.nn import functional as F
from eegmmidb_phase3.common import require,sha256,write_json,seed_everything,apply_budget,hardware,memory_sample
from eegmmidb_phase3.model import FusedDecoder,load_backbone,static_prompt
from eegmmidb_phase2.model import FusedDecoder as FullDecoder
from eegmmidb_phase2.data_contract import EEGStore,load_corpus,TASKS
ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'protocol.head_equivalence.v1.json'

def gradients(model,component):
 vals=[]
 for name,param in getattr(model,component).named_parameters():
  require(param.grad is not None,'Missing '+component+' gradient: '+name)
  vals.append(param.grad.detach().float().cpu().reshape(-1))
 result=torch.cat(vals)
 require(bool(torch.isfinite(result).all()) and float(result.norm())>0,'Nonfinite/zero gradient')
 return result

def compare(base,eeg,prompts,targets,limits):
 full=FullDecoder(base).to(eeg.device)
 optimized=FusedDecoder(base).to(eeg.device)
 optimized.encoder.load_state_dict(copy.deepcopy(full.encoder.state_dict()),strict=True)
 optimized.projector.load_state_dict(copy.deepcopy(full.projector.state_dict()),strict=True)
 outputs=[]
 for model in (full,optimized):
  model.train();model.zero_grad(set_to_none=True);seed_everything(20261122)
  logits=model(eeg,prompts,'fused');loss=F.cross_entropy(logits,targets)
  require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss)),'Nonfinite numerical guard')
  loss.backward()
  require(all(not p.requires_grad and p.grad is None for p in base.parameters()),'LLM was not frozen')
  outputs.append({'logits':logits.detach().float().cpu(),'loss':float(loss.detach()),
                  'encoder':gradients(model,'encoder'),'projector':gradients(model,'projector')})
 a,b=outputs
 errors={'logits_max_absolute_error':float((a['logits']-b['logits']).abs().max()),
         'loss_absolute_error':abs(a['loss']-b['loss'])}
 diagnostics={}
 for key in ('encoder','projector'):
  delta=(a[key]-b[key]);reference=float(a[key].norm())
  errors[key+'_gradient_relative_L2_error']=float(delta.norm())/max(reference,1e-30)
  diagnostics[key]={'reference_gradient_L2':reference,'candidate_gradient_L2':float(b[key].norm()),
                    'max_absolute_error':float(delta.abs().max()),'parameter_gradient_values':len(delta)}
 checks={key:errors[key]<=float(limit) for key,limit in limits.items() if key!='fixture'}
 return {'status':'passed' if all(checks.values()) else 'failed','errors':errors,'checks':checks,
         'gradient_diagnostics':diagnostics,'reference_logits':a['logits'].tolist(),
         'candidate_logits':b['logits'].tolist(),'reference_loss':a['loss'],'candidate_loss':b['loss'],
         'frozen_LLM_gradients_absent':True,'optimizer_steps':0,'bitwise_identity_claimed':False}

def run(args):
 protocol=json.loads(PROTOCOL.read_text());require(not Path(args.output).exists(),'Numerical guard receipt already exists')
 torch.set_num_threads(2);seed_everything(protocol['seed']);started=time.monotonic()
 if args.scope=='CPU_FP32':
  from transformers import LlamaConfig,LlamaForCausalLM
  device=torch.device('cpu')
  cfg=LlamaConfig(vocab_size=512,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                  num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=256,
                  attention_dropout=0.0,pad_token_id=0,bos_token_id=1,eos_token_id=2)
  cfg._attn_implementation='eager';base=LlamaForCausalLM(cfg).float()
  eeg=torch.randn(3,64,600)*2e-5;prompts=[[1,3,4,7],[1,8,9],[1,3,4,5,6]];targets=torch.tensor([0,1,0])
  source={'synthetic_fixture':True,'not_pretrained_model_performance':True}
 else:
  require(torch.cuda.is_available(),'Actual CUDA required for L4 BF16 guard')
  require('L4' in torch.cuda.get_device_name(0),'Registered actual L4 device required')
  device=torch.device('cuda');apply_budget(device,21)
  require(args.prepared is not None,'Actual train manifest required')
  rows,_,audit=load_corpus(Path(args.prepared)/'manifest.jsonl',Path(args.prepared)/'artifacts.json')
  train=sorted((r for r in rows if r['split']=='train'),key=lambda r:r['id'])
  chosen=[next(r for r in train if r['task_family']==t) for t in ('left_right','fists_feet')]
  chosen.append(next(r for r in train if r['id'] not in {x['id'] for x in chosen}))
  store=EEGStore();eeg=torch.from_numpy(np.stack([store.get(r) for r in chosen])).to(device)
  base,tokenizer,evidence=load_backbone(device,torch.bfloat16,local_files_only=True)
  prompts=[tokenizer.encode(static_prompt(r['task_family']),add_special_tokens=True) for r in chosen]
  targets=torch.tensor([TASKS[r['task_family']].index(r['label']) for r in chosen],device=device)
  source={'synthetic_fixture':False,'training_ids':[r['id'] for r in chosen],
          'prepared_manifest_sha256':audit['manifest_sha256'],'artifacts_sha256':audit['artifacts_sha256'],
          'backbone':evidence,'validation_forward_calls':0,'test_forward_calls':0}
 value=compare(base,eeg,prompts,targets,protocol[args.scope])
 value.update(schema='xavier-phase3-restricted-head-comparison-v1',scope=args.scope,source=source,
              protocol_sha256=sha256(PROTOCOL),comparison_source_sha256=sha256(__file__),
              candidate_source_sha256=sha256(ROOT/'model.py'),hardware=hardware(device),
              memory=memory_sample(device),elapsed_seconds=time.monotonic()-started)
 write_json(args.output,value);print(json.dumps(value,allow_nan=False),flush=True)
 return value

def combine(cpu_path,l4_path,output):
 require(not Path(output).exists(),'Combined numerical receipt already exists')
 receipts={name:json.loads(Path(path).read_text()) for name,path in [('CPU_FP32',cpu_path),('L4_BF16',l4_path)]}
 for name,value in receipts.items():
  require(value['scope']==name and value['protocol_sha256']==sha256(PROTOCOL),'Numerical receipt protocol/scope differs')
  require(value['comparison_source_sha256']==sha256(__file__) and value['candidate_source_sha256']==sha256(ROOT/'model.py'),'Numerical implementation differs')
  require(value['status'] in ('passed','failed') and value['optimizer_steps']==0,'Invalid numerical comparison')
 result={'schema':'xavier-phase3-restricted-head-combined-v1','protocol_sha256':sha256(PROTOCOL),
         'comparison_source_sha256':sha256(__file__),'candidate_source_sha256':sha256(ROOT/'model.py'),
         'receipts':{name:{'sha256':sha256(path),'contents':receipts[name]} for name,path in [('CPU_FP32',cpu_path),('L4_BF16',l4_path)]},
         'optimized_symbol_head_admitted':all(r['status']=='passed' for r in receipts.values()),
         'fallback':'phase2 full-vocabulary head if either guard fails','bitwise_identity_claimed':False}
 write_json(output,result);return result

def validate_combined_receipt(path):
 value=json.loads(Path(path).read_text())
 require(value['schema']=='xavier-phase3-restricted-head-combined-v1','Wrong combined numerical schema')
 require(value['protocol_sha256']==sha256(PROTOCOL) and value['comparison_source_sha256']==sha256(__file__) and value['candidate_source_sha256']==sha256(ROOT/'model.py'),'Head numerical provenance changed')
 protocol=json.loads(PROTOCOL.read_text());passed=[]
 require(set(value['receipts'])=={'CPU_FP32','L4_BF16'},'Both numerical scopes required')
 for scope,receipt in value['receipts'].items():
  record=receipt['contents'];require(record['scope']==scope and record['optimizer_steps']==0,'Scope/update mismatch')
  require(record['protocol_sha256']==value['protocol_sha256'] and record['candidate_source_sha256']==value['candidate_source_sha256'],'Receipt/model mismatch')
  checks={key:record['errors'][key]<=threshold for key,threshold in protocol[scope].items() if key!='fixture'}
  require(checks==record['checks'],'Numerical checks do not match actual errors')
  require(record['status']==('passed' if all(checks.values()) else 'failed'),'Numerical result differs from thresholds')
  passed.append(all(checks.values()))
 require(value['optimized_symbol_head_admitted']==all(passed),'Improper head adoption')
 return value

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 r=sub.add_parser('run');r.add_argument('--scope',choices=('CPU_FP32','L4_BF16'),required=True);r.add_argument('--prepared');r.add_argument('--output',required=True)
 c=sub.add_parser('combine');c.add_argument('--cpu',required=True);c.add_argument('--l4',required=True);c.add_argument('--output',required=True)
 a=p.parse_args()
 if a.command=='run':run(a)
 else:print(json.dumps(combine(a.cpu,a.l4,a.output)))
