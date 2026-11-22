"""Registered all-recording masked reconstruction, followed by encoder sealing."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json,math,os,random,shutil,sys,time
from pathlib import Path
if not __package__:sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from eegmmidb_phase3.common import (
 clone_cpu,require,write_json,sha256,object_hash,tensor_sha,rng_state,restore_rng,
 seed_everything,scientific_plan,source_seals,source_interval_union,
 select_device,apply_budget,hardware,memory_sample,synchronize,within_budget,
)
from eegmmidb_phase3.data_ssl import load_corpus,membership,SSLStore
from eegmmidb_phase3.ssl_model import MaskedEEGModel,patch_mask,make_masked_input,losses_per_window

SCHEMA="xavier-phase3-masked-EEG-checkpoint-v1"
CONTRACT_SCHEMA="xavier-phase3-SSL-execution-contract-v1"

class BoundedStop(RuntimeError):pass

def read(path):return json.loads(Path(path).read_text())

def save(path,model,optimizer,state):
 value={**state,"schema":SCHEMA,"model":clone_cpu(model.state_dict()),
        "optimizer":clone_cpu(optimizer.state_dict()),"rng":clone_cpu(rng_state())}
 p=Path(path);temp=p.with_suffix(".tmp");torch.save(value,temp);temp.replace(p)
 return sha256(p)

def guard(start,seconds,deadline):
 if time.monotonic()-start>=seconds:raise BoundedStop("stage_time_budget")
 if time.time()>=deadline:raise BoundedStop("absolute_deadline")

def register(args):
 root=Path(args.prepared);plan=scientific_plan();budget=read(args.budget)
 require(budget["absolute_deadline_epoch"]>time.time(),"Phase3 resource deadline has passed")
 rows,_,audit=load_corpus(root)
 require(not Path(args.out).exists(),"Execution contract already exists")
 proofs={n:sha256(root/n) for n in ("metadata_protocol.json","runtime_seal.json","source_records.jsonl","corpus_summary.json","completion.json","resource_receipt.json")}
 contract={"schema":CONTRACT_SCHEMA,"registered_epoch":time.time(),"scientific_protocol_sha256":source_seals()["protocol.scientific.v1.json"],
           "source_sha256":source_seals(),"data_audit":audit,"preparation_proofs_sha256":proofs,
           "budget_sha256":sha256(args.budget),"budget":budget,"plan":plan,"test_fit_or_forward_calls":0}
 write_json(args.out,contract)
 print(json.dumps({"status":"registered","contract_sha256":sha256(args.out),"data_audit":audit}))
 return contract

def load_execution(args):
 contract=read(args.contract)
 require(contract["schema"]==CONTRACT_SCHEMA and contract["source_sha256"]==source_seals(),"SSL source identity changed")
 require(contract["plan"]==scientific_plan(),"Scientific plan changed")
 root=Path(args.prepared)
 for name,digest in contract["preparation_proofs_sha256"].items():require(sha256(root/name)==digest,"Preparation certificate changed")
 rows,_,audit=load_corpus(root)
 require(audit==contract["data_audit"],"SSL corpus identity changed")
 return contract,rows,audit

def batches(rows,batch_size,seed,epoch,start_position=0):
 order=list(range(len(rows)));random.Random(seed+epoch).shuffle(order)
 require(0<=start_position<=len(order),"Bad restart position")
 for position in range(start_position,len(order),batch_size):
  chosen=[rows[i] for i in order[position:position+batch_size]]
  yield position,chosen

def tensor_batch(rows,store,device,split,epoch):
 require(rows and all(r["split"]==split for r in rows),"Wrong split in model input")
 eeg=torch.from_numpy(np.stack([store.get(r) for r in rows])).to(device)
 masks=torch.stack([patch_mask(r["id"],split,epoch) for r in rows]).to(device)
 return eeg,masks

@torch.inference_mode()
def validate(model,rows,store,device,batch_size,started,seconds,deadline,budget):
 require(rows and all(r["split"]=="val" for r in rows),"Only independent SSL val is permitted")
 model.eval();subjects=defaultdict(list);values=[]
 for begin in range(0,len(rows),batch_size):
  guard(started,seconds,deadline)
  batch=rows[begin:begin+batch_size]
  eeg,masks=tensor_batch(batch,store,device,"val",0)
  losses=losses_per_window(model(make_masked_input(eeg,masks)),eeg,masks).cpu().tolist()
  if not within_budget(memory_sample(device),budget,device):raise BoundedStop("memory_budget_in_validation")
  for row,value in zip(batch,losses,strict=True):
   require(math.isfinite(value),"Nonfinite validation loss")
   subjects[row["subject_group"]].append(value)
   values.append({"id":row["id"],"subject_group":row["subject_group"],"masked_MSE":value})
 means={key:float(np.mean(value)) for key,value in sorted(subjects.items())}
 return {"mean_subject_masked_MSE":float(np.mean(list(means.values()))),"mean_window_masked_MSE":float(np.mean([v["masked_MSE"] for v in values])),
         "per_subject":means,"windows":len(rows),"mask_condition":"fixed_ID_seed_same_at_initial_and_every_epoch"},values

def make_config(contract,args,batch):
 plan=contract["plan"]
 return {
  "schema":"xavier-phase3-SSL-scientific-config-v1","execution_contract_sha256":sha256(args.contract),
  "source_sha256":source_seals(),"seed":plan["seed"],"max_epochs":plan["optimization"]["max_epochs"],
  "batch_size":batch,"validation_batch_size":min(batch,512),"learning_rate":plan["optimization"]["learning_rate"],
  "weight_decay":plan["optimization"]["weight_decay"],"gradient_clip_norm":plan["optimization"]["gradient_clip_norm"],
  "max_seconds":args.max_seconds,"memory_budget_gib":21,"training_device":str(args.device),
  "precision":"float32","selection_metric":"mean_subject_masked_MSE","initial_state_is_candidate":True,
  "plan_sha256":contract["scientific_protocol_sha256"],"manifest_sha256":contract["data_audit"]["manifest_sha256"],
  "artifacts_sha256":contract["data_audit"]["artifacts_sha256"],
 }

def select_capacity(paths,contract):
 reports=[read(p) for p in paths]
 for value in reports:
  require(value["execution_contract_sha256"]==contract and value["source_sha256"]==source_seals(),"Capacity provenance differs")
 require(len({r["batch_size"] for r in reports})==len(reports),"Duplicate capacity candidate")
 require(all(r["batch_size"] in scientific_plan()["optimization"]["candidate_batches"] for r in reports),"Unregistered batch in capacity evidence")
 valid=[r for r in reports if r["status"]=="passed" and r["completed_optimizer_steps"]==50]
 require(valid,"No passing train-only capacity evidence")
 chosen=max(valid,key=lambda r:r["batch_size"])
 return chosen["batch_size"],{"rule":"largest stable registered train-only candidate",
                              "selected_batch_size":chosen["batch_size"],
                              "receipts":{str(p):sha256(p) for p in paths}}

def probe(args):
 start=time.monotonic();contract,rows,audit=load_execution(args)
 require(args.batch_size in contract["plan"]["optimization"]["candidate_batches"],"Unregistered capacity candidate")
 require(0<args.max_seconds<=contract["budget"]["phase_limits_seconds"]["train_only_capacity"] and args.absolute_deadline_epoch<=contract["budget"]["absolute_deadline_epoch"],"Capacity budget exceeds registration")
 output=Path(args.run_dir);require(not output.exists(),"Probe already exists");output.mkdir(parents=True)
 device=select_device(args.device);torch.set_num_threads(2);allocator=apply_budget(device,21)
 seed_everything(contract["plan"]["seed"]);model=MaskedEEGModel().to(device).train()
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 train=sorted((r for r in rows if r["split"]=="train"),key=lambda r:r["id"]);store=SSLStore()
 order=list(range(len(train)));random.Random(contract["plan"]["seed"]).shuffle(order)
 seen=set();steps=0;samples=[];reason=None
 try:
  for step in range(50):
   guard(start,args.max_seconds,args.absolute_deadline_epoch)
   batch=[train[order[(step*args.batch_size+i)%len(train)]] for i in range(args.batch_size)]
   eeg,masks=tensor_batch(batch,store,device,"train",0)
   optimizer.zero_grad(set_to_none=True)
   loss=losses_per_window(model(make_masked_input(eeg,masks)),eeg,masks).mean()
   require(bool(torch.isfinite(loss)),"Nonfinite masked loss")
   loss.backward()
   require(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),"Nonfinite gradient")
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
   optimizer.step();synchronize(device);steps+=1;seen.update(r["id"] for r in batch)
   sample=memory_sample(device);samples.append(sample)
   if not within_budget(sample,21,device):raise BoundedStop("memory_budget")
 except (RuntimeError,ValueError,BoundedStop) as exc:reason=str(exc)
 result={"schema":"xavier-phase3-SSL-capacity-v1","status":"passed" if steps==50 and reason is None else "failed",
         "failure_reason":reason,"batch_size":args.batch_size,"completed_optimizer_steps":steps,
         "examples_presented":steps*args.batch_size,"unique_training_ids":sorted(seen),
         "execution_contract_sha256":sha256(args.contract),"source_sha256":source_seals(),
         "elapsed_seconds":time.monotonic()-start,"memory_observations":samples,"hardware":hardware(device),
         "allocator":allocator,"training_parameters_discarded":True,"validation_and_test_forward_calls":0,
         "data_audit":audit}
 write_json(output/"capacity.json",result)
 print(json.dumps({k:result[k] for k in ("status","batch_size","completed_optimizer_steps","failure_reason","elapsed_seconds")}),flush=True)
 return result

def state_for(config,member_sha,epoch,cursor,step,seen,presentations,elapsed,initial_tensor,best_score,best_epoch,complete,resumable=True):
 return {"schema":SCHEMA,"config":config,"source_sha256":source_seals(),"membership_sha256":member_sha,
         "epoch_zero_based":epoch,"next_position":cursor,"global_step":step,
         "seen_train_ids":sorted(seen),"training_examples_presented":presentations,"elapsed_stage_seconds":elapsed,
         "initial_encoder_tensor_sha256":initial_tensor,"best_validation_score":best_score,"best_epoch":best_epoch,
         "epoch_completed":complete,"resumable":resumable}

def save_validation(path,rows,checkpoint_sha):
 require(not Path(path).exists(),"Complete validation output already exists")
 with Path(path).open("x") as stream:
  for row in rows:stream.write(json.dumps({**row,"checkpoint_sha256":checkpoint_sha})+"\n")

def record_history(path,value):
 with Path(path).open("a") as stream:stream.write(json.dumps(value,allow_nan=False)+"\n")

def select_best(run_dir,candidate,score,best_score):
 if best_score is None or score<best_score:
  tmp=Path(run_dir)/"best.tmp";shutil.copyfile(candidate,tmp);tmp.replace(Path(run_dir)/"best.pt")
  return score,True
 return best_score,False

def seal_encoder(output):
 config=read(output/"run_config.json");state=torch.load(output/"best.pt",map_location="cpu",weights_only=True)
 require(state["schema"]==SCHEMA and state["config"]==config and state["source_sha256"]==source_seals(),"Selected SSL state identity differs")
 require(state["membership_sha256"]==sha256(output/"training_membership.json"),"Selected membership changed")
 encoder={name.removeprefix("encoder."):value for name,value in state["model"].items() if name.startswith("encoder.")}
 from eegmmidb_phase2.model import EEGEncoder
 EEGEncoder().load_state_dict(encoder,strict=True)
 current=tensor_sha(encoder);initial=state["initial_encoder_tensor_sha256"]
 path=output/"selected_encoder.pt"
 require(not path.exists(),"Selected encoder already sealed")
 torch.save(encoder,path)
 # Independently prove exported tensors equal the selected whole model.
 actual=torch.load(path,map_location="cpu",weights_only=True)
 require(set(actual)==set(encoder) and all(torch.equal(actual[n],encoder[n]) for n in encoder),"Exported encoder tensor identity differs")
 selected_epoch=state["best_epoch"]
 result={"schema":"xavier-phase3-selected-encoder-seal-v1","weights_sha256":sha256(path),
         "tensor_sha256":current,"initial_encoder_tensor_sha256":initial,
         "updated_from_initial":current!=initial,"selected_epoch":selected_epoch,
         "best_checkpoint_sha256":sha256(output/"best.pt"),
         "selected_validation_mean_subject_MSE":state["best_validation_score"],
         "training_membership_sha256":sha256(output/"training_membership.json"),
         "run_config_sha256":sha256(output/"run_config.json"),"source_sha256":source_seals(),
         "SSL_manifest_sha256":config["manifest_sha256"],"SSL_artifacts_sha256":config["artifacts_sha256"],
         "initial_state_was_explicit_candidate":True,
         "selected_initial_means_no_validated_adaptation":selected_epoch==0,
         "decoder_discarded_for_downstream":True}
 write_json(output/"selected_encoder_seal.json",result)
 return result

def finalize_partial_encoder(output,exit_receipt):
 output=Path(output).resolve();receipt=read(exit_receipt);summary=read(output/"run_summary.json")
 require(summary["status"]=="partial" and summary["stop_reason"] in ("stage_time_budget","absolute_deadline"),"Only registered time-limited partial SSL may advance")
 require(receipt.get("exit_code")==2 and receipt.get("process_returned_normally") is True and receipt.get("terminated_by_supervisor") is False,"Owned child did not return normally")
 require(Path(receipt["run_dir"]).resolve()==output and receipt["source_sha256"]==source_seals(),"Child receipt identity mismatch")
 command=receipt["command"]
 require(len(command)>3 and Path(command[1]).resolve()==Path(__file__).resolve() and command[2]=="train","Wrong child command")
 require("--run-dir" in command and Path(command[command.index("--run-dir")+1]).resolve()==output,"Wrong child run")
 require((output/"best.pt").is_file() and (output/"initial_validation.json").is_file(),"No complete validation baseline")
 state=torch.load(output/"best.pt",map_location="cpu",weights_only=True)
 require(state["best_epoch"]==0 or state["epoch_completed"] is True,"Partial validation may not select an encoder")
 selected=seal_encoder(output)
 summary.update(selected_encoder=selected,selected_encoder_seal_sha256=sha256(output/"selected_encoder_seal.json"),best_checkpoint_sha256=sha256(output/"best.pt"),partial_advancement_exit_receipt_sha256=sha256(exit_receipt))
 write_json(output/"run_summary.json",summary)
 return selected

def train(args):
 start=time.monotonic();contract,rows,audit=load_execution(args);device=select_device(args.device)
 require(args.max_seconds==contract["budget"]["phase_limits_seconds"]["SSL_training_and_validation"],"SSL stage duration must equal registered budget")
 require(args.absolute_deadline_epoch<=contract["budget"]["absolute_deadline_epoch"],"Stage deadline exceeds registered budget")
 torch.set_num_threads(2);allocator=apply_budget(device,21)
 batch,capacity=select_capacity(args.capacity_reports,sha256(args.contract));config=make_config(contract,args,batch)
 out=Path(args.run_dir);member=membership(rows,audit)
 if args.resume:
  require(not (out/"selected_encoder_seal.json").exists(),"A terminal selected encoder cannot be resumed")
  require(out.is_dir() and read(out/"run_config.json")==config and read(out/"training_membership.json")==member,"Restart changed scientific identity")
 else:
  require(not out.exists(),"Run exists; explicit resume required");out.mkdir(parents=True)
  write_json(out/"run_config.json",config);write_json(out/"training_membership.json",member)
  write_json(out/"capacity_selection.json",capacity);write_json(out/"execution_contract.json",contract)
  for source_name,destination in (("manifest.jsonl","ssl_manifest.jsonl"),("artifacts.json","ssl_artifacts.json"),("source_allowlist.jsonl","source_allowlist.jsonl"),("source_denylist.jsonl","source_denylist.jsonl"),("subject_split.json","subject_split.json")):
   shutil.copyfile(Path(args.prepared)/source_name,out/destination)
 member_sha=sha256(out/"training_membership.json")
 attempt=len(list(out.glob("attempt_*.json")))+1
 write_json(out/f"attempt_{attempt:03d}.json",{"arguments":vars(args),"started_epoch":time.time(),"scientific_config_sha256":sha256(out/"run_config.json")})
 seed_everything(config["seed"]);model=MaskedEEGModel().to(device)
 initial_tensor=tensor_sha(model.encoder.state_dict())
 optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
 train_rows=sorted((r for r in rows if r["split"]=="train"),key=lambda r:r["id"])
 val_rows=sorted((r for r in rows if r["split"]=="val"),key=lambda r:r["id"]);store=SSLStore()
 epoch=cursor=step=presentations=completed_epochs=0;seen=set();elapsed_before=0.
 best_score=None;best_epoch=0;initial_metrics=None;stop=None;consistent=True;epoch_completed=False
 if args.resume:
  state=torch.load(args.resume,map_location="cpu",weights_only=True)
  require(state["schema"]==SCHEMA and state["config"]==config and state["source_sha256"]==source_seals(),"Resume source or config changed")
  require(state["membership_sha256"]==member_sha and state["resumable"],"Unsafe or different-membership restart")
  model.load_state_dict(state["model"],strict=True);optimizer.load_state_dict(state["optimizer"]);restore_rng(state["rng"])
  epoch,cursor,step=state["epoch_zero_based"],state["next_position"],state["global_step"]
  seen=set(state["seen_train_ids"]);presentations=state["training_examples_presented"];elapsed_before=state["elapsed_stage_seconds"]
  initial_tensor=state["initial_encoder_tensor_sha256"];best_score=state["best_validation_score"];best_epoch=state["best_epoch"]
  initial_metrics=read(out/"initial_validation.json")
  if state["epoch_completed"]:epoch+=1;cursor=0
  completed_epochs=epoch
 remaining=max(0,config["max_seconds"]-elapsed_before)
 try:
  if not args.resume:
   initial_metrics,initial_rows=validate(model,val_rows,store,device,config["validation_batch_size"],start,remaining,args.absolute_deadline_epoch,21)
   best_score=initial_metrics["mean_subject_masked_MSE"]
   state=state_for(config,member_sha,0,0,0,set(),0,time.monotonic()-start,initial_tensor,best_score,0,False)
   initial_path=out/"initial_epoch_000.pt";initial_sha=save(initial_path,model,optimizer,state)
   shutil.copyfile(initial_path,out/"best.pt")
   write_json(out/"initial_validation.json",initial_metrics)
   save_validation(out/"validation_initial.jsonl",initial_rows,initial_sha)
   print(json.dumps({"stage":"initial_validation","mean_subject_masked_MSE":best_score}),flush=True)
  while epoch<config["max_epochs"]:
   model.train();epoch_completed=False;loss_sum=0.;n=0
   for position,batch_rows in batches(train_rows,config["batch_size"],config["seed"],epoch,cursor):
    guard(start,remaining,args.absolute_deadline_epoch)
    eeg,masks=tensor_batch(batch_rows,store,device,"train",epoch)
    optimizer.zero_grad(set_to_none=True);consistent=False
    loss=losses_per_window(model(make_masked_input(eeg,masks)),eeg,masks).mean()
    require(bool(torch.isfinite(loss)),"Nonfinite SSL loss")
    loss.backward();require(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),"Nonfinite SSL gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(),config["gradient_clip_norm"])
    optimizer.step();synchronize(device);consistent=True
    step+=1;cursor=position+len(batch_rows);seen.update(r["id"] for r in batch_rows);presentations+=len(batch_rows)
    n+=len(batch_rows);loss_sum+=float(loss.detach())*len(batch_rows)
    if not within_budget(memory_sample(device),21,device):raise BoundedStop("memory_budget")
    if step%100==0:
     state=state_for(config,member_sha,epoch,cursor,step,seen,presentations,elapsed_before+time.monotonic()-start,initial_tensor,best_score,best_epoch,False)
     save(out/f"progress_epoch_{epoch+1:03d}_step_{step:07d}.pt",model,optimizer,state)
   metrics,predictions=validate(model,val_rows,store,device,config["validation_batch_size"],start,remaining,args.absolute_deadline_epoch,21)
   score=metrics["mean_subject_masked_MSE"];improved=score<best_score
   candidate_score=score if improved else best_score;candidate_epoch=epoch+1 if improved else best_epoch
   epoch_completed=True
   state=state_for(config,member_sha,epoch,cursor,step,seen,presentations,elapsed_before+time.monotonic()-start,initial_tensor,candidate_score,candidate_epoch,True)
   path=out/f"epoch_{epoch+1:03d}_step_{step:07d}.pt";checkpoint_sha=save(path,model,optimizer,state)
   best_score,selected=select_best(out,path,score,best_score)
   if selected:best_epoch=epoch+1
   save_validation(out/f"validation_epoch_{epoch+1:03d}.jsonl",predictions,checkpoint_sha)
   history={"epoch":epoch+1,"global_step":step,"training_loss_this_attempt_epoch":loss_sum/n if n else None,
            "validation":metrics,"selected_best":selected,"checkpoint":path.name,"checkpoint_sha256":checkpoint_sha,
            "elapsed_seconds":elapsed_before+time.monotonic()-start,"memory":memory_sample(device)}
   record_history(out/"history.jsonl",history)
   print(json.dumps({"epoch":epoch+1,"global_step":step,"validation_mean_subject_MSE":score,"selected_best":selected,"elapsed_seconds":history["elapsed_seconds"]}),flush=True)
   completed_epochs=epoch+1;epoch+=1;cursor=0
 except (BoundedStop,RuntimeError,ValueError) as exc:stop=str(exc)
 if stop:
  state=state_for(config,member_sha,epoch,cursor,step,seen,presentations,elapsed_before+time.monotonic()-start,initial_tensor,best_score,best_epoch,epoch_completed,consistent)
  save(out/f"stopped_attempt_{attempt:03d}_step_{step:07d}.pt",model,optimizer,state)
  write_json(out/f"stop_attempt_{attempt:03d}.json",{"reason":stop,"resumable":consistent,"validation_skipped_after_stop":True})
 selected_seal=seal_encoder(out) if stop is None and (out/"best.pt").exists() else None
 actual=[r for r in train_rows if r["id"] in seen]
 summary={"schema":"xavier-phase3-SSL-training-summary-v1","status":"completed" if stop is None and completed_epochs==config["max_epochs"] else "partial",
          "stop_reason":stop,"completed_epochs":completed_epochs,"optimizer_steps":step,
          "initial_validation":initial_metrics,"selected_epoch":best_epoch,"selected_validation_mean_subject_MSE":best_score,
          "selected_improved_over_initial":bool(initial_metrics and best_score<initial_metrics["mean_subject_masked_MSE"]),
          "seen_train_ids":sorted(seen),"seen_train_ids_sha256":object_hash(sorted(seen)),
          "unique_seen_training_windows":len(seen),"unique_source_interval_union_seconds":source_interval_union(actual),
          "training_examples_presented":presentations,"summed_presented_window_seconds":presentations*3.,
          "all_registered_training_windows_seen":seen=={r["id"] for r in train_rows},
          "selected_encoder_seal_sha256":sha256(out/"selected_encoder_seal.json") if selected_seal else None,
          "selected_encoder":selected_seal,"best_checkpoint_sha256":sha256(out/"best.pt") if selected_seal else None,
          "training_membership_sha256":member_sha,"elapsed_seconds":elapsed_before+time.monotonic()-start,
          "memory":memory_sample(device),"hardware":hardware(device),"allocator":allocator,
          "data_audit":audit,"source_sha256":source_seals(),"test_or_action_label_fitting":False,
          "SSL_reconstruction_improvement_does_not_prove_downstream_intent_performance":True}
 write_json(out/"run_summary.json",summary)
 print(json.dumps({k:summary[k] for k in ("status","completed_epochs","optimizer_steps","selected_epoch","selected_improved_over_initial","stop_reason")}),flush=True)
 return summary

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
 register0=sub.add_parser("register")
 for name in ("prepared","budget","out"):register0.add_argument("--"+name,required=True)
 seal_parser=sub.add_parser("seal")
 seal_parser.add_argument("--run-dir",required=True)
 seal_parser.add_argument("--exit-receipt",required=True)
 for name in ("probe","train"):
  q=sub.add_parser(name)
  for key in ("prepared","contract","run-dir"):q.add_argument("--"+key,required=True)
  q.add_argument("--device",default="cuda");q.add_argument("--max-seconds",type=float,required=True)
  q.add_argument("--absolute-deadline-epoch",type=float,required=True)
  if name=="probe":q.add_argument("--batch-size",type=int,required=True)
  else:q.add_argument("--capacity-reports",nargs="+",required=True);q.add_argument("--resume")
 args=p.parse_args()
 if args.command=="register":register(args)
 elif args.command=="seal":
  finalize_partial_encoder(args.run_dir,args.exit_receipt)
 elif args.command=="probe":
  value=probe(args);raise SystemExit(0 if value["status"]=="passed" else 2)
 else:
  value=train(args);raise SystemExit(0 if value["status"]=="completed" else 2)

if __name__=="__main__":main()

