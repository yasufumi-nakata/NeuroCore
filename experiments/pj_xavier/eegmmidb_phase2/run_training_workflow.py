"""Operate the registered phase2 GPU jobs, one process at a time.

All data/scientific source identities are frozen by an execution contract.
This operational runner never performs test prediction or changes a model plan.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(2**20),b""):h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");tmp.replace(path)

def require(condition,message):
    if not condition:raise RuntimeError(message)

def subprocess_environment():
    env=os.environ.copy()
    if env.get("CUDA_VISIBLE_DEVICES")=="":
        del env["CUDA_VISIBLE_DEVICES"]
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        env[name]="2"
    env["PYTORCH_ALLOC_CONF"]="expandable_segments:True"
    env["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
    return env

def run_child(command,operation,timeout,absolute_deadline):
    operation=Path(operation)
    require(not operation.with_suffix(".exit.json").exists(),"Terminal receipt already exists")
    started=time.time()
    remaining=min(timeout,absolute_deadline-started)
    require(remaining>10,"Insufficient child budget")
    log=operation.with_suffix(".log").open("xb")
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=subprocess_environment(),start_new_session=True)
    save(operation.with_suffix(".launch.json"),{
        "pid":child.pid,"command":command,"started_epoch":started,
        "child_deadline_epoch":started+remaining,"global_deadline_epoch":absolute_deadline,
        "environment_overrides":{"PYTORCH_ALLOC_CONF":"expandable_segments:True","CPU_threads":"2"},
    })
    timed_out=False
    try:
        code=child.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        timed_out=True
        os.killpg(child.pid,signal.SIGTERM)
        try:code=child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
    finally:log.close()
    receipt={"status":"terminal","pid":child.pid,"exit_code":code,"timed_out":timed_out,
             "started_epoch":started,"finished_epoch":time.time(),"elapsed_seconds":time.time()-started,
             "command":command,"log_sha256":sha(operation.with_suffix(".log"))}
    save(operation.with_suffix(".exit.json"),receipt)
    return receipt

def verify(root,args):
    contract_path=root/"training_execution_contract.v2.json"
    budget_path=root/"gpu_budget_v1.json"
    require(sha(contract_path)==args.contract_sha256,"Training execution contract changed")
    require(sha(budget_path)==args.budget_sha256,"Budget receipt changed")
    contract,budget=read(contract_path),read(budget_path)
    require(budget["remote_root"]==str(root) and budget["session"]=="xavier-eegmmidb-20261122-p2","Wrong owned session root")
    require(budget["absolute_deadline_epoch"]==1795325994.596278,"Original deadline changed")
    require(time.time()<budget["absolute_deadline_epoch"]-2100,"Recovery/evaluation/shutdown reserve exhausted")
    code=root/"code/eegmmidb_phase2"
    for name,digest in contract["scientific_source_sha256"].items():
        require(sha(code/name)==digest,"Model source changed: "+name)
    require(sha(code/"protocol.training.v1.json")==contract["protocol_sha256"],"Training protocol changed")
    tests=read(root/"operations/cpu_tests_receipt.v2.json")
    require(tests["status"]=="passed","CPU gradient/restart tests not passed")
    for name,digest in contract["scientific_source_sha256"].items():
        require(tests["source_sha256"][name]==digest,"CPU tests ran a different source: "+name)
    require(read(root/"operations/backbone_download_receipt.json")["status"]=="completed","Pinned backbone download incomplete")
    require(sha(root/"prepared/manifest.jsonl")==contract["prepared_manifest_sha256"],"Main manifest changed")
    require(sha(root/"prepared/artifacts.json")==contract["artifacts_sha256"],"Array seals changed")
    gpu=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader"],capture_output=True,text=True,check=True).stdout.strip()
    require(not gpu,"Another CUDA process is running")
    return contract,budget,code

def probe_kind(root,kind,contract,budget,code,capacity_spent):
    report_paths=[]
    candidates=sorted(contract["protocol"]["stages"][kind]["capacity_batch_candidates"],reverse=True)
    training_deadline=budget["absolute_deadline_epoch"]-2100
    parent=["--scratch-run",str(root/"runs/eegnet_scratch_v1")] if kind=="fused" else []
    for batch in candidates:
        left=budget["phases_seconds"]["capacity_probe"]-capacity_spent
        if left<=15:break
        out=root/"probes"/f"{kind}_batch_{batch}"
        operation=root/"workflow"/f"probe_{kind}_batch_{batch}"
        command=[sys.executable,str(code/"train_models.py"),"probe","--prepared",str(root/"prepared"),
                 "--contract",str(root/"training_execution_contract.v2.json"),"--kind",kind,
                 "--run-dir",str(out),"--device","cuda","--batch-size",str(batch),"--local-files-only",
                 "--absolute-deadline-epoch",str(min(training_deadline,time.time()+left-5)),*parent]
        terminal=run_child(command,operation,min(170,left),training_deadline)
        capacity_spent+=terminal["elapsed_seconds"]
        report=out/"capacity.json"
        if report.exists():
            value=read(report)
            require(value["execution_contract_sha256"]==sha(root/"training_execution_contract.v2.json"),"Probe identity changed")
            report_paths.append(report)
            if value["status"]=="passed" and value["completed_optimizer_steps"]==50:
                break
    valid=[p for p in report_paths if read(p)["status"]=="passed" and read(p)["completed_optimizer_steps"]==50]
    require(valid,"No successful bounded "+kind+" capacity probe")
    best=max(valid,key=lambda p:read(p)["batch_size"])
    summary={"status":"ready","kind":kind,"selected_batch_size":read(best)["batch_size"],
             "capacity_reports":[str(p) for p in report_paths],
             "capacity_report_sha256":{str(p):sha(p) for p in report_paths},
             "total_capacity_seconds_used":capacity_spent,"GPU_training_updates_used_for_main":0,
             "batch_selection_rule":"highest registered candidate tested first; back off only after resource failure",
             "source_contract_sha256":sha(root/"training_execution_contract.v2.json"),
             "observed_epoch":time.time()}
    save(root/"workflow"/(kind+"_capacity_ready.json"),summary)
    return summary

def train_kind(root,kind,contract,budget,code,capacity):
    name="eegnet_scratch_v1" if kind=="scratch" else "eegnet_smollm_fused_v1"
    out=root/"runs"/name
    training_deadline=budget["absolute_deadline_epoch"]-2100
    parent=["--scratch-run",str(root/"runs/eegnet_scratch_v1")] if kind=="fused" else []
    command=[sys.executable,str(code/"train_models.py"),"train","--prepared",str(root/"prepared"),
             "--contract",str(root/"training_execution_contract.v2.json"),"--kind",kind,
             "--run-dir",str(out),"--device","cuda","--local-files-only",
             "--absolute-deadline-epoch",str(training_deadline),
             "--capacity-reports",*capacity["capacity_reports"],*parent]
    terminal=run_child(command,root/"workflow"/("train_"+kind),contract["protocol"]["stages"][kind]["max_seconds"]+20,training_deadline+10)
    require((out/"run_summary.json").exists(),"Training produced no terminal scientific summary")
    summary=read(out/"run_summary.json")
    require(summary["status"] in ("completed","partial"),"Unknown training status")
    require(summary["test_forward_calls"]==0 and summary["test_used_for_tuning"] is False,"Held-out data entered training")
    require(summary["best_checkpoint_sha256"] is not None,"No validation-selected checkpoint")
    require(sha(out/"best.pt")==summary["best_checkpoint_sha256"],"Selected checkpoint bytes differ")
    return {"run_name":name,"summary_sha256":sha(out/"run_summary.json"),
            "best_checkpoint_sha256":summary["best_checkpoint_sha256"],
            "training_status":summary["status"],"terminal_exit_code":terminal["exit_code"]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True)
    p.add_argument("--contract-sha256",required=True)
    p.add_argument("--budget-sha256",required=True)
    p.add_argument("--stage",choices=("scratch-probe","train"),required=True)
    p.add_argument("--admission")
    args=p.parse_args()
    root=Path(args.root).resolve()
    contract,budget,code=verify(root,args)
    workflow=root/"workflow";workflow.mkdir(exist_ok=True)
    lock=workflow/"active_owned_job.json"
    with lock.open("x") as stream:
        json.dump({"pid":os.getpid(),"stage":args.stage,"started_epoch":time.time(),"boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip()},stream)
    try:
        if args.stage=="scratch-probe":
            prior=root/"operations/prior_capacity_seconds.json"
            spent=read(prior)["elapsed_seconds"] if prior.exists() else 0.
            result=probe_kind(root,"scratch",contract,budget,code,spent)
        else:
            require(args.admission is not None,"Training admission receipt required")
            admission=read(args.admission)
            require(admission["training_contract_sha256"]==args.contract_sha256 and admission["budget_sha256"]==args.budget_sha256,"Training admission identity changed")
            require(admission["scratch_capacity_sha256"]==sha(workflow/"scratch_capacity_ready.json"),"Scratch capacity admission changed")
            require(admission["source_and_data_sealed"] is True and admission["benchmark_selection_before_test"] is True,"Training gates incomplete")
            scratch_capacity=read(workflow/"scratch_capacity_ready.json")
            scratch=train_kind(root,"scratch",contract,budget,code,scratch_capacity)
            fused_capacity=probe_kind(root,"fused",contract,budget,code,scratch_capacity["total_capacity_seconds_used"])
            fused=train_kind(root,"fused",contract,budget,code,fused_capacity)
            result={"status":"training_terminal_pending_independent_test_and_recovery",
                    "scratch":scratch,"fused":fused,"completed_epoch":time.time(),
                    "test_forward_calls":0,"goal_validated":False,
                    "training_contract_sha256":args.contract_sha256,"budget_sha256":args.budget_sha256}
            save(workflow/"training_completion.json",result)
        print(json.dumps(result),flush=True)
    except Exception as exc:
        save(workflow/(args.stage+"_failure.json"),{"status":"failed","error":str(exc),"epoch":time.time(),"test_forward_calls":0})
        raise
    finally:
        if lock.exists() and read(lock).get("pid")==os.getpid():lock.unlink()

if __name__=="__main__":
    main()

