"""Serialized bounded phase3 training after explicit phase2 evidence handover.

This operational supervisor never reads held-out predictions or changes scientific
parameters. It preserves every capacity, checkpoint, exception and terminal receipt.
Independent evaluation and artifact recovery follow the final training handover.
"""
from __future__ import annotations
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
 if not value:raise RuntimeError(message)
def save(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def env():
 value=os.environ.copy()
 if value.get('CUDA_VISIBLE_DEVICES')=='':del value['CUDA_VISIBLE_DEVICES']
 for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):value[name]='2'
 value['PYTORCH_ALLOC_CONF']='expandable_segments:True';value['HF_HUB_DISABLE_PROGRESS_BARS']='1'
 return value

def run_child(command,operation,timeout,deadline,*,run_dir=None,scientific_source=None):
 operation=Path(operation);require(not operation.with_suffix('.exit.json').exists(),'Child terminal evidence exists')
 started=time.time();seconds=min(timeout,deadline-started);require(seconds>10,'Child budget exhausted')
 log=operation.with_suffix('.log').open('xb')
 child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env(),start_new_session=True)
 save(operation.with_suffix('.launch.json'),{'pid':child.pid,'command':command,'started_epoch':started,
                                          'deadline_epoch':started+seconds,'GPU_jobs_are_serial':True})
 killed=False
 try:code=child.wait(timeout=seconds)
 except subprocess.TimeoutExpired:
  killed=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
 finally:log.close()
 value={'status':'terminal','pid':child.pid,'command':command,'exit_code':code,
        'process_returned_normally':code>=0 and not killed,'terminated_by_supervisor':killed,
        'started_epoch':started,'finished_epoch':time.time(),'elapsed_seconds':time.time()-started,
        'run_dir':str(run_dir) if run_dir else None,'source_sha256':scientific_source,
        'log_sha256':sha(operation.with_suffix('.log'))}
 save(operation.with_suffix('.exit.json'),value);return value

def validate_terminal(run,receipt,expected_kind,exit_path):
 summary=read(run/'run_summary.json')
 require(receipt['process_returned_normally'] and not receipt['terminated_by_supervisor'],'Child terminated abnormally')
 require(summary['status'] in ('completed','partial'),'No scientific terminal status')
 if summary['status']=='completed':require(receipt['exit_code']==0,'Completed trainer nonzero exit')
 else:
  require(receipt['exit_code']==2 and summary['stop_reason'] in ('stage_time_budget','absolute_deadline'),'Non-time scientific failure may not advance')
 if expected_kind!='ssl':
  require(summary['kind']==expected_kind and summary['test_forward_calls']==0 and summary['test_used_for_tuning'] is False,'Wrong training task or test exposure')
  require(summary['best_checkpoint_sha256'] and sha(run/'best.pt')==summary['best_checkpoint_sha256'],'No complete validation best')
 return summary

def verify(args):
 root=Path(args.root).resolve();budget_path=root/'gpu_budget_v1.json';budget=read(budget_path)
 require(sha(budget_path)==args.budget_sha256,'Registered phase3 budget changed')
 require(budget['session']=='xavier-eegmmidb-20261122-p2' and budget['remote_root']==str(root),'Wrong session/root')
 require(budget['absolute_deadline_epoch']==1795331083.750155,'Absolute phase3 deadline changed')
 handover=read(args.handover)
 require(sha(args.handover)==args.handover_sha256,'Handover receipt changed')
 require(handover['session']==budget['session'] and handover['phase3_budget_sha256']==args.budget_sha256,'Handover lease identity mismatch')
 for key in ('phase2_all_training_and_evaluation_processes_terminal','phase2_all_selected_models_and_evaluations_recovered','all_recovery_member_hashes_verified','phase3_GPU_authorized'):
  require(handover.get(key) is True,'Incomplete owned-work handover: '+key)
 require(handover['physical_session_continues'] is True and handover['physical_session_stopped'] is False,'Misrepresented session transfer')
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();require(boot==handover['boot_id'],'Physical GPU boot changed')
 recovery=Path(handover['phase2_recovery_ledger_remote_path'])
 require(sha(recovery)==handover['phase2_recovery_ledger_sha256'],'Completed phase2 local recovery proof changed')
 recovery_value=read(recovery)
 require(recovery_value.get('status')=='completed' and recovery_value.get('all_member_hashes_verified') is True,'Phase2 artifacts not recovered')
 current=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
 require(not current,'GPU occupied at handover')
 code=root/'code/eegmmidb_phase3';seal=read(args.source_seal)
 require(sha(args.source_seal)==args.source_seal_sha256,'Source/test seal changed')
 for name,digest in seal['files'].items():require(sha(root/'code'/name)==digest,'Frozen execution source changed: '+name)
 require(seal['CPU_tests_status']=='passed','CPU tests not complete')
 observed=read(args.resource_observation)
 require(observed['session']==budget['session'] and observed['boot_id']==boot,'Resource observation belongs elsewhere')
 require(time.time()-observed['observed_epoch']<300,'Resource/CU observation is stale')
 require(observed['compute_units_remaining']>=budget['initial_observed_compute_units']-budget['maximum_compute_units_from_registration_observation'],'CU ceiling already exceeded')
 require(observed['compute_units_remaining']>=budget['minimum_compute_unit_reserve'],'CU reserve unavailable')
 require(observed['remote_disk_free_bytes']>=budget['remote_disk_reserve_bytes'],'Remote disk reserve unavailable')
 reserve=sum(budget['phase_limits_seconds'][k] for k in ('heldout_evaluation_and_routing','recovery_and_verification','shutdown'))
 deadline=budget['absolute_deadline_epoch']-reserve
 require(time.time()+1800<deadline,'Insufficient minimum fit time while preserving full evaluation/recovery/shutdown reserve')
 return root,budget,code,deadline

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('root','phase2-root','budget-sha256','handover','handover-sha256','source-seal','source-seal-sha256','resource-observation'):
  p.add_argument('--'+name,required=True)
 args=p.parse_args();root,budget,code,deadline=verify(args);phase2=Path(args.phase2_root).resolve()
 workflow=root/'workflow';workflow.mkdir(exist_ok=True)
 lock=workflow/'active_owned_job.json'
 with lock.open('x') as f:json.dump({'pid':os.getpid(),'started_epoch':time.time(),'budget_sha256':args.budget_sha256},f)
 started=time.time();capacity_spent=0.;selected={}
 ssl_contract=root/'ssl_execution_contract.v1.json';supervised_contract=root/'supervised_execution_contract.v1.json'
 ssl_run=root/'runs/masked_ssl_v1'
 try:
  # CPU input registration performs independent complete raw-identity and array-byte validation.
  command=[sys.executable,str(code/'train_ssl.py'),'register','--prepared',str(root/'prepared'),'--budget',str(root/'gpu_budget_v1.json'),'--out',str(ssl_contract)]
  terminal=run_child(command,workflow/'register_ssl',180,deadline)
  require(terminal['exit_code']==0,'SSL registration/input proof failed')
  ssl=read(ssl_contract)
  # Predeclared numerical equivalence is train-only; failures select the unchanged reference head.
  numeric=root/'operations/head_L4_BF16.json'
  command=[sys.executable,str(code/'compare_symbol_head.py'),'run','--scope','L4_BF16','--prepared',str(phase2/'prepared'),'--output',str(numeric)]
  terminal=run_child(command,workflow/'numeric_head_L4',120,deadline)
  capacity_spent+=terminal['elapsed_seconds'];require(terminal['exit_code']==0 and numeric.is_file(),'Numerical comparison could not complete')
  combined=root/'operations/head_equivalence_combined.json'
  command=[sys.executable,str(code/'compare_symbol_head.py'),'combine','--cpu',str(root/'operations/cpu_head_equivalence.json'),'--l4',str(numeric),'--output',str(combined)]
  terminal=run_child(command,workflow/'combine_head_guard',30,deadline);require(terminal['exit_code']==0,'Numerical evidence did not validate')

  def probe(kind,contract_path,candidates):
   nonlocal capacity_spent
   reports=[]
   for batch in sorted(candidates,reverse=True):
    left=budget['phase_limits_seconds']['train_only_capacity']-capacity_spent
    require(left>15,'Cumulative capacity budget exhausted')
    out=root/'probes'/f'{kind}_batch_{batch}'
    if kind=='ssl':
     command=[sys.executable,str(code/'train_ssl.py'),'probe','--prepared',str(root/'prepared'),'--contract',str(contract_path),'--run-dir',str(out),'--device','cuda','--batch-size',str(batch),'--max-seconds',str(min(150,left-10)),'--absolute-deadline-epoch',str(min(deadline,time.time()+left-10))]
    else:
     command=[sys.executable,str(code/'train_supervised.py'),'probe','--prepared',str(phase2/'prepared'),'--contract',str(contract_path),'--ssl-run',str(ssl_run),'--kind',kind,'--run-dir',str(out),'--device','cuda','--batch-size',str(batch),'--local-files-only','--absolute-deadline-epoch',str(min(deadline,time.time()+left-10))]
     if kind=='fused':command+=['--scratch-run',str(root/'runs/eegnet_ssl_supervised_v1')]
    terminal=run_child(command,workflow/f'probe_{kind}_{batch}',min(170,left),deadline)
    capacity_spent+=terminal['elapsed_seconds'];report=out/'capacity.json'
    if report.exists():
     value=read(report);require(value['execution_contract_sha256']==sha(contract_path),'Capacity identity changed')
     reports.append(str(report))
     if value['status']=='passed' and value['completed_optimizer_steps']==50:break
   valid=[read(path) for path in reports if read(path)['status']=='passed' and read(path)['completed_optimizer_steps']==50]
   require(valid,'No admitted capacity for '+kind)
   result={'status':'ready','kind':kind,'selected_batch':max(r['batch_size'] for r in valid),'reports':reports,
           'report_sha256':{path:sha(path) for path in reports},'cumulative_capacity_seconds':capacity_spent,'updates_reused_for_training':False}
   save(workflow/(kind+'_capacity_ready.json'),result);return result

  ssl_capacity=probe('ssl',ssl_contract,ssl['plan']['optimization']['candidate_batches'])
  command=[sys.executable,str(code/'train_ssl.py'),'train','--prepared',str(root/'prepared'),'--contract',str(ssl_contract),'--run-dir',str(ssl_run),'--device','cuda','--max-seconds',str(budget['phase_limits_seconds']['SSL_training_and_validation']),'--absolute-deadline-epoch',str(deadline),'--capacity-reports',*ssl_capacity['reports']]
  operation=workflow/'train_ssl'
  terminal=run_child(command,operation,budget['phase_limits_seconds']['SSL_training_and_validation']+20,deadline+10,run_dir=ssl_run,scientific_source=ssl['source_sha256'])
  summary=validate_terminal(ssl_run,terminal,'ssl',operation.with_suffix('.exit.json'))
  if summary['status']=='partial':
   command=[sys.executable,str(code/'train_ssl.py'),'seal','--run-dir',str(ssl_run),'--exit-receipt',str(operation.with_suffix('.exit.json'))]
   result=run_child(command,workflow/'seal_time_partial_ssl',60,deadline);require(result['exit_code']==0,'Time-limited SSL could not be sealed')
  selected['ssl']={'run':ssl_run.name,'summary_sha256':sha(ssl_run/'run_summary.json'),'encoder_seal_sha256':sha(ssl_run/'selected_encoder_seal.json')}

  command=[sys.executable,str(code/'train_supervised.py'),'register','--prepared',str(phase2/'prepared'),'--protocol',str(code/'protocol.supervised.v1.json'),'--ssl-run',str(ssl_run),'--head-equivalence',str(combined),'--budget',str(root/'gpu_budget_v1.json'),'--out',str(supervised_contract)]
  result=run_child(command,workflow/'register_supervised',180,deadline);require(result['exit_code']==0,'Supervised source/input/SSL lineage failed')
  supervised=read(supervised_contract)
  for kind,name in (('scratch','eegnet_ssl_supervised_v1'),('fused','eegnet_ssl_smollm_fused_v1')):
   capacity=probe(kind,supervised_contract,supervised['protocol']['stages'][kind]['capacity_batch_candidates'])
   out=root/'runs'/name
   command=[sys.executable,str(code/'train_supervised.py'),'train','--prepared',str(phase2/'prepared'),'--contract',str(supervised_contract),'--ssl-run',str(ssl_run),'--kind',kind,'--run-dir',str(out),'--device','cuda','--local-files-only','--absolute-deadline-epoch',str(deadline),'--capacity-reports',*capacity['reports']]
   if kind=='fused':command+=['--scratch-run',str(root/'runs/eegnet_ssl_supervised_v1')]
   duration=supervised['protocol']['stages'][kind]['max_seconds'];operation=workflow/('train_'+kind)
   terminal=run_child(command,operation,duration+20,deadline+10,run_dir=out,scientific_source=supervised['scientific_source_sha256'])
   summary=validate_terminal(out,terminal,kind,operation.with_suffix('.exit.json'))
   selected[kind]={'run':name,'status':summary['status'],'summary_sha256':sha(out/'run_summary.json'),
                   'best_checkpoint_sha256':summary['best_checkpoint_sha256'],'completed_epochs':summary['completed_epochs'],
                   'optimizer_steps':summary['optimizer_steps'],'validation_batch_size':read(out/'run_config.json')['validation_batch_size']}
  result={'status':'training_terminal_pending_independent_test_and_recovery','models':selected,
          'started_epoch':started,'finished_epoch':time.time(),'elapsed_seconds':time.time()-started,
          'cumulative_capacity_seconds':capacity_spent,'budget_sha256':args.budget_sha256,
          'source_seal_sha256':args.source_seal_sha256,'handover_sha256':args.handover_sha256,
          'test_forward_calls':0,'goal_validated':False,'physical_session_stopped':False}
  save(workflow/'training_completion.json',result);print(json.dumps(result),flush=True)
 except Exception as exc:
  save(workflow/'training_failure.json',{'status':'failed','error':type(exc).__name__+': '+str(exc),'epoch':time.time(),'models_finished':selected,'cumulative_capacity_seconds':capacity_spent,'test_forward_calls':0})
  raise
 finally:
  if lock.exists() and read(lock).get('pid')==os.getpid():lock.unlink()

if __name__=='__main__':main()
