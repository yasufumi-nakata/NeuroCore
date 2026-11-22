"""New phase3 admission around the byte-identical phase2 evaluator science."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

PHASE2_SEAL='e1847ae39ef284a49cdc34dbe1695593b6d9a3c46b8d4db046842c403b26eb7f'
PHASE3_BUDGET='018cdaa6c81eba94178ce8e465b89360e425691c4d083c5e25855b6d5c6d25a5'
PHASE3_SOURCE='4c713b5112ebae88885a7c6dae7dc429722df2543d78278a38de7174cf396be7'
TASK_RUNS={'eegnet_ssl_supervised_v1':'MI_supervised','eegnet_ssl_smollm_fused_v1':'fused_alignment'}
HELPERS=('benchmark_adapter.py','benchmark_ssl_audit.py','harness_compare.py','protocol.evaluation.v1.json')

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def normalized_budget(value):
    require(value['session']=='xavier-eegmmidb-20261122-p2' and value['remote_root']=='/content/xavier_phase3','wrong phase3 lease')
    require(value['GPU_memory_budget_gib']==21 and value['colab_access']=='workstation_CLI_only','phase3 hardware/access changed')
    require(value['phase_limits_seconds']['heldout_evaluation_and_routing']==900,'phase3 total evaluation allocation changed')
    require(value['absolute_deadline_epoch']==1795331083.750155,'phase3 absolute deadline changed')
    result=copy.deepcopy(value);result['gpu_memory_budget_gib']=value['GPU_memory_budget_gib']
    return result

def register(phase2_seal,source_seal,budget,phase2_code,output):
    phase2_code=Path(phase2_code).resolve();phase3_code=Path(__file__).resolve().parent
    require(sha(phase2_seal)==PHASE2_SEAL and sha(source_seal)==PHASE3_SOURCE and sha(budget)==PHASE3_BUDGET,'parent source/budget contract changed')
    original=read(phase2_seal);source=read(source_seal);allocation=read(budget)
    require(source['CPU_tests_status']=='passed','phase3 CPU verification incomplete')
    for name,digest in original['source_sha256'].items():require(sha(phase2_code/name)==digest,'immutable phase2 source changed')
    for name,digest in source['files'].items():require(sha(phase3_code.parent/name)==digest,'sealed phase3/source dependency changed')
    seal=copy.deepcopy(original);seal['schema']='xavier-phase3-same-EEGMMIDB-evaluation-seal-v1'
    seal['source_sha256']['runtime.py']=source['files']['eegmmidb_phase3/runtime.py']
    seal['budget']=normalized_budget(allocation)
    seal.update(phase3_registered_epoch=time.time(),phase2_parent_execution_seal_sha256=PHASE2_SEAL,phase3_source_seal_sha256=PHASE3_SOURCE,
                phase3_budget_sha256=PHASE3_BUDGET,phase3_source_sha256=source['files'],phase3_evaluation_helper_sha256={n:sha(phase3_code/n) for n in HELPERS},
                phase3_source_seal_document=Path(source_seal).read_text(),phase3_budget_document=Path(budget).read_text(),
                budget_field_alias={'gpu_memory_budget_gib':'same numeric value as original GPU_memory_budget_gib; original budget bytes unchanged'},
                comparison_scope='Additional pre-registered SSL models reuse the original test people exploratorily. Original phase2 model/report/source bytes unchanged. No threshold, prompt, sample, control or statistic changes.')
    save(output,seal)
    return {'status':'registered','execution_seal_sha256':sha(output),'model_forward_calls':0,'test_based_parameter_selection':False}

def verify_scope(seal_path,expected,phase2_seal,phase2_code):
    require(sha(seal_path)==expected and sha(phase2_seal)==PHASE2_SEAL,'evaluation seal identity differs')
    value=read(seal_path);original=read(phase2_seal);phase2_code=Path(phase2_code).resolve();phase3_code=Path(__file__).resolve().parent
    require(value['phase2_parent_execution_seal_sha256']==PHASE2_SEAL and value['phase3_source_seal_sha256']==PHASE3_SOURCE and value['phase3_budget_sha256']==PHASE3_BUDGET,'wrong parent/new resource source')
    require(hashlib.sha256(value['phase3_source_seal_document'].encode()).hexdigest()==PHASE3_SOURCE and hashlib.sha256(value['phase3_budget_document'].encode()).hexdigest()==PHASE3_BUDGET,'embedded original source/budget bytes changed')
    require(value['phase3_source_sha256']==json.loads(value['phase3_source_seal_document'])['files'],'phase3 scientific source map differs from original signed-off source')
    require(value['budget']==normalized_budget(json.loads(value['phase3_budget_document'])),'resource alias changed original allocation fields')
    for key,item in original.items():
        if key not in ('schema','source_sha256','budget'):require(value[key]==item,'original data/analysis contract changed: '+key)
    for name,digest in original['source_sha256'].items():require(sha(phase2_code/name)==digest,'immutable evaluator/model dependency changed')
    expected_source=dict(original['source_sha256']);expected_source['runtime.py']=value['phase3_source_sha256']['eegmmidb_phase3/runtime.py']
    require(value['source_sha256']==expected_source,'unexpected evaluator implementation replacement')
    for name,digest in value['phase3_source_sha256'].items():require(sha(phase3_code.parent/name)==digest,'phase3 scientific source changed')
    require(set(value['phase3_evaluation_helper_sha256'])==set(HELPERS),'missing independent admission helper')
    for name,digest in value['phase3_evaluation_helper_sha256'].items():require(sha(phase3_code/name)==digest,'phase3 evaluation admission changed')
    require(value['budget']['gpu_memory_budget_gib']==value['budget']['GPU_memory_budget_gib']==21,'explicit budget alias changed')
    return value

def verify_ssl_audit(path,expected_sha,run,checkpoint):
    require(sha(path)==expected_sha,'independent SSL audit changed');value=read(path);run=Path(run).resolve()
    require(value['status']=='passed' and value['model_forward_calls']==0 and value['audit_source_sha256']==sha(Path(__file__).with_name('benchmark_ssl_audit.py')),'independent selection audit unavailable/different code')
    for name,digest in value['inputs_sha256'].items():require(sha(name)==digest,'audited SSL/model/source artifact changed')
    require(run.name in TASK_RUNS and value['selected_downstream_path_exposure'][TASK_RUNS[run.name]]['checkpoint_sha256']==checkpoint,'audited downstream best differs')
    require(sha(run.parent/'masked_ssl_v1/best.pt')==value['SSL_best_checkpoint_sha256'] and sha(run.parent/'masked_ssl_v1/selected_encoder.pt')==value['selected_encoder_sha256'],'sibling SSL selected artifacts differ')
    return value

def evaluate(args):
    phase2_code=Path(args.phase2_code).resolve();run=Path(args.run_dir).resolve();phase3=run.parent.parent
    seal=verify_scope(args.seal,args.expected_seal_sha256,args.phase2_seal,phase2_code)
    require(run.name in TASK_RUNS and run.parent.name=='runs','unregistered phase3 run')
    require(args.device=='cuda' and args.memory_budget_gib==21 and 0<args.max_seconds<=900,'runtime outside fixed evaluation budget')
    require(args.collect_routing==(run.name=='eegnet_ssl_smollm_fused_v1'),'only fused model collects the same16 routing prompts')
    require(sha(run/'best.pt')==args.checkpoint_sha256,'selected checkpoint differs from admission')
    config=read(run/'run_config.json');require(args.batch_size==config['validation_batch_size'],'evaluation batch differs from measured fixed batch')
    completed=read(phase3/'workflow/training_completion.json')
    require(completed['status']=='training_terminal_pending_independent_test_and_recovery' and completed['test_forward_calls']==0,'all training must finish before test')
    require(not (phase3/'workflow/active_owned_job.json').exists(),'training still owns its process marker')
    for key,name in [('scratch','eegnet_ssl_supervised_v1'),('fused','eegnet_ssl_smollm_fused_v1')]:
        require(sha(run.parent/name/'best.pt')==completed['models'][key]['best_checkpoint_sha256'],'a selected best changed')
    verified=verify_ssl_audit(args.ssl_audit,args.expected_ssl_audit_sha256,run,args.checkpoint_sha256)
    admission={'schema':'xavier-phase3-independent-evaluation-admission-v1','execution_seal_sha256':args.expected_seal_sha256,'checkpoint_sha256':args.checkpoint_sha256,
               'SSL_selection_audit_sha256':args.expected_ssl_audit_sha256,'initial_epoch_candidate_verified':True,'source_exclusion_and_selected_path_verified':True,
               'analysis_changes_from_phase2':False,'exploratory_test_reuse':True,'created_epoch':time.time()}
    save(Path(args.output).with_suffix('.admission.json'),admission)
    sys.path.insert(0,str(phase2_code));sys.path.insert(1,str(phase2_code.parent))
    frozen=importlib.import_module('benchmark_evaluate')
    require(Path(frozen.__file__).resolve()==phase2_code/'benchmark_evaluate.py','wrong evaluator module was imported')
    report=frozen.evaluate(args.prepared,args.seal,args.expected_seal_sha256,Path(__file__).with_name('runtime.py'),run,args.checkpoint_sha256,args.output,
                           args.device,args.batch_size,args.memory_budget_gib,args.max_seconds,args.collect_routing)
    verify_scope(args.seal,args.expected_seal_sha256,args.phase2_seal,phase2_code)
    verify_ssl_audit(args.ssl_audit,args.expected_ssl_audit_sha256,run,args.checkpoint_sha256)
    save(Path(args.output)/'phase3_context.json',{**admission,'report_sha256':sha(Path(args.output)/'evaluation_report.json'),
          'status':'completed','actual_variant_count':5,'same_test_records':1489,'same_paired_records':1254,'same_T0_records':1357,'freeform_intention_validated':False})
    return {'status':'completed','report_sha256':sha(Path(args.output)/'evaluation_report.json'),'phase3_context_sha256':sha(Path(args.output)/'phase3_context.json')}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('register')
    for name in ['phase2-seal','source-seal','budget','phase2-code','output']:r.add_argument('--'+name,required=True)
    e=sub.add_parser('evaluate')
    for name in ['prepared','seal','expected-seal-sha256','phase2-seal','phase2-code','run-dir','checkpoint-sha256','ssl-audit','expected-ssl-audit-sha256','output']:e.add_argument('--'+name,required=True)
    e.add_argument('--device',default='cuda');e.add_argument('--batch-size',type=int,required=True);e.add_argument('--memory-budget-gib',type=float,default=21)
    e.add_argument('--max-seconds',type=int,required=True);e.add_argument('--collect-routing',action='store_true');a=p.parse_args()
    value=register(a.phase2_seal,a.source_seal,a.budget,a.phase2_code,a.output) if a.command=='register' else evaluate(a)
    print(json.dumps(value))

if __name__=='__main__':main()
