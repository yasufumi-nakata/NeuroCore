"""Immutable all-artifact recovery and owned-session shutdown for phase3.

Calls Colab only through the workstation CLI. Raw ZIP and reproducible prepared
waveform caches remain excluded with exact identities; scientific metadata,
checkpoints, every optimizer/RNG state, predictions, controls, logs and source are
preserved. Export/transport directories are outside the scientific remote root.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile,time,zipfile
from pathlib import Path
REPO=Path.home()/'Documents/GitHub/NeuroCore-xavier-training'
PHASE=REPO/'private/xavier/eegmmidb_phase3_v1';OPS=PHASE/'recovery_ops_v1'
CLI=str(Path.home()/'.local/bin/colab');SESSION='xavier-eegmmidb-20261122-p2'
SERIAL=REPO/'private/xavier/eegmmidb_phase2_v1/operations/colab_serial.py'
REMOTE='/content/xavier_phase3';EXPORT='/content/xavier_phase3_export_v1'
DEADLINE=1795331083.750155;RESERVE=35*1024**3;MARKER='XAVIER_PHASE3_RECOVERY_JSON:'
REFERENCE_HELPER_SHA='f188f6a2dd66132545ff851ae683fa5ad39b37afa7c08ec966a2a3d401f0edc4'

def require(ok,text):
 if not ok:raise ValueError(text)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def read(path):return json.loads(Path(path).read_text())
def bounded_timeout(wanted):
 remaining=DEADLINE-time.time();require(remaining>1,'Phase3 original deadline reached');return min(wanted,remaining)
def remote(script,timeout=60,*,stop_only_deadline=None):
 if stop_only_deadline is None:seconds=bounded_timeout(timeout)
 else:
  require(script==TERMINAL_PROBE_SCRIPT,'Postdeadline grace permits only the fixed terminal/stop proof')
  seconds=stop_timeout(timeout,stop_only_deadline)
 source='import json,hashlib,time,os,sys,subprocess,shutil,zipfile\nfrom pathlib import Path\n'+script+'\nprint('+repr(MARKER)+'+json.dumps(_result,allow_nan=False))\n'
 with tempfile.TemporaryDirectory(prefix='xavier_phase3_recovery_') as tmp:
  path=Path(tmp)/'request.py';path.write_text(source)
  done=subprocess.run([sys.executable,str(SERIAL),'exec','--session',SESSION,'--file',str(path),'--timeout',str(max(1,int(seconds)-5))],capture_output=True,text=True,stdin=subprocess.DEVNULL,timeout=seconds)
 require(done.returncode==0,'Serialized Colab exec failed; raw API response withheld')
 for line in reversed(done.stdout.splitlines()):
  if MARKER in line:return json.loads(line.split(MARKER,1)[1])
 raise RuntimeError('Remote receipt marker missing; raw API response withheld')

REMOTE_PACKAGE=r'''
root=Path('/content/xavier_phase3');export=Path('/content/xavier_phase3_export_v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
training=read(root/'workflow/training_completion.json')
assert training['status']=='training_terminal_pending_independent_test_and_recovery' and training['test_forward_calls']==0
assert not (root/'workflow/active_owned_job.json').exists()
# This attestation is emitted by the independent benchmark owner after both
# fixed-checkpoint five-condition evaluations and the saved-loss SSL audit.
attestation=read(ATTESTATION)
assert attestation['status']=='completed' and attestation['new_training_or_parameter_selection']==False
assert set(attestation['models'])=={'scratch','fused'}
for kind,name in [('scratch','eegnet_ssl_supervised_v1'),('fused','eegnet_ssl_smollm_fused_v1')]:
 run=root/'runs'/name;summary=read(run/'run_summary.json');checkpoint=sha(run/'best.pt')
 assert checkpoint==summary['best_checkpoint_sha256']==training['models'][kind]['best_checkpoint_sha256']==attestation['models'][kind]['checkpoint_sha256']
 item=attestation['models'][kind];directory=(root/item['evaluation_relative_path']).resolve();assert directory.is_relative_to(root)
 completion=read(directory/'completion.json');report=read(directory/'evaluation_report.json')
 assert completion['status']=='completed' and completion['exit_code']==0 and completion['checkpoint_sha256']==checkpoint
 assert completion['report_sha256']==sha(directory/'evaluation_report.json')==item['report_sha256']
 assert set(completion['variant_names'])=={'fused','context_only','eeg_only','eeg_shuffled','eeg_time_shifted'}
 assert report['full_test_records']==1489 and report['metadata_paired_subset_records']==1254
 assert report['resources']['status']=='passed'
auditpath=(root/attestation['SSL_audit_relative_path']).resolve();assert auditpath.is_relative_to(root)
assert sha(auditpath)==attestation['SSL_audit_sha256']
audit=read(auditpath);assert audit['status']=='passed'
for key in ('earliest_minimum_including_epoch0_verified','all_validation_IDs_and_checkpoint_hashes_verified','source_person_and_payload_exclusions_verified','same_seeded_initial_encoder_verified','exported_encoder_tensors_equal_selected_checkpoint'):assert audit[key] is True
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
  argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 owned=any(x.startswith(('/content/xavier_phase2/','/content/xavier_phase3/')) and x.endswith('.py') for x in argv)
 assert state in ('Z','X') or not owned,'owned phase3 process remains active '+proc.name
for launch in (root/'workflow').glob('*.launch.json'):
 terminal=launch.with_name(launch.name.replace('.launch.json','.exit.json'));assert terminal.exists()
 a=read(launch);b=read(terminal);assert a['pid']==b['pid'] and a['command']==b['command']
assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
if export.exists():
 existing=read(export/'ledger.json');archive=export/'phase3_artifacts.zip'
 assert archive.stat().st_size==existing['archive_bytes'] and sha(archive)==existing['archive_sha256']
 for chunk in existing['chunks']:
  p=Path(chunk['remote_path']);assert p.resolve().is_relative_to(export) and p.stat().st_size==chunk['bytes'] and sha(p)==chunk['sha256']
 for name,item in existing['files'].items():
  p=(root/name).resolve();assert p.is_relative_to(root) and p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 return existing
artifacts=read(root/'prepared/artifacts.json');excluded={}
for relative,item in artifacts.items():
 p=(root/'prepared'/relative).resolve();assert p.is_relative_to(root/'prepared/waveforms')
 assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 excluded[str(p.relative_to(root))]={**item,'reason':'regenerable prepared EEG; source/contracts/runtime and all exact array hashes preserved'}
raw=root/'raw/eegmmidb-1.0.0.zip';assert raw.stat().st_size==1997435063 and sha(raw)=='0ff5f7f38d1d14b8e485581ad4cf5b327e6494bc2ac7c7d7d4f7e9518a950cbe'
excluded[str(raw.relative_to(root))]={'sha256':sha(raw),'bytes':raw.stat().st_size,'reason':'identical verified raw release already on workstation'}
files={}
for path in sorted(root.rglob('*')):
 if path.is_dir():continue
 assert path.is_file() and not path.is_symlink(),'nonregular scientific artifact'
 relative=str(path.relative_to(root))
 if relative in excluded:continue
 assert not path.name.endswith(('.tmp','.part')),'incomplete artifact '+relative
 files[relative]={'sha256':sha(path),'bytes':path.stat().st_size}
assert files and not set(files)&set(excluded)
for name in ['prepared/manifest.jsonl','prepared/artifacts.json','prepared/source_allowlist.jsonl','prepared/source_denylist.jsonl','source_seal_v1.json','gpu_budget_v1.json','runs/masked_ssl_v1/selected_encoder_seal.json','operations/head_equivalence_combined.json']:
 assert name in files,'mandatory artifact missing: '+name
capacity={}
for kind in ('ssl','scratch','fused'):
 ready=read(root/'workflow'/(kind+'_capacity_ready.json'))
 for path,digest in ready['report_sha256'].items():
  p=Path(path).resolve();assert p.is_relative_to(root);key=str(p.relative_to(root));assert files[key]['sha256']==digest
  capacity[key]=digest
uncompressed=sum(item['bytes'] for item in files.values())
assert shutil.disk_usage(root).free>35*1024**3+2*uncompressed+64*1024**2
export.mkdir();archive=export/'phase3_artifacts.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for name,item in sorted(files.items()):
  p=root/name;before=p.stat();assert before.st_size==item['bytes'];z.write(p,name)
  after=p.stat();assert after.st_size==before.st_size and after.st_mtime_ns==before.st_mtime_ns and sha(p)==item['sha256']
chunks=[]
with archive.open('rb') as f:
 index=0
 while True:
  data=f.read(48*1024**2)
  if not data:break
  p=export/('chunk_%04d.bin'%index);p.write_bytes(data);chunks.append({'remote_path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});index+=1
ledger={'schema':'xavier-phase3-remote-artifact-export-v1','status':'packaged','archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'uncompressed_bytes':uncompressed,'member_count':len(files),'files':files,'chunks':chunks,'excluded_regenerable_artifacts':excluded,'checkpoint_sha256':{k:v['checkpoint_sha256'] for k,v in attestation['models'].items()},'SSL_encoder_seal_sha256':sha(root/'runs/masked_ssl_v1/selected_encoder_seal.json'),'SSL_audit_sha256':sha(auditpath),'benchmark_attestation_sha256':sha(ATTESTATION),'referenced_capacity_reports':capacity,'owned_training_and_evaluation_processes_terminal':True,'GPU_compute_processes_empty':True,'created_epoch':time.time()}
(export/'ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
return ledger
'''

def package(attestation):
 require(attestation.startswith(REMOTE+'/'),'Attestation outside phase3')
 script=REMOTE_PACKAGE.replace('ATTESTATION',repr(attestation))
 script='def _package_phase3():\n'+'\n'.join(' '+line for line in script.splitlines())+'\n_result=_package_phase3()\n'
 result=remote(script,timeout=180);target=OPS/'remote_export_ledger.json'
 if target.exists():require(read(target)==result,'Existing export identity differs')
 else:write(target,result)
 return {k:v for k,v in result.items() if k not in ('files','excluded_regenerable_artifacts')}

def download(remote_path,target,expected):
 require(remote_path.startswith(EXPORT+'/'),'Download outside owned export')
 target=Path(target);require(not target.exists(),'Preserve existing chunk')
 require(shutil.disk_usage(target.parent).free>RESERVE+expected['bytes'],'Insufficient reserve')
 with tempfile.TemporaryFile() as log:
  child=subprocess.Popen([sys.executable,str(SERIAL),'download',remote_path,str(target),'--session',SESSION],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  end=time.time()+bounded_timeout(240)
  try:
   while child.poll() is None:
    require(time.time()<end and shutil.disk_usage(target.parent).free>RESERVE,'Download time/disk bound');time.sleep(1)
   require(child.returncode==0,'Download failed with exit'+str(child.returncode))
  except BaseException:
   if child.poll() is None:
    child.terminate()
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:child.kill();child.wait()
   raise
 require(target.stat().st_size==expected['bytes'] and sha(target)==expected['sha256'],'Chunk size/hash mismatch')

def extract_archive(archive,ledger,destination,staging):
 destination,staging=Path(destination).resolve(),Path(staging).resolve();recovered={}
 with zipfile.ZipFile(archive) as z:
  names=z.namelist();require(len(names)==len(set(names)) and set(names)==set(ledger['files']),'Archive membership differs')
  for name in names:
   parts=Path(name);require(not parts.is_absolute() and '..' not in parts.parts and '\\' not in name,'Unsafe archive path')
   item=ledger['files'][name];info=z.getinfo(name);require(info.file_size==item['bytes'] and not info.is_dir(),'Archive size/type differs')
   require((info.external_attr>>16 & 0o170000) in (0,0o100000),'Nonregular ZIP member')
   dest=destination/name;require(dest.resolve().is_relative_to(destination) and not dest.is_symlink(),'Destination escaped')
   dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():
    if not (dest.stat().st_size==item['bytes'] and sha(dest)==item['sha256']):
     conflict=staging/'conflicts'/name;conflict.parent.mkdir(parents=True,exist_ok=True)
     if not conflict.exists():
      with z.open(name) as src,conflict.open('xb') as out:shutil.copyfileobj(src,out,1024**2)
     require(conflict.stat().st_size==item['bytes'] and sha(conflict)==item['sha256'],'Conflict copy differs')
     receipt=staging/'conflict_receipt.json'
     if not receipt.exists():write(receipt,{'status':'blocked_conflict','original':str(dest),'original_sha256':sha(dest),'retained_remote_copy':str(conflict),'remote_sha256':item['sha256'],'original_overwritten':False})
     raise ValueError('Original and conflicting remote evidence retained separately')
   else:
    part=dest.with_name(dest.name+'.recovery.part');require(not part.exists(),'Preserve interrupted extraction')
    with z.open(name) as src,part.open('xb') as out:shutil.copyfileobj(src,out,1024**2)
    require(part.stat().st_size==item['bytes'] and sha(part)==item['sha256'],'Extracted member mismatch')
    os.link(part,dest);part.unlink()
   recovered[name]={**item,'local_path':str(dest)}
 return recovered

def recover():
 ledger=read(OPS/'remote_export_ledger.json');staging=PHASE/'recovery_v1';staging.mkdir(exist_ok=True)
 require(shutil.disk_usage(staging).free>=RESERVE+2*ledger['archive_bytes']+ledger['uncompressed_bytes']+64*1024**2,'Chunks/archive/extraction would cross35GiB reserve')
 raw=REPO/'private/xavier/raw/physionet-eegmmidb-1.0.0/eegmmidb-1.0.0.zip'
 raw_item=ledger['excluded_regenerable_artifacts']['raw/eegmmidb-1.0.0.zip']
 require(raw.stat().st_size==raw_item['bytes'] and sha(raw)==raw_item['sha256'],'Excluded raw has no identical local original')
 chunks=[]
 for index,item in enumerate(ledger['chunks']):
  bounded_timeout(1);path=staging/f'chunk_{index:04d}.bin'
  if path.exists():require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Existing partial chunk preserved')
  else:download(item['remote_path'],path,item)
  chunks.append(path)
 archive=staging/'phase3_artifacts.zip'
 if not archive.exists():
  with archive.open('xb') as out:
   for path in chunks:
    with path.open('rb') as source:shutil.copyfileobj(source,out,1024**2)
 require(archive.stat().st_size==ledger['archive_bytes'] and sha(archive)==ledger['archive_sha256'],'Complete archive identity differs')
 require(shutil.disk_usage(staging).free>=RESERVE+ledger['uncompressed_bytes'],'Extraction reserve insufficient')
 recovered=extract_archive(archive,ledger,PHASE,staging)
 receipt={'schema':'xavier-phase3-complete-artifact-recovery-v1','status':'completed','all_member_hashes_verified':True,'member_count':len(recovered),'files':recovered,
          'checkpoint_sha256':ledger['checkpoint_sha256'],'SSL_encoder_seal_sha256':ledger['SSL_encoder_seal_sha256'],'SSL_audit_sha256':ledger['SSL_audit_sha256'],
          'archive_sha256':ledger['archive_sha256'],'archive_bytes':ledger['archive_bytes'],'archive_path':str(archive),'uncompressed_bytes':ledger['uncompressed_bytes'],
          'completed_epoch':time.time(),'session_stopped':False,'referenced_capacity_reports':ledger['referenced_capacity_reports'],
          'excluded_regenerable_artifacts':ledger['excluded_regenerable_artifacts'],'original_raw_local_sha256':sha(raw),'original_raw_local_path':str(raw),
          'owned_training_and_evaluation_processes_terminal':True,'GPU_compute_processes_empty_at_export':True,
          'source_helper_sha256':sha(__file__),'reference_safe_export_helper_sha256':REFERENCE_HELPER_SHA}
 path=PHASE/'phase3_recovery_receipt_v1.json';write(path,receipt)
 return {'status':'completed','receipt_sha256':sha(path),'member_count':len(recovered),'checkpoint_sha256':receipt['checkpoint_sha256'],'session_stopped':False}

def verify_recovery():
 path=PHASE/'phase3_recovery_receipt_v1.json';receipt=read(path)
 require(receipt['status']=='completed' and receipt['all_member_hashes_verified'] and receipt['member_count']==len(receipt['files']),'Recovery ledger incomplete')
 for name,item in receipt['files'].items():
  file=Path(item['local_path']);require(file.resolve()==(PHASE/name).resolve() and file.resolve().is_relative_to(PHASE.resolve()),'Recovered path differs')
  require(file.stat().st_size==item['bytes'] and sha(file)==item['sha256'],'Recovered evidence changed: '+name)
 require(shutil.disk_usage(PHASE).free>=RESERVE,'Local35GiB reserve absent')
 return {'status':'passed','recovery_receipt_sha256':sha(path),'member_count':receipt['member_count'],'verified_epoch':time.time()}

TERMINAL_PROBE_SCRIPT="\nroot=Path('/content/xavier_phase3')\nactive=[]\nfor proc in Path('/proc').iterdir():\n if not proc.name.isdigit():continue\n try:\n  state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]\n  argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]\n except (FileNotFoundError,PermissionError,ProcessLookupError):continue\n if state not in ('Z','X') and any(x.startswith(('/content/xavier_phase2/','/content/xavier_phase3/')) and x.endswith('.py') for x in argv):active.append({'pid':int(proc.name),'argv':argv})\ngpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()\nassert not active and not gpu,'Owned compute not terminal'\n_result={'status':'terminal','owned_active_processes':active,'GPU_compute_processes':gpu,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'observed_epoch':time.time()}\n"


def stop_timeout(wanted,deadline):
 remaining=deadline-time.time()
 require(0<remaining<=180,'Stop-only bounded grace expired')
 return min(wanted,remaining)

def stop():
 verified=verify_recovery();stop_start=time.time();stop_deadline=stop_start+180
 proof=remote(TERMINAL_PROBE_SCRIPT,timeout=45,stop_only_deadline=stop_deadline)
 started=OPS/'session_stop_started.json';write(started,{'session':SESSION,'started_epoch':time.time(),'recovery':verified,'remote_terminal':proof,'original_compute_deadline_epoch':DEADLINE,'stop_only_grace_deadline_epoch':stop_deadline,'overrun_seconds':max(0.,stop_start-DEADLINE),'postdeadline_training_or_download_authorized':False})
 done=subprocess.run([CLI,'stop','--session',SESSION],capture_output=True,text=True,stdin=subprocess.DEVNULL,timeout=stop_timeout(90,stop_deadline))
 require(done.returncode==0,'Session stop CLI failed; start receipt preserved')
 sessions=subprocess.run([CLI,'sessions'],capture_output=True,text=True,stdin=subprocess.DEVNULL,timeout=stop_timeout(45,stop_deadline))
 require(sessions.returncode==0,'Cannot independently observe sessions after stop')
 require(SESSION not in sessions.stdout,'Owned session still appears in active sessions')
 # Keep the complete CLI evidence private; no raw proxy/API content is printed.
 stoplog=OPS/'session_stop_cli.txt';stoplog.write_text(done.stdout+'\n'+done.stderr)
 sessionlog=OPS/'sessions_after_stop.txt';sessionlog.write_text(sessions.stdout+'\n'+sessions.stderr)
 value={'schema':'xavier-phase3-owned-session-stop-v1','status':'stopped','session':SESSION,'observed_epoch':time.time(),'stop_exit_code':done.returncode,
        'sessions_query_exit_code':sessions.returncode,'owned_session_absent_from_active_listing':True,'stop_cli_log_sha256':sha(stoplog),'sessions_log_sha256':sha(sessionlog),
        'all_mandatory_artifacts_recovered_before_stop':True,'recovery_receipt_sha256':verified['recovery_receipt_sha256'],'new_GPU_session_started':False,'original_compute_deadline_epoch':DEADLINE,'stop_only_grace_seconds':180,'stop_started_after_deadline_seconds':max(0.,stop_start-DEADLINE),'postdeadline_training_or_download_authorized':False}
 path=PHASE/'phase3_session_stop_receipt_v1.json';write(path,value);return value

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('package','recover','verify','stop'));p.add_argument('--attestation')
 args=p.parse_args();OPS.mkdir(parents=True,exist_ok=True)
 if args.action=='package':require(args.attestation is not None,'Benchmark completion attestation required');result=package(args.attestation)
 elif args.action=='recover':result=recover()
 elif args.action=='verify':result=verify_recovery()
 else:result=stop()
 print(json.dumps(result,ensure_ascii=False))
