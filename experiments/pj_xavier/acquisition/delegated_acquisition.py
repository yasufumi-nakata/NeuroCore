"""Small, reserved public acquisition using exact queued-row delegation.

Never modifies running/completed/leased original jobs. The original queue and
the delegated worker cannot concurrently select these files. Receipts are
immutable; data/checksum verification precedes reconciliation.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time


def require(x,message):
    if not x:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def immutable(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    b=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    temp=path.with_name(path.name+'.part')
    require(not path.exists() and not temp.exists(),'Receipt already exists')
    with temp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    os.link(temp,path);temp.unlink()


def identity(row):
    return {k:row[k] for k in ('id','catalog_id','dataset_key','relative_path','expected_bytes')}|{
        'checksum':json.loads(row['checksum_json'])}


def original_matches(original,selected):
    require(identity(original)==identity(selected),'Original source identity/size/checksum differs')
    for key in ('provider_file_id','provider_file_identity'):
        if original.get(key) and selected.get(key):
            require(original[key]==selected[key],'Pinned provider file identity differs')


def delegate(db, selected, raw_root, state_name, plan_path, commit_path):
    require(state_name=='delegated_compact_v1','Unexpected delegation scope')
    require(selected and len({x['id'] for x in selected})==len(selected),'Empty/repeated selected jobs')
    db.row_factory=sqlite3.Row
    db.execute('BEGIN IMMEDIATE')
    try:
        originals=[]
        for job in selected:
            found=db.execute('SELECT * FROM downloads WHERE id=?',(job['id'],)).fetchone()
            require(found is not None,'Expected source job absent')
            row=dict(found);original_matches(row,job)
            require(row['status']=='queued' and row['lease_until']==0,'Only exact unleased queued rows can be delegated')
            path=Path(raw_root)/row['dataset_key']/row['relative_path']
            require(not path.exists() and not path.with_name(path.name+'.part').exists() and not path.with_name(path.name+'.part.meta.json').exists(),
                    'Existing complete/partial prevents a new compact delegation')
            originals.append(row)
        immutable(plan_path,{'status':'precommit_original_state','created_epoch':time.time(),
                            'target_status':state_name,'original_rows':originals})
        for row in originals:
            changed=db.execute('UPDATE downloads SET status=?,updated_at=? WHERE id=? AND status=\'queued\' AND lease_until=0',
                               (state_name,time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),row['id'])).rowcount
            require(changed==1,'Concurrent status change; rollback')
        db.commit()
    except BaseException:
        db.rollback();raise
    receipt={'status':'delegated','job_count':len(selected),'original_state_sha256':sha(plan_path),
             'job_ids':[r['id'] for r in selected],'target_status':state_name,
             'running_complete_or_leased_jobs_changed':0,'committed_epoch':time.time()}
    immutable(commit_path,receipt)
    return receipt


def reconcile(db, verified, raw_root, acquire, receipt_path):
    # Rehash before acquiring SQLite writer lock; recheck stat under lock.
    checked=[]
    for item in verified:
        path=Path(raw_root)/item['dataset_key']/item['relative_path']
        before=path.stat()
        digest=acquire.verify_file(path,item['expected_bytes'],json.loads(item['checksum_json']))
        after=path.stat()
        require((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'File changed during verification')
        require(item['status']=='complete' and digest==item['sha256'],'Delegated file not checksum verified complete')
        checked.append((item,path,(after.st_ino,after.st_size,after.st_mtime_ns)))
    db.row_factory=sqlite3.Row;db.execute('BEGIN IMMEDIATE')
    try:
        for item,path,observed in checked:
            original=dict(db.execute('SELECT * FROM downloads WHERE id=?',(item['id'],)).fetchone())
            original_matches(original,item)
            require(original['status']=='delegated_compact_v1' and original['lease_until']==0,'Original job is no longer unleased delegated')
            st=path.stat();require((st.st_ino,st.st_size,st.st_mtime_ns)==observed,'Verified local file stat changed')
            changed=db.execute("UPDATE downloads SET status='complete',bytes_present=?,sha256=?,error=NULL,updated_at=? WHERE id=? AND status='delegated_compact_v1' AND lease_until=0",
                (st.st_size,item['sha256'],time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),item['id'])).rowcount
            require(changed==1,'Reconciliation race')
        db.commit()
    except BaseException:db.rollback();raise
    value={'status':'reconciled','job_count':len(checked),'bytes':sum(x[0]['expected_bytes'] for x in checked),
           'verified_ids':[x[0]['id'] for x in checked],'running_complete_or_leased_original_jobs_changed':0,'completed_epoch':time.time()}
    immutable(receipt_path,value)
    return value


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resource=json.loads(Path(args.resource).read_text())
        require(resource['status']=='authorized_registered' and time.time()<resource['absolute_deadline_epoch'],'Budget unregistered/expired')
        require(resource['concurrent_workers']==1,'Exactly one compact worker required')
        spec=importlib.util.spec_from_file_location('fixed_public_acquirer',args.acquire_source)
        a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
        records=[json.loads(x) for x in Path(args.manifest).read_text().splitlines()]
        require(sorted(r['catalog_id'] for r in records)==sorted(resource['catalog_ids']),'Budget source scope differs')
        require(sum(len(r['files']) for r in records)==resource['expected_files'],'File count differs')
        expected=sum(f['size_bytes'] for r in records for f in r['files'])
        require(expected==resource['known_source_bytes'],'Whole-release bytes differ')
        reserve=resource['effective_download_reserve_bytes']
        require(shutil.disk_usage(args.raw_root).free>=reserve+expected,'Whole releases do not fit reserved space')
        db_path=out/'downloads.sqlite'
        require(not db_path.exists(),'New delegated run requires a fresh node-local DB')
        db=a.init_db(db_path);a.ingest(db,records,quarantine_conflicts=False);db.row_factory=sqlite3.Row
        selected=[dict(r) for r in db.execute('SELECT * FROM downloads ORDER BY id')]
        require(len(selected)==resource['expected_files'],'Selected rows incomplete')
        original=sqlite3.connect(args.original_db,timeout=30)
        delegate(original,selected,args.raw_root,'delegated_compact_v1',out/'original_state.json',out/'delegation_receipt.json')
        immutable(out/'worker_launch.json',{'pid':os.getpid(),'started_epoch':time.time(),'absolute_deadline_epoch':resource['absolute_deadline_epoch'],
            'source_sha256':sha(__file__),'acquire_source_sha256':sha(args.acquire_source),'manifest_sha256':sha(args.manifest),
            'resource_sha256':sha(args.resource),'maximum_received_payload_bytes':resource['maximum_received_payload_bytes'],
            'whole_release_bytes_reserved':expected,'effective_reserve_bytes':reserve})
        traffic=0;attempts=[]
        while time.time()<resource['absolute_deadline_epoch'] and traffic<resource['maximum_received_payload_bytes']:
            remaining=max(1,math.floor(resource['absolute_deadline_epoch']-time.time()))
            result=a.worker(db,args.raw_root,reserve_gib=reserve/2**30,seconds=remaining,
                max_bytes=resource['maximum_received_payload_bytes']-traffic)
            traffic+=result['bytes_transferred_this_run'];attempts.append(result)
            states={r[0] for r in db.execute('SELECT status FROM downloads')}
            print(json.dumps({'time':time.time(),'received_payload_bytes':traffic,'states':sorted(states)}),flush=True)
            if states=={'complete'}:break
            if states-{'complete','queued','paused','retry'}:break
            retry=db.execute("SELECT min(retry_at) FROM downloads WHERE status IN ('queued','paused','retry')").fetchone()[0]
            if retry is None:break
            sleep=min(30,max(1,retry-time.time()),max(0,resource['absolute_deadline_epoch']-time.time()))
            if sleep:time.sleep(sleep)
        final=[dict(r) for r in db.execute('SELECT * FROM downloads ORDER BY id')]
        complete=all(r['status']=='complete' for r in final)
        if complete:
            for job in final:
                path=Path(args.raw_root)/job['dataset_key']/job['relative_path']
                require(a.verify_file(path,job['expected_bytes'],json.loads(job['checksum_json']))==job['sha256'],'Final complete checksum differs')
        report={'status':'all_files_downloaded_checksum_verified' if complete else 'incomplete',
            'received_payload_bytes_this_run':traffic,'files':len(final),'complete_files':sum(r['status']=='complete' for r in final),
            'complete_bytes':sum(r['expected_bytes'] for r in final if r['status']=='complete'),'attempt_reports':attempts,
            'header_audit_pending':True,'original_reconciliation_performed':False,'finished_epoch':time.time()}
        immutable(out/'download_completion.json',report);db.close();original.close()
        return report


def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','original-db','raw-root','resource','output','acquire-source'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    from smac_storage_guard import require_smac_raw_root
    require_smac_raw_root(args.raw_root)
    print(json.dumps(run(args),sort_keys=True),flush=True)


if __name__=='__main__':main()
