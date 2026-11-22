"""Resume an already-delegated stopped run within its original time/traffic caps.

Capacity reservations belonging to a now-complete other source may be released
only by a new immutable resource receipt. Original receipt, DB/file identities,
partials, previous received traffic, and absolute deadline remain binding.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time
import acquire
from delegated_acquisition_v2 import immutable, require, sha, original_matches


def validate_budget(old,new,previous):
    require(old['status']==new['status']=='authorized_registered','Unregistered budget')
    require(new['absolute_deadline_epoch']==old['absolute_deadline_epoch'],'Original absolute deadline must not extend')
    require(new['catalog_ids']==old['catalog_ids'] and new['expected_files']==old['expected_files']
            and new['known_source_bytes']==old['known_source_bytes'],'Source scope changed')
    require(new['manifest_sha256']==old['manifest_sha256'],'Pinned manifest changed')
    used=previous['received_payload_bytes_this_run']
    require(isinstance(used,int) and used>=0,'Previous actual received traffic absent')
    require(0<=new['maximum_received_payload_bytes']<=old['maximum_received_payload_bytes']-used,'Original traffic limit exceeded')
    require(new['effective_download_reserve_bytes']>=36*2**30,'36GiB raw/recovery reserve required')
    require(new['concurrent_workers']==old['concurrent_workers']==1,'One worker per delegated ledger')
    require(previous['status']=='incomplete','Resume only a terminal incomplete receipt')
    return used


def validate_rows(rows,expected_files):
    require(len(rows)==expected_files and len({r['id'] for r in rows})==len(rows),'Incomplete or repeated ledger')
    require(all(r['lease_until']==0 and r['status'] in {'queued','retry','paused','capacity_blocked','complete'} for r in rows),'Active/held/integrity-failed row prohibits resume')


def run(args):
    oldroot=Path(args.operation_root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    with (oldroot/'worker.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=json.loads(Path(args.old_resource).read_text());new=json.loads(Path(args.resource).read_text());previous=json.loads((oldroot/'download_completion.json').read_text())
        used=validate_budget(old,new,previous);require(time.time()<new['absolute_deadline_epoch'],'Original deadline expired')
        require(sha(args.old_resource)==new['original_resource_sha256'] and sha(oldroot/'download_completion.json')==new['previous_completion_sha256'],'Resume receipt lineage mismatch')
        other=json.loads(Path(args.released_reservation_completion).read_text())
        require(other['status']=='all_files_downloaded_checksum_verified' and other['files']==other['complete_files'],'Other source reservation not fully released')
        require(sha(args.released_reservation_completion)==new['released_reservation_completion_sha256'],'Released reservation source changed')
        launch=json.loads((oldroot/'worker_launch.json').read_text())
        try:os.kill(int(launch['pid']),0)
        except ProcessLookupError:pass
        else:raise ValueError('Previous worker PID still exists; require explicit ownership review')
        db=sqlite3.connect(oldroot/'downloads.sqlite',timeout=30);db.row_factory=sqlite3.Row;rows=[dict(x) for x in db.execute('select * from downloads order by id')]
        validate_rows(rows,new['expected_files']);original=sqlite3.connect(Path(args.original_db).resolve().as_uri()+'?mode=ro',uri=True);original.row_factory=sqlite3.Row
        before=[];remaining=0
        for row in rows:
            original_row=dict(original.execute('select * from downloads where id=?',(row['id'],)).fetchone());original_matches(original_row,row)
            require(original_row['status']=='delegated_compact_v1' and original_row['lease_until']==0,'Original queue no longer exclusively delegated')
            target=acquire.destination(args.raw_root,row['dataset_key'],row['relative_path']);part=target.with_name(target.name+'.part')
            if row['status']=='complete':
                require(acquire.verify_file(target,row['expected_bytes'],json.loads(row['checksum_json']))==row['sha256'],'Completed file lost integrity')
                before.append({'id':row['id'],'status':'complete','sha256':row['sha256'],'bytes':row['expected_bytes']})
            else:
                require(not target.exists(),'Noncomplete row already has a final file')
                n=part.stat().st_size if part.exists() else 0;require(n<=row['expected_bytes'],'Partial exceeds source size')
                remaining+=row['expected_bytes']-n
                before.append({'id':row['id'],'status':row['status'],'partial_bytes':n,'partial_sha256':sha(part) if n else None})
        require(shutil.disk_usage(args.raw_root).free>=new['effective_download_reserve_bytes']+remaining,'Remaining whole scope no longer fits')
        immutable(out/'resume_preflight.json',{'status':'passed','remaining_source_bytes':remaining,'original_deadline_epoch':new['absolute_deadline_epoch'],'previous_received_bytes':used,'source_rows':before,'new_source_ingests':0,'original_queue_mutations':0})
        # Only node-local capacity holds can be released after the full remaining
        # scope fits. Source metadata, partial bytes, and original queue stay intact.
        db.execute('BEGIN IMMEDIATE')
        current=[dict(x) for x in db.execute('select * from downloads order by id')]
        require(current==rows,'Node-local ledger changed after reserve verification')
        released=db.execute("update downloads set status='queued',retry_at=0 where status='capacity_blocked' and lease_until=0").rowcount
        db.commit()
        immutable(out/'worker_launch.json',{'pid':os.getpid(),'started_epoch':time.time(),'source_sha256':sha(__file__),'acquire_source_sha256':sha(acquire.__file__),'resource_sha256':sha(args.resource),'original_resource_sha256':sha(args.old_resource),'deadline_epoch':new['absolute_deadline_epoch'],'previous_received_bytes':used,'remaining_received_budget':new['maximum_received_payload_bytes'],'local_capacity_holds_released':released})
        traffic=0;attempts=[]
        while time.time()<new['absolute_deadline_epoch'] and traffic<new['maximum_received_payload_bytes']:
            result=acquire.worker(db,args.raw_root,reserve_gib=new['effective_download_reserve_bytes']/2**30,seconds=max(1,min(180,math.floor(new['absolute_deadline_epoch']-time.time()))),max_bytes=new['maximum_received_payload_bytes']-traffic,max_files=1)
            traffic+=result['bytes_transferred_this_run'];attempts.append(result)
            states={r[0] for r in db.execute('select status from downloads')}
            if states=={'complete'}:break
            if states-{'complete','queued','retry','paused'}:break
            if result['files_processed']==0:
                retry=db.execute("select min(retry_at) from downloads where status in ('queued','retry','paused')").fetchone()[0]
                if retry is None:break
                time.sleep(min(30,max(5,retry-time.time()),max(0,new['absolute_deadline_epoch']-time.time())))
        final=[dict(x) for x in db.execute('select * from downloads order by id')];complete=all(x['status']=='complete' for x in final)
        if complete:
            for j in final:require(acquire.verify_file(acquire.destination(args.raw_root,j['dataset_key'],j['relative_path']),j['expected_bytes'],json.loads(j['checksum_json']))==j['sha256'],'Final source checksum differs')
        report={'status':'all_files_downloaded_checksum_verified' if complete else 'incomplete','files':len(final),'complete_files':sum(j['status']=='complete' for j in final),'complete_bytes':sum(j['expected_bytes'] for j in final if j['status']=='complete'),
                'received_payload_bytes_this_run':traffic,'previous_received_bytes':used,'total_received_bytes_original_plus_resume':used+traffic,'original_received_budget':old['maximum_received_payload_bytes'],'absolute_deadline_epoch':old['absolute_deadline_epoch'],
                'source_scope_unchanged':True,'original_queue_mutations':0,'finished_epoch':time.time(),'attempt_reports':attempts}
        require(used+traffic<=old['maximum_received_payload_bytes'],'Traffic accounting exceeded original cap')
        immutable(out/'download_completion.json',report);return report


def main():
    p=argparse.ArgumentParser()
    for n in ['operation-root','output','old-resource','resource','released-reservation-completion','original-db','raw-root']:p.add_argument('--'+n,required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True),flush=True)


if __name__=='__main__':main()
