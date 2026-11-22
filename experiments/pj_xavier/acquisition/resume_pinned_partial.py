"""One-file continuation with strict206, immutable budget, and source custody.

The old worker must be absent and its target unleased. Only this original row
is delegated/reconciled. A200 Range fallback never truncates the old partial.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,importlib.util,json,math,os,re,shutil,sqlite3,time
from pathlib import Path


def require(ok,reason):
    if not ok:raise ValueError(reason)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def save(path,obj):
    p=Path(path);temp=p.with_name(p.name+'.part')
    with temp.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(temp,p);temp.unlink()


def check_range_response(response,headers,total):
    range_header=(headers or {}).get('Range','')
    m=re.fullmatch(r'bytes=(\d+)-',range_header)
    require(m is not None,'Strict continuation requires a Range request')
    start=int(m[1]);match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
    require(response.status==206,'Strict206 continuation refused full/restarted response; existing prefix retained')
    require(match and int(match[1])==start and start<=int(match[2])<total and int(match[3])==total,'Strict206 range start/end/total mismatch')
    length=response.headers.get('Content-Length')
    require(length is None or int(length)==int(match[2])-start+1,'Strict206 declared payload length mismatch')
    return {'status':206,'start':start,'end':int(match[2]),'total':total,'response_length':None if length is None else int(length)}


class StrictRangeHTTP:
    def __init__(self,underlying,total,evidence):self.underlying=underlying;self.total=total;self.evidence=Path(evidence)
    def request(self,url,headers=None):
        response=self.underlying.request(url,headers)
        try:
            record=check_range_response(response,headers,self.total)
            with self.evidence.open('a') as f:f.write(json.dumps({'time':time.time(),**record})+'\n');f.flush();os.fsync(f.fileno())
            return response
        except BaseException:response.close();raise


def pinned(row):return {k:row[k] for k in ['id','catalog_id','dataset_key','relative_path','url','expected_bytes','checksum_json','provider_file_id','provider_file_identity']}


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    budget=json.loads(Path(args.resource).read_text());deadline=budget['absolute_deadline_epoch']
    require(budget['status']=='authorized_registered_waiting_original_worker_terminal' and time.time()<deadline,'Budget not active')
    require(budget['max_files']==budget['max_workers']==1,'One-file/one-worker budget required')
    try:os.kill(budget['original_bulk_PID'],0)
    except ProcessLookupError:pass
    else:raise ValueError('Original worker PID still exists; do not acquire its partial')
    spec=importlib.util.spec_from_file_location('pinned_existing_acquirer',args.acquire_source);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    require(sha(args.acquire_source)==args.acquire_source_sha256,'New fixed acquisition source changed')
    require(sha(args.original_launch)==budget['original_launch_sha256'],'Original launch receipt changed')
    original=sqlite3.connect(args.original_db,timeout=30);original.row_factory=sqlite3.Row
    r=original.execute('select * from downloads where id=?',(args.job_id,)).fetchone();require(r is not None,'Original pinned job absent');job=dict(r)
    require(job['relative_path']==budget['selected_original_filename'] and job['expected_bytes']==budget['expected_whole_file_bytes'],'Budget identity/size mismatch')
    require(job['status'] in ['paused','retry','queued'] and job['lease_until']==0,'Original target is active/terminal/leased')
    path=Path(args.raw_root)/job['dataset_key']/job['relative_path'];partial=path.with_name(path.name+'.part');meta=path.with_name(path.name+'.part.meta.json')
    require(not path.exists() and partial.is_file() and not partial.is_symlink() and meta.is_file() and not meta.is_symlink(),'Expected unmodified regular partial missing')
    with path.with_name(path.name+'.continuation.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        st=partial.stat();size=st.st_size;remaining=job['expected_bytes']-size;md=json.loads(meta.read_text())
        require(0<size<job['expected_bytes'] and size==job['bytes_present'],'Stable partial length differs from old completed lease receipt')
        require(md['url_sha256']==hashlib.sha256(job['url'].encode()).hexdigest() and md['expected_bytes']==job['expected_bytes'],'Partial metadata/source mismatch')
        require(remaining<=budget['maximum_received_payload_bytes'],'Remaining full file exceeds new received-byte budget')
        require(shutil.disk_usage(partial.parent).free>=budget['effective_download_reserve_bytes']+remaining,'Remaining full file does not fit reserve')
        checksum=json.loads(job['checksum_json']);require(checksum and checksum['algorithm'] in ['md5','sha256','sha512'],'Published full checksum required')
        with partial.open('rb') as f:
            head=f.read(min(size,2**20));f.seek(max(0,size-2**20));tail=f.read()
        custody={'status':'stable_published_checksum_partial','observed_epoch':time.time(),'source_job':job,'partial_size':size,
          'partial_stat':[st.st_ino,st.st_size,st.st_mtime_ns],'prefix_1MiB_sha256':hashlib.sha256(head).hexdigest(),
          'suffix_1MiB_sha256':hashlib.sha256(tail).hexdigest(),'partial_metadata_sha256':sha(meta),
          'full_partial_SHA_not_claimed':True,'final_full_provider_checksum_required':True,
          'source_sha256':sha(__file__),'resource_sha256':sha(args.resource),'absolute_deadline_epoch':deadline,
          'original_acquisition_source_sha256':budget['original_acquisition_source_sha256'],
          'continuation_acquisition_source_sha256':sha(args.acquire_source),'original_launch_sha256':sha(args.original_launch)}
        strict=StrictRangeHTTP(a.Http(),job['expected_bytes'],out/'range_responses.jsonl')
        with strict.request(job['url'],{'Range':f'bytes={size}-','Accept-Encoding':'identity'}):pass
        now=partial.stat();require((now.st_ino,now.st_size,now.st_mtime_ns)==(st.st_ino,st.st_size,st.st_mtime_ns),'Partial changed during preflight')
        save(out/'original_partial_custody.json',custody)
        original.execute('BEGIN IMMEDIATE')
        try:
            check=dict(original.execute('select * from downloads where id=?',(args.job_id,)).fetchone())
            require(check==job,'Original target changed during continuation preflight')
            changed=original.execute("update downloads set status='delegated_BCI_continuation_v1',updated_at=? where id=? and status=? and lease_until=0",(a.now(),job['id'],job['status'])).rowcount
            require(changed==1,'Original delegation race');original.commit()
        except BaseException:original.rollback();raise
        save(out/'delegation_receipt.json',{'status':'delegated','job_id':job['id'],'original_state_sha256':sha(out/'original_partial_custody.json'),'other_rows_changed':0})
        db=a.init_db(out/'downloads.sqlite');db.row_factory=sqlite3.Row;cols=list(job);copy={**job,'status':'paused','attempts':0,'retry_at':0.,'lease_until':0.}
        db.execute('insert into downloads ('+','.join(cols)+') values('+','.join('?' for _ in cols)+')',[copy[k] for k in cols]);db.commit()
        save(out/'worker_launch.json',{'pid':os.getpid(),'started_epoch':time.time(),'absolute_deadline_epoch':deadline,'initial_partial_size':size,
          'full_remaining_bytes_reserved':remaining,'maximum_received_payload_bytes':budget['maximum_received_payload_bytes'],'resource_sha256':sha(args.resource),'source_sha256':sha(__file__)})
        traffic=0;attempts=[]
        while time.time()<deadline and traffic<budget['maximum_received_payload_bytes']:
            current=dict(db.execute('select * from downloads').fetchone())
            if current['retry_at']>time.time():time.sleep(min(30,current['retry_at']-time.time(),max(0,deadline-time.time())));continue
            report=a.worker(db,args.raw_root,reserve_gib=budget['effective_download_reserve_bytes']/2**30,seconds=max(1,math.floor(deadline-time.time())),max_bytes=budget['maximum_received_payload_bytes']-traffic,max_files=1,http=strict)
            traffic+=report['bytes_transferred_this_run'];attempts.append(report)
            current=dict(db.execute('select * from downloads').fetchone())
            print(json.dumps({'time':time.time(),'status':current['status'],'bytes_present':current['bytes_present'],'received_bytes_this_budget':traffic}),flush=True)
            if current['status'] not in ['paused','retry']:break
        current=dict(db.execute('select * from downloads').fetchone())
        # A transfer cap reached exactly at the last byte can pause before EOF.
        # Finalize the existing complete prefix without receiving another byte.
        if current['status'] in ['paused','retry'] and partial.exists() and partial.stat().st_size==job['expected_bytes'] and time.time()<deadline:
            verified=a.transfer(current,args.raw_root,strict,budget['effective_download_reserve_bytes'],time.monotonic()+max(0,deadline-time.time()),0)
            require(verified['status']=='complete' and verified['transferred_bytes']==0,'Full partial finalization attempted new traffic')
            db.execute("update downloads set status='complete',bytes_present=?,sha256=?,error=NULL,lease_until=0 where id=?",(verified['bytes_present'],verified['sha256'],job['id']));db.commit()
            current=dict(db.execute('select * from downloads').fetchone())
        result={'status':current['status'],'job_id':job['id'],'initial_partial_size':size,'received_bytes_this_budget':traffic,'final_size':current['bytes_present'],
          'full_SHA256':current['sha256'],'attempts':attempts,'finished_epoch':time.time(),'archive_header_audit_pending':True,'other_original_rows_changed':0}
        if current['status']=='complete':
            require(path.stat().st_size==job['expected_bytes'] and current['sha256'],'Complete size/SHA absent')
            # transfer already computed both the published MD5 and full SHA.
            # Rehash independently before updating the original single row.
            require(a.verify_file(path,job['expected_bytes'],checksum)==current['sha256'],'Independent complete checksum failed')
            original.execute('BEGIN IMMEDIATE')
            try:
                old=dict(original.execute('select * from downloads where id=?',(job['id'],)).fetchone())
                require(pinned(old)==pinned(job) and old['status']=='delegated_BCI_continuation_v1' and old['lease_until']==0,'Original reconciliation scope changed')
                original.execute("update downloads set status='complete',bytes_present=?,sha256=?,error=NULL,updated_at=? where id=?",(job['expected_bytes'],current['sha256'],a.now(),job['id']));original.commit()
            except BaseException:original.rollback();raise
            result['original_target_reconciled_after_full_checksum']=True
        save(out/'completion.json',result);return result


def main():
    p=argparse.ArgumentParser()
    for name in ['original-db','job-id','raw-root','resource','output','acquire-source','acquire-source-sha256','original-launch']:p.add_argument('--'+name,required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True),flush=True)
if __name__=='__main__':main()
