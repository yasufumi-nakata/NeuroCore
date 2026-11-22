"""Read-only sample/payload audit of Figshare 9970619 v1 processed exports.

This source has continuous EEGLAB SET/FDT exports, not the original 512 Hz
BioSemi files. No event, amplitude unit, target class, or online causality is
inferred from its paper. Auxiliary EXG/Status bytes are kept separate.
"""
from __future__ import annotations
import argparse
import collections
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import time
import numpy as np
from scipy.io import loadmat
import acquire
from delegated_acquisition_v2 import immutable, require, sha

SCALP = 'Fp1 AF7 AF3 F1 F3 F5 F7 FT7 FC5 FC3 FC1 C1 C3 C5 T7 TP7 CP5 CP3 CP1 P1 P3 P5 P7 P9 PO7 PO3 O1 Iz Oz POz Pz CPz Fpz Fp2 AF8 AF4 AFz Fz F2 F4 F6 F8 FT8 FC6 FC4 FC2 FCz Cz C2 C4 C6 T8 TP8 CP6 CP4 CP2 P2 P4 P6 P8 P10 PO8 PO4 O2'.split()
CHANNELS = SCALP + ['EXG'+str(i) for i in range(1,9)] + ['Status']
FORMAT_URL = 'https://eeglab.org/tutorials/ConceptsGuide/Data_Structures.html'
PAPER_URL = 'https://doi.org/10.1371/journal.pone.0235309'


def integer(x, name):
    require(not isinstance(x, (bool,np.bool_)) and np.ndim(x)==0, name+' not scalar integer')
    n=float(x);require(math.isfinite(n) and n.is_integer() and n>0,name+' invalid')
    return int(n)


def header(e, set_name, fdt_bytes):
    nchan=integer(e['nbchan'],'nbchan');n=integer(e['pnts'],'pnts');trials=integer(e['trials'],'trials')
    fs=float(e['srate']);require(math.isfinite(fs) and fs==200,'Unexpected processed rate')
    require(nchan==73 and trials==1,'Unexpected processed array dimensions')
    names=[str(x['labels']) for x in e['chanlocs']]
    require(names==CHANNELS,'Unreviewed channel montage/order')
    require(re.fullmatch(r's[0-9]+_exp[1-4]\.set',set_name) is not None,'Unexpected source record name')
    companion=Path(set_name).with_suffix('.fdt').name
    require(isinstance(e['data'],str) and e['data']==companion and e.get('datfile')==companion,
            'Companion must be exact declared sibling; external paths prohibited')
    require(n*nchan*4==fdt_bytes,'FDT bytes do not exhaust float32 sample frames')
    require(np.size(e.get('event',[]))==0,'Nonempty event array requires separate ontology audit')
    require('unit' not in e and 'units' not in e,'Unexpected explicit unit needs separate review')
    require(all(not str(x.get('type','')) for x in e['chanlocs']),'Unexpected channel type metadata')
    xmin=float(e['xmin']);xmax=float(e['xmax'])
    require(xmin==0 and abs(xmax-(n-1)/fs)<1e-8,'Stored sample time coordinates disagree')
    return {'sample_count':n,'sample_rate_hz':fs,'signals':nchan,'channel_names':names,
        'EEG_indices':list(range(64)),'auxiliary_EXG_indices':list(range(64,72)),'status_index':72,
        'EEG_channels':64,'auxiliary_EXG_channels':8,'status_channels':1,'trials':trials,
        'declared_xmin_s':xmin,'declared_last_sample_time_s':xmax,'stored_sample_duration_s':n/fs,
        'original_EEG_event_count':0,'original_EEG_event_array_empty':True,
        'unit_status':'absent_in_SET; EEGLAB reader convention is not independent export calibration proof',
        'explicit_amplitude_unit':None,'EEG_channel_type_basis':'Fixed source-specific named scalp montage; not chanlocs.type',
        'EXG_role_status':'8 external channels; individual EOG/other roles not assigned',
        'stored_scalar_dtype':'<f4','companion_name':companion,
        'export_software_metadata':str(e.get('etc',{}).get('eeglabvers','')),
        'preprocessing_history':str(e.get('history',''))}


def stream_fdt(path, h, expected_sha):
    path=Path(path);before=path.stat();whole=hashlib.sha256();eeg=hashlib.sha256();aux=hashlib.sha256();status_hash=hashlib.sha256()
    nonfinite=np.zeros(73,dtype=np.int64);minimum=np.full(73,np.inf);maximum=np.full(73,-np.inf)
    status_values=collections.Counter();status_unique_overflow=False;transitions=0;noninteger=0;previous=None;done=0
    with path.open('rb') as f:
        while done<h['sample_count']:
            take=min(4096,h['sample_count']-done);raw=f.read(take*73*4)
            require(len(raw)==take*73*4,'Truncated FDT frame')
            whole.update(raw);x=np.frombuffer(raw,dtype='<f4').reshape(take,73)
            eeg.update(np.ascontiguousarray(x[:,:64]).tobytes());aux.update(np.ascontiguousarray(x[:,64:72]).tobytes());status_hash.update(np.ascontiguousarray(x[:,72]).tobytes())
            finite=np.isfinite(x);nonfinite+=(~finite).sum(axis=0)
            minimum=np.minimum(minimum,np.where(finite,x,np.inf).min(axis=0));maximum=np.maximum(maximum,np.where(finite,x,-np.inf).max(axis=0))
            s=x[:,72];ok=np.isfinite(s);noninteger+=int(np.count_nonzero(ok&(s!=np.rint(s))))
            transitions+=int(np.count_nonzero(s[1:]!=s[:-1]))
            if previous is not None:transitions+=int(s[0]!=previous)
            previous=s[-1]
            if not status_unique_overflow:
                u,n=np.unique(s[ok],return_counts=True);status_values.update({str(float(v)):int(c) for v,c in zip(u,n)})
                if len(status_values)>1024:status_unique_overflow=True;status_values.clear()
            done+=take
        require(f.read(1)==b'','Unexpected FDT tail')
    after=path.stat();require((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'FDT changed during audit')
    require(whole.hexdigest()==expected_sha,'FDT full SHA mismatch')
    return {'FDT_sha256':whole.hexdigest(),'EEG_payload_sha256':eeg.hexdigest(),'EXG_payload_sha256':aux.hexdigest(),'Status_payload_sha256':status_hash.hexdigest(),
        'EEG_sample_bytes':done*64*4,'EXG_sample_bytes':done*8*4,'Status_sample_bytes':done*4,
        'nonfinite_values_per_channel':[int(v) for v in nonfinite],
        'unscaled_stored_min_per_channel':[float(v) if np.isfinite(v) else None for v in minimum],
        'unscaled_stored_max_per_channel':[float(v) if np.isfinite(v) else None for v in maximum],
        'Status_value_sample_counts':dict(status_values),'Status_distinct_value_count_exceeds_1024':status_unique_overflow,
        'Status_noninteger_sample_count':noninteger,'Status_value_transition_count':transitions,
        'Status_transitions_are_validated_stimulus_events':False,'waveform_modified':False}


def run(args):
    from threadpoolctl import threadpool_limits
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    budget=json.loads(Path(args.resource).read_text());done=json.loads(Path(args.download_completion).read_text())
    require(done['status']=='all_files_downloaded_checksum_verified' and done['complete_files']==88,'Whole release not verified')
    require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline expired')
    require(shutil.disk_usage(args.raw_root).free>=36*2**30+32*2**20,'Audit disk reserve unavailable')
    db=sqlite3.connect(Path(args.ledger).resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    jobs=[dict(x) for x in db.execute('SELECT * FROM downloads ORDER BY relative_path')]
    require(len(jobs)==88 and all(j['status']=='complete' and j['dataset_key']=='figshare-9970619-1' and j['catalog_id']=='000668' for j in jobs),'Pinned release identity differs')
    protocol={'schema':'xavier_processed_SSVEP_header_audit_v1','source_sha256':sha(__file__),'acquire_source_sha256':sha(acquire.__file__),
        'resource_sha256':sha(args.resource),'download_completion_sha256':sha(args.download_completion),'absolute_deadline_epoch':budget['absolute_deadline_epoch'],
        'format_source':FORMAT_URL,'paper_source':PAPER_URL,'catalog_id':'000668','figshare_version':'9970619-v1',
        'dtype_policy':'EEGLAB separate FDT little-endian float32, channels fastest; sample count must exhaust whole file',
        'channel_policy':'Exact fixed 64 named scalp EEG, 8 EXG with unresolved role, 1 Status',
        'unit_policy':'Retain absent explicit SET units; no rescaling or physical calibration claim',
        'label_policy':'Preserve empty event arrays and unscaled Status statistics; no paper-derived event reconstruction',
        'quantity_policy':'Processed-export sample duration, not newly acquired original 512Hz raw hours',
        'no_model_input_or_label_creation':True,'cpu_threads':2}
    immutable(out/'protocol.json',protocol)
    by_name={j['relative_path']:j for j in jobs};stats={};files=[];records=[];groups=set();unique={};duplicates=[]
    require(len(by_name)==88,'Repeated source filename')
    with threadpool_limits(limits=2):
        for j in jobs:
            require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline exceeded')
            path=acquire.destination(args.raw_root,j['dataset_key'],j['relative_path']);require(not path.is_symlink(),'Source symlink prohibited')
            st=path.stat();digest=acquire.verify_file(path,j['expected_bytes'],json.loads(j['checksum_json']));require(digest==j['sha256'],'Source final checksum differs')
            after=path.stat();require((st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'Source changed during hash')
            stats[str(path)]=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
            files.append({'job_id':j['id'],'source_path':j['relative_path'],'bytes':st.st_size,'sha256':digest,'kind':path.suffix.lower()})
        with (out/'records.jsonl').open('x') as output:
            for j in jobs:
                if not j['relative_path'].endswith('.set'):continue
                require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline exceeded')
                name=j['relative_path'];require(Path(name).name==name,'Unexpected nested source filename')
                fj=by_name.get(str(Path(name).with_suffix('.fdt')));require(fj is not None,'Missing FDT companion')
                sp=acquire.destination(args.raw_root,j['dataset_key'],name);fp=sp.with_suffix('.fdt')
                e=loadmat(sp,simplify_cells=True)['EEG'];h=header(e,name,fj['expected_bytes']);stream=stream_fdt(fp,h,fj['sha256'])
                group='figshare:9970619:'+re.match(r'(s[0-9]+)_',name).group(1);groups.add(group)
                row={'source_SET_path':name,'source_SET_sha256':j['sha256'],'source_FDT_path':fj['relative_path'],'source_group':group,
                     'scope':'processed_export_recording',**h,**stream,'supervised_label_eligible':False,'online_causality_verified':False,
                     'original_raw_recording_hours_added':0,'actual_gradient_training_hours_added':0}
                key=(row['EEG_payload_sha256'],tuple(row['channel_names'][:64]),row['sample_count'],row['sample_rate_hz'])
                if key in unique:duplicates.append({'source_SET_path':name,'same_as':unique[key]['source_SET_path'],'must_link_if_different_groups':group!=unique[key]['source_group']})
                else:unique[key]=row
                records.append(row);output.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');output.flush()
    require(len(records)==44 and len(groups)==11,'Unexpected complete record/group count')
    for name,expected in stats.items():
        s=Path(name).stat();require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==expected,'Source changed before completion')
    seconds=sum((Fraction(r['sample_count'],200) for r in unique.values()),Fraction())
    metadata=sum(f['bytes'] for f in files if f['kind']=='.set');eeg=sum(r['EEG_sample_bytes'] for r in records);aux=sum(r['EXG_sample_bytes'] for r in records);status=sum(r['Status_sample_bytes'] for r in records)
    require(metadata+eeg+aux+status==sum(f['bytes'] for f in files),'Whole source byte accounting mismatch')
    with (out/'files.jsonl').open('x') as f:
        for row in files:f.write(json.dumps(row,sort_keys=True)+'\n')
    result={'schema':'xavier_processed_SSVEP_quantity_summary_v1','status':'completed','completed_epoch':time.time(),'acquired_distribution_bytes':sum(f['bytes'] for f in files),
        'source_files':88,'processed_export_records':44,'source_participant_groups':len(groups),'unique_processed_EEG_records':len(unique),'exact_EEG_duplicates':len(duplicates),
        'processed_export_EEG_seconds_exact_fraction':[seconds.numerator,seconds.denominator],'processed_export_EEG_hours_deduplicated':float(seconds/3600),
        'processed_export_EEG_scalar_bytes_deduplicated':sum(r['EEG_sample_bytes'] for r in unique.values()),'EEG_scalar_bytes_all':eeg,'EXG_scalar_bytes_all':aux,'Status_scalar_bytes_all':status,'SET_metadata_bytes':metadata,
        'stored_sample_rate_hz_distribution':{'200':44},'source_channel_partition':{'named_scalp_EEG':64,'EXG_roles_unassigned':8,'Status':1},
        'all_event_arrays_empty':all(r['original_EEG_event_array_empty'] for r in records),'explicit_unit_absent_records':44,'original_raw_512Hz_files_acquired':0,
        'original_raw_recording_hours_added':0,'amplitude_calibration_validated':False,'supervised_labels_validated':False,'online_causality_validated':False,
        'nonfinite_stored_scalar_count':sum(sum(r['nonfinite_values_per_channel']) for r in records),'Status_noninteger_sample_count':sum(r['Status_noninteger_sample_count'] for r in records),
        'Status_transition_count_unmapped':sum(r['Status_value_transition_count'] for r in records),'existing_model_cohort_changed':False,'actual_gradient_training_hours_added':0,
        'all88_published_MD5_and_own_SHA256_verified':True,'source_stats_checked_unchanged':True,'records_sha256':sha(out/'records.jsonl'),'files_sha256':sha(out/'files.jsonl'),'protocol_sha256':sha(out/'protocol.json')}
    immutable(out/'duplicates.json',duplicates);immutable(out/'summary.json',result)
    immutable(out/'completion.json',{'status':'completed','summary_sha256':sha(out/'summary.json'),'all_original_files_unchanged':True,'original_rows_reconciled':False})
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('ledger','raw-root','download-completion','resource','output'):p.add_argument('--'+name,required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True),flush=True)


if __name__=='__main__':main()
