"""Bounded read-only quantity audit of three already acquired small releases.

Epoch duration, converted continuous FIF duration and unassigned text-table
duration are separate. Source markers are counted without inventing semantics.
No fitting, scale correction, clipping, raw mutation or model inference.
"""
from __future__ import annotations
import argparse
import collections
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time
import numpy as np
from scipy.io import loadmat
import acquire
from delegated_acquisition_v2 import immutable, require, sha

MI_CHANNELS=['C5','C3','FC3','CP3','C1','CZ','FCZ','CPZ','C2','C4','FC4','CP4','C6']
FIF_CHANNELS=['Oz','O1','O2','PO3','POz','PO7','PO8','PO4']
SCOPE={'000541':(48,75523098),'001148':(36,110071616),'002355':(30,30504982)}
FIF_AUTHOR_BAD_DATES={'08':['2013-04-06T16:22:32'],'10':['2014-02-26T15:10:48','2014-02-26T15:50:09'],'11':['2014-02-24T17:56:37','2014-02-24T18:02:40']}


def digest_values(x,dtype='<f4'):
    return hashlib.sha256(np.ascontiguousarray(x,dtype=dtype).tobytes()).hexdigest()


def validate_epoch_arrays(mat,hdf,labels,channels,fs,points,trials):
    require(fs==128 and points==448 and channels==MI_CHANNELS,'Unreviewed MI-BMPI rate, channel order or epoch length')
    require(mat.dtype==np.float32 and hdf.dtype==np.float32,'Unexpected stored epoch precision')
    require(mat.shape==(13,points,trials) and hdf.shape==(trials,points,13),'Epoch axes or declared dimensions mismatch')
    require(np.array_equal(mat,hdf.transpose(2,1,0)),'MAT and SET exported waveforms differ')
    require(np.isfinite(mat).all(),'Nonfinite epoch data')
    labels=np.asarray(labels).reshape(-1)
    require(len(labels)==trials and np.issubdtype(labels.dtype,np.integer) and set(map(int,labels))<={1,2},'Labels length/type/original codes mismatch')
    return labels


def h5text(h,reference):
    return ''.join(chr(int(v)) for v in np.asarray(h[reference]).reshape(-1) if int(v))


def h5empty(value):
    return bool(value.attrs.get('MATLAB_empty',0)) or value.size==0


def audit_epochs(paths):
    import h5py
    base=Path(paths['.set']).stem;m=re.fullmatch(r'D(\d{2})_s([12])',base);require(m is not None,'Unreviewed MI source name')
    e=loadmat(paths['.mat'],simplify_cells=True)['EEG'];mat=np.asarray(e['data']);labels=loadmat(paths['_labels.mat'],simplify_cells=True)['labels']
    chans=[str(v['labels']) for v in np.asarray(e['chanlocs']).reshape(-1)]
    require(all(str(v.get('type',''))=='EEG' for v in np.asarray(e['chanlocs']).reshape(-1)),'Unexpected MAT channel role')
    with h5py.File(paths['.set'],'r') as h:
        scalar=lambda k:float(np.asarray(h[k]).reshape(-1)[0])
        fs=scalar('srate');n=int(scalar('pnts'));trials=int(scalar('trials'));require(scalar('nbchan')==13,'SET channel count differs')
        hdf=np.asarray(h['data']);refs=np.asarray(h['chanlocs']['labels']).reshape(-1);names=[h5text(h,r) for r in refs]
        require(names==chans and float(e['srate'])==fs and int(e['pnts'])==n and int(e['trials'])==trials,'SET/MAT metadata mismatch')
        labels=validate_epoch_arrays(mat,hdf,labels,chans,fs,n,trials)
        require(h5empty(h['event']) and h5empty(h['epoch']) and np.asarray(e['event']).size==0 and np.asarray(e['epoch']).size==0,'Unexpected event or epoch provenance; review rather than infer')
        require(scalar('xmin')==-1 and abs(scalar('xmax')-2.4921875)<1e-9,'Unexpected epoch endpoint fields')
    epochs=np.ascontiguousarray(mat.transpose(2,0,1),dtype='<f4');epoch_hashes=[digest_values(x) for x in epochs]
    return {'catalog_id':'000541','recording_id':base,'subject_group':'figshare:26893396:D'+m.group(1),'source_session':int(m.group(2)),
        'format':'EEGLAB epoched HDF5 SET and identical MATLAB EEG export','channels':chans,'sample_rate_hz':128,'points_per_epoch':448,'epoch_count':trials,
        'stored_epoch_duration_s':3.5,'stored_epoch_seconds':trials*3.5,'continuous_original_hours_confirmed':0,
        'original_recording_intervals_and_overlap_known':False,'SET_MAT_full_array_exact_equality':True,
        'one_copy_decoded_EEG_scalar_bytes':int(mat.nbytes),'compressed_container_bytes_are_not_scalar_bytes':True,
        'EEG_payload_sha256':digest_values(epochs),'per_epoch_payload_sha256':epoch_hashes,'explicit_amplitude_unit':'unresolved',
        'event_count':0,'epoch_provenance_entries':0,'original_label_code_counts':dict(collections.Counter(map(int,labels))),
        'numeric_code_to_task_semantics_validated':False,'labels_are_not_free_form_intent':True,
        'new_training_hours_added':0}


def fif_buffer_accounting(raw,eeg):
    nchan=len(raw.ch_names);payload=eegbytes=stored=0;types=collections.Counter();widths={2:2,3:4,4:4,5:8,16:2}
    for extra in raw._raw_extras:
        require(extra['orig_nchan']==nchan and len(extra['bounds'])==len(extra['ent'])+1,'FIF channel/buffer bounds mismatch')
        for i,entry in enumerate(extra['ent']):
            require(entry is not None and int(entry.type) in widths,'Unsupported FIF interval or scalar encoding')
            samples=int(extra['bounds'][i+1]-extra['bounds'][i]);width=widths[int(entry.type)]
            require(samples>=0 and int(entry.size)==samples*nchan*width,'FIF buffer byte count mismatch')
            payload+=int(entry.size);eegbytes+=samples*len(eeg)*width;stored+=samples;types[str(int(entry.type))]+=1
    require(stored==raw.n_times,'FIF missing/repeated sample buffer')
    return payload,eegbytes,dict(types)


def audit_fif(path):
    import mne
    m=re.fullmatch(r'subject(\d{2})_run(\d+)_raw.fif',Path(path).name);require(m is not None,'Unexpected FIF filename')
    raw=mne.io.read_raw_fif(path,preload=False,verbose='ERROR');types=raw.get_channel_types();eeg=[i for i,t in enumerate(types) if t=='eeg'];stim=[i for i,t in enumerate(types) if t=='stim']
    require([raw.ch_names[i] for i in eeg]==FIF_CHANNELS and len(stim)==1 and len(raw.ch_names)==9,'Unreviewed8 EEG+1stim channel map')
    fs=float(raw.info['sfreq']);require(fs==256,'Unexpected FIF native export rate');n=int(raw.n_times)
    total,eegbytes,enc=fif_buffer_accounting(raw,eeg);h=hashlib.sha256();lo=float('inf');hi=float('-inf')
    for start in range(0,n,15360):
        x=raw.get_data(picks=eeg,start=start,stop=min(n,start+15360));require(np.isfinite(x).all(),'Nonfinite decoded FIF EEG')
        h.update(np.ascontiguousarray(x.T,dtype='<f8').tobytes());lo=min(lo,float(x.min()));hi=max(hi,float(x.max()))
    events=mne.find_events(raw,stim_channel=raw.ch_names[stim[0]],shortest_event=1,consecutive=True,initial_event=True,verbose='ERROR')
    codes=collections.Counter(int(x[2]) for x in events)
    require(all(0<=int(v[0])-raw.first_samp<n for v in events),'FIF stim transition outside stored record')
    dt=raw.info['meas_date'];iso=dt.isoformat() if dt else None
    bad=bool(iso and any(iso.startswith(t) for t in FIF_AUTHOR_BAD_DATES.get(m.group(1),[])))
    calibration=[{k:(int(ch[k]) if k in ['kind','unit','unit_mul'] else float(ch[k])) for k in ['kind','unit','unit_mul','cal','range']} for ch in (raw.info['chs'][i] for i in eeg)]
    row={'catalog_id':'001148','recording_id':Path(path).name,'subject_group':'zenodo:2392979:subject'+m.group(1),'source_run':int(m.group(2)),
        'format':'Converted MNE FIF continuous EEG export','sample_rate_hz':256,'sample_count':n,'channels':FIF_CHANNELS,'total_signals':9,
        'processed_continuous_seconds':n/256,'original_GDF_hours_confirmed':0,'EEG_scalar_bytes':eegbytes,'auxiliary_stim_scalar_bytes':total-eegbytes,
        'FIF_tag_and_metadata_bytes':Path(path).stat().st_size-total,'data_buffer_type_counts':enc,'decoded_EEG_V_sha256':h.hexdigest(),
        'decoded_min_V':lo,'decoded_max_V':hi,'channel_calibration':calibration,'physical_unit_scope':'FIF declared calibration to volts; not independent amplifier validation',
        'original_stim_code_counts':{str(k):v for k,v in sorted(codes.items())},'original_stim_events':[{'sample_index':int(s)-raw.first_samp,'original_code':int(c)} for s,_,c in events],
        'original_OpenVibe_code_to_converted_FIF_code_mapping_verified':False,'semantic_labels_or_exact_stimulus_schedule_created':False,
        'measurement_datetime':iso,'author_protocol_failure_datetime_candidate_match':bad,'author_flag_matching_scope':'Subject ID plus identical displayed date/time; timezone equivalence not independently established',
        'exoskeleton_actuated_during_recording':False,'new_training_hours_added':0}
    raw.close();return row


def audit_text(path):
    columns=collections.Counter();tokens_bytes=[0]*5;rows=[]
    with Path(path).open('rb') as f:
        for line in f:
            tokens=line.split()
            if not tokens:continue
            require(len(tokens)==5 and all(re.fullmatch(rb'[+-]?\d+',x) for x in tokens),'Expected exactly five integer columns without header')
            columns[len(tokens)]+=1;rows.append([int(x) for x in tokens])
            for i,x in enumerate(tokens):tokens_bytes[i]+=len(x)
    require(rows,'Empty text signal table');x=np.asarray(rows,dtype=np.int64)
    m=re.fullmatch(r'S(\d+)_2Pan_(12|14|16|18|20|22)\.txt',Path(path).name);require(m is not None,'Unreviewed Starlab filename')
    return {'catalog_id':'002355','recording_id':Path(path).name,'subject_group':'zenodo:1226586:S'+m.group(1),
        'format':'Headerless5column integer ASCII table','table_rows':len(rows),'table_columns':5,'publisher_declared_rate_hz':250,
        'declared_rate_table_duration_seconds':len(rows)/250,'declared_EEG_channel_count':3,'declared_EEG_electrodes':['O1','Oz','O2'],
        'actual_column_to_EEG_reference_marker_mapping_verified':False,'EEG_specific_payload_bytes':None,'original_EEG_hours_confirmed':0,
        'numeric_UTF8_token_bytes_by_column':tokens_bytes,'whitespace_bytes':Path(path).stat().st_size-sum(tokens_bytes),
        'numeric_min_by_column':x.min(axis=0).tolist(),'numeric_max_by_column':x.max(axis=0).tolist(),
        'nonzero_count_by_column':np.count_nonzero(x,axis=0).tolist(),'all_column_payload_sha256':digest_values(x,'<i8'),
        'filename_frequency_condition_hz':int(m.group(2)),'physical_unit':'unresolved','filename_condition_is_not_trial_marker':True,
        'new_training_hours_added':0}


def summarize(records,files):
    out={}
    for cid,(count,size) in SCOPE.items():
        rr=[r for r in records if r['catalog_id']==cid];ff=[r for r in files if r['catalog_id']==cid]
        require(len(ff)==count and sum(r['bytes'] for r in ff)==size,'Pinned release files/bytes mismatch')
        s={'catalog_id':cid,'distribution_files':count,'distribution_bytes':size,'source_record_ID_groups':len({r['subject_group'] for r in rr}),
           'audited_records':len(rr),'new_training_hours_added':0,'original_raw_EEG_hours_added':0,'license_API':'CC BY4.0'}
        if cid=='000541':
            unique={};dup=0;epochs=[]
            for r in rr:
                key=(r['EEG_payload_sha256'],tuple(r['channels']),r['epoch_count'])
                if key in unique:dup+=1
                else:unique[key]=r
                epochs.extend(r['per_epoch_payload_sha256'])
            secs=sum((Fraction(r['epoch_count']*448,128) for r in unique.values()),Fraction())
            s.update(stored_epoch_count=sum(r['epoch_count'] for r in rr),stored_epoch_seconds_exact_fraction=[secs.numerator,secs.denominator],stored_epoch_hours_deduplicated_whole_records=float(secs/3600),
                exact_whole_record_duplicates=dup,exact_repeated_epoch_payload_occurrences=len(epochs)-len(set(epochs)),
                one_copy_decoded_EEG_scalar_bytes=sum(r['one_copy_decoded_EEG_scalar_bytes'] for r in unique.values()),physical_recording_interval_union_unknown=True,
                SET_MAT_duplicates_not_double_counted=True,physical_amplitude_units_verified=False,commercial_permission_notice_also_present=True,
                commercial_use_eligibility='API CCBY4.0 alongside publisher description requesting permission for commercial studies; unresolved before commercial use',
                original_label_code_counts=dict(sum((collections.Counter(r['original_label_code_counts']) for r in rr),collections.Counter())))
        elif cid=='001148':
            seen={};dup=0
            for r in rr:
                key=(r['decoded_EEG_V_sha256'],tuple(r['channels']),r['sample_count'],r['sample_rate_hz'])
                if key in seen:dup+=1
                else:seen[key]=r
            secs=sum((Fraction(r['sample_count'],256) for r in seen.values()),Fraction())
            s.update(processed_continuous_seconds_exact_fraction=[secs.numerator,secs.denominator],processed_continuous_EEG_hours_deduplicated=float(secs/3600),exact_whole_record_duplicates=dup,
                EEG_scalar_bytes_deduplicated=sum(r['EEG_scalar_bytes'] for r in seen.values()),EEG_scalar_bytes_all=sum(r['EEG_scalar_bytes'] for r in rr),
                auxiliary_stim_bytes_all=sum(r['auxiliary_stim_scalar_bytes'] for r in rr),FIF_metadata_bytes_all=sum(r['FIF_tag_and_metadata_bytes'] for r in rr),
                author_flag_datetime_candidate_records=sum(r['author_protocol_failure_datetime_candidate_match'] for r in rr),original_GDF_not_present=True,
                converted_event_mapping_unverified=True,original_stim_code_counts=dict(sum((collections.Counter(r['original_stim_code_counts']) for r in rr),collections.Counter())))
            require(s['EEG_scalar_bytes_all']+s['auxiliary_stim_bytes_all']+s['FIF_metadata_bytes_all']==size,'FIF byte partition mismatch')
        else:
            seen={};dup=0
            for r in rr:
                key=(r['all_column_payload_sha256'],r['table_rows'])
                if key in seen:dup+=1
                else:seen[key]=r
            secs=sum((Fraction(r['table_rows'],250) for r in seen.values()),Fraction())
            s.update(declared_rate_table_seconds_exact_fraction=[secs.numerator,secs.denominator],declared_rate_table_hours_deduplicated=float(secs/3600),exact_whole_table_duplicates=dup,
                confirmed_EEG_specific_hours=0,EEG_specific_bytes=None,actual_column_mapping_and_amplitude_units_unresolved=True,
                total_numeric_UTF8_token_bytes=sum(sum(r['numeric_UTF8_token_bytes_by_column']) for r in rr),whitespace_bytes=sum(r['whitespace_bytes'] for r in rr))
            require(s['total_numeric_UTF8_token_bytes']+s['whitespace_bytes']==size,'ASCII bytes not partitioned')
        out[cid]=s
    return out


def run(args):
    import scipy,h5py,mne
    from threadpoolctl import threadpool_limits
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);budget=json.loads(Path(args.resource).read_text());done=json.loads(Path(args.download_completion).read_text())
    require(done['status']=='all_files_downloaded_checksum_verified' and done['complete_files']==114,'All114 source files required')
    require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline expired');require(shutil.disk_usage(args.raw_root).free>=36*2**30+64*2**20,'Audit reserve unavailable')
    db=sqlite3.connect(Path(args.ledger).resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;jobs=[dict(x) for x in db.execute('select * from downloads order by catalog_id,relative_path')]
    require(len(jobs)==114 and all(j['status']=='complete' for j in jobs),'Unexpected/incomplete ledger')
    immutable(out/'protocol.json',{'schema':'xavier_three_small_release_quantity_audit_v1','source_sha256':sha(__file__),'resource_sha256':sha(args.resource),'expected_scope':SCOPE,
        'absolute_deadline_epoch':budget['absolute_deadline_epoch'],'cpu_threads':2,'all_original_checksums_and_before_after_stat_required':True,
        'MI_rule':'HDF5 SET and MATLAB array all values/axes compared; once-only epoch sample duration, not independent continuous time',
        'FIF_rule':'Count actual EEG buffer bytes and continuous samples; original GDF absent; stim mapping not inferred from source OpenVibe codes',
        'text_rule':'Five-column table row duration only at publisher250Hz; no EEG column map or physical unit assigned',
        'runtime':{'numpy':np.__version__,'scipy':scipy.__version__,'mne':mne.__version__,'h5py':h5py.__version__},'no_training_or_raw_mutation':True})
    files=[];paths={};stats={};records=[]
    with threadpool_limits(limits=2):
        for job in jobs:
            require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline exceeded')
            path=acquire.destination(args.raw_root,job['dataset_key'],job['relative_path']);st=path.stat();h=acquire.verify_file(path,job['expected_bytes'],json.loads(job['checksum_json']));require(h==job['sha256'],'Completed SHA mismatch')
            paths[(job['catalog_id'],Path(job['relative_path']).name)]=path;stats[str(path)]=(st.st_ino,st.st_size,st.st_mtime_ns)
            files.append({'catalog_id':job['catalog_id'],'source_path':job['relative_path'],'bytes':st.st_size,'sha256':h,'job_id':job['id']})
        with (out/'records.jsonl').open('x') as f:
            for (cid,name),path in paths.items():
                require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline exceeded')
                if cid=='000541':
                    if path.suffix!='.set':continue
                    base=path.stem;p={k:paths[(cid,base+k)] for k in ['.set','.mat','_labels.mat']};r=audit_epochs(p)
                    r['source_paths']={k:v.name for k,v in p.items()};r['source_sha256']={k:sha(v) for k,v in p.items()}
                elif cid=='001148':r=audit_fif(path);r['source_sha256']=sha(path)
                elif cid=='002355':r=audit_text(path);r['source_sha256']=sha(path)
                else:raise ValueError('Out-of-scope catalog')
                records.append(r);f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n');f.flush()
    require(collections.Counter(r['catalog_id'] for r in records)=={'000541':16,'001148':36,'002355':30},'Record coverage incomplete')
    for job in jobs:
        path=acquire.destination(args.raw_root,job['dataset_key'],job['relative_path']);st=path.stat();require((st.st_ino,st.st_size,st.st_mtime_ns)==stats[str(path)] and sha(path)==job['sha256'],'Original changed during audit')
    with (out/'files.jsonl').open('x') as f:
        for row in files:f.write(json.dumps(row,sort_keys=True)+'\n')
    result={'schema':'xavier_three_small_release_quantity_summary_v1','status':'completed','completed_epoch':time.time(),'all_source_files':114,'distribution_bytes':216099696,
        'datasets':summarize(records,files),'all_source_files_checksum_verified':True,'original_files_unchanged':True,'new_training_hours_added':0,
        'records_sha256':sha(out/'records.jsonl'),'files_sha256':sha(out/'files.jsonl'),'protocol_sha256':sha(out/'protocol.json')}
    immutable(out/'summary.json',result);immutable(out/'completion.json',{'status':'completed','summary_sha256':sha(out/'summary.json'),'original_files_unchanged':True,'new_model_inferences':0})
    return result


def main():
    p=argparse.ArgumentParser()
    for n in ['ledger','raw-root','resource','download-completion','output']:p.add_argument('--'+n,required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True),flush=True)


if __name__=='__main__':main()
