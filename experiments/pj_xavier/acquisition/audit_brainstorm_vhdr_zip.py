"""Read-only Figshare5756277v2 raw BrainVision/companion audit.

The bundled software/tutorial EEG samples are preserved but not counted as
additional study recordings. Two original channel labels IO/ECG are retained
as auxiliary-or-unresolved; the paper and montage do not silently relabel them.
"""
from __future__ import annotations
import argparse
import collections
from fractions import Fraction
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import tempfile
import time
import warnings
import zipfile
import numpy as np
import acquire
from delegated_acquisition_v2 import immutable, require, sha

PREFIX='EEG_brainstorm/rawdata_vhdr/'
PAPER='https://doi.org/10.3389/fnins.2018.00309'


def sections(raw):
    text=raw.decode('utf-8-sig');out={};current=None
    for line in text.splitlines()[1:]:
        line=line.strip()
        if not line or line.startswith(';'):continue
        if line.startswith('[') and line.endswith(']'):
            current=line[1:-1]
            if current=='Comment':break
            require(current not in out,'Repeated section');out[current]={};continue
        if current is None:continue
        require('=' in line,'Malformed header field');key,value=line.split('=',1)
        require(key not in out[current],'Repeated header field');out[current][key]=value
    return out


def parse_header(raw,name,signal_bytes):
    require(raw.startswith(b'Brain Vision Data Exchange Header File Version 1.0'),'Unreviewed BrainVision version')
    s=sections(raw);c=s['Common Infos'];b=s['Binary Infos'];ch=s['Channel Infos']
    require(c['Codepage']=='UTF-8' and c['DataFormat']=='BINARY' and c['DataOrientation']=='MULTIPLEXED' and b['BinaryFormat']=='INT_16','Unreviewed binary format')
    stem=PurePosixPath(name).stem;require(c['DataFile']==stem+'.eeg' and c['MarkerFile']==stem+'.vmrk','Unsafe or mismatched companion')
    require(c['NumberOfChannels']=='64' and c['SamplingInterval']=='1000','Unreviewed channel/rate dimensions')
    require(signal_bytes>0 and signal_bytes%(64*2)==0,'Partial int16 signal frame')
    require(set(ch)=={'Ch'+str(i) for i in range(1,65)},'Channel list does not cover64')
    channels=[]
    for i in range(1,65):
        f=ch['Ch'+str(i)].split(',');require(len(f)==4,'Unexpected channel field schema')
        expected='IO' if i==19 else 'ECG' if i==20 else str(i)
        require(f[0]==expected and f[2]=='0.1' and f[3] in ['µV','μV','uV'],'Unexpected source labels or calibration')
        channels.append({'index':i-1,'label':f[0],'reference':f[1],'resolution':float(f[2]),'unit':f[3],'scale_V_per_AD':1e-7,
                         'role':'auxiliary_or_unresolved_header_label' if i in (19,20) else 'source_cap_EEG_candidate'})
    n=signal_bytes//128
    return {'channels':channels,'signals':64,'EEG_candidate_channels':62,'auxiliary_or_unresolved_channels':2,
            'sample_rate_hz':1000,'sample_count':n,'stored_signal_seconds':n/1000,'scalar_bytes':2,'dtype':'<i2',
            'header_sha256':hashlib.sha256(raw).hexdigest(),'header_bytes':len(raw),
            'EEG_indices':[i for i in range(64) if i not in (18,19)],'auxiliary_indices':[18,19],
            'source_channel_role_scope':'Conservative source cap candidates excluding literal IO/ECG; exact assignment of these2 requires publisher clarification',
            'signal_member':str(PurePosixPath(name).with_suffix('.eeg')),'marker_member':str(PurePosixPath(name).with_suffix('.vmrk'))}


def parse_markers(raw,h):
    s=sections(raw);require(s['Common Infos']['DataFile']==PurePosixPath(h['signal_member']).name,'Marker companion mismatch')
    entries=[];invalid=0;counter=collections.Counter()
    for key,value in s['Marker Infos'].items():
        require(re.fullmatch(r'Mk[0-9]+',key) is not None,'Unrecognized marker key')
        f=value.split(',');require(5<=len(f)<=6,'Unreviewed marker layout')
        pos,size,ch=map(int,f[2:5]);bad=pos<1 or size<0 or pos-1+size>h['sample_count'] or ch<0 or ch>64
        invalid+=bad;counter[(f[0],f[1])]+=1
        entries.append({'source_marker_key':key,'source_type':f[0].replace('\\1',','),'source_description':f[1].replace('\\1',','),
            'source_position_1based':pos,'source_size_samples':size,'source_channel_1based_or0_all':ch,
            'derived_onset_s':(pos-1)/1000,'outside_or_invalid_without_clipping':bad,
            'absolute_datetime_present':len(f)==6 and bool(f[5])})
    return {'original_marker_count':len(entries),'original_marker_entries':entries,
            'marker_outside_or_invalid_count':int(invalid),'marker_type_description_counts':{'|'.join(k):v for k,v in sorted(counter.items())},
            'stimulus_semantics_validated':False,'marker_time_is_independent_acoustic_onset':False}


def stream_signal(stream,h):
    whole=hashlib.sha256();eeg=hashlib.sha256();aux=hashlib.sha256();done=0
    minimum=np.full(64,32767,dtype=np.int32);maximum=np.full(64,-32768,dtype=np.int32)
    while done<h['sample_count']:
        take=min(4096,h['sample_count']-done);raw=stream.read(take*128);require(len(raw)==take*128,'Truncated sample payload')
        whole.update(raw);x=np.frombuffer(raw,dtype='<i2').reshape(take,64)
        eeg.update(np.ascontiguousarray(x[:,h['EEG_indices']]).tobytes());aux.update(np.ascontiguousarray(x[:,18:20]).tobytes())
        minimum=np.minimum(minimum,x.min(axis=0));maximum=np.maximum(maximum,x.max(axis=0))
        done+=take
    require(stream.read(1)==b'','Extra signal payload')
    return {'source_signal_sha256':whole.hexdigest(),'EEG_payload_sha256':eeg.hexdigest(),'auxiliary_payload_sha256':aux.hexdigest(),
            'EEG_candidate_sample_bytes':done*62*2,'auxiliary_or_unresolved_sample_bytes':done*2*2,
            'raw_AD_min_per_channel':[int(v) for v in minimum],'raw_AD_max_per_channel':[int(v) for v in maximum],
            'physical_scale_scope':'Declared BrainVision0.1 microV per AD; not independently calibrated hardware',
            'waveform_changes':0}


def run(args):
    from threadpoolctl import threadpool_limits
    import mne
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);budget=json.loads(Path(args.resource).read_text())
    done=json.loads(Path(args.download_completion).read_text());require(done['status']=='all_files_downloaded_checksum_verified' and done['complete_files']==1,'Archive not verified complete')
    require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline expired')
    require(shutil.disk_usage(args.raw_root).free>=36*2**30+32*2**20,'Audit reserve unavailable')
    db=sqlite3.connect(Path(args.ledger).resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;jobs=[dict(x) for x in db.execute('select * from downloads')]
    require(len(jobs)==1 and jobs[0]['dataset_key']=='figshare-5756277-2' and jobs[0]['catalog_id']=='002070' and jobs[0]['status']=='complete','Source identity differs')
    job=jobs[0];path=acquire.destination(args.raw_root,job['dataset_key'],job['relative_path']);initial=acquire.verify_file(path,job['expected_bytes'],json.loads(job['checksum_json']));require(initial==job['sha256'],'Archive source checksum mismatch')
    immutable(out/'protocol.json',{'schema':'xavier_brainstorm_original_BrainVision_audit_v1','source_sha256':sha(__file__),'acquire_source_sha256':sha(acquire.__file__),
        'source_archive_sha256':initial,'resource_sha256':sha(args.resource),'original_archive_immutable':True,'paper_source':PAPER,'MNE_version':mne.__version__,
        'scope':'Only10 rawdata_vhdr study triplets counted; software and bundled tutorial waveform samples preserved without new study hours',
        'source_roles':'62numeric cap EEG candidates plus2 IO/ECG unresolved/auxiliary; publisher mapping clarification required',
        'temporary_extraction_limit_bytes':32*2**20,'CPU_threads':2,'source_event_repair_or_new_gold_labels':False})
    records=[];members=[];rolebytes=collections.Counter();markers=collections.Counter();seen={};duplicates=[]
    with threadpool_limits(limits=2),zipfile.ZipFile(path) as z:
        infos=z.infolist();names=[i.filename for i in infos];require(len(set(names))==len(names),'Repeated ZIP path')
        for i in infos:
            n=PurePosixPath(i.filename);require(not n.is_absolute() and '..' not in n.parts and '\\' not in i.filename,'Unsafe archive member')
            if i.is_dir():continue
            require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline exceeded')
            h=hashlib.sha256()
            with z.open(i) as f:
                for b in iter(lambda:f.read(2**20),b''):h.update(b)
            role='study_signal' if i.filename.startswith(PREFIX) and i.filename.endswith('.eeg') else 'study_header_or_marker' if i.filename.startswith(PREFIX) else 'software_or_companion'
            members.append({'member':i.filename,'bytes':i.file_size,'sha256':h.hexdigest(),'CRC32':i.CRC,'role':role});rolebytes[role]+=i.file_size
        by={x['member']:x for x in members};headers=[x for x in by if x.startswith(PREFIX) and x.endswith('.vhdr')];require(len(headers)==10,'Expected10 source study headers')
        with (out/'records.jsonl').open('x') as ledger:
            for name in sorted(headers):
                require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline exceeded')
                signal=str(PurePosixPath(name).with_suffix('.eeg'));marker=str(PurePosixPath(name).with_suffix('.vmrk'))
                h=parse_header(z.read(name),name,by[signal]['bytes']);m=parse_markers(z.read(marker),h)
                with z.open(signal) as f:s=stream_signal(f,h)
                require(s['source_signal_sha256']==by[signal]['sha256'],'Stream hash mismatch')
                match=re.fullmatch(r'mas_AEP_(nh[0-9]+)\.vhdr',PurePosixPath(name).name);require(match is not None,'Unexpected study subject naming')
                group='figshare:5756277:'+match.group(1)
                with tempfile.TemporaryDirectory(prefix='probe_',dir=out) as td:
                    total=sum(by[n]['bytes'] for n in [name,signal,marker]);require(total<32*2**20 and shutil.disk_usage(out).free>=36*2**30+total,'Probe extraction reserve unavailable')
                    for n in [name,signal,marker]:
                        with z.open(n) as f,(Path(td)/PurePosixPath(n).name).open('xb') as o:shutil.copyfileobj(f,o,2**20)
                    with warnings.catch_warnings(record=True) as ws:
                        raw=mne.io.read_raw_brainvision(Path(td)/PurePosixPath(name).name,preload=False,verbose='ERROR')
                    require(raw.n_times==h['sample_count'] and raw.info['sfreq']==1000 and raw.ch_names==[c['label'] for c in h['channels']],'MNE header disagrees')
                    maximum_difference=0.
                    with (Path(td)/PurePosixPath(signal).name).open('rb') as f:
                        for start in [0,h['sample_count']//2,h['sample_count']-8]:
                            f.seek(start*128);expected=np.frombuffer(f.read(8*128),dtype='<i2').reshape(8,64).T.astype(np.float64)*1e-7
                            actual=raw.get_data(start=start,stop=start+8);diff=float(np.max(np.abs(actual-expected)));maximum_difference=max(maximum_difference,diff);require(np.allclose(actual,expected,rtol=1e-12,atol=1e-15),'MNE calibration sample differs')
                    raw.close()
                row={'source_header_member':name,'source_header_sha256':by[name]['sha256'],'source_marker_sha256':by[marker]['sha256'],'subject_group':group,**h,**m,**s,
                    'MNE_independent_header_and_calibration_probes_passed':True,'maximum_probe_difference_V':maximum_difference,'MNE_warning_categories':[w.category.__name__ for w in ws],
                    'new_training_hours_added':0,'supervised_gold_eligibility':False}
                calibration=hashlib.sha256(json.dumps(h['channels'],sort_keys=True).encode()).hexdigest();row['calibration_sha256']=calibration
                key=(row['EEG_payload_sha256'],calibration,row['sample_count'],row['sample_rate_hz'])
                if key in seen:duplicates.append({'source':name,'same_as':seen[key]['source_header_member'],'must_link_groups':group!=seen[key]['subject_group']})
                else:seen[key]=row
                records.append(row);markers.update(row['marker_type_description_counts']);ledger.write(json.dumps(row,sort_keys=True)+'\n');ledger.flush()
    require(acquire.verify_file(path,job['expected_bytes'],json.loads(job['checksum_json']))==initial,'Archive changed after member audit')
    seconds=sum((Fraction(r['sample_count'],1000) for r in seen.values()),Fraction());eeg=sum(r['EEG_candidate_sample_bytes'] for r in records);aux=sum(r['auxiliary_or_unresolved_sample_bytes'] for r in records)
    require(eeg+aux==rolebytes['study_signal'],'Signal byte partition mismatch')
    immutable(out/'archive_members.json',members);immutable(out/'duplicates.json',duplicates)
    result={'schema':'xavier_brainstorm_raw_quantity_summary_v1','status':'completed','completed_epoch':time.time(),'catalog_id':'002070','source_version':'Figshare5756277v2','license':'CC BY4.0',
        'source_archive_bytes':job['expected_bytes'],'source_archive_sha256':initial,'source_uncompressed_bytes':sum(x['bytes'] for x in members),'source_non_directory_members':len(members),
        'study_records':len(records),'source_ID_groups':len({r['subject_group'] for r in records}),'unique_study_EEG_records':len(seen),'exact_EEG_duplicates':len(duplicates),
        'original_recording_seconds_exact_fraction':[seconds.numerator,seconds.denominator],'original_recording_hours_deduplicated':float(seconds/3600),
        'conservative_EEG_candidate_channels':62,'auxiliary_or_unresolved_channels':2,'EEG_candidate_scalar_bytes_deduplicated':sum(r['EEG_candidate_sample_bytes'] for r in seen.values()),
        'auxiliary_or_unresolved_scalar_bytes_all':aux,'source_role_bytes':dict(rolebytes),'native_rate_hz_distribution':{'1000':10},'source_header_resolution':'0.1 microV per int16 AD',
        'MNE_independent_header_calibration_probe_records':sum(r['MNE_independent_header_and_calibration_probes_passed'] for r in records),
        'original_marker_count':sum(r['original_marker_count'] for r in records),'source_marker_type_description_counts':dict(markers),'marker_outside_or_invalid_without_clip':sum(r['marker_outside_or_invalid_count'] for r in records),
        'all_member_CRC_SHA256_verified':True,'original_archive_checksum_before_after_verified':True,'bundled_tutorial_EEG_additional_study_hours':0,
        'channel19_20_mapping_needs_publisher_clarification':True,'independent_physical_QC_or_stimulus_latency_verified':False,'supervised_gold_labels_validated':False,
        'existing_inputs_or_models_changed':False,'new_training_hours_added':0,'records_sha256':sha(out/'records.jsonl'),'members_sha256':sha(out/'archive_members.json'),'protocol_sha256':sha(out/'protocol.json')}
    immutable(out/'summary.json',result);immutable(out/'completion.json',{'status':'completed','summary_sha256':sha(out/'summary.json'),'original_archive_unchanged':True,'original_rows_reconciled':False})
    return result


def main():
    p=argparse.ArgumentParser()
    for n in ['ledger','raw-root','download-completion','resource','output']:p.add_argument('--'+n,required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True),flush=True)


if __name__=='__main__':main()
