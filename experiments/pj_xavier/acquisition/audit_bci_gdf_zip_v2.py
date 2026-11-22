"""Measure every GDF/source component in pinned Zenodo8089820 BCI archive.

Single-member temporary extraction, whole archive checksum, member CRC/SHA,
source-specific27EEG/3EOG/2EMG separation, exact payload/calibration dedup.
No signals enter training, participant splitting, or physiological QC here.
"""
from __future__ import annotations
import argparse,collections,hashlib,inspect,json,os,re,shutil,struct,tempfile,time,zipfile
from pathlib import Path,PurePosixPath
import numpy as np

EEG_NAMES=['Fz','FCz','Cz','CPz','Pz','C1','C3','C5','C2','C4','C6','F4','FC2','FC4','FC6','CP2','CP4','CP6','P4','F3','FC1','FC3','FC5','CP1','CP3','CP5','P3']
AUX={'EOG':{'EOG1','EOG2','EOG3'},'EMG':{'EMGg','EMGd'}}
ARTICLE='https://www.nature.com/articles/s41597-023-02445-z'


def require(ok,reason):
    if not ok:raise ValueError(reason)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def save(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def classify_channels(names):
    require(len(names)==len(set(names)),'Duplicate channel names')
    out={'EEG':[],'EOG':[],'EMG':[],'unknown':[]}
    for i,name in enumerate(names):
        kind='EEG' if name in EEG_NAMES else next((k for k,v in AUX.items() if name in v),'unknown')
        out[kind].append(i)
    require(not out['unknown'] and len(out['EEG'])==27 and len(out['EOG'])==3 and len(out['EMG'])==2,'Published27/3/2channel mapping does not cover header')
    return out


def event_table_125(data,expected_native_fs,record_seconds):
    require(len(data)>=8,'GDF event table absent/truncated')
    mode=data[0];require(mode in [1,3],'Unknown GDF1.x event mode')
    fs=int.from_bytes(data[1:4],'little');n=struct.unpack_from('<I',data,4)[0]
    require(n<1000000,'Event count invalid')
    explicit=fs>0;used_rate=fs if explicit else expected_native_fs
    require(used_rate>0,'Native fallback rate invalid')
    expected=8+n*(6 if mode==1 else 12);require(len(data)==expected,'GDF event bytes not exactly accounted')
    pos=np.frombuffer(data,dtype='<u4',count=n,offset=8).astype(np.int64)
    codes=np.frombuffer(data,dtype='<u2',count=n,offset=8+4*n)
    channels=np.frombuffer(data,dtype='<u2',count=n,offset=8+6*n) if mode==3 else None
    duration=np.frombuffer(data,dtype='<u4',count=n,offset=8+8*n) if mode==3 else None
    counts=collections.Counter(int(x) for x in codes);outside=0;zero=0;labels=[]
    for i in range(n):
        on=(int(pos[i])-1)/used_rate;fallback_on=(int(pos[i])-1)/expected_native_fs;valid=0<=on<record_seconds
        outside+=not valid;zero+=int(pos[i])==0
        if int(codes[i]) in [769,770]:
            labels.append({'original_one_based_position':int(pos[i]),'onset_s':on if explicit else None,'native_reader_fallback_onset_s':fallback_on,
                'explicit_event_timebase_verified':explicit,'code':int(codes[i]),
                'label':'left_hand_imagery' if int(codes[i])==769 else 'right_hand_imagery',
                'within_record':bool(valid) if explicit else None,'within_record_under_native_reader_fallback':bool(0<=fallback_on<record_seconds),'original_duration_samples':None if duration is None else int(duration[i]),
                'duration_absent_in_mode1_not_imputed':duration is None})
    return {'mode':mode,'event_sample_rate_hz':fs,'event_rate_field_original':fs,
      'explicit_event_timebase_verified':explicit,'native_reader_fallback_rate_hz':expected_native_fs,
      'native_reader_fallback_evidence':'Pinned MNE _get_annotations_gdf(edf_info,sfreq) divides raw positions by native signal sfreq; this is not proof of missing publisher rate',
      'rate_equals_signal':fs==expected_native_fs,'event_count':n,
      'event_code_counts':{str(k):v for k,v in sorted(counts.items())},'event_table_bytes':len(data),
      'events_outside_record':outside if explicit else None,'events_outside_record_using_native_reader_fallback':sum(not (0<=(int(x)-1)/expected_native_fs<record_seconds) for x in pos),'zero_one_based_positions':zero,'MI_cue_events':labels,
      'label_source':'GDF event table; publisher769=leftMI,770=rightMI','event_repairs':0,
      'not_free_form_intent':True,'raw_duration_zero_count':None if duration is None else int(np.count_nonzero(duration==0))}


def payload_hashes(path,header,indices):
    nchan=int(header['nchan']);records=int(header['n_records']);n_samps=[int(x) for x in header['n_samps']]
    width=int(header['dtype_byte']);require(len(set(n_samps))==1 and n_samps[0]>0,'Mixed native channel sampling needs separate audit')
    channelbytes=n_samps[0]*width;recordbytes=nchan*channelbytes
    require(recordbytes==int(header['bytes_tot']),'Header record byte count differs')
    hashers={k:hashlib.sha256() for k in ['EEG','EOG','EMG']};remaining=records
    with Path(path).open('rb') as f:
        f.seek(int(header['data_offset']))
        while remaining:
            batch=min(4096,remaining);block=f.read(batch*recordbytes);require(len(block)==batch*recordbytes,'GDF sample payload truncated')
            values=np.frombuffer(block,dtype=np.uint8).reshape(batch,nchan,channelbytes)
            for kind in hashers:hashers[kind].update(np.ascontiguousarray(values[:,indices[kind],:]).tobytes())
            remaining-=batch
        trailing=f.read()
    return {k:h.hexdigest() for k,h in hashers.items()}, {k:records*len(indices[k])*channelbytes for k in hashers},trailing


def audit_gdf(path,member):
    from mne.io.edf.edf import _read_gdf_header
    h=_read_gdf_header(path,exclude=[]);require(float(h['number'])==1.25,'Unreviewed GDF version')
    names=h['ch_names'];indices=classify_channels(names);nrec=int(h['n_records']);den=int(h['record_length'][1]);num=int(h['record_length'][0])
    require(nrec>0 and den>0 and num>0,'Invalid native record duration')
    rate=float(h['n_samps'][0])*den/num;duration=nrec*num/den
    hashes,bytes_by_kind,events=payload_hashes(path,h,indices)
    event=event_table_125(events,rate,duration)
    require(int(h['data_offset'])+sum(bytes_by_kind.values())+len(events)==Path(path).stat().st_size,'GDF bytes not exhaustive')
    require(event['event_count']==int(h['events'][0]),'MNE/raw event count mismatch')
    require({str(int(k)):int(v) for k,v in collections.Counter(h['events'][2]).items()}==event['event_code_counts'],'MNE/raw event codes mismatch')
    positions=h['events'][1]
    # MNE independently parses original positions. Its minimum duration
    # adjustment is not adopted; raw mode1 duration remains absent.
    raw_positions=np.frombuffer(events,dtype='<u4',count=event['event_count'],offset=8).astype(np.int64)-1
    require(np.array_equal(positions,raw_positions),'MNE/raw event positions mismatch')
    m=re.search(r'/Signals/DATA ([ABC])/([^/]+)/([^/]+)\.gdf$',member);require(m is not None,'Unrecognized source group path')
    record_name=m.group(3);suffix=record_name.split('_',1)[1] if '_' in record_name else 'unclassified'
    calibration={k:[float(h[k][i]) for i in indices['EEG']] for k in ['physical_min','physical_max','digital_min','digital_max','units']}
    calibration_sha=hashlib.sha256(json.dumps(calibration,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return {'source_member':member,'subject_group':'zenodo:8089820:'+m.group(1)+':'+m.group(2),'subdataset':m.group(1),
      'recording_kind':suffix,'source_file_bytes':Path(path).stat().st_size,'source_file_sha256':sha(path),
      'GDF_version':float(h['number']),'all_channels':names,'eeg_channels':[names[i] for i in indices['EEG']],
      'eog_channels':[names[i] for i in indices['EOG']],'emg_channels':[names[i] for i in indices['EMG']],
      'native_sample_rate_hz':rate,'samples_per_EEG_channel':nrec*int(h['n_samps'][0]),
      'stored_signal_seconds':duration,'sample_scalar_bytes':int(h['dtype_byte']),'scalar_dtype':str(h['dtype_np']),
      'data_records':nrec,'record_duration_numerator':num,'record_duration_denominator':den,
      'EEG_sample_bytes':bytes_by_kind['EEG'],'EOG_sample_bytes':bytes_by_kind['EOG'],'EMG_sample_bytes':bytes_by_kind['EMG'],
      'header_bytes':int(h['data_offset']),'event_bytes':len(events),'eeg_payload_sha256':hashes['EEG'],
      'EOG_payload_sha256':hashes['EOG'],'EMG_payload_sha256':hashes['EMG'],
      'EEG_header_calibration':calibration,'EEG_calibration_sha256':calibration_sha,
      'unit_scope':'GDF declared physical range/unit and digital range; header conversion, not independent amplifier calibration',
      'quantity_dedup_identity':['eeg_payload_sha256','EEG_calibration_sha256','eeg_channels','native_sample_rate_hz','samples_per_EEG_channel'],
      'annotation_audit':event,'source_specific_channel_map_primary':ARTICLE,
      'physiological_QC_or_training_eligibility_assessed':False,'training_uses':0}


def run(args):
    import mne
    from mne.io.edf.edf import _read_gdf_header,_get_annotations_gdf
    from threadpoolctl import threadpool_limits
    archive=Path(args.archive);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads(Path(args.acquisition_receipt).read_text());budget=json.loads(Path(args.resource).read_text())
    require(receipt['status']=='complete' and receipt['original_target_reconciled_after_full_checksum'],'Acquisition not fully verified/reconciled')
    require(archive.stat().st_size==receipt['final_size']==budget['expected_whole_file_bytes'],'Archive size mismatch')
    require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline expired')
    initial=archive.stat();reserve=budget['effective_download_reserve_bytes']
    protocol={'schema':'xavier_zenodo8089820_GDF_quantity_audit_v2','created_epoch':time.time(),'source_sha256':sha(__file__),
      'source_archive_sha256':receipt['full_SHA256'],'source_archive_bytes':initial.st_size,'acquisition_receipt_sha256':sha(args.acquisition_receipt),
      'resource_receipt_sha256':sha(args.resource),'absolute_deadline_epoch':budget['absolute_deadline_epoch'],'cpu_threads':2,
      'mne_version':mne.__version__,'mne_GDF_header_parser_source_sha256':hashlib.sha256(inspect.getsource(_read_gdf_header).encode()).hexdigest(),
      'channel_map':{'EEG':EEG_NAMES,'EOG':sorted(AUX['EOG']),'EMG':sorted(AUX['EMG'])},'primary_evidence':ARTICLE,
      'event_map':{'769':'left_hand_imagery','770':'right_hand_imagery'},
      'v1_failure_preserved':'Original event rate0 was detected before quantity ledger completed; v2 preserves zero and separates native reader fallback',
      'GDF125_specification':'https://pub.ista.ac.at/~schloegl/matlab/eeg/gdf4/TR_GDF.pdf',
      'MNE_annotation_converter_source_sha256':hashlib.sha256(inspect.getsource(_get_annotations_gdf).encode()).hexdigest(),
      'raw_event_duration_correction_performed':False,'full_unzip':False,'max_single_temporary_member_bytes':128*2**20,
      'dedup':'EEG stored scalar bytes plus calibration/channel/rate/sample count; exact representations only, not approximate waveform equivalence',
      'training_model_calls':0,'source_prior_full_MD5_SHA_verified':True,'final_whole_archive_SHA_will_be_reverified':True}
    save(out/'protocol.json',protocol);records=[];other=collections.Counter();other_bytes=collections.Counter();members=[]
    with threadpool_limits(limits=2),zipfile.ZipFile(archive) as z,(out/'recordings.jsonl').open('x') as ledger,(out/'archive_members.jsonl').open('x') as memberledger,tempfile.TemporaryDirectory(prefix='one_member_',dir=out) as temp:
        files=[x for x in z.infolist() if not x.is_dir()];require(len({x.filename for x in files})==len(files),'Duplicate ZIP member path')
        for info in files:
            require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline exceeded')
            name=PurePosixPath(info.filename);require(not name.is_absolute() and '..' not in name.parts,'Unsafe ZIP member path')
            require((info.external_attr>>16)&0o170000!=0o120000,'Symlink member disallowed')
            require(info.file_size<=128*2**20,'Oversize temporary member')
            suffix=name.suffix.lower();record=info.filename.lower().endswith('.gdf')
            require(shutil.disk_usage(out).free>reserve+(info.file_size if record else 2**20),'Audit would breach retained free space')
            digest=hashlib.sha256();count=0;target=Path(temp)/'current.gdf'
            dst=target.open('xb') if record else None
            try:
                with z.open(info) as src:
                    while True:
                        block=src.read(2**20)
                        if not block:break
                        count+=len(block);digest.update(block)
                        if dst:dst.write(block)
                if dst:dst.close()
                require(count==info.file_size,'ZIP member size mismatch')
                meta={'member':info.filename,'bytes':count,'compressed_bytes':info.compress_size,'crc32':f'{info.CRC:08x}','sha256':digest.hexdigest(),'zip_CRC_verified_on_full_read':True,'kind':'GDF_signal' if record else 'companion_'+(suffix or 'no_extension')}
                memberledger.write(json.dumps(meta,sort_keys=True)+'\n');memberledger.flush();members.append(meta)
                if record:
                    r=audit_gdf(target,info.filename);require(r['source_file_sha256']==meta['sha256'],'Extracted GDF differs from member stream')
                    records.append(r);ledger.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n');ledger.flush()
                    if len(records)%50==0:print(json.dumps({'records_audited':len(records),'members_audited':len(members),'time':time.time()}),flush=True)
                else:other[suffix or 'no_extension']+=1;other_bytes[suffix or 'no_extension']+=count
            finally:
                if dst and not dst.closed:dst.close()
                if target.exists():target.unlink()
    final_sha=sha(archive);final=archive.stat()
    require(final_sha==receipt['full_SHA256'] and (initial.st_size,initial.st_mtime_ns,initial.st_ino)==(final.st_size,final.st_mtime_ns,final.st_ino),'Archive changed or whole SHA differs')
    unique={};duplicates=[]
    for r in records:
        key=(r['eeg_payload_sha256'],r['EEG_calibration_sha256'],tuple(r['eeg_channels']),r['native_sample_rate_hz'],r['samples_per_EEG_channel'])
        if key in unique:duplicates.append({'member':r['source_member'],'same_as':unique[key]['source_member']})
        else:unique[key]=r
    kept=list(unique.values());groups=collections.defaultdict(list)
    for r in records:groups[r['subject_group']].append(r)
    maincodes=collections.Counter()
    for r in records:maincodes.update(r['annotation_audit']['event_code_counts'])
    result={'schema':'xavier_zenodo8089820_GDF_quantity_summary_v2','status':'completed','completed_epoch':time.time(),
      'source_archive_sha256':final_sha,'source_archive_bytes':final.st_size,'all_members_CRC_SHA_verified':True,'archive_members':len(members),
      'archive_uncompressed_bytes':sum(x['bytes'] for x in members),'GDF_records':len(records),'unique_EEG_records':len(kept),'duplicate_EEG_records':len(duplicates),
      'source_group_count':len(groups),'subdataset_group_counts':dict(collections.Counter(k.split(':')[2] for k in groups)),
      'record_count_per_group_distribution':dict(collections.Counter(str(len(v)) for v in groups.values())),
      'stored_signal_seconds_deduplicated':sum(r['stored_signal_seconds'] for r in kept),
      'stored_signal_hours_deduplicated':sum(r['stored_signal_seconds'] for r in kept)/3600,
      'EEG_sample_bytes_deduplicated':sum(r['EEG_sample_bytes'] for r in kept),
      'EOG_sample_bytes_all_records':sum(r['EOG_sample_bytes'] for r in records),'EMG_sample_bytes_all_records':sum(r['EMG_sample_bytes'] for r in records),
      'GDF_header_bytes_all_records':sum(r['header_bytes'] for r in records),'GDF_event_bytes_all_records':sum(r['event_bytes'] for r in records),
      'GDF_total_bytes':sum(r['source_file_bytes'] for r in records),
      'native_rate_distribution':dict(collections.Counter(str(r['native_sample_rate_hz']) for r in records)),
      'channel_count_distribution':dict(collections.Counter('27EEG+3EOG+2EMG/'+str(len(r['all_channels'])) for r in records)),
      'GDF_version_distribution':dict(collections.Counter(str(r['GDF_version']) for r in records)),
      'recording_kind_counts':dict(collections.Counter(r['recording_kind'] for r in records)),
      'all_event_code_counts':dict(sorted(maincodes.items(),key=lambda x:int(x[0]))),
      'MI_cue_event_counts':dict(collections.Counter(e['label'] for r in records for e in r['annotation_audit']['MI_cue_events'])),
      'records_with_event_outside_record':sum(bool(r['annotation_audit']['events_outside_record']) for r in records),
      'event_timebase_differs_from_signal_records':sum(not r['annotation_audit']['rate_equals_signal'] for r in records),
      'records_missing_explicit_event_timebase':sum(not r['annotation_audit']['explicit_event_timebase_verified'] for r in records),
      'records_with_outside_event_under_native_reader_fallback':sum(bool(r['annotation_audit']['events_outside_record_using_native_reader_fallback']) for r in records),
      'MI_cues_outside_record_under_native_reader_fallback':sum(not e['within_record_under_native_reader_fallback'] for r in records for e in r['annotation_audit']['MI_cue_events']),
      'event_rate_field_value_counts':dict(collections.Counter(str(r['annotation_audit']['event_rate_field_original']) for r in records)),
      'action_label_timestamp_validated_claim':False,
      'companion_count_by_suffix':dict(other),'companion_uncompressed_bytes_by_suffix':dict(other_bytes),
      'recordings_sha256':sha(out/'recordings.jsonl'),'archive_members_sha256':sha(out/'archive_members.jsonl'),'protocol_sha256':sha(out/'protocol.json'),
      'no_waveform_transform_or_relabeling':True,'trained_gradient_hours_added':0,
      'limitations':['Quantity/header/annotation audit is not a physiological quality approval or training split.',
         'Original source groups are publisher IDs; membership across unrelated releases is not inferred.',
         'Exact EEG payload/calibration duplicates are removed; resampled or differently encoded equivalents are not claimed exhaustively detected.',
         '27EEG channels require a new input protocol; existing64channel cohorts/models remain unchanged.']}
    save(out/'duplicates.json',duplicates);save(out/'summary.json',result)
    save(out/'completion.json',{'status':'completed','summary_sha256':sha(out/'summary.json'),'archive_sha256':final_sha,'source_stat_unchanged':True,'all_members_SHA_CRC_verified':True})
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ['archive','acquisition-receipt','resource','output']:p.add_argument('--'+name,required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True))
if __name__=='__main__':main()
