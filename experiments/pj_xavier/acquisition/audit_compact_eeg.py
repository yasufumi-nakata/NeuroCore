"""Read-only measured quantity/label audit of two pinned public EEG releases.

No fitting, relabeling, normalization, raw mutation, or benchmark inclusion.
Text units absent from publisher metadata remain unknown. MNE's FIF header
calibration is reported as declared, without asserting physical validity.
"""
from __future__ import annotations
import argparse, collections, hashlib, importlib.util, json, os, re, sqlite3, time
from pathlib import Path
import numpy as np

TEXT_CHANNELS=['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
FIF_CHANNELS=['Fpz','F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8']


def require(ok,reason):
    if not ok:raise ValueError(reason)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def decoded_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values,dtype='<f8').tobytes()).hexdigest()


def audit_text(path):
    rows=[];widths=collections.Counter();eeg_bytes=gyro_bytes=other_bytes=0
    with Path(path).open('rb') as f:
        for line in f:
            tokens=line.split()
            if not tokens:continue
            require(len(tokens)>=18,'Text record has fewer than declared EEG/gyro columns')
            widths[len(tokens)]+=1
            rows.append([float(x) for x in tokens])
            eeg_bytes+=sum(len(x) for x in tokens[2:16]);gyro_bytes+=sum(len(x) for x in tokens[16:18])
            other_bytes+=sum(len(x) for x in tokens[:2]+tokens[18:])
    require(rows and len(widths)==1,'Empty/inconsistent text table')
    x=np.asarray(rows,dtype=np.float64);require(np.isfinite(x).all(),'Nonfinite raw text numeric value')
    m=re.fullmatch(r'user([0-9]{3})_(10|40|70)_([12])\.txt',Path(path).name)
    require(m is not None,'Unexpected publisher filename schema')
    n=len(x);fs=128.;time_s=x[:,0]/12.8
    increments=np.diff(time_s);duration=n/fs
    counter=x[:,1];counter_integral=bool(np.equal(counter,np.floor(counter)).all())
    counter_steps=np.diff(counter);counter_bad=int(np.count_nonzero(~np.isin(counter_steps,[1,-127])))
    schedule=[{'onset_s':2.9+8*i,'duration_s':5.,'within_stored_record':bool(2.9+8*i+5<=duration)} for i in range(5)]
    return {'format':'publisher_ascii_numeric_table','subject_group':'figshare:25773342:1:user'+m.group(1),
        'sample_count_per_EEG_channel':n,'eeg_channels':TEXT_CHANNELS,'eeg_channel_count':14,
        'sample_rate_hz':fs,'sample_rate_evidence':'official_release_description; clock column divided by12.8 audited separately',
        'stored_signal_seconds':duration,'eeg_scalar_count':n*14,'gyro_scalar_count':n*2,
        'eeg_numeric_payload_bytes':eeg_bytes,'gyro_numeric_payload_bytes':gyro_bytes,
        'other_numeric_payload_bytes':other_bytes,
        'payload_byte_definition':'Exact UTF8 byte lengths of EEG numeric tokens; excludes whitespace and other columns; not a float32 allocation estimate',
        'eeg_float32_equivalent_bytes_not_stored_binary':n*14*4,
        'unit':'unresolved_publisher_does_not_state_amplitude_unit','physical_V_conversion_verified':False,
        'decoded_payload_sha256':decoded_sha(x[:,2:16]),
        'decoded_payload_hash_basis':'row-major little-endian float64 numeric EEG values in unresolved publisher units',
        'observed_numeric_column_count':x.shape[1],
        'clock_audit':{'first_s':float(time_s[0]),'last_s':float(time_s[-1]),
          'elapsed_plus_one_sample_s':float(time_s[-1]-time_s[0]+1/fs),'monotonic_strict':bool((increments>0).all()),
          'max_step_deviation_from_1_over_128_s':float(np.max(np.abs(increments-1/fs))) if n>1 else None,
          'counter_integral':counter_integral,'counter_nonincrement_or_nonwrap_transitions':counter_bad},
        'label_audit':{'label_source':'publisher_filename_condition_and_description_schedule; per-trial markers not embedded',
          'force_percent_MVC_condition':int(m.group(2)),'repetition':int(m.group(3)),
          'scheduled_trials':schedule,'task':'right-hand kinesthetic motor imagery force level',
          'not_free_form_intent_label':True,'training_eligible_under_current_64channel_V_protocol':False}}


def audit_fif(path,expected_channels=None):
    import mne
    raw=mne.io.read_raw_fif(path,preload=False,verbose=False)
    types=raw.get_channel_types();eeg=[i for i,t in enumerate(types) if t=='eeg'];stim=[i for i,t in enumerate(types) if t=='stim']
    require(eeg and stim,'Expected EEG and stimulation channels absent')
    names=[raw.ch_names[i] for i in eeg]
    if expected_channels is not None:require(names==expected_channels,'FIF EEG channel order differs from pinned publisher order')
    fs=float(raw.info['sfreq']);n=int(raw.n_times);nchan=len(raw.ch_names)
    payload=0;eeg_payload=0;buffer_types=collections.Counter();stored_samples=0
    # FIF data buffers interleave all declared channels in a single scalar
    # representation. Check every buffer against its actual sample bounds.
    size_for_type={2:2,3:4,4:4,5:8,16:2}
    for extra in raw._raw_extras:
        require(extra['orig_nchan']==nchan,'FIF original channel count mismatch')
        bounds=extra['bounds'];entries=extra['ent'];require(len(bounds)==len(entries)+1,'FIF bounds mismatch')
        for k,entry in enumerate(entries):
            require(entry is not None,'FIF skipped interval requires separate quantity review')
            typ=int(entry.type);require(typ in size_for_type,'Unsupported FIF data buffer encoding')
            samples=int(bounds[k+1]-bounds[k]);width=size_for_type[typ]
            require(int(entry.size)==samples*nchan*width,'FIF payload size/sample count mismatch')
            payload+=int(entry.size);eeg_payload+=samples*len(eeg)*width;stored_samples+=samples;buffer_types[str(typ)]+=1
    require(stored_samples==n,'Stored FIFO sample count differs from header length')
    hasher=hashlib.sha256();minimum=float('inf');maximum=float('-inf');finite=True
    for start in range(0,n,max(1,int(fs*60))):
        x=raw.get_data(picks=eeg,start=start,stop=min(n,start+int(fs*60)))
        finite=finite and bool(np.isfinite(x).all());minimum=min(minimum,float(x.min()));maximum=max(maximum,float(x.max()))
        # Stable sample-major stream independent of read chunk size.
        hasher.update(np.ascontiguousarray(x.T,dtype='<f8').tobytes())
    require(finite,'Nonfinite decoded FIF EEG')
    events=mne.find_events(raw,stim_channel=[raw.ch_names[i] for i in stim],shortest_event=1,consecutive=True,initial_event=True,verbose=False)
    counts=collections.Counter(int(e[2]) for e in events);semantic={2:'right_hand_imagery',3:'both_feet_imagery',4:'rest'}
    labels=[];outside=0
    for sample,previous,code in events:
        if int(code) in semantic:
            onset=(int(sample)-raw.first_samp)/fs;valid=onset>=0 and onset+3<=n/fs
            outside+=not valid;labels.append({'sample_index':int(sample)-raw.first_samp,'onset_s':onset,'duration_s':3.,'code':int(code),'label':semantic[int(code)],'within_stored_record':bool(valid)})
    calibrated=[{'name':raw.ch_names[i],'kind':int(raw.info['chs'][i]['kind']),'unit':int(raw.info['chs'][i]['unit']),
                 'unit_mul':int(raw.info['chs'][i]['unit_mul']),'cal':float(raw.info['chs'][i]['cal']),'range':float(raw.info['chs'][i]['range'])} for i in eeg]
    result={'format':'MNE_FIF','subject_group':'zenodo:806023:'+Path(path).name.split('.')[0],
      'sample_count_per_EEG_channel':n,'sample_rate_hz':fs,'sample_rate_evidence':'actual_FIF_header',
      'stored_signal_seconds':n/fs,'eeg_channels':names,'eeg_channel_count':len(eeg),
      'all_channel_types':dict(collections.Counter(types)),'all_channel_names':raw.ch_names,
      'stim_channel_names':[raw.ch_names[i] for i in stim],'eeg_scalar_count':n*len(eeg),
      'eeg_numeric_payload_bytes':eeg_payload,'all_signal_data_buffer_bytes':payload,'non_EEG_signal_data_buffer_bytes':payload-eeg_payload,
      'data_buffer_type_counts':dict(buffer_types),'payload_byte_definition':'Actual FIF data-buffer scalar bytes for EEG channel positions; excludes FIF tags/metadata and non-EEG channels',
      'unit':'V_as_decoded_using_FIF_header_calibration','EEG_channel_calibration':calibrated,
      'physical_calibration_validity':'Declared header conversion only; external amplifier calibration not independently proven',
      'decoded_payload_sha256':hasher.hexdigest(),'decoded_payload_hash_basis':'sample-major little-endian float64 header-calibrated EEG volts',
      'finite':finite,'decoded_min_V':minimum,'decoded_max_V':maximum,
      'label_audit':{'label_source':'embedded_stimulation_channel_with_official_release_code_mapping',
        'event_code_counts':{str(k):v for k,v in sorted(counts.items())},'trial_start_code':1,'semantic_trial_events':labels,
        'unmapped_positive_codes':[k for k in counts if k not in [1,2,3,4]],'semantic_windows_outside_record':outside,
        'declared_trials_per_class':20,'declared_trial_duration_seconds':3,
        'not_free_form_intent_label':True,'training_eligible_under_current_64channel_V_protocol':False},'mne_version':mne.__version__}
    raw.close();return result


def run(args):
    from threadpoolctl import threadpool_limits
    budget=json.loads(Path(args.resource).read_text());require(time.time()<budget['absolute_deadline_epoch'],'Audit budget expired')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect(Path(args.db).resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    jobs=[dict(r) for r in db.execute('select * from downloads order by catalog_id,relative_path')]
    require(len(jobs)==68 and {r['catalog_id'] for r in jobs}=={'000413','001035'},'Pinned compact scope differs')
    require(all(r['status']=='complete' for r in jobs),'All downloaded files must complete before header audit')
    spec=importlib.util.spec_from_file_location('compact_fixed_acquirer',args.acquire_source);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    protocol={'schema':'pj-xavier-compact-public-EEG-audit-v1','created_epoch':time.time(),'source_sha256':sha(__file__),
      'manifest_sha256':sha(args.manifest),'resource_sha256':sha(args.resource),'expected_files':68,
      'known_source_bytes':186109962,'raw_mutation':False,'training_or_model_inference':False,
      'text_rate_and_column_map':'Official Figshare25773342 release metadata; actual per-record row count and token bytes',
      'text_amplitude_unit':'Unresolved; no conversion or model eligibility assumed',
      'fif_quantity':'Actual MNE FIF header and every data buffer; declared calibrated volts',
      'fif_labels':'Actual embedded stim transitions; mapping1 trial start/2 right hand/3 feet/4 rest from official806023 metadata',
      'duration':'Stored samples / declared native fs per record; no channel or condition multiplier',
      'deduplication':'Within each source: decoded EEG payload SHA and sample/rate/channel count. Cross-release identity is not inferred from participant names.',
      'float32_equivalent_not_actual_bytes':True,'cpu_threads':2,
      'source_metadata_hashes':{Path(f).name:sha(f) for f in args.metadata_files}}
    save(out/'audit_protocol.json',protocol)
    records=[];started=time.time()
    with threadpool_limits(limits=2), (out/'records.jsonl').open('x') as ledger:
        for job in jobs:
            require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline exceeded')
            path=Path(args.raw_root)/job['dataset_key']/job['relative_path'];st=path.stat()
            digest=a.verify_file(path,job['expected_bytes'],json.loads(job['checksum_json']))
            require(digest==job['sha256'],'Downloaded complete SHA differs')
            rec=audit_text(path) if job['catalog_id']=='000413' else audit_fif(path,FIF_CHANNELS)
            require(sha(path)==digest,'Raw file changed during audit');end=path.stat()
            require((st.st_size,st.st_mtime_ns,st.st_ino)==(end.st_size,end.st_mtime_ns,end.st_ino),'Raw stat changed during audit')
            rec.update({'catalog_id':job['catalog_id'],'dataset_key':job['dataset_key'],'relative_path':job['relative_path'],
              'source_file_bytes':st.st_size,'source_sha256':digest,'publisher_checksum':json.loads(job['checksum_json']),
              'source_file_id':job.get('provider_file_id'),'source_before_after_SHA_matches':True})
            records.append(rec);ledger.write(json.dumps(rec,sort_keys=True,allow_nan=False)+'\n');ledger.flush()
    datasets={}
    for key in sorted({r['catalog_id'] for r in records}):
        subset=[r for r in records if r['catalog_id']==key];unique={}
        for r in subset:
            identity=(r['decoded_payload_sha256'],r['sample_count_per_EEG_channel'],r['sample_rate_hz'],r['eeg_channel_count'])
            unique.setdefault(identity,r)
        selected=list(unique.values())
        datasets[key]={'record_count':len(subset),'unique_decoded_EEG_records':len(selected),'duplicate_decoded_EEG_records':len(subset)-len(selected),
          'participant_groups':len({r['subject_group'] for r in selected}),
          'source_file_bytes':sum(r['source_file_bytes'] for r in subset),
          'stored_signal_seconds_deduplicated':sum(r['stored_signal_seconds'] for r in selected),
          'stored_signal_hours_deduplicated':sum(r['stored_signal_seconds'] for r in selected)/3600,
          'eeg_numeric_payload_bytes_deduplicated':sum(r['eeg_numeric_payload_bytes'] for r in selected),
          'eeg_scalar_count_deduplicated':sum(r['eeg_scalar_count'] for r in selected),
          'rate_channel_combinations':dict(collections.Counter(str(r['sample_rate_hz'])+'Hz/'+str(r['eeg_channel_count'])+'ch' for r in selected)),
          'new_protocol_required_before_training':True,'record_ledger_contains_original_label_evidence':True}
        if key=='000413':datasets[key].update({'amplitude_unit_unresolved':True,
          'clock_not_strict_monotonic_records':sum(not r['clock_audit']['monotonic_strict'] for r in subset),
          'counter_discontinuity_records':sum(r['clock_audit']['counter_nonincrement_or_nonwrap_transitions']>0 for r in subset),
          'force_condition_record_counts':dict(collections.Counter(str(r['label_audit']['force_percent_MVC_condition']) for r in subset)),
          'scheduled_trials_per_record':5,'event_onsets_are_publisher_schedule_not_observed_markers':True})
        else:datasets[key].update({'semantic_event_counts':dict(collections.Counter(e['label'] for r in subset for e in r['label_audit']['semantic_trial_events'])),
          'semantic_event_windows_outside_record':sum(r['label_audit']['semantic_windows_outside_record'] for r in subset),
          'unmapped_stim_code_records':sum(bool(r['label_audit']['unmapped_positive_codes']) for r in subset),
          'non_EEG_signal_data_buffer_bytes':sum(r['non_EEG_signal_data_buffer_bytes'] for r in subset)})
    result={'schema':'pj-xavier-compact-public-EEG-audit-summary-v1','status':'completed','started_epoch':started,'completed_epoch':time.time(),
      'record_count':len(records),'source_file_bytes':sum(x['source_file_bytes'] for x in records),
      'all_original_checksums_and_before_after_SHA_verified':True,'protocol_sha256':sha(out/'audit_protocol.json'),
      'records_sha256':sha(out/'records.jsonl'),'datasets':datasets,
      'stored_signal_seconds_deduplicated':sum(d['stored_signal_seconds_deduplicated'] for d in datasets.values()),
      'stored_signal_hours_deduplicated':sum(d['stored_signal_hours_deduplicated'] for d in datasets.values()),
      'actual_gradient_input_hours_added':0,'existing_frozen_datasets_or_models_modified':False,
      'limitations':['Text amplitude unit unresolved; its time is counted as declared-rate stored EEG, not physically calibrated volts.',
       '14ch/16ch sources do not match frozen64ch input; no arbitrary channel padding or relabeling was performed.',
       'Filename/scheduled KMI labels and actual FIF stimulation labels are distinguished; neither demonstrates free-form intent.']}
    save(out/'summary.json',result);save(out/'completion.json',{'status':'completed','summary_sha256':sha(out/'summary.json'),'records_sha256':sha(out/'records.jsonl'),'protocol_sha256':sha(out/'audit_protocol.json'),'raw_files_changed':0})
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ['db','raw-root','resource','manifest','output','acquire-source']:p.add_argument('--'+name,required=True)
    p.add_argument('--metadata-files',nargs='+',required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True))

if __name__=='__main__':main()
