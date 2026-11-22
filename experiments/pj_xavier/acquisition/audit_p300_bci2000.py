"""Read-only quantity and state audit for pinned OpenNeuro ds005028.

Canonical durations come from BCI2000 samples/rate, never the paired EDF's
conversion metadata. The audit does not establish physiological quality,
state-to-stimulus latency, or free-form intent labels, and creates no split.
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
from urllib.parse import unquote
import numpy as np
import acquire_openneuro_pinned as pinned

FORMAT_URL = 'https://www.bci2000.org/mediawiki/index.php/Technical_Reference:BCI2000_File_Format'
UNITS_URL = 'https://www.bci2000.org/mediawiki/index.php/User_Reference:DataIOFilter'
STATES_URL = 'https://www.bci2000.org/mediawiki/index.php/Technical_Reference:State_Definition'
SCALP = ['Fz','FC1','FCz','FC2','FC4','FC6','C4','C6','CP4','CP6','FC3','FC5','C3','C5','CP3','CP5',
         'CP1','P1','Cz','CPz','Pz','POz','CP2','P2','PO7','PO3','O1','Oz','Iz','O2','PO4','PO8']
STATE_FIELDS = ['StimulusCode','StimulusType','StimulusBegin','SelectedTarget','PhaseInSequence','Running','Recording']


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def number(token, suffix=''):
    if suffix and token.endswith(suffix):
        token = token[:-len(suffix)]
    require(re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', token) is not None,
            'Unrecognized numeric parameter unit or syntax')
    value = float(token)
    require(math.isfinite(value), 'Nonfinite header parameter')
    return value


def parse_header_bytes(header, file_bytes):
    text = header.decode('ascii')
    first = text.splitlines()[0]
    fields = {k.lower(): v for k, v in re.findall(r'(\w+)=\s*([^\s]+)', first)}
    require(fields.get('bci2000v') == '1.1', 'Unreviewed BCI2000 version')
    header_len = int(fields['headerlen']); channels = int(fields['sourcech'])
    state_bytes = int(fields.get('statevectorlen', fields.get('statevectorlength', '-1')))
    formats = {'int16': ('<i2',2), 'int32': ('<i4',4), 'float32': ('<f4',4)}
    require(fields.get('dataformat') in formats, 'Unsupported binary scalar format')
    dtype, width = formats[fields['dataformat']]
    require(len(header) == header_len and 0 < header_len <= 2**20 and 0 < channels <= 256
            and 0 < state_bytes <= 1024, 'Header dimensions invalid')
    frame_bytes = channels*width+state_bytes
    require(file_bytes > header_len and (file_bytes-header_len) % frame_bytes == 0, 'Truncated sample/state frame')
    state_section = re.search(r'\[\s*State Vector Definition\s*\](.*?)\[\s*Parameter Definition\s*\]',text,re.S)
    require(state_section is not None, 'State/parameter sections absent')
    states = {}; occupied = set()
    for line in state_section.group(1).splitlines():
        if not line.strip():
            continue
        parts = line.split(); require(len(parts) == 5, 'Unexpected state definition')
        name = parts[0]; length, initial, byte, bit = map(int, parts[1:])
        require(name not in states and 0 < length <= 64 and 0 <= bit < 8 and byte >= 0
                and bit+length <= 64 and byte*8+bit+length <= state_bytes*8
                and 0 <= initial < 2**length, 'Invalid state bit layout')
        bits = set(range(byte*8+bit, byte*8+bit+length))
        require(not occupied.intersection(bits), 'Overlapping state fields')
        occupied.update(bits)
        states[name] = {'length':length,'initial':initial,'byte':byte,'bit':bit}
    needed = {'SamplingRate','SourceCh','SampleBlockSize','ChannelNames','SourceChGain','SourceChOffset'}
    parameters = {}
    for line in text[state_section.end():].splitlines():
        match = re.match(r'\S+\s+\S+\s+(\w+)=\s*(.*)',line)
        if match and match.group(1) in needed:
            key = match.group(1); require(key not in parameters, 'Duplicate required parameter')
            parameters[key] = match.group(2).split('//',1)[0].split()
    require(set(parameters) == needed, 'Required calibration/rate/name parameters absent')
    require(int(parameters['SourceCh'][0]) == channels, 'Channel counts disagree')
    fs = number(parameters['SamplingRate'][0],'Hz'); block = int(parameters['SampleBlockSize'][0])
    require(fs > 0 and block > 0, 'Invalid sample rate or block size')
    lists = {}
    for key in ('ChannelNames','SourceChGain','SourceChOffset'):
        values = parameters[key]; require(int(values[0]) == channels and len(values) >= channels+1, 'Channel list count differs')
        lists[key] = values[1:channels+1]
    names = [unquote(x) for x in lists['ChannelNames']]
    require(len(set(names)) == channels, 'Repeated channel name')
    gain = [number(x,'uV') for x in lists['SourceChGain']]
    offset = [number(x) for x in lists['SourceChOffset']]
    samples = (file_bytes-header_len)//frame_bytes
    return {'version':'1.1','header_bytes':header_len,'source_channels':channels,'state_bytes_per_sample':state_bytes,
            'scalar_dtype':dtype,'scalar_bytes':width,'frame_bytes':frame_bytes,'sample_count':samples,
            'sample_rate_hz':fs,'sample_block_size':block,'stored_signal_seconds':samples/fs,
            'channel_names':names,'gain_AD_to_microV':gain,'offset_AD_units':offset,'states':states,
            'calibration_formula':'physical_microV=(raw_AD-SourceChOffset)*SourceChGain',
            'calibration_scope':'Declared BCI2000 software calibration; not independent hardware validation',
            'header_sha256':hashlib.sha256(header).hexdigest()}


def dat_header(path):
    path = Path(path)
    with path.open('rb') as f:
        first = f.readline(2048)
        m = re.search(br'HeaderLen=\s*(\d+)',first); require(m is not None, 'Header length absent')
        n = int(m.group(1)); require(0 < n <= 2**20, 'Header length bound exceeded')
        f.seek(0); header = f.read(n)
    return parse_header_bytes(header,path.stat().st_size),header


def decode_state(raw, state):
    require(raw.ndim == 2 and raw.dtype == np.uint8, 'State buffer must be uint8 samples by bytes')
    nbytes = (state['bit']+state['length']+7)//8
    require(state['byte']+nbytes <= raw.shape[1] and nbytes <= 8, 'State outside frame')
    values = np.zeros(raw.shape[0],dtype=np.uint64)
    for i in range(nbytes):
        values |= raw[:,state['byte']+i].astype(np.uint64) << np.uint64(8*i)
    return (values >> np.uint64(state['bit'])) & np.uint64(2**state['length']-1)


def audit_dat(path, relative, expected_sha):
    path = Path(path); before = path.stat(); h,header = dat_header(path)
    require(h['source_channels'] == 32 and set(h['channel_names']) == set(SCALP), 'Unreviewed source-specific EEG montage')
    require(h['sample_rate_hz'] == 256, 'Unreviewed native sample rate')
    whole = hashlib.sha256(header); eeg_hash = hashlib.sha256(); state_hash = hashlib.sha256()
    counts = {k:collections.Counter() for k in STATE_FIELDS if k in h['states']}
    previous = {k:0 for k in counts}; transitions = collections.Counter(); events=[]; selected=[]
    nonfinite = np.zeros(32,dtype=np.int64); minimum=np.full(32,np.inf); maximum=np.full(32,-np.inf)
    done = 0; signal_bytes = h['source_channels']*h['scalar_bytes']
    with path.open('rb') as f:
        f.seek(h['header_bytes'])
        while done < h['sample_count']:
            take = min(8192,h['sample_count']-done); raw=f.read(take*h['frame_bytes'])
            require(len(raw) == take*h['frame_bytes'], 'Truncated DAT payload')
            whole.update(raw); frames=np.frombuffer(raw,dtype=np.uint8).reshape(take,h['frame_bytes'])
            signal=np.ascontiguousarray(frames[:,:signal_bytes]); states=np.ascontiguousarray(frames[:,signal_bytes:])
            eeg_hash.update(signal.tobytes());state_hash.update(states.tobytes())
            values=np.frombuffer(signal.tobytes(),dtype=h['scalar_dtype']).reshape(take,32)
            finite=np.isfinite(values);nonfinite+=(~finite).sum(axis=0)
            minimum=np.minimum(minimum,np.where(finite,values,np.inf).min(axis=0));maximum=np.maximum(maximum,np.where(finite,values,-np.inf).max(axis=0))
            decoded={k:decode_state(states,h['states'][k]) for k in counts}
            for key,x in decoded.items():
                u,n=np.unique(x,return_counts=True);counts[key].update({int(v):int(c) for v,c in zip(u,n)})
                prev=np.concatenate((np.array([previous[key]],dtype=np.uint64),x[:-1]));changed=x!=prev
                transitions[key]+=int(np.count_nonzero(changed))
                if key=='StimulusBegin':
                    for index in np.flatnonzero((x!=0)&(prev==0)):
                        events.append({'sample_index':int(done+index),'stored_sample_time_s':float((done+index)/h['sample_rate_hz']),
                         'StimulusCode':int(decoded['StimulusCode'][index]) if 'StimulusCode' in decoded else None,
                         'StimulusType':int(decoded['StimulusType'][index]) if 'StimulusType' in decoded else None,
                         'first_sample_carry_state_possible':bool(done+index==0)})
                elif key=='SelectedTarget':
                    for index in np.flatnonzero(changed&(x!=0)):
                        selected.append({'sample_index':int(done+index),'stored_sample_time_s':float((done+index)/h['sample_rate_hz']),
                                         'source_selected_target_code':int(x[index])})
                previous[key]=int(x[-1])
            done+=take
        require(f.read(1)==b'', 'Unexpected payload tail')
    after=path.stat();require((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns), 'DAT changed during stream')
    require(whole.hexdigest()==expected_sha, 'DAT full SHA differs after stream')
    match=re.search(r'(sub-[^/]+)/([^/]+)/eeg/[^/]+\.dat$',relative);require(match is not None, 'Unrecognized original subject/session path')
    calibration={'channels':h['channel_names'],'gain':h['gain_AD_to_microV'],'offset':h['offset_AD_units']}
    return {'source_path':relative,'source_file_sha256':expected_sha,'source_file_bytes':before.st_size,
      'subject_group':'openneuro:ds005028:'+match.group(1),'publisher_session':match.group(2),
      'publisher_session_is_not_project_split':True,**h,'EEG_payload_sha256':eeg_hash.hexdigest(),
      'state_payload_sha256':state_hash.hexdigest(),'EEG_sample_bytes':h['sample_count']*signal_bytes,
      'state_payload_bytes':h['sample_count']*h['state_bytes_per_sample'],
      'calibration_sha256':hashlib.sha256(json.dumps(calibration,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
      'nonfinite_raw_values_per_channel':[int(x) for x in nonfinite],
      'raw_AD_min_per_channel':[float(x) if np.isfinite(x) else None for x in minimum],
      'raw_AD_max_per_channel':[float(x) if np.isfinite(x) else None for x in maximum],
      'source_state_value_sample_counts':{k:{str(a):b for a,b in sorted(v.items())} for k,v in counts.items()},
      'source_state_transition_counts':dict(transitions),'stored_stimulus_begin_rising_transitions':events,
      'stored_selected_target_nonzero_transitions':selected,
      'labels_scope':'Observed application state codes; not gold free-form intent. Timing/target ontology require a separate protocol.',
      'plain_state_one_block_delay_possible':True,'nominal_block_duration_s':h['sample_block_size']/h['sample_rate_hz'],
      'waveform_filters_or_rescaling_written':False,'actual_gradient_training_hours_added':0}


def edf_header(path):
    path=Path(path)
    with path.open('rb') as f:
        fixed=f.read(256);require(len(fixed)==256,'Truncated EDF prefix')
        ns=int(fixed[252:256]);header_bytes=int(fixed[184:192]);records=int(fixed[236:244]);duration=float(fixed[244:252])
        require(0 < ns <= 256 and header_bytes == (ns+1)*256 and duration>0 and records>0,'Unreviewed EDF layout')
        rest=f.read(header_bytes-256);require(len(rest)==header_bytes-256,'Truncated EDF channel headers')
    offset=0;fields={}
    for key,width in [('labels',16),('transducer',80),('units',8),('physical_min',8),('physical_max',8),('digital_min',8),('digital_max',8),('prefilter',80),('samples_per_record',8),('reserved',32)]:
        fields[key]=[rest[offset+i*width:offset+(i+1)*width].decode('ascii').strip() for i in range(ns)];offset+=ns*width
    samples=[int(x) for x in fields['samples_per_record']]
    require(all(n>0 for n in samples),'Invalid EDF samples per record')
    require(header_bytes+2*records*sum(samples)==path.stat().st_size,'EDF sample bytes do not exhaust file')
    return {'header_bytes':header_bytes,'channels':ns,'channel_labels':fields['labels'],'physical_units':fields['units'],
      'declared_record_duration_s':duration,'records':records,'declared_stored_duration_s':records*duration,
      'samples_per_channel':[n*records for n in samples],'declared_rate_per_channel_hz':[n/duration for n in samples],
      'header_sha256':hashlib.sha256(fixed+rest).hexdigest(),'data_bytes':2*records*sum(samples),
      'physical_min':fields['physical_min'],'physical_max':fields['physical_max'],
      'digital_min':fields['digital_min'],'digital_max':fields['digital_max']}


def run(args):
    from threadpoolctl import threadpool_limits
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    completion=json.loads(Path(args.download_completion).read_text());budget=json.loads(Path(args.resource).read_text())
    require(completion['status']=='all_files_downloaded_checksum_verified' and completion['complete_files']==216,'Release not fully acquired')
    require(time.time()<budget['absolute_deadline_epoch'],'Audit resource deadline expired')
    require(shutil.disk_usage(args.raw_root).free>budget['effective_download_reserve_bytes']+64*2**20,'Audit reserve not available')
    db=sqlite3.connect(Path(args.ledger).resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    jobs=[dict(x) for x in db.execute('SELECT * FROM downloads ORDER BY relative_path')]
    require(len(jobs)==216 and all(x['status']=='complete' for x in jobs),'Complete ledger differs')
    protocol={'schema':'xavier_P300_original_DAT_audit_v1','source_sha256':pinned.sha(__file__),
      'checksum_adapter_sha256':pinned.sha(pinned.__file__),'resource_sha256':pinned.sha(args.resource),
      'download_completion_sha256':pinned.sha(args.download_completion),'absolute_deadline_epoch':budget['absolute_deadline_epoch'],
      'source_release_key':pinned.RELEASE,'source_catalog':pinned.CATALOG,'cpu_threads':2,'canonical_format':'BCI2000 DAT',
      'derived_EDF_time_not_added':True,'BCI2000_format':FORMAT_URL,'BCI2000_calibration':UNITS_URL,'BCI2000_states':STATES_URL,
      'no_signal_filter_rescale_or_new_training':True,'source_version_policy':budget['source_version_policy']}
    pinned.immutable(out/'protocol.json',protocol)
    records=[];derived=[];file_ledger=[];stats={};canonical={}
    with threadpool_limits(limits=2),(out/'records.jsonl').open('x') as record_file,(out/'files.jsonl').open('x') as files:
        for job in jobs:
            require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline exceeded')
            path=Path(args.raw_root)/job['dataset_key']/job['relative_path']
            require(path.resolve().is_relative_to(Path(args.raw_root).resolve()),'Source escaped raw root')
            before=path.stat();digest=pinned.verify_file(path,job['expected_bytes'],json.loads(job['checksum_json']))
            require(digest==job['sha256'],'Completion SHA differs from verified source')
            suffix=path.suffix.lower()
            if suffix=='.dat':
                r=audit_dat(path,job['relative_path'],digest);records.append(r);canonical[job['relative_path'][:-4]]=r
                record_file.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n');record_file.flush()
            elif suffix=='.edf':
                derived.append({'source_path':job['relative_path'],'source_file_sha256':digest,'source_file_bytes':before.st_size,**edf_header(path)})
            after=path.stat();require((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'Source changed during audit')
            stats[str(path)]=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
            entry={'job_id':job['id'],'source_path':job['relative_path'],'bytes':before.st_size,'sha256':digest,
                   'kind':'original_BCI2000' if suffix=='.dat' else 'paired_EDF_export' if suffix=='.edf' else 'metadata',
                   'published_checksum_semantics':pinned.checksum_semantics(json.loads(job['checksum_json']),job['expected_bytes'])}
            files.write(json.dumps(entry,sort_keys=True)+'\n');files.flush();file_ledger.append(entry)
            if len(file_ledger)%20==0:print(json.dumps({'audited_files':len(file_ledger),'original_DAT_records':len(records),'epoch':time.time()}),flush=True)
    require(len(records)==105 and len(derived)==105,'Original/paired record count differs')
    pairs=[]
    for r in derived:
        original=canonical.get(r['source_path'][:-4]);require(original is not None,'EDF lacks same-path DAT counterpart')
        pairs.append({**r,'canonical_DAT_path':original['source_path'],'canonical_DAT_SHA256':original['source_file_sha256'],
         'sample_count_equal_to_DAT':r['channels']==32 and r['samples_per_channel']==[original['sample_count']]*32,
         'declared_rate_equal_to_DAT':r['declared_rate_per_channel_hz']==[original['sample_rate_hz']]*r['channels'],
         'all_physical_units_empty':not any(r['physical_units']),
         'canonical_original_hours_added_for_this_export':0,
         'lineage_scope':'Same-path paired export and sample-count relation; exact decoded waveform equivalence not claimed'})
    with (out/'paired_EDF_exports.jsonl').open('x') as f:
        for p in pairs:f.write(json.dumps(p,sort_keys=True)+'\n')
    unique={};duplicates=[]
    for r in records:
        key=(r['EEG_payload_sha256'],r['calibration_sha256'],tuple(r['channel_names']),r['sample_count'],r['sample_rate_hz'])
        if key in unique:duplicates.append({'source_path':r['source_path'],'same_as':unique[key]['source_path'],'subject_groups':sorted({r['subject_group'],unique[key]['subject_group']}),'must_link_if_cross_group':r['subject_group']!=unique[key]['subject_group']})
        else:unique[key]=r
    seconds=sum((Fraction(r['sample_count'],256) for r in unique.values()),Fraction())
    for name,expected in stats.items():
        st=Path(name).stat();require((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==expected,'Source stat changed before audit completion')
    result={'schema':'xavier_P300_original_DAT_quantity_summary_v1','status':'completed','completed_epoch':time.time(),
      'all_source_files':len(jobs),'acquired_distribution_bytes':sum(x['bytes'] for x in file_ledger),
      'all_published_checksums_verified':True,'original_DAT_records':len(records),'unique_original_EEG_records':len(unique),
      'exact_original_EEG_duplicates':len(duplicates),'source_participant_groups':len({r['subject_group'] for r in records}),
      'original_EEG_seconds_exact_fraction':[seconds.numerator,seconds.denominator],
      'original_EEG_hours_deduplicated':float(seconds/3600),
      'original_EEG_sample_bytes_deduplicated':sum(r['EEG_sample_bytes'] for r in unique.values()),
      'original_DAT_header_bytes_all':sum(r['header_bytes'] for r in records),
      'original_DAT_state_payload_bytes_all':sum(r['state_payload_bytes'] for r in records),
      'original_DAT_file_bytes_all':sum(r['source_file_bytes'] for r in records),
      'paired_EDF_file_bytes':sum(r['source_file_bytes'] for r in pairs),'paired_EDF_records':len(pairs),
      'paired_EDF_sample_count_mismatch_records':sum(not r['sample_count_equal_to_DAT'] for r in pairs),
      'paired_EDF_declared_rate_mismatch_records':sum(not r['declared_rate_equal_to_DAT'] for r in pairs),
      'paired_EDF_empty_physical_unit_records':sum(r['all_physical_units_empty'] for r in pairs),
      'paired_EDF_recording_hours_added':0,'native_rate_hz_distribution':dict(collections.Counter(str(r['sample_rate_hz']) for r in records)),
      'EEG_channel_count_distribution':dict(collections.Counter(str(r['source_channels']) for r in records)),
      'scalar_dtype_distribution':dict(collections.Counter(r['scalar_dtype'] for r in records)),
      'original_stored_stimulus_begin_rising_transition_count':sum(len(r['stored_stimulus_begin_rising_transitions']) for r in records),
      'original_stored_selected_target_nonzero_transition_count':sum(len(r['stored_selected_target_nonzero_transitions']) for r in records),
      'raw_nonfinite_value_count':sum(sum(r['nonfinite_raw_values_per_channel']) for r in records),
      'source_README_analysis_exclusion_flag_present':True,
      'source_README_flag_scope':'One source participant omitted by authors because testing did not include all flashing paradigms; source records remain counted',
      'labels_are_validated_free_form_intent':False,'external_stimulus_latency_validated':False,
      'state_timebase_scope':'Per-sample saved state positions; plain BCI2000 states may be delayed one block relative to signals; no delay correction applied',
      'new_source_split_or_supervised_protocol_required':True,'actual_gradient_training_hours_added':0,
      'existing_model_cohort_changed':False,'records_sha256':pinned.sha(out/'records.jsonl'),
      'paired_exports_sha256':pinned.sha(out/'paired_EDF_exports.jsonl'),'files_sha256':pinned.sha(out/'files.jsonl'),
      'protocol_sha256':pinned.sha(out/'protocol.json'),'source_stats_checked_unchanged':True}
    pinned.immutable(out/'duplicates.json',duplicates);pinned.immutable(out/'summary.json',result)
    pinned.immutable(out/'completion.json',{'status':'completed','summary_sha256':pinned.sha(out/'summary.json'),
                     'all216_published_checksums_and_source_stat_verified':True,'original_rows_reconciled':False})
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ['ledger','raw-root','download-completion','resource','output']:p.add_argument('--'+name,required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True),flush=True)


if __name__=='__main__':main()
