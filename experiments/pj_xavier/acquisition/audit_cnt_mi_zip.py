"""Audit the 26 original CNT members of Figshare2061654v1, without repair.

The observed legacy sample-count field is inconsistent. Precision is resolved
only from a present, structurally verified end tag used by the pinned official
EEGLAB reader, then sample count is derived from the actual signal byte span.
"""
from __future__ import annotations
import argparse
import collections
from fractions import Fraction
import hashlib
import inspect
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import struct
import tempfile
import time
import warnings
import zipfile
import numpy as np
import delegated_acquisition_v2 as custody

EEG = ['Fp1','Fp2','FC3','FCz','FC4','C3','Cz','C4','CP3','CPz','CP4','O1','Oz','O2']
EOG = ['VEOU','VEOL']
READER_SHA = '3156ac652687ef119db883db8f9a75424f1840a6803d9fee021bffec800d208b'


def require(ok, reason):
    if not ok:raise ValueError(reason)


def precision_from_tag(*, nextfile, event_offset, event_length, file_size, tag):
    require(0 < event_offset < file_size and event_length >= 0
            and nextfile == event_offset+9+event_length
            and nextfile+53 <= file_size, 'Missing or structurally inconsistent original format tag')
    require(tag in (0,1), 'Unknown original precision tag; no default int16 guess')
    return 4 if tag == 1 else 2


def layout(path):
    path = Path(path)
    with path.open('rb') as f:
        fixed = f.read(900);require(len(fixed)==900 and fixed[:12]==b'Version 3.0\0','Unreviewed CNT revision')
        n = struct.unpack_from('<H',fixed,370)[0];fs=struct.unpack_from('<H',fixed,376)[0]
        original_n=struct.unpack_from('<i',fixed,864)[0];event_offset=struct.unpack_from('<i',fixed,886)[0]
        nextfile=struct.unpack_from('<i',fixed,12)[0];channel_offset=struct.unpack_from('<i',fixed,894)[0]
        require(n==16 and fs==250 and channel_offset==1,'Unreviewed montage, rate or sample-block layout')
        channel_header=f.read(n*75);require(len(channel_header)==n*75,'Truncated channel header')
        f.seek(event_offset);teeg=f.read(9);require(len(teeg)==9,'Truncated event table header')
        event_type,event_length,event_relative_offset=struct.unpack('<Bii',teeg)
        f.seek(nextfile+52);b=f.read(1);require(len(b)==1,'Original precision marker absent')
        width=precision_from_tag(nextfile=nextfile,event_offset=event_offset,event_length=event_length,
                                 file_size=path.stat().st_size,tag=b[0])
        data_offset=900+n*75;data_bytes=event_offset-data_offset
        require(data_bytes>0 and data_bytes%(n*width)==0,'Signal bytes not a whole sample frame')
        samples=data_bytes//(n*width)
        other_width=2 if width==4 else 4
        require(not (original_n>0 and data_bytes%(n*other_width)==0 and original_n==data_bytes//(n*other_width)),
                'Header sample count directly contradicts precision marker')
        channels=[]
        for i in range(n):
            h=channel_header[i*75:(i+1)*75];name=h[:10].split(b'\0')[0].decode('ascii')
            baseline=struct.unpack_from('<h',h,47)[0];sensitivity=struct.unpack_from('<f',h,59)[0];cal=struct.unpack_from('<f',h,71)[0]
            require(math.isfinite(sensitivity) and math.isfinite(cal),'Nonfinite header calibration')
            channels.append({'name':name,'baseline':baseline,'sensitivity':sensitivity,'calibration':cal,
                             'V_per_AD_unit':cal*sensitivity*1e-6/204.8})
        require(len({c['name'] for c in channels})==n and set(c['name'] for c in channels)==set(EEG+EOG),
                'Source-specific EEG/EOG channels differ')
        f.seek(event_offset+9);events=f.read(event_length);require(len(events)==event_length,'Truncated original event entries')
    return {'revision':'Version 3.0','n_channels':n,'sample_rate_hz':fs,'original_header_sample_count':original_n,
      'actual_sample_count_from_byte_span':samples,'source_sample_count_field_matches':original_n==samples,
      'original_header_continuous_seconds':struct.unpack_from('<f',fixed,890)[0],
      'data_offset':data_offset,'signal_data_bytes':data_bytes,'scalar_bytes':width,'channel_offset':channel_offset,
      'original_nextfile':nextfile,'original_is32bit_tag':b[0],'format_tag_offset':nextfile+52,
      'original_event_table_offset':event_offset,'original_event_table_type':event_type,
      'original_event_table_length':event_length,'original_event_relative_offset':event_relative_offset,
      'tail_metadata_bytes':path.stat().st_size-nextfile,'channels':channels},events


def untranslated_event_sample(offset, event_type, *, n_channels, scalar_bytes, data_offset):
    # Mirrors the public reader's addressing definition but intentionally does
    # not clip a negative result or impute any duration.
    raw_offset = int(offset)*n_channels*scalar_bytes if event_type==3 else int(offset)
    numerator=raw_offset-data_offset;frame=n_channels*scalar_bytes
    return numerator//frame-1, numerator%frame


def audit_member(path, member, expected_tag):
    import mne
    from mne.io.cnt._utils import _get_event_parser
    h,event_bytes=layout(path);require(h['original_is32bit_tag']==expected_tag,'Original tag differs from preflight proof')
    names=[c['name'] for c in h['channels']];eeg_indices=[i for i,n in enumerate(names) if n in EEG];eog_indices=[i for i,n in enumerate(names) if n in EOG]
    hashes={k:hashlib.sha256() for k in ['EEG','EOG']};remaining=h['actual_sample_count_from_byte_span']
    with Path(path).open('rb') as f:
        f.seek(h['data_offset'])
        while remaining:
            n=min(16384,remaining);b=f.read(n*h['n_channels']*h['scalar_bytes']);require(len(b)==n*h['n_channels']*h['scalar_bytes'],'Truncated signal span')
            cells=np.frombuffer(b,dtype=np.uint8).reshape(n,h['n_channels'],h['scalar_bytes'])
            for kind,indices in [('EEG',eeg_indices),('EOG',eog_indices)]:hashes[kind].update(np.ascontiguousarray(cells[:,indices,:]).tobytes())
            remaining-=n
    parsed=list(_get_event_parser(event_type=h['original_event_table_type'])(event_bytes));events=[]
    for e in parsed:
        sample,rem=untranslated_event_sample(e.Offset,h['original_event_table_type'],n_channels=h['n_channels'],scalar_bytes=h['scalar_bytes'],data_offset=h['data_offset'])
        values={}
        for k,v in e._asdict().items():
            if isinstance(v,bytes):values[k]={'raw_hex':v.hex()}
            elif isinstance(v,float) and not math.isfinite(v):values[k]={'nonfinite_original_value':str(v)}
            else:values[k]=v
        events.append({'original_fields':values,'derived_sample_index_without_negative_clip':sample,
          'byte_offset_remainder_within_sample_frame':rem,'derived_onset_s':sample/h['sample_rate_hz'],
          'within_record':0<=sample<h['actual_sample_count_from_byte_span'],'semantic_class':None,
          'duration_imputed':False,'event_offset_repaired':False})
    with warnings.catch_warnings(record=True) as captured:
        raw=mne.io.read_raw_cnt(path,eog=EOG,preload=False,data_format='int16' if h['scalar_bytes']==2 else 'int32',
             recompute_n_samples=True,verbose='ERROR')
        require(raw.n_times==h['actual_sample_count_from_byte_span'] and raw.info['sfreq']==h['sample_rate_hz'] and raw.ch_names==names,'Independent MNE header interpretation differs')
        require(raw.get_channel_types().count('eeg')==14 and raw.get_channel_types().count('eog')==2,'Independent channel roles differ')
        probes=sorted(set([0,max(0,raw.n_times//2-128),max(0,raw.n_times-256)]));max_error=0.0
        for start in probes:
            stop=min(raw.n_times,start+256)
            with Path(path).open('rb') as f:
                f.seek(h['data_offset']+start*h['n_channels']*h['scalar_bytes'])
                b=f.read((stop-start)*h['n_channels']*h['scalar_bytes'])
            ad=np.frombuffer(b,dtype='<i2' if h['scalar_bytes']==2 else '<i4').reshape(stop-start,16).T.astype(np.float64)
            base=np.array([x['baseline'] for x in h['channels']])[:,None];gain=np.array([x['V_per_AD_unit'] for x in h['channels']])[:,None]
            expected=(ad-base)*gain;observed=raw.get_data(start=start,stop=stop)
            error=float(np.max(np.abs(expected-observed)));max_error=max(max_error,error)
            require(error<=1e-12,'Independent MNE sample calibration differs')
        mne_annotation_count=len(raw.annotations);raw.close()
    import re
    match=re.search(r'/(S\d+)_',member);require(match is not None,'Unknown source participant naming')
    calibration=[h['channels'][i] for i in eeg_indices]
    result={'source_member':member,'source_file_sha256':custody.sha(path),'source_file_bytes':Path(path).stat().st_size,
      'subject_group':'figshare:2061654:'+match.group(1),**h,
      'EEG_channels':[names[i] for i in eeg_indices],'EOG_channels':[names[i] for i in eog_indices],
      'EEG_payload_sha256':hashes['EEG'].hexdigest(),'EOG_payload_sha256':hashes['EOG'].hexdigest(),
      'EEG_sample_bytes':h['actual_sample_count_from_byte_span']*14*h['scalar_bytes'],
      'EOG_sample_bytes':h['actual_sample_count_from_byte_span']*2*h['scalar_bytes'],
      'stored_signal_seconds':h['actual_sample_count_from_byte_span']/h['sample_rate_hz'],
      'EEG_calibration_sha256':hashlib.sha256(json.dumps(calibration,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
      'original_event_entries':events,'raw_event_count':len(events),'MNE_annotations_after_reader_rules':mne_annotation_count,
      'independent_MNE_header_and_three_calibrated_sample_probes_match':True,'maximum_probe_difference_V':max_error,
      'MNE_reader_warnings':[str(x.message)[:1200] for x in captured],
      'unit_scope':'Original sensitivity/baseline/calibration interpreted with pinned MNE conversion to V; independent physical amplification calibration not established',
      'original_header_or_event_changes':0,'event_class_ontology_verified':False,
      'supervised_eligibility':'held_for_original_class_mapping_and_new_protocol',
      'SSL_eligibility':'candidate_only_new_units_QC_split_and_14channel_protocol_required','actual_gradient_training_hours_added':0}
    require(result['source_file_bytes']==result['data_offset']+result['EEG_sample_bytes']+result['EOG_sample_bytes']+9+result['original_event_table_length']+result['tail_metadata_bytes'], 'CNT byte partition not exhaustive')
    return result


def run(args):
    import mne
    from mne.io.cnt.cnt import _get_cnt_info
    from mne.io.cnt._utils import _get_event_parser
    from threadpoolctl import threadpool_limits
    archive=Path(args.archive);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    budget=json.loads(Path(args.resource).read_text());proof=json.loads(Path(args.format_proof).read_text())
    receipt=json.loads(Path(args.download_completion).read_text())
    require(receipt['status']=='all_files_downloaded_checksum_verified' and receipt['complete_files']==1,'ZIP not fully acquired')
    require(proof['official_reader_source_sha256']==READER_SHA and proof['nextfile_equals_event_table_end_all'],'Missing primary reader format proof')
    md5=hashlib.md5()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):md5.update(block)
    require(archive.stat().st_size==156181084 and md5.hexdigest()=='c48c82db4c51330285d513c4dbfd11a5','ZIP checksum differs')
    # Whole-file hashing is streamed. No expanded multi-record raw dataset
    # or complete archive is held in memory.
    whole=custody.sha(archive);initial=archive.stat();expected_tags={x['member']:x['original_is32bit_tag'] for x in proof['records']}
    protocol={'schema':'xavier_threeclass_CNT_quantity_protocol_v1','source_sha256':custody.sha(__file__),
      'resource_sha256':custody.sha(args.resource),'source_archive_SHA256':whole,'source_archive_bytes':initial.st_size,
      'format_proof_sha256':custody.sha(args.format_proof),'official_reader_source_URL':proof['official_source_url'],
      'official_reader_source_sha256':READER_SHA,'precision_rule':'Only present nextfile+52 tag0 or1 with nextfile exactly equal to event-table end; no absent-tag default',
      'header_count_policy':'Original sample count preserved; derive positive sample count from exact signal byte span and independently evidenced scalar width',
      'mne_version':mne.__version__,'mne_header_reader_sha256':hashlib.sha256(inspect.getsource(_get_cnt_info).encode()).hexdigest(),
      'mne_original_event_parser_sha256':hashlib.sha256(inspect.getsource(_get_event_parser).encode()).hexdigest(),
      'absolute_deadline_epoch':budget['absolute_deadline_epoch'],'cpu_threads':2,'single_member_limit_bytes':32*2**20,
      'supervised_event_class_mapping_verified':False,'no_waveform_filters_or_relabeling':True,'training_calls':0}
    custody.immutable(out/'protocol.json',protocol);records=[];members=[]
    with threadpool_limits(limits=2),zipfile.ZipFile(archive) as z,tempfile.TemporaryDirectory(dir=out,prefix='one_CNT_') as temp,(out/'records.jsonl').open('x') as ledger:
        infos=[x for x in z.infolist() if not x.is_dir()];require(len(infos)==26 and len({x.filename for x in infos})==26,'Unexpected member count or duplicate path')
        for info in sorted(infos,key=lambda x:x.filename):
            require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline exceeded')
            require(info.filename in expected_tags and info.filename.lower().endswith('.cnt'),'Unreviewed archive member')
            pp=PurePosixPath(info.filename);require(not pp.is_absolute() and '..' not in pp.parts and (info.external_attr>>16)&0o170000!=0o120000,'Unsafe member')
            require(info.file_size<=32*2**20 and shutil.disk_usage(out).free>budget['effective_download_reserve_bytes']+info.file_size,'Temporary member exceeds reserve')
            target=Path(temp)/'current.cnt'
            with z.open(info) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,length=2**20)
            require(target.stat().st_size==info.file_size,'Extracted member size mismatch')
            r=audit_member(target,info.filename,expected_tags[info.filename]);records.append(r)
            ledger.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n');ledger.flush()
            members.append({'member':info.filename,'bytes':info.file_size,'compressed_bytes':info.compress_size,'CRC32':f'{info.CRC:08x}','sha256':r['source_file_sha256'],'full_ZIP_CRC_verified':True})
            target.unlink();print(json.dumps({'CNT_records_audited':len(records),'epoch':time.time()}),flush=True)
    require(custody.sha(archive)==whole and (initial.st_ino,initial.st_size,initial.st_mtime_ns)==(archive.stat().st_ino,archive.stat().st_size,archive.stat().st_mtime_ns),'ZIP changed during audit')
    unique={};duplicates=[]
    for r in records:
        k=(r['EEG_payload_sha256'],r['EEG_calibration_sha256'],tuple(r['EEG_channels']),r['actual_sample_count_from_byte_span'],r['sample_rate_hz'])
        if k in unique:duplicates.append({'member':r['source_member'],'same_as':unique[k]['source_member'],'must_link_groups':sorted({r['subject_group'],unique[k]['subject_group']})})
        else:unique[k]=r
    seconds=sum((Fraction(r['actual_sample_count_from_byte_span'],r['sample_rate_hz']) for r in unique.values()),Fraction())
    summary={'schema':'xavier_threeclass_CNT_quantity_summary_v1','status':'completed','completed_epoch':time.time(),
      'catalog_id':'001708','source_article':'https://figshare.com/articles/dataset/2061654','source_version':1,'license':'CC0',
      'source_archive_bytes':initial.st_size,'source_archive_sha256':whole,'all_members_full_CRC_SHA_verified':True,
      'CNT_records':len(records),'source_participant_groups':len({r['subject_group'] for r in records}),
      'unique_EEG_records':len(unique),'exact_EEG_duplicate_records':len(duplicates),'source_uncompressed_bytes':sum(x['bytes'] for x in members),
      'EEG_hours_deduplicated':float(seconds/3600),'EEG_seconds_exact_fraction':[seconds.numerator,seconds.denominator],
      'EEG_scalar_bytes_deduplicated':sum(r['EEG_sample_bytes'] for r in unique.values()),
      'EOG_scalar_bytes_all_records':sum(r['EOG_sample_bytes'] for r in records),
      'header_bytes_all_records':sum(r['data_offset'] for r in records),
      'event_table_bytes_all_records':sum(9+r['original_event_table_length'] for r in records),
      'tail_metadata_bytes_all_records':sum(r['tail_metadata_bytes'] for r in records),
      'channels':'14 scalp EEG + 2 EOG','native_rate_hz':250,'original_end_tag_value_counts':dict(collections.Counter(str(r['original_is32bit_tag']) for r in records)),
      'scalar_bytes_counts':dict(collections.Counter(str(r['scalar_bytes']) for r in records)),
      'original_sample_count_field_mismatch_records':sum(not r['source_sample_count_field_matches'] for r in records),
      'MNE_independent_calibration_probe_records_passed':sum(r['independent_MNE_header_and_three_calibrated_sample_probes_match'] for r in records),
      'original_event_entries':sum(r['raw_event_count'] for r in records),
      'original_stimulation_code_counts':dict(collections.Counter(str(e['original_fields']['StimType']) for r in records for e in r['original_event_entries'])),
      'derived_event_outside_record_count_without_clip':sum(not e['within_record'] for r in records for e in r['original_event_entries']),
      'original_event_offset_not_frame_aligned_count':sum(e['byte_offset_remainder_within_sample_frame']!=0 for r in records for e in r['original_event_entries']),
      'original_header_or_event_repairs':0,'event_semantic_class_ontology_verified':False,'supervised_training_eligible':False,
      'additional_gradient_training_hours':0,'signal_QC_or_new_split_completed':False,
      'records_sha256':custody.sha(out/'records.jsonl'),'format_proof_sha256':custody.sha(args.format_proof),'protocol_sha256':custody.sha(out/'protocol.json')}
    custody.immutable(out/'archive_members.json',members);custody.immutable(out/'duplicates.json',duplicates)
    custody.immutable(out/'summary.json',summary);custody.immutable(out/'completion.json',{'status':'completed','summary_sha256':custody.sha(out/'summary.json'),'archive_sha256':whole,'source_unchanged':True})
    return summary


def main():
    p=argparse.ArgumentParser()
    for name in ['archive','output','resource','format-proof','download-completion']:p.add_argument('--'+name,required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True),flush=True)


if __name__=='__main__':main()
