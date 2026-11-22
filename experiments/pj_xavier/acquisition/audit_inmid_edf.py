"""Read-only InMID rjx76wd5v6 v2 audit, separating14 EEG from58 EDF signals.

MAT exports/video/questionnaires are preserved and checksum verified but not
added as independent recording hours. No state-to-action or intent labels
are invented from application markers or filenames.
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
import warnings
import numpy as np
from scipy.io import whosmat
import acquire
import audit_p300_bci2000 as edf_reader
from delegated_acquisition_v2 import immutable, require, sha

EEG=['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
STATES=['MarkerIndex','MarkerType','MarkerValueInt','MARKER_HARDWARE','INTERPOLATED']
PREFIX=['TIME_STAMP_s','TIME_STAMP_ms','COUNTER','INTERPOLATED']
SOURCE='https://data.mendeley.com/datasets/rjx76wd5v6/2'


def source_group(name):
    m=re.fullmatch(r'\d{4}-\d{2}-\d{2}-(P\d{2})-([12])-([12])\.edf',Path(name).name)
    require(m is not None,'Unreviewed InMID filename')
    return 'mendeley:rjx76wd5v6:'+m.group(1),int(m.group(2)),int(m.group(3))


def header(path):
    h=edf_reader.edf_header(path);names=h['channel_labels']
    require(h['channels']==58 and len(set(names))==58 and names[:4]==PREFIX and names[4:18]==EEG,'Unreviewed Emotiv58signal layout')
    require(all(s in names for s in STATES),'Application state fields absent')
    require(h['declared_record_duration_s']==1 and h['declared_rate_per_channel_hz']==[128.]*58,'Unexpected native EDF rate')
    require(all(h['physical_units'][i]=='uV' for i in range(4,18)),'EEG physical unit missing or changed')
    h['EEG_indices']=list(range(4,18));h['auxiliary_indices']=[i for i in range(58) if i not in range(4,18)]
    h['EEG_channels']=EEG;h['sample_count']=h['samples_per_channel'][4];h['sample_rate_hz']=128;h['samples_per_record']=128
    h['physical_min']=list(map(float,h['physical_min']));h['physical_max']=list(map(float,h['physical_max']))
    h['digital_min']=list(map(float,h['digital_min']));h['digital_max']=list(map(float,h['digital_max']))
    require(all(h['physical_max'][i]>h['physical_min'][i] and h['digital_max'][i]>h['digital_min'][i] for i in h['EEG_indices']),'Invalid EEG calibration')
    return h


def physical_EEG_volts(digital,h):
    ix=h['EEG_indices'];pmin=np.asarray(h['physical_min'])[ix,None];pmax=np.asarray(h['physical_max'])[ix,None]
    dmin=np.asarray(h['digital_min'])[ix,None];dmax=np.asarray(h['digital_max'])[ix,None]
    return ((digital.astype(np.float64)-dmin)*(pmax-pmin)/(dmax-dmin)+pmin)*1e-6


def stream(path,h,expected_sha):
    path=Path(path);st=path.stat();whole=hashlib.sha256();eeg=hashlib.sha256();aux=hashlib.sha256();cal_hash=hashlib.sha256()
    minimum=np.full(14,np.inf);maximum=np.full(14,-np.inf);counts={k:collections.Counter() for k in STATES};transitions=collections.Counter();previous={k:None for k in STATES};events=[]
    with path.open('rb') as f:
        b=f.read(h['header_bytes']);require(len(b)==h['header_bytes'],'Header truncated');whole.update(b)
        for block in range(h['records']):
            raw=f.read(58*128*2);require(len(raw)==58*128*2,'EDF record truncated');whole.update(raw)
            x=np.frombuffer(raw,dtype='<i2').reshape(58,128);selected=np.ascontiguousarray(x[h['EEG_indices']]);other=np.ascontiguousarray(x[h['auxiliary_indices']])
            eeg.update(selected.tobytes());aux.update(other.tobytes());volts=physical_EEG_volts(selected,h)
            require(np.isfinite(volts).all(),'Nonfinite calibrated EEG');minimum=np.minimum(minimum,volts.min(axis=1));maximum=np.maximum(maximum,volts.max(axis=1))
            cal_hash.update(np.ascontiguousarray(volts.T,dtype='<f8').tobytes())
            for key in STATES:
                y=x[h['channel_labels'].index(key)];u,n=np.unique(y,return_counts=True);counts[key].update({int(v):int(c) for v,c in zip(u,n)})
                prior=np.r_[y[0] if previous[key] is None else previous[key],y[:-1]];changed=y!=prior;transitions[key]+=int(np.count_nonzero(changed))
                if key=='MarkerValueInt':
                    for index in np.flatnonzero(changed&(y!=0)):events.append({'stored_sample_index':block*128+int(index),'original_digital_marker_value':int(y[index])})
                previous[key]=int(y[-1])
        require(f.read(1)==b'','Unexpected EDF tail')
    end=path.stat();require((st.st_size,st.st_mtime_ns,st.st_ino)==(end.st_size,end.st_mtime_ns,end.st_ino),'Source changed during stream')
    require(whole.hexdigest()==expected_sha,'EDF full checksum mismatch after decoding')
    calibration={k:[h[k][i] for i in h['EEG_indices']] for k in ['physical_min','physical_max','digital_min','digital_max','physical_units']};calibration['channels']=EEG
    return {'source_sha256':whole.hexdigest(),'EEG_payload_sha256':eeg.hexdigest(),'auxiliary_payload_sha256':aux.hexdigest(),
        'calibration_sha256':hashlib.sha256(json.dumps(calibration,sort_keys=True).encode()).hexdigest(),'decoded_EEG_V_sha256':cal_hash.hexdigest(),
        'EEG_sample_bytes':h['sample_count']*14*2,'auxiliary_sample_bytes':h['sample_count']*44*2,
        'declared_physical_EEG_min_V_per_channel':[float(v) for v in minimum],'declared_physical_EEG_max_V_per_channel':[float(v) for v in maximum],
        'original_digital_state_value_counts':{k:{str(a):b for a,b in sorted(v.items())} for k,v in counts.items()},
        'original_digital_state_transition_counts':dict(transitions),'MarkerValueInt_nonzero_change_events':events,
        'marker_codes_mapped_to_action_or_intent':False,'waveform_modified':False}


def mne_probe(path,h):
    import mne
    with warnings.catch_warnings(record=True) as ws:raw=mne.io.read_raw_edf(path,preload=False,verbose='ERROR')
    require(raw.info['sfreq']==128 and raw.n_times==h['sample_count'] and raw.ch_names==h['channel_labels'],'MNE dimensions disagree')
    maxdiff=0.
    with Path(path).open('rb') as f:
        for start in [0,h['sample_count']//2,h['sample_count']-8]:
            block=start//128;offset=start%128;take=min(8,128-offset)
            f.seek(h['header_bytes']+block*58*128*2);x=np.frombuffer(f.read(58*128*2),dtype='<i2').reshape(58,128)
            expected=physical_EEG_volts(x[h['EEG_indices'],offset:offset+take],h);actual=raw.get_data(picks=EEG,start=start,stop=start+take)
            maxdiff=max(maxdiff,float(np.max(np.abs(actual-expected))));require(np.allclose(actual,expected,rtol=1e-12,atol=1e-15),'MNE EEG calibration differs')
    raw.close();return {'MNE_header_and_three_EEG_calibration_probes_passed':True,'MNE_max_probe_difference_V':maxdiff,'MNE_warning_categories':[w.category.__name__ for w in ws]}


def run(args):
    import mne
    from threadpoolctl import threadpool_limits
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);budget=json.loads(Path(args.resource).read_text());done=json.loads(Path(args.download_completion).read_text())
    require(done['status']=='all_files_downloaded_checksum_verified' and done['complete_files']==237,'Complete InMID release required')
    require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline expired');require(shutil.disk_usage(args.raw_root).free>=36*2**30+32*2**20,'Audit disk reserve unavailable')
    db=sqlite3.connect(Path(args.ledger).resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;jobs=[dict(x) for x in db.execute('select * from downloads order by relative_path')]
    require(len(jobs)==237 and all(j['status']=='complete' and j['catalog_id']=='001085' for j in jobs),'Unreviewed source ledger')
    immutable(out/'protocol.json',{'schema':'xavier_InMID_original_EDF_audit_v1','source_sha256':sha(__file__),'EDF_header_helper_sha256':sha(edf_reader.__file__),'resource_sha256':sha(args.resource),
        'source_dataset':SOURCE,'source_official_version':2,'official_metadata_sha256':budget['official_metadata_sha256'],'published_SHA256_all237_required':True,
        'channel_policy':'Exactly14 AF3..AF4 EEG names,58 total signals; timestamp/counter/quality/battery/markers are auxiliary even if their unit fields say uV',
        'quantity_policy':'Original92EDF recording samples/128Hz only; MAT derivative/time overlap not added; video and metadata excluded from EEG duration',
        'units_policy':'EEG-specific physical/digital header calibration compared to MNE; non-EEG generic uV fields not used for physical semantics',
        'MNE_version':mne.__version__,'CPU_threads':2,'new_training_or_label_creation':False,'absolute_deadline_epoch':budget['absolute_deadline_epoch']})
    records=[];files=[];mats=[];stats={};unique={};duplicates=[];typebytes=collections.Counter()
    with threadpool_limits(limits=2),(out/'records.jsonl').open('x') as ledger:
        for job in jobs:
            require(time.time()<budget['absolute_deadline_epoch'],'Audit deadline exceeded')
            path=acquire.destination(args.raw_root,job['dataset_key'],job['relative_path']);st=path.stat();digest=acquire.verify_file(path,job['expected_bytes'],json.loads(job['checksum_json']));require(digest==job['sha256'],'Complete SHA differs')
            suffix=path.suffix.lower();typebytes[suffix]+=st.st_size
            if suffix=='.edf':
                h=header(path);group,session,run=source_group(path.name);s=stream(path,h,digest);probe=mne_probe(path,h)
                row={'source_path':job['relative_path'],'source_bytes':st.st_size,'subject_group':group,'source_session':session,'source_recording_in_session':run,**h,**s,**probe,
                    'physical_unit_scope':'Header-declared microvolts decoded to V, not independent amplifier or physiology validation',
                    'non_EEG_units_are_generic_header_values_not_physical_roles':True,'new_training_hours_added':0}
                key=(row['EEG_payload_sha256'],row['calibration_sha256'],tuple(EEG),row['sample_count'],128)
                if key in unique:duplicates.append({'source_path':job['relative_path'],'same_as':unique[key]['source_path'],'must_link_groups':group!=unique[key]['subject_group']})
                else:unique[key]=row
                records.append(row);ledger.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');ledger.flush()
            elif suffix=='.mat':
                # MATLAB metadata inspection only; no execution or full variable loading.
                mats.append({'source_path':job['relative_path'],'source_sha256':digest,'variables':[{'name':n,'shape':list(shape),'type':typ} for n,shape,typ in whosmat(path)],'additional_original_recording_hours':0})
            after=path.stat();require((st.st_size,st.st_ino,st.st_mtime_ns)==(after.st_size,after.st_ino,after.st_mtime_ns),'Raw file changed during audit');stats[str(path)]=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
            files.append({'source_path':job['relative_path'],'bytes':st.st_size,'sha256':digest,'suffix':suffix,'job_id':job['id']})
    require(len(records)==92 and len(mats)==138,'Expected92EDF/138MAT source scope differs')
    require(len({r['subject_group'] for r in records})==23,'Expected23 source ID groups')
    group_counts=collections.Counter(r['subject_group'] for r in records);require(set(group_counts.values())=={4},'Not4 records per source group')
    seconds=sum((Fraction(r['sample_count'],128) for r in unique.values()),Fraction());eeg=sum(r['EEG_sample_bytes'] for r in records);aux=sum(r['auxiliary_sample_bytes'] for r in records);headers=sum(r['header_bytes'] for r in records)
    require(eeg+aux+headers==typebytes['.edf'],'EDF bytes do not partition');require(sum(typebytes.values())==1410402579,'Release bytes differ')
    for name,expected in stats.items():
        st=Path(name).stat();require((st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==expected,'Source changed before completion')
    with (out/'files.jsonl').open('x') as f:
        for row in files:f.write(json.dumps(row,sort_keys=True)+'\n')
    immutable(out/'MAT_variable_metadata.json',mats);immutable(out/'duplicates.json',duplicates)
    result={'schema':'xavier_InMID_original_EDF_quantity_summary_v1','status':'completed','completed_epoch':time.time(),'catalog_id':'001085','source_version':'Mendeley rjx76wd5v6 v2','license':'CC BY4.0','all_source_files':237,
        'distribution_bytes':sum(typebytes.values()),'bytes_by_file_suffix':dict(typebytes),'original_EDF_records':92,'source_ID_groups':23,'records_per_group':4,'unique_original_EEG_records':len(unique),'exact_EEG_duplicates':len(duplicates),
        'original_EEG_seconds_exact_fraction':[seconds.numerator,seconds.denominator],'original_EEG_hours_deduplicated':float(seconds/3600),'EEG_scalar_bytes_deduplicated':sum(r['EEG_sample_bytes'] for r in unique.values()),
        'EEG_scalar_bytes_all':eeg,'non_EEG_44channel_scalar_bytes_all':aux,'EDF_header_bytes_all':headers,'native_rate_hz_distribution':{'128':92},'EEG_channel_names':EEG,
        'EDF_total_channels':58,'EEG_channels':14,'non_EEG_auxiliary_channels':44,'MNE_independent_calibration_records':sum(r['MNE_header_and_three_EEG_calibration_probes_passed'] for r in records),
        'MAT_derived_or_overlap_unresolved_files':len(mats),'additional_original_hours_from_MAT':0,'source_MAT_conversion_script_present':True,
        'raw_MarkerValueInt_nonzero_change_count':sum(len(r['MarkerValueInt_nonzero_change_events']) for r in records),'marker_ontology_and_clock_alignment_validated':False,
        'non_EEG_generic_uV_field_does_not_make_all58_EEG':True,'new_units_QC_and_split_protocol_required':True,'supervised_gold_validated':False,
        'source_all237_SHA256_verified':True,'raw_stats_unchanged':True,'existing_inputs_or_models_changed':False,'new_training_hours_added':0,
        'records_sha256':sha(out/'records.jsonl'),'files_sha256':sha(out/'files.jsonl'),'MAT_variable_metadata_sha256':sha(out/'MAT_variable_metadata.json'),'protocol_sha256':sha(out/'protocol.json')}
    immutable(out/'summary.json',result);immutable(out/'completion.json',{'status':'completed','summary_sha256':sha(out/'summary.json'),'all_original_files_unchanged':True,'original_rows_reconciled':False})
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ['ledger','raw-root','download-completion','resource','output']:p.add_argument('--'+name,required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True),flush=True)


if __name__=='__main__':main()
