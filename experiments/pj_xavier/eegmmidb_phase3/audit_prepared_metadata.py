"""Independent metadata-only accounting for sealed continuous SSL preparation.

Checks source/split exclusion, every array slot, causal times, and exact raw
interval unions. Does not claim remote waveform rehash or training exposure.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,time
from fractions import Fraction
from pathlib import Path


def require(ok,reason):
    if not ok:raise ValueError(reason)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def lines(path):
    with Path(path).open() as f:
        for line in f:yield json.loads(line)


def union_length(intervals):
    ordered=sorted(intervals)
    if not ordered:return 0
    total=0;start,end=ordered[0]
    for a,b in ordered[1:]:
        require(b>a,'Nonpositive interval')
        if a>end:total+=end-start;start,end=a,b
        else:end=max(end,b)
    return total+end-start


def require_integer(x):
    require(isinstance(x,int) and not isinstance(x,bool),'Integer index required')


def validate_row(r,allow,denied_sha,denied_payload,assignments,artifacts):
    require(r['split'] in ['train','val'],'Test or unknown split in SSL manifest')
    require(not ({'label','task_family','event','events','event_onset_s','annotation','run','trial'} & set(r)),'Annotation/action metadata in SSL manifest')
    source=r['source_eeg_sha256'];require(source==r['source_sha256'],'Source alias differs')
    require(source in allow and source not in denied_sha and r['eeg_payload_sha256'] not in denied_payload,'Denied/unlisted raw source')
    a=allow[source];require(r['split']==a['split']==assignments[r['subject_group']],'Source/subject split mismatch')
    for k in ['subject_group','source_recording_id','eeg_payload_sha256','channels','native_sample_rate_hz']:
        require(r[k]==a[k],'Source metadata mismatch: '+k)
    require(r['unit']=='V' and r['sample_rate_hz']==200 and len(r['channels'])==64,'Wrong physical input schema')
    require(r['preprocessing']['online_causality_verified'] is True and r['preprocessing']['future_samples_used'] is False,'Causality not sealed')
    start,end=r['window_start_s'],r['window_end_s'];require(start>=5 and abs(end-start-3)<1e-9,'Window duration/warmup mismatch')
    require(abs((start-5)/3-round((start-5)/3))<1e-8,'Window not on fixed3s recording grid')
    for k in ['sample_start','sample_stop','eeg_index']:require_integer(r[k])
    require(r['sample_start']==round(start*200) and r['sample_stop']==r['sample_start']+600,'Target sample bounds mismatch')
    require(r['decision_time_s']==end and r['feature_available_time_s']<=end,'Future feature timestamp')
    fs=r['native_sample_rate_hz'];raw=r['raw_interval'];history=r['causal_history']
    for k in ['native_sample_start','native_sample_stop']:require_integer(raw[k])
    require(raw['native_sample_start']==math.ceil(start*fs-1e-8) and raw['native_sample_stop']==math.ceil(end*fs-1e-8),'Native source interval mismatch')
    require(raw['native_sample_stop']<=round(a['source_seconds']*fs),'Raw interval beyond source')
    require(raw['start_s']==start and raw['end_s']==end and raw['source_recording_id']==r['source_recording_id'],'Raw timing/identity differs')
    require(history['native_sample_start']==0 and history['native_sample_stop']==raw['native_sample_stop'],'Causal history not same source prefix')
    require(history['source_recording_id']==r['source_recording_id'] and history['last_sample_time_s']<end,'Causal source/future mismatch')
    q=r['QC'];require(q['passed'] is True and q['finite'] is True and q['maximum_absolute_V']<=.0005 and q['minimum_channel_population_std_V']>=1e-8,'Fixed QC differs')
    p=r['eeg_path'];require(p in artifacts and not Path(p).is_absolute() and '..' not in Path(p).parts,'Unknown/unsafe artifact path')
    f=artifacts[p];require(f['dtype']=='float32' and f['shape'][1:]==[64,600] and r['array_sha256']==f['sha256'],'Artifact shape/dtype/SHA mismatch')
    require(0<=r['eeg_index']<f['shape'][0],'Array index outside shape')
    return source,p,r['eeg_index'],(raw['native_sample_start'],raw['native_sample_stop']),fs


def run(args):
    p=Path(args.prepared);summary=json.loads((p/'corpus_summary.json').read_text());completion=json.loads((p/'completion.json').read_text())
    require(completion['status']=='completed' and completion['corpus_summary_sha256']==sha(p/'corpus_summary.json'),'Prep completion seal differs')
    required={'manifest.jsonl':'manifest_sha256','artifacts.json':'artifacts_sha256','source_allowlist.jsonl':'source_allowlist_sha256','source_denylist.jsonl':'source_denylist_sha256','metadata_protocol.json':'protocol_sha256','runtime_seal.json':'runtime_seal_sha256'}
    for f,k in required.items():require(sha(p/f)==summary[k],'Prepared input SHA differs: '+f)
    allowed=list(lines(p/'source_allowlist.jsonl'));denied=list(lines(p/'source_denylist.jsonl'));allow={x['source_eeg_sha256']:x for x in allowed}
    require(len(allow)==len(allowed)==1238 and len(denied)==288,'Source coverage count differs')
    dsha={x['source_eeg_sha256'] for x in denied};dpayload={x['eeg_payload_sha256'] for x in denied}
    require(not set(allow)&dsha and not {x['eeg_payload_sha256'] for x in allowed}&dpayload,'Allowed source aliases excluded payload')
    assignments=json.loads((p/'subject_split.json').read_text())['assignments'];art=json.loads((p/'artifacts.json').read_text())
    seen_id=set();slots=collections.defaultdict(set);pathsources={};source_split={};payload_split={};groups=collections.defaultdict(set);counts=collections.Counter();intervals=collections.defaultdict(list);rates={};target=collections.defaultdict(list)
    for r in lines(p/'manifest.jsonl'):
        source,path,index,native,fs=validate_row(r,allow,dsha,dpayload,assignments,art)
        require(r['id'] not in seen_id,'Repeated window ID');seen_id.add(r['id'])
        require(index not in slots[path],'Repeated array slot');slots[path].add(index)
        pathsources.setdefault(path,source);require(pathsources[path]==source,'Array mixes source recordings')
        source_split.setdefault(source,r['split']);require(source_split[source]==r['split'],'Raw source crosses split')
        payload_split.setdefault(r['eeg_payload_sha256'],r['split']);require(payload_split[r['eeg_payload_sha256']]==r['split'],'EEG payload crosses split')
        groups[r['split']].add(r['subject_group']);counts[r['split']]+=1;key=(r['split'],source)
        intervals[key].append(native);rates[key]=fs;target[key].append((Fraction(str(r['window_start_s'])),Fraction(str(r['window_end_s']))))
    require(set(slots)==set(art),'Unreferenced/missing artifact')
    for path,a in art.items():
        require(slots[path]==set(range(a['shape'][0])),'Array slots are not unique complete coverage')
        payload=a['shape'][0]*64*600*4;require(payload<=a['bytes']<=payload+4096,'Array metadata bytes not compatible with payload')
    require(not groups['train']&groups['val'],'Participant crosses split')
    for key,values in intervals.items():require(union_length(values)==sum(b-a for a,b in values),'SSL raw intervals overlap')
    require(dict(counts)==summary['window_counts'] and sum(counts.values())==summary['accepted_windows'],'Summary window counts differ')
    records=list(lines(p/'source_records.jsonl'));require(len(records)==1238 and {r['source_sha256'] for r in records}==set(allow),'Decoded source accounting differs')
    require(sum(r['accepted'] for r in records)==sum(counts.values()),'Per-source acceptance count differs')
    rejected=list(lines(p/'rejected_windows.jsonl'));require(len(rejected)==summary['rejected_windows'],'Rejected accounting differs')
    require(sum(r['candidates'] for r in records)==len(seen_id)+len(rejected),'Candidate/accept/reject not exhaustive')
    native_union={s:sum(Fraction(union_length(v),1)/Fraction(str(rates[k])) for k,v in intervals.items() if k[0]==s) for s in ['train','val']}
    target_union={s:sum((union_length(v) for k,v in target.items() if k[0]==s),Fraction(0)) for s in ['train','val']}
    combined={k:list(v) for k,v in intervals.items()};combined_target={k:list(v) for k,v in target.items()};p2counts=collections.Counter();p2seconds=collections.Counter()
    for r in lines(args.phase2_manifest):
        if r['split'] not in ['train','val']:continue
        source=r['source_eeg_sha256'];require(source in allow and allow[source]['split']==r['split'],'Phase2 train/val source not in SSL source allowlist')
        key=(r['split'],source);raw=r['raw_interval'];combined.setdefault(key,[]).append((raw['native_sample_start'],raw['native_sample_stop']));rates[key]=allow[source]['native_sample_rate_hz']
        combined_target.setdefault(key,[]).append((Fraction(str(r['window_start_s'])),Fraction(str(r['window_end_s']))))
        p2counts[r['split']]+=1;p2seconds[r['split']]+=Fraction(raw['native_sample_stop']-raw['native_sample_start'],1)/Fraction(str(rates[key]))
    joint={s:sum(Fraction(union_length(v),1)/Fraction(str(rates[k])) for k,v in combined.items() if k[0]==s) for s in ['train','val']}
    joint_target={s:sum((union_length(v) for k,v in combined_target.items() if k[0]==s),Fraction(0)) for s in ['train','val']}
    report={'schema':'pj-xavier-phase3-independent-metadata-audit-v1','status':'passed','completed_epoch':time.time(),
      'audit_source_sha256':sha(__file__),'prepared_manifest_sha256':sha(p/'manifest.jsonl'),'phase2_manifest_sha256':sha(args.phase2_manifest),
      'source_records_allowed':len(allow),'source_records_denied':len(denied),'candidate_windows':sum(r['candidates'] for r in records),
      'accepted_windows':dict(counts),'participant_groups':{s:len(v) for s,v in groups.items()},'rejected_windows':len(rejected),
      'artifact_count':len(art),'declared_array_bytes':sum(x['bytes'] for x in art.values()),
      'source_group_payload_split_exclusion_passed':True,'each_array_slot_unique_complete_passed':True,'raw_source_intervals_nonoverlap_passed':True,
      'native_source_target_interval_union_hours':{s:float(v/3600) for s,v in native_union.items()},
      'target_clock_interval_union_hours':{s:float(v/3600) for s,v in target_union.items()},
      'combined_prepared_SSL_and_MI':{s:{'SSL_native_seconds':float(native_union[s]),'MI_native_seconds':float(p2seconds[s]),
          'native_interval_intersection_seconds':float(native_union[s]+p2seconds[s]-joint[s]),
          'native_interval_union_seconds':float(joint[s]),'native_interval_union_hours':float(joint[s]/3600),
          'target_clock_interval_union_seconds':float(joint_target[s]),'MI_windows':p2counts[s]} for s in ['train','val']},
      'scope':'Prepared windows, not actual seen-ID training exposure or selected checkpoint exposure. Source-native integer index union excludes filter history and never includes test.',
      'remote_waveforms_rehashed_by_this_metadata_audit':False,'remote_waveform_validation_owner':'model loader independent source/array verification',
      'actual_training_seen_ID_accounting_pending':True,'model_forward_calls':0}
    with Path(args.output).open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--phase2-manifest',required=True);p.add_argument('--output',required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True))
if __name__=='__main__':main()
