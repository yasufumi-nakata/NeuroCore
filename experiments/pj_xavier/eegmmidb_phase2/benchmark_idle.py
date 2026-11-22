"""Test-only cued-rest diagnostics, never a fitted idle/intention classifier."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

import benchmark_common as common


def verify_idle(prepared,seal,main_rows):
    prepared=Path(prepared)
    for name,key in [('idle_manifest.jsonl','idle_manifest_sha256'),('idle_artifacts.json','idle_artifacts_sha256'),('idle_metadata_protocol.json','idle_metadata_protocol_sha256')]:
        common.require(common.sha256(prepared/name)==seal[key],'idle-only artifact seal changed: '+name)
    rows=common.read_rows(prepared/'idle_manifest.jsonl'); ledger=common.read_json(prepared/'idle_artifacts.json')
    original=common.read_json(prepared/'subject_split.json')['assignments']
    main_ids={r['id'] for r in main_rows}; all_ids=set(); slots=defaultdict(set)
    held_sources={r['source_eeg_sha256'] for r in main_rows if r['split']!='test'}
    held_payload={r['eeg_payload_sha256'] for r in main_rows if r['split']!='test'}
    other_files=set(common.read_json(prepared/'artifacts.json'))|set(common.read_json(prepared/'control_artifacts.json'))
    common.require(not other_files&set(ledger),'rest diagnostic arrays must be separate')
    spans=defaultdict(list)
    for r in main_rows:spans[r['source_eeg_sha256']].append((r['window_start_s'],r['window_end_s']))
    for r in rows:
        common.require(r['id'] not in main_ids|all_ids,'idle ID repeated or reused as MI target');all_ids.add(r['id'])
        common.require(r['split']=='test' and original.get(r['subject_group'])=='test' and r['subject_group']==r['subject'],'idle participant outside frozen test')
        common.require(r['label']=='rest' and r['label_source']=='recorded_event' and r['gold_annotation_code']=='T0','idle is not recorded T0')
        common.require(r.get('diagnostic_only') is True and r.get('use_for_fit_or_selection') is False,'idle diagnostic use restriction missing')
        common.require(r['source_eeg_sha256'] not in held_sources and r['eeg_payload_sha256'] not in held_payload,'idle waveform overlaps training/selection participant source')
        common.require(r['task_family'] in common.TASKS and r['unit']=='V' and r['sample_rate_hz']==200,'idle geometry changed')
        common.require(r['channels']==main_rows[0]['channels'],'idle channel order differs')
        common.require(r.get('annotation_review_required') is False and r['preprocessing'].get('online_causality_verified') is True,'unresolved idle provenance')
        common.require(r['preprocessing']['protocol_sha256']==seal['preprocessing_protocol_sha256'],'idle preprocessing differs')
        common.validate_causal_grid(r,r)
        common.require(type(r['eeg_index']) is int and r['eeg_path'] in ledger and 0<=r['eeg_index']<ledger[r['eeg_path']]['shape'][0],'idle array index absent or fractional')
        common.require(r['eeg_index'] not in slots[r['eeg_path']],'duplicate idle array slot');slots[r['eeg_path']].add(r['eeg_index'])
        common.require(r['array_sha256']==ledger[r['eeg_path']]['sha256'],'idle row array hash differs')
        spans[r['source_eeg_sha256']].append((r['window_start_s'],r['window_end_s']))
    for values in spans.values():
        values.sort();common.require(all(a[1]<=b[0]+1e-8 for a,b in zip(values,values[1:])),'idle overlaps an MI or another idle target window')
    common.require(set(ledger)==set(slots),'idle array without provenance')
    for name,item in ledger.items():
        common.require(slots[name]==set(range(item['shape'][0])),'uncovered idle array slots')
        p=common.safe_path(prepared,name)
        common.require(p.stat().st_size==item['bytes'] and common.sha256(p)==item['sha256'],'idle waveform changed')
        a=np.load(p,mmap_mode='r',allow_pickle=False)
        common.require(a.dtype==np.float32 and list(a.shape)==item['shape'] and tuple(a.shape[1:])==(64,600),'idle array geometry differs')
    return rows


def evaluate_idle(runtime,prepared,seal,main_rows,output,batch_size,checkpoint,deadline_monotonic,verified_rows=None):
    rows=verify_idle(prepared,seal,main_rows) if verified_rows is None else verified_rows
    output=Path(output);store=common.EEGStore(prepared)
    predictions=[]
    with (output/'idle_predictions.jsonl').open('x') as stream:
        for i in range(0,len(rows),batch_size):
            common.require(time.monotonic()<deadline_monotonic,'idle evaluation budget expired')
            batch=rows[i:i+batch_size]
            values=runtime.predict_batch(np.stack([store.get(r) for r in batch]),[r['task_family'] for r in batch],'fused')
            common.require(len(values)==len(batch),'idle prediction count differs')
            for r,v in zip(batch,values,strict=True):
                labels=common.TASKS[r['task_family']];p=common.checked_probabilities(v['probabilities'],labels)
                common.require(v['predicted_label']==labels[int(p.argmax())],'idle argmax changed')
                value={'id':r['id'],'subject':r['subject'],'dataset':r['dataset'],'task_family':r['task_family'],
                       'true_label':'rest','predicted_binary_action':v['predicted_label'],'probabilities':dict(zip(labels,map(float,p))),
                       'accepted_as_action':bool(p.max()>=.8),'threshold':.8,'checkpoint_sha256':checkpoint,'diagnostic_only':True,
                       'latency_ms':v.get('latency_ms'),'natural_idle_ground_truth':False}
                predictions.append(value);stream.write(json.dumps(value,allow_nan=False)+'\n')
    return summarize_idle(predictions)


def summarize_idle(rows):
    result={}
    for task in common.TASKS:
        values=[r for r in rows if r['task_family']==task];groups=defaultdict(list)
        for r in values:groups[(r['dataset'],r['subject'])].append(r)
        rates=[sum(r['accepted_as_action'] for r in v)/len(v) for v in groups.values()]
        result[task]={'n':len(values),'subject_groups':len(groups),'status':'measured' if values else 'blocked_no_eligible_T0_windows',
                      'fraction_cued_rest_accepted_as_binary_action':sum(r['accepted_as_action'] for r in values)/len(values) if values else None,
                      'mean_subject_fraction_accepted':float(np.mean(rates)) if rates else None}
        if len(rates)>=2:
            rng=np.random.default_rng(20261122);x=np.array(rates);draw=np.mean(x[rng.integers(0,len(x),(1000,len(x)))],axis=1)
            result[task]['mean_subject_95_interval']=[float(v) for v in np.percentile(draw,[2.5,97.5])]
    return {'conditions':result,'threshold':.8,'model_fit_or_selection_on_rest':False,'natural_idle_ground_truth':False,
            'false_actions_per_minute':None,'scope':'One fixed window per public instructed T0 interval. Not a continuous natural-idle or spontaneous-intent safety estimate.'}
