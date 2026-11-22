"""Shared phase2 evidence checks and scoring. No global GPU/environment mutation."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import roc_auc_score

TASKS = {"left_right": ("left_fist", "right_fist"), "fists_feet": ("both_fists", "both_feet")}
CONDITIONS = ("fused", "context_only", "eeg_only", "eeg_shuffled", "eeg_time_shifted")
SHA = re.compile(r"[a-f0-9]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def safe_path(root, relative):
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in str(relative), 'artifact path must be contained and relative')
    out = Path(root)/p
    require(out.resolve().is_relative_to(Path(root).resolve()) and not out.is_symlink(), 'artifact escaped prepared root')
    return out


def validate_rows(rows):
    ids, slots = set(), set()
    group_splits, source_splits, payload_splits = defaultdict(set), defaultdict(set), defaultdict(set)
    intervals = defaultdict(list)
    counts = Counter()
    channel_order = rows[0]['channels'] if rows else None
    for row in rows:
        require(isinstance(row.get('id'), str) and row['id'] not in ids, 'missing or repeated record ID')
        ids.add(row['id'])
        require(row['split'] in ('train', 'val', 'test'), 'unsupported partition')
        task = row['task_family']
        require(task in TASKS and row['label'] in TASKS[task], 'invalid task-specific label')
        require(row.get('label_source') == 'recorded_event', 'pseudo or synthetic target is not EEG ground truth')
        require(float(row['sample_rate_hz']) == 200 and len(row['channels']) == 64 and len(set(row['channels'])) == 64 and row['channels']==channel_order and row.get('unit')=='V', 'input geometry or physical units changed')
        source = row.get('source_eeg_sha256', row.get('eeg_source_sha256'))
        require(SHA.fullmatch(str(source)) is not None, 'original source hash missing')
        require(all(row[k]==source for k in ('source_sha256','eeg_source_sha256') if k in row),'source aliases disagree')
        row['source_eeg_sha256'] = source
        payload=row.get('eeg_payload_sha256')
        require(SHA.fullmatch(str(payload)) is not None,'original EEG-only payload hash missing')
        payload_splits[payload].add(row['split'])
        group = (row['dataset'], row['subject'])
        group_splits[group].add(row['split'])
        source_splits[source].add(row['split'])
        require(type(row['eeg_index']) is int,'array index must be an integer, not float/bool')
        slot = (row['eeg_path'], row['eeg_index'])
        require(slot not in slots and slot[1] >= 0, 'duplicate or negative EEG array index')
        slots.add(slot)
        start, end = float(row['window_start_s']), float(row['window_end_s'])
        decision, available = float(row['decision_time_s']), float(row['feature_available_time_s'])
        require(all(math.isfinite(x) for x in (start, end, decision, available)), 'nonfinite time')
        require(start >= 0 and abs(end-start-3) < 1e-7 and available <= decision+1e-8 and end <= decision+1e-8, 'invalid causal decision/window times')
        prep = row.get('preprocessing', {})
        require(prep.get('online_causality_verified') is True, 'causal preprocessing proof required')
        require(row.get('annotation_review_required', False) is False, 'unresolved annotation review record')
        intervals[source].append((start, end, row['id']))
        counts[(row['split'], task, row['label'])] += 1
    require(all(len(v) == 1 for v in group_splits.values()), 'participant crosses partitions')
    require(all(len(v) == 1 for v in source_splits.values()), 'physical recording crosses partitions')
    require(all(len(v) == 1 for v in payload_splits.values()), 'identical EEG payload crosses partitions')
    for spans in intervals.values():
        spans.sort()
        require(all(a[1] <= b[0]+1e-8 for a,b in zip(spans, spans[1:])), 'overlapping primary EEG windows')
    require(all(any(r['split'] == s for r in rows) for s in ('train','val','test')), 'empty partition')
    return {'records': len(rows), 'participants_by_split': {s: sum(v == {s} for v in group_splits.values()) for s in ('train','val','test')},
            'label_counts': {'|'.join(k): v for k,v in sorted(counts.items())}, 'source_partition_disjoint': True}


def verify_prepared(root, seal):
    root = Path(root)
    for name, key in [('manifest.jsonl','prepared_manifest_sha256'),('artifacts.json','artifacts_sha256'),('control_artifacts.json','control_artifacts_sha256'),('subject_split.json','subject_split_sha256'),('metadata_protocol.json','metadata_protocol_sha256'),('preprocessing_protocol.json','preprocessing_protocol_sha256')]:
        require(sha256(root/name) == seal[key], 'sealed prepared source changed: '+name)
    rows = read_rows(root/'manifest.jsonl')
    audit = validate_rows(rows)
    proto=read_json(root/'preprocessing_protocol.json')
    require(proto['status']=='frozen_before_val_test_signal_processing' and proto['subject_split_sha256']==seal['subject_split_sha256'] and proto['metadata_protocol_sha256']==seal['metadata_protocol_sha256'],'preprocessing does not reference original metadata split')
    for name,key,reference in [('final_QC_decision.json','final_QC_decision_sha256','train_only_QC_decision_sha256'),('preprocessing_tests_v2.json','preprocessing_tests_sha256','tests_receipt_sha256')]:
        require(sha256(root/name)==seal[key]==proto[reference],'causal/QC proof hash changed')
    proof=read_json(root/'preprocessing_tests_v2.json');decision=read_json(root/'final_QC_decision.json')
    require(proof['status']=='passed' and proof['failures']==proof['errors']==0 and proof['prep_source_sha256']==proto['source_code_sha256'],'causal preprocessing fixtures not passed on actual source')
    require(set(proof['future_changes_prefix_invariance_native_rates'])=={128,160} and proof['independent_MNE_calibration_fixture'] is True,'causal/unit conversion fixture coverage incomplete')
    require(decision['val_waveform_reads_at_decision']==decision['test_waveform_reads_at_decision']==0 and decision['model_performance_used'] is False,'QC threshold used held-out signals or performance')
    require(proto['QC']['maximum_absolute_V']==.0005 and proto['QC']['minimum_channel_population_std_V']==1e-8 and proto['QC']['finite_required'] is True,'frozen exploratory QC changed')
    assignments=read_json(root/'subject_split.json')['assignments']
    require(assignments==canonical_subject_split(),'canonical original 109-participant split changed')
    for row in rows:
        require(row.get('subject_group')==row['subject'] and assignments.get(row['subject'])==row['split'],'manifest subject differs from frozen split')
        require(row['preprocessing'].get('protocol_sha256')==seal['preprocessing_protocol_sha256'] and row['preprocessing'].get('metadata_protocol_sha256')==seal['metadata_protocol_sha256'],'row preprocessing differs from sealed causal recipe')
        validate_causal_grid(row,row)
    artifacts = read_json(root/'artifacts.json')
    controls = read_json(root/'control_artifacts.json')
    require(not set(artifacts)&set(controls),'primary/control artifact files must be separate')
    for relative, item in {**artifacts, **controls}.items():
        file = safe_path(root, relative)
        require(file.stat().st_size == item['bytes'] and sha256(file) == item['sha256'], 'EEG artifact bytes changed')
        arr = np.load(file, mmap_mode='r', allow_pickle=False)
        require(str(arr.dtype) == item['dtype'] and list(arr.shape) == item['shape'], 'artifact header changed')
        require(arr.ndim == 3 and tuple(arr.shape[1:]) == (64,600) and arr.dtype == np.float32, 'expected float32 N64x600')
    file_splits = defaultdict(set); main_slots=defaultdict(set); control_slots=defaultdict(set)
    for row in rows:
        item = artifacts.get(row['eeg_path'])
        require(item is not None and row['eeg_index'] < item['shape'][0], 'array member/index absent')
        if 'array_sha256' in row:require(row['array_sha256']==item['sha256'],'main array row hash changed')
        main_slots[row['eeg_path']].add(row['eeg_index'])
        file_splits[row['eeg_path']].add(row['split'])
        ctrl = row.get('time_shift_control', {})
        if ctrl.get('status') == 'available':
            validate_shift(row, ctrl, controls)
            validate_causal_grid(ctrl,row,shift=True)
            require(ctrl['eeg_index'] not in control_slots[ctrl['eeg_path']],'control array slot repeated')
            control_slots[ctrl['eeg_path']].add(ctrl['eeg_index'])
            file_splits[ctrl['eeg_path']].add(row['split'])
        else:require(ctrl.get('status')=='blocked' and isinstance(ctrl.get('reason'),str) and bool(ctrl['reason']),'missing control must have an explicit reason')
    for ledger,slots in ((artifacts,main_slots),(controls,control_slots)):
        require(set(ledger)==set(slots),'extra or missing array without provenance rows')
        require(all(slots[k]==set(range(v['shape'][0])) for k,v in ledger.items()),'array index provenance incomplete')
    require(all(len(v) == 1 for v in file_splits.values()), 'array file spans partitions')
    audit['original_split_assignments_verified']=True
    return rows, {**artifacts,**controls}, audit


def canonical_subject_split():
    groups=[f'physionet:eegmmidb:1.0.0:S{i:03d}' for i in range(1,110)]
    groups.sort(key=lambda x:hashlib.sha256(('20261122|'+x).encode()).hexdigest())
    return {g:'train' if i<76 else 'val' if i<92 else 'test' for i,g in enumerate(groups)}


def validate_causal_grid(ref,source,shift=False):
    fs=source['native_sample_rate_hz']
    require(fs in (128,160),'unexpected original sample rate')
    start,stop=ref['sample_start'],ref['sample_stop']
    require(type(start) is int and type(stop) is int and stop-start==600 and start>=1000,'causal stream warmup or sample indices changed')
    require(abs(start/200-ref['window_start_s'])<1e-8 and abs(stop/200-ref['window_end_s'])<1e-8,'target grid disagreement')
    require(abs(ref['decision_time_s']-stop/200)<1e-8 and abs(ref['feature_available_time_s']-(stop-1)/200)<1e-8,'availability is not the latest actually used target sample')
    history=ref['causal_history']; last=((stop-1)*int(fs))//200
    require(history['source_recording_id']==source['recording_id'] and history['native_sample_start']==0 and history['native_sample_stop']==last+1,'causal native history differs')
    require(abs(history['last_sample_time_s']-last/fs)<1e-8 and history['last_sample_time_s']<=ref['feature_available_time_s']+1e-8,'causal history uses unavailable future native samples')
    if not shift:
        require(source['preprocessing'].get('future_samples_used') is False,'future-sample proof missing')
        require(-1e-8<=ref['window_start_s']-(source['event_onset_s']+1)<.005+1e-8 and ref['window_end_s']<=source['event_offset_s']+1e-8,'cue-relative interval differs from frozen candidate')


def validate_shift(row, ctrl, artifacts):
    require(ctrl.get('status') == 'available' and float(ctrl['shift_seconds']) == -10, 'raw shift contract changed')
    require(ctrl['source_eeg_sha256'] == row['source_eeg_sha256'], 'raw shift source differs')
    require(ctrl['source_recording_id'] == row['recording_id'], 'raw shift recording differs')
    if 'eeg_payload_sha256' in ctrl:require(ctrl['eeg_payload_sha256']==row['eeg_payload_sha256'],'raw shift EEG payload differs')
    require(all(type(ctrl[k]) is int for k in ('eeg_index','sample_start','sample_stop')),'control indices must be integers')
    require(abs(ctrl['window_start_s'] - (row['window_start_s']-10)) < 1e-7 and abs(ctrl['window_end_s']-(row['window_end_s']-10)) < 1e-7, 'raw shift times differ from actual minus10s')
    require(ctrl['window_start_s'] >= 0 and ctrl['sample_stop']-ctrl['sample_start'] == 600, 'raw shift outside stream or wrong width')
    require(abs(ctrl['sample_start']/200-ctrl['window_start_s']) < 1e-7 and abs(ctrl['sample_stop']/200-ctrl['window_end_s']) < 1e-7, 'raw shift grid disagrees with interval')
    require(ctrl['eeg_path'] in artifacts and 0 <= ctrl['eeg_index'] < artifacts[ctrl['eeg_path']]['shape'][0], 'raw shift array absent')
    if 'array_sha256' in ctrl:
        require(ctrl['array_sha256'] == artifacts[ctrl['eeg_path']]['sha256'], 'raw shift array hash changed')


class EEGStore:
    def __init__(self, root):
        self.root, self.arrays = Path(root), {}

    def get(self, row, shift=False):
        ref = row['time_shift_control'] if shift else row
        relative = ref['eeg_path']
        if relative not in self.arrays:
            self.arrays[relative] = np.load(safe_path(self.root, relative), mmap_mode='r', allow_pickle=False)
        a = np.array(self.arrays[relative][ref['eeg_index']], dtype=np.float32, copy=True)
        require(a.shape == (64,600) and np.isfinite(a).all(), 'invalid input array')
        return a


def shuffle_mapping(rows, seed):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['dataset'],r['subject'],r['session'],r['recording_id'],r['task_family'])].append(r)
    mapping, blocked = {}, []
    for group, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda r: hashlib.sha256((str(seed)+'|'+r['id']).encode()).hexdigest())
        if len(ordered) < 2:
            blocked.extend(r['id'] for r in ordered)
            continue
        shift = 1+int(hashlib.sha256(repr(group).encode()).hexdigest()[:12],16) % (len(ordered)-1)
        for i, row in enumerate(ordered):
            mapping[row['id']] = ordered[(i+shift)%len(ordered)]
    return mapping, blocked


def checked_probabilities(values, labels):
    require(set(values) == set(labels), 'class probability keys differ')
    p = np.array([values[x] for x in labels], dtype=np.float64)
    require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all() and abs(p.sum()-1) <= 1e-5, 'invalid probabilities')
    return p/p.sum()


def metrics(y, p, threshold=.8):
    y, p = np.asarray(y, int), np.asarray(p, float)
    require(len(y) > 0 and p.shape == (len(y),2), 'empty or malformed predictions')
    require(set(y.tolist()) <= {0,1} and np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=1e-5,rtol=0),'metric inputs invalid')
    pred = p.argmax(1); conf = p.max(1); accepted = conf >= threshold
    cm = np.bincount(y*2+pred, minlength=4).reshape(2,2)
    support = cm.sum(1); recalls = np.divide(cm.diagonal(),support,out=np.zeros(2),where=support>0)
    denominator = cm.sum(0)+support
    f1 = np.divide(2*cm.diagonal(),denominator,out=np.zeros(2),where=denominator>0)
    strict = np.bincount(y[(pred==y)&accepted],minlength=2)
    ece = 0.0
    for i in range(10):
        take = (conf >= i/10)&((conf < (i+1)/10) if i<9 else (conf <= 1))
        if take.any(): ece += float(take.mean()*abs((pred[take]==y[take]).mean()-conf[take].mean()))
    return {'n':len(y),'confusion':cm.tolist(),'support':support.tolist(),'accuracy':float((pred==y).mean()),
            'balanced_accuracy':float(recalls.mean()) if (support>0).all() else None,'macro_f1':float(f1.mean()),
            'auroc':float(roc_auc_score(y,p[:,1])) if len(set(y.tolist()))==2 else None,
            'negative_log_likelihood':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),
            'brier':float(((p[:,1]-y)**2).mean()),'ece':ece,'acceptance_threshold':threshold,
            'coverage':float(accepted.mean()),'accepted':int(accepted.sum()),
            'selective_accuracy':float((pred[accepted]==y[accepted]).mean()) if accepted.any() else None,
            'strict_balanced_accuracy':float(np.mean(strict/support)) if (support>0).all() else None}


def score_rows(rows, threshold=.8):
    groups = defaultdict(list)
    seen = set()
    for r in rows:
        key = (r['condition'],r['task_family'],r['id'])
        require(key not in seen,'duplicate prediction identity');seen.add(key)
        groups[key[:2]].append(r)
    result = {}
    for (condition,task), records in sorted(groups.items()):
        labels = TASKS[task]
        y = [labels.index(r['true_label']) for r in records]
        p = np.stack([checked_probabilities(r['probabilities'],labels) for r in records])
        value = metrics(y,p,threshold)
        subject = defaultdict(list)
        for i,r in enumerate(records): subject[(r['dataset'],r['subject'])].append(i)
        per_subject = {}
        for key,index in subject.items():per_subject['|'.join(key)]=metrics(np.array(y)[index],p[index],threshold)
        bas = [v['balanced_accuracy'] for v in per_subject.values() if v['balanced_accuracy'] is not None]
        value.update(mean_subject_balanced_accuracy=float(np.mean(bas)) if bas else None,
                     evaluated_subject_groups=len(subject),subjects_missing_either_class=len(subject)-len(bas),per_subject=per_subject)
        latencies = [r.get('latency_ms') for r in records if isinstance(r.get('latency_ms'),(int,float))]
        value['latency_ms_p50'] = float(np.percentile(latencies,50)) if latencies else None
        value['latency_ms_p95'] = float(np.percentile(latencies,95)) if latencies else None
        result[condition+'::'+task] = value
    return result


def participant_bootstrap(rows, repetitions=1000, seed=20261122, paired_ids=None):
    """Paired bootstrap uses the same participant draws and exact common IDs."""
    result = {}
    for task, labels in TASKS.items():
        by_condition = defaultdict(dict)
        for r in rows:
            if r['task_family']==task: by_condition[r['condition']][r['id']]=r
        if 'fused' not in by_condition: continue
        task_result=[]
        for other, alternative in sorted(by_condition.items()):
            if other=='fused':continue
            base=by_condition['fused']
            if paired_ids is not None:
                expected={i for i in paired_ids if i in base}
                require(expected <= set(alternative),'paired metadata subset has an unavailable scored control')
                ids=sorted(expected)
            else:
                ids=sorted(set(base)&set(alternative))
            if not ids:
                task_result.append({'first':'fused','second':other,'status':'blocked_no_common_ids'});continue
            groups=defaultdict(list)
            for i,key in enumerate(ids):
                a,b=base[key],alternative[key]
                require(a['true_label']==b['true_label'] and a['subject']==b['subject'],'paired target or subject changed')
                groups[(a['dataset'],a['subject'])].append(i)
            if len(groups)<2:
                task_result.append({'first':'fused','second':other,'status':'blocked_insufficient_subject_groups'});continue
            y=np.array([labels.index(base[k]['true_label']) for k in ids])
            p=np.stack([checked_probabilities(base[k]['probabilities'],labels) for k in ids])
            q=np.stack([checked_probabilities(alternative[k]['probabilities'],labels) for k in ids])
            indices=list(groups.values());rng=np.random.default_rng(seed)
            draws=defaultdict(list)
            subject_scores=[]
            for index in indices:
                ma,mb=metrics(y[index],p[index]),metrics(y[index],q[index])
                subject_scores.append(None if ma['balanced_accuracy'] is None or mb['balanced_accuracy'] is None else ma['balanced_accuracy']-mb['balanced_accuracy'])
            for _ in range(repetitions):
                selected=rng.integers(0,len(indices),len(indices)); loc=np.concatenate([indices[j] for j in selected])
                a,b=metrics(y[loc],p[loc]),metrics(y[loc],q[loc])
                for name in ['balanced_accuracy','macro_f1','auroc','coverage']:
                    if a[name] is not None and b[name] is not None:draws[name].append(a[name]-b[name])
                values=[subject_scores[j] for j in selected if subject_scores[j] is not None]
                if values:draws['mean_subject_balanced_accuracy'].append(float(np.mean(values)))
            task_result.append({'first':'fused','second':other,'status':'measured','n_common_records':len(ids),'n_subject_groups':len(groups),
                'repetitions':repetitions,'intervals':{k:{'lower_95':float(np.percentile(v,2.5)),'upper_95':float(np.percentile(v,97.5)),'valid_resamples':len(v)} for k,v in draws.items()}})
        result[task]=task_result
    return {'method':'paired participant-cluster percentile bootstrap on common IDs','comparisons':result,'multiplicity_adjusted':False,'retraining_uncertainty_measured':False}


def per_condition_subject_intervals(rows,repetitions=1000,seed=20261122):
    result={};conditions=defaultdict(list)
    for r in rows:conditions[(r['condition'],r['task_family'])].append(r)
    for (condition,task),values in sorted(conditions.items()):
        groups=defaultdict(list);labels=TASKS[task]
        for i,r in enumerate(values):groups[(r['dataset'],r['subject'])].append(i)
        key=condition+'::'+task
        if len(groups)<2:result[key]={'status':'blocked_insufficient_subject_groups'};continue
        y=np.array([labels.index(r['true_label']) for r in values]);p=np.stack([checked_probabilities(r['probabilities'],labels) for r in values])
        locs=list(groups.values());subject_scores=[metrics(y[ix],p[ix])['balanced_accuracy'] for ix in locs]
        rng=np.random.default_rng(seed);draws=defaultdict(list)
        for _ in range(repetitions):
            take=rng.integers(0,len(locs),len(locs));ix=np.concatenate([locs[j] for j in take]);m=metrics(y[ix],p[ix])
            for name in ('balanced_accuracy','macro_f1','auroc','coverage'):
                if m[name] is not None:draws[name].append(m[name])
            scores=[subject_scores[j] for j in take if subject_scores[j] is not None]
            if scores:draws['mean_subject_balanced_accuracy'].append(float(np.mean(scores)))
        result[key]={'status':'measured','subject_groups':len(groups),'repetitions':repetitions,
                     'intervals':{k:{'lower_95':float(np.percentile(v,2.5)),'upper_95':float(np.percentile(v,97.5)),'valid_resamples':len(v)} for k,v in draws.items()}}
    return {'method':'participant-cluster percentile bootstrap; trained checkpoint fixed','conditions':result,'multiplicity_adjusted':False}
