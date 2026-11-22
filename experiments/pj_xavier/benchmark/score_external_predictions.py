#!/usr/bin/env python3
"""CPU-only inference-free summaries of separately sealed external predictions."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import history_baselines as history
from xavier_benchmark import classification_metrics

CLASSES = ['ai_prompt_submit', 'keyboard', 'mouse_click']
HISTORY_SHA = '5be7b7c4e91f6776895e2454adbec2ba12bfba1548e572e69b61ce7b9704ea13'


def read(path):
    return json.loads(Path(path).read_text())


def readrows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def multiclass_metrics(y, predicted, abstain):
    y, predicted, abstain = np.asarray(y), np.asarray(predicted), np.asarray(abstain, dtype=bool)
    full = np.bincount(y*3+predicted, minlength=9).reshape(3, 3)
    accepted = np.bincount(y[~abstain]*3+predicted[~abstain], minlength=9).reshape(3, 3)
    support = full.sum(axis=1)
    def scored(matrix):
        recall = np.divide(np.diag(matrix), support, out=np.zeros(3), where=support!=0)
        denominator = support+matrix.sum(axis=0)
        f1 = np.divide(2*np.diag(matrix), denominator, out=np.zeros(3), where=denominator!=0)
        return float(recall[support!=0].mean()), float(f1.mean())
    forced_ba, forced_f1 = scored(full)
    strict_ba, strict_f1 = scored(accepted)
    return {'forced_balanced_accuracy': forced_ba, 'forced_macro_f1': forced_f1,
            'strict_balanced_accuracy': strict_ba, 'strict_macro_f1': strict_f1,
            'coverage': float(np.mean(~abstain)),
            'selective_accuracy': float(np.mean(y[~abstain]==predicted[~abstain])) if (~abstain).any() else None}


def interval(values, repetitions):
    return {'lower_95': float(np.quantile(values, .025)) if values else None,
            'upper_95': float(np.quantile(values, .975)) if values else None,
            'valid_resamples': len(values), 'invalid_resamples': repetitions-len(values)}


def multiclass_uncertainty(truth, subjects, models, repetitions=1000):
    y = np.asarray(truth)
    collected = {name: defaultdict(list) for name in models}
    differences = {}
    for name in models:
        if name.endswith('::fused'):
            prefix = name[:-len('fused')]
            for other in ['context_only', 'eeg_only', 'eeg_shuffled', 'eeg_time_shifted']:
                if prefix+other in models:
                    differences[(name, prefix+other)] = defaultdict(list)
    for indices in history.cluster_indices(subjects, repetitions, 20261122):
        metrics = {name: multiclass_metrics(y[indices], predicted[indices], abstain[indices]) for name,(predicted,abstain) in models.items()}
        for name,result in metrics.items():
            for key,value in result.items():
                if value is not None: collected[name][key].append(value)
        for (first,second), values in differences.items():
            for key in metrics[first]:
                if metrics[first][key] is not None and metrics[second][key] is not None:
                    values[key].append(metrics[first][key]-metrics[second][key])
    return {'status':'measured', 'method':'paired participant-cluster percentile bootstrap conditional on fixed models and external enriched sample',
            'n_subjects':len(set(subjects)), 'repetitions':repetitions, 'seed':20261122,
            'models':{name:{key:interval(value,repetitions) for key,value in metrics.items()} for name,metrics in collected.items()},
            'paired_fused_minus_ablation':[{ 'first':first,'second':second,'intervals':{key:interval(value,repetitions) for key,value in metrics.items()}} for (first,second),metrics in differences.items()],
            'multiplicity_adjusted':False,'training_repeat_uncertainty_measured':False}


def collect_predictions(inputs, external):
    ids = [r['id'] for r in external]
    index = {r['id']:r for r in external}
    if len(ids)!=len(index): raise ValueError('duplicate external manifest IDs')
    collected, provenance = {}, {}
    for prefix,source in inputs.items():
        path = Path(source['predictions_path'])
        if history.sha(path)!=source['predictions_sha256']: raise ValueError('recorded external predictions changed')
        if source.get('report_path'):
            report=read(source['report_path'])
            if history.sha(source['report_path'])!=source['report_sha256'] or report.get('status')!='completed' or report.get('evaluation_scope')!='external_only': raise ValueError('external source report is not verified complete')
        per_condition=defaultdict(dict)
        checkpoint={}
        for p in readrows(path):
            name=p.get('variant',p.get('condition'))
            if not name or p['id'] not in index or p['id'] in per_condition[name]: raise ValueError('missing, duplicate or nonexternal prediction identity')
            if p.get('true_label')!=index[p['id']]['label']: raise ValueError('external prediction ground truth changed')
            probs=p['probabilities']
            if 'ai_prompt_submit' not in probs or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=1 for v in probs.values()) or abs(sum(probs.values())-1)>1e-5: raise ValueError('invalid external probabilities')
            artifact=p.get('checkpoint_sha256',p.get('model_artifact_sha256'))
            if not isinstance(artifact,str) or len(artifact)!=64: raise ValueError('model artifact SHA is required')
            if name in checkpoint and checkpoint[name]!=artifact: raise ValueError('condition mixes external model artifacts')
            if source.get('checkpoint_sha256') and source['checkpoint_sha256']!=artifact: raise ValueError('prediction differs from frozen external checkpoint')
            checkpoint[name]=artifact;per_condition[name][p['id']]=p
        if set(per_condition)!=set(source['expected_conditions']): raise ValueError('external condition coverage mismatch')
        for name,values in per_condition.items():
            if set(values)!=set(ids): raise ValueError('incomplete external condition; no subset score allowed')
            if set(next(iter(values.values()))['probabilities'])==set(CLASSES):
                for p in values.values():
                    if p.get('abstain')!=(max(p['probabilities'].values())<.8): raise ValueError('fixed external abstention threshold changed')
            full_name=prefix+'::'+name
            collected[full_name]=[values[rid] for rid in ids]
        provenance[prefix]={'prediction_sha256':source['predictions_sha256'],'report_sha256':source.get('report_sha256'),'checkpoint_sha256_per_condition':checkpoint}
    return collected,provenance


def run(args):
    protocol=read(args.protocol)
    if history.sha(args.protocol)!=args.expected_protocol_sha256 or history.sha(__file__)!=protocol['scorer_sha256'] or history.sha(history.__file__)!=HISTORY_SHA: raise ValueError('external score source/protocol changed')
    if history.sha(args.manifest)!=protocol['external_cohort']['cohort_files']['manifest.jsonl']: raise ValueError('external source manifest changed')
    external=readrows(args.manifest)
    if any(r['split']!='external_test' or r['site']!='itu' for r in external): raise ValueError('scoring requires the separately registered ITU external_test partition')
    output=Path(args.output)
    if output.exists(): raise ValueError('external score output exists; preserve the original report')
    inputs=read(args.inputs)
    predictions,provenance=collect_predictions(inputs,external)
    binary_truth=np.asarray([r['label']=='ai_prompt_submit' for r in external],dtype=int)
    three_truth=np.asarray([CLASSES.index(r['label']) for r in external],dtype=int)
    subjects=[r['site']+':'+r['subject'] for r in external]
    probabilities={name:[r['probabilities']['ai_prompt_submit'] for r in rows] for name,rows in predictions.items()}
    models,three,calibration={},{},{}
    for name,predicted in predictions.items():
        models[name]=history.binary_metrics(binary_truth,probabilities[name],threshold=.5)
        if set(predicted[0]['probabilities'])==set(CLASSES):
            yhat=np.asarray([int(np.argmax([r['probabilities'][c] for c in CLASSES])) for r in predicted])
            abstain=np.asarray([r['abstain'] for r in predicted],dtype=bool)
            three[name]=(yhat,abstain)
            calibration[name]=classification_metrics([(row['label'],p['probabilities'],p['abstain']) for row,p in zip(external,predicted)],CLASSES)
    result={'schema':'xavier-external-only-prediction-score-v1','status':'completed','evaluation_scope':'external_only',
            'manifest_sha256':history.sha(args.manifest),'protocol_sha256':args.expected_protocol_sha256,'inputs_sha256':history.sha(args.inputs),
            'source_provenance':provenance,'binary_ai_vs_other':models,
            'binary_participant_uncertainty':history.clustered_comparison(binary_truth,subjects,probabilities,1000,20261122),
            'multiclass_with_abstention_and_calibration':calibration,
            'multiclass_participant_uncertainty':multiclass_uncertainty(three_truth,subjects,three,1000),
            'site_cluster_uncertainty':{'status':'blocked','n_external_sites':1,'reason':'Only ITU is external; no uncertainty distribution over multiple unseen sites can be estimated.'},
            'new_model_inference_calls':0,'new_fit_calls':0,'thresholds':{'ai_vs_other':.5,'action_acceptance':.8},
            'sampling_design':'case_control_enriched','natural_population_precision_estimated':False,
            'false_actions_per_minute':None,'harness_completion':None,'online_causality_verified':False,
            'human_identity_disjointness_verified':False,'autonomous_agent_validated':False,
            'scope':'Fixed models on independently sealed ITU observations. Participant bootstrap is conditional on this enriched sample; no multiplicity adjustment, model retraining variability or unseen-site guarantee.'}
    output.parent.mkdir(parents=True,exist_ok=True);history.save(output,result)
    print(json.dumps({'status':'completed','conditions':len(models),'external_records':len(external),'observation_subjects':len(set(subjects)),'report_sha256':history.sha(output)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','expected-protocol-sha256','manifest','inputs','output'):parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
