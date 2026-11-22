#!/usr/bin/env python3
"""Independently enforce the complete registered ITU comparison before publication."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

CLASSES = ['ai_prompt_submit', 'keyboard', 'mouse_click']
VARIANTS = ['fused', 'context_only', 'eeg_only', 'eeg_shuffled', 'eeg_time_shifted']
def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def require(value,reason):
    if not value: raise ValueError(reason)
def identity(r):return [r.get('canonical_dataset_id',r['dataset']),r.get('canonical_subject_id',r['subject']),r['session'],r['recording_id']]
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def verify_predictions(predictions, truth, artifact, variant, neural=True):
    require(len(predictions)==len(truth),'prediction count differs from frozen cohort')
    require({p['id'] for p in predictions}==set(truth),'missing or nonexternal prediction ID')
    require(len({p['id'] for p in predictions})==len(predictions),'duplicate external prediction ID')
    confusion=[[0]*3 for _ in range(3)];accepted=0
    for p in predictions:
        require(p.get('variant',p.get('condition'))==variant,'wrong prediction condition')
        require(p['true_label']==truth[p['id']]['label'],'external truth label changed')
        require(p.get('checkpoint_sha256',p.get('model_artifact_sha256'))==artifact,'unregistered prediction artifact')
        probability=p['probabilities']
        expected=set(CLASSES) if neural else {'ai_prompt_submit','other_observed_action'}
        require(set(probability)==expected,'probability class contract differs')
        require(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=1 for v in probability.values()) and abs(sum(probability.values())-1)<1e-5,'invalid probabilities')
        if neural:
            winner=max(CLASSES,key=lambda c:probability[c]);reject=max(probability.values())<.8
            require(p.get('abstain')==reject,'fixed 0.8 acceptance changed')
            confusion[CLASSES.index(p['true_label'])][CLASSES.index(winner)]+=1
            accepted+=not reject
    if not neural:return {'records':len(predictions),'artifact_sha256':artifact}
    recalls=[confusion[i][i]/sum(confusion[i]) for i in range(3)]
    return {'records':len(predictions),'artifact_sha256':artifact,'forced_confusion_matrix':confusion,'forced_balanced_accuracy':sum(recalls)/3,'coverage':accepted/len(predictions)}

def verify_controls(variant, predictions, truth, output, recovered_root, protocol_sha, manifest_sha, checkpoint):
    if variant=='eeg_shuffled':
        sources=[]
        for p in predictions:
            target=p['id'];source=p['perturbation']['eeg_source_id']
            require(source in truth and source!=target,'invalid external shuffle source')
            require(identity(truth[target])==identity(truth[source]),'external shuffle crossed a recording')
            require(p['eeg_source_id']==source and p['perturbation']['original_eeg_id']==target,'shuffle prediction/certificate disagreement')
            sources.append(source)
        require(len(set(sources))==len(truth) and set(sources)==set(truth),'shuffle is not a complete permutation')
        return {'status':'passed','records':len(sources),'fixed_points':0,'same_recording':True}
    if variant!='eeg_time_shifted':return None
    cert_rows=rows(output/'raw_shift_sources.jsonl');certs={r['record_id']:r['perturbation'] for r in cert_rows}
    require(len(certs)==len(cert_rows)==len(truth) and set(certs)==set(truth),'incomplete shift certificates')
    for p in predictions:
        row=truth[p['id']];cert=certs[p['id']];source=cert['source']
        require(p['perturbation']==cert and p['eeg_source_id']==cert['eeg_source_id'],'shift prediction differs from certificate')
        require(cert['method']=='external_raw_recording_time_shift' and cert['shift_seconds']==-10 and cert['original_eeg_id']==p['id'],'wrong external shift method')
        require(source['external_manifest_sha256']==manifest_sha and source['external_protocol_sha256']==protocol_sha and source['checkpoint_sha256']==checkpoint,'shift not sealed to this external contract')
        require(source['external_arrays_sealed_in_checkpoint'] is False,'external data inserted into checkpoint seal')
        require(source['recording_identity']==identity(row) and source['raw_eeg_sha256']==row['eeg_source_sha256'] and source['raw_vhdr_sha256']==row['eeg_header_sha256'],'shift source recording/header mismatch')
        first=round(row['window_start_s']*256)-2560;last=round(row['window_end_s']*256)-2560
        require(source['sample_rate_hz']==256 and source['start_sample']==first and source['stop_sample']==last and source['shift_samples']==-2560,'shift is not the real minus10-second sample interval')
        require(0<=first<last<=source['raw_n_samples'] and last-first==448 and source['start_s']==first/256 and source['end_s']==last/256,'shift bounds or time fields invalid')
        derived={k:source[k] for k in ('raw_eeg_sha256','sample_rate_hz','start_sample','stop_sample','channels')}
        require(cert['eeg_source_id']=='raw-window:'+canonical(derived),'derived shift ID forged')
        path=(recovered_root/source['extracted_array_path']).resolve()
        require(path.is_relative_to(recovered_root.resolve()),'shift archive path escapes recovery root')
        require(sha(path)==source['extracted_array_sha256'],'recovered raw-derived array changed')
    return {'status':'passed','records':len(predictions),'actual_raw_sample_rederivation_scope':'performed by sealed external runner on Colab; workstation verifies recovered evidence and array bytes','shift_seconds':-10}

def run(args):
    protocol_path=Path(args.protocol);require(sha(protocol_path)==args.expected_protocol_sha256,'protocol changed');protocol=read(protocol_path)
    require(protocol['conditions']==VARIANTS and len(protocol['models'])==3 and len(protocol['history_models'])==4,'registered 3x5 plus4 contract required')
    manifest=Path(args.manifest);manifest_sha=sha(manifest)
    require(manifest_sha==protocol['external_cohort']['cohort_files']['manifest.jsonl'],'external cohort changed')
    records=rows(manifest);truth={r['id']:r for r in records}
    require(len(records)==len(truth)==630 and len({r['subject'] for r in records})==25,'external 630/25 membership mismatch')
    require(all(r['split']=='external_test' and r['site']=='itu' for r in records),'external partition/site changed')
    inputs=read(args.inputs);require(set(inputs)==set(protocol['models'])|{'history'},'missing or extra registered models')
    score=read(args.score);require(score['status']=='completed' and score['protocol_sha256']==args.expected_protocol_sha256 and score['manifest_sha256']==manifest_sha and score['inputs_sha256']==sha(args.inputs),'score does not seal these exact inputs')
    require(score['new_model_inference_calls']==0 and score['new_fit_calls']==0 and score['thresholds']=={'ai_vs_other':.5,'action_acceptance':.8},'scorer performed an unregistered operation')
    recovered=Path(args.recovered_root);results={};source_reports={}
    for key,spec in protocol['models'].items():
        inp=inputs[key];require(inp['checkpoint_sha256']==spec['checkpoint_sha256'],'input checkpoint differs from registration')
        output=Path(inp['report_path']).parent;report=read(inp['report_path']);receipt=read(output/'variant_evaluation_completion.pending.json')
        require(sha(inp['report_path'])==inp['report_sha256']==receipt['report_sha256'],'report/receipt hash mismatch')
        require(report['status']=='completed' and report['evaluation_scope']=='external_only' and report['external_protocol_sha256']==args.expected_protocol_sha256 and report['checkpoint_sha256']==spec['checkpoint_sha256'] and report['model_module_sha256']==spec['source_sha256'],'report source/checkpoint/contract mismatch')
        require(receipt['status']=='completed' and receipt['exit_code']==0 and receipt['checkpoint_sha256']==spec['checkpoint_sha256'],'inference completion missing')
        require(report['test_record_count']==630 and report['variant_names']==VARIANTS and set(report['conditions'])==set(VARIANTS),'not all registered neural conditions evaluated')
        require(report['resource_feasibility']['status']=='passed' and report['memory']['cuda_peak_reserved_bytes']<=21*1024**3,'actual GPU memory gate failed')
        require(sha(inp['predictions_path'])==inp['predictions_sha256'],'combined predictions changed')
        combined=rows(inp['predictions_path']);require(len(combined)==3150,'five neural conditions must each contain630 rows')
        mapping=defaultdict(list)
        for p in combined:mapping[p['variant']].append(p)
        require(set(mapping)==set(VARIANTS),'wrong neural condition coverage')
        for variant in VARIANTS:
            metrics=report['conditions'][variant];source=output/(variant+'_predictions.jsonl')
            require(metrics['status']=='measured' and sha(source)==metrics['prediction_sha256'],'condition unavailable or source hash mismatch')
            original=rows(source);require(original==mapping[variant],'combined source predictions differ')
            result=verify_predictions(original,truth,spec['checkpoint_sha256'],variant)
            require(abs(result['forced_balanced_accuracy']-metrics['balanced_accuracy'])<1e-7 and abs(result['coverage']-metrics['coverage'])<1e-7,'reported metrics differ from direct arithmetic')
            result['negative_control']=verify_controls(variant,original,truth,output,recovered,args.expected_protocol_sha256,manifest_sha,spec['checkpoint_sha256'])
            results[key+'::'+variant]=result
        source_reports[key]=sha(inp['report_path'])
    history=Path(args.history_run);baseline=Path(args.baseline_run);hr=read(history/'external_history_report.json');hc=read(history/'completion.json');hp=inputs['history']
    require(hc['status']=='completed' and hc['exit_code']==0 and hc['report_sha256']==sha(history/'external_history_report.json'),'history completion changed')
    require(hc['protocol_sha256']==args.expected_protocol_sha256 and hc['source_sha256']==protocol['history_adapter_source_sha256'],'history source/protocol mismatch')
    require(hc['prediction_sha256']==sha(hp['predictions_path'])==hp['predictions_sha256'],'history predictions changed')
    require(hr['fit_calls']==hr['vocabulary_fit_calls']==hr['C_selection_calls']==0 and hr['threshold']==.5 and hr['external_used_for_tuning'] is False,'history refit/tuning forbidden')
    for name,expected in hr['frozen_baseline_files'].items():require(sha(baseline/name)==expected,'frozen history coefficient/vocabulary/selection artifact changed')
    histrows=rows(hp['predictions_path']);require(len(histrows)==2520,'four history conditions must each contain630 rows')
    grouped=defaultdict(list)
    for p in histrows:grouped[p['variant']].append(p)
    require(set(grouped)==set(protocol['history_models'])==set(hr['models']),'history condition registration mismatch')
    for name,values in grouped.items():
        artifact=hr['models'][name]['model_artifact_sha256'];require(sha(baseline/(name.replace(':','__')+'.npz'))==artifact,'history prediction coefficient mismatch')
        results['history::'+name]=verify_predictions(values,truth,artifact,name,neural=False)
    require(len(results)==19 and set(score['binary_ai_vs_other'])==set(results),'score must contain exactly19 registered conditions')
    require(all(v['n']==630 for v in score['binary_ai_vs_other'].values()),'score cohort size changed')
    require(set(score['multiclass_with_abstention_and_calibration'])=={k for k in results if not k.startswith('history::')},'three-class score coverage mismatch')
    require(score['site_cluster_uncertainty']['status']=='blocked' and score['human_identity_disjointness_verified'] is False and score['autonomous_agent_validated'] is False,'unproved site/human/agent gate silently promoted')
    result={'schema':'xavier-external-19-condition-independent-completion-audit-v1','status':'passed','protocol_sha256':args.expected_protocol_sha256,'manifest_sha256':manifest_sha,'score_sha256':sha(args.score),'inputs_sha256':sha(args.inputs),'conditions':19,'records_per_condition':630,'total_recorded_prediction_rows':11970,'observation_subjects':25,'neural_checkpoints':{k:v['checkpoint_sha256'] for k,v in protocol['models'].items()},'source_reports':source_reports,'history_report_sha256':hc['report_sha256'],'condition_identity_and_arithmetic_checks':results,'new_inference_calls':0,'new_training':False,'goal_validated':False,'scope':'Completion, frozen model membership, coefficient identity, prediction correspondence, direct arithmetic, negative-control evidence and recovered byte custody. This is not a performance acceptance claim.'}
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','conditions':19,'records_per_condition':630,'audit_sha256':sha(target)}))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','expected-protocol-sha256','manifest','inputs','score','recovered-root','history-run','baseline-run','output'):parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
