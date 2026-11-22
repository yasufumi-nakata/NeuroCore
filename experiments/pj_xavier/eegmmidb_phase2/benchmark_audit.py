"""Independent identity/control audit on saved phase2 predictions, no inference."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import benchmark_common as common


def audit_predictions(rows,predictions,checkpoint,seed=20261122):
    test={r['id']:r for r in rows if r['split']=='test'}
    mapping,blocked=common.shuffle_mapping(list(test.values()),seed)
    expected={c:set(test) for c in common.CONDITIONS}
    expected['eeg_shuffled']=set(mapping)
    expected['eeg_time_shifted']={i for i,r in test.items() if r['time_shift_control']['status']=='available'}
    seen=defaultdict(set)
    for p in predictions:
        condition=p['condition'];identity=p['id']
        common.require(condition in expected and identity in expected[condition],'unexpected evaluation condition/record')
        common.require(identity not in seen[condition],'duplicate evaluated condition/record');seen[condition].add(identity)
        source=test[identity]
        for field in ['dataset','subject','session','recording_id','task_family']:common.require(p[field]==source[field],'prediction provenance differs: '+field)
        common.require(p['true_label']==source['label'] and p['checkpoint_sha256']==checkpoint,'target/checkpoint changed')
        labels=common.TASKS[source['task_family']];q=common.checked_probabilities(p['probabilities'],labels)
        common.require(p['predicted_label']==labels[int(q.argmax())],'classification differs from probabilities')
        perturbation=p['perturbation']
        if condition=='eeg_shuffled':
            other=mapping[identity]
            common.require(perturbation=={'method':'within_recording_derangement','eeg_source_id':other['id'],'original_eeg_id':identity,'fixed_points':0},'shuffle substitution evidence differs')
        elif condition=='eeg_time_shifted':common.require(perturbation==dict(source['time_shift_control'],method='actual_causal_stream_minus10seconds'),'raw-shift interval/source evidence differs')
        elif condition=='context_only':common.require(perturbation=={'method':'EEG_removed'},'context-only evidence differs')
        else:common.require(perturbation=={'method':'original_EEG'},'original EEG evidence differs')
    common.require(all(seen[c]==ids for c,ids in expected.items()),'not all required condition/record identities were measured')
    return {'status':'passed','checkpoint_sha256':checkpoint,'condition_counts':{c:len(ids) for c,ids in expected.items()},
            'all_test_target_and_subject_identities_unchanged':True,'negative_control_identity_verified':True,
            'scope':'Prediction provenance and sealed sample substitution. Does not infer intention or independent real-world task success.'}


def audit_folder(prepared,evaluation):
    root=Path(evaluation);report=common.read_json(root/'evaluation_report.json')
    common.require(common.sha256(root/'predictions.jsonl')==report['predictions_sha256'],'saved predictions changed')
    rows=common.read_rows(Path(prepared)/'manifest.jsonl')
    common.require(common.sha256(Path(prepared)/'manifest.jsonl')==report['manifest_sha256'],'cohort changed')
    result=audit_predictions(rows,common.read_rows(root/'predictions.jsonl'),report['checkpoint_sha256'])
    return result


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();value=audit_folder(a.prepared,a.evaluation);common.save_json(a.output,value);print(json.dumps(value))
