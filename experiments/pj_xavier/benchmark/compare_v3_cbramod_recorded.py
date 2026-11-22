#!/usr/bin/env python3
"""Add frozen CBraMod predictions to the descriptive UniBa v3 comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import compare_frozen_v3_models as previous

PREVIOUS_SHA = 'fa8db13dccfa2558fcf143efbab90a884aac6da53e51ee6a32e212b9bd9dad86'
CBR_CHECKPOINT = '3019cddd288a9052fa91f9af0a7bf81b5662106178f16367525a98e6667f3a4c'
CBR_REPORT = '11d30f5afee0ab492c2af73c8e4e1307ec8bddcdb56377bdcd61149542b03099'

def run(args):
    root = Path(args.worktree).resolve()
    private = root / 'private/xavier'
    public = root / 'experiments/pj_xavier/benchmark'
    manifest = private / 'prepared/ai_coding_uniba_v3_natural_casecontrol_verified/manifest.jsonl'
    scorer = public / 'history_baselines.py'
    if previous.sha(previous.__file__) != PREVIOUS_SHA or previous.sha(scorer) != previous.SCORER_SHA or previous.sha(manifest) != previous.MANIFEST_SHA:
        raise ValueError('frozen comparison utilities or manifest changed')
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    test_ids = {r['id'] for r in records if r['split'] == 'test'}
    if len(test_ids) != 203:
        raise ValueError('expected exactly 203 original test IDs')
    earlier = private / 'comparisons/uniba_v3_fixed_models_v2'
    old_receipt = previous.read(earlier / 'completion.json')
    if old_receipt['status'] != 'completed' or previous.sha(earlier/'aggregate.json') != old_receipt['aggregate_sha256']:
        raise ValueError('previous recorded comparison seal mismatch')
    old_provenance = previous.read(earlier / 'derivation_provenance.json')
    if previous.sha(earlier/'derivation_provenance.json') != previous.read(earlier/'aggregate.json')['derivation_provenance_sha256']:
        raise ValueError('previous source derivation seal mismatch')
    inputs = []
    for prefix, filename in [('smol17b_3epochs','smol17b_3epochs_derived_predictions.jsonl'), ('qwen8b_30epochs','qwen8b_30epochs_derived_predictions.jsonl'), ('history','history_binary_derived_predictions.jsonl')]:
        source = earlier / filename
        if previous.sha(source) != old_provenance['sources'][prefix]['derived_prediction_sha256']:
            raise ValueError('previous derived probabilities changed')
        inputs.append(source)
    run_dir = private / 'runs/cbramod_smollm2_uniba_v3_symbol_v1'
    evaluation = private / 'evaluations/cbramod_smollm2_uniba_v3_symbol_fiveconditions_v1'
    if previous.sha(run_dir/'best.pt') != CBR_CHECKPOINT or previous.sha(evaluation/'variant_evaluation_report.json') != CBR_REPORT:
        raise ValueError('frozen CBraMod checkpoint/report changed')
    receipt_path = run_dir/'variant_evaluation_completion.pending.json'
    receipt = previous.read(receipt_path)
    if receipt['status'] != 'completed' or receipt['exit_code'] != 0 or receipt['checkpoint_sha256'] != CBR_CHECKPOINT or receipt['report_sha256'] != CBR_REPORT:
        raise ValueError('CBraMod actual evaluation completion not authenticated')
    cbr_rows, cbr_source = previous.derived_rows('cbramod_smol17b_30epochs', evaluation, CBR_CHECKPOINT, test_ids)
    out = private/'comparisons/uniba_v3_with_cbramod_recorded_v1'
    out.mkdir(parents=True, exist_ok=False)
    predictions = out/'cbramod_derived_predictions.jsonl'
    with predictions.open('x') as stream:
        for row in cbr_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    inputs.append(predictions)
    cbr_source['derived_prediction_sha256'] = previous.sha(predictions)
    sources = {**old_provenance['sources'], 'cbramod_smol17b_30epochs': cbr_source}
    previous.save(out/'derivation_provenance.json', {'schema':'xavier-frozen-v3-cbramod-display-name-only-v1','manifest_sha256':previous.MANIFEST_SHA,'scorer_sha256':previous.SCORER_SHA,'previous_comparison_sha256':previous.sha(earlier/'aggregate.json'),'sources':sources,'probabilities_truth_and_ids_unchanged':True,'new_model_inference_calls':0})
    score_path = out/'paired_ai_comparison.json'
    command = [sys.executable,str(scorer),'score-predictions','--manifest',str(manifest),'--expected-manifest-sha256',previous.MANIFEST_SHA,'--output',str(score_path)]
    for source in inputs:
        command += ['--predictions',str(source)]
    with (out/'scoring.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600, env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    score = previous.read(score_path)
    expected = set(previous.HISTORY_VARIANTS) | {prefix+'::'+variant for prefix in ('smol17b_3epochs','qwen8b_30epochs','cbramod_smol17b_30epochs') for variant in previous.VARIANTS}
    if set(score['models']) != expected:
        raise ValueError('all 19 frozen conditions required')
    result = {'schema':'xavier-exploratory-v3-three-models-history-comparison-v1','status':'completed','conditions':19,'records_per_condition':203,'comparison':score,'sources':sources,'derivation_provenance_sha256':previous.sha(out/'derivation_provenance.json'),'recorded_comparison_sha256':previous.sha(score_path),'new_model_inference_calls':0,'new_parameter_fits':0,'analysis_scope':'Descriptive analysis of already recorded outputs. CBraMod experiment was designed after prior UniBa v3 results, so this comparison is exploratory. Fixed probability threshold0.5; 1000 paired participant-cluster replicates; no multiplicity adjustment or repeated-training uncertainty.','confounding':'Smol3epochs versus Qwen30epochs changes backbone and training budget. CBraMod also changes EEG encoder and label objective together; no isolated architectural cause is established.','goal_validated':False,'natural_population_precision_estimated':False}
    previous.save(out/'aggregate.json',result)
    previous.save(out/'completion.json',{'status':'completed','aggregate_sha256':previous.sha(out/'aggregate.json'),'comparison_sha256':previous.sha(score_path),'conditions':19,'records_per_condition':203,'new_model_inference_calls':0})
    print(json.dumps({'status':'completed','output':str(out),'aggregate_sha256':previous.sha(out/'aggregate.json')}))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worktree',required=True)
    run(parser.parse_args())
