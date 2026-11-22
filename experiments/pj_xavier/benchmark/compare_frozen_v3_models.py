#!/usr/bin/env python3
"""Compare already-recorded v3 predictions; never loads or fits a neural model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

MANIFEST_SHA = '030e8550bd6f4c79005722eb617e9b64f5f80e87b77866b7f17bec4e7e2a1093'
SCORER_SHA = '5be7b7c4e91f6776895e2454adbec2ba12bfba1548e572e69b61ce7b9704ea13'
VARIANTS = ['fused', 'context_only', 'eeg_only', 'eeg_shuffled', 'eeg_time_shifted']
HISTORY_VARIANTS = {'ai_vs_other:' + name for name in ('history_pre_eeg', 'history_pre_eeg_plus_context', 'history_pre_decision_guarded', 'history_pre_decision_guarded_plus_context')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def derived_rows(prefix, evaluation, checkpoint, test_ids):
    """Authenticate each condition artifact before changing only its display name."""
    report_path = evaluation / 'variant_evaluation_report.json'
    report = read(report_path)
    if report['status'] != 'completed' or report['source_manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('completed report with the frozen source manifest required')
    if report['checkpoint_sha256'] != checkpoint or report['variant_names'] != VARIANTS:
        raise ValueError('frozen checkpoint or exact five-condition contract mismatch')
    rows, provenance = [], {'report_sha256': sha(report_path), 'checkpoint_sha256': checkpoint, 'conditions': {}}
    for variant in VARIANTS:
        metrics = report['conditions'][variant]
        if metrics['status'] != 'measured':
            raise ValueError('all conditions must be measured for this paired comparison')
        source = evaluation / (variant + '_predictions.jsonl')
        source_sha = sha(source)
        if source_sha != metrics['prediction_sha256'] or metrics['checkpoint_sha256'] != checkpoint:
            raise ValueError('condition file hash or checkpoint mismatch')
        original = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if len(original) != len(test_ids) or {r['id'] for r in original} != test_ids:
            raise ValueError('exact frozen held-out IDs required')
        if any(row['variant'] != variant or row['checkpoint_sha256'] != checkpoint for row in original):
            raise ValueError('mixed original condition or checkpoint')
        for row in original:
            derived = dict(row)
            derived['variant'] = prefix + '::' + variant
            derived['original_variant'] = variant
            derived['source_prediction_sha256'] = source_sha
            rows.append(derived)
        provenance['conditions'][variant] = {'original_prediction_sha256': source_sha, 'records': len(original)}
    return rows, provenance


def binary_history_lines(source, test_ids):
    selected, by_condition = [], {}
    for line in source.read_text().splitlines():
        row = json.loads(line)
        name = row.get('variant', row.get('condition', ''))
        if name in HISTORY_VARIANTS:
            selected.append(line)
            by_condition.setdefault(name, []).append(row['id'])
    if set(by_condition) != HISTORY_VARIANTS or any(len(ids) != len(test_ids) or set(ids) != test_ids for ids in by_condition.values()):
        raise ValueError('four exact primary history variants with complete held-out IDs required')
    return selected


def run(args):
    root = Path(args.worktree).resolve()
    private = root / 'private/xavier'
    public = root / 'experiments/pj_xavier/benchmark'
    manifest = private / 'prepared/ai_coding_uniba_v3_natural_casecontrol_verified/manifest.jsonl'
    scorer = public / 'history_baselines.py'
    if sha(manifest) != MANIFEST_SHA or sha(scorer) != SCORER_SHA:
        raise ValueError('frozen manifest or scorer changed')
    test = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    test_ids = {row['id'] for row in test if row['split'] == 'test'}
    if len(test_ids) != 203:
        raise ValueError('expected 203 held-out records')
    output = private / 'comparisons/uniba_v3_fixed_models_v2'
    output.mkdir(parents=True, exist_ok=False)
    qwen_root = private / 'colab_recovered_v3_qwen30e/runs/qwen3_8b_aicoding_uniba_natural_v3_fused_30e_v1'
    qwen_recovery = read(private / 'colab_recovered_v3_qwen30e/workstation_recovery_receipt_uniba_v3_qwen30e.json')
    if qwen_recovery['status'] != 'verified_workstation_recovery':
        raise ValueError('Qwen private artifact recovery not verified')
    qwen_eval = qwen_root / 'variants_v1'
    if sha(qwen_eval / 'variant_evaluation_report.json') != qwen_recovery['variant_report_sha256'] or sha(qwen_root / 'best.pt') != qwen_recovery['checkpoint_sha256']:
        raise ValueError('recovered Qwen report or checkpoint changed')
    smol_root = private / 'runs/smollm2_1_7b_uniba_natural_v3_fused_v1'
    smol_eval = private / 'evaluations/smollm2_1_7b_uniba_natural_v3_fiveconditions_v1'
    smol_receipt = read(smol_root / 'variant_evaluation_completion.json')
    if smol_receipt['status'] != 'completed' or smol_receipt['exit_code'] != 0 or sha(smol_eval / 'variant_evaluation_report.json') != smol_receipt['report_sha256'] or sha(smol_root / 'best.pt') != smol_receipt['checkpoint_sha256']:
        raise ValueError('Smol completion/checkpoint/report mismatch')
    sources, prediction_files = {}, []
    for prefix, evaluation, checkpoint in [('smol17b_3epochs', smol_eval, smol_receipt['checkpoint_sha256']), ('qwen8b_30epochs', qwen_eval, qwen_recovery['checkpoint_sha256'])]:
        rows, sources[prefix] = derived_rows(prefix, evaluation, checkpoint, test_ids)
        target = output / (prefix + '_derived_predictions.jsonl')
        with target.open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        sources[prefix]['derived_prediction_sha256'] = sha(target)
        prediction_files.append(target)
    baseline = private / 'runs/history_baselines_uniba_v3_v1'
    baseline_source = baseline / 'test_predictions.jsonl'
    if sha(baseline_source) != read(baseline / 'completion.json')['test_prediction_sha256']:
        raise ValueError('recorded history baseline predictions changed')
    target = output / 'history_binary_derived_predictions.jsonl'
    with target.open('x') as stream:
        for line in binary_history_lines(baseline_source, test_ids):
            stream.write(line + '\n')
    sources['history'] = {'original_prediction_sha256': sha(baseline_source), 'derived_prediction_sha256': sha(target), 'selection': 'binary primary endpoints only; original probabilities and record IDs unchanged'}
    prediction_files.append(target)
    save(output / 'derivation_provenance.json', {'schema': 'xavier-fixed-prediction-display-name-derivation-v1', 'manifest_sha256': MANIFEST_SHA, 'scorer_sha256': SCORER_SHA, 'sources': sources, 'probabilities_truth_and_ids_unchanged': True, 'new_model_inference_calls': 0, 'test_used_for_tuning': False})
    comparison_path = output / 'paired_ai_comparison.json'
    command = [sys.executable, str(scorer), 'score-predictions', '--manifest', str(manifest), '--expected-manifest-sha256', MANIFEST_SHA, '--output', str(comparison_path)]
    for path in prediction_files:
        command += ['--predictions', str(path)]
    with (output / 'scoring.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300, env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2'))
    if len(read(comparison_path)['models']) != 14:
        raise ValueError('expected ten model ablations and four history baselines')
    result = {'schema': 'xavier-fixed-v3-model-and-history-comparison-v1', 'status': 'completed', 'comparison': read(comparison_path), 'sources': sources, 'derivation_provenance_sha256': sha(output / 'derivation_provenance.json'), 'recorded_comparison_sha256': sha(comparison_path), 'analysis_scope': 'Descriptive comparison of existing held-out predictions; 1000 paired participant-cluster bootstrap replicates, no multiplicity adjustment. No inference, parameter fitting, checkpoint selection or probability threshold selection was performed.', 'training_budget_warning': 'Smol1.7B ran 3 epochs; Qwen8B ran 30 epochs. Different backbone, tokenizer, and training budget prevent assigning differences to parameter count alone.', 'goal_validated': False, 'natural_population_precision_estimated': False}
    save(output / 'aggregate.json', result)
    save(output / 'completion.json', {'status': 'completed', 'aggregate_sha256': sha(output / 'aggregate.json'), 'comparison_sha256': sha(comparison_path), 'new_model_inference_calls': 0})
    print(json.dumps({'status': 'completed', 'output': str(output), 'aggregate_sha256': sha(output / 'aggregate.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worktree', required=True)
    run(parser.parse_args())
