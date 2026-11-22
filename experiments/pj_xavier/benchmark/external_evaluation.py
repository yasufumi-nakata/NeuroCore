#!/usr/bin/env python3
"""Frozen model inference on a separately sealed, external-only EEG cohort.

The training checkpoint's manifest and arrays stay unchanged. External arrays
are authenticated by their independent cohort contract, never injected into a
checkpoint or represented as training-owned artifacts.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import evaluate_variants as shared
from xavier_benchmark import classification_metrics

SCHEMA = 'xavier-external-only-evaluation-v1'
SHARED_SHA = 'bd92732e088ff633d2ff299161cf47a8a5630435c4c425dbae7992f563bc0447'
BENCHMARK_SHA = '3044e8e406f8fe73c9b5a7e58d23d1011d370241637e282f125175e018595956'
CLASSES = ['ai_prompt_submit', 'keyboard', 'mouse_click']
CHANNELS = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
SHIFT_METHOD = 'external_raw_recording_time_shift'
sha = shared.file_hash
save = shared.write_json


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(value, reason):
    if not value:
        raise ValueError(reason)


def verify_external_rows(external, original, expected_counts):
    require(bool(external), 'external cohort is empty')
    ids = [r['id'] for r in external]
    require(len(ids) == len(set(ids)), 'duplicate external IDs')
    require(not set(ids) & {r['id'] for r in original}, 'external observation IDs overlap the original cohort')
    source_hashes = {r['eeg_source_sha256'] for r in original}
    require(not source_hashes & {r['eeg_source_sha256'] for r in external}, 'external raw EEG bytes overlap an original source')
    intervals = defaultdict(list)
    for row in external:
        require(row.get('split') == 'external_test' and row.get('site') == 'itu', 'external-only site/partition required')
        require(row['subject'].startswith('itu:') and row['recording_id'].startswith('itu:'), 'site-qualified observation identity required')
        require(row.get('data_origin') == 'real_recording' and row.get('label_source') == 'recorded_event' and row.get('target_kind') == 'observed_action', 'external labels must be recorded observed actions')
        require(row['label'] in CLASSES, 'unknown external class')
        require(row.get('channels') == CHANNELS and row.get('sample_rate_hz') == 256, 'external montage/sample rate mismatch')
        values = [row[k] for k in ('window_start_s', 'window_end_s', 'decision_time_s', 'event_time_s', 'environment_time_s', 'clock_uncertainty_guard_s', 'feature_available_time_s')]
        require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values), 'nonfinite external timing')
        start, end, decision, event, context, guard, available = values
        require(0 <= start < end <= decision < event and guard >= 0, 'invalid external causal ordering')
        require(abs(end-start-1.75) < 1e-10 and end <= event-4.5-guard+1e-8 and abs(decision-end-4.125) < 1e-10, 'external signal/event gap changed')
        require(context <= start-guard+1e-8 and available <= decision+1e-8 and row.get('environment_source') == 'predecision_harness' and row.get('environment_contains_target_label') is False, 'external context or availability reaches future information')
        require(row.get('clock_sync', {}).get('verified') is True and bool(row.get('environment_text')), 'verified external clock/context required')
        require(row.get('preprocessing', {}).get('online_causality_verified') is False, 'offline preprocessing uncertainty must stay explicit')
        for field in ('eeg_source_sha256', 'label_evidence_sha256', 'environment_sha256'):
            require(bool(shared.SHA.fullmatch(str(row.get(field, '')))), 'source SHA missing: '+field)
        for name, boundary in [('history_pre_eeg', start-guard), ('history_pre_decision_guarded', decision-guard)]:
            history = row[name]
            latest = history['max_source_onset_used_s']
            require(abs(history['cutoff_s']-boundary) < 1e-8 and boundary < event and (latest is None or latest < boundary) and history.get('target_or_future_events_included') is False, 'external history crosses its past-only boundary')
        intervals[row['eeg_source_sha256']].append((start, end, row['id']))
    for group in intervals.values():
        ordered = sorted(group)
        require(all(a[1] <= b[0]+1e-10 for a, b in zip(ordered, ordered[1:])), 'overlapping external EEG windows')
    require(dict(Counter(r['label'] for r in external)) == expected_counts, 'external class counts differ from contract')
    return {'records': len(external), 'observation_subjects': len({r['subject'] for r in external}),
            'external_sites': ['itu'], 'raw_binary_hash_disjoint': True, 'observation_ids_disjoint': True,
            'human_identity_disjointness_verified': False, 'source_pretraining_overlap_excluded': False,
            'original_training_partitions_unchanged': True}


def verify_external_contract(prepared, original, contract, cache):
    prepared = Path(prepared)
    for name, expected in contract['cohort_files'].items():
        cache.checked(prepared/name, expected)
    manifest = prepared/'manifest.jsonl'
    external = rows(manifest)
    validation = verify_external_rows(external, original, contract['class_counts'])
    independent = read(prepared/'independent_validation.json')
    require(independent.get('status') == 'passed' and independent.get('manifest_sha256') == sha(manifest), 'independent raw/event/history validation not passed')
    require(independent.get('actual_waveform_windows_compared_sample_for_sample') == len(external) and independent.get('recorded_event_labels_rederived') == len(external) and independent.get('past_history_feature_sets_rederived') == 2*len(external) and independent.get('train_or_validation_rows') == 0, 'incomplete independent external validation')
    membership = read(prepared/'external_membership.json')
    require(membership.get('training_ids') == [] and membership.get('validation_ids') == [] and membership.get('external_test_ids') == sorted(r['id'] for r in external), 'external membership is not a pure test contract')
    artifacts = read(prepared/'eeg_artifacts.json')
    require(set(artifacts) == {r['eeg_path'] for r in external}, 'external artifact membership mismatch')
    for relative, evidence in artifacts.items():
        path = shared.inside(prepared, relative)
        cache.checked(path, evidence['sha256'])
        require(path.stat().st_size == evidence['bytes'], 'external artifact size mismatch')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        selected = [r for r in external if r['eeg_path'] == relative]
        require(array.dtype == np.float32 and array.shape == (len(selected), 8, 448) and np.isfinite(array).all(), 'external array shape/dtype/finite guard failed')
        require(sorted(r['eeg_index'] for r in selected) == list(range(len(selected))), 'external indices are duplicated or missing')
    return external, artifacts, validation


def checked_external_array(row, prepared, artifacts, cache):
    evidence = artifacts[row['eeg_path']]
    path = shared.inside(prepared, row['eeg_path'])
    cache.checked(path, evidence['sha256'])
    require(path.stat().st_size == evidence['bytes'], 'external prepared bytes changed')
    index = row.get('eeg_index')
    require(isinstance(index, int) and not isinstance(index, bool) and index >= 0, 'external array index invalid')
    array = np.load(path, mmap_mode='r', allow_pickle=False)[index]
    require(array.dtype == np.float32 and array.shape == (8, 448) and np.isfinite(array).all(), 'external model input differs from contract')
    return array


def verify_external_shift(detail, original, prepared, artifacts, cache, manifest_sha, contract_sha, checkpoint_sha):
    require(detail.get('method') == SHIFT_METHOD and detail.get('original_eeg_id') == original['id'], 'external shift target identity mismatch')
    source = detail['source']
    require(source.get('external_manifest_sha256') == manifest_sha and source.get('external_protocol_sha256') == contract_sha, 'external shift contract mismatch')
    require(source.get('checkpoint_sha256') == checkpoint_sha and source.get('external_arrays_sealed_in_checkpoint') is False, 'external shift must separate training seal from external evidence')
    require(source.get('recording_identity') == list(shared.flat_identity(original)), 'external shift crosses observation recording')
    require(source.get('raw_eeg_sha256') == original['eeg_source_sha256'], 'external shift raw source mismatch')
    require(source.get('raw_vhdr_sha256') == original['eeg_header_sha256'], 'external shift header differs from frozen source')
    require(source.get('external_prepared_array_sha256') == artifacts[original['eeg_path']]['sha256'], 'external prepared array evidence mismatch')
    for name in ('raw_eeg', 'raw_vhdr', 'raw_vmrk'):
        cache.checked(source[name+'_path'], source[name+'_sha256'])
    raw = shared.brainvision_reader(cache, source['raw_vhdr_path'], source['raw_eeg_path'], source['raw_vmrk_path'])
    fs = float(raw.info['sfreq'])
    require(fs == 256 == source['sample_rate_hz'] and source['channels'] == original['channels'], 'external shift montage/rate mismatch')
    first0, last0 = (shared.integral_sample(original[k], fs) for k in ('window_start_s', 'window_end_s'))
    first, last = source['start_sample'], source['stop_sample']
    require(all(isinstance(x, int) and not isinstance(x, bool) for x in (first, last)), 'external shift requires integer samples')
    require(0 <= first < last <= raw.n_times and source['raw_n_samples'] == raw.n_times, 'external shift outside physical raw bounds')
    require(detail['shift_seconds'] == -10 and first == first0-2560 and last == last0-2560 and source['shift_samples'] == -2560, 'external shift is not the actual minus10-second interval')
    require(source['start_s'] == first/fs and source['end_s'] == last/fs and last <= first0, 'external shifted timestamps inconsistent or overlapping')
    require(detail['eeg_source_id'] == shared.source_id(source) and detail['eeg_source_id'] != original['id'], 'external derived source identity is forged')
    original_array = checked_external_array(original, prepared, artifacts, cache)
    require(np.array_equal(raw.get_data(picks=source['channels'], start=first0, stop=last0).astype(np.float32), original_array), 'actual raw does not reproduce the sealed external window')
    path, _ = cache.checked(source['extracted_array_path'], source['extracted_array_sha256'])
    extracted = np.load(path, allow_pickle=False)
    require(extracted.dtype == np.float32 and extracted.shape == (8, 448), 'external shifted array shape/dtype mismatch')
    require(np.array_equal(raw.get_data(picks=source['channels'], start=first, stop=last).astype(np.float32), extracted), 'external saved shift does not match actual raw samples')
    return {'status': 'passed', 'original_id': original['id'], 'source_id': detail['eeg_source_id'], 'sample_interval': [first, last], 'training_checkpoint_unmodified': True, 'online_causality_verified': False}


def prepare_external_shifts(external, prepared, artifacts, raw_root, out, cache, manifest_sha, protocol_sha, checkpoint_sha, source_verification_sha):
    verification_path, _ = cache.checked(Path(raw_root)/'source_verification.json', source_verification_sha)
    verification = read(verification_path)
    complete = {r.get('filename') for r in verification.get('sources', []) if r.get('ledger_status') == 'complete' and r.get('reverified_before_extraction') is True}
    require(verification.get('verified') is True and {'bids_eeg_itu.tar.gz', 'bids_behavioral_itu.tar.gz'} <= complete, 'external original EEG/behavioral archives not verified')
    members = read(Path(prepared)/'source_members_verified.json')
    destination = out/'raw_shift_arrays'; destination.mkdir()
    certificates, blocked, checks = {}, [], []
    for row in external:
        try:
            header, binary, marker = shared.raw_paths_for(row, raw_root, cache)
            for path in (header, binary, marker):
                member = str(path.relative_to(Path(raw_root).resolve()))
                cache.checked(path, members[member]['sha256'])
                require(path.stat().st_size == members[member]['bytes'], 'external original member size mismatch')
            raw = shared.brainvision_reader(cache, header, binary, marker)
            first = shared.integral_sample(row['window_start_s'], 256)-2560
            last = shared.integral_sample(row['window_end_s'], 256)-2560
            require(0 <= first < last <= raw.n_times, 'minus10-second window outside external raw bounds')
            value = raw.get_data(picks=CHANNELS, start=first, stop=last).astype(np.float32)
            target = destination/(hashlib.sha256(row['id'].encode()).hexdigest()+'.npy')
            np.save(target, value, allow_pickle=False)
            source = {'external_manifest_sha256': manifest_sha, 'external_protocol_sha256': protocol_sha,
                      'checkpoint_sha256': checkpoint_sha, 'external_arrays_sealed_in_checkpoint': False,
                      'external_prepared_array_sha256': artifacts[row['eeg_path']]['sha256'],
                      'recording_identity': list(shared.flat_identity(row)), 'sample_rate_hz': 256,
                      'start_sample': first, 'stop_sample': last, 'start_s': first/256, 'end_s': last/256,
                      'shift_samples': -2560, 'raw_n_samples': int(raw.n_times), 'channels': CHANNELS,
                      'extracted_array_path': shared.relative(cache.root, target), 'extracted_array_sha256': sha(target),
                      'source_verification_sha256': source_verification_sha,
                      'source_preprocessing_unchanged': True, 'online_causality_verified': False,
                      'feature_available_time_s_under_source_assumption': last/256+4.125}
            for name, path in [('raw_eeg', binary), ('raw_vhdr', header), ('raw_vmrk', marker)]:
                source[name+'_path'], source[name+'_sha256'] = shared.relative(cache.root, path), cache.checked(path)[1]
            detail = {'method': SHIFT_METHOD, 'original_eeg_id': row['id'], 'shift_seconds': -10,
                      'source': source, 'eeg_source_id': shared.source_id(source)}
            checks.append(verify_external_shift(detail, row, prepared, artifacts, cache, manifest_sha, protocol_sha, checkpoint_sha))
            certificates[row['id']] = detail
        except (OSError, ValueError, KeyError, IndexError) as error:
            blocked.append({'record_id': row['id'], 'reason': type(error).__name__+': '+str(error)})
    status = {'status': 'ready' if len(certificates) == len(external) and not blocked else 'blocked',
              'expected_records': len(external), 'verified_records': len(certificates), 'blocked_records': blocked,
              'shift_seconds': -10, 'reason': None if not blocked else 'Complete external cohort cannot be shifted; no partial-cohort score is permitted.',
              'external_arrays_sealed_in_checkpoint': False, 'external_cohort_contract_verified': True}
    shared.write_jsonl(out/'raw_shift_sources.jsonl', [{'record_id': key, 'perturbation': value} for key, value in certificates.items()])
    save(out/'raw_shift_preparation.json', status)
    save(out/'raw_shift_independent_checks.json', {'records': checks, 'status': status['status']})
    return certificates, status


def check_shuffle(predictions, external):
    index = {r['id']: r for r in external}
    observed = []
    for p in predictions:
        if p['variant'] != 'eeg_shuffled':
            continue
        target, source = p['id'], p.get('perturbation', {}).get('eeg_source_id')
        require(target in index and source in index and target != source, 'external shuffle has missing/identity source')
        require(shared.flat_identity(index[target]) == shared.flat_identity(index[source]), 'external shuffle crosses a recording')
        require(p.get('eeg_source_id') == source and p['perturbation'].get('original_eeg_id') == target, 'external shuffle correspondence mismatch')
        observed.append(source)
    require(len(observed) == len(external) and set(observed) == set(index), 'external shuffle is not a complete permutation')
    return {'status': 'passed', 'records_checked': len(observed), 'fixed_points': 0, 'same_recording': True}


def run(args):
    started = time.monotonic()
    protocol_path = Path(args.protocol)
    require(sha(protocol_path) == args.expected_protocol_sha256, 'external evaluation protocol changed')
    protocol = read(protocol_path)
    require(protocol['schema'] == SCHEMA and protocol['runner_sha256'] == sha(__file__), 'external protocol does not seal this runner')
    require(sha(shared.__file__) == SHARED_SHA and sha(Path(__file__).with_name('xavier_benchmark.py')) == BENCHMARK_SHA, 'frozen evaluation utilities changed')
    require(time.time() < protocol['absolute_deadline_epoch'], 'original absolute deadline reached')
    root = Path(args.artifact_root).resolve()
    prepared = shared.inside(root, args.external_prepared)
    checkpoint = shared.inside(root, args.checkpoint)
    original_manifest = shared.inside(root, args.training_manifest)
    reference_run = checkpoint.parent
    out = shared.inside(root, args.output)
    require(not out.exists(), 'external output already exists; preserve final-test execution')
    out.parent.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(out.parent).free >= protocol['disk_reserve_gib']*1024**3 + 64*1024**2, 'external evaluation disk reserve')
    spec = protocol['models'][args.model_key]
    require(sha(args.model_module) == spec['source_sha256'], 'external model source differs from registration')
    require(sha(original_manifest) == protocol['training_manifest_sha256'], 'original training manifest changed')
    checkpoint_sha = sha(checkpoint)
    if spec.get('checkpoint_sha256'):
        require(checkpoint_sha == spec['checkpoint_sha256'], 'frozen model checkpoint changed')
    summary = read(reference_run/'run_summary.json')
    require(summary.get('checkpoint_sha256') == checkpoint_sha and summary.get('model_selection_partition') == 'validation' and summary.get('test_used_for_tuning') is False, 'checkpoint was not sealed by validation-only within-run selection')
    require(summary.get('status') in ('completed_epochs', 'stopped_time_budget'), 'training failure cannot be used as final external model')
    require(summary.get('manifest_sha256') == protocol['training_manifest_sha256'], 'training summary refers to another cohort')
    if args.device == 'cuda':
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], capture_output=True, text=True, check=True)
        require(not gpu.stdout.strip(), 'external evaluation cannot overlap another GPU job')
    out.mkdir(parents=True)
    save(out/'started.json', {'schema': SCHEMA, 'status': 'started', 'model_key': args.model_key, 'checkpoint_sha256': checkpoint_sha, 'protocol_sha256': args.expected_protocol_sha256, 'started_at': shared.utc_now(), 'pid': os.getpid(), 'new_training': False})
    import torch
    api = shared.load_api(args.model_module)
    cache = shared.EvidenceCache(root)
    cache.checked(protocol_path, args.expected_protocol_sha256)
    cache.checked(checkpoint, checkpoint_sha)
    cache.checked(args.model_module, spec['source_sha256'])
    cache.checked(original_manifest, protocol['training_manifest_sha256'])
    for name in ('training_membership.json', 'run_summary.json', 'eeg_artifacts.json'):
        cache.checked(reference_run/name)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    original_rows = api.read_rows(original_manifest)
    membership = read(reference_run/'training_membership.json')
    shared.verify_membership(original_rows, membership, protocol['training_manifest_sha256'], state, 203)
    api.validate_rows(original_rows, 'subject', 'fused')
    actual_training_artifacts = api.eeg_artifact_manifest(original_rows)
    require(actual_training_artifacts == state['eeg_artifacts'] == read(reference_run/'eeg_artifacts.json'), 'checkpoint-original training EEG seal mismatch')
    for row in original_rows:
        cache.checked(row['eeg_path'], actual_training_artifacts[row['_manifest_eeg_path']]['sha256'])
    external, artifacts, cohort = verify_external_contract(prepared, original_rows, protocol['external_cohort'], cache)
    require(not set(artifacts) & set(actual_training_artifacts), 'external and original artifact names unexpectedly collide')
    config = dict(state['config'])
    require(config.get('abstain_threshold') == .8 and state['classes'] == CLASSES, 'frozen acceptance threshold/class order changed')
    require(config.get('model') == spec['backbone'] and config.get('revision') == spec['revision'], 'frozen backbone identity changed')
    require(config.get('mode') == 'fused', 'external ablations require a fused trained checkpoint')
    save(out/'reference_freeze.json', {'checkpoint_sha256': checkpoint_sha, 'model_source_sha256': spec['source_sha256'], 'reference_membership_sha256': sha(reference_run/'training_membership.json'), 'reference_artifacts_sha256': sha(reference_run/'eeg_artifacts.json'), 'reference_summary_sha256': sha(reference_run/'run_summary.json'), 'original_manifest_sha256': protocol['training_manifest_sha256'], 'external_manifest_sha256': protocol['external_cohort']['cohort_files']['manifest.jsonl'], 'external_protocol_sha256': args.expected_protocol_sha256, 'within_run_selection': 'UniBa validation only', 'ITU_used_for_model_selection': False, 'new_training': False})
    certificates, preparation = prepare_external_shifts(external, prepared, artifacts, args.raw_bids_root, out, cache,
        protocol['external_cohort']['cohort_files']['manifest.jsonl'], args.expected_protocol_sha256, checkpoint_sha,
        protocol['external_cohort']['source_verification_sha256'])
    model_rows = api.read_rows(prepared/'manifest.jsonl')
    config.update(device=args.device, memory_budget_gib=protocol['gpu_memory_budget_gib'], candidate_batch_size=spec['candidate_batch_size'])
    device = api.select_device(args.device)
    allocator = api.apply_allocator_limit(device, config['memory_budget_gib'])
    torch.set_num_threads(2)
    base, tokenizer, backbone = api.load_backbone(config, device)
    model = api.EEGCausalModel(base, config['channels'], config['prefix_tokens'], config['encoder_width']).to(device)
    api.attach_lora(model, config['lora_rank'])
    shared.restore_loaded_state(api, state, model)
    conditions = shared.evaluate_conditions(api, model, tokenizer, model_rows, CLASSES, api.EEGStore(config['samples']), config, device, checkpoint_sha, out, certificates, preparation, root)
    predictions = rows(out/'predictions.jsonl')
    require(all(conditions[name]['status'] == 'measured' for name in ('fused', 'context_only', 'eeg_only')), 'one or more primary external input conditions were not measured')
    controls = {}
    if conditions['eeg_shuffled']['status'] == 'measured':
        controls['eeg_shuffled'] = check_shuffle(predictions, external)
    else:
        controls['eeg_shuffled'] = conditions['eeg_shuffled']
    if conditions['eeg_time_shifted']['status'] == 'measured':
        for row in external:
            verify_external_shift(certificates[row['id']], row, prepared, artifacts, cache, protocol['external_cohort']['cohort_files']['manifest.jsonl'], args.expected_protocol_sha256, checkpoint_sha)
        controls['eeg_time_shifted'] = {'status': 'passed', 'records_checked': len(external), 'external_array_contract_separate_from_checkpoint': True}
    else:
        controls['eeg_time_shifted'] = conditions['eeg_time_shifted']
    by_id = {r['id']: r for r in external}
    calibration = {}
    for name in shared.VARIANTS:
        subset = [p for p in predictions if p['variant'] == name]
        if conditions[name]['status'] == 'measured':
            calibration[name] = classification_metrics([(by_id[p['id']]['label'], p['probabilities'], p['abstain']) for p in subset], CLASSES)
    memory, hardware = api.memory_sample(device), {**api.hardware(device), **backbone}
    require(memory.get('cuda_peak_reserved_bytes', math.inf) <= protocol['gpu_memory_budget_gib']*1024**3, 'external GPU memory budget exceeded or actual peak missing')
    require(memory['process_peak_rss_bytes'] < hardware['system_memory_bytes'], 'host memory evidence exceeds physical capacity')
    cache.unchanged()
    report = {'schema_version': shared.VERSION, 'external_schema_version': SCHEMA, 'evaluation_scope': 'external_only',
              'model_key': args.model_key, 'checkpoint_sha256': checkpoint_sha, 'source_manifest_sha256': protocol['external_cohort']['cohort_files']['manifest.jsonl'],
              'training_manifest_sha256': protocol['training_manifest_sha256'], 'external_protocol_sha256': args.expected_protocol_sha256,
              'model_module_sha256': spec['source_sha256'], 'evaluator_sha256': sha(__file__), 'shared_evaluator_sha256': SHARED_SHA,
              'variant_names': list(shared.VARIANTS), 'conditions': conditions, 'test_record_count': len(external),
              'frozen_test_ids_sha256': shared.canonical_hash([r['id'] for r in external]), 'classes': CLASSES,
              'external_cohort_verification': cohort, 'negative_control_verification': controls,
              'strict_with_abstention': calibration, 'memory': memory, 'hardware': hardware, 'allocator_limit': allocator,
              'resource_feasibility': {'status': 'passed', 'scope': 'actual Colab CUDA inference only; component latency excludes preprocessing and harness', 'budget_gib': protocol['gpu_memory_budget_gib']},
              'raw_shift_preparation': preparation, 'elapsed_seconds': time.monotonic()-started,
              'weights_updated': False, 'model_selection_changed': False, 'test_used_for_tuning': False, 'os_actions_executed': 0,
              'original_training_membership_or_checkpoint_modified': False, 'human_identity_disjointness_verified': False,
              'site_cluster_uncertainty': {'status': 'blocked', 'n_external_sites': 1, 'reason': 'Only ITU is an external site; a distribution across unseen sites cannot be estimated.'},
              'binary_and_subject_cluster_analysis': {'status': 'pending_cpu_scoring_of_recorded_predictions', 'AI_probability_threshold': .5, 'bootstrap_repetitions': 1000},
              'goal_validation_blockers': ['offline preprocessing causality unverified', 'no independent latent-intent ground truth', 'no idle exposure labels or measured false actions per minute', 'no real harness task completion', 'human identity disjointness across sites unverified', 'single external site', 'component latency is not end-to-end latency']}
    receipt = shared.finish_report(report, out/'external_evaluation_report.json', out/'variant_evaluation_completion.pending.json')
    print(json.dumps({'status': 'external_predictions_completed_pending_cpu_analysis_and_recovery', 'receipt': receipt}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'expected-protocol-sha256', 'model-key', 'model-module', 'checkpoint', 'training-manifest', 'external-prepared', 'raw-bids-root', 'artifact-root', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--device', choices=['cuda'], default='cuda')
    args = p.parse_args()
    try:
        run(args)
    except Exception as error:
        out = Path(args.output)
        if (out/'started.json').exists() and not (out/'failure.json').exists():
            save(out/'failure.json', {'schema': SCHEMA, 'status': 'failed_or_blocked', 'error': type(error).__name__+': '+str(error), 'inference_success_claimed': False, 'automatic_retry_permitted': False})
        raise


if __name__ == '__main__':
    main()
