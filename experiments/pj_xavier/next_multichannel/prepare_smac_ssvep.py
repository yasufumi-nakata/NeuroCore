#!/usr/bin/env python3
"""Add eligible Zenodo2392979 train records to the frozen 806023 SSL corpus.

The 2392979 split has no validation fold. Its 10 test files remain unopened;
the validation set remains 806023 only. This stage makes no claim to estimate
2392979 generalization. Calibrated V are not relabelled or rescaled to force QC.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy import signal

from prepare_smac_fif import (FS_OUT, PATCHES, PEAK_V, MIN_STD_V,
                              dump, read_denies, require, rows, sha)

CHANNELS = ['Oz', 'O1', 'O2', 'PO3', 'POz', 'PO7', 'PO8', 'PO4']
REFERENCE = 'publisher_converted_FIF_reference_unspecified_no_rereference'


def plan_records(ledger, metadata, constraints, denied_sources, denied_payloads, denied_groups=()):
    selected = [r for r in ledger if r.get('catalog_id') == '001148'
                and r.get('subject_group', '').startswith('zenodo:2392979:')]
    require(len(selected) == 36, 'All 36 source audit records are required')
    require(len({r['recording_id'] for r in selected}) == 36, 'Duplicate file identity')
    by_name = {r['recording_id']: r for r in selected}
    mrows = [r for r in metadata if r.get('source_group', '').startswith('small_bci_mixed|zenodo:2392979:')]
    require(len(mrows) == 36, 'All 36 frozen metadata split records are required')
    folds = {}
    metadata_identities = set()
    for r in mrows:
        g = r['source_group'].removeprefix('small_bci_mixed|')
        fold = r['metadata_only_proposed_split']
        require(fold in {'train', 'test'}, 'Do not invent a new validation fold')
        require(g not in folds or folds[g] == fold, 'One group crosses frozen folds')
        folds[g] = fold
        metadata_identities.add((g, r['source_file_sha256']))
    components = constraints['all_group_components']
    require(len(components) == 8 and sum(map(len, components)) == 12
            and len({g for gs in components for g in gs}) == 12,
            'The retained 8 must-link components / 12 nominal IDs are required')
    require({g for gs in components for g in gs} == set(folds), 'Must-link group coverage mismatch')
    component_for = {}
    for groups in components:
        require(len({folds[g] for g in groups}) == 1, 'Must-link component crosses frozen folds')
        component = hashlib.sha256(json.dumps(sorted(groups)).encode()).hexdigest()
        for g in groups:
            component_for[g] = component
    require(len(constraints['pairs']) == 7, 'All 7 retained duplicate proofs are required')
    for pair in constraints['pairs']:
        a, b = by_name[pair['left_record']], by_name[pair['right_record']]
        require(a['decoded_EEG_V_sha256'] == b['decoded_EEG_V_sha256'] == pair['decoded_EEG_V_sha256'],
                'Retained exact EEG duplicate proof disagrees with source audit')
        require(component_for[a['subject_group']] == component_for[b['subject_group']],
                'Duplicate EEG groups must be linked before preparing')
    denied_components = {component_for[r['subject_group']] for r in selected
                         if r['subject_group'] in denied_groups or r['source_sha256'] in denied_sources
                         or r['decoded_EEG_V_sha256'] in denied_payloads}
    seen, planned = {}, []
    for r in sorted(selected, key=lambda r: r['recording_id']):
        group, source, payload = r['subject_group'], r['source_sha256'], r['decoded_EEG_V_sha256']
        require((group, source) in metadata_identities, 'Source SHA / group absent from frozen split')
        require(r['channels'] == CHANNELS and r['sample_rate_hz'] == 256, 'Unexpected montage/rate')
        fold = folds[group]
        key = (payload, tuple(r['channels']), r['sample_count'], r['sample_rate_hz'])
        duplicate = seen.get(key)
        if duplicate:
            require(duplicate['split'] == fold, 'Identical EEG crosses frozen folds')
        if fold == 'test':
            status, reason = 'excluded', 'frozen_test_not_opened'
        elif component_for[group] in denied_components:
            status, reason = 'excluded', 'prior_heldout_or_review_identity'
        elif r.get('author_protocol_failure_datetime_candidate_match'):
            status, reason = 'excluded', 'author_flag_review_hold'
        elif duplicate:
            status, reason = 'duplicate_omitted', 'exact_EEG_already_represented'
        else:
            status, reason = 'candidate', 'requires_current_hash_calibration_and_fixed_QC'
        item = {'record': r, 'split': fold, 'status': status, 'reason': reason,
                'component_id_sha256': component_for[group],
                'duplicate_of': duplicate['record']['recording_id'] if duplicate else None}
        planned.append(item)
        seen.setdefault(key, item)
    require(Counter(p['split'] for p in planned) == {'train': 26, 'test': 10}, 'Frozen split changed')
    require(len(seen) == 29, 'Retained 29 unique calibrated EEG records changed')
    return planned


def causal_windows8(x):
    x = np.asarray(x, dtype=np.float64)
    require(x.ndim == 2 and x.shape[0] == 8, 'Expected 8-channel physical V')
    require(np.isfinite(x).all(), 'Nonfinite native source')
    # Preserve the first-stage physical-V sanity bound. No guessed unit correction.
    require(np.max(np.abs(x)) <= .1, 'native_peak_above_fixed_0_1V_source_bound')
    y = signal.sosfilt(signal.butter(4, (3, 40), btype='bandpass', fs=256, output='sos'), x, axis=1)
    y = signal.sosfilt(signal.butter(6, 80, btype='lowpass', fs=256, output='sos'), y, axis=1)
    js = np.arange(2 * FS_OUT, ((x.shape[1]-1) * FS_OUT)//256 + 1, dtype=np.int64)
    n = (js * 256)//FS_OUT
    a = ((js * 256) % FS_OUT).astype(np.float64)/FS_OUT
    z = (y[:, n-1]*(1-a) + y[:, n]*a).astype(np.float32)
    arrays, intervals, reject = [], [], Counter()
    width = PATCHES * FS_OUT
    for start in range(0, len(js)-width+1, width):
        w = z[:, start:start+width]
        if np.max(np.abs(w)) > PEAK_V:
            reject['peak_above_fixed_100uV'] += 1
        elif np.any(w.astype(np.float64).std(axis=1, ddof=0) < MIN_STD_V):
            reject['flat_below_fixed_0_01uV'] += 1
        else:
            arrays.append((w*1e4).reshape(8, PATCHES, FS_OUT).astype(np.float32))
            intervals.append({'window_start_s': int(js[start])/FS_OUT,
                              'window_end_s': (int(js[start+width-1])+1)/FS_OUT,
                              'latest_native_source_index': int(n[start+width-1]),
                              'feature_available_native_s': int(n[start+width-1])/256})
    return (np.stack(arrays) if arrays else np.empty((0, 8, PATCHES, FS_OUT), np.float32)), intervals, dict(reject)


def verify_fif(path, audit):
    import mne
    require(sha(path) == audit['source_sha256'], 'FIF source SHA mismatch')
    raw = mne.io.read_raw_fif(path, preload=False, verbose='ERROR')
    try:
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        require([raw.ch_names[i] for i in picks] == CHANNELS, 'Audited EEG channel order changed')
        require(raw.info['sfreq'] == 256 and raw.n_times == audit['sample_count'], 'Audited rate/sample count changed')
        calibration = []
        for i in picks:
            ch = raw.info['chs'][i]
            require(int(ch['unit']) == 107 and int(ch['unit_mul']) == 0, 'Header physical volts not confirmed')
            require(np.isfinite(ch['cal']) and ch['cal'] > 0 and np.isfinite(ch['range']) and ch['range'] > 0,
                    'Invalid source calibration')
            calibration.append({k: int(ch[k]) if k in ['kind', 'unit', 'unit_mul'] else float(ch[k])
                                for k in ['kind', 'unit', 'unit_mul', 'cal', 'range']})
        require(calibration == audit['channel_calibration'], 'Retained calibration changed')
        x = raw.get_data(picks=picks)
        payload = hashlib.sha256(np.asarray(x.T, dtype='<f8', order='C').tobytes()).hexdigest()
        require(payload == audit['decoded_EEG_V_sha256'], 'Calibrated physical-V payload mismatch')
        require(sha(path) == audit['source_sha256'], 'Source changed during read')
        return x, {'physical_unit_verified': True, 'verification_scope': 'FIF declared V and matching retained full decoded payload; external amplifier calibration unverified',
                   'observed_native_peak_V': float(np.max(np.abs(x))), 'source_causality_verified': False,
                   'source_kind': 'converted_FIF_not_original_GDF', 'online_harness_allowed': False,
                   'original_event_mapping_used': False}
    finally:
        raw.close()


def validate_base(base, expected_completion):
    base = Path(base).expanduser().resolve()
    require(sha(base/'completion.json') == expected_completion, 'Base completion SHA changed')
    completion = json.loads((base/'completion.json').read_text())
    require(completion['status'] == 'prepared_verified_new_offline_SSL_corpus', 'Expected verified 806023 base corpus')
    for name, field in [('manifest.jsonl', 'manifest_sha256'), ('protocol.json', 'protocol_sha256'),
                        ('artifacts.json', 'artifacts_sha256'), ('source_admission.json', 'source_admission_sha256')]:
        require(sha(base/name) == completion[field], 'Base corpus metadata changed: '+name)
    manifest = rows(base/'manifest.jsonl')
    admission = json.loads((base/'source_admission.json').read_text())
    artifacts = json.loads((base/'artifacts.json').read_text())
    require({r['split'] for r in manifest} == {'train', 'val'}, 'Base must keep train and validation')
    require(all(r['source_recording_id'].startswith('zenodo:806023:') for r in manifest), 'Base validation domain must remain 806023')
    for rel, spec in artifacts.items():
        p = (base/rel).resolve()
        require(base in p.parents and p.stat().st_size == spec['bytes'] and sha(p) == spec['sha256'],
                'Base array identity changed or escaped corpus')
    return base, completion, manifest, admission, artifacts


def prepare(args):
    out = Path(args.output).expanduser().absolute()
    require(not out.exists(), 'Use a new immutable output directory')
    raw_root = Path(args.raw_root).expanduser().resolve()
    base, base_completion, bm, ba, bart = validate_base(args.base_corpus, args.expected_base_completion_sha256)
    ds, dp, dg, deny_inputs = read_denies(args.deny_json)
    ds.update(ba['excluded_source_sha256']); dp.update(ba['excluded_payload_sha256'])
    metadata = rows(args.metadata_admission)
    constraints = json.loads(Path(args.must_link).read_text())
    plan = plan_records(rows(args.source_ledger), metadata, constraints, ds, dp, dg)
    new_tests = [p for p in plan if p['split'] == 'test']
    ds.update(p['record']['source_sha256'] for p in new_tests)
    dp.update(p['record']['decoded_EEG_V_sha256'] for p in new_tests)
    base_identities = set(ba['sources'])
    base_payloads = {v['eeg_payload_sha256'] for v in ba['sources'].values()}
    require(not base_identities.intersection(ds) and not base_payloads.intersection(dp), 'New heldout overlaps base corpus')
    require(shutil.disk_usage(raw_root).free >= args.reserve_gib*1024**3 + 512*1024**2 + sum(v['bytes'] for v in bart.values()),
            'Insufficient reserve for source preparation and exact base copy')
    out.mkdir(parents=True)
    (out/'waveforms').mkdir()
    protocol = {'schema': 'xavier-smac-2392979-source-addition-v1', 'created_epoch': time.time(), 'code_sha256': sha(__file__),
                'source_ledger_sha256': sha(args.source_ledger), 'metadata_admission_sha256': sha(args.metadata_admission),
                'must_link_sha256': sha(args.must_link), 'prior_deny_input_hashes': deny_inputs,
                'base_completion_sha256': args.expected_base_completion_sha256,
                'new_source_split': 'frozen_train26_test10; 10 test files unopened; 21 unique train candidates before QC',
                'validation_domain': 'retained 806023 only; no 2392979 generalization estimate',
                'QC': {'native_max_abs_V': .1, 'postprocess_max_abs_V': PEAK_V, 'minimum_channel_std_V': MIN_STD_V,
                       'thresholds_fixed_before_current_reads': True, 'unit_correction_or_clipping': False},
                'preprocessing': 'causal SOS3-40 and LP80; 256->200Hz delayed interpolation; warmup2s; nonoverlap3s; V_x_1e4',
                'source_causality_verified': False, 'original_event_mapping_used': False, 'online_harness_allowed': False}
    dump(out/'protocol.json', protocol)
    artifacts, new_manifest, new_sources, certificates = {}, [], {}, []
    montage = hashlib.sha256(json.dumps([CHANNELS, REFERENCE, 256, 'zenodo:2392979', 'physical_V']).encode()).hexdigest()
    for p in plan:
        r = p['record']
        cert = {k: v for k, v in p.items() if k != 'record'}
        cert.update(recording_id=r['recording_id'], source_sha256=r['source_sha256'],
                    eeg_payload_sha256=r['decoded_EEG_V_sha256'], subject_group=r['subject_group'], signal_opened=False)
        if p['status'] != 'candidate':
            certificates.append(cert); continue
        if r['subject_group'] in dg or r['source_sha256'] in base_identities or r['decoded_EEG_V_sha256'] in base_payloads:
            cert.update(status='excluded', reason='prior_group_or_base_identity_overlap'); certificates.append(cert); continue
        rel = Path(r['recording_id'])
        require(not rel.is_absolute() and len(rel.parts) == 1, 'Unsafe source filename')
        file = (raw_root/rel).resolve()
        require(raw_root in file.parents, 'Source escaped dataset folder')
        x, proof = verify_fif(file, r)
        cert.update(proof, signal_opened=True)
        try:
            array, windows, rejected = causal_windows8(x)
        except ValueError as error:
            cert.update(status='held_QC', reason=str(error)); certificates.append(cert); continue
        finally:
            del x
        cert.update(accepted_windows=len(windows), rejected_windows=rejected)
        if not len(windows):
            cert.update(status='held_QC', reason='no_window_meets_fixed_V_QC'); certificates.append(cert); continue
        relarray = 'waveforms/'+r['source_sha256']+'.npy'
        np.save(out/relarray, array, allow_pickle=False)
        artifacts[relarray] = {'shape': list(array.shape), 'dtype': 'float32', 'sha256': sha(out/relarray), 'bytes': (out/relarray).stat().st_size}
        new_sources[r['source_sha256']] = {'split': 'train', 'subject_group': r['subject_group'],
            'eeg_payload_sha256': r['decoded_EEG_V_sha256'], 'montage_key': montage, 'physical_unit_verified': True,
            'fit_allowed': True, 'validation_allowed': False, 'online_harness_allowed': False,
            'must_link_component_id': p['component_id_sha256'], 'source_recording_id': 'zenodo:2392979:'+r['recording_id']}
        for index, window in enumerate(windows):
            new_manifest.append({'id': r['source_sha256']+':'+str(index), 'split': 'train', 'subject_group': r['subject_group'],
                'source_sha256': r['source_sha256'], 'eeg_payload_sha256': r['decoded_EEG_V_sha256'],
                'source_recording_id': 'zenodo:2392979:'+r['recording_id'], **window, 'channels': CHANNELS,
                'sample_rate_hz': FS_OUT, 'unit': 'V_x_1e4', 'eeg_path': relarray, 'eeg_index': index,
                'montage_key': montage, 'reference': REFERENCE, 'native_sample_rate_hz': 256,
                'physical_unit_verified': True, 'source_causality': 'unverified_converted_FIF',
                'online_harness_allowed': False, 'must_link_component_id': p['component_id_sha256']})
        cert.update(status='admitted_offline_SSL', prepared_seconds=len(windows)*PATCHES)
        certificates.append(cert)
    dump(out/'source_certificates.json', certificates)
    if not new_manifest:
        result = {'status': 'blocked_no_eligible_new_source_windows', 'new_source_windows': 0,
                  'new_source_records': 0, 'new_source_unique_hours': 0, 'new_gradient_exposure_hours': 0,
                  'base_corpus_not_copied': True, 'test_records_not_opened': 10,
                  'candidate_records_before_current_QC': sum(p['status'] == 'candidate' for p in plan),
                  'reasons': dict(Counter(c['reason'] for c in certificates if c['status'] == 'held_QC')),
                  'protocol_sha256': sha(out/'protocol.json'), 'certificates_sha256': sha(out/'source_certificates.json')}
        dump(out/'preparation_result.json', result)
        return result
    # Preserve exact base arrays and unchanged validation rows after a verified new-source admission.
    for rel, spec in bart.items():
        target = out/'base'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base/rel, target)
        require(sha(target) == spec['sha256'], 'Base array copy hash mismatch')
        artifacts['base/'+rel] = spec
    manifest = [{**r, 'eeg_path': 'base/'+r['eeg_path']} for r in bm] + new_manifest
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False)+'\n' for r in manifest))
    dump(out/'artifacts.json', artifacts)
    dump(out/'source_admission.json', {'schema': 'xavier-smac-source-addition-offline-SSL-admission-v1',
         'sources': {**ba['sources'], **new_sources}, 'excluded_source_sha256': sorted(ds), 'excluded_payload_sha256': sorted(dp),
         'protocol_sha256': sha(out/'protocol.json'), 'online_harness_allowed': False,
         'validation_domain': '806023 only; no 2392979 validation or generalization claim'})
    counts = Counter(r['split'] for r in manifest)
    result = {'status': 'prepared_verified_new_offline_SSL_corpus', 'windows': dict(counts),
              'new_source_windows': len(new_manifest), 'new_source_records': len(new_sources),
              'new_source_unique_hours': len(new_manifest)*PATCHES/3600,
              'candidate_records_before_current_QC': sum(p['status'] == 'candidate' for p in plan),
              'retained_base_windows': len(bm), 'validation_domain': '806023 only', '2392979_generalization_evaluated': False,
              'test_records_not_opened': 10, 'new_gradient_exposure_hours': 0, 'online_harness_allowed': False,
              'protocol_sha256': sha(out/'protocol.json'), 'manifest_sha256': sha(out/'manifest.jsonl'),
              'artifacts_sha256': sha(out/'artifacts.json'), 'source_admission_sha256': sha(out/'source_admission.json'),
              'certificates_sha256': sha(out/'source_certificates.json'),
              'base_completion_sha256': args.expected_base_completion_sha256}
    dump(out/'completion.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('raw-root', 'source-ledger', 'metadata-admission', 'must-link', 'base-corpus', 'expected-base-completion-sha256', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--deny-json', action='append', required=True)
    p.add_argument('--reserve-gib', type=float, default=30)
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]/'acquisition'))
    from smac_storage_guard import require_smac_raw_root
    evidence = require_smac_raw_root(args.raw_root)
    node_root = (Path.home()/'pj-xavier'/evidence['node_id']).resolve()
    require(node_root in Path(args.output).expanduser().resolve().parents, 'Output must stay on this smac node')
    require(node_root in Path(args.base_corpus).expanduser().resolve().parents, 'Base corpus must be present on this smac node')
    require(args.reserve_gib >= 1, 'Positive disk reserve required')
    result = prepare(args)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] == 'prepared_verified_new_offline_SSL_corpus' else 2


if __name__ == '__main__':
    raise SystemExit(main())
