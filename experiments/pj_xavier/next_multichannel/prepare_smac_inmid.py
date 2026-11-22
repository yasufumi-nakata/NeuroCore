#!/usr/bin/env python3
"""Add only verified InMID train EDF EEG to any sealed offline SSL base.

The 92-record frozen metadata has 76 train and 16 test records. Only the
train signals are opened. Exactly 14 of 58 EDF channels are EEG. Native
128 Hz data are causally upsampled, not relabelled as new independent data.
The existing 806023 validation remains unchanged and is not an InMID test.
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
import scipy
from scipy import signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'acquisition'))
sys.path.insert(0, str(HERE.parent/'model'))
import audit_inmid_edf as native
import cbramod_ssl_smac as trainer
import prepare_smac_fif as common
from prepare_smac_fif import FS_OUT, PATCHES, PEAK_V, MIN_STD_V, dump, read_denies, require, rows, sha

LEDGER_SHA256 = 'e25206409c48ffd854cf3a3f45f3624d5471426a933f82885b1396b8d9af6669'
METADATA_SHA256 = '555b3247b6a9eadcf48bed22f29ef381b06a601e87909241aedcc60900c70e1d'
AUDIT_PROTOCOL_SHA256 = '9ab3b3053aeea7276290757d91150579778c858f05128d6bf33a015ffa4f979d'
AUDIT_MODULE_SHA256 = '8288d629afefa11e653acb8788edb8d3da641e7d7b5658ab7d28137713c99b75'
EDF_HELPER_SHA256 = 'fc61fed2388ad26b5cf2b0c4bba907c3e1b5d0725bb06c989deed90d6728189b'
CHANNELS = list(native.EEG)
NATIVE_HZ = 128
REFERENCE = 'publisher_Emotiv_EDF_reference_unspecified_no_rereference'
GROUP_PREFIX = 'mendeley:rjx76wd5v6:P'
TEST_GROUPS = {GROUP_PREFIX+f'{n:02d}' for n in (1, 5, 8, 11)}
ALL_GROUPS = {GROUP_PREFIX+f'{n:02d}' for n in range(1, 24)}
RELEASE_KEY = 'landing-https___data.mendeley.com_datasets_rjx76wd5v6-observed-unpinned-4e394f5ce88c'


def payloads(record):
    return {record['EEG_payload_sha256'], record['decoded_EEG_V_sha256']}


def plan_records(ledger, metadata, denied_sources, denied_payloads, denied_groups=()):
    require(len(ledger) == 92 and len({r['source_path'] for r in ledger}) == 92,
            'All 92 unique original EDF audit records are required')
    selected = [r for r in metadata if r.get('source') == 'inmid14']
    require(len(selected) == 92, 'All 92 frozen InMID metadata rows are required')
    by_path = {r['source_member_or_path']: r for r in selected}
    require(len(by_path) == 92, 'Duplicate metadata source path')
    held_groups = {r['subject_group'] for r in ledger if r['subject_group'] in denied_groups
                   or 'inmid14|'+r['subject_group'] in denied_groups
                   or r['source_sha256'] in denied_sources or payloads(r) & set(denied_payloads)}
    plan, source_ids, physical_ids, groups = [], set(), set(), Counter()
    for record in sorted(ledger, key=lambda r: r['source_path']):
        name, group = record['source_path'], record['subject_group']
        require(group in ALL_GROUPS, 'Unreviewed InMID source group')
        m = by_path.get(name)
        require(m is not None and m['source_group'] == 'inmid14|'+group,
                'Frozen InMID metadata group/path mismatch')
        require(m['source_file_sha256'] == record['source_sha256']
                and m['eeg_payload_sha256'] == record['EEG_payload_sha256']
                and m['calibration_sha256'] == record['calibration_sha256'],
                'Frozen InMID source/digital payload/calibration differs')
        fold = 'test' if group in TEST_GROUPS else 'train'
        require(m['metadata_only_proposed_split'] == fold, 'Frozen InMID split changed')
        require(record['channels'] == 58 and record['EEG_channels'] == CHANNELS
                and record['EEG_indices'] == list(range(4, 18))
                and len(record['auxiliary_indices']) == 44 and record['sample_rate_hz'] == NATIVE_HZ,
                'Expected audited 14 EEG plus 44 auxiliary at 128 Hz')
        require(record['MNE_header_and_three_EEG_calibration_probes_passed'] is True
                and record['MNE_max_probe_difference_V'] <= 1e-12,
                'Retained independent physical-V calibration is unverified')
        require(record['source_sha256'] not in source_ids
                and record['decoded_EEG_V_sha256'] not in physical_ids,
                'InMID duplicate requires explicit must-link review')
        source_ids.add(record['source_sha256']); physical_ids.add(record['decoded_EEG_V_sha256'])
        groups[group] += 1
        if fold != 'train':
            status, reason = 'excluded_frozen_holdout', 'new_InMID_test_signal_not_opened'
        elif group in held_groups:
            status, reason = 'excluded_prior_identity', 'prior_source_payload_or_group_deny'
        else:
            status, reason = 'candidate', 'requires_current_physical_V_and_fixed_QC'
        plan.append({'record': record, 'split': fold, 'status': status, 'reason': reason,
                     'component_id_sha256': m['component_id_sha256']})
    require(set(groups) == ALL_GROUPS and set(groups.values()) == {4}, 'Expected 23 groups with 4 recordings each')
    require(Counter(p['split'] for p in plan) == {'train': 76, 'test': 16}, 'Frozen split counts differ')
    return plan


def validate_base(path, expected_completion_sha):
    base = Path(path).expanduser().resolve()
    require(sha(base/'completion.json') == expected_completion_sha, 'Base completion SHA changed')
    completion = json.loads((base/'completion.json').read_text())
    require(completion['status'] == 'prepared_verified_new_offline_SSL_corpus', 'Base is not a verified SSL corpus')
    for name, key in [('manifest.jsonl', 'manifest_sha256'), ('artifacts.json', 'artifacts_sha256'),
                      ('source_admission.json', 'source_admission_sha256'), ('protocol.json', 'protocol_sha256')]:
        require(sha(base/name) == completion[key], 'Base metadata seal changed: '+name)
    trainer.load_corpus(base/'manifest.jsonl', base/'artifacts.json', base/'source_admission.json')
    manifest = rows(base/'manifest.jsonl')
    require(all(r.get('online_harness_allowed') is False for r in manifest), 'Base must remain offline-only')
    require(all(r['source_recording_id'].startswith('zenodo:806023:') for r in manifest if r['split'] == 'val'),
            'Preserve only the existing 806023 validation domain')
    return base, manifest, json.loads((base/'source_admission.json').read_text()), json.loads((base/'artifacts.json').read_text())


def verify_edf(path, record):
    """Replay the audited integer and physical-V identities; never decode labels."""
    path = Path(path)
    before = path.stat()
    require(before.st_size == record['source_bytes'], 'Original EDF size changed')
    header = native.header(path)
    for key, value in header.items():
        require(value == record[key], 'Audited EDF header changed: '+key)
    require(native.source_group(path.name) == (record['subject_group'], record['source_session'],
                                              record['source_recording_in_session']), 'Original EDF filename/group changed')
    whole, digital, physical = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    chunks = []
    with path.open('rb') as stream:
        whole.update(stream.read(header['header_bytes']))
        for _ in range(header['records']):
            blob = stream.read(58*NATIVE_HZ*2)
            require(len(blob) == 58*NATIVE_HZ*2, 'Truncated original EDF record')
            whole.update(blob)
            raw = np.frombuffer(blob, dtype='<i2').reshape(58, NATIVE_HZ)
            eeg = np.ascontiguousarray(raw[header['EEG_indices']])
            digital.update(eeg.tobytes())
            volts = native.physical_EEG_volts(eeg, header)
            require(np.isfinite(volts).all(), 'Nonfinite physical EEG volts')
            physical.update(np.ascontiguousarray(volts.T, dtype='<f8').tobytes())
            chunks.append(volts)
        require(stream.read(1) == b'', 'Unexpected EDF tail')
    require(whole.hexdigest() == record['source_sha256'], 'Original EDF SHA changed')
    require(digital.hexdigest() == record['EEG_payload_sha256'], 'Original integer EEG payload changed')
    require(physical.hexdigest() == record['decoded_EEG_V_sha256'], 'Calibrated physical-V EEG payload changed')
    calibration = {k: [header[k][i] for i in header['EEG_indices']]
                   for k in ('physical_min', 'physical_max', 'digital_min', 'digital_max', 'physical_units')}
    calibration['channels'] = CHANNELS
    require(hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest() == record['calibration_sha256'],
            'Original EEG calibration fingerprint changed')
    probe = native.mne_probe(path, header)
    require(probe['MNE_header_and_three_EEG_calibration_probes_passed'], 'Independent MNE calibration failed')
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
            and sha(path) == record['source_sha256'], 'EDF changed during verification')
    x = np.concatenate(chunks, axis=1)
    require(x.shape == (14, record['sample_count']), 'Decoded EEG shape mismatch')
    return x, {'physical_unit_verified': True, 'verification_scope': 'EDF declared microvolt calibration to V, full retained physical payload and independent MNE probes; amplifier calibration unverified',
               'original_integer_EEG_payload_sha256': digital.hexdigest(), 'decoded_EEG_V_sha256': physical.hexdigest(),
               'calibration_sha256': record['calibration_sha256'], 'EEG_channel_count': 14, 'auxiliary_channels_excluded': 44,
               'original_source_bytes': before.st_size, 'observed_native_peak_V': float(np.max(np.abs(x))),
               **probe, 'native_marker_or_label_used': False, 'MAT_or_video_used': False,
               'source_causality_verified': False, 'online_harness_allowed': False}


def filter_sos():
    # A fixed source-specific filter: 3 < 40 < native Nyquist 64 Hz.
    return signal.butter(4, (3, 40), btype='bandpass', fs=NATIVE_HZ, output='sos')


def causal_transform128(x):
    x = np.asarray(x, dtype=np.float64)
    require(x.ndim == 2 and x.shape[0] == 14 and x.shape[1] > 0, 'Expected [14,N] physical EEG volts')
    require(np.isfinite(x).all(), 'Nonfinite native physical EEG')
    require(float(np.max(np.abs(x))) <= .1, 'native_peak_above_fixed_0_1V_source_bound')
    y = signal.sosfilt(filter_sos(), x, axis=1)
    js = np.arange(2*FS_OUT, ((x.shape[1]-1)*FS_OUT)//NATIVE_HZ+1, dtype=np.int64)
    cutoffs = (js*NATIVE_HZ)//FS_OUT
    fraction = ((js*NATIVE_HZ) % FS_OUT).astype(np.float64)/FS_OUT
    # At output time j/200 this uses only native samples n-1 and n,
    # n=floor(128*j/200); its represented signal time is j/200 - 1/128.
    z = y[:, cutoffs-1]*(1-fraction) + y[:, cutoffs]*fraction
    return z, js, cutoffs


def causal_windows14(x):
    y, js, cutoffs = causal_transform128(x)
    arrays, windows, rejected = [], [], Counter()
    width = PATCHES*FS_OUT
    for start in range(0, len(js)-width+1, width):
        w = y[:, start:start+width]
        if not np.isfinite(w).all():
            rejected['nonfinite'] += 1
        elif np.max(np.abs(w)) > PEAK_V:
            rejected['peak_above_fixed_100uV'] += 1
        elif np.any(w.std(axis=1, ddof=0) < MIN_STD_V):
            rejected['flat_below_fixed_0_01uV'] += 1
        else:
            arrays.append((w*1e4).reshape(14, PATCHES, FS_OUT).astype(np.float32))
            windows.append({'window_start_s': int(js[start])/FS_OUT,
                            'window_end_s': (int(js[start+width-1])+1)/FS_OUT,
                            'latest_native_source_index': int(cutoffs[start+width-1]),
                            'feature_available_native_s': int(cutoffs[start+width-1])/NATIVE_HZ})
    return (np.stack(arrays) if arrays else np.empty((0, 14, PATCHES, FS_OUT), np.float32)), windows, dict(rejected)


def verify_input_seals(args):
    for path, expected in [(args.source_ledger, LEDGER_SHA256), (args.metadata_admission, METADATA_SHA256),
                           (args.audit_protocol, AUDIT_PROTOCOL_SHA256), (native.__file__, AUDIT_MODULE_SHA256),
                           (native.edf_reader.__file__, EDF_HELPER_SHA256)]:
        require(sha(path) == expected, 'InMID audited input/parser seal changed: '+str(path))
    old = json.loads(Path(args.audit_protocol).read_text())
    require(old['source_official_version'] == 2 and old['source_sha256'] == AUDIT_MODULE_SHA256
            and old['EDF_header_helper_sha256'] == EDF_HELPER_SHA256, 'Retained original EDF audit proof differs')


def prepare(args):
    out, raw_root = Path(args.output).expanduser().absolute(), Path(args.raw_root).expanduser().resolve()
    require(not out.exists(), 'Use a new immutable InMID corpus directory')
    verify_input_seals(args)
    base, bm, ba, bart = validate_base(args.base_corpus, args.expected_base_completion_sha256)
    ds, dp, dg, deny_inputs = read_denies(args.deny_json)
    ds.update(ba['excluded_source_sha256']); dp.update(ba['excluded_payload_sha256'])
    plan = plan_records(rows(args.source_ledger), rows(args.metadata_admission), ds, dp, dg)
    for p in plan:
        if p['status'].startswith('excluded'):
            ds.add(p['record']['source_sha256']); dp.update(payloads(p['record']))
    base_sources, base_payloads = set(ba['sources']), {r['eeg_payload_sha256'] for r in ba['sources'].values()}
    require(not base_sources & ds and not base_payloads & dp, 'New exclusions overlap retained base')
    possible_bytes = sum(r['record']['sample_count']*FS_OUT//NATIVE_HZ*14*4 for r in plan if r['status'] == 'candidate')
    require(args.reserve_gib >= 1 and shutil.disk_usage(raw_root).free >= args.reserve_gib*1024**3
            + possible_bytes + 512*1024**2 + sum(v['bytes'] for v in bart.values()), 'Insufficient fixed storage reserve')
    out.mkdir(parents=True); (out/'waveforms').mkdir()
    protocol = {'schema': 'xavier-smac-InMID-source-addition-v1', 'created_epoch': time.time(), 'code_sha256': sha(__file__),
                'source_ledger_sha256': LEDGER_SHA256, 'metadata_admission_sha256': METADATA_SHA256,
                'audit_protocol_sha256': AUDIT_PROTOCOL_SHA256, 'native_decoder_sha256': AUDIT_MODULE_SHA256,
                'EDF_header_helper_sha256': EDF_HELPER_SHA256, 'common_helpers_sha256': sha(common.__file__),
                'trainer_contract_sha256': sha(trainer.__file__), 'base_completion_sha256': args.expected_base_completion_sha256,
                'prior_deny_input_hashes': deny_inputs, 'source': 'Mendeley rjx76wd5v6 v2', 'catalog_id': '001085',
                'source_files_in_distribution': 237, 'source_EDF_metadata_records': 92,
                'non_EEG_files_required_for_preparation': False, 'train_records_frozen': 76, 'test_records_frozen': 16,
                'newsource_val_or_test_signals_opened': 0, 'fixed_test_groups': sorted(TEST_GROUPS),
                'validation_domain': 'retained 806023 only; no InMID generalization estimate',
                'QC': {'native_max_abs_V': .1, 'postprocess_max_abs_V': PEAK_V, 'minimum_channel_std_V': MIN_STD_V,
                       'fixed_before_current_signal_reads': True, 'unit_correction_or_clipping': False},
                'preprocessing': {'native_hz': NATIVE_HZ, 'output_hz': FS_OUT, 'EEG_channels': CHANNELS,
                    'excluded_auxiliary_channels': 44, 'scipy_version': scipy.__version__, 'bandpass_design_N': 4,
                    'bandpass_final_order': 8, 'bandpass_Hz': [3, 40], 'sos_coefficients': filter_sos().tolist(),
                    'initial_state': 'zero separately per original recording', 'filter_method': 'forward-only sosfilt',
                    'additional_LP80_used': False, 'upsampling': 'z[j]=(1-a)*y[n-1]+a*y[n]; n=floor(128*j/200); a=(128*j mod 200)/200',
                    'represented_signal_delay_seconds': 1/NATIVE_HZ, 'warmup_seconds': 2,
                    'nonoverlap_window_seconds': PATCHES, 'output_shape': '[N,14,3,200]', 'output_unit': 'V_x_1e4'},
                'design_rationale': 'Native Nyquist is 64Hz, so an 80Hz native cutoff is invalid. This stage increases rate 128->200, with no decimation anti-alias stage. Fixed 3-40Hz SOS and delayed convex linear interpolation use only already available native samples. Upsampling does not add bandwidth, observations, or independent EEG hours; interpolation attenuation and filter phase remain source-specific.',
                'design_references': ['https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html',
                                      'https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html'],
                'physical_payload_identity': 'sample-major little-endian float64 calibrated V; audited integer block EEG hash also retained',
                'MAT_video_auxiliary_or_labels_independent_EEG_hours_added': 0, 'behavior_labels_used': False,
                'source_causality_verified': False, 'online_harness_allowed': False}
    dump(out/'protocol.json', protocol)
    arrays_meta, new_manifest, new_sources, certificates = {}, [], {}, []
    montage = hashlib.sha256(json.dumps([CHANNELS, REFERENCE, NATIVE_HZ, 'mendeley:rjx76wd5v6:v2', 'physical_V']).encode()).hexdigest()
    base_heldout_groups = {r['subject_group'] for r in bm if r['split'] != 'train'}
    for p in plan:
        r = p['record']
        cert = {k: v for k, v in p.items() if k != 'record'}
        cert.update(source_path=r['source_path'], source_sha256=r['source_sha256'], subject_group=r['subject_group'],
                    original_integer_EEG_payload_sha256=r['EEG_payload_sha256'], eeg_payload_sha256=r['decoded_EEG_V_sha256'],
                    signal_opened=False)
        if p['status'] != 'candidate':
            certificates.append(cert); continue
        if r['source_sha256'] in base_sources or payloads(r) & base_payloads or r['subject_group'] in base_heldout_groups:
            cert.update(status='excluded_base_identity', reason='already_represented_or_base_heldout_identity')
            certificates.append(cert); continue
        rel = Path(r['source_path'])
        require(not rel.is_absolute() and '..' not in rel.parts and rel.suffix.lower() == '.edf', 'Unsafe original EDF relative path')
        file = (raw_root/rel).resolve()
        require(raw_root in file.parents, 'Original EDF escaped designated release root')
        x, proof = verify_edf(file, r)
        cert.update(proof, signal_opened=True)
        try:
            array, windows, rejected = causal_windows14(x)
        except ValueError as error:
            cert.update(status='held_QC', reason=str(error)); certificates.append(cert); continue
        finally:
            del x
        cert.update(accepted_windows=len(windows), rejected_windows=rejected)
        if not windows:
            cert.update(status='held_QC', reason='no_window_meets_fixed_V_QC'); certificates.append(cert); continue
        relarray = 'waveforms/'+r['source_sha256']+'.npy'
        np.save(out/relarray, array, allow_pickle=False)
        arrays_meta[relarray] = {'shape': list(array.shape), 'dtype': 'float32', 'sha256': sha(out/relarray), 'bytes': (out/relarray).stat().st_size}
        new_sources[r['source_sha256']] = {'split': 'train', 'subject_group': r['subject_group'],
            'eeg_payload_sha256': r['decoded_EEG_V_sha256'], 'original_integer_EEG_payload_sha256': r['EEG_payload_sha256'],
            'montage_key': montage, 'physical_unit_verified': True, 'fit_allowed': True, 'validation_allowed': False,
            'online_harness_allowed': False, 'source_recording_id': 'mendeley:rjx76wd5v6:'+r['source_path'],
            'component_id_sha256': p['component_id_sha256']}
        for index, window in enumerate(windows):
            new_manifest.append({'id': r['source_sha256']+':'+str(index), 'split': 'train', 'subject_group': r['subject_group'],
                'source_sha256': r['source_sha256'], 'eeg_payload_sha256': r['decoded_EEG_V_sha256'],
                'source_recording_id': 'mendeley:rjx76wd5v6:'+r['source_path'], **window, 'channels': CHANNELS,
                'sample_rate_hz': FS_OUT, 'unit': 'V_x_1e4', 'eeg_path': relarray, 'eeg_index': index,
                'montage_key': montage, 'reference': REFERENCE, 'native_sample_rate_hz': NATIVE_HZ,
                'physical_unit_verified': True, 'source_causality': 'unverified_publisher_EDF_preprocessing',
                'online_harness_allowed': False})
        cert.update(status='admitted_offline_SSL', prepared_seconds=len(windows)*PATCHES)
        certificates.append(cert)
    dump(out/'source_certificates.json', certificates)
    if not new_manifest:
        result = {'status': 'blocked_no_eligible_new_source_windows', 'new_source_records': 0, 'new_source_windows': 0,
                  'new_source_unique_hours': 0, 'new_gradient_exposure_hours': 0, 'base_corpus_not_copied': True,
                  'test_records_not_opened': 16, 'reasons': dict(Counter(c['reason'] for c in certificates if c['status'] != 'admitted_offline_SSL')),
                  'protocol_sha256': sha(out/'protocol.json'), 'certificates_sha256': sha(out/'source_certificates.json')}
        dump(out/'preparation_result.json', result); return result
    for rel, spec in bart.items():
        target = out/'base'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base/rel, target)
        require(sha(target) == spec['sha256'], 'Retained base array copy hash mismatch')
        arrays_meta['base/'+rel] = spec
    manifest = [{**r, 'eeg_path': 'base/'+r['eeg_path']} for r in bm]+new_manifest
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False)+'\n' for r in manifest))
    dump(out/'artifacts.json', arrays_meta)
    dump(out/'source_admission.json', {'schema': 'xavier-smac-InMID-offline-SSL-admission-v1', 'sources': {**ba['sources'], **new_sources},
         'excluded_source_sha256': sorted(ds), 'excluded_payload_sha256': sorted(dp), 'protocol_sha256': sha(out/'protocol.json'),
         'online_harness_allowed': False, 'validation_domain': '806023 only; no InMID generalization claim'})
    trainer.load_corpus(out/'manifest.jsonl', out/'artifacts.json', out/'source_admission.json')
    result = {'status': 'prepared_verified_new_offline_SSL_corpus', 'windows': dict(Counter(r['split'] for r in manifest)),
              'new_source_records': len(new_sources), 'new_source_windows': len(new_manifest),
              'new_source_unique_hours': len(new_manifest)*PATCHES/3600, 'new_gradient_exposure_hours': 0,
              'retained_base_windows': len(bm), 'test_records_not_opened': 16, 'newsource_val_records_opened': 0,
              'validation_domain': '806023 only', 'InMID_generalization_evaluated': False,
              'MAT_video_auxiliary_or_labels_independent_EEG_hours_added': 0, 'online_harness_allowed': False,
              'base_completion_sha256': args.expected_base_completion_sha256,
              'protocol_sha256': sha(out/'protocol.json'), 'manifest_sha256': sha(out/'manifest.jsonl'),
              'artifacts_sha256': sha(out/'artifacts.json'), 'source_admission_sha256': sha(out/'source_admission.json'),
              'certificates_sha256': sha(out/'source_certificates.json')}
    dump(out/'completion.json', result); return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('raw-root', 'source-ledger', 'metadata-admission', 'audit-protocol', 'base-corpus', 'expected-base-completion-sha256', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--deny-json', action='append', required=True)
    p.add_argument('--reserve-gib', type=float, default=30)
    args = p.parse_args()
    from smac_storage_guard import require_smac_raw_root
    evidence = require_smac_raw_root(args.raw_root)
    node_root = (Path.home()/'pj-xavier'/evidence['node_id']).resolve()
    require(node_root in Path(args.output).expanduser().resolve().parents, 'Output must remain on the current S Mac node')
    require(node_root in Path(args.base_corpus).expanduser().resolve().parents, 'Base must be present on the current S Mac node')
    result = prepare(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['status'] == 'prepared_verified_new_offline_SSL_corpus' else 2


if __name__ == '__main__':
    raise SystemExit(main())
