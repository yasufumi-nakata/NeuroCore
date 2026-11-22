#!/usr/bin/env python3
"""Add sealed Figshare2061654 train CNT signals to an existing smac SSL corpus.

Only S1/S3 (14 records) are eligible. S2 test and S4 validation members are
never decompressed. The existing 806023 validation remains unchanged, so it
does not estimate CNT generalization. No event code, EOG, or behavioral label
is a model input. The original negative sample-count fields are not repaired.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import time
import warnings
import zipfile

import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'acquisition'))
sys.path.insert(0, str(HERE.parent / 'model'))
import audit_cnt_mi_zip as cnt
import cbramod_ssl_smac as trainer
import prepare_smac_fif as common
from prepare_smac_fif import FS_OUT, PATCHES, PEAK_V, MIN_STD_V, dump, read_denies, require, rows, sha

ARCHIVE_SHA256 = '1cf9c47c55c18f3bc272b86e1849ebfb7a2dca08159409de67b53a8a9263c41b'
ARCHIVE_BYTES = 156181084
LEDGER_SHA256 = 'c8b8ad27c34c072cc26bf371a61282c646ddf39977677de503fc41646bbcfa22'
METADATA_SHA256 = '555b3247b6a9eadcf48bed22f29ef381b06a601e87909241aedcc60900c70e1d'
FORMAT_PROOF_SHA256 = 'fdf6c962529055b5286b90489c663b5a6c9d9e247b2d160381d3114a7ade71e5'
LAYOUT_MODULE_SHA256 = '04f8ba3053dc940ea2ed1f04df6496b2484f500c131bfdc5d23ee22963efa904'
CHANNELS = list(cnt.EEG)
REFERENCE = 'publisher_CNT_reference_unspecified_no_rereference'
FIXED_SPLIT = {'figshare:2061654:S1': 'train', 'figshare:2061654:S2': 'test',
               'figshare:2061654:S3': 'train', 'figshare:2061654:S4': 'val'}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def plan_records(ledger, metadata, denied_sources, denied_payloads, denied_groups=()):
    require(len(ledger) == 26 and len({r['source_member'] for r in ledger}) == 26,
            'All 26 unique original CNT records are required')
    selected = [r for r in metadata if r.get('source') == 'cnt14']
    require(len(selected) == 26, 'All 26 frozen CNT metadata split rows are required')
    by_member = {r['source_member_or_path']: r for r in selected}
    require(len(by_member) == 26, 'Duplicate metadata member identity')
    plan, raw_identities, payloads, groups = [], set(), {}, Counter()
    for record in sorted(ledger, key=lambda r: r['source_member']):
        name, group = record['source_member'], record['subject_group']
        require(group in FIXED_SPLIT, 'Unknown CNT subject group')
        m = by_member.get(name)
        require(m is not None and m['source_group'] == 'cnt14|' + group, 'CNT metadata group/member mismatch')
        require(m['source_file_sha256'] == record['source_file_sha256']
                and m['eeg_payload_sha256'] == record['EEG_payload_sha256']
                and m['calibration_sha256'] == record['EEG_calibration_sha256'], 'CNT metadata source/payload/calibration mismatch')
        split = m['metadata_only_proposed_split']
        require(split == FIXED_SPLIT[group], 'Frozen CNT subject split changed')
        require(record['EEG_channels'] == CHANNELS and record['EOG_channels'] == cnt.EOG
                and record['n_channels'] == 16 and record['sample_rate_hz'] == 250,
                'Expected the audited 14 EEG plus 2 EOG at 250 Hz')
        require(record['scalar_bytes'] == 2 and record['original_is32bit_tag'] == 0
                and record['channel_offset'] == 1 and record['original_header_or_event_changes'] == 0,
                'Unreviewed original CNT layout or repairs')
        require(record['independent_MNE_header_and_three_calibrated_sample_probes_match'] is True
                and record['maximum_probe_difference_V'] <= 1e-12,
                'Prior independent physical-V calibration is unverified')
        source = record['source_file_sha256']
        require(source not in raw_identities, 'Duplicate CNT whole-file identity')
        raw_identities.add(source)
        identity = (record['EEG_payload_sha256'], record['EEG_calibration_sha256'])
        require(identity not in payloads, 'CNT duplicate requires new must-link review')
        payloads[identity] = split
        groups[group] += 1
        if split != 'train':
            status, reason = 'excluded_frozen_holdout', 'new_CNT_val_test_members_not_decompressed'
        elif source in denied_sources or record['EEG_payload_sha256'] in denied_payloads or group in denied_groups:
            status, reason = 'excluded_prior_identity', 'prior_raw_payload_or_group_deny'
        else:
            status, reason = 'candidate', 'requires_current_raw_calibration_and_fixed_QC'
        plan.append({'record': record, 'split': split, 'status': status, 'reason': reason,
                     'component_id_sha256': m['component_id_sha256']})
    require(dict(groups) == {'figshare:2061654:S1': 8, 'figshare:2061654:S2': 6,
                             'figshare:2061654:S3': 6, 'figshare:2061654:S4': 6}, 'CNT subject record counts changed')
    return plan


def validate_base(path, expected_completion_sha):
    base = Path(path).expanduser().resolve()
    require(sha(base / 'completion.json') == expected_completion_sha, 'Base completion SHA changed')
    completion = json.loads((base / 'completion.json').read_text())
    require(completion['status'] == 'prepared_verified_new_offline_SSL_corpus', 'Base is not a verified SSL corpus')
    for name, key in [('manifest.jsonl', 'manifest_sha256'), ('artifacts.json', 'artifacts_sha256'),
                      ('source_admission.json', 'source_admission_sha256'), ('protocol.json', 'protocol_sha256')]:
        require(sha(base / name) == completion[key], 'Base metadata seal changed: ' + name)
    # The trainer checks every array byte, provenance slot, unit and partition.
    trainer.load_corpus(base / 'manifest.jsonl', base / 'artifacts.json', base / 'source_admission.json')
    manifest = rows(base / 'manifest.jsonl')
    allowed = ('zenodo:806023:', 'zenodo:2392979:')
    require(all(r['source_recording_id'].startswith(allowed) and r.get('online_harness_allowed') is False for r in manifest),
            'Base must be the sealed 806023 or 806023+2392979 offline corpus')
    require(all(r['source_recording_id'].startswith('zenodo:806023:') for r in manifest if r['split'] == 'val'),
            'Preserve the existing 806023 validation domain')
    return base, manifest, json.loads((base / 'source_admission.json').read_text()), json.loads((base / 'artifacts.json').read_text())


def verify_cnt(path, record, format_record):
    """Verify original bytes and calibrate only the 14 EEG signals, in volts."""
    import mne
    path = Path(path)
    require(path.stat().st_size == record['source_file_bytes'] and sha(path) == record['source_file_sha256'],
            'CNT original file bytes changed')
    header, _unused_events = cnt.layout(path)
    for key, value in header.items():
        require(value == record[key], 'CNT audited header/layout changed: ' + key)
    require(format_record['original_is32bit_tag'] == header['original_is32bit_tag']
            and format_record['format_tag_offset'] == header['format_tag_offset']
            and format_record['nextfile'] == header['original_nextfile'], 'CNT pinned end-tag proof differs')
    names = [c['name'] for c in header['channels']]
    indices = [names.index(name) for name in CHANNELS]
    calibration = [header['channels'][i] for i in indices]
    require(canonical(calibration) == record['EEG_calibration_sha256'], 'CNT calibration fingerprint changed')
    require(all(c['V_per_AD_unit'] > 0 and math.isfinite(c['V_per_AD_unit']) for c in calibration),
            'CNT volts per AD unit must be finite and positive')
    data = np.memmap(path, mode='r', dtype='<i2', offset=header['data_offset'],
                     shape=(header['actual_sample_count_from_byte_span'], 16))
    eeg_ad = np.ascontiguousarray(data[:, indices])
    require(hashlib.sha256(eeg_ad.tobytes()).hexdigest() == record['EEG_payload_sha256'],
            'CNT original integer EEG payload differs')
    baseline = np.asarray([c['baseline'] for c in calibration], dtype=np.float64)
    gain = np.asarray([c['V_per_AD_unit'] for c in calibration], dtype=np.float64)
    physical = (eeg_ad.astype(np.float64) - baseline) * gain
    del data, eeg_ad
    physical_sha = hashlib.sha256(np.asarray(physical, dtype='<f8', order='C').tobytes()).hexdigest()
    x = physical.T
    require(np.isfinite(x).all(), 'Nonfinite calibrated CNT signal')
    with warnings.catch_warnings(record=True) as captured:
        raw = mne.io.read_raw_cnt(path, eog=cnt.EOG, preload=False, data_format='int16',
                                 recompute_n_samples=True, verbose='ERROR')
        try:
            require(raw.n_times == x.shape[1] and raw.info['sfreq'] == 250 and raw.ch_names == names,
                    'Independent MNE CNT layout differs')
            require(raw.get_channel_types().count('eeg') == 14 and raw.get_channel_types().count('eog') == 2,
                    'Independent MNE EEG/EOG roles differ')
            error = 0.
            for start in sorted({0, max(0, raw.n_times // 2 - 128), max(0, raw.n_times - 256)}):
                stop = min(raw.n_times, start + 256)
                observed = raw.get_data(picks=CHANNELS, start=start, stop=stop)
                error = max(error, float(np.max(np.abs(observed - x[:, start:stop]))))
            require(error <= 1e-12, 'Independent MNE physical-V calibration differs')
        finally:
            raw.close()
    return x, {'decoded_EEG_V_sha256': physical_sha, 'original_integer_EEG_payload_sha256': record['EEG_payload_sha256'],
               'EEG_calibration_sha256': record['EEG_calibration_sha256'], 'MNE_calibration_max_error_V': error,
               'MNE_warning_categories': sorted({type(w.message).__name__ for w in captured}),
               'physical_unit_verified': True, 'verification_scope': 'original header gain/baseline to V plus independent MNE probes; not amplifier calibration',
               'EEG_channels': CHANNELS, 'EOG_channels_excluded': cnt.EOG, 'original_header_changes': 0,
               'event_code_mapping_used': False}


def causal_windows14(x):
    x = np.asarray(x, dtype=np.float64)
    require(x.ndim == 2 and x.shape[0] == 14 and x.shape[1] > 0, 'Expected [14,N] calibrated CNT EEG volts')
    require(np.isfinite(x).all() and float(np.max(np.abs(x))) <= .1,
            'CNT raw physical-V magnitude/nonfinite gate failed; no unit correction')
    y = signal.sosfilt(signal.butter(4, (3, 40), btype='bandpass', fs=250, output='sos'), x, axis=1)
    y = signal.sosfilt(signal.butter(6, 80, btype='lowpass', fs=250, output='sos'), y, axis=1)
    js = np.arange(2 * FS_OUT, ((x.shape[1] - 1) * FS_OUT) // 250 + 1, dtype=np.int64)
    cutoffs = (js * 250) // FS_OUT
    fraction = ((js * 250) % FS_OUT).astype(np.float64) / FS_OUT
    y = y[:, cutoffs - 1] * (1 - fraction) + y[:, cutoffs] * fraction
    accepted, windows, rejected = [], [], Counter()
    width = PATCHES * FS_OUT
    for start in range(0, len(js) - width + 1, width):
        window = y[:, start:start + width]
        if not np.isfinite(window).all():
            rejected['nonfinite'] += 1
        elif np.max(np.abs(window)) > PEAK_V:
            rejected['peak_above_fixed_100uV'] += 1
        elif np.any(window.std(axis=1, ddof=0) < MIN_STD_V):
            rejected['flat_below_fixed_0_01uV'] += 1
        else:
            accepted.append((window * 1e4).reshape(14, PATCHES, FS_OUT).astype(np.float32))
            windows.append({'window_start_s': int(js[start]) / FS_OUT,
                            'window_end_s': (int(js[start + width - 1]) + 1) / FS_OUT,
                            'latest_native_source_index': int(cutoffs[start + width - 1]),
                            'feature_available_native_s': int(cutoffs[start + width - 1]) / 250})
    array = np.stack(accepted) if accepted else np.empty((0, 14, PATCHES, FS_OUT), np.float32)
    return array, windows, dict(rejected)


def prepare(args):
    out, archive = Path(args.output).expanduser().absolute(), Path(args.archive).expanduser().resolve()
    require(not out.exists(), 'Use a new immutable CNT corpus directory')
    for path, expected in [(args.source_ledger, LEDGER_SHA256), (args.metadata_admission, METADATA_SHA256),
                           (args.format_proof, FORMAT_PROOF_SHA256), (cnt.__file__, LAYOUT_MODULE_SHA256)]:
        require(sha(path) == expected, 'CNT audited input or parser seal changed: ' + str(path))
    require(archive.stat().st_size == ARCHIVE_BYTES and sha(archive) == ARCHIVE_SHA256, 'CNT source ZIP bytes differ')
    archive_stat = archive.stat()
    base, base_manifest, base_admission, base_artifacts = validate_base(args.base_corpus, args.expected_base_completion_sha256)
    denied_sources, denied_payloads, denied_groups, deny_inputs = read_denies(args.deny_json)
    denied_sources.update(base_admission['excluded_source_sha256'])
    denied_payloads.update(base_admission['excluded_payload_sha256'])
    ledger, metadata = rows(args.source_ledger), rows(args.metadata_admission)
    plan = plan_records(ledger, metadata, denied_sources, denied_payloads, denied_groups)
    proof = json.loads(Path(args.format_proof).read_text())
    require(proof['official_reader_source_sha256'] == cnt.READER_SHA and proof['nextfile_equals_event_table_end_all'] is True
            and proof['source_header_repair_performed'] is False, 'Primary CNT format proof is incomplete')
    format_records = {r['member']: r for r in proof['records']}
    require(set(format_records) == {r['source_member'] for r in ledger}, 'End-tag proof does not cover all CNT files')
    for p in plan:
        if p['status'] != 'candidate':
            denied_sources.add(p['record']['source_file_sha256'])
            denied_payloads.add(p['record']['EEG_payload_sha256'])
    base_sources = set(base_admission['sources'])
    base_payloads = {r['eeg_payload_sha256'] for r in base_admission['sources'].values()}
    require(not base_sources & denied_sources and not base_payloads & denied_payloads, 'CNT exclusions overlap retained base')
    need = 512 * 1024**2 + sum(s['bytes'] for s in base_artifacts.values())
    require(args.reserve_gib >= 1 and shutil.disk_usage(archive.parent).free >= args.reserve_gib * 1024**3 + need,
            'Insufficient reserve for bounded CNT preparation and exact base copy')
    out.mkdir(parents=True); (out / 'waveforms').mkdir()
    protocol = {'schema': 'xavier-smac-CNT2061654-source-addition-v1', 'created_epoch': time.time(),
                'code_sha256': sha(__file__), 'source_archive_sha256': ARCHIVE_SHA256, 'source_archive_bytes': ARCHIVE_BYTES,
                'source_ledger_sha256': LEDGER_SHA256, 'metadata_admission_sha256': METADATA_SHA256,
                'format_proof_sha256': FORMAT_PROOF_SHA256, 'layout_source_sha256': LAYOUT_MODULE_SHA256,
                'helpers_sha256': sha(common.__file__), 'trainer_contract_sha256': sha(trainer.__file__),
                'base_completion_sha256': args.expected_base_completion_sha256, 'prior_deny_input_hashes': deny_inputs,
                'fixed_split': FIXED_SPLIT, 'train_candidate_records_before_prior_denies': 14,
                'new_val_members_decompressed': 0, 'new_test_members_decompressed': 0,
                'validation_domain': 'retained 806023 only; no CNT generalization estimate',
                'QC': {'native_max_abs_V': .1, 'postprocess_max_abs_V': PEAK_V, 'minimum_channel_std_V': MIN_STD_V,
                       'fixed_before_current_signal_reads': True, 'unit_correction_or_clipping': False},
                'preprocessing': {'channels': CHANNELS, 'excluded_EOG': cnt.EOG, 'native_hz': 250, 'output_hz': FS_OUT,
                                  'bandpass': 'causal Butterworth4 SOS 3-40Hz, zero initial state',
                                  'anti_alias': 'causal Butterworth6 SOS 80Hz, zero initial state',
                                  'resampling': 'one native sample delayed linear interpolation',
                                  'warmup_seconds': 2, 'nonoverlap_window_seconds': PATCHES, 'output_unit': 'V_x_1e4'},
                'payload_identity': 'sample-major little-endian float64 physical V; original integer EEG hash separately verified',
                'source_causality_verified': False, 'online_harness_allowed': False,
                'original_sample_count_repaired': False, 'event_code_mapping_used': False, 'behavior_labels_used': False,
                'temporary_member_limit_bytes': 32 * 1024**2}
    dump(out / 'protocol.json', protocol)
    manifest, sources, artifacts, certificates = [], {}, {}, []
    new_payloads = set()
    montage = canonical([CHANNELS, REFERENCE, 250, 'figshare:2061654:v1', 'physical_V'])
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(dir=out, prefix='one_CNT_') as work:
        members = {i.filename: i for i in z.infolist() if not i.is_dir()}
        require(len(members) == 26 and set(members) == {r['source_member'] for r in ledger}, 'Unexpected CNT archive members')
        for p in plan:
            r = p['record']; name = r['source_member']
            cert = {k: v for k, v in p.items() if k != 'record'}
            cert.update(source_member=name, source_sha256=r['source_file_sha256'], subject_group=r['subject_group'],
                        original_integer_EEG_payload_sha256=r['EEG_payload_sha256'], signal_opened=False)
            if p['status'] != 'candidate':
                certificates.append(cert); continue
            info, relative = members[name], PurePosixPath(name)
            require(not relative.is_absolute() and '..' not in relative.parts and not stat.S_ISLNK(info.external_attr >> 16)
                    and info.file_size == r['source_file_bytes'] and 0 < info.file_size <= 32 * 1024**2,
                    'Unsafe or changed CNT archive member')
            file = Path(work) / 'current.cnt'
            with z.open(info) as src, file.open('xb') as dst:
                shutil.copyfileobj(src, dst, 1024**2)
            x, verified = verify_cnt(file, r, format_records[name])
            cert.update(verified, signal_opened=True, ZIP_CRC_verified=True)
            payload = verified['decoded_EEG_V_sha256']
            if payload in denied_payloads or payload in base_payloads or payload in new_payloads or r['source_file_sha256'] in base_sources:
                cert.update(status='excluded_identity', reason='heldout_or_duplicate_calibrated_payload')
                certificates.append(cert); del x; file.unlink(); continue
            try:
                array, windows, rejected = causal_windows14(x)
            except ValueError as error:
                cert.update(status='held_QC', reason=str(error)); certificates.append(cert); file.unlink(); continue
            finally:
                del x
            file.unlink()
            cert.update(accepted_windows=len(windows), rejected_windows=rejected)
            if not windows:
                cert.update(status='held_QC', reason='no_window_meets_fixed_V_QC'); certificates.append(cert); continue
            new_payloads.add(payload)
            rel = 'waveforms/' + r['source_file_sha256'] + '.npy'
            np.save(out / rel, array, allow_pickle=False)
            artifacts[rel] = {'shape': list(array.shape), 'dtype': 'float32', 'sha256': sha(out / rel), 'bytes': (out / rel).stat().st_size}
            sources[r['source_file_sha256']] = {'split': 'train', 'subject_group': r['subject_group'], 'eeg_payload_sha256': payload,
                'montage_key': montage, 'physical_unit_verified': True, 'fit_allowed': True, 'validation_allowed': False,
                'source_recording_id': 'figshare:2061654:' + name, 'online_harness_allowed': False,
                'original_integer_EEG_payload_sha256': r['EEG_payload_sha256'], 'EEG_calibration_sha256': r['EEG_calibration_sha256']}
            for index, window in enumerate(windows):
                manifest.append({'id': r['source_file_sha256'] + ':' + str(index), 'split': 'train',
                    'subject_group': r['subject_group'], 'source_sha256': r['source_file_sha256'], 'eeg_payload_sha256': payload,
                    'source_recording_id': 'figshare:2061654:' + name, **window, 'channels': CHANNELS,
                    'sample_rate_hz': FS_OUT, 'unit': 'V_x_1e4', 'eeg_path': rel, 'eeg_index': index,
                    'montage_key': montage, 'reference': REFERENCE, 'native_sample_rate_hz': 250, 'physical_unit_verified': True,
                    'source_causality': 'original_CNT_acquisition_causality_not_certified', 'online_harness_allowed': False})
            cert.update(status='admitted_offline_SSL', prepared_seconds=len(windows) * PATCHES)
            certificates.append(cert)
    require(sha(archive) == ARCHIVE_SHA256 and (archive.stat().st_ino, archive.stat().st_size, archive.stat().st_mtime_ns)
            == (archive_stat.st_ino, archive_stat.st_size, archive_stat.st_mtime_ns), 'CNT archive changed during preparation')
    dump(out / 'source_certificates.json', certificates)
    if not manifest:
        result = {'status': 'blocked_no_eligible_new_source_windows', 'new_source_windows': 0, 'new_source_records': 0,
                  'new_source_unique_hours': 0, 'new_gradient_exposure_hours': 0, 'base_corpus_not_copied': True,
                  'new_val_members_decompressed': 0, 'new_test_members_decompressed': 0,
                  'reasons': dict(Counter(c['reason'] for c in certificates if c['status'] in {'held_QC', 'excluded_identity'})),
                  'protocol_sha256': sha(out / 'protocol.json'), 'certificates_sha256': sha(out / 'source_certificates.json')}
        dump(out / 'preparation_result.json', result); return result
    new_count = len(manifest)
    for rel, spec in base_artifacts.items():
        target = out / 'base' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / rel, target)
        require(sha(target) == spec['sha256'], 'Base array copy changed')
        artifacts['base/' + rel] = spec
    manifest = [{**r, 'eeg_path': 'base/' + r['eeg_path']} for r in base_manifest] + manifest
    (out / 'manifest.jsonl').write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in manifest))
    dump(out / 'artifacts.json', artifacts)
    dump(out / 'source_admission.json', {'schema': 'xavier-smac-CNT-offline-SSL-admission-v1',
        'sources': {**base_admission['sources'], **sources}, 'excluded_source_sha256': sorted(denied_sources),
        'excluded_payload_sha256': sorted(denied_payloads), 'protocol_sha256': sha(out / 'protocol.json'),
        'online_harness_allowed': False, 'validation_domain': '806023 only; no CNT validation or generalization claim'})
    trainer.load_corpus(out / 'manifest.jsonl', out / 'artifacts.json', out / 'source_admission.json')
    result = {'status': 'prepared_verified_new_offline_SSL_corpus', 'windows': dict(Counter(r['split'] for r in manifest)),
              'new_source_windows': new_count, 'new_source_records': len(sources), 'new_source_unique_hours': new_count * PATCHES / 3600,
              'candidate_records_before_current_QC': sum(p['status'] == 'candidate' for p in plan),
              'retained_base_windows': len(base_manifest), 'validation_domain': '806023 only',
              'CNT_generalization_evaluated': False, 'new_val_members_decompressed': 0, 'new_test_members_decompressed': 0,
              'new_gradient_exposure_hours': 0, 'online_harness_allowed': False,
              'protocol_sha256': sha(out / 'protocol.json'), 'manifest_sha256': sha(out / 'manifest.jsonl'),
              'artifacts_sha256': sha(out / 'artifacts.json'), 'source_admission_sha256': sha(out / 'source_admission.json'),
              'certificates_sha256': sha(out / 'source_certificates.json'), 'base_completion_sha256': args.expected_base_completion_sha256}
    dump(out / 'completion.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('archive', 'source-ledger', 'metadata-admission', 'format-proof', 'base-corpus',
                'expected-base-completion-sha256', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--deny-json', action='append', required=True)
    p.add_argument('--reserve-gib', type=float, default=30)
    args = p.parse_args()
    from smac_storage_guard import require_smac_raw_root
    evidence = require_smac_raw_root(Path(args.archive).expanduser().resolve().parent)
    node_root = (Path.home() / 'pj-xavier' / evidence['node_id']).resolve()
    require(node_root in Path(args.output).expanduser().resolve().parents, 'Prepared output must remain in this smac node workspace')
    result = prepare(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    if result['status'].startswith('blocked'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
