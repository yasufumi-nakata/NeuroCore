#!/usr/bin/env python3
"""Add the frozen train-only Zenodo8089820 GDF source to offline smac SSL.

Raw scalar identity and calibrated physical-V identity remain distinct. Every
train candidate is checked; metadata-only validation/test/review members are
never decompressed. Event timing, cue labels and amplifier calibration are not
certified by this representation-learning admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import warnings
import zipfile

import mne
import numpy as np
from scipy import signal

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE.parent / 'acquisition'))
sys.path.insert(0, str(HERE.parent / 'model'))
import audit_bci_gdf_zip_v2 as original
import cbramod_ssl_smac as trainer
import freeze_smac_bci8089820_metadata as frozen
import preflight_smac_bci8089820_zip as preflight
from prepare_smac_fif import require, dump, rows, sha

SOURCE = 'zenodo:8089820'
NATIVE_FS, FS_OUT, PATCHES = 512, 200, 3
CHANNELS = original.EEG_NAMES
PEAK_V, MIN_STD_V, NATIVE_PEAK_V = 1e-4, 1e-8, .1
MAX_MEMBER_BYTES = 128 * 1024**2
EXPECTED_FOLDS = {'train': 446, 'val': 96, 'test': 96, 'review': 56}
REFERENCE = 'publisher_GDF_reference_unspecified_no_rereference'
IDENTITY_SCOPE = ('Known exact raw bytes, original scalar bytes and ordered sample-major physical-V identities; '
                  'unknown heldout payloads, transformed equivalents and undocumented cross-release participant identity remain unverified')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class PhysicalCalibrationHold(ValueError):
    def __init__(self, reason, evidence):
        super().__init__(reason)
        self.reason, self.evidence = reason, evidence


def load_metadata(frozen_path, recordings_path, constraints_path):
    require(sha(frozen_path) == preflight.FROZEN_MANIFEST_SHA, 'Frozen source split changed')
    require(sha(recordings_path) == frozen.RECORDINGS_SHA, 'Original recording ledger changed')
    require(sha(constraints_path) == frozen.CONSTRAINTS_SHA, 'Original duplicate/review constraints changed')
    plan = json.loads(Path(frozen_path).read_text())
    constraint = json.loads(Path(constraints_path).read_text())
    expected = frozen.make_plan(constraint['record_metadata'], constraint['must_link_groups'], constraint['author_reported_QC_flags'])
    require(frozen.canonical_json(expected) == Path(frozen_path).read_bytes(), 'Metadata split is not the frozen must-link/review plan')
    require(plan['summary']['offline_ssl_training_allowed_now'] is False, 'Metadata is a prerequisite, not existing fit approval')
    require(plan['summary']['records_by_split'] == EXPECTED_FOLDS, 'Frozen fold cardinality changed')
    records = {r['source_member']: r for r in rows(recordings_path)}
    require(len(records) == 694 and set(records) == {r['source_member'] for r in plan['members']}, 'Original GDF roster changed')
    for member in plan['members']:
        record = records[member['source_member']]
        for key, original_key in [('source_file_sha256', 'source_file_sha256'), ('eeg_payload_sha256', 'eeg_payload_sha256'),
                                  ('calibration_sha256', 'EEG_calibration_sha256'), ('subject_group', 'subject_group')]:
            require(member[key] == record[original_key], 'Frozen record identity differs: ' + key)
        require(record['native_sample_rate_hz'] == NATIVE_FS and record['eeg_channels'] == CHANNELS,
                'Unreviewed GDF native rate or ordered montage')
        require(record['annotation_audit']['event_sample_rate_hz'] == 0
                and record['annotation_audit']['explicit_event_timebase_verified'] is False,
                'Missing event timebase must remain uncertified')
    return plan, records


def validate_base(path, expected_completion):
    base = Path(path).expanduser().resolve()
    require(sha(base / 'completion.json') == expected_completion, 'Parent corpus completion changed')
    complete = json.loads((base / 'completion.json').read_text())
    require(complete['status'] == 'prepared_verified_new_offline_SSL_corpus', 'Parent corpus is not complete')
    optional = {}
    for filename, key in [('manifest.jsonl', 'manifest_sha256'), ('artifacts.json', 'artifacts_sha256'),
                           ('source_admission.json', 'source_admission_sha256'), ('protocol.json', 'protocol_sha256'),
                           ('source_certificates.json', 'certificates_sha256'), ('split_manifest.json', 'split_manifest_sha256'),
                           ('parent_provenance.json', 'parent_provenance_sha256')]:
        if key in complete:
            require(sha(base / filename) == complete[key], 'Parent metadata seal changed: ' + filename)
            if filename in ('source_certificates.json', 'split_manifest.json', 'parent_provenance.json'):
                optional[filename] = json.loads((base / filename).read_text())
    trainer.load_corpus(base / 'manifest.jsonl', base / 'artifacts.json', base / 'source_admission.json')
    manifest = rows(base / 'manifest.jsonl')
    require(all(r.get('online_harness_allowed') is False for r in manifest), 'Parent must remain offline-only')
    require(all(r['source_recording_id'].startswith('zenodo:806023:') for r in manifest if r['split'] == 'val'),
            'Preserve only the existing 806023 validation domain')
    return (base, manifest, json.loads((base / 'source_admission.json').read_text()),
            json.loads((base / 'artifacts.json').read_text()), {'completion': complete, 'optional_metadata': optional})


def calibration_affine(header, indices):
    """Convert endpoints to float BEFORE subtraction and retain exact integers."""
    ix = indices['EEG']
    pmn = np.asarray([header['physical_min'][i] for i in ix], dtype=np.float64)
    pmx = np.asarray([header['physical_max'][i] for i in ix], dtype=np.float64)
    dmn = np.asarray([header['digital_min'][i] for i in ix], dtype=np.float64)
    dmx = np.asarray([header['digital_max'][i] for i in ix], dtype=np.float64)
    units = np.asarray([header['units'][i] for i in ix], dtype=np.float64)
    require(np.isfinite([pmn, pmx, dmn, dmx, units]).all() and np.all(dmx > dmn)
            and np.all(pmx > pmn) and np.all(units == 1e-6), 'Invalid/unverified GDF physical calibration')
    gain = (pmx - pmn) / (dmx - dmn)
    offset = pmn - dmn * gain
    require(np.isfinite(gain).all() and np.isfinite(offset).all() and np.all(gain > 0), 'Nonfinite or nonpositive calibration')
    exact_min = [int(header['digital_min'][i]) for i in ix]
    exact_max = [int(header['digital_max'][i]) for i in ix]
    ranges = [b - a for a, b in zip(exact_min, exact_max)]
    evidence = {'original_digital_min_decimal_strings': [str(x) for x in exact_min],
                'original_digital_max_decimal_strings': [str(x) for x in exact_max],
                'original_digital_range_decimal_strings': [str(x) for x in ranges],
                'any_integer_range_exceeds_signed_int64': any(x > np.iinfo(np.int64).max for x in ranges),
                'gain_V_per_original_scalar': (gain * units).tolist(),
                'offset_V': (offset * units).tolist(),
                'formula': '(raw_scalar * float64_gain + float64_offset) * header_units',
                'digital_endpoints_converted_before_subtraction': True}
    return gain, offset, units, evidence


def verify_gdf(path, record):
    """Whole stored EEG hash, all channel calibration, independent MNE probes."""
    from mne.io.edf.edf import _read_gdf_header
    path = Path(path)
    require(path.stat().st_size == record['source_file_bytes'] and sha(path) == record['source_file_sha256'],
            'Original GDF source bytes differ')
    with warnings.catch_warnings(record=True) as captured:
        header = _read_gdf_header(path, exclude=[])
    require(float(header['number']) == record['GDF_version'] == 1.25, 'Unreviewed GDF version')
    require(header['ch_names'] == record['all_channels'], 'Original GDF channel layout changed')
    indices = original.classify_channels(header['ch_names'])
    channels = [header['ch_names'][i] for i in indices['EEG']]
    require(channels == record['eeg_channels'] == CHANNELS, 'Ordered 27 EEG montage changed')
    ns = [int(x) for x in header['n_samps']]
    require(len(set(ns)) == 1 and ns[0] > 0 and int(header['nchan']) == 32, 'Unexpected source GDF record layout')
    count = int(header['n_records'])
    rate = ns[0] * int(header['record_length'][1]) / int(header['record_length'][0])
    require(rate == NATIVE_FS and count * ns[0] == record['samples_per_EEG_channel'], 'GDF original sample count/rate changed')
    require(int(header['data_offset']) == record['header_bytes'] and int(header['dtype_byte']) == 8
            and str(header['dtype_np']) == record['scalar_dtype'], 'GDF storage header/type changed')
    hashes, byte_counts, trailing = original.payload_hashes(path, header, indices)
    require(hashes['EEG'] == record['eeg_payload_sha256'], 'Original scalar EEG payload differs')
    require(byte_counts['EEG'] == record['EEG_sample_bytes'] and int(header['data_offset']) + sum(byte_counts.values()) + len(trailing) == path.stat().st_size,
            'Original scalar payload bytes are not exhaustive')
    del trailing
    calibration = {key: [float(header[key][i]) for i in indices['EEG']]
                   for key in ['physical_min', 'physical_max', 'digital_min', 'digital_max', 'units']}
    require(canonical(calibration) == record['EEG_calibration_sha256'], 'Original calibration fingerprint changed')
    gain, offset, units, proof = calibration_affine(header, indices)
    raw = np.memmap(path, dtype=header['dtype_np'], mode='r', offset=int(header['data_offset']),
                    shape=(count, int(header['nchan']), ns[0]))
    scalar = np.asarray(raw[:, indices['EEG'], :].transpose(0, 2, 1).reshape(-1, len(CHANNELS)), dtype=np.float64)
    # Subtracting digital_min from tiny float scalars first would lose signal.
    physical = (scalar * gain + offset) * units
    del scalar, raw
    proof.update({'original_EEG_scalar_payload_sha256': hashes['EEG'], 'EEG_calibration_sha256': record['EEG_calibration_sha256'],
                  'EEG_channels': channels, 'excluded_EOG_channels': record['eog_channels'], 'excluded_EMG_channels': record['emg_channels'],
                  'native_samples': count * ns[0], 'native_sample_rate_hz': rate, 'event_timebase_verified': False,
                  'behavior_labels_or_event_codes_used': False, 'original_GDF_modified': False})
    if not np.isfinite(physical).all():
        raise PhysicalCalibrationHold('held_nonfinite_independent_physical_calibration', proof)
    physical_sha = hashlib.sha256(np.asarray(physical, dtype='<f8', order='C').tobytes()).hexdigest()
    proof['decoded_EEG_V_sha256'] = physical_sha
    probe = None
    try:
        with warnings.catch_warnings(record=True) as reader_warnings:
            probe = mne.io.read_raw_gdf(path, eog=record['eog_channels'], misc=record['emg_channels'], preload=False, verbose='ERROR')
            require(probe.n_times == physical.shape[0] and probe.info['sfreq'] == NATIVE_FS, 'Independent MNE layout differs')
            error, scale = 0., 0.
            for start in sorted({0, max(0, probe.n_times // 2 - 128), max(0, probe.n_times - 256)}):
                stop = min(probe.n_times, start + 256)
                observed = probe.get_data(picks=CHANNELS, start=start, stop=stop, verbose='ERROR')
                expected = physical[start:stop].T
                if not np.isfinite(observed).all():
                    raise PhysicalCalibrationHold('held_nonfinite_MNE_calibration_probe', proof)
                error = max(error, float(np.max(np.abs(observed - expected))))
                scale = max(scale, float(np.max(np.abs(expected))))
            proof['MNE_all_channel_three_interval_max_error_V'] = error
            proof['MNE_probe_tolerance_V'] = 1e-11 + 1e-7 * scale
            proof['MNE_warning_categories'] = sorted({type(w.message).__name__ for w in list(captured) + list(reader_warnings)})
            if error > proof['MNE_probe_tolerance_V']:
                raise PhysicalCalibrationHold('held_physical_calibration_MNE_disagreement', proof)
    finally:
        if probe is not None:
            probe.close()
    proof['physical_unit_verified'] = True
    proof['observed_native_peak_V'] = float(np.max(np.abs(physical)))
    proof['verification_scope'] = 'Pinned original stored EEG/calibration plus independent all-channel MNE probes; amplifier calibration unverified'
    return physical.T, proof


def causal_windows(x):
    require(x.ndim == 2 and x.shape[0] == len(CHANNELS) and np.isfinite(x).all(), 'Expected finite [27,N] physical-V EEG')
    y = signal.sosfilt(signal.butter(4, (3, 40), btype='bandpass', fs=NATIVE_FS, output='sos'), x, axis=1)
    js = np.arange(2 * FS_OUT, ((x.shape[1] - 1) * FS_OUT) // NATIVE_FS, dtype=np.int64)
    cutoffs = (js * NATIVE_FS) // FS_OUT
    fraction = ((js * NATIVE_FS) % FS_OUT).astype(np.float64) / FS_OUT
    y = y[:, cutoffs - 1] * (1 - fraction) + y[:, cutoffs] * fraction
    arrays, intervals, rejected = [], [], Counter()
    width = FS_OUT * PATCHES
    for start in range(0, len(js) - width + 1, width):
        value = y[:, start:start + width]
        if not np.isfinite(value).all(): rejected['nonfinite'] += 1
        elif np.max(np.abs(value)) > PEAK_V: rejected['peak_above_fixed_100uV'] += 1
        elif np.any(value.std(axis=1, ddof=0) < MIN_STD_V): rejected['flat_below_fixed_0_01uV'] += 1
        else:
            arrays.append((value * 1e4).reshape(len(CHANNELS), PATCHES, FS_OUT).astype(np.float32))
            intervals.append({'window_start_s': int(js[start]) / FS_OUT, 'window_end_s': (int(js[start + width - 1]) + 1) / FS_OUT,
                              'latest_native_source_index': int(cutoffs[start + width - 1]),
                              'feature_available_native_s': int(cutoffs[start + width - 1]) / NATIVE_FS})
    array = np.stack(arrays) if arrays else np.empty((0, len(CHANNELS), PATCHES, FS_OUT), np.float32)
    return array, intervals, dict(rejected)


def prepare_sources(archive, out, plan, records, base_admission, deadline, reserve_bytes):
    """Single temporary train member, full source checks, explicit holds."""
    denied_source = set(base_admission['excluded_source_sha256'])
    denied_payload = set(base_admission['excluded_payload_sha256'])
    for member in plan['members']:
        if member['split'] != 'train':
            denied_source.add(member['source_file_sha256']); denied_payload.add(member['eeg_payload_sha256'])
    old_source = set(base_admission['sources'])
    old_payload = {v['eeg_payload_sha256'] for v in base_admission['sources'].values()}
    added, admitted, artifacts, certificates = [], {}, {}, []
    seen_source, seen_scalar, seen_physical = {}, {}, {}
    montage = canonical([CHANNELS, REFERENCE, NATIVE_FS, SOURCE])
    (out / 'waveforms').mkdir()
    with zipfile.ZipFile(archive, 'r') as zf, tempfile.TemporaryDirectory(prefix='temporary_train_GDF_', dir=out) as temporary:
        names = {x.filename: x for x in zf.infolist()}
        for member in sorted(plan['members'], key=lambda r: r['source_member']):
            require(time.time() < deadline, 'New preparation time budget expired')
            record = records[member['source_member']]
            cert = {'source_member': member['source_member'], 'source_sha256': member['source_file_sha256'],
                    'original_EEG_scalar_payload_sha256': member['eeg_payload_sha256'], 'EEG_calibration_sha256': member['calibration_sha256'],
                    'subject_group': member['subject_group'], 'split': member['split'], 'signal_opened': False}
            if member['split'] != 'train':
                cert['status'] = 'excluded_frozen_' + member['split']; certificates.append(cert); continue
            info = names[member['source_member']]
            require(not info.is_dir() and info.file_size == record['source_file_bytes'] <= MAX_MEMBER_BYTES,
                    'Train GDF size differs or exceeds fixed member cap')
            require((info.external_attr >> 16) & 0o170000 != 0o120000, 'GDF ZIP symlink disallowed')
            require(shutil.disk_usage(out).free > reserve_bytes + MAX_MEMBER_BYTES * 4, 'Preparation free-space reserve exhausted')
            target = Path(temporary) / 'current.gdf'
            digest = hashlib.sha256(); total = 0
            with zf.open(info) as src, target.open('xb') as dest:
                for block in iter(lambda: src.read(1024**2), b''):
                    digest.update(block); total += len(block); dest.write(block)
            require(total == info.file_size and digest.hexdigest() == record['source_file_sha256'], 'GDF extraction CRC/source SHA mismatch')
            cert.update(signal_opened=True, source_bytes=total, ZIP_CRC_verified=True)
            try:
                # All frozen train candidates are physically checked, including aliases.
                try:
                    x, proof = verify_gdf(target, record)
                    cert.update(proof)
                except PhysicalCalibrationHold as hold:
                    cert.update(hold.evidence, status=hold.reason); certificates.append(cert); continue
                physical = proof['decoded_EEG_V_sha256']
                scalar_identity = (member['eeg_payload_sha256'], member['calibration_sha256'], tuple(CHANNELS), NATIVE_FS, record['samples_per_EEG_channel'])
                if cert['source_sha256'] in denied_source or member['eeg_payload_sha256'] in denied_payload or physical in denied_payload:
                    cert['status'] = 'excluded_known_heldout_raw_scalar_or_physical_payload'
                elif cert['source_sha256'] in old_source or physical in old_payload:
                    cert['status'] = 'excluded_previously_admitted_source_or_physical_payload'
                elif cert['source_sha256'] in seen_source or scalar_identity in seen_scalar or physical in seen_physical:
                    cert.update(status='excluded_exact_duplicate_train_alias', same_as_source_sha256=seen_source.get(cert['source_sha256']) or seen_scalar.get(scalar_identity) or seen_physical.get(physical))
                elif proof['observed_native_peak_V'] > NATIVE_PEAK_V:
                    cert['status'] = 'held_native_peak_above_fixed_0_1V'
                else:
                    array, intervals, rejected = causal_windows(x)
                    cert.update(accepted_windows=len(intervals), rejected_windows=rejected)
                    if not intervals:
                        cert['status'] = 'held_no_window_meets_fixed_QC'
                    else:
                        source = cert['source_sha256']; rel = 'waveforms/' + source + '.npy'
                        np.save(out / rel, array, allow_pickle=False)
                        artifacts[rel] = {'shape': list(array.shape), 'dtype': 'float32', 'bytes': (out / rel).stat().st_size, 'sha256': sha(out / rel)}
                        admitted[source] = {'split': 'train', 'subject_group': member['subject_group'], 'eeg_payload_sha256': physical,
                                            'original_EEG_scalar_payload_sha256': member['eeg_payload_sha256'], 'montage_key': montage,
                                            'physical_unit_verified': True, 'fit_allowed': True, 'validation_allowed': False,
                                            'online_harness_allowed': False, 'source_event_timebase_verified': False,
                                            'supervised_cue_training_allowed': False, 'EEG_calibration_sha256': member['calibration_sha256']}
                        for index, interval in enumerate(intervals):
                            added.append({'id': source + ':' + str(index), 'source_sha256': source, 'eeg_payload_sha256': physical,
                                          'source_recording_id': SOURCE + ':' + member['source_member'], 'split': 'train', 'subject_group': member['subject_group'],
                                          'eeg_path': rel, 'eeg_index': index, **interval, 'channels': CHANNELS,
                                          'sample_rate_hz': FS_OUT, 'unit': 'V_x_1e4', 'native_sample_rate_hz': NATIVE_FS,
                                          'montage_key': montage, 'reference': REFERENCE, 'physical_unit_verified': True,
                                          'source_causality': 'unverified_original_GDF_acquisition_and_preprocessing', 'online_harness_allowed': False})
                        cert.update(status='admitted_offline_SSL', prepared_seconds=len(intervals) * PATCHES)
                        seen_source[source] = source; seen_scalar[scalar_identity] = source; seen_physical[physical] = source
                del x
                certificates.append(cert)
            finally:
                target.unlink()
                dump(out / 'progress.json', {'phase': 'train_only_physical_QC', 'updated_epoch': time.time(),
                     'processed_source_certificates': len(certificates), 'source_status_counts': dict(Counter(c['status'] for c in certificates)),
                     'accepted_unique_windows': len(added), 'gradient_steps': 0})
    return added, admitted, artifacts, certificates, denied_source, denied_payload


def prepare(args):
    from smac_storage_guard import require_smac_raw_root
    guard = require_smac_raw_root(args.raw_root)
    node = Path.home() / 'pj-xavier' / guard['node_id']
    require(guard['node_id'] != 'smac19', 'Excluded node')
    out, archive = Path(args.output).expanduser().absolute(), Path(args.archive).expanduser().absolute()
    require(not out.exists(), 'Use a new immutable GDF preparation directory')
    require(node in out.parents and node in archive.parents and node in Path(args.base_corpus).expanduser().absolute().parents,
            'Raw, base and output must be on the same authorized S Mac node')
    require(archive == node / 'raw' / preflight.SOURCE_KEY / 'BCI Database.zip', 'Use the existing acquired source; do not create a second download')
    require(0 < args.max_seconds <= 7200 and args.reserve_gib >= 100, 'Invalid fresh preparation budget/reserve')
    started = time.time(); deadline = started + args.max_seconds
    plan, record_map = load_metadata(args.frozen_manifest, args.recordings, args.constraints)
    base, base_rows, base_admission, base_artifacts, base_provenance = validate_base(args.base_corpus, args.expected_base_completion_sha256)
    require(not any(r['source_recording_id'].startswith(SOURCE + ':') for r in base_rows), 'Do not add an already trained GDF source again')
    source_preflight = preflight.run(node, Path(args.frozen_manifest))
    require(source_preflight['status'] == 'ready_for_train_only_physical_qc', 'Original ZIP is incomplete/unverified')
    source_stat = archive.stat()
    out.parent.mkdir(parents=True, exist_ok=True)
    estimated_arrays = sum(int(record_map[m['source_member']]['stored_signal_seconds'] * FS_OUT * len(CHANNELS) * 4)
                           for m in plan['members'] if m['split'] == 'train')
    require(shutil.disk_usage(out.parent).free > args.reserve_gib * 1024**3 + estimated_arrays + sum(a['bytes'] for a in base_artifacts.values()) + 2 * 1024**3,
            'Insufficient disk budget for cumulative corpus and recovery')
    out.mkdir()
    shutil.copyfile(args.frozen_manifest, out / 'split_manifest.json')
    dump(out / 'parent_provenance.json', base_provenance)
    protocol = {'schema': 'xavier-smac-BCI8089820-offline-source-addition-v1', 'created_epoch': started,
                'absolute_preparation_deadline_epoch': deadline, 'max_preparation_seconds': args.max_seconds,
                'source_archive_sha256': preflight.PINNED_SHA, 'source_archive_bytes': preflight.PINNED_BYTES,
                'frozen_split_sha256': preflight.FROZEN_MANIFEST_SHA, 'source_ledger_sha256': frozen.RECORDINGS_SHA,
                'source_constraints_sha256': frozen.CONSTRAINTS_SHA, 'base_completion_sha256': args.expected_base_completion_sha256,
                'preparation_sha256': sha(__file__), 'audit_helper_sha256': sha(original.__file__), 'trainer_sha256': sha(trainer.__file__),
                'metadata_freeze_helper_sha256': sha(frozen.__file__), 'preflight_helper_sha256': sha(preflight.__file__),
                'source_preflight': source_preflight, 'source_fold_counts': EXPECTED_FOLDS, 'cross_release_identity_scope': IDENTITY_SCOPE,
                'all_frozen_train_candidates_to_be_checked': EXPECTED_FOLDS['train'], 'new_val_test_review_signals_decoded': 0,
                'mne_version': mne.__version__, 'numpy_version': np.__version__, 'reference': REFERENCE,
                'preprocessing': {'channels': CHANNELS, 'excluded_EOG': sorted(original.AUX['EOG']), 'excluded_EMG': sorted(original.AUX['EMG']),
                    'native_sample_rate_hz': NATIVE_FS, 'output_sample_rate_hz': FS_OUT,
                    'bandpass': 'forward-only Butterworth4 SOS 3-40Hz with zero initial state per original recording',
                    'sos_coefficients': signal.butter(4, (3, 40), btype='bandpass', fs=NATIVE_FS, output='sos').tolist(),
                    'additional_LP80_used': False, 'interpolation': 'one native sample delayed convex linear interpolation; no future sample',
                    'warmup_seconds': 2, 'nonoverlap_window_seconds': PATCHES, 'output_unit': 'V_x_1e4',
                    'physical_conversion': 'Stable affine digital-to-physical conversion, float endpoint differences before division'},
                'QC': {'fixed_before_current_signal_reads': True, 'native_max_abs_V': NATIVE_PEAK_V, 'postprocess_max_abs_V': PEAK_V,
                       'minimum_channel_std_V': MIN_STD_V, 'unit_correction_or_clipping': False},
                'event_timebase_verified': False, 'behavior_labels_used': False, 'online_harness_allowed': False,
                'validation_domain': 'Retained 806023 only; no new GDF generalization estimate'}
    dump(out / 'protocol.json', protocol)
    added, admitted, artifacts, certificates, denied_source, denied_payload = prepare_sources(
        archive, out, plan, record_map, base_admission, deadline, args.reserve_gib * 1024**3)
    dump(out / 'source_certificates.json', certificates)
    require(Counter(c['split'] for c in certificates) == EXPECTED_FOLDS, 'Not every frozen source was accounted for')
    require(sum(c['signal_opened'] for c in certificates) == EXPECTED_FOLDS['train'] and all(not c['signal_opened'] for c in certificates if c['split'] != 'train'),
            'Train-only signal access contract changed')
    current = archive.stat()
    require((source_stat.st_size, source_stat.st_mtime_ns, source_stat.st_ino) == (current.st_size, current.st_mtime_ns, current.st_ino),
            'Pinned source archive changed during preparation')
    result = {'status': 'prepared_verified_new_offline_SSL_corpus' if added else 'blocked_no_eligible_new_source_windows',
              'all_frozen_train_records_checked': EXPECTED_FOLDS['train'], 'new_source_records': len(admitted), 'new_source_windows': len(added),
              'new_source_unique_seconds': len(added) * PATCHES, 'new_source_unique_hours': len(added) * PATCHES / 3600,
              'new_gradient_exposure_hours': 0, 'source_status_counts': dict(Counter(c['status'] for c in certificates)),
              'new_val_test_review_signals_decoded': 0, 'event_timebase_verified': False, 'behavior_labels_used': False,
              'online_harness_allowed': False, 'new_source_generalization_evaluated': False,
              'base_completion_sha256': args.expected_base_completion_sha256, 'cross_release_identity_scope': IDENTITY_SCOPE,
              'split_manifest_sha256': sha(out / 'split_manifest.json'), 'certificates_sha256': sha(out / 'source_certificates.json'),
              'parent_provenance_sha256': sha(out / 'parent_provenance.json'), 'protocol_sha256': sha(out / 'protocol.json')}
    if not added:
        dump(out / 'preparation_result.json', result); return result
    for name, spec in base_artifacts.items():
        target = out / 'base' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / name, target)
        require(sha(target) == spec['sha256'], 'Parent array copy changed')
        artifacts['base/' + name] = spec
    manifest = [{**r, 'eeg_path': 'base/' + r['eeg_path']} for r in base_rows] + added
    (out / 'manifest.jsonl').write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False) + '\n' for r in manifest))
    dump(out / 'artifacts.json', artifacts)
    dump(out / 'source_admission.json', {'schema': 'xavier-smac-BCI8089820-physical-QC-admission-v1',
         'sources': {**base_admission['sources'], **admitted}, 'excluded_source_sha256': sorted(denied_source),
         'excluded_payload_sha256': sorted(denied_payload), 'excluded_source_groups': sorted(m['subject_group'] for m in plan['members'] if m['split'] != 'train'),
         'split_manifest_sha256': sha(out / 'split_manifest.json'), 'protocol_sha256': sha(out / 'protocol.json'),
         'parent_provenance_sha256': sha(out / 'parent_provenance.json'), 'cross_release_identity_scope': IDENTITY_SCOPE,
         'online_harness_allowed': False, 'validation_domain': '806023 only'})
    trainer.load_corpus(out / 'manifest.jsonl', out / 'artifacts.json', out / 'source_admission.json')
    result.update(windows=dict(Counter(r['split'] for r in manifest)), retained_base_windows=len(base_rows),
                  manifest_sha256=sha(out / 'manifest.jsonl'), artifacts_sha256=sha(out / 'artifacts.json'),
                  source_admission_sha256=sha(out / 'source_admission.json'), elapsed_preparation_seconds=time.time() - started)
    dump(out / 'completion.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['raw-root', 'archive', 'frozen-manifest', 'recordings', 'constraints', 'base-corpus', 'expected-base-completion-sha256', 'output']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--max-seconds', type=float, default=1800)
    parser.add_argument('--reserve-gib', type=float, default=100)
    result = prepare(parser.parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 2 if result['status'].startswith('blocked') else 0


if __name__ == '__main__':
    raise SystemExit(main())
