"""Synthetic GDF fixtures only: no source EEG, network, or model training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest import mock
import zipfile

import numpy as np
import prepare_smac_bci8089820 as p


def synthetic_gdf(path, group='A1', amplitude=8e-6, phase=0., huge=False, overflow=False, units='uV'):
    """Original GDF 1.25 with interleaved EEG/auxiliary and float64 scalars."""
    channels = p.CHANNELS[:11] + ['EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd'] + p.CHANNELS[11:]
    n, count = len(channels), 9 * p.NATIVE_FS
    if huge:
        lo, hi = -float(np.float32(1.701411733e38)), float(np.float32(1.701411733e38))
        bound = 2**62 if overflow else 2**62 - 1
        dlo, dhi = -bound, bound
    else:
        lo, hi, dlo, dhi = -1e6, 1e6, -1000000, 1000000
    gain = (hi - lo) / (float(dhi) - float(dlo))
    offset = lo - float(dlo) * gain
    t = np.arange(count, dtype=np.float64) / p.NATIVE_FS
    physical = np.stack([amplitude * np.sin(2 * np.pi * 10 * t + phase + i / 25) for i in range(27)], axis=1)
    values = np.zeros((count, n), dtype='<f8')
    for i, name in enumerate(channels):
        if name in p.CHANNELS:
            values[:, i] = (physical[:, p.CHANNELS.index(name)] / 1e-6 - offset) / gain
        else:
            values[:, i] = 8e5  # Auxiliary amplitude must not become an EEG QC input.
    fixed = (b'GDF 1.25' + b'SYNTHETIC'.ljust(80) + group.encode().ljust(80)
             + b'2026112300000000' + struct.pack('<q', (n + 1) * 256)
             + bytes(24 + 20) + struct.pack('<qIII', count, 1, p.NATIVE_FS, n))
    variable = (b''.join(x.encode().ljust(16) for x in channels) + b' ' * (80 * n)
                + b''.join(units.encode().ljust(8) for _ in channels)
                + np.full(n, lo, dtype='<f8').tobytes() + np.full(n, hi, dtype='<f8').tobytes()
                + np.full(n, dlo, dtype='<i8').tobytes() + np.full(n, dhi, dtype='<i8').tobytes()
                + b' ' * (80 * n) + np.ones(n, dtype='<i4').tobytes()
                + np.full(n, 17, dtype='<i4').tobytes() + bytes(32 * n))
    assert len(fixed) == 256 and len(variable) == 256 * n
    path.write_bytes(fixed + variable + values.tobytes() + bytes([1, 0, 0, 0]) + struct.pack('<I', 0))
    member = 'BCI Database/Signals/DATA A/' + group + '/' + group + '_synthetic.gdf'
    record = p.original.audit_gdf(path, member)
    return record, physical.T


def source_fixture(root):
    records, members = {}, []
    archive = root / 'fixture.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        for group, phase in [('A1', 0.), ('A2', 0.), ('A3', .4)]:
            source = root / (group + '.gdf')
            record, _ = synthetic_gdf(source, group, phase=phase)
            records[record['source_member']] = record
            zf.write(source, record['source_member'])
            members.append({'source_member': record['source_member'], 'source_file_sha256': record['source_file_sha256'],
                            'eeg_payload_sha256': record['eeg_payload_sha256'], 'calibration_sha256': record['EEG_calibration_sha256'],
                            'subject_group': record['subject_group'], 'split': 'train'})
        for group, split in [('A4', 'val'), ('A5', 'test'), ('A6', 'review')]:
            name = 'BCI Database/Signals/DATA A/' + group + '/' + group + '_never_open.gdf'
            value = ('NOT A GDF; NEVER DECOMPRESS ' + split).encode()
            digest = hashlib.sha256(value).hexdigest()
            record = {'source_member': name, 'source_file_sha256': digest, 'source_file_bytes': len(value),
                      'eeg_payload_sha256': hashlib.sha256(split.encode()).hexdigest(), 'EEG_calibration_sha256': '0' * 64,
                      'subject_group': 'zenodo:8089820:A:' + group}
            records[name] = record; zf.writestr(name, value)
            members.append({'source_member': name, 'source_file_sha256': digest, 'eeg_payload_sha256': record['eeg_payload_sha256'],
                            'calibration_sha256': record['EEG_calibration_sha256'], 'subject_group': record['subject_group'], 'split': split})
    return archive, {'members': members}, records


def base_fixture(root):
    base = root / 'base'; base.mkdir()
    manifest, sources, artifacts = [], {}, {}
    for split, letter in [('train', 'a'), ('val', 'b')]:
        source, payload = letter * 64, ('c' if split == 'train' else 'd') * 64
        array = np.full((1, 16, 3, 200), .05, np.float32)
        rel = split + '.npy'; np.save(base / rel, array, allow_pickle=False)
        artifacts[rel] = {'shape': list(array.shape), 'dtype': 'float32', 'bytes': (base / rel).stat().st_size, 'sha256': p.sha(base / rel)}
        group = 'zenodo:806023:' + split; montage = 'synthetic16'
        sources[source] = {'split': split, 'subject_group': group, 'eeg_payload_sha256': payload, 'montage_key': montage,
                           'physical_unit_verified': True, 'fit_allowed': split == 'train', 'validation_allowed': split == 'val'}
        manifest.append({'id': source + ':0', 'source_sha256': source, 'eeg_payload_sha256': payload, 'split': split,
                         'subject_group': group, 'source_recording_id': 'zenodo:806023:' + split, 'channels': ['C' + str(i) for i in range(16)],
                         'montage_key': montage, 'reference': 'synthetic', 'native_sample_rate_hz': 200, 'sample_rate_hz': 200,
                         'unit': 'V_x_1e4', 'window_start_s': 0., 'window_end_s': 3., 'eeg_path': rel, 'eeg_index': 0,
                         'physical_unit_verified': True, 'online_harness_allowed': False})
    (base / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in manifest))
    p.dump(base / 'artifacts.json', artifacts)
    p.dump(base / 'source_admission.json', {'sources': sources, 'excluded_source_sha256': [], 'excluded_payload_sha256': []})
    p.dump(base / 'protocol.json', {'synthetic': True})
    p.dump(base / 'split_manifest.json', {'frozen_heldout': ['synthetic_BDF_holdout']})
    p.dump(base / 'source_certificates.json', [{'signal_opened': False, 'split': 'test'}])
    completion = {'status': 'prepared_verified_new_offline_SSL_corpus'}
    for filename, key in [('manifest.jsonl', 'manifest_sha256'), ('artifacts.json', 'artifacts_sha256'),
                          ('source_admission.json', 'source_admission_sha256'), ('protocol.json', 'protocol_sha256'),
                          ('split_manifest.json', 'split_manifest_sha256'), ('source_certificates.json', 'certificates_sha256')]:
        completion[key] = p.sha(base / filename)
    p.dump(base / 'completion.json', completion)
    return base


class GDFPreparation(unittest.TestCase):
    def test_original_all_channel_affine_calibration_and_auxiliary_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'synthetic.gdf'; record, expected = synthetic_gdf(path)
            x, proof = p.verify_gdf(path, record)
            np.testing.assert_allclose(x, expected, atol=1e-15, rtol=1e-12)
            self.assertTrue(proof['physical_unit_verified'])
            self.assertEqual(len(proof['excluded_EOG_channels']), 3)
            self.assertEqual(len(proof['excluded_EMG_channels']), 2)
            self.assertLess(proof['observed_native_peak_V'], 1e-5)
            self.assertNotEqual(proof['decoded_EEG_V_sha256'], proof['original_EEG_scalar_payload_sha256'])

    def test_huge_gain_tiny_scalar_survives_stable_affine_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'huge.gdf'; record, expected = synthetic_gdf(path, huge=True)
            x, proof = p.verify_gdf(path, record)
            np.testing.assert_allclose(x, expected, atol=1e-15, rtol=1e-12)
            self.assertGreater(min(proof['gain_V_per_original_scalar']), 1e13)
            self.assertEqual(proof['original_digital_max_decimal_strings'][0], str(2**62 - 1))
            self.assertFalse(proof['any_integer_range_exceeds_signed_int64'])
            dmin, gain = -float(2**62 - 1), proof['gain_V_per_original_scalar'][0]
            tiny = 1e-18
            cancelled = (tiny - dmin) * gain + dmin * gain
            self.assertEqual(cancelled, 0.)
            self.assertGreater(tiny * gain, 1e-5)

    def test_MNE_int64_range_overflow_is_quarantined_without_unit_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'overflow.gdf'; record, _ = synthetic_gdf(path, huge=True, overflow=True)
            with self.assertRaises(p.PhysicalCalibrationHold) as found:
                p.verify_gdf(path, record)
            self.assertTrue(found.exception.evidence['any_integer_range_exceeds_signed_int64'])
            self.assertEqual(found.exception.reason, 'held_physical_calibration_MNE_disagreement')

    def test_original_source_and_scalar_mutations_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mutable.gdf'; record, _ = synthetic_gdf(path)
            wrong = dict(record, eeg_payload_sha256='e' * 64)
            with self.assertRaisesRegex(ValueError, 'scalar EEG payload'):
                p.verify_gdf(path, wrong)
            path.write_bytes(path.read_bytes() + b'X')
            with self.assertRaisesRegex(ValueError, 'source bytes'):
                p.verify_gdf(path, record)

    def test_unit_and_calibration_provenance_cannot_be_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad_units.gdf'; record, _ = synthetic_gdf(path, units='mV')
            with self.assertRaisesRegex(ValueError, 'physical calibration'):
                p.verify_gdf(path, record)
            record, _ = synthetic_gdf(path)
            record['EEG_calibration_sha256'] = 'f' * 64
            with self.assertRaisesRegex(ValueError, 'calibration fingerprint'):
                p.verify_gdf(path, record)

    def test_full_train_candidates_alias_dedup_holdouts_and_real_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive, plan, record_map = source_fixture(root)
            base = base_fixture(root)
            _, base_rows, admission, base_artifacts, proof = p.validate_base(base, p.sha(base / 'completion.json'))
            self.assertIn('split_manifest.json', proof['optional_metadata'])
            out = root / 'combined'; out.mkdir()
            opened = []; original_open = zipfile.ZipFile.open
            def observing(zf, name, *args, **kwargs):
                opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else name)
                return original_open(zf, name, *args, **kwargs)
            with mock.patch.object(zipfile.ZipFile, 'open', observing):
                added, admitted, artifacts, certs, ds, dp = p.prepare_sources(archive, out, plan, record_map, admission, time.time() + 60, 0)
            self.assertEqual(len(opened), 3)
            self.assertTrue(all(record_map[name]['subject_group'][-2:] in ['A1', 'A2', 'A3'] for name in opened))
            self.assertEqual(Counter(c['status'] for c in certs)['excluded_exact_duplicate_train_alias'], 1)
            self.assertEqual(len(admitted), 2)
            self.assertEqual(len(added), 4)
            self.assertTrue(all(not c['signal_opened'] for c in certs if c['split'] != 'train'))
            self.assertEqual(len(ds), 3); self.assertEqual(len(dp), 3)
            for rel, spec in base_artifacts.items():
                destination = out / 'base' / rel; destination.parent.mkdir(exist_ok=True)
                destination.write_bytes((base / rel).read_bytes()); artifacts['base/' + rel] = spec
            manifest = [{**r, 'eeg_path': 'base/' + r['eeg_path']} for r in base_rows] + added
            (out / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in manifest))
            p.dump(out / 'artifacts.json', artifacts)
            p.dump(out / 'source_admission.json', {'sources': {**admission['sources'], **admitted}, 'excluded_source_sha256': sorted(ds), 'excluded_payload_sha256': sorted(dp)})
            loaded, _, _, _ = p.trainer.load_corpus(out / 'manifest.jsonl', out / 'artifacts.json', out / 'source_admission.json')
            self.assertEqual(len(loaded), 6)
            self.assertEqual({len(r['channels']) for r in loaded}, {16, 27})

    def test_previous_raw_scalar_and_physical_denies_are_all_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive, plan, record_map = source_fixture(root)
            record = next(r for r in record_map.values() if r['subject_group'].endswith('A1'))
            _, proof = p.verify_gdf(root / 'A1.gdf', record)
            for index, (source_denies, payload_denies) in enumerate([
                ([record['source_file_sha256']], []), ([], [record['eeg_payload_sha256']]), ([], [proof['decoded_EEG_V_sha256']])]):
                out = root / str(index); out.mkdir()
                admission = {'sources': {}, 'excluded_source_sha256': source_denies, 'excluded_payload_sha256': payload_denies}
                _, admitted, _, certs, _, _ = p.prepare_sources(archive, out, plan, record_map, admission, time.time() + 60, 0)
                self.assertNotIn(record['source_file_sha256'], admitted)
                target = next(c for c in certs if c['source_sha256'] == record['source_file_sha256'])
                self.assertEqual(target['status'], 'excluded_known_heldout_raw_scalar_or_physical_payload')

    def test_fixed_QC_drops_high_amplitude_and_flat_windows(self):
        t = np.arange(9 * 512) / 512
        x = np.tile(500e-6 * np.sin(2 * np.pi * 10 * t), (27, 1))
        arrays, intervals, rejected = p.causal_windows(x)
        self.assertEqual(len(arrays), 0); self.assertEqual(intervals, [])
        self.assertGreater(rejected['peak_above_fixed_100uV'], 0)
        arrays, _, rejected = p.causal_windows(np.zeros_like(x))
        self.assertEqual(len(arrays), 0); self.assertGreater(rejected['flat_below_fixed_0_01uV'], 0)

    def test_future_samples_cannot_change_an_earlier_prepared_window(self):
        t = np.arange(9 * 512) / 512
        x = np.tile(8e-6 * np.sin(2 * np.pi * 10 * t), (27, 1))
        before, intervals, _ = p.causal_windows(x)
        x[:, 3000:] *= 50
        after, _, _ = p.causal_windows(x)
        np.testing.assert_array_equal(before[0], after[0])
        self.assertLess(intervals[0]['latest_native_source_index'], 3000)

    def test_parent_optional_split_and_certificate_seals_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = base_fixture(Path(tmp)); seal = p.sha(base / 'completion.json')
            p.validate_base(base, seal)
            (base / 'split_manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'split_manifest.json'):
                p.validate_base(base, seal)


if __name__ == '__main__':
    unittest.main()
