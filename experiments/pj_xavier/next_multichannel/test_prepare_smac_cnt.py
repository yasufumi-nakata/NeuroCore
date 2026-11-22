"""Synthetic CNT/CPU-only tests; no original EEG or model training."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import prepare_smac_cnt as c


def jsonl(path, records):
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))


def make_base(root):
    base = root / 'base_corpus'; base.mkdir()
    manifest, sources, artifacts = [], {}, {}
    for index, fold in enumerate(('train', 'val')):
        rel = 'base_' + fold + '.npy'
        np.save(base / rel, np.full((2, 16, 3, 200), .03, np.float32), allow_pickle=False)
        source, payload, group = f'{1000 + index:064x}', f'{1100 + index:064x}', 'zenodo:806023:base_' + fold
        sources[source] = {'split': fold, 'subject_group': group, 'eeg_payload_sha256': payload,
            'montage_key': 'base16', 'physical_unit_verified': True, 'fit_allowed': fold == 'train',
            'validation_allowed': fold == 'val'}
        artifacts[rel] = {'shape': [2, 16, 3, 200], 'dtype': 'float32', 'sha256': c.sha(base / rel), 'bytes': (base / rel).stat().st_size}
        for j in range(2):
            manifest.append({'id': source + ':' + str(j), 'split': fold, 'subject_group': group,
                'source_sha256': source, 'eeg_payload_sha256': payload, 'source_recording_id': 'zenodo:806023:' + fold + '.fif',
                'window_start_s': 2 + j * 3, 'window_end_s': 5 + j * 3, 'channels': ['C' + str(i) for i in range(16)],
                'sample_rate_hz': 200, 'unit': 'V_x_1e4', 'eeg_path': rel, 'eeg_index': j, 'montage_key': 'base16',
                'reference': 'synthetic', 'native_sample_rate_hz': 512, 'physical_unit_verified': True,
                'online_harness_allowed': False})
    jsonl(base / 'manifest.jsonl', manifest)
    (base / 'protocol.json').write_text(json.dumps({'synthetic': True}))
    (base / 'artifacts.json').write_text(json.dumps(artifacts))
    (base / 'source_admission.json').write_text(json.dumps({'sources': sources, 'excluded_source_sha256': [], 'excluded_payload_sha256': []}))
    return base, reseal_base(base), manifest, artifacts


def synthetic_cnt(index=0, amplitude=8e-6):
    samples, n, width, data_offset = 2500, 16, 2, 2100
    event_offset = data_offset + samples * n * width
    nextfile = event_offset + 9
    contents = bytearray(nextfile + 64)
    contents[:12] = b'Version 3.0\0'
    struct.pack_into('<i', contents, 12, nextfile)
    struct.pack_into('<H', contents, 370, n)
    struct.pack_into('<H', contents, 376, 250)
    struct.pack_into('<i', contents, 864, -525)
    struct.pack_into('<i', contents, 886, event_offset)
    struct.pack_into('<i', contents, 894, 1)
    names = c.cnt.EOG + c.CHANNELS
    for i, name in enumerate(names):
        off = 900 + i * 75
        contents[off:off + len(name)] = name.encode()
        struct.pack_into('<f', contents, off + 59, 12.8)
        struct.pack_into('<f', contents, off + 71, 1.)
    gain = struct.unpack('<f', struct.pack('<f', 12.8))[0] * 1e-6 / 204.8
    t = np.arange(samples) / 250
    # Large EOG ensures treating all sixteen channels as EEG would fail QC.
    physical = np.stack([.001 * np.sin(2 * np.pi * 5 * t)] * 2 +
                        [amplitude * np.sin(2 * np.pi * (8 + channel / 8 + index / 70) * t)
                         for channel in range(14)], axis=1)
    digital = np.rint(physical / gain).astype('<i2')
    contents[data_offset:event_offset] = digital.tobytes()
    struct.pack_into('<Bii', contents, event_offset, 2, 0, 0)
    contents[nextfile + 52] = 0
    return bytes(contents)


def fixture(root):
    ledger, metadata, proof_rows = [], [], []
    archive = root / 'data.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        index = 0
        for subject, count in ((1, 8), (2, 6), (3, 6), (4, 6)):
            for run in range(count):
                member = f'data/S{subject}_{run + 1}.cnt'
                file = root / 'one.cnt'; file.write_bytes(synthetic_cnt(index))
                record = c.cnt.audit_member(file, member, 0)
                ledger.append(record)
                metadata.append({'source': 'cnt14', 'source_group': 'cnt14|' + record['subject_group'],
                    'source_member_or_path': member, 'source_file_sha256': record['source_file_sha256'],
                    'eeg_payload_sha256': record['EEG_payload_sha256'], 'calibration_sha256': record['EEG_calibration_sha256'],
                    'metadata_only_proposed_split': c.FIXED_SPLIT[record['subject_group']],
                    'component_id_sha256': hashlib.sha256(record['subject_group'].encode()).hexdigest()})
                proof_rows.append({'member': member, 'original_is32bit_tag': 0,
                                   'format_tag_offset': record['format_tag_offset'], 'nextfile': record['original_nextfile']})
                z.write(file, member); index += 1
    file.unlink()
    jsonl(root / 'ledger.jsonl', ledger); jsonl(root / 'metadata.jsonl', metadata)
    (root / 'proof.json').write_text(json.dumps({'official_reader_source_sha256': c.cnt.READER_SHA,
        'nextfile_equals_event_table_end_all': True, 'source_header_repair_performed': False, 'records': proof_rows}))
    (root / 'deny.json').write_text(json.dumps({'excluded_source_sha256': ['f' * 64]}))
    base, completion, _, _ = make_base(root)
    return SimpleNamespace(archive=str(archive), source_ledger=str(root / 'ledger.jsonl'),
        metadata_admission=str(root / 'metadata.jsonl'), format_proof=str(root / 'proof.json'),
        deny_json=[str(root / 'deny.json')], base_corpus=str(base), expected_base_completion_sha256=completion,
        reserve_gib=1), ledger, metadata, proof_rows


def reseal_base(base):
    completion = {'status': 'prepared_verified_new_offline_SSL_corpus'}
    for name, key in [('manifest.jsonl', 'manifest_sha256'), ('artifacts.json', 'artifacts_sha256'),
                      ('source_admission.json', 'source_admission_sha256'), ('protocol.json', 'protocol_sha256')]:
        completion[key] = c.sha(base / name)
    (base / 'completion.json').write_text(json.dumps(completion))
    return c.sha(base / 'completion.json')


class CNTPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.args, cls.ledger, cls.metadata, cls.proof = fixture(cls.root)
        cls.seals = dict(ARCHIVE_SHA256=c.sha(cls.args.archive), ARCHIVE_BYTES=Path(cls.args.archive).stat().st_size,
                         LEDGER_SHA256=c.sha(cls.args.source_ledger), METADATA_SHA256=c.sha(cls.args.metadata_admission),
                         FORMAT_PROOF_SHA256=c.sha(cls.args.format_proof))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_frozen_split_deny_and_no_reassignment(self):
        plan = c.plan_records(self.ledger, self.metadata, set(), set())
        self.assertEqual(sum(p['status'] == 'candidate' for p in plan), 14)
        self.assertEqual(sum(p['split'] == 'val' for p in plan), 6)
        self.assertEqual(sum(p['split'] == 'test' for p in plan), 6)
        denied = c.plan_records(self.ledger, self.metadata, {self.ledger[0]['source_file_sha256']}, set())
        self.assertEqual(sum(p['status'] == 'candidate' for p in denied), 13)
        changed = copy.deepcopy(self.metadata)
        changed[8]['metadata_only_proposed_split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'Frozen CNT subject split'):
            c.plan_records(self.ledger, changed, set(), set())

    def test_source_calibration_and_duplicate_admission_are_checked(self):
        changed = copy.deepcopy(self.metadata); changed[0]['calibration_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'calibration mismatch'):
            c.plan_records(self.ledger, changed, set(), set())
        changed_ledger = copy.deepcopy(self.ledger)
        changed_ledger[0]['independent_MNE_header_and_three_calibrated_sample_probes_match'] = False
        with self.assertRaisesRegex(ValueError, 'calibration is unverified'):
            c.plan_records(changed_ledger, self.metadata, set(), set())

    def test_actual_CNT_integer_payload_and_independent_MNE_volts(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'current.cnt'
            with zipfile.ZipFile(self.args.archive) as z:
                file.write_bytes(z.read(self.ledger[0]['source_member']))
            x, proof = c.verify_cnt(file, self.ledger[0], self.proof[0])
            self.assertEqual(x.shape, (14, 2500))
            self.assertLess(float(np.max(np.abs(x))), 9e-6)
            self.assertLessEqual(proof['MNE_calibration_max_error_V'], 1e-12)
            self.assertEqual(proof['decoded_EEG_V_sha256'], hashlib.sha256(np.asarray(x.T, dtype='<f8', order='C').tobytes()).hexdigest())
            changed = copy.deepcopy(self.ledger[0]); changed['EEG_payload_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'integer EEG payload'):
                c.verify_cnt(file, changed, self.proof[0])
            file.write_bytes(file.read_bytes() + b'X')
            with self.assertRaisesRegex(ValueError, 'original file bytes'):
                c.verify_cnt(file, self.ledger[0], self.proof[0])

    def test_prepare_only_decompresses_train_and_matches_trainer_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            args = copy.copy(self.args); args.output = str(Path(temp) / 'prepared')
            opened = []; original_open = zipfile.ZipFile.open

            def tracked(z, name, *values, **options):
                opened.append(name.filename if hasattr(name, 'filename') else name)
                return original_open(z, name, *values, **options)

            with patch.multiple(c, **self.seals), patch.object(zipfile.ZipFile, 'open', tracked):
                result = c.prepare(args)
            self.assertEqual(result['new_source_records'], 14)
            self.assertEqual(result['new_source_windows'], 28)
            self.assertEqual(result['windows'], {'train': 30, 'val': 2})
            expected = {r['source_member'] for r in self.ledger if c.FIXED_SPLIT[r['subject_group']] == 'train'}
            self.assertEqual(set(opened), expected)
            self.assertEqual(len(opened), 14)
            out = Path(args.output)
            loaded, _, admission, _ = c.trainer.load_corpus(out / 'manifest.jsonl', out / 'artifacts.json', out / 'source_admission.json')
            self.assertEqual({len(r['channels']) for r in loaded}, {14, 16})
            self.assertEqual({r['native_sample_rate_hz'] for r in loaded}, {250, 512})
            base_val = [r for r in c.rows(Path(args.base_corpus) / 'manifest.jsonl') if r['split'] == 'val']
            val = [r for r in c.rows(out / 'manifest.jsonl') if r['split'] == 'val']
            self.assertEqual(val, [{**r, 'eeg_path': 'base/' + r['eeg_path']} for r in base_val])
            self.assertTrue(all(r['source_file_sha256'] in admission['excluded_source_sha256'] for r in self.ledger
                                if c.FIXED_SPLIT[r['subject_group']] != 'train'))
            self.assertFalse(result['CNT_generalization_evaluated'])

    def test_combined_806023_2392979_base_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); args = copy.copy(self.args)
            base = root / 'combined'; shutil.copytree(args.base_corpus, base)
            manifest = c.rows(base / 'manifest.jsonl')
            admission = json.loads((base / 'source_admission.json').read_text())
            artifacts = json.loads((base / 'artifacts.json').read_text())
            source, payload = f'{2000:064x}', f'{2100:064x}'
            extra = {**manifest[0], 'id': source + ':0', 'source_sha256': source, 'eeg_payload_sha256': payload,
                     'source_recording_id': 'zenodo:2392979:synthetic.fif', 'subject_group': 'zenodo:2392979:synthetic',
                     'channels': ['C' + str(i) for i in range(8)], 'native_sample_rate_hz': 256,
                     'montage_key': 'synthetic8', 'eeg_path': 'extra8.npy', 'eeg_index': 0}
            np.save(base / 'extra8.npy', np.full((1, 8, 3, 200), .02, np.float32), allow_pickle=False)
            artifacts['extra8.npy'] = {'shape': [1, 8, 3, 200], 'dtype': 'float32', 'sha256': c.sha(base / 'extra8.npy'), 'bytes': (base / 'extra8.npy').stat().st_size}
            admission['sources'][source] = {**admission['sources'][manifest[0]['source_sha256']],
                'eeg_payload_sha256': payload, 'subject_group': extra['subject_group'], 'montage_key': extra['montage_key']}
            jsonl(base / 'manifest.jsonl', manifest + [extra])
            (base / 'artifacts.json').write_text(json.dumps(artifacts)); (base / 'source_admission.json').write_text(json.dumps(admission))
            args.base_corpus = str(base); args.expected_base_completion_sha256 = reseal_base(base)
            args.output = str(root / 'prepared')
            with patch.multiple(c, **self.seals):
                result = c.prepare(args)
            self.assertEqual(result['retained_base_windows'], 5)
            self.assertEqual(result['windows'], {'train': 31, 'val': 2})
            prepared = c.rows(Path(args.output) / 'manifest.jsonl')
            self.assertEqual(prepared[:5], [{**r, 'eeg_path': 'base/' + r['eeg_path']} for r in manifest + [extra]])

    def test_all_QC_rejected_never_creates_base_only_training_corpus(self):
        with tempfile.TemporaryDirectory() as temp:
            args = copy.copy(self.args); args.output = str(Path(temp) / 'prepared')
            rejected = (np.empty((0, 14, 3, 200), np.float32), [], {'synthetic_forced_QC_reject': 2})
            with patch.multiple(c, **self.seals), patch.object(c, 'causal_windows14', return_value=rejected):
                result = c.prepare(args)
            self.assertEqual(result['status'], 'blocked_no_eligible_new_source_windows')
            self.assertEqual(result['new_source_windows'], 0)
            self.assertFalse((Path(args.output) / 'manifest.jsonl').exists())
            self.assertFalse((Path(args.output) / 'base').exists())
            self.assertFalse((Path(args.output) / 'completion.json').exists())

    def test_causal_cutoff_and_fixed_QC_without_rescaling(self):
        t = np.arange(5000) / 250
        x = np.stack([8e-6 * np.sin(2 * np.pi * (8 + i / 7) * t) for i in range(14)])
        a, windows, _ = c.causal_windows14(x)
        future = x.copy(); future[:, 2000:] += 5e-6 * np.sin(2 * np.pi * 13 * t[2000:])
        b, future_windows, _ = c.causal_windows14(future)
        for index, window in enumerate(windows):
            if window['latest_native_source_index'] < 2000:
                j = next(j for j, other in enumerate(future_windows) if other['window_start_s'] == window['window_start_s'])
                np.testing.assert_array_equal(a[index], b[j])
        self.assertEqual(c.causal_windows14(np.zeros_like(x))[0].shape[0], 0)
        self.assertEqual(c.causal_windows14(x * 1000)[0].shape[0], 0)
        with self.assertRaisesRegex(ValueError, 'physical-V magnitude'):
            c.causal_windows14(x * 100000)

    def test_base_mutation_or_new_validation_domain_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp) / 'base'; shutil.copytree(self.args.base_corpus, base)
            file = next(base.glob('*.npy'))
            with file.open('r+b') as stream:
                stream.seek(-1, 2); stream.write(b'X')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                c.validate_base(base, self.args.expected_base_completion_sha256)
            shutil.copyfile(Path(self.args.base_corpus) / file.name, file)
            manifest = c.rows(base / 'manifest.jsonl')
            for row in manifest:
                if row['split'] == 'val':
                    row['source_recording_id'] = 'zenodo:2392979:invalid-new-validation.fif'
            jsonl(base / 'manifest.jsonl', manifest)
            with self.assertRaisesRegex(ValueError, 'existing 806023 validation'):
                c.validate_base(base, reseal_base(base))


if __name__ == '__main__':
    unittest.main()
