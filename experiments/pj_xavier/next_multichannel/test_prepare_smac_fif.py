import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from prepare_smac_fif import (causal_transform, make_windows, prepare,
                              read_denies, register_sources, sha)


CHANNELS = ['Fpz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3',
            'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8']


def planned_fixture():
    ledger, metadata = [], []
    for i in range(8):
        split = 'train' if i < 4 else 'val' if i < 6 else 'test'
        source, payload = f'{i + 1:064x}', f'{i + 101:064x}'
        name = f'subject{i+1}.raw.fif'
        group = f'zenodo:806023:subject{i+1}'
        ledger.append({'dataset_key': 'zenodo-806023-806023',
                       'source_sha256': source, 'decoded_payload_sha256': payload,
                       'relative_path': name, 'subject_group': group,
                       'eeg_channel_count': 16, 'eeg_channels': CHANNELS,
                       'unit': 'V_as_decoded_using_FIF_header_calibration',
                       'sample_count_per_EEG_channel': 5120})
        metadata.append({'source_group': 'compact14_16|' + group,
                         'source_file_sha256': source, 'eeg_payload_sha256': payload,
                         'source_member_or_path': name,
                         'metadata_only_proposed_split': split, 'held_reasons': []})
    return ledger, metadata


class PreparationTests(unittest.TestCase):
    def test_source_registration_preserves_tests_and_old_denies(self):
        ledger, metadata = planned_fixture()
        result = register_sources(ledger, metadata, {ledger[0]['source_sha256']}, set(), set())
        self.assertEqual(sum(x['split'] == 'train' for x in result), 4)
        self.assertEqual(sum(x['reason'] == 'frozen_test_not_opened' for x in result), 2)
        self.assertEqual(result[0]['reason'], 'prior_heldout_or_review_identity')

    def test_duplicate_cross_fold_is_rejected(self):
        ledger, metadata = planned_fixture()
        ledger[4]['decoded_payload_sha256'] = ledger[0]['decoded_payload_sha256']
        metadata[4]['eeg_payload_sha256'] = ledger[0]['decoded_payload_sha256']
        with self.assertRaisesRegex(ValueError, 'crosses frozen folds'):
            register_sources(ledger, metadata, set(), set(), set())

    def test_causal_output_does_not_use_future_source(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0, 4e-6, (16, 512 * 12))
        before, _, cutoffs = causal_transform(x, 512)
        changed = x.copy()
        boundary = 512 * 8
        changed[:, boundary:] += 4e-5
        after, _, _ = causal_transform(changed, 512)
        np.testing.assert_array_equal(before[:, cutoffs < boundary], after[:, cutoffs < boundary])

    def test_fixed_QC_rejects_flat_and_large_signal_without_clipping(self):
        flat, _, rejected = make_windows(np.zeros((16, 512 * 12)), 512)
        self.assertEqual(len(flat), 0)
        self.assertGreater(rejected['flat_below_fixed_0_01uV'], 0)
        t = np.arange(512 * 12) / 512
        large = np.tile(500e-6 * np.sin(2 * np.pi * 10 * t), (16, 1))
        accepted, _, rejected = make_windows(large, 512)
        self.assertEqual(len(accepted), 0)
        self.assertGreater(rejected['peak_above_fixed_100uV'], 0)

    def test_prior_native_payload_and_exclusion_identities_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'deny.json'
            p.write_text(json.dumps({'sources': {'a'*64: {'split': 'val', 'subject_group': 'g'}},
                'excluded_source_sha256': ['b'*64],
                'records': [{'split': 'test', 'source_eeg_sha256': 'c'*64, 'eeg_payload_sha256': 'd'*64}]}))
            sources, payloads, groups, _ = read_denies([p])
            self.assertEqual(sources, {'a'*64, 'b'*64, 'c'*64})
            self.assertEqual(payloads, {'d'*64})
            self.assertEqual(groups, {'g'})

    def test_synthetic_end_to_end_does_not_open_test_files(self):
        import mne
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rawdir = root / 'raw'
            rawdir.mkdir()
            ledger, metadata = planned_fixture()
            for i in range(6):
                # Tests intentionally do not exist. All six actual files are synthetic.
                t = np.arange(5120) / 512
                x = np.stack([8e-6 * np.sin(2 * np.pi * (8 + c/8 + i/20) * t)
                              for c in range(16)])
                raw = mne.io.RawArray(x, mne.create_info(CHANNELS, 512, 'eeg'), verbose='ERROR')
                file = rawdir / ledger[i]['relative_path']
                raw.save(file, overwrite=False, verbose='ERROR')
                decoded = mne.io.read_raw_fif(file, preload=False, verbose='ERROR')
                payload = hashlib.sha256(np.asarray(decoded.get_data().T, dtype='<f8', order='C').tobytes()).hexdigest()
                decoded.close()
                ledger[i]['source_sha256'] = sha(file)
                ledger[i]['decoded_payload_sha256'] = payload
                metadata[i]['source_file_sha256'] = sha(file)
                metadata[i]['eeg_payload_sha256'] = payload
            lp, mp, dp = root/'ledger.jsonl', root/'metadata.jsonl', root/'deny.json'
            lp.write_text(''.join(json.dumps(x)+'\n' for x in ledger))
            mp.write_text(''.join(json.dumps(x)+'\n' for x in metadata))
            dp.write_text(json.dumps({'excluded_source_sha256': ['f'*64]}))
            out = root/'prepared'
            summary = prepare(SimpleNamespace(raw_root=str(rawdir), source_ledger=str(lp),
                metadata_admission=str(mp), deny_json=[str(dp)], output=str(out), reserve_gib=1))
            self.assertEqual(summary['records_admitted'], 6)
            self.assertEqual(summary['test_records_not_opened'], 2)
            self.assertFalse(summary['online_harness_allowed'])
            manifest = [json.loads(line) for line in (out/'manifest.jsonl').read_text().splitlines()]
            self.assertEqual({r['split'] for r in manifest}, {'train', 'val'})
            for row in manifest:
                array = np.load(out/row['eeg_path'], allow_pickle=False)
                self.assertEqual(array[row['eeg_index']].shape, (16, 3, 200))
                self.assertTrue(row['physical_unit_verified'])
                self.assertFalse(row['online_harness_allowed'])
            admission = json.loads((out/'source_admission.json').read_text())
            self.assertIn(ledger[7]['source_sha256'], admission['excluded_source_sha256'])
            certificates = json.loads((out/'source_certificates.json').read_text())
            self.assertEqual(sum(not c['signal_opened'] for c in certificates), 2)
            with self.assertRaisesRegex(ValueError, 'immutable'):
                prepare(SimpleNamespace(raw_root=str(rawdir), source_ledger=str(lp),
                    metadata_admission=str(mp), deny_json=[str(dp)], output=str(out), reserve_gib=1))


if __name__ == '__main__':
    unittest.main()
