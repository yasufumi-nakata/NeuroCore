import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from prepare_smac_ssvep import (CHANNELS, causal_windows8, plan_records, prepare,
                                sha, validate_base)


GROUP = 'zenodo:2392979:subject'
PAIR_NAMES = [('03_run1', '07_run1'), ('06_run3', '07_run3'),
              ('09_run3', '10_run3'), ('09_run4', '10_run4'),
              ('11_run3', '12_run3'), ('11_run4', '12_run4'), ('11_run5', '12_run5')]


def fixture():
    ledger, metadata = [], []
    counts = [2, 2, 2, 2, 2, 3, 3, 2, 4, 4, 5, 5]
    for number, count in enumerate(counts, 1):
        for run in range(1, count+1):
            index = len(ledger)+1
            source, payload = f'{index:064x}', f'{index+100:064x}'
            name, group = f'subject{number:02d}_run{run}_raw.fif', f'{GROUP}{number:02d}'
            ledger.append({'catalog_id': '001148', 'recording_id': name,
                'subject_group': group, 'source_sha256': source, 'decoded_EEG_V_sha256': payload,
                'sample_count': 2560, 'sample_rate_hz': 256, 'channels': CHANNELS,
                'channel_calibration': [{'kind': 2, 'unit': 107, 'unit_mul': 0, 'cal': 1., 'range': 1.}]*8,
                'author_protocol_failure_datetime_candidate_match': False})
            metadata.append({'source_group': 'small_bci_mixed|'+group, 'source_file_sha256': source,
                'metadata_only_proposed_split': 'test' if number in (5, 9, 10) else 'train'})
    by_name = {r['recording_id']: r for r in ledger}
    pairs = []
    for left, right in PAIR_NAMES:
        a, b = by_name['subject'+left+'_raw.fif'], by_name['subject'+right+'_raw.fif']
        b['decoded_EEG_V_sha256'] = a['decoded_EEG_V_sha256']
        pairs.append({'left_record': a['recording_id'], 'right_record': b['recording_id'],
                      'decoded_EEG_V_sha256': a['decoded_EEG_V_sha256']})
    components = [[1], [2], [3, 6, 7], [4], [5], [8], [9, 10], [11, 12]]
    constraints = {'all_group_components': [[f'{GROUP}{n:02d}' for n in ns] for ns in components],
                   'pairs': pairs}
    return ledger, metadata, constraints


def jsonl(path, records):
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))


def make_base(root):
    base = root/'base_corpus'
    base.mkdir()
    manifest, sources, artifacts = [], {}, {}
    for index, fold in enumerate(('train', 'val')):
        rel = f'base_{fold}.npy'
        np.save(base/rel, np.full((2, 16, 3, 200), .03, dtype=np.float32), allow_pickle=False)
        source, payload = f'{1000+index:064x}', f'{1100+index:064x}'
        group = 'zenodo:806023:base_'+fold
        sources[source] = {'split': fold, 'subject_group': group, 'eeg_payload_sha256': payload,
            'montage_key': 'base16', 'physical_unit_verified': True,
            'fit_allowed': fold == 'train', 'validation_allowed': fold == 'val'}
        artifacts[rel] = {'shape': [2, 16, 3, 200], 'dtype': 'float32',
                          'sha256': sha(base/rel), 'bytes': (base/rel).stat().st_size}
        for j in range(2):
            manifest.append({'id': source+':'+str(j), 'split': fold, 'subject_group': group,
                'source_sha256': source, 'eeg_payload_sha256': payload,
                'source_recording_id': 'zenodo:806023:'+fold+'.fif',
                'window_start_s': 2+j*3, 'window_end_s': 5+j*3, 'channels': ['C'+str(c) for c in range(16)],
                'sample_rate_hz': 200, 'unit': 'V_x_1e4', 'eeg_path': rel, 'eeg_index': j,
                'montage_key': 'base16', 'reference': 'synthetic', 'native_sample_rate_hz': 512,
                'physical_unit_verified': True, 'online_harness_allowed': False})
    jsonl(base/'manifest.jsonl', manifest)
    (base/'protocol.json').write_text(json.dumps({'synthetic': True}))
    (base/'artifacts.json').write_text(json.dumps(artifacts))
    (base/'source_admission.json').write_text(json.dumps({'sources': sources,
        'excluded_source_sha256': [], 'excluded_payload_sha256': []}))
    completion = {'status': 'prepared_verified_new_offline_SSL_corpus'}
    for name, field in [('manifest.jsonl', 'manifest_sha256'), ('protocol.json', 'protocol_sha256'),
                        ('artifacts.json', 'artifacts_sha256'), ('source_admission.json', 'source_admission_sha256')]:
        completion[field] = sha(base/name)
    (base/'completion.json').write_text(json.dumps(completion))
    return base, sha(base/'completion.json'), manifest, artifacts


def preparation_fixture(root, amplitude):
    import mne
    ledger, metadata, constraints = fixture()
    rawdir = root/'raw'
    rawdir.mkdir()
    plan = plan_records(ledger, metadata, constraints, set(), set())
    by_name = {r['recording_id']: r for r in ledger}
    for index, item in enumerate(plan):
        r = item['record']
        if item['status'] != 'candidate':
            continue
        t = np.arange(2560)/256
        x = np.stack([amplitude*np.sin(2*np.pi*(8+c/8+index/60)*t) for c in range(8)])
        raw = mne.io.RawArray(x, mne.create_info(CHANNELS, 256, 'eeg'), verbose='ERROR')
        file = rawdir/r['recording_id']
        raw.save(file, overwrite=False, verbose='ERROR')
        raw.close()
        decoded = mne.io.read_raw_fif(file, preload=False, verbose='ERROR')
        r['source_sha256'] = sha(file)
        r['decoded_EEG_V_sha256'] = hashlib.sha256(np.asarray(decoded.get_data().T, dtype='<f8', order='C').tobytes()).hexdigest()
        r['channel_calibration'] = [{k: int(ch[k]) if k in ('kind', 'unit', 'unit_mul') else float(ch[k])
            for k in ('kind', 'unit', 'unit_mul', 'cal', 'range')} for ch in decoded.info['chs']]
        decoded.close()
    for pair in constraints['pairs']:
        a, b = by_name[pair['left_record']], by_name[pair['right_record']]
        b['decoded_EEG_V_sha256'] = a['decoded_EEG_V_sha256']
        # An identical whole-file alias must not globally deny its admitted canonical source.
        b['source_sha256'] = a['source_sha256']
        pair['decoded_EEG_V_sha256'] = a['decoded_EEG_V_sha256']
    for m, r in zip(metadata, ledger):
        m['source_file_sha256'] = r['source_sha256']
    lp, mp, cp, dp = root/'ledger.jsonl', root/'metadata.jsonl', root/'constraints.json', root/'deny.json'
    jsonl(lp, ledger); jsonl(mp, metadata)
    cp.write_text(json.dumps(constraints))
    dp.write_text(json.dumps({'excluded_source_sha256': ['f'*64]}))
    base, base_sha, base_manifest, base_artifacts = make_base(root)
    args = SimpleNamespace(raw_root=str(rawdir), source_ledger=str(lp), metadata_admission=str(mp),
        must_link=str(cp), base_corpus=str(base), expected_base_completion_sha256=base_sha,
        output=str(root/'prepared'), deny_json=[str(dp)], reserve_gib=1)
    return args, ledger, base_manifest, base_artifacts


class SSVEPAdditionTests(unittest.TestCase):
    def test_frozen_plan_has_21_unique_train_and_keeps_ten_tests(self):
        ledger, metadata, constraints = fixture()
        plan = plan_records(ledger, metadata, constraints, set(), set())
        self.assertEqual(sum(p['status'] == 'candidate' for p in plan), 21)
        self.assertEqual(sum(p['status'] == 'duplicate_omitted' for p in plan), 5)
        self.assertEqual(sum(p['reason'] == 'frozen_test_not_opened' for p in plan), 10)
        self.assertEqual({p['split'] for p in plan}, {'train', 'test'})

    def test_must_link_cross_fold_and_duplicate_proof_mismatch_stop(self):
        ledger, metadata, constraints = fixture()
        for m in metadata:
            if m['source_group'].endswith('subject07'):
                m['metadata_only_proposed_split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'component crosses frozen folds'):
            plan_records(ledger, metadata, constraints, set(), set())
        ledger, metadata, constraints = fixture()
        constraints['pairs'][0]['decoded_EEG_V_sha256'] = 'e'*64
        with self.assertRaisesRegex(ValueError, 'duplicate proof'):
            plan_records(ledger, metadata, constraints, set(), set())

    def test_prior_deny_propagates_to_must_link_component(self):
        ledger, metadata, constraints = fixture()
        r = next(r for r in ledger if r['subject_group'].endswith('subject07'))
        plan = plan_records(ledger, metadata, constraints, {r['source_sha256']}, set())
        related = [p for p in plan if p['record']['subject_group'][-2:] in ('03', '06', '07')]
        self.assertTrue(all(p['reason'] == 'prior_heldout_or_review_identity' for p in related))

    def test_fixed_QC_and_causality_without_unit_adjustment(self):
        t = np.arange(256*12)/256
        x = np.stack([8e-6*np.sin(2*np.pi*(8+c/8)*t) for c in range(8)])
        a, windows, _ = causal_windows8(x)
        self.assertEqual(a.shape[1:], (8, 3, 200))
        changed = x.copy()
        changed[:, 8*256:] += 20e-6
        b, _, _ = causal_windows8(changed)
        earlier = [i for i, w in enumerate(windows) if w['latest_native_source_index'] < 8*256]
        self.assertTrue(earlier)
        np.testing.assert_array_equal(a[earlier], b[earlier])
        with self.assertRaisesRegex(ValueError, 'fixed_0_1V_source_bound'):
            causal_windows8(x+1.)
        big, _, rejected = causal_windows8(x*100)
        self.assertEqual(len(big), 0)
        self.assertGreater(rejected['peak_above_fixed_100uV'], 0)

    def test_base_array_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            base, digest, _, _ = make_base(Path(d))
            with (base/'base_train.npy').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Base array identity'):
                validate_base(base, digest)

    def test_synthetic_merge_keeps_base_validation_and_never_opens_tests_or_aliases(self):
        with tempfile.TemporaryDirectory() as d:
            args, ledger, bm, bart = preparation_fixture(Path(d), 8e-6)
            result = prepare(args)
            out = Path(args.output)
            self.assertEqual(result['status'], 'prepared_verified_new_offline_SSL_corpus')
            self.assertEqual(result['new_source_windows'], 42)
            self.assertEqual(result['windows'], {'train': 44, 'val': 2})
            self.assertFalse(result['2392979_generalization_evaluated'])
            manifest = [json.loads(s) for s in (out/'manifest.jsonl').read_text().splitlines()]
            expected = [{**r, 'eeg_path': 'base/'+r['eeg_path']} for r in bm if r['split'] == 'val']
            self.assertEqual([r for r in manifest if r['split'] == 'val'], expected)
            for rel, spec in bart.items():
                self.assertEqual(sha(out/'base'/rel), spec['sha256'])
            certs = json.loads((out/'source_certificates.json').read_text())
            self.assertEqual(sum(c['signal_opened'] for c in certs), 21)
            self.assertEqual(sum(c['split'] == 'test' and not c['signal_opened'] for c in certs), 10)
            self.assertEqual(sum(c['status'] == 'duplicate_omitted' and not c['signal_opened'] for c in certs), 5)
            admission = json.loads((out/'source_admission.json').read_text())
            self.assertFalse(set(admission['sources']) & set(admission['excluded_source_sha256']))
            self.assertTrue(all(v['split'] == 'train' and not v['validation_allowed'] for v in admission['sources'].values()
                                if v.get('source_recording_id', '').startswith('zenodo:2392979:')))

    def test_all_QC_fail_reports_zero_new_data_without_copying_or_completion(self):
        with tempfile.TemporaryDirectory() as d:
            args, _, _, _ = preparation_fixture(Path(d), .2)
            result = prepare(args)
            out = Path(args.output)
            self.assertEqual(result['status'], 'blocked_no_eligible_new_source_windows')
            self.assertEqual(result['new_source_windows'], 0)
            self.assertEqual(result['reasons']['native_peak_above_fixed_0_1V_source_bound'], 21)
            self.assertFalse((out/'completion.json').exists())
            self.assertFalse((out/'base').exists())
            self.assertFalse((out/'manifest.jsonl').exists())
            self.assertTrue((out/'preparation_result.json').exists())


if __name__ == '__main__':
    unittest.main()
