import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

import prepare_smac_inmid as a


def make_edf(path, index=0, amplitude_ad=16, auxiliary_value=0, unit='uV'):
    names = a.native.PREFIX+a.CHANNELS+['RAW_CQ', 'BATTERY', 'BATTERY_PERCENT', 'MarkerIndex',
        'MarkerType', 'MarkerValueInt', 'MARKER_HARDWARE']+[f'aux{i}' for i in range(33)]
    n, records = 58, 12
    prefix = (b'0       '+b' '*160+b'24.07.23'+b'00.00.00'+str((n+1)*256).ljust(8).encode()
              +b' '*44+str(records).ljust(8).encode()+b'1       '+str(n).ljust(4).encode())
    fields = [(names, 16), (['']*n, 80), ([unit]*n, 8), (['0']*n, 8), (['16000']*n, 8),
              (['0']*n, 8), (['31200']*n, 8), (['']*n, 80), (['128']*n, 8), (['']*n, 32)]
    header = prefix+b''.join(b''.join(str(v).ljust(width).encode() for v in vals) for vals, width in fields)
    t = np.arange(records*128)/128
    data = np.full((58, records*128), auxiliary_value, dtype='<i2')
    data[4:18] = np.rint(9000+np.stack([amplitude_ad*np.sin(2*np.pi*(8+c/20+index/100)*t+index/40)
                                     for c in range(14)])).astype('<i2')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(header)
        for start in range(0, records*128, 128):
            f.write(np.ascontiguousarray(data[:, start:start+128]).tobytes())


def audit_record(path, relative):
    h = a.native.header(path)
    group, session, recording = a.native.source_group(path.name)
    result = a.native.stream(path, h, a.sha(path))
    return {'source_path': relative, 'source_bytes': path.stat().st_size, 'subject_group': group,
            'source_session': session, 'source_recording_in_session': recording, **h, **result,
            'MNE_header_and_three_EEG_calibration_probes_passed': True, 'MNE_max_probe_difference_V': 0.}


def metadata_for(ledger):
    return [{'source': 'inmid14', 'source_group': 'inmid14|'+r['subject_group'],
             'source_member_or_path': r['source_path'], 'source_file_sha256': r['source_sha256'],
             'eeg_payload_sha256': r['EEG_payload_sha256'], 'calibration_sha256': r['calibration_sha256'],
             'metadata_only_proposed_split': 'test' if r['subject_group'] in a.TEST_GROUPS else 'train',
             'component_id_sha256': hashlib.sha256(r['subject_group'].encode()).hexdigest(), 'fit_allowed': False}
            for r in ledger]


def planned_fixture():
    ledger = []
    for n in range(1, 24):
        for session in (1, 2):
            for recording in (1, 2):
                i = len(ledger)+1
                ledger.append({'source_path': f'source_67710cd9f99e/2023-07-24-P{n:02d}-{session}-{recording}.edf',
                    'subject_group': a.GROUP_PREFIX+f'{n:02d}', 'source_session': session,
                    'source_recording_in_session': recording, 'source_sha256': f'{i:064x}',
                    'EEG_payload_sha256': f'{i+100:064x}', 'decoded_EEG_V_sha256': f'{i+200:064x}',
                    'calibration_sha256': 'c'*64, 'channels': 58, 'EEG_channels': a.CHANNELS,
                    'EEG_indices': list(range(4, 18)), 'auxiliary_indices': list(range(4))+list(range(18, 58)),
                    'sample_rate_hz': 128, 'sample_count': 1536, 'MNE_header_and_three_EEG_calibration_probes_passed': True,
                    'MNE_max_probe_difference_V': 0.})
    return ledger, metadata_for(ledger)


def make_base(root, arbitrary_train_source=True):
    base = root/'base';base.mkdir()
    manifest, sources, artifacts = [], {}, {}
    for i, fold in enumerate(('train', 'val')):
        rel = fold+'.npy';source, payload = f'{1000+i:064x}', f'{1100+i:064x}'
        np.save(base/rel, np.full((1, 16, 3, 200), .03, np.float32), allow_pickle=False)
        group = 'zenodo:806023:base_'+fold
        recording = 'zenodo:806023:base_'+fold
        if arbitrary_train_source and fold == 'train':
            group, recording = 'synthetic_prior_source:group', 'synthetic_prior_source:record'
        sources[source] = {'split': fold, 'subject_group': group, 'eeg_payload_sha256': payload,
            'montage_key': 'base16', 'physical_unit_verified': True, 'fit_allowed': fold == 'train',
            'validation_allowed': fold == 'val'}
        artifacts[rel] = {'shape': [1, 16, 3, 200], 'dtype': 'float32', 'sha256': a.sha(base/rel), 'bytes': (base/rel).stat().st_size}
        manifest.append({'id': source+':0', 'split': fold, 'subject_group': group, 'source_sha256': source,
            'eeg_payload_sha256': payload, 'source_recording_id': recording, 'window_start_s': 2., 'window_end_s': 5.,
            'channels': ['C'+str(c) for c in range(16)], 'sample_rate_hz': 200, 'unit': 'V_x_1e4',
            'eeg_path': rel, 'eeg_index': 0, 'montage_key': 'base16', 'reference': 'synthetic',
            'native_sample_rate_hz': 512, 'physical_unit_verified': True, 'online_harness_allowed': False})
    (base/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    a.dump(base/'artifacts.json', artifacts)
    a.dump(base/'source_admission.json', {'sources': sources, 'excluded_source_sha256': [], 'excluded_payload_sha256': []})
    a.dump(base/'protocol.json', {'synthetic': True})
    completion = {'status': 'prepared_verified_new_offline_SSL_corpus'}
    for filename, key in [('manifest.jsonl', 'manifest_sha256'), ('artifacts.json', 'artifacts_sha256'),
                          ('source_admission.json', 'source_admission_sha256'), ('protocol.json', 'protocol_sha256')]:
        completion[key] = a.sha(base/filename)
    a.dump(base/'completion.json', completion)
    return base, a.sha(base/'completion.json'), manifest


def preparation_fixture(root, amplitude_ad=16):
    ledger, _ = planned_fixture()
    raw = root/'raw';raw.mkdir()
    for i, row in enumerate(ledger):
        if row['subject_group'] in a.TEST_GROUPS:
            continue  # Test files intentionally do not exist.
        file = raw/row['source_path']
        make_edf(file, i, amplitude_ad=amplitude_ad)
        ledger[i] = audit_record(file, row['source_path'])
    metadata = metadata_for(ledger)
    lp, mp, ap, dp = root/'ledger.jsonl', root/'metadata.jsonl', root/'audit_protocol.json', root/'deny.json'
    lp.write_text(''.join(json.dumps(r)+'\n' for r in ledger));mp.write_text(''.join(json.dumps(r)+'\n' for r in metadata))
    a.dump(ap, {'source_official_version': 2, 'source_sha256': a.AUDIT_MODULE_SHA256,
                'EDF_header_helper_sha256': a.EDF_HELPER_SHA256})
    a.dump(dp, {'excluded_source_sha256': ['f'*64]})
    base, digest, bm = make_base(root)
    args = SimpleNamespace(raw_root=str(raw), source_ledger=str(lp), metadata_admission=str(mp),
        audit_protocol=str(ap), base_corpus=str(base), expected_base_completion_sha256=digest,
        deny_json=[str(dp)], output=str(root/'prepared'), reserve_gib=1)
    # Tests register their own synthetic input fingerprints; production CLI has no bypass.
    seals = {'LEDGER_SHA256': a.sha(lp), 'METADATA_SHA256': a.sha(mp), 'AUDIT_PROTOCOL_SHA256': a.sha(ap)}
    return args, ledger, bm, seals


class InMIDPreparationTests(unittest.TestCase):
    def test_frozen_train76_test16_and_split_change_rejected(self):
        ledger, metadata = planned_fixture()
        plan = a.plan_records(ledger, metadata, set(), set())
        self.assertEqual(sum(p['status'] == 'candidate' for p in plan), 76)
        self.assertEqual(sum(p['reason'] == 'new_InMID_test_signal_not_opened' for p in plan), 16)
        metadata[0]['metadata_only_proposed_split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'Frozen InMID split'):
            a.plan_records(ledger, metadata, set(), set())

    def test_digital_and_physical_denies_propagate_to_group_and_duplicates_fail(self):
        ledger, metadata = planned_fixture()
        row = ledger[4]
        for identity in (row['EEG_payload_sha256'], row['decoded_EEG_V_sha256']):
            plan = a.plan_records(ledger, metadata, set(), {identity})
            self.assertEqual(sum(p['status'] == 'excluded_prior_identity' for p in plan), 4)
        ledger[5]['decoded_EEG_V_sha256'] = ledger[4]['decoded_EEG_V_sha256']
        with self.assertRaisesRegex(ValueError, 'must-link'):
            a.plan_records(ledger, metadata, set(), set())

    def test_physical_V_decoding_excludes_generic_uV_aux_and_matches_MNE(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'2023-07-24-P02-1-1.edf'
            make_edf(p, auxiliary_value=0);r = audit_record(p, p.name);x, proof = a.verify_edf(p, r)
            make_edf(p, auxiliary_value=30000);s = audit_record(p, p.name);y, _ = a.verify_edf(p, s)
            np.testing.assert_array_equal(x, y)
            self.assertEqual(x.shape, (14, 1536));self.assertEqual(proof['auxiliary_channels_excluded'], 44)
            self.assertTrue(proof['MNE_header_and_three_EEG_calibration_probes_passed'])
            self.assertNotEqual(r['source_sha256'], s['source_sha256'])
            self.assertEqual(r['decoded_EEG_V_sha256'], s['decoded_EEG_V_sha256'])
            self.assertFalse(proof['native_marker_or_label_used'])
            with self.assertRaisesRegex(ValueError, 'Original EDF SHA'):
                a.verify_edf(p, r)

    def test_mislabeled_physical_unit_is_not_corrected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'2023-07-24-P02-1-1.edf';make_edf(p);r = audit_record(p, p.name)
            make_edf(p, unit='mV')
            with self.assertRaisesRegex(ValueError, 'physical unit'):
                a.verify_edf(p, r)

    def test_128Hz_causality_fixed_QC_and_no_added_duration(self):
        t = np.arange(128*12)/128
        x = np.stack([8e-6*np.sin(2*np.pi*(8+c/20)*t) for c in range(14)])
        before, js, cutoff = a.causal_transform128(x)
        changed = x.copy();changed[:, 8*128:] += 20e-6
        after, _, _ = a.causal_transform128(changed)
        np.testing.assert_array_equal(before[:, cutoff < 8*128], after[:, cutoff < 8*128])
        self.assertTrue(np.all(cutoff/128 <= js/200))
        array, windows, _ = a.causal_windows14(x)
        self.assertEqual(array.shape, (3, 14, 3, 200))
        self.assertEqual(sum(w['window_end_s']-w['window_start_s'] for w in windows), 9.)
        self.assertLessEqual(windows[-1]['window_end_s'], x.shape[1]/128)
        array, _, rejected = a.causal_windows14(x*100)
        self.assertEqual(len(array), 0);self.assertGreater(rejected['peak_above_fixed_100uV'], 0)
        array, _, rejected = a.causal_windows14(np.zeros_like(x))
        self.assertEqual(len(array), 0);self.assertGreater(rejected['flat_below_fixed_0_01uV'], 0)
        with self.assertRaisesRegex(ValueError, 'fixed_0_1V_source_bound'):
            a.causal_windows14(x+1.)

    def test_arbitrary_verified_train_base_supported_and_changed_array_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            base, digest, _ = make_base(Path(d))
            a.validate_base(base, digest)
            with (base/'train.npy').open('ab') as f:f.write(b'corrupt')
            with self.assertRaises(ValueError):a.validate_base(base, digest)

    def test_full_synthetic_EDF76_merge_preserves_val_and_skips_test16_and_nonEEG_files(self):
        with tempfile.TemporaryDirectory() as d:
            args, ledger, bm, seals = preparation_fixture(Path(d))
            with mock.patch.multiple(a, **seals):result = a.prepare(args)
            out = Path(args.output)
            self.assertEqual(result['status'], 'prepared_verified_new_offline_SSL_corpus')
            self.assertEqual(result['new_source_records'], 76)
            self.assertEqual(result['new_source_windows'], 228)
            self.assertEqual(result['windows'], {'train': 229, 'val': 1})
            self.assertEqual(result['MAT_video_auxiliary_or_labels_independent_EEG_hours_added'], 0)
            manifest = a.rows(out/'manifest.jsonl')
            expected_val = [{**r, 'eeg_path': 'base/'+r['eeg_path']} for r in bm if r['split'] == 'val']
            self.assertEqual([r for r in manifest if r['split'] == 'val'], expected_val)
            certs = json.loads((out/'source_certificates.json').read_text())
            self.assertEqual(sum(c['signal_opened'] for c in certs), 76)
            self.assertEqual(sum(c['split'] == 'test' and not c['signal_opened'] for c in certs), 16)
            admission = json.loads((out/'source_admission.json').read_text())
            for r in ledger:
                if r['subject_group'] in a.TEST_GROUPS:
                    self.assertIn(r['source_sha256'], admission['excluded_source_sha256'])
                    self.assertTrue(a.payloads(r) <= set(admission['excluded_payload_sha256']))
            protocol = json.loads((out/'protocol.json').read_text())
            self.assertFalse(protocol['preprocessing']['additional_LP80_used'])
            self.assertFalse(protocol['non_EEG_files_required_for_preparation'])
            self.assertFalse(protocol['source_causality_verified'])
            # prepare also executes the real trainer's byte/provenance loader before completion.
            a.trainer.load_corpus(out/'manifest.jsonl', out/'artifacts.json', out/'source_admission.json')

    def test_all_new_windows_rejected_has_receipt_but_no_trainable_completion(self):
        with tempfile.TemporaryDirectory() as d:
            args, _, _, seals = preparation_fixture(Path(d), amplitude_ad=6000)
            with mock.patch.multiple(a, **seals):result = a.prepare(args)
            out = Path(args.output)
            self.assertEqual(result['status'], 'blocked_no_eligible_new_source_windows')
            self.assertEqual(result['new_source_windows'], 0)
            self.assertFalse((out/'completion.json').exists());self.assertFalse((out/'base').exists())
            self.assertFalse((out/'manifest.jsonl').exists());self.assertTrue((out/'preparation_result.json').exists())


if __name__ == '__main__':
    unittest.main()
