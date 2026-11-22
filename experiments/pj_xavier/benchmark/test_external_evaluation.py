import copy
import hashlib
import json
from pathlib import Path
import tempfile
import sys
import unittest

import numpy as np
import external_evaluation as api


def row(rid='external1', start=15):
    end = start+1.75
    return {'id': rid, 'dataset': 'source', 'subject': 'itu:p1', 'session': 's1', 'recording_id': 'itu:r1',
            'site': 'itu', 'split': 'external_test', 'data_origin': 'real_recording', 'label_source': 'recorded_event',
            'target_kind': 'observed_action', 'label': 'keyboard', 'channels': api.CHANNELS, 'sample_rate_hz': 256,
            'window_start_s': start, 'window_end_s': end, 'decision_time_s': end+4.125, 'event_time_s': end+5,
            'environment_time_s': start-.2, 'clock_uncertainty_guard_s': .1, 'feature_available_time_s': end+4.125,
            'environment_source': 'predecision_harness', 'environment_contains_target_label': False,
            'environment_text': 'past context', 'clock_sync': {'verified': True}, 'preprocessing': {'online_causality_verified': False},
            'eeg_source_sha256': 'a'*64, 'label_evidence_sha256': 'b'*64, 'environment_sha256': 'c'*64,
            'history_pre_eeg': {'cutoff_s': start-.1, 'max_source_onset_used_s': start-1, 'target_or_future_events_included': False},
            'history_pre_decision_guarded': {'cutoff_s': end+4.025, 'max_source_onset_used_s': start-1, 'target_or_future_events_included': False}}


class ExternalContractTests(unittest.TestCase):
    def setUp(self):
        self.original = [{'id': 'train1', 'eeg_source_sha256': '0'*64}]

    def test_separate_contract_accepts_real_external_observations_without_relabeling_split(self):
        r = row(); before = copy.deepcopy(r)
        result = api.verify_external_rows([r], self.original, {'keyboard': 1})
        self.assertEqual(r, before)
        self.assertFalse(result['human_identity_disjointness_verified'])
        self.assertEqual(r['split'], 'external_test')

    def test_original_source_overlap_rejected_even_with_site_qualified_names(self):
        r = row(); r['eeg_source_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'raw EEG bytes overlap'):
            api.verify_external_rows([r], self.original, {'keyboard': 1})

    def test_split_and_pseudo_label_and_future_context_rejected(self):
        for field, value in [('split', 'train'), ('label_source', 'ai_pseudo'), ('environment_time_s', 99)]:
            with self.subTest(field=field):
                r = row(); r[field] = value
                with self.assertRaises(ValueError):
                    api.verify_external_rows([r], self.original, {'keyboard': 1})

    def test_overlapping_external_windows_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            api.verify_external_rows([row(), row('other', 16)], self.original, {'keyboard': 2})

    def test_shuffle_identity_or_cross_record_rejected(self):
        rows = [row(), row('external2', 20)]
        predictions = [{'id': r['id'], 'variant': 'eeg_shuffled', 'eeg_source_id': rows[1-i]['id'],
                        'perturbation': {'original_eeg_id': r['id'], 'eeg_source_id': rows[1-i]['id']}}
                       for i, r in enumerate(rows)]
        self.assertEqual(api.check_shuffle(predictions, rows)['fixed_points'], 0)
        changed = copy.deepcopy(predictions); changed[0]['perturbation']['eeg_source_id'] = rows[0]['id']
        with self.assertRaises(ValueError): api.check_shuffle(changed, rows)
        rows[1]['recording_id'] = 'itu:r2'
        with self.assertRaisesRegex(ValueError, 'crosses'): api.check_shuffle(predictions, rows)


class ExternalRawEvidenceTests(unittest.TestCase):
    def fixture(self, root):
        rawroot = root/'raw'; rawroot.mkdir()
        prepared = root/'prepared'; prepared.mkdir()
        output = root/'evaluation'; output.mkdir()
        data = (np.arange(8*30*256, dtype=np.float32).reshape(30*256, 8)/100).astype('<f4')
        binary = rawroot/'test.eeg'; data.tofile(binary)
        header = rawroot/'test.vhdr'; marker = rawroot/'test.vmrk'
        header.write_text('Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\nDataFile=test.eeg\nMarkerFile=test.vmrk\nDataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=8\nSamplingInterval=3906.25\n[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\n'+''.join(f'Ch{i+1}={ch},,1,V\n' for i,ch in enumerate(api.CHANNELS)))
        marker.write_text('Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\nDataFile=test.eeg\n[Marker Infos]\n')
        r = row(); r['eeg_source_sha256'] = api.sha(binary); r['eeg_source_uri'] = 'https://example.invalid/archive#test.eeg'
        r['eeg_header_sha256'] = api.sha(header)
        r['eeg_path'] = 'windows.npy'; r['eeg_index'] = 0
        np.save(prepared/'windows.npy', data[15*256:int(16.75*256)].T[None].copy())
        artifacts = {'windows.npy': {'sha256': api.sha(prepared/'windows.npy'), 'bytes': (prepared/'windows.npy').stat().st_size}}
        (prepared/'source_members_verified.json').write_text(json.dumps({p.name:{'sha256':api.sha(p),'bytes':p.stat().st_size} for p in (header,binary,marker)}))
        verification = rawroot/'source_verification.json';verification.write_text(json.dumps({'verified': True, 'sources': [{'filename': 'bids_'+kind+'_itu.tar.gz', 'ledger_status': 'complete', 'reverified_before_extraction': True} for kind in ['eeg', 'behavioral']]}))
        cache = api.shared.EvidenceCache(root)
        cert, report = api.prepare_external_shifts([r], prepared, artifacts, rawroot, output, cache, '1'*64, '2'*64, '3'*64, api.sha(verification))
        self.assertEqual(report['status'], 'ready', report)
        return r, prepared, artifacts, cache, cert[r['id']]

    def test_actual_raw_shift_independently_matches_external_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, prepared, artifacts, cache, detail = self.fixture(Path(tmp))
            check = api.verify_external_shift(detail, r, prepared, artifacts, cache, '1'*64, '2'*64, '3'*64)
            self.assertEqual(check['sample_interval'], [5*256, int(6.75*256)])
            self.assertFalse(detail['source']['external_arrays_sealed_in_checkpoint'])

    def test_forged_shift_or_external_training_ownership_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r, prepared, artifacts, cache, detail = self.fixture(Path(tmp))
            for key, value in [('start_sample', 6*256), ('external_arrays_sealed_in_checkpoint', True), ('recording_identity', ['other']*4), ('raw_eeg_sha256', 'f'*64)]:
                with self.subTest(key=key):
                    modified = copy.deepcopy(detail); modified['source'][key] = value
                    with self.assertRaises(ValueError): api.verify_external_shift(modified, r, prepared, artifacts, cache, '1'*64, '2'*64, '3'*64)

    def test_shift_array_changed_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);r, prepared, artifacts, cache, detail = self.fixture(root)
            path = root/detail['source']['extracted_array_path'];x=np.load(path);x[0,0]+=1;np.save(path,x)
            with self.assertRaises(ValueError): api.verify_external_shift(detail, r, prepared, artifacts, api.shared.EvidenceCache(root), '1'*64, '2'*64, '3'*64)


class FrozenModelAPICompatibilityTests(unittest.TestCase):
    def test_both_model_apis_keep_external_split_ids_and_weights_in_all_five_conditions(self):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'model'))
        import eeg_llm as legacy
        import cbramod_token_llm as symbol
        import test_cbramod_token_llm as fixtures
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.random.default_rng(14).normal(size=(3, 8, 448)).astype(np.float32)*1e-5
            np.save(root/'original.npy', x)
            external = [dict(row('external'+str(i), 15+i*3), label=label, eeg_path=str(root/'original.npy'), eeg_index=i) for i,label in enumerate(api.CLASSES)]
            config = {'mode': 'fused', 'max_prompt_tokens': 256, 'candidate_batch_size': 4, 'memory_budget_gib': 8, 'abstain_threshold': .8,
                      'dtype': 'float32', 'model': 'offline-tiny-random', 'seed': 19}
            base, tokenizer, _ = legacy.load_backbone(config, torch.device('cpu'))
            candidates = [(legacy, legacy.EEGCausalModel(base, 8, 2, 8), tokenizer),
                          (symbol, fixtures.tiny_model(), fixtures.TestTokenizer())]
            for number,(model_api, model, tokenizer) in enumerate(candidates):
                with self.subTest(api=model_api.SCHEMA_VERSION):
                    destination = root/str(number);destination.mkdir()
                    certificate = {}
                    for i,r in enumerate(external):
                        path = destination/(str(i)+'.npy');np.save(path, x[(i+1)%3])
                        certificate[r['id']] = {'method': 'fixture_only', 'eeg_source_id': 'fixture-shift-'+str(i), 'source': {'extracted_array_path': str(path.relative_to(root))}}
                    before = {k:v.detach().clone() for k,v in model.state_dict().items()}
                    state = {'schema': model_api.SCHEMA_VERSION, 'connector': model.connector.state_dict(), 'trainable_base': {}}
                    api.shared.restore_loaded_state(model_api, state, model)
                    conditions = api.shared.evaluate_conditions(model_api, model, tokenizer, external, api.CLASSES, model_api.EEGStore(448), config,
                        torch.device('cpu'), 'f'*64, destination, certificate, {'status':'ready'}, root)
                    self.assertEqual(set(conditions), set(api.shared.VARIANTS))
                    self.assertTrue(all(v['status']=='measured' and v['n']==3 for v in conditions.values()), conditions)
                    self.assertTrue(all(r['split']=='external_test' for r in external))
                    for key,value in model.state_dict().items(): torch.testing.assert_close(value, before[key], rtol=0, atol=0)
                    predictions = api.rows(destination/'predictions.jsonl')
                    self.assertEqual(len(predictions), 15)
                    self.assertTrue(all(not p['action_executed'] for p in predictions))
                    self.assertEqual(api.check_shuffle(predictions, external)['records_checked'], 3)


if __name__ == '__main__':
    unittest.main()
