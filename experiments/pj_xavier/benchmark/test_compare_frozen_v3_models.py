import json
from pathlib import Path
import tempfile
import unittest

import compare_frozen_v3_models as api


class RecordedPredictionTests(unittest.TestCase):
    def fixture(self, folder):
        checkpoint = 'a' * 64
        report = {'status': 'completed', 'source_manifest_sha256': api.MANIFEST_SHA,
                  'checkpoint_sha256': checkpoint, 'variant_names': api.VARIANTS, 'conditions': {}}
        for variant in api.VARIANTS:
            rows = [{'id': rid, 'variant': variant, 'checkpoint_sha256': checkpoint,
                     'true_label': 'ai_prompt_submit', 'probabilities': {'ai_prompt_submit': .3, 'other': .7}}
                    for rid in ['heldout1', 'heldout2']]
            path = folder / (variant + '_predictions.jsonl')
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            report['conditions'][variant] = {'status': 'measured', 'prediction_sha256': api.sha(path),
                                             'checkpoint_sha256': checkpoint}
        (folder / 'variant_evaluation_report.json').write_text(json.dumps(report))
        return checkpoint

    def test_derivation_only_changes_display_name_and_adds_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            checkpoint = self.fixture(folder)
            rows, provenance = api.derived_rows('model1', folder, checkpoint, {'heldout1', 'heldout2'})
            self.assertEqual(len(rows), 10)
            for row in rows:
                self.assertEqual(row['probabilities'], {'ai_prompt_submit': .3, 'other': .7})
                self.assertEqual(row['true_label'], 'ai_prompt_submit')
                self.assertEqual(row['variant'], 'model1::' + row['original_variant'])
                self.assertEqual(row['source_prediction_sha256'], provenance['conditions'][row['original_variant']]['original_prediction_sha256'])

    def test_changed_source_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            checkpoint = self.fixture(folder)
            path = folder / 'fused_predictions.jsonl'
            path.write_text(path.read_text().replace('0.3', '0.4'))
            with self.assertRaisesRegex(ValueError, 'hash'):
                api.derived_rows('model1', folder, checkpoint, {'heldout1', 'heldout2'})

    def test_heldout_id_replacement_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            checkpoint = self.fixture(folder)
            with self.assertRaisesRegex(ValueError, 'held-out IDs'):
                api.derived_rows('model1', folder, checkpoint, {'heldout1', 'training1'})

    def test_mixed_checkpoint_row_rejected_even_with_refreshed_file_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            checkpoint = self.fixture(folder)
            path = folder / 'eeg_shuffled_predictions.jsonl'
            path.write_text(path.read_text().replace(checkpoint, 'b' * 64))
            report_path = folder / 'variant_evaluation_report.json'
            report = json.loads(report_path.read_text())
            report['conditions']['eeg_shuffled']['prediction_sha256'] = api.sha(path)
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'mixed'):
                api.derived_rows('model1', folder, checkpoint, {'heldout1', 'heldout2'})

    def test_exact_history_variants_and_no_silent_empty_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'history.jsonl'
            source.write_text(''.join(json.dumps({'id': rid, 'variant': name}) + '\n' for name in api.HISTORY_VARIANTS for rid in ['x', 'y']))
            self.assertEqual(len(api.binary_history_lines(source, {'x', 'y'})), 8)
            source.write_text(source.read_text().replace('ai_vs_other:', 'binary:'))
            with self.assertRaisesRegex(ValueError, 'four exact'):
                api.binary_history_lines(source, {'x', 'y'})


if __name__ == '__main__':
    unittest.main()
