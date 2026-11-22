import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import score_external_predictions as api


class ExternalScoreTests(unittest.TestCase):
    def test_abstentions_are_false_negatives_without_changing_forced_accuracy(self):
        result=api.multiclass_metrics(np.array([0,1,2]),np.array([0,1,2]),np.array([True,False,True]))
        self.assertEqual(result['forced_balanced_accuracy'],1)
        self.assertAlmostEqual(result['strict_balanced_accuracy'],1/3)
        self.assertAlmostEqual(result['strict_macro_f1'],1/3)
        self.assertEqual(result['selective_accuracy'],1)

    def test_collection_requires_full_identity_truth_threshold_and_artifact(self):
        rows=[{'id':'e1','label':'keyboard'},{'id':'e2','label':'mouse_click'}]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'predictions.jsonl'
            predictions=[{'id':r['id'],'true_label':r['label'],'variant':'fused','probabilities':{'ai_prompt_submit':.2,'keyboard':.5,'mouse_click':.3},'checkpoint_sha256':'f'*64,'abstain':True} for r in rows]
            def write():path.write_text(''.join(json.dumps(r)+'\n' for r in predictions));return {'m':{'predictions_path':str(path),'predictions_sha256':api.history.sha(path),'expected_conditions':['fused'],'checkpoint_sha256':'f'*64}}
            values,_=api.collect_predictions(write(),rows);self.assertEqual(len(values['m::fused']),2)
            predictions[0]['abstain']=False
            with self.assertRaisesRegex(ValueError,'threshold'):api.collect_predictions(write(),rows)
            predictions[0]['abstain']=True;predictions[0]['id']='training1'
            with self.assertRaisesRegex(ValueError,'nonexternal'):api.collect_predictions(write(),rows)


if __name__=='__main__':unittest.main()
