import copy
import unittest
from audit_external_completion import verify_predictions

class CompletionAuditTests(unittest.TestCase):
    def setUp(self):
        self.truth={str(i):{'label':label} for i,label in enumerate(['ai_prompt_submit','keyboard','mouse_click'])}
        self.predictions=[]
        for i,label in enumerate(['ai_prompt_submit','keyboard','mouse_click']):
            probability={k:.05 for k in ['ai_prompt_submit','keyboard','mouse_click']};probability[label]=.9
            self.predictions.append({'id':str(i),'true_label':label,'variant':'fused','checkpoint_sha256':'a'*64,'probabilities':probability,'abstain':False})
    def test_direct_arithmetic(self):
        result=verify_predictions(self.predictions,self.truth,'a'*64,'fused')
        self.assertEqual(result['forced_confusion_matrix'],[[1,0,0],[0,1,0],[0,0,1]])
        self.assertEqual(result['forced_balanced_accuracy'],1.)
    def test_missing_and_duplicate_ids_rejected(self):
        for value in (self.predictions[:-1],[self.predictions[0]]*3):
            with self.assertRaises(ValueError):verify_predictions(value,self.truth,'a'*64,'fused')
    def test_wrong_checkpoint_truth_and_threshold_rejected(self):
        for key,value in [('checkpoint_sha256','b'*64),('true_label','keyboard'),('abstain',True)]:
            altered=copy.deepcopy(self.predictions);altered[0][key]=value
            with self.assertRaises(ValueError):verify_predictions(altered,self.truth,'a'*64,'fused')
    def test_binary_preserves_full_observed_truth(self):
        values=[]
        for i,row in self.truth.items():values.append({'id':i,'true_label':row['label'],'variant':'history','model_artifact_sha256':'b'*64,'probabilities':{'ai_prompt_submit':.7,'other_observed_action':.3}})
        self.assertEqual(verify_predictions(values,self.truth,'b'*64,'history',False)['records'],3)
        values[0]['probabilities']={'ai_prompt_submit':.7,'fake':.3}
        with self.assertRaises(ValueError):verify_predictions(values,self.truth,'b'*64,'history',False)
if __name__=='__main__':unittest.main()
