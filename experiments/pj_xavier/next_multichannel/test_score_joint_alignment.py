"""CPU-only protocol validation. Synthetic fixtures are not model results."""
import hashlib
import math
import sys
import unittest
from pathlib import Path
HERE=Path(__file__).absolute().parent
sys.path.insert(0,str(HERE))
import joint_alignment as j
import score_joint_alignment as s


def digest(x):return hashlib.sha256(x.encode()).hexdigest()

def registry(n=3):
    return [{
       'signal_trial_id_sha256':digest('t'+str(i)),
       'must_link_component_sha256':digest('g'+str(i//2)),
       'window_sha256':digest('window'+str(i)),
       'split':'val', 'original_source_cue_gold':'AB'[i%2],
       'label_provenance':'cue_original_plus_synthetic_environment_proxy',
       'training_exposure_proven_zero':True,
    } for i in range(n)]

def evaluate(rows, **kw):
    frozen=registry()
    return s.score(rows, frozen_registry=frozen,
         expected_registry_sha256=s.digest_registry(frozen), **kw)

def fixtures(n=3):
    out=[]
    for sc in s.SCENARIOS:
        for ix in range(n):
            for row in j.make_counterbalanced_trials('t'+str(ix),'AB'[ix%2],allowed_split='val'):
                p = .85 if row['synthetic_mapping_action_gold']=='X' else .15
                if sc not in ('model','correct_class_routing_upper_bound'):
                    p=.5
                if sc=='correct_class_routing_upper_bound':
                    p=float(row['synthetic_mapping_action_gold']=='X')
                out.append({**row,
                    'scenario':sc,'must_link_component_sha256':digest('g'+str(ix//2)),
                    'window_sha256':digest('window'+str(ix)),
                    'model_prob_x':p,
                    'prediction_receipt_sha256':digest(sc+'prediction'),
                    'model_checkpoint_sha256':digest('best'),
                    'intervention_receipt_sha256':digest(sc+'intervention'),
                    'os_action_executed':False})
    return out

class ContractTests(unittest.TestCase):
    def test_scores_paired_and_grouped(self):
        r=evaluate(fixtures(),bootstrap=200)
        self.assertEqual(r['n_unique_EEG_trials'],3)
        self.assertEqual(r['n_independent_must_link_components'],2)
        self.assertEqual(r['metrics']['model']['mean_group_balanced_accuracy'],1)
        self.assertEqual(r['metrics']['context_only']['mean_group_balanced_accuracy'],.5)
        self.assertEqual(r['metrics']['correct_class_routing_upper_bound']['mean_group_balanced_accuracy'],1)
        self.assertEqual(r['paired_group_bootstrap_95pct']['lower'],.5)
        self.assertFalse(r['free_intent_evaluated'])
    def test_reject_missing_controls(self):
        with self.assertRaisesRegex(ValueError,'controls'):
            evaluate([r for r in fixtures() if r['scenario']!='EEG_removed'])
    def test_reject_heldout_change(self):
        x=fixtures();x[0]['split']='train'
        with self.assertRaises(ValueError):evaluate(x)
    def test_reject_prob_nan(self):
        x=fixtures();x[0]['model_prob_x']=float('nan')
        with self.assertRaises(ValueError):evaluate(x)
    def test_reject_gold_leak(self):
        x=fixtures();x[0]['input_environment']['TextResult']='future'
        with self.assertRaises(ValueError):evaluate(x)
    def test_reject_layout_geometry_forgery(self):
        x=fixtures()
        x[0]['input_environment']['slot_to_id']={'left':'Y','right':'X'}
        x[0]['synthetic_mapping_action_gold']=j.synthetic_action_gold(
              x[0]['original_source_cue_gold'],x[0]['input_environment'])
        with self.assertRaisesRegex(ValueError,'geometry'):evaluate(x)

    def test_reject_missing_frozen_trial_even_if_all_controls_pair(self):
        x=[r for r in fixtures() if r['signal_trial_id_sha256']!=digest('t2')]
        with self.assertRaisesRegex(ValueError,'registry trials'):evaluate(x)

    def test_reject_oracle_ceiling_incorrect(self):
        x=fixtures();r=next(z for z in x if z['scenario']=='correct_class_routing_upper_bound')
        r['model_prob_x']=.5
        with self.assertRaisesRegex(ValueError,'ceiling'):evaluate(x)
    def test_reject_duplicate_layout(self):
        x=fixtures();x.append(x[0].copy())
        with self.assertRaisesRegex(ValueError,'Repeated'):evaluate(x)
    def test_reject_unpaired_negative_control(self):
        x=fixtures();x=[r for r in x if not (r['scenario']=='EEG_removed' and r['input_environment']['layout_id']=='normal' and r['signal_trial_id_sha256']==digest('t0'))]
        with self.assertRaisesRegex(ValueError,'paired'):evaluate(x)
    def test_reject_false_intent_or_os_claim(self):
        x=fixtures();x[0]['os_action_executed']=True
        with self.assertRaisesRegex(ValueError,'intent'):evaluate(x)
    def test_reject_receipt_reuse(self):
        x=fixtures();forbidden=digest('modelintervention')
        for row in x:
            if row['scenario']=='EEG_removed':row['intervention_receipt_sha256']=forbidden
        with self.assertRaisesRegex(ValueError,'Different control'):evaluate(x)
    def test_reject_consistently_forged_gold(self):
        x=fixtures()
        for row in x:
            if row['signal_trial_id_sha256']==digest('t0'):
                row['original_source_cue_gold']='B'
                row['synthetic_mapping_action_gold']=j.synthetic_action_gold('B',row['input_environment'])
        with self.assertRaisesRegex(ValueError,'frozen registry'):evaluate(x)

    def test_reject_changed_registry_hash(self):
        frozen=registry()
        frozen[0]['original_source_cue_gold']='B'
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):
            s.score(fixtures(),frozen_registry=frozen,expected_registry_sha256=s.digest_registry(registry()))

    def test_reject_shuffled_cue_gold(self):
        x=fixtures();x[0]['synthetic_mapping_action_gold']='Y' if x[0]['synthetic_mapping_action_gold']=='X' else 'X'
        with self.assertRaisesRegex(ValueError,'gold mismatch'):evaluate(x)

if __name__=='__main__':unittest.main()
