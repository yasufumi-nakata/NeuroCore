"""Leakage, complete selection and native-interval accounting fixtures."""
from pathlib import Path
import unittest
import benchmark_ssl_audit as audit

class SSLSelectionAuditTests(unittest.TestCase):
    def expected(self):return {str(i):{'subject_group':'a' if i==0 else 'b'} for i in range(4)}
    def values(self):return [{'id':str(i),'subject_group':'a' if i==0 else 'b','masked_MSE':2. if i==0 else 10.,'checkpoint_sha256':'fixed'} for i in range(4)]
    def test_subject_macro_is_not_window_weighted(self):
        m=audit.validation_metrics(self.values(),self.expected(),'fixed')
        self.assertEqual(m['mean_subject_masked_MSE'],6.);self.assertEqual(m['mean_window_masked_MSE'],8.)
    def test_all_validation_IDs_and_checkpoint_are_mandatory(self):
        for mutate in ('missing','duplicate','wrong_group','wrong_checkpoint'):
            with self.subTest(mutate=mutate):
                rows=self.values()
                if mutate=='missing':rows.pop()
                elif mutate=='duplicate':rows.append(dict(rows[0]))
                elif mutate=='wrong_group':rows[0]['subject_group']='future'
                else:rows[0]['checkpoint_sha256']='another'
                with self.assertRaises(ValueError):audit.validation_metrics(rows,self.expected(),'fixed')
    def test_nonfinite_negative_and_boolean_losses_rejected(self):
        for value in (float('nan'),float('inf'),-1.,True):
            with self.subTest(value=value):
                rows=self.values();rows[0]['masked_MSE']=value
                with self.assertRaises(ValueError):audit.validation_metrics(rows,self.expected(),'fixed')
    def test_initial_checkpoint_and_earliest_tie_win(self):
        initial={'epoch':0,'score':2.};same={'epoch':1,'score':2.,'selected_best':False};worse={'epoch':2,'score':3.,'selected_best':False}
        self.assertEqual(audit.earliest_minimum([initial,same,worse])['epoch'],0)
        better={'epoch':3,'score':1.,'selected_best':True};tie={'epoch':4,'score':1.,'selected_best':False}
        self.assertEqual(audit.earliest_minimum([initial,same,worse,better,tie])['epoch'],3)
        with self.assertRaises(ValueError):audit.earliest_minimum([initial,dict(same,selected_best=True)])
        with self.assertRaises(ValueError):audit.earliest_minimum([same,worse])
    def test_recorded_metric_cannot_override_actual_losses(self):
        actual=audit.validation_metrics(self.values(),self.expected(),'fixed')
        recorded={**actual,'mask_condition':'fixed_ID_seed_same_at_initial_and_every_epoch'}
        audit.check_recorded_metrics(actual,recorded)
        with self.assertRaises(ValueError):audit.check_recorded_metrics(actual,{**recorded,'mean_subject_masked_MSE':8.})
    def row(self,identity,source,payload,start,end,rate=160):
        return {'id':identity,'source_eeg_sha256':source,'eeg_payload_sha256':payload,'native_sample_rate_hz':rate,'raw_interval':{'native_sample_start':start,'native_sample_stop':end},'window_start_s':start/rate,'window_end_s':end/rate}
    def test_native_union_counts_overlap_once_and_deduplicates_payload_alias(self):
        rows=[self.row('a','source1','signal1',800,1280),self.row('b','source1','signal1',1120,1600),self.row('c','source_alias','signal1',800,1280)]
        result=audit.exposure(rows)
        self.assertEqual(result['native_source_interval_union_seconds'],8.)
        self.assertEqual(result['physical_payload_interval_union_seconds'],5.)
        self.assertEqual(result['unique_window_ids'],3)
    def test_128_and_160Hz_intervals_remain_physical_seconds(self):
        result=audit.exposure([self.row('a','160','one',800,1280),self.row('b','128','two',640,1024,128)])
        self.assertEqual(result['physical_payload_interval_union_seconds'],6.)
    def test_seen_membership_rejects_heldout_unknown_and_duplicate(self):
        original={'train':{'split':'train'},'val':{'split':'val'},'test':{'split':'test'}}
        self.assertEqual(audit.checked_seen(['train'],original),[original['train']])
        for ids in (['val'],['test'],['unknown'],['train','train']):
            with self.subTest(ids=ids),self.assertRaises(ValueError):audit.checked_seen(ids,original)

if __name__=='__main__':unittest.main()
