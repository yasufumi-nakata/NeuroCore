import tempfile,unittest
from pathlib import Path
import numpy as np
import audit_compact_eeg as a


class TextQuantity(unittest.TestCase):
    def test_actual_rows_and_EEG_tokens_not_gyro_or_counter(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'user001_10_1.txt'
            lines=[]
            for i in range(6):lines.append(' '.join([str(i*.1),str((126+i)%128+1)]+['1.25']*14+['999','888']+['0']*5))
            p.write_text('\n'.join(lines)+'\n')
            r=a.audit_text(p)
            self.assertEqual(r['stored_signal_seconds'],6/128)
            self.assertEqual(r['eeg_numeric_payload_bytes'],6*14*4)
            self.assertEqual(r['eeg_scalar_count'],84)
            self.assertEqual(r['clock_audit']['counter_nonincrement_or_nonwrap_transitions'],0)
            self.assertFalse(r['physical_V_conversion_verified'])
            self.assertEqual(r['unit'],'unresolved_publisher_does_not_state_amplitude_unit')
            self.assertFalse(r['label_audit']['scheduled_trials'][0]['within_stored_record'])

    def test_nonfinite_and_wrong_width_are_not_quantity_verified(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'user001_10_1.txt';p.write_text('1 2 3\n')
            with self.assertRaisesRegex(ValueError,'fewer'):a.audit_text(p)
            p.write_text(' '.join(['0','1','nan']+['0']*20)+'\n')
            with self.assertRaisesRegex(ValueError,'Nonfinite'):a.audit_text(p)


class FIFQuantity(unittest.TestCase):
    def test_native_rate_payload_and_stim_labels_are_separated(self):
        import mne
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'subject1.raw.fif';x=np.zeros((3,1280));x[0]=1e-6;x[1]=2e-6
            x[2,100:110]=1;x[2,256:266]=2;x[2,768:778]=4
            r=mne.io.RawArray(x,mne.create_info(['EEG1','EEG2','STI'],128,['eeg','eeg','stim']),verbose=False)
            r.save(p,fmt='single',overwrite=False,verbose=False)
            q=a.audit_fif(p,['EEG1','EEG2'])
            self.assertEqual(q['stored_signal_seconds'],10)
            self.assertEqual(q['eeg_numeric_payload_bytes'],2*1280*4)
            self.assertEqual(q['non_EEG_signal_data_buffer_bytes'],1280*4)
            self.assertEqual(q['label_audit']['event_code_counts'],{'1':1,'2':1,'4':1})
            self.assertEqual([e['label'] for e in q['label_audit']['semantic_trial_events']],['right_hand_imagery','rest'])
            self.assertEqual(q['label_audit']['semantic_windows_outside_record'],0)
            self.assertEqual(q['decoded_payload_sha256'],a.decoded_sha(np.asarray(x[:2].T,dtype=np.float32).astype(float)))
            self.assertFalse(q['label_audit']['training_eligible_under_current_64channel_V_protocol'])


if __name__=='__main__':unittest.main()
