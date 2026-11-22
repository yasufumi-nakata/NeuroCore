import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import audit_ssvep_eeglab_v2 as a


def fixture(n=5):
    return {'nbchan':73,'pnts':n,'trials':1,'srate':200.,'xmin':0.,'xmax':(n-1)/200,
            'chanlocs':[{'labels':x,'type':''} for x in a.CHANNELS],
            'data':'s1_exp1.fdt','datfile':'s1_exp1.fdt','event':np.array([])}


class AuditTests(unittest.TestCase):
    def test_matlab_empty_channel_type_is_not_text(self):
        e=fixture()
        for ch in e['chanlocs']:ch['type']=np.array([],dtype=float)
        h=a.header(e,'s1_exp1.set',5*73*4)
        self.assertEqual(h['EEG_channels'],64)
        e['chanlocs'][0]['type']=np.array(['EOG'])
        with self.assertRaises(ValueError):a.header(e,'s1_exp1.set',5*73*4)

    def test_duration_and_auxiliary_byte_partition(self):
        e=fixture();h=a.header(e,'s1_exp1.set',5*73*4)
        self.assertEqual(h['stored_sample_duration_s'],.025)
        self.assertEqual(h['declared_last_sample_time_s'],.020)
        self.assertEqual(h['EEG_indices'],list(range(64)))
        self.assertIsNone(h['explicit_amplitude_unit'])

    def test_companion_escape_and_interpretation_changes_rejected(self):
        mutations=[{'data':'../s1_exp1.fdt'},{'datfile':'s2_exp1.fdt'},{'event':[{'type':1}]},
                   {'units':'uV'},{'trials':2},{'pnts':5.5}]
        for change in mutations:
            with self.subTest(change=change),self.assertRaises(ValueError):a.header(fixture()|change,'s1_exp1.set',5*73*4)

    def test_payload_excludes_aux_and_status_preserves_unmapped_fractional_values(self):
        h=a.header(fixture(),'s1_exp1.set',5*73*4);x=np.arange(5*73,dtype='<f4').reshape(5,73);x[:,72]=[0,0,1.5,1.5,0]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.fdt';p.write_bytes(x.tobytes());r=a.stream_fdt(p,h,hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(r['EEG_payload_sha256'],hashlib.sha256(np.ascontiguousarray(x[:,:64]).tobytes()).hexdigest())
            self.assertEqual(r['EEG_sample_bytes']+r['EXG_sample_bytes']+r['Status_sample_bytes'],p.stat().st_size)
            self.assertEqual(r['Status_value_transition_count'],2);self.assertEqual(r['Status_noninteger_sample_count'],2)
            self.assertFalse(r['Status_transitions_are_validated_stimulus_events'])
            x[:,64:]+=9;p.write_bytes(x.tobytes());r2=a.stream_fdt(p,h,hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(r['EEG_payload_sha256'],r2['EEG_payload_sha256'])

    def test_truncation_and_wrong_digest_rejected(self):
        h=a.header(fixture(),'s1_exp1.set',5*73*4)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.fdt';p.write_bytes(b'\0'*(5*73*4-1))
            with self.assertRaises(ValueError):a.stream_fdt(p,h,hashlib.sha256(p.read_bytes()).hexdigest())
            p.write_bytes(b'\0'*(5*73*4))
            with self.assertRaises(ValueError):a.stream_fdt(p,h,'0'*64)


if __name__=='__main__':unittest.main()
