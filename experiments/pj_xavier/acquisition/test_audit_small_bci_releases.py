import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import audit_small_bci_releases as a


class AuditTests(unittest.TestCase):
    def test_epoch_axis_equality_and_label_length(self):
        mat=np.arange(13*448*2,dtype=np.float32).reshape(13,448,2);hdf=mat.transpose(2,1,0)
        self.assertEqual(a.validate_epoch_arrays(mat,hdf,np.array([1,2]),a.MI_CHANNELS,128,448,2).tolist(),[1,2])
        bad=hdf.copy();bad[0,0,0]+=1
        with self.assertRaisesRegex(ValueError,'waveforms differ'):a.validate_epoch_arrays(mat,bad,np.array([1,2]),a.MI_CHANNELS,128,448,2)
        with self.assertRaisesRegex(ValueError,'Labels length'):a.validate_epoch_arrays(mat,hdf,np.array([1]),a.MI_CHANNELS,128,448,2)

    def test_buffer_bytes_exclude_stim(self):
        raw=SimpleNamespace(ch_names=list(range(9)),n_times=7,_raw_extras=[{'orig_nchan':9,'bounds':[0,3,7],'ent':[SimpleNamespace(type=4,size=3*9*4),SimpleNamespace(type=4,size=4*9*4)]}])
        self.assertEqual(a.fif_buffer_accounting(raw,list(range(8)))[:2],(252,224))
        raw._raw_extras[0]['ent'][1].size-=4
        with self.assertRaisesRegex(ValueError,'byte count'):a.fif_buffer_accounting(raw,list(range(8)))

    def test_text_does_not_assign_unknown_column_or_unit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'S2_2Pan_18.txt';p.write_bytes(b'1 2 3 4 0\n5 6 7 8 9\n')
            r=a.audit_text(p)
            self.assertEqual(r['table_rows'],2);self.assertEqual(r['declared_rate_table_duration_seconds'],.008)
            self.assertIsNone(r['EEG_specific_payload_bytes']);self.assertFalse(r['actual_column_to_EEG_reference_marker_mapping_verified'])
            self.assertEqual(sum(r['numeric_UTF8_token_bytes_by_column'])+r['whitespace_bytes'],p.stat().st_size)

    def test_text_rejects_bad_shape_and_fractional_values(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'S2_2Pan_18.txt'
            for bad in [b'1 2 3 4\n',b'1 2 3 4 .5\n',b'O1 Oz O2 ref marker\n']:
                p.write_bytes(bad)
                with self.assertRaises(ValueError):a.audit_text(p)

    def test_missing_buffer_not_credited_as_recording(self):
        raw=SimpleNamespace(ch_names=list(range(9)),n_times=7,_raw_extras=[{'orig_nchan':9,'bounds':[0,7],'ent':[None]}])
        with self.assertRaisesRegex(ValueError,'Unsupported'):a.fif_buffer_accounting(raw,list(range(8)))

    def test_unknown_original_label_not_repaired(self):
        x=np.zeros((13,448,1),np.float32)
        with self.assertRaisesRegex(ValueError,'Labels length'):a.validate_epoch_arrays(x,x.transpose(2,1,0),np.array([3]),a.MI_CHANNELS,128,448,1)


if __name__=='__main__':unittest.main()
