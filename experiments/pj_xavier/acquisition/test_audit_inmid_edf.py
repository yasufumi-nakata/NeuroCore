import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import audit_inmid_edf as a


def make_file(path,marker=0):
    names=a.PREFIX+a.EEG+['RAW_CQ','BATTERY','BATTERY_PERCENT','MarkerIndex','MarkerType','MarkerValueInt','MARKER_HARDWARE']+[f'aux{i}' for i in range(33)]
    assert len(names)==58
    n=58;prefix=(b'0       '+b' '*160+b'01.01.01'+b'00.00.00'+str((n+1)*256).ljust(8).encode()+b' '*44+b'2       '+b'1       '+str(n).ljust(4).encode())
    fields=[(names,16),(['']*n,80),(['uV']*n,8),(['0']*n,8),(['16000']*n,8),(['0']*n,8),(['31200']*n,8),(['']*n,80),(['128']*n,8),(['']*n,32)]
    h=prefix+b''.join(b''.join(str(v).ljust(width).encode() for v in vals) for vals,width in fields)
    x=np.arange(58*128,dtype='<i2').reshape(58,128);x[names.index('MarkerValueInt'),:]=marker
    path.write_bytes(h+x.tobytes()*2);return x


class Tests(unittest.TestCase):
    def test_only14EEG_and44auxiliary_byte_partition(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'2023-07-24-P01-1-1.edf';make_file(p);h=a.header(p)
            self.assertEqual(h['sample_count'],256);self.assertEqual(len(h['auxiliary_indices']),44)
            r=a.stream(p,h,hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(r['EEG_sample_bytes'],256*14*2);self.assertEqual(r['EEG_sample_bytes']+r['auxiliary_sample_bytes']+h['header_bytes'],p.stat().st_size)

    def test_declared_EEG_calibration_not_applied_to_markers(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.edf';make_file(p);h=a.header(p);x=np.tile([0,31200],(14,1));v=a.physical_EEG_volts(x,h)
            np.testing.assert_array_equal(v[:,0],0);np.testing.assert_allclose(v[:,1],.016)

    def test_auxiliary_change_does_not_change_EEG_identity_or_invent_gold(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.edf';make_file(p,0);h=a.header(p);r=a.stream(p,h,hashlib.sha256(p.read_bytes()).hexdigest());make_file(p,3);s=a.stream(p,h,hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(r['EEG_payload_sha256'],s['EEG_payload_sha256']);self.assertNotEqual(r['auxiliary_payload_sha256'],s['auxiliary_payload_sha256']);self.assertFalse(s['marker_codes_mapped_to_action_or_intent'])

    def test_source_group_shared_across_sessions(self):
        self.assertEqual(a.source_group('2023-07-24-P01-1-1.edf')[0],a.source_group('2023-08-24-P01-2-2.edf')[0])
        with self.assertRaises(ValueError):a.source_group('P01.edf')

    def test_truncated_payload_and_wrong_digest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.edf';make_file(p);h=a.header(p)
            with self.assertRaises(ValueError):a.stream(p,h,'0'*64)
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):a.header(p)


if __name__=='__main__':unittest.main()
