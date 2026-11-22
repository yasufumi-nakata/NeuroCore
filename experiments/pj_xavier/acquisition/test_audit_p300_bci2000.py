import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import audit_p300_bci2000 as a


def header(channels=2, state_line='StimulusCode 16 0 0 3', state_bytes=3, gain='1 2', names='C3 C4'):
    prefix=f'BCI2000V= 1.1 HeaderLen= 01024 SourceCh= {channels} StatevectorLen= {state_bytes} DataFormat= float32\r\n'
    body=('[ State Vector Definition ]\r\n'+state_line+'\r\n[ Parameter Definition ]\r\n'
          f'Source int SourceCh= {channels}\r\nSource int SamplingRate= 256Hz\r\nSource int SampleBlockSize= 8\r\n'
          f'Source list ChannelNames= {channels} {names}\r\nSource floatlist SourceChGain= {channels} {gain}\r\n'
          f'Source floatlist SourceChOffset= {channels} '+('0 '*channels)+'\r\n')
    data=(prefix+body).encode('ascii');return data+b' '*(1024-len(data))


class TestBCI2000Audit(unittest.TestCase):
    def test_frames_use_signal_and_state_bytes_and_native_rate(self):
        h=a.parse_header_bytes(header(),1024+(2*4+3)*512)
        self.assertEqual(h['sample_count'],512);self.assertEqual(h['stored_signal_seconds'],2)
        self.assertEqual(h['gain_AD_to_microV'],[1,2])
        with self.assertRaisesRegex(ValueError,'Truncated'):
            a.parse_header_bytes(header(),1024+(2*4+3)*512+1)

    def test_state_bit_cross_byte_values_are_little_endian(self):
        values=[0,1,0xABCD,65535]
        raw=np.array([list((v<<3).to_bytes(3,'little')) for v in values],dtype=np.uint8)
        self.assertEqual(a.decode_state(raw,{'byte':0,'bit':3,'length':16}).tolist(),values)
        with self.assertRaises(ValueError):a.decode_state(raw[:,:1],{'byte':0,'bit':3,'length':16})

    def test_unknown_units_overlap_and_inconsistent_name_counts_rejected(self):
        for h in [header(gain='1mV 2'),header(state_line='A 4 0 0 0\r\nB 3 0 0 2'),header(names='C3 C3')]:
            with self.subTest(),self.assertRaises(ValueError):a.parse_header_bytes(h,1024+11*2)

    def test_signal_hash_excludes_state_changes_but_whole_hash_does_not(self):
        n=32;names=' '.join(a.SCALP);h=header(n,'StimulusBegin 1 0 0 0\r\nStimulusCode 16 0 1 0\r\nStimulusType 1 0 3 0',4,' '.join(['1']*n),names)
        # Increase the fixture header to contain the full montage safely.
        h=h.rstrip(b' '); h=h.replace(b'HeaderLen= 01024',b'HeaderLen= 04096');h=h+b' '*(4096-len(h))
        signal=np.arange(n*4,dtype='<f4').reshape(4,n)
        states=np.array([[0,0,0,0],[1,3,0,1],[1,3,0,1],[0,0,0,0]],dtype=np.uint8)
        def payload(st):return b''.join(signal[i].tobytes()+st[i].tobytes() for i in range(4))
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.dat';b=h+payload(states);p.write_bytes(b)
            r=a.audit_dat(p,'root/sub-01/ses-test/eeg/r.dat',hashlib.sha256(b).hexdigest())
            self.assertEqual(r['stored_stimulus_begin_rising_transitions'][0]['sample_index'],1)
            self.assertEqual(r['stored_stimulus_begin_rising_transitions'][0]['StimulusCode'],3)
            self.assertEqual(r['stored_stimulus_begin_rising_transitions'][0]['StimulusType'],1)
            states[1,1]=4;b2=h+payload(states);p.write_bytes(b2)
            r2=a.audit_dat(p,'root/sub-01/ses-test/eeg/r.dat',hashlib.sha256(b2).hexdigest())
            self.assertEqual(r['EEG_payload_sha256'],r2['EEG_payload_sha256'])
            self.assertNotEqual(r['state_payload_sha256'],r2['state_payload_sha256'])
            self.assertTrue(r['publisher_session_is_not_project_split'])

    def test_edf_declared_one_second_is_kept_not_repaired_to_DAT_rate(self):
        fixed=bytearray(b' '*256);fixed[:8]=b'0       ';fixed[184:192]=b'512     ';fixed[236:244]=b'1       ';fixed[244:252]=b'1       ';fixed[252:256]=b'1   '
        fields=[('Signal_1',16),('',80),('',8),('-1',8),('1',8),('-32768',8),('32767',8),('',80),('256',8),('',32)]
        data=bytes(fixed)+b''.join(x.encode().ljust(n) for x,n in fields)+b'\0'*512
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.edf';p.write_bytes(data);r=a.edf_header(p)
            self.assertEqual(r['declared_stored_duration_s'],1);self.assertEqual(r['physical_units'],[''])
            self.assertEqual(r['samples_per_channel'],[256])
            p.write_bytes(data[:-1])
            with self.assertRaisesRegex(ValueError,'exhaust'):a.edf_header(p)


if __name__=='__main__':unittest.main()
