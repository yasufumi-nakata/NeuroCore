import hashlib
import io
import unittest
import numpy as np
import audit_brainstorm_vhdr_zip as a


def fixture():
    ch='\n'.join(f'Ch{i}={"IO" if i==19 else "ECG" if i==20 else i},,0.1,µV' for i in range(1,65))
    return ('Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\nCodepage=UTF-8\nDataFile=mas_AEP_nh01.eeg\nMarkerFile=mas_AEP_nh01.vmrk\nDataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=64\nSamplingInterval=1000\n[Binary Infos]\nBinaryFormat=INT_16\n[Channel Infos]\n'+ch+'\n[Comment]\nnot parsed = metadata\n').encode()


class Tests(unittest.TestCase):
    def test_montage_precision_and_companion_identity(self):
        h=a.parse_header(fixture(),a.PREFIX+'mas_AEP_nh01.vhdr',1280)
        self.assertEqual(h['sample_count'],10);self.assertEqual(h['EEG_candidate_channels'],62)
        self.assertEqual(h['auxiliary_indices'],[18,19]);self.assertEqual(h['stored_signal_seconds'],.01)
        for b in [fixture().replace(b'INT_16',b'IEEE_FLOAT_32'),fixture().replace(b'DataFile=mas_AEP_nh01.eeg',b'DataFile=../a.eeg')]:
            with self.assertRaises(ValueError):a.parse_header(b,a.PREFIX+'mas_AEP_nh01.vhdr',1280)

    def test_unclipped_marker_and_one_based_time(self):
        h=a.parse_header(fixture(),a.PREFIX+'mas_AEP_nh01.vhdr',1280)
        m=a.parse_markers(b'Brain Vision Marker\n[Common Infos]\nDataFile=mas_AEP_nh01.eeg\n[Marker Infos]\nMk1=Stimulus,S 10,1,1,0\nMk2=Stimulus,S 99,11,1,0\n',h)
        self.assertEqual(m['original_marker_entries'][0]['derived_onset_s'],0)
        self.assertEqual(m['marker_outside_or_invalid_count'],1);self.assertFalse(m['stimulus_semantics_validated'])

    def test_EEG_hash_excludes_IO_ECG_with_unmodified_binary_samples(self):
        h=a.parse_header(fixture(),a.PREFIX+'mas_AEP_nh01.vhdr',1280);x=np.arange(640,dtype='<i2').reshape(10,64)
        r=a.stream_signal(io.BytesIO(x.tobytes()),h);x[:,18:20]=0;r2=a.stream_signal(io.BytesIO(x.tobytes()),h)
        self.assertEqual(r['EEG_payload_sha256'],r2['EEG_payload_sha256']);self.assertNotEqual(r['auxiliary_payload_sha256'],r2['auxiliary_payload_sha256'])
        self.assertEqual(r['EEG_candidate_sample_bytes']+r['auxiliary_or_unresolved_sample_bytes'],1280)
        self.assertEqual(r2['source_signal_sha256'],hashlib.sha256(x.tobytes()).hexdigest())
        with self.assertRaises(ValueError):a.stream_signal(io.BytesIO(x.tobytes()[:-1]),h)


if __name__=='__main__':unittest.main()
