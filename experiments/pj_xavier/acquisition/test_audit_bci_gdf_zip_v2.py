import unittest,tempfile,struct,hashlib
from pathlib import Path
import numpy as np
import audit_bci_gdf_zip_v2 as a
class TestBCIQuantity(unittest.TestCase):
 def test_source_mapping_separates_EOG_EMG_not_every_signal_EEG(self):
  names=a.EEG_NAMES[:11]+['EOG1','EOG2','EOG3','EMGg','EMGd']+a.EEG_NAMES[11:]
  q=a.classify_channels(names);self.assertEqual({k:len(v) for k,v in q.items()},{'EEG':27,'EOG':3,'EMG':2,'unknown':0})
  with self.assertRaises(ValueError):a.classify_channels(names[:-1]+['AUXunknown'])
 def test_raw_event_positions_and_absent_durations_not_repaired(self):
  data=bytes([1])+(512).to_bytes(3,'little')+struct.pack('<I',3)+struct.pack('<III',1,513,1025)+struct.pack('<HHH',768,769,770)
  q=a.event_table_125(data,512,2)
  self.assertEqual(q['events_outside_record'],1)
  self.assertEqual([x['onset_s'] for x in q['MI_cue_events']],[1,2])
  self.assertIsNone(q['MI_cue_events'][0]['original_duration_samples'])
  self.assertEqual(q['event_repairs'],0)
 def test_mode3_zero_duration_is_preserved(self):
  data=bytes([3])+(512).to_bytes(3,'little')+struct.pack('<I',1)+struct.pack('<I',20)+struct.pack('<H',769)+struct.pack('<H',0)+struct.pack('<I',0)
  q=a.event_table_125(data,512,2);self.assertEqual(q['MI_cue_events'][0]['original_duration_samples'],0);self.assertEqual(q['raw_duration_zero_count'],1)
  with self.assertRaises(ValueError):a.event_table_125(data+b'x',512,2)
 def test_payload_byte_count_and_hash_exclude_header_events_aux(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'data.gdf';x=np.arange(6*3,dtype='<f8').reshape(6,3)
   p.write_bytes(b'HEAD'+x.tobytes()+b'EVENTS')
   h={'nchan':3,'n_records':6,'n_samps':[1,1,1],'dtype_byte':8,'bytes_tot':24,'data_offset':4}
   hashes,counts,events=a.payload_hashes(p,h,{'EEG':[0],'EOG':[1],'EMG':[2]})
   self.assertEqual(counts,{'EEG':48,'EOG':48,'EMG':48});self.assertEqual(events,b'EVENTS')
   self.assertEqual(hashes['EEG'],hashlib.sha256(x[:,0].tobytes()).hexdigest())
 def test_observed_zero_event_rate_is_preserved_as_unresolved(self):
  data=bytes([1,0,0,0])+struct.pack('<I',1)+struct.pack('<I',513)+struct.pack('<H',769)
  q=a.event_table_125(data,512,10)
  self.assertEqual(q['event_sample_rate_hz'],0)
  self.assertFalse(q['explicit_event_timebase_verified'])
  self.assertIsNone(q['MI_cue_events'][0]['onset_s'])
  self.assertEqual(q['MI_cue_events'][0]['native_reader_fallback_onset_s'],1)
  self.assertEqual(q['event_repairs'],0)
  self.assertIsNone(q['events_outside_record'])
if __name__=='__main__':unittest.main()
