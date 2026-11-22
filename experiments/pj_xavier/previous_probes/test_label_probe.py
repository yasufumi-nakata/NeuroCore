import unittest
from label_probe import chbmit_events, event_window_classification


class EventMappingTests(unittest.TestCase):
    def test_single_event_and_boundary(self):
        text='File Name: x.edf\nNumber of Seizures in File: 1\nSeizure Start Time: 2996 seconds\nSeizure End Time: 3036 seconds\n'
        ev=chbmit_events(text,'x.edf')
        self.assertEqual(ev,[(2996,3036)])
        agg=event_window_classification(3600,ev)
        self.assertEqual(agg['n_windows'],360)
        self.assertEqual(agg['positive'],4)
        self.assertEqual(agg['negative'],355)
        self.assertEqual(agg['ambiguous'],1)
        self.assertEqual(agg['positive_window_starts_seconds'],[3000,3010,3020,3030])
        self.assertEqual(agg['ambiguous_window_starts_seconds'],[2990])
    def test_missing_annotation_fails_closed(self):
        with self.assertRaises(ValueError):
            chbmit_events('File Name: x.edf\nNumber of Seizures in File: 1\n','x.edf')
    def test_multiple_events(self):
        text='File Name: x.edf\nNumber of Seizures in File: 2\nSeizure 1 Start Time: 5 seconds\nSeizure 1 End Time: 15 seconds\nSeizure 2 Start Time: 30 seconds\nSeizure 2 End Time: 50 seconds\n'
        self.assertEqual(chbmit_events(text,'x.edf'),[(5,15),(30,50)])
    def test_wrong_edf_name_fails(self):
        with self.assertRaises(ValueError):
            chbmit_events('File Name: y.edf\nNumber of Seizures in File: 0\n','x.edf')


if __name__=='__main__':unittest.main()
