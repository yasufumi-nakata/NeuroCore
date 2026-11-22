import unittest
import derive_fif_duplicate_constraints as a


def row(name,group,digest,whole):
    return {'recording_id':name,'subject_group':group,'catalog_id':'001148','sample_rate_hz':256,'sample_count':256,'channels':['Oz'],
            'decoded_EEG_V_sha256':digest,'source_sha256':whole,'EEG_scalar_bytes':1024,'original_stim_events':[],
            'measurement_datetime':'2020-01-01','channel_calibration':[{'cal':1.}]}


class Tests(unittest.TestCase):
    def test_transitive_must_link_even_distinct_header_bytes(self):
        rr=[row('a1','a','x','h1'),row('b1','b','x','h2'),row('b2','b','y','h3'),row('c1','c','y','h3'),row('d1','d','z','h4')]
        s,c=a.derive(rr)
        self.assertEqual(c['must_link_subject_groups'],[['a','b','c']]);self.assertEqual(s['unique_decoded_EEG_records'],3)
        self.assertEqual(s['whole_file_identical_duplicate_pairs'],1);self.assertEqual(s['cross_subject_ID_duplicate_pairs'],2)
        self.assertEqual(s['unique_processed_EEG_seconds_exact_fraction'],[3,1]);self.assertTrue(s['components_are_not_claimed_actual_people'])

    def test_same_subject_session_duplicate_does_not_join_others(self):
        s,c=a.derive([row('a1','a','x','h1'),row('a2','a','x','h1'),row('b1','b','y','h2')])
        self.assertEqual(c['must_link_subject_groups'],[]);self.assertEqual(s['cross_subject_ID_duplicate_pairs'],0)


if __name__=='__main__':unittest.main()
