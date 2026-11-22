import copy
import unittest
import resume_delegated_acquisition as a


class Tests(unittest.TestCase):
    def fixture(self):
        old={'status':'authorized_registered','absolute_deadline_epoch':2000,'catalog_ids':['a','b'],'expected_files':2,'known_source_bytes':200,'manifest_sha256':'a'*64,'maximum_received_payload_bytes':230,'effective_download_reserve_bytes':38*2**30,'concurrent_workers':1}
        new=copy.deepcopy(old);new['maximum_received_payload_bytes']=130;new['effective_download_reserve_bytes']=36*2**30
        return old,new,{'status':'incomplete','received_payload_bytes_this_run':100}

    def test_prior_traffic_and_deadline_remain_binding(self):
        old,new,previous=self.fixture();self.assertEqual(a.validate_budget(old,new,previous),100)
        for change in [{'maximum_received_payload_bytes':131},{'absolute_deadline_epoch':2001},{'catalog_ids':['b']},{'effective_download_reserve_bytes':35*2**30},{'manifest_sha256':'b'*64}]:
            with self.subTest(change=change),self.assertRaises(ValueError):a.validate_budget(old,new|change,previous)

    def test_running_lease_integrity_failure_and_duplicate_refused(self):
        rows=[{'id':'one','lease_until':0,'status':'complete'},{'id':'two','lease_until':0,'status':'capacity_blocked'}];a.validate_rows(rows,2)
        for change in [{'lease_until':1},{'status':'running'},{'status':'integrity_failed'},{'id':'one'}]:
            with self.subTest(change=change),self.assertRaises(ValueError):a.validate_rows([rows[0],rows[1]|change],2)


if __name__=='__main__':unittest.main()
