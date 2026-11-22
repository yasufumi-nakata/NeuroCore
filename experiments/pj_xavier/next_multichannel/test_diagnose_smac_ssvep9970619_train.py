import unittest
from diagnose_smac_ssvep9970619_train import train_rows
from freeze_smac_ssvep9970619_metadata import SOURCE, sha
import json

class TrainOnlyTests(unittest.TestCase):
    def make(self):
        rows=[]
        for i in range(44):
            n=f'P{i:02}.fdt'
            rows.append({'split':('train' if i<28 else 'val' if i<36 else 'test'),
                         'fdt_content_opened':False,'fdt_name':n,'set_name':f'P{i:02}.set',
                         'source':SOURCE,'fdt_expected_bytes':200,'set_header_sha256':'0'*64})
        digest=sha(json.dumps(rows,sort_keys=True,separators=(',',':')).encode())
        return {'schema':'xavier-figshare9970619-metadata-only-split-v1','source':SOURCE,
                'fit_allowed':False,'validation_signal_files_opened':0,'test_signal_files_opened':0,
                'metadata_manifest_sha256':digest,'items':rows}
    def verify(self, x):
        return train_rows(x,expected_metadata_manifest_sha256=x['metadata_manifest_sha256'])
    def test_only_train_records(self):
        rows=self.verify(self.make())
        self.assertEqual(len(rows),28)
        self.assertTrue(all(r['split']=='train' for r in rows))
    def test_refuse_already_opened_heldout(self):
        x=self.make();x['items'][-1]['fdt_content_opened']=True
        with self.assertRaises(ValueError):self.verify(x)
    def test_refuse_miscount(self):
        x=self.make();x['items'][-1]['split']='train'
        with self.assertRaises(ValueError):self.verify(x)
    def test_refuse_fit_enabled(self):
        x=self.make();x['fit_allowed']=True
        with self.assertRaises(ValueError):self.verify(x)
    def test_refuse_tampered_manifest(self):
        x=self.make();x['items'][0]['fdt_expected_bytes']+=1
        with self.assertRaises(ValueError):self.verify(x)
    def test_refuse_untrusted_expected_digest(self):
        x=self.make()
        with self.assertRaises(ValueError):train_rows(x,expected_metadata_manifest_sha256='f'*64)
    def test_refuse_escape_path(self):
        x=self.make();x['items'][0]['fdt_name']='../../../other.fdt'
        x['metadata_manifest_sha256']=sha(json.dumps(x['items'],sort_keys=True,separators=(',',':')).encode())
        with self.assertRaises(ValueError):self.verify(x)

if __name__=='__main__':unittest.main()
