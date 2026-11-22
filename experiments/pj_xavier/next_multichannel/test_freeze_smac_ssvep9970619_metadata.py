import tempfile
import unittest
from pathlib import Path

from freeze_smac_ssvep9970619_metadata import approved_host, split_tokens, discover_headers, prepare, SOURCE


class SplitAuditTests(unittest.TestCase):
    def test_excluded_hosts(self):
        for host in ['smac19','smac31','smac40','workstation','smac1','smac99x']:
            self.assertFalse(approved_host(host))
        self.assertTrue(approved_host('smac02.sfc.keio.ac.jp'))

    def test_group_determinism_and_balance(self):
        ids=[f'P{x:02}' for x in range(11)]
        a=split_tokens(ids)
        self.assertEqual(a,split_tokens(list(reversed(ids))))
        self.assertEqual((list(a.values()).count('train'),list(a.values()).count('val'),list(a.values()).count('test')),(7,2,2))

    def test_split_rejects_duplicate_or_incomplete(self):
        with self.assertRaises(ValueError):split_tokens(['a']*11)
        with self.assertRaises(ValueError):split_tokens(['a']*10)

    def test_source_geometry_and_no_heldout_signal_open(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/SOURCE;root.mkdir()
            for i in range(11):
                for j in range(4):
                    stem=f'P{i:02}_block{j}'
                    (root/(stem+'.set')).write_bytes(b'metadata-header')
                    (root/(stem+'.fdt')).write_bytes(b'0'*(73*4*200))
            for token,s,f in discover_headers(root,host='smac02'):
                self.assertTrue(s.exists());self.assertTrue(f.exists())
            r=prepare(root,host='smac02')
            self.assertEqual(r['aggregate']['train_records'],28)
            self.assertEqual(r['aggregate']['val_records'],8)
            self.assertEqual(r['aggregate']['test_records'],8)
            self.assertEqual(r['test_signal_files_opened'],0)
            self.assertIsNone(r['unit_voltage_scale'])
            self.assertFalse(r['fit_allowed'])
            self.assertEqual(len(r['metadata_manifest_sha256']),64)

    def test_wrong_geometry_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/SOURCE;root.mkdir()
            for i in range(11):
                for j in range(4):
                    name=f'P{i:02}_block{j}'
                    (root/(name+'.set')).write_bytes(b'm')
                    (root/(name+'.fdt')).write_bytes(b'bad')
            with self.assertRaises(ValueError):discover_headers(root,host='smac02')


if __name__=='__main__':unittest.main()
