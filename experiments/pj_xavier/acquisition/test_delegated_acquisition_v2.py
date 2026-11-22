import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import delegated_acquisition_v2 as d


def setup_db():
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
    db.execute('CREATE TABLE downloads(id TEXT PRIMARY KEY,catalog_id TEXT,dataset_key TEXT,relative_path TEXT,expected_bytes INTEGER,checksum_json TEXT,status TEXT,lease_until REAL,updated_at TEXT,provider_file_id TEXT,provider_file_identity TEXT,bytes_present INTEGER,sha256 TEXT,error TEXT)')
    for id,status in [('a','queued'),('b','queued'),('untouched','running')]:
        db.execute('INSERT INTO downloads VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(id,'000413','data',id+'.txt',3,json.dumps({'algorithm':'md5','value':hashlib.md5(b'abc').hexdigest()}),status,0 if status=='queued' else 99999,'before','p'+id,'p'+id,0,None,None))
    db.commit();return db


class TestDelegation(unittest.TestCase):
    def test_exact_queued_only_and_running_unchanged(self):
        db=setup_db();selected=[dict(r) for r in db.execute("SELECT * FROM downloads WHERE id IN ('a','b')")]
        with tempfile.TemporaryDirectory() as t:
            r=d.delegate(db,selected,t,'delegated_compact_v1',Path(t)/'before.json',Path(t)/'commit.json')
            self.assertEqual(r['job_count'],2)
            self.assertEqual(dict(db.execute("SELECT * FROM downloads WHERE id='untouched'").fetchone())['status'],'running')
            self.assertEqual(db.execute("SELECT lease_until FROM downloads WHERE id='untouched'").fetchone()[0],99999)

    def test_any_active_or_complete_or_leased_source_rolls_back_all(self):
        for status,lease in [('running',0),('complete',0),('queued',30)]:
            db=setup_db();selected=[dict(r) for r in db.execute("SELECT * FROM downloads WHERE id IN ('a','b')")]
            db.execute("UPDATE downloads SET status=?,lease_until=? WHERE id='b'",(status,lease));db.commit()
            with tempfile.TemporaryDirectory() as t,self.subTest(status=status,lease=lease):
                with self.assertRaisesRegex(ValueError,'unleased queued'):
                    d.delegate(db,selected,t,'delegated_compact_v1',Path(t)/'before.json',Path(t)/'commit.json')
                self.assertEqual(db.execute("SELECT status FROM downloads WHERE id='a'").fetchone()[0],'queued')

    def test_existing_partial_or_changed_checksum_prevents_delegation(self):
        db=setup_db();selected=[dict(db.execute("SELECT * FROM downloads WHERE id='a'").fetchone())]
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'data/a.txt.part';p.parent.mkdir();p.write_bytes(b'a')
            with self.assertRaisesRegex(ValueError,'complete/partial'):
                d.delegate(db,selected,t,'delegated_compact_v1',Path(t)/'before.json',Path(t)/'commit.json')
            p.unlink();selected[0]['expected_bytes']=4
            with self.assertRaisesRegex(ValueError,'identity/size/checksum'):
                d.delegate(db,selected,t,'delegated_compact_v1',Path(t)/'before.json',Path(t)/'commit.json')

    def test_reconcile_requires_verified_complete_and_unchanged_delegation(self):
        db=setup_db();selected=[dict(db.execute("SELECT * FROM downloads WHERE id='a'").fetchone())]
        class A:
            @staticmethod
            def verify_file(path,size,checksum):
                b=Path(path).read_bytes()
                if len(b)!=size or hashlib.md5(b).hexdigest()!=checksum['value']:raise ValueError('checksum')
                return hashlib.sha256(b).hexdigest()
        with tempfile.TemporaryDirectory() as t:
            d.delegate(db,selected,t,'delegated_compact_v1',Path(t)/'before.json',Path(t)/'commit.json')
            p=Path(t)/'data/a.txt';p.parent.mkdir();p.write_bytes(b'abc')
            verified={**selected[0],'status':'complete','sha256':hashlib.sha256(b'abc').hexdigest()}
            d.reconcile(db,[verified],t,A,Path(t)/'reconcile.json')
            row=dict(db.execute("SELECT * FROM downloads WHERE id='a'").fetchone())
            self.assertEqual(row['status'],'complete');self.assertEqual(row['bytes_present'],3)
            with self.assertRaisesRegex(ValueError,'no longer unleased delegated'):
                d.reconcile(db,[verified],t,A,Path(t)/'again.json')


class TestPinnedIdentityEnrichment(unittest.TestCase):
    def rows(self):
        old={'id':'a','catalog_id':'001035','dataset_key':'zenodo-806023-806023','relative_path':'subject1.raw.fif',
             'expected_bytes':3,'checksum_json':json.dumps({'algorithm':'md5','value':hashlib.md5(b'abc').hexdigest()}),
             'provider_file_id':None,'url':'https://zenodo.org/api/records/806023/files/subject1.raw.fif/content'}
        old['provider_file_identity']='zenodo:url:'+old['url']
        new={**old,'provider_file_id':'e86b197d-ae4c-4dc8-b38d-9205679196f8',
             'provider_file_identity':'zenodo:file:e86b197d-ae4c-4dc8-b38d-9205679196f8'}
        return old,new

    def test_identical_URL_size_checksum_can_enrich_missing_ID(self):
        old,new=self.rows();before=dict(old)
        alias=d.original_matches(old,new)
        self.assertEqual(alias['reason'],'live_Zenodo_UUID_enrichment_of_identical_pinned_content_URL')
        self.assertFalse(alias['original_identity_metadata_modified'])
        self.assertEqual(old,before)

    def test_content_changes_and_unproven_aliases_fail_closed(self):
        cases=[('url','https://zenodo.org/api/records/806023/files/different/content'),
               ('expected_bytes',4),('checksum_json',json.dumps({'algorithm':'md5','value':'0'*32})),
               ('provider_file_identity','zenodo:file:unrecognized'),
               ('dataset_key','zenodo-999-999'),('provider_file_id','not-a-uuid')]
        for key,value in cases:
            old,new=self.rows();new[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):d.original_matches(old,new)
        old,new=self.rows();old['provider_file_id']='23e8b972-597f-49b4-9127-9efe5fe9ded2'
        with self.assertRaisesRegex(ValueError,'file ID differs'):d.original_matches(old,new)
        for url in ['https://evilzenodo.org/api/records/806023/files/subject1.raw.fif/content',
                    'https://zenodo.org/api/records/999/files/subject1.raw.fif/content',
                    'https://zenodo.org/api/records/806023/files/subject1.raw.fif/content?token=x']:
            old,new=self.rows();old['url']=new['url']=url;old['provider_file_identity']='zenodo:url:'+url
            with self.subTest(url=url),self.assertRaises(ValueError):d.original_matches(old,new)

    def test_identity_enrichment_is_in_durable_delegation_ledger(self):
        db=setup_db();db.execute('ALTER TABLE downloads ADD COLUMN url TEXT')
        old,new=self.rows()
        for key,value in old.items():
            if key!='id':db.execute('UPDATE downloads SET '+key+'=? WHERE id=?',(value,'a'))
        db.commit()
        selected=dict(db.execute("SELECT * FROM downloads WHERE id='a'").fetchone());selected.update(new)
        with tempfile.TemporaryDirectory() as t:
            r=d.delegate(db,[selected],t,'delegated_compact_v1',Path(t)/'before.json',Path(t)/'commit.json')
            self.assertEqual(r['verified_provider_identity_alias_count'],1)
            original=dict(db.execute("SELECT * FROM downloads WHERE id='a'").fetchone())
            self.assertIsNone(original['provider_file_id'])
            self.assertEqual(original['provider_file_identity'],old['provider_file_identity'])
            ledger=json.loads((Path(t)/'before.json').read_text())
            self.assertEqual(len(ledger['verified_provider_identity_aliases']),1)


if __name__=='__main__':unittest.main()
