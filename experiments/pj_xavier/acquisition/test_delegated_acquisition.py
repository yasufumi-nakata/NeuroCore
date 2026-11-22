import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import delegated_acquisition as d


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


if __name__=='__main__':unittest.main()
