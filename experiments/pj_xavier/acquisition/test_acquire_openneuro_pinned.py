import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import acquire_openneuro_pinned as p
import acquire


class ChecksumTests(unittest.TestCase):
    def test_git_blob_is_not_raw_sha1(self):
        data = b'hello\n'
        spec = {'algorithm': 'unidentified', 'value': hashlib.sha1(b'blob 6\0'+data).hexdigest()}
        with tempfile.TemporaryDirectory() as t:
            f = Path(t)/'README'; f.write_bytes(data)
            self.assertEqual(p.verify_file(f, 6, spec), hashlib.sha256(data).hexdigest())
            with self.assertRaisesRegex(ValueError, 'Published checksum'):
                p.verify_file(f, 6, {'algorithm': 'unidentified', 'value': hashlib.sha1(data).hexdigest()})

    def test_annex_embedded_size_and_content_are_required(self):
        data = b'abc'
        spec = {'algorithm': 'unidentified', 'value': 'sha256e-s3--'+hashlib.sha256(data).hexdigest()+'.dat'}
        with tempfile.TemporaryDirectory() as t:
            f = Path(t)/'x.dat'; f.write_bytes(data)
            self.assertEqual(p.verify_file(f, 3, spec), hashlib.sha256(data).hexdigest())
            with self.assertRaisesRegex(ValueError, 'key size'):
                p.verify_file(f, 4, spec)
            f.write_bytes(b'xyz')
            with self.assertRaisesRegex(ValueError, 'Published checksum'):
                p.verify_file(f, 3, spec)

    def test_unknown_checksums_and_symlinks_fail_closed(self):
        for value in ('etag', 'A'*40, 'md5e-s3--'+'0'*32+'.dat', 'sha256e-s3--'+'0'*64+'.zip'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                p.checksum_semantics({'algorithm': 'unidentified', 'value': value}, 3)
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'x'; f.write_bytes(b'abc'); link=Path(t)/'link'; link.symlink_to(f)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                p.verify_file(link, 3, {'algorithm': 'unidentified', 'value': hashlib.sha1(b'blob 3\0abc').hexdigest()})


def rows(db):
    r={'id': hashlib.sha256((p.RELEASE+'/README').encode()).hexdigest(), 'catalog_id': p.CATALOG,
       'dataset_key': p.RELEASE, 'relative_path': 'README', 'url': 'https://s3.amazonaws.com/openneuro.org/ds005028/README?versionId=v1',
       'expected_bytes': 3, 'checksum_json': json.dumps({'algorithm':'unidentified','value':hashlib.sha1(b'blob 3\0abc').hexdigest()}),
       'status':'queued', 'attempts':0, 'retry_at':0, 'lease_until':0, 'bytes_present':0, 'sha256':None,
       'error':None, 'updated_at':'before', 'original_relative_path':'README', 'provider_file_id':None}
    r['provider_file_identity']='openneuro:url:'+r['url']
    columns=list(r)
    db.execute('INSERT INTO downloads ('+','.join(columns)+') VALUES ('+','.join('?' for _ in columns)+')',[r[k] for k in columns])
    db.commit();return r


class SourceTests(unittest.TestCase):
    def test_only_exact_public_version_pinned_release(self):
        db=acquire.init_db(':memory:');r=rows(db)
        p.validate_source_job(r, acquire)
        for url in ['https://s3.amazonaws.com.evil.test/openneuro.org/ds005028/README?versionId=v1',
                    'https://s3.amazonaws.com/openneuro.org/ds005028/README',
                    'https://s3.amazonaws.com/openneuro.org/ds005028/README?versionId=null',
                    'https://s3.amazonaws.com/openneuro.org/ds005028/README?versionId=v1&token=x']:
            x={**r, 'url':url, 'provider_file_identity':'openneuro:url:'+url}
            with self.subTest(url=url),self.assertRaises(ValueError):p.validate_source_job(x, acquire)

    def test_delegation_does_not_take_active_or_existing_partial(self):
        for state,lease in [('running',0),('queued',999),('complete',0),('queued',0)]:
            with tempfile.TemporaryDirectory() as t:
                db=acquire.init_db(Path(t)/'original.sqlite');r=rows(db)
                db.execute('UPDATE downloads SET status=?,lease_until=?',(state,lease));db.commit()
                if state=='queued' and lease==0:
                    f=Path(t)/p.RELEASE/'README.part';f.parent.mkdir();f.write_bytes(b'a')
                with self.subTest(state=state,lease=lease),self.assertRaises(ValueError):
                    p.delegate(db,[r],t,acquire,t)
                self.assertEqual(db.execute('SELECT status FROM downloads').fetchone()[0],state)

    def test_reconcile_exact_delegation_after_content_verification(self):
        with tempfile.TemporaryDirectory() as t:
            db=acquire.init_db(Path(t)/'original.sqlite');r=rows(db)
            p.delegate(db,[r],t,acquire,t)
            f=Path(t)/p.RELEASE/'README';f.parent.mkdir(exist_ok=True);f.write_bytes(b'abc')
            complete={**r,'status':'complete','sha256':hashlib.sha256(b'abc').hexdigest()}
            result=p.reconcile(db,[complete],t,acquire,t)
            self.assertEqual(result['files'],1)
            self.assertEqual(db.execute('SELECT status FROM downloads').fetchone()[0],'complete')
            with self.assertRaisesRegex(ValueError,'no longer delegated'):
                p.reconcile(db,[complete],t,acquire,t)


if __name__ == '__main__':
    unittest.main()
