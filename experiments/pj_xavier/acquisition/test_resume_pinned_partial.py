import tempfile,unittest
from pathlib import Path
import resume_pinned_partial as r

class Response:
    def __init__(self,status=206,start=8,end=9,total=10,length='2'):
        self.status=status;self.headers={'Content-Range':f'bytes {start}-{end}/{total}','Content-Length':length};self.closed=False
    def close(self):self.closed=True

class TestStrict206(unittest.TestCase):
    def test_exact_start_total_allowed(self):
        self.assertEqual(r.check_range_response(Response(),{'Range':'bytes=8-'},10)['start'],8)
    def test_ignored_range_or_changed_source_or_length_refused(self):
        for response in [Response(status=200),Response(start=0),Response(total=11),Response(end=11),Response(length='3')]:
            with self.subTest(status=response.status,headers=response.headers),self.assertRaises(ValueError):r.check_range_response(response,{'Range':'bytes=8-'},10)
    def test_rejected_response_closed_without_touching_partial(self):
        class H:
            def __init__(self):self.r=Response(status=200)
            def request(self,url,headers):return self.r
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'raw.part';p.write_bytes(b'oldprefix');h=H();http=r.StrictRangeHTTP(h,10,Path(t)/'range.jsonl')
            with self.assertRaises(ValueError):http.request('https://example.invalid/data',{'Range':'bytes=8-'})
            self.assertTrue(h.r.closed);self.assertEqual(p.read_bytes(),b'oldprefix');self.assertFalse((Path(t)/'range.jsonl').exists())

class TestOneFileCustody(unittest.TestCase):
    def test_end_to_end_resume_and_only_original_target_reconciled(self):
        import hashlib,json,sqlite3,time,types,importlib.util,io
        from unittest.mock import patch
        source=Path(__file__).with_name('acquire.py')
        spec=importlib.util.spec_from_file_location('fixture_a',source);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
        class FakeResponse(Response):
            def __init__(self):
                super().__init__(start=10,end=19,total=20,length='10');self.data=io.BytesIO(b'new_data10')
            def read(self,n=-1):return self.data.read(n)
            def __enter__(self):return self
            def __exit__(self,*args):self.close()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);raw=root/'raw';raw.mkdir();folder=raw/'zenodo-8089820-8089820';folder.mkdir()
            data=b'old_data10new_data10';self.assertEqual(len(data),20)
            part=folder/'BCI Database.zip.part';part.write_bytes(data[:10])
            url='https://zenodo.org/api/records/8089820/files/BCI%20Database.zip/content'
            (folder/'BCI Database.zip.part.meta.json').write_text(json.dumps({'url_sha256':hashlib.sha256(url.encode()).hexdigest(),'expected_bytes':20}))
            original=root/'original.sqlite';db=a.init_db(original)
            job={'id':'target','catalog_id':'000038','dataset_key':folder.name,'relative_path':'BCI Database.zip','url':url,'expected_bytes':20,
                 'checksum_json':json.dumps({'algorithm':'md5','value':hashlib.md5(data).hexdigest()}),'status':'paused','bytes_present':10,'lease_until':0}
            cols=list(job);db.execute('insert into downloads ('+','.join(cols)+') values('+','.join('?' for _ in cols)+')',[job[k] for k in cols])
            other={**job,'id':'other','relative_path':'unrelated.zip','status':'running','lease_until':9999999999}
            db.execute('insert into downloads ('+','.join(cols)+') values('+','.join('?' for _ in cols)+')',[other[k] for k in cols]);db.commit()
            launch=root/'oldlaunch.json';launch.write_text('{}')
            budget=root/'resource.json';budget.write_text(json.dumps({'status':'authorized_registered_waiting_original_worker_terminal','absolute_deadline_epoch':time.time()+60,
              'max_files':1,'max_workers':1,'original_bulk_PID':99999999,'original_acquisition_source_sha256':r.sha(source),'original_launch_sha256':r.sha(launch),
              'selected_original_filename':'BCI Database.zip','expected_whole_file_bytes':20,'maximum_received_payload_bytes':10,'effective_download_reserve_bytes':0}))
            args=types.SimpleNamespace(output=root/'run',resource=budget,original_db=original,job_id='target',raw_root=raw,acquire_source=source,acquire_source_sha256=r.sha(source),original_launch=launch)
            with patch('urllib.request.urlopen',side_effect=lambda *a,**k:FakeResponse()):result=r.run(args)
            self.assertEqual(result['status'],'complete');self.assertEqual(result['received_bytes_this_budget'],10)
            self.assertEqual((folder/'BCI Database.zip').read_bytes(),data)
            self.assertEqual(db.execute("select status,bytes_present from downloads where id='target'").fetchone(),('complete',20))
            self.assertEqual(db.execute("select status,lease_until from downloads where id='other'").fetchone(),('running',9999999999))
            self.assertFalse(part.exists())

if __name__=='__main__':unittest.main()
