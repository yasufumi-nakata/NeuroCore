import hashlib
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

import acquire


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status, self.headers = status, headers or {}


class HttpFixture:
    def __init__(self, response):
        self.response, self.headers = response, None

    def request(self, url, headers=None):
        self.headers = headers
        return self.response


def job(data=b"abcdefgh"):
    return {"dataset_key": "zenodo-1-1", "relative_path": "a.eeg",
            "url": "https://zenodo.org/records/1/files/a.eeg",
            "expected_bytes": len(data),
            "checksum_json": json.dumps({"algorithm": "md5", "value": hashlib.md5(data).hexdigest()})}


class AcquisitionTests(unittest.TestCase):
    def test_figshare_inferred_identity_requires_exact_host_or_subdomain(self):
        cases = {
            "https://figshare.com/files/7": "7",
            "https://ndownloader.figshare.com/files/7": "7",
            "https://NDOWNLOADER.FIGSHARE.COM/files/7": "7",
            "https://figshare.com./files/7": "7",
            "https://evilfigshare.com/files/7": None,
            "https://figshare.com.other.example/files/7": None,
            "https://other.example/figshare.com/files/7": None,
            "https://other.example/files/7?provider=figshare.com": None,
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                identity = acquire.provider_identity({"provider": "figshare"}, {"url": url})
                self.assertEqual(identity["provider_file_id"], expected)
                if expected is None:
                    self.assertIn(":url:", identity["identity"])

    def test_physionet_detection_uses_url_hostname_not_embedded_text(self):
        cases = {
            "https://physionet.org/content/demo/1.0/": "physionet",
            "https://data.physionet.org/content/demo/1.0/": "physionet",
            "https://PHYSIONET.ORG./content/demo/1.0/": "physionet",
            "https://evilphysionet.org/content/demo/1.0/": "unsupported",
            "https://physionet.org.other.example/content/demo/1.0/": "unsupported",
            "https://other.example/physionet.org/demo": "unsupported",
            "https://other.example/?provider=physionet.org": "unsupported",
            "https://physionet.org@other.example/demo": "unsupported",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["catalog_id", "primary_url"])
                writer.writeheader()
                writer.writerows({"catalog_id": str(i), "primary_url": url}
                                 for i, url in enumerate(cases, start=1))
            rows = acquire.normalize_catalog(path)
        for row, (url, expected) in zip(rows, cases.items()):
            with self.subTest(url=url):
                self.assertEqual(row["provider"], expected)

    def test_explicit_dataverse_guestbook_requires_action_without_logging_url_token(self):
        body = json.dumps({"status": "ERROR", "message": "You may not download this file without the required Guestbook response for guestbookID 217."}).encode()
        error = urllib.error.HTTPError("https://dataverse.harvard.edu/api/access/datafile/3144692?token=DO_NOT_LOG", 400, "Bad Request", {}, io.BytesIO(body))
        result = acquire.access_requirement_from_http_error(error)
        self.assertEqual(result["code"], "guestbook_required")
        self.assertEqual(result["guestbook_id"], 217)
        self.assertNotIn("DO_NOT_LOG", json.dumps(result))
        self.assertFalse(result["submission_performed"])

    def test_other_http400_or_unbounded_body_does_not_become_access_restriction(self):
        for body in (b'{"status":"ERROR","message":"Bad file format"}', b'x' * 9000,
                     b'{"status":"ERROR","message":"Guestbook mentioned in unrelated debugging text"}'):
            with self.subTest(body_length=len(body)):
                error = urllib.error.HTTPError("https://dataverse.harvard.edu/api/access/datafile/1", 400, "Bad Request", {}, io.BytesIO(body))
                self.assertIsNone(acquire.access_requirement_from_http_error(error))

    def test_guestbook_worker_holds_partial_and_never_retries_held_file(self):
        class GuestbookHttp:
            calls = 0
            def request(self, url, headers=None):
                self.calls += 1
                body = b'{"status":"ERROR","message":"You may not download this file without the required Guestbook response for guestbookID 217."}'
                raise urllib.error.HTTPError(url + "?token=DO_NOT_LOG", 400, "Bad Request", {}, io.BytesIO(body))
        with tempfile.TemporaryDirectory() as tmp:
            db = acquire.init_db(Path(tmp) / "download.sqlite")
            row = {"catalog_id": "000037", "provider": "dataverse", "source_id": "doi:test", "source_version": "1.1", "status": "resolved",
                   "files": [acquire.make_file("a.eeg", "https://dataverse.harvard.edu/api/access/datafile/1", 8, evidence_url="https://dataverse.harvard.edu/api/datasets/1")]}
            acquire.ingest(db, [row])
            p = acquire.destination(Path(tmp) / "raw", acquire.dataset_storage_key(row), "a.eeg")
            p.with_name("a.eeg.part").write_bytes(b"old")
            http = GuestbookHttp()
            acquire.worker(db, Path(tmp) / "raw", seconds=2, reserve_gib=0, http=http)
            status = db.execute("SELECT status,bytes_present,error FROM downloads").fetchone()
            self.assertEqual(status[0], "access_action_required")
            self.assertEqual(status[1], 3)
            self.assertNotIn("DO_NOT_LOG", status[2])
            self.assertEqual(p.with_name("a.eeg.part").read_bytes(), b"old")
            acquire.worker(db, Path(tmp) / "raw", seconds=2, reserve_gib=0, http=http)
            self.assertEqual(http.calls, 1)

    def test_japanese_audit_scope_exclusions_remain_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["ID", "対象判定（暫定）", "元の一次資料URL", "出典系列キー"])
                writer.writeheader()
                writer.writerow({"ID": "1", "対象判定（暫定）": "対象外候補：頭蓋内", "元の一次資料URL": "https://zenodo.org/records/123", "出典系列キー": "zenodo_record:123"})
            rows = acquire.normalize_catalog(path)
            self.assertEqual(rows[0]["status"], "excluded")
            self.assertEqual(rows[0]["provider"], "zenodo")

    def test_restricted_zenodo_cannot_enqueue_files(self):
        record = acquire.parse_zenodo({"id": 1, "metadata": {"access_right": "restricted"},
                                      "files": [{"key": "a.edf", "size": 2}]}, "https://zenodo.org/api/records/1")
        self.assertEqual(record["status"], "restricted")
        self.assertEqual(record["files"], [])

    def test_dataverse_only_open_files_are_selected(self):
        data = {"data": {"latestVersion": {"files": [
            {"restricted": True, "dataFile": {"id": 1, "filename": "secret.edf", "filesize": 10}},
            {"restricted": False, "dataFile": {"id": 2, "filename": "open.edf", "filesize": 4}}
        ]}}}
        result = acquire.parse_dataverse(data, "https://dataverse.harvard.edu/api/datasets/1")
        self.assertEqual(result["restricted_file_count"], 1)
        self.assertEqual([x["path"] for x in result["files"]], ["open.edf"])

    def test_multimodal_release_preserves_screen_and_event_files(self):
        data = {"id": 1, "metadata": {"access_right": "open", "license": {"id": "cc-by-4.0"}},
                "files": [{"key": n, "size": 2} for n in ("bids_eeg.tar.gz", "screen.tar.gz", "events.tsv", "README.md")]}
        result = acquire.parse_zenodo(data, "https://zenodo.org/api/records/1")
        self.assertEqual(len(result["files"]), 4)
        self.assertTrue(all(x["declared_public"] for x in result["files"]))

    def test_path_traversal_rejected(self):
        for value in ("../a", "a/../../b", "/tmp/a", "a\\..\\b", "a/./b", "a//b"):
            with self.subTest(path=value), self.assertRaises(ValueError):
                acquire.safe_relative(value)

    def test_resume_206_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = job()
            path = acquire.destination(tmp, spec["dataset_key"], spec["relative_path"])
            path.with_name(path.name + ".part").write_bytes(b"abc")
            http = HttpFixture(Response(b"defgh", 206, {"Content-Range": "bytes 3-7/8"}))
            result = acquire.transfer(spec, tmp, http, 0)
            self.assertEqual(http.headers["Range"], "bytes=3-")
            self.assertEqual(path.read_bytes(), b"abcdefgh")
            self.assertEqual(result["sha256"], hashlib.sha256(b"abcdefgh").hexdigest())

    def test_server_ignoring_range_restarts_not_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = job()
            path = acquire.destination(tmp, spec["dataset_key"], spec["relative_path"])
            path.with_name(path.name + ".part").write_bytes(b"abc")
            result = acquire.transfer(spec, tmp, HttpFixture(Response(b"abcdefgh", 200, {"Content-Length": "8"})), 0)
            self.assertEqual(path.read_bytes(), b"abcdefgh")
            self.assertEqual(result["bytes_present"], 8)

    def test_incorrect_range_cannot_corrupt_existing_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = job()
            path = acquire.destination(tmp, spec["dataset_key"], spec["relative_path"])
            part = path.with_name(path.name + ".part")
            part.write_bytes(b"abc")
            with self.assertRaisesRegex(ValueError, "Range start"):
                acquire.transfer(spec, tmp, HttpFixture(Response(b"efgh", 206, {"Content-Range": "bytes 4-7/8"})), 0)
            self.assertEqual(part.read_bytes(), b"abc")
            self.assertFalse(path.exists())

    def test_integrity_failure_is_not_published_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = acquire.transfer(job(), tmp, HttpFixture(Response(b"XXXXXXXX", 200, {"Content-Length": "8"})), 0)
            self.assertEqual(result["status"], "integrity_failed")
            self.assertIn("checksum", result["error"])
            self.assertEqual(result["transferred_bytes"], 8)
            self.assertFalse((Path(tmp) / "zenodo-1-1/a.eeg").exists())

    def test_capacity_limit_stops_before_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            http = HttpFixture(Response(b"abcdefgh"))
            with patch("acquire.shutil.disk_usage", return_value=SimpleNamespace(free=15)):
                result = acquire.transfer(job(), tmp, http, reserve_bytes=10)
            self.assertEqual(result["status"], "capacity_blocked")
            self.assertIsNone(http.headers)

    def test_run_byte_limit_is_exact_and_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = job()
            result = acquire.transfer(spec, tmp, HttpFixture(Response(b"abcdefgh", 200, {"Content-Length": "8"})), 0, transfer_limit=3)
            self.assertEqual(result["status"], "paused")
            self.assertEqual(result["transferred_bytes"], 3)
            self.assertEqual((Path(tmp) / "zenodo-1-1/a.eeg.part").read_bytes(), b"abc")

    def test_existing_download_requires_checksum_revalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = job()
            path = acquire.destination(tmp, spec["dataset_key"], spec["relative_path"])
            path.write_bytes(b"XXXXXXXX")
            with self.assertRaisesRegex(ValueError, "checksum"):
                acquire.transfer(spec, tmp, HttpFixture(Response(b"abcdefgh")), 0)

    def test_200_html_signin_cannot_become_eeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "HTML response"):
                acquire.transfer(job(), tmp, HttpFixture(Response(b"abcdefgh", headers={"Content-Type": "text/html"})), 0)

    def test_database_restricted_files_not_ingested_and_resume_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = acquire.init_db(Path(tmp) / "state.sqlite")
            row = {"status": "resolved", "provider": "zenodo", "source_id": "1", "source_version": "1", "catalog_id": "000001",
                   "files": [acquire.make_file("a.eeg", "https://zenodo.org/a.eeg", 8, evidence_url="https://zenodo.org/api/records/1")]}
            self.assertEqual(acquire.ingest(db, [row]), 1)
            self.assertEqual(acquire.ingest(db, [row]), 0)
            restricted = {**row, "source_id": "2", "status": "restricted"}
            self.assertEqual(acquire.ingest(db, [restricted]), 0)
            self.assertEqual(db.execute("SELECT count(*) FROM downloads").fetchone()[0], 1)

    def test_checksum_without_validator_restart_policy_does_not_join_unknown_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = {**job(), "checksum_json": "null"}
            path = acquire.destination(tmp, spec["dataset_key"], spec["relative_path"])
            path.with_name(path.name + ".part").write_bytes(b"OLD")
            http = HttpFixture(Response(b"abcdefgh", 200, {"Content-Length": "8"}))
            acquire.transfer(spec, tmp, http, 0)
            self.assertNotIn("Range", http.headers)
            self.assertEqual(path.read_bytes(), b"abcdefgh")

    def test_changed_manifest_size_is_not_silently_treated_as_same_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = acquire.init_db(Path(tmp) / "state.sqlite")
            row = {"status": "resolved", "provider": "zenodo", "source_id": "1", "source_version": "1", "catalog_id": "000001",
                   "files": [acquire.make_file("a.eeg", "https://zenodo.org/a.eeg", 8, evidence_url="https://zenodo.org/api/records/1")]}
            acquire.ingest(db, [row])
            row["files"][0]["size_bytes"] = 12
            with self.assertRaisesRegex(ValueError, "manifest conflict"):
                acquire.ingest(db, [row], quarantine_conflicts=False)

    def test_distinct_figshare_file_ids_with_same_name_are_all_preserved(self):
        files = [acquire.make_file("sub01.7z", f"https://ndownloader.figshare.com/files/{fid}", size,
                                   evidence_url="https://api.figshare.com/v2/articles/1")
                 for fid, size in ((100, 10), (200, 20))]
        row = {"catalog_id": "000443", "provider": "figshare", "source_id": "1", "source_version": 4,
               "status": "resolved", "files": files}
        migrated, ledger = acquire.migrate_manifest_records([row])
        self.assertEqual(ledger["file_entries_before"], ledger["file_entries_after"])
        self.assertEqual(ledger["renamed_file_entries"], 2)
        self.assertTrue(migrated[0]["companion_resolution_required"])
        self.assertEqual({f["provider_file_id"] for f in migrated[0]["files"]}, {"100", "200"})
        self.assertEqual(len({f["path"] for f in migrated[0]["files"]}), 2)
        with tempfile.TemporaryDirectory() as tmp:
            db = acquire.init_db(Path(tmp) / "state.sqlite")
            self.assertEqual(acquire.ingest(db, migrated), 2)
            self.assertEqual(db.execute("SELECT sum(expected_bytes) FROM downloads").fetchone()[0], 30)

    def test_same_provider_id_conflict_is_quarantined_without_losing_other_release(self):
        def release(ident, files):
            return {"catalog_id": ident, "provider": "figshare", "source_id": ident, "source_version": 1, "status": "resolved", "files": files}
        bad = release("1", [acquire.make_file("s.edf", "https://ndownloader.figshare.com/files/100", size,
                                             evidence_url="https://api.figshare.com/v2/articles/1") for size in (10, 20)])
        good = release("2", [acquire.make_file("a.edf", "https://ndownloader.figshare.com/files/200", 30,
                                              evidence_url="https://api.figshare.com/v2/articles/2")])
        with tempfile.TemporaryDirectory() as tmp:
            db = acquire.init_db(Path(tmp) / "state.sqlite")
            self.assertEqual(acquire.ingest(db, [bad, good]), 1)
            self.assertEqual(db.execute("SELECT catalog_id FROM downloads").fetchone()[0], "2")
            self.assertEqual(db.execute("SELECT count(*) FROM quarantined_releases").fetchone()[0], 1)

    def test_migration_is_idempotent_and_seed_paths_do_not_change(self):
        files = [acquire.make_file(n, "https://zenodo.org/records/1/files/" + n, 2,
                                   evidence_url="https://zenodo.org/api/records/1") for n in ("a.set", "a.fdt")]
        rows = [{"catalog_id": "1", "provider": "zenodo", "source_id": "1", "source_version": "1", "status": "resolved", "files": files}]
        first, ledger = acquire.migrate_manifest_records(rows)
        second, again = acquire.migrate_manifest_records(first)
        self.assertEqual(first, second)
        self.assertEqual(ledger["renamed_file_entries"], 0)
        self.assertEqual(again["renamed_file_entries"], 0)
        self.assertEqual([f["path"] for f in first[0]["files"]], ["a.set", "a.fdt"])

    def test_unknown_file_size_is_not_promoted_to_downloadable(self):
        row = {"catalog_id": "1", "provider": "zenodo", "source_id": "1", "source_version": "1", "status": "resolved",
               "files": [acquire.make_file("a.edf", "https://zenodo.org/records/1/files/a.edf", None,
                                            evidence_url="https://zenodo.org/api/records/1")]}
        with tempfile.TemporaryDirectory() as tmp:
            db = acquire.init_db(Path(tmp) / "state.sqlite")
            acquire.ingest(db, [row])
            status, size = db.execute("SELECT status,expected_bytes FROM downloads").fetchone()
            self.assertEqual(status, "metadata_blocked")
            self.assertIsNone(size)

    def test_changed_etag_206_cannot_mix_versions_without_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = {**job(), "checksum_json": "null"}
            path = acquire.destination(tmp, spec["dataset_key"], spec["relative_path"])
            part = path.with_name(path.name + ".part")
            part.write_bytes(b"AAAA")
            acquire.atomic_json(path.with_name(path.name + ".part.meta.json"),
                                {"url_sha256": hashlib.sha256(spec["url"].encode()).hexdigest(), "etag": '"old"'})
            http = HttpFixture(Response(b"BBBB", 206, {"Content-Range": "bytes 4-7/8", "ETag": '"new"'}))
            with self.assertRaisesRegex(ValueError, "validator changed"):
                acquire.transfer(spec, tmp, http, 0)
            self.assertEqual(part.read_bytes(), b"AAAA")
            self.assertFalse(path.exists())

    def test_missing_or_changed_last_modified_206_is_not_complete(self):
        for response_value in (None, "Thu, 02 Jan 2025 00:00:00 GMT"):
            with self.subTest(value=response_value), tempfile.TemporaryDirectory() as tmp:
                spec = {**job(), "checksum_json": "null"}
                path = acquire.destination(tmp, spec["dataset_key"], spec["relative_path"])
                part = path.with_name(path.name + ".part")
                part.write_bytes(b"AAAA")
                acquire.atomic_json(path.with_name(path.name + ".part.meta.json"),
                                    {"url_sha256": hashlib.sha256(spec["url"].encode()).hexdigest(), "last_modified": "Wed, 01 Jan 2025 00:00:00 GMT"})
                headers = {"Content-Range": "bytes 4-7/8"}
                if response_value:
                    headers["Last-Modified"] = response_value
                with self.assertRaisesRegex(ValueError, "validator"):
                    acquire.transfer(spec, tmp, HttpFixture(Response(b"BBBB", 206, headers)), 0)
                self.assertEqual(part.read_bytes(), b"AAAA")

    def test_collision_migration_retains_existing_raw_copy_and_records_alias(self):
        first = acquire.make_file("sub01.7z", "https://ndownloader.figshare.com/files/100", 10,
                                  evidence_url="https://api.figshare.com/v2/articles/1")
        second = acquire.make_file("sub01.7z", "https://ndownloader.figshare.com/files/200", 20,
                                   evidence_url="https://api.figshare.com/v2/articles/1")
        row = {"catalog_id": "1", "provider": "figshare", "source_id": "1", "source_version": 1, "status": "resolved", "files": [first]}
        with tempfile.TemporaryDirectory() as tmp:
            db = acquire.init_db(Path(tmp) / "state.sqlite")
            acquire.ingest(db, [row])
            db.execute("UPDATE downloads SET status='complete',bytes_present=10")
            db.commit()
            original = acquire.destination(tmp, acquire.dataset_storage_key(row), "sub01.7z")
            original.write_bytes(b"0123456789")
            self.assertEqual(acquire.ingest(db, [{**row, "files": [first, second]}]), 2)
            self.assertEqual(original.read_bytes(), b"0123456789")
            self.assertEqual(db.execute("SELECT count(*) FROM preserved_manifest_aliases").fetchone()[0], 1)
            self.assertEqual(dict(db.execute("SELECT status,count(*) FROM downloads GROUP BY status")), {"legacy_preserved": 1, "queued": 2})

    def test_short_clean_eof_keeps_partial_and_can_resume_to_verified_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = job()
            first = acquire.transfer(spec, tmp, HttpFixture(Response(b"abc", 200, {"Content-Length": "8", "ETag": '"one"'})), 0)
            self.assertEqual(first["status"], "retry")
            self.assertEqual(first["bytes_present"], 3)
            self.assertEqual(first["transferred_bytes"], 3)
            path = Path(tmp) / "zenodo-1-1/a.eeg"
            self.assertFalse(path.exists())
            second_http = HttpFixture(Response(b"defgh", 206, {"Content-Range": "bytes 3-7/8", "ETag": '"one"'}))
            second = acquire.transfer(spec, tmp, second_http, 0)
            self.assertEqual(second_http.headers["Range"], "bytes=3-")
            self.assertEqual(second["status"], "complete")
            self.assertEqual(second["transferred_bytes"], 5)
            self.assertEqual(path.read_bytes(), b"abcdefgh")

    def test_short_eof_traffic_and_backoff_persist_in_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = acquire.init_db(Path(tmp) / "state.sqlite")
            record = {"catalog_id": "1", "provider": "zenodo", "source_id": "1", "source_version": "1", "status": "resolved",
                      "files": [acquire.make_file("a.eeg", "https://zenodo.org/records/1/files/a.eeg", 8,
                                                   evidence_url="https://zenodo.org/api/records/1")]}
            acquire.ingest(db, [record])
            result = acquire.worker(db, Path(tmp) / "raw", reserve_gib=0, seconds=5, max_files=1,
                                    http=HttpFixture(Response(b"abc", 200, {"Content-Length": "8", "ETag": '"one"'})))
            self.assertEqual(result["bytes_transferred_this_run"], 3)
            row = db.execute("SELECT status,bytes_present,retry_at,error FROM downloads").fetchone()
            self.assertEqual(row[0], "retry")
            self.assertEqual(row[1], 3)
            self.assertGreater(row[2], __import__('time').time())
            self.assertIn("incomplete_transfer_eof", row[3])

    def test_response_larger_than_expected_remains_integrity_failure_and_counts_traffic(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = acquire.transfer(job(), tmp, HttpFixture(Response(b"abcdefghi", 200)), 0)
            self.assertEqual(result["status"], "integrity_failed")
            self.assertEqual(result["transferred_bytes"], 9)

    def test_interrupted_read_keeps_received_partial_and_counts_traffic(self):
        class BrokenResponse(Response):
            def read(self, _size):
                raise acquire.IncompleteRead(b"abc", 8)
        with tempfile.TemporaryDirectory() as tmp:
            result = acquire.transfer(job(), tmp, HttpFixture(BrokenResponse(b"", 200, {"Content-Length": "8"})), 0)
            self.assertEqual(result["status"], "retry")
            self.assertEqual(result["bytes_present"], 3)
            self.assertEqual(result["transferred_bytes"], 3)
            self.assertEqual((Path(tmp) / "zenodo-1-1/a.eeg.part").read_bytes(), b"abc")


if __name__ == "__main__":
    unittest.main()
