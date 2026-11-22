"""Synthetic ZIP/HTTP fixtures for bounded metadata reconciliation; no network."""
import base64
import copy
import hashlib
import io
import re
import unittest
from unittest import mock
import zipfile

import reconcile_kaggle_zip_transport as r


MEMBER = "dataset/subject/eeg/file.set"
URL = "https://www.kaggle.com/api/v1/datasets/download/owner/source?file_name=" + r.urllib.parse.quote(MEMBER, safe="")


def fixture(member=MEMBER, data=b"synthetic EEG container " * 300, extra=False, symlink=False):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if symlink:
            item = zipfile.ZipInfo(member); item.create_system = 3; item.external_attr = 0o120777 << 16
            archive.writestr(item, data)
        else:
            archive.writestr(member, data)
        if extra:
            archive.writestr("unrequested.bin", b"x")
    return buffer.getvalue(), {"name": MEMBER, "totalBytes": len(data)}


class Response(io.BytesIO):
    def __init__(self, value, status, headers, url):
        super().__init__(value); self.status = status; self.headers = headers; self.url = url


class Remote:
    def __init__(self, payload, *, ignore_range=False, change_validator=False, missing_md5=False, origin=None):
        self.payload = payload
        self.calls = []
        self.ignore_range, self.change_validator, self.missing_md5 = ignore_range, change_validator, missing_md5
        self.origin = origin or "https://storage.googleapis.com/kaggle-data-sets/123/456/compressed/" + MEMBER + ".zip"

    def __call__(self, request, timeout):
        self.calls.append(request)
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        start, end = map(int, match.groups())
        etag = '"generation-1"' if not self.change_validator or len(self.calls) == 1 else '"generation-2"'
        headers = {"Content-Range": f"bytes {start}-{end}/{len(self.payload)}", "ETag": etag}
        if not self.missing_md5:
            headers["x-goog-hash"] = "md5=" + base64.b64encode(hashlib.md5(self.payload).digest()).decode()
        return Response(self.payload[start:end + 1], 200 if self.ignore_range else 206, headers, self.origin)


class TestKaggleZipTransport(unittest.TestCase):
    def test_binds_source_version_member_uncompressed_size_and_archive_checksum(self):
        payload, official = fixture()
        remote = Remote(payload)
        proof = r.inspect_transport(URL, 6, official, opener=remote)
        self.assertEqual(proof["source"]["version"], 6)
        self.assertEqual(proof["source_uncompressed_bytes"], official["totalBytes"])
        self.assertEqual(proof["archive_transport_bytes"], len(payload))
        self.assertEqual(proof["published_archive_checksum"]["value"], hashlib.md5(payload).hexdigest())
        self.assertFalse(proof["archive_downloaded"])
        self.assertFalse(proof["full_member_crc_verified"])
        self.assertLess(proof["metadata_range_bytes"], r.MAX_METADATA_BYTES)
        self.assertTrue(all("datasetVersionNumber=6" in q.full_url for q in remote.calls))
        self.assertTrue(all(q.get_header("If-match") == '"generation-1"' for q in remote.calls[1:]))

    def test_corrected_proposal_preserves_original_metadata_and_does_not_mutate_input(self):
        payload, official = fixture()
        proof = r.inspect_transport(URL, 6, official, opener=Remote(payload))
        original = {"url": URL, "size_bytes": official["totalBytes"], "path": "source_123/" + MEMBER + ".zip",
                    "published_checksum": None, "declared_public": True}
        before = copy.deepcopy(original)
        corrected = r.corrected_file_metadata(original, proof)
        self.assertEqual(original, before)
        self.assertEqual(corrected["size_bytes"], len(payload))
        self.assertEqual(corrected["path"], original["path"])
        self.assertEqual(corrected["transport_metadata_correction"]["original_size_bytes"], official["totalBytes"])
        self.assertTrue(corrected["transport_metadata_correction"]["full_member_crc_and_source_checks_required_before_training"])

    def test_rejects_source_size_mismatch(self):
        payload, official = fixture(); official["totalBytes"] += 1
        with self.assertRaisesRegex(ValueError, "uncompressed size"):
            r.inspect_transport(URL, 6, official, opener=Remote(payload))

    def test_rejects_wrong_member_or_multiple_payloads(self):
        for kwargs in [{"member": "other.set"}, {"extra": True}]:
            with self.subTest(kwargs=kwargs):
                payload, official = fixture(**kwargs)
                with self.assertRaises(ValueError):
                    r.inspect_transport(URL, 6, official, opener=Remote(payload))

    def test_rejects_symlink_and_traversal(self):
        for kwargs in [{"symlink": True}, {"member": "../file.set"}]:
            with self.subTest(kwargs=kwargs):
                payload, official = fixture(**kwargs)
                with self.assertRaises(ValueError):
                    r.inspect_transport(URL, 6, official, opener=Remote(payload))

    def test_requires_real_partial_responses_stable_validator_and_object_md5(self):
        payload, official = fixture()
        for kwargs in [{"ignore_range": True}, {"change_validator": True}, {"missing_md5": True}]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    r.inspect_transport(URL, 6, official, opener=Remote(payload, **kwargs))

    def test_rejects_other_origin_and_object(self):
        payload, official = fixture()
        for origin in ["https://example.com/file.zip",
                       "https://storage.googleapis.com/kaggle-data-sets/123/456/compressed/other.zip"]:
            with self.subTest(origin=origin):
                with self.assertRaises(ValueError):
                    r.inspect_transport(URL, 6, official, opener=Remote(payload, origin=origin))

    def test_metadata_ranges_are_bounded(self):
        payload, _ = fixture(data=bytes(range(256)) * 1000)
        reader = r.MetadataRangeReader(r.source_descriptor(URL, 6), Remote(payload))
        with mock.patch.object(r, "MAX_SINGLE_RANGE", 3):
            with self.assertRaisesRegex(ValueError, "Unbounded"):
                reader.read(4)
        with mock.patch.object(r, "MAX_RANGE_REQUESTS", reader.requests):
            with self.assertRaisesRegex(ValueError, "budget"):
                reader.read(1)

    def test_version_url_and_member_are_strict(self):
        cases = [("http://www.kaggle.com/api/v1/datasets/download/a/b?file_name=x", 1),
                 (URL + "&datasetVersionNumber=5", 6), (URL + "&extra=1", 6),
                 ("https://www.kaggle.com/api/v1/datasets/download/a/b?file_name=../x", 1),
                 (URL, 0)]
        for url, version in cases:
            with self.subTest(url=url, version=version):
                with self.assertRaises(ValueError):
                    r.source_descriptor(url, version)

    def test_multiple_google_hash_headers_preserve_md5_after_crc32c(self):
        from email.message import Message
        headers = Message()
        headers.add_header("x-goog-hash", "crc32c=AAAAAA==")
        headers.add_header("x-goog-hash", "md5=" + base64.b64encode(bytes(range(16))).decode())
        self.assertEqual(r.google_md5(headers), {"algorithm": "md5", "value": bytes(range(16)).hex()})

    def test_cannot_replace_checksum_or_apply_modified_proof(self):
        payload, official = fixture()
        proof = r.inspect_transport(URL, 6, official, opener=Remote(payload))
        file = {"url": URL, "size_bytes": official["totalBytes"], "path": MEMBER + ".zip",
                "published_checksum": {"algorithm": "md5", "value": "0" * 32}}
        with self.assertRaisesRegex(ValueError, "must not be replaced"):
            r.corrected_file_metadata(file, proof)
        file["published_checksum"] = None
        proof["archive_transport_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "seal differs"):
            r.corrected_file_metadata(file, proof)


if __name__ == "__main__":
    unittest.main()
