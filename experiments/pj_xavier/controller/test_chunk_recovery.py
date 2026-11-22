import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import controller as c
import remote_core as r
from test_controller import ledger, fixture_config


class ChunkClient:
    def __init__(self, root):
        self.root, self.requests, self.downloads = root, [], []
        self.interrupt_offset, self.corrupt, self.after_download = None, False, None

    def remote(self, action, config, request, **kwargs):
        if action == "raw_chunk":
            self.requests.append(dict(request))
            if request["offset"] == self.interrupt_offset:
                raise RuntimeError("simulated connection interruption")
            return r.raw_chunk(config, request, root_override=self.root)
        if action == "ack_raw_chunk":
            return r.acknowledge_raw_chunk(config, request, root_override=self.root)
        raise AssertionError(action)

    def download(self, remote, local, **kwargs):
        self.downloads.append(remote)
        shutil.copyfile(remote, local)
        if self.corrupt:
            local.write_bytes(b"X" * local.stat().st_size)
        if self.after_download:
            self.after_download(remote, local)


class ChunkRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.body = b"0123456789abcdefghij"
        ledger(self.root / "downloads.sqlite", "complete", self.body)
        self.source = self.root / "raw/zenodo-1-v1/signal.bin"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(self.body)
        self.config = fixture_config(self.root)
        self.config.update(remote_root=str(self.root), raw_chunk_bytes=4)
        self.item = r.raw_inventory(self.config, root_override=self.root)["files"][0]
        self.partial = self.root / "archive/signal.controller.download.part"
        self.client = ChunkClient(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def recover(self):
        return c.download_raw_chunks(self.client, self.config, self.item, self.partial, reserve_bytes=0)

    def test_resume_uses_verified_prefix_and_never_downloads_whole_source(self):
        self.client.interrupt_offset = 8
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.recover()
        self.assertEqual(self.partial.read_bytes(), self.body[:8])
        self.client.interrupt_offset = None
        receipt = self.recover()
        self.assertEqual(receipt["status"], "complete_sha256_verified")
        self.assertEqual(self.partial.read_bytes(), self.body)
        self.assertEqual([v["offset"] for v in receipt["chunks"]], [0, 4, 8, 12, 16])
        self.assertEqual(len(self.client.downloads), 5)
        self.assertTrue(all("controller_chunks" in p for p in self.client.downloads))
        self.assertEqual(list((self.root / "controller_chunks").rglob("*.bin")), [])
        self.assertEqual(self.source.read_bytes(), self.body)

    def test_chunk_checksum_failure_never_appends_unverified_bytes(self):
        self.client.corrupt = True
        with self.assertRaisesRegex(ValueError, "download_checksum"):
            self.recover()
        self.assertEqual(self.partial.stat().st_size, 0)
        receipt = json.loads(self.partial.with_name(self.partial.name + ".chunks.json").read_text())
        self.assertEqual(receipt["confirmed_bytes"], 0)

    def test_crash_between_append_and_receipt_commit_resumes_without_duplicate_bytes(self):
        original = c.json_save
        crashed = []

        def interrupted_save(path, value):
            if value.get("confirmed_bytes") == 4 and value.get("pending_ack") and not crashed:
                crashed.append(True)
                raise RuntimeError("power loss after durable append")
            return original(path, value)

        with patch.object(c, "json_save", side_effect=interrupted_save):
            with self.assertRaisesRegex(RuntimeError, "power loss"):
                self.recover()
        self.assertEqual(self.partial.read_bytes(), self.body[:4])
        receipt = self.recover()
        self.assertEqual(receipt["confirmed_bytes"], len(self.body))
        self.assertEqual(self.partial.read_bytes(), self.body)
        self.assertEqual(len(self.client.downloads), 5)

    def test_changed_remote_source_identity_is_held_with_prefix_preserved(self):
        self.client.interrupt_offset = 4
        with self.assertRaises(RuntimeError):
            self.recover()
        self.source.write_bytes(b"X" * len(self.body))
        self.client.interrupt_offset = None
        with self.assertRaisesRegex(ValueError, "source_identity_changed"):
            self.recover()
        self.assertEqual(self.partial.read_bytes(), self.body[:4])

    def test_changed_local_committed_prefix_is_detected_before_more_downloads(self):
        self.client.interrupt_offset = 4
        with self.assertRaises(RuntimeError):
            self.recover()
        self.partial.write_bytes(b"BAD!")
        calls = len(self.client.requests)
        self.client.interrupt_offset = None
        with self.assertRaisesRegex(ValueError, "committed_prefix_checksum"):
            self.recover()
        self.assertEqual(len(self.client.requests), calls)

    def test_final_source_hash_is_required_even_when_all_chunk_hashes_match(self):
        import sqlite3
        self.item["sha256"] = "0" * 64
        with sqlite3.connect(self.root / "downloads.sqlite") as db:
            db.execute("UPDATE downloads SET sha256=?", (self.item["sha256"],))
        with self.assertRaisesRegex(ValueError, "final_source_checksum"):
            self.recover()
        receipt = json.loads(self.partial.with_name(self.partial.name + ".chunks.json").read_text())
        self.assertNotEqual(receipt["status"], "complete_sha256_verified")
        self.assertEqual(self.partial.read_bytes(), self.body)

    def test_expired_absolute_deadline_performs_no_remote_transfer(self):
        self.config["controller_deadline_epoch"] = 1
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            self.recover()
        self.assertEqual(self.client.requests, [])
        self.assertEqual(self.client.downloads, [])

    def test_remote_size_bound_and_ack_identity_preserve_source(self):
        request = {"source_path": str(self.source), "source_sha256": self.item["sha256"],
                   "source_size": len(self.body), "offset": 0, "length": r.RAW_CHUNK_LIMIT + 1}
        with self.assertRaisesRegex(ValueError, "out_of_range"):
            r.raw_chunk(self.config, request, root_override=self.root)
        request["length"] = 4
        descriptor = r.raw_chunk(self.config, request, root_override=self.root)
        changed = dict(descriptor, chunk_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "content_mismatch"):
            r.acknowledge_raw_chunk(self.config, changed, root_override=self.root)
        self.assertTrue(Path(descriptor["remote_path"]).is_file())
        self.assertEqual(self.source.read_bytes(), self.body)

    def test_wrong_offset_or_source_descriptor_cannot_be_appended(self):
        original = self.client.remote

        def changed(action, config, request, **kwargs):
            result = original(action, config, request, **kwargs)
            return dict(result, offset=1) if action == "raw_chunk" else result

        self.client.remote = changed
        with self.assertRaisesRegex(ValueError, "source_offset_or_size"):
            self.recover()
        self.assertEqual(self.partial.stat().st_size, 0)

    def test_interrupted_ack_finishes_only_matching_retained_sidecar(self):
        request = {"source_path": str(self.source), "source_sha256": self.item["sha256"],
                   "source_size": len(self.body), "offset": 0, "length": 4}
        descriptor = r.raw_chunk(self.config, request, root_override=self.root)
        target = Path(descriptor["remote_path"])
        target.unlink()  # interruption between binary and sidecar cleanup
        with self.assertRaisesRegex(ValueError, "content_mismatch"):
            r.acknowledge_raw_chunk(self.config, dict(descriptor, chunk_sha256="0" * 64), root_override=self.root)
        self.assertTrue(target.with_suffix(".json").exists())
        self.assertTrue(r.acknowledge_raw_chunk(self.config, descriptor, root_override=self.root)["recovered_interrupted_ack"])
        self.assertTrue(r.acknowledge_raw_chunk(self.config, descriptor, root_override=self.root)["already_absent"])
        self.assertEqual(self.source.read_bytes(), self.body)


if __name__ == "__main__":
    unittest.main()
