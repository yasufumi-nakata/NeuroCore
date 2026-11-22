import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import smac_storage_guard as g
import acquire


class SmacStorageGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.policy = self.home / "policy.json"
        self.nodes = self.home / "nodes.json"
        self.policy.write_text(json.dumps({"download_location":"smac_only", "local_bulk_downloads_allowed":False, "local_fallback_allowed":False, "excluded_nodes":["smac19"]}))
        self.nodes.write_text(json.dumps({"nodes":[{"node_id":"smac31","transport":"ssh"},{"node_id":"smac06","transport":"ssh"},{"node_id":"smac19","transport":"ssh"}]}))
        self.raw = self.home / "pj-xavier/smac31/raw"

    def check(self, root=None, host="smac31", **kw):
        return g.require_smac_raw_root(root if root is not None else self.raw, policy_path=kw.pop("policy",self.policy), nodes_path=self.nodes, hostname=host, home=self.home)

    def test_correct_node_and_root(self):
        self.assertEqual(self.check()['node_id'],"smac31")
        self.assertEqual(self.check(self.raw / "dataset")['node_id'],"smac31")
        self.assertFalse(self.raw.exists())  # preflight never creates storage

    def test_workstation_and_excluded_node_rejected(self):
        for host in ("vpnclient231.sfc.keio.ac.jp", "smac19.sfc.keio.ac.jp", "smac99"):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError, "authorized smac"):
                self.check(host=host)

    def test_wrong_node_root_rejected(self):
        with self.assertRaisesRegex(ValueError,"dedicated raw"):self.check(self.home/"pj-xavier/smac06/raw")
        with self.assertRaisesRegex(ValueError,"absolute"):self.check(Path("relative/raw"))

    def test_symlink_escaping_root_rejected(self):
        self.raw.mkdir(parents=True)
        (self.raw / "link").symlink_to(self.home)
        with self.assertRaisesRegex(ValueError,"dedicated raw"):self.check(self.raw / "link" / "dataset")

    def test_missing_policy_blocks(self):
        with self.assertRaisesRegex(ValueError,"absent"):self.check(policy=self.home/"missing.json")

    def test_cli_download_refuses_before_touching_manifest_or_db(self):
        with patch.object(g,"DEFAULT_POLICY_PATH",self.policy),patch.object(g,"DEFAULT_NODES_PATH",self.nodes),patch.object(g.socket,"gethostname",return_value="vpnclient231.sfc.keio.ac.jp"):
            db=self.home/"should-not-exist.sqlite"
            with self.assertRaisesRegex(ValueError,"authorized smac"):
                acquire.main(["download","--manifest",str(self.home/"nonexistent.jsonl"),"--db",str(db),"--root",str(self.raw)])
            self.assertFalse(db.exists())


if __name__ == "__main__": unittest.main()
