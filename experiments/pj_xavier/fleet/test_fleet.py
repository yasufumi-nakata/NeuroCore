import datetime as dt
import json
import importlib.util
import subprocess
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fleet
import node_worker

NOW = dt.datetime(2026, 11, 22, tzinfo=dt.timezone.utc)
GIB = 1024**3


def node(name, usable=100, **extra):
    return {"node_id": name, "transport": "ssh", "ssh_target": name, "status": "ready",
            "preflight_success": True, "ssh_auth": "batch_authenticated", "hostkey_verified": True,
            "free_bytes": GIB + usable, "reserve_bytes": GIB, "measured_at": NOW.isoformat(), **extra}


def row(ident="000001", sizes=(20, 30), names=None, **extra):
    names = names or [f"f{i}.edf" for i in range(len(sizes))]
    return {"catalog_id": ident, "provider": "zenodo", "source_id": ident, "source_version": ident,
            "scope_decision": "include_provisional", "status": "resolved",
            "files": [{"path": name, "url": "https://zenodo.org/records/1/files/" + name,
                       "size_bytes": size, "declared_public": True} for name, size in zip(names, sizes)], **extra}


class PlanningTests(unittest.TestCase):
    def test_unresolved_file_extension_is_never_accepted_as_EEG_proof(self):
        plan = fleet.make_plan([row(scope_decision="unresolved")], {"nodes": [node("smac31")]}, at=NOW)
        self.assertEqual(plan["summary"]["assigned_files"], 0)
        self.assertEqual(len(plan["review"]), 1)

    def test_excluded_and_duplicate_are_not_assigned(self):
        records = [row("1", scope_decision="exclude_intracranial_provisional"), row("2", status="duplicate"),
                   row("3", scope_decision="対象外候補：非ヒト")]
        plan = fleet.make_plan(records, {"nodes": [node("smac31")]}, at=NOW)
        self.assertEqual(len(plan["excluded"]), 3)
        self.assertEqual(plan["summary"]["assigned_files"], 0)

    def test_unknown_sizes_and_oversized_atomic_groups_are_blocked(self):
        records = [row("1", sizes=(None,)), row("2", sizes=(70, 40), names=("a.set", "a.fdt"))]
        plan = fleet.make_plan(records, {"nodes": [node("a", usable=100), node("b", usable=100)]}, at=NOW)
        self.assertEqual(plan["summary"]["assigned_files"], 0)
        self.assertEqual(len(plan["blocked"]), 2)

    def test_capacity_reserve_pending_and_workspace_all_subtracted(self):
        inventory = {"nodes": [node("a", usable=200, reserved_pending_bytes=40, preprocessing_headroom_bytes=30)]}
        plan = fleet.make_plan([row(sizes=(100, 60))], inventory, at=NOW)
        self.assertEqual(plan["nodes"]["a"]["allocatable_bytes"], 130)
        self.assertEqual(plan["nodes"]["a"]["assigned_bytes"], 100)
        self.assertEqual(len(plan["blocked"]), 1)

    def test_stale_historical_nodes_have_zero_capacity(self):
        historical = node("smac01", usable=10000, measured_at="2026-10-30T00:00:00+00:00")
        unverified = node("smac31", usable=10000, preflight_success=False)
        plan = fleet.make_plan([row()], {"nodes": [historical, unverified]}, at=NOW)
        self.assertEqual(plan["summary"]["assigned_file_bytes"], 0)
        self.assertEqual(sum(n["allocatable_bytes"] for n in plan["nodes"].values()), 0)

    def test_shards_deterministic_disjoint_and_companions_stay_together(self):
        records = [row("1", sizes=(40, 40, 20), names=("a.vhdr", "a.eeg", "events.tsv")), row("2", sizes=(80,))]
        inventory = {"nodes": [node("b", 100), node("a", 100)]}
        plan = fleet.make_plan(records, inventory, at=NOW)
        reversed_plan = fleet.make_plan(list(reversed(records)), {"nodes": list(reversed(inventory["nodes"]))}, at=NOW)
        self.assertEqual(plan["plan_sha256"], reversed_plan["plan_sha256"])
        locations = {}
        for host, assigned in plan["assignments"].items():
            for rec in assigned:
                for f in rec["files"]:
                    key = (rec["catalog_id"], f["path"])
                    self.assertNotIn(key, locations)
                    locations[key] = host
        self.assertEqual(locations[("1", "a.vhdr")], locations[("1", "a.eeg")])
        self.assertEqual(len(locations), 4)

    def test_ssh_preflight_never_prompts_or_disables_host_check(self):
        calls = []
        def fake_run(args, **kw):
            calls.append((args, kw))
            return SimpleNamespace(returncode=255, stdout="", stderr="Permission denied")
        result = fleet.preflight_node(node("smac31"), run=fake_run)
        argv, kwargs = calls[0]
        self.assertIn("BatchMode=yes", argv)
        self.assertIn("StrictHostKeyChecking=yes", argv)
        self.assertNotIn("StrictHostKeyChecking=no", argv)
        self.assertIn("shutil.disk_usage", kwargs["input"])
        self.assertFalse(result["preflight_success"])
        self.assertEqual(result["status"], "blocked")

    def test_colab_is_never_accessed_with_ssh_preflight(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("no subprocess should run")
        result = fleet.preflight_node({"node_id": "colab_l4", "transport": "workstation_colab_cli"}, run=forbidden)
        self.assertIn("workstation_CLI", result["reason"])

    def test_duplicate_provider_path_conflict_blocks_release(self):
        rec = row(sizes=(10, 20), names=("a.edf", "a.edf"))
        plan = fleet.make_plan([rec], {"nodes": [node("a")]}, at=NOW)
        self.assertEqual(plan["summary"]["assigned_files"], 0)
        self.assertEqual(plan["blocked"][0]["reason"], "conflicting_files_for_same_provider_path")


class BundlingTests(unittest.TestCase):
    def policy_files(self, root):
        policy = root / "storage_policy.json"
        registry = root / "nodes.smac_only.json"
        policy.write_text(json.dumps({"download_location": "smac_only",
                                      "local_bulk_downloads_allowed": False,
                                      "local_fallback_allowed": False,
                                      "excluded_nodes": ["smac19"]}))
        registry.write_text(json.dumps({"nodes": [node("smac31"), node("smac19")]}))
        return policy, registry

    def build(self, root, target="smac31"):
        policy, registry = self.policy_files(root)
        plan = fleet.make_plan([row()], {"nodes": [node(target)]}, at=NOW)
        source = Path(__file__).resolve().parent.parent / "acquisition/acquire.py"
        commands = fleet.write_bundle(plan, root / "bundle", source,
                                     Path(node_worker.__file__), storage_policy=policy,
                                     nodes_registry=registry)
        return root / "bundle" / target, commands, source

    def test_bundle_preserves_guard_code_and_repository_relative_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, commands, source = self.build(root)
            code = bundle / "code/experiments/pj_xavier/acquisition"
            self.assertEqual((code / "acquire.py").read_bytes(), source.read_bytes())
            guard_path = code / "smac_storage_guard.py"
            self.assertEqual(guard_path.read_bytes(), source.with_name("smac_storage_guard.py").read_bytes())
            spec = importlib.util.spec_from_file_location("bundle_storage_guard", guard_path)
            guard = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(guard)
            self.assertEqual(guard.DEFAULT_POLICY_PATH.read_bytes(), (root / "storage_policy.json").read_bytes())
            self.assertEqual(guard.DEFAULT_NODES_PATH.read_bytes(), (root / "nodes.smac_only.json").read_bytes())
            home = root / "remote-home"
            valid = guard.require_smac_raw_root(home / "pj-xavier/smac31/raw", hostname="smac31", home=home)
            self.assertEqual(valid["node_id"], "smac31")
            with self.assertRaises(ValueError):
                guard.require_smac_raw_root(home / "pj-xavier/smac31/raw", hostname="workstation", home=home)
            job = json.loads((bundle / "download.json").read_text())
            self.assertEqual(job["command"][1], "~/pj-xavier/smac31/code/experiments/pj_xavier/acquisition/acquire.py")
            upload = commands[0]["steps"][1]["argv"]
            self.assertIn("-r", upload)
            self.assertIn(str((bundle / "code").resolve()), upload)
            self.assertEqual(json.loads((bundle / "train.json").read_text())["state"], "blocked")

    def test_bundled_cli_blocks_wrong_local_root_before_ledger_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, _, _ = self.build(root)
            db, raw = root / "forbidden.sqlite", root / "forbidden-raw"
            result = subprocess.run([sys.executable,
                                     str(bundle / "code/experiments/pj_xavier/acquisition/acquire.py"),
                                     "download", "--manifest", str(bundle / "assigned.jsonl"),
                                     "--db", str(db), "--root", str(raw)],
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertIn("ValueError", result.stderr)
            self.assertFalse(db.exists())
            self.assertFalse(raw.exists())

    def test_missing_policy_blocks_bundle_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, registry = self.policy_files(root)
            plan = fleet.make_plan([row()], {"nodes": [node("smac31")]}, at=NOW)
            source = Path(__file__).resolve().parent.parent / "acquisition/acquire.py"
            with self.assertRaisesRegex(ValueError, "dependency missing"):
                fleet.write_bundle(plan, root / "bundle", source, Path(node_worker.__file__),
                                   storage_policy=root / "absent.json", nodes_registry=registry)
            self.assertFalse((root / "bundle").exists())

    def test_smac19_is_not_deployed_even_if_registry_mentions_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "not an authorized smac"):
                self.build(root, target="smac19")
            self.assertFalse((root / "bundle").exists())


class WorkerTests(unittest.TestCase):
    def test_quarantined_release_prevents_false_fleet_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "download.sqlite"
            db = sqlite3.connect(path)
            db.execute("CREATE TABLE downloads(status TEXT)")
            db.execute("INSERT INTO downloads VALUES('complete')")
            db.execute("CREATE TABLE quarantined_releases(dataset_key TEXT)")
            db.execute("INSERT INTO quarantined_releases VALUES('bad-release')")
            db.commit()
            db.close()
            done, evidence = node_worker.completed({"completion_check": {"kind": "download_sqlite", "path": str(path)}})
            self.assertFalse(done)
            self.assertEqual(evidence["quarantined_release_count"], 1)

    def test_download_exit_zero_is_not_all_files_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = sqlite3.connect(root / "download.sqlite")
            db.execute("CREATE TABLE downloads(status TEXT)")
            db.executemany("INSERT INTO downloads VALUES(?)", [("complete",), ("paused",)])
            db.commit()
            db.close()
            job = {"kind": "download", "state": "ready", "data_root": tmp, "reserve_bytes": GIB,
                   "command": [sys.executable, "-c", "pass"],
                   "completion_check": {"kind": "download_sqlite", "path": str(root / "download.sqlite")}}
            path = root / "job.json"
            path.write_text(json.dumps(job))
            with patch("node_worker.time.sleep"), patch("node_worker.shutil.disk_usage", return_value=SimpleNamespace(free=3 * GIB)):
                result = node_worker.run_job(path)
            self.assertEqual(result["state"], "paused")
            self.assertEqual(result["download_status_counts"]["paused"], 1)

    def test_unconfigured_AI_label_job_stays_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "label.json"
            path.write_text(json.dumps({"kind": "label", "state": "blocked", "reason": "model_not_verified"}))
            result = node_worker.run_job(path)
            self.assertEqual(result["state"], "blocked")
            self.assertEqual(result["reason"], "model_not_verified")

    def test_kind_lock_prevents_second_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "locks").mkdir()
            path = root / "job.json"
            path.write_text(json.dumps({"kind": "train", "state": "blocked"}))
            import fcntl
            with (root / "locks/train.lock").open("w") as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = node_worker.run_job(path)
            self.assertEqual(result["state"], "already_running")


if __name__ == "__main__":
    unittest.main()
