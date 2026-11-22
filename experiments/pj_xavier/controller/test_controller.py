import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import controller as c
import remote_core as r


SCHEMA = """CREATE TABLE downloads (
id TEXT PRIMARY KEY, dataset_key TEXT, relative_path TEXT, url TEXT,
expected_bytes INTEGER, checksum_json TEXT, status TEXT, bytes_present INTEGER,
sha256 TEXT, error TEXT, updated_at TEXT, lease_until REAL DEFAULT 0, retry_at REAL DEFAULT 0)"""


def ledger(path, status, body=b"EEG DATA", sha=None, expected=None):
    with sqlite3.connect(path) as db:
        db.execute(SCHEMA)
        db.execute("INSERT INTO downloads(id,dataset_key,relative_path,url,expected_bytes,checksum_json,status,bytes_present,sha256) VALUES(?,?,?,?,?,?,?,?,?)",
                   ("file1", "zenodo-1-v1", "signal.bin", "https://zenodo.org/records/1/files/signal.bin",
                    len(body) if expected is None else expected,
                    json.dumps({"algorithm": "md5", "value": hashlib.md5(body).hexdigest()}),
                    status, len(body), sha if sha is not None else hashlib.sha256(body).hexdigest()))


def fixture_config(root):
    config = c.default_config()
    config.update(local_root=str(root / "local"), raw_archive_root=str(root / "archive"),
                  local_reserve_bytes=0, remote_reserve_bytes=0, local_jobs=[], completion_gates=[], allow_idle_stop=True, idle_grace_seconds=0)
    config["remote_jobs"] = [{"job_id": "download", "kind": "download", "pid_file": "download.pid", "db": "downloads.sqlite",
                              "max_runtime_seconds": 14400, "allow_resume": True,
                              "expected_tokens": ["/content/xavier/code/acquire.py", "download"]}]
    return config


class FakeDownloader:
    def __init__(self, bodies):
        self.bodies, self.calls = bodies, []

    def download(self, source, target, **kwargs):
        self.calls.append((source, str(target)))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.bodies[source])


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = fixture_config(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def write_proc(self, state="S", start_ticks=1234, argv=None):
        proc = self.root / "proc"
        (proc / "42").mkdir(parents=True, exist_ok=True)
        (proc / "stat").write_text("btime 1000\n")
        fields = [state, "1"] + ["0"] * 17 + [str(start_ticks)] + ["0"] * 4
        (proc / "42/stat").write_text("42 (a process with ) spaces) " + " ".join(fields))
        (proc / "42/cmdline").write_bytes(b"\0".join(s.encode() for s in (argv or ["python", "/content/xavier/code/acquire.py", "download"])))
        return proc

    def raw_fixture(self, body=b"EEG DATA"):
        ledger(self.root / "downloads.sqlite", "complete", body)
        target = self.root / "raw/zenodo-1-v1/signal.bin"
        target.parent.mkdir(parents=True)
        target.write_bytes(body)
        return r.raw_inventory(self.config, root_override=self.root)

    def idle_state(self, inventory):
        archive = self.root / "snapshot.zip"
        archive.write_bytes(b"verified snapshot fixture")
        return ({"ok": True, "process_scan_complete": True, "project_processes": [], "jobs": {},
                 "gpu_utilization_percent": [0], "now_epoch": 5000, "boot_id": "boot-1", "runs": {},
                 "raw_inventory": inventory, "raw_partial_file_count": 0},
                {"idle_since_epoch": 4000, "boot_id": "boot-1", "last_snapshot_verified": {
                    "path": str(archive), "local_stat": c.local_fingerprint(archive)}})

    def test_zombie_is_not_running_and_starttime_is_parsed(self):
        p = r.proc_identity(42, proc_root=self.write_proc("Z"), clock_ticks=100)
        self.assertFalse(p["alive"])
        self.assertEqual(p["start_ticks"], 1234)
        self.assertEqual(p["start_epoch"], 1012.34)

    def test_pid_reuse_or_command_change_does_not_match(self):
        proc = self.write_proc()
        before = r.proc_identity(42, proc_root=proc)
        self.assertTrue(r.identity_matches(before, before))
        self.write_proc(start_ticks=1235)
        self.assertFalse(r.identity_matches(r.proc_identity(42, proc_root=proc), before))
        self.write_proc(argv=["python", "other_job.py"])
        self.assertFalse(r.identity_matches(r.proc_identity(42, proc_root=proc), before))

    def test_mac_ps_is_pinned_to_start_and_command(self):
        result = subprocess.CompletedProcess([], 0, "42 S Sun Nov 22 08:41:36 2026 python /tmp/job.py\n")
        p = c.mac_process(42, runner=lambda *a, **k: result)
        self.assertTrue(p["alive"])
        self.assertEqual(p["argv"], ["python", "/tmp/job.py"])
        zombie = subprocess.CompletedProcess([], 0, "42 Z Sun Nov 22 08:41:36 2026 python /tmp/job.py\n")
        self.assertFalse(c.mac_process(42, runner=lambda *a, **k: zombie)["alive"])

    def test_raw_inventory_deduplicates_same_file_across_ledgers(self):
        inv = self.raw_fixture()
        ledger(self.root / "priority_eeg.sqlite", "complete")
        self.config["remote_jobs"].append({"job_id": "priority", "kind": "download", "db": "priority_eeg.sqlite"})
        inv2 = r.raw_inventory(self.config, root_override=self.root)
        self.assertTrue(inv2["ok"])
        self.assertEqual(inv2["file_count"], 1)
        self.assertEqual(inv["inventory_sha256"], inv2["inventory_sha256"])

    def test_completed_missing_or_changed_size_is_never_omitted_as_success(self):
        self.raw_fixture()
        (self.root / "raw/zenodo-1-v1/signal.bin").write_bytes(b"short")
        inv = r.raw_inventory(self.config, root_override=self.root)
        self.assertFalse(inv["ok"])
        self.assertEqual(len(inv["held"]), 1)

    def test_reconcile_delegated_priority_requires_body_sha_and_published_md5(self):
        body = b"valid full waveform"
        ledger(self.root / "downloads.sqlite", "delegated_priority", body)
        ledger(self.root / "priority_eeg.sqlite", "complete", body)
        p = self.root / "raw/zenodo-1-v1/signal.bin"
        p.parent.mkdir(parents=True)
        p.write_bytes(body)
        result = r.reconcile_priority(self.root)
        self.assertEqual(len(result["reconciled"]), 1)
        with sqlite3.connect(self.root / "downloads.sqlite") as db:
            self.assertEqual(db.execute("SELECT status FROM downloads").fetchone()[0], "complete")
        self.assertEqual(r.reconcile_priority(self.root)["reconciled"], [])

    def test_bad_priority_sha_is_held_without_reclassifying_main(self):
        ledger(self.root / "downloads.sqlite", "delegated_priority")
        ledger(self.root / "priority_eeg.sqlite", "complete", sha="0" * 64)
        p = self.root / "raw/zenodo-1-v1/signal.bin"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"EEG DATA")
        self.assertEqual(len(r.reconcile_priority(self.root)["held"]), 1)
        with sqlite3.connect(self.root / "downloads.sqlite") as db:
            self.assertEqual(db.execute("SELECT status FROM downloads").fetchone()[0], "delegated_priority")

    def restart_fixture(self):
        job = self.config["remote_jobs"][0]
        receipt = {"original_identity": {"pid": 42}, "deadline_epoch": 6000,
                   "original_argv": ["python", "/content/xavier/code/acquire.py", "download", "--seconds", "14400"]}
        status = {"now_epoch": 5000, "free_bytes": 1024**3,
                  "jobs": {"download": {"process": {"alive": False}, "ledger": {"pending_transfer_files": 1}}}}
        usage = {"verified": True, "observed_epoch": 5000, "balance_units": 100, "usage_units_per_hour": 2}
        return job, receipt, status, usage

    def test_restart_preserves_deadlines_reserve_and_credit_gates(self):
        job, receipt, status, usage = self.restart_fixture()
        self.assertIsNotNone(c.restart_request(job, receipt, status, usage, {}, self.config))
        self.assertIsNone(c.restart_request(job, receipt, status, usage, {"session_deadline_epoch": 4999}, self.config))
        self.assertIsNone(c.restart_request(job, receipt, dict(status, free_bytes=0), usage, {}, self.config))
        self.assertIsNone(c.restart_request(job, receipt, status, dict(usage, balance_units=0), {}, self.config))
        self.assertIsNone(c.restart_request(job, dict(receipt, completed_epoch=4900), status, usage, {}, self.config))
        self.assertIsNone(c.restart_request(dict(job, kind="train"), receipt, status, usage, {}, self.config))

    def test_byte_capped_original_job_requires_verified_traffic_receipt(self):
        job, receipt, status, usage = self.restart_fixture()
        receipt["original_argv"] += ["--max-bytes", "12000000000"]
        self.assertIsNone(c.restart_request(job, receipt, status, usage, {}, self.config))
        self.assertIn("cumulative_transfer", receipt["resume_hold_reason"])

    def test_original_deadline_never_moves_forward_after_new_observation(self):
        job = self.config["remote_jobs"][0]
        proc = {"pid": 42, "alive": True, "start_ticks": 1, "start_epoch": 1000, "cmd_sha256": "abc", "argv": ["python", "download"]}
        state = {}
        status = {"now_epoch": 1500, "jobs": {job["job_id"]: {"process": proc, "command_matches": True}}}
        c.update_job_receipts(self.config, state, status)
        status["jobs"][job["job_id"]]["process"] = dict(proc, start_epoch=9999, start_ticks=999, pid=43)
        c.update_job_receipts(self.config, state, status)
        self.assertEqual(state["remote_jobs"]["download"]["deadline_epoch"], 15400)
        self.assertFalse(state["remote_jobs"]["download"]["identity_matches"])

    def test_flat_recovery_already_done_by_parent_is_verified_without_redownload(self):
        inv = self.raw_fixture()
        destination = Path(self.config["raw_archive_root"])
        destination.mkdir()
        (destination / "signal.bin").write_bytes(b"EEG DATA")
        client = FakeDownloader({})
        status, state = self.idle_state(inv)
        receipt = c.recover_raw(self.config, state, inv, client)
        self.assertTrue(receipt["all_complete"])
        self.assertEqual(client.calls, [])
        self.assertTrue(c.can_stop(status, state, self.config)[0])

    def test_raw_download_is_verified_and_checksum_failure_blocks_stop(self):
        inv = self.raw_fixture()
        client = FakeDownloader({inv["files"][0]["remote_path"]: b"BAD DATA"})
        status, state = self.idle_state(inv)
        receipt = c.recover_raw(self.config, state, inv, client)
        self.assertFalse(receipt["all_complete"])
        self.assertFalse(c.can_stop(status, state, self.config)[0])
        self.assertFalse((Path(self.config["raw_archive_root"]) / inv["files"][0]["relative_path"]).exists())

    def test_valid_raw_download_is_atomically_promoted_and_cached(self):
        inv = self.raw_fixture()
        client = FakeDownloader({inv["files"][0]["remote_path"]: b"EEG DATA"})
        _, state = self.idle_state(inv)
        self.assertTrue(c.recover_raw(self.config, state, inv, client)["all_complete"])
        self.assertTrue(c.recover_raw(self.config, state, inv, client)["all_complete"])
        self.assertEqual(len(client.calls), 1)

    def test_new_remote_file_or_missing_local_file_invalidates_stop_receipt(self):
        inv = self.raw_fixture()
        status, state = self.idle_state(inv)
        c.recover_raw(self.config, state, inv, FakeDownloader({inv["files"][0]["remote_path"]: b"EEG DATA"}))
        changed = dict(status, raw_inventory=dict(inv, inventory_sha256="new"))
        self.assertFalse(c.can_stop(changed, state, self.config)[0])
        Path(next(iter(state["raw_preservation_receipt"]["files"].values()))["local_path"]).unlink()
        self.assertFalse(c.can_stop(status, state, self.config)[0])

    def test_cpu_download_or_partial_raw_never_counts_as_idle_stop(self):
        inv = self.raw_fixture()
        status, state = self.idle_state(inv)
        status["project_processes"] = [{"pid": 42, "argv": ["python", "acquire.py"], "alive": True}]
        self.assertIn("still_active", c.can_stop(status, state, self.config)[1])
        status["project_processes"] = []
        status["raw_partial_file_count"] = 1
        self.assertIn("partials", c.can_stop(status, state, self.config)[1])

    def test_cli_only_allows_owned_session_and_no_purchase_or_new_runtime(self):
        runner = lambda *a, **k: subprocess.CompletedProcess([], 0, "")
        cli = c.ColabCLI(self.config, runner=runner)
        for args in (["new"], ["pay"], ["update"], ["stop"], ["stop", "--session", "someone-else"]):
            with self.assertRaises(ValueError):
                cli.call(args)
        self.assertEqual(cli.stop_owned_session(), "")

    def make_zip(self, duplicate=False):
        target = self.root / "artifact.zip"
        body = b"checkpoint"
        manifest = {"files": [{"path": "runs/model/best.pt", "size": len(body), "sha256": hashlib.sha256(body).hexdigest()}], "included_runs": ["model"]}
        with zipfile.ZipFile(target, "w") as z:
            z.writestr("runs/model/best.pt", body)
            if duplicate:
                z.writestr("extra.txt", "unlisted payload")
            z.writestr("snapshot_manifest.json", json.dumps(manifest))
        return target, {"size": target.stat().st_size, "sha256": r.digest_file(target)}

    def test_snapshot_membership_and_size_sha_are_verified_before_extract(self):
        archive, receipt = self.make_zip()
        manifest = c.verify_and_extract(archive, receipt, self.root / "verified")
        self.assertEqual(manifest["included_runs"], ["model"])
        archive, receipt = self.make_zip(duplicate=True)
        with self.assertRaisesRegex(ValueError, "membership"):
            c.verify_and_extract(archive, receipt, self.root / "invalid")
        self.assertFalse((self.root / "invalid").exists())

    def test_remote_launch_rejects_deadline_and_wrong_kind_without_spawning(self):
        with patch.object(r.subprocess, "Popen", side_effect=AssertionError("must not spawn")):
            out = r.launch_download(self.config, {"job_id": "download", "deadline_epoch": 1}, root_override=self.root, now_epoch=2)
            self.assertFalse(out["started"])
            self.config["remote_jobs"][0]["kind"] = "train"
            out = r.launch_download(self.config, {"job_id": "download"}, root_override=self.root)
            self.assertIn("never_relaunched", out["reason"])

    def test_local_monitor_prefers_v3_and_retains_original_absolute_deadline(self):
        v2, v3 = self.root / "v2.json", self.root / "v3.json"
        v2.write_text(json.dumps({"pid": 2, "started_epoch": 100, "deadline_epoch": 18100, "command": ["old"]}))
        v3.write_text(json.dumps({"pid": 3, "started_epoch": 200, "deadline_epoch": 18100, "command": ["new"]}))
        with patch.object(c, "mac_process", return_value={"pid": 3, "alive": True, "start_epoch": 200, "argv": ["new"]}):
            result = c.local_job_status({"launch_receipts": [str(v3), str(v2)]})
        self.assertEqual(result["state"], "running")
        self.assertEqual(result["deadline_epoch"], 18100)
        self.assertFalse(result["automatic_restart"])

    def test_saved_stop_receipt_makes_later_cycles_idempotent_without_cli(self):
        state = self.root / "state.json"
        state.write_text(json.dumps({"session": c.SESSION, "session_stop_receipt": {"session": c.SESSION}}))
        with patch.object(c, "ColabCLI", side_effect=AssertionError("must not touch stopped runtime")):
            result = c.tick(self.config, state, apply=True)
        self.assertEqual(result["state"], "owned_session_already_stopped")

    def complete_cycle_client(self, *, changed_inventory=False, changed_boot=False):
        inventory = self.raw_fixture()
        archive, zip_receipt = self.make_zip()
        bodies = {inventory["files"][0]["remote_path"]: b"EEG DATA", "/content/xavier/export.zip": archive.read_bytes()}
        status = {"ok": True, "now_epoch": 5000, "boot_id": "boot-1", "free_bytes": 1024**3,
                  "jobs": {"download": {"process": {"alive": False}, "command_matches": False,
                                          "ledger": {"all_complete": True, "pending_transfer_files": 0}}},
                  "runs": {}, "project_processes": [], "process_scan_complete": True,
                  "gpu_utilization_percent": [0], "raw_partial_file_count": 0}

        class Client(FakeDownloader):
            def __init__(self):
                super().__init__(bodies)
                self.stops, self.probes, self.inventories = 0, 0, 0

            def usage(self):
                return {"verified": True, "observed_epoch": 5000, "balance_units": 100, "usage_units_per_hour": 2}

            def remote(self, action, config, request=None, **kwargs):
                if action == "probe":
                    self.probes += 1
                    return dict(status, boot_id="new-boot" if changed_boot and self.probes > 1 else "boot-1")
                if action == "reconcile":
                    return {"reconciled": [], "held": []}
                if action == "snapshot":
                    return dict(zip_receipt, ok=True, archive="/content/xavier/export.zip", included_runs=[],
                                run_fingerprints={}, prepared_included=True)
                if action == "raw_inventory":
                    self.inventories += 1
                    return dict(inventory, inventory_sha256="new-inventory") if changed_inventory and self.inventories > 1 else inventory
                raise AssertionError("unexpected action " + action)

            def stop_owned_session(self):
                self.stops += 1

        return Client()

    def test_complete_apply_cycle_recovers_raw_then_stops_only_owned_session_once(self):
        client = self.complete_cycle_client()
        state_file = self.root / "state.json"
        result = c.tick(self.config, state_file, apply=True, client=client)
        self.assertEqual(result["state"], "cycle_completed")
        self.assertEqual(client.stops, 1)
        state = json.loads(state_file.read_text())
        self.assertTrue(state["raw_preservation_receipt"]["all_complete"])
        self.assertEqual(state["raw_preservation_receipt"]["raw_payload_bytes_verified"], 8)
        c.tick(self.config, state_file, apply=True, client=client)
        self.assertEqual(client.stops, 1)

    def test_apply_cycle_does_not_stop_if_another_completed_raw_arrives(self):
        client = self.complete_cycle_client(changed_inventory=True)
        state_file = self.root / "state.json"
        result = c.tick(self.config, state_file, apply=True, client=client)
        self.assertEqual(result["state"], "cycle_completed")
        self.assertEqual(client.stops, 0)
        self.assertFalse(json.loads(state_file.read_text())["idle_stop_gate"]["allowed"])

    def test_reused_runtime_name_with_different_boot_blocks_cycle(self):
        client = self.complete_cycle_client(changed_boot=True)
        result = c.tick(self.config, self.root / "state.json", apply=True, client=client)
        self.assertEqual(result["state"], "blocked")
        self.assertEqual(client.calls, [])
        self.assertEqual(client.stops, 0)

    def variant_fixture(self, report_change=None):
        directory = self.root / "runs/uniba"
        directory.mkdir(parents=True)
        checkpoint = directory / "best.pt"
        checkpoint.write_bytes(b"trained checkpoint fixture")
        report = {"schema_version": "pj-xavier-variant-evaluation-v1", "status": "completed",
                  "checkpoint_sha256": r.digest_file(checkpoint), "variant_names": list(r.VARIANT_NAMES),
                  "conditions": {name: {"status": "measured"} for name in r.VARIANT_NAMES},
                  "autonomous_agent_validated": False}
        if report_change:
            report_change(report)
        path = directory / "evaluation.json"
        path.write_text(json.dumps(report))
        receipt = {"status": "completed", "exit_code": 0, "checkpoint_sha256": r.digest_file(checkpoint),
                   "variant_names": list(r.VARIANT_NAMES), "report_path": str(path), "report_sha256": r.digest_file(path)}
        (directory / "variant_evaluation_completion.json").write_text(json.dumps(receipt))
        return {"gate_id": "uniba_variants", "receipt_path": "runs/uniba/variant_evaluation_completion.json",
                "checkpoint_path": "runs/uniba/best.pt"}

    def test_five_variant_completion_accepts_honest_blocked_condition_not_goal_success(self):
        gate = self.variant_fixture(lambda report: report["conditions"].update(
            eeg_time_shifted={"status": "blocked", "reason": "insufficient earlier paired samples"}))
        result = r.verify_variant_completion(self.root, gate)
        self.assertTrue(result["passed"])
        self.assertFalse(result["model_goal_validated"])
        self.assertEqual(result["variant_statuses"]["eeg_time_shifted"], "blocked")

    def test_variant_receipt_does_not_excuse_missing_condition_or_changed_checkpoint(self):
        gate = self.variant_fixture(lambda report: report["conditions"].pop("eeg_only"))
        self.assertFalse(r.verify_variant_completion(self.root, gate)["passed"])
        (self.root / "runs/uniba/best.pt").write_bytes(b"different checkpoint")
        self.assertIn("sha256_mismatch", r.verify_variant_completion(self.root, gate)["reason"])

    def test_idle_stop_requires_five_variant_report_to_be_verified_and_exported(self):
        inv = self.raw_fixture()
        status, state = self.idle_state(inv)
        c.recover_raw(self.config, state, inv, FakeDownloader({inv["files"][0]["remote_path"]: b"EEG DATA"}))
        gate = self.variant_fixture()
        self.config["completion_gates"] = [gate]
        self.assertIn("required_evaluation", c.can_stop(status, state, self.config)[1])
        result = r.verify_variant_completion(self.root, gate)
        status["completion_gates"] = {gate["gate_id"]: result}
        self.assertIn("not_yet_exported", c.can_stop(status, state, self.config)[1])
        state["exported_completion_gates"] = {gate["gate_id"]: result["report_sha256"]}
        state["gate_archive_receipts"] = {gate["gate_id"]: state["last_snapshot_verified"]}
        self.assertTrue(c.can_stop(status, state, self.config)[0])


if __name__ == "__main__":
    unittest.main()
