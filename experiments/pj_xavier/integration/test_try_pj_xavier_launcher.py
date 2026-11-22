"""No-inference checks for the pinned one-command recorded-EEG launcher."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SOURCE = Path(os.environ.get("XAVIER_LAUNCHER_TEST_SOURCE", Path(__file__).resolve().parents[3] / "scripts/try_pj_xavier_recorded.py"))
spec = importlib.util.spec_from_file_location("phase3_trial_launcher", SOURCE)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def fixture(repo):
    contract = {
        "schema": "xavier-phase3-SSL-train-three-M1-contract-v1", "partition": "train",
        "selected_ids": ["a", "b", "c"],
        "run": "private/xavier/eegmmidb_phase3_v1/runs/eegnet_ssl_smollm_fused_v1",
        "device": "mps", "memory_budget_gib": 8, "acceptance_threshold": .8,
        "route_calls": 12, "max_new_tokens": 8, "test_forward_calls": 0,
        "routing_cache_reuse": False, "OS_actions_allowed": False,
        "cloud_calls_allowed": False, "checkpoint_sha256": "checkpoint",
    }
    for name, content in ((launcher.CONTRACT, json.dumps(contract)), (launcher.WRAPPER, "fixture wrapper"), (".venv-xavier/bin/python", "fixture interpreter")):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return contract, launcher.sha(repo / launcher.CONTRACT), launcher.sha(repo / launcher.WRAPPER)


def result(contract_sha):
    return {
        "status": "completed", "contract_sha256": contract_sha, "checkpoint_sha256": "checkpoint",
        "train_records": 3, "actual_fresh_LLM_routing_calls": 12, "test_forward_calls": 0,
        "OS_actions": 0, "cloud_API_calls": 0, "local_files_only": True,
        "abstained_episodes": 12, "successful_synthetic_episodes": 0,
        "invalid_actions": 0, "incorrect_target_actions": 0,
    }


class LauncherTests(unittest.TestCase):
    def test_unsealed_contract_never_spawns_a_model(self):
        with mock.patch.object(launcher, "EXPECTED_CONTRACT_SHA256", "UNSEALED"), mock.patch.object(launcher.subprocess, "Popen") as child:
            with self.assertRaises(ValueError):
                launcher.run(Path("/nonexistent"))
            child.assert_not_called()

    def test_changed_contract_or_wrapper_blocks_before_child(self):
        for changed in (launcher.CONTRACT, launcher.WRAPPER):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary:
                repo = Path(temporary)
                _, csha, wsha = fixture(repo)
                (repo / changed).write_text("changed after sealing")
                with mock.patch.object(launcher, "EXPECTED_CONTRACT_SHA256", csha), mock.patch.object(launcher, "EXPECTED_WRAPPER_SHA256", wsha), mock.patch.object(launcher.subprocess, "Popen") as child:
                    with self.assertRaises(ValueError):
                        launcher.run(repo)
                    child.assert_not_called()

    def test_test_scope_and_resource_changes_are_rejected(self):
        for key, wrong in (("partition", "test"), ("memory_budget_gib", 21), ("route_calls", 16), ("acceptance_threshold", .5), ("cloud_calls_allowed", True)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                repo = Path(temporary)
                contract, _, wsha = fixture(repo)
                contract[key] = wrong
                (repo / launcher.CONTRACT).write_text(json.dumps(contract))
                csha = launcher.sha(repo / launcher.CONTRACT)
                with mock.patch.object(launcher, "EXPECTED_CONTRACT_SHA256", csha), mock.patch.object(launcher, "EXPECTED_WRAPPER_SHA256", wsha):
                    with self.assertRaises(ValueError):
                        launcher.verify(repo)

    def test_abstention_is_reported_as_completed_execution_without_goal_claim(self):
        csha = "a" * 64
        with mock.patch.object(launcher, "EXPECTED_CONTRACT_SHA256", csha):
            actual = launcher.summarize(result(csha), {"checkpoint_sha256": "checkpoint"})
            self.assertEqual(actual["status"], "completed")
            self.assertEqual(actual["abstained_episodes"], 12)
            self.assertEqual(actual["successful_synthetic_episodes"], 0)
            self.assertFalse(actual["heldout_performance"])
            self.assertFalse(actual["autonomous_goal_validated"])
            bad = result(csha)
            bad["OS_actions"] = 1
            with self.assertRaises(ValueError):
                launcher.summarize(bad, {"checkpoint_sha256": "checkpoint"})

    def test_fixed_command_offline_and_exclusive_new_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            _, csha, wsha = fixture(repo)
            commands = []

            def child(command, **kwargs):
                commands.append((command, kwargs))
                output = Path(command[command.index("--output") + 1])
                output.mkdir()
                (output / "summary.json").write_text(json.dumps(result(csha)))
                return SimpleNamespace(wait=lambda timeout: 0)

            date = SimpleNamespace(strftime=lambda format: "fixed-time")
            with mock.patch.object(launcher, "EXPECTED_CONTRACT_SHA256", csha), mock.patch.object(launcher, "EXPECTED_WRAPPER_SHA256", wsha), mock.patch.object(launcher.subprocess, "Popen", side_effect=child), mock.patch.object(launcher.uuid, "uuid4", return_value=SimpleNamespace(hex="fixed-random")), mock.patch.object(launcher, "datetime", SimpleNamespace(now=lambda zone: date)), mock.patch("builtins.print"):
                launcher.run(repo)
                with self.assertRaises(FileExistsError):
                    launcher.run(repo)
            self.assertEqual(len(commands), 1)
            command, keywords = commands[0]
            self.assertEqual(command[-2:], ["--max-seconds", "300"])
            self.assertEqual(keywords["env"]["HF_HUB_OFFLINE"], "1")
            self.assertEqual(keywords["env"]["TRANSFORMERS_OFFLINE"], "1")
            self.assertNotIn("--device", command)
            self.assertNotIn("--checkpoint", command)

    def test_timeout_terminates_only_the_owned_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            _, csha, wsha = fixture(repo)
            child = mock.Mock(pid=4321, returncode=-15)
            child.wait.side_effect = [subprocess.TimeoutExpired("fixture", 330), 0]
            with mock.patch.object(launcher, "EXPECTED_CONTRACT_SHA256", csha), mock.patch.object(launcher, "EXPECTED_WRAPPER_SHA256", wsha), mock.patch.object(launcher.subprocess, "Popen", return_value=child), mock.patch.object(launcher.os, "killpg") as kill:
                with self.assertRaises(subprocess.TimeoutExpired):
                    launcher.run(repo)
            kill.assert_called_once_with(4321, launcher.signal.SIGTERM)
            receipts = list((repo / launcher.OUTPUT_ROOT).glob("*/launcher_completion.json"))
            self.assertEqual(json.loads(receipts[0].read_text())["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
