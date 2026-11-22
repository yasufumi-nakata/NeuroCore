import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import try_cbramod_recorded_eeg as demo


def fixture_contract():
    return {"schema": "xavier-cbramod-train-demo-contract-v1", "selected_partition": "train", "heldout_records_used_for_demo": 0,
            "abstain_threshold": .8, "os_execution_allowed": False, "manifest_sha256": demo.KNOWN["manifest"],
            "run_files": {name: demo.KNOWN.get(name, "a" * 64) for name in demo.RUN_FILES},
            "gates": {name: {"path": "private/xavier/" + name + ".json", "sha256": "a" * 64} for name in demo.GATE_FILES}}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def gate_fixture(root):
    root = root.resolve()  # macOS /var -> /private/var must match file_record.
    base = root / "private/xavier/evaluation"
    report = {"schema_version": "pj-xavier-variant-evaluation-v1", "status": "completed", "checkpoint_sha256": demo.KNOWN["best.pt"], "variant_names": sorted(demo.VARIANTS),
              "conditions": {key: {"status": "measured"} for key in demo.VARIANTS}}
    report_path = base / "report.json"
    write_json(report_path, report)
    receipt = {"status": "completed", "exit_code": 0, "checkpoint_sha256": demo.KNOWN["best.pt"],
               "variant_names": sorted(demo.VARIANTS), "report_path": "/content/xavier/evaluation/report.json",
               "report_sha256": demo.sha256(report_path)}
    contents = {"workflow_completion": {"status": "training_and_variants_completed_pending_strict_and_recovery", "checkpoint_sha256": demo.KNOWN["best.pt"], "training_repeated": False, "os_actions_executed": 0},
                "evaluation_exit": {"exit_code": 0, "checkpoint_sha256": demo.KNOWN["best.pt"]}, "variant_receipt": receipt, "variant_report": report,
                "strict_report": {"schema_version": "xavier-benchmark-v1", "status": "evaluated_goal_unproven", "checkpoint_sha256": demo.KNOWN["best.pt"],
                                  "integrity_errors": [], "data_validation": {"errors": []}, "resource_feasibility": {"status": "passed"}, "autonomous_agent_validated": False}}
    records = {"best.pt": {"sha256": demo.KNOWN["best.pt"], "path": str(base / "best.pt")}}
    for key, value in contents.items():
        path = report_path if key == "variant_report" else base / (key + ".json")
        write_json(path, value)
        records[key] = demo.file_record(path)
    members = {Path(record["path"]).relative_to(root / "private/xavier").as_posix(): Path(record["path"]).relative_to(root / "private/xavier").as_posix() for record in records.values()}
    members.update({"other/" + str(i): "other/" + str(i) for i in range(371 - len(members))})
    recovery_path = base / "recovery_receipt.json"
    write_json(recovery_path, {"status": "completed", "checkpoint_sha256": demo.KNOWN["best.pt"], "all_member_hashes_verified": True,
                               "member_count": 371, "member_destinations": members})
    records["recovery_receipt"] = demo.file_record(recovery_path)
    return {"repo": str(root), "sources": records}


class CBraModDemoTests(unittest.TestCase):
    def test_contract_requires_exact_hash_and_train_only_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            contract = fixture_contract()
            write_json(path, contract)
            self.assertEqual(demo.load_contract(path, demo.sha256(path)), contract)
            with self.assertRaisesRegex(ValueError, "SHA"):
                demo.load_contract(path, "b" * 64)
            for change in ({"os_execution_allowed": True}, {"selected_partition": "test"}, {"abstain_threshold": .7}, {"heldout_records_used_for_demo": 3}):
                with self.subTest(change=change):
                    write_json(path, {**contract, **change})
                    with self.assertRaises(ValueError):
                        demo.load_contract(path, demo.sha256(path))

    def test_contract_paths_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for bad in ("../outside", "/tmp/other", "private\\other"):
                with self.subTest(path=bad), self.assertRaises(ValueError):
                    demo.safe_relative(root, bad)
            self.assertEqual(demo.receipt_local_path(root, "/content/xavier/evaluation/report.json"), root / "private/xavier/evaluation/report.json")
            with self.assertRaises(ValueError):
                demo.receipt_local_path(root, "/content/other/report.json")

    def test_pending_receipt_can_prove_completed_evaluation_after_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = gate_fixture(Path(directory))
            demo.evaluation_is_complete(plan)
            self.assertFalse((Path(directory) / "variant_evaluation_completion.json").exists())

    def test_incomplete_or_wrong_evaluation_blocks_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key, value in (("workflow_completion", {"status": "running"}), ("evaluation_exit", {"exit_code": 1}),
                               ("variant_receipt", {"status": "completed", "exit_code": 0, "checkpoint_sha256": "b" * 64, "variant_names": sorted(demo.VARIANTS)})):
                with self.subTest(key=key):
                    plan = gate_fixture(root)
                    write_json(Path(plan["sources"][key]["path"]), value)
                    with self.assertRaises(ValueError):
                        demo.evaluation_is_complete(plan)

    def test_blocked_condition_needs_reason_and_report_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = gate_fixture(Path(directory))
            report_path = Path(plan["sources"]["variant_report"]["path"])
            report = demo.read_json(report_path)
            report["conditions"]["eeg_shuffled"] = {"status": "blocked"}
            write_json(report_path, report)
            receipt_path = Path(plan["sources"]["variant_receipt"]["path"])
            receipt = demo.read_json(receipt_path)
            with self.assertRaisesRegex(ValueError, "hash"):
                demo.evaluation_is_complete(plan)
            receipt["report_sha256"] = demo.sha256(report_path)
            write_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "unfinished"):
                demo.evaluation_is_complete(plan)

    def test_historical_pending_workflow_requires_later_recovery_and_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            for key, changes in (("recovery_receipt", {"all_member_hashes_verified": False}),
                                 ("strict_report", {"integrity_errors": ["hash mismatch"]}),
                                 ("strict_report", {"autonomous_agent_validated": True})):
                with self.subTest(key=key, changes=changes):
                    plan = gate_fixture(Path(directory))
                    path = Path(plan["sources"][key]["path"])
                    write_json(path, {**demo.read_json(path), **changes})
                    with self.assertRaises(ValueError):
                        demo.evaluation_is_complete(plan)
    def test_commands_pin_mps_memory_cpu_encoder_sources_and_dry_run(self):
        source = {key: {"path": "/repo/" + key} for key in ("model_source", "best.pt", "upstream_weights", "replay_source")}
        predict, replay = demo.commands_for({"sources": source, "upstream": "/repo/upstream"}, Path("/demo"))
        self.assertEqual(predict[predict.index("--device") + 1], "mps")
        self.assertEqual(predict[predict.index("--memory-budget-gib") + 1], "8")
        self.assertIn("--cbramod-upstream", predict)
        self.assertNotIn("--candidate-batch-size", predict)
        self.assertNotIn("--abstain-threshold", predict)
        self.assertEqual(replay[2:], ["--predictions", "/demo/predictions.jsonl", "--output", "/demo/dry_run_audit.jsonl"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch not installed in scratch")
    def test_checkpoint_rejects_trainable_backbone_tokens_and_bad_connector(self):
        import torch
        config = {"mode": "fused", "lora_rank": 0, "abstain_threshold": .8, "dtype": "bfloat16", "channels": 8, "samples": 448,
                  "prefix_tokens": 8, "encoder_width": 200, "eeg_unit": "V", "eeg_encoder_identity": demo.ENCODER_IDENTITY,
                  "class_token_ids": [330, 389, 340], "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "revision": "31b70e2e869a7173562077fd711b654946d38674",
                  "model_module_sha256": demo.KNOWN["model_source"], "core_source_sha256": demo.KNOWN["core_source"]}
        state = {"schema": "xavier-cbramod-symbol-llm-v1", "manifest_sha256": demo.KNOWN["manifest"], "config": config, "classes": demo.CLASSES,
                 "eeg_artifacts": {}, "trainable_base": {}, "eeg_encoder_identity": demo.ENCODER_IDENTITY, "token_mapping": demo.TOKEN_MAPPING,
                 "connector": {"0.weight": torch.ones(200), "0.bias": torch.zeros(200), "1.weight": torch.zeros((2048, 200)), "1.bias": torch.zeros(2048)}}
        summary = {"eeg_encoder_frozen": True, "base_frozen": True, "eeg_encoder_cpu_float32": True, "trainable_parameters": 412048}
        demo.verify_checkpoint(state, config, {}, summary)
        for bad in ({"trainable_base": {"edited": torch.ones(1)}}, {"token_mapping": {"ai_prompt_submit": 331}}, {"schema": "legacy"}):
            with self.subTest(key=list(bad)), self.assertRaises(ValueError):
                demo.verify_checkpoint({**state, **bad}, config, {}, summary)
        malformed = copy.deepcopy(state)
        malformed["connector"]["1.weight"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            demo.verify_checkpoint(malformed, config, {}, summary)

    def test_existing_attempt_is_never_relaunched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = [{"id": label, "label": label, "split": "train"} for label in demo.CLASSES]
            verified = {"repo": str(root), "run_dir": "run", "manifest": "manifest", "sources": {"contract": {"path": "contract"}},
                        "arrays": {}, "backend": {}, "selected": selected, "upstream": "upstream", "contract_sha256": "a" * 64}
            commands = (["python", "predict"], ["python", "replay"])
            data = demo.manifest_bytes(selected)
            (root / "train_examples.jsonl").write_bytes(data)
            plan = {**verified, "wrapper_sha256": demo.sha256(demo.__file__), "python_executable": str(Path(demo.sys.executable).absolute()),
                    "predict_command": commands[0], "replay_command": commands[1], "selected_ids": [row["id"] for row in selected],
                    "train_examples_sha256": hashlib.sha256(data).hexdigest()}
            write_json(root / "plan.json", plan)
            write_json(root / "plan.sha256.json", {"sha256": demo.sha256(root / "plan.json")})
            write_json(root / "execution_started.json", {"attempt": 1})
            runner = mock.Mock(side_effect=AssertionError("must not start"))
            with mock.patch.object(demo, "assert_originals_unchanged"), mock.patch.object(demo, "verify_sources", return_value=verified), mock.patch.object(demo, "commands_for", return_value=commands):
                with self.assertRaisesRegex(ValueError, "already started"):
                    demo.execute_prepared(root, runner=runner)
            runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
