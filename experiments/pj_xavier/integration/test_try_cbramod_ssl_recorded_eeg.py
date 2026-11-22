import copy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import try_cbramod_ssl_recorded_eeg as demo


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def seal_fixture():
    return {"schema": "xavier-selected-ssl-encoder-seal-v1", "initial_weights_sha256": demo.KNOWN["initial_weights"],
            "weights_sha256": "1" * 64, "selection_partition": "portable_SSL_validation", "downstream_test_source_used": False}


def contract_fixture():
    return {"schema": "xavier-cbramod-ssl-train-demo-contract-v1", "selected_partition": "train", "heldout_records_used_for_demo": 0,
            "abstain_threshold": .8, "os_execution_allowed": False, "manifest_sha256": demo.KNOWN["manifest"],
            "run_files": {key: "a" * 64 for key in demo.RUN_FILES}, "ssl_files": {key: "b" * 64 for key in demo.SSL_FILES},
            "ssl_corpus_files": demo.CORPUS_HASHES, "expected_recovery_member_count": 100,
            "gates": {key: {"path": "private/xavier/" + key + ".json", "sha256": "c" * 64} for key in demo.GATE_FILES},
            "ssl_protocol": {"path": "private/xavier/ssl_protocol.json", "sha256": "d" * 64},
            "alignment_protocol": {"path": "private/xavier/alignment_protocol.json", "sha256": "e" * 64}}


def gate_fixture(root):
    root = root.resolve()
    records = {}
    for key, path in [(name, root / demo.RUN / name) for name in demo.RUN_FILES] + [("ssl:" + name, root / demo.SSL_RUN / name) for name in demo.SSL_FILES]:
        write(path, {"fixture": key})
        records[key] = demo.file_record(path)
    protocol = root / "private/xavier/workflows/cbramod_ssl_v1/alignment_protocol.json"
    write(protocol, {"fixture": "alignment"})
    records["alignment_protocol"] = demo.file_record(protocol)
    best_sha = records["best.pt"]["sha256"]
    path = root / "private/xavier/evaluation/report.json"
    report = {"schema_version": "pj-xavier-variant-evaluation-v1", "status": "completed", "checkpoint_sha256": best_sha,
              "variant_names": sorted(demo.VARIANTS), "conditions": {key: {"status": "measured"} for key in demo.VARIANTS}}
    write(path, report)
    records["variant_report"] = demo.file_record(path)
    command = ["python", "/content/xavier/benchmark/evaluate_variants.py", "--checkpoint", "/content/xavier/" + demo.RUN.removeprefix("private/xavier/") + "/best.pt",
               "--model-module", "/content/xavier/code/model/cbramod_ssl_token_llm.py", "--expected-model-sha256", demo.KNOWN["model_source"],
               "--expected-manifest-sha256", demo.KNOWN["manifest"]]
    gates = {"workflow_completion": {"status": "SSL_alignment_and_five_conditions_completed_pending_strict_recovery_external",
                                    "alignment_checkpoint_sha256": best_sha, "ssl_encoder_seal_sha256": records["ssl:selected_encoder_seal.json"]["sha256"],
                                    "alignment_protocol_sha256": records["alignment_protocol"]["sha256"], "evaluation_report_sha256": records["variant_report"]["sha256"],
                                    "os_actions_executed": 0},
             "evaluation_exit": {"stage": "five_conditions", "exit_code": 0, "timed_out": False, "command": command},
             "variant_receipt": {"status": "completed", "exit_code": 0, "checkpoint_sha256": best_sha, "variant_names": sorted(demo.VARIANTS),
                                 "report_path": "/content/xavier/evaluation/report.json", "report_sha256": records["variant_report"]["sha256"]},
             "strict_report": {"schema_version": "xavier-benchmark-v1", "status": "evaluated_goal_unproven", "checkpoint_sha256": best_sha,
                               "integrity_errors": [], "data_validation": {"errors": []}, "resource_feasibility": {"status": "passed"}, "autonomous_agent_validated": False}}
    for key, value in gates.items():
        path = root / "private/xavier/evaluation" / (key + ".json")
        write(path, value); records[key] = demo.file_record(path)
    members = {Path(record["path"]).relative_to(root / "private/xavier").as_posix(): Path(record["path"]).relative_to(root / "private/xavier").as_posix() for key, record in records.items() if key != "strict_report"}
    members.update({"other/" + str(i): "other/" + str(i) for i in range(100-len(members))})
    path = root / "private/xavier/evaluation/recovery_receipt.json"
    write(path, {"status": "completed", "all_member_hashes_verified": True, "checkpoint_sha256": best_sha,
                 "ssl_encoder_seal_sha256": records["ssl:selected_encoder_seal.json"]["sha256"], "member_count": 100, "member_destinations": members})
    records["recovery_receipt"] = demo.file_record(path)
    return {"repo": str(root), "sources": records, "expected_recovery_member_count": 100}


def ssl_tensor_fixture():
    import torch
    sources = {}
    for index in range(75):
        split = "val" if 46 <= index < 58 else "train"
        sources[f"{index:064x}"] = {"split": split, "subject_group": split + str(index % (4 if split == "val" else 18)), "source_recording_id": "r" + str(index)}
    corpus = []
    for split, count, indices in (("train", 1678, range(46)), ("val", 522, range(46, 58))):
        ids = list(indices)
        for number in range(count):
            digest = f"{ids[number % len(ids)]:064x}"
            corpus.append({"id": split + str(number), "source_sha256": digest, **sources[digest]})
    membership = {"manifest_sha256": demo.CORPUS_HASHES["manifest.jsonl"], "test_source_used": False, "behavior_labels_used": 0,
                  "train_ids": sorted(r["id"] for r in corpus if r["split"] == "train"), "val_ids": sorted(r["id"] for r in corpus if r["split"] == "val"),
                  "source_sha256_by_partition": {s: sorted({r["source_sha256"] for r in corpus if r["split"] == s}) for s in ("train", "val")},
                  "subject_groups_by_partition": {s: sorted({r["subject_group"] for r in corpus if r["split"] == s}) for s in ("train", "val")}}
    records = {key: {"sha256": character * 64, "bytes": 20} for key, character in (("ssl:best_encoder_weights.pth", "1"), ("ssl:best.pt", "2"),
                  ("ssl_protocol", "3"), ("ssl:training_membership.json", "4"), ("ssl:run_summary.json", "5"))}
    config = {"command": "train", "model_source_sha256": demo.KNOWN["ssl_trainer_source"], "protocol_sha256": "3" * 64,
              "manifest_sha256": demo.CORPUS_HASHES["manifest.jsonl"], "artifacts_sha256": demo.CORPUS_HASHES["artifacts.json"], "source_allowlist_sha256": demo.CORPUS_HASHES["source_allowlist.json"]}
    weights, initial = {"w": torch.ones(2)}, {"w": torch.zeros(2)}
    state = {"schema": "xavier-cbramod-masked-ssl-v1", "config": config, "initial_weights_sha256": demo.KNOWN["initial_weights"], "behavior_labels_used": 0,
             "model": {"w": torch.ones(2)}, "seen_training_ids": ["train0"], "training_examples_presented": 1, "global_step": 2, "epoch": 0, "best_val_masked_mse": 1.}
    summary = {"status": "completed_epochs", "global_step": 2, "best_checkpoint_sha256": "2" * 64, "best_encoder_weights_sha256": "1" * 64,
               "downstream_test_source_used": False, "itu_sources_used": 0, "behavior_labels_used": 0, "os_actions_executed": 0, "best_validation_masked_mse": 1.}
    seal = {**seal_fixture(), "run_name": Path(demo.SSL_RUN).name, "weights_bytes": 20, "ssl_best_checkpoint_sha256": "2" * 64,
            "ssl_protocol_sha256": "3" * 64, "ssl_source_sha256": demo.KNOWN["ssl_trainer_source"], "ssl_manifest_sha256": demo.CORPUS_HASHES["manifest.jsonl"],
            "ssl_source_allowlist_sha256": demo.CORPUS_HASHES["source_allowlist.json"], "ssl_training_membership_sha256": "4" * 64, "ssl_run_summary_sha256": "5" * 64,
            "itu_sources_used": 0, "behavior_labels_used": 0, "updated_tensor_count": 1, "unique_training_windows_before_selection": 1,
            "unique_training_seconds_before_selection": 30, "training_examples_presented_before_selection": 1, "selected_global_step": 2, "selected_epoch": 1}
    history = [{"global_step": 1, "val": {"masked_mse": 2.}}, {"global_step": 2, "val": {"masked_mse": 1.}}]
    allowlist = {"sources": sources, "excluded_source_sha256": [f"{index:064x}" for index in range(100, 142)]}
    return [state, weights, initial, config, summary, membership, history, seal, corpus, allowlist, records]


class SSLDemoTests(unittest.TestCase):
    def test_actual_workflow_model_directory_matches_the_child_receipt_gate(self):
        # Inspect the independently implemented real workflow, not the gate's
        # fixture constant. No workflow code or external command is executed.
        source = Path(demo.__file__).resolve().parents[1] / "model/run_ssl_workflow.py"
        tree = ast.parse(source.read_text())
        assignments = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                               and target.value.id == "self" and target.attr == "model" for target in node.targets)]
        self.assertEqual(len(assignments), 1)
        value = assignments[0]
        self.assertIsInstance(value, ast.BinOp)
        self.assertIsInstance(value.op, ast.Div)
        self.assertEqual(ast.unparse(value.left), "self.root")
        self.assertIsInstance(value.right, ast.Constant)
        self.assertEqual(demo.REMOTE_MODEL_MODULE, "/content/xavier/" + value.right.value + "/cbramod_ssl_token_llm.py")

    def test_contract_keeps_fixed_scope_and_full_SSL_seals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"; contract = contract_fixture(); write(path, contract)
            self.assertEqual(demo.load_contract(path, demo.sha256(path)), contract)
            with self.assertRaisesRegex(ValueError, "SHA"):
                demo.load_contract(path, "0" * 64)
            for change in ({"selected_partition": "test"}, {"abstain_threshold": .6}, {"os_execution_allowed": True}, {"ssl_corpus_files": {}}, {"ssl_files": {}}):
                with self.subTest(change=change):
                    write(path, {**contract, **change})
                    with self.assertRaises(ValueError):demo.load_contract(path, demo.sha256(path))

    def test_unchanged_official_weights_or_wrong_selection_cannot_be_SSL(self):
        demo.encoder_identity(seal_fixture())
        for changes in ({"weights_sha256": demo.KNOWN["initial_weights"]}, {"selection_partition": "test"}, {"downstream_test_source_used": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):demo.encoder_identity({**seal_fixture(), **changes})

    def test_recovered_five_condition_evidence_unblocks_CPU_preparation(self):
        with tempfile.TemporaryDirectory() as directory:demo.evaluation_is_complete(gate_fixture(Path(directory)))

    def test_missing_real_child_exit_or_wrong_model_remains_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            for changes in ({"exit_code": 1}, {"timed_out": True}, {"command": ["python", "some_other_model"]}, {"stage": "supervisor_exit"}):
                with self.subTest(changes=changes):
                    plan = gate_fixture(Path(directory)); path = Path(plan["sources"]["evaluation_exit"]["path"])
                    write(path, {**demo.read_json(path), **changes})
                    with self.assertRaises(ValueError):demo.evaluation_is_complete(plan)

    def test_partial_recovery_changed_SSL_seal_or_strict_error_blocks_MPS(self):
        with tempfile.TemporaryDirectory() as directory:
            for key, changes in (("recovery_receipt", {"all_member_hashes_verified": False}), ("recovery_receipt", {"ssl_encoder_seal_sha256": "f" * 64}),
                                 ("recovery_receipt", {"member_count": 99}), ("strict_report", {"integrity_errors": ["changed"]})):
                with self.subTest(key=key, changes=changes):
                    plan = gate_fixture(Path(directory)); path = Path(plan["sources"][key]["path"])
                    write(path, {**demo.read_json(path), **changes})
                    with self.assertRaises(ValueError):demo.evaluation_is_complete(plan)

    def test_missing_recovered_encoder_member_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = gate_fixture(Path(directory)); path = Path(plan["sources"]["recovery_receipt"]["path"]); value = demo.read_json(path)
            key = demo.SSL_RUN.removeprefix("private/xavier/") + "/best_encoder_weights.pth"
            del value["member_destinations"][key]; value["member_destinations"]["irrelevant"] = "irrelevant"
            write(path, value)
            with self.assertRaisesRegex(ValueError, "recovered archive"):demo.evaluation_is_complete(plan)

    def test_recovery_absolute_destinations_must_match_the_actual_private_files(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = gate_fixture(Path(directory)); path = Path(plan["sources"]["recovery_receipt"]["path"]); value = demo.read_json(path)
            value["member_destinations"] = {key: str(Path(plan["repo"]) / "private/xavier" / target) for key, target in value["member_destinations"].items()}
            write(path, value); demo.evaluation_is_complete(plan)
            key = demo.SSL_RUN.removeprefix("private/xavier/") + "/best_encoder_weights.pth"
            value["member_destinations"][key] = "/tmp/another_encoder.pth"; write(path, value)
            with self.assertRaises(ValueError):demo.evaluation_is_complete(plan)

    def test_demo_command_uses_adapted_weights_and_unchanged_dry_run(self):
        source = {key: {"path": "/repo/" + key} for key in ("model_source", "best.pt", "ssl:best_encoder_weights.pth", "replay_source")}
        predict, replay = demo.commands_for({"sources": source, "upstream": "/repo/upstream"}, Path("/demo"))
        self.assertEqual(predict[predict.index("--cbramod-weights")+1], "/repo/ssl:best_encoder_weights.pth")
        self.assertEqual(predict[predict.index("--device")+1], "mps")
        self.assertEqual(predict[predict.index("--memory-budget-gib")+1], "8")
        self.assertNotIn("--abstain-threshold", predict)
        self.assertEqual(replay[2:], ["--predictions", "/demo/predictions.jsonl", "--output", "/demo/dry_run_audit.jsonl"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch unavailable in scratch")
    def test_adapted_tensor_selection_corpus_membership_and_true_update(self):
        demo.verify_ssl_selection(*ssl_tensor_fixture())
        for kind in ("tensor", "no_update", "selection", "heldout_seen", "excluded_source"):
            with self.subTest(kind=kind):
                args = ssl_tensor_fixture()
                if kind == "tensor":args[1]["w"][0] = 3
                elif kind == "no_update":
                    args[1]["w"].zero_(); args[0]["model"]["w"].zero_(); args[7]["updated_tensor_count"] = 0
                elif kind == "selection":args[6][0]["val"]["masked_mse"] = .5
                elif kind == "heldout_seen":args[0]["seen_training_ids"] = ["val0"]
                else:args[9]["excluded_source_sha256"][0] = args[8][0]["source_sha256"]
                with self.assertRaises(ValueError):demo.verify_ssl_selection(*args)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch unavailable in scratch")
    def test_alignment_checkpoint_exact_SSL_identity_and_frozen_base(self):
        import torch
        seal = seal_fixture(); identity = demo.encoder_identity(seal)
        config = {"mode": "fused", "lora_rank": 0, "abstain_threshold": .8, "dtype": "bfloat16", "channels": 8, "samples": 448,
                  "prefix_tokens": 8, "encoder_width": 200, "eeg_unit": "V", "eeg_encoder_identity": identity, "class_token_ids": [330,389,340],
                  "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "revision": "31b70e2e869a7173562077fd711b654946d38674",
                  "model_module_sha256": demo.KNOWN["model_source"], "core_source_sha256": demo.KNOWN["core_source"],
                  "ssl_encoder_seal": seal, "ssl_encoder_seal_sha256": "f"*64, "prior_test_results_seen_for_experiment_design": True,
                  "exploratory_reuse_of_v3_test": True}
        state = {"schema": "xavier-cbramod-ssl-symbol-llm-v1", "config": config, "manifest_sha256": demo.KNOWN["manifest"], "classes": demo.CLASSES,
                 "eeg_artifacts": {}, "trainable_base": {}, "eeg_encoder_identity": identity, "token_mapping": demo.TOKEN_MAPPING,
                 "connector": {"0.weight": torch.ones(200), "0.bias": torch.zeros(200), "1.weight": torch.zeros((2048,200)), "1.bias": torch.zeros(2048)}}
        summary = {"eeg_encoder_frozen": True, "base_frozen": True, "eeg_encoder_cpu_float32": True, "trainable_parameters": 412048}
        demo.verify_checkpoint(state, config, {}, summary, seal, "f"*64)
        for changes in ({"trainable_base": {"x": torch.ones(1)}}, {"schema": "xavier-cbramod-symbol-llm-v1"}, {"eeg_encoder_identity": {}}):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):demo.verify_checkpoint({**state, **changes}, config, {}, summary, seal, "f"*64)

    def test_existing_attempt_never_relaunches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); selected = [{"id": label, "label": label, "split": "train"} for label in demo.CLASSES]
            verified = {"repo": str(root), "run_dir": "run", "manifest": "manifest", "sources": {"contract": {"path": "contract"}}, "arrays": {}, "backend": {},
                        "selected": selected, "upstream": "upstream", "contract_sha256": "a"*64}
            commands = (["python", "predict"], ["python", "replay"]); data = demo.manifest_bytes(selected)
            (root/"train_examples.jsonl").write_bytes(data)
            plan = {**verified, "wrapper_sha256": demo.sha256(demo.__file__), "python_executable": str(Path(demo.sys.executable).absolute()),
                    "predict_command": commands[0], "replay_command": commands[1], "selected_ids": [row["id"] for row in selected],
                    "train_examples_sha256": hashlib.sha256(data).hexdigest()}
            write(root/"plan.json", plan); write(root/"plan.sha256.json", {"sha256": demo.sha256(root/"plan.json")}); write(root/"execution_started.json", {"attempt": 1})
            runner = mock.Mock(side_effect=AssertionError("must not start"))
            with mock.patch.object(demo, "assert_originals_unchanged"), mock.patch.object(demo, "verify_sources", return_value=verified), mock.patch.object(demo, "commands_for", return_value=commands):
                with self.assertRaisesRegex(ValueError, "already started"):demo.execute_prepared(root, runner=runner)
            runner.assert_not_called()


if __name__ == "__main__":unittest.main()
