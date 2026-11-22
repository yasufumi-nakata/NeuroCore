#!/usr/bin/env python3
"""Three training-only CBraMod/SmolLM EEG examples and a blocked dry-run route.

The older demo and all model sources remain immutable. Preparation is CPU-only
weights_only inspection; execution is one explicit MPS call after completed
five-condition evaluation evidence is recovered on the workstation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

LEGACY_SHA = "8387729dedf3bacba2c68d496db489079b03709d7e209f3f9b24bfd6cfdbc7d6"
LEGACY_PATH = Path(__file__).with_name("try_recorded_eeg.py")
if hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest() != LEGACY_SHA:
    raise RuntimeError("Immutable existing demo helper changed")
_spec = importlib.util.spec_from_file_location("xavier_immutable_demo_helpers", LEGACY_PATH)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
sha256 = _helpers.sha256
file_record = _helpers.file_record
read_json = _helpers.read_json
write_new_json = _helpers.write_new_json
manifest_bytes = _helpers.manifest_bytes
assert_originals_unchanged = _helpers.assert_originals_unchanged

RUN = "private/xavier/runs/cbramod_smollm2_uniba_v3_symbol_v1"
MANIFEST = "private/xavier/prepared/ai_coding_uniba_v3_natural_casecontrol_verified/manifest.jsonl"
CLASSES = ["ai_prompt_submit", "keyboard", "mouse_click"]
TOKEN_MAPPING = dict(zip(CLASSES, [330, 389, 340]))
VARIANTS = {"fused", "context_only", "eeg_only", "eeg_shuffled", "eeg_time_shifted"}
KNOWN = {
    "best.pt": "3019cddd288a9052fa91f9af0a7bf81b5662106178f16367525a98e6667f3a4c",
    "manifest": "030e8550bd6f4c79005722eb617e9b64f5f80e87b77866b7f17bec4e7e2a1093",
    "model_source": "b0c397176f8cb9e6dd759ed7169c0029c870f756c501ce317c2e6c7c41f32505",
    "core_source": "e30811b027414426cdfbed14a822c27695af2ab1be00106d89fe32a5fb740d2f",
    "replay_source": "87ea9eddaf839c1ec900350aca20b2baca374d560b0fdec1a7fb686673e56c1e",
    "legacy_helper_source": LEGACY_SHA,
    "upstream_weights": "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178",
}
UPSTREAM_SOURCE = {
    "models/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models/cbramod.py": "caa7c5a0e4acd488a6f625b56f7b54dc4e7190d6b68ef5b76dff2cef86874a14",
    "models/criss_cross_transformer.py": "90e90b8c98ca7eb543c44e32afc29674d3f71cb090692ad9f82db3a43edf334c",
}
ENCODER_IDENTITY = {
    "source_commit": "b9e961003214326972c567eff390e75b0287e32a",
    "source_sha256": UPSTREAM_SOURCE,
    "weights_revision": "500543c7e30bda1b22bfd51a49301b238dee21fd",
    "weights_sha256": KNOWN["upstream_weights"],
    "encoder_device": "cpu", "encoder_dtype": "float32", "encoder_frozen": True,
}
RUN_FILES = ("best.pt", "run_summary.json", "run_config.json", "training_membership.json", "eeg_artifacts.json")
GATE_FILES = ("workflow_completion", "evaluation_exit", "variant_receipt", "variant_report", "recovery_receipt", "strict_report")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_relative(root, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in relative, "contract path must be relative and contained")
    target = Path(root) / path
    require(target.resolve().is_relative_to(Path(root).resolve()) and not target.is_symlink(), "contract path escapes project or is a symlink")
    return target


def load_contract(path, expected_sha):
    require(sha256(path) == expected_sha, "reviewed demo contract SHA mismatch")
    contract = read_json(path)
    require(contract.get("schema") == "xavier-cbramod-train-demo-contract-v1"
            and contract.get("selected_partition") == "train" and contract.get("heldout_records_used_for_demo") == 0
            and contract.get("abstain_threshold") == .8 and contract.get("os_execution_allowed") is False
            and set(contract.get("run_files", {})) == set(RUN_FILES) and set(contract.get("gates", {})) == set(GATE_FILES),
            "demo contract changes scope or lacks frozen evidence")
    require(contract["run_files"]["best.pt"] == KNOWN["best.pt"] and contract.get("manifest_sha256") == KNOWN["manifest"], "demo contract uses another checkpoint or cohort")
    for item in contract["gates"].values():
        require(set(item) == {"path", "sha256"} and len(item["sha256"]) == 64, "completion evidence contract invalid")
    return contract


def verify_checkpoint(state, config, artifacts, summary):
    import torch
    require(state.get("schema") == "xavier-cbramod-symbol-llm-v1" and state.get("manifest_sha256") == KNOWN["manifest"]
            and state.get("config") == config and state.get("classes") == CLASSES and state.get("eeg_artifacts") == artifacts
            and state.get("trainable_base") == {} and state.get("eeg_encoder_identity") == ENCODER_IDENTITY
            and state.get("token_mapping") == TOKEN_MAPPING, "CBraMod checkpoint schema/provenance/frozen backbone differs")
    expected_shapes = {"0.weight": (200,), "0.bias": (200,), "1.weight": (2048, 200), "1.bias": (2048,)}
    connector = state.get("connector", {})
    require(set(connector) == set(expected_shapes), "connector must be the fixed LayerNorm200 plus Linear200-to-2048")
    for name, shape in expected_shapes.items():
        value = connector[name]
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape and torch.isfinite(value).all().item(), "connector shape/finite values invalid")
    require(config.get("mode") == "fused" and config.get("lora_rank") == 0 and config.get("abstain_threshold") == .8
            and config.get("dtype") == "bfloat16" and config.get("channels") == 8 and config.get("samples") == 448
            and config.get("prefix_tokens") == 8 and config.get("encoder_width") == 200 and config.get("eeg_unit") == "V"
            and config.get("eeg_encoder_identity") == ENCODER_IDENTITY and config.get("class_token_ids") == [330, 389, 340]
            and config.get("model") == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
            and config.get("revision") == "31b70e2e869a7173562077fd711b654946d38674"
            and config.get("model_module_sha256") == KNOWN["model_source"] and config.get("core_source_sha256") == KNOWN["core_source"],
            "fixed model/class/threshold configuration differs")
    require(summary.get("eeg_encoder_frozen") is True and summary.get("base_frozen") is True
            and summary.get("eeg_encoder_cpu_float32") is True and summary.get("trainable_parameters") == 412048,
            "training frozen-backbone evidence differs")


def verify_sources(repo, run_dir, manifest, contract_path, contract_sha):
    repo, run_dir, manifest = Path(repo).resolve(), Path(run_dir).resolve(), Path(manifest).resolve()
    require(run_dir == (repo / RUN).resolve() and manifest == (repo / MANIFEST).resolve(), "fixed CBraMod run/cohort path required")
    contract = load_contract(contract_path, contract_sha)
    upstream = repo / "private/xavier/upstream/CBraMod"
    paths = {name: run_dir / name for name in RUN_FILES}
    paths.update(manifest=manifest, contract=Path(contract_path).resolve(),
                 model_source=repo / "experiments/pj_xavier/model/cbramod_token_llm.py",
                 core_source=repo / "experiments/pj_xavier/model/eeg_llm.py",
                 replay_source=repo / "experiments/pj_xavier/integration/replay_to_neurocore.py",
                 legacy_helper_source=repo / "experiments/pj_xavier/integration/try_recorded_eeg.py",
                 upstream_weights=repo / "private/xavier/upstream/cbramod_pretrained_weights.pth")
    for name in UPSTREAM_SOURCE:
        paths["upstream:" + name] = upstream / name
    for key, value in contract["gates"].items():
        paths[key] = safe_relative(repo, value["path"])
    records = {key: file_record(path) for key, path in paths.items()}
    expected = {**KNOWN, **contract["run_files"], "contract": contract_sha,
                **{"upstream:" + key: value for key, value in UPSTREAM_SOURCE.items()},
                **{key: value["sha256"] for key, value in contract["gates"].items()}}
    for key, record in records.items():
        require(record["sha256"] == expected[key], "immutable source changed:" + key)
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    require(commit == ENCODER_IDENTITY["source_commit"], "official CBraMod source commit mismatch")
    for name in UPSTREAM_SOURCE:
        original = subprocess.check_output(["git", "-C", str(upstream), "show", commit + ":" + name])
        require(original == (upstream / name).read_bytes(), "upstream source differs from pinned commit")
    summary, config, membership, artifacts = (read_json(paths[key]) for key in ("run_summary.json", "run_config.json", "training_membership.json", "eeg_artifacts.json"))
    require(summary.get("status") == "completed_epochs" and summary.get("checkpoint_sha256") == KNOWN["best.pt"]
            and summary.get("manifest_sha256") == KNOWN["manifest"] and summary.get("model_selection_partition") == "validation"
            and summary.get("test_used_for_tuning") is False and summary.get("os_actions_executed") == 0,
            "completed validation-selected training run required")
    import torch
    import numpy as np
    torch.set_num_threads(2)
    state = torch.load(paths["best.pt"], map_location="cpu", weights_only=True)
    verify_checkpoint(state, config, artifacts, summary)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    require(len(rows) == 758 and len({row["id"] for row in rows}) == len(rows), "frozen source IDs/count mismatch")
    for split, key, count in (("train", "training_ids", 400), ("val", "validation_ids", 155), ("test", "test_ids", 203)):
        ids = sorted(row["id"] for row in rows if row["split"] == split)
        require(len(ids) == count and sorted(membership.get(key, [])) == ids, "source partition differs from training membership:" + split)
    ids_hash = hashlib.sha256("\n".join(sorted(membership["training_ids"])).encode()).hexdigest()
    require(membership.get("manifest_sha256") == KNOWN["manifest"] and membership.get("training_ids_sha256") == ids_hash
            and membership.get("model_selection_partition") == "validation" and membership.get("test_used_for_tuning") is False,
            "membership provenance mismatch")
    require(set(artifacts) == {row["eeg_path"] for row in rows} and len(artifacts) == 32, "sealed EEG artifact inventory differs")
    arrays = {}
    for name, expected in artifacts.items():
        path = safe_relative(manifest.parent, name)
        record = file_record(path)
        require({key: record[key] for key in ("bytes", "sha256")} == expected, "sealed EEG bytes changed:" + name)
        arrays[name] = record
    selected = []
    for label in CLASSES:
        row = min((row for row in rows if row["split"] == "train" and row["label"] == label), key=lambda row: row["id"])
        require(row.get("data_origin") == "real_recording" and row.get("label_source") == "recorded_event", "demo requires recorded training EEG")
        data = np.load(arrays[row["eeg_path"]]["path"], mmap_mode="r", allow_pickle=False)
        value = data[int(row["eeg_index"])]
        require(value.shape == (8, 448) and np.isfinite(value).all(), "selected training EEG shape/values invalid")
        selected.append({**row, "eeg_path": arrays[row["eeg_path"]]["path"]})
    backend = {name: file_record(repo / "backend/neurocore" / name) for name in ("audit.py", "control.py", "settings.py")}
    result = {"repo": str(repo), "run_dir": str(run_dir), "manifest": str(manifest), "sources": records, "arrays": arrays,
              "backend": backend, "selected": selected, "config": config, "upstream": str(upstream),
              "contract_sha256": contract_sha, "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__}}
    evaluation_is_complete(result)
    return result


def receipt_local_path(repo, raw):
    if raw.startswith("/content/xavier/"):
        return safe_relative(repo, "private/xavier/" + raw.removeprefix("/content/xavier/"))
    path = Path(raw)
    if path.is_absolute():
        require(path.resolve().is_relative_to(Path(repo).resolve()), "evaluation report outside project")
        return path
    return safe_relative(repo, raw)


def evaluation_is_complete(plan):
    records = plan["sources"]
    values = {name: read_json(records[name]["path"]) for name in GATE_FILES}
    workflow, exited, receipt, report, recovery, strict = (values[key] for key in GATE_FILES)
    require(workflow.get("status") == "training_and_variants_completed_pending_strict_and_recovery"
            and workflow.get("checkpoint_sha256") == records["best.pt"]["sha256"] and workflow.get("training_repeated") is False
            and workflow.get("os_actions_executed") == 0 and exited.get("exit_code") == 0
            and exited.get("checkpoint_sha256") == records["best.pt"]["sha256"]
            and receipt.get("status") == "completed" and receipt.get("exit_code") == 0,
            "completed evaluation/workflow evidence required before MPS demonstration")
    # The original workflow's historically accurate pending text remains intact.
    # Later recovery and strict evidence closes those two steps independently.
    require(recovery.get("status") == "completed" and recovery.get("all_member_hashes_verified") is True
            and recovery.get("checkpoint_sha256") == records["best.pt"]["sha256"] and recovery.get("member_count") == 371
            and len(recovery.get("member_destinations", {})) == 371, "complete recovered artifact receipt required")
    for key in ("best.pt", "workflow_completion", "evaluation_exit", "variant_receipt", "variant_report", "strict_report"):
        relative = Path(records[key]["path"]).relative_to(Path(plan["repo"]) / "private/xavier").as_posix()
        require(recovery["member_destinations"].get(relative) == relative, "demo evidence not present in recovered archive:" + key)
    require(strict.get("schema_version") == "xavier-benchmark-v1" and strict.get("status") == "evaluated_goal_unproven"
            and strict.get("checkpoint_sha256") == records["best.pt"]["sha256"] and strict.get("integrity_errors") == []
            and strict.get("data_validation", {}).get("errors") == [] and strict.get("resource_feasibility", {}).get("status") == "passed"
            and strict.get("autonomous_agent_validated") is False, "strict evaluation completion/provenance missing")
    require(receipt.get("checkpoint_sha256") == records["best.pt"]["sha256"]
            and len(receipt.get("variant_names", [])) == 5 and set(receipt["variant_names"]) == VARIANTS,
            "evaluation checkpoint/conditions mismatch")
    report_path = receipt_local_path(plan["repo"], receipt["report_path"])
    require(report_path.resolve() == Path(records["variant_report"]["path"]).resolve()
            and sha256(report_path) == receipt.get("report_sha256"), "recovered evaluation report path/hash mismatch")
    require(report.get("schema_version") == "pj-xavier-variant-evaluation-v1" and report.get("status") == "completed"
            and report.get("checkpoint_sha256") == receipt["checkpoint_sha256"]
            and len(report.get("variant_names", [])) == 5 and set(report["variant_names"]) == VARIANTS,
            "five-condition report provenance mismatch")
    for name in VARIANTS:
        condition = report.get("conditions", {}).get(name, {})
        require(condition.get("status") in {"measured", "blocked"}
                and (condition["status"] != "blocked" or bool(condition.get("reason"))), "unfinished evaluation condition:" + name)


def commands_for(verified, output):
    python = str(Path(sys.executable).absolute())
    predict = [python, verified["sources"]["model_source"]["path"], "predict", "--checkpoint", verified["sources"]["best.pt"]["path"],
               "--manifest", str(output / "train_examples.jsonl"), "--output", str(output / "predictions.jsonl"),
               "--device", "mps", "--memory-budget-gib", "8", "--dtype", "bfloat16", "--max-examples", "3",
               "--cbramod-upstream", verified["upstream"], "--cbramod-weights", verified["sources"]["upstream_weights"]["path"]]
    replay = [python, verified["sources"]["replay_source"]["path"], "--predictions", str(output / "predictions.jsonl"),
              "--output", str(output / "dry_run_audit.jsonl")]
    return predict, replay


def prepare(repo, contract_path, contract_sha, output_dir):
    repo, output = Path(repo).resolve(), Path(output_dir).resolve()
    require(not output.exists(), "existing output is never overwritten")
    verified = verify_sources(repo, repo / RUN, repo / MANIFEST, contract_path, contract_sha)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "train_examples.jsonl").open("xb") as stream:
        stream.write(manifest_bytes(verified["selected"]))
    predict, replay = commands_for(verified, output)
    plan = {"schema_version": "pj-xavier-cbramod-recorded-train-demo-v1", "status": "prepared_only", "created_epoch": time.time(),
            **{key: value for key, value in verified.items() if key != "selected"},
            "selected_ids": [row["id"] for row in verified["selected"]], "selected_labels": CLASSES,
            "selection": "lexicographically first train record ID per class; same method as prior demo; no prediction-based selection",
            "train_examples_sha256": sha256(output / "train_examples.jsonl"), "predict_command": predict, "replay_command": replay,
            "wrapper_sha256": sha256(__file__), "python_executable": str(Path(sys.executable).absolute()),
            "heldout_records_used_for_demo": 0, "model_or_threshold_changed": False, "os_actions_executed": 0}
    write_new_json(output / "plan.json", plan)
    write_new_json(output / "plan.sha256.json", {"sha256": sha256(output / "plan.json")})
    write_new_json(output / "prepared_summary.json", {"status": "prepared_only", "examples": 3, "selected_partition": "train",
                   "classes": CLASSES, "arrays_verified": len(verified["arrays"]), "inference_executed": False,
                   "heldout_records_used_for_demo": 0, "os_actions_executed": 0, "plan_sha256": sha256(output / "plan.json")})
    return plan


def summarize_outputs(plan, selected, output):
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines() if line.strip()]
    audit = [json.loads(line) for line in (output / "dry_run_audit.jsonl").read_text().splitlines() if line.strip()]
    metadata, routing = read_json(output / "predictions.metadata.json"), read_json(output / "dry_run_audit.summary.json")
    require(metadata.get("checkpoint_sha256") == plan["sources"]["best.pt"]["sha256"] and metadata.get("abstain_threshold") == .8
            and metadata.get("checkpoint_modified") is False and metadata.get("model_or_threshold_changed") is False
            and metadata.get("candidate_forward_passes") == 1 and routing.get("mode") == "dry_run_only"
            and routing.get("os_actions_executed") == 0 and routing.get("cloud_agent_calls") == 0
            and routing.get("predictions_sha256") == sha256(output / "predictions.jsonl")
            and routing.get("audit_sha256") == sha256(output / "dry_run_audit.jsonl") and len(predictions) == len(audit) == 3,
            "predictor/dry-run output contract mismatch")
    examples = []
    for original, prediction, event in zip(selected, predictions, audit):
        probs = prediction.get("probabilities", {})
        require(prediction.get("id") == original["id"] and prediction.get("eeg_source_id") == original["id"]
                and prediction.get("condition") == "normal" and prediction.get("action_executed") is False
                and prediction.get("probability_kind") == "restricted_three_next_token_softmax_uncalibrated"
                and prediction.get("candidate_forward_passes") == 1 and set(probs) == set(CLASSES)
                and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in probs.values())
                and math.isclose(sum(probs.values()), 1, abs_tol=1e-5), "prediction identity/probabilities invalid")
        label = prediction.get("predicted_label")
        require(label in probs and probs[label] == max(probs.values()) and prediction.get("abstain") is (probs[label] < .8)
                and prediction.get("decision_time_s") == original["decision_time_s"], "prediction label/threshold/timing changed")
        action = event.get("action", {})
        require(action.get("metadata", {}).get("record_id") == original["id"] and action.get("blocked") is True
                and event.get("note") == "blocked" and bool(action.get("reason")), "default router must block unmapped/abstained proxy actions")
        examples.append({"record_id": original["id"], "source_split": "train", "recorded_action_label": original["label"],
                         "predicted_label": label, "probabilities": probs, "abstain": prediction["abstain"], "abstain_threshold": .8,
                         "router_blocked": True, "reason": action["reason"], "os_action_executed": False})
    assert_originals_unchanged(plan)
    return {"schema_version": "pj-xavier-cbramod-recorded-train-demo-result-v1", "status": "completed", "completed_epoch": time.time(),
            "examples": examples, "selected_partition": "train", "heldout_records_used_for_demo": 0, "model_or_threshold_changed": False,
            "os_actions_executed": 0, "cloud_agent_calls": 0, "autonomous_agent_validated": False, "generalization_accuracy_estimated": False,
            "probabilities_are_calibrated_intent_confidence": False, "probability_kind": "restricted_three_next_token_softmax_uncalibrated",
            "checkpoint_sha256": plan["sources"]["best.pt"]["sha256"], "source_manifest_sha256": plan["sources"]["manifest"]["sha256"],
            "contract_sha256": plan["contract_sha256"], "artifacts": {name: file_record(output / name) for name in
                ("predictions.jsonl", "predictions.metadata.json", "dry_run_audit.jsonl", "dry_run_audit.summary.json")}}


def execute_prepared(output_dir, *, runner=subprocess.run):
    output = Path(output_dir).resolve()
    plan = read_json(output / "plan.json")
    require(sha256(output / "plan.json") == read_json(output / "plan.sha256.json")["sha256"] and sha256(__file__) == plan["wrapper_sha256"], "prepared plan/wrapper changed")
    require(str(Path(sys.executable).absolute()) == plan["python_executable"], "use the prepared Python environment")
    assert_originals_unchanged(plan)
    verified = verify_sources(plan["repo"], plan["run_dir"], plan["manifest"], plan["sources"]["contract"]["path"], plan["contract_sha256"])
    require(all(verified[group] == plan[group] for group in ("sources", "arrays", "backend")), "prepared file inventory changed")
    require((plan["predict_command"], plan["replay_command"]) == commands_for(verified, output), "prepared commands changed")
    require(plan["selected_ids"] == [row["id"] for row in verified["selected"]], "prepared train selection changed")
    expected = manifest_bytes(verified["selected"])
    require((output / "train_examples.jsonl").read_bytes() == expected and hashlib.sha256(expected).hexdigest() == plan["train_examples_sha256"], "train-only sample changed")
    for name in ("execution_started.json", "predictions.jsonl", "predictions.metadata.json", "dry_run_audit.jsonl", "dry_run_audit.summary.json", "summary.json", "failure.json"):
        require(not (output / name).exists(), "execution already started; automatic rerun forbidden")
    write_new_json(output / "execution_started.json", {"started_epoch": time.time(), "pid": os.getpid(), "plan_sha256": sha256(output / "plan.json"), "execution_attempt": 1})
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONPATH=str(Path(plan["repo"]) / "backend"))
    env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[key] = "2"
    try:
        for name, command in (("predict", plan["predict_command"]), ("replay", plan["replay_command"])):
            with (output / (name + ".log")).open("xb") as log:
                result = runner(command, cwd=plan["repo"], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=False)
            require(result.returncode == 0, name + " failed; keep this attempt evidence")
        summary = summarize_outputs(plan, verified["selected"], output)
        write_new_json(output / "summary.json", summary)
        return summary
    except BaseException as error:
        write_new_json(output / "failure.json", {"status": "failed_or_interrupted", "type": type(error).__name__, "reason": str(error)[:500], "automatic_rerun_allowed": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--expected-contract-sha256")
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute-prepared", type=Path)
    args = parser.parse_args()
    if args.execute_prepared:
        if args.contract or args.expected_contract_sha256 or args.output_dir:
            parser.error("execute-prepared uses sealed inputs; overrides are not accepted")
        result = execute_prepared(args.execute_prepared)
    else:
        if not all((args.contract, args.expected_contract_sha256, args.output_dir)):
            parser.error("new preparation requires --contract, --expected-contract-sha256, --output-dir")
        prepare(args.repo, args.contract, args.expected_contract_sha256, args.output_dir)
        result = read_json(args.output_dir / "prepared_summary.json") if args.prepare_only else execute_prepared(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
