#!/usr/bin/env python3
"""Try three frozen training EEG examples through the existing predictor and dry-run router.

This is a recorded-data demonstration, not a held-out benchmark or an OS agent.
No model, threshold, source array, checkpoint, or existing output is changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


PINS = {
    "manifest": "030e8550bd6f4c79005722eb617e9b64f5f80e87b77866b7f17bec4e7e2a1093",
    "best.pt": "281716da86293d2b700ec003c967ca2c141515cc565b136a12137c34b73a3475",
    "run_summary.json": "9d9b8d65cde7636dfe07d73b74766e6b0d8e1abfd915c6ac47ace27adfd87065",
    "run_config.json": "400cee905b3ae2223b1de8e0dfc23f551c3e27e196fa2bf9c7d0db583aa51074",
    "training_membership.json": "5d8636f427f705d7c914797f3b505cc158225684ef278f730c92b1ef45b647f0",
    "eeg_artifacts.json": "0a2989891e46e2aacab785df162a84376f18acde444deb49e341a3b96969192f",
    "exit_receipt.json": "8d676c87725603a56388644ff88ae1acdec457cac00863610470aff1a1122358",
    "model_source": "e30811b027414426cdfbed14a822c27695af2ab1be00106d89fe32a5fb740d2f",
    "replay_source": "87ea9eddaf839c1ec900350aca20b2baca374d560b0fdec1a7fb686673e56c1e",
}
RUN = "private/xavier/runs/smollm2_1_7b_uniba_natural_v3_fused_v1"
MANIFEST = "private/xavier/prepared/ai_coding_uniba_v3_natural_casecontrol_verified/manifest.jsonl"
EVALUATION_EXIT = "private/xavier/m1_natural_v3_final_evaluation/exit_receipt.json"
CLASSES = ["ai_prompt_submit", "keyboard", "mouse_click"]
VARIANTS = {"fused", "context_only", "eeg_only", "eeg_shuffled", "eeg_time_shifted"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024**2), b""):
            h.update(data)
    return h.hexdigest()


def file_record(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("source must be an existing non-symlink file:" + str(path))
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("source changed during verification")
    return {"path": str(path.resolve()), "bytes": before.st_size, "sha256": digest}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")


def verify_sources(repo, run_dir, manifest, *, pins=None):
    """No model import/backbone load: only CPU weights_only checkpoint inspection."""
    pins = PINS if pins is None else pins
    repo, run_dir, manifest = Path(repo).resolve(), Path(run_dir).resolve(), Path(manifest).resolve()
    paths = {name: run_dir / name for name in ("best.pt", "run_summary.json", "run_config.json",
             "training_membership.json", "eeg_artifacts.json", "exit_receipt.json")}
    paths.update(manifest=manifest, model_source=repo / "experiments/pj_xavier/model/eeg_llm.py",
                 replay_source=repo / "experiments/pj_xavier/integration/replay_to_neurocore.py")
    records = {name: file_record(path) for name, path in paths.items()}
    for name, record in records.items():
        if record["sha256"] != pins[name]:
            raise ValueError("frozen source hash mismatch:" + name)
    summary, config, membership, artifacts, exited = (read_json(paths[name]) for name in
        ("run_summary.json", "run_config.json", "training_membership.json", "eeg_artifacts.json", "exit_receipt.json"))
    if (summary.get("status") != "completed_epochs" or summary.get("checkpoint_sha256") != pins["best.pt"]
            or summary.get("manifest_sha256") != pins["manifest"] or summary.get("test_used_for_tuning") is not False
            or summary.get("model_selection_partition") != "validation" or exited.get("exit_code") != 0
            or exited.get("source_sha256") != pins["model_source"] or exited.get("manifest_sha256") != pins["manifest"]):
        raise ValueError("run is not the frozen completed validation-selected training run")
    import torch
    import numpy as np
    torch.set_num_threads(2)
    state = torch.load(paths["best.pt"], map_location="cpu", weights_only=True)
    if (state.get("schema") != "xavier-eeg-llm-v1" or state.get("manifest_sha256") != pins["manifest"]
            or state.get("config") != config or state.get("classes") != CLASSES
            or state.get("eeg_artifacts") != artifacts or state.get("trainable_base") != {}):
        raise ValueError("checkpoint provenance or frozen-backbone contract mismatch")
    if not state.get("connector") or not all(isinstance(x, torch.Tensor) and torch.isfinite(x).all().item() for x in state["connector"].values()):
        raise ValueError("connector weights are absent or non-finite")
    if (config.get("mode") != "fused" or config.get("lora_rank") != 0 or config.get("abstain_threshold") != .8
            or config.get("device") != "mps" or config.get("dtype") != "bfloat16"
            or config.get("memory_budget_gib") != 8 or config.get("candidate_batch_size") != 2):
        raise ValueError("frozen model, threshold, or runtime settings differ")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate source record IDs")
    for split, key in (("train", "training_ids"), ("val", "validation_ids"), ("test", "test_ids")):
        expected = sorted(r["id"] for r in rows if r["split"] == split)
        if sorted(membership.get(key, [])) != expected or not expected:
            raise ValueError("source split does not match training membership:" + split)
    ids_hash = hashlib.sha256("\n".join(sorted(membership["training_ids"])).encode()).hexdigest()
    if (membership.get("manifest_sha256") != pins["manifest"] or membership.get("training_ids_sha256") != ids_hash
            or membership.get("model_selection_partition") != "validation" or membership.get("test_used_for_tuning") is not False):
        raise ValueError("training membership provenance mismatch")
    if set(artifacts) != {r["eeg_path"] for r in rows}:
        raise ValueError("prepared EEG artifact inventory differs from the source manifest")
    array_records = {}
    for key, expected in artifacts.items():
        path = Path(key)
        path = path if path.is_absolute() else manifest.parent / path
        record = file_record(path)
        if {k: record[k] for k in ("sha256", "bytes")} != expected:
            raise ValueError("prepared EEG bytes changed:" + key)
        array_records[key] = record
    selected = []
    for label in CLASSES:
        candidates = [r for r in rows if r["split"] == "train" and r["label"] == label]
        if not candidates:
            raise ValueError("training class has no demonstration example:" + label)
        row = min(candidates, key=lambda r: r["id"])
        if row.get("data_origin") != "real_recording" or row.get("label_source") not in {"recorded_event", "human_verified", "dataset_annotation"}:
            raise ValueError("demo requires recorded training EEG with source labels")
        array = np.load(array_records[row["eeg_path"]]["path"], mmap_mode="r", allow_pickle=False)
        value = array[int(row["eeg_index"])]
        if value.shape != (config["channels"], config["samples"]) or not np.isfinite(value).all():
            raise ValueError("selected EEG window shape or values invalid")
        selected.append({**row, "eeg_path": array_records[row["eeg_path"]]["path"]})
    backend = {name: file_record(repo / "backend/neurocore" / name) for name in ("audit.py", "control.py", "settings.py")}
    return {"repo": str(repo), "run_dir": str(run_dir), "manifest": str(manifest), "sources": records,
            "arrays": array_records, "backend": backend, "selected": selected, "config": config,
            "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__}}


def manifest_bytes(rows):
    return ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows)).encode()


def commands_for(verified, output):
    # Preserve the virtualenv invocation path; resolving this symlink would
    # bypass pyvenv.cfg and silently select the base interpreter's packages.
    python = str(Path(sys.executable).absolute())
    predict = [python, verified["sources"]["model_source"]["path"], "predict", "--checkpoint",
               verified["sources"]["best.pt"]["path"], "--manifest", str(output / "train_examples.jsonl"),
               "--output", str(output / "predictions.jsonl"), "--device", "mps", "--memory-budget-gib", "8",
               "--candidate-batch-size", "2", "--dtype", "bfloat16", "--max-examples", "3"]
    replay = [python, verified["sources"]["replay_source"]["path"], "--predictions", str(output / "predictions.jsonl"),
              "--output", str(output / "dry_run_audit.jsonl")]
    return predict, replay


def prepare(repo, run_dir, manifest, output_dir, *, pins=None):
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("existing output directory is never overwritten; choose a new directory")
    verified = verify_sources(repo, run_dir, manifest, pins=pins)
    output.mkdir(parents=True, exist_ok=False)
    sample = output / "train_examples.jsonl"
    with sample.open("xb") as f:
        f.write(manifest_bytes(verified["selected"]))
    python = str(Path(sys.executable).absolute())
    predict, replay = commands_for(verified, output)
    plan = {"schema_version": "pj-xavier-recorded-train-demo-v1", "status": "prepared_only", "created_epoch": time.time(),
            **{k: v for k, v in verified.items() if k != "selected"}, "selected_ids": [r["id"] for r in verified["selected"]],
            "selected_labels": CLASSES, "selection": "lexicographically first training record ID per class, independent of predictions",
            "train_examples_sha256": sha256(sample), "predict_command": predict, "replay_command": replay,
            "wrapper_sha256": sha256(__file__), "python_executable": python,
            "heldout_records_used_for_demo": 0, "model_or_threshold_changed": False, "os_actions_executed": 0,
            "evaluation_exit_receipt": str(Path(verified["repo"]) / EVALUATION_EXIT),
            "evaluation_completion": str(Path(verified["run_dir"]) / "variant_evaluation_completion.json")}
    write_new_json(output / "plan.json", plan)
    write_new_json(output / "plan.sha256.json", {"sha256": sha256(output / "plan.json")})
    write_new_json(output / "prepared_summary.json", {"status": "prepared_only", "examples": 3,
                   "selected_partition": "train", "classes": CLASSES, "arrays_verified": len(verified["arrays"]),
                   "inference_executed": False, "os_actions_executed": 0, "heldout_records_used_for_demo": 0,
                   "plan_sha256": sha256(output / "plan.json")})
    return plan


def evaluation_is_complete(plan):
    exited, receipt = read_json(plan["evaluation_exit_receipt"]), read_json(plan["evaluation_completion"])
    if exited.get("exit_code") != 0 or receipt.get("status") != "completed" or receipt.get("exit_code") != 0:
        raise ValueError("final M1 evaluation is not complete; do not overlap accelerator work")
    if (receipt.get("checkpoint_sha256") != plan["sources"]["best.pt"]["sha256"]
            or len(receipt.get("variant_names", [])) != 5 or set(receipt.get("variant_names", [])) != VARIANTS):
        raise ValueError("final evaluation refers to a different checkpoint or conditions")
    report_path = Path(receipt["report_path"])
    if not report_path.is_absolute():
        report_path = Path(plan["repo"]) / report_path
    if not report_path.resolve().is_relative_to(Path(plan["repo"]).resolve()):
        raise ValueError("final evaluation report lies outside the project")
    if sha256(report_path) != receipt["report_sha256"]:
        raise ValueError("final evaluation report hash mismatch")
    report = read_json(report_path)
    if (report.get("schema_version") != "pj-xavier-variant-evaluation-v1" or report.get("status") != "completed"
            or report.get("checkpoint_sha256") != receipt["checkpoint_sha256"]
            or len(report.get("variant_names", [])) != 5 or set(report.get("variant_names", [])) != VARIANTS):
        raise ValueError("final evaluation report provenance mismatch")
    for name in VARIANTS:
        value = report.get("conditions", {}).get(name, {})
        if value.get("status") not in {"measured", "blocked"} or (value["status"] == "blocked" and not value.get("reason")):
            raise ValueError("final evaluation condition not finalized:" + name)


def assert_originals_unchanged(plan):
    for group in ("sources", "arrays", "backend"):
        for record in plan[group].values():
            if file_record(record["path"]) != record:
                raise ValueError("frozen input changed after preparation:" + record["path"])


def execute_prepared(output_dir, *, pins=None, runner=subprocess.run):
    output = Path(output_dir).resolve()
    plan = read_json(output / "plan.json")
    if sha256(output / "plan.json") != read_json(output / "plan.sha256.json")["sha256"] or sha256(__file__) != plan["wrapper_sha256"]:
        raise ValueError("prepared plan or wrapper source changed")
    if str(Path(sys.executable).absolute()) != plan["python_executable"]:
        raise ValueError("use the same prepared Python environment")
    assert_originals_unchanged(plan)
    verified = verify_sources(plan["repo"], plan["run_dir"], plan["manifest"], pins=pins)
    if any(plan[group] != verified[group] for group in ("sources", "arrays", "backend")):
        raise ValueError("prepared file identities differ from the verified source inventory")
    if (plan["predict_command"], plan["replay_command"]) != commands_for(verified, output):
        raise ValueError("prepared command differs from the fixed predictor and dry-run router")
    if plan["selected_ids"] != [r["id"] for r in verified["selected"]]:
        raise ValueError("prepared selection IDs changed")
    expected = manifest_bytes(verified["selected"])
    if (output / "train_examples.jsonl").read_bytes() != expected or hashlib.sha256(expected).hexdigest() != plan["train_examples_sha256"]:
        raise ValueError("demo manifest differs from deterministic training-only selection")
    evaluation_is_complete(plan)
    for name in ("execution_started.json", "predictions.jsonl", "predictions.metadata.json", "dry_run_audit.jsonl",
                 "dry_run_audit.summary.json", "summary.json", "failure.json"):
        if (output / name).exists():
            raise ValueError("demo execution already started or output exists; automatic rerun refused")
    write_new_json(output / "execution_started.json", {"started_epoch": time.time(), "pid": os.getpid(),
                   "plan_sha256": sha256(output / "plan.json"), "execution_attempt": 1})
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONPATH=str(Path(plan["repo"]) / "backend"))
    env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "2"
    try:
        for name, command in (("predict", plan["predict_command"]), ("replay", plan["replay_command"])):
            with (output / (name + ".log")).open("xb") as log:
                result = runner(command, cwd=plan["repo"], env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, timeout=600, check=False)
            if result.returncode:
                raise RuntimeError(name + " failed with exit " + str(result.returncode))
        predictions = [json.loads(x) for x in (output / "predictions.jsonl").read_text().splitlines() if x.strip()]
        audit = [json.loads(x) for x in (output / "dry_run_audit.jsonl").read_text().splitlines() if x.strip()]
        metadata, routing = read_json(output / "predictions.metadata.json"), read_json(output / "dry_run_audit.summary.json")
        if (metadata.get("checkpoint_sha256") != plan["sources"]["best.pt"]["sha256"]
                or metadata.get("abstain_threshold") != .8 or metadata.get("checkpoint_modified") is not False
                or routing.get("mode") != "dry_run_only" or routing.get("os_actions_executed") != 0
                or routing.get("cloud_agent_calls") != 0 or routing.get("predictions_sha256") != sha256(output / "predictions.jsonl")
                or routing.get("audit_sha256") != sha256(output / "dry_run_audit.jsonl")
                or len(predictions) != 3 or len(audit) != 3):
            raise ValueError("prediction or dry-run output contract mismatch")
        examples = []
        for source, prediction, event in zip(verified["selected"], predictions, audit):
            probs = prediction.get("probabilities", {})
            if (prediction.get("id") != source["id"] or prediction.get("eeg_source_id") != source["id"]
                    or prediction.get("condition") != "normal" or prediction.get("action_executed") is not False
                    or set(probs) != set(CLASSES) or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values())
                    or not math.isclose(sum(probs.values()), 1, abs_tol=1e-5)
                    or prediction.get("predicted_label") not in probs):
                raise ValueError("prediction identities or class probabilities invalid")
            label = prediction["predicted_label"]
            if (prediction.get("abstain") is not (probs[label] < .8) or probs[label] != max(probs.values())
                    or prediction.get("decision_time_s") != source["decision_time_s"]):
                raise ValueError("prediction abstention threshold changed")
            action = event.get("action", {})
            if action.get("metadata", {}).get("record_id") != source["id"] or event.get("note") not in {"blocked", "dry_run_only"}:
                raise ValueError("dry-run audit record identity mismatch")
            examples.append({"record_id": source["id"], "source_split": "train", "recorded_action_label": source["label"],
                             "predicted_label": label, "probabilities": probs, "abstain": prediction["abstain"],
                             "abstain_threshold": .8, "router_blocked": action.get("blocked"),
                             "reason": action.get("reason"), "os_action_executed": False})
        assert_originals_unchanged(plan)
        result = {"schema_version": "pj-xavier-recorded-train-demo-result-v1", "status": "completed",
                  "completed_epoch": time.time(), "examples": examples, "selected_partition": "train",
                  "heldout_records_used_for_demo": 0, "model_or_threshold_changed": False,
                  "os_actions_executed": 0, "cloud_agent_calls": 0, "autonomous_agent_validated": False,
                  "generalization_accuracy_estimated": False, "probabilities_are_calibrated_intent_confidence": False,
                  "checkpoint_sha256": plan["sources"]["best.pt"]["sha256"], "source_manifest_sha256": plan["sources"]["manifest"]["sha256"],
                  "artifacts": {name: file_record(output / name) for name in ("predictions.jsonl", "predictions.metadata.json", "dry_run_audit.jsonl", "dry_run_audit.summary.json")}}
        write_new_json(output / "summary.json", result)
        return result
    except BaseException as error:
        write_new_json(output / "failure.json", {"status": "failed_or_interrupted", "type": type(error).__name__,
                       "reason": str(error)[:500], "automatic_rerun_allowed": False})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output-dir", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute-prepared", type=Path)
    a = p.parse_args()
    if a.execute_prepared:
        if a.run_dir or a.manifest or a.output_dir:
            p.error("--execute-prepared uses its sealed paths; do not override inputs")
        result = execute_prepared(a.execute_prepared)
    else:
        if a.output_dir is None:
            p.error("--output-dir is required and must not exist")
        prepare(a.repo, a.run_dir or a.repo / RUN, a.manifest or a.repo / MANIFEST, a.output_dir)
        result = read_json(a.output_dir / "prepared_summary.json") if a.prepare_only else execute_prepared(a.output_dir)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
