#!/usr/bin/env python3
"""Run a pinned, bounded EVLM curriculum without reusing held-out data.

Each stage receives the previous verified adapter and a fresh Adam optimizer.
This is explicit stage-wise warm-start, never an exact optimizer resume.
The train runner owns the node GPU lock. This controller owns a separate
curriculum lock, launches one child at a time, and stops on failed receipts.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import random
import signal
import subprocess
import sys
import time

SCHEMA = "evlm-curriculum-v1"
TERMINAL = {"completed", "stopped", "failed", "paused_update_budget", "stopped_numeric_gate"}

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def bound_file(item, label):
    p = Path(item["path"]).expanduser().resolve()
    require(p.is_file(), label + "_missing")
    require(re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")), label + "_hash_missing")
    require(sha256(p) == item["sha256"], label + "_hash_mismatch")
    return p

def numeric_policy(plan):
    if "numeric_retention_gate" not in plan:
        return None
    policy = {"enabled": True, "relative_tolerance": .05, "invalid_output_penalty": 40000,
              "min_paired_examples": 30, "stagnation_patience": 2,
              "development_examples": 36, "targets_per_example": 4}
    supplied = plan["numeric_retention_gate"]
    require(isinstance(supplied, dict) and set(supplied) <= set(policy), "unknown_numeric_gate_setting")
    require(supplied.get("enabled") is True, "numeric_gate_must_be_explicitly_enabled")
    for key, value in supplied.items():
        require(value == policy[key] and (key == "enabled" or type(value) in (int, float)),
                "unregistered_numeric_gate_threshold")
    return policy

def remaining_presentations(plan):
    return sum(s["train_records"] - s.get("start_order_offset", 0) for s in plan["sources"])

def planned_updates(plan):
    return min(remaining_presentations(plan), plan.get("max_total_optimizer_updates", remaining_presentations(plan)))

def read_plan(path):
    plan = json.loads(Path(path).read_text())
    require(plan.get("schema") == SCHEMA, "unsupported_curriculum_schema")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", plan.get("curriculum_id", "")), "unsafe_curriculum_id")
    root = Path(plan["run_root"]).expanduser()
    require(root.is_absolute(), "run_root_must_be_absolute")
    require(isinstance(plan.get("max_stage_updates"), int) and
            1 <= plan["max_stage_updates"] <= 400, "stage_budget_out_of_range")
    require(type(plan.get("initial_stage_updates", plan["max_stage_updates"])) is int and
            1 <= plan.get("initial_stage_updates", plan["max_stage_updates"]) <= plan["max_stage_updates"],
            "invalid_initial_stage_budget")
    if "stage_update_schedule" in plan:
        schedule = plan["stage_update_schedule"]
        require(isinstance(schedule, list) and schedule and
                all(type(x) is int and 1 <= x <= plan["max_stage_updates"] for x in schedule),
                "invalid_stage_update_schedule")
        require("initial_stage_updates" not in plan or schedule[0] == plan["initial_stage_updates"],
                "conflicting_initial_stage_schedule")
    if "max_total_optimizer_updates" in plan:
        require(type(plan["max_total_optimizer_updates"]) is int and plan["max_total_optimizer_updates"] > 0,
                "invalid_total_update_budget")
    policy = numeric_policy(plan)
    require(60 <= plan.get("max_controller_wall_seconds", 0) <= 7 * 86400,
            "controller_must_have_bounded_wall_budget")
    require(plan.get("scope_approved_by_probe", {}).get("passed") is True,
            "requires_reviewed_engineering_probe")
    probe_path = bound_file(plan["scope_approved_by_probe"]["receipt"], "scope_probe")
    probe = json.loads(probe_path.read_text())
    require(probe.get("mode") == "engineering_probe", "wrong_probe_receipt_mode")
    require(probe.get("quality_gates_passed") is True, "probe_quality_failed")
    require(probe.get("completed_optimizer_steps") == 2, "probe_updates_not_verified")
    require(probe.get("test_opened") is False, "probe_test_policy_missing")
    bound_file(plan["runner"], "runner")
    for item in plan.get("supporting_sources", []):
        bound_file(item, "supporting_source")
    py = Path(plan["python_path"]).expanduser()
    require(py.is_absolute() and py.is_file(), "python_missing")
    sources = plan.get("sources", [])
    require(sources and isinstance(sources, list), "empty_curriculum")
    ids, hashes, validation_identity = set(), set(), None
    for src in sources:
        sid = src.get("source_id", "")
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", sid), "unsafe_source_id")
        require(sid not in ids, "duplicate_source_id")
        ids.add(sid)
        require(isinstance(src.get("train_records"), int) and src["train_records"] > 0,
                "invalid_train_count")
        require(type(src.get("start_order_offset", 0)) is int and
                0 <= src.get("start_order_offset", 0) < src["train_records"],
                "invalid_explicit_continuation_offset")
        require(src.get("admission_status") == "admitted_train_only", "source_not_admitted")
        require(src.get("test_or_review_used") is False, "source_heldout_policy_missing")
        p = bound_file(src["data_manifest"], "data_manifest")
        require(src["data_manifest"]["sha256"] not in hashes, "duplicate_data_manifest")
        hashes.add(src["data_manifest"]["sha256"])
        data = json.loads(p.read_text())
        # The training runner revalidates full data, source/subject splits and
        # token lengths before every fit. This binding prevents queue substitution.
        require(src.get("data_schema") == data.get("schema"), "data_schema_binding_mismatch")
        require(src["train_records"] == data.get("inputs", {}).get("train", {}).get("count"),
                "source_train_count_differs_from_manifest")
        canonical_source = src.get("canonical_source_id", src["source_id"])
        require(canonical_source in data.get("source_ids", []), "source_id_not_in_manifest")
        if policy is not None:
            require(data.get("task_kind") == "quantized_eeg_reconstruction"
                    and data.get("quantization_schema") == "eeg-raw32-uv1-channel-major-cycle19-v1"
                    and data.get("quantization_bounds") == [-100, 100], "unsupported_numeric_gate_task")
            val = data.get("inputs", {}).get("val", {})
            bound_file(val, "numeric_validation")
            require(type(val.get("count")) is int and val["count"] >= 36, "insufficient_fixed_numeric_validation")
            identity = {"sha256": val["sha256"], "count": val["count"]}
            require(validation_identity is None or validation_identity == identity,
                    "numeric_validation_differs_across_sources")
            validation_identity = identity
    template = plan["stage_template"]
    require(template.get("schema") == "evlm-mlx-train-v1", "wrong_runner_schema")
    require(template.get("mode") == "train_stage", "not_training_template")
    require(template.get("num_layers") == plan["scope_approved_by_probe"].get("num_layers"),
            "scope_differs_from_probe")
    require(template.get("num_layers") == probe.get("num_layers"), "probe_actual_scope_mismatch")
    require(template.get("rank") == probe.get("rank"), "probe_rank_mismatch")
    require(template.get("max_seq_length") == probe.get("max_seq_length"), "probe_context_mismatch")
    require(template.get("expected_model_config_sha256") == probe.get("expected_model_config_sha256"),
            "probe_base_model_mismatch")
    expected_base_identity = hashlib.sha256(json.dumps({
        "model_path": str(Path(template["model_path"]).expanduser().resolve()),
        "config_sha256": template["expected_model_config_sha256"],
        "receipt_sha256": template["base_receipt"]["sha256"],
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    require(probe.get("base_identity") == expected_base_identity, "probe_base_identity_mismatch")
    require(template.get("max_seq_length") in (801, 1024), "unregistered_context_length")
    require(template.get("rank") == 8, "unregistered_rank")
    require(template.get("memory_limit_gb", 0) <= 54, "memory_budget_exceeds_approved_probe")
    if policy is not None:
        require(template.get("eval_max_records", 36) == 36 and type(template.get("seed")) is int,
                "numeric_gate_requires_fixed_dev36_and_seed")
    gate = plan.get("vision_gate", {})
    require(gate.get("enabled") is True, "evlm_curriculum_requires_vision_retention_gate")
    bound_file(gate["runtime"], "vlm_runtime")
    require(Path(gate["python_path"]).is_absolute() and Path(gate["python_path"]).is_file(),
            "vlm_python_missing")
    require(re.fullmatch(r"[0-9a-f]{40}", gate.get("base_revision", "")), "missing_pinned_base_revision")
    require(60 <= gate.get("max_wall_seconds", 0) <= 900, "invalid_vision_gate_budget")
    require(gate.get("memory_gb") == 54, "vision_gate_memory_policy_changed")
    require(plan.get("final_test_policy") == "never_open", "final_test_policy_missing")
    require(plan.get("automatic_scope_expansion") is False, "scope_expansion_forbidden")
    return plan

def stage_specs(plan):
    step_cap = plan["max_stage_updates"]
    number = 0
    budget = planned_updates(plan)
    schedule = plan.get("stage_update_schedule", [])
    for src in plan["sources"]:
        offset = src.get("start_order_offset", 0)
        while offset < src["train_records"]:
            if budget <= 0:
                return
            cap = schedule[number] if number < len(schedule) else (
                plan.get("initial_stage_updates", step_cap) if number == 0 else step_cap)
            updates = min(cap, src["train_records"] - offset, budget)
            sid = "stage_%05d_%s_%07d" % (number, src["source_id"], offset)
            yield {
                "stage_id": sid, "stage_index": number,
                "source_id": src["source_id"],
                "canonical_source_id": src.get("canonical_source_id", src["source_id"]),
                "order_offset": offset,
                "max_updates": updates, "data_manifest": src["data_manifest"],
                "train_records_in_source": src["train_records"],
            }
            offset += updates
            budget -= updates
            number += 1

def checkpoint_steps(updates):
    # Early sanity evaluation plus a final receipt, without ever silently
    # turning a stage into a full held-out-test evaluation.
    return sorted(set([n for n in (20, 100, 200) if n < updates] + [updates]))

def make_stage_config(plan, spec, parent):
    cfg = dict(plan["stage_template"])
    cfg.update(
        run_id=plan["curriculum_id"] + "_" + spec["stage_id"],
        run_dir=str(Path(plan["run_root"]) / "stages" / spec["stage_id"]),
        data_manifest=spec["data_manifest"],
        order_offset=spec["order_offset"],
        max_updates=spec["max_updates"],
        checkpoint_steps=checkpoint_steps(spec["max_updates"]),
        warm_start_adapter=parent,
    )
    return cfg

def verified_checkpoint(cfg):
    run = Path(cfg["run_dir"])
    state = json.loads((run / "state.json").read_text())
    require(state.get("status") == "completed", "stage_not_completed")
    completed = state.get("completed_optimizer_steps", state.get("step"))
    require(completed == cfg["max_updates"], "stage_update_count_mismatch")
    directory = run / "checkpoints" / ("%07d" % cfg["max_updates"])
    cp = json.loads((directory / "checkpoint.json").read_text())
    ev = json.loads((directory / "evaluation.json").read_text())
    require(ev.get("quality_gates_passed") is True, "stage_quality_gates_not_passed")
    require(ev.get("status") == "completed", "evaluation_not_completed")
    require(ev.get("test_opened") is False, "heldout_policy_not_proven")
    require(cp.get("completed_optimizer_steps") == cfg["max_updates"]
            and ev.get("completed_optimizer_steps") == cfg["max_updates"],
            "checkpoint_update_count_mismatch")
    require(cp.get("evaluation_sha256") == sha256(directory / "evaluation.json"),
            "checkpoint_evaluation_hash_mismatch")
    adapter_sha = sha256(directory / "adapters.safetensors")
    require(cp.get("adapter_sha256") == adapter_sha, "saved_adapter_hash_mismatch")
    require(ev.get("adapter_sha256") == adapter_sha,
            "evaluated_adapter_differs_from_saved_adapter")
    return {
        "checkpoint_dir": str(directory),
        "adapter_sha256": adapter_sha,
        "checkpoint_sha256": sha256(directory / "checkpoint.json"),
    }, state, ev


def finite_nonnegative(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "unknown_numeric_value_" + label)
    return float(value)

def numeric_score(metrics, policy):
    n, valid = metrics.get("n"), metrics.get("valid_count")
    require(type(n) is int and n == policy["development_examples"] and
            type(valid) is int and 0 <= valid <= n, "numeric_development_count_mismatch")
    rate = finite_nonnegative(metrics.get("valid_rate"), "valid_rate")
    require(rate <= 1 and abs(rate - valid / n) < 1e-12, "numeric_valid_count_rate_mismatch")
    invalid = finite_nonnegative(metrics.get("invalid_rate"), "invalid_rate")
    require(abs(invalid - (1 - rate)) < 1e-12, "numeric_invalid_rate_mismatch")
    mse = metrics.get("mse_on_valid_predictions")
    if valid:
        mse = finite_nonnegative(mse, "valid_mse")
        require(mse <= policy["invalid_output_penalty"], "numeric_mse_exceeds_registered_range")
    else:
        require(mse is None, "mse_present_without_valid_targets")
        mse = 0.0
    return rate * mse + (1 - rate) * policy["invalid_output_penalty"]

def evaluate_numeric_retention(plan, cfg, parent, evaluation, prior=None):
    policy = numeric_policy(plan)
    if policy is None:
        return None, prior
    run = Path(cfg["run_dir"])
    baseline_path = run / "baseline_task_evaluation.json"
    baseline_sha = sha256(baseline_path)
    require(evaluation.get("baseline_task_evaluation_sha256") == baseline_sha,
            "numeric_baseline_hash_mismatch")
    baseline = json.loads(baseline_path.read_text())
    require(baseline.get("completed_optimizer_steps") == 0 and baseline.get("test_opened") is False
            and baseline.get("task_kind") == "quantized_eeg_reconstruction", "unexpected_numeric_baseline_scope")
    require(baseline.get("data_manifest_sha256") == cfg["data_manifest"]["sha256"],
            "numeric_baseline_data_manifest_mismatch")
    data = json.loads(bound_file(cfg["data_manifest"], "numeric_stage_data").read_text())
    val = data["inputs"]["val"]
    bound_file(val, "numeric_stage_validation")
    indices = list(range(val["count"]))
    random.Random(cfg["seed"] + 2).shuffle(indices)
    indices = sorted(indices[:36])
    current_private = json.loads((Path(parent["checkpoint_dir"]) / "predictions.private.json").read_text())
    initial_private = json.loads((run / "baseline_task_predictions.private.json").read_text())
    require(baseline.get("selected_validation_indices") == indices
            and current_private.get("indices") == indices and initial_private.get("indices") == indices,
            "numeric_development_selection_mismatch")
    require(len(current_private.get("outputs", [])) == 36 and len(initial_private.get("outputs", [])) == 36,
            "numeric_development_output_count_mismatch")
    identity = {"validation_sha256": val["sha256"], "validation_records": val["count"],
                "selection_indices_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
                "seed": cfg["seed"], "n": 36}
    require(prior is None or prior.get("development_identity") == identity, "numeric_development_identity_changed")
    before, current = baseline["metrics"], evaluation["metrics"]
    initial_score, score = numeric_score(before, policy), numeric_score(current, policy)
    require(current["valid_rate"] >= .9, "numeric_stage_output_format_below_existing_gate")
    paired = current.get("initial_model_comparison", {})
    count, targets = paired.get("paired_valid_examples"), paired.get("paired_valid_target_values")
    require(paired.get("n") == 36 and type(count) is int and
            max(0, before["valid_count"] + current["valid_count"] - 36) <= count <= min(before["valid_count"], current["valid_count"])
            and targets == count * 4, "unknown_numeric_paired_population")
    require(paired.get("initial_valid_rate") == before["valid_rate"]
            and paired.get("current_valid_rate") == current["valid_rate"], "numeric_paired_rate_mismatch")
    delta, paired_tolerance = None, None
    if count:
        a = finite_nonnegative(paired.get("initial_mse_paired_valid"), "initial_paired_mse")
        b = finite_nonnegative(paired.get("current_mse_paired_valid"), "current_paired_mse")
        delta = b - a
        stated_delta = paired.get("mse_delta_current_minus_initial")
        require(type(stated_delta) in (int, float) and math.isfinite(stated_delta)
                and abs(stated_delta - delta) < 1e-9 and max(a, b) <= 40000,
                "numeric_paired_delta_mismatch")
        paired_tolerance = max(policy["relative_tolerance"] * a, 1 / targets)
    else:
        require(all(paired.get(k) is None for k in ("initial_mse_paired_valid", "current_mse_paired_valid", "mse_delta_current_minus_initial")),
                "numeric_paired_mse_without_population")
    previous = dict(prior) if prior is not None else {
        "development_identity": identity, "initial_penalized_score": initial_score,
        "best_penalized_score": initial_score, "best_checkpoint": cfg.get("warm_start_adapter"),
        "best_reference": "first_stage_initial_model", "stagnant_stages": 0, "evaluated_stages": 0}
    best = finite_nonnegative(previous["best_penalized_score"], "previous_best_score")
    tolerance = max(policy["relative_tolerance"] * best, 1 / (36 * 4))
    improved = score < best
    stagnant = 0 if improved else previous["stagnant_stages"] + 1
    reasons = []
    if score > best + tolerance:
        reasons.append("penalized_score_regressed_from_best")
    paired_evaluated = count >= policy["min_paired_examples"]
    if paired_evaluated and delta > paired_tolerance:
        reasons.append("paired_numeric_mse_regressed_from_stage_start")
    if stagnant >= policy["stagnation_patience"]:
        reasons.append("two_stages_without_new_best")
    passed = not reasons
    proposed = dict(previous, stagnant_stages=stagnant, evaluated_stages=previous["evaluated_stages"] + 1)
    if passed and improved:
        proposed.update(best_penalized_score=score, best_checkpoint=parent, best_reference="verified_stage_checkpoint")
    receipt = {"schema": "evlm-numeric-retention-v1", "passed": passed,
        "status": "passed" if passed else "stopped", "reasons": reasons, "policy": policy,
        "development_identity": identity, "baseline_task_evaluation_sha256": baseline_sha,
        "evaluation_sha256": sha256(Path(parent["checkpoint_dir"]) / "evaluation.json"),
        "initial_model_penalized_score": initial_score, "current_penalized_score": score,
        "score_name": "all-example score with 40000 invalid-output penalty; not ordinary MSE",
        "best_penalized_score_before": best, "best_penalized_score_after": proposed["best_penalized_score"],
        "best_score_tolerance": tolerance, "new_best": passed and improved,
        "stagnant_stages": stagnant, "paired_gate_evaluated": paired_evaluated,
        "paired_numeric_improvement_observed": paired_evaluated and delta < 0,
        "paired_valid_examples": count, "paired_mse_delta": delta, "paired_mse_tolerance": paired_tolerance,
        "ordinary_mse_on_valid_predictions": current["mse_on_valid_predictions"],
        "valid_rate": current["valid_rate"],
        "limitation": "fixed development 36 examples / 2 source groups; budget stop rule, not generalization evidence",
        "test_opened": False, "candidate_efficacy_demonstrated": False}
    return receipt, proposed


def bounded_command(command, log, timeout, stop_path, on_pid):
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=log, start_new_session=True)
    on_pid(child.pid)
    until = time.monotonic() + timeout
    try:
        while child.poll() is None:
            require(not stop_path.exists(), "operator_stop_requested")
            require(time.monotonic() < until, "vision_gate_wall_budget_exceeded")
            time.sleep(1)
        return child.returncode
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        on_pid(None)

def verify_vision_receipt(bundle, parent):
    bundle = Path(bundle).resolve()
    status = json.loads((bundle / "candidate_status.json").read_text())
    require(status.get("status") == "synthetic_smoke_passed", "vision_smoke_not_passed")
    receipt_path = (bundle / status["receipt"]).resolve()
    require(receipt_path.parent == bundle, "unsafe_vision_receipt_path")
    require(sha256(receipt_path) == status.get("receipt_sha256"), "vision_receipt_hash_mismatch")
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("status") == "passed" and receipt.get("synthetic_only") is True,
            "unexpected_vision_evaluation_scope")
    require(receipt.get("cases_per_mode") == 2 and receipt.get("base_correct") == 2
            and receipt.get("adapter_correct") == 2, "vision_control_or_adapter_failed")
    require(receipt.get("test_opened") is False and receipt.get("executed_actions") == 0,
            "vision_gate_must_remain_synthetic_and_dry_run")
    require(receipt.get("manifest_sha256") == sha256(bundle / "manifest.json"),
            "vision_bundle_manifest_changed")
    require(sha256(bundle / "adapter/adapters.safetensors") == parent["adapter_sha256"],
            "vision_gate_did_not_use_new_checkpoint")
    return {"bundle": str(bundle), "receipt": str(receipt_path),
            "receipt_sha256": sha256(receipt_path), "base_correct": 2,
            "adapter_correct": 2, "synthetic_only": True,
            "eeg_effectiveness_claim": False}

def run_vision_gate(plan, spec, cfg, parent, deadline, on_pid):
    gate = plan["vision_gate"]
    runtime = bound_file(gate["runtime"], "vlm_runtime")
    bundle = Path(plan["run_root"]) / "bundles" / spec["stage_id"]
    require(not bundle.exists(), "vision_bundle_namespace_exists")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(plan["run_root"]) / "logs" / (spec["stage_id"] + ".vision.log")
    commands = [
        [gate["python_path"], str(runtime), "prepare", "--base", cfg["model_path"],
         "--adapter", parent["checkpoint_dir"], "--output", str(bundle),
         "--expected-adapter-sha256", parent["adapter_sha256"],
         "--base-revision", gate["base_revision"]],
        [gate["python_path"], str(runtime), "smoke", "--bundle", str(bundle),
         "--gpu-lock", cfg["lock_path"], "--memory-gb", "54", "--max-tokens", "128"],
    ]
    until = min(deadline, time.monotonic() + gate["max_wall_seconds"])
    with log_path.open("ab") as log:
        for command in commands:
            require(time.monotonic() < until, "vision_gate_wall_budget_exceeded")
            rc = bounded_command(command, log, until-time.monotonic(),
                                 Path(plan["run_root"]) / "STOP_CURRICULUM", on_pid)
            require(rc == 0, "vision_gate_process_failed")
    return verify_vision_receipt(bundle, parent)

def run(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read_plan(plan_path)
    root = Path(plan["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "curriculum_state.json"
    lock = (root / "curriculum.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not state_path.exists(), "existing_curriculum_requires_explicit_new_continuation_plan")
    started = time.monotonic()
    deadline = started + plan["max_controller_wall_seconds"]
    state = {
        "schema": SCHEMA, "curriculum_id": plan["curriculum_id"],
        "status": "starting", "started_at": utc_now(), "updated_at": utc_now(),
        "controller_pid": os.getpid(), "plan_sha256": sha256(plan_path),
        "completed_stages": 0, "completed_optimizer_updates": 0,
        "planned_train_presentations": planned_updates(plan),
        "planned_optimizer_updates_this_run": planned_updates(plan),
        "total_remaining_source_presentations": remaining_presentations(plan),
        "full_queue_train_records": sum(x["train_records"] for x in plan["sources"]),
        "numeric_retention_policy": numeric_policy(plan),
        "unique_eeg_hours_claimed": False, "test_opened": False,
        "optimizer_continuity": "fresh_Adam_per_stage; adapter_warm_start",
        "exact_resume": False, "latest_candidate": plan.get("initial_adapter"),
        "candidate_efficacy_demonstrated": False,
    }
    atomic(root / "plan.json", plan)
    atomic(state_path, state)
    child = None
    def stop_signal(signum, frame):
        raise RuntimeError("controller_received_signal_" + str(signum))
    old_handlers = {sig: signal.signal(sig, stop_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        parent = plan.get("initial_adapter")
        for spec in stage_specs(plan):
            require(not (root / "STOP_CURRICULUM").exists(), "operator_stop_requested")
            require(time.monotonic() < deadline, "controller_wall_budget_exceeded")
            cfg = make_stage_config(plan, spec, parent)
            # Recheck pinned source and runner before a new stage. A changed
            # acquisition queue cannot rewrite the current immutable plan.
            bound_file(plan["runner"], "runner")
            bound_file(spec["data_manifest"], "data_manifest")
            cfg_path = root / "configs" / (spec["stage_id"] + ".json")
            atomic(cfg_path, cfg)
            state.update(status="preflight", current_stage=spec, child_pid=None,
                         updated_at=utc_now())
            atomic(state_path, state)
            cmd = [plan["python_path"], plan["runner"]["path"]]
            stage_log = root / "logs" / (spec["stage_id"] + ".log")
            stage_log.parent.mkdir(exist_ok=True)
            with stage_log.open("ab") as log:
                receipt = root / "preflight" / (spec["stage_id"] + ".json")
                receipt.parent.mkdir(exist_ok=True)
                pre = subprocess.run(cmd + ["preflight", "--config", str(cfg_path),
                                     "--receipt", str(receipt)],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     timeout=min(600, max(1, deadline-time.monotonic())))
                require(pre.returncode == 0 and receipt.is_file(), "stage_preflight_failed")
                child = subprocess.Popen(cmd + ["run", "--config", str(cfg_path)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                state.update(status="training", child_pid=child.pid, updated_at=utc_now())
                atomic(state_path, state)
                while child.poll() is None:
                    if (root / "STOP_CURRICULUM").exists() or time.monotonic() >= deadline:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait(timeout=10)
                        raise RuntimeError("operator_stop_or_controller_budget")
                    time.sleep(5)
                    state.update(updated_at=utc_now())
                    atomic(state_path, state)
                require(child.returncode == 0, "stage_process_failed")
                child = None
            parent, child_state, evaluation = verified_checkpoint(cfg)
            # Training already happened even if a subsequent retention gate stops.
            state.update(completed_optimizer_updates=state["completed_optimizer_updates"]+cfg["max_updates"],
                         last_trained_checkpoint=parent, last_completed_training_stage=spec,
                         updated_at=utc_now())
            atomic(state_path, state)
            numeric = None
            numeric_next = state.get("numeric_retention")
            if numeric_policy(plan) is not None:
                numeric_path = root / "numeric_retention" / (spec["stage_id"] + ".json")
                try:
                    numeric, numeric_next = evaluate_numeric_retention(
                        plan, cfg, parent, evaluation, state.get("numeric_retention"))
                except Exception as exc:
                    atomic(numeric_path, {"status": "failed_validation", "stage": spec,
                        "checkpoint": parent, "error_type": type(exc).__name__, "error": str(exc)[:500],
                        "test_opened": False, "next_stage_started": False})
                    raise
                atomic(numeric_path, {**numeric, "stage": spec, "checkpoint": parent})
                state.update(last_numeric_retention_receipt=str(numeric_path),
                             last_numeric_retention_sha256=sha256(numeric_path))
                if not numeric["passed"]:
                    state.update(status="stopped_numeric_gate", ended_at=utc_now(), child_pid=None,
                                 numeric_retention=numeric_next, numeric_stop_reasons=numeric["reasons"],
                                 next_action="review_preserved_best_and_metrics_before_any_new_plan")
                    atomic(state_path, state)
                    return 2
            def gate_pid(pid):
                state.update(status="vision_validation", child_pid=pid, updated_at=utc_now())
                atomic(state_path, state)
            vision = run_vision_gate(plan, spec, cfg, parent, deadline, gate_pid)
            receipt = {
                "schema": SCHEMA, "stage": spec, "config_sha256": sha256(cfg_path),
                "checkpoint": parent, "vision_retention": vision, "completed_at": utc_now(),
                "numeric_retention": numeric,
                "completed_optimizer_updates": cfg["max_updates"],
                "quality_gates_passed": True,
                "test_opened": False, "candidate_efficacy_demonstrated": False,
                "evaluation_sha256": sha256(Path(parent["checkpoint_dir"]) / "evaluation.json"),
            }
            atomic(root / "completed" / (spec["stage_id"] + ".json"), receipt)
            state.update(completed_stages=state["completed_stages"]+1,
                         latest_candidate=parent, latest_vlm_bundle=vision["bundle"], child_pid=None, updated_at=utc_now())
            if numeric is not None:
                state["numeric_retention"] = numeric_next
            atomic(state_path, state)
        paused = state["completed_optimizer_updates"] < remaining_presentations(plan)
        state.update(status="paused_update_budget" if paused else "completed", ended_at=utc_now(),
                     remaining_unpresented_records=remaining_presentations(plan) - state["completed_optimizer_updates"],
                     next_action="create_explicit_continuation_plan_from_completed_offsets" if paused else
                                 "evaluate_saved_candidate_and_admit_next_verified_source")
        atomic(state_path, state)
        return 0
    except Exception as exc:
        if child is not None and child.poll() is None:
            # This is solely our recorded child/process group, never a broad kill.
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        state.update(status="failed", ended_at=utc_now(), error_type=type(exc).__name__,
                     error=str(exc)[:500], next_action="review_receipt_before_new_continuation_plan")
        atomic(state_path, state)
        raise
    finally:
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)
        lock.close()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for action in ("validate", "run"):
        p = sub.add_parser(action)
        p.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        plan = read_plan(args.plan)
        specs = list(stage_specs(plan))
        print(json.dumps({
            "schema": SCHEMA, "status": "plan_valid",
            "stages": len(specs), "presentations": sum(x["max_updates"] for x in specs),
            "planned_optimizer_updates_this_run": planned_updates(plan),
            "total_remaining_source_presentations": remaining_presentations(plan),
            "full_queue_train_records": sum(x["train_records"] for x in plan["sources"]),
            "sources": len(plan["sources"]), "test_opened": False,
            "scope": plan["stage_template"]["num_layers"],
        }))
        return 0
    return run(args.plan)

if __name__ == "__main__":
    sys.exit(main())
