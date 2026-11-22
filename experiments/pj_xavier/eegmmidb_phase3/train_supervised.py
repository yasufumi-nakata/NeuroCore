"""Bounded supervised adaptation of the selected phase3 SSL encoder.

This module never evaluates test waveforms. The held-out test metadata are sealed
for the independent once-only evaluator; model selection uses validation only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.nn import functional as F

from eegmmidb_phase2.data_contract import (
    EEGStore, TASKS, load_corpus, object_hash, partition_metrics, require, sha256,
    source_interval_union, write_json,
)
from eegmmidb_phase3.model import (
    ARCHITECTURE, BACKBONE_ID, BACKBONE_REVISION, CLASS_TOKEN_IDS,
    FusedDecoder, ScratchDecoder, learned_state, load_backbone,
    restore_learned, static_prompt, trainable_parameter_count,
)
from eegmmidb_phase3.runtime import (
    CHECKPOINT_SCHEMA, apply_budget, dtype_value, hardware, memory_sample,
    select_device, source_seals, synchronize, within_budget, verify_ssl_origin,
)

CONTRACT_SCHEMA = "xavier-phase3-training-execution-contract-v1"
PLAN_SCHEMA = "xavier-phase3-neural-training-plan-v1"


class BoundedStop(RuntimeError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text())


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def rng_state():
    numpy = np.random.get_state()
    result = {
        "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]],
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(value):
    random.setstate(value["python"])
    numpy = value["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], np.uint32), numpy[2], numpy[3], numpy[4]))
    torch.set_rng_state(value["torch"])
    if "cuda" in value:
        require(torch.cuda.is_available(), "Cannot restore CUDA RNG on a different backend")
        torch.cuda.set_rng_state_all(value["cuda"])
    if "mps" in value:
        require(torch.backends.mps.is_available(), "Cannot restore MPS RNG on a different backend")
        torch.mps.set_rng_state(value["mps"])


def clone_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_cpu(item) for item in value)
    return copy.deepcopy(value)


def save_checkpoint(path, model, optimizer, state):
    value = dict(state)
    value.update(
        schema=CHECKPOINT_SCHEMA, learned=clone_cpu(learned_state(model, state["kind"])),
        optimizer=clone_cpu(optimizer.state_dict()), rng=clone_cpu(rng_state()),
    )
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)
    return sha256(path)


def epoch_order(length, seed, epoch):
    order = list(range(length))
    random.Random(seed + epoch).shuffle(order)
    return order


def batch_rows(rows, store, device):
    require(rows and all(row["split"] in ("train", "val") for row in rows), "Test waveform access is prohibited in trainer")
    eeg = torch.from_numpy(np.stack([store.get(row) for row in rows])).to(device)
    families = [row["task_family"] for row in rows]
    labels = torch.tensor([TASKS[row["task_family"]].index(row["label"]) for row in rows], device=device)
    return eeg, families, labels


def forward_logits(model, kind, eeg, families, tokenizer):
    if kind == "scratch":
        return model(eeg, families, "fused")
    prompts = [tokenizer.encode(static_prompt(task, "fused"), add_special_tokens=True) for task in families]
    return model(eeg, prompts, "fused")


def check_time(start, seconds, absolute_deadline):
    if time.monotonic() - start >= seconds:
        raise BoundedStop("stage_time_budget")
    if absolute_deadline is not None and time.time() >= absolute_deadline:
        raise BoundedStop("absolute_deadline")


def verify_plan(plan):
    require(plan.get("schema") == PLAN_SCHEMA, "Unknown training plan")
    require(plan["seed"] == 20261122 and plan["acceptance_threshold"] == 0.8, "Registered seed/threshold changed")
    require(plan["architecture"] == ARCHITECTURE, "Architecture changed")
    require(plan["backbone"] == {"id": BACKBONE_ID, "revision": BACKBONE_REVISION, "frozen": True}, "Frozen LLM identity changed")
    require(plan["selection"] == {
        "partition": "val", "task": "left_right", "metric": "mean_subject_balanced_accuracy",
        "missing_class_subjects": "omit_and_report", "zero_valid_subjects": "blocked",
        "ties": "earliest_checkpoint", "test_used_for_tuning": False,
    }, "Model selection changed")
    from eegmmidb_phase3.common import SCIENTIFIC_PROTOCOL_SHA
    require(plan["scientific_protocol_sha256"] == SCIENTIFIC_PROTOCOL_SHA, "Phase3 scientific design changed")
    for kind, epochs, lr, candidates in (("scratch", 60, .001, [64,128,256]), ("fused",30,.0003,[8,16,32,64,128])):
        stage = plan["stages"][kind]
        require(stage["max_epochs"] == epochs and stage["learning_rate"] == lr and stage["weight_decay"] == .0001, "Registered optimization changed")
        require(stage["capacity_batch_candidates"] == candidates, "Unregistered batch search")
    require(plan["gradient_clip_norm"] == 1 and plan["capacity_probe"]["optimizer_steps"] == 50, "Gradient/probe plan changed")
    require(plan["memory_budget_gib"] <= 21, "Memory ceiling changed")
    require(plan["stages"]["scratch"]["max_seconds"] == 900 and plan["stages"]["fused"]["max_seconds"] == 1800, "Stage budget exceeded")
    return plan


def ssl_origin_for_rows(ssl_run, rows):
    test = [row for row in rows if row["split"] == "test"]
    return verify_ssl_origin(
        ssl_run, sorted({r["subject_group"] for r in test}),
        sorted({r["source_eeg_sha256"] for r in test}),
        sorted({r["eeg_payload_sha256"] for r in test}),
    )


def verify_head_equivalence(path):
    from eegmmidb_phase3.compare_symbol_head import validate_combined_receipt
    return validate_combined_receipt(path)


def register(prepared, protocol, output, ssl_run, head_equivalence, budget_path):
    prepared, protocol, output = Path(prepared), Path(protocol), Path(output)
    require(not output.exists(), "Execution contract already exists")
    plan = verify_plan(read_json(protocol))
    rows, _, data_audit = load_corpus(prepared / "manifest.jsonl", prepared / "artifacts.json")
    origin = ssl_origin_for_rows(ssl_run, rows)
    numeric = verify_head_equivalence(head_equivalence)
    budget = read_json(budget_path)
    require(plan["stages"]["scratch"]["max_seconds"] == budget["phase_limits_seconds"]["MI_supervised_training_and_validation"], "MI budget differs")
    require(plan["stages"]["fused"]["max_seconds"] == budget["phase_limits_seconds"]["frozen_LLM_alignment_and_validation"], "Fused budget differs")
    metadata = {}
    for name in ("metadata_protocol.json", "preprocessing_protocol.json", "subject_split.json", "corpus_summary.json"):
        target = prepared / name
        require(target.is_file(), "Missing preparation proof: " + name)
        metadata[name] = sha256(target)
    contract = {
        "schema": CONTRACT_SCHEMA,
        "registered_epoch": time.time(),
        "ssl_encoder_origin": origin,
        "budget_sha256": sha256(budget_path),
        "budget": budget,
        "head_equivalence_sha256": sha256(head_equivalence),
        "optimized_symbol_head": numeric["optimized_symbol_head_admitted"],
        "protocol_sha256": sha256(protocol),
        "protocol": plan,
        "scientific_source_sha256": source_seals(),
        "prepared_manifest_sha256": data_audit["manifest_sha256"],
        "artifacts_sha256": data_audit["artifacts_sha256"],
        "preparation_proofs_sha256": metadata,
        "data_audit": data_audit,
        "test_forward_calls": 0,
        "signal_statistics_used_to_set_model_hyperparameters": False,
    }
    write_json(output, contract)
    print(json.dumps({"status": "registered", "contract_sha256": sha256(output), "data_audit": data_audit}))
    return contract


def load_execution(contract_path, prepared, ssl_run):
    contract_path, prepared = Path(contract_path), Path(prepared)
    contract = read_json(contract_path)
    require(contract.get("schema") == CONTRACT_SCHEMA, "Unknown execution contract")
    verify_plan(contract["protocol"])
    require(contract["scientific_source_sha256"] == source_seals(), "Scientific source changed after registration")
    rows, artifacts, audit = load_corpus(prepared / "manifest.jsonl", prepared / "artifacts.json")
    require(audit["manifest_sha256"] == contract["prepared_manifest_sha256"], "Prepared manifest changed")
    require(audit["artifacts_sha256"] == contract["artifacts_sha256"], "Prepared array seals changed")
    for name, digest in contract["preparation_proofs_sha256"].items():
        require(sha256(prepared / name) == digest, "Preparation proof changed: " + name)
    require(ssl_origin_for_rows(ssl_run, rows) == contract["ssl_encoder_origin"], "Selected SSL origin changed")
    return contract, rows, artifacts, audit


def membership(rows, contract):
    parts = {part: sorted(row["id"] for row in rows if row["split"] == part) for part in ("train", "val", "test")}
    return {
        "schema": "xavier-phase3-training-membership-v1",
        "manifest_sha256": contract["prepared_manifest_sha256"],
        "artifacts_sha256": contract["artifacts_sha256"],
        "training_ids": parts["train"], "validation_ids": parts["val"], "test_ids": parts["test"],
        "split_ids_sha256": object_hash(parts),
        "subject_groups": {
            part: sorted({row["subject_group"] for row in rows if row["split"] == part})
            for part in parts
        },
        "raw_source_sha256": {
            part: sorted({row["source_eeg_sha256"] for row in rows if row["split"] == part})
            for part in parts
        },
        "payload_sha256": {
            part: sorted({row["eeg_payload_sha256"] for row in rows if row["split"] == part})
            for part in parts
        },
        "model_selection_partition": "val", "test_used_for_tuning": False,
    }


def scratch_identity(scratch_run, contract):
    root = Path(scratch_run)
    summary, config = read_json(root / "run_summary.json"), read_json(root / "run_config.json")
    digest = sha256(root / "best.pt")
    require(digest == summary["best_checkpoint_sha256"], "Parent scratch best identity changed")
    require(config["kind"] == "scratch", "Encoder parent must be scratch model")
    require(config["execution_contract_sha256"] == contract, "Parent was trained against a different data/plan contract")
    state = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
    require(state["schema"] == CHECKPOINT_SCHEMA and state["source_sha256"] == source_seals(), "Parent scratch source identity differs")
    require(state["config"] == config, "Parent scratch config differs")
    require(state["training_membership_sha256"] == sha256(root / "training_membership.json"), "Parent membership differs")
    return digest, state


def build_model(kind, device, dtype, local_files_only, scratch_state=None, *, ssl_run, optimized_symbol_head):
    if kind == "scratch":
        model, tokenizer, evidence = ScratchDecoder().to(device), None, {}
        selected = torch.load(Path(ssl_run) / "selected_encoder.pt", map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(selected, strict=True)
    else:
        require(scratch_state is not None, "Fused stage requires validation-selected scratch encoder")
        base, tokenizer, evidence = load_backbone(device, dtype, local_files_only=local_files_only)
        model = FusedDecoder(base, optimized_symbol_head=optimized_symbol_head)
        model.encoder.load_state_dict(scratch_state["learned"]["encoder"], strict=True)
        model.encoder.to(device)
        model.projector.to(device)
    return model, tokenizer, evidence


def configuration(kind, batch_size, contract, contract_sha, parent_sha, device):
    plan = contract["protocol"]
    stage = plan["stages"][kind]
    return {
        "schema": "xavier-phase3-scientific-config-v1",
        "kind": kind,
        "ssl_encoder_origin": contract["ssl_encoder_origin"],
        "optimized_symbol_head": contract["optimized_symbol_head"],
        "head_equivalence_sha256": contract["head_equivalence_sha256"],
        "architecture": ARCHITECTURE,
        "task_mapping": {key: list(value) for key, value in TASKS.items()},
        "execution_contract_sha256": contract_sha,
        "protocol_sha256": contract["protocol_sha256"],
        "manifest_sha256": contract["prepared_manifest_sha256"],
        "artifacts_sha256": contract["artifacts_sha256"],
        "source_sha256": source_seals(),
        "seed": plan["seed"], "batch_size": batch_size,
        "validation_batch_size": min(batch_size, stage["validation_batch_ceiling"]),
        "learning_rate": stage["learning_rate"], "weight_decay": stage["weight_decay"],
        "max_epochs": stage["max_epochs"], "max_seconds": stage["max_seconds"],
        "optimizer": "AdamW", "gradient_clip_norm": plan["gradient_clip_norm"],
        "class_weights": None, "learning_rate_schedule": None,
        "selection": plan["selection"], "acceptance_threshold": plan["acceptance_threshold"],
        "memory_budget_gib": plan["memory_budget_gib"],
        "training_device": str(device),
        "backbone_dtype": "bfloat16" if device.type != "cpu" else "float32",
        "class_token_ids": list(CLASS_TOKEN_IDS),
        "backbone": plan["backbone"] if kind == "fused" else None,
        "scratch_parent_best_sha256": parent_sha,
        "encoder_trainable": True, "projector_trainable": kind == "fused",
        "frozen_backbone": kind == "fused",
        "classification_loss": "cross_entropy_of_same_two_next_symbol_logits" if kind == "fused" else "cross_entropy_task_specific_binary_head",
    }


def select_capacity(receipts, kind, contract_sha, plan):
    reports = [read_json(path) for path in receipts]
    require(bool(reports), "No measured capacity probes")
    candidates = plan["stages"][kind]["capacity_batch_candidates"]
    by_batch = {}
    for path, report in zip(receipts, reports, strict=True):
        require(report["kind"] == kind and report["execution_contract_sha256"] == contract_sha, "Capacity identity differs")
        require(report["source_sha256"] == source_seals(), "Capacity implementation differs")
        batch = report["batch_size"]
        require(batch in candidates and batch not in by_batch, "Unregistered or duplicate batch probe")
        by_batch[batch] = report
        report["_receipt_sha256"] = sha256(path)
    stable = [r for r in reports if r["status"] == "passed" and r["completed_optimizer_steps"] == plan["capacity_probe"]["optimizer_steps"]]
    require(stable, "No stable measured capacity")
    chosen = max(stable, key=lambda r: r["batch_size"])
    return {
        "batch_size": chosen["batch_size"], "rule": "largest_completed_registered_train_only_probe",
        "tested_batch_sizes": sorted(by_batch), "untested_batch_sizes": sorted(set(candidates) - set(by_batch)),
        "receipts": [{"batch_size": r["batch_size"], "status": r["status"], "sha256": r["_receipt_sha256"]} for r in reports],
        "selected_receipt_sha256": chosen["_receipt_sha256"],
        "does_not_guarantee_no_later_fragmentation": True,
    }


def probe(args):
    started = time.monotonic()
    contract, rows, _, data_audit = load_execution(args.contract, args.prepared, args.ssl_run)
    contract_sha, plan = sha256(args.contract), contract["protocol"]
    require(args.absolute_deadline_epoch is not None and args.absolute_deadline_epoch <= contract["budget"]["absolute_deadline_epoch"], "Training deadline exceeds registration")
    require(args.batch_size in plan["stages"][args.kind]["capacity_batch_candidates"], "Batch candidate not registered")
    output = Path(args.run_dir)
    require(not output.exists(), "Probe output exists")
    output.mkdir(parents=True)
    device = select_device(args.device)
    torch.set_num_threads(2)
    allocator = apply_budget(device, plan["memory_budget_gib"])
    train = sorted((r for r in rows if r["split"] == "train"), key=lambda r:r["id"])
    parent_sha, parent = (None, None)
    if args.kind == "fused":
        parent_sha, parent = scratch_identity(args.scratch_run, contract_sha)
    seed_everything(plan["seed"])
    model, tokenizer, backbone = build_model(args.kind, device, dtype_value("bfloat16" if device.type != "cpu" else "float32", device), args.local_files_only, parent, ssl_run=args.ssl_run, optimized_symbol_head=contract["optimized_symbol_head"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=plan["stages"][args.kind]["learning_rate"],
        weight_decay=plan["stages"][args.kind]["weight_decay"],
    )
    store, steps, seen, samples, losses = EEGStore(), 0, set(), [], []
    order = epoch_order(len(train), plan["seed"], 0)
    status, reason = "passed", None
    model.train()
    try:
        for step in range(plan["capacity_probe"]["optimizer_steps"]):
            check_time(started, plan["capacity_probe"]["per_candidate_max_seconds"], args.absolute_deadline_epoch)
            chosen = [train[order[(step * args.batch_size + i) % len(train)]] for i in range(args.batch_size)]
            eeg, families, labels = batch_rows(chosen, store, device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_logits(model, args.kind, eeg, families, tokenizer)
            loss = F.cross_entropy(logits, labels)
            require(bool(torch.isfinite(loss)), "Nonfinite probe loss")
            loss.backward()
            require(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), "Nonfinite probe gradients")
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], plan["gradient_clip_norm"])
            optimizer.step()
            synchronize(device)
            steps += 1
            seen.update(r["id"] for r in chosen)
            losses.append(float(loss.detach()))
            sample = memory_sample(device)
            samples.append(sample)
            if not within_budget(sample, plan["memory_budget_gib"], device):
                raise BoundedStop("memory_budget")
    except (BoundedStop, RuntimeError, ValueError) as exc:
        status, reason = "failed", str(exc)
    report = {
        "schema": "xavier-phase3-train-only-capacity-v1",
        "status": status, "failure_reason": reason, "kind": args.kind,
        "execution_contract_sha256": contract_sha, "source_sha256": source_seals(),
        "batch_size": args.batch_size, "completed_optimizer_steps": steps,
        "examples_presented": steps * args.batch_size,
        "unique_training_ids": sorted(seen), "unique_training_ids_sha256": object_hash(sorted(seen)),
        "validation_forward_calls": 0, "test_forward_calls": 0,
        "probe_parameters_discarded": True, "main_training_starts_from_fixed_seed_fresh_state": True,
        "scratch_parent_best_sha256": parent_sha, "data_audit": data_audit,
        "hardware": hardware(device), "allocator": allocator,
        "memory_observations": samples, "elapsed_seconds": time.monotonic() - started,
        "last_training_loss": losses[-1] if losses else None,
        "backbone": backbone, "trainable_parameters": trainable_parameter_count(model),
        "capacity_scope": "50 sustained optimizer steps on actual training rows; not a guarantee for the entire later run",
    }
    write_json(output / "capacity.json", report)
    print(json.dumps({key: report[key] for key in ("status", "kind", "batch_size", "completed_optimizer_steps", "elapsed_seconds", "failure_reason")}))
    return report


@torch.inference_mode()
def validate(model, kind, rows, store, tokenizer, device, batch_size, started, max_seconds, deadline, memory_budget):
    require(rows and all(row["split"] == "val" for row in rows), "Validation must be isolated from test")
    model.eval()
    probabilities, predictions = [], []
    for begin in range(0, len(rows), batch_size):
        check_time(started, max_seconds, deadline)
        batch = rows[begin:begin+batch_size]
        eeg, families, _ = batch_rows(batch, store, device)
        values = torch.softmax(forward_logits(model, kind, eeg, families, tokenizer).float(), -1).cpu().tolist()
        sample = memory_sample(device)
        if not within_budget(sample, memory_budget, device):
            raise BoundedStop("memory_budget_in_validation")
        probabilities.extend(values)
        for row, probability in zip(batch, values, strict=True):
            labels = TASKS[row["task_family"]]
            predictions.append({
                "id": row["id"], "split": "val", "subject_group": row["subject_group"],
                "task_family": row["task_family"], "true_label": row["label"],
                "probabilities": dict(zip(labels, probability, strict=True)),
                "predicted_label": labels[int(np.argmax(probability))],
            })
    metrics = partition_metrics(rows, probabilities)
    require(metrics["selection_score"] is not None, "No validation subject contains both primary classes")
    return metrics, predictions


def update_best(run_dir, checkpoint, score, current):
    if current is None or score > current:
        destination = Path(run_dir) / "best.pt"
        temp = destination.with_suffix(".tmp")
        shutil.copyfile(checkpoint, temp)
        temp.replace(destination)
        return score, sha256(destination), True
    return current, sha256(Path(run_dir) / "best.pt"), False


def training_state(config, membership_sha, epoch, next_position, step, best_score, best_sha,
                   seen, presentations, elapsed, epoch_completed, *, resumable=True):
    return {
        "schema": CHECKPOINT_SCHEMA, "kind": config["kind"], "config": config,
        "source_sha256": source_seals(), "training_membership_sha256": membership_sha,
        "epoch": epoch, "next_position": next_position, "global_step": step,
        "epoch_completed": epoch_completed,
        "best_score": best_score, "best_checkpoint_sha256_before_this_save": best_sha,
        "seen_train_ids": sorted(seen), "training_examples_presented": presentations,
        "elapsed_stage_seconds": elapsed, "resumable": resumable,
    }


def restore_training(checkpoint, model, optimizer, config, membership_sha):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(state["schema"] == CHECKPOINT_SCHEMA and state["config"] == config, "Resume scientific identity differs")
    require(state["source_sha256"] == source_seals() and state["training_membership_sha256"] == membership_sha, "Resume provenance differs")
    require(state.get("resumable") is True, "This diagnostic checkpoint is not a consistent restart point")
    restore_learned(model, config["kind"], state["learned"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return state


def train(args):
    started = time.monotonic()
    contract, rows, _, data_audit = load_execution(args.contract, args.prepared, args.ssl_run)
    contract_sha, plan = sha256(args.contract), contract["protocol"]
    require(args.absolute_deadline_epoch is not None and args.absolute_deadline_epoch <= contract["budget"]["absolute_deadline_epoch"], "Training deadline exceeds registration")
    device = select_device(args.device)
    torch.set_num_threads(2)
    allocator = apply_budget(device, plan["memory_budget_gib"])
    chosen = select_capacity(args.capacity_reports, args.kind, contract_sha, plan)
    parent_sha, parent = (None, None)
    if args.kind == "fused":
        parent_sha, parent = scratch_identity(args.scratch_run, contract_sha)
    config = configuration(args.kind, chosen["batch_size"], contract, contract_sha, parent_sha, device)
    output = Path(args.run_dir)
    member = membership(rows, contract)
    if args.resume:
        require(output.is_dir() and read_json(output / "run_config.json") == config, "Resume must keep original immutable scientific config")
        require(read_json(output / "training_membership.json") == member, "Resume membership differs")
    else:
        require(not output.exists(), "Run output exists; resume is explicit")
        output.mkdir(parents=True)
        write_json(output / "run_config.json", config)
        write_json(output / "training_membership.json", member)
        write_json(output / "capacity_selection.json", chosen)
        write_json(output / "execution_contract.json", contract)
    membership_sha = sha256(output / "training_membership.json")
    attempt = len(list(output.glob("attempt_*.json"))) + 1
    write_json(output / f"attempt_{attempt:03d}.json", {
        "started_epoch": time.time(), "arguments": vars(args),
        "scientific_config_sha256": sha256(output / "run_config.json"),
        "execution_contract_sha256": contract_sha,
    })
    seed_everything(plan["seed"])
    model, tokenizer, backbone = build_model(
        args.kind, device, dtype_value(config["backbone_dtype"], device),
        args.local_files_only, parent, ssl_run=args.ssl_run,
        optimized_symbol_head=contract["optimized_symbol_head"],
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    train_rows = sorted((r for r in rows if r["split"] == "train"), key=lambda r:r["id"])
    val_rows = sorted((r for r in rows if r["split"] == "val"), key=lambda r:r["id"])
    seen, presentations, step = set(), 0, 0
    epoch, cursor, elapsed_before, best_score, best_sha = 0, 0, 0.0, None, None
    completed_epochs = 0
    if args.resume:
        state = restore_training(args.resume, model, optimizer, config, membership_sha)
        epoch, cursor = state["epoch"], state["next_position"]
        seen, presentations, step = set(state["seen_train_ids"]), state["training_examples_presented"], state["global_step"]
        elapsed_before, best_score = state["elapsed_stage_seconds"], state["best_score"]
        if (output / "best.pt").exists():
            best_sha = sha256(output / "best.pt")
        if state["epoch_completed"]:
            epoch, cursor = epoch + 1, 0
        completed_epochs = epoch
    write_json(output / f"hardware_attempt_{attempt:03d}.json", {
        "hardware": hardware(device), "allocator": allocator, "backbone": backbone,
        "trainable_parameters": trainable_parameter_count(model),
        "all_LLM_parameters_frozen": args.kind != "fused" or all(not p.requires_grad for p in model.base.parameters()),
    })
    store = EEGStore()
    stop, consistent, last_checkpoint = None, True, None
    history_path = output / "history.jsonl"
    current_epoch_completed = False
    epoch_loss_sum, epoch_examples = 0.0, 0
    epoch_start_step = step
    max_seconds_remaining = max(0.0, config["max_seconds"] - elapsed_before)
    try:
        while epoch < config["max_epochs"]:
            order = epoch_order(len(train_rows), plan["seed"], epoch)
            require(0 <= cursor <= len(order), "Resume batch cursor out of range")
            model.train()
            current_epoch_completed = False
            epoch_loss_sum, epoch_examples, epoch_start_step = 0.0, 0, step
            while cursor < len(order):
                check_time(started, max_seconds_remaining, args.absolute_deadline_epoch)
                selected = order[cursor:cursor + config["batch_size"]]
                batch = [train_rows[i] for i in selected]
                eeg, families, labels = batch_rows(batch, store, device)
                optimizer.zero_grad(set_to_none=True)
                consistent = False  # RNG/BatchNorm may mutate even if a later allocation fails.
                logits = forward_logits(model, args.kind, eeg, families, tokenizer)
                loss = F.cross_entropy(logits, labels)
                require(bool(torch.isfinite(loss)), "Nonfinite training loss")
                loss.backward()
                require(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), "Nonfinite training gradient")
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["gradient_clip_norm"])
                optimizer.step()
                synchronize(device)
                consistent = True
                step += 1
                cursor += len(batch)
                seen.update(r["id"] for r in batch)
                presentations += len(batch)
                epoch_examples += len(batch)
                epoch_loss_sum += float(loss.detach()) * len(batch)
                if not within_budget(memory_sample(device), config["memory_budget_gib"], device):
                    raise BoundedStop("memory_budget")
                if step % plan["periodic_checkpoint_steps"] == 0:
                    state = training_state(config, membership_sha, epoch, cursor, step, best_score, best_sha, seen, presentations,
                                           elapsed_before + time.monotonic() - started, False)
                    last_checkpoint = output / f"progress_epoch_{epoch+1:03d}_step_{step:07d}.pt"
                    save_checkpoint(last_checkpoint, model, optimizer, state)
            # Partial validation is discarded and cannot select a checkpoint.
            metrics, predictions = validate(
                model, args.kind, val_rows, store, tokenizer, device,
                config["validation_batch_size"], started, max_seconds_remaining,
                args.absolute_deadline_epoch, config["memory_budget_gib"],
            )
            score = metrics["selection_score"]
            current_epoch_completed = True
            candidate_best = score if best_score is None or score > best_score else best_score
            state = training_state(config, membership_sha, epoch, cursor, step, candidate_best, best_sha, seen, presentations,
                                   elapsed_before + time.monotonic() - started, True)
            last_checkpoint = output / f"epoch_{epoch+1:03d}_step_{step:07d}.pt"
            checkpoint_sha = save_checkpoint(last_checkpoint, model, optimizer, state)
            best_score, best_sha, selected_best = update_best(output, last_checkpoint, score, best_score)
            val_path = output / f"validation_epoch_{epoch+1:03d}.jsonl"
            require(not val_path.exists(), "Preserve previous complete validation predictions")
            with val_path.open("x") as stream:
                for row in predictions:
                    stream.write(json.dumps({**row, "checkpoint_sha256": checkpoint_sha}, allow_nan=False) + "\n")
            record = {
                "epoch": epoch + 1, "global_step": step, "optimizer_steps_this_attempt_epoch": step - epoch_start_step,
                "training_examples_this_attempt_epoch": epoch_examples,
                "training_loss_this_attempt_epoch": epoch_loss_sum / epoch_examples if epoch_examples else None,
                "validation": metrics, "checkpoint": last_checkpoint.name, "checkpoint_sha256": checkpoint_sha,
                "validation_predictions_sha256": sha256(val_path), "selected_best": selected_best,
                "stage_elapsed_seconds": elapsed_before + time.monotonic() - started,
                "memory": memory_sample(device),
            }
            append_json(history_path, record)
            print(json.dumps({"epoch": epoch + 1, "global_step": step, "primary_validation_subject_BA": score,
                              "selected_best": selected_best, "elapsed_seconds": record["stage_elapsed_seconds"]}), flush=True)
            completed_epochs = epoch + 1
            epoch, cursor = epoch + 1, 0
    except BoundedStop as exc:
        stop = str(exc)
    except (RuntimeError, ValueError) as exc:
        stop = type(exc).__name__ + ": " + str(exc)
    if stop:
        # No validation is attempted after memory/partial-update failure.
        state = training_state(config, membership_sha, epoch, cursor, step, best_score, best_sha, seen, presentations,
                               elapsed_before + time.monotonic() - started, current_epoch_completed, resumable=consistent)
        last_checkpoint = output / f"stopped_attempt_{attempt:03d}_step_{step:07d}.pt"
        save_checkpoint(last_checkpoint, model, optimizer, state)
        write_json(output / f"stop_attempt_{attempt:03d}.json", {
            "reason": stop, "checkpoint": last_checkpoint.name, "checkpoint_sha256": sha256(last_checkpoint),
            "resumable": consistent, "test_forward_calls": 0,
            "validation_skipped_after_stop": True, "next_epoch_zero_based": epoch, "next_position": cursor,
        })
    actual_seen_rows = [r for r in train_rows if r["id"] in seen]
    if (output / "best.pt").exists():
        best_sha = sha256(output / "best.pt")
    summary = {
        "schema": "xavier-phase3-neural-training-summary-v1",
        "status": "completed" if stop is None and completed_epochs == config["max_epochs"] else "partial",
        "stop_reason": stop, "kind": args.kind,
        "best_checkpoint_sha256": best_sha, "best_primary_validation_subject_BA": best_score,
        "completed_epochs": completed_epochs, "optimizer_steps": step,
        "training_examples_presented": presentations,
        "summed_presented_window_seconds": presentations * 3.0,
        "unique_seen_training_examples": len(seen),
        "seen_train_ids": sorted(seen), "seen_train_ids_sha256": object_hash(sorted(seen)),
        "unique_source_interval_union_seconds": source_interval_union(actual_seen_rows),
        "raw_recording_source_hours_not_equivalent_to_selected_window_hours": True,
        "all_registered_training_ids_seen": seen == {r["id"] for r in train_rows},
        "validation_records": len(val_rows), "test_forward_calls": 0,
        "test_used_for_tuning": False, "model_selection_partition": "val",
        "elapsed_stage_seconds": elapsed_before + time.monotonic() - started,
        "data_audit": data_audit, "memory": memory_sample(device),
        "hardware": hardware(device), "allocator": allocator,
        "execution_contract_sha256": contract_sha, "scientific_source_sha256": source_seals(),
        "run_config_sha256": sha256(output / "run_config.json"),
        "training_membership_sha256": membership_sha,
        "history_sha256": sha256(history_path) if history_path.exists() else None,
        "last_checkpoint": last_checkpoint.name if last_checkpoint else None,
        "goal_validated": False,
        "ssl_encoder_origin": contract["ssl_encoder_origin"],
        "previous_phase2_test_reused_exploratorily": True,
    }
    write_json(output / "run_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "kind", "completed_epochs", "optimizer_steps", "best_checkpoint_sha256", "stop_reason")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register_parser = sub.add_parser("register")
    for name in ("prepared", "protocol", "out", "ssl-run", "head-equivalence", "budget"):
        register_parser.add_argument("--" + name, required=True)
    for command in ("probe", "train"):
        p = sub.add_parser(command)
        for name in ("prepared", "contract", "run-dir", "ssl-run"):
            p.add_argument("--" + name, required=True)
        p.add_argument("--kind", choices=("scratch", "fused"), required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--scratch-run")
        p.add_argument("--local-files-only", action="store_true")
        p.add_argument("--absolute-deadline-epoch", type=float)
        if command == "probe":
            p.add_argument("--batch-size", type=int, required=True)
        else:
            p.add_argument("--capacity-reports", nargs="+", required=True)
            p.add_argument("--resume")
    args = parser.parse_args()
    if args.command == "register":
        register(args.prepared, args.protocol, args.out, args.ssl_run, args.head_equivalence, args.budget)
    elif args.command == "probe":
        result = probe(args)
        raise SystemExit(0 if result["status"] == "passed" else 2)
    else:
        result = train(args)
        raise SystemExit(0 if result["status"] == "completed" else 2)


if __name__ == "__main__":
    main()

