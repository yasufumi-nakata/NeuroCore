#!/usr/bin/env python3
"""Sealed multi-source CBraMod SSL stages on smac Apple Silicon.

The completed CUDA experiments and their source hashes are unchanged. This
module reuses their verified official encoder loader and masked objective.
Only explicitly admitted, physically calibrated, non-overlapping train/val
shards enter the model. Capacity probes are train-only and never seed a run.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import time

import numpy as np
import torch

if __package__:
    from . import cbramod_ssl as legacy
else:
    import cbramod_ssl as legacy

SCHEMA = "xavier-smac-multichannel-ssl-v1"
ALLOWED_CHANNELS = {8, 14, 16, 27, 32, 64}
sha256 = legacy.sha256
write_json = legacy.dump_json


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def bucket(row):
    return (row["montage_key"], tuple(row["channels"]), row["reference"],
            float(row["native_sample_rate_hz"]), row["patches"])


def bucket_id(row):
    return object_hash(bucket(row))


def check_history(rows, history=None):
    history = {kind: dict((history or {}).get(kind, {})) for kind in ("source", "payload", "group")}
    for row in rows:
        for kind, key in (("source", "source_sha256"), ("payload", "eeg_payload_sha256"), ("group", "subject_group")):
            previous = history[kind].setdefault(row[key], row["split"])
            require(previous == row["split"], "Source/payload/group changes partition across stages")
    return history


def load_corpus(manifest_path, artifacts_path, admission_path):
    root = Path(manifest_path).resolve().parent
    rows = [json.loads(line) for line in Path(manifest_path).read_text().splitlines() if line.strip()]
    artifacts, admission = read(artifacts_path), read(admission_path)
    require(rows and {r.get("split") for r in rows} == {"train", "val"}, "Sealed train and validation are required")
    require(isinstance(artifacts, dict) and isinstance(admission.get("sources"), dict), "Missing corpus seals")
    denied_sources = set(admission.get("excluded_source_sha256", []))
    denied_payloads = set(admission.get("excluded_payload_sha256", []))
    ids, signal_slots, slots, source_intervals = set(), set(), defaultdict(set), defaultdict(list)
    file_layout = {}
    for row in rows:
        require(not {"label", "target", "action_label", "gold", "environment_text"} & row.keys(), "Labels or context are not SSL inputs")
        require(isinstance(row.get("id"), str) and row["id"] and row["id"] not in ids, "Duplicate or absent window ID")
        ids.add(row["id"])
        require(valid_sha(row.get("source_sha256")) and valid_sha(row.get("eeg_payload_sha256")), "Raw and calibrated payload hashes required")
        source, payload = row["source_sha256"], row["eeg_payload_sha256"]
        require(source not in denied_sources and payload not in denied_payloads, "Heldout raw or payload entered corpus")
        sealed = admission["sources"].get(source)
        require(isinstance(sealed, dict), "Raw source is not admitted")
        for key in ("split", "subject_group", "eeg_payload_sha256", "montage_key"):
            require(row.get(key) == sealed.get(key) and row.get(key) is not None, "Source admission mismatch: " + key)
        require(sealed.get("physical_unit_verified") is True and row.get("physical_unit_verified") is True, "Physical volts provenance is unverified")
        require(sealed.get("fit_allowed") is (row["split"] == "train"), "Train admission is not explicit")
        require(sealed.get("validation_allowed") is (row["split"] == "val"), "Validation admission is not explicit")
        require(isinstance(row["subject_group"], str) and row["subject_group"], "Verified participant/component group is required")
        channels = row.get("channels")
        require(isinstance(channels, list) and len(channels) in ALLOWED_CHANNELS and len(set(channels)) == len(channels), "Invalid ordered montage")
        require(all(isinstance(c, str) and c for c in channels), "Channel names must be explicit")
        require(row.get("unit") == "V_x_1e4" and row.get("sample_rate_hz") == 200, "Only once-scaled physical volts at 200Hz are admitted")
        require(isinstance(row.get("reference"), str) and row["reference"] and isinstance(row.get("montage_key"), str) and row["montage_key"], "Reference and montage identity required")
        require(isinstance(row.get("native_sample_rate_hz"), (int, float)) and math.isfinite(row["native_sample_rate_hz"]) and row["native_sample_rate_hz"] > 80, "Invalid native sampling rate for the 40Hz causal filter")
        start, end = float(row["window_start_s"]), float(row["window_end_s"])
        duration = end - start
        require(math.isfinite(start) and math.isfinite(end) and start >= 0 and duration > 0 and abs(duration - round(duration)) < 1e-7, "Invalid original interval")
        patches = int(round(duration))
        require(1 <= patches <= 30, "Capacity protocol supports 1..30 one-second patches")
        require(row.get("patches", patches) == patches, "Patch count differs from original duration")
        row["patches"] = patches
        signal_slot = (payload, row["montage_key"], start, end)
        require(signal_slot not in signal_slots, "Duplicate calibrated EEG interval")
        signal_slots.add(signal_slot)
        source_intervals[source].append((start, end))
        name = row["eeg_path"]
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and relative.suffix == ".npy", "Bounded relative npy path required")
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "Prepared array escapes corpus root")
        seal = artifacts.get(name)
        require(isinstance(seal, dict) and valid_sha(seal.get("sha256")), "Array bytes are not sealed")
        shape = seal.get("shape")
        require(isinstance(shape, list) and len(shape) == 4 and all(type(v) is int and v > 0 for v in shape), "Stacked [N,C,P,200] shape required")
        require(shape[1:] == [len(channels), patches, 200] and seal.get("dtype") == "float32", "Prepared shape or dtype differs")
        index = row.get("eeg_index")
        require(type(index) is int and 0 <= index < shape[0] and index not in slots[name], "Duplicate or invalid array slot")
        slots[name].add(index)
        require(file_layout.setdefault(name, bucket(row)) == bucket(row), "One array mixes montage/reference/rate/shape")
        row["_path"] = str(path)
    history = check_history(rows)
    for intervals in source_intervals.values():
        ordered = sorted(intervals)
        require(all(a[1] <= b[0] + 1e-8 for a, b in zip(ordered, ordered[1:])), "Original EEG intervals overlap")
    require(set(slots) == set(artifacts), "Extra or absent prepared array")
    for name, seal in artifacts.items():
        require(slots[name] == set(range(seal["shape"][0])), "Every prepared array slot needs provenance")
        path = (root / name).resolve()
        require(path.stat().st_size == seal["bytes"] and sha256(path) == seal["sha256"], "Prepared EEG bytes changed")
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        require(array.dtype == np.float32 and list(array.shape) == seal["shape"], "Array header changed")
    return rows, artifacts, admission, history


class Store:
    def __init__(self):
        self.arrays = {}

    def get(self, row):
        path = row["_path"]
        if path not in self.arrays:
            self.arrays[path] = np.load(path, allow_pickle=False, mmap_mode="r")
        value = self.arrays[path][row["eeg_index"]]
        require(value.shape == (len(row["channels"]), row["patches"], 200), "Window shape changed")
        require(np.isfinite(value).all() and np.abs(value).max() <= 1.00001, "Signal fails sealed physical-volts/QC range")
        return torch.from_numpy(np.array(value, dtype=np.float32, copy=True))


def epoch_batches(rows, batch_size, seed, epoch):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[bucket(row)].append(i)
    rng, result = random.Random(seed + epoch), []
    for key in sorted(groups):
        rng.shuffle(groups[key])
        result.extend(groups[key][j:j + batch_size] for j in range(0, len(groups[key]), batch_size))
    rng.shuffle(result)
    return result


def probe_batches(rows, batch_size, seed, steps_per_shape):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[bucket(row)].append(i)
    result = []
    for key in sorted(groups):
        indices = list(groups[key]); random.Random(seed).shuffle(indices)
        for step in range(steps_per_shape):
            result.append([indices[(step * batch_size + j) % len(indices)] for j in range(batch_size)])
    return result


def batch_tensor(rows, store, device):
    require(len({bucket(row) for row in rows}) == 1, "Batch mixes montage/reference/rate/shape")
    return torch.stack([store.get(row) for row in rows]).to(device=device, dtype=torch.float32)


def rng_state(device):
    state = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, device):
    torch.set_rng_state(state["torch"]); random.setstate(state["python"])
    if device.type == "mps":
        require("mps" in state, "MPS RNG checkpoint missing")
        torch.mps.set_rng_state(state["mps"])


def seed_all(seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)


def memory_bytes(device):
    sample = legacy.frozen.memory_sample(device)
    return sample, max(sample.values(), default=0)


def source_seals():
    return {"trainer": sha256(Path(__file__)), "legacy": sha256(Path(legacy.__file__)),
            "frozen": legacy.FROZEN_MODULE_SHA256, "official_weights": legacy.frozen.WEIGHTS_SHA256}


def corpus_seals(args):
    return {key: sha256(Path(getattr(args, key))) for key in ("manifest", "artifacts", "admission")}


def read_parent(path):
    if not path:
        return None
    root = Path(path).resolve(); summary = read(root / "run_summary.json")
    require(summary.get("schema") == SCHEMA and summary.get("kind") == "train" and summary.get("global_step", 0) > 0, "Parent is not a real smac training stage")
    require(summary.get("best_checkpoint_sha256") == sha256(root / "best.pt"), "Parent selected checkpoint changed")
    member = read(root / "training_membership.json")
    require(summary.get("membership_sha256") == sha256(root / "training_membership.json"), "Parent membership changed")
    return {"path": str(root), "checkpoint_sha256": summary["best_checkpoint_sha256"],
            "membership_sha256": summary["membership_sha256"], "history": member["cumulative_partition_history"],
            "denied_sources": member["denied_sources"], "denied_payloads": member["denied_payloads"]}


def register(args):
    rows, _, admission, history = load_corpus(args.manifest, args.artifacts, args.admission)
    parent = read_parent(args.parent_run)
    if parent:
        history = check_history(rows, parent["history"])
        require(set(parent["denied_sources"]) <= set(admission.get("excluded_source_sha256", [])), "Previous source denylist was dropped")
        require(set(parent["denied_payloads"]) <= set(admission.get("excluded_payload_sha256", [])), "Previous payload denylist was dropped")
    require(0 < args.max_seconds <= 86400 and 0 < args.memory_budget_gib <= 48 and 1 <= args.epochs <= 100, "Invalid bounded stage budget")
    require(args.device in ("mps", "cpu") and args.batch_candidates and all(v > 0 for v in args.batch_candidates), "Invalid device or capacity candidates")
    require(not Path(args.protocol).exists(), "Preserve existing protocol")
    value = {"schema": SCHEMA, "registered_epoch": time.time(), "sources": source_seals(), "corpus": corpus_seals(args),
             "training": {"device": args.device, "precision": "float32", "seed": args.seed, "epochs": args.epochs,
                          "max_seconds": args.max_seconds, "memory_budget_gib": args.memory_budget_gib,
                          "learning_rate": 1e-4, "weight_decay": .05, "cpu_threads": 4, "checkpoint_steps": 50},
             "capacity": {"batch_candidates": sorted(set(args.batch_candidates)), "steps_per_shape": args.probe_steps_per_shape,
                          "train_buckets": sorted({bucket_id(r) for r in rows if r["split"] == "train"})},
             "parent": parent, "cumulative_partition_history": history,
             "test_used": False, "labels_used": False, "online_intent_claim": False}
    require(3 <= args.probe_steps_per_shape <= 50, "At least three real backward steps per shape are required")
    write_json(Path(args.protocol), value)
    return {"status": "registered", "protocol_sha256": sha256(Path(args.protocol)), "windows": len(rows), "train_shapes": len(value["capacity"]["train_buckets"])}


def environment(protocol):
    host = platform.node().split(".")[0].lower()
    require(re.fullmatch(r"smac\d+", host) is not None and host != "smac19", "Actual probes/training are restricted to smac, excluding smac19")
    device = legacy.frozen.select_device(protocol["training"]["device"])
    require(device.type in ("mps", "cpu"), "This execution protocol is for Apple Silicon")
    torch.set_num_threads(protocol["training"]["cpu_threads"])
    allocator = legacy.frozen.apply_allocator_limit(device, protocol["training"]["memory_budget_gib"])
    seed_all(protocol["training"]["seed"], device)
    return device, allocator


def initialize(args, protocol, device):
    model = legacy.load_full_encoder(args.cbramod_upstream, args.cbramod_weights)
    if protocol["parent"]:
        p = protocol["parent"]
        path = Path(p["path"]) / "best.pt"
        require(sha256(path) == p["checkpoint_sha256"], "Parent weights changed after registration")
        parent = torch.load(path, map_location="cpu", weights_only=True)
        require(parent["kind"] == "train", "Capacity checkpoint cannot seed a new stage")
        model.load_state_dict(parent["model"], strict=True)
    return model.to(device=device, dtype=torch.float32)


def checkpoint(path, model, optimizer, state, device):
    state = {**state, "schema": SCHEMA, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "optimizer": optimizer.state_dict(), "rng": rng_state(device)}
    legacy.atomic_torch_save(path, state)
    return sha256(path)


@torch.no_grad()
def validate(model, rows, store, device, batch_size, seed, budget):
    require(rows and all(r["split"] == "val" for r in rows), "Only fixed independent validation is allowed")
    model.eval(); groups = defaultdict(list)
    for indices in epoch_batches(rows, batch_size, seed, 0):
        batch = [rows[i] for i in indices]; x = batch_tensor(batch, store, device)
        masks = torch.stack([legacy.fixed_mask(r["id"], x.shape[1], x.shape[2], seed) for r in batch]).to(device)
        prediction = model(x, mask=masks)
        for row, pred, target, mask in zip(batch, prediction, x, masks):
            value = float(legacy.masked_loss(pred[None], target[None], mask[None]).cpu())
            require(math.isfinite(value), "Nonfinite validation loss")
            groups[row["subject_group"]].append(value)
        _, used = memory_bytes(device)
        require(used <= budget * 1024**3, "Validation memory budget exceeded")
    return {"mean_subject_masked_mse": float(np.mean([np.mean(v) for v in groups.values()])),
            "groups": len(groups), "windows": len(rows), "test_windows": 0}


def select_batch(reports, protocol, protocol_sha):
    eligible = []
    for path in reports:
        report = read(path)
        require(report.get("protocol_sha256") == protocol_sha and report.get("sources") == protocol["sources"], "Capacity receipt belongs to different source/protocol")
        require(report.get("kind") == "probe" and report.get("weights_reused") is False, "Training state cannot be a capacity probe")
        if (report.get("status") == "passed" and report.get("all_registered_buckets_observed") is True
                and report.get("all_gradients_finite_nonzero") is True and report.get("memory_within_budget") is True):
            eligible.append(report["batch_size"])
    require(eligible and all(v in protocol["capacity"]["batch_candidates"] for v in eligible), "No registered measured capacity passed")
    return max(eligible)


def execute(args):
    protocol = read(args.protocol); digest = sha256(Path(args.protocol))
    require(protocol.get("schema") == SCHEMA and protocol.get("sources") == source_seals(), "Trainer/encoder source seal changed")
    require(protocol["corpus"] == corpus_seals(args), "Corpus bytes changed after registration")
    rows, _, admission, history = load_corpus(args.manifest, args.artifacts, args.admission)
    history = check_history(rows, protocol["cumulative_partition_history"])
    device, allocator = environment(protocol); cfg = protocol["training"]
    out = Path(args.output).resolve()
    require(not out.exists() or args.resume, "Preserve existing run/probe directory")
    require(not args.resume or args.command == "train", "Capacity probes cannot resume")
    batch_size = args.batch_size if args.command == "probe" else select_batch(args.capacity_reports, protocol, digest)
    require(batch_size in protocol["capacity"]["batch_candidates"], "Unregistered batch")
    out.mkdir(parents=True, exist_ok=True)
    config = {"schema": SCHEMA, "kind": args.command, "protocol_sha256": digest, "sources": protocol["sources"], "batch_size": batch_size,
              "corpus": protocol["corpus"], "training": cfg, "parent_checkpoint_sha256": (protocol["parent"] or {}).get("checkpoint_sha256")}
    member = {"train_ids": sorted(r["id"] for r in rows if r["split"] == "train"), "val_ids": sorted(r["id"] for r in rows if r["split"] == "val"),
              "cumulative_partition_history": history, "denied_sources": sorted(admission.get("excluded_source_sha256", [])),
              "denied_payloads": sorted(admission.get("excluded_payload_sha256", [])), "test_used": False}
    if args.resume:
        require(read(out / "run_config.json") == config and read(out / "training_membership.json") == member, "Resume changes scientific identity")
    else:
        write_json(out / "run_config.json", config); write_json(out / "training_membership.json", member)
        write_json(out / "protocol.json", protocol)
    member_sha = sha256(out / "training_membership.json")
    model = initialize(args, protocol, device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    train_rows = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["id"])
    val_rows = sorted((r for r in rows if r["split"] == "val"), key=lambda r: r["id"])
    step = epoch = cursor = presented = 0; elapsed_before = 0.; seen = set(); best_score = math.inf
    if args.resume:
        resume = Path(args.resume).resolve(); require(resume.parent == out, "Resume must be in original run")
        state = torch.load(resume, map_location="cpu", weights_only=True)
        require(state["config"] == config and state["membership_sha256"] == member_sha, "Checkpoint identity mismatch")
        model.load_state_dict(state["model"], strict=True); optimizer.load_state_dict(state["optimizer"])
        step, epoch, cursor, presented = state["step"], state["epoch"], state["next_batch"], state["presented"]
        seen = set(state["seen_ids"]); best_score = state["best_score"]; elapsed_before = state["elapsed_seconds"]
        restore_rng(state["rng"], device)
    write_json(out / "hardware.json", {**legacy.frozen.hardware(device), "allocator": allocator, "dtype": "float32", "MPS_memory_is_sampled_not_true_peak": device.type == "mps"})
    store = Store(); start = time.monotonic(); observed = set(); finite_nonzero = True; peak = 0; durations = []
    status = "completed"; best_hash = sha256(out / "best.pt") if (out / "best.pt").exists() else None
    initial_hash = object_hash({k: sha256_tensor(v) for k, v in model.state_dict().items()})
    if args.command == "train" and not args.resume:
        write_json(out / "initial_validation.json", validate(model, val_rows, store, device, batch_size, cfg["seed"], cfg["memory_budget_gib"]))
    while epoch < (1 if args.command == "probe" else cfg["epochs"]):
        batches = (probe_batches(train_rows, batch_size, cfg["seed"], protocol["capacity"]["steps_per_shape"])
                   if args.command == "probe" else epoch_batches(train_rows, batch_size, cfg["seed"], epoch))
        loss_total = count = 0; model.train()
        while cursor < len(batches):
            if elapsed_before + time.monotonic() - start >= cfg["max_seconds"] * .85:
                status = "stopped_time_budget"; break
            batch = [train_rows[i] for i in batches[cursor]]; x = batch_tensor(batch, store, device)
            mask = legacy.random_mask(len(batch), x.shape[1], x.shape[2], device=device)
            optimizer.zero_grad(set_to_none=True); legacy.frozen.sync(device); tick = time.perf_counter()
            loss = legacy.masked_loss(model(x, mask=mask), x, mask)
            require(bool(torch.isfinite(loss)), "Nonfinite training loss")
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            good = bool(torch.isfinite(norm)) and float(norm.detach().cpu()) > 0
            finite_nonzero &= good; require(good, "Missing or nonfinite encoder gradient")
            optimizer.step(); legacy.frozen.sync(device)
            durations.append(time.perf_counter() - tick); step += 1; cursor += 1; presented += len(batch)
            seen.update(r["id"] for r in batch); observed.update(bucket_id(r) for r in batch)
            loss_total += float(loss.detach().cpu()) * len(batch); count += len(batch)
            _, used = memory_bytes(device); peak = max(peak, used)
            if peak > cfg["memory_budget_gib"] * 1024**3:
                status = "stopped_memory_budget"; break
            state = make_state(config, member_sha, epoch, cursor, step, presented, seen, best_score, elapsed_before + time.monotonic() - start)
            if step % cfg["checkpoint_steps"] == 0:
                checkpoint(out / "latest.pt", model, optimizer, state, device)
                print(json.dumps({"step": step, "train_masked_mse": loss_total / count, "unique_windows_seen": len(seen), "device": str(device)}), flush=True)
        complete = cursor == len(batches)
        validation = None
        if args.command == "train" and status != "stopped_memory_budget":
            validation = validate(model, val_rows, store, device, batch_size, cfg["seed"], cfg["memory_budget_gib"])
        better = validation is not None and validation["mean_subject_masked_mse"] < best_score
        if better:
            best_score = validation["mean_subject_masked_mse"]
        next_epoch, next_cursor = (epoch + 1, 0) if complete else (epoch, cursor)
        state = make_state(config, member_sha, next_epoch, next_cursor, step, presented, seen, best_score, elapsed_before + time.monotonic() - start)
        checkpoint(out / "latest.pt", model, optimizer, state, device)
        if better:
            best_hash = checkpoint(out / "best.pt", model, optimizer, state, device)
        record = {"epoch": epoch + 1, "step": step, "epoch_completed": complete, "validation": validation,
                  "train_masked_mse": loss_total / count if count else None, "selected_best": better, "elapsed_seconds": state["elapsed_seconds"]}
        with (out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        epoch, cursor = next_epoch, next_cursor
        if args.command == "probe" or status != "completed":
            break
    changed = initial_hash != object_hash({k: sha256_tensor(v) for k, v in model.state_dict().items()})
    seen_seconds = sum(r["window_end_s"] - r["window_start_s"] for r in train_rows if r["id"] in seen)
    result = {"schema": SCHEMA, "kind": args.command, "status": status, "protocol_sha256": digest, "sources": protocol["sources"],
              "batch_size": batch_size, "global_step": step, "completed_epochs": epoch, "unique_training_windows": len(seen),
              "unique_training_seconds": seen_seconds, "training_examples_presented": presented,
              "membership_sha256": member_sha, "best_checkpoint_sha256": best_hash,
              "best_validation_mean_subject_MSE": best_score if math.isfinite(best_score) else None,
              "elapsed_seconds": elapsed_before + time.monotonic() - start, "peak_sampled_memory_bytes": peak,
              "memory_within_budget": peak <= cfg["memory_budget_gib"] * 1024**3,
              "all_registered_buckets_observed": observed == set(protocol["capacity"]["train_buckets"]),
              "all_gradients_finite_nonzero": finite_nonzero and step > 0 and changed,
              "weights_reused": False if args.command == "probe" else None, "test_windows": 0, "behavior_labels_used": 0,
              "step_seconds_p50": float(np.median(durations)) if durations else None}
    if args.command == "probe":
        result["status"] = "passed" if status == "completed" and result["all_registered_buckets_observed"] and result["all_gradients_finite_nonzero"] and result["memory_within_budget"] else "failed"
    write_json(out / "run_summary.json", result)
    return result


def sha256_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_state(config, membership_sha, epoch, cursor, step, presented, seen, best, elapsed):
    return {"kind": config["kind"], "config": config, "membership_sha256": membership_sha, "epoch": epoch,
            "next_batch": cursor, "step": step, "presented": presented, "seen_ids": sorted(seen), "best_score": best,
            "elapsed_seconds": elapsed}


def parser():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    for name in ("register", "probe", "train"):
        q = sub.add_parser(name)
        for key in ("manifest", "artifacts", "admission", "protocol"):
            q.add_argument("--" + key, required=True)
        if name == "register":
            q.add_argument("--device", choices=("mps", "cpu"), default="mps")
            q.add_argument("--seed", type=int, default=20261122); q.add_argument("--epochs", type=int, default=10)
            q.add_argument("--max-seconds", type=float, default=3600); q.add_argument("--memory-budget-gib", type=float, default=40)
            q.add_argument("--batch-candidates", nargs="+", type=int, default=[1, 2, 4]); q.add_argument("--probe-steps-per-shape", type=int, default=5)
            q.add_argument("--parent-run")
        else:
            for key in ("output", "cbramod-upstream", "cbramod-weights"):
                q.add_argument("--" + key, required=True)
            q.add_argument("--resume")
            if name == "probe":
                q.add_argument("--batch-size", type=int, required=True)
            else:
                q.add_argument("--capacity-reports", nargs="+", required=True)
    return p


def main():
    args = parser().parse_args()
    result = register(args) if args.command == "register" else execute(args)
    print(json.dumps(result, allow_nan=False), flush=True)
    if result.get("status") in ("failed", "stopped_memory_budget"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
