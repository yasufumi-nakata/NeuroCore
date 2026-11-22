#!/usr/bin/env python3
"""Bounded, sealed masked-EEG continuation of the official CBraMod checkpoint.

No language model, behavior label, UniBa held-out source, or ITU signal is used
by this program. Corpus preparation and its participant allowlist are separate
artifacts. This script verifies every prepared array before model loading.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn


FROZEN_MODULE_SHA256 = "b0c397176f8cb9e6dd759ed7169c0029c870f756c501ce317c2e6c7c41f32505"
FROZEN_PATH = Path(__file__).with_name("cbramod_token_llm.py")
if hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest() != FROZEN_MODULE_SHA256:
    raise RuntimeError("Original frozen-encoder experiment source changed")
_spec = importlib.util.spec_from_file_location("xavier_frozen_cbramod_for_ssl", FROZEN_PATH)
frozen = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = frozen
_spec.loader.exec_module(frozen)

SCHEMA = "xavier-cbramod-masked-ssl-v1"
UNIT = "V_x_1e4"
sha256 = frozen.sha256
dump_json = frozen.dump_json


def read_corpus(manifest, artifact_file, *, allowlist=None, verify=True):
    """Validate the actual bytes, participant partitions, and original intervals."""
    manifest, artifact_file = Path(manifest).resolve(), Path(artifact_file).resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    artifacts = json.loads(artifact_file.read_text())
    if "artifacts" in artifacts:
        artifacts = artifacts["artifacts"]
    if not rows or not isinstance(artifacts, dict):
        raise ValueError("Empty corpus or invalid array seal")
    allowed = json.loads(Path(allowlist).read_text()) if allowlist else None
    groups, source_parts, intervals, ids, files = {}, {}, defaultdict(list), set(), {}
    slots, file_channels = defaultdict(set), {}
    for row in rows:
        required = {"id", "split", "subject_group", "source_sha256", "source_recording_id",
                    "window_start_s", "window_end_s", "channels", "sample_rate_hz", "unit", "eeg_path"}
        if not required <= row.keys():
            raise ValueError("Missing source provenance in SSL row")
        if {"label", "action_label", "target", "environment_text"} & row.keys():
            raise ValueError("Behavior labels or environment text are not SSL inputs")
        if row["id"] in ids or row["split"] not in ("train", "val"):
            raise ValueError("Duplicate ID or unadmitted partition")
        ids.add(row["id"])
        if row["unit"] != UNIT or row["sample_rate_hz"] != 200 or len(row["channels"]) not in (8, 32):
            raise ValueError("Wrong SSL units, sample rate, or actual channel count")
        if len(set(row["channels"])) != len(row["channels"]):
            raise ValueError("Duplicate channel names")
        if len(row["source_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["source_sha256"]):
            raise ValueError("Raw source SHA-256 is required")
        if "itu" in row["subject_group"].lower() or "itu" in row["source_recording_id"].lower():
            raise ValueError("ITU is excluded from self-supervised learning")
        if allowed is not None:
            source = allowed["sources"].get(row["source_sha256"])
            if source is None or row["source_sha256"] in allowed["excluded_source_sha256"]:
                raise ValueError("Raw EEG was not admitted by the preassigned source allowlist")
            for field in ("split", "subject_group", "source_recording_id"):
                if source[field] != row[field]:
                    raise ValueError("SSL row misrepresents its original subject or partition")
        for key, group in (("subject_group", groups), ("source_sha256", source_parts)):
            old = group.setdefault(row[key], row["split"])
            if old != row["split"]:
                raise ValueError("Subject or raw waveform appears in both SSL partitions")
        start, end = float(row["window_start_s"]), float(row["window_end_s"])
        if not math.isfinite(start + end) or start < 0 or abs(end - start - 30.) > 1e-6:
            raise ValueError("SSL windows must cover exactly 30 original seconds")
        intervals[row["source_sha256"]].append((start, end, row["id"]))
        relative = Path(row["eeg_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".npy":
            raise ValueError("Only bounded relative .npy corpus paths are admitted")
        path = (manifest.parent / relative).resolve()
        if not path.is_relative_to(manifest.parent):
            raise ValueError("Corpus array escapes its root")
        if row["eeg_path"] not in artifacts:
            raise ValueError("Array is not sealed in the artifact manifest")
        shape = artifacts[row["eeg_path"]]["shape"]
        if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in shape):
            raise ValueError("Invalid sealed array dimensions")
        if len(shape) == 4:
            index = row.get("eeg_index")
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < shape[0]:
                raise ValueError("eeg_index must be a nonnegative in-range integer, not bool/float")
            actual_window_shape = shape[1:]
        elif len(shape) == 3:
            if "eeg_index" in row:
                raise ValueError("A standalone single window must not specify eeg_index")
            index, actual_window_shape = 0, shape
        else:
            raise ValueError("Only individual or stacked channel x 30 x 200 windows are allowed")
        if actual_window_shape != [len(row["channels"]), 30, 200]:
            raise ValueError("Row channel count differs from the sealed array dimensions")
        if file_channels.setdefault(row["eeg_path"], tuple(row["channels"])) != tuple(row["channels"]):
            raise ValueError("Rows of one array disagree about channel order")
        if index in slots[row["eeg_path"]]:
            raise ValueError("The same prepared array slot was assigned to multiple source intervals")
        slots[row["eeg_path"]].add(index)
        files[row["eeg_path"]] = path
        row["_path"] = str(path)
    for items in intervals.values():
        previous_end = -math.inf
        for start, end, _ in sorted(items):
            if start < previous_end - 1e-8:
                raise ValueError("Overlapping or duplicate raw time windows")
            previous_end = end
    if set(r["split"] for r in rows) != {"train", "val"}:
        raise ValueError("Both SSL train and independent reconstruction validation are required")
    if set(files) != set(artifacts):
        raise ValueError("Artifact manifest contains extra or missing arrays")
    for name, seal in artifacts.items():
        expected_slots = set(range(seal["shape"][0])) if len(seal["shape"]) == 4 else {0}
        if slots[name] != expected_slots:
            raise ValueError("Every prepared array slot must have exactly one manifest provenance row")
    if verify:
        for name, path in files.items():
            seal = artifacts[name]
            if path.stat().st_size != seal["bytes"] or sha256(path) != seal["sha256"]:
                raise ValueError("Prepared SSL waveform bytes changed: " + name)
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            if arr.dtype != np.float32 or list(arr.shape) != seal["shape"]:
                raise ValueError("Sealed array shape or float32 dtype changed")
    return rows, artifacts


class SSLStore:
    def __init__(self):
        self.cache = {}

    def get(self, row):
        path = row["_path"]
        if path not in self.cache:
            self.cache[path] = np.load(path, allow_pickle=False, mmap_mode="r")
        value = self.cache[path]
        if "eeg_index" in row:
            index = row["eeg_index"]
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError("eeg_index is not a nonnegative integer")
            value = value[index]
        shape = (len(row["channels"]), 30, 200)
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Nonfinite or incorrectly shaped SSL signal")
        if float(np.abs(value).max()) > 1.00001:
            raise ValueError("Signal violates the sealed post-resample 100 microvolt QC ceiling")
        # A writable copy avoids undefined behavior if downstream code changes a
        # tensor backed by a read-only mmap. No time/channel normalization occurs.
        return torch.from_numpy(np.array(value, dtype=np.float32, copy=True))


def fixed_mask(record_id, channels, patches=30, seed=20261122):
    digest = hashlib.sha256(f"{seed}|ssl-validation-mask|{record_id}".encode()).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))
    count = channels * patches
    mask = torch.zeros(count, dtype=torch.bool)
    mask[torch.randperm(count, generator=generator)[:count // 2]] = True
    return mask.reshape(channels, patches)


def random_mask(batch, channels, patches=30, *, device):
    count = channels * patches
    ordering = torch.rand((batch, count), device=device).argsort(dim=1)
    result = torch.zeros((batch, count), device=device, dtype=torch.bool)
    result.scatter_(1, ordering[:, :count // 2], True)
    return result.reshape(batch, channels, patches)


def masked_loss(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape[:-1] or mask.dtype != torch.bool or not mask.any():
        raise ValueError("Masked reconstruction requires a nonempty aligned boolean patch mask")
    # Selection precedes MSE: unmasked output errors never enter this objective.
    return (prediction[mask].float() - target[mask].float()).square().mean()


def epoch_batches(rows, batch_size, seed, epoch):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[len(row["channels"])].append(index)
    rng = random.Random(seed + epoch)
    result = []
    for channels in sorted(groups):
        ids = groups[channels]
        rng.shuffle(ids)
        result.extend(ids[pos:pos + batch_size] for pos in range(0, len(ids), batch_size))
    rng.shuffle(result)
    return result


def probe_batches(rows, batch_size, steps, seed):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[len(row["channels"])].append(index)
    if set(groups) != {8, 32}:
        raise ValueError("Train-only capacity probe must observe both real channel shapes")
    rng = random.Random(seed)
    for ids in groups.values():
        rng.shuffle(ids)
    result, counters = [], {8: 0, 32: 0}
    for step in range(steps):
        channels = (32, 8)[step % 2]
        ids = groups[channels]
        # This is capacity work only: small corpora may repeat true train rows.
        result.append([ids[(counters[channels] + k) % len(ids)] for k in range(batch_size)])
        counters[channels] += batch_size
    return result


def load_full_encoder(upstream, weights):
    if sha256(Path(weights)) != frozen.WEIGHTS_SHA256:
        raise ValueError("SSL must initialize the exact official checkpoint")
    model = frozen.load_official_encoder({"cbramod_upstream": upstream, "cbramod_weights": weights})
    model.proj_out = nn.Sequential(nn.Linear(200, 200))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.endswith("mask_encoding"))
    return model.float()


def autocast(device, precision):
    # The source signal and model parameters stay float32. CUDA autocast leaves
    # the length-200 FFT in float32, avoiding unsupported half FFT lengths.
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and precision == "bfloat16" else nullcontext()


@torch.no_grad()
def reconstruction_validation(model, rows, store, batch_size, seed, device, precision, budget_gib):
    if any(row["split"] != "val" for row in rows):
        raise ValueError("SSL checkpoint selection uses reconstruction validation only")
    model.eval()
    weighted, count = 0., 0
    for indices in epoch_batches(rows, batch_size, seed, 0):
        batch = [rows[i] for i in indices]
        x = torch.stack([store.get(r) for r in batch]).to(device)
        masks = torch.stack([fixed_mask(r["id"], len(r["channels"]), seed=seed) for r in batch]).to(device)
        with autocast(device, precision):
            loss = masked_loss(model(x, mask=masks), x, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite validation reconstruction loss")
        weighted += float(loss.cpu()) * len(batch)
        count += len(batch)
        if max(frozen.memory_sample(device).values()) > budget_gib * 1024**3:
            raise RuntimeError("memory_budget_exceeded_during_ssl_validation")
    return {"masked_mse": weighted / count, "examples": count,
            "fixed_mask": "sha256(seed|ssl-validation-mask|record_id), exact 50% per example",
            "selection_partition": "portable_SSL_validation", "behavior_labels_used": 0}


def atomic_torch_save(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def save_checkpoint(path, model, optimizer, config, epoch, next_batch, completed, step, elapsed,
                    best_loss, presented_examples, seen_ids):
    state = {"schema": SCHEMA, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "optimizer": optimizer.state_dict(), "config": config, "epoch": epoch,
             "next_batch": next_batch, "epoch_completed": completed, "global_step": step,
             "elapsed_training_seconds": elapsed, "best_val_masked_mse": best_loss,
             "training_examples_presented": presented_examples, "seen_training_ids": sorted(seen_ids),
             "torch_rng": torch.get_rng_state(),
             "cuda_rng_states": torch.cuda.get_rng_state_all() if next(model.parameters()).device.type == "cuda" else None,
             "initial_weights_sha256": frozen.WEIGHTS_SHA256, "behavior_labels_used": 0}
    atomic_torch_save(path, state)


def validate_protocol(args, protocol):
    if args.epochs > 10 or args.max_seconds > 1800 or args.memory_budget_gib > 21:
        raise ValueError("This continuation is bounded by 10 epochs / 1800 seconds / 21 GiB")
    if protocol.get("schema") != "xavier-cbramod-ssl-protocol-v1" or protocol.get("model_source_sha256") != sha256(Path(__file__)):
        raise ValueError("Protocol does not seal this source")
    if protocol.get("frozen_encoder_module_sha256") != FROZEN_MODULE_SHA256:
        raise ValueError("Official model loader identity changed")
    for key in ("epochs", "lr", "seed", "precision", "max_seconds", "memory_budget_gib", "cpu_threads"):
        if getattr(args, key) != protocol["training"][key]:
            raise ValueError("Registered SSL setting changed: " + key)
    for name, path in (("manifest", args.manifest), ("artifacts", args.artifacts),
                       ("corpus_summary", args.corpus_summary), ("source_allowlist", args.source_allowlist)):
        if sha256(Path(path)) != protocol["corpus"][name + "_sha256"]:
            raise ValueError("SSL corpus seal changed: " + name)
    if args.batch_size not in protocol["capacity"]["candidate_batches"]:
        raise ValueError("Unregistered training batch candidate")
    if args.command == "probe":
        if args.probe_steps != protocol["capacity"]["steps_per_candidate"]:
            raise ValueError("Capacity probe step count changed")
    else:
        if not args.capacity_selection:
            raise ValueError("Main SSL run requires a measured capacity selection")
        receipt = json.loads(Path(args.capacity_selection).read_text())
        if receipt["protocol_sha256"] != sha256(Path(args.protocol)) or receipt["selected_batch"] != args.batch_size or receipt.get("weights_reused") is not False:
            raise ValueError("Capacity receipt does not match this fresh training run")
        eligible = []
        for item in receipt["evidence"]:
            path = Path(item["report_path"])
            if sha256(path) != item["report_sha256"]:
                raise ValueError("Capacity report bytes changed")
            report = json.loads(path.read_text())
            valid = capacity_eligible(report, protocol)
            if valid != item["eligible"] or report["protocol_sha256"] != receipt["protocol_sha256"]:
                raise ValueError("Capacity decision contradicts measured evidence")
            if valid:
                eligible.append(item["batch"])
        if not eligible or max(eligible) != args.batch_size:
            raise ValueError("Batch is not the largest measured stable candidate")


def train(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Preserve existing SSL run; output is nonempty")
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(args.protocol).read_text())
    validate_protocol(args, protocol)
    config = {k: v for k, v in vars(args).items() if k != "func"}
    config["output"] = str(output)
    config.update(schema=SCHEMA, protocol_sha256=sha256(Path(args.protocol)),
                  model_source_sha256=sha256(Path(__file__)), initial_weights_sha256=frozen.WEIGHTS_SHA256,
                  manifest_sha256=sha256(Path(args.manifest)), artifacts_sha256=sha256(Path(args.artifacts)),
                  corpus_summary_sha256=sha256(Path(args.corpus_summary)),
                  source_allowlist_sha256=sha256(Path(args.source_allowlist)))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = frozen.select_device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Self-supervised continuation is verified for CPU diagnostics or CUDA only")
    allocator = frozen.apply_allocator_limit(device, args.memory_budget_gib)
    rows, artifacts = read_corpus(args.manifest, args.artifacts, allowlist=args.source_allowlist)
    parts = {s: [r for r in rows if r["split"] == s] for s in ("train", "val")}
    store = SSLStore()
    model = load_full_encoder(args.cbramod_upstream, args.cbramod_weights).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=.05)
    step = epoch_start = next_batch = presented = 0
    prior_elapsed, best_loss, seen = 0., math.inf, set()
    if args.resume:
        if Path(args.resume).resolve().parent != output:
            raise ValueError("Resume in the original run directory to preserve its selected best")
        old = torch.load(args.resume, map_location="cpu", weights_only=True)
        if old["schema"] != SCHEMA:
            raise ValueError("Wrong SSL checkpoint")
        for key in ("protocol_sha256", "batch_size", "seed", "model_source_sha256", "manifest_sha256", "artifacts_sha256", "source_allowlist_sha256"):
            if old["config"][key] != config[key]:
                raise ValueError("Resume changes the registered source, corpus, or optimizer batch")
        model.load_state_dict(old["model"], strict=True)
        optimizer.load_state_dict(old["optimizer"])
        torch.set_rng_state(old["torch_rng"])
        if device.type == "cuda" and old["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(old["cuda_rng_states"])
        step, presented = old["global_step"], old["training_examples_presented"]
        seen = set(old["seen_training_ids"])
        epoch_start = old["epoch"] + int(old["epoch_completed"])
        next_batch = 0 if old["epoch_completed"] else old["next_batch"]
        prior_elapsed, best_loss = old["elapsed_training_seconds"], old["best_val_masked_mse"]
    dump_json(output / "run_config.json", config)
    dump_json(output / "hardware.json", {**frozen.hardware(device), "allocator": allocator,
              "total_encoder_parameters": sum(p.numel() for p in model.parameters()),
              "trainable_encoder_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "parameter_dtype": "float32", "activation_autocast": args.precision if device.type == "cuda" else "disabled",
              "fft_input_dtype": "float32", "behavior_labels_used": 0})
    dump_json(output / "training_membership.json", {"manifest_sha256": config["manifest_sha256"],
              "train_ids": sorted(r["id"] for r in parts["train"]), "val_ids": sorted(r["id"] for r in parts["val"]),
              "source_sha256_by_partition": {s: sorted({r["source_sha256"] for r in v}) for s, v in parts.items()},
              "subject_groups_by_partition": {s: sorted({r["subject_group"] for r in v}) for s, v in parts.items()},
              "test_source_used": False, "behavior_labels_used": 0})
    shutil.copyfile(args.artifacts, output / "eeg_artifacts.json")
    started, durations, observations, peak = time.monotonic(), [], [], {}
    status, last, epochs_completed = "completed_epochs", None, epoch_start
    if args.command == "train" and not args.resume:
        initial = reconstruction_validation(model, parts["val"], store, args.batch_size, args.seed, device, args.precision, args.memory_budget_gib)
        dump_json(output / "initial_validation.json", initial)
    for epoch in range(epoch_start, 1 if args.command == "probe" else args.epochs):
        batches = probe_batches(parts["train"], args.batch_size, args.probe_steps, args.seed) if args.command == "probe" else epoch_batches(parts["train"], args.batch_size, args.seed, epoch)
        losses, nloss = 0., 0
        model.train()
        for position in range(next_batch, len(batches)):
            if prior_elapsed + time.monotonic() - started >= args.max_seconds:
                status = "stopped_time_budget"; break
            batch = [parts["train"][i] for i in batches[position]]
            x = torch.stack([store.get(row) for row in batch]).to(device=device, dtype=torch.float32)
            mask = random_mask(len(batch), x.shape[1], device=device)
            optimizer.zero_grad(set_to_none=True)
            frozen.sync(device); tick = time.perf_counter()
            with autocast(device, args.precision):
                loss = masked_loss(model(x, mask=mask), x, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training masked MSE")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite SSL encoder gradient")
            optimizer.step(); frozen.sync(device)
            durations.append(time.perf_counter() - tick)
            losses += float(loss.detach().cpu()) * len(batch); nloss += len(batch)
            step += 1; next_batch = position + 1; presented += len(batch)
            seen.update(row["id"] for row in batch)
            memory = frozen.memory_sample(device)
            for key, value in memory.items():
                peak[key] = max(peak.get(key, 0), value)
            current = {**memory, "step": step, "channels": int(x.shape[1]), "batch": len(batch)}
            if device.type == "cuda":
                current.update(cuda_reserved_bytes=torch.cuda.memory_reserved(device), cuda_allocated_bytes=torch.cuda.memory_allocated(device))
            observations.append(current)
            if max(peak.values()) > args.memory_budget_gib * 1024**3:
                status = "stopped_memory_budget"; break
            if step % 50 == 0:
                save_checkpoint(output / "latest.pt", model, optimizer, config, epoch, next_batch, False, step,
                                prior_elapsed + time.monotonic() - started, best_loss, presented, seen)
                print(json.dumps({"epoch": epoch + 1, "step": step, "train_masked_mse": losses / nloss,
                                  "examples_presented": presented, "unique_windows_seen": len(seen)}), flush=True)
        complete = next_batch == len(batches)
        if complete and args.command == "train":
            epochs_completed = epoch + 1
        last = output / f"epoch_{epoch + 1:03d}_step_{step:07d}.pt"
        elapsed = prior_elapsed + time.monotonic() - started
        save_checkpoint(last, model, optimizer, config, epoch, next_batch, complete, step, elapsed, best_loss, presented, seen)
        validation = {"status": "not_run", "reason": "train-only probe or memory guard"}
        if args.command == "train" and status != "stopped_memory_budget":
            validation = reconstruction_validation(model, parts["val"], store, args.batch_size, args.seed, device, args.precision, args.memory_budget_gib)
            score = validation["masked_mse"]
            improved = score < best_loss
            if improved:
                best_loss = score
            save_checkpoint(last, model, optimizer, config, epoch, next_batch, complete, step,
                            prior_elapsed + time.monotonic() - started, best_loss, presented, seen)
            if improved or not (output / "best.pt").exists():
                shutil.copyfile(last, output / "best.pt.tmp"); os.replace(output / "best.pt.tmp", output / "best.pt")
                atomic_torch_save(output / "best_encoder_weights.pth", {k: v.detach().cpu() for k, v in model.state_dict().items()})
        shutil.copyfile(last, output / "latest.pt.tmp"); os.replace(output / "latest.pt.tmp", output / "latest.pt")
        record = {"epoch": epoch + 1, "global_step": step, "train_masked_mse": losses / nloss if nloss else None,
                  "val": validation, "epoch_completed": complete, "next_batch": next_batch,
                  "checkpoint": last.name, "elapsed_seconds": prior_elapsed + time.monotonic() - started}
        with (output / "history.jsonl").open("a") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        if status != "completed_epochs":
            break
        if args.command == "probe":
            status = "completed_train_only_probe"; break
        next_batch = 0
    frozen.core.update_memory(peak, device)
    tail = observations[-10:]
    values = [r["cuda_reserved_bytes"] for r in tail if "cuda_reserved_bytes" in r]
    best = output / "best_encoder_weights.pth"
    summary = {"schema": SCHEMA, "status": status, "global_step": step, "epochs_completed": epochs_completed,
               "protocol_sha256": config["protocol_sha256"], "model_source_sha256": config["model_source_sha256"],
               "manifest_sha256": config["manifest_sha256"], "batch_size": args.batch_size,
               "elapsed_seconds": prior_elapsed + time.monotonic() - started,
               "step_seconds_p50": float(np.median(durations)) if durations else None,
               "step_seconds_p95": float(np.quantile(durations, .95)) if durations else None,
               "memory": peak, "memory_budget_gib": args.memory_budget_gib,
               "training_examples_presented": presented, "unique_training_windows_seen": len(seen),
               "unique_training_seconds_seen": 30 * len(seen), "repeated_training_seconds_presented": 30 * presented,
               "eligible_train_windows": len(parts["train"]), "eligible_train_union_seconds": 30 * len(parts["train"]),
               "training_source_sha256": sorted({r["source_sha256"] for r in parts["train"]}),
               "best_validation_masked_mse": best_loss if math.isfinite(best_loss) else None,
               "best_encoder_weights_sha256": sha256(best) if best.exists() else None,
               "best_checkpoint_sha256": sha256(output / "best.pt") if (output / "best.pt").exists() else None,
               "initial_official_weights_sha256": frozen.WEIGHTS_SHA256, "last_checkpoint": last.name if last else None,
               "behavior_labels_used": 0, "itu_sources_used": 0, "downstream_test_source_used": False,
               "agentic_readiness": "not_validated", "os_actions_executed": 0}
    if args.command == "probe":
        summary["probe"] = {"steps": args.probe_steps, "completed_steps": step == args.probe_steps,
                            "both_channel_shapes_observed": {r["channels"] for r in observations} == {8, 32},
                            "validation_examples": 0, "test_examples": 0, "weights_reused": False,
                            "last_ten_reserved_range_bytes": max(values) - min(values) if len(values) == 10 else None,
                            "allocator_stable_last_ten": len(values) == 10 and max(values) - min(values) <= 256 * 1024**2}
    dump_json(output / "memory_observations.json", observations)
    dump_json(output / "run_summary.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return summary


def capacity_eligible(report, protocol):
    probe = report.get("probe", {})
    return (report.get("status") == "completed_train_only_probe" and probe.get("completed_steps") is True
            and probe.get("both_channel_shapes_observed") is True and probe.get("allocator_stable_last_ten") is True
            and max(report.get("memory", {"missing": math.inf}).values()) <= protocol["training"]["memory_budget_gib"] * 1024**3)


def select_capacity(args):
    protocol = json.loads(Path(args.protocol).read_text())
    evidence, eligible = [], []
    for name in args.reports:
        path = Path(name).resolve(); report = json.loads(path.read_text())
        if report.get("protocol_sha256") != sha256(Path(args.protocol)) or report.get("model_source_sha256") != sha256(Path(__file__)):
            raise ValueError("Capacity report belongs to another source or protocol")
        batch = report["batch_size"]
        if batch not in protocol["capacity"]["candidate_batches"] or batch in [e["batch"] for e in evidence]:
            raise ValueError("Unregistered or duplicate capacity candidate")
        passed = capacity_eligible(report, protocol)
        evidence.append({"batch": batch, "eligible": passed, "report_path": str(path), "report_sha256": sha256(path)})
        if passed:
            eligible.append(batch)
    if not eligible:
        raise ValueError("No sustained measured capacity candidate passed")
    result = {"protocol_sha256": sha256(Path(args.protocol)), "selected_batch": max(eligible),
              "weights_reused": False, "selected_by": "largest_measured_stable_train_only_batch", "evidence": evidence}
    if Path(args.output).exists():
        raise ValueError("Do not overwrite capacity selection")
    dump_json(Path(args.output), result)
    print(json.dumps(result), flush=True)
    return result


def seal_encoder(args):
    """Bind selected adapted weights to the completed SSL evidence, on CPU only."""
    root = Path(args.ssl_run).resolve()
    summary = json.loads((root / "run_summary.json").read_text())
    config = json.loads((root / "run_config.json").read_text())
    membership = json.loads((root / "training_membership.json").read_text())
    if summary["status"] not in ("completed_epochs", "stopped_time_budget") or summary["global_step"] <= 0:
        raise ValueError("Only a completed bounded SSL run can seed alignment")
    if config["command"] != "train" or config["model_source_sha256"] != sha256(Path(__file__)):
        raise ValueError("Capacity weights or changed trainer source cannot seed alignment")
    best, weights = root / "best.pt", root / "best_encoder_weights.pth"
    if sha256(best) != summary["best_checkpoint_sha256"] or sha256(weights) != summary["best_encoder_weights_sha256"]:
        raise ValueError("Selected SSL weight/checkpoint bytes changed")
    state = torch.load(best, map_location="cpu", weights_only=True)
    selected = torch.load(weights, map_location="cpu", weights_only=True)
    immutable = ("protocol_sha256", "model_source_sha256", "manifest_sha256", "artifacts_sha256",
                 "source_allowlist_sha256", "batch_size", "seed", "lr", "precision")
    if (state["schema"] != SCHEMA or any(state["config"][key] != config[key] for key in immutable)
            or selected.keys() != state["model"].keys()):
        raise ValueError("SSL state/config identity mismatch")
    if any(not torch.equal(value, state["model"][name]) for name, value in selected.items()):
        raise ValueError("Published encoder does not equal the validation-selected checkpoint")
    initial_path = Path(config["cbramod_weights"])
    if sha256(initial_path) != frozen.WEIGHTS_SHA256:
        raise ValueError("Official initialization bytes changed")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    changed_tensors = sum(not torch.equal(value, initial[name]) for name, value in selected.items())
    if changed_tensors == 0:
        raise ValueError("No model tensor changed during SSL")
    history = [json.loads(line) for line in (root / "history.jsonl").read_text().splitlines() if line.strip()]
    candidates = [r for r in history if "masked_mse" in r["val"]]
    winner = min(candidates, key=lambda r: r["val"]["masked_mse"])
    if winner["global_step"] != state["global_step"] or winner["val"]["masked_mse"] != summary["best_validation_masked_mse"]:
        raise ValueError("SSL checkpoint was not selected by the registered validation loss")
    if sha256(Path(config["protocol"])) != config["protocol_sha256"] or sha256(Path(config["source_allowlist"])) != config["source_allowlist_sha256"]:
        raise ValueError("SSL protocol or raw-source allowlist changed")
    allowlist = json.loads(Path(config["source_allowlist"]).read_text())
    exclusions = set(allowlist["excluded_source_sha256"])
    used = set().union(*(set(v) for v in membership["source_sha256_by_partition"].values()))
    if used & exclusions or used - set(allowlist["sources"]):
        raise ValueError("An excluded raw source reached SSL")
    result = {"schema": "xavier-selected-ssl-encoder-seal-v1", "run_name": root.name,
              "weights_sha256": sha256(weights), "weights_bytes": weights.stat().st_size,
              "ssl_best_checkpoint_sha256": sha256(best), "ssl_protocol_sha256": config["protocol_sha256"],
              "ssl_source_sha256": config["model_source_sha256"], "ssl_manifest_sha256": config["manifest_sha256"],
              "ssl_source_allowlist_sha256": config["source_allowlist_sha256"],
              "ssl_training_membership_sha256": sha256(root / "training_membership.json"),
              "ssl_run_summary_sha256": sha256(root / "run_summary.json"),
              "initial_weights_sha256": frozen.WEIGHTS_SHA256, "selection_partition": "portable_SSL_validation",
              "updated_tensor_count": changed_tensors,
              "selection_metric": "fixed-mask raw-sample MSE; earliest minimum", "selected_epoch": state["epoch"] + 1,
              "selected_global_step": state["global_step"], "training_examples_presented_before_selection": state["training_examples_presented"],
              "unique_training_windows_before_selection": len(state["seen_training_ids"]),
              "unique_training_seconds_before_selection": 30 * len(state["seen_training_ids"]),
              "validation_mask_identity": "ID and seed fixed before training", "downstream_test_source_used": False,
              "itu_sources_used": 0, "behavior_labels_used": 0,
              "exclusion_scope": "This new continuation only; original upstream pretraining membership was not independently re-audited"}
    if result["weights_sha256"] == frozen.WEIGHTS_SHA256:
        raise ValueError("SSL artifact is unchanged from the original checkpoint")
    if Path(args.output).exists():
        raise ValueError("Do not overwrite a selected encoder seal")
    dump_json(Path(args.output), result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("train", "probe"):
        cmd = sub.add_parser(name)
        for key in ("manifest", "artifacts", "source-allowlist", "corpus-summary", "protocol", "output", "cbramod-upstream", "cbramod-weights"):
            cmd.add_argument("--" + key, required=True)
        cmd.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        cmd.add_argument("--epochs", type=int, default=10)
        cmd.add_argument("--max-seconds", type=float, default=1800)
        cmd.add_argument("--lr", type=float, default=1e-4)
        cmd.add_argument("--seed", type=int, default=20261122)
        cmd.add_argument("--precision", choices=("float32", "bfloat16"), default="bfloat16")
        cmd.add_argument("--memory-budget-gib", type=float, default=21)
        cmd.add_argument("--cpu-threads", type=int, default=2)
        cmd.add_argument("--batch-size", type=int, default=8)
        cmd.add_argument("--probe-steps", type=int, default=50)
        cmd.add_argument("--capacity-selection")
        cmd.add_argument("--resume")
        cmd.set_defaults(func=train)
    cap = sub.add_parser("select-capacity")
    cap.add_argument("--protocol", required=True)
    cap.add_argument("--reports", nargs="+", required=True)
    cap.add_argument("--output", required=True)
    cap.set_defaults(func=select_capacity)
    seal = sub.add_parser("seal-encoder")
    seal.add_argument("--ssl-run", required=True)
    seal.add_argument("--output", required=True)
    seal.set_defaults(func=seal_encoder)
    return result


def main():
    args = parser().parse_args()
    preserve = (args.command in ("train", "probe") and Path(args.output).exists() and any(Path(args.output).iterdir()) and not args.resume)
    try:
        summary = args.func(args)
        return 2 if summary.get("status") == "stopped_memory_budget" else 0
    except Exception as exc:
        if args.command in ("train", "probe") and not preserve:
            dump_json(Path(args.output) / "failure.json", {"error_type": type(exc).__name__, "message": str(exc),
                       "partial_update_checkpoint_not_created": True})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
