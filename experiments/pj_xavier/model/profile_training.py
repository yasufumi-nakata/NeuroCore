#!/usr/bin/env python3
"""Bounded training-only resource probe; discard weights and never score held-out data.

Run each batch size in a fresh process. This probe selects the longest tokenized
training examples to avoid claiming a short-prompt peak as a worst-case result.
The actual training run must start from its fixed seed, not these probe weights.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

from eeg_llm import (EEGCausalModel, EEGStore, apply_allocator_limit, dump_json,
                     hardware, load_backbone, read_rows, select_device, sha256,
                     sync, tokenized_answer, tokenized_prompt, update_memory)


def profile(args):
    config = vars(args).copy()
    device = select_device(args.device)
    result = {"status": "started", "configuration": config,
              "partitions_used": ["train"], "validation_or_test_scored": False,
              "weights_retained": False, "scope": "training-only resource diagnostic; no model performance estimate"}
    peak = {}
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        result["hardware"] = hardware(device)
        result["allocator_limit"] = apply_allocator_limit(device, args.memory_budget_gib)
        rows = [r for r in read_rows(Path(args.manifest)) if r["split"] == "train"]
        if not rows:
            raise ValueError("No training rows")
        result["manifest_sha256"] = sha256(Path(args.manifest))
        classes = sorted({r["label"] for r in rows})
        base, tokenizer, backbone = load_backbone(config, device)
        result["backbone"] = backbone
        store = EEGStore(args.samples)
        channels = store.get(rows[0]).shape[0]
        model = EEGCausalModel(base, channels, args.prefix_tokens, args.encoder_width).to(device)
        optimizer = torch.optim.AdamW(model.connector.parameters(), lr=args.lr, weight_decay=.01)
        encoded = [(r, tokenized_prompt(tokenizer, r, classes, args.mode, args.max_prompt_tokens),
                    tokenized_answer(tokenizer, r["label"])) for r in rows]
        encoded.sort(key=lambda entry: (-(len(entry[1]) + len(entry[2])), entry[0]["id"]))
        batch = [encoded[i % len(encoded)] for i in range(args.batch_size)]
        result["training_example_ids"] = [r[0]["id"] for r in batch]
        result["sequence_lengths_including_prefix"] = [len(r[1]) + len(r[2]) + args.prefix_tokens for r in batch]
        eeg = torch.stack([store.get(r[0]) for r in batch]).to(device)
        times = []
        losses = []
        model.train()
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            sync(device)
            started = time.perf_counter()
            loss = model.loss(eeg, [r[1] for r in batch], [r[2] for r in batch], args.mode)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite diagnostic loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.connector.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite diagnostic gradient")
            optimizer.step()
            sync(device)
            times.append(time.perf_counter() - started)
            losses.append(float(loss.detach().cpu()))
            update_memory(peak, device)
            if max(peak.values()) > args.memory_budget_gib * 1024 ** 3:
                raise MemoryError("Observed memory exceeds diagnostic budget")
        result.update(status="measured", completed_steps=len(times), losses=losses,
                      step_seconds=times, warm_median_step_seconds=statistics.median(times[1:] or times),
                      warm_examples_per_second=args.batch_size / statistics.median(times[1:] or times))
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        if device.type == "cuda" and torch.cuda.is_available():
            update_memory(peak, device)
        result["memory"] = peak
        dump_json(Path(args.output), result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("training_example_ids", "configuration")}, ensure_ascii=False))
    return 0 if result["status"] == "measured" else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--mode", choices=("fused", "eeg_only", "context_only"), default="fused")
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--samples", type=int, default=448)
    p.add_argument("--prefix-tokens", type=int, default=8)
    p.add_argument("--encoder-width", type=int, default=64)
    p.add_argument("--max-prompt-tokens", type=int, default=256)
    p.add_argument("--max-base-params", type=int, default=9_000_000_000)
    p.add_argument("--memory-budget-gib", type=float, default=21)
    p.add_argument("--seed", type=int, default=20261122)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=3)
    args = p.parse_args()
    if args.batch_size < 1 or args.steps < 1:
        p.error("Positive batch size and step count required")
    raise SystemExit(profile(args))
