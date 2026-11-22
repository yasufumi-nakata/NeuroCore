#!/usr/bin/env python3
"""EEG prefix -> causal LLM training pilot, with honest held-out evaluation.

This is an engineering implementation inspired by modality connectors, not a
reproduction of UniMind and not a general brain-to-agent model. No custom remote
model code is executed. Outputs are categorical action likelihoods, not OS calls.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import resource
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


SCHEMA_VERSION = "xavier-eeg-llm-v1"
GOLD_SOURCES = {"recorded_event", "human_verified", "dataset_annotation"}
MODES = ("eeg_only", "fused", "context_only")


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["_manifest_eeg_path"] = row["eeg_path"]
        p = Path(row["eeg_path"])
        if not p.is_absolute():
            row["eeg_path"] = str((path.parent / p).resolve())
    return rows


def eeg_artifact_manifest(rows: list[dict]) -> dict:
    """Seal actual prepared EEG bytes, with portable keys from the source JSONL."""
    hashed = {}
    artifacts = {}
    for row in rows:
        path = str(Path(row["eeg_path"]).resolve())
        if path not in hashed:
            p = Path(path)
            hashed[path] = {"sha256": sha256(p), "bytes": p.stat().st_size}
        artifacts[row.get("_manifest_eeg_path", row["eeg_path"])] = hashed[path]
    return dict(sorted(artifacts.items()))


def validate_rows(rows: list[dict], split_policy: str, mode: str, allow_synthetic_diagnostic=False) -> dict:
    """Fail closed on observed schema/split leakage; cannot detect mislabeled clocks."""
    if not rows:
        raise ValueError("No examples")
    ids: set[str] = set()
    groups: dict[tuple, set] = {}
    recordings: dict[tuple, list] = {}
    physical_sources: dict[str, list] = {}
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for key in ("id", "dataset", "subject", "session", "recording_id", "eeg_path",
                    "window_start_s", "window_end_s", "decision_time_s", "label",
                    "label_source", "split"):
            if key not in row:
                raise ValueError(f"Missing {key} in {row.get('id', '<unknown>')}")
        if row["id"] in ids:
            raise ValueError("Duplicate sample id")
        ids.add(row["id"])
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Split must be train, val or test")
        start, end, decision = (float(row[k]) for k in
                               ("window_start_s", "window_end_s", "decision_time_s"))
        if not all(math.isfinite(x) for x in (start, end, decision)) or not 0 <= start < end <= decision:
            raise ValueError(f"Invalid causal EEG window: {row['id']}")
        if "event_time_s" in row:
            event = float(row["event_time_s"])
            lead = float(row.get("minimum_lead_s", 0))
            if not math.isfinite(event) or not math.isfinite(lead) or lead < 0 or decision > event - lead + 1e-8:
                raise ValueError(f"Post-event EEG or violated pre-action guard: {row['id']}")
        if "feature_available_time_s" in row:
            available = float(row["feature_available_time_s"])
            if not math.isfinite(available) or available > decision + 1e-8:
                raise ValueError(f"Feature availability is later than decision time: {row['id']}")
        if row.get("data_origin") == "synthetic" and not allow_synthetic_diagnostic:
            raise ValueError("Synthetic examples are restricted to the offline random-backbone diagnostic")
        if (row["split"] != "train" and row["label_source"] not in GOLD_SOURCES
                and not (allow_synthetic_diagnostic and row["label_source"] == "synthetic")):
            raise ValueError(f"Non-gold evaluation label: {row['id']}")
        if mode != "eeg_only":
            if not row.get("environment_text", "").strip():
                raise ValueError("Real environment_text required for fused/context_only")
            if row.get("environment_source") != "predecision_harness":
                raise ValueError("Environment source must be predecision_harness")
            et = float(row.get("environment_time_s", float("nan")))
            if not math.isfinite(et) or et > start:
                raise ValueError("Environment snapshot must precede EEG input window")
        dataset_key = row.get("canonical_dataset_id", row["dataset"])
        group = (dataset_key, row.get("canonical_subject_id", row["subject"]))
        if split_policy == "session":
            group += (row["session"],)
        if split_policy == "dataset":
            group = (dataset_key,)
        groups.setdefault(group, set()).add(row["split"])
        recording = (dataset_key, row.get("canonical_subject_id", row["subject"]), row["session"], row["recording_id"])
        recordings.setdefault(recording, []).append((start, end, row["split"], row["id"]))
        if row.get("eeg_source_sha256"):
            physical_sources.setdefault(row["eeg_source_sha256"], []).append((start, end, row["split"], row["id"]))
        counts.setdefault(row["split"], {})
        counts[row["split"]][row["label"]] = counts[row["split"]].get(row["label"], 0) + 1
    if split_policy != "within_subject_time" and any(len(v) > 1 for v in groups.values()):
        raise ValueError(f"{split_policy} group crosses train/val/test")
    for windows in physical_sources.values():
        active = []
        for current in sorted(windows):
            active = [prev for prev in active if prev[1] > current[0]]
            if any(prev[2] != current[2] for prev in active):
                raise ValueError("Identical physical EEG source hash has overlapping intervals across data splits")
            active.append(current)
    for windows in recordings.values():
        ordered = sorted(windows)
        active: list[tuple] = []
        for current in ordered:
            active = [prev for prev in active if prev[1] > current[0]]
            if any(prev[2] != current[2] for prev in active):
                raise ValueError("Overlapping original EEG intervals cross data splits")
            active.append(current)
        if split_policy == "within_subject_time":
            rank = {"train": 0, "val": 1, "test": 2}
            if any(rank[a[2]] > rank[b[2]] for a, b in zip(ordered, ordered[1:])):
                raise ValueError("Within-subject splits must follow chronological train -> val -> test order")
    if not all(counts.get(split) for split in ("train", "val", "test")):
        raise ValueError("Nonempty train, val and test required")
    train_labels = set(counts["train"])
    if len(train_labels) < 2 or any(set(counts[s]) - train_labels for s in ("val", "test")):
        raise ValueError("Need >=2 training classes and no unseen evaluation classes")
    return {"split_policy": split_policy, "counts": counts, "synthetic_diagnostic": allow_synthetic_diagnostic,
            "subjects": len({(r.get('canonical_dataset_id', r['dataset']), r.get('canonical_subject_id', r['subject'])) for r in rows}),
            "physical_source_hashes_checked": len(physical_sources),
            "scope": "within_subject_session_pilot" if split_policy == "within_subject_time" else split_policy,
            "checks": "metadata-based; semantic intent correctness and source clock correctness are not proven"}


class EEGStore:
    """Lazy file-backed arrays. No pickle loading; input must be [channels,time]."""
    def __init__(self, samples: int):
        self.samples = samples
        self.cache: dict[str, Any] = {}

    def get(self, row: dict) -> torch.Tensor:
        key = row["eeg_path"]
        if key not in self.cache:
            self.cache[key] = np.load(key, allow_pickle=False, mmap_mode="r")
        value = self.cache[key]
        if isinstance(value, np.lib.npyio.NpzFile):
            value = value[row.get("eeg_key", "eeg")]
        if "eeg_index" in row:
            value = value[int(row["eeg_index"])]
        x = np.array(value, dtype=np.float32, copy=True)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 8 or not np.isfinite(x).all():
            raise ValueError(f"Invalid EEG shape/values for {row['id']}: {x.shape}")
        x -= x.mean(axis=-1, keepdims=True)
        x /= np.maximum(x.std(axis=-1, keepdims=True), 1e-8)
        t = torch.from_numpy(x)
        # This interpolation standardizes tensor length, not a claimed signal-processing
        # anti-aliased resampler. Data preparation must resample to a common rate first.
        if t.shape[1] != self.samples:
            raise ValueError(f"Expected {self.samples} samples, got {t.shape[1]}; prepare common-rate windows first")
        return t


class PortableAdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    """Keep native CPU/CUDA pooling; use equivalent bins on MPS.

    MPS adaptive pooling rejects some non-divisible lengths (for example the
    110 feature samples produced from 438 EEG samples into eight prefix tokens).
    Slice means use the same floor/ceil bin boundaries, remain on the device,
    preserve gradients, and add no state-dict parameters or checkpoint keys.
    """
    @staticmethod
    def slice_average(x: torch.Tensor, bins: int) -> torch.Tensor:
        length = x.shape[-1]
        return torch.stack([
            x[..., (index * length) // bins:((index + 1) * length + bins - 1) // bins].mean(dim=-1)
            for index in range(bins)
        ], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "mps":
            return self.slice_average(x, int(self.output_size))
        return super().forward(x)


class EEGPrefix(nn.Module):
    def __init__(self, channels: int, hidden_size: int, prefix_tokens: int = 8, width: int = 64):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, width, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=7, stride=2, padding=3),
            nn.GELU(), PortableAdaptiveAvgPool1d(prefix_tokens),
        )
        self.projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden_size))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.projection(self.temporal(eeg).transpose(1, 2))


class EEGCausalModel(nn.Module):
    def __init__(self, base: nn.Module, channels: int, prefix_tokens: int = 8, width: int = 64):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        hidden = int(self.base.get_input_embeddings().weight.shape[1])
        self.connector = EEGPrefix(channels, hidden, prefix_tokens, width)
        self.prefix_tokens = prefix_tokens
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen backbone remains deterministic. LoRA dropout may stay active when
        # adapters are explicitly requested; parameter freezing alone does not do it.
        if not any(p.requires_grad for p in self.base.parameters()):
            self.base.eval()
        return self

    def pack(self, eeg: torch.Tensor, prompt_ids: list[list[int]], answer_ids: list[list[int]],
             mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode == "context_only":
            eeg = torch.zeros_like(eeg)
        prefix = self.connector(eeg)
        sequences, masks, labels = [], [], []
        embed = self.base.get_input_embeddings()
        for i, (prompt, answer) in enumerate(zip(prompt_ids, answer_ids)):
            if not prompt or not answer:
                raise ValueError("Prompt and answer must be nonempty")
            ids = torch.tensor(prompt + answer, dtype=torch.long, device=eeg.device)
            # Do not wrap the base forward in no_grad: gradients must flow from the
            # answer loss through the frozen transformer into the EEG embeddings.
            text = embed(ids)
            sequence = torch.cat((prefix[i].to(text.dtype), text), dim=0)
            label = torch.full((len(sequence),), -100, dtype=torch.long, device=eeg.device)
            label[self.prefix_tokens + len(prompt):] = torch.tensor(answer, device=eeg.device)
            sequences.append(sequence)
            masks.append(torch.ones(len(sequence), dtype=torch.long, device=eeg.device))
            labels.append(label)
        return (pad_sequence(sequences, batch_first=True),
                pad_sequence(masks, batch_first=True),
                pad_sequence(labels, batch_first=True, padding_value=-100))

    def logits_and_targets(self, eeg, prompts, answers, mode):
        embeds, mask, labels = self.pack(eeg, prompts, answers, mode)
        outputs = self.base(inputs_embeds=embeds, attention_mask=mask, use_cache=False, return_dict=True)
        return outputs.logits[:, :-1, :], labels[:, 1:]

    def loss(self, eeg, prompts, answers, mode):
        logits, targets = self.logits_and_targets(eeg, prompts, answers, mode)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)

    def answer_log_likelihood(self, eeg, prompts, answers, mode):
        logits, targets = self.logits_and_targets(eeg, prompts, answers, mode)
        token_loss = F.cross_entropy(logits.float().transpose(1, 2), targets, ignore_index=-100, reduction="none")
        return -token_loss.sum(dim=1)


class ByteTokenizer:
    """Offline diagnostic only. Never substitute for a pretrained tokenizer."""
    eos_token_id, bos_token_id, pad_token_id = 1, 1, 0
    def encode(self, text, add_special_tokens=False):
        return [int(x) + 2 for x in text.encode("utf-8")]


def tokenized_prompt(tokenizer, row: dict, classes: list[str], mode: str, max_prompt_tokens: int):
    instruction = "Classify the recorded upcoming action. Allowed labels: " + ", ".join(classes) + ".\n"
    context = ""
    if mode != "eeg_only":
        context = "Environment before the EEG window:\n" + row["environment_text"] + "\n"
    prompt = instruction + context + "Action label:"
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_prompt_tokens:
        raise ValueError(f"Prompt too long ({len(ids)}>{max_prompt_tokens}); explicitly prepare a shorter causal context")
    if getattr(tokenizer, "bos_token_id", None) is not None:
        ids = [tokenizer.bos_token_id] + ids
    return ids


def tokenized_answer(tokenizer, label: str):
    result = tokenizer.encode(" " + label, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        result.append(tokenizer.eos_token_id)
    return result


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return torch.device(requested)


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_sample(device: torch.device) -> dict:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024)}
    if device.type == "cuda":
        result["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    elif device.type == "mps":
        result["mps_sampled_allocated_bytes"] = int(torch.mps.current_allocated_memory())
        result["mps_sampled_driver_bytes"] = int(torch.mps.driver_allocated_memory())
    return result


def update_memory(peak: dict, device: torch.device) -> None:
    for key, value in memory_sample(device).items():
        peak[key] = max(peak.get(key, 0), value)


def hardware(device: torch.device) -> dict:
    result = {"platform": platform.platform(), "machine": platform.machine(),
              "processor": platform.processor(), "device": str(device),
              "python": platform.python_version(), "torch": torch.__version__,
              "transformers": importlib.metadata.version("transformers")}
    if device.type == "cuda":
        result["accelerator"] = torch.cuda.get_device_name(device)
        result["accelerator_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
    elif device.type == "mps":
        result["mps_recommended_max_memory_bytes"] = int(torch.mps.recommended_max_memory())
    if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
        result["system_memory_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return result


def apply_allocator_limit(device: torch.device, budget_gib: float) -> dict:
    """Limit the framework GPU allocator; this is not a whole-machine RAM quota."""
    if not math.isfinite(budget_gib) or budget_gib <= 0:
        raise ValueError("Positive finite memory budget required")
    requested = int(budget_gib * (1024 ** 3))
    if device.type == "mps":
        recommended = int(torch.mps.recommended_max_memory())
        if recommended <= 0:
            raise ValueError("Cannot determine MPS recommended memory for an allocator limit")
        fraction = min(requested / recommended, 1.)
        torch.mps.set_per_process_memory_fraction(fraction)
        return {"kind": "mps_allocator", "fraction": fraction,
                "allocator_limit_bytes": int(recommended * fraction), "recommended_bytes": recommended,
                "scope": "PyTorch MPS allocator only; CPU allocations and other processes are outside this quota"}
    if device.type == "cuda":
        # set_per_process_memory_fraction rejects torch.device('cuda') without an
        # index in the verified CUDA runtime, even though other CUDA APIs accept it.
        index = device.index if device.index is not None else torch.cuda.current_device()
        total = int(torch.cuda.get_device_properties(index).total_memory)
        fraction = min(requested / total, 1.)
        torch.cuda.set_per_process_memory_fraction(fraction, index)
        return {"kind": "cuda_allocator", "fraction": fraction,
                "allocator_limit_bytes": int(total * fraction), "device_total_bytes": total,
                "scope": "PyTorch CUDA allocator only; host RAM and external allocations are outside this quota"}
    return {"kind": "sampled_rss_only", "allocator_limit_bytes": None,
            "scope": "CPU RSS monitoring is not an allocator-enforced hard limit"}


def load_backbone(config: dict, device: torch.device):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU pilot requires float32; choose an explicitly supported accelerator for reduced precision")
    if config["model"] == "offline-tiny-random":
        # Seeded random weights establish plumbing only, never pretrained performance.
        torch.manual_seed(config["seed"])
        base = GPT2LMHeadModel(GPT2Config(vocab_size=258, n_positions=512, n_ctx=512,
                                        n_embd=32, n_layer=2, n_head=4,
                                        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
        return base.to(device), ByteTokenizer(), {"base_parameter_count": sum(p.numel() for p in base.parameters()),
                                                 "resolved_revision": None, "pretrained": False}
    # Transformers 5 may omit config._commit_hash. Pin the Hub revision before any
    # config, weight or tokenizer read, using a concrete snapshot path or Hub SHA.
    requested_revision = config.get("revision")
    if requested_revision and re.fullmatch(r"[0-9a-fA-F]{40}", requested_revision):
        resolved_revision = requested_revision.lower()
    else:
        from huggingface_hub import HfApi, hf_hub_download
        cached_config = hf_hub_download(config["model"], "config.json", revision=requested_revision)
        parts = Path(cached_config).parts  # Do not resolve symlinks into the blobs directory.
        resolved_revision = None
        for index, part in enumerate(parts[:-1]):
            if part == "snapshots" and re.fullmatch(r"[0-9a-fA-F]{40}", parts[index + 1]):
                resolved_revision = parts[index + 1].lower()
                break
        if resolved_revision is None:
            resolved_revision = HfApi().model_info(config["model"], revision=requested_revision).sha
        if not resolved_revision or not re.fullmatch(r"[0-9a-fA-F]{40}", resolved_revision):
            raise ValueError("Could not resolve an immutable Hugging Face model commit; refusing an unpinned load")
    source_config = AutoConfig.from_pretrained(config["model"], revision=resolved_revision, trust_remote_code=False)
    with torch.device("meta"):
        meta = AutoModelForCausalLM.from_config(source_config, trust_remote_code=False)
    params = sum(p.numel() for p in meta.parameters())
    del meta
    if params > config["max_base_params"]:
        raise ValueError(f"Backbone has {params:,} parameters; explicit limit is {config['max_base_params']:,}")
    weight_bytes = params * torch.tensor([], dtype=dtype).element_size()
    if weight_bytes > config["memory_budget_gib"] * (1024 ** 3) * 0.80:
        raise ValueError("Backbone weights alone exceed 80% of configured memory budget; reduce size/precision or use larger hardware")
    base = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=resolved_revision, torch_dtype=dtype,
        trust_remote_code=False, attn_implementation="eager",
    ).to(device)
    tok = AutoTokenizer.from_pretrained(config["model"], revision=resolved_revision, trust_remote_code=False)
    return base, tok, {"base_parameter_count": params,
                       "resolved_revision": resolved_revision,
                       "pretrained": True, "estimated_weight_bytes": weight_bytes}


def attach_lora(model: EEGCausalModel, rank: int):
    if rank:
        from peft import LoraConfig, get_peft_model
        model.base = get_peft_model(model.base, LoraConfig(
            r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM", bias="none"))


def class_metrics(labels: list[str], probabilities: np.ndarray, classes: list[str]) -> dict:
    actual = np.array([classes.index(x) for x in labels])
    predicted = probabilities.argmax(1)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(matrix, (actual, predicted), 1)
    support, pred_support = matrix.sum(1), matrix.sum(0)
    recalls = np.divide(matrix.diagonal(), support, out=np.zeros(len(classes), dtype=float), where=support != 0)
    f1 = np.divide(2 * matrix.diagonal(), support + pred_support,
                   out=np.zeros(len(classes), dtype=float), where=(support + pred_support) != 0)
    return {"n": len(labels), "accuracy": float((predicted == actual).mean()),
            "balanced_accuracy": float(recalls[support > 0].mean()), "macro_f1": float(f1.mean()),
            "all_classes_present": bool((support > 0).all()),
            "classes": classes, "class_support": support.tolist(), "confusion_matrix": matrix.tolist(),
            "categorical_nll": float(-np.log(np.maximum(probabilities[np.arange(len(actual)), actual], 1e-30)).mean()),
            "probability_note": "renormalized conditional candidate-sequence likelihoods; calibration not established"}


def within_recording_derangement(rows: list[dict]) -> list[dict]:
    """Change EEG identity without changing participant/session/recording domain."""
    groups = {}
    for index, row in enumerate(rows):
        key = (row.get("canonical_dataset_id", row["dataset"]), row["subject"], row["session"], row["recording_id"])
        groups.setdefault(key, []).append(index)
    if any(len(indices) < 2 for indices in groups.values()):
        raise ValueError("A recording group has fewer than two evaluation examples")
    result = list(rows)
    for indices in groups.values():
        offset = max(1, len(indices) // 2)
        for local_index, original_index in enumerate(indices):
            result[original_index] = rows[indices[(local_index + offset) % len(indices)]]
    return result


@torch.no_grad()
def evaluate(model, tokenizer, rows, classes, store, config, device, checkpoint,
             condition="normal", output: Path | None = None, include_truth=True):
    model.eval()
    source_rows = list(rows)
    if condition == "shuffled_eeg":
        # Labels/context remain original. Subject/session changes must not serve
        # as the perturbation, because that would confound the EEG-use test.
        try:
            source_rows = within_recording_derangement(rows)
        except ValueError as exc:
            return {"status": "blocked", "reason": str(exc)}, []
    predictions = []
    all_probs = []
    candidate_answers = [tokenized_answer(tokenizer, c) for c in classes]
    for original, source in zip(rows, source_rows):
        eeg = store.get(source).unsqueeze(0).to(device)
        prompt = tokenized_prompt(tokenizer, original, classes, config["mode"], config["max_prompt_tokens"])
        sync(device)
        started = time.perf_counter()
        score_chunks = []
        candidate_batch_size = int(config.get("candidate_batch_size", 4))
        for start in range(0, len(classes), candidate_batch_size):
            answers = candidate_answers[start:start + candidate_batch_size]
            score_chunks.append(model.answer_log_likelihood(eeg.repeat(len(answers), 1, 1),
                                                            [prompt] * len(answers), answers, config["mode"]))
            if max(memory_sample(device).values()) > config["memory_budget_gib"] * (1024 ** 3):
                raise RuntimeError("memory_budget_exceeded_during_evaluation; stopping before another candidate batch")
        scores = torch.cat(score_chunks)
        probs = torch.softmax(scores.float(), dim=0).cpu().numpy()
        sync(device)
        elapsed = (time.perf_counter() - started) * 1000
        index = int(probs.argmax())
        all_probs.append(probs)
        prediction = {"id": original["id"], "predicted_label": classes[index],
                      "probabilities": {c: float(p) for c, p in zip(classes, probs)},
                      "condition": condition, "latency_ms": elapsed, "checkpoint": checkpoint,
                      "mode": config["mode"], "eeg_source_id": source["id"],
                      "abstain": bool(probs[index] < config["abstain_threshold"]),
                      "decision_time_s": original["decision_time_s"],
                      "action_executed": False}
        if condition == "shuffled_eeg":
            prediction["perturbation"] = {"method": "nonzero_cyclic_shift", "eeg_source_id": source["id"],
                                          "original_eeg_id": original["id"], "fixed_points": 0,
                                          "within": "dataset,participant,session,recording"}
        if include_truth:
            prediction["true_label"] = original["label"]
        predictions.append(prediction)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(p, ensure_ascii=False, allow_nan=False) + "\n" for p in predictions))
    metrics = class_metrics([r["label"] for r in rows], np.asarray(all_probs), classes) if include_truth else {}
    latencies = [p["latency_ms"] for p in predictions]
    metrics.update({"status": "measured", "condition": condition,
                    "latency_ms_p50": float(np.median(latencies)), "latency_ms_p95": float(np.quantile(latencies, .95)),
                    "latency_scope": "candidate scoring, includes accelerator sync; excludes EEG loading/preprocessing and harness execution",
                    "abstain_threshold": config["abstain_threshold"],
                    "coverage": float(np.mean([not p["abstain"] for p in predictions])),
                    "task_completion": None, "false_actions_per_minute": None,
                    "agentic_benchmark_status": "unavailable_without_real_harness_episodes_and_idle_periods"})
    return metrics, predictions


def save_checkpoint(path: Path, model, optimizer, config, epoch, global_step, classes, data_sha, best_score,
                    *, epoch_completed=True, next_batch_position=0, eeg_artifacts=None):
    trainable_names = {n for n, p in model.base.named_parameters() if p.requires_grad}
    base_trainable = {n: p.detach().cpu() for n, p in model.base.named_parameters() if n in trainable_names}
    state = {"schema": SCHEMA_VERSION, "connector": {k: v.detach().cpu() for k, v in model.connector.state_dict().items()},
             "trainable_base": base_trainable, "optimizer": optimizer.state_dict(),
             "config": config, "epoch": epoch, "global_step": global_step, "classes": classes,
             "manifest_sha256": data_sha, "best_score": best_score,
             "torch_rng": torch.get_rng_state(),
             "epoch_completed": epoch_completed, "next_batch_position": next_batch_position,
             "eeg_artifacts": eeg_artifacts,
             "cuda_rng_states": torch.cuda.get_rng_state_all() if next(model.connector.parameters()).device.type == "cuda" else None,
             "mps_rng_state": torch.mps.get_rng_state() if next(model.connector.parameters()).device.type == "mps" else None,
             "purpose": "categorical observed-action pilot; no claim of general intent/agent capability"}
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def restore_weights(path: Path, model):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != SCHEMA_VERSION:
        raise ValueError("Checkpoint schema mismatch")
    model.connector.load_state_dict(state["connector"])
    params = dict(model.base.named_parameters())
    with torch.no_grad():
        for key, value in state["trainable_base"].items():
            if key not in params:
                raise ValueError(f"Missing adapter parameter {key}")
            params[key].copy_(value)
    return state


def train(args):
    output = Path(args.output).resolve()
    if (output / "test_metrics.json").exists():
        raise ValueError("This run already consumed its test set; use a new predefined experiment, not repeated test-guided tuning")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_config.json").exists() and not args.resume:
        raise ValueError("Existing run directory: use --resume with its checkpoint or choose a fresh output")
    config = vars(args).copy()
    config.pop("func", None)
    config["manifest"] = str(Path(args.manifest).resolve())
    config["output"] = str(output)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = select_device(args.device)
    rows = read_rows(Path(args.manifest))
    split_info = validate_rows(rows, args.split_policy, args.mode,
                               allow_synthetic_diagnostic=args.model == "offline-tiny-random")
    data_sha = sha256(Path(args.manifest))
    artifacts = eeg_artifact_manifest(rows)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    classes = sorted({r["label"] for r in train_rows})
    store = EEGStore(args.samples)
    config["channels"] = int(store.get(train_rows[0]).shape[0])
    if any(int(store.get(r).shape[0]) != config["channels"] for r in rows):
        raise ValueError("This pilot requires a common explicit channel montage")
    if args.resume:
        resume_header = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume_header.get("eeg_artifacts") != artifacts:
            raise ValueError("Prepared EEG bytes changed, or legacy checkpoint lacks the required EEG artifact hashes")
        previous_revision = resume_header["config"].get("revision")
        if args.revision and args.revision != previous_revision:
            raise ValueError("Resume must use the checkpoint's recorded immutable backbone revision")
        if args.model != "offline-tiny-random" and not previous_revision:
            raise ValueError("Legacy checkpoint lacks a resolved backbone revision")
        config["revision"] = previous_revision
    base, tokenizer, backbone = load_backbone(config, device)
    # Resolve the mutable model alias once; every reload uses the exact revision.
    config["revision"] = backbone["resolved_revision"]
    model = EEGCausalModel(base, config["channels"], args.prefix_tokens, args.encoder_width).to(device)
    attach_lora(model, args.lora_rank)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01)
    best_score, epoch_start, global_step, batch_start = -math.inf, 0, 0, 0
    if args.resume:
        saved = restore_weights(Path(args.resume), model)
        if saved["manifest_sha256"] != data_sha or saved["classes"] != classes:
            raise ValueError("Resume manifest/classes changed")
        if saved.get("eeg_artifacts") != artifacts:
            raise ValueError("Prepared EEG bytes changed, or legacy checkpoint lacks the required EEG artifact hashes")
        for k in ("model", "revision", "mode", "samples", "channels", "prefix_tokens", "encoder_width", "dtype", "lora_rank", "seed", "batch_size"):
            if saved["config"][k] != config[k]:
                raise ValueError(f"Resume configuration mismatch: {k}")
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
        if device.type == "mps" and saved.get("mps_rng_state") is not None:
            torch.mps.set_rng_state(saved["mps_rng_state"])
        completed = bool(saved.get("epoch_completed", True))
        best_score, epoch_start, global_step = saved["best_score"], saved["epoch"] + int(completed), saved["global_step"]
        batch_start = 0 if completed else saved["next_batch_position"]
    dump_json(output / "run_config.json", config)
    dump_json(output / "data_validation.json", {**split_info, "manifest_sha256": data_sha})
    dump_json(output / "eeg_artifacts.json", artifacts)
    dump_json(output / "training_membership.json", {
        "training_ids": sorted(r["id"] for r in train_rows),
        "validation_ids": sorted(r["id"] for r in val_rows),
        "test_ids": sorted(r["id"] for r in test_rows),
        "manifest_sha256": data_sha,
        "training_ids_sha256": hashlib.sha256("\n".join(sorted(r["id"] for r in train_rows)).encode()).hexdigest(),
        "model_selection_partition": "validation", "test_used_for_tuning": False,
        "selection_claim_scope": "within_this_run; external human decisions are not observable",
    })
    dump_json(output / "hardware.json", {**hardware(device), **backbone,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad)})
    started, step_times, peak, reports = time.monotonic(), [], {}, []
    status = "completed_epochs"
    last_checkpoint = Path(args.resume) if args.resume else None
    for epoch in range(epoch_start, args.epochs):
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        next_position = batch_start if epoch == epoch_start else 0
        model.train()
        for pos in range(next_position, len(order), args.batch_size):
            if args.max_steps and global_step >= args.max_steps:
                status = "stopped_max_steps"
                break
            if args.max_seconds and time.monotonic() - started >= args.max_seconds:
                status = "stopped_time_budget"
                break
            batch = [train_rows[i] for i in order[pos:pos + args.batch_size]]
            x = torch.stack([store.get(r) for r in batch]).to(device)
            prompts = [tokenized_prompt(tokenizer, r, classes, args.mode, args.max_prompt_tokens) for r in batch]
            answers = [tokenized_answer(tokenizer, r["label"]) for r in batch]
            optimizer.zero_grad(set_to_none=True)
            sync(device)
            tick = time.perf_counter()
            loss = model.loss(x, prompts, answers, args.mode)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite connector/adapter gradient")
            optimizer.step()
            sync(device)
            step_times.append(time.perf_counter() - tick)
            losses.append(float(loss.detach().cpu()))
            global_step += 1
            next_position = min(pos + len(batch), len(order))
            update_memory(peak, device)
            observed = max(peak.values(), default=0)
            if observed > args.memory_budget_gib * (1024 ** 3):
                status = "stopped_memory_budget"
                break
        epoch_completed = next_position >= len(order)
        checkpoint = output / f"epoch_{epoch + 1:03d}_step_{global_step:07d}.pt"
        last_checkpoint = checkpoint
        # Persist training before a potentially expensive validation pass. A failed
        # evaluation must not erase completed optimization work.
        save_checkpoint(checkpoint, model, optimizer, config, epoch, global_step, classes, data_sha, best_score,
                        epoch_completed=epoch_completed, next_batch_position=next_position, eeg_artifacts=artifacts)
        if status == "stopped_memory_budget":
            val = {"status": "skipped", "reason": "memory_budget_exceeded"}
            score, improved = None, False
        else:
            val, _ = evaluate(model, tokenizer, val_rows, classes, store, config, device, checkpoint.name,
                              output=output / f"val_epoch_{epoch + 1:03d}_step_{global_step:07d}.jsonl")
            score = val["balanced_accuracy"]
            improved = score > best_score
        if improved:
            best_score = score
        save_checkpoint(checkpoint, model, optimizer, config, epoch, global_step, classes, data_sha, best_score,
                        epoch_completed=epoch_completed, next_batch_position=next_position, eeg_artifacts=artifacts)
        dump_json(checkpoint.with_suffix(".metadata.json"), {
            "checkpoint_sha256": sha256(checkpoint), "manifest_sha256": data_sha,
            "global_step": global_step, "epoch": epoch + 1,
            "training_ids_sha256": hashlib.sha256("\n".join(sorted(r["id"] for r in train_rows)).encode()).hexdigest(),
        })
        if status != "stopped_memory_budget" and (improved or not (output / "best.pt").exists()):
            # Atomic copy to avoid a reader seeing a partially written checkpoint.
            import shutil
            shutil.copyfile(checkpoint, output / "best.pt.tmp")
            os.replace(output / "best.pt.tmp", output / "best.pt")
        record = {"epoch": epoch + 1, "global_step": global_step,
                  "mean_train_token_loss": float(np.mean(losses)) if losses else None,
                  "val": val, "checkpoint": checkpoint.name,
                  "epoch_completed": epoch_completed, "next_batch_position": next_position,
                  "best_by": "validation balanced accuracy; earliest checkpoint wins ties"}
        reports.append(record)
        with (output / "history.jsonl").open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps({"epoch": epoch + 1, "global_step": global_step,
                          "train_loss": record["mean_train_token_loss"],
                          "val_balanced_accuracy": score, "status": status}), flush=True)
        if status != "completed_epochs":
            break
    if last_checkpoint is None or not last_checkpoint.exists():
        raise ValueError("No checkpoint created")
    best_path = output / "best.pt"
    test_result = {"status": "not_run", "reason": "--finalize-test required after validation-stage model selection"}
    if status == "stopped_memory_budget":
        test_result = {"status": "blocked", "reason": "memory_budget_exceeded; no validation or test was run after the limit"}
    elif args.finalize_test:
        if not best_path.exists():
            raise ValueError("Final test requires a validation-selected checkpoint")
        restore_weights(output / "best.pt", model)
        test, _ = evaluate(model, tokenizer, test_rows, classes, store, config, device, "best.pt",
                           output=output / "test_predictions.jsonl")
        shuffled, _ = evaluate(model, tokenizer, test_rows, classes, store, config, device, "best.pt", "shuffled_eeg",
                               output=output / "test_shuffled_predictions.jsonl")
        test_result = {"normal": test, "shuffled_eeg": shuffled,
                       "eeg_increment_balanced_accuracy": test["balanced_accuracy"] - shuffled.get("balanced_accuracy", test["balanced_accuracy"]),
                       "status": "single_final_test_run", "heldout_scope": split_info["scope"]}
        dump_json(output / "test_metrics.json", test_result)
    update_memory(peak, device)
    elapsed = time.monotonic() - started
    summary = {"status": status, "global_step": global_step, "elapsed_seconds": elapsed,
               "train_step_seconds_p50": float(np.median(step_times)) if step_times else None,
               "train_step_seconds_p95": float(np.quantile(step_times, .95)) if step_times else None,
               "memory": peak, "memory_budget_gib": args.memory_budget_gib,
               "observed_memory_within_budget": max(peak.values(), default=0) <= args.memory_budget_gib * (1024 ** 3),
               "memory_note": "CUDA reports allocator peaks; MPS values are sampled, not guaranteed peak; RSS is process high-water mark",
               "best_checkpoint": "best.pt" if best_path.exists() else None,
               "last_checkpoint": last_checkpoint.name, "test": test_result,
               "checkpoint_sha256": sha256(best_path) if best_path.exists() else None, "manifest_sha256": data_sha,
               "model_selection_partition": "validation", "test_used_for_tuning": False,
               "model_kind": "pretrained_llm_with_trained_eeg_connector" if backbone["pretrained"] else "offline_random_backbone_plumbing_test",
               "agentic_readiness": "not_validated", "os_actions_executed": 0}
    dump_json(output / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    return summary


def predict(args):
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"].copy()
    training_dtype = config["dtype"]
    config["device"] = args.device
    if args.memory_budget_gib is not None:
        config["memory_budget_gib"] = args.memory_budget_gib
    if args.candidate_batch_size is not None:
        if args.candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be positive")
        config["candidate_batch_size"] = args.candidate_batch_size
    if args.dtype is not None:
        config["dtype"] = args.dtype
    device = select_device(args.device)
    allocator = apply_allocator_limit(device, config["memory_budget_gib"])
    base, tokenizer, backbone = load_backbone(config, device)
    model = EEGCausalModel(base, config["channels"], config["prefix_tokens"], config["encoder_width"]).to(device)
    attach_lora(model, config["lora_rank"])
    restore_weights(Path(args.checkpoint), model)
    rows = read_rows(Path(args.manifest))
    if args.max_examples:
        if args.max_examples < 1:
            raise ValueError("max_examples must be nonnegative")
        rows = rows[:args.max_examples]
    if not rows:
        raise ValueError("No prediction rows")
    for row in rows:
        if float(row["window_end_s"]) > float(row["decision_time_s"]):
            raise ValueError("Prediction contains post-decision EEG")
        if config["mode"] != "eeg_only":
            if row.get("environment_source") != "predecision_harness" or float(row.get("environment_time_s", math.inf)) > float(row["window_start_s"]):
                raise ValueError("Invalid predecision environment context")
    metrics, _ = evaluate(model, tokenizer, rows, state["classes"], EEGStore(config["samples"]),
                          config, device, str(args.checkpoint), output=Path(args.output), include_truth=False)
    metrics.update({"n_examples": len(rows), "hardware": hardware(device), "backbone": backbone,
                    "runtime_dtype": config["dtype"], "checkpoint_training_dtype": training_dtype,
                    "memory_budget_gib": config["memory_budget_gib"],
                    "candidate_batch_size": config.get("candidate_batch_size", 4),
                    "memory": memory_sample(device), "allocator_limit": allocator,
                    "checkpoint_sha256": sha256(Path(args.checkpoint)), "checkpoint_modified": False})
    dump_json(Path(args.output).with_suffix(".metadata.json"), metrics)
    print(json.dumps(metrics, ensure_ascii=False, allow_nan=False))
    return metrics


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train")
    t.add_argument("--manifest", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    t.add_argument("--revision")
    t.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    t.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    t.add_argument("--mode", choices=MODES, default="eeg_only")
    t.add_argument("--split-policy", choices=("subject", "session", "dataset", "within_subject_time"), default="subject")
    t.add_argument("--samples", type=int, required=True)
    t.add_argument("--prefix-tokens", type=int, default=8)
    t.add_argument("--encoder-width", type=int, default=64)
    t.add_argument("--max-prompt-tokens", type=int, default=256)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--candidate-batch-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=20261122)
    t.add_argument("--max-steps", type=int, default=0)
    t.add_argument("--max-seconds", type=float, default=1800)
    t.add_argument("--max-base-params", type=int, default=2_000_000_000)
    t.add_argument("--memory-budget-gib", type=float, default=12)
    t.add_argument("--lora-rank", type=int, default=0)
    t.add_argument("--abstain-threshold", type=float, default=.8)
    t.add_argument("--resume")
    t.add_argument("--finalize-test", action="store_true")
    t.set_defaults(func=train)
    q = commands.add_parser("predict")
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--manifest", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    q.add_argument("--memory-budget-gib", type=float)
    q.add_argument("--candidate-batch-size", type=int)
    q.add_argument("--dtype", choices=("float32", "bfloat16", "float16"))
    q.add_argument("--max-examples", type=int, default=0, help="Limit a capacity diagnostic; 0 scores all supplied examples")
    q.set_defaults(func=predict)
    return p


def main():
    args = parser().parse_args()
    try:
        result = args.func(args)
        if isinstance(result, dict) and result.get("status") == "stopped_memory_budget":
            raise SystemExit(2)
    except Exception as exc:
        if hasattr(args, "output") and getattr(args, "command", "") == "train":
            # A blocked run is reviewable, never silently converted into a CPU success.
            dump_json(Path(args.output) / "failure.json", {"status": "failed", "type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
