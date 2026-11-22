"""Portable phase2 inference, resource accounting and frozen-LLM routing."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import resource
import sys
import time

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from eegmmidb_phase2.data_contract import TASKS, require, sha256
from eegmmidb_phase2.model import (
    ARCHITECTURE, CLASS_TOKEN_IDS, FusedDecoder, ROUTE_SYSTEM,
    ScratchDecoder, load_backbone, load_tokenizer, restore_learned, static_prompt,
)

CHECKPOINT_SCHEMA = "xavier-eegmmidb-phase2-checkpoint-v1"
SCIENTIFIC_FILES = ("__init__.py", "data_contract.py", "model.py", "runtime.py", "train_models.py")


def source_seals():
    root = Path(__file__).resolve().parent
    return {name: sha256(root / name) for name in SCIENTIFIC_FILES}


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    require(name in ("cpu", "mps", "cuda"), "Unsupported execution device")
    if name == "cuda":
        require(torch.cuda.is_available(), "CUDA is unavailable")
    if name == "mps":
        require(torch.backends.mps.is_available(), "MPS is unavailable")
    return torch.device(name)


def dtype_value(name, device):
    require(name in ("float32", "bfloat16"), "Only measured float32/BF16 paths are admitted")
    return torch.float32 if name == "float32" else torch.bfloat16


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def hardware(device):
    try:
        import psutil
        virtual = psutil.virtual_memory()
        total, available = int(virtual.total), int(virtual.available)
    except ImportError:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        available = None
    value = {
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "device": str(device),
        "system_memory_bytes": total, "available_host_memory_bytes_at_observation": available,
    }
    if device.type == "cuda":
        info = torch.cuda.get_device_properties(device)
        value.update(accelerator=info.name, accelerator_memory_bytes=info.total_memory)
    elif device.type == "mps":
        value["mps_recommended_max_memory_bytes"] = torch.mps.recommended_max_memory()
    return value


def apply_budget(device, gib):
    require(0 < gib <= 21, "Resource budget must remain at most21 GiB")
    amount = int(gib * 1024**3)
    if device.type == "cuda":
        capacity = torch.cuda.get_device_properties(device).total_memory
        require(amount <= capacity, "Budget exceeds physical CUDA memory")
        torch.cuda.set_per_process_memory_fraction(amount / capacity, device)
        torch.cuda.reset_peak_memory_stats(device)
        return {"kind": "cuda_allocator", "budget_bytes": amount, "capacity_bytes": capacity}
    if device.type == "mps":
        recommended = torch.mps.recommended_max_memory()
        torch.mps.set_per_process_memory_fraction(min(amount / recommended, 1.0))
        return {
            "kind": "mps_allocator", "budget_bytes": min(amount, recommended),
            "recommended_bytes": recommended,
            "scope": "PyTorch MPS allocator; host allocations and other applications excluded",
        }
    return {"kind": "CPU_observation_only", "budget_bytes": amount, "hard_allocator_limit": False}


def memory_sample(device):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024)}
    if device.type == "cuda":
        result.update(
            cuda_allocated_bytes=torch.cuda.memory_allocated(device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(device),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    elif device.type == "mps":
        result.update(
            mps_sampled_allocated_bytes=torch.mps.current_allocated_memory(),
            mps_sampled_driver_bytes=torch.mps.driver_allocated_memory(),
        )
    return result


def within_budget(sample, gib, device):
    if device.type == "cuda":
        value = max(sample.get("cuda_peak_allocated_bytes", 0), sample.get("cuda_peak_reserved_bytes", 0))
    elif device.type == "mps":
        value = max(sample.get("mps_sampled_allocated_bytes", 0), sample.get("mps_sampled_driver_bytes", 0))
    else:
        value = sample["process_peak_rss_bytes"]
    return value <= gib * 1024**3


class ModelRuntime:
    def __init__(self, model, kind, tokenizer, device, config, checkpoint_sha256, allocator):
        self.model = model.eval()
        self.kind, self.tokenizer, self.device = kind, tokenizer, device
        self.config, self.checkpoint_sha256 = config, checkpoint_sha256
        self.allocator = allocator
        self.memory_observations = []
        self.route_records = []
        self.router_base = model.base if kind == "fused" else None
        self.prompt_cache = {}

    def _prompts(self, families, condition):
        values = []
        for task in families:
            key = (task, condition == "eeg_only")
            if key not in self.prompt_cache:
                self.prompt_cache[key] = self.tokenizer.encode(
                    static_prompt(task, condition), add_special_tokens=True
                )
            values.append(self.prompt_cache[key])
        return values

    @torch.inference_mode()
    def predict_batch(self, eeg, task_families, condition="fused"):
        require(condition in ("fused", "context_only", "eeg_only", "eeg_shuffled", "eeg_time_shifted"), "Unknown condition")
        tensor = torch.as_tensor(eeg, dtype=torch.float32, device=self.device)
        require(len(tensor) == len(task_families) and len(tensor) > 0, "Empty or mismatched batch")
        require(all(task in TASKS for task in task_families), "Unknown task family")
        prompts = self._prompts(task_families, condition) if self.kind == "fused" else None
        synchronize(self.device)
        start = time.perf_counter()
        if self.kind == "scratch":
            logits = self.model(tensor, task_families, condition)
        else:
            logits = self.model(tensor, prompts, condition)
        probability = torch.softmax(logits.float(), dim=-1)
        synchronize(self.device)
        elapsed = (time.perf_counter() - start) * 1000
        sample = memory_sample(self.device)
        self.memory_observations.append(sample)
        require(within_budget(sample, self.config["runtime_memory_budget_gib"], self.device), "Inference memory budget exceeded")
        output = []
        for task, values in zip(task_families, probability.cpu().tolist(), strict=True):
            labels = TASKS[task]
            predicted = labels[int(np.argmax(values))]
            output.append({
                "task_family": task,
                "predicted_label": predicted,
                "probabilities": dict(zip(labels, values, strict=True)),
                "condition": condition,
                "checkpoint_sha256": self.checkpoint_sha256,
                "latency_ms": elapsed / len(task_families),
                "batch_latency_ms": elapsed,
                "batch_size": len(task_families),
                "latency_scope": "amortized model call including EEG encoder; excludes array I/O, transfer before timer, prompt encoding and harness",
                "abstain": max(values) < self.config["acceptance_threshold"],
                "abstain_threshold": self.config["acceptance_threshold"],
                "probabilities_are_calibrated_intent_confidence": False,
            })
        return output

    def _ensure_router(self):
        if self.router_base is None:
            self.router_base, self.tokenizer, evidence = load_backbone(
                self.device, dtype_value(self.config["runtime_dtype"], self.device),
                local_files_only=self.config.get("local_files_only", False),
            )
            self.config["routing_backbone_evidence"] = evidence
        self.router_base.eval()
        require(all(not parameter.requires_grad for parameter in self.router_base.parameters()), "Routing backbone must remain frozen")

    def _route_prompt(self, text):
        require(isinstance(text, str), "Routing input must be a string")
        messages = [{"role": "system", "content": ROUTE_SYSTEM}, {"role": "user", "content": text}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        require(0 < len(ids) <= 256, "Routing prompt exceeds fixed256-token bound")
        return ids

    @torch.inference_mode()
    def route_text(self, prompt, max_new_tokens=8):
        """Greedy complete continuation from frozen base, without EEG prefix."""
        require(isinstance(max_new_tokens, int) and 1 <= max_new_tokens <= 8, "Routing output bound changed")
        self._ensure_router()
        ids = self._route_prompt(prompt)
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        synchronize(self.device)
        start = time.perf_counter()
        generated = self.router_base.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs),
            do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        synchronize(self.device)
        elapsed = (time.perf_counter() - start) * 1000
        new_ids = generated[0, len(ids):].cpu().tolist()
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        from eegmmidb_phase2.data_contract import object_hash
        sample = memory_sample(self.device)
        require(within_budget(sample, self.config["runtime_memory_budget_gib"], self.device), "Routing memory budget exceeded")
        self.memory_observations.append(sample)
        self.route_records.append({
            "prompt_sha256": object_hash({"system": ROUTE_SYSTEM, "user": prompt}),
            "output_text": text, "generated_token_ids": new_ids, "latency_ms": elapsed,
            "EEG_prefix_used": False, "new_parameter_updates": 0,
            "max_new_tokens": max_new_tokens, "do_sample": False,
        })
        return text

    @torch.inference_mode()
    def score_routes(self, prompts, symbols=(" X", " Y")):
        """Auxiliary restricted next-token routing; not the primary greedy output."""
        self._ensure_router()
        candidate = [self.tokenizer.encode(symbol, add_special_tokens=False) for symbol in symbols]
        require(len(candidate) == 2 and all(len(value) == 1 for value in candidate), "Routing candidates must be two single tokens")
        sequences = [self._route_prompt(prompt) for prompt in prompts]
        require(bool(sequences), "No routing prompts")
        maximum = max(map(len, sequences))
        ids = torch.full((len(sequences), maximum), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for index, sequence in enumerate(sequences):
            ids[index, :len(sequence)] = torch.tensor(sequence, device=self.device)
            mask[index, :len(sequence)] = 1
        result = self.router_base(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = result.logits[torch.arange(len(sequences), device=self.device), torch.tensor([len(value)-1 for value in sequences], device=self.device)]
        probability = torch.softmax(logits[:, [value[0] for value in candidate]].float(), -1).cpu().tolist()
        return [
            {"symbol": symbols[int(np.argmax(value))].strip(), "probabilities": dict(zip([x.strip() for x in symbols], value, strict=True)), "auxiliary_restricted_scoring": True}
            for value in probability
        ]

    def resource_summary(self):
        samples = self.memory_observations + [memory_sample(self.device)]
        keys = set().union(*(sample.keys() for sample in samples))
        return {
            "hardware": hardware(self.device),
            "memory": {key: max(sample.get(key, 0) for sample in samples) for key in keys},
            "allocator": self.allocator,
            "MPS_values_are_samples_not_true_peaks": self.device.type == "mps",
            "routing_generation_calls": len(self.route_records),
            "os_actions_executed": 0,
        }


def load_runtime(run_dir, device="auto", memory_budget_gib=8, batch_size=None, dtype=None,
                 expected_checkpoint_sha256=None, local_files_only=False):
    """Public evaluator API. Batch size is advisory; caller controls actual batches."""
    import json

    run_dir = Path(run_dir).resolve()
    checkpoint = run_dir / "best.pt"
    checkpoint_sha = sha256(checkpoint)
    summary = json.loads((run_dir / "run_summary.json").read_text())
    require(checkpoint_sha == summary["best_checkpoint_sha256"], "Run summary/best checkpoint mismatch")
    if expected_checkpoint_sha256:
        require(checkpoint_sha == expected_checkpoint_sha256, "Evaluation checkpoint seal changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(state.get("schema") == CHECKPOINT_SCHEMA, "Unsupported checkpoint schema")
    require(state["source_sha256"] == source_seals(), "Phase2 scientific source files changed")
    require(state["config"]["architecture"] == ARCHITECTURE, "Architecture identity changed")
    require(state["config"]["task_mapping"] == {key: list(value) for key, value in TASKS.items()}, "Task labels changed")
    require(state["training_membership_sha256"] == sha256(run_dir / "training_membership.json"), "Training membership changed")
    require(state["config"] == json.loads((run_dir / "run_config.json").read_text()), "Scientific config changed")
    device = select_device(device)
    allocator = apply_budget(device, memory_budget_gib)
    torch.set_num_threads(2)
    config = dict(state["config"])
    config.update(
        runtime_memory_budget_gib=memory_budget_gib,
        runtime_device=str(device),
        runtime_dtype=dtype or ("bfloat16" if device.type != "cpu" else "float32"),
        runtime_batch_size=batch_size,
        local_files_only=local_files_only,
    )
    if state["kind"] == "scratch":
        model = ScratchDecoder().to(device)
        tokenizer, token_audit = load_tokenizer(local_files_only=local_files_only)
        config["routing_tokenizer_evidence"] = token_audit
    else:
        base, tokenizer, evidence = load_backbone(
            device, dtype_value(config["runtime_dtype"], device),
            local_files_only=local_files_only,
        )
        model = FusedDecoder(base)
        model.encoder.to(device)
        model.projector.to(device)
        config["backbone_evidence"] = evidence
        require(tuple(state["config"]["class_token_ids"]) == CLASS_TOKEN_IDS, "A/B candidate mapping changed")
    restore_learned(model, state["kind"], state["learned"])
    return ModelRuntime(model, state["kind"], tokenizer, device, config, checkpoint_sha, allocator)

