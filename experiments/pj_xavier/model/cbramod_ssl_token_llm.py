#!/usr/bin/env python3
"""Frozen SSL-adapted CBraMod -> same channel projection -> frozen SmolLM.

This separate continuation uses the same fixed alignment settings after unlabeled EEG SSL. The
three allowed next-token logits are used by both cross-entropy training and
evaluation. It does not alter eeg_llm.py, its checkpoints, or the v3 manifest.
The official EEG encoder runs explicitly on CPU/float32 on every platform.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from scipy.signal import resample_poly


CORE_SHA256 = "e30811b027414426cdfbed14a822c27695af2ab1be00106d89fe32a5fb740d2f"
CORE_PATH = Path(__file__).with_name("eeg_llm.py")
if hashlib.sha256(CORE_PATH.read_bytes()).hexdigest() != CORE_SHA256:
    raise RuntimeError("Frozen shared EEG utility module changed")
_spec = importlib.util.spec_from_file_location("xavier_frozen_core_for_cbramod", CORE_PATH)
core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = core
_spec.loader.exec_module(core)

SCHEMA_VERSION = "xavier-cbramod-ssl-symbol-llm-v1"
UPSTREAM_COMMIT = "b9e961003214326972c567eff390e75b0287e32a"
WEIGHTS_REVISION = "500543c7e30bda1b22bfd51a49301b238dee21fd"
INITIAL_WEIGHTS_SHA256 = "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"
SOURCE_SHA256 = {
    "models/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models/cbramod.py": "caa7c5a0e4acd488a6f625b56f7b54dc4e7190d6b68ef5b76dff2cef86874a14",
    "models/criss_cross_transformer.py": "90e90b8c98ca7eb543c44e32afc29674d3f71cb090692ad9f82db3a43edf334c",
}
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MODEL_REVISION = "31b70e2e869a7173562077fd711b654946d38674"
MANIFEST_SHA256 = "030e8550bd6f4c79005722eb617e9b64f5f80e87b77866b7f17bec4e7e2a1093"
CLASSES = ["ai_prompt_submit", "keyboard", "mouse_click"]
SYMBOLS = [" A", " B", " C"]
TOKEN_IDS = [330, 389, 340]
CHANNELS = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]
MODES = ("fused", "context_only", "eeg_only")

# Public evaluator API; these shared functions are imported without monkeypatching.
read_rows = core.read_rows
validate_rows = core.validate_rows
eeg_artifact_manifest = core.eeg_artifact_manifest
select_device = core.select_device
apply_allocator_limit = core.apply_allocator_limit
memory_sample = core.memory_sample
hardware = core.hardware
sha256 = core.sha256
dump_json = core.dump_json
sync = core.sync


def identity(config):
    seal = config.get("ssl_encoder_seal", {})
    required = ("weights_sha256", "ssl_best_checkpoint_sha256", "ssl_protocol_sha256",
                "ssl_source_sha256", "ssl_manifest_sha256", "ssl_source_allowlist_sha256")
    if seal.get("schema") != "xavier-selected-ssl-encoder-seal-v1":
        raise ValueError("An explicit selected SSL encoder seal is required")
    for key in required:
        value = seal.get(key, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Missing immutable SSL provenance: " + key)
    if seal.get("initial_weights_sha256") != INITIAL_WEIGHTS_SHA256 or seal["weights_sha256"] == INITIAL_WEIGHTS_SHA256:
        raise ValueError("This experiment requires updated SSL weights from the official initial checkpoint")
    if seal.get("selection_partition") != "portable_SSL_validation" or seal.get("downstream_test_source_used") is not False:
        raise ValueError("SSL selection/source partition does not satisfy the registered protocol")
    return {"source_commit": UPSTREAM_COMMIT, "source_sha256": SOURCE_SHA256,
            "initial_weights_revision": WEIGHTS_REVISION, "initial_weights_sha256": INITIAL_WEIGHTS_SHA256,
            "weights_sha256": seal["weights_sha256"], "ssl_encoder_seal": seal,
            "encoder_device": "cpu", "encoder_dtype": "float32", "encoder_frozen": True}


def latest_second(values, *, sample_rate_hz=256, unit="V", channel_order=CHANNELS):
    """Use only the recorded final second. No outside-window padding or z-score."""
    values = np.asarray(values)
    if unit != "V" or sample_rate_hz != 256 or list(channel_order) != CHANNELS:
        raise ValueError("Explicit V, 256 Hz, and fixed eight-channel order required")
    if values.ndim != 2 or values.shape != (8, 448) or not np.isfinite(values).all():
        raise ValueError("Expected finite prepared 8 x 448 EEG in volts")
    # Upstream downstream loaders use microvolts / 100 = volts * 1e4.
    # Polyphase FIR uses only this extracted past second and its fixed zero pad.
    result = resample_poly(values[:, -256:].astype(np.float32) * 1e4, 25, 32, axis=-1)
    if result.shape != (8, 200) or not np.isfinite(result).all():
        raise ValueError("Invalid resampled CBraMod input")
    return np.ascontiguousarray(result, dtype=np.float32)


class EEGStore:
    """Frozen v3 volts -> final one second -> official encoder input scaling."""
    def __init__(self, samples):
        if samples != 448:
            raise ValueError("Source windows must remain the frozen 448-sample v3 windows")
        self.samples, self.cache = samples, {}

    def get(self, row):
        key = row["eeg_path"]
        if key not in self.cache:
            self.cache[key] = np.load(key, allow_pickle=False, mmap_mode="r")
        x = self.cache[key]
        if isinstance(x, np.lib.npyio.NpzFile):
            x = x[row.get("eeg_key", "eeg")]
        if "eeg_index" in row:
            x = x[int(row["eeg_index"])]
        if row.get("sample_rate_hz") != 256:
            raise ValueError("Manifest sample rate differs from fixed v3 input")
        # v3 channel field is frozen in the sealed manifest; require it when present.
        channel_order = row.get("channels", CHANNELS)
        if isinstance(channel_order, int):
            if channel_order != 8:
                raise ValueError("Wrong channel count")
            channel_order = CHANNELS
        return torch.from_numpy(latest_second(x, sample_rate_hz=row["sample_rate_hz"],
                                             channel_order=channel_order))


def load_adapted_encoder(config):
    upstream = Path(os.environ.get("XAVIER_CBRAMOD_UPSTREAM", config.get("cbramod_upstream", ""))).resolve()
    weights = Path(os.environ.get("XAVIER_CBRAMOD_SSL_WEIGHTS", config.get("cbramod_weights", ""))).resolve()
    if sha256(weights) != identity(config)["weights_sha256"]:
        raise ValueError("Selected SSL CBraMod checkpoint bytes differ from the sealed experiment")
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("Official CBraMod source commit differs")
    for name, expected in SOURCE_SHA256.items():
        p = upstream / name
        if sha256(p) != expected:
            raise ValueError("Official CBraMod model source modified: " + name)
        original = subprocess.check_output(["git", "-C", str(upstream), "show", commit + ":" + name])
        if p.read_bytes() != original:
            raise ValueError("Working tree does not match pinned upstream source")
    # Upstream uses an absolute models.* import. Refuse accidental reuse of a
    # different already-imported package rather than silently overriding it.
    if "models" in sys.modules:
        origin = getattr(sys.modules["models"], "__file__", None)
        if origin is None or Path(origin).resolve() != (upstream / "models/__init__.py").resolve():
            raise ValueError("Conflicting models package is already imported")
    sys.path.insert(0, str(upstream))
    try:
        from models.cbramod import CBraMod
    finally:
        sys.path.remove(str(upstream))
    encoder = CBraMod().cpu().float().eval()
    encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    encoder.proj_out = nn.Identity()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


class FrozenCPUEncoder(nn.Module):
    """Registered, auditable parameters that explicitly never move to MPS/CUDA."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.cpu().float().eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def _apply(self, fn, recurse=True):
        # Parent model.to(device/dtype) must not move or quantize this CPU branch.
        return self

    def train(self, mode=True):
        super().train(False)
        self.encoder.eval()
        return self

    def forward(self, eeg):
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != (8, 200):
            raise ValueError("CBraMod receives batch x 8 x 200 only")
        if not bool(torch.isfinite(eeg).all()):
            raise ValueError("Nonfinite CBraMod input")
        with torch.no_grad():
            output = self.encoder(eeg.detach().to(device="cpu", dtype=torch.float32).unsqueeze(2))
        if tuple(output.shape) != (eeg.shape[0], 8, 1, 200) or not bool(torch.isfinite(output).all()):
            raise ValueError("Unexpected official CBraMod output")
        return output[:, :, 0, :]


def prompt_text(row, mode):
    if mode not in MODES:
        raise ValueError("Unknown input condition")
    text = ("Classify the recorded upcoming action. Reply with one code.\n"
            "A = AI prompt submit; B = keyboard; C = mouse click.\n")
    if mode != "eeg_only":
        if not row.get("environment_text"):
            raise ValueError("Pre-window coarse context is required")
        text += "Environment before the EEG window:\n" + row["environment_text"] + "\n"
    return text + "Action code:"


def tokenized_prompt(tokenizer, row, classes, mode, max_prompt_tokens):
    if list(classes) != CLASSES:
        raise ValueError("Fixed A/B/C mapping requires the three preregistered classes")
    ids = tokenizer.encode(prompt_text(row, mode), add_special_tokens=False)
    if getattr(tokenizer, "bos_token_id", None) is not None:
        ids = [tokenizer.bos_token_id] + ids
    if not ids or len(ids) > max_prompt_tokens:
        raise ValueError("Prompt is empty or exceeds the explicitly fixed token budget")
    return ids


def audit_tokenizer(tokenizer, train_rows):
    if any(r["split"] != "train" for r in train_rows):
        raise ValueError("Token-boundary audit is restricted to the train partition")
    actual = [tokenizer.encode(s, add_special_tokens=False) for s in SYMBOLS]
    if actual != [[x] for x in TOKEN_IDS]:
        raise ValueError("Fixed Smol A/B/C single-token IDs changed")
    checks, max_length = 0, 0
    for row in train_rows:
        for mode in ("fused", "eeg_only"):
            text = prompt_text(row, mode)
            prefix = tokenizer.encode(text, add_special_tokens=False)
            max_length = max(max_length, len(prefix) + int(tokenizer.bos_token_id is not None))
            for symbol, token in zip(SYMBOLS, TOKEN_IDS):
                if tokenizer.encode(text + symbol, add_special_tokens=False) != prefix + [token]:
                    raise ValueError("Prompt/answer BPE boundary changes tokenization")
                checks += 1
    return {"partition": "train", "records": len(train_rows), "boundary_checks": checks,
            "max_prompt_tokens_with_bos": max_length,
            "mapping": {c: {"text": s, "token_id": t} for c, s, t in zip(CLASSES, SYMBOLS, TOKEN_IDS)}}


def load_backbone(config, device):
    if config["model"] != MODEL_ID or config.get("revision") != MODEL_REVISION:
        raise ValueError("This experiment pins the SmolLM2-1.7B backbone and revision")
    for key, expected in (("eeg_encoder_identity", identity(config)), ("class_token_ids", TOKEN_IDS),
                          ("core_source_sha256", CORE_SHA256),
                          ("model_module_sha256", sha256(Path(__file__)))):
        if config.get(key) != expected:
            raise ValueError("Runtime checkpoint/config identity changed: " + key)
    base, tokenizer, evidence = core.load_backbone(config, device)
    if [tokenizer.encode(s, add_special_tokens=False) for s in SYMBOLS] != [[x] for x in TOKEN_IDS]:
        raise ValueError("Fixed tokenizer output mapping changed")
    base._xavier_symbol_config = dict(config)
    evidence.update(cbramod=identity(config), core_source_sha256=CORE_SHA256,
                    candidate_scoring="one forward; three restricted next-token logits",
                    eeg_encoder_input_seconds=1, source_window_seconds=1.75)
    return base, tokenizer, evidence


class EEGCausalModel(nn.Module):
    def __init__(self, base, channels, prefix_tokens=8, width=200, *, encoder=None, token_ids=TOKEN_IDS):
        super().__init__()
        if channels != 8 or prefix_tokens != 8 or width != 200:
            raise ValueError("Fixed encoder: eight 200-dimensional channel tokens")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        if encoder is None:
            encoder = load_adapted_encoder(getattr(base, "_xavier_symbol_config", {}))
        self.eeg_encoder = FrozenCPUEncoder(encoder)
        hidden = int(base.get_input_embeddings().weight.shape[1])
        self.connector = nn.Sequential(nn.LayerNorm(200), nn.Linear(200, hidden))
        self.prefix_tokens, self.token_ids = prefix_tokens, list(token_ids)
        if len(self.token_ids) != 3 or len(set(self.token_ids)) != 3:
            raise ValueError("Three distinct next-token classes required")
        if max(self.token_ids) >= base.get_input_embeddings().weight.shape[0]:
            raise ValueError("Output token is outside frozen backbone vocabulary")

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        self.eeg_encoder.eval()
        return self

    def class_logits(self, eeg, prompts, mode):
        if mode not in MODES or len(prompts) != len(eeg) or any(not p for p in prompts):
            raise ValueError("Invalid prompt batch or input condition")
        if mode == "context_only":
            eeg = torch.zeros_like(eeg)
        features = self.eeg_encoder(eeg)
        parameter = next(self.connector.parameters())
        prefix = self.connector(features.to(device=parameter.device, dtype=parameter.dtype))
        embedding = self.base.get_input_embeddings()
        sequences, masks, last = [], [], []
        for index, prompt in enumerate(prompts):
            ids = torch.tensor(prompt, device=embedding.weight.device, dtype=torch.long)
            text = embedding(ids)
            sequence = torch.cat((prefix[index].to(text.dtype), text), dim=0)
            sequences.append(sequence)
            masks.append(torch.ones(len(sequence), device=text.device, dtype=torch.long))
            last.append(len(sequence) - 1)
        # No answer token is fed to the model. The last prompt position predicts
        # the same A/B/C logits for both optimization and all evaluation modes.
        result = self.base(inputs_embeds=pad_sequence(sequences, batch_first=True),
                           attention_mask=pad_sequence(masks, batch_first=True),
                           use_cache=False, return_dict=True)
        index = torch.arange(len(eeg), device=result.logits.device)
        logits = result.logits[index, torch.tensor(last, device=index.device), :]
        return logits[:, self.token_ids].float()

    def loss(self, eeg, prompts, targets, mode):
        return F.cross_entropy(self.class_logits(eeg, prompts, mode), targets)


def attach_lora(model, rank):
    if rank:
        raise ValueError("LoRA is not part of this separate frozen-backbone experiment")


@torch.no_grad()
def evaluate(model, tokenizer, rows, classes, store, config, device, checkpoint,
             condition="normal", output=None, include_truth=True):
    if list(classes) != CLASSES:
        raise ValueError("Class mapping changed")
    model.eval()
    source_rows = list(rows)
    if condition == "shuffled_eeg":
        try:
            source_rows = core.within_recording_derangement(rows)
        except ValueError as exc:
            return {"status": "blocked", "reason": str(exc)}, []
    elif condition != "normal":
        raise ValueError("Unknown perturbation condition")
    predictions, all_probs = [], []
    for original, source in zip(rows, source_rows):
        eeg = store.get(source).unsqueeze(0)  # CPU EEG encoder is explicit.
        prompt = tokenized_prompt(tokenizer, original, classes, config["mode"], config["max_prompt_tokens"])
        sync(device)
        start = time.perf_counter()
        logits = model.class_logits(eeg, [prompt], config["mode"])
        probs = logits.softmax(-1)[0].cpu().numpy()
        sync(device)
        elapsed = 1000 * (time.perf_counter() - start)
        if max(memory_sample(device).values()) > config["memory_budget_gib"] * 1024**3:
            raise RuntimeError("memory_budget_exceeded_during_evaluation")
        selected = int(probs.argmax())
        row = {"id": original["id"], "predicted_label": classes[selected],
               "probabilities": {c: float(p) for c, p in zip(classes, probs)},
               "condition": condition, "latency_ms": elapsed, "checkpoint": checkpoint,
               "mode": config["mode"], "eeg_source_id": source["id"],
               "abstain": bool(probs[selected] < config["abstain_threshold"]),
               "decision_time_s": original["decision_time_s"], "action_executed": False,
               "probability_kind": "restricted_three_next_token_softmax_uncalibrated",
               "candidate_forward_passes": 1}
        if condition == "shuffled_eeg":
            row["perturbation"] = {"method": "nonzero_cyclic_shift", "eeg_source_id": source["id"],
                                   "original_eeg_id": original["id"], "fixed_points": 0,
                                   "within": "dataset,participant,session,recording"}
        if include_truth:
            row["true_label"] = original["label"]
        predictions.append(row)
        all_probs.append(probs)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text("".join(json.dumps(p, allow_nan=False) + "\n" for p in predictions))
    metrics = core.class_metrics([r["label"] for r in rows], np.asarray(all_probs), classes) if include_truth else {}
    latencies = [p["latency_ms"] for p in predictions]
    metrics.update(status="measured", condition=condition, probability_note="restricted A/B/C next-token softmax; natural-use calibration not established",
                   latency_ms_p50=float(np.median(latencies)), latency_ms_p95=float(np.quantile(latencies, .95)),
                   latency_scope="CPU EEG encoder, transfer, projection and one LLM forward; excludes array I/O, resampling, tokenization and harness",
                   abstain_threshold=config["abstain_threshold"], coverage=float(np.mean([not p["abstain"] for p in predictions])),
                   task_completion=None, false_actions_per_minute=None, candidate_forward_passes=1,
                   input_ablation_note="Same selected checkpoint; context-only passes zero EEG through the fixed encoder, a distribution-shift ablation",
                   agentic_benchmark_status="unavailable_without_real_harness_episodes_and_idle_periods")
    return metrics, predictions


def save_checkpoint(path, model, optimizer, config, epoch, global_step, artifacts, best_score,
                    epoch_completed, next_position, elapsed):
    state = {"schema": SCHEMA_VERSION, "connector": {k: v.detach().cpu() for k, v in model.connector.state_dict().items()},
             "trainable_base": {}, "optimizer": optimizer.state_dict(), "config": config,
             "classes": CLASSES, "manifest_sha256": config["manifest_sha256"], "eeg_artifacts": artifacts,
             "eeg_encoder_identity": identity(config), "token_mapping": dict(zip(CLASSES, TOKEN_IDS)),
             "epoch": epoch, "global_step": global_step, "epoch_completed": epoch_completed,
             "next_batch_position": next_position, "best_score": best_score,
             "elapsed_training_seconds": elapsed, "torch_rng": torch.get_rng_state(),
             "cuda_rng_states": torch.cuda.get_rng_state_all() if next(model.connector.parameters()).device.type == "cuda" else None,
             "purpose": "fixed alignment after additional unlabeled EEG SSL; exploratory reused UniBa test; no agent acceptance"}
    temp = Path(path).with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def restore_weights(path, model):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != SCHEMA_VERSION or state.get("eeg_encoder_identity") != identity(state.get("config", {})):
        raise ValueError("Wrong checkpoint schema or encoder identity")
    if state.get("classes") != CLASSES or state.get("token_mapping") != dict(zip(CLASSES, TOKEN_IDS)) or state.get("trainable_base"):
        raise ValueError("Frozen class mapping/backbone contract changed")
    model.connector.load_state_dict(state["connector"], strict=True)
    return state


def frozen_evidence(model):
    return {"eeg_encoder_frozen": all(not p.requires_grad for p in model.eeg_encoder.parameters()),
            "base_frozen": all(not p.requires_grad for p in model.base.parameters()),
            "eeg_encoder_cpu_float32": all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.eeg_encoder.parameters()),
            "eeg_encoder_parameters": sum(p.numel() for p in model.eeg_encoder.parameters()),
            "base_parameters": sum(p.numel() for p in model.base.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def build_config(args):
    config = vars(args).copy()
    config.pop("func", None)
    config["ssl_encoder_seal"] = json.loads(Path(args.ssl_encoder_seal).read_text())
    config["ssl_encoder_seal_sha256"] = sha256(Path(args.ssl_encoder_seal))
    config.update(channels=8, samples=448, prefix_tokens=8, encoder_width=200, lora_rank=0,
                  model=MODEL_ID, revision=MODEL_REVISION, mode="fused", split_policy="subject",
                  max_base_params=2_000_000_000, candidate_batch_size=1, manifest_sha256=MANIFEST_SHA256,
                  eeg_unit="V", eeg_encoder_identity=identity(config), class_token_ids=TOKEN_IDS,
                  core_source_sha256=CORE_SHA256, model_module_sha256=sha256(Path(__file__)),
                  prior_test_results_seen_for_experiment_design=True, exploratory_reuse_of_v3_test=True)
    return config


def validate_protocol(args, protocol):
    if (protocol.get("schema") != "xavier-cbramod-ssl-symbol-experiment-v1"
            or protocol.get("manifest_sha256") != MANIFEST_SHA256
            or protocol.get("model_source_sha256") != sha256(Path(__file__))
            or protocol.get("core_source_sha256") != CORE_SHA256):
        raise ValueError("Registered protocol does not seal this model and source cohort")
    if protocol.get("ssl_encoder_seal_sha256") != sha256(Path(args.ssl_encoder_seal)):
        raise ValueError("Registered SSL encoder selection changed")
    fixed = protocol["training"]
    for key in ("epochs", "lr", "seed", "dtype", "max_prompt_tokens", "abstain_threshold", "cpu_threads"):
        if getattr(args, key) != fixed[key]:
            raise ValueError("Registered training setting changed: " + key)
    if args.batch_size not in protocol["capacity_selection"]["candidate_batches"]:
        raise ValueError("Batch size is not a registered train-only probe candidate")
    if not 0 < args.max_seconds <= fixed["max_seconds"] or args.memory_budget_gib != fixed["memory_budget_gib"]:
        raise ValueError("Registered resource limit changed")
    if args.command == "probe":
        if args.probe_steps != protocol["capacity_selection"]["minimum_steps"]:
            raise ValueError("Capacity probe requires the registered sustained step budget")
    elif args.max_seconds != fixed["max_seconds"]:
        raise ValueError("Main training wall-time budget differs from registration")
    elif not args.capacity_selection:
        raise ValueError("Main training requires the saved train-only batch-selection receipt")
    else:
        receipt = json.loads(Path(args.capacity_selection).read_text())
        if (receipt.get("protocol_sha256") != sha256(Path(args.protocol))
                or receipt.get("selected_batch") != args.batch_size
                or receipt.get("selected_by") != "largest_stable_train_only_candidate"
                or receipt.get("weights_reused") is not False):
            raise ValueError("Invalid capacity-selection provenance")
        eligible = []
        for item in receipt.get("evidence", []):
            path = Path(item["report_path"])
            report_config_path = path.with_name("run_config.json")
            if sha256(path) != item["report_sha256"] or sha256(report_config_path) != item["run_config_sha256"]:
                raise ValueError("Capacity evidence bytes changed after selection")
            probe_config = json.loads(report_config_path.read_text())
            if (probe_config.get("protocol_sha256") != receipt["protocol_sha256"] or probe_config.get("command") != "probe"
                    or probe_config.get("model_module_sha256") != sha256(Path(__file__))):
                raise ValueError("Capacity evidence is not a train-only probe of this protocol")
            if item["eligible"] != capacity_eligible(json.loads(path.read_text()), protocol):
                raise ValueError("Capacity eligibility differs from the measured report")
            if item["eligible"]:
                eligible.append(item["batch"])
        if not eligible or max(eligible) != args.batch_size:
            raise ValueError("Selected batch is not the largest declared eligible measurement")


def train(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output is nonempty; preserve the prior experiment")
    output.mkdir(parents=True, exist_ok=True)
    config = build_config(args)
    config["output"], config["manifest"] = str(output), str(Path(args.manifest).resolve())
    if sha256(Path(args.manifest)) != MANIFEST_SHA256:
        raise ValueError("Only the sealed v3 source cohort is admitted")
    protocol = json.loads(Path(args.protocol).read_text())
    validate_protocol(args, protocol)
    config["protocol_sha256"] = sha256(Path(args.protocol))
    if args.epochs > 30 or args.max_seconds > 900 or args.memory_budget_gib > 21:
        raise ValueError("This experiment is bounded by 30 epochs / 900 s / 21 GiB")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = select_device(args.device)
    allocator = apply_allocator_limit(device, args.memory_budget_gib)
    rows = read_rows(Path(args.manifest))
    validation = validate_rows(rows, "subject", "fused")
    artifacts = eeg_artifact_manifest(rows)
    parts = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    if sorted({r["label"] for r in parts["train"]}) != CLASSES:
        raise ValueError("Training classes differ")
    store = EEGStore(448)
    for row in parts["train"]:
        store.get(row)
    base, tokenizer, backbone = load_backbone(config, device)
    token_audit = audit_tokenizer(tokenizer, parts["train"])
    model = EEGCausalModel(base, 8, 8, 200).to(device)
    optimizer = torch.optim.AdamW(model.connector.parameters(), lr=args.lr, weight_decay=.01)
    best_score, epoch_start, position, step, prior_elapsed = -math.inf, 0, 0, 0, 0.
    if args.resume:
        old = restore_weights(Path(args.resume), model)
        if old["eeg_artifacts"] != artifacts or old["manifest_sha256"] != MANIFEST_SHA256:
            raise ValueError("Resume prepared EEG bytes changed")
        for key in ("seed", "batch_size", "lr", "dtype", "protocol_sha256", "model_module_sha256"):
            if old["config"][key] != config[key]:
                raise ValueError("Resume config changed: " + key)
        optimizer.load_state_dict(old["optimizer"])
        torch.set_rng_state(old["torch_rng"])
        if device.type == "cuda" and old["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(old["cuda_rng_states"])
        epoch_start = old["epoch"] + int(old["epoch_completed"])
        position = 0 if old["epoch_completed"] else old["next_batch_position"]
        step, best_score, prior_elapsed = old["global_step"], old["best_score"], old["elapsed_training_seconds"]
    for name, content in [("run_config.json", config), ("runtime_allocator_limit.json", allocator),
                          ("data_validation.json", validation), ("eeg_artifacts.json", artifacts),
                          ("label_token_audit.json", token_audit),
                          ("hardware.json", {**hardware(device), **backbone, **frozen_evidence(model)})]:
        dump_json(output / name, content)
    membership = {"training_ids": sorted(r["id"] for r in parts["train"]),
                  "validation_ids": sorted(r["id"] for r in parts["val"]),
                  "test_ids": sorted(r["id"] for r in parts["test"]),
                  "manifest_sha256": MANIFEST_SHA256, "model_selection_partition": "validation",
                  "test_used_for_tuning": False,
                  "selection_claim_scope": "within_this_run_only; prior v3 test results motivated this exploratory experiment",
                  "prior_test_results_seen_for_experiment_design": True}
    membership["training_ids_sha256"] = hashlib.sha256("\n".join(membership["training_ids"]).encode()).hexdigest()
    dump_json(output / "training_membership.json", membership)
    started, durations, peak, observations = time.monotonic(), [], {}, []
    status, last_checkpoint = "completed_epochs", None
    for epoch in range(epoch_start, args.epochs):
        order = list(range(len(parts["train"])))
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        model.train()
        for pos in range(position, len(order), args.batch_size):
            if prior_elapsed + time.monotonic() - started >= args.max_seconds:
                status = "stopped_time_budget"; break
            if args.command == "probe" and step >= args.probe_steps:
                status = "completed_train_only_probe"; break
            batch = [parts["train"][i] for i in order[pos:pos + args.batch_size]]
            x = torch.stack([store.get(r) for r in batch])
            prompts = [tokenized_prompt(tokenizer, r, CLASSES, "fused", args.max_prompt_tokens) for r in batch]
            targets = torch.tensor([CLASSES.index(r["label"]) for r in batch], device=device)
            optimizer.zero_grad(set_to_none=True)
            sync(device); tick = time.perf_counter()
            loss = model.loss(x, prompts, targets, "fused")
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite classification loss")
            loss.backward()
            if any(p.grad is not None for p in model.base.parameters()) or any(p.grad is not None for p in model.eeg_encoder.parameters()):
                raise ValueError("Frozen base or EEG encoder received parameter gradients")
            norm = nn.utils.clip_grad_norm_(model.connector.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite projection gradients")
            optimizer.step(); sync(device)
            durations.append(time.perf_counter() - tick)
            losses.append(float(loss.detach().cpu()))
            step += 1; position = min(pos + len(batch), len(order))
            sample = memory_sample(device)
            for key, value in sample.items(): peak[key] = max(peak.get(key, 0), value)
            current = dict(sample)
            if device.type == "cuda":
                current["cuda_reserved_bytes"] = torch.cuda.memory_reserved(device)
                current["cuda_allocated_bytes"] = torch.cuda.memory_allocated(device)
            observations.append({"step": step, "batch_records": len(batch), **current})
            if max(peak.values()) > args.memory_budget_gib * 1024**3:
                status = "stopped_memory_budget"; break
        completed = position == len(order)
        elapsed = prior_elapsed + time.monotonic() - started
        last_checkpoint = output / f"epoch_{epoch + 1:03d}_step_{step:07d}.pt"
        save_checkpoint(last_checkpoint, model, optimizer, config, epoch, step, artifacts, best_score, completed, position, elapsed)
        val, score = {"status": "not_run", "reason": "train-only capacity probe"}, None
        if args.command == "train" and status != "stopped_memory_budget":
            val, _ = evaluate(model, tokenizer, parts["val"], CLASSES, store, config, device, last_checkpoint.name,
                              output=output / f"val_epoch_{epoch + 1:03d}_step_{step:07d}.jsonl")
            score = val["balanced_accuracy"]
            improved = score > best_score
            if improved: best_score = score
            save_checkpoint(last_checkpoint, model, optimizer, config, epoch, step, artifacts, best_score, completed, position,
                            prior_elapsed + time.monotonic() - started)
            if improved or not (output / "best.pt").exists():
                shutil.copyfile(last_checkpoint, output / "best.pt.tmp"); os.replace(output / "best.pt.tmp", output / "best.pt")
        report = {"epoch": epoch + 1, "global_step": step, "mean_train_classification_loss": float(np.mean(losses)) if losses else None,
                  "val": val, "checkpoint": last_checkpoint.name, "epoch_completed": completed,
                  "next_batch_position": position, "best_by": "validation balanced accuracy; earliest tie wins"}
        with (output / "history.jsonl").open("a") as f: f.write(json.dumps(report, allow_nan=False) + "\n")
        print(json.dumps({"epoch": epoch + 1, "global_step": step, "val_balanced_accuracy": score, "status": status}), flush=True)
        if status != "completed_epochs": break
        if args.command == "probe" and step >= args.probe_steps:
            status = "completed_train_only_probe"; break
        position = 0
    core.update_memory(peak, device)
    best = output / "best.pt"
    summary = {"status": status, "global_step": step, "elapsed_seconds": prior_elapsed + time.monotonic() - started,
               "training_loop_seconds_this_process": time.monotonic() - started,
               "train_step_seconds_p50": float(np.median(durations)) if durations else None,
               "train_step_seconds_p95": float(np.quantile(durations, .95)) if durations else None,
               "memory": peak, "memory_budget_gib": args.memory_budget_gib,
               "best_checkpoint": "best.pt" if best.exists() else None,
               "last_checkpoint": last_checkpoint.name if last_checkpoint else None,
               "checkpoint_sha256": sha256(best) if best.exists() else None,
               "manifest_sha256": MANIFEST_SHA256, "model_selection_partition": "validation",
               "test_used_for_tuning": False, "prior_test_results_seen_for_experiment_design": True,
               "test": {"status": "not_run", "reason": "separate gate-controlled final evaluation required"},
               "model_kind": "frozen_ssl_adapted_cbramod_frozen_smollm_same_symbol_projection",
               "agentic_readiness": "not_validated", "os_actions_executed": 0,
               "memory_note": "CUDA allocator peaks measured; MPS readings are samples only; CPU EEG branch and CPU RSS are distinct scopes",
               **frozen_evidence(model)}
    if args.command == "probe":
        tail = observations[-10:]
        key = "cuda_reserved_bytes" if device.type == "cuda" else "mps_sampled_driver_bytes"
        values = [o[key] for o in tail if key in o]
        summary["probe"] = {"training_examples_processed": sum(o["batch_records"] for o in observations),
                            "validation_examples": 0, "test_examples": 0, "requested_steps": args.probe_steps,
                            "completed_requested_steps": step >= args.probe_steps,
                            "at_least_one_train_epoch_equivalent": sum(o["batch_records"] for o in observations) >= len(parts["train"]),
                            "last_ten_reserved_range_bytes": max(values) - min(values) if len(values) == 10 else None,
                            "allocator_stable_last_ten": len(values) == 10 and max(values) - min(values) <= 256 * 1024**2,
                            "weights_must_not_seed_main_training": True}
        dump_json(output / "memory_observations.json", observations)
    dump_json(output / "run_summary.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return summary


def predict(args):
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = dict(state["config"])
    for key in ("device", "memory_budget_gib", "dtype", "cbramod_upstream", "cbramod_weights"):
        if getattr(args, key, None) is not None: config[key] = getattr(args, key)
    device = select_device(config["device"])
    allocator = apply_allocator_limit(device, config["memory_budget_gib"])
    torch.set_num_threads(2)
    base, tokenizer, backbone = load_backbone(config, device)
    model = EEGCausalModel(base, 8, 8, 200).to(device)
    restore_weights(Path(args.checkpoint), model)
    rows = read_rows(Path(args.manifest))
    if args.max_examples: rows = rows[:args.max_examples]
    if not rows: raise ValueError("No supplied prediction windows")
    # Demo rows may resolve paths absolutely; match each actual file to any
    # checkpoint-sealed array by bytes before reading signals.
    sealed = {(v["sha256"], v["bytes"]) for v in state["eeg_artifacts"].values()}
    for entry in eeg_artifact_manifest(rows).values():
        if (entry["sha256"], entry["bytes"]) not in sealed:
            raise ValueError("Prediction EEG bytes are not present in the sealed cohort")
    for r in rows:
        if r["window_end_s"] > r["decision_time_s"] or r["environment_time_s"] > r["window_start_s"]:
            raise ValueError("Noncausal prediction metadata")
    metrics, _ = evaluate(model, tokenizer, rows, CLASSES, EEGStore(448), config, device,
                          str(args.checkpoint), output=Path(args.output), include_truth=False)
    metrics.update(n_examples=len(rows), hardware={**hardware(device), **backbone},
                   memory=memory_sample(device), allocator_limit=allocator,
                   checkpoint_sha256=sha256(Path(args.checkpoint)), checkpoint_modified=False,
                   model_or_threshold_changed=False, os_actions_executed=0)
    dump_json(Path(args.output).with_suffix(".metadata.json"), metrics)
    print(json.dumps(metrics, allow_nan=False))
    return metrics


def select_capacity(args):
    protocol = json.loads(Path(args.protocol).read_text())
    protocol_sha = sha256(Path(args.protocol))
    candidates, evidence = [], []
    for path in args.reports:
        path = Path(path)
        report = json.loads(path.read_text())
        config = json.loads(path.with_name("run_config.json").read_text())
        if (config.get("protocol_sha256") != protocol_sha or config.get("model_module_sha256") != sha256(Path(__file__))
                or report.get("manifest_sha256") != MANIFEST_SHA256 or config.get("command") != "probe"):
            raise ValueError("Capacity report has another protocol/model/data identity")
        batch = config["batch_size"]
        eligible = capacity_eligible(report, protocol)
        if batch not in protocol["capacity_selection"]["candidate_batches"]:
            raise ValueError("Unregistered capacity candidate")
        evidence.append({"batch": batch, "eligible": eligible, "report_path": str(path.resolve()),
                         "report_sha256": sha256(path), "run_config_sha256": sha256(path.with_name("run_config.json"))})
        if eligible: candidates.append(batch)
    if len(evidence) != len({e["batch"] for e in evidence}):
        raise ValueError("Duplicate batch capacity reports")
    if not candidates: raise ValueError("No measured stable capacity candidate; do not launch training")
    result = {"schema": "xavier-cbramod-capacity-selection-v1", "protocol_sha256": protocol_sha,
              "selected_batch": max(candidates), "selected_by": "largest_stable_train_only_candidate",
              "weights_reused": False, "validation_or_test_scores_used": False, "evidence": evidence}
    output = Path(args.output)
    if output.exists(): raise ValueError("Preserve prior capacity selection receipt")
    dump_json(output, result)
    print(json.dumps(result))
    return result


def capacity_eligible(report, protocol):
    probe = report.get("probe", {})
    return (report.get("status") == "completed_train_only_probe"
            and probe.get("completed_requested_steps") is True
            and probe.get("at_least_one_train_epoch_equivalent") is True
            and probe.get("allocator_stable_last_ten") is True
            and max(report.get("memory", {"missing": math.inf}).values()) <= protocol["training"]["memory_budget_gib"] * 1024**3)


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "probe"):
        t = commands.add_parser(command)
        for name in ("manifest", "output", "protocol", "cbramod-upstream", "cbramod-weights", "ssl-encoder-seal"):
            t.add_argument("--" + name, required=True)
        t.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
        t.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
        t.add_argument("--epochs", type=int, default=30)
        t.add_argument("--batch-size", type=int, default=8)
        t.add_argument("--lr", type=float, default=1e-3)
        t.add_argument("--seed", type=int, default=20261122)
        t.add_argument("--memory-budget-gib", type=float, default=21)
        t.add_argument("--max-seconds", type=float, default=900)
        t.add_argument("--max-prompt-tokens", type=int, default=128)
        t.add_argument("--abstain-threshold", type=float, default=.8)
        t.add_argument("--cpu-threads", type=int, default=2)
        t.add_argument("--resume")
        t.add_argument("--capacity-selection")
        t.add_argument("--probe-steps", type=int, default=50)
        t.set_defaults(func=train)
    p = commands.add_parser("predict")
    for name in ("checkpoint", "manifest", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--cbramod-upstream")
    p.add_argument("--cbramod-weights")
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), default="mps")
    p.add_argument("--memory-budget-gib", type=float, default=8)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--max-examples", type=int, default=3)
    p.set_defaults(func=predict)
    s = commands.add_parser("select-capacity")
    s.add_argument("--protocol", required=True)
    s.add_argument("--reports", nargs="+", required=True)
    s.add_argument("--output", required=True)
    s.set_defaults(func=select_capacity)
    return parser


def main():
    args = parser().parse_args()
    preserve_existing_output = (args.command in ("train", "probe") and Path(args.output).exists()
                                and any(Path(args.output).iterdir()) and not args.resume)
    try:
        summary = args.func(args)
        return 2 if summary.get("status") == "stopped_memory_budget" else 0
    except Exception as exc:
        if args.command in ("train", "probe") and not preserve_existing_output:
            dump_json(Path(args.output) / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
