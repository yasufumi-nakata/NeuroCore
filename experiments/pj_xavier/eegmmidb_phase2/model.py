"""Scratch EEG decoder and its trainable prefix connection to frozen Smol."""
from __future__ import annotations

import math
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from eegmmidb_phase2.data_contract import SHAPE, TASKS, require, sha256

BACKBONE_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
BACKBONE_REVISION = "31b70e2e869a7173562077fd711b654946d38674"
CLASS_SYMBOLS = (" A", " B")
CLASS_TOKEN_IDS = (330, 389)
ROUTING_SYMBOLS = (" X", " Y")
ARCHITECTURE = {
    "kind": "compact_EEGNet_style_from_scratch",
    "channels": 64,
    "samples": 600,
    "sample_rate_hz": 200,
    "input_unit": "V",
    "fixed_input_multiplier": 10000.0,
    "F1": 16,
    "depth_multiplier": 2,
    "F2": 32,
    "temporal_kernel": 64,
    "separable_kernel": 16,
    "dropout": 0.5,
    "prefix_tokens": 8,
}
PROMPT_VERSION = "static-task-and-two-symbols-v1"
PROMPTS = {
    "left_right": (
        "Classify the imagined movement represented by the EEG. "
        "The task has two possible classes: A = left-fist imagery; B = right-fist imagery. "
        "Return the class symbol only. Answer:"
    ),
    "fists_feet": (
        "Classify the imagined movement represented by the EEG. "
        "The task has two possible classes: A = both-fists imagery; B = both-feet imagery. "
        "Return the class symbol only. Answer:"
    ),
}
EEG_ONLY_PROMPT = "Classify the observed EEG into one of the two permitted symbols, A or B. Answer:"
ROUTE_SYSTEM = "Select the permitted identifier using the supplied mapping. Reply with the identifier only."


def static_prompt(task_family, condition="fused"):
    require(task_family in TASKS, "Unknown static task family")
    require(condition in ("fused", "eeg_only", "context_only", "eeg_shuffled", "eeg_time_shifted"), "Unknown input condition")
    return EEG_ONLY_PROMPT if condition == "eeg_only" else PROMPTS[task_family]


def audit_tokenizer(tokenizer):
    ids = [tokenizer.encode(symbol, add_special_tokens=False) for symbol in CLASS_SYMBOLS]
    require(ids == [[value] for value in CLASS_TOKEN_IDS], "Pinned Smol A/B single-token mapping changed")
    route_ids = [tokenizer.encode(symbol, add_special_tokens=False) for symbol in ROUTING_SYMBOLS]
    require(all(len(values) == 1 for values in route_ids) and route_ids[0] != route_ids[1], "Routing IDs must be distinct single tokens")
    prompts = {task: tokenizer.encode(text, add_special_tokens=True) for task, text in PROMPTS.items()}
    prompts["eeg_only"] = tokenizer.encode(EEG_ONLY_PROMPT, add_special_tokens=True)
    require(all(0 < len(values) <= 128 for values in prompts.values()), "Static prompt exceeds fixed capacity")
    return {
        "model": BACKBONE_ID, "revision": BACKBONE_REVISION,
        "classification_symbols": list(CLASS_SYMBOLS), "classification_token_ids": list(CLASS_TOKEN_IDS),
        "routing_symbols": list(ROUTING_SYMBOLS), "routing_token_ids": [value[0] for value in route_ids],
        "static_prompts": PROMPTS, "eeg_only_prompt": EEG_ONLY_PROMPT,
        "prompt_token_ids": prompts,
        "prompt_input_fields": ["task_family"],
        "target_or_subject_or_recording_or_trial_fields_used": False,
    }


def load_tokenizer(*, local_files_only=False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BACKBONE_ID, revision=BACKBONE_REVISION, trust_remote_code=False,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, audit_tokenizer(tokenizer)


def load_backbone(device, dtype, *, local_files_only=False):
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM

    common = {
        "revision": BACKBONE_REVISION,
        "trust_remote_code": False,
        "local_files_only": local_files_only,
    }
    config_path = Path(hf_hub_download(
        BACKBONE_ID, "config.json", revision=BACKBONE_REVISION,
        local_files_only=local_files_only,
    ))
    require("snapshots" in config_path.parts and config_path.parent.name == BACKBONE_REVISION,
            "Hugging Face config did not resolve to the pinned commit")
    configuration = AutoConfig.from_pretrained(BACKBONE_ID, **common)
    tokenizer, audit = load_tokenizer(local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE_ID, config=configuration, torch_dtype=dtype,
        attn_implementation="eager", **common,
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    require(count <= 2_000_000_000, "Backbone exceeds approved size")
    model.to(device=device, dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, {
        "backbone_id": BACKBONE_ID, "resolved_revision": BACKBONE_REVISION,
        "config_sha256": sha256(config_path), "config_snapshot_commit": config_path.parent.name,
        "parameter_count": count, "all_parameters_frozen": True,
        "tokenizer_audit": audit,
    }


class SamePadTemporal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, groups=1):
        super().__init__()
        self.left = (kernel - 1) // 2
        self.right = kernel - 1 - self.left
        self.convolution = nn.Conv2d(
            in_channels, out_channels, (1, kernel), groups=groups, bias=False
        )

    def forward(self, values):
        return self.convolution(F.pad(values, (self.left, self.right, 0, 0)))


class EEGEncoder(nn.Module):
    """No external EEG pretraining; all operations remain within observed window."""
    def __init__(self):
        super().__init__()
        self.temporal = SamePadTemporal(1, 16, 64)
        self.temporal_norm = nn.BatchNorm2d(16)
        self.spatial = nn.Conv2d(16, 32, (64, 1), groups=16, bias=False)
        self.spatial_norm = nn.BatchNorm2d(32)
        self.first_pool = nn.AvgPool2d((1, 4))
        self.first_dropout = nn.Dropout(0.5)
        self.depth_temporal = SamePadTemporal(32, 32, 16, groups=32)
        self.pointwise = nn.Conv2d(32, 32, 1, bias=False)
        self.output_norm = nn.BatchNorm2d(32)
        self.second_pool = nn.AvgPool2d((1, 8))
        self.second_dropout = nn.Dropout(0.5)
        self.width = 32

    def forward(self, eeg):
        require(eeg.ndim == 3 and tuple(eeg.shape[1:]) == SHAPE, "EEG tensor must be N x 64 x 600")
        require(bool(torch.isfinite(eeg).all()), "Nonfinite input EEG")
        values = eeg.float().unsqueeze(1) * 10000.0
        values = self.temporal_norm(self.temporal(values))
        values = F.elu(self.spatial_norm(self.spatial(values)))
        values = self.first_dropout(self.first_pool(values))
        values = self.depth_temporal(values)
        values = F.elu(self.output_norm(self.pointwise(values)))
        values = self.second_dropout(self.second_pool(values))
        return values.squeeze(2).transpose(1, 2)

    def prefix_features(self, eeg, count=8):
        values = self(eeg)
        length = values.shape[1]
        # This explicit adaptive-mean definition is MPS-compatible for 18 -> 8.
        pieces = [
            values[:, math.floor(index * length / count):math.ceil((index + 1) * length / count)].mean(1)
            for index in range(count)
        ]
        return torch.stack(pieces, dim=1)


class ScratchDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EEGEncoder()
        self.heads = nn.ModuleDict({task: nn.Linear(32, 2) for task in TASKS})

    def logits(self, eeg, task_families, condition="fused"):
        require(len(task_families) == len(eeg), "Task routing length differs")
        require(all(task in TASKS for task in task_families), "Unknown task family")
        if condition == "context_only":
            eeg = torch.zeros_like(eeg)
        features = self.encoder(eeg).mean(1)
        return torch.stack([
            self.heads[task](features[index]) for index, task in enumerate(task_families)
        ])

    def forward(self, eeg, task_families, condition="fused"):
        return self.logits(eeg, task_families, condition)


class FusedDecoder(nn.Module):
    def __init__(self, base, *, encoder=None, class_token_ids=CLASS_TOKEN_IDS):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.encoder = encoder if encoder is not None else EEGEncoder()
        hidden = base.get_input_embeddings().embedding_dim
        self.projector = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, hidden))
        self.class_token_ids = tuple(class_token_ids)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def logits_from_ids(self, eeg, prompt_ids, condition="fused"):
        require(len(prompt_ids) == len(eeg) and all(len(ids) > 0 for ids in prompt_ids), "Invalid prompt batch")
        if condition == "context_only":
            eeg = torch.zeros_like(eeg)
        features = self.encoder.prefix_features(eeg, 8)
        prefix = self.projector(features)
        embeddings = self.base.get_input_embeddings()
        sequences = []
        lengths = []
        for index, ids in enumerate(prompt_ids):
            ids = torch.as_tensor(ids, device=prefix.device, dtype=torch.long)
            text = embeddings(ids)
            sequence = torch.cat((prefix[index].to(text.dtype), text), dim=0)
            sequences.append(sequence)
            lengths.append(len(sequence))
        padded = pad_sequence(sequences, batch_first=True)
        mask = torch.arange(padded.shape[1], device=padded.device)[None] < torch.tensor(lengths, device=padded.device)[:, None]
        result = self.base(inputs_embeds=padded, attention_mask=mask.long(), use_cache=False)
        last = result.logits[
            torch.arange(len(lengths), device=padded.device),
            torch.tensor(lengths, device=padded.device) - 1,
        ]
        return last[:, list(self.class_token_ids)].float()

    def forward(self, eeg, prompt_ids, condition="fused"):
        return self.logits_from_ids(eeg, prompt_ids, condition)


def learned_state(model, kind):
    if kind == "scratch":
        return {"encoder": model.encoder.state_dict(), "heads": model.heads.state_dict()}
    if kind == "fused":
        return {"encoder": model.encoder.state_dict(), "projector": model.projector.state_dict()}
    raise ValueError("Unknown model kind")


def restore_learned(model, kind, state):
    require(set(state) == ({"encoder", "heads"} if kind == "scratch" else {"encoder", "projector"}), "Learned module identity differs")
    model.encoder.load_state_dict(state["encoder"], strict=True)
    if kind == "scratch":
        model.heads.load_state_dict(state["heads"], strict=True)
    else:
        model.projector.load_state_dict(state["projector"], strict=True)


def trainable_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

