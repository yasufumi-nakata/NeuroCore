#!/usr/bin/env python3
"""Guarded, validation-only direct EEG LoRA experiments for an existing MLX model.

No learned EEG encoder is introduced.  The base and historical runs are read-only.
The supervisor owns the existing GPU lock; one child keeps its Adam state while
evaluating checkpoints.  This runner never opens a held-out test set and never
automatically starts a larger model/layer scope.  Validation is for development,
not a final claim of EEG benefit, general intelligence, or computer-use ability.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

LABELS = ("mouse_click", "keyboard", "ai_prompt_submit")
EEG_MARKER = "Fixed-discretized EEG measurement (not a learned encoder):\n"
EEG_TAIL = "\nOutput exactly one label."
SCHEMA = "direct-qwen-mlx-progressive-v2"


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    require(cfg.get("schema") == SCHEMA, "unsupported_config_schema")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", cfg.get("run_id", "")), "unsafe_run_id")
    require(cfg.get("scope") in ("corrected4", "all64"), "unsupported_scope")
    require(cfg.get("condition") in ("eeg", "noeeg"), "unsupported_condition")
    require(cfg.get("rank") == 8, "this_protocol_only_allows_rank8")
    require(cfg.get("batch_size") == 1, "paired_order_protocol_requires_batch1")
    require(cfg.get("enable_thinking") is False, "thinking_must_be_explicitly_false")
    require(cfg.get("max_seq_length") == 1024, "registered_context_must_be_1024")
    require(isinstance(cfg.get("iters"), int) and 1 <= cfg["iters"] <= 80,
            "initial_stage_budget_is_1_to_80_steps")
    points = cfg.get("checkpoint_steps", [])
    require(points == sorted(set(points)) and points and points[-1] == cfg["iters"]
            and all(isinstance(n, int) and 0 < n <= cfg["iters"] for n in points),
            "invalid_checkpoint_steps")
    require(60 <= cfg.get("max_wall_seconds", 0) <= 14400, "invalid_wall_budget")
    require(1 <= cfg.get("memory_limit_gb", 0) <= 55, "invalid_memory_budget")
    require(0 < cfg.get("learning_rate", 0) <= 0.0001, "invalid_learning_rate")
    require(0 <= cfg.get("invalid_rate_limit", -1) <= 0.2, "invalid_invalid_rate_limit")
    require(0 < cfg.get("max_loss", 0) <= 100, "invalid_loss_guard")
    require(isinstance(cfg.get("max_sentry_drop"), int) and 0 <= cfg["max_sentry_drop"] <= 2,
            "invalid_sentry_drop_guard")
    require(isinstance(cfg.get("seed"), int), "missing_seed")
    require(cfg.get("expected_counts") == {"train": 400, "val": 155},
            "registered_split_counts_changed")
    for name in ("train", "val"):
        item = cfg["inputs"][name]
        p = Path(item["path"]).expanduser().resolve()
        allowed = {"train.jsonl"} if name == "train" else {"valid.jsonl", "val.jsonl"}
        require(p.name in allowed, "only_explicit_train_and_validation_inputs_allowed")
        require(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), "missing_input_hash")
    for name in ("model_path", "run_dir", "lock_path"):
        require(Path(cfg[name]).expanduser().is_absolute(), "paths_must_be_absolute")
    require(Path(cfg["lock_path"]).name == "train_gpu.lock", "must_use_existing_gpu_lock")
    expected_lock = Path(cfg["model_path"]).expanduser().resolve().parents[1] / "train_gpu.lock"
    require(Path(cfg["lock_path"]).expanduser().resolve() == expected_lock,
            "gpu_lock_not_in_registered_model_node_root")
    for key in ("expected_model_config_sha256",):
        require(re.fullmatch(r"[0-9a-f]{64}", cfg.get(key, "")), "missing_model_hash")
    require(re.fullmatch(r"[0-9a-f]{64}", cfg.get("base_receipt", {}).get("sha256", "")),
            "missing_immutable_base_receipt")
    require(Path(cfg["base_receipt"]["path"]).expanduser().is_absolute(), "base_receipt_path_not_absolute")
    # Code-generated progression is deliberately disabled.  A larger scope needs
    # an explicit audit receipt from the coordinator, not merely a falling loss.
    if cfg["scope"] == "all64":
        require(cfg.get("progression_receipt"), "all64_requires_progression_receipt")
        receipt = json.loads(Path(cfg["progression_receipt"]).read_text())
        require(receipt.get("allow_all64_probe") is True
                and receipt.get("purpose") == "bounded_engineering_probe", "unapproved_all64_probe")
        comparison_path = Path(receipt["comparison_report_path"])
        require(sha256(comparison_path) == receipt.get("comparison_report_sha256"),
                "progression_comparison_hash_mismatch")
        comparison = json.loads(comparison_path.read_text())
        require(comparison.get("schema") == SCHEMA and comparison.get("scope") == "corrected4"
                and comparison.get("step") == 80 and comparison.get("quality_gates_passed") is True
                and comparison.get("experiment_identity_sha256") == experiment_identity(cfg),
                "progression_comparison_not_same_completed_control_protocol")
    return cfg


def read_training_rows(cfg):
    """Only the two explicitly registered files are ever opened."""
    result = {}
    for split in ("train", "val"):
        source = cfg["inputs"][split]
        path = Path(source["path"]).expanduser().resolve()
        require(sha256(path) == source["sha256"], "immutable_input_hash_mismatch_" + split)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        require(len(rows) == cfg["expected_counts"][split], "wrong_split_count_" + split)
        seen = set()
        for row in rows:
            require(row.get("split") in (("val", "validation") if split == "val" else ("train",)),
                    "unexpected_split_membership")
            require(row.get("id") not in seen and row.get("id") is not None, "duplicate_or_missing_id")
            seen.add(row["id"])
            messages = row.get("messages", [])
            require(len(messages) == 2 and messages[0].get("role") == "user"
                    and messages[1].get("role") == "assistant", "unexpected_message_structure")
            require(messages[1].get("content") in LABELS, "unexpected_gold_label")
            require(row.get("canonical_subject_id") is not None, "missing_subject_key")
            split_prompt(messages[0]["content"])
        result[split] = rows
    train_ids = {r["id"] for r in result["train"]}
    val_ids = {r["id"] for r in result["val"]}
    train_subjects = {r["canonical_subject_id"] for r in result["train"]}
    val_subjects = {r["canonical_subject_id"] for r in result["val"]}
    require(not train_ids & val_ids, "train_val_id_overlap")
    require(not train_subjects & val_subjects, "train_val_subject_overlap")
    return result


def split_prompt(prompt):
    require(prompt.count(EEG_MARKER) == 1 and prompt.count(EEG_TAIL) == 1,
            "unexpected_eeg_serialization")
    before, remaining = prompt.split(EEG_MARKER, 1)
    eeg, after = remaining.rsplit(EEG_TAIL, 1)
    require(eeg.strip() and not after.strip(), "unexpected_eeg_prompt_suffix")
    return before, eeg, EEG_TAIL + after


def no_eeg_prompt(prompt):
    before, _, after = split_prompt(prompt)
    return before + "EEG measurement withheld for the matched context-only control.\n" + after


def shuffled_prompts(rows, seed):
    """Within-subject cyclic derangement; context/gold always stay with recipient."""
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        groups[str(row["canonical_subject_id"])].append(i)
    rng = random.Random(seed)
    donor = {}
    for subject in sorted(groups):
        indices = groups[subject]
        require(len(indices) >= 2, "cannot_shuffle_singleton_subject")
        shift = rng.randrange(1, len(indices))
        for i, j in zip(indices, indices[shift:] + indices[:shift]):
            donor[i] = j
    require(len(set(donor.values())) == len(rows) and all(i != j for i, j in donor.items()),
            "invalid_derangement")
    parts = [split_prompt(r["messages"][0]["content"]) for r in rows]
    prompts = [parts[i][0] + EEG_MARKER + parts[donor[i]][1] + parts[i][2]
               for i in range(len(rows))]
    require(all(p != rows[i]["messages"][0]["content"] for i, p in enumerate(prompts)),
            "shuffle_did_not_change_measurement")
    return prompts, [donor[i] for i in range(len(rows))]


def render_example(tokenizer, row, prompt=None, max_length=1024):
    prompt = row["messages"][0]["content"] if prompt is None else prompt
    label = row["messages"][1]["content"]
    user = [{"role": "user", "content": prompt}]
    common = dict(tokenize=True, return_dict=False, enable_thinking=False)
    prefix = list(tokenizer.apply_chat_template(user, add_generation_prompt=True, **common))
    tokens = list(tokenizer.apply_chat_template(user + [{"role": "assistant", "content": label}],
                                               add_generation_prompt=False, **common))
    require(tokens[:len(prefix)] == prefix, "train_eval_prefix_mismatch")
    require(0 < len(prefix) < len(tokens) <= max_length, "truncated_or_empty_completion")
    suffix = tokenizer.decode(tokens[len(prefix):], skip_special_tokens=True).strip()
    require(suffix == label and "think" not in suffix, "completion_is_not_exact_registered_label")
    # Trainer indices are target-token positions in the original (unshifted)
    # sequence.  Exclude position len(tokens): that is the first padding token.
    selected = [i for i in range(1, len(tokens) + 32)
                if i >= len(prefix) and i < len(tokens)]
    require(selected == list(range(len(prefix), len(tokens))), "loss_mask_boundary_error")
    return tokens, len(prefix)


def preflight(cfg, tokenizer=None):
    model_path = Path(cfg["model_path"]).expanduser().resolve()
    require(sha256(model_path / "config.json") == cfg["expected_model_config_sha256"],
            "base_model_config_changed")
    if "base_receipt" in cfg:
        require(sha256(Path(cfg["base_receipt"]["path"]).expanduser()) == cfg["base_receipt"]["sha256"],
                "immutable_base_receipt_changed")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False,
                                                  local_files_only=True)
    rows = read_training_rows(cfg)
    processed = {}
    report = {"schema": SCHEMA, "status": "passed", "created_at": utc_now(),
              "test_opened": False, "enable_thinking": False,
              "config_sha256": canonical_digest(cfg), "splits": {}}
    for split, records in rows.items():
        for condition in ("eeg", "noeeg"):
            prompts = [r["messages"][0]["content"] for r in records]
            if condition == "noeeg":
                prompts = [no_eeg_prompt(p) for p in prompts]
            items = [render_example(tokenizer, r, p, cfg["max_seq_length"])
                     for r, p in zip(records, prompts)]
            processed[(split, condition)] = items
            report["splits"][split + "/" + condition] = {
                "count": len(items), "min_tokens": min(len(t) for t, _ in items),
                "max_tokens": max(len(t) for t, _ in items),
                "min_completion_tokens": min(len(t) - o for t, o in items),
                "subjects": len({r["canonical_subject_id"] for r in records}),
                "all_prefixes_equal": True, "all_completion_masks_exact": True,
            }
    shuffled, donor = shuffled_prompts(rows["val"], cfg["seed"] + 1)
    for r, prompt in zip(rows["val"], shuffled):
        render_example(tokenizer, r, prompt, cfg["max_seq_length"])
    report["shuffle_mapping_sha256"] = canonical_digest(donor)
    report["val_pairing_sha256"] = canonical_digest([
        [r["id"], r["canonical_subject_id"], r["messages"][1]["content"]] for r in rows["val"]])
    report["train_order_sha256"] = canonical_digest(training_order(len(rows["train"]), cfg["seed"], cfg["iters"]))
    return report, rows, processed


def training_order(size, seed, count):
    rng = random.Random(seed)
    order = []
    while len(order) < count:
        batch = list(range(size))
        rng.shuffle(batch)
        order.extend(batch)
    return order[:count]


def fixed_probe_indices(rows, seed, per_subject=None, per_class=None):
    """Freeze selection before training; only development splits are accepted."""
    require(bool(per_subject) != bool(per_class), "select_one_probe_rule")
    require(all(r.get("split") in ("train", "val", "validation") for r in rows),
            "probe_selection_only_train_and_validation")
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        group = str(row["canonical_subject_id"]) if per_subject else row["messages"][1]["content"]
        groups[group].append(i)
    chosen = []
    for group in sorted(groups):
        indices = list(groups[group])
        rng.shuffle(indices)
        count = per_subject or per_class
        if per_subject:
            # Up to two examples/class per subject avoids an accidental probe
            # containing only a majority class; fill from remaining trials.
            picked = []
            for label in LABELS:
                picked.extend([i for i in indices if rows[i]["messages"][1]["content"] == label][:2])
            picked = picked[:count]
            picked.extend(i for i in indices if i not in picked)
            chosen.extend(picked[:count])
        else:
            chosen.extend(indices[:count])
    chosen = sorted(chosen)
    require(all(any(rows[i]["messages"][1]["content"] == label for i in chosen) for label in LABELS),
            "fixed_probe_missing_class")
    return chosen


def parse_prediction(text):
    # Exact label matching prevents a label followed by explanation from passing
    # the format gate. Whitespace surrounding a valid label is harmless.
    normalized = text.strip().lower()
    return normalized if normalized in LABELS else "invalid"


def metrics(predictions):
    require(predictions, "empty_evaluation")
    counts = collections.Counter(x["gold"] for x in predictions)
    require(all(counts[label] for label in LABELS), "missing_validation_class")
    acc = sum(x["gold"] == x["pred"] for x in predictions) / len(predictions)
    recall, f1 = [], []
    for label in LABELS:
        tp = sum(x["gold"] == label and x["pred"] == label for x in predictions)
        fp = sum(x["gold"] != label and x["pred"] == label for x in predictions)
        fn = counts[label] - tp
        recall.append(tp / counts[label])
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    return {"n": len(predictions), "accuracy": acc,
            "balanced_accuracy": sum(recall) / len(LABELS),
            "macro_f1": sum(f1) / len(LABELS),
            "invalid_rate": sum(x["pred"] == "invalid" for x in predictions) / len(predictions),
            "prediction_counts": dict(collections.Counter(x["pred"] for x in predictions))}


def paired_difference(a, b):
    require(len(a) == len(b) and a, "unpaired_prediction_count")
    require(all(x["index"] == y["index"] and x["gold"] == y["gold"]
                and x["row_sha256"] == y["row_sha256"]
                and x["subject_sha256"] == y["subject_sha256"] for x, y in zip(a, b)),
            "unpaired_predictions")
    wins = sum(x["pred"] == x["gold"] and y["pred"] != y["gold"] for x, y in zip(a, b))
    losses = sum(x["pred"] != x["gold"] and y["pred"] == y["gold"] for x, y in zip(a, b))
    n = wins + losses
    exact_p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0
    # A small subject count makes this exploratory interval broad; retain groups
    # in the bootstrap instead of pretending that adjacent EEG trials are IID.
    groups = collections.defaultdict(list)
    for x, y in zip(a, b):
        groups[x["subject_sha256"]].append(int(x["pred"] == x["gold"]) - int(y["pred"] == y["gold"]))
    values = list(groups.values())
    rng = random.Random(20261123)
    draws = []
    for _ in range(1000):
        sampled = [v for _ in values for v in values[rng.randrange(len(values))]]
        draws.append(sum(sampled) / len(sampled))
    draws.sort()
    return {"accuracy_delta": (wins - losses) / len(a), "paired_wins": wins,
            "paired_losses": losses, "mcnemar_exact_p": exact_p,
            "mcnemar_note": "uncalibrated IID-trial reference; subject clustering may violate its assumptions",
            "subject_cluster_bootstrap_95pct": [draws[24], draws[974]],
            "subjects": len(values), "development_evaluation_only": True}


def save_next(run_dir, action, **details):
    atomic_json(Path(run_dir) / "next_action.json", {
        "schema": SCHEMA, "updated_at": utc_now(), "action": action,
        "auto_scale_allowed": False, "test_opened": False, **details})


def run_supervisor(config_path, cfg):
    run_dir = Path(cfg["run_dir"]).expanduser().resolve()
    model_dir = Path(cfg["model_path"]).expanduser().resolve()
    require(run_dir != model_dir and model_dir not in run_dir.parents
            and run_dir not in model_dir.parents, "run_directory_overlaps_base_model")
    lock_path = Path(cfg["lock_path"]).expanduser()
    require(lock_path.is_file() and not lock_path.is_symlink(), "registered_gpu_lock_missing_or_not_regular")
    require(not run_dir.exists(), "refusing_to_overwrite_or_repeat_existing_run")
    run_dir.mkdir(parents=True)
    atomic_json(run_dir / "config.json", cfg)
    atomic_json(run_dir / "supervisor.json", {"status": "waiting_for_gpu_lock", "pid": os.getpid(),
                                             "started_at": utc_now(), "test_opened": False})
    fd = os.open(lock_path, os.O_RDWR)
    child = None
    status = "failed"
    exit_code = 2
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            status, exit_code = "blocked_gpu_lock", 11
            save_next(run_dir, "retry_in_new_run_after_gpu_lock_released")
            return exit_code
        # The registered lock is the primary exclusion mechanism. Do not launch
        # over a historical trainer that does not cooperate with that lock.
        commands = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True,
                                  timeout=10, check=True).stdout.splitlines()
        if any("mlx_lm lora" in c or "eeg_llm.py" in c for c in commands):
            status, exit_code = "blocked_other_gpu_trainer", 12
            save_next(run_dir, "inspect_existing_gpu_job")
            return exit_code
        with (run_dir / "worker.log").open("x") as log:
            env = os.environ.copy()
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                       TOKENIZERS_PARALLELISM="false")
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_train",
                                      "--config", str(run_dir / "config.json"), "--lock-fd", str(fd)],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     pass_fds=(fd,), start_new_session=True, env=env)
            atomic_json(run_dir / "supervisor.json", {"status": "running", "pid": os.getpid(),
                        "child_pid": child.pid, "started_at": utc_now(),
                        "max_wall_seconds": cfg["max_wall_seconds"], "test_opened": False})
            try:
                exit_code = child.wait(timeout=cfg["max_wall_seconds"])
                status = "completed" if exit_code == 0 else "failed"
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=15)
                status, exit_code = "stopped_wall_budget", 124
                save_next(run_dir, "review_wall_budget_failure_and_last_checkpoint")
        return exit_code
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=15)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=15)
        atomic_json(run_dir / "supervisor.json", {"status": status, "exit_code": exit_code,
                    "pid": os.getpid(), "child_pid": child.pid if child else None,
                    "ended_at": utc_now(), "test_opened": False})
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def train_worker(cfg, lock_fd):
    run_dir = Path(cfg["run_dir"]).expanduser()
    lock_stat = os.fstat(lock_fd)
    require((lock_stat.st_dev, lock_stat.st_ino) ==
            (os.stat(cfg["lock_path"]).st_dev, os.stat(cfg["lock_path"]).st_ino),
            "worker_missing_supervisor_lock")
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"schema": SCHEMA, "status": "preflight", "run_id": cfg["run_id"],
             "scope": cfg["scope"], "condition": cfg["condition"], "step": 0,
             "started_at": utc_now(), "test_opened": False, "auto_scale_allowed": False}

    def update(**changes):
        state.update(changes, updated_at=utc_now())
        atomic_json(run_dir / "state.json", state)

    started = time.monotonic()
    try:
        update()
        receipt, rows, processed = preflight(cfg)
        atomic_json(run_dir / "preflight.json", receipt)
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        import numpy as np
        from mlx.utils import tree_flatten
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.tuner.trainer import train, TrainingArgs
        from mlx_lm.tuner.utils import linear_to_lora_layers
        require(mx.distributed.init().size() == 1, "this_runner_is_single_gpu_only")
        mx.set_memory_limit(int(cfg["memory_limit_gb"] * 1e9))
        mx.set_cache_limit(min(2_000_000_000, int(cfg["memory_limit_gb"] * 1e8)))
        mx.random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        update(status="loading_model")
        model, tokenizer = load(cfg["model_path"], lazy=False)
        # Validate again with the tokenizer wrapper actually used by generate.
        actual_receipt, rows, processed = preflight(cfg, tokenizer)
        require(actual_receipt["splits"] == receipt["splits"], "loaded_tokenizer_preflight_changed")
        require(len(model.layers) == 64, "unexpected_qwen_decoder_layer_count")
        model.freeze()
        n_layers = 4 if cfg["scope"] == "corrected4" else 64
        selected_layers = list(enumerate(model.layers))[-n_layers:]
        from mlx_lm.models.switch_layers import SwitchLinear, QuantizedSwitchLinear
        legacy_types = (nn.Linear, nn.QuantizedLinear, nn.Embedding, nn.QuantizedEmbedding,
                        SwitchLinear, QuantizedSwitchLinear)
        legacy_targets = set()
        keys = set()
        modules = []
        for i, layer in selected_layers:
            for name, module in layer.named_modules():
                if isinstance(module, legacy_types) or hasattr(module, "to_lora"):
                    legacy_targets.add((i, name))
                if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                    keys.add(name)
                    modules.append({"layer": i, "name": name, "type": type(module).__name__})
        require(modules and all(any(x["layer"] == i for x in modules) for i, _ in selected_layers),
                "decoder_layer_without_linear_lora_target")
        require(not any(any(s in x["name"] for s in ("embed_tokens", "lm_head", "vision")) for x in modules),
                "unexpected_embedding_head_or_vision_target")
        lora_parameters = {"rank": cfg["rank"], "scale": 20.0, "dropout": 0.0, "keys": sorted(keys)}
        linear_to_lora_layers(model, n_layers, lora_parameters)
        trainable = dict(tree_flatten(model.trainable_parameters()))
        require(trainable and all(k.rsplit(".", 1)[-1] in ("lora_a", "lora_b") for k in trainable),
                "non_lora_parameter_unfrozen")
        audit = {"num_layers": n_layers, "targets": modules,
                 "target_count": len(modules), "trainable_parameters": sum(v.size for v in trainable.values()),
                 "trainable_names": sorted(trainable), "embedding_head_vision_excluded": True,
                 "layer_types": dict(collections.Counter(type(x).__name__ for _, x in selected_layers)),
                 "base_weights_frozen": True, "rank": cfg["rank"], "lora_parameters": lora_parameters}
        require(len(trainable) == 2 * len(modules), "lora_target_parameter_count_mismatch")
        audit["matrix_selection_rule"] = "all Linear and QuantizedLinear modules inside selected decoder layers"
        audit["training_budget"] = {"examples": len(rows["train"]), "optimizer_steps": cfg["iters"],
                                     "batch_size": 1, "nominal_epochs": cfg["iters"] / len(rows["train"]),
                                     "interpretation": "bounded adaptation probe; not full-dataset training completion"}
        current_targets = {(x["layer"], x["name"]) for x in modules}
        audit["legacy_default_rule"] = "Linear/QuantizedLinear/SwitchLinear/Embedding or module.to_lora"
        audit["legacy_default_target_count"] = len(legacy_targets)
        audit["legacy_targets_excluded"] = sorted(legacy_targets - current_targets)
        audit["new_targets_not_in_legacy"] = sorted(current_targets - legacy_targets)
        audit["legacy_target_rule_identical_for_selected_layers"] = legacy_targets == current_targets
        atomic_json(run_dir / "module_audit.json", audit)
        adapter_config = {"fine_tune_type": "lora", "num_layers": n_layers,
                          "lora_parameters": lora_parameters, "model": cfg["model_path"],
                          "enable_thinking": False, "mask_prompt": True,
                          "loss_mask": "prefix_length <= target_position < token_length"}
        adapter_dir = run_dir / "adapter"
        adapter_dir.mkdir()
        atomic_json(adapter_dir / "adapter_config.json", adapter_config)
        mx.eval(model.parameters())
        optimizer = optim.Adam(learning_rate=cfg["learning_rate"])
        sampler = make_sampler(temp=0.0)
        val_rows = rows["val"]
        eeg_prompts = [r["messages"][0]["content"] for r in val_rows]
        shuffled, donors = shuffled_prompts(val_rows, cfg["seed"] + 1)
        no_prompts = [no_eeg_prompt(p) for p in eeg_prompts]
        val_probe = fixed_probe_indices(val_rows, cfg["seed"] + 2, per_subject=6)
        train_probe = fixed_probe_indices(rows["train"], cfg["seed"] + 3, per_class=4)
        atomic_json(run_dir / "probe_selection.json", {
            "validation_rule": "up_to_6_per_subject_with_class_coverage; development_selection_only",
            "val_indices": val_probe, "train_indices": train_probe,
            "val_selection_sha256": canonical_digest([val_rows[i]["id"] for i in val_probe]),
            "train_selection_sha256": canonical_digest([rows["train"][i]["id"] for i in train_probe]),
            "final_checkpoint_val_count": 155, "test_opened": False})

        def guard(loss=None):
            require(time.monotonic() - started < cfg["max_wall_seconds"], "wall_budget_exceeded")
            peak = mx.get_peak_memory() / 1e9
            require(peak <= cfg["memory_limit_gb"], "peak_memory_limit_exceeded")
            if loss is not None:
                require(math.isfinite(loss) and 0 <= loss <= cfg["max_loss"], "nonfinite_or_excessive_loss")
            return peak

        def exact_loss(current_model, batch, lengths):
            logits = current_model(batch[:, :-1])
            targets = batch[:, 1:]
            positions = mx.arange(1, targets.shape[1] + 1)
            mask = (positions >= lengths[:, 0:1]) & (positions < lengths[:, 1:])
            ntokens = mask.sum()
            loss = (nn.losses.cross_entropy(logits, targets) * mask).astype(mx.float32).sum() / ntokens
            return loss, ntokens

        def paired_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
            require(batch_size == 1 and (comm_group is None or comm_group.size() == 1),
                    "unexpected_batch_topology")
            if loop:
                indices = training_order(len(dataset), cfg["seed"], cfg["iters"])
            else:
                indices = list(range(len(dataset)))
            for i in indices:
                tokens, offset = dataset[i]
                require(len(tokens) <= max_seq_length, "refusing_silent_truncation")
                padded = min(max_seq_length, 1 + 32 * ((len(tokens) + 31) // 32))
                values = np.zeros((1, padded), np.int32)
                values[0, :len(tokens)] = tokens
                yield mx.array(values), mx.array([[offset, len(tokens)]])

        def capability_sentry():
            # These tiny fixed probes catch formatting/basic instruction regressions;
            # they are explicitly not a computer-use or broad capability benchmark.
            probes = [
                ("What is 7 + 5? Reply with only the number.", "12"),
                ("Sort these numbers ascending: 9, 2, 5. Reply only: a,b,c with no spaces.", "2,5,9"),
                ("Return only the lowercase English translation of 猫.", "cat"),
                ("Output exactly this JSON and nothing else: {\"ok\":true}", '{"ok":true}'),
                ("Reply with exactly the word READY and nothing else.", "READY"),
                ("日本語で「了解しました」とだけ答えてください．", "了解しました"),
            ]
            answers = []
            for prompt, expected in probes:
                guard()
                ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                    tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
                answer = generate(model, tokenizer, ids, verbose=False, max_tokens=24, sampler=sampler)
                answers.append(answer.strip() == expected)
                mx.clear_cache()
            return {"n": len(probes), "passed": sum(answers), "checks": answers,
                    "scope": "basic_format_and_instruction_smoke_only"}

        model.eval()
        baseline_sentry = capability_sentry()
        atomic_json(run_dir / "baseline_sentry.json", baseline_sentry)
        model.train()

        def evaluate_checkpoint(step):
            guard()
            directory = run_dir / "checkpoints" / f"{step:07d}"
            directory.mkdir(parents=True, exist_ok=False)
            weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(directory / "adapters.safetensors"), weights)
            atomic_json(directory / "adapter_config.json", adapter_config)
            atomic_json(directory / "checkpoint.json", {
                "step": step, "adapter_sha256": sha256(directory / "adapters.safetensors"),
                "optimizer_continuity": "same_live_Adam_instance; no restart between evaluations",
                "crash_resume_supported": False})
            update(status="evaluating", step=step, eval_done=0, eval_checkpoint=str(directory))
            arms = {"eeg": eeg_prompts, "shuffle": shuffled} if cfg["condition"] == "eeg" else {"noeeg": no_prompts}
            predictions, reports = {}, {}
            selected = list(range(len(val_rows))) if step == cfg["iters"] else val_probe
            model.eval()
            try:
                for arm, prompts in arms.items():
                    outputs = []
                    for done, i in enumerate(selected, 1):
                        row, prompt = val_rows[i], prompts[i]
                        guard()
                        tokens, offset = render_example(tokenizer, row, prompt, cfg["max_seq_length"])
                        out = generate(model, tokenizer, tokens[:offset], verbose=False,
                                       max_tokens=12, sampler=sampler)
                        outputs.append({"index": i, "gold": row["messages"][1]["content"],
                            "pred": parse_prediction(out), "row_sha256": canonical_digest(row["id"]),
                            "subject_sha256": canonical_digest(row["canonical_subject_id"])})
                        if done % 10 == 0 or done == len(selected):
                            update(eval_arm=arm, eval_done=done, eval_target=len(selected), peak_memory_gb=guard())
                            atomic_json(directory / (arm + "_partial.private.json"), outputs)
                        mx.clear_cache()
                    predictions[arm], reports[arm] = outputs, metrics(outputs)
                train_outputs = []
                for i in train_probe:
                    guard()
                    row = rows["train"][i]
                    prompt = row["messages"][0]["content"]
                    if cfg["condition"] == "noeeg":
                        prompt = no_eeg_prompt(prompt)
                    tokens, offset = render_example(tokenizer, row, prompt, cfg["max_seq_length"])
                    out = generate(model, tokenizer, tokens[:offset], verbose=False,
                                   max_tokens=12, sampler=sampler)
                    train_outputs.append({"gold": row["messages"][1]["content"], "pred": parse_prediction(out)})
                    mx.clear_cache()
                train_probe_metrics = metrics(train_outputs)
                sentry = capability_sentry()
            finally:
                model.train()
                mx.clear_cache()
            quality_gates = {
                "output_format": all(v["invalid_rate"] <= cfg["invalid_rate_limit"] for v in reports.values()),
                "basic_instruction_sentry": sentry["passed"] >= baseline_sentry["passed"] - cfg["max_sentry_drop"],
            }
            quality_passed = all(quality_gates.values())
            report = {"schema": SCHEMA, "status": "completed" if quality_passed else "failed_quality_gates", "step": step,
                      "run_id": cfg["run_id"], "condition": cfg["condition"], "scope": cfg["scope"],
                      "seed": cfg["seed"], "config_sha256": canonical_digest(cfg),
                      "comparison_contract_sha256": comparison_contract(cfg),
                      "experiment_identity_sha256": experiment_identity(cfg),
                      "val_pairing_sha256": receipt["val_pairing_sha256"],
                      "shuffle_mapping_sha256": canonical_digest(donors), "metrics": reports,
                      "selected_indices": selected,
                      "selection": "all_validation" if step == cfg["iters"] else "fixed_development_probe",
                      "train_probe_metrics": train_probe_metrics,
                      "sentry": sentry, "baseline_sentry": baseline_sentry,
                      "quality_gates": quality_gates, "quality_gates_passed": quality_passed,
                      "test_opened": False, "development_only": True,
                      "adapter_sha256": sha256(directory / "adapters.safetensors")}
            if cfg["condition"] == "eeg":
                report["eeg_vs_shuffle"] = paired_difference(predictions["eeg"], predictions["shuffle"])
            atomic_json(directory / "predictions.private.json", predictions)
            atomic_json(directory / "evaluation.json", report)
            require(quality_gates["output_format"],
                    "invalid_output_gate_failed")
            require(quality_gates["basic_instruction_sentry"],
                    "basic_instruction_regression_gate_failed")
            update(status="training", last_evaluated_step=step, last_evaluation=str(directory / "evaluation.json"))
            save_next(run_dir, "observe_current_stage_without_scaling", last_evaluated_step=step)

        class Callback:
            def on_val_loss_report(self, info):
                guard(float(info["val_loss"]))
                update(val_loss=float(info["val_loss"]), val_loss_step=info["iteration"])

            def on_train_loss_report(self, info):
                step = int(info["iteration"])
                peak = guard(float(info["train_loss"]))
                seen_order = training_order(len(rows["train"]), cfg["seed"], step)
                update(status="training", step=step, train_loss=float(info["train_loss"]), peak_memory_gb=peak,
                       train_presentations=step, unique_train_examples=len(set(seen_order)),
                       epoch_equivalent=step / len(rows["train"]),
                       ordered_training_indices_sha256=canonical_digest(seen_order))
                if step in cfg["checkpoint_steps"]:
                    evaluate_checkpoint(step)

        args = TrainingArgs(batch_size=1, iters=cfg["iters"], val_batches=20,
                            steps_per_report=1, steps_per_eval=40,
                            steps_per_save=cfg["iters"] + 1,
                            max_seq_length=1024, adapter_file=str(adapter_dir / "adapters.safetensors"),
                            grad_checkpoint=True, grad_accumulation_steps=1,
                            clear_cache_threshold=1_000_000_000)
        update(status="training", module_audit="module_audit.json", optimizer="Adam", optimizer_restarts=0)
        train(model=model, optimizer=optimizer, train_dataset=processed[("train", cfg["condition"])],
              val_dataset=processed[("val", cfg["condition"])], args=args, loss=exact_loss,
              iterate_batches=paired_batches, training_callback=Callback())
        update(status="completed", ended_at=utc_now(), step=cfg["iters"], optimizer_restarts=0)
        save_next(run_dir, "compare_matched_control_and_shuffle_before_larger_scope",
                  final_checkpoint=str(run_dir / "checkpoints" / f"{cfg['iters']:07d}"),
                  generalization_status="validation_only; external_and_final_test_not_evaluated")
        return 0
    except Exception as exc:
        update(status="failed", ended_at=utc_now(), error_type=type(exc).__name__,
               error=str(exc)[:400])
        save_next(run_dir, "review_failure_before_any_training_restart", reason=type(exc).__name__)
        raise


def comparison_contract(cfg):
    keys = ("scope", "rank", "seed", "learning_rate", "batch_size", "max_seq_length", "iters",
            "checkpoint_steps", "expected_model_config_sha256", "enable_thinking")
    return canonical_digest({**{k: cfg[k] for k in keys}, "inputs": cfg["inputs"],
                             "experiment_identity_sha256": experiment_identity(cfg),
                             "protocol": SCHEMA, "noeeg_transform": "fixed_withheld_block_v2"})


def experiment_identity(cfg):
    keys = ("rank", "seed", "learning_rate", "batch_size", "max_seq_length", "enable_thinking")
    return canonical_digest({**{k: cfg[k] for k in keys}, "inputs": cfg["inputs"],
        "model_path": str(Path(cfg["model_path"]).expanduser().resolve()),
        "model_config_sha256": cfg["expected_model_config_sha256"],
        "base_receipt_sha256": cfg["base_receipt"]["sha256"], "protocol": SCHEMA})


def compare_checkpoints(eeg_dir, no_dir, output):
    eeg_dir, no_dir = Path(eeg_dir), Path(no_dir)
    eeg = json.loads((eeg_dir / "evaluation.json").read_text())
    no = json.loads((no_dir / "evaluation.json").read_text())
    require(eeg["condition"] == "eeg" and no["condition"] == "noeeg", "wrong_comparison_arms")
    require(all(x.get("status") == "completed" and x.get("quality_gates_passed") is True for x in (eeg, no)),
            "failed_or_incomplete_checkpoint_not_comparable")
    for key in ("step", "scope", "seed", "comparison_contract_sha256", "val_pairing_sha256"):
        require(eeg[key] == no[key], "comparison_contract_mismatch_" + key)
    ep = json.loads((eeg_dir / "predictions.private.json").read_text())
    npred = json.loads((no_dir / "predictions.private.json").read_text())
    difference = paired_difference(ep["eeg"], npred["noeeg"])
    report = {"schema": SCHEMA, "created_at": utc_now(), "step": eeg["step"],
              "scope": eeg["scope"], "metrics": {"eeg": eeg["metrics"]["eeg"],
              "noeeg": no["metrics"]["noeeg"], "shuffle": eeg["metrics"]["shuffle"]},
              "eeg_vs_noeeg": difference, "eeg_vs_shuffle": eeg["eeg_vs_shuffle"],
              "eeg_checkpoint_sha256": eeg["adapter_sha256"], "noeeg_checkpoint_sha256": no["adapter_sha256"],
              "test_opened": False, "development_only": True, "auto_scale_allowed": False,
              "experiment_identity_sha256": eeg["experiment_identity_sha256"],
              "quality_gates_passed": True,
              "next_action": "review_paired_EEG_effect_and_model_audit_before_bounded_all64_probe"}
    atomic_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run", "_train"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        if command == "preflight":
            p.add_argument("--receipt", required=True)
        if command == "_train":
            p.add_argument("--lock-fd", type=int, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--eeg-checkpoint", required=True)
    p.add_argument("--noeeg-checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "compare":
        compare_checkpoints(args.eeg_checkpoint, args.noeeg_checkpoint, args.output)
        return 0
    cfg = read_config(args.config)
    if args.command == "preflight":
        receipt, _, _ = preflight(cfg)
        atomic_json(args.receipt, receipt)
        print(json.dumps(receipt, ensure_ascii=False))
        return 0
    if args.command == "run":
        return run_supervisor(args.config, cfg)
    return train_worker(cfg, args.lock_fd)


if __name__ == "__main__":
    sys.exit(main())
