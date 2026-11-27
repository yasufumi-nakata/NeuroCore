"""Optional Metal implementation of the verified source30 model.

The native VLM image merger follows the existing A06 implementation. Forecasts
use the text core's ordinary hidden-state return, not locally patched
``return_hidden`` or ``skip_logits`` generation keywords.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
import sys
import time

import numpy as np

from .model import MAIN_KEYS, strict_json


def sanitize_history(history):
    if not isinstance(history, list) or len(history) > 32:
        raise ValueError("at most 32 prior actions are accepted")
    copied = []
    for event in history:
        if not isinstance(event, dict):
            raise ValueError("history events must be objects")
        if set(event) == {"action", "x", "y"} and isinstance(event["action"], str) and event["action"] in {"click", "move"}:
            if any(type(event[k]) is not int or not 0 <= event[k] <= 1000 for k in ("x", "y")):
                raise ValueError("history coordinates must use normalized integer units")
        elif set(event) == {"kind", "mono_ns"} and isinstance(event["kind"], str) and event["kind"] in {"key_down", "click", "scroll", "move", "focus", "submit"}:
            if type(event["mono_ns"]) is not int or event["mono_ns"] < 0:
                raise ValueError("history timestamps must be nonnegative integers")
        else:
            raise ValueError("history may contain only sanitized prior event kinds or clicks")
        copied.append(dict(event))
    timed = [event["mono_ns"] for event in copied if "mono_ns" in event]
    if any(a > b for a, b in zip(timed, timed[1:])):
        raise ValueError("history timestamps must be ordered")
    return copied


def parse_action(text):
    """Treat an unparseable or unsupported model response as abstention."""
    try:
        action = strict_json(text.strip())
        if not isinstance(action, dict):
            raise ValueError("object required")
        if set(action) == {"action"} and action["action"] in {"finish", "abstain"}:
            return action, None
        if (set(action) == {"action", "x", "y"} and action["action"] == "click"
                and all(type(action[k]) is int and 0 <= action[k] <= 1000 for k in ("x", "y"))):
            return action, None
    except (ValueError, TypeError, KeyError):
        pass
    return {"action": "abstain"}, "model_response_did_not_match_action_schema"


class MLXBackend:
    def __init__(self, package):
        self.package = package
        self.spec = package.manifest
        try:
            import mlx.core as mx
            from mlx_vlm import load
            from mlx_vlm.trainer.peft.lora_layers import LoRALinear
        except ImportError as exc:
            raise RuntimeError("unsupported_hardware: install the pinned optional MLX runtime on an Apple Silicon Mac") from exc
        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            raise RuntimeError("unsupported_hardware: an actual MLX Metal GPU is required")
        actual_device = mx.metal.device_info()
        self.device_info = {key: actual_device[key] for key in (
            "device_name", "name", "architecture", "memory_size", "max_recommended_working_set_size"
        ) if key in actual_device}
        for name, expected in self.spec["runtime_versions"].items():
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError("runtime_version_mismatch: missing " + name) from exc
            if actual != expected:
                raise RuntimeError("runtime_version_mismatch: " + name)
        self.mx = mx
        self.limit = 50_000_000_000
        mx.set_memory_limit(self.limit)
        mx.set_cache_limit(0)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        with contextlib.redirect_stdout(sys.stderr):
            self.model, self.processor = load(str(package.base_model_dir), strict=True, lazy=False,
                                              trust_remote_code=False, local_files_only=True)
        self.model.freeze()
        self.model.eval()
        self.core = self.model.language_model.model
        width = int(self.core.norm.weight.shape[0])
        if width != self.spec["architecture"]["hidden_size"] or len(self.core.layers) != 64:
            raise RuntimeError("loaded base does not match the registered 27B architecture")
        final = self.core.layers[-1].mlp
        adapter = LoRALinear.from_base(final.down_proj, r=4, scale=1.0, dropout=0.0)
        weights = mx.load(str(package.package_dir / self.spec["weights"]["file"]))
        matrices = mx.load(str(package.package_dir / self.spec["matrices"]["file"]))
        if set(weights) != MAIN_KEYS or set(matrices) != {"fixed", "compress"}:
            raise ValueError("loaded tensor keys differ from the inference manifest")
        for kind, values in (("weights", weights), ("matrices", matrices)):
            for name, value in values.items():
                if list(value.shape) != self.spec[kind]["tensors"][name] or not bool(mx.all(mx.isfinite(value)).item()):
                    raise ValueError("loaded tensor shape or finiteness mismatch")
        if weights["adapter.lora_a"].shape != adapter.lora_a.shape or weights["adapter.lora_b"].shape != adapter.lora_b.shape:
            raise ValueError("LoRA dimensions do not match the fixed base")
        adapter.lora_a = weights["adapter.lora_a"]
        adapter.lora_b = weights["adapter.lora_b"]
        adapter.freeze()
        final.down_proj = adapter
        self.adapter = adapter
        self.weights = weights
        self.fixed = matrices["fixed"]
        self.compress = matrices["compress"]
        token_ids = [self.processor.tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 7)]
        if any(len(row) != 1 for row in token_ids) or [row[0] for row in token_ids] != self.spec["position_encoding"]["token_ids"]:
            raise ValueError("position tokenizer identity mismatch")
        self.one = mx.array([[row[0] for row in token_ids]], dtype=mx.int32)
        self.embedding = self.core.embed_tokens(self.one)
        embedding32 = np.ascontiguousarray(np.asarray(self.embedding.astype(mx.float32)))
        digest = hashlib.sha256(embedding32.tobytes()).hexdigest()
        self.amplitude = float(mx.sqrt(mx.mean(self.embedding.astype(mx.float32) ** 2)).item())
        pos = self.spec["position_encoding"]
        if digest != pos["embedding_sha256"] or self.amplitude != float(pos["amplitude"]):
            raise ValueError("position embedding or amplitude identity mismatch")
        self.model.freeze()
        from mlx.utils import tree_flatten
        if tree_flatten(self.model.trainable_parameters()):
            raise RuntimeError("inference model still contains trainable parameters")
        self._budget()

    def _budget(self, deadline=None):
        if int(self.mx.get_peak_memory()) > self.limit:
            raise RuntimeError("model memory budget exceeded")
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("inference deadline exceeded")

    def _mode(self, forecast):
        # Match the verified learner's unfused rotary arithmetic for EEG forecasts.
        for layer in self.core.layers:
            layer.train() if forecast else layer.eval()
            if not layer.is_linear and hasattr(layer.self_attn.rotary_emb, "fused_apply"):
                layer.self_attn.rotary_emb.fused_apply = not forecast

    def _eeg_embedding(self, features):
        mx = self.mx
        x = mx.array(features[None], dtype=mx.float32)
        down = self.weights["project_down.weight"]
        up = self.weights["project_up.weight"]
        z = x @ self.fixed + (x @ down.T) @ up.T
        return self.embedding[:, :5, :] + (self.amplitude * z).astype(self.embedding.dtype)

    def forecast(self, features, anchor):
        mx = self.mx
        self._mode(True)
        measured = self._eeg_embedding(features)
        inputs = mx.concatenate([measured, self.embedding[:, 5:, :]], axis=1)
        positions = mx.broadcast_to(mx.arange(6)[None, None, :], (3, 1, 6))
        # This is the native text core return; no patched hidden-output API is used.
        hidden = self.core(self.one, inputs_embeds=inputs, position_ids=positions, cache=None)
        h = hidden[:, -1, :].astype(mx.float32)
        h = h * mx.rsqrt(mx.mean(h * h, axis=-1, keepdims=True) + 1e-8)
        compressed = h @ self.compress
        logits = compressed @ self.weights["tuned.weight"].T + self.weights["tuned.bias"]
        value = mx.array(anchor)[None] + 3.0 * mx.tanh(logits)
        mx.eval(value)
        result = np.asarray(value[0].astype(mx.float32)).copy()
        self._budget()
        mx.clear_cache()
        return result

    def propose_action(self, features, image, history, goal, max_tokens, timeout_seconds):
        if not isinstance(image, bytes) or not 0 < len(image) <= 4 * 1024 * 1024:
            raise ValueError("bounded PNG bytes required")
        if not isinstance(goal, str) or not 0 < len(goal) <= 4096:
            raise ValueError("a bounded explicit goal is required")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 64:
            raise ValueError("generation is limited to 64 tokens")
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120:
            raise ValueError("inference timeout must be in (0, 120] seconds")
        history = sanitize_history(history)
        from PIL import Image
        with Image.open(io.BytesIO(image)) as candidate:
            if candidate.format != "PNG" or candidate.width * candidate.height > 2_000_000:
                raise ValueError("bounded PNG dimensions required")
            candidate.verify()
        with Image.open(io.BytesIO(image)) as candidate:
            picture = candidate.convert("RGB")
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs
        mx = self.mx
        self._mode(False)
        deadline = time.monotonic() + float(timeout_seconds)
        prompt = ("You control only the shown isolated application. Explicit goal: " + goal
                  + "\nPrior permitted actions (coordinates on BOTH axes are normalized 0 through 1000): "
                  + json.dumps(history, ensure_ascii=False, separators=(",", ":"))
                  + '\nReturn exactly one JSON object: {"action":"click","x":INTEGER,"y":INTEGER}, '
                  + '{"action":"finish"}, or {"action":"abstain"}. Both coordinates use 0 through 1000. '
                  + "Use only the visible screen and past observations. Finish only when the goal is visibly satisfied. "
                  + "Do not emit code, URLs, explanations or additional fields.")
        formatted = apply_chat_template(self.processor, self.model.config, prompt, num_images=1,
                                        enable_thinking=False, add_generation_prompt=True)
        image_id = self.model.config.image_token_index
        prepared = prepare_inputs(self.processor, images=[picture], prompts=[formatted],
                                  image_token_index=image_id, return_tensors="mlx")
        ids, pixels, mask = prepared["input_ids"], prepared["pixel_values"], prepared["attention_mask"]
        if ids.shape[0] != 1 or int(mask.min().item()) != 1 or ids.shape[1] > 4096:
            raise ValueError("bounded unpadded single-image prompt required")
        native = self.model.get_input_embeddings(ids, pixels, image_grid_thw=prepared["image_grid_thw"], mask=mask)
        joined, positions = native.inputs_embeds, native.position_ids
        count = 0 if features is None else 5
        if count:
            ids = mx.concatenate([self.one[:, :5], ids], axis=1)
            joined = mx.concatenate([self._eeg_embedding(features), joined], axis=1)
            positions = mx.concatenate([mx.broadcast_to(mx.arange(5)[None, None, :], (3, 1, 5)), positions + 5], axis=-1)
        if positions.shape != (3, 1, ids.shape[1]) or joined.shape[:2] != ids.shape:
            raise RuntimeError("native image / EEG rotary geometry mismatch")
        next_position = int(positions.max().item()) + 1
        tokens = []
        started = time.monotonic()
        # Qwen3.5 uses a hybrid SSM/KV cache; only the wrapper knows the
        # appropriate cache type for each layer. A fresh instance is mandatory
        # for each request to avoid mixing screen or EEG sessions.
        cache = self.model.language_model.make_cache()
        extra = None
        try:
            for _ in range(max_tokens):
                self._budget(deadline)
                if extra is None:
                    hidden = self.core(ids, inputs_embeds=joined,
                                       position_ids=positions, cache=cache)
                else:
                    # Process only the previously generated token. Replaying the
                    # entire visual prompt on every step was O(n^2) per token
                    # and routinely exhausted the 60-second product deadline.
                    hidden = self.core(
                        extra, inputs_embeds=self.core.embed_tokens(extra),
                        position_ids=mx.full((3, 1, 1), next_position,
                                             dtype=positions.dtype),
                        cache=cache)
                    next_position += 1
                last = hidden[:, -1:, :]
                language = self.model.language_model
                logits = (self.core.embed_tokens.as_linear(last)
                          if language.args.tie_word_embeddings else
                          language.lm_head(last))
                token = int(mx.argmax(logits[0, -1, :]).item())
                self._budget(deadline)
                if self.processor.tokenizer.stopping_criteria(token):
                    break
                tokens.append(token)
                text = self.processor.tokenizer.decode(tokens)
                if len(text) > 2048:
                    raise RuntimeError("generation character budget exceeded")
                action, error = parse_action(text)
                if error is None:
                    break
                extra = mx.array([[token]], dtype=mx.int32)
        finally:
            # Free per-request multimodal KV and recurrent state even if timed
            # out. The registered model weights remain loaded for future use.
            del cache
            mx.clear_cache()
        text = self.processor.tokenizer.decode(tokens)
        action, reason = parse_action(text)
        mx.clear_cache()
        return {"action": action, "reason": reason, "eeg_positions": count, "eeg_input_used": bool(count),
                "image_sha256": hashlib.sha256(image).hexdigest(), "generated_tokens": len(tokens),
                "seconds": time.monotonic() - started, "causal_timestamp_verified": False,
                "coordinates": "normalized_0_through_1000_both_axes"}

    def close(self):
        self.model = None
        self.processor = None
        self.core = None
        self.adapter = None
        self.weights = None
        self.fixed = None
        self.compress = None
        self.embedding = None
        self.one = None
        import gc
        gc.collect()
        mx = getattr(self, "mx", None)
        if mx is not None:
            mx.clear_cache()
