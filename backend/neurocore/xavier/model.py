"""Verified, inference-only Xavier model package. MLX is imported only on load()."""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURE_CONTRACT, encode_prefix, validate_sensor

SCHEMA = "neurocore-xavier-source30-v1"
MAIN_KEYS = {"adapter.lora_a", "adapter.lora_b", "project_down.weight", "project_up.weight", "tuned.weight", "tuned.bias"}
MATRIX_KEYS = {"fixed", "compress"}
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("duplicate JSON field")
        out[key] = value
    return out


def strict_json(text: str) -> Any:
    def reject(_):
        raise ValueError("non-finite JSON value")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=reject)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(root: Path, name: str, expected_sha: str, expected_bytes: int | None = None) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("package assets must be local relative paths")
    if not isinstance(expected_sha, str) or not _HEX.fullmatch(expected_sha):
        raise ValueError("asset SHA-256 is required")
    target = root / name
    if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("asset must be a regular contained file")
    before = target.stat()
    if expected_bytes is not None and (type(expected_bytes) is not int or before.st_size != expected_bytes):
        raise ValueError("asset size mismatch")
    if sha256(target) != expected_sha:
        raise ValueError("asset hash mismatch")
    after = target.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("asset changed while being verified")
    return target


def safetensors_header(path: Path, shapes: dict[str, list[int]], *, matrices: bool = False) -> None:
    """Check exact keys, shapes and byte ranges without importing a model framework."""
    size = path.stat().st_size
    if not 8 <= size <= 64 * 1024 * 1024:
        raise ValueError("inference tensor asset exceeds its size budget")
    with path.open("rb") as handle:
        raw = handle.read(8)
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= min(size - 8, 1024 * 1024):
            raise ValueError("invalid safetensors header length")
        header = strict_json(handle.read(length).decode("utf-8"))
    if not isinstance(header, dict) or "__metadata__" in header or set(header) != set(shapes):
        raise ValueError("unexpected tensor keys or metadata")
    intervals = []
    for name, entry in header.items():
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("invalid tensor header")
        shape = entry["shape"]
        if shape != shapes[name] or not shape or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError("tensor shape mismatch")
        dtype = entry["dtype"]
        if dtype not in ({"F32"} if matrices else {"F32", "F16", "BF16"}):
            raise ValueError("unsupported inference tensor dtype")
        offsets = entry["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2 or any(type(v) is not int for v in offsets)
                or offsets[0] < 0 or offsets[1] - offsets[0] != math.prod(shape) * (4 if dtype == "F32" else 2)
                or offsets[1] > size - 8 - length):
            raise ValueError("invalid tensor byte range")
        intervals.append(tuple(offsets))
    intervals.sort()
    if intervals[0][0] != 0 or any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])) or intervals[-1][1] != size - 8 - length:
        raise ValueError("tensor data has gaps, overlaps, or trailing bytes")


def validate_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported Xavier model package schema")
    if not isinstance(manifest.get("model_id"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", manifest["model_id"]):
        raise ValueError("bounded model ID required")
    spec = manifest["input"]
    channels = spec["channels"]
    if (not isinstance(channels, list) or len(channels) != 30 or len(set(channels)) != 30
            or any(not isinstance(c, str) or not c or len(c) > 32 for c in channels)
            or spec["sample_rate_hz"] != 200 or spec["units"] != "uV" or spec["prefix_samples"] != 300
            or spec["feature_positions"] != 5 or spec["feature_contract"] != FEATURE_CONTRACT):
        raise ValueError("exact source30 input contract required")
    arch = manifest["architecture"]
    width = arch["hidden_size"]
    if (type(width) is not int or not 256 <= width <= 16384 or arch["rank"] != 4 or arch["project_rank"] != 8
            or arch["compressed_size"] != 96 or arch["residual_limit_uv"] != 3.0):
        raise ValueError("unsupported source30 architecture")
    expected = {"project_down.weight": [8, 30], "project_up.weight": [width, 8],
                "tuned.weight": [30, 96], "tuned.bias": [30]}
    weights = manifest["weights"]["tensors"]
    if set(weights) != MAIN_KEYS or any(weights[k] != v for k, v in expected.items()):
        raise ValueError("exact six inference tensors required")
    for name in ("adapter.lora_a", "adapter.lora_b"):
        shape = weights[name]
        if not isinstance(shape, list) or len(shape) != 2 or any(type(x) is not int or x <= 0 or x > 100000 for x in shape) or 4 not in shape:
            raise ValueError("invalid rank-four LoRA tensor")
    if manifest["matrices"]["tensors"] != {"fixed": [30, width], "compress": [width, 96]}:
        raise ValueError("fixed projection matrices must be explicit")
    base = manifest["base"]
    if (base.get("repository") != "mlx-community/Qwen3.8-27B-4bit"
            or base.get("revision") != "b935172e4bb5dcd0543774a34748fb92422e3027"
            or not isinstance(base.get("assets_sha256"), dict) or "config.json" not in base["assets_sha256"]
            or not isinstance(base.get("shards"), list) or len(base["shards"]) != 3):
        raise ValueError("fixed public base revision and assets required")
    pos = manifest["position_encoding"]
    if (not isinstance(pos["token_ids"], list) or len(pos["token_ids"]) != 6
            or any(type(v) is not int or v < 0 for v in pos["token_ids"])
            or not _HEX.fullmatch(str(pos["embedding_sha256"]))
            or not isinstance(pos["amplitude"], (int, float)) or not math.isfinite(pos["amplitude"]) or pos["amplitude"] <= 0):
        raise ValueError("position token and embedding identity required")
    training = manifest["training"]
    if any(training.get(k) is not False for k in ("quality_accepted", "scale_authorized", "independent_evaluation")):
        raise ValueError("this artifact has no quality, scaling, or independent-evaluation acceptance")
    if type(training.get("optimizer_updates")) is not int or not 1 <= training["optimizer_updates"] <= 16:
        raise ValueError("bounded completed source30 update count required")
    versions = manifest["runtime_versions"]
    if not isinstance(versions, dict) or not {"mlx", "mlx-vlm", "transformers", "numpy"} <= set(versions):
        raise ValueError("pinned runtime versions required")


class XavierModel:
    """A CPU-verifiable model artifact, with explicit optional GPU initialization."""

    def __init__(self, package_dir: str | Path, base_model_dir: str | Path,
                 expected_manifest_sha256: str | None = None):
        self.package_dir = Path(package_dir).resolve()
        self.base_model_dir = Path(base_model_dir).resolve()
        path = self.package_dir / "manifest.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise ValueError("bounded regular inference manifest required")
        self.manifest_sha256 = sha256(path)
        if expected_manifest_sha256 is not None and self.manifest_sha256 != expected_manifest_sha256:
            raise ValueError("manifest SHA-256 mismatch")
        self.manifest = strict_json(path.read_text(encoding="utf-8"))
        try:
            validate_manifest(self.manifest)
        except (KeyError, TypeError) as exc:
            raise ValueError("incomplete or malformed inference manifest") from exc
        self._verify_package()
        self._backend = None
        self._loaded = False
        self._last_inference = None
        self._lock = threading.RLock()

    def _verify_package(self) -> None:
        if sha256(self.package_dir / "manifest.json") != self.manifest_sha256:
            raise ValueError("manifest changed")
        for name in ("weights", "matrices"):
            record = self.manifest[name]
            path = checked_file(self.package_dir, record["file"], record["sha256"], record["bytes"])
            safetensors_header(path, record["tensors"], matrices=name == "matrices")

    def verify_base(self) -> None:
        base = self.manifest["base"]
        for name, digest in base["assets_sha256"].items():
            checked_file(self.base_model_dir, name, digest)
        for record in base["shards"]:
            checked_file(self.base_model_dir, record["name"], record["sha256"], record["bytes"])

    def status(self) -> dict[str, Any]:
        return {"model_id": self.manifest["model_id"], "manifest_sha256": self.manifest_sha256,
                "package_verified": True, "loaded": self._loaded,
                "backend": "mlx-metal" if self._loaded else None,
                "device": getattr(self._backend, "device_info", None),
                "physical_gpu_utilization_percent_measured": False,
                "input": self.manifest["input"], "training": self.manifest["training"],
                "last_inference": self._last_inference,
                "capabilities": {"causal_processed_eeg_forecast": True, "explicit_goal_action_proposal": True,
                                 "autonomous_eeg_intent_validated": False, "os_action_execution": False}}

    def load(self) -> "XavierModel":
        with self._lock:
            if self._backend is not None and not self._loaded:
                raise RuntimeError("previous model load requires cleanup before reuse")
            if self._backend is None:
                self._verify_package()
                self.verify_base()
                from .mlx_backend import MLXBackend
                # Register the partially initialized object before allocating the
                # base so a failed initialization remains explicitly cleanable.
                self._backend = MLXBackend.__new__(MLXBackend)
                try:
                    self._backend.__init__(self)
                    self._loaded = True
                except BaseException as exc:
                    import traceback
                    traceback.clear_frames(exc.__traceback__)
                    self._backend.close()
                    self._backend = None
                    self._loaded = False
                    raise
        return self

    def close(self) -> None:
        with self._lock:
            if self._backend is not None:
                self._backend.close()
                self._backend = None
                self._loaded = False

    def unload(self) -> None:
        """Release this service's model; equivalent to close()."""
        self.close()

    def _features(self, prefix, channels, sample_rate_hz, units):
        validate_sensor(channels, self.manifest["input"]["channels"], sample_rate_hz, units)
        return encode_prefix(prefix)

    def forecast(self, prefix: np.ndarray, *, channels, sample_rate_hz=200, units="uV") -> dict[str, Any]:
        features, anchor = self._features(prefix, channels, sample_rate_hz, units)
        with self._lock:
            if self._backend is None or not self._loaded:
                raise RuntimeError("call load() before model inference")
            value = np.asarray(self._backend.forecast(features, anchor), dtype=np.float32)
        if value.shape != (30,) or not np.isfinite(value).all():
            raise RuntimeError("model returned an invalid forecast")
        return {"schema": "neurocore-xavier-forecast-v1", "model_id": self.manifest["model_id"],
                "manifest_sha256": self.manifest_sha256, "prediction_uv": value.tolist(),
                "channels": list(channels), "context_seconds": 1.5, "horizon_seconds": 0.1,
                "prediction_of": "causal processed-stream mean; filter delay is uncompensated",
                "physical_raw_timing_certified": False, "gradient_updates": 0,
                "independent_evaluation": False}

    def propose_action(self, prefix, image: bytes, history: list, goal: str, *, channels=None,
                       sample_rate_hz=200, units="uV", max_tokens=64, timeout_seconds=60) -> dict[str, Any]:
        features = None if prefix is None else self._features(prefix, channels, sample_rate_hz, units)[0]
        with self._lock:
            if self._backend is None or not self._loaded:
                raise RuntimeError("call load() before model inference")
            result = self._backend.propose_action(features, image, history, goal, max_tokens, timeout_seconds)
            self._last_inference = {key: value for key, value in result.items() if key not in {"action", "reason"}}
            self._last_inference.update(kind="action_proposal", gradient_updates=0,
                                        independent_eeg_intent_validated=False)
        return {"action": result["action"], "reason": result.get("reason") or "", "eeg_positions": result["eeg_positions"],
                "schema": "neurocore-xavier-action-proposal-v1", "model_id": self.manifest["model_id"],
                "manifest_sha256": self.manifest_sha256, "confidence": None, "confidence_calibrated": False,
                "executed": False, "mode": "explicit_goal_engineering"}
