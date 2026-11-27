"""Persistent, single-model inference lifecycle. No input or prediction storage."""
from __future__ import annotations

import base64
import binascii
from contextlib import contextmanager
import json
try:
    import fcntl
except ImportError:  # Windows clients use the remote inference transport.
    fcntl = None
import os
from pathlib import Path
import stat
from threading import Lock
from typing import Any, Callable

import numpy as np

from .registry import ModelRegistry


class XavierServiceError(RuntimeError):
    def __init__(self, code: str, message: str, http_status: int = 422) -> None:
        super().__init__(message)
        self.code, self.message, self.http_status = code, message, http_status

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message,
                "scientific_validation": "not_established"}


def _factory(*args: Any, **kwargs: Any) -> Any:
    from .model import XavierModel
    return XavierModel(*args, **kwargs)


class XavierService:
    def __init__(self, registry: ModelRegistry, *, model_factory: Callable[..., Any] | None = None,
                 configuration_error: bool = False, gpu_lock_path: str | Path | None = None) -> None:
        self.registry = registry
        self._factory = model_factory or _factory
        self._gate = Lock()
        self._snapshot_lock = Lock()
        self._model: Any = None
        self._model_id: str | None = None
        self._phase = "configuration_error" if configuration_error else "unloaded"
        self._operation: str | None = None
        self._configuration_error = configuration_error
        self._model_status: dict[str, Any] = {}
        self._gpu_fd: int | None = None
        self._gpu_lock_path = Path(gpu_lock_path or os.environ.get("XAVIER_GPU_LOCK",
                                                f"/tmp/neurocore-xavier-{getattr(os, 'getuid', lambda: 0)()}.gpu.lock"))

    @classmethod
    def from_environment(cls) -> XavierService:
        try:
            return cls(ModelRegistry.from_environment())
        except (OSError, ValueError, TypeError):
            return cls(ModelRegistry(), configuration_error=True)

    def list_models(self) -> dict[str, Any]:
        return {"models": self.registry.list(), "configured": not self._configuration_error,
                "scientific_validation": "not_established"}

    def status(self) -> dict[str, Any]:
        with self._snapshot_lock:
            return {"status": self._phase, "model_id": self._model_id,
                    "busy": self._operation is not None, "operation": self._operation,
                    "model": dict(self._model_status), "persistence": "server_process_memory",
                    "input_storage": "none", "execution": "proposal_only",
                    "scientific_validation": "not_established"}

    def _record(self, *, phase: str | None = None, model_status: dict[str, Any] | None = None) -> None:
        with self._snapshot_lock:
            if phase is not None:
                self._phase = phase
            if model_status is not None:
                self._model_status = model_status

    @contextmanager
    def _operation_guard(self, name: str):
        if self._configuration_error:
            raise XavierServiceError("registry_invalid", "Server model registry is invalid", 503)
        if not self._gate.acquire(blocking=False):
            raise XavierServiceError("busy", "Another model operation is in progress", 409)
        with self._snapshot_lock:
            self._operation = name
        try:
            yield
        finally:
            with self._snapshot_lock:
                self._operation = None
            self._gate.release()

    def _entry(self, model_id: str):
        try:
            return self.registry.get(model_id)
        except KeyError:
            raise XavierServiceError("model_not_registered", "Choose a registered model ID", 404) from None

    def _construct(self, model_id: str):
        entry = self._entry(model_id)
        return self._factory(entry.package_dir, entry.base_model_dir,
                             expected_manifest_sha256=entry.manifest_sha256)

    @staticmethod
    def _error(exc: Exception, *, loading: bool = False) -> XavierServiceError:
        if isinstance(exc, XavierServiceError):
            return exc
        code = getattr(exc, "code", "")
        if code in {"unsupported_hardware", "unsupported_platform", "backend_unavailable"} or (
            isinstance(exc, (ImportError, RuntimeError)) and
            any(term in str(exc).lower() for term in ("unsupported_hardware", "apple silicon", "mlx is", "mlx unavailable"))
        ):
            return XavierServiceError("unsupported_hardware", "This model requires its supported MLX hardware runtime", 503)
        if isinstance(exc, RuntimeError) and str(exc).startswith("runtime_version_mismatch"):
            return XavierServiceError("runtime_version_mismatch", "The model requires its pinned backend package versions", 503)
        if isinstance(exc, TimeoutError):
            return XavierServiceError("inference_timeout", "The finite inference deadline was exceeded", 504)
        if isinstance(exc, OSError):
            return XavierServiceError("artifact_unavailable", "A registered model artifact is unavailable", 503)
        if isinstance(exc, ValueError) or code in {"hash_mismatch", "manifest_invalid", "contract_mismatch"}:
            return XavierServiceError("contract_mismatch" if loading else "invalid_input",
                                      "Model artifact contract does not match" if loading else "Input does not match model contract")
        if isinstance(exc, ImportError):
            return XavierServiceError("backend_unavailable", "Install the optional model backend on supported hardware", 503)
        return XavierServiceError("backend_failure", "Model operation failed", 500)

    def inspect(self, model_id: str) -> dict[str, Any]:
        with self._operation_guard("inspect"):
            try:
                model = self._construct(model_id)
                result = model.status()
                return {"registered": self._entry(model_id).public(), "model": result,
                        "scientific_validation": "not_established"}
            except Exception as exc:
                raise self._error(exc, loading=True) from None

    def _acquire_gpu_lock(self) -> None:
        if fcntl is None:
            raise XavierServiceError("unsupported_hardware", "Use an Apple Silicon model server through --remote-model", 503)
        if not self._gpu_lock_path.is_absolute():
            raise XavierServiceError("gpu_lock_invalid", "GPU lock path must be configured as absolute", 503)
        fd = None
        try:
            fd = os.open(self._gpu_lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            value = os.fstat(fd)
            if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid() or value.st_nlink != 1:
                raise OSError("Unsafe GPU lock file")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            current = self._gpu_lock_path.stat(follow_symlinks=False)
            if (value.st_dev, value.st_ino) != (current.st_dev, current.st_ino):
                raise OSError("GPU lock replaced")
            self._gpu_fd, fd = fd, None
        except BlockingIOError:
            raise XavierServiceError("gpu_busy", "GPU is owned by another process", 409) from None
        except OSError:
            raise XavierServiceError("gpu_lock_invalid", "GPU ownership lock is unavailable", 503) from None
        finally:
            if fd is not None:
                os.close(fd)

    def _release_gpu_lock(self) -> None:
        if self._gpu_fd is not None:
            fcntl.flock(self._gpu_fd, fcntl.LOCK_UN)
            os.close(self._gpu_fd)
            self._gpu_fd = None

    def load(self, model_id: str) -> dict[str, Any]:
        with self._operation_guard("load"):
            self._entry(model_id)
            if self._model is not None:
                if self._model_id == model_id and self._phase == "loaded":
                    result = self.status()
                    result.update(busy=False, operation=None)
                    return result
                raise XavierServiceError("model_already_loaded", "Unload the current model before loading another", 409)
            self._record(phase="loading")
            try:
                self._model = self._construct(model_id)
                self._model_id = model_id
                self._acquire_gpu_lock()
                self._model.load()
                self._record(phase="loaded", model_status=self._model.status())
            except Exception as exc:
                cleanup_ok = True
                if self._model is not None:
                    try:
                        self._model.unload()
                    except Exception:
                        cleanup_ok = False
                if cleanup_ok:
                    self._release_gpu_lock()
                    self._model = None
                    self._model_id = None
                    self._record(phase="unloaded", model_status={})
                else:
                    self._record(phase="failed_cleanup_required")
                raise self._error(exc, loading=True) from None
        return self.status()

    def unload(self) -> dict[str, Any]:
        with self._operation_guard("unload"):
            if self._model is not None:
                try:
                    self._model.unload()
                except Exception as exc:
                    self._record(phase="failed_cleanup_required")
                    raise self._error(exc) from None
                self._model = None
                self._model_id = None
            self._release_gpu_lock()
            self._record(phase="unloaded", model_status={})
        return self.status()

    @staticmethod
    def _input(payload: dict[str, Any], *, propose: bool = False):
        allowed = {"prefix", "channels", "sample_rate_hz", "units", "preprocessing"}
        if propose:
            allowed |= {"image_png_base64", "history", "goal", "eeg_condition"}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise XavierServiceError("invalid_input", "Unexpected input fields")
        if "preprocessing" in payload:
            metadata = payload["preprocessing"]
            expected = {"id", "filter_coefficients_sha256", "native_sample_rate_hz", "native_samples",
                        "warmup_seconds", "context_seconds", "hardware_clock_certified", "physical_raw_timing_certified"}
            if not isinstance(metadata, dict) or set(metadata) != expected or (
                metadata["id"] != "ds006735_causal30_v1" or
                metadata["filter_coefficients_sha256"] != "ab896fb5b6313fb585c33d52ff6e3599ae0d39e9c7e1b18616aa7ec29ed5a4fe" or
                metadata["native_sample_rate_hz"] != 10000 or metadata["native_samples"] != 25000 or
                metadata["warmup_seconds"] != 1.0 or metadata["context_seconds"] != 1.5 or
                metadata["hardware_clock_certified"] is not False or metadata["physical_raw_timing_certified"] is not False
            ):
                raise XavierServiceError("invalid_input", "Unknown preprocessing metadata")
        if propose and payload.get("eeg_condition", "present") not in {"present", "no_eeg"}:
            raise XavierServiceError("invalid_input", "EEG condition must be present or no_eeg")
        rows = payload.get("prefix")
        if not isinstance(rows, list) or len(rows) != 300 or any(not isinstance(row, list) or len(row) != 30 for row in rows):
            raise XavierServiceError("invalid_input", "prefix must contain exactly 300 samples by 30 channels")
        if any(type(value) not in (int, float) for row in rows for value in row):
            raise XavierServiceError("invalid_input", "prefix must contain numeric samples")
        try:
            prefix = np.asarray(rows, dtype=np.float32)
        except (ValueError, TypeError, OverflowError):
            raise XavierServiceError("invalid_input", "Invalid numeric prefix") from None
        if not np.isfinite(prefix).all():
            raise XavierServiceError("invalid_input", "prefix must contain only finite values")
        channels = payload.get("channels")
        if not isinstance(channels, list) or len(channels) != 30 or any(
            not isinstance(channel, str) or len(channel) > 64 for channel in channels
        ) or len(set(channels)) != 30:
            raise XavierServiceError("invalid_input", "Provide the model's ordered 30 channel names")
        if type(payload.get("sample_rate_hz", 200)) not in (int, float) or payload.get("sample_rate_hz", 200) != 200:
            raise XavierServiceError("invalid_input", "This model expects preprocessed input at 200 Hz")
        if payload.get("units", "uV") != "uV":
            raise XavierServiceError("invalid_input", "Input units must be uV")
        return prefix, {"channels": channels, "sample_rate_hz": 200, "units": "uV"}

    def _require_loaded(self):
        if self._model is None or self._phase != "loaded":
            raise XavierServiceError("model_not_loaded", "Load a registered model first", 409)
        return self._model

    def forecast(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._operation_guard("forecast"):
            model = self._require_loaded()
            prefix, kwargs = self._input(payload)
            try:
                result = model.forecast(prefix, **kwargs)
                self._record(model_status=model.status())
                return {**result, "scientific_validation": "not_established", "executed": False}
            except Exception as exc:
                raise self._error(exc) from None

    def propose(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._operation_guard("propose"):
            model = self._require_loaded()
            prefix, kwargs = self._input(payload, propose=True)
            image_text, goal, history = payload.get("image_png_base64"), payload.get("goal"), payload.get("history", [])
            if not isinstance(image_text, str) or len(image_text) > 5_592_408:
                raise XavierServiceError("invalid_input", "Provide a PNG image of at most 4 MiB")
            try:
                png = base64.b64decode(image_text, validate=True)
            except (ValueError, binascii.Error):
                raise XavierServiceError("invalid_input", "Invalid base64 PNG") from None
            if len(png) > 4 * 1024 * 1024 or not png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise XavierServiceError("invalid_input", "Invalid PNG input")
            if not isinstance(goal, str) or not goal.strip() or len(goal) > 4096:
                raise XavierServiceError("invalid_input", "An explicit goal of at most 4096 characters is required")
            if not isinstance(history, list) or len(history) > 32 or any(not isinstance(row, dict) for row in history):
                raise XavierServiceError("invalid_input", "history must contain at most 32 past action objects")
            try:
                if len(json.dumps(history, allow_nan=False)) > 16384:
                    raise ValueError("history too large")
                # Exact per-trial EEG ablation: same screen, prompt, and model;
                # omit the five EEG input positions without changing any weights.
                # This is an inference control, not a separately trained no-EEG baseline.
                condition = payload.get("eeg_condition", "present")
                result = model.propose_action(prefix if condition == "present" else None,
                                              png, history, goal, **kwargs)
                self._record(model_status=model.status())
                return {**result, "input_condition": condition,
                        "scientific_validation": "not_established", "executed": False,
                        "execution_policy": "proposal_only_no_driver"}
            except Exception as exc:
                raise self._error(exc) from None
