"""Local, user-driven desktop companion for Xavier proposals.

The HTTP host owns this manager and must bind it to loopback, validate Origin,
and require its local session token on mutations.  Model output never selects a
driver or authorises execution.  ``apply`` is a separate, explicit UI action and
only consumes an opaque ID for a proposal this manager obtained itself.

Screenshots, the explicit goal and EEG input stay in process memory.  Only the
requested screenshot and inference are sent to the configured Xavier service;
history exports contain operational metadata, not screenshots, EEG, or goals.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import secrets
import threading
import time
from typing import Any, Callable, Protocol

import numpy as np

from .xavier.actions import validate_proposal


class DesktopError(RuntimeError):
    def __init__(self, code: str, message: str, http_status: int = 409) -> None:
        super().__init__(message)
        self.code, self.message, self.http_status = code, message, http_status

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class DesktopCapture:
    png: bytes
    width: int
    height: int
    screen_width: int
    screen_height: int


class DesktopDriver(Protocol):
    def capture(self) -> DesktopCapture: ...

    def click(self, x: int, y: int) -> None: ...


class PyAutoGuiDesktopDriver:
    """Primary-display desktop access, imported only on the first capture.

    Screenshot pixels may differ from OS coordinates on Retina/HiDPI displays.
    Proposals use normalised coordinates; the manager maps them to the logical
    display size reported by PyAutoGUI before dispatching one left click.
    """

    def __init__(self) -> None:
        try:
            import pyautogui
        except Exception as exc:
            raise DesktopError("desktop_unavailable", "Install the desktop extra and open a graphical session", 503) from exc
        self._gui = pyautogui
        # Preserve PyAutoGUI's emergency mouse-corner stop.
        self._gui.FAILSAFE = True

    def capture(self) -> DesktopCapture:
        try:
            width, height = map(int, self._gui.size())
            if width < 2 or height < 2:
                raise ValueError("No primary display")
            image = self._gui.screenshot().convert("RGB")
            # The native screenshot uses display pixels, not logical coordinates.
            # A different aspect ratio can indicate a multi-monitor capture; do
            # not guess which part of it is the primary display.
            if abs(image.width / image.height - width / height) > 0.03:
                raise DesktopError("display_layout_unsupported", "Capture must contain the primary display only", 422)
            image.thumbnail((1600, 1200))
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=True)
            if output.tell() > 4 * 1024 * 1024:
                raise DesktopError("capture_too_large", "Screen image exceeds the model image limit", 422)
            return DesktopCapture(output.getvalue(), image.width, image.height, width, height)
        except DesktopError:
            raise
        except Exception as exc:
            raise DesktopError("screen_capture_failed", "Screen capture failed; check screen-recording permission", 503) from exc

    def click(self, x: int, y: int) -> None:
        # ``click`` is the only supported OS operation.  No string, key sequence,
        # URL, shell command, or module name from a proposal reaches this driver.
        self._gui.click(x=x, y=y, button="left")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _image(capture: DesktopCapture):
    try:
        from PIL import Image
        if not isinstance(capture, DesktopCapture) or not isinstance(capture.png, bytes):
            raise ValueError("Invalid capture")
        if len(capture.png) > 4 * 1024 * 1024 or not capture.png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Invalid PNG")
        if any(type(n) is not int or n < 2 for n in (
            capture.width, capture.height, capture.screen_width, capture.screen_height
        )) or capture.width * capture.height > 2_000_000:
            raise ValueError("Invalid display geometry")
        image = Image.open(io.BytesIO(capture.png))
        if image.format != "PNG" or image.size != (capture.width, capture.height):
            raise ValueError("Capture dimensions changed")
        image.load()
        return image.convert("RGB")
    except ImportError as exc:
        raise DesktopError("desktop_unavailable", "Install the desktop extra to read screenshots", 503) from exc
    except (OSError, ValueError, TypeError) as exc:
        raise DesktopError("invalid_capture", "The desktop driver returned an invalid screenshot", 503) from exc


def _same_screen(before: DesktopCapture, after: DesktopCapture, action: dict[str, Any]) -> bool:
    """Allow clocks/cursors/minor animation, reject navigation and target changes.

    A local target patch is checked in addition to the whole screen.  This is a
    change guard, not a semantic assertion that the target is safe to click.
    """
    if (before.width, before.height, before.screen_width, before.screen_height) != (
        after.width, after.height, after.screen_width, after.screen_height
    ):
        return False
    first, second = _image(before), _image(after)
    left = np.asarray(first.resize((128, 80)), dtype=np.float32)
    right = np.asarray(second.resize((128, 80)), dtype=np.float32)
    delta = np.abs(left - right)
    if float(delta.mean()) > 12 or float((delta.mean(axis=2) > 32).mean()) > 0.12:
        return False
    x, y = action["x"] * (before.width - 1) // 1000, action["y"] * (before.height - 1) // 1000
    half = max(16, min(before.width, before.height) // 40)
    box = (max(0, x - half), max(0, y - half), min(before.width, x + half + 1), min(before.height, y + half + 1))
    local = np.abs(np.asarray(first.crop(box), dtype=np.float32) - np.asarray(second.crop(box), dtype=np.float32))
    return float(local.mean()) <= 24


class DesktopSessionManager:
    """A single finite local session, with one-use server-owned proposals."""

    def __init__(self, xavier_service: Any, *, enabled: bool = False,
                 driver_factory: Callable[[], DesktopDriver] = PyAutoGuiDesktopDriver,
                 max_actions: int = 20, max_attempts: int = 40,
                 session_seconds: float = 900, proposal_seconds: float = 120,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if type(max_actions) is not int or not 1 <= max_actions <= 64:
            raise ValueError("max_actions must be in 1..64")
        if type(max_attempts) is not int or not max_actions <= max_attempts <= 128:
            raise ValueError("max_attempts must be between max_actions and 128")
        if any(type(n) not in (int, float) or not math.isfinite(n) or n <= 0 for n in (
            session_seconds, proposal_seconds
        )) or session_seconds > 1800 or proposal_seconds > 300:
            raise ValueError("Use finite session and proposal lifetimes")
        self._service, self.enabled, self._factory = xavier_service, enabled, driver_factory
        self._max_actions, self._max_attempts = max_actions, max_attempts
        self._session_seconds, self._proposal_seconds, self._clock = session_seconds, proposal_seconds, clock
        self._state_lock, self._gate = threading.RLock(), threading.Lock()
        self._cancel = threading.Event()
        self._driver: DesktopDriver | None = None
        self._session_id: str | None = None
        self._goal = ""
        self._model_id: str | None = None
        self._manifest_sha256: str | None = None
        self._started = self._clock()
        self._state = "idle"
        self._operation: str | None = None
        self._capture: DesktopCapture | None = None
        self._capture_id: str | None = None
        self._capture_time = 0.0
        self._pending: dict[str, Any] | None = None
        self._actions, self._attempts = 0, 0
        self._events: list[dict[str, Any]] = []
        self._used: set[str] = set()
        self._receipts: dict[str, dict[str, Any]] = {}
        self._model_history: list[dict[str, Any]] = []

    def _event(self, event: str, **metadata: Any) -> None:
        self._events.append({"at": _utc(), "event": event, **metadata})
        del self._events[:-256]

    def _clear_inputs(self) -> None:
        self._capture, self._capture_id, self._pending = None, None, None

    def _require(self, session_id: str, *, allow_terminal: bool = False) -> None:
        if not self.enabled:
            raise DesktopError("desktop_disabled", "Start NeuroCore with desktop access enabled", 403)
        if not isinstance(session_id, str) or self._session_id is None or not secrets.compare_digest(session_id, self._session_id):
            raise DesktopError("session_not_found", "Start a current desktop session", 404)
        now = self._clock()
        if not math.isfinite(now) or now < self._started:
            self._state = "failed"
            self._cancel.set()
            self._clear_inputs()
            raise DesktopError("clock_invalid", "The session clock changed; start a new session")
        if self._state not in {"stopped", "expired", "failed", "finished"} and now - self._started >= self._session_seconds:
            self._state = "expired"
            self._cancel.set()
            self._clear_inputs()
            self._event("session_expired")
        if not allow_terminal and self._state in {"stopped", "expired", "failed", "finished"}:
            raise DesktopError("session_" + self._state, "This desktop session has ended; start a new one")

    def _snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled, "state": self._state, "session_id": self._session_id,
            "busy": self._operation is not None, "operation": self._operation, "model_id": self._model_id,
            "capture_id": self._capture_id,
            "proposal_id": self._pending["proposal_id"] if self._pending else None,
            "actions_remaining": max(0, self._max_actions - self._actions),
            "attempts_remaining": max(0, self._max_attempts - self._attempts),
            "expires_in_seconds": max(0, round(self._session_seconds - (self._clock() - self._started), 1)) if self._session_id else 0,
            "events": _json_copy(self._events[-30:]),
        }

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            if self._session_id is not None:
                self._require(self._session_id, allow_terminal=True)
            return self._snapshot()

    def history(self) -> dict[str, Any]:
        with self._state_lock:
            return {"schema": "neurocore-desktop-history-v1", "session_id": self._session_id,
                    "state": self._state, "events": _json_copy(self._events)}

    @contextmanager
    def _operation_guard(self, name: str, session_id: str | None = None):
        if not self._gate.acquire(blocking=False):
            raise DesktopError("desktop_busy", "Another desktop operation is in progress")
        try:
            with self._state_lock:
                if session_id is not None:
                    self._require(session_id)
                self._operation = name
            yield
        finally:
            with self._state_lock:
                self._operation = None
            self._gate.release()

    @staticmethod
    def _delay_value(seconds: float) -> float:
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= 5:
            raise DesktopError("invalid_delay", "Screen switching delay must be between 0 and 5 seconds", 422)
        return float(seconds)

    def _wait(self, session_id: str, seconds: float) -> None:
        self._cancel.wait(seconds)
        with self._state_lock:
            self._require(session_id)

    def _take_capture(self) -> DesktopCapture:
        if self._driver is None:
            self._driver = self._factory()
        try:
            captured = self._driver.capture()
            _image(captured)
            return captured
        except DesktopError:
            raise
        except Exception as exc:
            raise DesktopError("screen_capture_failed", "Could not capture the local desktop", 503) from exc

    def start(self, goal: str, model_id: str | None = None) -> dict[str, Any]:
        if not self.enabled:
            raise DesktopError("desktop_disabled", "Start NeuroCore with desktop access enabled", 403)
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 4096:
            raise DesktopError("invalid_goal", "Provide a goal of at most 4096 characters", 422)
        if model_id is not None and (not isinstance(model_id, str) or not 1 <= len(model_id) <= 200):
            raise DesktopError("invalid_model", "Choose a loaded model", 422)
        with self._operation_guard("start"):
            model = self._service.status()
            loaded_id = model.get("model_id")
            manifest = model.get("model", {}).get("manifest_sha256")
            if model.get("status") != "loaded" or not loaded_id:
                raise DesktopError("model_not_loaded", "Load a Xavier model before starting desktop assistance")
            if model_id is not None and model_id != loaded_id:
                raise DesktopError("model_changed", "The selected model is not the loaded model")
            if not isinstance(manifest, str) or len(manifest) != 64 or any(c not in "0123456789abcdef" for c in manifest):
                raise DesktopError("model_contract_missing", "The loaded model has no verified manifest", 503)
            with self._state_lock:
                self._session_id, self._goal = secrets.token_urlsafe(24), goal.strip()
                self._model_id, self._manifest_sha256 = loaded_id, manifest
                self._started, self._state = self._clock(), "active"
                self._actions, self._attempts = 0, 0
                self._events, self._model_history, self._used, self._receipts = [], [], set(), {}
                self._cancel = threading.Event()
                self._clear_inputs()
                self._event("session_started", model_id=loaded_id, goal_sha256=hashlib.sha256(self._goal.encode()).hexdigest())
        return self.status()

    def capture(self, session_id: str, delay_seconds: float = 3) -> dict[str, Any]:
        delay = self._delay_value(delay_seconds)
        with self._operation_guard("capture", session_id):
            with self._state_lock:
                self._clear_inputs()
                self._state = "active"
            self._wait(session_id, delay)
            captured = self._take_capture()
            with self._state_lock:
                self._require(session_id)
                self._capture, self._capture_id = captured, secrets.token_urlsafe(24)
                self._capture_time, self._state = self._clock(), "captured"
                stamp = _utc()
                self._event("screen_captured", capture_id=self._capture_id, width=captured.width, height=captured.height)
                response = {"capture_id": self._capture_id, "image_png_base64": base64.b64encode(captured.png).decode("ascii"),
                            "width": captured.width, "height": captured.height, "captured_at": stamp}
        return {**self.status(), **response}

    def propose(self, session_id: str, eeg: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(eeg, dict) or set(eeg) - {"prefix", "channels", "sample_rate_hz", "units", "preprocessing"}:
            raise DesktopError("invalid_eeg", "Supply a prepared EEG input object", 422)
        # Copy the request once: application callers cannot mutate input during inference.
        try:
            eeg = _json_copy(eeg)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DesktopError("invalid_eeg", "EEG input must contain finite JSON values", 422) from exc
        with self._operation_guard("propose", session_id):
            with self._state_lock:
                if self._capture is None:
                    raise DesktopError("capture_required", "Capture the operation target first")
                if self._clock() - self._capture_time >= self._proposal_seconds:
                    self._clear_inputs()
                    self._state = "active"
                    raise DesktopError("capture_expired", "The screenshot expired; capture it again")
                if self._attempts >= self._max_attempts:
                    raise DesktopError("attempt_budget_exhausted", "The session inference limit has been reached")
                self._attempts += 1
                self._pending = None
                self._state = "captured"
                captured, capture_id = self._capture, self._capture_id
                payload = {**eeg, "image_png_base64": base64.b64encode(captured.png).decode("ascii"),
                           "goal": self._goal, "history": _json_copy(self._model_history[-20:])}
            try:
                result = self._service.propose(payload)
            except Exception:
                with self._state_lock:
                    self._event("proposal_failed")
                raise
            try:
                if not isinstance(result, dict):
                    raise ValueError("Invalid service proposal")
                clean = {key: value for key, value in result.items() if key not in {"scientific_validation", "execution_policy"}}
                result = validate_proposal(clean)
            except (ValueError, TypeError) as exc:
                with self._state_lock:
                    self._event("proposal_invalid")
                raise DesktopError("invalid_model_proposal", "The model returned an unsupported action; request another proposal", 502) from exc
            with self._state_lock:
                self._require(session_id)
                if result["model_id"] != self._model_id or result["manifest_sha256"] != self._manifest_sha256:
                    self._clear_inputs()
                    self._state = "active"
                    raise DesktopError("model_changed", "The model changed; start a new session")
                if self._clock() - self._capture_time >= self._proposal_seconds:
                    self._clear_inputs()
                    self._state = "active"
                    raise DesktopError("capture_expired", "The screenshot expired during inference; capture again")
                proposal_id = secrets.token_urlsafe(24)
                self._pending = {"proposal_id": proposal_id, "proposal": result, "capture_id": capture_id,
                                 "expires": min(self._capture_time + self._proposal_seconds, self._started + self._session_seconds)}
                self._state = "proposed"
                self._event("action_proposed", proposal_id=proposal_id, action=result["action"]["action"], model_id=self._model_id)
                response = {"proposal_id": proposal_id, "action": _json_copy(result["action"]), "reason": result.get("reason", ""),
                            "model_id": result["model_id"], "manifest_sha256": result["manifest_sha256"],
                            "eeg_positions": result["eeg_positions"], "proposal_expires_in_seconds": max(0, self._pending["expires"] - self._clock())}
        return {**self.status(), **response}

    def apply(self, session_id: str, proposal_id: str, delay_seconds: float = 3) -> dict[str, Any]:
        delay = self._delay_value(delay_seconds)
        if not isinstance(proposal_id, str) or not 1 <= len(proposal_id) <= 128:
            raise DesktopError("invalid_proposal", "Choose the current proposal", 422)
        with self._state_lock:
            self._require(session_id, allow_terminal=True)
            if proposal_id in self._receipts:
                return {**_json_copy(self._receipts[proposal_id]), "replayed": True}
        with self._operation_guard("apply", session_id):
            with self._state_lock:
                if proposal_id in self._used:
                    raise DesktopError("proposal_consumed", "This proposal was already consumed; capture again")
                pending = self._pending
                if pending is None or not secrets.compare_digest(proposal_id, pending["proposal_id"]):
                    raise DesktopError("proposal_not_found", "Choose the current server-issued proposal", 404)
                captured = self._capture
                self._pending = None
                self._used.add(proposal_id)
                action = _json_copy(pending["proposal"]["action"])
                if self._clock() >= pending["expires"] or captured is None:
                    self._clear_inputs()
                    self._state = "active"
                    raise DesktopError("proposal_expired", "The proposal expired; capture again")
                if action["action"] == "click" and self._actions >= self._max_actions:
                    self._clear_inputs()
                    self._state = "active"
                    raise DesktopError("action_budget_exhausted", "The session action limit has been reached")
            if action["action"] == "click":
                self._wait(session_id, delay)
                try:
                    current = self._take_capture()
                except Exception:
                    with self._state_lock:
                        self._clear_inputs()
                        if not self._cancel.is_set():
                            self._state = "active"
                        self._event("screen_verification_failed", proposal_id=proposal_id)
                    raise
                with self._state_lock:
                    self._require(session_id)
                    if self._clock() >= pending["expires"]:
                        self._clear_inputs()
                        self._state = "active"
                        raise DesktopError("proposal_expired", "The proposal expired; capture again")
                    if not _same_screen(captured, current, action):
                        self._clear_inputs()
                        self._state = "active"
                        self._event("screen_changed", proposal_id=proposal_id)
                        raise DesktopError("screen_changed", "The target screen changed; capture and request a new proposal")
                    x = action["x"] * (current.screen_width - 1) // 1000
                    y = action["y"] * (current.screen_height - 1) // 1000
                    # Consume before dispatch.  If the OS call fails its outcome
                    # may be unknown; it must never be automatically retried.
                    self._actions += 1
                    self._event("action_dispatch", proposal_id=proposal_id, action="click", authority="user_apply")
                # Do not hold the state mutex in native OS code.  Stop can then
                # return immediately; an already-dispatched click may complete,
                # and its receipt still truthfully records that one action.
                try:
                    assert self._driver is not None
                    self._driver.click(x, y)
                except Exception as exc:
                    with self._state_lock:
                        self._state = "failed"
                        self._cancel.set()
                        self._clear_inputs()
                        self._event("action_outcome_unknown", proposal_id=proposal_id, action="click")
                    raise DesktopError("action_outcome_unknown", "The OS action outcome is unknown; inspect the target before starting a new session", 503) from exc
                with self._state_lock:
                    if not self._cancel.is_set():
                        self._model_history.append({"action": action, "executed": True})
                        self._state = "active"
                    executed, outcome = True, "applied"
            else:
                with self._state_lock:
                    self._require(session_id)
                    self._state = "finished"
                    self._cancel.set()
                    executed, outcome = False, "finished" if action["action"] == "finish" else "abstained"
            with self._state_lock:
                self._clear_inputs()
                self._event("action_result", proposal_id=proposal_id, action=action["action"], executed=executed, result=outcome)
                response = {"proposal_id": proposal_id, "action": action, "executed": executed, "result": outcome, "replayed": False}
        response = {**self.status(), **response}
        with self._state_lock:
            self._receipts[proposal_id] = _json_copy(response)
        return response

    def stop(self, session_id: str) -> dict[str, Any]:
        # This path does not wait for the inference/operation gate.  A late
        # inference result cannot recreate a proposal after stop.
        with self._state_lock:
            self._require(session_id, allow_terminal=True)
            self._cancel.set()
            self._state = "stopped"
            self._clear_inputs()
            self._goal = ""
            self._model_history = []
            self._event("session_stopped")
            return self._snapshot()
