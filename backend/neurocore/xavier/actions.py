"""Finite action bridge for an explicitly authorised, isolated GUI session.

The model only proposes actions.  The application owns this object's arm state,
driver, goal, approved calibration IDs, and single-action human approvals.  None
of those capabilities may be populated from a decoder proposal or HTTP body.
Unknown model confidence is never treated as calibrated confidence.  A trusted
human may separately approve one exact proposal; the audit then records human
authority, not a model confidence of one.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlsplit

from ..control import ControlRouter, IntentCommand
from ..settings import ActionBinding, NeuroCoreSettings


SCHEMA = "neurocore-xavier-action-proposal-v1"
ISOLATED_SCOPE = "isolated_loopback_fixture"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = {
    "schema", "action", "confidence", "confidence_calibrated", "calibration_id",
    "executed", "eeg_positions", "model_id", "manifest_sha256", "mode", "reason",
}


def _digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _origin(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.username is not None or parsed.password is not None
            or parsed.port is None or not 1 <= parsed.port <= 65535
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("An explicit HTTP loopback fixture origin is required")
    return value.rstrip("/")


@dataclass(frozen=True)
class ActionBridgePolicy:
    session_id: str
    explicit_goal: str
    max_actions: int = 6
    max_attempts: int = 32
    wall_seconds: float = 600.0
    action_timeout_seconds: float = 15.0
    stop_paths: tuple[Path, ...] = ()
    allowed_model_ids: frozenset[str] = field(default_factory=frozenset)
    allowed_manifest_sha256: frozenset[str] = field(default_factory=frozenset)
    approved_calibration_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not _IDENTIFIER.fullmatch(self.session_id):
            raise ValueError("A bounded session identifier is required")
        if not isinstance(self.explicit_goal, str) or not 1 <= len(self.explicit_goal.strip()) <= 4096:
            raise ValueError("A bounded explicit goal is required")
        if type(self.max_actions) is not int or not 1 <= self.max_actions <= 64:
            raise ValueError("max_actions must be an integer in 1..64")
        if type(self.max_attempts) is not int or not self.max_actions <= self.max_attempts <= 256:
            raise ValueError("max_attempts must be an integer between max_actions and 256")
        if not _finite(self.wall_seconds) or not 0 < self.wall_seconds <= 1800:
            raise ValueError("A finite session deadline of at most 1800 seconds is required")
        if not _finite(self.action_timeout_seconds) or not 0 < self.action_timeout_seconds <= 30:
            raise ValueError("An action timeout of at most 30 seconds is required")
        if any(not isinstance(x, Path) or not x.is_absolute() for x in self.stop_paths):
            raise ValueError("STOP/HOLD paths must be explicit absolute paths")
        if any(not _SHA256.fullmatch(x) for x in self.allowed_manifest_sha256):
            raise ValueError("Invalid approved manifest SHA256")


@dataclass(frozen=True)
class HumanActionApproval:
    """Opaque, one-use approval issued by a bridge's trusted host application."""

    token: str
    proposal_sha256: str
    session_id: str
    goal_sha256: str


class IsolatedActionDriver(Protocol):
    session_id: str
    origin: str
    scope: str

    def execute(self, action: Mapping[str, Any], *, timeout_seconds: float) -> None:
        """Execute one click within the supplied finite timeout or raise."""


class IsolatedChromeDriver:
    """Adapter for the existing dedicated ``browser_pipe.Chrome`` fixture.

    The caller creates the reviewed Chrome implementation with its dedicated
    profile, CDP pipe, and network interception.  This adapter creates no browser,
    navigates nowhere, and exposes no arbitrary CDP method to the proposal.
    """

    scope = ISOLATED_SCOPE

    def __init__(self, browser: Any, *, session_id: str, origin: str):
        self.session_id = session_id
        self.origin = _origin(origin)
        self._browser = browser
        root = Path(browser.root).resolve(strict=True)
        profile = Path(browser.profile_path)
        if (profile.is_symlink() or profile.resolve(strict=True).parent != root
                or not profile.name.startswith("dedicated_chrome_")
                or browser.origin != self.origin):
            raise ValueError("The driver must use its dedicated fixture/profile")

    def execute(self, action: Mapping[str, Any], *, timeout_seconds: float) -> None:
        browser = self._browser
        if browser.closed or browser.p.poll() is not None:
            raise RuntimeError("Dedicated Chrome is no longer running")
        if not _finite(timeout_seconds) or timeout_seconds <= 0:
            raise TimeoutError("Action deadline elapsed")
        if action.get("action") != "click" or set(action) != {"action", "x", "y"}:
            raise ValueError("The fixture driver only accepts a bounded click")
        if any(type(action[k]) is not int or not 0 <= action[k] <= 1000 for k in ("x", "y")):
            raise ValueError("Normalized coordinates must be integers in 0..1000")
        previous_deadline = browser.deadline
        browser.deadline = min(previous_deadline, time.time() + timeout_seconds)
        try:
            if browser.origin != self.origin or browser.js("location.origin") != self.origin:
                raise RuntimeError("The active page left the isolated fixture")
            viewport = browser.js("({width:innerWidth,height:innerHeight})")
            if viewport != {"width": 1000, "height": 700}:
                raise RuntimeError("The reviewed fixture viewport changed")
            browser.click(action["x"] * 999 // 1000, action["y"] * 699 // 1000)
        finally:
            browser.deadline = previous_deadline


def validate_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the model contract without executing or recording private inputs."""

    if not isinstance(proposal, Mapping) or set(proposal) - _FIELDS:
        raise ValueError("Unexpected proposal fields")
    if proposal.get("schema") != SCHEMA or proposal.get("mode") != "explicit_goal_engineering":
        raise ValueError("Only the explicit-goal proposal contract is supported")
    if proposal.get("executed") is not False:
        raise ValueError("A proposal must be unexecuted")
    action = proposal.get("action")
    if not isinstance(action, Mapping):
        raise ValueError("Action must be an object")
    kind = action.get("action")
    if kind == "click":
        if set(action) != {"action", "x", "y"} or any(
            type(action[k]) is not int or not 0 <= action[k] <= 1000 for k in ("x", "y")
        ):
            raise ValueError("Click coordinates must be exact normalized integers")
    elif kind in {"finish", "abstain"}:
        if set(action) != {"action"}:
            raise ValueError("Terminal actions have no additional payload")
    else:
        raise ValueError("Only click, finish, and abstain are supported")
    confidence = proposal.get("confidence")
    if confidence is not None and (not _finite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("Confidence must be null or a finite probability")
    if type(proposal.get("confidence_calibrated")) is not bool:
        raise ValueError("Calibration status must be an explicit boolean")
    if type(proposal.get("eeg_positions")) is not int or proposal["eeg_positions"] not in {0, 5}:
        raise ValueError("The registered proposal uses zero or five EEG positions")
    if not isinstance(proposal.get("model_id"), str) or not 1 <= len(proposal["model_id"]) <= 200:
        raise ValueError("A bounded model identifier is required")
    if not isinstance(proposal.get("manifest_sha256"), str) or not _SHA256.fullmatch(proposal["manifest_sha256"]):
        raise ValueError("A fixed model manifest SHA256 is required")
    if "calibration_id" in proposal and (
        not isinstance(proposal["calibration_id"], str) or not _IDENTIFIER.fullmatch(proposal["calibration_id"])
    ):
        raise ValueError("Invalid calibration identifier")
    if "reason" in proposal and (not isinstance(proposal["reason"], str) or len(proposal["reason"]) > 2048):
        raise ValueError("Proposal reason must be bounded text")
    result = dict(proposal)
    result["action"] = dict(action)
    # Ensure there are no non-JSON values, NaNs, or custom objects hiding in input.
    return json.loads(json.dumps(result, allow_nan=False))


class XavierActionBridge:
    """One persistent router and finite budget per isolated GUI session.

    ``arm`` and ``approve`` are trusted application operations, not model tools.
    API endpoints should return proposals or call ``preview`` only.  A failed or
    timed-out driver attempt consumes its slot and closes the session because
    whether a click reached the fixture may be unknown; it is never retried.
    """

    def __init__(self, policy: ActionBridgePolicy, *, driver: IsolatedActionDriver | None = None,
                 settings: NeuroCoreSettings | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.policy = policy
        self.driver = driver
        self._clock = clock
        self._started = clock()
        if not _finite(self._started):
            raise ValueError("A finite monotonic clock is required")
        original = settings or NeuroCoreSettings.default()
        if any(x["severity"] == "error" for x in original.validate()):
            raise ValueError("Invalid NeuroCore settings")
        if not _finite(original.safety.max_actions_per_second) or original.safety.max_actions_per_second <= 0:
            raise ValueError("The action bridge requires a positive finite action rate")
        if not _finite(original.safety.min_confidence):
            raise ValueError("The action bridge requires finite confidence policy")
        configured = replace(original, safety=replace(original.safety, require_human_arm=True, human_armed=False),
                             bindings=(ActionBinding("xavier_click", "mouse", "click", "left"),))
        self.router = ControlRouter(configured)
        self._lock = threading.RLock()
        self._approvals: dict[str, tuple[HumanActionApproval, float]] = {}
        self._attempts = 0
        self._dispatches = 0
        self._completed_clicks = 0
        self._closed_reason: str | None = None
        self._audit: list[dict[str, Any]] = []
        self._driver_origin: str | None = None
        if driver is not None:
            if driver.session_id != policy.session_id or driver.scope != ISOLATED_SCOPE:
                raise ValueError("Driver/session isolation contract mismatch")
            if not policy.stop_paths:
                raise ValueError("An execution driver requires explicit STOP/HOLD file paths")
            self._driver_origin = _origin(driver.origin)

    def arm(self, armed: bool = True) -> None:
        if type(armed) is not bool:
            raise ValueError("Arm state must be a trusted boolean")
        with self._lock:
            self.router.settings = replace(self.router.settings,
                                           safety=replace(self.router.settings.safety, human_armed=armed))
            if not armed:
                self._approvals.clear()

    def stop(self) -> None:
        with self._lock:
            self.router.settings = replace(self.router.settings,
                                           safety=replace(self.router.settings.safety, emergency_stop=True))
            self._closed_reason = "emergency_stop"
            self._approvals.clear()

    def _guard(self) -> str | None:
        if self.router.settings.safety.emergency_stop:
            return "emergency_stop"
        for path in self.policy.stop_paths:
            if path.exists() or path.is_symlink():
                return "stop_or_hold_present"
        now = self._clock()
        if not _finite(now) or now < self._started:
            return "clock_invalid"
        if now - self._started >= self.policy.wall_seconds:
            return "session_deadline"
        if self._closed_reason:
            return self._closed_reason
        return None

    def approve(self, proposal: Mapping[str, Any], *, goal: str, session_id: str,
                ttl_seconds: float = 30.0) -> HumanActionApproval:
        """Record a human's approval of this exact action, for at most 30 seconds."""

        with self._lock:
            if self._guard() or not self.router.settings.safety.human_armed:
                raise RuntimeError("A live, armed session is required for human approval")
            if goal != self.policy.explicit_goal or session_id != self.policy.session_id:
                raise ValueError("Human approval goal/session mismatch")
            if not _finite(ttl_seconds) or not 0 < ttl_seconds <= 30:
                raise ValueError("Human approval lifetime must be at most 30 seconds")
            validated = validate_proposal(proposal)
            if validated["action"]["action"] != "click":
                raise ValueError("Human approval only applies to one click")
            now = self._clock()
            self._approvals = {k: v for k, v in self._approvals.items() if v[1] > now}
            if len(self._approvals) >= self.policy.max_actions:
                raise RuntimeError("Outstanding approval budget exhausted")
            approval = HumanActionApproval(secrets.token_hex(24), _digest(validated), session_id, _digest(goal))
            self._approvals[approval.token] = (approval, min(now + ttl_seconds, self._started + self.policy.wall_seconds))
            return approval

    def preview(self, proposal: Mapping[str, Any], *, goal: str, session_id: str) -> dict[str, Any]:
        return self.submit(proposal, goal=goal, session_id=session_id, execute=False)

    def submit(self, proposal: Mapping[str, Any], *, goal: str, session_id: str,
               human_approval: HumanActionApproval | None = None, execute: bool = False) -> dict[str, Any]:
        with self._lock:
            base: dict[str, Any] = {"session_id": self.policy.session_id, "executed": False,
                                   "blocked": True, "goal_completion_verified": False}

            def result(reason: str, **extra: Any) -> dict[str, Any]:
                output = {**base, "reason": reason, "attempts": self._attempts,
                          "action_dispatch_attempts": self._dispatches,
                          "completed_clicks": self._completed_clicks,
                          "actions_remaining": self.policy.max_actions - self._dispatches, **extra}
                if len(self._audit) < self.policy.max_attempts:
                    self._audit.append(dict(output))
                return output

            guard = self._guard()
            if guard:
                return result(guard)
            if self._attempts >= self.policy.max_attempts:
                return result("attempt_budget_exhausted")
            self._attempts += 1
            if type(execute) is not bool or goal != self.policy.explicit_goal or session_id != self.policy.session_id:
                return result("execution_context_mismatch")
            try:
                validated = validate_proposal(proposal)
            except (TypeError, ValueError, OverflowError):
                return result("invalid_proposal")
            proposal_sha = _digest(validated)
            base.update(proposal_sha256=proposal_sha, model_confidence=validated["confidence"],
                        confidence_calibrated=validated["confidence_calibrated"])
            if self.policy.allowed_model_ids and validated["model_id"] not in self.policy.allowed_model_ids:
                return result("model_not_approved")
            if self.policy.allowed_manifest_sha256 and validated["manifest_sha256"] not in self.policy.allowed_manifest_sha256:
                return result("model_manifest_not_approved")
            kind = validated["action"]["action"]
            if not execute:
                return result("preview_only", blocked=False, execution_authorized=False)
            if kind in {"finish", "abstain"}:
                self._closed_reason = "model_finished" if kind == "finish" else "model_abstained"
                self._approvals.clear()
                return result(self._closed_reason, blocked=False)
            if self._dispatches >= self.policy.max_actions:
                return result("action_budget_exhausted")
            if not self.router.settings.safety.human_armed:
                return result("human_arm_required")
            authority = "calibrated_model"
            accepted_approval = None
            if human_approval is not None:
                if not isinstance(human_approval, HumanActionApproval):
                    return result("invalid_human_approval")
                entry = self._approvals.get(human_approval.token)
                if (entry is None or entry[0] != human_approval or entry[1] <= self._clock()
                        or human_approval.proposal_sha256 != proposal_sha
                        or human_approval.session_id != session_id or human_approval.goal_sha256 != _digest(goal)):
                    return result("invalid_human_approval")
                accepted_approval = human_approval
                authority = "human_approved_exact_action"
                route_confidence = 1.0  # human authority only; model_confidence remains unchanged above.
            else:
                if validated["confidence"] is None or not validated["confidence_calibrated"]:
                    return result("uncalibrated_model_confidence")
                if validated.get("calibration_id") not in self.policy.approved_calibration_ids:
                    return result("calibration_not_approved")
                route_confidence = validated["confidence"]
            if self.driver is None:
                return result("isolated_driver_not_configured")
            if (self.driver.scope != ISOLATED_SCOPE or self.driver.session_id != session_id
                    or self.driver.origin != self._driver_origin):
                return result("driver_context_changed")
            now = self._clock()
            # A model-supplied timestamp is never allowed to reset or evade this persistent router.
            command = IntentCommand("xavier_click", route_confidence, source="xavier-bridge", timestamp=now)
            routed = self.router.route(command)
            if routed.blocked:
                return result(routed.reason, routing_authority=authority)
            guard = self._guard()
            if guard:
                return result(guard)
            if accepted_approval is not None:
                del self._approvals[accepted_approval.token]
            self._dispatches += 1  # reserve before attempting I/O; an uncertain failure cannot be replayed.
            remaining = min(self.policy.action_timeout_seconds,
                            self.policy.wall_seconds - (self._clock() - self._started))
            try:
                self.driver.execute(validated["action"], timeout_seconds=remaining)
            except Exception as exc:
                self._closed_reason = "driver_outcome_unresolved"
                self._approvals.clear()
                return result("driver_outcome_unresolved", routing_authority=authority,
                              error_type=type(exc).__name__, outcome_unknown=True)
            self._completed_clicks += 1
            return result("executed_in_isolated_fixture", blocked=False, executed=True,
                          routing_authority=authority, scope=ISOLATED_SCOPE)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"session_id": self.policy.session_id, "human_armed": self.router.settings.safety.human_armed,
                    "blocked_reason": self._guard(), "attempts": self._attempts,
                    "action_dispatch_attempts": self._dispatches, "completed_clicks": self._completed_clicks,
                    "actions_remaining": self.policy.max_actions - self._dispatches,
                    "isolated_driver_configured": self.driver is not None,
                    "goal_completion_verified": False, "audit_records": len(self._audit)}

    def audit(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._audit]
