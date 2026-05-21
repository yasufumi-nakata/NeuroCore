from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .settings import ActionBinding, NeuroCoreSettings


PROMPT_LIKE_PATTERN = re.compile(
    r"(ignore\s+previous|system\s+prompt|developer\s+message|reveal\s+instructions|sudo|rm\s+-rf|curl\s+.+\|\s*sh)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IntentCommand:
    intent: str
    confidence: float
    source: str = "external-decoder"
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ControlAction:
    kind: str
    target: str
    value: str | int | float | bool | dict[str, Any] | None
    blocked: bool = False
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "value": self.value,
            "blocked": self.blocked,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class ControlRouter:
    """Routes externally decoded intents into actions.

    NeuroCore deliberately does not contain classifier weights or decision models.
    A decoder supplies an IntentCommand; this router only validates and envelopes it.
    """

    def __init__(self, settings: NeuroCoreSettings | None = None):
        self.settings = settings or NeuroCoreSettings.default()
        self._bindings = {binding.intent: binding for binding in self.settings.bindings}
        self._last_action_at: float | None = None

    def route(self, command: IntentCommand) -> ControlAction:
        if self.settings.safety.emergency_stop:
            return _blocked("system", "emergency_stop", "Emergency stop is active")
        if command.confidence < self.settings.safety.min_confidence:
            return _blocked(command.intent, "low_confidence", "Intent confidence is below the configured threshold")
        now = command.timestamp
        if self.settings.safety.max_actions_per_second > 0:
            min_interval = 1.0 / self.settings.safety.max_actions_per_second
            if self._last_action_at is not None and now - self._last_action_at < min_interval:
                return _blocked(command.intent, "rate_limited", "Action rate limit is active")
        binding = self._bindings.get(command.intent)
        if binding is None:
            return _blocked(command.intent, "unbound_intent", "No action binding is configured for this intent")
        action = self._build_action(binding, command)
        if not action.blocked:
            self._last_action_at = now
        return action

    def _build_action(self, binding: ActionBinding, command: IntentCommand) -> ControlAction:
        if binding.kind not in {"mouse", "keyboard", "agent"}:
            return _blocked(binding.intent, "unsupported_action_kind", f"Unsupported action kind: {binding.kind}")
        if binding.kind == "agent":
            return self._build_agent_action(binding, command)
        return ControlAction(
            kind=binding.kind,
            target=binding.target,
            value=binding.value,
            metadata={"intent": command.intent, "confidence": command.confidence, "source": command.source},
        )

    def _build_agent_action(self, binding: ActionBinding, command: IntentCommand) -> ControlAction:
        if binding.target not in self.settings.agent.allowed_tools:
            return _blocked(command.intent, "agent_tool_not_allowed", "Agent target is not in the allowlist")
        payload_text = str(command.payload.get("text", ""))
        if len(payload_text) > self.settings.agent.max_payload_chars:
            return _blocked(command.intent, "agent_payload_too_large", "Agent payload exceeds max_payload_chars")
        if self.settings.safety.block_prompt_like_agent_payloads and PROMPT_LIKE_PATTERN.search(payload_text):
            return _blocked(command.intent, "agent_prompt_like_payload", "Prompt-like agent payload was blocked")
        envelope = {
            "tool": binding.target,
            "intent": command.intent,
            "confidence": command.confidence,
            "payload": command.payload,
            "trust": "untrusted_decoded_intent",
            "execution": "requires_agent_policy",
        }
        if self.settings.agent.require_untrusted_envelope:
            value: dict[str, Any] = envelope
        else:
            value = command.payload
        return ControlAction(
            kind="agent",
            target=binding.target,
            value=value,
            metadata={"source": command.source, "enveloped": self.settings.agent.require_untrusted_envelope},
        )


def _blocked(target: str, reason: str, message: str) -> ControlAction:
    return ControlAction(kind="blocked", target=target, value=None, blocked=True, reason=reason, metadata={"message": message})
