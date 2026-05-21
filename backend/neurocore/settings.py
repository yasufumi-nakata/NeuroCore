from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_ACTION_KINDS = {"mouse", "keyboard", "agent"}
ALLOWED_MOUSE_TARGETS = {"move_x", "move_y", "click", "scroll_x", "scroll_y"}
ALLOWED_KEYBOARD_TARGETS = {"key", "text"}


@dataclass(frozen=True)
class DeviceProfile:
    name: str = "generic-eeg"
    sampling_rate: float = 250.0
    channels: tuple[str, ...] = ("Fz", "Cz", "Pz", "Oz")
    channel_aliases: dict[str, str] = field(default_factory=dict)
    reference: str = "common"


@dataclass(frozen=True)
class SignalPolicy:
    highpass_hz: float = 1.0
    lowpass_hz: float = 40.0
    target_sampling_rate: float = 250.0
    window_seconds: float = 1.0
    step_seconds: float = 0.25
    max_clock_drift_ppm: float = 100.0


@dataclass(frozen=True)
class SafetyPolicy:
    min_confidence: float = 0.75
    max_actions_per_second: float = 4.0
    require_human_arm: bool = True
    emergency_stop: bool = False
    block_prompt_like_agent_payloads: bool = True


@dataclass(frozen=True)
class ActionBinding:
    intent: str
    kind: str
    target: str
    value: str | float | int | bool | None = None


@dataclass(frozen=True)
class AgentPolicy:
    allowed_tools: tuple[str, ...] = ("open_task", "summarize_context", "draft_reply", "run_local_plan")
    max_payload_chars: int = 512
    require_untrusted_envelope: bool = True


@dataclass(frozen=True)
class NeuroCoreSettings:
    device: DeviceProfile = field(default_factory=DeviceProfile)
    signal: SignalPolicy = field(default_factory=SignalPolicy)
    safety: SafetyPolicy = field(default_factory=SafetyPolicy)
    agent: AgentPolicy = field(default_factory=AgentPolicy)
    bindings: tuple[ActionBinding, ...] = (
        ActionBinding("cursor_left", "mouse", "move_x", -24),
        ActionBinding("cursor_right", "mouse", "move_x", 24),
        ActionBinding("select", "mouse", "click", "left"),
        ActionBinding("cancel", "keyboard", "key", "Escape"),
        ActionBinding("agent_focus", "agent", "open_task", None),
    )

    @classmethod
    def default(cls) -> "NeuroCoreSettings":
        return cls()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NeuroCoreSettings":
        device_payload = payload.get("device", {})
        signal_payload = payload.get("signal", {})
        safety_payload = payload.get("safety", {})
        agent_payload = payload.get("agent", {})
        binding_payloads = payload.get("bindings", None)
        return cls(
            device=DeviceProfile(
                channels=tuple(device_payload.get("channels", DeviceProfile().channels)),
                channel_aliases=dict(device_payload.get("channel_aliases", {})),
                name=device_payload.get("name", DeviceProfile().name),
                reference=device_payload.get("reference", DeviceProfile().reference),
                sampling_rate=float(device_payload.get("sampling_rate", DeviceProfile().sampling_rate)),
            ),
            signal=SignalPolicy(**{**asdict(SignalPolicy()), **signal_payload}),
            safety=SafetyPolicy(**{**asdict(SafetyPolicy()), **safety_payload}),
            agent=AgentPolicy(
                allowed_tools=tuple(agent_payload.get("allowed_tools", AgentPolicy().allowed_tools)),
                max_payload_chars=int(agent_payload.get("max_payload_chars", AgentPolicy().max_payload_chars)),
                require_untrusted_envelope=bool(
                    agent_payload.get("require_untrusted_envelope", AgentPolicy().require_untrusted_envelope)
                ),
            ),
            bindings=tuple(ActionBinding(**item) for item in binding_payloads)
            if binding_payloads is not None
            else cls().bindings,
        )

    @classmethod
    def load(cls, path: str | Path) -> "NeuroCoreSettings":
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))

    def save(self, path: str | Path) -> None:
        Path(path).expanduser().write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def validate(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        if self.device.sampling_rate <= 0:
            issues.append({"severity": "error", "code": "invalid_sampling_rate", "message": "sampling_rate must be positive"})
        if not self.device.channels:
            issues.append({"severity": "error", "code": "no_channels", "message": "At least one EEG channel is required"})
        if len(self.device.channels) != len(set(self.device.channels)):
            issues.append({"severity": "error", "code": "duplicate_channels", "message": "Device channels must be unique"})
        if not (0.0 <= self.safety.min_confidence <= 1.0):
            issues.append({"severity": "error", "code": "invalid_confidence", "message": "min_confidence must be between 0 and 1"})
        if self.safety.max_actions_per_second < 0:
            issues.append(
                {"severity": "error", "code": "invalid_action_rate", "message": "max_actions_per_second must be non-negative"}
            )
        if self.signal.lowpass_hz <= self.signal.highpass_hz:
            issues.append({"severity": "error", "code": "invalid_band", "message": "lowpass_hz must be greater than highpass_hz"})
        if self.signal.highpass_hz < 0:
            issues.append({"severity": "error", "code": "invalid_highpass", "message": "highpass_hz must be non-negative"})
        if self.signal.lowpass_hz >= self.signal.target_sampling_rate / 2:
            issues.append(
                {
                    "severity": "error",
                    "code": "band_exceeds_target_nyquist",
                    "message": "lowpass_hz must be below half the target sampling rate",
                    "details": {"lowpass_hz": self.signal.lowpass_hz, "target_sampling_rate": self.signal.target_sampling_rate},
                }
            )
        if self.signal.window_seconds <= 0:
            issues.append({"severity": "error", "code": "invalid_window", "message": "window_seconds must be positive"})
        if self.signal.step_seconds <= 0:
            issues.append({"severity": "error", "code": "invalid_step", "message": "step_seconds must be positive"})
        if self.signal.step_seconds > self.signal.window_seconds:
            issues.append({"severity": "warning", "code": "step_exceeds_window", "message": "step_seconds exceeds window_seconds"})
        binding_intents = [binding.intent for binding in self.bindings]
        if len(binding_intents) != len(set(binding_intents)):
            issues.append({"severity": "error", "code": "duplicate_binding", "message": "Each intent can have one binding"})
        for binding in self.bindings:
            if binding.kind not in ALLOWED_ACTION_KINDS:
                issues.append(
                    {
                        "severity": "error",
                        "code": "unsupported_action_kind",
                        "message": "Binding kind must be mouse, keyboard, or agent",
                        "details": {"intent": binding.intent, "kind": binding.kind},
                    }
                )
            if binding.kind == "mouse" and binding.target not in ALLOWED_MOUSE_TARGETS:
                issues.append(
                    {
                        "severity": "error",
                        "code": "unsupported_mouse_target",
                        "message": "Mouse binding target is not supported",
                        "details": {"intent": binding.intent, "target": binding.target},
                    }
                )
            if binding.kind == "keyboard" and binding.target not in ALLOWED_KEYBOARD_TARGETS:
                issues.append(
                    {
                        "severity": "error",
                        "code": "unsupported_keyboard_target",
                        "message": "Keyboard binding target is not supported",
                        "details": {"intent": binding.intent, "target": binding.target},
                    }
                )
        unsupported_agent_tools = [
            binding.target for binding in self.bindings if binding.kind == "agent" and binding.target not in self.agent.allowed_tools
        ]
        if unsupported_agent_tools:
            issues.append(
                {
                    "severity": "error",
                    "code": "unsupported_agent_tool",
                    "message": "Agent bindings must target allowed tools",
                    "details": {"tools": sorted(set(unsupported_agent_tools))},
                }
            )
        return issues
