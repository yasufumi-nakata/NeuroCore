from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .control import ControlAction, IntentCommand


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    timestamp: float
    intent: dict[str, Any] | None
    action: dict[str, Any]
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "intent": self.intent,
            "action": self.action,
            "note": self.note,
        }


@dataclass
class ActionAuditLog:
    events: list[AuditEvent] = field(default_factory=list)

    def record(self, action: ControlAction, command: IntentCommand | None = None, *, note: str = "") -> AuditEvent:
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            intent=_command_to_dict(command) if command is not None else None,
            action=action.to_dict(),
            note=note,
        )
        self.events.append(event)
        return event

    def extend(self, events: Iterable[AuditEvent]) -> None:
        self.events.extend(events)

    def to_dict(self) -> dict[str, Any]:
        return {"events": [event.to_dict() for event in self.events]}

    def write_jsonl(self, path: str | Path) -> None:
        resolved = Path(path).expanduser()
        resolved.write_text(
            "".join(json.dumps(event.to_dict(), ensure_ascii=False) + "\n" for event in self.events),
            encoding="utf-8",
        )


class DryRunActionSink:
    """Collects actions without touching the OS or a live agent runtime."""

    def __init__(self, audit_log: ActionAuditLog | None = None):
        self.audit_log = audit_log or ActionAuditLog()

    def submit(self, action: ControlAction, command: IntentCommand | None = None) -> AuditEvent:
        note = "blocked" if action.blocked else "dry_run_only"
        return self.audit_log.record(action, command, note=note)


def _command_to_dict(command: IntentCommand) -> dict[str, Any]:
    return {
        "intent": command.intent,
        "confidence": command.confidence,
        "source": command.source,
        "payload": command.payload,
        "timestamp": command.timestamp,
    }
