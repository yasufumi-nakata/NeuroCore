from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Channel:
    name: str
    type: str = "eeg"
    unit: str = "uV"
    position: tuple[float, float, float] | None = None
    reference: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Timebase:
    sampling_rate: float
    start_time: float = 0.0
    timestamps: tuple[float, ...] | None = None
    clock_source: str = "local"
    drift_ppm: float = 0.0


@dataclass(frozen=True)
class Event:
    onset_seconds: float
    label: str
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidityIssue:
    severity: str
    code: str
    message: str
    location: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "details": self.details,
        }


class FrameValidationError(ValueError):
    def __init__(self, issues: Iterable[ValidityIssue]):
        self.issues = list(issues)
        joined = "; ".join(f"{issue.code}: {issue.message}" for issue in self.issues)
        super().__init__(joined or "NeuroFrame validation failed")


@dataclass(frozen=True)
class NeuroFrame:
    data: np.ndarray
    channels: tuple[Channel, ...]
    timebase: Timebase
    events: tuple[Event, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    validity: tuple[ValidityIssue, ...] = ()

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=float)
        if data.ndim != 2:
            raise ValueError("NeuroFrame.data must be a 2D array shaped as samples x channels")
        if len(self.channels) != data.shape[1]:
            raise ValueError("channel count must match data.shape[1]")
        if self.timebase.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "validity", tuple(self.validity))

    @property
    def samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def channel_count(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.samples / self.timebase.sampling_rate

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

    def eeg_channel_indices(self) -> tuple[int, ...]:
        return tuple(index for index, channel in enumerate(self.channels) if channel.type.lower() == "eeg")

    def validate(self) -> tuple[ValidityIssue, ...]:
        issues: list[ValidityIssue] = list(self.validity)
        if self.samples == 0:
            issues.append(ValidityIssue("error", "empty_data", "NeuroFrame has no samples", "data"))
        if self.channel_count == 0:
            issues.append(ValidityIssue("error", "empty_channels", "NeuroFrame has no channels", "channels"))
        if not np.isfinite(self.data).all():
            bad_count = int(np.size(self.data) - np.isfinite(self.data).sum())
            issues.append(
                ValidityIssue(
                    "error",
                    "non_finite_data",
                    "NeuroFrame contains NaN or infinite values",
                    "data",
                    {"bad_value_count": bad_count},
                )
            )
        duplicate_names = _duplicates(self.channel_names)
        if duplicate_names:
            issues.append(
                ValidityIssue(
                    "error",
                    "duplicate_channels",
                    "Channel names must be unique",
                    "channels",
                    {"duplicates": duplicate_names},
                )
            )
        if not self.eeg_channel_indices():
            issues.append(ValidityIssue("warning", "no_eeg_channels", "No channels are typed as EEG", "channels"))
        if self.timebase.timestamps is not None and len(self.timebase.timestamps) != self.samples:
            issues.append(
                ValidityIssue(
                    "error",
                    "timestamp_length_mismatch",
                    "timestamps length must match sample count",
                    "timebase.timestamps",
                    {"timestamps": len(self.timebase.timestamps), "samples": self.samples},
                )
            )
        if abs(self.timebase.drift_ppm) > 100:
            issues.append(
                ValidityIssue(
                    "warning",
                    "clock_drift_high",
                    "Clock drift is high enough to require event realignment review",
                    "timebase.drift_ppm",
                    {"drift_ppm": self.timebase.drift_ppm},
                )
            )
        return tuple(issues)

    def require_valid(self) -> None:
        issues = [issue for issue in self.validate() if issue.severity in {"error", "critical"}]
        if issues:
            raise FrameValidationError(issues)

    def with_data(
        self,
        data: np.ndarray,
        *,
        channels: Iterable[Channel] | None = None,
        timebase: Timebase | None = None,
        operation: str | None = None,
        validity: Iterable[ValidityIssue] | None = None,
    ) -> "NeuroFrame":
        provenance = dict(self.provenance)
        if operation:
            steps = list(provenance.get("steps", []))
            steps.append(operation)
            provenance["steps"] = steps
        return NeuroFrame(
            data=np.asarray(data, dtype=float),
            channels=tuple(channels) if channels is not None else self.channels,
            timebase=timebase or self.timebase,
            events=self.events,
            provenance=provenance,
            validity=tuple(validity) if validity is not None else self.validity,
        )

    def to_summary(self) -> dict[str, Any]:
        return {
            "shape": [self.samples, self.channel_count],
            "duration_seconds": self.duration_seconds,
            "sampling_rate": self.timebase.sampling_rate,
            "channels": [
                {
                    "name": channel.name,
                    "type": channel.type,
                    "unit": channel.unit,
                    "reference": channel.reference,
                    "aliases": list(channel.aliases),
                }
                for channel in self.channels
            ],
            "events": [
                {
                    "onset_seconds": event.onset_seconds,
                    "label": event.label,
                    "duration_seconds": event.duration_seconds,
                    "metadata": event.metadata,
                }
                for event in self.events
            ],
            "validity": [issue.to_dict() for issue in self.validate()],
            "provenance": self.provenance,
        }

    def replace_timebase(self, **changes: Any) -> "NeuroFrame":
        return self.with_data(self.data.copy(), timebase=replace(self.timebase, **changes))


def _duplicates(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return sorted(duplicates)
