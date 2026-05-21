from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .frame import FrameValidationError, NeuroFrame, Timebase, ValidityIssue


class Kernel(Protocol):
    name: str

    def validate(self, frame: NeuroFrame) -> tuple[ValidityIssue, ...]:
        ...

    def run(self, frame: NeuroFrame) -> NeuroFrame | "FeatureSet":
        ...


@dataclass(frozen=True)
class FeatureSet:
    name: str
    values: dict[str, float]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "values": self.values, "metadata": self.metadata}


@dataclass(frozen=True)
class ValidateEEG:
    min_channels: int = 1
    name: str = "validate_eeg"

    def validate(self, frame: NeuroFrame) -> tuple[ValidityIssue, ...]:
        return ()

    def run(self, frame: NeuroFrame) -> NeuroFrame:
        issues = list(frame.validate())
        eeg_count = len(frame.eeg_channel_indices())
        if eeg_count < self.min_channels:
            issues.append(
                ValidityIssue(
                    "error",
                    "insufficient_eeg_channels",
                    "Not enough EEG channels for this pipeline",
                    "channels",
                    {"required": self.min_channels, "actual": eeg_count},
                )
            )
        errors = [issue for issue in issues if issue.severity in {"error", "critical"}]
        if errors:
            raise FrameValidationError(errors)
        return frame.with_data(frame.data.copy(), operation=self.name, validity=issues)


@dataclass(frozen=True)
class Resample:
    target_rate: float
    name: str = "resample"

    def validate(self, frame: NeuroFrame) -> tuple[ValidityIssue, ...]:
        if self.target_rate <= 0:
            return (ValidityIssue("error", "invalid_target_rate", "target_rate must be positive", "kernel.target_rate"),)
        return ()

    def run(self, frame: NeuroFrame) -> NeuroFrame:
        issues = self.validate(frame)
        if issues:
            raise FrameValidationError(issues)
        source_rate = frame.timebase.sampling_rate
        if np.isclose(source_rate, self.target_rate):
            return frame.with_data(frame.data.copy(), operation=f"{self.name}:{self.target_rate:g}Hz")
        duration = frame.duration_seconds
        source_t = np.arange(frame.samples, dtype=float) / source_rate
        target_samples = max(1, int(round(duration * self.target_rate)))
        target_t = np.arange(target_samples, dtype=float) / self.target_rate
        out = np.empty((target_samples, frame.channel_count), dtype=float)
        for channel_index in range(frame.channel_count):
            out[:, channel_index] = np.interp(target_t, source_t, frame.data[:, channel_index])
        timebase = Timebase(
            sampling_rate=float(self.target_rate),
            start_time=frame.timebase.start_time,
            clock_source=frame.timebase.clock_source,
            drift_ppm=frame.timebase.drift_ppm,
        )
        return frame.with_data(out, timebase=timebase, operation=f"{self.name}:{source_rate:g}->{self.target_rate:g}Hz")


@dataclass(frozen=True)
class Bandpass:
    low_hz: float
    high_hz: float
    name: str = "bandpass"

    def validate(self, frame: NeuroFrame) -> tuple[ValidityIssue, ...]:
        nyquist = frame.timebase.sampling_rate / 2.0
        issues: list[ValidityIssue] = []
        if self.low_hz < 0:
            issues.append(ValidityIssue("error", "invalid_low_cut", "low_hz must be non-negative", "kernel.low_hz"))
        if self.high_hz <= self.low_hz:
            issues.append(ValidityIssue("error", "invalid_band", "high_hz must be greater than low_hz", "kernel.high_hz"))
        if self.high_hz >= nyquist:
            issues.append(
                ValidityIssue(
                    "error",
                    "band_exceeds_nyquist",
                    "high_hz must be below Nyquist for the current sampling rate",
                    "kernel.high_hz",
                    {"high_hz": self.high_hz, "nyquist": nyquist},
                )
            )
        return tuple(issues)

    def run(self, frame: NeuroFrame) -> NeuroFrame:
        issues = self.validate(frame)
        if issues:
            raise FrameValidationError(issues)
        freqs = np.fft.rfftfreq(frame.samples, d=1.0 / frame.timebase.sampling_rate)
        spectrum = np.fft.rfft(frame.data, axis=0)
        mask = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        filtered = np.fft.irfft(spectrum * mask[:, None], n=frame.samples, axis=0)
        return frame.with_data(filtered, operation=f"{self.name}:{self.low_hz:g}-{self.high_hz:g}Hz")


@dataclass(frozen=True)
class ReReference:
    strategy: str = "average"
    name: str = "re_reference"

    def validate(self, frame: NeuroFrame) -> tuple[ValidityIssue, ...]:
        if self.strategy != "average":
            return (ValidityIssue("error", "unsupported_reference", "Only average reference is implemented", "kernel.strategy"),)
        if len(frame.eeg_channel_indices()) < 2:
            return (
                ValidityIssue(
                    "warning",
                    "single_channel_reference",
                    "Average reference with fewer than two EEG channels is a no-op",
                    "channels",
                ),
            )
        return ()

    def run(self, frame: NeuroFrame) -> NeuroFrame:
        issues = self.validate(frame)
        errors = [issue for issue in issues if issue.severity in {"error", "critical"}]
        if errors:
            raise FrameValidationError(errors)
        eeg_indices = frame.eeg_channel_indices()
        out = frame.data.copy()
        if len(eeg_indices) >= 2:
            out[:, eeg_indices] = out[:, eeg_indices] - out[:, eeg_indices].mean(axis=1, keepdims=True)
        return frame.with_data(out, operation=f"{self.name}:{self.strategy}", validity=tuple(frame.validity) + tuple(issues))


@dataclass(frozen=True)
class SpectralFeatures:
    bands: dict[str, tuple[float, float]] | None = None
    name: str = "spectral_features"

    def __post_init__(self) -> None:
        if self.bands is None:
            object.__setattr__(
                self,
                "bands",
                {
                    "theta": (4.0, 8.0),
                    "alpha": (8.0, 13.0),
                    "beta": (13.0, 30.0),
                },
            )

    def validate(self, frame: NeuroFrame) -> tuple[ValidityIssue, ...]:
        nyquist = frame.timebase.sampling_rate / 2.0
        issues = []
        for name, (low, high) in (self.bands or {}).items():
            if low < 0 or high <= low:
                issues.append(ValidityIssue("error", "invalid_feature_band", "Invalid spectral band", f"bands.{name}"))
            if high >= nyquist:
                issues.append(
                    ValidityIssue(
                        "error",
                        "feature_band_exceeds_nyquist",
                        "Spectral feature band exceeds Nyquist",
                        f"bands.{name}",
                        {"high_hz": high, "nyquist": nyquist},
                    )
                )
        return tuple(issues)

    def run(self, frame: NeuroFrame) -> FeatureSet:
        issues = self.validate(frame)
        if issues:
            raise FrameValidationError(issues)
        freqs = np.fft.rfftfreq(frame.samples, d=1.0 / frame.timebase.sampling_rate)
        power = np.abs(np.fft.rfft(frame.data, axis=0)) ** 2 / max(frame.samples, 1)
        eeg_indices = frame.eeg_channel_indices() or tuple(range(frame.channel_count))
        values: dict[str, float] = {}
        for band_name, (low, high) in (self.bands or {}).items():
            mask = (freqs >= low) & (freqs <= high)
            band_power = power[mask][:, eeg_indices].mean() if mask.any() else 0.0
            values[f"{band_name}_power"] = float(band_power)
        return FeatureSet(
            name=self.name,
            values=values,
            metadata={
                "sampling_rate": frame.timebase.sampling_rate,
                "channels_used": [frame.channels[index].name for index in eeg_indices],
                "bands": self.bands,
            },
        )
