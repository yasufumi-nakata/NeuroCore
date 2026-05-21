from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frame import NeuroFrame


@dataclass(frozen=True)
class SignalQualityReport:
    status: str
    flat_channels: tuple[str, ...]
    high_amplitude_channels: tuple[str, ...]
    rms_by_channel: dict[str, float]
    peak_to_peak_by_channel: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "flat_channels": list(self.flat_channels),
            "high_amplitude_channels": list(self.high_amplitude_channels),
            "rms_by_channel": self.rms_by_channel,
            "peak_to_peak_by_channel": self.peak_to_peak_by_channel,
        }


def score_signal_quality(
    frame: NeuroFrame,
    *,
    flat_peak_to_peak_uv: float = 1.0,
    high_peak_to_peak_uv: float = 500.0,
) -> SignalQualityReport:
    peak_to_peak = np.ptp(frame.data, axis=0)
    rms = np.sqrt(np.mean(np.square(frame.data), axis=0))
    flat_channels = []
    high_amplitude_channels = []
    for index, channel in enumerate(frame.channels):
        if peak_to_peak[index] <= flat_peak_to_peak_uv:
            flat_channels.append(channel.name)
        if peak_to_peak[index] >= high_peak_to_peak_uv:
            high_amplitude_channels.append(channel.name)
    status = "passed" if not flat_channels and not high_amplitude_channels else "warning"
    return SignalQualityReport(
        status=status,
        flat_channels=tuple(flat_channels),
        high_amplitude_channels=tuple(high_amplitude_channels),
        rms_by_channel={channel.name: float(rms[index]) for index, channel in enumerate(frame.channels)},
        peak_to_peak_by_channel={channel.name: float(peak_to_peak[index]) for index, channel in enumerate(frame.channels)},
    )
