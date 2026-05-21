from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .frame import Channel, NeuroFrame, Timebase


@dataclass(frozen=True)
class StreamWindow:
    frame: NeuroFrame
    start_sample: int
    end_sample: int

    def to_summary(self) -> dict[str, object]:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "frame": self.frame.to_summary(),
        }


@dataclass
class StreamBuffer:
    channels: tuple[Channel, ...]
    sampling_rate: float
    window_seconds: float
    step_seconds: float
    max_seconds: float = 30.0
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        self.channels = tuple(self.channels)
        self._data = np.empty((0, len(self.channels)), dtype=float)
        self._absolute_start = 0
        self._next_window_start = 0

    @property
    def sample_count(self) -> int:
        return int(self._data.shape[0])

    def append(self, samples: Iterable[Iterable[float]]) -> tuple[StreamWindow, ...]:
        incoming = np.asarray(list(samples), dtype=float)
        if incoming.ndim == 1:
            incoming = incoming.reshape(1, -1)
        if incoming.ndim != 2 or incoming.shape[1] != len(self.channels):
            raise ValueError("samples must be shaped as samples x channels")
        self._data = np.vstack([self._data, incoming])
        windows = self._emit_ready_windows()
        self._trim()
        return windows

    def _emit_ready_windows(self) -> tuple[StreamWindow, ...]:
        window_size = max(1, int(round(self.window_seconds * self.sampling_rate)))
        step_size = max(1, int(round(self.step_seconds * self.sampling_rate)))
        windows: list[StreamWindow] = []
        while self._next_window_start + window_size <= self._absolute_start + self.sample_count:
            local_start = self._next_window_start - self._absolute_start
            local_end = local_start + window_size
            frame = NeuroFrame(
                data=self._data[local_start:local_end].copy(),
                channels=self.channels,
                timebase=Timebase(
                    sampling_rate=self.sampling_rate,
                    start_time=self._next_window_start / self.sampling_rate,
                ),
                provenance={**self.provenance, "source": "stream_window"},
            )
            windows.append(StreamWindow(frame, self._next_window_start, self._next_window_start + window_size))
            self._next_window_start += step_size
        return tuple(windows)

    def _trim(self) -> None:
        max_samples = max(1, int(round(self.max_seconds * self.sampling_rate)))
        keep_from_next = max(0, self._next_window_start - self._absolute_start)
        keep_from = min(keep_from_next, max(0, self.sample_count - max_samples))
        if keep_from <= 0:
            return
        self._data = self._data[keep_from:].copy()
        self._absolute_start += keep_from
