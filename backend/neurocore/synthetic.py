from __future__ import annotations

import numpy as np

from .frame import Channel, NeuroFrame, Timebase


def synthetic_eeg_frame(
    *,
    seconds: float = 4.0,
    sampling_rate: float = 250.0,
    channels: tuple[str, ...] = ("Fz", "Cz", "Pz", "Oz"),
    seed: int = 7,
) -> NeuroFrame:
    rng = np.random.default_rng(seed)
    sample_count = int(seconds * sampling_rate)
    t = np.arange(sample_count, dtype=float) / sampling_rate
    signals = []
    for index, _name in enumerate(channels):
        alpha = 10.0 + index * 0.3
        beta = 20.0 + index * 0.7
        signal = 14.0 * np.sin(2 * np.pi * alpha * t) + 4.0 * np.sin(2 * np.pi * beta * t)
        signal += rng.normal(0.0, 1.5, size=sample_count)
        signals.append(signal)
    data = np.stack(signals, axis=1)
    return NeuroFrame(
        data=data,
        channels=tuple(Channel(name=name, type="eeg", unit="uV", reference="common") for name in channels),
        timebase=Timebase(sampling_rate=sampling_rate),
        provenance={"source": "synthetic", "seed": seed},
    )
