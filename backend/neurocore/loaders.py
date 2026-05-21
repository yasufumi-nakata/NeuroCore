from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .frame import Channel, NeuroFrame, Timebase


def load(path: str | Path, *, sampling_rate: float | None = None) -> NeuroFrame:
    resolved = Path(path).expanduser()
    if resolved.suffix.lower() == ".csv":
        if sampling_rate is None:
            raise ValueError("sampling_rate is required when loading CSV EEG data")
        return load_csv(resolved, sampling_rate=sampling_rate)
    raise ValueError(f"unsupported input format: {resolved.suffix}")


def load_csv(
    path: str | Path,
    *,
    sampling_rate: float,
    channel_type: str = "eeg",
    unit: str = "uV",
) -> NeuroFrame:
    resolved = Path(path).expanduser()
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV file is empty") from exc
        names = [item.strip() for item in header if item.strip()]
        rows: list[list[float]] = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(names):
                raise ValueError(f"CSV row {line_number} has {len(row)} values, expected {len(names)}")
            try:
                rows.append([float(item) for item in row])
            except ValueError as exc:
                raise ValueError(f"CSV row {line_number} contains a non-numeric value") from exc
    if not rows:
        raise ValueError("CSV file contains no samples")
    return NeuroFrame(
        data=np.asarray(rows, dtype=float),
        channels=tuple(Channel(name=name, type=channel_type, unit=unit) for name in names),
        timebase=Timebase(sampling_rate=sampling_rate),
        provenance={"source": "csv", "path": str(resolved)},
    )
