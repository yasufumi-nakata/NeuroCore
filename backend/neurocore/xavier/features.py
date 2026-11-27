"""Causal source30 inference features. No target or future sample is accepted."""
from __future__ import annotations

import numpy as np

FEATURE_CONTRACT = "ds006735_causal30_features5_future20mean_v1"
CHANNELS = 30
PREFIX_SAMPLES = 300
SAMPLE_RATE_HZ = 200
INPUT_SCALE_UV = np.float32(20.0)
CONTEXT_BINS = ((0, 75), (75, 150), (150, 225), (225, 300))


def encode_prefix(prefix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return five EEG positions and the last observed sample, in training order.

    The input is the already causally filtered/downsampled 200 Hz stream in µV.
    This function deliberately does not estimate a target, resample, pad channels,
    remove DC, or silently change a sensor's physical units.
    """
    if not isinstance(prefix, np.ndarray) or prefix.dtype != np.float32:
        raise ValueError("prefix must be a float32 NumPy array")
    if prefix.shape != (PREFIX_SAMPLES, CHANNELS) or not np.isfinite(prefix).all():
        raise ValueError("a finite 300 by 30 causal prefix is required")
    means64 = [np.mean(prefix[a:b], axis=0, dtype=np.float64) for a, b in CONTEXT_BINS]
    means = [value.astype(np.float32) for value in means64]
    innovation = (prefix[299].astype(np.float64) - means64[-1]).astype(np.float32)
    features = np.stack(means + [innovation]) / INPUT_SCALE_UV
    return features, prefix[299].copy()


def validate_sensor(channels: list[str] | tuple[str, ...], expected: list[str],
                    sample_rate_hz: int | float, units: str) -> None:
    if (not isinstance(channels, (list, tuple)) or list(channels) != expected
            or type(sample_rate_hz) not in (int, float) or sample_rate_hz != SAMPLE_RATE_HZ
            or not isinstance(units, str) or units not in {"uV", "µV"}):
        raise ValueError("sensor channel order, 200 Hz rate, and microvolt units must match the model")
