#!/usr/bin/env python3
"""Reconstruct the published AI-coding exporter FIR support on a zero fixture.

This runs our reconstruction, not executable code from a downloaded notebook.
It establishes a conditional filter-support bound, not a claim about unseen XDF.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import scipy


def audit():
    original = mne.filter.create_filter
    filters = []

    def capture(*args, **kwargs):
        coefficients = original(*args, **kwargs)
        if isinstance(coefficients, np.ndarray):
            filters.append({"taps": len(coefficients),
                            "coefficient_sha256": hashlib.sha256(coefficients.astype("<f8").tobytes()).hexdigest(),
                            "phase": kwargs.get("phase", "zero")})
        return coefficients

    raw = mne.io.RawArray(np.zeros((1, 8192)), mne.create_info(["C3"], 256, "eeg"), verbose="ERROR")
    with patch("mne.filter.create_filter", side_effect=capture):
        raw.filter(3, 40, verbose="ERROR").notch_filter([50, 100], verbose="ERROR")
    return {
        "method": "Reconstruct finite impulse support of the published export recipe",
        "mne": mne.__version__, "scipy": scipy.__version__, "numpy": np.__version__,
        "release_requirements_mne": "1.10.2", "mne_version_matches_release": mne.__version__ == "1.10.2",
        "sample_rate_hz": 256, "filters": filters,
        "combined_fir_taps": 1 + sum(f["taps"] - 1 for f in filters),
        "future_support_seconds": sum((f["taps"] - 1) / 2 / 256 for f in filters),
        "source": "https://zenodo.org/records/20801101/files/scripts.tar.gz#scripts/01-b_load.ipynb",
        "source_requirements": "https://zenodo.org/records/20801101/files/requirements.txt",
        "interpretation": "Conditional on release binaries being generated exactly by the published export. This reconstructs filter support, not original XDF or online data availability. A 0.25-second pre-action gap is insufficient under this recipe.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="filter_support_evidence.json")
    args = p.parse_args()
    result = audit()
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
