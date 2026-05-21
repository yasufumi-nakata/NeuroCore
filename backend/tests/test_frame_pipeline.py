from __future__ import annotations

import numpy as np
import pytest

from neurocore import Bandpass, Pipeline, PipelineExecutionError, ReReference, Resample, SpectralFeatures, ValidateEEG
from neurocore.frame import Channel, NeuroFrame, Timebase
from neurocore.loaders import load_csv
from neurocore.synthetic import synthetic_eeg_frame


def test_frame_validation_catches_duplicate_channels() -> None:
    frame = NeuroFrame(
        data=np.zeros((10, 2)),
        channels=(Channel("Cz"), Channel("Cz")),
        timebase=Timebase(250),
    )

    issues = frame.validate()

    assert any(issue.code == "duplicate_channels" for issue in issues)


def test_reference_pipeline_returns_features() -> None:
    frame = synthetic_eeg_frame(seconds=2, sampling_rate=250)
    pipeline = Pipeline([ValidateEEG(min_channels=2), Resample(200), Bandpass(1, 40), ReReference(), SpectralFeatures()])

    result = pipeline.run(frame)

    assert result.report.status == "passed"
    assert result.output.name == "spectral_features"
    assert "alpha_power" in result.output.values
    assert result.report.steps[-1].output["kind"] == "features"


def test_pipeline_rejects_invalid_filter_band() -> None:
    frame = synthetic_eeg_frame(seconds=2, sampling_rate=100)
    pipeline = Pipeline([Bandpass(1, 80)])

    with pytest.raises(PipelineExecutionError) as excinfo:
        pipeline.run(frame)

    assert excinfo.value.report is not None
    assert excinfo.value.report.status == "failed"
    assert any(issue.code == "band_exceeds_nyquist" for issue in excinfo.value.report.issues)


def test_csv_loader_round_trip(tmp_path) -> None:
    csv_path = tmp_path / "eeg.csv"
    csv_path.write_text("Fz,Cz\n1.0,2.0\n3.0,4.0\n", encoding="utf-8")

    frame = load_csv(csv_path, sampling_rate=250)

    assert frame.samples == 2
    assert frame.channel_names == ("Fz", "Cz")
    assert frame.timebase.sampling_rate == 250
