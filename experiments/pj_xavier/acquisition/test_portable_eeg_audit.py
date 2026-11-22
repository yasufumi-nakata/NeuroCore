import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("portable_audit", Path(__file__).with_name("audit_portable_eeg_zip.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def bv():
    return b"Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\nDataFile=a.eeg\nDataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=2\nSamplingInterval=4000\n[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\nCh1=C3,,1,uV\nCh2=C4,,1,uV\n"


def edf(records="3"):
    def field(value, width):
        return str(value).ljust(width).encode("ascii")
    fixed = (field(0, 8) + b" " * 160 + b" " * 16 + field(768, 8) + field("EDF+C", 44)
             + field(records, 8) + field(2, 8) + field(2, 4))
    signal = (field("C3", 16) + field("EDF Annotations", 16) + b" " * (2 * 200)
              + field(128, 8) + field(16, 8) + b" " * 64)
    return fixed + signal


def test_bv_uses_sample_frames_not_channel_seconds():
    result = mod.brainvision_metadata(bv(), 8000)
    assert result["samples_per_channel"] == 1000
    assert result["duration_seconds"] == 4


def test_bv_truncated_binary_rejected():
    with pytest.raises(ValueError, match="whole sample"):
        mod.brainvision_metadata(bv(), 8001)


def test_edf_per_signal_rates_and_annotations():
    result = mod.edf_metadata(edf(), 768 + 3 * (128 + 16) * 2)
    assert result["duration_seconds"] == 6
    assert result["sample_rates_hz"] == [64, 8]
    assert result["sample_counts"] == [384, 48]


def test_edf_unknown_record_count_from_exact_body():
    assert mod.edf_metadata(edf("-1"), 768 + 3 * 288)["data_records"] == 3


def test_edf_declared_count_conflict_rejected():
    with pytest.raises(ValueError, match="declared record count"):
        mod.edf_metadata(edf("4"), 768 + 3 * 288)
