from __future__ import annotations

import pickle

import numpy as np
import pytest

from neurocore import loaders


@pytest.mark.parametrize("entry_point", ["load", "load_pickle", "load_directory"])
def test_pickle_default_rejection_happens_before_deserialization(tmp_path, monkeypatch, entry_point):
    path = tmp_path / "untrusted.pkl"
    path.write_bytes(b"this input must never reach an unpickler")

    def forbidden(_path):
        pytest.fail("untrusted pickle was deserialized")

    monkeypatch.setattr(loaders, "_load_pickle_payload", forbidden)
    target = tmp_path if entry_point == "load_directory" else path
    with pytest.raises(ValueError, match="pickle loading is disabled by default"):
        getattr(loaders, entry_point)(target, sampling_rate=250)


@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_pickle_trust_requires_literal_true(tmp_path, monkeypatch, value):
    path = tmp_path / "input.pkl"
    path.write_bytes(b"not a pickle")

    def forbidden(_path):
        pytest.fail("non-boolean trust bypassed the boundary")

    monkeypatch.setattr(loaders, "_load_pickle_payload", forbidden)
    with pytest.raises(ValueError, match="pickle loading is disabled by default"):
        loaders.load_pickle(path, trusted=value, sampling_rate=250)
    with pytest.raises(ValueError, match="pickle loading is disabled by default"):
        loaders.load(path, trusted_pickle=value, sampling_rate=250)


@pytest.mark.parametrize("entry_point", ["load", "load_pickle", "load_directory"])
def test_verified_numeric_fixture_remains_loadable_with_explicit_trust(tmp_path, entry_point):
    path = tmp_path / "trusted.pkl"
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    path.write_bytes(pickle.dumps({"data": expected, "sampling_rate": 250.0}))
    target = tmp_path if entry_point == "load_directory" else path
    keyword = "trusted" if entry_point == "load_pickle" else "trusted_pickle"
    frame = getattr(loaders, entry_point)(target, **{keyword: True})
    np.testing.assert_array_equal(frame.data, expected)
    assert frame.timebase.sampling_rate == 250.0
