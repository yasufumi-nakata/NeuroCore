from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np

from neurocore.cli import main
from neurocore.loaders import find_supported_signal_files, load, load_mne_raw, supported_extensions


def test_numpy_npz_loader_uses_embedded_metadata(tmp_path) -> None:
    path = tmp_path / "eeg.npz"
    np.savez(
        path,
        data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        sampling_rate=np.array([250.0]),
        channel_names=np.array(["Fz", "Cz"]),
    )

    frame = load(path)

    assert frame.samples == 2
    assert frame.channel_names == ("Fz", "Cz")
    assert frame.timebase.sampling_rate == 250.0
    assert frame.provenance["source"] == "numpy"


def test_run_file_cli_accepts_existing_csv_fixture(capsys) -> None:
    exit_code = main(["run-file", "samples/synthetic_eeg.csv", "--sampling-rate", "250", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["status"] == "passed"
    assert payload["output"]["kind"] == "features"


def test_directory_loader_prefers_signal_files_under_eeg_folder(tmp_path) -> None:
    dataset = tmp_path / "bids_like"
    misc = dataset / "misc"
    eeg = dataset / "sub-01" / "eeg"
    misc.mkdir(parents=True)
    eeg.mkdir(parents=True)
    (misc / "table.csv").write_text("a\n1\n", encoding="utf-8")
    signal = eeg / "sub-01_task-test_eeg.csv"
    signal.write_text("Fz,Cz\n1,2\n3,4\n", encoding="utf-8")

    files = find_supported_signal_files(dataset)
    frame = load(dataset, sampling_rate=250)

    assert files[0] == signal
    assert frame.channel_names == ("Fz", "Cz")


def test_mne_loader_dispatches_eeglab_and_edf_through_optional_adapter(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, str, bool]] = []

    class FakeRaw:
        info = {"sfreq": 500.0}
        ch_names = ["Fz", "Cz"]

        def copy(self):
            return self

        def pick(self, picks):
            assert picks == "data"
            return self

        def get_data(self):
            return np.array([[1e-6, 2e-6, 3e-6], [4e-6, 5e-6, 6e-6]])

        def get_channel_types(self):
            return ["eeg", "eeg"]

    def reader(name):
        def _read(path, *, preload, verbose):
            calls.append((name, path, preload))
            assert verbose == "ERROR"
            return FakeRaw()

        return _read

    fake_mne = SimpleNamespace(
        io=SimpleNamespace(
            read_raw_edf=reader("read_raw_edf"),
            read_raw_eeglab=reader("read_raw_eeglab"),
        )
    )
    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    edf = tmp_path / "sample.edf"
    eeglab = tmp_path / "sample.set"
    edf.write_text("", encoding="utf-8")
    eeglab.write_text("", encoding="utf-8")

    edf_frame = load_mne_raw(edf)
    eeglab_frame = load(eeglab)

    assert edf_frame.data.shape == (3, 2)
    assert edf_frame.data[0, 0] == 1.0
    assert eeglab_frame.channel_names == ("Fz", "Cz")
    assert calls[0][0] == "read_raw_edf"
    assert calls[1][0] == "read_raw_eeglab"


def test_supported_extensions_include_major_eeg_formats() -> None:
    extensions = set(supported_extensions())

    assert {".edf", ".bdf", ".vhdr", ".set", ".fif", ".xdf", ".mat", ".npy", ".npz"}.issubset(extensions)
