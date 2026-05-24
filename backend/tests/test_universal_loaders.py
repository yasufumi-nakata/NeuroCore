from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from neurocore.cli import main
from neurocore.loaders import find_supported_signal_files, load, load_mat, load_mne_raw, load_nwb, supported_extensions


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


def test_csv_loader_drops_non_numeric_label_columns(tmp_path) -> None:
    path = tmp_path / "features.csv"
    path.write_text("Fz,Cz,label\n1.0,2.0,NEGATIVE\n3.0,4.0,POSITIVE\n", encoding="utf-8")

    frame = load(path, sampling_rate=250)

    assert frame.data.shape == (2, 2)
    assert frame.channel_names == ("Fz", "Cz")
    assert frame.provenance["dropped_non_numeric_columns"] == 1


def test_csv_loader_drops_index_and_timestamp_metadata_columns(tmp_path) -> None:
    path = tmp_path / "raw.csv"
    path.write_text(
        ",unixTimestamp,CP3,C3\n"
        "0,1674926223312,-318.8,1.1\n"
        "1,1674926223316,-1881.0,4.7\n",
        encoding="utf-8",
    )

    frame = load(path, sampling_rate=250)

    assert frame.channel_names == ("CP3", "C3")
    assert frame.data.shape == (2, 2)
    assert frame.provenance["dropped_metadata_columns"] == 2


def test_txt_loader_uses_delimited_numeric_eeg_columns(tmp_path) -> None:
    path = tmp_path / "user001_10_1.txt"
    path.write_text("Time\tSample\tAF3\tF7\n0.0\t1\t10.5\t11.5\n0.1\t2\t12.5\t13.5\n", encoding="utf-8")

    frame = load(path, sampling_rate=128)

    assert frame.channel_names == ("AF3", "F7")
    assert frame.data.shape == (2, 2)
    assert frame.provenance["source"] == "csv"


def test_mat_loader_flattens_multidimensional_eeg_array(tmp_path, monkeypatch) -> None:
    path = tmp_path / "subject.mat"
    payload = {
        "eeg": np.zeros((12, 8, 1114, 15)),
        "list_sub": np.arange(10).reshape(1, 10),
        "fs": np.array([[256.0]]),
    }
    monkeypatch.setattr("neurocore.loaders._load_mat_payload", lambda _path: payload)

    frame = load_mat(path)

    assert frame.data.shape == (12 * 1114 * 15, 8)
    assert frame.timebase.sampling_rate == 256.0
    assert frame.channel_names == ("Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7", "Ch8")


def test_mat_loader_finds_numeric_matrix_inside_matlab_struct(tmp_path, monkeypatch) -> None:
    class FakeStruct:
        _fieldnames = ["train", "test"]

        def __init__(self):
            self.train = np.zeros((3, 10))
            self.test = np.ones((3, 12))

    path = tmp_path / "bnci.mat"
    payload = {"s1": np.array([[FakeStruct()]], dtype=object)}
    monkeypatch.setattr("neurocore.loaders._load_mat_payload", lambda _path: payload)

    frame = load_mat(path, sampling_rate=250)

    assert frame.data.shape == (12, 3)
    assert frame.timebase.sampling_rate == 250.0
    assert frame.data[0, 0] == 1.0


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
    (misc / "README.txt").write_text("metadata", encoding="utf-8")
    signal = eeg / "sub-01_task-test_eeg.csv"
    signal.write_text("Fz,Cz\n1,2\n3,4\n", encoding="utf-8")
    text_signal = eeg / "user001_10_1.txt"
    text_signal.write_text("Fz\tCz\n1\t2\n", encoding="utf-8")

    files = find_supported_signal_files(dataset)
    frame = load(dataset, sampling_rate=250)

    assert files[0] == signal
    assert text_signal in files
    assert misc / "README.txt" not in files
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


def test_mne_loader_falls_back_to_ant_cnt_reader(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    class FakeRaw:
        info = {"sfreq": 250.0}
        ch_names = ["Fz"]

        def copy(self):
            return self

        def pick(self, picks):
            assert picks == "data"
            return self

        def get_data(self):
            return np.array([[1e-6, 2e-6]])

        def get_channel_types(self):
            return ["eeg"]

    def read_raw_cnt(_path, *, preload, verbose):
        calls.append("read_raw_cnt")
        assert preload is False
        assert verbose == "ERROR"
        raise RuntimeError("WARNING: mne.io.read_raw_cnt supports Neuroscan CNT files only")

    def read_raw_ant(_path, *, preload, verbose):
        calls.append("read_raw_ant")
        assert preload is False
        assert verbose == "ERROR"
        return FakeRaw()

    fake_mne = SimpleNamespace(io=SimpleNamespace(read_raw_cnt=read_raw_cnt, read_raw_ant=read_raw_ant))
    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    path = tmp_path / "sample.cnt"
    path.write_text("", encoding="utf-8")

    frame = load_mne_raw(path, preload=False)

    assert calls == ["read_raw_cnt", "read_raw_ant"]
    assert frame.data.shape == (2, 1)
    assert frame.provenance["reader"] == "mne.io.read_raw_ant"


def test_nwb_loader_reads_electrical_series_through_pynwb_adapter(tmp_path, monkeypatch) -> None:
    class FakeData:
        shape = (3, 2)

        def __getitem__(self, item):
            assert item == slice(None, None, None)
            return np.array([[1e-6, 2e-6], [3e-6, 4e-6], [5e-6, 6e-6]])

    class FakeElectrodes:
        def to_dataframe(self):
            return {"label": SimpleNamespace(tolist=lambda: ["Fz", "Cz"])}

    class FakeSeries:
        name = "raw_voltage"
        data = FakeData()
        rate = 500.0
        electrodes = FakeElectrodes()
        unit = "volts"

    class FakeNWBFile:
        session_description = "fixture"
        identifier = "nwb-fixture"
        acquisition = {"raw_voltage": FakeSeries()}
        processing = {}

    class FakeNWBHDF5IO:
        def __init__(self, path, mode):
            assert mode == "r"
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return FakeNWBFile()

    fake_pynwb = ModuleType("pynwb")
    fake_pynwb.NWBHDF5IO = FakeNWBHDF5IO
    monkeypatch.setitem(sys.modules, "pynwb", fake_pynwb)
    path = tmp_path / "sample.nwb"
    path.write_text("", encoding="utf-8")

    frame = load_nwb(path)

    assert frame.data.shape == (3, 2)
    assert frame.data[0, 0] == 1.0
    assert frame.channel_names == ("Fz", "Cz")
    assert frame.timebase.sampling_rate == 500.0
    assert frame.provenance["series_name"] == "raw_voltage"


def test_supported_extensions_include_major_eeg_formats() -> None:
    extensions = set(supported_extensions())

    assert {
        ".edf",
        ".bdf",
        ".vhdr",
        ".set",
        ".fif",
        ".cnt",
        ".gdf",
        ".egi",
        ".mff",
        ".nxe",
        ".data",
        ".lay",
        ".mefd",
        ".xdf",
        ".nwb",
        ".mat",
        ".npy",
        ".npz",
    }.issubset(extensions)
