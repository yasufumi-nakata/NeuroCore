from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Any

import numpy as np

from .frame import Channel, NeuroFrame, Timebase


MNE_RAW_READERS = {
    ".edf": "read_raw_edf",
    ".bdf": "read_raw_bdf",
    ".vhdr": "read_raw_brainvision",
    ".fif": "read_raw_fif",
    ".set": "read_raw_eeglab",
    ".cnt": "read_raw_cnt",
    ".gdf": "read_raw_gdf",
    ".egi": "read_raw_egi",
    ".mff": "read_raw_egi",
    ".nxe": "read_raw_eximia",
    ".data": "read_raw_nicolet",
    ".lay": "read_raw_persyst",
    ".mefd": "read_raw_mef",
}

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".npy",
    ".npz",
    ".mat",
    ".xdf",
    *MNE_RAW_READERS,
}


def load(
    path: str | Path,
    *,
    sampling_rate: float | None = None,
    channel_names: tuple[str, ...] | list[str] | None = None,
    mne_preload: bool = True,
) -> NeuroFrame:
    resolved = Path(path).expanduser()
    if resolved.is_dir():
        return load_directory(resolved, sampling_rate=sampling_rate, mne_preload=mne_preload)
    suffix = _loader_suffix(resolved)
    if suffix == ".csv":
        if sampling_rate is None:
            raise ValueError("sampling_rate is required when loading CSV EEG data")
        return load_csv(resolved, sampling_rate=sampling_rate, channel_names=channel_names)
    if suffix in MNE_RAW_READERS:
        return load_mne_raw(resolved, preload=mne_preload)
    if suffix == ".xdf":
        return load_xdf(resolved)
    if suffix == ".npy":
        if sampling_rate is None:
            raise ValueError("sampling_rate is required when loading NPY EEG data")
        return load_numpy(resolved, sampling_rate=sampling_rate, channel_names=channel_names)
    if suffix == ".npz":
        return load_numpy(resolved, sampling_rate=sampling_rate, channel_names=channel_names)
    if suffix == ".mat":
        return load_mat(resolved, sampling_rate=sampling_rate, channel_names=channel_names)
    raise ValueError(f"unsupported input format: {resolved.suffix}")


def load_directory(
    path: str | Path,
    *,
    sampling_rate: float | None = None,
    mne_preload: bool = True,
) -> NeuroFrame:
    resolved = Path(path).expanduser()
    files = find_supported_signal_files(resolved)
    if not files:
        raise ValueError(f"directory does not contain a supported EEG signal file: {resolved}")
    return load(files[0], sampling_rate=sampling_rate, mne_preload=mne_preload)


def load_csv(
    path: str | Path,
    *,
    sampling_rate: float,
    channel_names: tuple[str, ...] | list[str] | None = None,
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
        if channel_names is not None:
            names = [str(item).strip() for item in channel_names]
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


def load_mne_raw(path: str | Path, *, preload: bool = True, unit: str = "uV") -> NeuroFrame:
    resolved = Path(path).expanduser()
    mne = _require_module("mne", extra="io")
    suffix = _loader_suffix(resolved)
    reader_name = MNE_RAW_READERS.get(suffix)
    if reader_name is None:
        raise ValueError(f"MNE raw loader does not support format: {resolved.suffix}")
    reader = getattr(mne.io, reader_name)
    raw = reader(str(resolved), preload=preload, verbose="ERROR")
    data, names, types = _mne_raw_to_samples(raw)
    scale = 1_000_000.0 if unit == "uV" else 1.0
    return NeuroFrame(
        data=np.asarray(data, dtype=float) * scale,
        channels=tuple(Channel(name=name, type=_channel_type(kind), unit=unit) for name, kind in zip(names, types)),
        timebase=Timebase(sampling_rate=float(raw.info["sfreq"])),
        provenance={
            "source": "mne",
            "format": suffix.lstrip("."),
            "reader": f"mne.io.{reader_name}",
            "path": str(resolved),
            "native_unit": "V",
        },
    )


def load_xdf(path: str | Path, *, stream_name: str | None = None, unit: str = "uV") -> NeuroFrame:
    resolved = Path(path).expanduser()
    pyxdf = _require_module("pyxdf", extra="xdf")
    streams, _header = pyxdf.load_xdf(str(resolved))
    stream = _select_xdf_stream(streams, stream_name=stream_name)
    data = np.asarray(stream.get("time_series"), dtype=float)
    if data.ndim != 2:
        raise ValueError("selected XDF stream is not shaped as samples x channels")
    timestamps = tuple(float(item) for item in stream.get("time_stamps", ()))
    sampling_rate = _xdf_sampling_rate(stream, timestamps)
    names = _xdf_channel_names(stream, data.shape[1])
    return NeuroFrame(
        data=data,
        channels=tuple(Channel(name=name, type="eeg", unit=unit) for name in names),
        timebase=Timebase(sampling_rate=sampling_rate, timestamps=timestamps or None, clock_source="xdf"),
        provenance={
            "source": "xdf",
            "path": str(resolved),
            "stream_name": _xdf_info_value(stream, "name"),
            "stream_type": _xdf_info_value(stream, "type"),
        },
    )


def load_numpy(
    path: str | Path,
    *,
    sampling_rate: float | None = None,
    channel_names: tuple[str, ...] | list[str] | None = None,
    channel_type: str = "eeg",
    unit: str = "uV",
) -> NeuroFrame:
    resolved = Path(path).expanduser()
    if resolved.suffix.lower() == ".npz":
        with np.load(resolved, allow_pickle=False) as archive:
            data = _npz_array(archive)
            sampling_rate = sampling_rate or _npz_sampling_rate(archive)
            names = channel_names or _npz_channel_names(archive)
    else:
        data = np.load(resolved, allow_pickle=False)
        names = channel_names
    if sampling_rate is None:
        raise ValueError("sampling_rate is required when the NumPy archive does not include it")
    data = _samples_by_channels(data)
    channel_tuple = _channels_for_data(data, names, channel_type=channel_type, unit=unit)
    return NeuroFrame(
        data=data,
        channels=channel_tuple,
        timebase=Timebase(sampling_rate=float(sampling_rate)),
        provenance={"source": "numpy", "path": str(resolved), "format": resolved.suffix.lower().lstrip(".")},
    )


def load_mat(
    path: str | Path,
    *,
    sampling_rate: float | None = None,
    channel_names: tuple[str, ...] | list[str] | None = None,
    channel_type: str = "eeg",
    unit: str = "uV",
) -> NeuroFrame:
    resolved = Path(path).expanduser()
    payload = _load_mat_payload(resolved)
    data = _find_numeric_matrix(payload)
    sampling_rate = sampling_rate or _find_sampling_rate(payload)
    if sampling_rate is None:
        raise ValueError("sampling_rate is required when the MAT file does not expose fs/sfreq/sampling_rate")
    names = channel_names or _find_channel_names(payload)
    data = _samples_by_channels(data)
    return NeuroFrame(
        data=data,
        channels=_channels_for_data(data, names, channel_type=channel_type, unit=unit),
        timebase=Timebase(sampling_rate=float(sampling_rate)),
        provenance={"source": "mat", "path": str(resolved), "format": "mat"},
    )


def can_load_extension(path: str | Path) -> bool:
    return _loader_suffix(Path(path)) in SUPPORTED_EXTENSIONS


def supported_extensions() -> tuple[str, ...]:
    return tuple(sorted(SUPPORTED_EXTENSIONS))


def find_supported_signal_files(path: str | Path) -> tuple[Path, ...]:
    resolved = Path(path).expanduser()
    if resolved.is_file():
        return (resolved,) if can_load_extension(resolved) else ()
    if not resolved.is_dir():
        return ()
    files = [item for item in resolved.rglob("*") if item.is_file() and can_load_extension(item)]
    return tuple(sorted(files, key=_signal_file_sort_key))


def _loader_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".fif.gz"):
        return ".fif"
    return path.suffix.lower()


def _signal_file_sort_key(path: Path) -> tuple[int, int, str]:
    parts = {part.lower() for part in path.parts}
    suffix = _loader_suffix(path)
    eeg_dir_rank = 0 if "eeg" in parts else 1
    preferred = [".vhdr", ".edf", ".bdf", ".set", ".fif", ".gdf", ".cnt", ".egi", ".mff", ".xdf", ".mat", ".npz", ".npy", ".csv"]
    try:
        suffix_rank = preferred.index(suffix)
    except ValueError:
        suffix_rank = len(preferred)
    return eeg_dir_rank, suffix_rank, str(path)


def _require_module(name: str, *, extra: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(
            f"loading this EEG format requires optional dependency {name!r}; install NeuroCore with the {extra!r} extra"
        ) from exc


def _mne_raw_to_samples(raw: Any) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    picked = raw.copy().pick("data") if hasattr(raw, "copy") else raw
    data = np.asarray(picked.get_data(), dtype=float).T
    names = tuple(str(name) for name in picked.ch_names)
    if hasattr(picked, "get_channel_types"):
        types = tuple(str(kind) for kind in picked.get_channel_types())
    else:
        types = tuple("eeg" for _ in names)
    return data, names, types


def _channel_type(kind: str) -> str:
    normalized = kind.lower()
    if normalized in {"eeg", "ecog", "seeg", "dbs"}:
        return normalized
    return "aux"


def _select_xdf_stream(streams: list[dict[str, Any]], *, stream_name: str | None) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for stream in streams:
        if stream_name and _xdf_info_value(stream, "name") != stream_name:
            continue
        data = np.asarray(stream.get("time_series"))
        if data.ndim == 2 and data.size:
            candidates.append(stream)
    if not candidates:
        raise ValueError("XDF file does not contain a numeric samples x channels stream")
    candidates.sort(
        key=lambda item: (_xdf_info_value(item, "type").lower() == "eeg", np.asarray(item["time_series"]).shape[1]),
        reverse=True,
    )
    return candidates[0]


def _xdf_info_value(stream: dict[str, Any], key: str) -> str:
    value = stream.get("info", {}).get(key, [""])
    if isinstance(value, list) and value:
        return str(value[0])
    return str(value or "")


def _xdf_sampling_rate(stream: dict[str, Any], timestamps: tuple[float, ...]) -> float:
    nominal = _xdf_info_value(stream, "nominal_srate")
    try:
        sampling_rate = float(nominal)
    except ValueError:
        sampling_rate = 0.0
    if sampling_rate > 0:
        return sampling_rate
    if len(timestamps) > 1:
        duration = timestamps[-1] - timestamps[0]
        if duration > 0:
            return (len(timestamps) - 1) / duration
    raise ValueError("XDF stream does not expose a usable sampling rate")


def _xdf_channel_names(stream: dict[str, Any], count: int) -> tuple[str, ...]:
    desc = stream.get("info", {}).get("desc", [])
    try:
        channels = desc[0]["channels"][0]["channel"]
        labels = tuple(str(channel["label"][0]) for channel in channels if channel.get("label"))
    except (KeyError, IndexError, TypeError):
        labels = ()
    if len(labels) == count:
        return labels
    return tuple(f"Ch{index + 1}" for index in range(count))


def _samples_by_channels(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("EEG data must be a 2D array")
    if array.shape[0] < array.shape[1]:
        array = array.T
    if array.size == 0:
        raise ValueError("EEG data contains no samples")
    return array


def _channels_for_data(
    data: np.ndarray,
    names: tuple[str, ...] | list[str] | None,
    *,
    channel_type: str,
    unit: str,
) -> tuple[Channel, ...]:
    if names is None:
        names = [f"Ch{index + 1}" for index in range(data.shape[1])]
    if len(names) != data.shape[1]:
        raise ValueError(f"channel_names has {len(names)} values, expected {data.shape[1]}")
    return tuple(Channel(name=str(name), type=channel_type, unit=unit) for name in names)


def _npz_array(archive: Any) -> np.ndarray:
    for key in ("data", "eeg", "signal", "signals", "x", "X"):
        if key in archive:
            return np.asarray(archive[key], dtype=float)
    arrays = [np.asarray(archive[key]) for key in archive.files if np.asarray(archive[key]).ndim == 2]
    if not arrays:
        raise ValueError("NPZ archive does not contain a 2D EEG array")
    return np.asarray(max(arrays, key=lambda item: item.size), dtype=float)


def _npz_sampling_rate(archive: Any) -> float | None:
    for key in ("sampling_rate", "sfreq", "fs", "srate"):
        if key in archive:
            return float(np.asarray(archive[key]).reshape(-1)[0])
    return None


def _npz_channel_names(archive: Any) -> tuple[str, ...] | None:
    for key in ("channel_names", "channels", "ch_names"):
        if key in archive:
            return tuple(str(item) for item in np.asarray(archive[key]).reshape(-1))
    return None


def _load_mat_payload(path: Path) -> dict[str, Any]:
    try:
        scipy_io = importlib.import_module("scipy.io")
        return {key: value for key, value in scipy_io.loadmat(path).items() if not key.startswith("__")}
    except ImportError:
        h5py = _require_module("h5py", extra="io")
        with h5py.File(path, "r") as handle:
            return {key: np.asarray(value) for key, value in handle.items()}
    except NotImplementedError:
        h5py = _require_module("h5py", extra="io")
        with h5py.File(path, "r") as handle:
            return {key: np.asarray(value) for key, value in handle.items()}


def _find_numeric_matrix(payload: dict[str, Any]) -> np.ndarray:
    preferred = ("data", "eeg", "EEG", "signal", "signals", "X", "x")
    for key in preferred:
        if (
            key in payload
            and np.asarray(payload[key]).ndim == 2
            and np.issubdtype(np.asarray(payload[key]).dtype, np.number)
        ):
            return np.asarray(payload[key], dtype=float)
    matrices = [
        np.asarray(value, dtype=float)
        for value in payload.values()
        if np.asarray(value).ndim == 2 and np.issubdtype(np.asarray(value).dtype, np.number)
    ]
    if not matrices:
        raise ValueError("MAT file does not contain a 2D numeric EEG matrix")
    return max(matrices, key=lambda item: item.size)


def _find_sampling_rate(payload: dict[str, Any]) -> float | None:
    for key in ("sampling_rate", "sfreq", "fs", "srate"):
        if key in payload:
            return float(np.asarray(payload[key]).reshape(-1)[0])
    return None


def _find_channel_names(payload: dict[str, Any]) -> tuple[str, ...] | None:
    for key in ("channel_names", "channels", "ch_names"):
        if key in payload:
            return tuple(str(item) for item in np.asarray(payload[key]).reshape(-1))
    return None
