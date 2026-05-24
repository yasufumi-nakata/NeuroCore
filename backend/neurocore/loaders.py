from __future__ import annotations

import csv
import importlib
import re
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
    ".tab",
    ".ts",
    ".txt",
    ".tsv",
    ".nwb",
    ".npy",
    ".npz",
    ".mat",
    ".pt",
    ".pth",
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
    if suffix in {".csv", ".tab", ".txt", ".tsv"}:
        if sampling_rate is None:
            raise ValueError("sampling_rate is required when loading delimited EEG data")
        return load_csv(resolved, sampling_rate=sampling_rate, channel_names=channel_names)
    if suffix == ".ts":
        if sampling_rate is None:
            raise ValueError("sampling_rate is required when loading Time Series Classification EEG data")
        return load_ts(resolved, sampling_rate=sampling_rate, channel_names=channel_names)
    if suffix in MNE_RAW_READERS:
        return load_mne_raw(resolved, preload=mne_preload)
    if suffix == ".xdf":
        return load_xdf(resolved)
    if suffix == ".nwb":
        return load_nwb(resolved)
    if suffix == ".npy":
        if sampling_rate is None:
            raise ValueError("sampling_rate is required when loading NPY EEG data")
        return load_numpy(resolved, sampling_rate=sampling_rate, channel_names=channel_names)
    if suffix == ".npz":
        return load_numpy(resolved, sampling_rate=sampling_rate, channel_names=channel_names)
    if suffix == ".mat":
        return load_mat(resolved, sampling_rate=sampling_rate, channel_names=channel_names)
    if suffix in {".pt", ".pth"}:
        return load_torch(resolved, sampling_rate=sampling_rate, channel_names=channel_names)
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
    source_rows = _read_delimited_rows(resolved)
    if not source_rows:
        raise ValueError("CSV file is empty")
    headerless = _row_is_numeric(source_rows[0])
    header = [f"Ch{index + 1}" for index in range(len(source_rows[0]))] if headerless else source_rows[0]
    raw_rows = source_rows if headerless else source_rows[1:]
    if not raw_rows:
        raise ValueError("CSV file contains no samples")
    for line_number, row in enumerate(raw_rows, start=1 if headerless else 2):
        if len(row) != len(header):
            raise ValueError(f"CSV row {line_number} has {len(row)} values, expected {len(header)}")
    if channel_names is not None:
        names = [str(item).strip() for item in channel_names]
        if len(names) != len(header):
            raise ValueError(f"channel_names has {len(names)} values, expected {len(header)}")
        numeric_indices = tuple(range(len(header)))
    else:
        names, numeric_indices, dropped_non_numeric_columns, dropped_metadata_columns = _csv_numeric_columns(
            header, raw_rows, drop_metadata=not headerless
        )
    rows: list[list[float]] = []
    for line_number, row in enumerate(raw_rows, start=1 if headerless else 2):
        try:
            rows.append([float(row[index]) for index in numeric_indices])
        except ValueError as exc:
            raise ValueError(f"CSV row {line_number} contains a non-numeric value in selected columns") from exc
    if not rows:
        raise ValueError("CSV file contains no samples")
    if channel_names is not None:
        dropped_non_numeric_columns = 0
        dropped_metadata_columns = 0
    return NeuroFrame(
        data=np.asarray(rows, dtype=float),
        channels=tuple(Channel(name=name, type=channel_type, unit=unit) for name in names),
        timebase=Timebase(sampling_rate=sampling_rate),
        provenance={
            "source": "csv",
            "path": str(resolved),
            "dropped_non_numeric_columns": dropped_non_numeric_columns,
            "dropped_metadata_columns": dropped_metadata_columns,
            "header_inferred": headerless,
        },
    )


def load_ts(
    path: str | Path,
    *,
    sampling_rate: float,
    channel_names: tuple[str, ...] | list[str] | None = None,
    channel_type: str = "eeg",
    unit: str = "uV",
) -> NeuroFrame:
    resolved = Path(path).expanduser()
    lines = [
        line.strip()
        for line in resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    data_start = next((index + 1 for index, line in enumerate(lines) if line.lower() == "@data"), 0)
    metadata_lines = lines[: data_start - 1] if data_start else [line for line in lines if line.startswith("@")]
    has_class_label = _ts_has_class_label(metadata_lines)
    rows = lines[data_start:] if data_start else [line for line in lines if not line.startswith("@")]
    series_rows = [_parse_ts_row(row, has_class_label=has_class_label) for row in rows if row and not row.startswith("@")]
    if not series_rows:
        raise ValueError("TS file does not contain EEG time-series rows")
    channel_count = len(series_rows[0])
    if any(len(row) != channel_count for row in series_rows):
        raise ValueError("TS file rows do not expose a consistent channel count")
    sample_blocks = []
    for row in series_rows:
        lengths = {len(channel) for channel in row}
        if len(lengths) != 1:
            raise ValueError("TS file contains channels with inconsistent sample counts")
        sample_blocks.append(np.asarray(row, dtype=float).T)
    data = np.vstack(sample_blocks)
    names = channel_names or [f"Ch{index + 1}" for index in range(channel_count)]
    return NeuroFrame(
        data=data,
        channels=_channels_for_data(data, names, channel_type=channel_type, unit=unit),
        timebase=Timebase(sampling_rate=float(sampling_rate)),
        provenance={
            "source": "ts",
            "path": str(resolved),
            "trial_count": len(series_rows),
        },
    )


def _ts_has_class_label(metadata_lines: list[str]) -> bool | None:
    for line in metadata_lines:
        if line.lower().startswith("@classlabel"):
            return line.lower().split()[1:2] == ["true"]
    return None


def _parse_ts_row(row: str, *, has_class_label: bool | None) -> list[list[float]]:
    parts = [part.strip() for part in row.split(":")]
    if len(parts) > 1 and (has_class_label is True or (has_class_label is None and _looks_like_ts_class_label(parts[-1]))):
        parts = parts[:-1]
    channels = [_parse_ts_channel(part) for part in parts if part]
    if not channels:
        raise ValueError("TS row does not contain numeric channel data")
    return channels


def _looks_like_ts_class_label(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if "," in text or "(" in text or ")" in text:
        return False
    try:
        float(text)
    except ValueError:
        return True
    return True


def _parse_ts_channel(value: str) -> list[float]:
    timestamped = re.findall(r"\([^,()]+,\s*([^()]+?)\)", value)
    tokens = timestamped or value.split(",")
    samples = []
    for token in tokens:
        item = token.strip().strip("()")
        if not item:
            continue
        if item == "?":
            samples.append(float("nan"))
            continue
        try:
            samples.append(float(item))
        except ValueError as exc:
            raise ValueError("TS channel contains a non-numeric sample") from exc
    if not samples:
        raise ValueError("TS channel does not contain numeric samples")
    return samples


def load_mne_raw(path: str | Path, *, preload: bool = True, unit: str = "uV") -> NeuroFrame:
    resolved = Path(path).expanduser()
    mne = _require_module("mne", extra="io")
    suffix = _loader_suffix(resolved)
    reader_names = _mne_reader_names_for_suffix(suffix, mne)
    if not reader_names:
        raise ValueError(f"MNE raw loader does not support format: {resolved.suffix}")
    raw = None
    used_reader_name = ""
    last_error: Exception | None = None
    for reader_name in reader_names:
        reader = getattr(mne.io, reader_name)
        try:
            raw = reader(str(resolved), preload=preload, verbose="ERROR")
        except Exception as exc:
            last_error = exc
            continue
        used_reader_name = reader_name
        break
    if raw is None:
        assert last_error is not None
        raise last_error
    data, names, types = _mne_raw_to_samples(raw)
    scale = 1_000_000.0 if unit == "uV" else 1.0
    return NeuroFrame(
        data=np.asarray(data, dtype=float) * scale,
        channels=tuple(Channel(name=name, type=_channel_type(kind), unit=unit) for name, kind in zip(names, types)),
        timebase=Timebase(sampling_rate=float(raw.info["sfreq"])),
        provenance={
            "source": "mne",
            "format": suffix.lstrip("."),
            "reader": f"mne.io.{used_reader_name}",
            "path": str(resolved),
            "native_unit": "V",
        },
    )


def _read_delimited_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        if _looks_whitespace_delimited(sample):
            return [line.strip().split() for line in handle if line.strip()]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(handle, dialect=dialect)
        return [row for row in reader if any(cell.strip() for cell in row)]


def _looks_whitespace_delimited(sample: str) -> bool:
    if any(delimiter in sample for delimiter in (",", "\t", ";")):
        return False
    non_empty = [line.strip() for line in sample.splitlines() if line.strip()]
    if not non_empty:
        return False
    return all(len(line.split()) > 1 for line in non_empty[:5])


def _row_is_numeric(row: list[str]) -> bool:
    if not row:
        return False
    try:
        for cell in row:
            float(cell)
    except ValueError:
        return False
    return True


def _csv_numeric_columns(
    header: list[str], rows: list[list[str]], *, drop_metadata: bool = True
) -> tuple[list[str], tuple[int, ...], int, int]:
    names = [item.strip() or f"Col{index + 1}" for index, item in enumerate(header)]
    numeric_indices = []
    dropped_non_numeric_columns = 0
    dropped_metadata_columns = 0
    for index in range(len(header)):
        try:
            for row in rows:
                float(row[index])
        except ValueError:
            dropped_non_numeric_columns += 1
            continue
        if drop_metadata and _is_csv_metadata_column(header[index], index, rows):
            dropped_metadata_columns += 1
            continue
        numeric_indices.append(index)
    if not numeric_indices:
        raise ValueError("CSV file does not contain numeric EEG columns")
    return [names[index] for index in numeric_indices], tuple(numeric_indices), dropped_non_numeric_columns, dropped_metadata_columns


def _is_csv_metadata_column(raw_name: str, index: int, rows: list[list[str]]) -> bool:
    normalized = "".join(ch for ch in raw_name.strip().lower() if ch.isalnum())
    if not normalized or normalized.startswith("unnamed"):
        return _is_index_like_column(index, rows)
    if normalized in {
        "index",
        "sample",
        "sampleid",
        "sampleindex",
        "time",
        "timepoint",
        "timestamp",
        "unixtimestamp",
        "datetime",
        "date",
    }:
        return True
    return normalized.endswith("timestamp")


def _is_index_like_column(index: int, rows: list[list[str]]) -> bool:
    if not rows:
        return False
    try:
        values = [float(row[index]) for row in rows]
    except ValueError:
        return False
    start = values[0]
    return all(value == start + offset for offset, value in enumerate(values))


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


def load_nwb(path: str | Path, *, series_name: str | None = None, unit: str = "uV") -> NeuroFrame:
    resolved = Path(path).expanduser()
    pynwb = _require_module("pynwb", extra="io")
    with pynwb.NWBHDF5IO(str(resolved), mode="r") as io:
        nwbfile = io.read()
        series = _select_nwb_electrical_series(nwbfile, series_name=series_name)
        data = np.asarray(series.data[:], dtype=float)
        data = _samples_by_channels(data)
        sampling_rate, timestamps = _nwb_timebase(series)
        names = _nwb_channel_names(series, data.shape[1])
        native_unit = str(getattr(series, "unit", "") or unit)
        scale = _unit_scale(native_unit, unit)
        return NeuroFrame(
            data=data * scale,
            channels=tuple(Channel(name=name, type="eeg", unit=unit) for name in names),
            timebase=Timebase(sampling_rate=sampling_rate, timestamps=timestamps, clock_source="nwb"),
            provenance={
                "source": "nwb",
                "path": str(resolved),
                "series_name": str(getattr(series, "name", series_name or "")),
                "native_unit": native_unit,
                "session_description": str(getattr(nwbfile, "session_description", "")),
                "identifier": str(getattr(nwbfile, "identifier", "")),
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


def load_torch(
    path: str | Path,
    *,
    sampling_rate: float | None = None,
    channel_names: tuple[str, ...] | list[str] | None = None,
    channel_type: str = "eeg",
    unit: str = "uV",
) -> NeuroFrame:
    resolved = Path(path).expanduser()
    payload = _load_torch_payload(resolved)
    data = _find_numeric_matrix({"payload": payload})
    sampling_rate = sampling_rate or _find_sampling_rate(payload if isinstance(payload, dict) else {})
    if sampling_rate is None:
        raise ValueError("sampling_rate is required when the PyTorch file does not expose fs/sfreq/sampling_rate")
    names = channel_names or _find_channel_names(payload if isinstance(payload, dict) else {})
    data = _samples_by_channels(data)
    return NeuroFrame(
        data=data,
        channels=_channels_for_data(data, names, channel_type=channel_type, unit=unit),
        timebase=Timebase(sampling_rate=float(sampling_rate)),
        provenance={"source": "torch", "path": str(resolved), "format": resolved.suffix.lower().lstrip(".")},
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
    files = [item for item in resolved.rglob("*") if item.is_file() and _looks_like_supported_signal_file(item)]
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
    preferred = [
        ".vhdr",
        ".edf",
        ".bdf",
        ".set",
        ".fif",
        ".gdf",
        ".cnt",
        ".egi",
        ".mff",
        ".xdf",
        ".mat",
        ".npz",
        ".npy",
        ".csv",
        ".ts",
        ".tsv",
        ".tab",
        ".txt",
    ]
    try:
        suffix_rank = preferred.index(suffix)
    except ValueError:
        suffix_rank = len(preferred)
    return eeg_dir_rank, suffix_rank, str(path)


def _looks_like_supported_signal_file(path: Path) -> bool:
    if not can_load_extension(path):
        return False
    if _loader_suffix(path) not in {".tab", ".txt", ".tsv"}:
        return True
    text = path.as_posix().casefold()
    if any(token in text for token in ("readme", "license", "stimuli", "stimulus", "questionnaire", "metadata")):
        return False
    return "eeg" in text or "raw" in text or bool(re.search(r"(^|[/_-])(subj?|subject|user|s)\d+", text))


def _require_module(name: str, *, extra: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(
            f"loading this EEG format requires optional dependency {name!r}; install NeuroCore with the {extra!r} extra"
        ) from exc


def _mne_reader_names_for_suffix(suffix: str, mne: Any) -> tuple[str, ...]:
    reader_name = MNE_RAW_READERS.get(suffix)
    if reader_name is None:
        return ()
    names = [reader_name]
    if suffix == ".cnt" and hasattr(mne.io, "read_raw_ant"):
        names.append("read_raw_ant")
    return tuple(dict.fromkeys(names))


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


def _select_nwb_electrical_series(nwbfile: Any, *, series_name: str | None) -> Any:
    candidates = []
    acquisition = getattr(nwbfile, "acquisition", {})
    candidates.extend(_nwb_series_from_mapping(acquisition))
    processing = getattr(nwbfile, "processing", {})
    for module in _mapping_values(processing):
        candidates.extend(_nwb_series_from_mapping(getattr(module, "data_interfaces", module)))
    if series_name:
        for series in candidates:
            if str(getattr(series, "name", "")) == series_name:
                return series
        raise ValueError(f"NWB file does not contain an ElectricalSeries named {series_name!r}")
    usable = [series for series in candidates if hasattr(series, "data")]
    if not usable:
        raise ValueError("NWB file does not contain an ElectricalSeries with data")
    usable.sort(key=lambda item: len(getattr(getattr(item, "data", None), "shape", ())) == 2, reverse=True)
    return usable[0]


def _nwb_series_from_mapping(mapping: Any) -> list[Any]:
    series = []
    for value in _mapping_values(mapping):
        if _looks_like_nwb_electrical_series(value):
            series.append(value)
        elif hasattr(value, "electrical_series"):
            series.extend(_nwb_series_from_mapping(getattr(value, "electrical_series")))
        elif hasattr(value, "data_interfaces"):
            series.extend(_nwb_series_from_mapping(getattr(value, "data_interfaces")))
    return series


def _mapping_values(mapping: Any) -> list[Any]:
    if isinstance(mapping, dict):
        return list(mapping.values())
    if hasattr(mapping, "values"):
        return list(mapping.values())
    if isinstance(mapping, (list, tuple)):
        return list(mapping)
    return []


def _looks_like_nwb_electrical_series(value: Any) -> bool:
    return hasattr(value, "data") and (hasattr(value, "electrodes") or "ElectricalSeries" in type(value).__name__)


def _nwb_timebase(series: Any) -> tuple[float, tuple[float, ...] | None]:
    rate = getattr(series, "rate", None)
    if rate:
        return float(rate), None
    timestamps = getattr(series, "timestamps", None)
    if timestamps is not None:
        values = tuple(float(item) for item in np.asarray(timestamps[:]).reshape(-1))
        if len(values) > 1:
            duration = values[-1] - values[0]
            if duration > 0:
                return (len(values) - 1) / duration, values
    raise ValueError("NWB ElectricalSeries does not expose rate or usable timestamps")


def _nwb_channel_names(series: Any, count: int) -> tuple[str, ...]:
    electrodes = getattr(series, "electrodes", None)
    dataframe = None
    if electrodes is not None and hasattr(electrodes, "to_dataframe"):
        try:
            dataframe = electrodes.to_dataframe()
        except Exception:
            dataframe = None
    if dataframe is not None:
        for column in ("label", "name", "electrode_name", "location"):
            if column in dataframe:
                names = tuple(str(item) for item in dataframe[column].tolist())
                if len(names) == count and all(name and name != "nan" for name in names):
                    return names
    return tuple(f"Ch{index + 1}" for index in range(count))


def _unit_scale(native_unit: str, target_unit: str) -> float:
    native = native_unit.strip().casefold()
    target = target_unit.strip().casefold()
    if target in {"uv", "µv", "μv"} and native in {"v", "volt", "volts"}:
        return 1_000_000.0
    if target in {"v", "volt", "volts"} and native in {"uv", "µv", "μv"}:
        return 0.000001
    return 1.0


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


def _load_torch_payload(path: Path) -> Any:
    torch = _require_module("torch", extra="io")
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise ValueError("PyTorch EEG loading requires torch.load(..., weights_only=True) support") from exc


def _find_numeric_matrix(payload: dict[str, Any]) -> np.ndarray:
    preferred = ("data", "eeg", "EEG", "signal", "signals", "X", "x")
    for key in preferred:
        if key not in payload:
            continue
        try:
            return _coerce_numeric_signal_array(payload[key])
        except ValueError:
            matrices = _nested_numeric_matrices(payload[key])
            if matrices:
                return max(matrices, key=lambda item: item.size)
    matrices: list[np.ndarray] = []
    for value in payload.values():
        matrices.extend(_nested_numeric_matrices(value))
    if not matrices:
        raise ValueError("MAT file does not contain a numeric EEG matrix")
    return max(matrices, key=lambda item: item.size)


def _nested_numeric_matrices(value: Any, *, depth: int = 0) -> list[np.ndarray]:
    if depth > 6:
        return []
    try:
        return [_coerce_numeric_signal_array(value)]
    except ValueError:
        pass
    matrices: list[np.ndarray] = []
    field_names = getattr(value, "_fieldnames", None)
    if field_names:
        for field in field_names:
            matrices.extend(_nested_numeric_matrices(getattr(value, field), depth=depth + 1))
        return matrices
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            for field in value.dtype.names:
                matrices.extend(_nested_numeric_matrices(value[field], depth=depth + 1))
        elif value.dtype == object:
            for item in value.flat:
                matrices.extend(_nested_numeric_matrices(item, depth=depth + 1))
    elif isinstance(value, dict):
        for item in value.values():
            matrices.extend(_nested_numeric_matrices(item, depth=depth + 1))
    elif isinstance(value, (list, tuple)):
        for item in value:
            matrices.extend(_nested_numeric_matrices(item, depth=depth + 1))
    return matrices


def _coerce_numeric_signal_array(value: Any) -> np.ndarray:
    array = _as_numpy_array(value)
    if array.ndim < 2 or not np.issubdtype(array.dtype, np.number):
        raise ValueError("value is not a numeric EEG array")
    if array.ndim == 2:
        return np.asarray(array, dtype=float)
    channel_axis = _guess_channel_axis(array.shape)
    moved = np.moveaxis(np.asarray(array, dtype=float), channel_axis, -1)
    return moved.reshape(-1, moved.shape[-1])


def _as_numpy_array(value: Any) -> np.ndarray:
    if all(hasattr(value, attr) for attr in ("detach", "cpu", "numpy")):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def _guess_channel_axis(shape: tuple[int, ...]) -> int:
    common_counts = {
        4,
        8,
        16,
        18,
        19,
        20,
        21,
        22,
        24,
        32,
        40,
        48,
        56,
        64,
        65,
        72,
        96,
        128,
        129,
        160,
        256,
    }
    sample_axis = max(range(len(shape)), key=lambda axis: shape[axis])

    def score(axis: int) -> tuple[int, int, int]:
        size = shape[axis]
        value = 0
        if axis == sample_axis:
            value -= 100
        if size in common_counts:
            value += 8
        if 4 <= size <= 256:
            value += 4
        if size <= 3:
            value -= 4
        if axis == 0:
            value -= 2
        if axis + 1 == sample_axis or axis - 1 == sample_axis:
            value += 2
        return (value, -size, -axis)

    return max(range(len(shape)), key=score)


def _find_sampling_rate(payload: dict[str, Any]) -> float | None:
    for key in ("sampling_rate", "sfreq", "fs", "srate"):
        if key in payload:
            return float(_as_numpy_array(payload[key]).reshape(-1)[0])
    return None


def _find_channel_names(payload: dict[str, Any]) -> tuple[str, ...] | None:
    for key in ("channel_names", "channels", "ch_names"):
        if key in payload:
            return tuple(str(item) for item in _as_numpy_array(payload[key]).reshape(-1))
    return None
