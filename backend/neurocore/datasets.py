from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EEG_DATA_JA_COLUMNS = {
    "ID": "record_id",
    "旧ID": "legacy_id",
    "データセット名": "name",
    "公開URL": "url",
    "DOI": "doi",
    "提供元ドメイン": "source_domain",
    "アクセス区分": "access_status",
    "評価点": "score",
    "サイズ": "size",
    "人数": "participants",
    "時間": "duration",
    "刺激の種類": "stimulus",
    "観測機器": "equipment",
    "実験条件": "conditions",
    "公開年": "year",
    "説明（日本語）": "description",
    "判定根拠": "evidence",
    "検索ソース": "search_sources",
}

CANONICAL_FIELDS = tuple(EEG_DATA_JA_COLUMNS.values())

COLUMN_ALIASES = {
    "record_id": ("ID", "id", "record_id", "dataset_id"),
    "legacy_id": ("旧ID", "legacy_id", "old_id"),
    "name": ("データセット名", "dataset_name", "name", "title"),
    "url": ("公開URL", "url", "public_url", "source_url"),
    "doi": ("DOI", "doi"),
    "source_domain": ("提供元ドメイン", "source_domain", "domain", "repository_domain"),
    "access_status": ("アクセス区分", "access_status", "access", "availability"),
    "score": ("評価点", "score", "rating"),
    "size": ("サイズ", "size", "data_size"),
    "participants": ("人数", "participants", "subjects", "n_participants"),
    "duration": ("時間", "duration", "recording_duration"),
    "stimulus": ("刺激の種類", "stimulus", "stimuli", "task"),
    "equipment": ("観測機器", "equipment", "device", "hardware"),
    "conditions": ("実験条件", "conditions", "condition"),
    "year": ("公開年", "year", "published_year", "publication_year"),
    "description": ("説明（日本語）", "description", "summary", "notes"),
    "evidence": ("判定根拠", "evidence", "basis", "rationale"),
    "search_sources": ("検索ソース", "search_sources", "sources", "references"),
}

REQUIRED_FIELDS = ("record_id", "name", "url", "doi", "source_domain", "access_status", "score", "description")

FORMAT_PATTERNS = {
    "bids": (r"\bBIDS\b",),
    "bdf": (r"\.bdf\b", r"\bBDF\b"),
    "cnt": (r"\.cnt\b", r"\bCNT\b"),
    "csv": (r"\.csv\b", r"\bCSV\b"),
    "edf": (r"\.edf\b", r"\bEDF\b"),
    "egi": (r"\.egi\b", r"\bEGI\b"),
    "eeglab_set": (r"\.set\b", r"\bEEGLAB\b"),
    "eximia": (r"\.nxe\b", r"\beXimia\b"),
    "fif": (r"\.fif\b", r"\bFIFF?\b"),
    "gdf": (r"\.gdf\b", r"\bGDF\b"),
    "hdf5": (r"\.h5\b", r"\.hdf5\b", r"\bHDF5\b", r"\bHDF\b"),
    "mat": (r"\.mat\b", r"\bMATLAB\b"),
    "mef": (r"\.mefd\b", r"\bMEF3?\b"),
    "mff": (r"\.mff\b", r"\bMFF\b"),
    "nicolet": (r"\.data\b", r"\bNicolet\b"),
    "nwb": (r"\.nwb\b", r"\bNWB\b", r"Neurodata Without Borders"),
    "npy_npz": (r"\.npy\b", r"\.npz\b"),
    "parquet": (r"\.parquet\b", r"\bParquet\b"),
    "persyst": (r"\.lay\b", r"\bPersyst\b"),
    "r_data": (r"\.rds\b", r"\.rda\b", r"\.rdata\b"),
    "torch": (r"\.pt\b", r"\.pth\b", r"\bPyTorch\b"),
    "ts": (r"\.ts\b", r"Time Series Classification"),
    "vhdr": (r"\.vhdr\b", r"\bBrainVision\b"),
    "xdf": (r"\.xdf\b", r"\bXDF\b"),
    "xlsx": (r"\.xlsx?\b", r"\bExcel\b", r"\bworkbook\b"),
}

SIGNAL_LOADER_FORMATS = {
    "bdf",
    "bids",
    "cnt",
    "csv",
    "edf",
    "egi",
    "eeglab_set",
    "eximia",
    "fif",
    "gdf",
    "hdf5",
    "mat",
    "mef",
    "mff",
    "nicolet",
    "nwb",
    "npy_npz",
    "parquet",
    "persyst",
    "r_data",
    "torch",
    "ts",
    "vhdr",
    "xdf",
}

CONTAINER_OR_METADATA_FORMATS = {"xlsx"}


@dataclass(frozen=True)
class DatasetRecord:
    record_id: str
    legacy_id: str
    name: str
    url: str
    doi: str
    source_domain: str
    access_status: str
    score: int | None
    size: str
    participants: str
    duration: str
    stimulus: str
    equipment: str
    conditions: str
    year: int | None
    description: str
    evidence: str
    search_sources: tuple[str, ...]

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "DatasetRecord":
        values = {field: str(row.get(field, "") or "").strip() for field in CANONICAL_FIELDS}
        return cls(
            record_id=values["record_id"],
            legacy_id=values["legacy_id"],
            name=values["name"],
            url=values["url"],
            doi=values["doi"],
            source_domain=values["source_domain"],
            access_status=values["access_status"],
            score=_parse_int(values["score"]),
            size=values["size"],
            participants=values["participants"],
            duration=values["duration"],
            stimulus=values["stimulus"],
            equipment=values["equipment"],
            conditions=values["conditions"],
            year=_parse_int(values["year"]),
            description=values["description"],
            evidence=values["evidence"],
            search_sources=_split_sources(values["search_sources"]),
        )

    @property
    def text_for_detection(self) -> str:
        return " ".join(
            [
                self.size,
                self.duration,
                self.stimulus,
                self.equipment,
                self.conditions,
                self.description,
                self.evidence,
                " ".join(self.search_sources),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "legacy_id": self.legacy_id,
            "name": self.name,
            "url": self.url,
            "doi": self.doi,
            "source_domain": self.source_domain,
            "access_status": self.access_status,
            "score": self.score,
            "size": self.size,
            "participants": self.participants,
            "duration": self.duration,
            "stimulus": self.stimulus,
            "equipment": self.equipment,
            "conditions": self.conditions,
            "year": self.year,
            "description": self.description,
            "evidence": self.evidence,
            "search_sources": list(self.search_sources),
        }


@dataclass(frozen=True)
class DatasetInventory:
    source: Path
    records: tuple[DatasetRecord, ...]
    header: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.records)

    def access_status_counts(self) -> dict[str, int]:
        return dict(Counter(record.access_status or "(missing)" for record in self.records))

    def domain_counts(self, *, limit: int | None = None) -> dict[str, int]:
        counts = Counter(record.source_domain or "(missing)" for record in self.records)
        items = counts.most_common(limit)
        return dict(items)

    def format_mentions(self) -> dict[str, int]:
        counts: dict[str, int] = {name: 0 for name in FORMAT_PATTERNS}
        for record in self.records:
            text = record.text_for_detection
            for name, patterns in FORMAT_PATTERNS.items():
                if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
                    counts[name] += 1
        return {name: count for name, count in counts.items() if count}

    def loader_coverage(self) -> dict[str, Any]:
        format_counts = self.format_mentions()
        supported = {name: count for name, count in format_counts.items() if name in SIGNAL_LOADER_FORMATS}
        container_or_metadata = {
            name: count for name, count in format_counts.items() if name in CONTAINER_OR_METADATA_FORMATS
        }
        unsupported = {
            name: count
            for name, count in format_counts.items()
            if name not in SIGNAL_LOADER_FORMATS and name not in CONTAINER_OR_METADATA_FORMATS
        }
        records_with_supported_signal_hint = 0
        records_with_only_container_or_metadata_hint = 0
        records_without_format_hint = 0
        for record in self.records:
            mentioned = _record_format_mentions(record)
            if mentioned & SIGNAL_LOADER_FORMATS:
                records_with_supported_signal_hint += 1
            elif mentioned & CONTAINER_OR_METADATA_FORMATS:
                records_with_only_container_or_metadata_hint += 1
            else:
                records_without_format_hint += 1
        return {
            "record_count": len(self.records),
            "supported_signal_format_counts": supported,
            "container_or_metadata_format_counts": container_or_metadata,
            "unsupported_format_counts": unsupported,
            "records_with_supported_signal_hint": records_with_supported_signal_hint,
            "records_with_only_container_or_metadata_hint": records_with_only_container_or_metadata_hint,
            "records_without_format_hint": records_without_format_hint,
        }

    def validate(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        header_fields = set(_canonical_header_map(self.header))
        missing_columns = [field for field in REQUIRED_FIELDS if field not in header_fields]
        if missing_columns:
            issues.append({"code": "missing_columns", "severity": "error", "columns": missing_columns})
        if not self.records:
            issues.append({"code": "empty_inventory", "severity": "error"})
        missing_names = [record.record_id for record in self.records if not record.name]
        if missing_names:
            issues.append(
                {
                    "code": "missing_dataset_name",
                    "severity": "error",
                    "record_ids": missing_names[:20],
                    "count": len(missing_names),
                }
            )
        missing_locations = [record.record_id for record in self.records if not record.url and not record.doi]
        if missing_locations:
            issues.append(
                {
                    "code": "missing_access_location",
                    "severity": "warning",
                    "record_ids": missing_locations[:20],
                    "count": len(missing_locations),
                }
            )
        return issues

    def summary(self) -> dict[str, Any]:
        issues = self.validate()
        return {
            "source": str(self.source),
            "record_count": len(self),
            "header_columns": list(self.header),
            "access_status_counts": self.access_status_counts(),
            "top_domains": self.domain_counts(limit=10),
            "format_mentions": self.format_mentions(),
            "loader_coverage": self.loader_coverage(),
            "issues": issues,
        }


def load_eeg_dataset_inventory(path: str | Path) -> DatasetInventory:
    resolved = Path(path).expanduser()
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("dataset inventory CSV is empty")
        header = tuple(item.strip() for item in reader.fieldnames)
        header_map = _canonical_header_map(header)
        missing_columns = [field for field in REQUIRED_FIELDS if field not in header_map]
        if missing_columns:
            raise ValueError(f"dataset inventory is missing required fields: {', '.join(missing_columns)}")
        records = tuple(DatasetRecord.from_row(_canonical_row(row, header_map)) for row in reader)
    return DatasetInventory(source=resolved, records=records, header=header)


def supported_signal_file_counts(paths: Iterable[Path]) -> dict[str, int]:
    raw_extensions = {
        ".bdf",
        ".cnt",
        ".edf",
        ".eeg",
        ".egi",
        ".fdt",
        ".fif",
        ".gdf",
        ".data",
        ".lay",
        ".mat",
        ".mefd",
        ".mff",
        ".npy",
        ".npz",
        ".nwb",
        ".nxe",
        ".set",
        ".vhdr",
        ".xdf",
    }
    counts = Counter(path.suffix.lower() for path in paths if path.suffix.lower() in raw_extensions)
    return dict(sorted(counts.items()))


def _parse_int(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _split_sources(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in re.split(r";\s*", value) if item.strip())


def _canonical_header_map(header: tuple[str, ...]) -> dict[str, str]:
    normalized = {_normalize_column_name(column): column for column in header}
    header_map: dict[str, str] = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            column = normalized.get(_normalize_column_name(alias))
            if column is not None:
                header_map[field] = column
                break
    return header_map


def _canonical_row(row: dict[str, str], header_map: dict[str, str]) -> dict[str, str]:
    return {field: row.get(column, "") for field, column in header_map.items()}


def _normalize_column_name(value: str) -> str:
    return re.sub(r"[\s_\-（）()]+", "", value.strip().casefold())


def _record_format_mentions(record: DatasetRecord) -> set[str]:
    text = record.text_for_detection
    return {
        name
        for name, patterns in FORMAT_PATTERNS.items()
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
    }
