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

REQUIRED_EEG_DATA_COLUMNS = tuple(EEG_DATA_JA_COLUMNS)

FORMAT_PATTERNS = {
    "bids": (r"\bBIDS\b",),
    "bdf": (r"\.bdf\b", r"\bBDF\b"),
    "csv": (r"\.csv\b", r"\bCSV\b"),
    "edf": (r"\.edf\b", r"\bEDF\b"),
    "eeglab_set": (r"\.set\b", r"\bEEGLAB\b"),
    "fif": (r"\.fif\b", r"\bFIFF?\b"),
    "mat": (r"\.mat\b", r"\bMATLAB\b"),
    "npy_npz": (r"\.npy\b", r"\.npz\b"),
    "vhdr": (r"\.vhdr\b", r"\bBrainVision\b"),
    "xdf": (r"\.xdf\b", r"\bXDF\b"),
    "xlsx": (r"\.xlsx?\b", r"\bExcel\b", r"\bworkbook\b"),
}


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
        values = {field: row.get(column, "").strip() for column, field in EEG_DATA_JA_COLUMNS.items()}
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

    def validate(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        missing_columns = [column for column in REQUIRED_EEG_DATA_COLUMNS if column not in self.header]
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
            "issues": issues,
        }


def load_eeg_dataset_inventory(path: str | Path) -> DatasetInventory:
    resolved = Path(path).expanduser()
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("dataset inventory CSV is empty")
        header = tuple(item.strip() for item in reader.fieldnames)
        missing_columns = [column for column in REQUIRED_EEG_DATA_COLUMNS if column not in header]
        if missing_columns:
            raise ValueError(f"dataset inventory is missing required columns: {', '.join(missing_columns)}")
        records = tuple(DatasetRecord.from_row(row) for row in reader)
    return DatasetInventory(source=resolved, records=records, header=header)


def supported_signal_file_counts(paths: Iterable[Path]) -> dict[str, int]:
    raw_extensions = {".bdf", ".edf", ".eeg", ".fdt", ".fif", ".mat", ".npy", ".npz", ".set", ".vhdr", ".xdf"}
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
