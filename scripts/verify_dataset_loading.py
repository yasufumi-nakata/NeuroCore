from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from neurocore.datasets import load_eeg_dataset_inventory, supported_signal_file_counts
from neurocore.kernels import Bandpass, ReReference, Resample, SpectralFeatures, ValidateEEG
from neurocore.loaders import load_csv, supported_extensions
from neurocore.pipeline import Pipeline, PipelineExecutionError
from neurocore.settings import NeuroCoreSettings


CONFIDENTIALITY = "private dataset loading verification; generated reports should stay out of public docs and packages"


def default_eeg_data_inventory() -> Path:
    configured = os.environ.get("EEG_DATA_INVENTORY")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(ROOT.parent / "EEG-DATA" / "eeg_dataset_summary_ja.csv")
    candidates.append(Path("eeg_dataset_summary_ja.csv"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


DEFAULT_EEG_DATA_INVENTORY = default_eeg_data_inventory()


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    expected: Any
    actual: Any
    group: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
        }


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.SubprocessError:
        return "unknown"


def percent(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def summarize_checks(group: str, checks: list[Check], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    passed = sum(check.passed for check in checks)
    payload = {
        "group": group,
        "total_checks": len(checks),
        "passed_checks": passed,
        "failed_checks": len(checks) - passed,
        "accuracy": percent(passed, len(checks)),
        "checks": [check.to_dict() for check in checks],
    }
    if extra:
        payload.update(extra)
    return payload


def evaluate_inventory(path: Path) -> dict[str, Any]:
    checks = [Check("inventory.exists", path.is_file(), True, path.is_file(), "inventory")]
    summary: dict[str, Any] = {"source": str(path)}
    sample_records: list[dict[str, Any]] = []
    if not path.is_file():
        return summarize_checks("inventory", checks, {"summary": summary, "sample_records": sample_records})
    try:
        inventory = load_eeg_dataset_inventory(path)
    except Exception as exc:  # pragma: no cover - defensive report path
        checks.append(Check("inventory.loads", False, "loaded", f"{type(exc).__name__}: {exc}", "inventory"))
        return summarize_checks("inventory", checks, {"summary": summary, "sample_records": sample_records})
    summary = inventory.summary()
    sample_records = [record.to_dict() for record in inventory.records[:5]]
    error_issues = [issue for issue in summary["issues"] if issue["severity"] == "error"]
    checks.extend(
        [
            Check("inventory.loads", True, "loaded", "loaded", "inventory"),
            Check("inventory.record_count", len(inventory) > 0, ">0", len(inventory), "inventory"),
            Check("inventory.no_error_issues", not error_issues, [], error_issues, "inventory"),
            Check(
                "inventory.access_status_present",
                bool(summary["access_status_counts"]),
                "nonempty_counts",
                summary["access_status_counts"],
                "inventory",
            ),
            Check(
                "inventory.format_mentions_present",
                bool(summary["format_mentions"]),
                "nonempty_counts",
                summary["format_mentions"],
                "inventory",
            ),
        ]
    )
    return summarize_checks("inventory", checks, {"summary": summary, "sample_records": sample_records})


def evaluate_signal_csv(path: Path, sampling_rate: float) -> dict[str, Any]:
    checks = [Check("signal_csv.exists", path.is_file(), True, path.is_file(), "signal_csv")]
    details: dict[str, Any] = {"source": str(path), "sampling_rate": sampling_rate}
    if not path.is_file():
        return summarize_checks("signal_csv", checks, details)
    try:
        frame = load_csv(path, sampling_rate=sampling_rate)
    except Exception as exc:  # pragma: no cover - defensive report path
        checks.append(Check("signal_csv.loads", False, "loaded", f"{type(exc).__name__}: {exc}", "signal_csv"))
        return summarize_checks("signal_csv", checks, details)
    details["frame"] = frame.to_summary()
    checks.extend(
        [
            Check("signal_csv.loads", True, "loaded", "loaded", "signal_csv"),
            Check("signal_csv.samples", frame.samples > 0, ">0", frame.samples, "signal_csv"),
            Check("signal_csv.channels", frame.channel_count >= 2, ">=2", frame.channel_count, "signal_csv"),
        ]
    )
    settings = NeuroCoreSettings.default()
    pipeline = Pipeline(
        [
            ValidateEEG(min_channels=2),
            Resample(settings.signal.target_sampling_rate),
            Bandpass(settings.signal.highpass_hz, settings.signal.lowpass_hz),
            ReReference("average"),
            SpectralFeatures(),
        ],
        name="dataset-loading-reference-pipeline",
    )
    try:
        result = pipeline.run(frame)
    except PipelineExecutionError as exc:
        report = exc.report.to_dict() if exc.report else None
        checks.append(Check("signal_csv.pipeline", False, "passed", report or str(exc), "signal_csv"))
    else:
        payload = result.to_dict()
        details["pipeline"] = payload["report"]
        details["output"] = payload["output"]
        checks.extend(
            [
                Check(
                    "signal_csv.pipeline",
                    payload["report"]["status"] == "passed",
                    "passed",
                    payload["report"]["status"],
                    "signal_csv",
                ),
                Check(
                    "signal_csv.features",
                    payload["output"]["kind"] == "features",
                    "features",
                    payload["output"]["kind"],
                    "signal_csv",
                ),
            ]
        )
    return summarize_checks("signal_csv", checks, details)


def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for current, dirs, names in os.walk(root):
        dirs[:] = [name for name in dirs if name != ".git"]
        current_path = Path(current)
        files.extend(current_path / name for name in names)
    return files


def evaluate_local_dataset_files(root: Path, *, skip_scan: bool) -> dict[str, Any]:
    checks = [Check("local_dataset_root.exists", root.is_dir(), True, root.is_dir(), "local_dataset_files")]
    if skip_scan or not root.is_dir():
        return summarize_checks("local_dataset_files", checks, {"root": str(root), "skipped": skip_scan})
    files = iter_files(root)
    extension_counts = Counter(path.suffix.lower() or "(none)" for path in files)
    raw_signal_counts = supported_signal_file_counts(files)
    csv_files = [str(path.relative_to(root)) for path in files if path.suffix.lower() == ".csv"][:20]
    checks.extend(
        [
            Check("local_dataset_files.scanned", True, "scanned", len(files), "local_dataset_files"),
            Check(
                "local_dataset_files.raw_signal_extensions_known",
                all(
                    extension in supported_extensions() or extension in {".eeg", ".fdt"}
                    for extension in raw_signal_counts
                ),
                "known raw EEG extensions",
                raw_signal_counts,
                "local_dataset_files",
            ),
        ]
    )
    return summarize_checks(
        "local_dataset_files",
        checks,
        {
            "root": str(root),
            "file_count": len(files),
            "extension_counts": dict(extension_counts.most_common(30)),
            "raw_signal_file_counts": raw_signal_counts,
            "supported_loader_extensions": supported_extensions(),
            "sample_csv_files": csv_files,
        },
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# NeuroCore Dataset Loading Verification",
        "",
        f"- confidentiality: {report['confidentiality']}",
        f"- generated_at: {report['generated_at']}",
        f"- git_commit: {report['git_commit']}",
        "- scope: EEG-DATA inventory CSV loading, local raw-file scan, and NeuroCore signal CSV pipeline smoke test.",
        "",
        "## Summary",
        "",
        "| Group | Checks | Passed | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, metric in report["metrics"].items():
        accuracy = metric["accuracy"]
        rendered_accuracy = "n/a" if accuracy is None else f"{accuracy:.6f}"
        lines.append(f"| {name} | {metric['total_checks']} | {metric['passed_checks']} | {rendered_accuracy} |")
    lines.extend(
        [
            f"| overall | {report['overall']['total_checks']} | {report['overall']['passed_checks']} | {report['overall']['accuracy']:.6f} |",
            "",
            "## Notes",
            "",
            "- EEG-DATA is treated as a dataset inventory checkout unless local raw EEG files are present.",
            "- NeuroCore can normalize CSV, NumPy, MNE-supported EEG files, XDF, and generic MAT files into `NeuroFrame` when the relevant optional dependencies are installed.",
            "- Dataset inventory CSV files are loaded through the inventory API, not treated as raw EEG signals.",
            "- Do not copy this report into public docs, releases, package metadata, or issue comments.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    inventory_path = args.inventory.expanduser()
    dataset_root = args.dataset_root.expanduser() if args.dataset_root else inventory_path.parent
    metrics = {
        "inventory": evaluate_inventory(inventory_path),
        "signal_csv": evaluate_signal_csv(args.signal.expanduser(), args.sampling_rate),
        "local_dataset_files": evaluate_local_dataset_files(dataset_root, skip_scan=args.skip_local_scan),
    }
    total = sum(metric["total_checks"] for metric in metrics.values())
    passed = sum(metric["passed_checks"] for metric in metrics.values())
    return {
        "confidentiality": CONFIDENTIALITY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "metrics": metrics,
        "overall": {
            "total_checks": total,
            "passed_checks": passed,
            "failed_checks": total - passed,
            "accuracy": percent(passed, total),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local EEG dataset inventory and NeuroCore signal loading.")
    parser.add_argument(
        "--inventory", type=Path, default=DEFAULT_EEG_DATA_INVENTORY, help="EEG-DATA Japanese inventory CSV."
    )
    parser.add_argument("--dataset-root", type=Path, default=None, help="Root to scan for local raw EEG files.")
    parser.add_argument(
        "--signal", type=Path, default=ROOT / "samples" / "synthetic_eeg.csv", help="Signal CSV fixture to load as EEG."
    )
    parser.add_argument("--sampling-rate", type=float, default=250.0)
    parser.add_argument(
        "--skip-local-scan", action="store_true", help="Skip scanning the dataset checkout for local raw EEG files."
    )
    parser.add_argument("--output", type=Path, default=None, help="Private JSON report path.")
    parser.add_argument("--markdown-output", type=Path, default=None, help="Private Markdown summary path.")
    args = parser.parse_args()
    stamp = now_stamp()
    output = args.output or ROOT / "private" / "reports" / f"neurocore_dataset_loading_{stamp}.json"
    markdown_output = args.markdown_output or output.with_suffix(".md")
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote private JSON report: {output}")
    print(f"wrote private Markdown report: {markdown_output}")
    print(
        json.dumps(
            {
                "confidentiality": report["confidentiality"],
                "overall": report["overall"],
                "groups": {
                    name: {
                        "total_checks": metric["total_checks"],
                        "passed_checks": metric["passed_checks"],
                        "accuracy": metric["accuracy"],
                    }
                    for name, metric in report["metrics"].items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if report["overall"]["failed_checks"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
