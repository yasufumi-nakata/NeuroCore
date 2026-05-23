from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from neurocore.acquisition import local_readiness_for_record, plan_inventory_acquisition, summarize_acquisition_plans
from neurocore.datasets import load_eeg_dataset_inventory
from neurocore.loaders import load


CONFIDENTIALITY = (
    "private dataset acquisition verification; raw dataset mirrors and generated reports should not be committed"
)
DEFAULT_INVENTORY = ROOT.parent / "EEG-DATA" / "eeg_dataset_summary_ja.csv"


@dataclass(frozen=True)
class LoadAttempt:
    path: str
    status: str
    summary: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "status": self.status, "summary": self.summary, "error": self.error}


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    inventory = load_eeg_dataset_inventory(args.inventory)
    plans = plan_inventory_acquisition(inventory)
    plan_summary = summarize_acquisition_plans(plans)
    readiness = []
    if args.cache_root:
        readiness = [local_readiness_for_record(record, args.cache_root) for record in inventory.records]
    load_attempts = []
    if args.load_local and args.cache_root:
        load_attempts = attempt_local_loads(readiness, max_files=args.max_loads, sampling_rate=args.sampling_rate)
    return {
        "confidentiality": CONFIDENTIALITY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inventory": str(args.inventory),
        "cache_root": str(args.cache_root) if args.cache_root else None,
        "plan_summary": plan_summary,
        "sample_plans": [plan.to_dict() for plan in plans[: args.sample_limit]],
        "local_readiness_summary": summarize_readiness(readiness),
        "sample_local_readiness": readiness[: args.sample_limit],
        "load_attempts": [attempt.to_dict() for attempt in load_attempts],
    }


def summarize_readiness(readiness: list[dict[str, Any]]) -> dict[str, Any]:
    if not readiness:
        return {"checked": False, "records_with_local_signal_files": 0, "local_signal_file_count": 0}
    local_counts = [item["local_signal_file_count"] for item in readiness]
    return {
        "checked": True,
        "records_checked": len(readiness),
        "records_with_local_signal_files": sum(1 for count in local_counts if count > 0),
        "local_signal_file_count": sum(local_counts),
    }


def attempt_local_loads(
    readiness: list[dict[str, Any]],
    *,
    max_files: int,
    sampling_rate: float,
) -> list[LoadAttempt]:
    attempts: list[LoadAttempt] = []
    for item in readiness:
        for path in item["local_signal_files"]:
            if len(attempts) >= max_files:
                return attempts
            try:
                frame = load(path, sampling_rate=sampling_rate)
            except Exception as exc:
                attempts.append(LoadAttempt(path=path, status="failed", error=f"{type(exc).__name__}: {exc}"))
            else:
                attempts.append(LoadAttempt(path=path, status="loaded", summary=frame.to_summary()))
    return attempts


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["plan_summary"]
    readiness = report["local_readiness_summary"]
    lines = [
        "# NeuroCore Dataset Acquisition Verification",
        "",
        f"- confidentiality: {report['confidentiality']}",
        f"- generated_at: {report['generated_at']}",
        f"- inventory: {report['inventory']}",
        f"- cache_root: {report['cache_root']}",
        "",
        "## Acquisition Plan",
        "",
        f"- records: {summary['record_count']}",
        f"- records_with_candidates: {summary['records_with_candidates']}",
        f"- records_without_candidates: {summary['records_without_candidates']}",
        f"- automation_status_counts: `{json.dumps(summary['automation_status_counts'], ensure_ascii=False)}`",
        "",
        "## Local Readiness",
        "",
        f"- checked: {readiness['checked']}",
        f"- records_with_local_signal_files: {readiness['records_with_local_signal_files']}",
        f"- local_signal_file_count: {readiness['local_signal_file_count']}",
        "",
        "## Notes",
        "",
        "- This report plans raw EEG acquisition and validates already-materialized local cache files only.",
        "- It does not claim that remote raw EEG files have all been downloaded.",
        "- Do not copy this report into public docs, releases, package metadata, or issue comments.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan EEG-DATA raw acquisition and verify any local raw cache.")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--cache-root", type=Path, default=None, help="Local raw dataset cache root to scan.")
    parser.add_argument("--load-local", action="store_true", help="Attempt to load local cached signal files.")
    parser.add_argument("--max-loads", type=int, default=25)
    parser.add_argument("--sampling-rate", type=float, default=250.0, help="Fallback rate for raw numeric files.")
    parser.add_argument("--sample-limit", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()
    stamp = now_stamp()
    output = args.output or ROOT / "private" / "reports" / f"neurocore_dataset_acquisition_{stamp}.json"
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
            {"plan_summary": report["plan_summary"], "local_readiness_summary": report["local_readiness_summary"]},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
