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
from neurocore.exercise import exercise_dataset_records, summarize_dataset_exercise
from neurocore.loaders import load
from neurocore.materialization import (
    extract_supported_signal_files_from_archive,
    materialize_remote_files,
    resolve_inventory_remote_files,
    summarize_remote_file_resolutions,
)


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
    remote_resolutions = []
    materialization_results = []
    archive_extractions = []
    if args.resolve_remote_files:
        remote_resolutions = list(
            resolve_inventory_remote_files(
                plans,
                limit=None if args.resolve_limit == 0 else args.resolve_limit,
                providers=set(args.resolve_provider) if args.resolve_provider else None,
                automation_statuses=(
                    set(args.resolve_status)
                    if args.resolve_status
                    else {"direct_api", "tooling_required", "doi_resolution_required", "manual_review", "account_required"}
                ),
                timeout=args.http_timeout,
                max_pages=args.max_pages,
            )
        )
    if args.materialize and args.cache_root:
        remote_files = [file for resolution in remote_resolutions for file in resolution.files]
        materialization_results = list(
            materialize_remote_files(
                remote_files,
                args.cache_root,
                max_files=None if args.max_downloads == 0 else args.max_downloads,
                max_bytes=args.max_download_bytes,
                direct_only=not args.include_archives,
            )
        )
    if args.extract_archives:
        archive_extractions = [
            extract_supported_signal_files_from_archive(
                result.path,
                max_members=args.max_archive_members,
                max_member_bytes=args.max_archive_member_bytes,
            )
            for result in materialization_results
            if result.path and result.path.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))
        ]
    readiness = []
    if args.cache_root:
        max_local_files = None if args.max_local_files_per_record == 0 else args.max_local_files_per_record
        readiness = [
            local_readiness_for_record(record, args.cache_root, max_files=max_local_files)
            for record in inventory.records
        ]
    load_attempts = []
    if args.load_local and args.cache_root:
        load_attempts = attempt_local_loads(
            readiness,
            max_files=None if args.max_loads == 0 else args.max_loads,
            sampling_rate=args.sampling_rate,
        )
    exercise_records = exercise_dataset_records(plans, remote_resolutions=remote_resolutions, local_readiness=readiness)
    return {
        "confidentiality": CONFIDENTIALITY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inventory": str(args.inventory),
        "cache_root": str(args.cache_root) if args.cache_root else None,
        "plan_summary": plan_summary,
        "sample_plans": [plan.to_dict() for plan in plans[: args.sample_limit]],
        "dataset_exercise_summary": summarize_dataset_exercise(exercise_records),
        "sample_dataset_exercise_records": [record.to_dict() for record in exercise_records[: args.sample_limit]],
        "remote_file_resolution_summary": summarize_remote_file_resolutions(remote_resolutions),
        "sample_remote_file_resolutions": [resolution.to_dict() for resolution in remote_resolutions[: args.sample_limit]],
        "materialization_summary": summarize_materialization(materialization_results),
        "sample_materialization_results": [result.to_dict() for result in materialization_results[: args.sample_limit]],
        "archive_extraction_summary": summarize_archive_extractions(archive_extractions),
        "sample_archive_extractions": [result.to_dict() for result in archive_extractions[: args.sample_limit]],
        "local_readiness_summary": summarize_readiness(readiness),
        "sample_local_readiness": readiness[: args.sample_limit],
        "load_attempt_summary": summarize_load_attempts(load_attempts),
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


def summarize_materialization(results: list[Any]) -> dict[str, Any]:
    if not results:
        return {"attempted": False, "result_count": 0, "status_counts": {}}
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    return {
        "attempted": True,
        "result_count": len(results),
        "status_counts": status_counts,
        "bytes_written": sum(result.bytes_written for result in results),
    }


def summarize_archive_extractions(results: list[Any]) -> dict[str, Any]:
    if not results:
        return {"attempted": False, "result_count": 0, "status_counts": {}, "signal_file_count": 0}
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    return {
        "attempted": True,
        "result_count": len(results),
        "status_counts": status_counts,
        "signal_file_count": sum(result.signal_file_count for result in results),
    }


def summarize_load_attempts(attempts: list[LoadAttempt]) -> dict[str, Any]:
    if not attempts:
        return {"attempted": False, "result_count": 0, "status_counts": {}}
    status_counts: dict[str, int] = {}
    for attempt in attempts:
        status_counts[attempt.status] = status_counts.get(attempt.status, 0) + 1
    return {
        "attempted": True,
        "result_count": len(attempts),
        "status_counts": status_counts,
    }


def attempt_local_loads(
    readiness: list[dict[str, Any]],
    *,
    max_files: int | None,
    sampling_rate: float,
) -> list[LoadAttempt]:
    attempts: list[LoadAttempt] = []
    for item in readiness:
        for path in item["local_signal_files"]:
            if max_files is not None and len(attempts) >= max_files:
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
        "## Remote File Resolution",
        "",
        f"- exercise_state_counts: `{json.dumps(report['dataset_exercise_summary']['state_counts'], ensure_ascii=False)}`",
        f"- records_with_actionable_raw_path: {report['dataset_exercise_summary']['records_with_actionable_raw_path']}",
        f"- records_with_external_blocker: {report['dataset_exercise_summary']['records_with_external_blocker']}",
        f"- records_needing_resolver_work: {report['dataset_exercise_summary']['records_needing_resolver_work']}",
        f"- records_not_yet_remotely_exercised: {report['dataset_exercise_summary']['records_not_yet_remotely_exercised']}",
        f"- records_resolved: {report['remote_file_resolution_summary']['records_resolved']}",
        f"- remote_file_count: {report['remote_file_resolution_summary']['remote_file_count']}",
        f"- directly_loadable_file_count: {report['remote_file_resolution_summary']['directly_loadable_file_count']}",
        f"- archive_file_count: {report['remote_file_resolution_summary']['archive_file_count']}",
        f"- error_count: {report['remote_file_resolution_summary']['error_count']}",
        "",
        "## Materialization",
        "",
        f"- attempted: {report['materialization_summary']['attempted']}",
        f"- result_count: {report['materialization_summary']['result_count']}",
        f"- status_counts: `{json.dumps(report['materialization_summary']['status_counts'], ensure_ascii=False)}`",
        "",
        "## Archive Extraction",
        "",
        f"- attempted: {report['archive_extraction_summary']['attempted']}",
        f"- result_count: {report['archive_extraction_summary']['result_count']}",
        f"- signal_file_count: {report['archive_extraction_summary']['signal_file_count']}",
        f"- status_counts: `{json.dumps(report['archive_extraction_summary']['status_counts'], ensure_ascii=False)}`",
        "",
        "## Local Readiness",
        "",
        f"- checked: {readiness['checked']}",
        f"- records_with_local_signal_files: {readiness['records_with_local_signal_files']}",
        f"- local_signal_file_count: {readiness['local_signal_file_count']}",
        "",
        "## Load Attempts",
        "",
        f"- attempted: {report['load_attempt_summary']['attempted']}",
        f"- result_count: {report['load_attempt_summary']['result_count']}",
        f"- status_counts: `{json.dumps(report['load_attempt_summary']['status_counts'], ensure_ascii=False)}`",
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
    parser.add_argument("--max-loads", type=int, default=25, help="Maximum local files to load; use 0 for every listed file.")
    parser.add_argument(
        "--max-local-files-per-record",
        type=int,
        default=20,
        help="Maximum local signal file paths to include per dataset; use 0 when auditing every cached file.",
    )
    parser.add_argument("--sampling-rate", type=float, default=250.0, help="Fallback rate for raw numeric files.")
    parser.add_argument("--resolve-remote-files", action="store_true", help="Query public provider APIs for file lists.")
    parser.add_argument(
        "--resolve-limit",
        type=int,
        default=25,
        help="Maximum records to resolve remotely; use 0 to attempt every selected record.",
    )
    parser.add_argument(
        "--resolve-provider",
        action="append",
        default=[],
        help="Restrict remote resolution to a provider; repeat for multiple providers.",
    )
    parser.add_argument(
        "--resolve-status",
        action="append",
        default=[],
        help="Restrict remote resolution to an automation status; defaults to every resolvable status, including manual/account rows with public candidates.",
    )
    parser.add_argument("--http-timeout", type=float, default=20.0)
    parser.add_argument("--max-pages", type=int, default=30)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Download resolved files into --cache-root. Defaults to directly loadable files only.",
    )
    parser.add_argument("--max-downloads", type=int, default=5, help="Maximum resolved files to download; use 0 for no cap.")
    parser.add_argument("--max-download-bytes", type=int, default=100_000_000)
    parser.add_argument(
        "--include-archives",
        action="store_true",
        help="Allow archive downloads. Archives still need extraction before raw files can be loaded.",
    )
    parser.add_argument("--extract-archives", action="store_true", help="Extract downloaded zip/tar archives safely.")
    parser.add_argument("--max-archive-members", type=int, default=20_000)
    parser.add_argument("--max-archive-member-bytes", type=int, default=2_000_000_000)
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
            {
                "plan_summary": report["plan_summary"],
                "dataset_exercise_summary": report["dataset_exercise_summary"],
                "remote_file_resolution_summary": report["remote_file_resolution_summary"],
                "materialization_summary": report["materialization_summary"],
                "archive_extraction_summary": report["archive_extraction_summary"],
                "local_readiness_summary": report["local_readiness_summary"],
                "load_attempt_summary": report["load_attempt_summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
