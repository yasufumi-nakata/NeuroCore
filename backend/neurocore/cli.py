from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .acquisition import plan_inventory_acquisition, summarize_acquisition_plans
from .datasets import load_eeg_dataset_inventory
from .audit import ActionAuditLog, DryRunActionSink
from . import __version__
from .control import ControlRouter, IntentCommand
from .frame import Channel
from .kernels import Bandpass, ReReference, Resample, SpectralFeatures, ValidateEEG
from .loaders import load, load_csv, supported_extensions
from .materialization import resolve_inventory_remote_files, summarize_remote_file_resolutions
from .pipeline import Pipeline, PipelineExecutionError
from .quality import score_signal_quality
from .selftest import run_self_tests
from .settings import NeuroCoreSettings
from .stream import StreamBuffer
from .synthetic import synthetic_eeg_frame


def cmd_settings(args: argparse.Namespace) -> int:
    settings = NeuroCoreSettings.default()
    payload = {"settings": settings.to_dict(), "issues": settings.validate()}
    if args.out:
        settings.save(args.out)
        print(f"wrote {args.out}")
        return 0
    _print(payload, json_mode=args.json)
    return 0


def cmd_self_test(args: argparse.Namespace) -> int:
    settings = NeuroCoreSettings.load(args.settings) if args.settings else NeuroCoreSettings.default()
    report = run_self_tests(settings)
    _print(report.to_dict(), json_mode=args.json)
    return 0 if report.status == "passed" else 1


def cmd_demo(args: argparse.Namespace) -> int:
    settings = NeuroCoreSettings.load(args.settings) if args.settings else NeuroCoreSettings.default()
    frame = synthetic_eeg_frame(
        seconds=args.seconds, sampling_rate=settings.device.sampling_rate, channels=settings.device.channels
    )
    return _run_reference_pipeline(frame, settings, json_mode=args.json)


def cmd_run_csv(args: argparse.Namespace) -> int:
    settings = NeuroCoreSettings.load(args.settings) if args.settings else NeuroCoreSettings.default()
    frame = load_csv(args.csv, sampling_rate=args.sampling_rate or settings.device.sampling_rate)
    return _run_reference_pipeline(frame, settings, json_mode=args.json)


def cmd_run_file(args: argparse.Namespace) -> int:
    settings = NeuroCoreSettings.load(args.settings) if args.settings else NeuroCoreSettings.default()
    frame = load(args.path, sampling_rate=args.sampling_rate or settings.device.sampling_rate)
    return _run_reference_pipeline(frame, settings, json_mode=args.json)


def cmd_dataset_inventory(args: argparse.Namespace) -> int:
    inventory = load_eeg_dataset_inventory(args.csv)
    payload = inventory.summary()
    payload["supported_loader_extensions"] = supported_extensions()
    payload["sample_records"] = [record.to_dict() for record in inventory.records[: args.limit]]
    _print(payload, json_mode=args.json)
    return 0 if not any(issue["severity"] == "error" for issue in payload["issues"]) else 1


def cmd_dataset_acquisition_plan(args: argparse.Namespace) -> int:
    inventory = load_eeg_dataset_inventory(args.csv)
    plans = plan_inventory_acquisition(inventory)
    payload = summarize_acquisition_plans(plans)
    payload["sample_plans"] = [plan.to_dict() for plan in plans[: args.limit]]
    _print(payload, json_mode=args.json)
    return 0


def cmd_dataset_resolve_files(args: argparse.Namespace) -> int:
    inventory = load_eeg_dataset_inventory(args.csv)
    plans = plan_inventory_acquisition(inventory)
    resolutions = resolve_inventory_remote_files(
        plans,
        limit=None if args.limit == 0 else args.limit,
        providers=set(args.provider) if args.provider else None,
        automation_statuses=set(args.status) if args.status else {"direct_api", "tooling_required", "doi_resolution_required"},
        timeout=args.http_timeout,
        max_pages=args.max_pages,
    )
    payload = summarize_remote_file_resolutions(resolutions)
    file_limit = None if args.file_limit == 0 else args.file_limit
    payload["sample_resolutions"] = [
        resolution.to_dict(file_limit=file_limit) for resolution in resolutions[: args.sample_limit]
    ]
    _print(payload, json_mode=args.json)
    return 0 if payload["error_count"] == 0 else 1


def cmd_route(args: argparse.Namespace) -> int:
    settings = NeuroCoreSettings.load(args.settings) if args.settings else NeuroCoreSettings.default()
    payload = json.loads(args.payload) if args.payload else {}
    command = IntentCommand(args.intent, args.confidence, source="cli", payload=payload)
    action = ControlRouter(settings).route(command)
    _print(action.to_dict(), json_mode=args.json)
    return 0 if not action.blocked else 1


def cmd_simulate(args: argparse.Namespace) -> int:
    settings = NeuroCoreSettings.load(args.settings) if args.settings else NeuroCoreSettings.default()
    command_payloads = json.loads(Path(args.commands).expanduser().read_text(encoding="utf-8"))
    router = ControlRouter(settings)
    audit = ActionAuditLog()
    sink = DryRunActionSink(audit)
    for item in command_payloads:
        command = IntentCommand(
            item["intent"],
            float(item.get("confidence", 0.0)),
            source=item.get("source", "cli"),
            payload=dict(item.get("payload", {})),
            timestamp=float(item.get("timestamp", time.time())),
        )
        sink.submit(router.route(command), command)
    payload = audit.to_dict()
    if args.out:
        audit.write_jsonl(args.out)
    _print(payload, json_mode=args.json)
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    settings = NeuroCoreSettings.load(args.settings) if args.settings else NeuroCoreSettings.default()
    frame = synthetic_eeg_frame(
        seconds=args.seconds, sampling_rate=settings.device.sampling_rate, channels=settings.device.channels
    )
    _print(score_signal_quality(frame).to_dict(), json_mode=args.json)
    return 0


def cmd_stream_demo(args: argparse.Namespace) -> int:
    settings = NeuroCoreSettings.load(args.settings) if args.settings else NeuroCoreSettings.default()
    frame = synthetic_eeg_frame(
        seconds=args.seconds, sampling_rate=settings.device.sampling_rate, channels=settings.device.channels
    )
    buffer = StreamBuffer(
        channels=tuple(Channel(name=name) for name in settings.device.channels),
        sampling_rate=settings.device.sampling_rate,
        window_seconds=settings.signal.window_seconds,
        step_seconds=settings.signal.step_seconds,
        provenance={"source": "cli_stream_demo"},
    )
    chunk_size = max(1, int(round(settings.device.sampling_rate * settings.signal.step_seconds)))
    emitted = []
    for start in range(0, frame.samples, chunk_size):
        emitted.extend(buffer.append(frame.data[start : start + chunk_size]))
    _print(
        {"window_count": len(emitted), "windows": [window.to_summary() for window in emitted[:10]]}, json_mode=args.json
    )
    return 0


def _run_reference_pipeline(frame, settings: NeuroCoreSettings, *, json_mode: bool) -> int:
    pipeline = Pipeline(
        [
            ValidateEEG(min_channels=2),
            Resample(settings.signal.target_sampling_rate),
            Bandpass(settings.signal.highpass_hz, settings.signal.lowpass_hz),
            ReReference("average"),
            SpectralFeatures(),
        ]
    )
    try:
        result = pipeline.run(frame)
    except PipelineExecutionError as exc:
        payload = {"status": "failed", "error": str(exc), "report": exc.report.to_dict() if exc.report else None}
        _print(payload, json_mode=json_mode)
        return 1
    payload = result.to_dict()
    payload["status"] = "passed"
    _print(payload, json_mode=json_mode)
    return 0


def _print(payload: dict, *, json_mode: bool) -> None:
    if json_mode:
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False))
        sys.stdout.write("\n")
        return
    if "settings" in payload:
        print("NeuroCore settings")
        print(f"  device: {payload['settings']['device']['name']}")
        print(f"  channels: {', '.join(payload['settings']['device']['channels'])}")
        print(f"  issues: {len(payload['issues'])}")
        return
    if "results" in payload:
        print(f"NeuroCore self-test: {payload['status']} ({payload['passed']} passed, {payload['failed']} failed)")
        for result in payload["results"]:
            print(f"  [{result['status'].upper()}] {result['name']}: {result['message']}")
        return
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurocore", description="NeuroCore EEG-to-control runtime")
    parser.add_argument("--version", action="version", version=f"neurocore {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_settings = sub.add_parser("settings", help="Print or write the default NeuroCore settings")
    p_settings.add_argument("--out", type=Path, default=None, help="Write default settings JSON to this path")
    p_settings.add_argument("--json", action="store_true")
    p_settings.set_defaults(func=cmd_settings)

    p_self = sub.add_parser("self-test", help="Run synthetic breakage and safety tests")
    p_self.add_argument("--settings", type=Path, default=None)
    p_self.add_argument("--json", action="store_true")
    p_self.set_defaults(func=cmd_self_test)

    p_demo = sub.add_parser("demo", help="Run the reference pipeline on synthetic EEG")
    p_demo.add_argument("--settings", type=Path, default=None)
    p_demo.add_argument("--seconds", type=float, default=3.0)
    p_demo.add_argument("--json", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    p_csv = sub.add_parser("run-csv", help="Run the reference pipeline on a CSV shaped as samples x channels")
    p_csv.add_argument("csv", type=Path)
    p_csv.add_argument("--settings", type=Path, default=None)
    p_csv.add_argument("--sampling-rate", type=float, default=None)
    p_csv.add_argument("--json", action="store_true")
    p_csv.set_defaults(func=cmd_run_csv)

    p_file = sub.add_parser("run-file", help="Run the reference pipeline on any supported EEG signal file")
    p_file.add_argument("path", type=Path)
    p_file.add_argument("--settings", type=Path, default=None)
    p_file.add_argument(
        "--sampling-rate", type=float, default=None, help="Required for raw numeric CSV/NPY when absent"
    )
    p_file.add_argument("--json", action="store_true")
    p_file.set_defaults(func=cmd_run_file)

    p_inventory = sub.add_parser("dataset-inventory", help="Inspect an EEG-DATA Japanese dataset inventory CSV")
    p_inventory.add_argument("csv", type=Path)
    p_inventory.add_argument("--limit", type=int, default=3, help="Number of sample records to include")
    p_inventory.add_argument("--json", action="store_true")
    p_inventory.set_defaults(func=cmd_dataset_inventory)

    p_acquisition = sub.add_parser("dataset-acquisition-plan", help="Plan provider-specific raw EEG acquisition")
    p_acquisition.add_argument("csv", type=Path)
    p_acquisition.add_argument("--limit", type=int, default=10, help="Number of sample plans to include")
    p_acquisition.add_argument("--json", action="store_true")
    p_acquisition.set_defaults(func=cmd_dataset_acquisition_plan)

    p_resolve = sub.add_parser("dataset-resolve-files", help="Resolve provider API file lists for EEG-DATA rows")
    p_resolve.add_argument("csv", type=Path)
    p_resolve.add_argument("--limit", type=int, default=25, help="Maximum records to resolve; use 0 for all selected rows")
    p_resolve.add_argument("--sample-limit", type=int, default=10)
    p_resolve.add_argument("--file-limit", type=int, default=50, help="Files to show per sample resolution; use 0 for all")
    p_resolve.add_argument("--provider", action="append", default=[], help="Provider filter; repeat for multiple providers")
    p_resolve.add_argument(
        "--status",
        action="append",
        default=[],
        help="Automation status filter; defaults to direct_api, tooling_required, and doi_resolution_required",
    )
    p_resolve.add_argument("--http-timeout", type=float, default=20.0)
    p_resolve.add_argument("--max-pages", type=int, default=30)
    p_resolve.add_argument("--json", action="store_true")
    p_resolve.set_defaults(func=cmd_dataset_resolve_files)

    p_route = sub.add_parser("route-intent", help="Route an externally decoded intent into a safe action envelope")
    p_route.add_argument("intent")
    p_route.add_argument("--confidence", type=float, default=1.0)
    p_route.add_argument("--payload", default="")
    p_route.add_argument("--settings", type=Path, default=None)
    p_route.add_argument("--json", action="store_true")
    p_route.set_defaults(func=cmd_route)

    p_simulate = sub.add_parser(
        "simulate-intents", help="Dry-run a JSON list of decoded intents and write an audit log"
    )
    p_simulate.add_argument("commands", type=Path, help="JSON file containing a list of intent command objects")
    p_simulate.add_argument("--out", type=Path, default=None, help="Optional JSONL audit log path")
    p_simulate.add_argument("--settings", type=Path, default=None)
    p_simulate.add_argument("--json", action="store_true")
    p_simulate.set_defaults(func=cmd_simulate)

    p_quality = sub.add_parser("quality", help="Score synthetic EEG signal quality for the current settings")
    p_quality.add_argument("--settings", type=Path, default=None)
    p_quality.add_argument("--seconds", type=float, default=3.0)
    p_quality.add_argument("--json", action="store_true")
    p_quality.set_defaults(func=cmd_quality)

    p_stream = sub.add_parser("stream-demo", help="Run a synthetic stream buffer demo and report emitted windows")
    p_stream.add_argument("--settings", type=Path, default=None)
    p_stream.add_argument("--seconds", type=float, default=3.0)
    p_stream.add_argument("--json", action="store_true")
    p_stream.set_defaults(func=cmd_stream_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
