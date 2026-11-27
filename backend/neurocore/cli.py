from __future__ import annotations

import argparse
import json
import hashlib
import os
import zipfile
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .acquisition import plan_inventory_acquisition, summarize_acquisition_plans
from .datasets import load_eeg_dataset_inventory
from .audit import ActionAuditLog, DryRunActionSink
from . import __version__
from .control import ControlRouter, IntentCommand
from .exercise import exercise_dataset_records, summarize_dataset_exercise
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
        offset=args.offset,
        providers=set(args.provider) if args.provider else None,
        automation_statuses=(
            set(args.status)
            if args.status
            else {"direct_api", "tooling_required", "doi_resolution_required", "manual_review", "account_required"}
        ),
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


def cmd_dataset_exercise(args: argparse.Namespace) -> int:
    inventory = load_eeg_dataset_inventory(args.csv)
    plans = plan_inventory_acquisition(inventory)
    resolutions = ()
    if args.resolve_remote_files:
        resolutions = resolve_inventory_remote_files(
            plans,
            limit=None if args.resolve_limit == 0 else args.resolve_limit,
            offset=args.resolve_offset,
            providers=set(args.provider) if args.provider else None,
            automation_statuses=set(args.status) if args.status else {"direct_api", "tooling_required"},
            timeout=args.http_timeout,
            max_pages=args.max_pages,
        )
    exercise_records = exercise_dataset_records(plans, remote_resolutions=resolutions)
    payload = summarize_dataset_exercise(exercise_records)
    payload["remote_file_resolution_summary"] = summarize_remote_file_resolutions(resolutions)
    payload["sample_records"] = [record.to_dict() for record in exercise_records[: args.sample_limit]]
    _print(payload, json_mode=args.json)
    return 0 if payload["error_count"] == 0 else 1


def cmd_route(args: argparse.Namespace) -> int:
    settings = _control_settings(args)
    payload = json.loads(args.payload) if args.payload else {}
    command = IntentCommand(args.intent, args.confidence, source="cli", payload=payload)
    action = ControlRouter(settings).route(command)
    _print(action.to_dict(), json_mode=args.json)
    return 0 if not action.blocked else 1


def cmd_simulate(args: argparse.Namespace) -> int:
    settings = _control_settings(args)
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


def _control_settings(args: argparse.Namespace) -> NeuroCoreSettings:
    settings = NeuroCoreSettings.load(args.settings) if args.settings else NeuroCoreSettings.default()
    if not getattr(args, "human_armed", False):
        return settings
    payload = settings.to_dict()
    payload["safety"]["human_armed"] = True
    return NeuroCoreSettings.from_dict(payload)


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


def _xavier_write_new_json(path: Path, payload: dict) -> None:
    data = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path.expanduser(), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def _xavier_validate_npy_header(stream) -> None:
    import numpy as np
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Unsupported NPY format")
    if shape != (25000, 34) or dtype.kind not in "iuf" or dtype.itemsize > 8:
        raise ValueError("Native array header does not match the bounded source contract")


def cmd_xavier_prepare(args: argparse.Namespace) -> int:
    try:
        import numpy as np
        from .xavier.preprocessing import prepare_native_prefix
        path = args.input.expanduser()
        if path.stat().st_size > 32 * 1024 * 1024 or path.suffix.lower() not in {".npy", ".npz"}:
            raise ValueError("Unsupported or oversized input")
        if path.suffix.lower() == ".npz":
            with zipfile.ZipFile(path) as archive:
                files = archive.infolist()
                if len(files) != 1 or files[0].filename != "native_uv.npy" or files[0].file_size > 16 * 1024 * 1024:
                    raise ValueError("NPZ requires only a bounded native_uv array")
                with archive.open(files[0]) as stream:
                    _xavier_validate_npy_header(stream)
            with np.load(path, allow_pickle=False) as archive:
                native_uv = archive["native_uv"]
        else:
            with path.open("rb") as stream:
                _xavier_validate_npy_header(stream)
            native_uv = np.load(path, allow_pickle=False)
        if args.channels.stat().st_size > 8192:
            raise ValueError("Channel metadata too large")
        channels = json.loads(args.channels.read_text(encoding="utf-8"))
        result = prepare_native_prefix(native_uv, channels=channels, sample_rate_hz=10000, units="uV")
        _xavier_write_new_json(args.output, result)
        output = {"status": "prepared", "output": str(args.output), "samples": len(result["prefix"]),
                  "channels": len(result["channels"]), "scientific_validation": "not_established"}
        code = 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, zipfile.BadZipFile):
        output = {"status": "error", "error": {"code": "prepare_failed",
                  "message": "Check native input, ordered channel metadata, and a new output path"}}
        code = 1
    _print(output, json_mode=args.json)
    return code


def cmd_xavier_configure(args: argparse.Namespace) -> int:
    try:
        from .xavier.registry import ModelRegistry
        from .xavier import registry as registry_module
        catalog = ModelRegistry.from_file(Path(registry_module.__file__).with_name("packagedcatalog.json"))
        models = catalog.list()
        selected = args.model_id or (models[0]["model_id"] if len(models) == 1 else "")
        entry = catalog.get(selected)
        base = args.base_model.expanduser().resolve(strict=True)
        if not base.is_dir():
            raise ValueError("Base model directory required")
        manifest = entry.package_dir / "manifest.json"
        if manifest.stat().st_size > 1024 * 1024:
            raise ValueError("Manifest too large")
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if digest != entry.manifest_sha256:
            raise ValueError("Packaged manifest mismatch")
        body = {"schema": "neurocore-xavier-registry-v1", "models": [{"id": entry.model_id, "title": entry.title,
                "package_dir": str(entry.package_dir), "base_model_dir": str(base), "manifest_sha256": digest}]}
        _xavier_write_new_json(args.output, body)
        output = {"status": "configured", "registry": str(args.output.resolve()), "model_id": entry.model_id,
                  "manifest_sha256": digest, "activation": "Set XAVIER_MODEL_REGISTRY before starting the API"}
        code = 0
    except (OSError, ValueError, TypeError, KeyError):
        output = {"status": "error", "error": {"code": "configure_failed",
                  "message": "Check the bundled model, base directory, and a new registry output path"}}
        code = 1
    _print(output, json_mode=args.json)
    return code


class _XavierNoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _xavier_http_request(url: str, method: str, payload: dict | None, timeout: float) -> dict:
    body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": "application/json"})
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _XavierNoRedirect())
        with opener.open(request, timeout=timeout) as response:
            body = response.read(8 * 1024 * 1024 + 1)
            if len(body) > 8 * 1024 * 1024:
                raise ValueError("API response exceeded the bound")
            result = json.loads(body)
            if not isinstance(result, dict):
                raise ValueError("API response must be an object")
            return result
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read(65536)).get("detail", {})
        except (ValueError, AttributeError):
            detail = {}
        if not isinstance(detail, dict):
            detail = {}
        return {"status": "error", "error": {"code": detail.get("code", "http_error"),
                                                "message": detail.get("message", "Xavier API rejected the request"),
                                                "http_status": exc.code}}


def cmd_xavier(args: argparse.Namespace) -> int:
    """Operate the persistent local API process rather than losing weights on CLI exit."""
    try:
        parsed = urllib.parse.urlsplit(args.api_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or (
            parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}
        ):
            raise ValueError("Xavier CLI requires a loopback HTTP API URL (use an SSH tunnel for a remote node)")
        if not 0 < args.timeout <= 600:
            raise ValueError("timeout must be greater than zero and at most 600 seconds")
        action, payload, method = args.xavier_action, None, "GET"
        if action == "list":
            path = "/models"
        elif action == "inspect":
            path = "/models/" + urllib.parse.quote(args.model_id, safe="")
        elif action == "status":
            path = "/status"
        elif action in {"load", "unload"}:
            path, method = "/" + action, "POST"
            payload = {"model_id": args.model_id} if action == "load" else {}
        else:
            path, method = "/" + action, "POST"
            if args.input == "-":
                source = sys.stdin.read(8 * 1024 * 1024 + 1)
            else:
                with Path(args.input).expanduser().open("r", encoding="utf-8") as stream:
                    source = stream.read(8 * 1024 * 1024 + 1)
            if len(source.encode("utf-8")) > 8 * 1024 * 1024:
                raise ValueError("Input JSON is limited to 8 MiB")
            payload = json.loads(source)
            if not isinstance(payload, dict):
                raise ValueError("Input JSON must be an object")
        result = _xavier_http_request(args.api_url.rstrip("/") + "/api/xavier" + path, method, payload, args.timeout)
    except (OSError, ValueError, urllib.error.URLError):
        result = {"status": "error", "error": {"code": "client_error",
                    "message": "Check the loopback API, command options, and input JSON; no local model was started"}}
    _print(result, json_mode=args.json)
    return 1 if result.get("status") == "error" else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurocore", description="NeuroCore EEG-to-control runtime")
    parser.add_argument("--version", action="version", version=f"neurocore {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Start the Japanese application and its local API")
    p_serve.add_argument("--port", type=int, default=8010)
    p_serve.add_argument("--registry", type=Path, help="Trusted local model registry (model server only)")
    p_serve.add_argument("--remote-model", help="Loopback HTTP URL of an SSH tunnel to the SMAC model server")
    p_serve.add_argument("--desktop", action="store_true", help="Enable explicit screen capture and one-action application on this computer")
    p_serve.add_argument("--api-only", action="store_true", help="Run without bundled UI assets")
    p_serve.set_defaults(func=lambda args: __import__("neurocore.product", fromlist=["cmd_serve"]).cmd_serve(args))

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
    p_resolve.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many selected resolvable records before querying provider APIs",
    )
    p_resolve.add_argument("--sample-limit", type=int, default=10)
    p_resolve.add_argument("--file-limit", type=int, default=50, help="Files to show per sample resolution; use 0 for all")
    p_resolve.add_argument("--provider", action="append", default=[], help="Provider filter; repeat for multiple providers")
    p_resolve.add_argument(
        "--status",
        action="append",
        default=[],
        help="Automation status filter; defaults to every resolvable status, including manual/account rows with public candidates",
    )
    p_resolve.add_argument("--http-timeout", type=float, default=20.0)
    p_resolve.add_argument("--max-pages", type=int, default=30)
    p_resolve.add_argument("--json", action="store_true")
    p_resolve.set_defaults(func=cmd_dataset_resolve_files)

    p_exercise = sub.add_parser(
        "dataset-exercise",
        help="Classify every EEG-DATA row by raw-readiness state and optional remote resolution evidence",
    )
    p_exercise.add_argument("csv", type=Path)
    p_exercise.add_argument("--resolve-remote-files", action="store_true")
    p_exercise.add_argument("--resolve-limit", type=int, default=25, help="Maximum records to resolve; use 0 for all")
    p_exercise.add_argument(
        "--resolve-offset",
        type=int,
        default=0,
        help="Skip this many selected resolvable records before querying provider APIs",
    )
    p_exercise.add_argument("--sample-limit", type=int, default=10)
    p_exercise.add_argument("--provider", action="append", default=[], help="Provider filter for remote resolution")
    p_exercise.add_argument("--status", action="append", default=[], help="Automation status filter for remote resolution")
    p_exercise.add_argument("--http-timeout", type=float, default=20.0)
    p_exercise.add_argument("--max-pages", type=int, default=30)
    p_exercise.add_argument("--json", action="store_true")
    p_exercise.set_defaults(func=cmd_dataset_exercise)

    p_route = sub.add_parser("route-intent", help="Route an externally decoded intent into a safe action envelope")
    p_route.add_argument("intent")
    p_route.add_argument("--confidence", type=float, default=1.0)
    p_route.add_argument("--payload", default="")
    p_route.add_argument("--settings", type=Path, default=None)
    p_route.add_argument("--human-armed", action="store_true", help="Mark trusted operator human-arm state as armed")
    p_route.add_argument("--json", action="store_true")
    p_route.set_defaults(func=cmd_route)

    p_simulate = sub.add_parser(
        "simulate-intents", help="Dry-run a JSON list of decoded intents and write an audit log"
    )
    p_simulate.add_argument("commands", type=Path, help="JSON file containing a list of intent command objects")
    p_simulate.add_argument("--out", type=Path, default=None, help="Optional JSONL audit log path")
    p_simulate.add_argument("--settings", type=Path, default=None)
    p_simulate.add_argument("--human-armed", action="store_true", help="Mark trusted operator human-arm state as armed")
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
    p_xavier = sub.add_parser("xavier", help="Use the persistent local Xavier inference API")
    xavier_sub = p_xavier.add_subparsers(dest="xavier_action", required=True)
    prepare = xavier_sub.add_parser("prepare", help="Prepare a native source-compatible EEG prefix without loading a model")
    prepare.add_argument("--input", type=Path, required=True, help="Numeric NPY or NPZ with native_uv only; pickle is forbidden")
    prepare.add_argument("--channels", type=Path, required=True, help="JSON ordered native channel names")
    prepare.add_argument("--output", type=Path, required=True, help="New private request JSON path; existing files are preserved")
    prepare.add_argument("--json", action="store_true")
    prepare.set_defaults(func=cmd_xavier_prepare)
    configure = xavier_sub.add_parser("configure", help="Write a registry for bundled weights and an existing base-model cache")
    configure.add_argument("--base-model", type=Path, required=True)
    configure.add_argument("--output", type=Path, required=True)
    configure.add_argument("--model-id", default=None)
    configure.add_argument("--json", action="store_true")
    configure.set_defaults(func=cmd_xavier_configure)
    for action in ("list", "inspect", "status", "load", "forecast", "propose", "unload"):
        item = xavier_sub.add_parser(action)
        item.add_argument("--api-url", default="http://127.0.0.1:8010", help="Persistent local API address")
        item.add_argument("--timeout", type=float, default=120.0)
        item.add_argument("--json", action="store_true")
        if action in {"inspect", "load"}:
            item.add_argument("model_id", help="ID from the server's fixed registry")
        if action in {"forecast", "propose"}:
            item.add_argument("--input", required=True, help="Request JSON file, or - for standard input")
        item.set_defaults(func=cmd_xavier)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
