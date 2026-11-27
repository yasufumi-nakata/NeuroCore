from __future__ import annotations

import os
import time
from typing import Any

from . import __version__
from .audit import ActionAuditLog, DryRunActionSink
from .control import ControlRouter, IntentCommand
from .frame import Channel
from .selftest import run_self_tests
from .settings import NeuroCoreSettings
from .stream import StreamBuffer
from .synthetic import synthetic_eeg_frame
from .kernels import Bandpass, ReReference, Resample, SpectralFeatures, ValidateEEG
from .pipeline import Pipeline, PipelineExecutionError
from .quality import score_signal_quality
from .xavier.service import XavierService, XavierServiceError

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
except ImportError as exc:  # pragma: no cover - exercised only when server extra is missing
    raise RuntimeError("Install neurocore[server] to use neurocore.api") from exc


app = FastAPI(
    title="NeuroCore API",
    version=__version__,
    description="EEG-to-control runtime API for settings, synthetic pipeline checks, and safety self-tests.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "neurocore", "version": __version__}


@app.get("/api/settings/default")
def default_settings() -> dict[str, Any]:
    settings = NeuroCoreSettings.default()
    return {"settings": settings.to_dict(), "issues": settings.validate()}


@app.post("/api/settings/validate")
def validate_settings(payload: dict[str, Any]) -> dict[str, Any]:
    settings = NeuroCoreSettings.from_dict(payload.get("settings", payload))
    return {"settings": settings.to_dict(), "issues": settings.validate()}


@app.post("/api/self-test")
def self_test(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = _settings_from_payload(payload or {})
    return run_self_tests(settings).to_dict()


@app.post("/api/pipeline/demo")
def pipeline_demo(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = _settings_from_payload(payload or {})
    frame = synthetic_eeg_frame(
        seconds=float((payload or {}).get("seconds", 3.0)),
        sampling_rate=settings.device.sampling_rate,
        channels=tuple(settings.device.channels),
    )
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
        return pipeline.run(frame).to_dict()
    except PipelineExecutionError as exc:
        raise HTTPException(status_code=422, detail=exc.report.to_dict() if exc.report else str(exc)) from exc


@app.post("/api/signal/quality")
def signal_quality(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = _settings_from_payload(payload or {})
    frame = synthetic_eeg_frame(
        seconds=float((payload or {}).get("seconds", 3.0)),
        sampling_rate=settings.device.sampling_rate,
        channels=tuple(settings.device.channels),
    )
    return score_signal_quality(frame).to_dict()


@app.post("/api/stream/demo")
def stream_demo(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = _settings_from_payload(payload or {})
    seconds = float((payload or {}).get("seconds", 3.0))
    frame = synthetic_eeg_frame(seconds=seconds, sampling_rate=settings.device.sampling_rate, channels=tuple(settings.device.channels))
    buffer = StreamBuffer(
        channels=tuple(Channel(name=name) for name in settings.device.channels),
        sampling_rate=settings.device.sampling_rate,
        window_seconds=settings.signal.window_seconds,
        step_seconds=settings.signal.step_seconds,
        provenance={"source": "api_stream_demo"},
    )
    chunk_size = max(1, int(round(settings.device.sampling_rate * settings.signal.step_seconds)))
    emitted = []
    for start in range(0, frame.samples, chunk_size):
        emitted.extend(buffer.append(frame.data[start : start + chunk_size]))
    return {"window_count": len(emitted), "windows": [window.to_summary() for window in emitted[:10]]}


@app.post("/api/control/route")
def route_control(payload: dict[str, Any]) -> dict[str, Any]:
    settings = _settings_from_payload(payload)
    command_payload = payload.get("command", payload)
    command = IntentCommand(
        intent=command_payload["intent"],
        confidence=float(command_payload.get("confidence", 0.0)),
        source=command_payload.get("source", "api"),
        payload=dict(command_payload.get("payload", {})),
    )
    return ControlRouter(settings).route(command).to_dict()


@app.post("/api/control/simulate")
def simulate_control(payload: dict[str, Any]) -> dict[str, Any]:
    settings = _settings_from_payload(payload)
    router = ControlRouter(settings)
    audit = ActionAuditLog()
    sink = DryRunActionSink(audit)
    commands = payload.get("commands", [])
    if not isinstance(commands, list):
        raise HTTPException(status_code=422, detail="commands must be a list")
    for item in commands:
        command = IntentCommand(
            intent=item["intent"],
            confidence=float(item.get("confidence", 0.0)),
            source=item.get("source", "api"),
            payload=dict(item.get("payload", {})),
            timestamp=float(item.get("timestamp", time.time())),
        )
        action = router.route(command)
        sink.submit(action, command)
    return audit.to_dict()


def _settings_from_payload(payload: dict[str, Any]) -> NeuroCoreSettings:
    settings_payload = payload.get("settings")
    if settings_payload is None:
        return NeuroCoreSettings.default()
    return NeuroCoreSettings.from_dict(settings_payload)


# Captured once at API import/startup; requests cannot change registry paths.
if os.environ.get("NEUROCORE_REMOTE_MODEL"):
    from .xavier.remote import RemoteXavierService
    _xavier_service = RemoteXavierService(os.environ["NEUROCORE_REMOTE_MODEL"])
else:
    _xavier_service = XavierService.from_environment()


class _XavierBodyLimit:
    def __init__(self, app, max_bytes: int = 8 * 1024 * 1024) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith("/api/xavier/") or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            total += len(body)
            if total > self.max_bytes:
                payload = b'{"detail":{"code":"request_too_large","message":"Xavier requests are limited to 8 MiB"}}'
                await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": payload})
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


app.add_middleware(_XavierBodyLimit)


def _xavier_call(method, *args):
    try:
        return method(*args)
    except XavierServiceError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.to_dict()) from None


@app.get("/api/xavier/models")
def xavier_models() -> dict[str, Any]:
    return _xavier_call(_xavier_service.list_models)


@app.get("/api/xavier/models/{model_id}")
def xavier_inspect(model_id: str) -> dict[str, Any]:
    return _xavier_call(_xavier_service.inspect, model_id)


@app.get("/api/xavier/status")
def xavier_status() -> dict[str, Any]:
    return _xavier_call(_xavier_service.status)


@app.post("/api/xavier/load")
def xavier_load(payload: dict[str, Any]) -> dict[str, Any]:
    if set(payload) != {"model_id"} or not isinstance(payload["model_id"], str):
        raise HTTPException(status_code=422, detail={"code": "invalid_input", "message": "Only a registered model_id is accepted"})
    return _xavier_call(_xavier_service.load, payload["model_id"])


@app.post("/api/xavier/forecast")
def xavier_forecast(payload: dict[str, Any]) -> dict[str, Any]:
    return _xavier_call(_xavier_service.forecast, payload)


@app.post("/api/xavier/propose")
def xavier_propose(payload: dict[str, Any]) -> dict[str, Any]:
    return _xavier_call(_xavier_service.propose, payload)


@app.post("/api/xavier/unload")
def xavier_unload(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if payload:
        raise HTTPException(status_code=422, detail={"code": "invalid_input", "message": "unload takes no parameters"})
    return _xavier_call(_xavier_service.unload)


from .product import install_product_routes
install_product_routes(app, lambda: _xavier_service)
