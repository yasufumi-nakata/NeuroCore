from __future__ import annotations

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
