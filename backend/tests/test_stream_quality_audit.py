from __future__ import annotations

import json

from neurocore.audit import ActionAuditLog, DryRunActionSink
from neurocore.control import ControlRouter, IntentCommand
from neurocore.frame import Channel
from neurocore.quality import score_signal_quality
from neurocore.settings import NeuroCoreSettings
from neurocore.stream import StreamBuffer
from neurocore.synthetic import synthetic_eeg_frame


def test_stream_buffer_emits_overlapping_windows() -> None:
    frame = synthetic_eeg_frame(seconds=2, sampling_rate=100, channels=("Fz", "Cz"))
    buffer = StreamBuffer(
        channels=(Channel("Fz"), Channel("Cz")),
        sampling_rate=100,
        window_seconds=1.0,
        step_seconds=0.5,
    )

    windows = []
    for start in range(0, frame.samples, 25):
        windows.extend(buffer.append(frame.data[start : start + 25]))

    assert len(windows) == 3
    assert [window.start_sample for window in windows] == [0, 50, 100]
    assert all(window.frame.samples == 100 for window in windows)


def test_signal_quality_flags_flat_channel() -> None:
    frame = synthetic_eeg_frame(seconds=2, sampling_rate=100, channels=("Fz", "Cz"))
    broken = frame.with_data(frame.data.copy())
    broken.data[:, 1] = 0

    report = score_signal_quality(broken)

    assert report.status == "warning"
    assert "Cz" in report.flat_channels


def test_dry_run_sink_records_blocked_and_allowed_actions(tmp_path) -> None:
    armed_settings = NeuroCoreSettings.default().to_dict()
    armed_settings["safety"]["human_armed"] = True
    router = ControlRouter(NeuroCoreSettings.from_dict(armed_settings))
    unarmed_router = ControlRouter()
    audit = ActionAuditLog()
    sink = DryRunActionSink(audit)

    allowed = IntentCommand("select", 0.99)
    blocked = IntentCommand("select", 0.1)
    unarmed = IntentCommand("select", 0.99)
    sink.submit(router.route(allowed), allowed)
    sink.submit(router.route(blocked), blocked)
    sink.submit(unarmed_router.route(unarmed), unarmed)

    out = tmp_path / "audit.jsonl"
    audit.write_jsonl(out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 3
    assert rows[0]["note"] == "dry_run_only"
    assert rows[1]["action"]["blocked"] is True
    assert rows[2]["note"] == "blocked"
    assert rows[2]["action"]["reason"] == "human_arm_required"
