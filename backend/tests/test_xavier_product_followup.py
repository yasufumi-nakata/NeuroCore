"""Multistep desktop and matched inference-ablation regression tests.

Synthetic in-memory images only. No real OS actions, EEG people, or GPU load.
"""
from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from neurocore.desktop import DesktopCapture, DesktopSessionManager
from neurocore.xavier.service import XavierService, XavierServiceError


MODEL_ID = "test-registered-model"
MANIFEST = "a" * 64
CHANNELS = [f"C{i:02d}" for i in range(30)]


def test_multistep_click_history_is_inference_compatible():
    img = Image.new("RGB", (120, 80), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    picture = DesktopCapture(buf.getvalue(), 120, 80, 120, 80)

    class Driver:
        clicks = []
        def capture(self):
            return picture
        def click(self, x, y):
            self.clicks.append((x, y))

    class ModelService:
        histories = []
        def status(self):
            return {"status": "loaded", "model_id": MODEL_ID, "model": {"manifest_sha256": MANIFEST}}
        def propose(self, payload):
            from neurocore.xavier.mlx_backend import sanitize_history
            self.histories.append(sanitize_history(payload["history"]))
            return {
                "schema": "neurocore-xavier-action-proposal-v1",
                "mode": "explicit_goal_engineering",
                "model_id": MODEL_ID,
                "manifest_sha256": MANIFEST,
                "eeg_positions": 5,
                "confidence": None,
                "confidence_calibrated": False,
                "action": {"action": "click", "x": 500, "y": 500},
                "executed": False,
                "input_condition": "present",
            }

    service, driver = ModelService(), Driver()
    manager = DesktopSessionManager(service, enabled=True, driver_factory=lambda: driver)
    sid = manager.start("Click two buttons")["session_id"]
    for expected in ([], [{"action": "click", "x": 500, "y": 500}]):
        manager.capture(sid, delay_seconds=0)
        proposed = manager.propose(sid, {"prefix": [], "channels": []})
        assert service.histories[-1] == expected
        applied = manager.apply(sid, proposed["proposal_id"], delay_seconds=0)
        assert applied["executed"] is True
        assert applied["goal_completion_verified"] is False
    assert driver.clicks == [(59, 39), (59, 39)]
    assert manager.history()["state"] == "active"


def test_finish_claim_does_not_claim_independent_success():
    img = Image.new("RGB", (120, 80), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    class Driver:
        def capture(self):
            return DesktopCapture(buf.getvalue(), 120, 80, 120, 80)
    class ModelService:
        def status(self):
            return {"status": "loaded", "model_id": MODEL_ID, "model": {"manifest_sha256": MANIFEST}}
        def propose(self, payload):
            return {
                "schema": "neurocore-xavier-action-proposal-v1",
                "mode": "explicit_goal_engineering", "model_id": MODEL_ID,
                "manifest_sha256": MANIFEST, "eeg_positions": 5,
                "confidence": None, "confidence_calibrated": False,
                "action": {"action": "finish"}, "executed": False,
            }
    manager = DesktopSessionManager(ModelService(), enabled=True, driver_factory=Driver)
    sid = manager.start("Test")["session_id"]
    manager.capture(sid, delay_seconds=0)
    proposal = manager.propose(sid, {"prefix": [], "channels": []})
    result = manager.apply(sid, proposal["proposal_id"], delay_seconds=0)
    assert result["state"] == "finished"
    assert result["goal_completion_verified"] is False
    assert result["result"] == "model_claimed_finish_unverified"
    assert result["executed"] is False


def test_inference_ablation_removes_eeg_positions_without_changing_model_or_screen():
    class Stub:
        def __init__(self):
            self.inputs = []
        def propose_action(self, prefix, image, history, goal, **kwargs):
            self.inputs.append((prefix, image, history, goal, kwargs))
            return {"action": {"action": "abstain"}, "eeg_positions": 0 if prefix is None else 5}
        def status(self):
            return {}
    class Registry:
        pass
    service = XavierService(Registry())
    service._model = Stub()
    service._phase = "loaded"
    service._model_id = MODEL_ID
    img = Image.new("RGB", (32, 32), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    common = {"prefix": [[0.] * 30 for _ in range(300)], "channels": CHANNELS,
              "image_png_base64": base64.b64encode(buf.getvalue()).decode(),
              "history": [], "goal": "Click blue circle"}
    with_eeg = service.propose(common)
    without_eeg = service.propose({**common, "eeg_condition": "no_eeg"})
    assert with_eeg["input_condition"] == "present"
    assert with_eeg["eeg_positions"] == 5
    assert without_eeg["input_condition"] == "no_eeg"
    assert without_eeg["eeg_positions"] == 0
    assert service._model.inputs[0][0] is not None
    assert service._model.inputs[1][0] is None
    assert service._model.inputs[0][1:] == service._model.inputs[1][1:]
    with pytest.raises(XavierServiceError) as error:
        service.propose({**common, "eeg_condition": "bad"})
    assert error.value.code == "invalid_input"
    assert len(service._model.inputs) == 2
