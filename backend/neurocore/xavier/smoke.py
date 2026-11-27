"""Finite public-synthetic startup/reload smoke; run inside the node's owner guard.

This command neither acquires nor replaces the cluster coordinator. Its caller
must hold the approved node GPU lease and enforce the external process deadline.
No EEG recordings, training batches, optimizer, or private checkpoint metadata
are opened. The exported inference package is read without modification.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np

from .model import XavierModel, strict_json


def public_prefix():
    # Exact basic arithmetic: no RNG/version-dependent synthetic generator.
    return ((np.arange(9000, dtype=np.float32).reshape(300, 30) % np.float32(29))
            / np.float32(10) - np.float32(1.4)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--proposal", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("smoke output must be new")
    started = time.monotonic()
    model = XavierModel(args.package, args.base_model, args.manifest_sha256)
    channels = model.manifest["input"]["channels"]
    prefix = public_prefix()
    report = {"schema": "neurocore-xavier-software-smoke-v1", "pid": os.getpid(),
              "manifest_sha256": model.manifest_sha256, "input_kind": "public_synthetic",
              "input_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
              "real_eeg_windows": 0, "gradient_updates": 0, "os_actions_executed": 0,
              "independent_evaluation": False, "scientific_level_certified": False}
    try:
        before = time.monotonic()
        model.load()
        report["load_seconds"] = time.monotonic() - before
        report["device"] = model.status()["device"]
        report["physical_gpu_utilization_percent_measured"] = False
        first = model.forecast(prefix, channels=channels)
        second = model.forecast(prefix, channels=channels)
        a = np.asarray(first["prediction_uv"], dtype=np.float32)
        b = np.asarray(second["prediction_uv"], dtype=np.float32)
        if not np.array_equal(a, b):
            raise RuntimeError("repeat prediction differs")
        report["prediction_sha256"] = hashlib.sha256(a.tobytes()).hexdigest()
        report["repeat_prediction_equal"] = True
        report["forecast_calls"] = 2
        report["separate_pid_reload_equal"] = None
        if args.compare:
            old = strict_json(args.compare.read_text())
            if (old["schema"] != report["schema"] or old["pid"] == report["pid"]
                    or old["manifest_sha256"] != report["manifest_sha256"]
                    or old["input_sha256"] != report["input_sha256"]
                    or old["prediction_sha256"] != report["prediction_sha256"]
                    or old["status"] != "passed"):
                raise RuntimeError("separate-process reload comparison failed")
            report["separate_pid_reload_equal"] = True
        if args.proposal:
            from PIL import Image, ImageDraw
            picture = Image.new("RGB", (1000, 700), "white")
            draw = ImageDraw.Draw(picture)
            draw.rectangle((100, 230, 430, 470), outline="black", width=6)
            draw.rectangle((570, 230, 900, 470), outline="black", width=6)
            draw.ellipse((200, 290, 330, 420), fill="blue")
            draw.rectangle((670, 290, 800, 420), fill="orange")
            data = io.BytesIO()
            picture.save(data, format="PNG")
            proposal = model.propose_action(prefix, data.getvalue(), [], "Click the button containing a blue circle.",
                                            channels=channels, max_tokens=64, timeout_seconds=120)
            from .actions import validate_proposal
            validate_proposal(proposal)
            report["proposal"] = proposal
            report["proposal_diagnostics"] = model.status()["last_inference"]
        report["status"] = "passed"
    finally:
        model.unload()
    report["seconds"] = time.monotonic() - started
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
