#!/usr/bin/env python3
"""CPU-only capacity probe of the official frozen CBraMod on real train EEG.

Uses only the latest one second inside an existing training window, at 200 Hz.
This is an encoder compatibility probe, not a trained decoder or an evaluation.
Upstream source and public weights must be acquired separately and pinned.
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np
from scipy.signal import resample_poly
import torch

UPSTREAM_COMMIT = "b9e961003214326972c567eff390e75b0287e32a"
WEIGHTS_REVISION = "500543c7e30bda1b22bfd51a49301b238dee21fd"
WEIGHTS_SHA256 = "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def latest_second(values, sample_rate):
    values = np.asarray(values)
    if sample_rate != 256 or values.ndim != 2 or values.shape[0] != 8 or values.shape[1] < 256:
        raise ValueError("probe requires the declared UniBa montage and 256 Hz")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite EEG")
    # Prepared MNE arrays are in volts. Upstream downstream loaders divide
    # microvolt inputs by 100, so volts * 1e6 / 100 is the declared scaling.
    result = resample_poly(values[:, -256:].astype(np.float32) * 1e4, 25, 32, axis=-1)
    if result.shape != (8, 200):
        raise AssertionError("unexpected resampling length")
    return np.ascontiguousarray(result[:, None, :], dtype=np.float32)


def run(upstream, weights, manifest, expected_manifest_sha, output):
    started = time.monotonic()
    upstream, weights, manifest, output = map(Path, (upstream, weights, manifest, output))
    if output.exists():
        raise ValueError("preserve prior probe output")
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT or digest(weights) != WEIGHTS_SHA256 or digest(manifest) != expected_manifest_sha:
        raise ValueError("fixed source, model, or data identity mismatch")
    required = ["models/__init__.py", "models/cbramod.py", "models/criss_cross_transformer.py"]
    source_hashes = {}
    for name in required:
        original = subprocess.check_output(["git", "-C", str(upstream), "show", commit + ":" + name])
        actual = (upstream / name).read_bytes()
        if actual != original:
            raise ValueError("upstream model source modified:" + name)
        source_hashes[name] = hashlib.sha256(actual).hexdigest()
    rows = []
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        if row["split"] == "train":
            rows.append(row)
        if len(rows) == 3:
            break
    if len(rows) != 3 or any(r.get("data_origin") != "real_recording" for r in rows):
        raise ValueError("three real training windows required")
    arrays, inputs = {}, []
    for row in rows:
        relative = Path(row["eeg_path"])
        path = manifest.parent / relative
        if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(manifest.parent.resolve()):
            raise ValueError("EEG path escapes manifest root")
        arrays[str(relative)] = digest(path)
        values = np.load(path, allow_pickle=False, mmap_mode="r")
        inputs.append(latest_second(values[row["eeg_index"]], row["sample_rate_hz"]))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    sys.path.insert(0, str(upstream.resolve()))
    from models.cbramod import CBraMod
    model = CBraMod().cpu().eval()
    weights_dict = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(weights_dict, strict=True)
    model.proj_out = torch.nn.Identity()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    data = torch.from_numpy(np.stack(inputs))
    tick = time.monotonic()
    with torch.inference_mode():
        features = model(data)
    inference_seconds = time.monotonic() - tick
    if features.shape != (3, 8, 1, 200) or not torch.isfinite(features).all():
        raise ValueError("invalid official encoder output")
    report = {"schema": "pj-xavier-cbramod-capacity-probe-v1", "status": "completed",
              "source_repo": "https://github.com/wjq-learning/CBraMod", "source_commit": commit,
              "source_sha256": source_hashes, "weights_repo": "weighting666/CBraMod",
              "weights_revision": WEIGHTS_REVISION, "weights_sha256": WEIGHTS_SHA256,
              "weights_bytes": weights.stat().st_size, "encoder_parameters": sum(p.numel() for p in model.parameters()),
              "manifest_sha256": expected_manifest_sha, "prepared_array_sha256": arrays,
              "train_windows": 3, "validation_windows": 0, "test_windows": 0,
              "input_shape": list(data.shape), "output_shape": list(features.shape),
              "input_preparation": {"original_unit": "V", "latest_seconds_only": 1,
                                    "input_sample_rate_hz": 256, "encoder_sample_rate_hz": 200,
                                    "scale_factor": 10000, "right_padding_samples": 0,
                                    "resample": "scipy.signal.resample_poly up=25 down=32 within existing past window"},
              "device": "cpu", "torch_threads": 2, "gpu_calls": 0,
              "inference_batch_seconds": inference_seconds, "elapsed_seconds": time.monotonic() - started,
              "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024),
              "torch_version": torch.__version__, "all_output_finite": True,
              "training_steps": 0, "decoder_performance_measured": False,
              "agentic_goal_validated": False,
              "next_use": "Frozen pretrained EEG encoder candidate for a separate declared training experiment; no replacement of active model or checkpoint.",
              "limitations": ["Three real training examples only; no performance comparison or backbone fusion was run.",
                              "Only the final one second of 1.75-second windows was used; a downstream protocol must fix this policy before comparison.",
                              "Official pretraining data provenance must be considered for each future benchmark."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(report, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--expected-manifest-sha256", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.upstream, a.weights, a.manifest, a.expected_manifest_sha256, a.output)


if __name__ == "__main__":
    main()
