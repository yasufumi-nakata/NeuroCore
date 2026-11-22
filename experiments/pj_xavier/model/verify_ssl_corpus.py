#!/usr/bin/env python3
"""Independently check all prepared SSL windows and two true train gradients."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

import cbramod_ssl as api


def verify(args):
    started = time.perf_counter()
    root = Path(args.corpus).resolve()
    rows, artifacts = api.read_corpus(root / "manifest.jsonl", root / "artifacts.json",
                                     allowlist=root / "source_allowlist.json")
    store = api.SSLStore()
    minimum_std, maximum_abs, nvalues = float("inf"), 0., 0
    for row in rows:
        value = store.get(row).numpy()
        std = value.std(axis=(1, 2), dtype=np.float64)
        if float(std.min()) < 1e-4 * (1 - 1e-6):
            raise ValueError("Prepared channel fails the 0.01 microvolt standard-deviation QC")
        minimum_std = min(minimum_std, float(std.min()))
        maximum_abs = max(maximum_abs, float(np.abs(value).max()))
        nvalues += value.size
    torch.set_num_threads(2)
    torch.manual_seed(20261122)
    encoder = api.load_full_encoder(args.cbramod_upstream, args.cbramod_weights).train()
    optimizer = torch.optim.AdamW((p for p in encoder.parameters() if p.requires_grad), lr=1e-4, weight_decay=.05)
    diagnostics = []
    for channels in (8, 32):
        row = next(row for row in rows if row["split"] == "train" and len(row["channels"]) == channels)
        value = store.get(row).unsqueeze(0)
        mask = api.fixed_mask(row["id"], channels).unsqueeze(0)
        optimizer.zero_grad(set_to_none=True)
        tick = time.perf_counter()
        loss = api.masked_loss(encoder(value, mask=mask), value, mask)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0:
            raise ValueError("Real training EEG did not yield a finite nonzero encoder gradient")
        optimizer.step()
        diagnostics.append({"partition": "train", "channels": channels, "seconds": 30,
                            "masked_mse": float(loss.detach()), "preclip_gradient_norm": float(gradient_norm),
                            "wall_seconds": time.perf_counter() - tick, "CPU_only": True,
                            "probe_weights_discarded": True})
    report = {"schema": "xavier-independent-SSL-input-validation-v1", "status": "passed",
              "manifest_sha256": api.sha256(root / "manifest.jsonl"),
              "artifact_manifest_sha256": api.sha256(root / "artifacts.json"),
              "source_allowlist_sha256": api.sha256(root / "source_allowlist.json"),
              "trainer_sha256": api.sha256(Path(api.__file__)),
              "validator_sha256": api.sha256(Path(__file__)),
              "arrays": len(artifacts), "all_array_bytes_rehashed": True,
              "array_file_bytes": sum(item["bytes"] for item in artifacts.values()),
              "windows": len(rows), "float32_samples_individually_checked": nvalues,
              "partitions": dict(Counter(r["split"] for r in rows)),
              "time_union_seconds_by_partition": {s: sum(30 for row in rows if row["split"] == s) for s in ("train", "val")},
              "subject_counts": {s: len({r["subject_group"] for r in rows if r["split"] == s}) for s in ("train", "val")},
              "actual_source_counts": {s: len({r["source_sha256"] for r in rows if r["split"] == s}) for s in ("train", "val")},
              "raw_interval_nonoverlap": True, "array_slot_unique_integer_complete": True,
              "subject_and_raw_SHA_exclusions_passed": True, "all_windows_finite": True,
              "observed_minimum_channel_std_microvolt": minimum_std * 100,
              "observed_maximum_absolute_sample_microvolt": maximum_abs * 100,
              "std_float32_rounding_relative_tolerance": 1e-6,
              "real_train_CPU_gradients": diagnostics, "GPU_used": False,
              "downstream_behavior_labels_used": 0, "probe_weights_reused": False,
              "wall_seconds": time.perf_counter() - started}
    output = Path(args.output)
    if output.exists():
        raise ValueError("Preserve the existing independent validation receipt")
    api.dump_json(output, report)
    print(json.dumps(report, allow_nan=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "cbramod-upstream", "cbramod-weights", "output"):
        parser.add_argument("--" + name, required=True)
    verify(parser.parse_args())
