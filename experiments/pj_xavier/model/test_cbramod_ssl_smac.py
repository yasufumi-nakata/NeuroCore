"""Synthetic CPU-only checks; no public EEG or model downloads."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import cbramod_ssl_smac as smac


def fixture(root, channels=16, patches=3):
    rows, artifacts, sources = [], {}, {}
    for i, split in enumerate(("train", "val")):
        name = split + ".npy"
        data = np.random.default_rng(i).normal(0, .01, (3, channels, patches, 200)).astype(np.float32)
        np.save(root / name, data)
        source = str(i + 1) * 64; payload = str(i + 3) * 64
        sources[source] = dict(split=split, subject_group="synthetic-" + split,
                               eeg_payload_sha256=payload, montage_key="synthetic-montage",
                               physical_unit_verified=True, fit_allowed=split == "train", validation_allowed=split == "val")
        artifacts[name] = dict(shape=list(data.shape), dtype="float32", bytes=(root / name).stat().st_size, sha256=smac.sha256(root / name))
        for j in range(3):
            rows.append(dict(id=f"synthetic-{split}-{j}", split=split, subject_group="synthetic-" + split,
                             source_sha256=source, eeg_payload_sha256=payload, source_recording_id="synthetic-" + split,
                             window_start_s=2 + j * patches, window_end_s=2 + (j + 1) * patches,
                             channels=[f"C{k}" for k in range(channels)], sample_rate_hz=200, unit="V_x_1e4",
                             eeg_path=name, eeg_index=j, montage_key="synthetic-montage", reference="synthetic-reference",
                             native_sample_rate_hz=512, physical_unit_verified=True,
                             online_harness_allowed=False, source_causality="synthetic_fixture"))
    admission = dict(sources=sources, excluded_source_sha256=[], excluded_payload_sha256=[])
    save(root, rows, artifacts, admission)
    return rows, artifacts, admission


def save(root, rows, artifacts, admission):
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "artifacts.json").write_text(json.dumps(artifacts))
    (root / "source_admission.json").write_text(json.dumps(admission))


def load(root):
    return smac.load_corpus(root / "manifest.jsonl", root / "artifacts.json", root / "source_admission.json")


class TinyMasked(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.proj = torch.nn.Linear(200, 200)

    def forward(self, x, mask=None):
        return self.proj(x.masked_fill(mask[..., None], 0.))


class SmacSSLTests(unittest.TestCase):
    def test_variable_channels_patches_and_no_montage_mixing(self):
        for channels, patches in ((14, 3), (16, 3), (27, 5), (32, 30), (64, 3)):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp); fixture(root, channels, patches)
                rows, _, _, _ = load(root)
                self.assertEqual(tuple(smac.Store().get(rows[0]).shape), (channels, patches, 200))
                other = copy.deepcopy(rows[0]); other["reference"] = "different"
                batches = smac.epoch_batches([rows[0], other], 5, 4, 0)
                self.assertEqual(sorted(map(len, batches)), [1, 1])

    def test_changed_array_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            with (root / "train.npy").open("r+b") as f:
                f.seek(-1, 2); f.write(b"X")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                load(root)

    def test_cross_partition_payload_and_parent_group_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); rows, artifacts, admission = fixture(root)
            for row in rows[3:]:
                row["eeg_payload_sha256"] = rows[0]["eeg_payload_sha256"]
            admission["sources"][rows[3]["source_sha256"]]["eeg_payload_sha256"] = rows[0]["eeg_payload_sha256"]
            save(root, rows, artifacts, admission)
            with self.assertRaises(ValueError):
                load(root)
            with self.assertRaisesRegex(ValueError, "changes partition"):
                smac.check_history([rows[0]], {"group": {rows[0]["subject_group"]: "test"}})

    def test_test_source_and_unverified_units_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); rows, artifacts, admission = fixture(root)
            admission["excluded_source_sha256"] = [rows[0]["source_sha256"]]
            save(root, rows, artifacts, admission)
            with self.assertRaisesRegex(ValueError, "Heldout"):
                load(root)
            admission["excluded_source_sha256"] = []
            admission["sources"][rows[0]["source_sha256"]]["physical_unit_verified"] = False
            save(root, rows, artifacts, admission)
            with self.assertRaisesRegex(ValueError, "Physical volts"):
                load(root)

    def test_missing_provenance_slot_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); rows, artifacts, admission = fixture(root)
            save(root, rows[1:], artifacts, admission)
            with self.assertRaisesRegex(ValueError, "Every prepared array slot"):
                load(root)

    def test_unadmitted_fit_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); rows, artifacts, admission = fixture(root)
            admission["sources"][rows[0]["source_sha256"]]["fit_allowed"] = False
            save(root, rows, artifacts, admission)
            with self.assertRaisesRegex(ValueError, "Train admission"):
                load(root)
            admission["sources"][rows[0]["source_sha256"]]["fit_allowed"] = True
            rows[0]["eeg_path"] = "../bad.npy"
            save(root, rows, artifacts, admission)
            with self.assertRaisesRegex(ValueError, "Bounded relative"):
                load(root)

    def test_workstation_and_excluded_node_cannot_execute(self):
        protocol = {"training": {"device": "mps"}}
        for host in ("workstation.local", "vpnclient231.sfc.keio.ac.jp", "smac19"):
            with patch.object(smac.platform, "node", return_value=host), self.assertRaisesRegex(ValueError, "restricted to smac"):
                smac.environment(protocol)

    def test_dynamic_mask_is_finite_with_nonzero_backward(self):
        model = TinyMasked()
        for ch, pa in ((14, 3), (16, 5), (27, 3), (32, 3), (64, 3)):
            x = torch.randn(2, ch, pa, 200) * .01
            mask = smac.legacy.random_mask(2, ch, pa, device=torch.device("cpu"))
            model.zero_grad()
            loss = smac.legacy.masked_loss(model(x, mask), x, mask)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(model.proj.bias.grad.norm()), 0)

    def test_cpu_fixture_probe_checkpoint_and_real_duration_accounting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            common = ["--manifest", str(root / "manifest.jsonl"), "--artifacts", str(root / "artifacts.json"),
                      "--admission", str(root / "source_admission.json"), "--protocol", str(root / "protocol.json")]
            registration = smac.parser().parse_args(["register", *common, "--device", "cpu", "--epochs", "1", "--max-seconds", "30", "--batch-candidates", "1", "--probe-steps-per-shape", "3"])
            smac.register(registration)
            probe = smac.parser().parse_args(["probe", *common, "--output", str(root / "probe"), "--cbramod-upstream", "unused", "--cbramod-weights", "unused", "--batch-size", "1"])
            with patch.object(smac, "environment", return_value=(torch.device("cpu"), {"kind": "synthetic-test"})), patch.object(smac, "initialize", side_effect=lambda *args: TinyMasked()):
                result = smac.execute(probe)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["unique_training_seconds"], 9)
                self.assertEqual(result["global_step"], 3)
                train = smac.parser().parse_args(["train", *common, "--output", str(root / "run"), "--cbramod-upstream", "unused", "--cbramod-weights", "unused", "--capacity-reports", str(root / "probe/run_summary.json")])
                trained = smac.execute(train)
                self.assertEqual(trained["global_step"], 3)
                self.assertEqual(trained["unique_training_seconds"], 9)
                self.assertIsNotNone(trained["best_checkpoint_sha256"])
                state = torch.load(root / "run/best.pt", map_location="cpu", weights_only=True)
                self.assertEqual(state["membership_sha256"], trained["membership_sha256"])
                self.assertEqual(state["kind"], "train")


if __name__ == "__main__":
    unittest.main()
