import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import cbramod_ssl as ssl


class SSLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_masked_loss_excludes_unmasked_targets_and_gradients(self):
        output = torch.zeros(1, 8, 30, 200, requires_grad=True)
        mask = ssl.fixed_mask("record-1", 8).unsqueeze(0)
        target = torch.ones_like(output)
        target[~mask] = 10000
        loss = ssl.masked_loss(output, target, mask)
        self.assertEqual(float(loss.detach()), 1.)
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(output.grad[~mask])), 0)
        self.assertGreater(float(output.grad[mask].abs().sum()), 0.)

    def test_validation_mask_identity_and_exact_half(self):
        one = ssl.fixed_mask("first", 32)
        self.assertTrue(torch.equal(one, ssl.fixed_mask("first", 32)))
        self.assertFalse(torch.equal(one, ssl.fixed_mask("second", 32)))
        self.assertEqual(int(one.sum()), 480)
        actual = ssl.random_mask(3, 8, device=torch.device("cpu"))
        self.assertEqual(actual.sum((1, 2)).tolist(), [120, 120, 120])

    def test_channel_bucketing_and_full_epoch_coverage(self):
        rows = [{"channels": [str(i) for i in range(c)]} for c in [8] * 11 + [32] * 19]
        batches = ssl.epoch_batches(rows, 7, 17, 0)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(30)))
        self.assertTrue(all(len({len(rows[i]["channels"]) for i in batch}) == 1 for batch in batches))
        self.assertEqual(batches, ssl.epoch_batches(rows, 7, 17, 0))
        probe = ssl.probe_batches(rows, 16, 50, 17)
        self.assertEqual(len(probe), 50)
        self.assertEqual([len(rows[batch[0]]["channels"]) for batch in probe[-10:]], [32, 8] * 5)

    def fixture(self, root):
        root = Path(root)
        rows, artifacts, sources = [], {}, {}
        for split, index in (("train", 0), ("val", 1)):
            path = root / f"{split}.npy"
            data = np.linspace(-.2, .2, 8 * 30 * 200, dtype=np.float32).reshape(1, 8, 30, 200)
            np.save(path, data)
            row = {"id": split, "split": split, "subject_group": f"portable:{index}",
                   "source_sha256": str(index) * 64, "source_recording_id": f"record:{index}",
                   "window_start_s": 5., "window_end_s": 35., "channels": [f"C{i}" for i in range(8)],
                   "sample_rate_hz": 200, "unit": ssl.UNIT, "eeg_path": path.name, "eeg_index": 0}
            rows.append(row)
            artifacts[path.name] = {"sha256": ssl.sha256(path), "bytes": path.stat().st_size,
                                    "shape": list(data.shape), "dtype": "float32"}
            sources[row["source_sha256"]] = {k: row[k] for k in ("split", "subject_group", "source_recording_id")}
        paths = [root / name for name in ("manifest.jsonl", "artifacts.json", "allowlist.json")]
        paths[0].write_text("".join(json.dumps(r) + "\n" for r in rows))
        paths[1].write_text(json.dumps(artifacts))
        paths[2].write_text(json.dumps({"sources": sources, "excluded_source_sha256": ["a" * 64]}))
        return rows, paths

    def test_corpus_actual_bytes_units_and_source_allowlist(self):
        with tempfile.TemporaryDirectory() as root:
            _, paths = self.fixture(root)
            rows, _ = ssl.read_corpus(paths[0], paths[1], allowlist=paths[2])
            value = ssl.SSLStore().get(rows[0])
            self.assertEqual(tuple(value.shape), (8, 30, 200))
            self.assertAlmostEqual(float(value[0, 0, 0]), -.2)
            with (Path(root) / "train.npy").open("ab") as f:
                f.write(b"changed")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                ssl.read_corpus(paths[0], paths[1], allowlist=paths[2])

    def test_disguised_source_or_subject_and_overlap_rejected(self):
        changes = [lambda r: r[1].update(subject_group=r[0]["subject_group"]),
                   lambda r: r[1].update(source_sha256=r[0]["source_sha256"]),
                   lambda r: r[0].update(source_sha256="a" * 64)]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as root:
                rows, paths = self.fixture(root)
                change(rows)
                paths[0].write_text("".join(json.dumps(r) + "\n" for r in rows))
                with self.assertRaises(ValueError):
                    ssl.read_corpus(paths[0], paths[1], allowlist=paths[2])
        with tempfile.TemporaryDirectory() as root:
            rows, paths = self.fixture(root)
            array_path = Path(root) / "train.npy"
            value = np.load(array_path)
            value = np.concatenate([value, value], axis=0)
            np.save(array_path, value)
            seals = json.loads(paths[1].read_text())
            seals["train.npy"].update(sha256=ssl.sha256(array_path), bytes=array_path.stat().st_size, shape=list(value.shape))
            paths[1].write_text(json.dumps(seals))
            duplicate = {**rows[0], "id": "overlap", "eeg_index": 1, "window_start_s": 20, "window_end_s": 50}
            rows.append(duplicate)
            paths[0].write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                ssl.read_corpus(paths[0], paths[1])

    def test_no_behavior_or_external_test_rows(self):
        for change in ({"label": "keyboard"}, {"split": "test"}, {"subject_group": "aicoding:itu:P001"}, {"eeg_path": "../escape.npy"}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as root:
                rows, paths = self.fixture(root)
                rows[0].update(change)
                paths[0].write_text("".join(json.dumps(r) + "\n" for r in rows))
                with self.assertRaises(ValueError):
                    ssl.read_corpus(paths[0], paths[1])

    def test_array_slot_integer_uniqueness_full_coverage_and_channel_order(self):
        for index in (-1, .5, 0., True, 1, None):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as root:
                rows, paths = self.fixture(root)
                rows[0]["eeg_index"] = index
                paths[0].write_text("".join(json.dumps(r) + "\n" for r in rows))
                with self.assertRaisesRegex(ValueError, "eeg_index"):
                    ssl.read_corpus(paths[0], paths[1])
        with tempfile.TemporaryDirectory() as root:
            rows, paths = self.fixture(root)
            rows.append({**rows[0], "id": "false-unique", "window_start_s": 35, "window_end_s": 65})
            paths[0].write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "same prepared array slot"):
                ssl.read_corpus(paths[0], paths[1])
        for failure in ("missing-slot", "different-channels"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                rows, paths = self.fixture(root)
                array_path = Path(root) / "train.npy"
                value = np.load(array_path)
                value = np.concatenate([value, value], axis=0)
                np.save(array_path, value)
                seals = json.loads(paths[1].read_text())
                seals["train.npy"].update(sha256=ssl.sha256(array_path), bytes=array_path.stat().st_size, shape=list(value.shape))
                paths[1].write_text(json.dumps(seals))
                if failure == "different-channels":
                    rows.append({**rows[0], "id": "second", "eeg_index": 1, "window_start_s": 35, "window_end_s": 65,
                                 "channels": list(reversed(rows[0]["channels"]))})
                paths[0].write_text("".join(json.dumps(r) + "\n" for r in rows))
                with self.assertRaisesRegex(ValueError, "slot must|channel order"):
                    ssl.read_corpus(paths[0], paths[1])

    def test_checkpoint_rng_optimizer_and_next_update_roundtrip(self):
        torch.manual_seed(19)
        model = nn.Sequential(nn.Linear(200, 32), nn.Dropout(.1), nn.GELU(), nn.Linear(32, 200))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.randn(2, 8, 30, 200) * .05
        mask = ssl.random_mask(2, 8, device=torch.device("cpu"))
        ssl.masked_loss(model(x), x, mask).backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "state.pt"
            ssl.save_checkpoint(path, model, optimizer, {}, 2, 9, False, 29, 14., .1, 58, {"a", "b"})
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(state["next_batch"], 9)
            self.assertFalse(state["epoch_completed"])
            restored = copy.deepcopy(model)
            opt2 = torch.optim.AdamW(restored.parameters(), lr=1e-4)
            restored.load_state_dict(state["model"]); opt2.load_state_dict(state["optimizer"])
            outcomes = []
            for m, opt in ((model, optimizer), (restored, opt2)):
                torch.set_rng_state(state["torch_rng"])
                mask = ssl.random_mask(2, 8, device=torch.device("cpu"))
                loss = ssl.masked_loss(m(x), x, mask)
                loss.backward(); opt.step()
                outcomes.append((float(loss.detach()), {k: v.clone() for k, v in m.state_dict().items()}))
            self.assertEqual(outcomes[0][0], outcomes[1][0])
            for key in outcomes[0][1]:
                torch.testing.assert_close(outcomes[0][1][key], outcomes[1][1][key], rtol=0, atol=0)

    def test_resume_without_validation_improvement_seals_previous_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial_path = root / "initial.pth"
            torch.save({"weight": torch.tensor([1.])}, initial_path)
            initial_sha = ssl.sha256(initial_path)
            protocol = root / "protocol.json"; protocol.write_text('{}\n')
            allowlist = root / "allowlist.json"
            allowlist.write_text(json.dumps({"sources": {"a" * 64: {}, "b" * 64: {}}, "excluded_source_sha256": ["c" * 64]}))
            config = {"command": "train", "protocol": str(protocol), "protocol_sha256": ssl.sha256(protocol),
                      "model_source_sha256": ssl.sha256(Path(ssl.__file__)), "manifest_sha256": "d" * 64,
                      "artifacts_sha256": "e" * 64, "source_allowlist": str(allowlist),
                      "source_allowlist_sha256": ssl.sha256(allowlist), "batch_size": 8, "seed": 20261122,
                      "lr": 1e-4, "precision": "float32", "cbramod_weights": str(initial_path), "resume": None}
            best = {"schema": ssl.SCHEMA, "config": copy.deepcopy(config), "model": {"weight": torch.tensor([2.])},
                    "epoch": 0, "global_step": 12, "training_examples_presented": 96, "seen_training_ids": ["one"]}
            torch.save(best, root / "best.pt")
            torch.save(best["model"], root / "best_encoder_weights.pth")
            # A later attempt resumed but did not improve validation. Its attempt
            # path differs, while all sealed scientific settings remain identical.
            config["resume"] = str(root / "latest.pt")
            (root / "run_config.json").write_text(json.dumps(config))
            summary = {"status": "completed_epochs", "global_step": 20,
                       "best_checkpoint_sha256": ssl.sha256(root / "best.pt"),
                       "best_encoder_weights_sha256": ssl.sha256(root / "best_encoder_weights.pth"),
                       "best_validation_masked_mse": .1}
            (root / "run_summary.json").write_text(json.dumps(summary))
            (root / "training_membership.json").write_text(json.dumps({"source_sha256_by_partition": {"train": ["a" * 64], "val": ["b" * 64]}}))
            (root / "history.jsonl").write_text(json.dumps({"global_step": 12, "val": {"masked_mse": .1}}) + '\n' + json.dumps({"global_step": 20, "val": {"masked_mse": .2}}) + '\n')
            with patch.object(ssl.frozen, "WEIGHTS_SHA256", initial_sha), redirect_stdout(io.StringIO()):
                sealed = ssl.seal_encoder(SimpleNamespace(ssl_run=str(root), output=str(root / "seal.json")))
            self.assertEqual(sealed["selected_global_step"], 12)
            self.assertEqual(sealed["updated_tensor_count"], 1)

    @unittest.skipUnless(os.environ.get("XAVIER_CBRAMOD_UPSTREAM") and os.environ.get("XAVIER_CBRAMOD_WEIGHTS"), "official private pinned source/weights not configured")
    def test_official_full_encoder_real_cpu_masked_gradient(self):
        model = ssl.load_full_encoder(os.environ["XAVIER_CBRAMOD_UPSTREAM"], os.environ["XAVIER_CBRAMOD_WEIGHTS"])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4_924_000)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 4_923_800)
        for channels in (8, 32):
            with self.subTest(channels=channels):
                model.zero_grad(set_to_none=True)
                x = torch.randn(1, channels, 30, 200) * .03
                mask = ssl.fixed_mask("cpu-shape-diagnostic", channels).unsqueeze(0)
                output = model(x, mask=mask)
                self.assertEqual(output.shape, x.shape)
                loss = ssl.masked_loss(output, x, mask)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(float(model.patch_embedding.spectral_proj[0].weight.grad.abs().sum()), 0.)
                self.assertIsNone(model.patch_embedding.mask_encoding.grad)
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
