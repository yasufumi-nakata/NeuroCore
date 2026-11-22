"""CPU tests for the separate CBraMod/symbol experiment; no download or GPU."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel

import cbramod_token_llm as api

torch.set_num_threads(2)


class StubEEGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(200, 200)
        self.drop = nn.Dropout(.8)

    def forward(self, x):
        return self.drop(self.layer(x))


def tiny_model():
    torch.manual_seed(12)
    base = GPT2LMHeadModel(GPT2Config(vocab_size=512, n_positions=128, n_embd=24,
                                    n_layer=1, n_head=4, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
                                    bos_token_id=1, eos_token_id=2, pad_token_id=0))
    return api.EEGCausalModel(base, 8, 8, 200, encoder=StubEEGEncoder())


def parameter_digest(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class TestTokenizer:
    bos_token_id = 1
    def encode(self, text, add_special_tokens=False):
        for symbol, token in zip(api.SYMBOLS, api.TOKEN_IDS):
            if text.endswith(symbol):
                return self.encode(text[:-len(symbol)]) + [token]
        if not text: return []
        return [5, 6, 7]  # Small deterministic CPU fixture, not a pretrained tokenizer.


def row(identifier="one", label="keyboard", index=0):
    return {"id": identifier, "label": label, "split": "train", "dataset": "fixture",
            "subject": "one", "session": "one", "recording_id": "one", "decision_time_s": 20.,
            "sample_rate_hz": 256, "channels": api.CHANNELS, "environment_text": "Neutral prior context.",
            "eeg_index": index}


class SymbolModelTests(unittest.TestCase):
    def test_resampling_unit_shape_and_only_latest_second(self):
        x = np.ones((8, 448), np.float32) * 1e-5
        x[:, :192] = 1e6
        out = api.latest_second(x)
        self.assertEqual(out.shape, (8, 200))
        np.testing.assert_allclose(out[:, 30:-30], .1, atol=.0001)
        y = x.copy();y[:, :192] = -1e9
        np.testing.assert_array_equal(api.latest_second(y), out)
        self.assertEqual(out.dtype, np.float32)
        for kwargs in ({"unit": "uV"}, {"sample_rate_hz": 200}, {"channel_order": list(reversed(api.CHANNELS))}):
            with self.assertRaises(ValueError): api.latest_second(x, **kwargs)
        with self.assertRaises(ValueError): api.latest_second(x[:, :-1])
        x[0, -1] = np.nan
        with self.assertRaises(ValueError): api.latest_second(x)

    def test_store_keeps_physical_amplitude_and_raw_shape_contract(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.npy"
            rng = np.random.default_rng(2)
            x = rng.normal(size=(2, 8, 448)).astype(np.float32) * 1e-5
            x[1] = x[0] * 2
            np.save(p, x)
            a = {**row(), "eeg_path": str(p)}
            b = {**row(index=1), "eeg_path": str(p)}
            store = api.EEGStore(448)
            torch.testing.assert_close(store.get(b), store.get(a) * 2)
            a["sample_rate_hz"] = 250
            with self.assertRaises(ValueError): store.get(a)
        with self.assertRaises(ValueError): api.EEGStore(200)

    def test_projection_gradient_through_frozen_llm_and_frozen_eeg_encoder(self):
        model = tiny_model().train()
        before_base, before_eeg = parameter_digest(model.base), parameter_digest(model.eeg_encoder)
        before_projection = parameter_digest(model.connector)
        x = torch.randn(3, 8, 200)
        prompts, targets = [[1, 3, 4], [1, 6], [1, 7, 8, 9]], torch.tensor([0, 1, 2])
        optimizer = torch.optim.AdamW(model.connector.parameters(), lr=.01)
        loss = model.loss(x, prompts, targets, "fused");loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.connector.parameters()), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.base.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.eeg_encoder.parameters()))
        optimizer.step()
        self.assertEqual(parameter_digest(model.base), before_base)
        self.assertEqual(parameter_digest(model.eeg_encoder), before_eeg)
        self.assertNotEqual(parameter_digest(model.connector), before_projection)

    def test_one_forward_and_training_evaluation_logit_identity(self):
        model = tiny_model().eval()
        x = torch.randn(2, 8, 200);prompts = [[1, 2], [1, 2, 3]];targets = torch.tensor([0, 2])
        with patch.object(model.base, "forward", wraps=model.base.forward) as forward:
            logits = model.class_logits(x, prompts, "fused")
            self.assertEqual(forward.call_count, 1)
            inputs = forward.call_args.kwargs
            self.assertEqual(inputs["inputs_embeds"].shape[1], 8 + 3)
            self.assertNotIn("labels", inputs)
        expected = F.cross_entropy(logits, targets)
        torch.testing.assert_close(model.loss(x, prompts, targets, "fused"), expected, rtol=0, atol=0)

    def test_cpu_encoder_stays_cpu_float32_when_parent_is_moved_or_cast(self):
        model = tiny_model().to(dtype=torch.float64).train()
        self.assertTrue(all(p.dtype == torch.float64 for p in model.connector.parameters()))
        self.assertTrue(all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.eeg_encoder.parameters()))
        self.assertFalse(model.base.training)
        self.assertFalse(model.eeg_encoder.training)
        self.assertFalse(model.eeg_encoder.encoder.drop.training)
        self.assertTrue(model.connector.training)
        # Meta is a real non-CPU destination and does not allocate an accelerator.
        model.to("meta")
        self.assertTrue(all(p.device.type == "meta" for p in model.connector.parameters()))
        self.assertTrue(all(p.device.type == "cpu" for p in model.eeg_encoder.parameters()))

    def test_context_only_removes_eeg_and_prompt_does_not_read_target(self):
        model = tiny_model().eval();x = torch.randn(2, 8, 200)
        a = model.class_logits(x, [[1, 2]] * 2, "context_only")
        b = model.class_logits(x * 50, [[1, 2]] * 2, "context_only")
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(api.prompt_text(row(label="keyboard"), "fused"), api.prompt_text(row(label="ai_prompt_submit"), "fused"))
        self.assertNotIn("Neutral prior context", api.prompt_text(row(), "eeg_only"))

    def test_token_audit_rejects_holdout_or_changed_mapping(self):
        audit = api.audit_tokenizer(TestTokenizer(), [row()])
        self.assertEqual(audit["boundary_checks"], 6)
        with self.assertRaises(ValueError): api.audit_tokenizer(TestTokenizer(), [{**row(), "split": "test"}])
        with self.assertRaises(ValueError): api.tokenized_prompt(TestTokenizer(), row(), list(reversed(api.CLASSES)), "fused", 128)

    def test_runtime_identity_guard_precedes_any_backbone_load(self):
        config = {"model": api.MODEL_ID, "revision": api.MODEL_REVISION,
                  "eeg_encoder_identity": api.identity(), "class_token_ids": api.TOKEN_IDS,
                  "core_source_sha256": api.CORE_SHA256, "model_module_sha256": api.sha256(Path(api.__file__))}
        for key in ("eeg_encoder_identity", "class_token_ids", "core_source_sha256", "model_module_sha256"):
            changed = copy.deepcopy(config);changed[key] = "changed"
            with patch.object(api.core, "load_backbone") as loader:
                with self.assertRaises(ValueError): api.load_backbone(changed, torch.device("cpu"))
                loader.assert_not_called()

    def test_protocol_hyperparameters_and_sustained_probe_are_enforced(self):
        fixed = {"epochs": 30, "lr": .001, "seed": 20261122, "dtype": "bfloat16",
                 "max_prompt_tokens": 128, "abstain_threshold": .8, "cpu_threads": 2,
                 "max_seconds": 900, "memory_budget_gib": 21}
        protocol = {"schema": "xavier-cbramod-symbol-experiment-v1", "manifest_sha256": api.MANIFEST_SHA256,
                    "model_source_sha256": api.sha256(Path(api.__file__)), "core_source_sha256": api.CORE_SHA256,
                    "training": fixed, "capacity_selection": {"candidate_batches": [8, 16, 32], "minimum_steps": 50}}
        args = SimpleNamespace(**fixed, command="probe", batch_size=8, probe_steps=50)
        api.validate_protocol(args, protocol)
        for key, value in (("lr", .1), ("seed", 1), ("probe_steps", 3), ("batch_size", 128), ("memory_budget_gib", 22)):
            changed = copy.deepcopy(args);setattr(changed, key, value)
            with self.assertRaises(ValueError): api.validate_protocol(changed, protocol)

    def test_capacity_gate_requires_sustained_stable_within_budget_report(self):
        protocol = {"training": {"memory_budget_gib": 21}}
        report = {"status": "completed_train_only_probe", "memory": {"cuda_peak_reserved_bytes": 6 * 1024**3},
                  "probe": {"completed_requested_steps": True, "at_least_one_train_epoch_equivalent": True,
                            "allocator_stable_last_ten": True}}
        self.assertTrue(api.capacity_eligible(report, protocol))
        for key in report["probe"]:
            changed = copy.deepcopy(report);changed["probe"][key] = False
            self.assertFalse(api.capacity_eligible(changed, protocol))
        changed = copy.deepcopy(report);changed["memory"]["cuda_peak_reserved_bytes"] = 22 * 1024**3
        self.assertFalse(api.capacity_eligible(changed, protocol))

    def test_checkpoint_roundtrip_and_encoder_identity_guard(self):
        model = tiny_model().eval();optimizer = torch.optim.AdamW(model.connector.parameters(), lr=.001)
        x, prompts = torch.randn(2, 8, 200), [[1, 3], [1, 5, 7]]
        model.loss(x, prompts, torch.tensor([0, 2]), "fused").backward();optimizer.step()
        expected = model.class_logits(x, prompts, "fused").detach()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checkpoint.pt"
            api.save_checkpoint(path, model, optimizer, {"manifest_sha256": "a" * 64}, 0, 1,
                                {"x.npy": {"sha256": "b" * 64, "bytes": 100}}, .3, False, 2, 1.)
            fresh = tiny_model().eval()
            state = api.restore_weights(path, fresh)
            self.assertEqual(state["next_batch_position"], 2)
            self.assertFalse(state["epoch_completed"])
            self.assertEqual(state["trainable_base"], {})
            torch.testing.assert_close(fresh.class_logits(x, prompts, "fused"), expected, rtol=0, atol=0)
            state["eeg_encoder_identity"]["weights_sha256"] = "0" * 64
            torch.save(state, path)
            with self.assertRaises(ValueError): api.restore_weights(path, fresh)

    def test_evaluator_one_forward_per_example_and_actual_shuffle(self):
        model = tiny_model().eval()
        rows = [row("one", "keyboard", 0), row("two", "mouse_click", 1)]
        class Store:
            def __init__(self): self.seen = []
            def get(self, source):
                self.seen.append(source["id"])
                return torch.full((8, 200), float(source["eeg_index"] + 1))
        store = Store()
        config = {"mode": "fused", "max_prompt_tokens": 128, "abstain_threshold": .8, "memory_budget_gib": 8}
        with patch.object(model.base, "forward", wraps=model.base.forward) as forward:
            metrics, predictions = api.evaluate(model, TestTokenizer(), rows, api.CLASSES, store, config,
                                                torch.device("cpu"), "a" * 64, condition="shuffled_eeg")
            self.assertEqual(forward.call_count, 2)
        self.assertEqual(store.seen, ["two", "one"])
        self.assertEqual([p["id"] for p in predictions], ["one", "two"])
        self.assertTrue(all(p["perturbation"]["eeg_source_id"] != p["id"] for p in predictions))
        self.assertTrue(all(abs(sum(p["probabilities"].values()) - 1) < 1e-6 for p in predictions))
        self.assertEqual(metrics["candidate_forward_passes"], 1)

    @unittest.skipUnless(os.environ.get("XAVIER_CBRAMOD_TEST_UPSTREAM"), "Official offline source/weights fixture not present")
    def test_official_frozen_encoder_real_train_windows_cpu_gradient(self):
        config = {"cbramod_upstream": os.environ["XAVIER_CBRAMOD_TEST_UPSTREAM"],
                  "cbramod_weights": os.environ["XAVIER_CBRAMOD_TEST_WEIGHTS"]}
        encoder = api.load_official_encoder(config)
        manifest = Path(os.environ["XAVIER_CBRAMOD_TEST_MANIFEST"])
        self.assertEqual(api.sha256(manifest), api.MANIFEST_SHA256)
        rows = [r for r in api.read_rows(manifest) if r["split"] == "train"][:3]
        x = torch.stack([api.EEGStore(448).get(r) for r in rows])
        before = parameter_digest(encoder)
        tiny = tiny_model()
        model = api.EEGCausalModel(tiny.base, 8, 8, 200, encoder=encoder).train()
        model.loss(x, [[1, 3, 5]] * 3, torch.tensor([api.CLASSES.index(r["label"]) for r in rows]), "fused").backward()
        self.assertTrue(all(p.grad is None for p in model.eeg_encoder.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.connector.parameters()), 0)
        self.assertEqual(parameter_digest(encoder), before)
        self.assertEqual(sum(p.numel() for p in encoder.parameters()), 4_883_800)


if __name__ == "__main__":
    unittest.main()
