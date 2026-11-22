"""CPU-only checks of the new adapted-weights alignment contract."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import cbramod_ssl_token_llm as api
import cbramod_token_llm as original
from test_cbramod_token_llm import tiny_model, parameter_digest, row


def seal_fixture():
    return {"schema": "xavier-selected-ssl-encoder-seal-v1", "weights_sha256": "a" * 64,
            "ssl_best_checkpoint_sha256": "b" * 64, "ssl_protocol_sha256": "c" * 64,
            "ssl_source_sha256": "d" * 64, "ssl_manifest_sha256": "e" * 64,
            "ssl_source_allowlist_sha256": "f" * 64, "initial_weights_sha256": api.INITIAL_WEIGHTS_SHA256,
            "selection_partition": "portable_SSL_validation", "downstream_test_source_used": False}


def model_pair():
    old = tiny_model().eval()
    new = api.EEGCausalModel(copy.deepcopy(old.base), 8, encoder=copy.deepcopy(old.eeg_encoder.encoder)).eval()
    new.connector.load_state_dict(old.connector.state_dict())
    return old, new


class AdaptedAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_same_preprocessing_prompt_tokens_logits_and_loss_for_same_weights(self):
        old, new = model_pair()
        source = np.random.default_rng(7).normal(0, 1e-5, size=(8, 448)).astype(np.float32)
        np.testing.assert_array_equal(original.latest_second(source), api.latest_second(source))
        self.assertEqual(api.TOKEN_IDS, original.TOKEN_IDS)
        self.assertEqual(api.CHANNELS, original.CHANNELS)
        x, prompts, targets = torch.randn(2, 8, 200), [[1, 3, 4], [1, 8]], torch.tensor([0, 2])
        for mode in api.MODES:
            self.assertEqual(original.prompt_text(row(), mode), api.prompt_text(row(), mode))
            torch.testing.assert_close(old.class_logits(x, prompts, mode), new.class_logits(x, prompts, mode), rtol=0, atol=0)
            torch.testing.assert_close(old.loss(x, prompts, targets, mode), new.loss(x, prompts, targets, mode), rtol=0, atol=0)

    def test_projection_updates_with_both_encoders_and_llm_frozen(self):
        _, new = model_pair()
        before_encoder, before_base = parameter_digest(new.eeg_encoder), parameter_digest(new.base)
        optimizer = torch.optim.AdamW(new.connector.parameters(), lr=.001)
        new.train()
        loss = new.loss(torch.randn(2, 8, 200), [[1, 4]] * 2, torch.tensor([0, 1]), "fused")
        loss.backward(); optimizer.step()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in new.connector.parameters()), 0.)
        self.assertTrue(all(p.grad is None for p in new.base.parameters()))
        self.assertTrue(all(p.grad is None for p in new.eeg_encoder.parameters()))
        self.assertEqual(parameter_digest(new.base), before_base)
        self.assertEqual(parameter_digest(new.eeg_encoder), before_encoder)
        new.to("meta")
        self.assertTrue(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in new.eeg_encoder.parameters()))

    def test_no_original_weights_or_external_selection_seal(self):
        config = {"ssl_encoder_seal": seal_fixture()}
        self.assertEqual(api.identity(config)["weights_sha256"], "a" * 64)
        for key, value in (("weights_sha256", api.INITIAL_WEIGHTS_SHA256),
                           ("selection_partition", "ITU_test"), ("downstream_test_source_used", True),
                           ("ssl_source_allowlist_sha256", None)):
            changed = copy.deepcopy(config); changed["ssl_encoder_seal"][key] = value
            with self.assertRaises((ValueError, TypeError)):
                api.identity(changed)

    def test_runtime_identity_rejection_before_loading_backbone(self):
        config = {"ssl_encoder_seal": seal_fixture(), "model": api.MODEL_ID, "revision": api.MODEL_REVISION,
                  "class_token_ids": api.TOKEN_IDS, "core_source_sha256": api.CORE_SHA256,
                  "model_module_sha256": api.sha256(Path(api.__file__))}
        config["eeg_encoder_identity"] = api.identity(config)
        config["eeg_encoder_identity"]["weights_sha256"] = "0" * 64
        with patch.object(api.core, "load_backbone") as loader:
            with self.assertRaises(ValueError):
                api.load_backbone(config, torch.device("cpu"))
            loader.assert_not_called()

    def test_checkpoint_schema_and_adapted_identity_roundtrip(self):
        _, model = model_pair()
        optimizer = torch.optim.AdamW(model.connector.parameters(), lr=.001)
        x, prompts = torch.randn(1, 8, 200), [[1, 2, 4]]
        model.loss(x, prompts, torch.tensor([0]), "fused").backward(); optimizer.step()
        expected = model.class_logits(x, prompts, "fused").detach()
        config = {"manifest_sha256": api.MANIFEST_SHA256, "ssl_encoder_seal": seal_fixture()}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "checkpoint.pt"
            api.save_checkpoint(path, model, optimizer, config, 0, 1, {}, .3, False, 1, 1.)
            _, fresh = model_pair()
            state = api.restore_weights(path, fresh)
            self.assertEqual(state["schema"], "xavier-cbramod-ssl-symbol-llm-v1")
            torch.testing.assert_close(fresh.class_logits(x, prompts, "fused"), expected, rtol=0, atol=0)
            state["schema"] = original.SCHEMA_VERSION
            torch.save(state, path)
            with self.assertRaises(ValueError):
                api.restore_weights(path, fresh)

    def test_protocol_seals_selected_encoder_and_fixed_alignment_settings(self):
        fixed = {"epochs": 30, "lr": .001, "seed": 20261122, "dtype": "bfloat16",
                 "max_prompt_tokens": 128, "abstain_threshold": .8, "cpu_threads": 2,
                 "max_seconds": 900, "memory_budget_gib": 21}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seal.json"; path.write_text(json.dumps(seal_fixture()))
            args = SimpleNamespace(**fixed, command="probe", batch_size=32, probe_steps=50, ssl_encoder_seal=str(path))
            protocol = {"schema": "xavier-cbramod-ssl-symbol-experiment-v1", "manifest_sha256": api.MANIFEST_SHA256,
                        "model_source_sha256": api.sha256(Path(api.__file__)), "core_source_sha256": api.CORE_SHA256,
                        "ssl_encoder_seal_sha256": api.sha256(path), "training": fixed,
                        "capacity_selection": {"candidate_batches": [32], "minimum_steps": 50}}
            api.validate_protocol(args, protocol)
            for key, value in (("lr", .1), ("seed", 1), ("batch_size", 16), ("probe_steps", 3)):
                changed = copy.deepcopy(args); setattr(changed, key, value)
                with self.assertRaises(ValueError):
                    api.validate_protocol(changed, protocol)
            path.write_text(path.read_text() + "\n")
            with self.assertRaises(ValueError):
                api.validate_protocol(args, protocol)

    @unittest.skipUnless(os.environ.get("XAVIER_CBRAMOD_UPSTREAM") and os.environ.get("XAVIER_CBRAMOD_WEIGHTS"), "official private fixture unavailable")
    def test_adapted_weight_loader_cpu_fixture_keeps_source_and_delta(self):
        # A one-element synthetic weight delta tests loading, not SSL performance.
        official = torch.load(os.environ["XAVIER_CBRAMOD_WEIGHTS"], map_location="cpu", weights_only=True)
        name = "patch_embedding.spectral_proj.0.weight"
        official[name] = official[name].clone()
        official[name].flatten()[0] += .001
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "synthetic_adapted_fixture.pth"
            torch.save(official, path)
            seal = seal_fixture(); seal["weights_sha256"] = api.sha256(path)
            config = {"ssl_encoder_seal": seal, "cbramod_weights": str(path), "cbramod_upstream": os.environ["XAVIER_CBRAMOD_UPSTREAM"]}
            loaded = api.load_adapted_encoder(config)
            torch.testing.assert_close(loaded.state_dict()[name], official[name], rtol=0, atol=0)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            self.assertEqual(sum(p.numel() for p in loaded.parameters()), 4_883_800)


if __name__ == "__main__":
    unittest.main()
