import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import torch

from eeg_llm import (EEGCausalModel, EEGStore, ByteTokenizer, PortableAdaptiveAvgPool1d,
                     EEGPrefix, class_metrics, load_backbone,
                     eeg_artifact_manifest, read_rows, parser, train,
                     apply_allocator_limit, predict,
                     within_recording_derangement,
                     restore_weights, save_checkpoint, validate_rows, SCHEMA_VERSION)


def tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(3)
    base = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=40, n_embd=16,
                                    n_layer=1, n_head=2, resid_pdrop=0,
                                    embd_pdrop=0, attn_pdrop=0))
    return EEGCausalModel(base, channels=3, prefix_tokens=3, width=8)


def rows():
    result = []
    for i, split in enumerate(("train", "val", "test")):
        for j, label in enumerate(("left", "right")):
            result.append(dict(id=f"{split}-{j}", dataset="data", subject=f"person-{i}",
                               session="s", recording_id="r", eeg_path="x.npy",
                               window_start_s=j * 3, window_end_s=j * 3 + 1,
                               decision_time_s=j * 3 + 2, label=label,
                               label_source="recorded_event", split=split))
    return result


class ConnectorTests(unittest.TestCase):
    def test_mps_pooling_bins_match_native_values_and_gradients(self):
        for length, bins in ((110, 8), (109, 8), (112, 8), (3, 8)):
            with self.subTest(length=length, bins=bins):
                actual_input = torch.randn(2, 3, length, dtype=torch.float64, requires_grad=True)
                expected_input = actual_input.detach().clone().requires_grad_(True)
                actual = PortableAdaptiveAvgPool1d.slice_average(actual_input, bins)
                expected = torch.nn.functional.adaptive_avg_pool1d(expected_input, bins)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                weights = torch.randn_like(actual)
                (actual * weights).sum().backward()
                (expected * weights).sum().backward()
                torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=1e-12, atol=1e-12)

    def test_portable_pooling_preserves_existing_checkpoint_keys_and_cpu_output(self):
        portable = EEGPrefix(3, 16, 8, 8)
        legacy = copy.deepcopy(portable)
        legacy.temporal[-1] = torch.nn.AdaptiveAvgPool1d(8)
        portable.load_state_dict(legacy.state_dict(), strict=True)
        eeg = torch.randn(2, 3, 438)
        torch.testing.assert_close(portable(eeg), legacy(eeg), rtol=0, atol=0)
        self.assertEqual(set(portable.state_dict()), set(legacy.state_dict()))

    def test_gradient_reaches_encoder_through_frozen_llm(self):
        model = tiny_model()
        before = {n: p.detach().clone() for n, p in model.base.named_parameters()}
        eeg = torch.randn(2, 3, 32)
        loss = model.loss(eeg, [[1, 2], [1]], [[3, 4], [5]], "eeg_only")
        loss.backward()
        gradient = sum(float(p.grad.abs().sum()) for p in model.connector.parameters() if p.grad is not None)
        self.assertGreater(gradient, 0)
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        opt = torch.optim.AdamW(model.connector.parameters(), lr=.01)
        opt.step()
        self.assertTrue(all(torch.equal(before[n], p) for n, p in model.base.named_parameters()))

    def test_prefix_prompt_and_padding_are_never_target_labels(self):
        model = tiny_model()
        _, attention, labels = model.pack(torch.randn(2, 3, 32), [[1, 2], [1]], [[3, 4], [5]], "eeg_only")
        self.assertEqual(labels[0].tolist(), [-100, -100, -100, -100, -100, 3, 4])
        self.assertEqual(labels[1].tolist(), [-100, -100, -100, -100, 5, -100, -100])
        self.assertEqual(attention[1].tolist(), [1, 1, 1, 1, 1, 0, 0])

    def test_checkpoint_roundtrip_preserves_prediction(self):
        model = tiny_model().eval()
        eeg = torch.randn(1, 3, 32)
        expected = model.answer_log_likelihood(eeg, [[1]], [[2, 3]], "eeg_only")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "checkpoint.pt"
            opt = torch.optim.AdamW(model.connector.parameters())
            save_checkpoint(p, model, opt, {"model": "offline-tiny-random"}, 0, 1, ["a", "b"], "hash", .5)
            with torch.no_grad():
                for param in model.connector.parameters():
                    param.add_(2)
            restore_weights(p, model)
            actual = model.answer_log_likelihood(eeg, [[1]], [[2, 3]], "eeg_only")
        torch.testing.assert_close(expected, actual)

    def test_context_only_does_not_see_eeg(self):
        model = tiny_model().eval()
        a = model.answer_log_likelihood(torch.randn(1, 3, 32), [[1]], [[2]], "context_only")
        b = model.answer_log_likelihood(torch.randn(1, 3, 32) * 100, [[1]], [[2]], "context_only")
        torch.testing.assert_close(a, b)

    def test_missing_transformers_commit_uses_same_hub_snapshot_for_all_reads(self):
        sha = "a" * 40
        cfg = {"model": "example/model", "revision": None, "dtype": "float32",
               "seed": 1, "max_base_params": 1000000, "memory_budget_gib": 1}
        fake_config = Mock(_commit_hash=None)
        fake_base = nn_model = torch.nn.Embedding(3, 4)
        with patch("huggingface_hub.hf_hub_download", return_value=f"/cache/models--example--model/snapshots/{sha}/config.json"), \
             patch("transformers.AutoConfig.from_pretrained", return_value=fake_config) as read_config, \
             patch("transformers.AutoModelForCausalLM.from_config", return_value=fake_base), \
             patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_base) as read_weights, \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=ByteTokenizer()) as read_tokenizer:
            _, _, metadata = load_backbone(cfg, torch.device("cpu"))
        self.assertEqual(metadata["resolved_revision"], sha)
        for call in (read_config, read_weights, read_tokenizer):
            self.assertEqual(call.call_args.kwargs["revision"], sha)

    def test_unresolvable_revision_refuses_weight_loading(self):
        cfg = {"model": "example/model", "revision": None, "dtype": "float32",
               "seed": 1, "max_base_params": 1000000, "memory_budget_gib": 1}
        with patch("huggingface_hub.hf_hub_download", return_value="/custom/config.json"), \
             patch("huggingface_hub.HfApi.model_info", return_value=Mock(sha=None)), \
             patch("transformers.AutoModelForCausalLM.from_pretrained") as load:
            with self.assertRaisesRegex(ValueError, "immutable"):
                load_backbone(cfg, torch.device("cpu"))
            load.assert_not_called()


class IntegrityTests(unittest.TestCase):
    def test_decision_must_include_filter_availability_and_preserve_event_lead(self):
        data = rows()
        data[0].update(event_time_s=2.2, minimum_lead_s=.25)
        with self.assertRaisesRegex(ValueError, "pre-action guard"):
            validate_rows(data, "subject", "eeg_only")
        data[0].update(event_time_s=3, feature_available_time_s=2.5)
        with self.assertRaisesRegex(ValueError, "Feature availability"):
            validate_rows(data, "subject", "eeg_only")

    def test_pseudo_labels_cannot_be_evaluation_gold(self):
        x = rows()
        x[-1]["label_source"] = "ai_pseudo"
        with self.assertRaisesRegex(ValueError, "Non-gold"):
            validate_rows(x, "subject", "eeg_only")

    def test_overlapping_raw_intervals_are_rejected(self):
        x = rows()
        for r in x:
            r["subject"] = "same-person"
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            validate_rows(x, "within_subject_time", "eeg_only")

    def test_future_context_is_rejected(self):
        x = rows()
        for r in x:
            r.update(environment_text="An editor is open", environment_source="predecision_harness",
                     environment_time_s=r["decision_time_s"] + 1)
        with self.assertRaisesRegex(ValueError, "precede EEG"):
            validate_rows(x, "subject", "fused")

    def test_valid_subject_split(self):
        result = validate_rows(rows(), "subject", "eeg_only")
        self.assertEqual(result["subjects"], 3)

    def test_dataset_versions_cannot_hide_same_subject_across_splits(self):
        x = rows()
        for r in x:
            r["dataset"] = "release-" + r["split"]
            r["canonical_dataset_id"] = "one-study"
            r["subject"] = "same-person"
        with self.assertRaisesRegex(ValueError, "group crosses"):
            validate_rows(x, "subject", "eeg_only")

    def test_same_physical_signal_cannot_hide_under_different_subject_ids(self):
        x = rows()
        for r in x:
            r["eeg_source_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "physical EEG source"):
            validate_rows(x, "subject", "eeg_only")

    def test_shuffle_preserves_subject_session_and_recording_without_self_pairs(self):
        x = rows()
        shuffled = within_recording_derangement(x)
        self.assertEqual({r["id"] for r in x}, {r["id"] for r in shuffled})
        for original, source in zip(x, shuffled):
            self.assertNotEqual(original["id"], source["id"])
            for k in ("dataset", "subject", "session", "recording_id"):
                self.assertEqual(original[k], source[k])

    def test_class_metrics_known_confusion(self):
        result = class_metrics(["a", "a", "b"], np.array([[.9, .1], [.2, .8], [.1, .9]]), ["a", "b"])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["balanced_accuracy"], .75)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 1]])

    def test_file_index_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "eeg.npy"
            x = np.arange(2 * 3 * 32).reshape(2, 3, 32).astype("float32")
            x[1, 0, 0] = np.nan
            np.save(path, x)
            store = EEGStore(32)
            good = store.get({"id": "ok", "eeg_path": str(path), "eeg_index": 0})
            self.assertEqual(good.shape, (3, 32))
            with self.assertRaisesRegex(ValueError, "Invalid EEG"):
                store.get({"id": "bad", "eeg_path": str(path), "eeg_index": 1})


class PersistenceTests(unittest.TestCase):
    def test_cuda_allocator_resolves_implicit_index_before_setting_quota(self):
        with patch("torch.cuda.current_device", return_value=2), \
             patch("torch.cuda.get_device_properties", return_value=Mock(total_memory=24 * 1024 ** 3)) as props, \
             patch("torch.cuda.set_per_process_memory_fraction") as quota:
            result = apply_allocator_limit(torch.device("cuda"), 21)
        props.assert_called_once_with(2)
        quota.assert_called_once_with(.875, 2)
        self.assertEqual(result["allocator_limit_bytes"], 21 * 1024 ** 3)

    def fixture(self, base):
        data = np.random.default_rng(1).normal(size=(10, 3, 32)).astype("float32")
        np.save(base / "eeg.npy", data)
        items = []
        for i in range(10):
            split = "train" if i < 6 else ("val" if i < 8 else "test")
            items.append(dict(id=f"synthetic-{i}", dataset="diagnostic", subject=split, session="s",
                              recording_id="r", eeg_path="eeg.npy", eeg_index=i,
                              window_start_s=i * 3, window_end_s=i * 3 + 1, decision_time_s=i * 3 + 2,
                              label="a" if i % 2 else "b", label_source="synthetic", data_origin="synthetic", split=split))
        path = base / "manifest.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in items))
        return path

    def args(self, manifest, output, extra=()):
        return parser().parse_args(["train", "--manifest", str(manifest), "--output", str(output),
                                   "--model", "offline-tiny-random", "--samples", "32", "--device", "cpu",
                                   "--epochs", "1", "--batch-size", "2", "--max-seconds", "0", *extra])

    def test_array_bytes_are_sealed_independently_of_manifest_text(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            manifest = self.fixture(base)
            x = read_rows(manifest)
            before = eeg_artifact_manifest(x)
            np.save(base / "eeg.npy", np.zeros((10, 3, 32), dtype="float32"))
            after = eeg_artifact_manifest(x)
        self.assertNotEqual(before, after)

    def test_resume_mid_epoch_processes_remaining_batch_and_matches_continuous_training(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            base = Path(d)
            manifest = self.fixture(base)
            complete = train(self.args(manifest, base / "continuous"))
            partial = train(self.args(manifest, base / "resumed", ("--max-steps", "2")))
            checkpoint = base / "resumed" / partial["last_checkpoint"]
            stopped = torch.load(checkpoint, weights_only=True)
            self.assertFalse(stopped["epoch_completed"])
            self.assertEqual(stopped["next_batch_position"], 4)
            resumed = train(self.args(manifest, base / "resumed", ("--resume", str(checkpoint))))
            a = torch.load(base / "continuous" / complete["last_checkpoint"], weights_only=True)
            b = torch.load(base / "resumed" / resumed["last_checkpoint"], weights_only=True)
            self.assertEqual(resumed["global_step"], 3)
            self.assertTrue(b["epoch_completed"])
            for name in a["connector"]:
                torch.testing.assert_close(a["connector"][name], b["connector"][name], rtol=0, atol=0)

    def test_memory_stop_saves_checkpoint_and_does_not_run_validation_or_test(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            base = Path(d)
            manifest = self.fixture(base)
            with patch("eeg_llm.evaluate") as evaluate:
                result = train(self.args(manifest, base / "memory_stop", ("--memory-budget-gib", "0.000001", "--finalize-test")))
                evaluate.assert_not_called()
            self.assertEqual(result["status"], "stopped_memory_budget")
            self.assertIsNone(result["best_checkpoint"])
            self.assertTrue((base / "memory_stop" / result["last_checkpoint"]).exists())
            self.assertEqual(result["test"]["status"], "blocked")

    def test_mps_allocator_limit_uses_recommended_memory_without_unlimited_fraction(self):
        with patch("torch.mps.recommended_max_memory", return_value=12 * 1024 ** 3), \
             patch("torch.mps.set_per_process_memory_fraction") as limit:
            result = apply_allocator_limit(torch.device("mps"), 8)
        limit.assert_called_once_with(8 / 12)
        self.assertEqual(result["allocator_limit_bytes"], 8 * 1024 ** 3)

    def test_predict_runtime_overrides_do_not_mutate_checkpoint(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            base = Path(d)
            manifest = self.fixture(base)
            result = train(self.args(manifest, base / "run"))
            checkpoint = base / "run" / result["last_checkpoint"]
            before = checkpoint.read_bytes()
            args = parser().parse_args(["predict", "--checkpoint", str(checkpoint),
                "--manifest", str(manifest), "--output", str(base / "predictions.jsonl"),
                "--device", "cpu", "--memory-budget-gib", "2", "--candidate-batch-size", "1",
                "--dtype", "float32", "--max-examples", "2"])
            inference = predict(args)
            self.assertEqual(before, checkpoint.read_bytes())
            self.assertEqual(inference["n_examples"], 2)
            self.assertEqual(inference["memory_budget_gib"], 2)
            self.assertEqual(inference["candidate_batch_size"], 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
