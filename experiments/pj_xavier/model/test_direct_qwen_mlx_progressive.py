"""CPU-only safety/measurement tests; no model load, GPU use, or held-out data."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import fcntl
import os
import signal
import subprocess

spec = importlib.util.spec_from_file_location("direct_qwen_progressive", Path(__file__).with_name("direct_qwen_mlx_progressive.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ToyTokenizer:
    def __init__(self, broken=False):
        self.broken = broken
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        self.calls.append(kwargs)
        prefix = [1001] + [ord(c) for c in messages[0]["content"]] + [1002]
        if len(messages) == 1:
            return prefix[:-1] + ([1003] if self.broken else [1002])
        return prefix + [ord(c) for c in messages[1]["content"]] + [1004]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if i < 1000)


def row(index, subject, split="val", label="keyboard"):
    prompt = ("Predecision context stays here.\n" + m.EEG_MARKER
              + "CP3 t:" + str(index) + "\nC3 p: 1 2 3 4" + m.EEG_TAIL)
    return {"id": f"{split}-{subject}-{index}", "canonical_subject_id": subject, "split": split,
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": label}]}


def cohort(split, subjects):
    return [row(j * 6 + i, subject, split, m.LABELS[i % 3])
            for j, subject in enumerate(subjects) for i in range(6)]


class DirectQwenTests(unittest.TestCase):
    def test_no_thinking_prefix_and_exact_label_mask_include_eos_exclude_padding(self):
        tokenizer = ToyTokenizer()
        record = row(1, "s1")
        tokens, offset = m.render_example(tokenizer, record)
        # IDs at indices offset...L-1 are exactly label plus EOS.  The next
        # position is padding and must never be included by the training loss.
        expected = [ord(c) for c in "keyboard"] + [1004]
        self.assertEqual(tokens[offset:], expected)
        padded = tokens + [0] * 8
        selected = [padded[pos] for pos in range(1, len(padded)) if offset <= pos < len(tokens)]
        self.assertEqual(selected, expected)
        self.assertNotIn(0, selected)
        self.assertTrue(all(c["enable_thinking"] is False for c in tokenizer.calls))

    def test_prefix_mismatch_and_truncation_fail_before_training(self):
        with self.assertRaisesRegex(ValueError, "prefix_mismatch"):
            m.render_example(ToyTokenizer(broken=True), row(1, "s1"))
        with self.assertRaisesRegex(ValueError, "truncated"):
            m.render_example(ToyTokenizer(), row(1, "s1"), max_length=10)

    def test_noeeg_keeps_context_gold_and_does_not_modify_source(self):
        record = row(1, "s1")
        original = copy.deepcopy(record)
        prompt = m.no_eeg_prompt(record["messages"][0]["content"])
        self.assertTrue(prompt.startswith("Predecision context stays here."))
        self.assertTrue(prompt.endswith(m.EEG_TAIL))
        self.assertNotIn("CP3 t:", prompt)
        self.assertEqual(record, original)

    def test_shuffle_is_paired_within_subject_and_changes_only_eeg(self):
        rows = cohort("val", ["v1", "v2"])
        prompts, donors = m.shuffled_prompts(rows, 7)
        self.assertEqual(sorted(donors), list(range(len(rows))))
        for i, j in enumerate(donors):
            self.assertNotEqual(i, j)
            self.assertEqual(rows[i]["canonical_subject_id"], rows[j]["canonical_subject_id"])
            before, _, after = m.split_prompt(rows[i]["messages"][0]["content"])
            _, eeg, _ = m.split_prompt(rows[j]["messages"][0]["content"])
            self.assertEqual(prompts[i], before + m.EEG_MARKER + eeg + after)

    def test_preflight_never_opens_test_and_rejects_subject_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            cfg = {"model_path": str(model), "expected_model_config_sha256": m.sha256(model / "config.json"),
                   "seed": 4, "iters": 8, "max_seq_length": 1024, "inputs": {}, "expected_counts": {}}
            for split, subjects in [("train", ["t1", "t2"]), ("val", ["v1", "v2"])]:
                path = root / (split + ".jsonl")
                data = cohort(split, subjects)
                path.write_text("".join(json.dumps(r) + "\n" for r in data))
                cfg["inputs"][split] = {"path": str(path), "sha256": m.sha256(path)}
                cfg["expected_counts"][split] = len(data)
            forbidden = root / "heldout_test.jsonl"
            forbidden.write_text("THIS IS NOT VALID JSON AND MUST NEVER BE OPENED")
            original_open = Path.open

            def guarded_open(path, *args, **kwargs):
                if path == forbidden:
                    raise AssertionError("heldout test was opened")
                return original_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", guarded_open):
                receipt, _, _ = m.preflight(cfg, ToyTokenizer())
            self.assertFalse(receipt["test_opened"])
            self.assertTrue(receipt["splits"]["val/eeg"]["all_prefixes_equal"])
            bad_rows = cohort("val", ["t1", "v2"])
            path = Path(cfg["inputs"]["val"]["path"])
            path.write_text("".join(json.dumps(r) + "\n" for r in bad_rows))
            cfg["inputs"]["val"]["sha256"] = m.sha256(path)
            with self.assertRaisesRegex(ValueError, "subject_overlap"):
                m.read_training_rows(cfg)

    def test_fixed_development_probes_and_training_order_match_across_arms(self):
        rows = cohort("val", [f"v{i}" for i in range(6)])
        first = m.fixed_probe_indices(rows, 4, per_subject=6)
        second = m.fixed_probe_indices(copy.deepcopy(rows), 4, per_subject=6)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 36)
        self.assertEqual(m.training_order(400, 7, 80), m.training_order(400, 7, 80))
        self.assertEqual(len(set(m.training_order(400, 7, 80))), 80)
        with self.assertRaisesRegex(ValueError, "only_train_and_validation"):
            m.fixed_probe_indices(cohort("test", ["heldout"]), 4, per_subject=6)

    def test_exact_output_format_and_paired_subject_comparison(self):
        self.assertEqual(m.parse_prediction("keyboard\n"), "keyboard")
        self.assertEqual(m.parse_prediction("keyboard because ..."), "invalid")
        a = [{"index": i, "gold": lab, "pred": lab, "row_sha256": str(i), "subject_sha256": "v1"}
             for i, lab in enumerate(m.LABELS)]
        b = copy.deepcopy(a)
        b[0]["pred"] = "invalid"
        comparison = m.paired_difference(a, b)
        self.assertAlmostEqual(comparison["accuracy_delta"], 1 / 3)
        self.assertEqual(comparison["paired_wins"], 1)
        self.assertEqual(m.metrics(a)["balanced_accuracy"], 1)
        b[0]["row_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "unpaired"):
            m.paired_difference(a, b)

    def test_atomic_json_does_not_replace_valid_state_with_nonfinite_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            m.atomic_json(path, {"status": "running"})
            with self.assertRaises(ValueError):
                m.atomic_json(path, {"loss": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"status": "running"})

    def test_gpu_lock_is_nonblocking_and_child_is_not_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            lock = base / "train_gpu.lock"
            fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            cfg = {"run_dir": str(base / "run"), "model_path": str(base / "model"), "lock_path": str(lock)}
            try:
                with mock.patch.object(subprocess, "Popen") as popen:
                    self.assertEqual(m.run_supervisor("unused", cfg), 11)
                    popen.assert_not_called()
                self.assertEqual(json.loads((base / "run/supervisor.json").read_text())["status"], "blocked_gpu_lock")
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def test_timeout_terminates_process_group_before_unlock(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = {"run_dir": str(base / "run"), "model_path": str(base / "model"),
                   "lock_path": str(base / "train_gpu.lock"), "max_wall_seconds": 60}
            (base / "train_gpu.lock").touch()
            child = mock.Mock(pid=12345)
            child.wait.side_effect = [subprocess.TimeoutExpired("worker", 60),
                                      subprocess.TimeoutExpired("worker", 15), 0]
            child.poll.return_value = 0
            with mock.patch.object(subprocess, "run", return_value=mock.Mock(stdout="")), \
                 mock.patch.object(subprocess, "Popen", return_value=child), \
                 mock.patch.object(os, "killpg") as killpg:
                self.assertEqual(m.run_supervisor("unused", cfg), 124)
                self.assertEqual(killpg.call_args_list, [mock.call(12345, signal.SIGTERM),
                                                        mock.call(12345, signal.SIGKILL)])
            self.assertEqual(json.loads((base / "run/supervisor.json").read_text())["status"], "stopped_wall_budget")

    def test_missing_registered_lock_is_not_silently_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {"run_dir": str(root / "run"), "model_path": str(root / "model"),
                   "lock_path": str(root / "train_gpu.lock")}
            with self.assertRaisesRegex(ValueError, "registered_gpu_lock_missing"):
                m.run_supervisor("unused", cfg)
            self.assertFalse((root / "train_gpu.lock").exists())
            self.assertFalse((root / "run").exists())

    def test_failed_checkpoint_cannot_be_compared_or_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg, no = root / "eeg", root / "noeeg"
            eeg.mkdir()
            no.mkdir()
            (eeg / "evaluation.json").write_text(json.dumps({"condition": "eeg",
                "status": "failed_quality_gates", "quality_gates_passed": False}))
            (no / "evaluation.json").write_text(json.dumps({"condition": "noeeg",
                "status": "completed", "quality_gates_passed": True}))
            with self.assertRaisesRegex(ValueError, "failed_or_incomplete"):
                m.compare_checkpoints(eeg, no, root / "comparison.json")
            self.assertFalse((root / "comparison.json").exists())

    def test_experiment_identity_pins_model_location_and_base_receipt(self):
        cfg = {"rank": 8, "seed": 1, "learning_rate": .0001, "batch_size": 1,
               "max_seq_length": 1024, "enable_thinking": False, "inputs": {},
               "model_path": "/tmp/base-a/model", "expected_model_config_sha256": "a" * 64,
               "base_receipt": {"sha256": "b" * 64}}
        before = m.experiment_identity(cfg)
        changed_location = {**cfg, "model_path": "/tmp/base-b/model"}
        self.assertNotEqual(before, m.experiment_identity(changed_location))
        changed_weights_receipt = {**cfg, "base_receipt": {"sha256": "c" * 64}}
        self.assertNotEqual(before, m.experiment_identity(changed_weights_receipt))

    def test_memory_failure_records_real_peak_and_completed_step(self):
        allocator = mock.Mock()
        allocator.get_active_memory.return_value = 17_000_000_000
        allocator.get_cache_memory.return_value = 0
        allocator.get_peak_memory.return_value = 50_123_456_789
        snapshot = m.memory_snapshot(allocator)
        self.assertAlmostEqual(snapshot["peak_gb"], 50.123456789)
        with self.assertRaisesRegex(ValueError, "completed_optimizer_steps.*2.*peak_gb.*50.123456789"):
            m.check_memory_snapshot(snapshot, 50, step=2)
        allocator.reset_peak_memory.assert_not_called()

    def test_completion_head_loss_and_gradients_equal_full_logits_on_mlx_cpu(self):
        if importlib.util.find_spec("mlx") is None:
            self.skipTest("MLX is unavailable; run this CPU test on the target Mac before GPU launch")
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            mx.random.seed(41)

            class Backbone(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.hidden = mx.random.normal((1, 16, 6)) * .1
                    self.lora_a = mx.random.normal((6, 2)) * .1
                    self.lora_b = mx.random.normal((2, 6)) * .1

                def __call__(self, ids):
                    hidden = self.hidden[:, :ids.shape[1], :]
                    return hidden + (hidden @ self.lora_a) @ self.lora_b

            class LanguageModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = Backbone()
                    self.lm_head = nn.Linear(6, 17, bias=False)
                    self.lm_head.freeze()

                def __call__(self, ids):
                    return self.lm_head(self.model(ids))

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.language_model = LanguageModel()

                def __call__(self, ids):
                    return self.language_model(ids)

            model = Model()

            def full_loss(current, batch, lengths):
                logits = current(batch[:, :-1])
                positions = mx.arange(1, batch.shape[1])
                mask = (positions >= lengths[:, :1]) & (positions < lengths[:, 1:])
                return (nn.losses.cross_entropy(logits, batch[:, 1:]) * mask).astype(mx.float32).sum() / mask.sum()

            def selected_loss(current, batch, lengths):
                return m.completion_head_loss(mx, nn, current, batch, lengths, 5)[0]

            full_grad = nn.value_and_grad(model, full_loss)
            selected_grad = nn.value_and_grad(model, selected_loss)

            @mx.compile
            def compiled_selected(batch, lengths):
                return selected_grad(model, batch, lengths)

            for completion_length in (3, 4, 5):
                with self.subTest(completion_length=completion_length):
                    token_length = 7 + completion_length
                    values = [(i + 2) % 17 if i < token_length else 0 for i in range(16)]
                    batch = mx.array([values], dtype=mx.int32)
                    lengths = mx.array([[7, token_length]], dtype=mx.int32)
                    full_value, original_gradient = full_grad(model, batch, lengths)
                    selected_value, selected_gradient = compiled_selected(batch, lengths)
                    mx.eval(full_value, selected_value, original_gradient, selected_gradient)
                    self.assertAlmostEqual(float(full_value.item()), float(selected_value.item()), places=5)
                    original = dict(tree_flatten(original_gradient))
                    selected = dict(tree_flatten(selected_gradient))
                    expected = {"language_model.model.hidden", "language_model.model.lora_a", "language_model.model.lora_b"}
                    self.assertEqual(set(original), expected)
                    self.assertEqual(set(original), set(selected))
                    for name in original:
                        self.assertTrue(mx.allclose(original[name], selected[name], atol=5e-6, rtol=5e-5).item(), name)
        finally:
            mx.set_default_device(previous_device)


if __name__ == "__main__":
    unittest.main()
