"""CPU contract tests; do not import MLX or touch model weights/GPU/test data."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evlm_mlx_train as runner


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["return_dict"] is False
        prompt = [1] + [ord(x) + 10 for x in messages[0]["content"]] + [2]
        if len(messages) == 1:
            return prompt
        return prompt + [ord(x) + 10 for x in messages[1]["content"]] + [3]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(t - 10) for t in tokens if t > 3)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.qc = self.root / "qc.json"
        self.qc.write_text('{"status":"admitted"}')
        self.manifest = {"schema": runner.DATA_SCHEMA, "task_kind": "quantized_eeg_reconstruction",
            "source_ids": ["source-a"], "protected_subject_keys": [],
            "quantization_schema": "uV_1_integer_raw_v1", "quantization_bounds": [-1000, 1000],
            "qc_receipt": self.entry(self.qc), "inputs": {}}
        self.train = [self.row("a", "train", "train-subject")]
        self.val = [self.row("b", "val", "val-subject")]
        self.cfg = {"mode": "train_stage", "seed": 123, "max_updates": 1,
                    "max_seq_length": 1024, "order_offset": 0,
                    "model_path": str(self.root / "model"), "expected_model_config_sha256": "a" * 64,
                    "base_receipt": {"sha256": "b" * 64}}
        self.write_manifest()

    def tearDown(self):
        self.tmp.cleanup()

    def entry(self, path):
        return {"path": str(path), "sha256": runner.sha256(path)}

    def row(self, id_, split, subject):
        observed, masked = [1, None, None, 4], [1, 2]
        geometry = {"shape": [1, 4], "channel_names": ["C3"], "sample_rate_hz": 200,
                    "quantum_uv": 1, "crop_start_sample": 0}
        return {"id": id_, "split": split, "source_id": "source-a",
            "canonical_subject_id": subject, "task_kind": "quantized_eeg_reconstruction",
            "reconstruction": {"observed": observed, "masked_indices": masked, "geometry": geometry},
            "messages": [{"role": "user", "content": runner.reconstruction_prompt(observed, masked, "uV_1_integer_raw_v1", geometry=geometry)},
                         {"role": "assistant", "content": "[2,3]"}]}

    def write_manifest(self):
        for split, records in (("train", self.train), ("val", self.val)):
            p = self.root / (split + ".jsonl")
            p.write_text("\n".join(json.dumps(x) for x in records) + "\n")
            self.manifest["inputs"][split] = {**self.entry(p), "count": len(records)}
        p = self.root / "manifest.json"
        p.write_text(json.dumps(self.manifest))
        self.cfg["data_manifest"] = self.entry(p)

    def test_valid_registered_subject_split(self):
        _, rows = runner.read_rows(self.cfg)
        self.assertEqual(len(rows["train"]), 1)

    def test_subject_overlap_rejected(self):
        self.val[0]["canonical_subject_id"] = "train-subject"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "subject_overlap"):
            runner.read_rows(self.cfg)

    def test_same_dataset_distinct_subject_allowed(self):
        # Dataset identity may overlap; person/recording identity may not.
        self.assertEqual(runner.read_rows(self.cfg)[1]["val"][0]["source_id"], "source-a")

    def test_recording_overlap_rejected(self):
        self.train[0]["recording_id"] = self.val[0]["recording_id"] = "same-recording"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "recording_overlap"):
            runner.read_rows(self.cfg)

    def test_test_input_never_opened(self):
        self.manifest["inputs"]["test"] = {"path": "/must/not/be/opened/test.jsonl", "sha256": "0" * 64}
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "only_train_and_val"):
            runner.read_rows(self.cfg)

    def test_protected_subject_rejected(self):
        self.manifest["protected_subject_keys"] = [runner.digest(["source-a", "train-subject"])]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "protected_subject"):
            runner.read_rows(self.cfg)

    def test_changed_payload_rejected(self):
        (self.root / "train.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            runner.read_rows(self.cfg)

    def test_input_cannot_reveal_masked_values(self):
        self.train[0]["reconstruction"]["observed"][1] = 2
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "mask_observation"):
            runner.read_rows(self.cfg)

    def test_unregistered_prompt_feature_rejected(self):
        self.train[0]["messages"][0]["content"] += "\nFull signal bandpower: 7"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "prompt_contract"):
            runner.read_rows(self.cfg)

    def test_boolean_and_float_targets_rejected(self):
        for target in ("[true,3]", "[2.0,3]"):
            self.train[0]["messages"][1]["content"] = target
            self.write_manifest()
            with self.assertRaisesRegex(ValueError, "reconstruction_target"):
                runner.read_rows(self.cfg)

    def test_mask_indices_must_be_unique_and_ordered(self):
        self.train[0]["reconstruction"]["masked_indices"] = [2, 1]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "masked_indices"):
            runner.read_rows(self.cfg)

    def test_unknown_source_and_task_rejected(self):
        self.train[0]["source_id"] = "not-admitted"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "not_admitted"):
            runner.read_rows(self.cfg)

    def test_no_implicit_epoch_repetition(self):
        self.cfg["max_updates"] = 2
        with self.assertRaisesRegex(ValueError, "silently_repeat"):
            runner.read_rows(self.cfg)
        self.cfg["allow_repeat_epochs"] = True
        runner.read_rows(self.cfg)

    def test_probe_repeats_maximum_length_only(self):
        cfg = {"mode": "engineering_probe"}
        self.assertEqual(runner.selected_order(cfg, [([1, 2], 1), ([1, 2, 3, 4], 2), ([1], 0)]), [1, 1])

    def test_probe_tests_full_registered_padded_shape(self):
        cfg = {"mode": "engineering_probe", "max_seq_length": 1024}
        self.assertEqual(runner.padded_length(cfg, 800), 1024)
        self.assertEqual(runner.padded_length({**cfg, "mode": "train_stage"}, 800), 801)
        with self.assertRaisesRegex(ValueError, "truncation"):
            runner.padded_length(cfg, 1025)
        self.assertEqual(runner.padded_length({**cfg, "max_seq_length": 801}, 683), 801)
        self.assertEqual(runner.padded_length({**cfg, "mode": "train_stage", "max_seq_length": 801}, 800), 801)

    def test_order_offset_is_continuous_not_restarted(self):
        cfg = {"mode": "train_stage", "seed": 11, "max_updates": 2, "order_offset": 2}
        processed = [([1, 2], 1)] * 6
        self.assertEqual(runner.selected_order(cfg, processed), runner.audited.training_order(6, 11, 4)[2:4])

    def test_streamed_later_stage_retains_only_selected_rows_and_fixed_development(self):
        self.train = [self.row(f"t{i}", "train", "train-subject") for i in range(620)]
        self.val = [self.row(f"v{i}", "val", "val-subject") for i in range(80)]
        self.cfg.update(max_updates=200, order_offset=400, max_seq_length=801)
        self.manifest.update(max_seq_length=801, max_channels_per_training_block=6)
        self.write_manifest()
        original_read_text = Path.read_text
        def forbid_full_jsonl(path, *args, **kwargs):
            if path.suffix == ".jsonl":
                raise AssertionError("JSONL must be streamed, not read_text")
            return original_read_text(path, *args, **kwargs)
        with patch.object(Path, "read_text", forbid_full_jsonl):
            report, _, rows, processed, local_order = runner.preflight(self.cfg, FakeTokenizer())
        expected = runner.audited.training_order(620, 123, 600)[400:600]
        self.assertEqual(report["selected_source_indices"], expected)
        self.assertEqual([rows["train"][i]["id"] for i in local_order], [f"t{i}" for i in expected])
        self.assertEqual(report["splits"]["train"]["count"], 620)
        self.assertEqual(report["splits"]["train"]["unique_ids"], 620)
        self.assertEqual(len(rows["train"]), 200)
        self.assertEqual(len(processed["train"]), 200)
        self.assertEqual(len(rows["val"]), 36)
        self.assertEqual(report["splits"]["val"]["count"], 80)
        again = runner.preflight(self.cfg, FakeTokenizer())[0]
        self.assertEqual(report["order_sha256"], again["order_sha256"])
        self.assertEqual(report["selected_tokenization_sha256"], again["selected_tokenization_sha256"])

    def test_streamed_probe_keeps_only_global_longest_even_at_file_end(self):
        self.train = [self.row(f"t{i}", "train", "train-subject") for i in range(50)]
        rec = self.train[-1]["reconstruction"]
        rec["geometry"]["channel_names"] = ["LONGEST_CHANNEL_IDENTIFIER"]
        self.train[-1]["messages"][0]["content"] = runner.reconstruction_prompt(
            rec["observed"], rec["masked_indices"], self.manifest["quantization_schema"], geometry=rec["geometry"])
        self.cfg.update(mode="engineering_probe", max_updates=2, max_seq_length=801)
        self.write_manifest()
        report, _, rows, processed, order = runner.preflight(self.cfg, FakeTokenizer())
        self.assertEqual(report["selected_source_indices"], [49, 49])
        self.assertEqual(order, [0, 0])
        self.assertEqual([r["id"] for r in rows["train"]], ["t49"])
        self.assertEqual(len(processed["train"]), 1)
        self.assertEqual(len(rows["val"]), 0)
        self.assertEqual(report["selected_padded_batch_lengths"], [801, 801])
        self.assertEqual(report["splits"]["train"]["count"], 50)

    def late_unselected_row(self):
        self.train = [self.row(f"t{i}", "train", "train-subject") for i in range(50)]
        self.cfg.update(max_updates=2, max_seq_length=801)
        selected = set(runner.selected_order(self.cfg, len(self.train)))
        index = max(i for i in range(len(self.train)) if i not in selected)
        return self.train[index]

    def test_unselected_late_heldout_membership_fails(self):
        self.late_unselected_row()["split"] = "test"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "split_membership"):
            runner.preflight(self.cfg, FakeTokenizer())

    def test_unselected_late_protected_subject_fails(self):
        self.late_unselected_row()["canonical_subject_id"] = "late-protected"
        self.manifest["protected_subject_keys"] = [runner.digest(["source-a", "late-protected"])]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "protected_subject"):
            runner.preflight(self.cfg, FakeTokenizer())

    def test_unselected_late_duplicate_id_fails(self):
        self.late_unselected_row()["id"] = "t0"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate_record_id"):
            runner.preflight(self.cfg, FakeTokenizer())

    def test_unselected_late_overlong_tokens_fail(self):
        row = self.late_unselected_row()
        rec = row["reconstruction"]
        rec["geometry"]["channel_names"] = ["L" * 900]
        row["messages"][0]["content"] = runner.reconstruction_prompt(
            rec["observed"], rec["masked_indices"], self.manifest["quantization_schema"], geometry=rec["geometry"])
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "truncated_or_empty"):
            runner.preflight(self.cfg, FakeTokenizer())

    def test_training_channel_cap_is_explicit_and_not_boolean(self):
        self.manifest["max_channels_per_training_block"] = True
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "training_channel_block"):
            runner.read_rows(self.cfg)

    def test_initial_model_comparison_uses_same_valid_targets(self):
        report = runner.compare_initial_task(["[3,4]", "bad"], ["[2,3]", "[100,100]"],
                                             self.train * 2, self.manifest)
        self.assertEqual(report["paired_valid_examples"], 1)
        self.assertEqual(report["initial_mse_paired_valid"], 1)
        self.assertEqual(report["current_mse_paired_valid"], 0)
        self.assertEqual(report["mse_delta_current_minus_initial"], -1)
        self.assertEqual(report["valid_rate_delta"], .5)
        invalid = runner.compare_initial_task(["bad"], ["[2,3]"], self.train, self.manifest)
        self.assertEqual(invalid["paired_valid_examples"], 0)
        self.assertIsNone(invalid["mse_delta_current_minus_initial"])

    def test_completion_prefix_and_target_mask_boundary(self):
        row = self.train[0]
        tokens, offset = runner.render(FakeTokenizer(), row)
        supervised_positions = list(range(offset, len(tokens)))
        self.assertEqual(supervised_positions[0], offset)
        self.assertEqual(tokens[supervised_positions[-1]], 3)  # EOS included
        self.assertNotIn(len(tokens), supervised_positions)   # first padding excluded
        self.assertEqual(tokens[offset], ord("[") + 10)

    def test_mismatched_prefix_is_rejected(self):
        class Broken(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                ids = super().apply_chat_template(messages, **kwargs)
                if len(messages) == 2:
                    ids[0] = 9
                return ids
        with self.assertRaisesRegex(ValueError, "prefix_mismatch"):
            runner.render(Broken(), self.train[0])

    def test_numeric_metrics_include_baselines_and_invalids(self):
        records = self.train * 3
        metrics = runner.reconstruction_metrics(["[2,3]", "[99,3]", "bad"], records, [-10, 10])
        self.assertEqual(metrics["valid_count"], 1)
        self.assertEqual(metrics["mse_on_valid_predictions"], 0)
        self.assertEqual(metrics["zero_baseline_mse_all_targets"], 6.5)
        self.assertEqual(metrics["visible_mean_baseline_mse_all_targets"], .25)
        self.assertEqual(metrics["visible_mean_baseline_mse_valid_targets"], .25)
        self.assertEqual(metrics["within_channel_interpolation_baseline_mse_valid_targets"], 0)
        self.assertAlmostEqual(metrics["invalid_rate"], 2/3)

    def test_interpolation_does_not_cross_channel_boundary(self):
        row = self.row("c", "train", "train-subject")
        row["reconstruction"] = {"observed": [1, 2, 3, None, None, 6, 7, 8],
                                 "masked_indices": [3, 4], "geometry": {"shape": [2, 4]}}
        with self.assertRaisesRegex(ValueError, "cross_channel"):
            runner.interpolation_baseline(row)

    def test_supervisor_signal_runs_finally(self):
        import signal
        cleaned = []
        with self.assertRaises(runner.SupervisorSignal):
            try:
                runner.supervisor_signal_handler(signal.SIGTERM, None)
            finally:
                cleaned.append(True)
        self.assertEqual(cleaned, [True])

    def test_warm_start_requires_completed_evaluation(self):
        d = self.root / "checkpoint"
        d.mkdir()
        (d / "adapters.safetensors").write_bytes(b"test")
        cfg = {"model_path": "/fake/base/model", "expected_model_config_sha256": "a" * 64,
               "base_receipt": {"sha256": "b" * 64}, "num_layers": 64, "rank": 8}
        meta = {"schema": runner.SCHEMA, "base_identity": runner.base_identity(cfg), "num_layers": 64,
                "rank": 8, "adapter_sha256": runner.sha256(d / "adapters.safetensors")}
        (d / "checkpoint.json").write_text(json.dumps(meta))
        (d / "evaluation.json").write_text(json.dumps({"status": "failed_quality_gates", "quality_gates_passed": False}))
        cfg["warm_start_adapter"] = {"checkpoint_dir": str(d), "adapter_sha256": meta["adapter_sha256"],
                                   "checkpoint_sha256": runner.sha256(d / "checkpoint.json")}
        with self.assertRaisesRegex(ValueError, "failed_or_unreviewed"):
            runner.warm_start_metadata(cfg)


if __name__ == "__main__":
    unittest.main()
