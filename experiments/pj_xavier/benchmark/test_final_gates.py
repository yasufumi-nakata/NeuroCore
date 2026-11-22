"""Regression fixtures for concrete audit defects; no empirical EEG claims."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import model_adapter
import xavier_benchmark as bench
from test_xavier_benchmark import fixture_record


class FinalGateTests(unittest.TestCase):
    def test_legacy_decision_migration_changes_only_metadata_and_rejects_future_eeg(self):
        original = {"id": "r0", "decision_time_s": 10., "event_time_s": 10., "window_start_s": 7.5, "window_end_s": 9.75,
                    "minimum_lead_s": .25, "label_source": "recorded_event", "target_kind": "observed_action", "label": "keyboard", "split": "train"}
        before = copy.deepcopy(original)
        fixed, migration = model_adapter.normalize_legacy_decisions([original], "a" * 64)
        self.assertEqual(original, before)
        self.assertEqual(fixed[0]["decision_time_s"], 9.75)
        self.assertTrue(migration["labels_and_split_unchanged"])
        self.assertFalse(migration["source_manifest_bytes_modified"])
        self.assertEqual({k: v for k, v in fixed[0].items() if k not in ("decision_time_s", "_timestamp_metadata_migration")},
                         {k: v for k, v in original.items() if k != "decision_time_s"})
        original["window_end_s"] = 9.9
        with self.assertRaisesRegex(ValueError, "legacy migration requires"):
            model_adapter.normalize_legacy_decisions([original], "a" * 64)

    def test_temporal_split_cannot_reverse_train_and_test(self):
        records = [fixture_record(f"r{i}", start=30. * i) for i in range(30)]
        split = bench.create_split(records, unit="temporal_block")
        self.assertFalse(bench.validate_split(records, split))
        split["assignments"] = {rid: {"train": "test", "test": "train"}.get(part, part) for rid, part in split["assignments"].items()}
        self.assertTrue(any("chronological" in error for error in bench.validate_split(records, split)))

    def test_negative_or_nan_embargo_is_rejected(self):
        records = [fixture_record(f"r{i}", f"s{i}") for i in range(6)]
        split = bench.create_split(records)
        for embargo in (-1., float("nan")):
            split["embargo_s"] = embargo
            self.assertIn("finite nonnegative embargo required", bench.validate_split(records, split))

    def test_subject_session_changes_cannot_reuse_a_frozen_split(self):
        records = [fixture_record(f"r{i}", f"s{i}") for i in range(6)]
        split = bench.create_split(records)
        for field in ("participant_id", "session_id", "source_dataset_id"):
            altered = copy.deepcopy(records)
            altered[0][field] += "-modified"
            self.assertIn("records differ from frozen split manifest", bench.validate_split(altered, split))

    def test_relabeling_subject_does_not_hide_the_same_raw_interval(self):
        records = [fixture_record(f"r{i}", f"s{i}") for i in range(3)]
        for row in records:
            row["eeg"]["sha256"] = "d" * 64
        split = {"unit": "participant", "records_sha256": bench.record_digest(records), "embargo_s": 0,
                 "assignments": {"r0": "train", "r1": "validation", "r2": "test"}, "excluded": {}}
        self.assertTrue(any("shared EEG interval" in error for error in bench.validate_split(records, split)))

    def test_unknown_or_offline_preprocessing_blocks_and_future_support_is_checked(self):
        row = fixture_record()
        for unavailable in (None, {}, {"online_causality_verified": False}):
            row["preprocessing"] = unavailable
            self.assertEqual(bench.preprocessing_checks([row])["status"], "blocked")
        row["preprocessing"] = {"online_causality_verified": True, "evidence_uri": "fixture://proof", "max_future_support_s": 4.125}
        self.assertEqual(bench.preprocessing_checks([row])["status"], "blocked")
        row["decision_time_s"] = row["eeg"]["end_s"] + 4.125
        self.assertEqual(bench.preprocessing_checks([row])["status"], "passed")
        row["preprocessing"]["max_future_support_s"] = 0
        del row["preprocessing"]["evidence_uri"]
        self.assertEqual(bench.preprocessing_checks([row])["status"], "blocked")

    def controls(self):
        records = [fixture_record(f"r{i}", start=i * 30.) for i in range(3)]
        predictions = {}
        for i, row in enumerate(records):
            source = records[(i + 1) % len(records)]
            for variant in ("eeg_shuffled", "eeg_time_shifted"):
                detail = {"eeg_source_id": source["record_id"], "original_eeg_id": row["record_id"],
                          "method": "nonzero_cyclic_shift" if variant == "eeg_shuffled" else "recording_time_shift",
                          "shift_seconds": source["eeg"]["start_s"] - row["eeg"]["start_s"]}
                predictions[(variant, row["record_id"])] = {"perturbation": detail}
        return records, predictions

    def test_valid_logged_controls_and_identity_control_rejected(self):
        records, predictions = self.controls()
        self.assertTrue(all(v["status"] == "passed" for v in bench.negative_control_checks(records, predictions).values()))
        predictions[("eeg_shuffled", "r0")]["perturbation"]["eeg_source_id"] = "r0"
        self.assertEqual(bench.negative_control_checks(records, predictions)["eeg_shuffled"]["status"], "failed")

    def test_repeated_shuffle_source_and_cross_session_control_rejected(self):
        records, predictions = self.controls()
        predictions[("eeg_shuffled", "r2")]["perturbation"]["eeg_source_id"] = "r1"
        self.assertEqual(bench.negative_control_checks(records, predictions)["eeg_shuffled"]["status"], "failed")
        records, predictions = self.controls()
        records[1]["session_id"] = "different-session"
        self.assertTrue(all(v["status"] == "failed" for v in bench.negative_control_checks(records, predictions).values()))

    def test_fabricated_time_shift_amount_is_rejected(self):
        records, predictions = self.controls()
        predictions[("eeg_time_shifted", "r0")]["perturbation"]["shift_seconds"] = 0.
        self.assertEqual(bench.negative_control_checks(records, predictions)["eeg_time_shifted"]["status"], "failed")

    def test_adapter_preserves_canonical_source_and_unverified_preprocessing(self):
        row = {"id": "r", "dataset": "mirror", "canonical_dataset_id": "original", "canonical_subject_id": "canonical-person", "label_source": "recorded_event",
               "preprocessing": {"online_causality_verified": False}, "environment_text": "prior context", "environment_scope": "task_protocol"}
        record = model_adapter.adapt_records([row])[0]
        self.assertEqual(record["source_dataset_id"], "original")
        self.assertFalse(record["preprocessing"]["online_causality_verified"])
        self.assertEqual(record["environment"]["scope"], "task_protocol")
        self.assertEqual(bench.group_key(record, "participant"), ("original", "canonical-person"))

    def test_adapter_rejects_modified_original_subject_or_membership(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.jsonl"
            original = {"id": "r0", "subject": "s0", "session": "one", "split": "train"}
            path.write_text(json.dumps(original) + "\n")
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            membership = {"manifest_sha256": checksum, "training_ids": ["r0"], "validation_ids": [], "test_ids": [],
                          "training_ids_sha256": hashlib.sha256(b"r0").hexdigest()}
            split = {"assignments": {"r0": "train"}}
            metadata = {"manifest_sha256": checksum, "checkpoint_sha256": "b" * 64}
            self.assertEqual(model_adapter.adapt_training(membership, path, split, metadata)["source_flat_manifest_sha256"], checksum)
            original["session"] = "changed"
            path.write_text(json.dumps(original) + "\n")
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                model_adapter.adapt_training(membership, path, split, metadata)

    def test_measured_resource_pass_is_separate_from_target_link_and_full_latency(self):
        resource = {"measurement_mode": "actual", "checkpoint_sha256": "b" * 64, "machine_id": "fixture", "runtime": "test", "accelerator": "CPU",
                    "measurement_method": "fixture", "memory_measurement_scope": "process_peak_rss", "hardware_compatible": True, "inference_succeeded": True,
                    "peak_memory_bytes": 8, "memory_budget_bytes": 16, "memory_capacity_bytes": 32, "p95_latency_ms": 10,
                    "latency_scope": "candidate_inference_only"}
        result = bench.resource_checks(resource, "b" * 64)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["target_environment_evidence"], "unverified")
        resource.update(target_environment_verified=True, target_environment_id="fixture-colab", hardware_evidence_sha256="c" * 64)
        self.assertEqual(bench.resource_checks(resource, "b" * 64)["target_environment_evidence"], "recorded")
        resource["memory_budget_bytes"] = 64
        self.assertEqual(bench.resource_checks(resource, "b" * 64)["status"], "failed")
        resource["memory_budget_bytes"] = 16
        resource.update(host_peak_rss_bytes=65, host_ram_bytes=64)
        self.assertEqual(bench.resource_checks(resource, "b" * 64)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
