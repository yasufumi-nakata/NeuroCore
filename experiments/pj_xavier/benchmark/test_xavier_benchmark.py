"""Synthetic unit fixtures verify accounting only; these are not EEG results."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import xavier_benchmark as bench
import model_adapter


def fixture_record(identifier="r0", subject="s0", start=0., label="left"):
    return {
        "record_id": identifier, "dataset_id": "test-fixture-v1", "source_dataset_id": "test-fixture", "participant_id": subject,
        "session_id": "session1", "recording_id": f"recording-{subject}", "target_kind": "observed_action", "data_origin": "real_recording",
        "decision_time_s": start + 2., "minimum_lead_s": .25,
        "eeg": {"uri": "fixture://eeg", "sha256": hashlib.sha256(subject.encode()).hexdigest(), "start_s": start, "end_s": start + 2., "sample_rate_hz": 250},
        "label": {"value": label, "provenance": "event_derived", "evidence_uri": "fixture://event", "event_time_s": start + 2.25},
        "environment": {"uri": "fixture://screen", "sha256": "a" * 64, "available_until_s": start + 2., "contains_target_label": False},
        "clock_sync": {"verified": True, "method": "test fixture clock", "max_error_ms": 1.},
    }


class IntegrityTests(unittest.TestCase):
    def test_group_split_deterministic_and_disjoint(self):
        rows = [fixture_record(f"r{p}-{i}", f"s{p}", i * 30.) for p in range(12) for i in range(3)]
        first = bench.create_split(rows)
        second = bench.create_split(list(reversed(rows)))
        self.assertEqual(first["assignments"], second["assignments"])
        self.assertEqual(first["records_sha256"], second["records_sha256"])
        self.assertEqual(first["status"], "ready")
        self.assertFalse(bench.validate_split(rows, first))

    def test_single_subject_does_not_become_cross_subject_benchmark(self):
        rows = [fixture_record(f"r{i}", start=i * 30.) for i in range(30)]
        split = bench.create_split(rows)
        self.assertEqual(split["status"], "blocked")
        temporal = bench.create_split(rows, unit="temporal_block")
        self.assertEqual(temporal["status"], "ready")
        self.assertIn("no new-person", temporal["generalization_scope"])
        self.assertFalse(bench.validate_split(rows, temporal))

    def test_pseudo_labels_never_become_test_gold(self):
        rows = [fixture_record(f"r{i}", f"s{i}") for i in range(15)]
        initial = bench.create_split(rows)
        test_id = next(rid for rid, split in initial["assignments"].items() if split == "test")
        row = next(row for row in rows if row["record_id"] == test_id)
        row["label"].update(provenance="ai_pseudo", teacher_id="teacher", teacher_revision="rev", prompt_sha256="f" * 64)
        final = bench.create_split(rows)
        self.assertNotIn(test_id, final["assignments"])
        self.assertIn(test_id, final["excluded"])

    def test_same_raw_interval_under_two_subject_names_is_rejected(self):
        rows = [fixture_record("r1", "s1"), fixture_record("r2", "s2"), fixture_record("r3", "s3")]
        rows[1]["eeg"]["sha256"] = rows[0]["eeg"]["sha256"]
        split = {"records_sha256": bench.record_digest(rows), "unit": "participant", "embargo_s": 10, "assignments": {"r1": "train", "r2": "test", "r3": "validation"}, "excluded": {}}
        self.assertTrue(any("shared EEG interval" in error for error in bench.validate_split(rows, split)))

    def test_future_context_and_concurrent_action_rejected(self):
        row = fixture_record()
        row["environment"]["available_until_s"] = row["decision_time_s"] + .01
        row["label"]["event_time_s"] = row["decision_time_s"]
        errors = bench.validate_records([row])["errors"]
        self.assertTrue(any("future" in error for error in errors))
        self.assertTrue(any("post-action" in error for error in errors))

    def test_observed_action_not_relabeled_as_hidden_intent(self):
        row = fixture_record()
        row["target_kind"] = "explicit_intent"
        self.assertTrue(any("unobserved intent" in error for error in bench.validate_records([row])["errors"]))

    def test_changed_record_invalidates_frozen_split(self):
        rows = [fixture_record(f"r{i}", f"s{i}") for i in range(6)]
        split = bench.create_split(rows)
        rows[0]["label"]["value"] = "right"
        self.assertIn("records differ from frozen split manifest", bench.validate_split(rows, split))

    def test_training_record_from_test_detected(self):
        rows = [fixture_record(f"r{i}", f"s{i}") for i in range(6)]
        split = bench.create_split(rows)
        test_id = next(rid for rid, name in split["assignments"].items() if name == "test")
        manifest = {"training_record_ids": [test_id], "split_manifest_sha256": bench.canonical_hash(split), "checkpoint_sha256": "b" * 64, "model_selection_partition": "validation", "test_used_for_tuning": False}
        self.assertTrue(any("outside training" in error for error in bench.training_checks(manifest, split, "b" * 64)))


class MetricTests(unittest.TestCase):
    def test_perfect_action_classification_does_not_validate_agent_goal(self):
        rows = [fixture_record(f"r{p}-{i}", f"s{p}", i * 30., ("left", "right")[i % 2]) for p in range(6) for i in range(2)]
        for row in rows:
            row["environment"] = None
        split = bench.create_split(rows)
        training = {"training_record_ids": [rid for rid, name in split["assignments"].items() if name == "train"], "split_manifest_sha256": bench.canonical_hash(split), "checkpoint_sha256": "b" * 64, "model_selection_partition": "validation", "test_used_for_tuning": False}
        predictions = []
        for row in rows:
            if split["assignments"][row["record_id"]] == "test":
                predictions.append({"record_id": row["record_id"], "variant": "eeg_only", "checkpoint_id": "fixture", "checkpoint_sha256": "b" * 64, "probabilities": {name: float(name == row["label"]["value"]) for name in ("left", "right")}})
        report = bench.evaluate(rows, split, predictions, training)
        self.assertEqual(report["metrics"]["eeg_only"]["macro_f1"], 1.)
        self.assertEqual(report["status"], "evaluated_goal_unproven")
        self.assertFalse(report["autonomous_agent_validated"])
        self.assertFalse(report["held_out_replay_validated"])
        self.assertIsNone(report["harness_replay"]["task_completion_rate"])

    def test_abstention_counts_as_false_negative_and_no_hidden_accuracy_gain(self):
        rows = [("A", {"A": .8, "B": .2}, False), ("A", {"A": .4, "B": .6}, False), ("B", {"A": .3, "B": .7}, True)]
        result = bench.classification_metrics(rows, ["A", "B"])
        self.assertAlmostEqual(result["forced_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["accepted_correct_over_all"], 1 / 3)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        self.assertAlmostEqual(result["selective_accuracy"], .5)
        self.assertAlmostEqual(result["balanced_accuracy"], .25)
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertAlmostEqual(result["brier_multiclass_sum"], .98 / 3)
        self.assertAlmostEqual(result["expected_calibration_error"], 1.1 / 3)

    def test_single_subject_has_no_population_confidence_interval(self):
        result = bench.bootstrap_increment([fixture_record()], {}, "fused", "context_only", ["left"])
        self.assertEqual(result["status"], "unavailable")

    def test_resource_budget_cannot_pass_with_more_memory_than_target(self):
        resource = {"measurement_mode": "actual", "checkpoint_sha256": "b" * 64, "machine_id": "fixture", "runtime": "test", "accelerator": "CPU", "measurement_method": "fixture", "memory_measurement_scope": "process_peak_rss", "hardware_compatible": True, "inference_succeeded": True, "peak_memory_bytes": 17, "memory_budget_bytes": 16, "p95_latency_ms": 10}
        self.assertEqual(bench.resource_checks(resource, "b" * 64)["status"], "failed")

    def test_no_idle_data_means_no_false_activation_rate(self):
        result = bench.replay_metrics([], None, set(), "b" * 64)
        self.assertIsNone(result["false_actions_per_minute"])
        self.assertIsNone(result["task_completion_rate"])

    def test_replay_verifies_trace_and_unions_idle_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = [
                {"kind": "observed_idle_interval", "independent_annotation": True, "stream_id": "stream1", "start_s": 0, "end_s": 60},
                {"kind": "observed_idle_interval", "independent_annotation": True, "stream_id": "stream1", "start_s": 30, "end_s": 90},
                {"kind": "executed_action", "is_false_activation": True, "independent_annotation": True, "stream_id": "stream1", "event_id": "action1", "time_s": 40},
                {"kind": "terminal_assertion", "goal_id": "goal", "passed": True},
            ]
            path = Path(directory) / "trace.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in trace))
            episode = {"episode_id": "e1", "checkpoint_sha256": "b" * 64, "variant": "fused", "record_ids": ["r1"], "ground_truth_source": "independent_harness_assertion", "environment_goal_seen_in_training": False, "action_logging_complete": True, "trace_path": "trace.jsonl", "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "goal_id": "goal"}
            result = bench.replay_metrics([episode, episode], directory, {"r1"}, "b" * 64)
            self.assertEqual(result["valid_episodes"], 1)
            self.assertEqual(len(result["rejected_episodes"]), 1)
            self.assertEqual(result["idle_seconds_observed"], 90)
            self.assertAlmostEqual(result["false_actions_per_minute"], 2 / 3)
            del episode["action_logging_complete"]
            self.assertIsNone(bench.replay_metrics([episode], directory, {"r1"}, "b" * 64)["false_actions_per_minute"])
            episode["trace_sha256"] = "0" * 64
            self.assertEqual(bench.replay_metrics([episode], directory, {"r1"}, "b" * 64)["status"], "blocked")

    def test_adapter_does_not_invent_real_origin_or_context(self):
        flat = {"id": "r1", "dataset": "d", "subject": "s", "session": "session", "recording_id": "rec", "label": "left", "label_source": "recorded_event"}
        record = model_adapter.adapt_records([flat])[0]
        self.assertIsNone(record["data_origin"])
        self.assertIsNone(record["environment"])
        self.assertTrue(bench.validate_records([record])["errors"])


if __name__ == "__main__":
    unittest.main()
