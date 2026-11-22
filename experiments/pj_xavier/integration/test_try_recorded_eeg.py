import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import try_recorded_eeg as demo


class UnsupportedCheckpointObject:
    pass


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch is needed for safe checkpoint fixtures")
class RecordedDemoTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        self.np, self.torch = np, torch
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.run = self.repo / demo.RUN
        self.manifest = self.repo / demo.MANIFEST
        self.run.mkdir(parents=True)
        self.manifest.parent.mkdir(parents=True)
        for name in ("model/eeg_llm.py", "integration/replay_to_neurocore.py"):
            file = self.repo / "experiments/pj_xavier" / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("# immutable CPU fixture source; never executed\n")
        for name in ("audit.py", "control.py", "settings.py"):
            file = self.repo / "backend/neurocore" / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("# fixture backend\n")
        array = self.manifest.parent / "waveforms/fixture.npy"
        array.parent.mkdir()
        np.save(array, np.arange(9 * 8 * 448, dtype=np.float32).reshape(9, 8, 448))
        self.rows = []
        for i, split in enumerate(("train", "val", "test")):
            for j, label in enumerate(demo.CLASSES):
                self.rows.append({"id": split + "-" + str(j), "split": split, "label": label,
                    "dataset": "fixture", "subject": "subject-" + split, "session": "session-1",
                    "eeg_path": "waveforms/fixture.npy", "eeg_index": 3 * i + j, "data_origin": "real_recording",
                    "label_source": "recorded_event", "window_start_s": 0., "window_end_s": 1.75,
                    "decision_time_s": 2., "environment_source": "predecision_harness", "environment_time_s": 0.})
        self.manifest.write_bytes(demo.manifest_bytes(self.rows))
        manifest_sha = demo.sha256(self.manifest)
        self.config = {"mode": "fused", "lora_rank": 0, "abstain_threshold": .8, "device": "mps",
                       "dtype": "bfloat16", "memory_budget_gib": 8, "candidate_batch_size": 2, "channels": 8, "samples": 448}
        self.artifacts = {"waveforms/fixture.npy": {"sha256": demo.sha256(array), "bytes": array.stat().st_size}}
        self.state = {"schema": "xavier-eeg-llm-v1", "manifest_sha256": manifest_sha, "config": self.config,
                      "classes": demo.CLASSES, "eeg_artifacts": self.artifacts, "trainable_base": {},
                      "connector": {"weight": torch.tensor([1., 2.])}}
        torch.save(self.state, self.run / "best.pt")
        membership = {"manifest_sha256": manifest_sha, "model_selection_partition": "validation", "test_used_for_tuning": False}
        for split, key in (("train", "training_ids"), ("val", "validation_ids"), ("test", "test_ids")):
            membership[key] = sorted(r["id"] for r in self.rows if r["split"] == split)
        membership["training_ids_sha256"] = hashlib.sha256("\n".join(membership["training_ids"]).encode()).hexdigest()
        summary = {"status": "completed_epochs", "checkpoint_sha256": demo.sha256(self.run / "best.pt"),
                   "manifest_sha256": manifest_sha, "test_used_for_tuning": False, "model_selection_partition": "validation"}
        exited = {"exit_code": 0, "manifest_sha256": manifest_sha,
                  "source_sha256": demo.sha256(self.repo / "experiments/pj_xavier/model/eeg_llm.py")}
        for name, value in {"run_summary.json": summary, "run_config.json": self.config,
                            "training_membership.json": membership, "eeg_artifacts.json": self.artifacts,
                            "exit_receipt.json": exited}.items():
            demo.write_new_json(self.run / name, value)
        self.pins = {name: demo.sha256(self.run / name) for name in
                     ("best.pt", "run_summary.json", "run_config.json", "training_membership.json", "eeg_artifacts.json", "exit_receipt.json")}
        self.pins.update(manifest=manifest_sha,
                         model_source=demo.sha256(self.repo / "experiments/pj_xavier/model/eeg_llm.py"),
                         replay_source=demo.sha256(self.repo / "experiments/pj_xavier/integration/replay_to_neurocore.py"))
        self.output = self.repo / "demo"

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self):
        return demo.prepare(self.repo, self.run, self.manifest, self.output, pins=self.pins)

    def finish_evaluation(self):
        exit_path = self.repo / demo.EVALUATION_EXIT
        exit_path.parent.mkdir(parents=True, exist_ok=True)
        demo.write_new_json(exit_path, {"exit_code": 0})
        report = {"schema_version": "pj-xavier-variant-evaluation-v1", "status": "completed",
                  "checkpoint_sha256": self.pins["best.pt"], "variant_names": sorted(demo.VARIANTS),
                  "conditions": {k: {"status": "blocked", "reason": "CPU fixture only"} for k in demo.VARIANTS}}
        file = self.run / "variant_report.json"
        demo.write_new_json(file, report)
        demo.write_new_json(self.run / "variant_evaluation_completion.json", {"status": "completed", "exit_code": 0,
            "checkpoint_sha256": self.pins["best.pt"], "variant_names": sorted(demo.VARIANTS),
            "report_path": str(file), "report_sha256": demo.sha256(file)})

    def test_prepare_uses_cpu_weights_only_and_train_examples_without_subprocess(self):
        load = self.torch.load
        with patch.object(self.torch, "load", wraps=load) as safe_load, patch.object(subprocess, "run", side_effect=AssertionError("no inference in prepare")):
            plan = self.prepare()
        self.assertTrue(safe_load.call_args.kwargs["weights_only"])
        self.assertEqual(safe_load.call_args.kwargs["map_location"], "cpu")
        selected = [json.loads(x) for x in (self.output / "train_examples.jsonl").read_text().splitlines()]
        self.assertEqual({r["split"] for r in selected}, {"train"})
        self.assertEqual([r["label"] for r in selected], demo.CLASSES)
        self.assertTrue(all(Path(r["eeg_path"]).is_absolute() for r in selected))
        self.assertNotIn("--human-armed", plan["replay_command"])
        self.assertFalse((self.output / "execution_started.json").exists())

    def test_array_mutation_is_rejected_before_output_directory_exists(self):
        array = self.manifest.parent / "waveforms/fixture.npy"
        with array.open("r+b") as f:
            f.seek(-1, 2);f.write(b"X")
        with self.assertRaisesRegex(ValueError, "EEG bytes changed"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_membership_cannot_move_heldout_record_into_train(self):
        path = self.run / "training_membership.json"
        value = demo.read_json(path);value["training_ids"][0] = "test-0"
        path.write_text(json.dumps(value));self.pins[path.name] = demo.sha256(path)
        with self.assertRaisesRegex(ValueError, "source split"):
            self.prepare()

    def test_non_weights_checkpoint_object_is_never_unsafe_loaded(self):
        path = self.run / "best.pt";self.torch.save(UnsupportedCheckpointObject(), path)
        self.pins["best.pt"] = demo.sha256(path)
        s = self.run / "run_summary.json";value = demo.read_json(s);value["checkpoint_sha256"] = self.pins["best.pt"]
        s.write_text(json.dumps(value));self.pins[s.name] = demo.sha256(s)
        with self.assertRaises(Exception) as failure:
            self.prepare()
        self.assertIn("Weights only", str(failure.exception))
        self.assertFalse(self.output.exists())

    def test_existing_output_is_preserved(self):
        self.output.mkdir();marker = self.output / "existing.txt";marker.write_text("retain")
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            self.prepare()
        self.assertEqual(marker.read_text(), "retain")

    def test_heldout_demo_injection_is_rejected_even_with_rehashed_plan(self):
        plan = self.prepare()
        selected = dict(self.rows[-1], eeg_path=str(self.manifest.parent / self.rows[-1]["eeg_path"]))
        (self.output / "train_examples.jsonl").write_bytes(demo.manifest_bytes([selected]))
        plan["train_examples_sha256"] = demo.sha256(self.output / "train_examples.jsonl")
        (self.output / "plan.json").write_text(json.dumps(plan))
        (self.output / "plan.sha256.json").write_text(json.dumps({"sha256": demo.sha256(self.output / "plan.json")}))
        with self.assertRaisesRegex(ValueError, "training-only selection"):
            demo.execute_prepared(self.output, pins=self.pins, runner=lambda *a, **k: self.fail("must not run"))

    def test_changed_prepared_command_cannot_execute_arbitrary_process(self):
        plan = self.prepare();plan["predict_command"] = ["different-command"]
        (self.output / "plan.json").write_text(json.dumps(plan))
        (self.output / "plan.sha256.json").write_text(json.dumps({"sha256": demo.sha256(self.output / "plan.json")}))
        with self.assertRaisesRegex(ValueError, "fixed predictor"):
            demo.execute_prepared(self.output, pins=self.pins, runner=lambda *a, **k: self.fail("must not run"))

    def test_incomplete_evaluation_blocks_before_start_marker(self):
        self.prepare()
        with self.assertRaises(FileNotFoundError):
            demo.execute_prepared(self.output, pins=self.pins, runner=lambda *a, **k: self.fail("must not run"))
        self.assertFalse((self.output / "execution_started.json").exists())

    def test_existing_started_marker_prevents_duplicate_inference(self):
        self.prepare();self.finish_evaluation()
        demo.write_new_json(self.output / "execution_started.json", {"old_attempt": True})
        with self.assertRaisesRegex(ValueError, "automatic rerun refused"):
            demo.execute_prepared(self.output, pins=self.pins, runner=lambda *a, **k: self.fail("must not run"))


if __name__ == "__main__":
    unittest.main()
