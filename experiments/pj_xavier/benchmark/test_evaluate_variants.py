"""CPU-only synthetic BrainVision fixtures; no EEG performance claims or GPU use."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import evaluate_variants as variants
import xavier_benchmark as bench


HAS_SIGNAL_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch", "mne"))


def make_fixture(root):
    import numpy as np
    import torch
    import mne
    root = Path(root)
    rawroot, prepared, output = root / "raw", root / "prepared", root / "controls"
    for path in (rawroot, prepared, output):
        path.mkdir()
    header, binary, marker = rawroot / "rec.vhdr", rawroot / "rec.eeg", rawroot / "rec.vmrk"
    samples = np.stack([np.arange(512, dtype=np.float32), np.arange(512, dtype=np.float32) ** 2], axis=1)
    samples.astype("<f4").tofile(binary)
    header.write_text("Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\nDataFile=rec.eeg\nMarkerFile=rec.vmrk\nDataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=2\nSamplingInterval=125000\n[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\nCh1=CP3,,1,uV\nCh2=CP4,,1,uV\n")
    marker.write_text("Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\nDataFile=rec.eeg\n[Marker Infos]\nMk1=New Segment,,1,1,0\n")
    raw = mne.io.read_raw_brainvision(header, preload=False, verbose="ERROR")
    rows, arrays = [], []
    for index, (partition, start) in enumerate((("train", 1.), ("val", 5.), ("test", 20.), ("test", 24.))):
        row = {"id": f"r{index}", "dataset": "fixture-version", "canonical_dataset_id": "fixture-source", "site": "uniba",
               "subject": "testperson" if partition == "test" else partition + "person", "session": "s1", "recording_id": "rec",
               "split": partition, "eeg_path": "waveforms.npy", "eeg_index": index, "channels": ["CP3", "CP4"], "sample_rate_hz": 8.,
               "window_start_s": start, "window_end_s": start + 1.75, "decision_time_s": start + 5.875,
               "label": "keyboard" if index % 2 == 0 else "mouse_click", "label_source": "recorded_event",
               "eeg_source_uri": "https://fixture.invalid/source.tar.gz#rec.eeg", "eeg_source_sha256": variants.file_hash(binary),
               "assumed_filter_lookahead_s": 4.125}
        rows.append(row)
        arrays.append(raw.get_data(picks=row["channels"], start=int(start * 8), stop=int((start + 1.75) * 8)).astype(np.float32))
    np.save(prepared / "waveforms.npy", np.asarray(arrays), allow_pickle=False)
    manifest = prepared / "manifest.jsonl"
    variants.write_jsonl(manifest, rows)
    state = {"schema": "xavier-eeg-llm-v1", "config": {"mode": "fused"}, "manifest_sha256": variants.file_hash(manifest),
             "eeg_artifacts": {"waveforms.npy": {"sha256": variants.file_hash(prepared / "waveforms.npy"), "bytes": (prepared / "waveforms.npy").stat().st_size}}}
    checkpoint = root / "best.pt"
    torch.save(state, checkpoint)
    variants.write_json(rawroot / "source_verification.json", {"verified": True, "sources": [
        {"filename": f"bids_{kind}_uniba.tar.gz", "ledger_status": "complete", "reverified_before_extraction": True}
        for kind in ("eeg", "behavioral")]})
    membership = {"manifest_sha256": variants.file_hash(manifest), "training_ids": ["r0"], "validation_ids": ["r1"], "test_ids": ["r2", "r3"],
                  "training_ids_sha256": hashlib.sha256(b"r0").hexdigest(), "model_selection_partition": "validation", "test_used_for_tuning": False}
    return {"root": root, "rawroot": rawroot, "prepared": prepared, "output": output, "manifest": manifest,
            "checkpoint": checkpoint, "state": state, "rows": rows, "test": rows[2:], "membership": membership}


def prepare(f):
    return variants.prepare_raw_shifts(f["test"], f["manifest"], f["checkpoint"], f["state"], f["rawroot"], f["output"], f["root"])


def nested(row):
    return {"record_id": row["id"], "source_dataset_id": row["canonical_dataset_id"], "participant_id": row["subject"],
            "session_id": row["session"], "recording_id": row["recording_id"],
            "eeg": {"sha256": row["eeg_source_sha256"], "start_s": row["window_start_s"], "end_s": row["window_end_s"]}}


def verify(f, detail):
    return variants.verify_raw_shift_certificate(detail, nested(f["test"][0]), f["root"], variants.file_hash(f["checkpoint"]), variants.file_hash(f["manifest"]))


@unittest.skipUnless(HAS_SIGNAL_RUNTIME, "CPU numpy/torch/MNE signal runtime required")
class RawShiftTests(unittest.TestCase):
    def test_actual_raw_shift_and_checkpoint_sealed_original_are_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            f = make_fixture(folder)
            certs, report, _ = prepare(f)
            self.assertEqual(report["status"], "ready")
            self.assertEqual(verify(f, certs["r2"])["status"], "passed")
            detail = certs["r2"]
            self.assertEqual(detail["source"]["start_sample"], 80)
            self.assertEqual(detail["source"]["stop_sample"], 94)
            self.assertNotIn(detail["eeg_source_id"], {row["id"] for row in f["rows"]})
            self.assertEqual(detail["original_eeg_id"], "r2")
            self.assertFalse(detail["source"]["online_causality_verified"])
            predictions = {("eeg_time_shifted", rid): {"perturbation": cert, "checkpoint_sha256": variants.file_hash(f["checkpoint"])} for rid, cert in certs.items()}
            controls = bench.negative_control_checks([nested(row) for row in f["test"]], predictions, f["root"], variants.file_hash(f["manifest"]))
            self.assertEqual(controls["eeg_time_shifted"]["status"], "passed")

    def test_changed_raw_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f = make_fixture(folder)
            certs, _, _ = prepare(f)
            path = f["rawroot"] / "rec.eeg"
            data = bytearray(path.read_bytes())
            data[0] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                verify(f, certs["r2"])

    def test_wrong_extracted_values_rejected_even_after_rehash(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            f = make_fixture(folder)
            certs, _, _ = prepare(f)
            detail = certs["r2"]
            path = f["root"] / detail["source"]["extracted_array_path"]
            np.save(path, np.zeros((2, 14), dtype=np.float32), allow_pickle=False)
            detail["source"]["extracted_array_sha256"] = variants.file_hash(path)
            with self.assertRaisesRegex(ValueError, "independently extracted"):
                verify(f, detail)

    def test_changed_header_gain_fails_original_prepared_window_match(self):
        with tempfile.TemporaryDirectory() as folder:
            f = make_fixture(folder)
            certs, _, _ = prepare(f)
            detail = certs["r2"]
            header = f["rawroot"] / "rec.vhdr"
            header.write_text(header.read_text().replace("Ch1=CP3,,1,uV", "Ch1=CP3,,2,uV"))
            detail["source"]["raw_vhdr_sha256"] = variants.file_hash(header)
            with self.assertRaisesRegex(ValueError, "does not reproduce"):
                verify(f, detail)

    def test_prepared_input_mutation_cannot_be_hidden_by_a_source_certificate(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            f = make_fixture(folder)
            certs, _, _ = prepare(f)
            np.save(f["prepared"] / "waveforms.npy", np.zeros((4, 2, 14), dtype=np.float32), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                verify(f, certs["r2"])

    def test_identity_time_sample_subject_and_path_forgery_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            f = make_fixture(folder)
            certs, _, _ = prepare(f)
            modifications = [
                ("existing ID", lambda d: d.update(eeg_source_id="r3")),
                ("wrong seconds", lambda d: d.update(shift_seconds=-9.)),
                ("wrong sample", lambda d: d["source"].update(start_sample=81)),
                ("different person", lambda d: d["source"]["recording_identity"].__setitem__(1, "other")),
                ("different session", lambda d: d["source"]["recording_identity"].__setitem__(2, "other")),
                ("escaped path", lambda d: d["source"].update(extracted_array_path="../outside.npy")),
            ]
            for name, change in modifications:
                with self.subTest(name=name):
                    detail = copy.deepcopy(certs["r2"])
                    change(detail)
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        verify(f, detail)

    def test_out_of_recording_and_unverified_source_are_blocked_not_filled(self):
        with tempfile.TemporaryDirectory() as folder:
            f = make_fixture(folder)
            target = copy.deepcopy(f["test"][0])
            target.update(window_start_s=5., window_end_s=6.75)
            f["test"] = [target]
            certificates, report, _ = prepare(f)
            self.assertEqual(report["status"], "blocked")
            self.assertFalse(certificates)
        with tempfile.TemporaryDirectory() as folder:
            f = make_fixture(folder)
            (f["rawroot"] / "source_verification.json").unlink()
            certificates, report, _ = prepare(f)
            self.assertEqual(report["status"], "blocked")
            self.assertFalse(certificates)

    def test_membership_and_exact_test_count_are_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            f = make_fixture(folder)
            sha = variants.file_hash(f["manifest"])
            self.assertEqual(len(variants.verify_membership(f["rows"], f["membership"], sha, f["state"], 2)), 2)
            with self.assertRaisesRegex(ValueError, "record count"):
                variants.verify_membership(f["rows"], f["membership"], sha, f["state"], 3)
            f["membership"]["test_ids"] = ["r2", "r0"]
            with self.assertRaisesRegex(ValueError, "test_ids"):
                variants.verify_membership(f["rows"], f["membership"], sha, f["state"], 2)

    def test_input_removal_conditions_keep_the_same_targets_and_shift_real_input(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            f = make_fixture(folder)
            certs, preparation, _ = prepare(f)
            test = [{**row, "eeg_path": str(f["prepared"] / row["eeg_path"])} for row in f["test"]]
            calls = []
            class Store:
                def __init__(self):
                    self.cache = {}
                def get(self, row):
                    if row["eeg_path"] not in self.cache:
                        self.cache[row["eeg_path"]] = np.load(row["eeg_path"], allow_pickle=False)
                    value = self.cache[row["eeg_path"]]
                    return value[row["eeg_index"]] if "eeg_index" in row else value
            def evaluate(model, tokenizer, rows, classes, store, config, device, checkpoint, condition, include_truth):
                source = list(reversed(rows)) if condition == "shuffled_eeg" else rows
                arrays = [store.get(row) for row in source]
                calls.append((config["mode"], condition, arrays))
                predictions = [{"id": row["id"], "probabilities": {"keyboard": .4, "mouse_click": .6}} for row in rows]
                return {"status": "measured", "n": len(rows)}, predictions
            api = types.SimpleNamespace(evaluate=evaluate)
            store = Store()
            report = variants.evaluate_conditions(api, object(), object(), test, ["keyboard", "mouse_click"], store, {"mode": "fused"}, "cpu",
                                                  variants.file_hash(f["checkpoint"]), f["output"], certs, preparation, f["root"])
            self.assertEqual(list(report), list(variants.VARIANTS))
            self.assertTrue(all(row["status"] == "measured" for row in report.values()))
            self.assertEqual([entry[0] for entry in calls], ["fused", "context_only", "eeg_only", "fused", "fused"])
            self.assertFalse(np.array_equal(calls[0][2][0], calls[4][2][0]))
            self.assertEqual(set(store.cache), {str(f["prepared"] / "waveforms.npy")})
            shifted = [json.loads(line) for line in (f["output"] / "eeg_time_shifted_predictions.jsonl").read_text().splitlines()]
            self.assertEqual([row["id"] for row in shifted], ["r2", "r3"])
            self.assertEqual(shifted[0]["perturbation"]["method"], variants.SHIFT_METHOD)
            self.assertFalse(any(row["action_executed"] for row in shifted))

    def test_cpu_main_loads_selected_checkpoint_and_backbone_once_and_writes_receipt(self):
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory() as folder:
            f = make_fixture(folder)
            f["state"].update(connector={}, trainable_base={}, classes=["keyboard", "mouse_click"])
            f["state"]["config"].update(revision="f" * 40, split_policy="subject", channels=2, prefix_tokens=8,
                                         encoder_width=8, samples=14, lora_rank=0, memory_budget_gib=1.)
            torch.save(f["state"], f["checkpoint"])
            variants.write_json(f["root"] / "training_membership.json", f["membership"])
            initial_checkpoint_hash = variants.file_hash(f["checkpoint"])
            counters = {"backbone_loads": 0}
            class Store:
                def __init__(self, samples):
                    pass
                def get(self, row):
                    value = np.load(row["eeg_path"], allow_pickle=False)
                    return value[row["eeg_index"]] if "eeg_index" in row else value
            class Model:
                def __init__(self, base, *args):
                    self.base = base
                    self.connector = types.SimpleNamespace(load_state_dict=lambda state: None)
                def to(self, device):
                    return self
                def eval(self):
                    return self
                def parameters(self):
                    return []
            def load_backbone(config, device):
                counters["backbone_loads"] += 1
                return types.SimpleNamespace(named_parameters=lambda: []), object(), {"pretrained": False}
            def evaluate(model, tokenizer, rows, classes, store, config, device, checkpoint, condition, include_truth):
                for row in rows:
                    store.get(row)
                return {"status": "measured", "n": len(rows)}, [{"id": row["id"], "probabilities": {"keyboard": .4, "mouse_click": .6}} for row in rows]
            api = types.SimpleNamespace(
                SCHEMA_VERSION="xavier-eeg-llm-v1", EEGCausalModel=Model, EEGStore=Store,
                read_rows=lambda path: [{**row, "_manifest_eeg_path": row["eeg_path"], "eeg_path": str(path.parent / row["eeg_path"])} for row in f["rows"]],
                validate_rows=lambda *args: {}, eeg_artifact_manifest=lambda rows: f["state"]["eeg_artifacts"],
                select_device=lambda name: "cpu", apply_allocator_limit=lambda *args: {"scope": "fixture"},
                load_backbone=load_backbone, attach_lora=lambda *args: None, evaluate=evaluate,
                memory_sample=lambda device: {"process_peak_rss_bytes": 1}, hardware=lambda device: {"device": "cpu_fixture"})
            with patch.object(variants, "load_api", return_value=api), patch.object(torch, "load", wraps=torch.load) as loading:
                result = variants.main(["--manifest", str(f["manifest"]), "--checkpoint", str(f["checkpoint"]), "--raw-bids-root", str(f["rawroot"]),
                                        "--artifact-root", str(f["root"]), "--output-dir", str(f["output"]), "--expected-test-records", "2", "--device", "cpu",
                                        "--model-module", str(Path(variants.__file__))])
            self.assertEqual(result, 0)
            self.assertEqual(loading.call_count, 1)
            self.assertEqual(counters["backbone_loads"], 1)
            self.assertEqual(initial_checkpoint_hash, variants.file_hash(f["checkpoint"]))
            receipt = json.loads((f["root"] / "variant_evaluation_completion.json").read_text())
            report = json.loads(Path(receipt["report_path"]).read_text())
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(set(report["conditions"]), set(variants.VARIANTS))
            self.assertFalse(report["weights_updated"] or report["model_selection_changed"] or report["autonomous_agent_validated"])


class ReceiptTests(unittest.TestCase):
    def test_measured_condition_without_matching_prediction_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            report = {"checkpoint_sha256": "a" * 64, "variant_names": list(variants.VARIANTS), "test_record_count": 2,
                      "frozen_test_ids_sha256": "b" * 64,
                      "conditions": {name: {"status": "blocked", "reason": "fixture unavailable"} for name in variants.VARIANTS}}
            report["conditions"]["fused"] = {"status": "measured", "checkpoint_sha256": "a" * 64, "n": 2, "frozen_test_ids_sha256": "b" * 64}
            with self.assertRaisesRegex(ValueError, "prediction bytes"):
                variants.finish_report(report, Path(folder) / "report.json", Path(folder) / "receipt.json")

    def test_complete_blocked_outcomes_are_execution_complete_not_goal_success(self):
        with tempfile.TemporaryDirectory() as folder:
            report = {"checkpoint_sha256": "a" * 64, "variant_names": list(variants.VARIANTS),
                      "conditions": {name: {"status": "blocked", "reason": "fixture unavailable"} for name in variants.VARIANTS}}
            receipt = variants.finish_report(report, Path(folder) / "report.json", Path(folder) / "receipt.json")
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(receipt["exit_code"], 0)
            self.assertEqual(receipt["report_sha256"], variants.file_hash(Path(folder) / "report.json"))
            self.assertFalse(report["autonomous_agent_validated"] or report["held_out_replay_validated"])

    def test_incomplete_or_unexplained_conditions_cannot_create_completion_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            report = {"checkpoint_sha256": "a" * 64, "variant_names": list(variants.VARIANTS), "conditions": {}}
            with self.assertRaisesRegex(ValueError, "all five conditions"):
                variants.finish_report(report, Path(folder) / "report.json", Path(folder) / "receipt.json")
            self.assertFalse((Path(folder) / "receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
