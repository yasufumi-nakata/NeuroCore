"""Synthetic, CPU-only checks for the frozen BCI independent validation route."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch
import evaluate_smac_bci8089820_validation as ev


def h(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def fixture(root):
    members, records = [], {}
    folds = [("train", 446, 56), ("val", 96, 12), ("test", 96, 12), ("review", 56, 7)]
    for fold, count, groups in folds:
        for index in range(count):
            group = "synthetic:" + fold + ":" + str(index % groups).zfill(2)
            name = "synthetic/" + fold + "/" + str(index).zfill(3) + ".gdf"
            data = ("SYNTHETIC_TEST_ONLY_" + name).encode()
            source = h(data)
            member = {
                "source_member": name, "subject_group": group, "split": fold,
                "source_file_sha256": source, "eeg_payload_sha256": h("scalar:" + name),
                "calibration_sha256": h("calibration:" + name),
            }
            members.append(member)
            records[name] = {
                **member, "source_file_bytes": len(data),
                "EEG_calibration_sha256": member["calibration_sha256"],
                "samples_per_EEG_channel": 4096, "_phase": index * .011,
                "_test_data": data,
            }
    archive = root / "synthetic.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, record in records.items():
            zf.writestr(name, record["_test_data"])
    return {"members": members}, records, archive


def synthetic_reader(path, record):
    if Path(path).read_bytes() != record["_test_data"]:
        raise AssertionError("Synthetic extraction changed")
    t = np.arange(4096, dtype=np.float64) / 512
    phase = np.arange(27, dtype=np.float64)[:, None] * .017 + record["_phase"]
    x = 8e-6 * np.sin(2 * np.pi * 10 * t[None, :] + phase)
    proof = {
        "physical_unit_verified": True,
        "decoded_EEG_V_sha256": h(np.asarray(x.T, dtype="<f8", order="C").tobytes()),
        "observed_native_peak_V": float(np.max(np.abs(x))),
        "original_EEG_scalar_payload_sha256": record["eeg_payload_sha256"],
    }
    return x, proof


class IndependentBciValidationTests(unittest.TestCase):
    def test_preregistered_protocol_and_production_dependencies(self):
        protocol = ev.load_protocol(Path(ev.__file__).with_name("BCI8089820_VALIDATION_PROTOCOL_V1.json"))
        self.assertEqual(protocol["scientific_spec"]["selection"]["candidate_records"], 96)
        self.assertEqual(ev.sha(ev.bci.__file__), ev.PREPARATION_SHA256)
        self.assertEqual(ev.sha(ev.trainer.__file__), ev.TRAINER_SHA256)

    def test_changed_QC_cannot_pass_protocol_hash(self):
        protocol = json.loads(Path(ev.__file__).with_name("BCI8089820_VALIDATION_PROTOCOL_V1.json").read_text())
        protocol["scientific_spec"]["QC"]["postprocess_max_abs_V"] = .001
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "changed.json"
            p.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                ev.load_protocol(p)

    def test_only_frozen_96_and_12_groups_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, records, _ = fixture(Path(tmp))
            selected, _ = ev.select_validation(plan, records, [], {"sources": {}})
            self.assertEqual(len(selected), 96)
            self.assertEqual(len({r["subject_group"] for r in selected}), 12)
            self.assertTrue(all(r["split"] == "val" for r in selected))

    def test_prior_group_source_and_scalar_contamination_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, records, _ = fixture(Path(tmp))
            selected, _ = ev.select_validation(plan, records, [], {"sources": {}})
            val = selected[0]
            baseline = {"source_sha256": "other-source", "subject_group": "other-group",
                        "eeg_payload_sha256": "other-payload"}
            for field, value in [
                ("subject_group", val["subject_group"]),
                ("source_sha256", val["source_file_sha256"]),
                ("eeg_payload_sha256", val["eeg_payload_sha256"]),
            ]:
                row = {**baseline, field: value}
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "overlaps earlier"):
                    ev.select_validation(plan, records, [row], {"sources": {}})

    def test_cross_fold_scalar_identity_rejected_without_signal_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, records, _ = fixture(Path(tmp))
            val = next(m for m in plan["members"] if m["split"] == "val")
            test = next(m for m in plan["members"] if m["split"] == "test")
            test["eeg_payload_sha256"] = val["eeg_payload_sha256"]
            with self.assertRaisesRegex(ValueError, "train/test/review identity"):
                ev.select_validation(plan, records, [], {"sources": {}})

    def test_test_and_review_extraction_rejected_before_zip_access(self):
        class NeverRead:
            def getinfo(self, name):
                raise AssertionError("ZIP should not be accessed")
        for fold in ("train", "test", "review"):
            with self.subTest(fold=fold), self.assertRaisesRegex(ValueError, "Only frozen validation"):
                ev.extract_validation_member(NeverRead(), {"split": fold}, {}, "unused.gdf")

    def test_source_SHA_mismatch_stops_before_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, records, archive = fixture(root)
            member = next(m for m in plan["members"] if m["split"] == "val")
            record = {**records[member["source_member"]], "source_file_sha256": "0" * 64}
            with zipfile.ZipFile(archive) as zf, self.assertRaisesRegex(ValueError, "source SHA mismatch"):
                ev.extract_validation_member(zf, member, record, root / "bad.gdf")

    def test_same_row_mask_ignores_global_rng_and_masks_40_patches(self):
        torch.manual_seed(1)
        before = ev.trainer.legacy.fixed_mask("synthetic-source:7", 27, 3, ev.SEED)
        torch.manual_seed(999)
        after = ev.trainer.legacy.fixed_mask("synthetic-source:7", 27, 3, ev.SEED)
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(before.shape, (27, 3))
        self.assertEqual(int(before.sum()), 40)

    def test_group_macro_is_not_window_weighted(self):
        score = ev.aggregate_group_scores([
            {"subject_group": "synthetic-small", "windows": 1, "mean_masked_mse": 1.},
            {"subject_group": "synthetic-large", "windows": 99, "mean_masked_mse": 3.},
        ])
        self.assertEqual(score["mean_subject_masked_mse"], 2.)
        self.assertEqual(score["windows"], 100)

    def test_all_96_synthetic_sources_and_one_missing_group_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, records, archive = fixture(root)
            selected, protected = ev.select_validation(plan, records, [], {"sources": {}})
            missing = selected[0]["subject_group"]
            out = root / "evaluation"
            out.mkdir()
            opened = []
            def reader(path, record):
                opened.append(record["source_member"])
                if record["subject_group"] == missing:
                    raise ev.bci.PhysicalCalibrationHold("held_synthetic_calibration", {"synthetic": True})
                return synthetic_reader(path, record)
            with patch.object(ev.bci, "verify_gdf", side_effect=reader):
                manifest, report = ev.prepare_validation(
                    {"archive": str(archive), "reserve_gib": 0}, out, selected, records,
                    protected, time.monotonic() + 60)
            self.assertEqual(len(opened), 96)
            self.assertEqual(set(opened), {m["source_member"] for m in selected})
            self.assertEqual(len(manifest), 88)
            self.assertEqual(report["candidate_groups"], 12)
            self.assertEqual(report["groups_with_accepted_windows"], 11)
            self.assertFalse(report["complete_12_group_coverage"])
            self.assertEqual(report["missing_groups"], [missing])
            self.assertEqual(report["test_GDF_opened"], 0)
            self.assertEqual(report["review_GDF_opened"], 0)
            self.assertEqual(report["fit_steps"], 0)
            self.assertFalse(list(out.glob("temporary_validation_GDF_*")))
            for row in manifest:
                self.assertEqual(row["split"], "val")
                self.assertFalse(row["fit_allowed"])
                self.assertEqual(tuple(ev.trainer.Store().get(row).shape), (27, 3, 200))

    def test_unexpected_physical_duplicate_across_groups_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, records, archive = fixture(root)
            selected, protected = ev.select_validation(plan, records, [], {"sources": {}})
            out = root / "evaluation"
            out.mkdir()
            physical = None
            first_group = None
            def reader(path, record):
                nonlocal physical, first_group
                x, proof = synthetic_reader(path, record)
                if physical is None:
                    physical = proof["decoded_EEG_V_sha256"]
                    first_group = record["subject_group"]
                elif record["subject_group"] != first_group:
                    proof["decoded_EEG_V_sha256"] = physical
                return x, proof
            with patch.object(ev.bci, "verify_gdf", side_effect=reader):
                with self.assertRaisesRegex(ValueError, "cross-group validation duplicate"):
                    ev.prepare_validation({"archive": str(archive), "reserve_gib": 0}, out,
                                          selected, records, protected, time.monotonic() + 60)

    def test_shared_GPU_lock_refuses_contention_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_gpu.lock"
            path.touch()
            with ev.exclusive_gpu(path):
                with self.assertRaises(BlockingIOError):
                    with ev.exclusive_gpu(path):
                        self.fail("Concurrent model process obtained lock")
            with ev.exclusive_gpu(path):
                pass


if __name__ == "__main__":
    unittest.main()
