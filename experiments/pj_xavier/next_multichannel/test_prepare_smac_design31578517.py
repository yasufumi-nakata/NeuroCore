import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

import prepare_smac_design31578517 as a
import freeze_smac_design31578517_metadata as f
from test_freeze_smac_design31578517_metadata import fixture
from test_prepare_smac_inmid import make_base


def write_synthetic_edf(path, index, amplitude=30, duration=12.0):
    n, samples = 32, 12288
    fixed = (b"0       " + b" " * 160 + b"01.01.24" + b"00.00.00"
             + b"8448    " + b" " * 44 + b"1       "
             + ("%g" % duration).ljust(8).encode() + b"32  ")
    c = {
        "label": f.CHANNELS[:], "transducer": ["Ag/AgCl electrodes"] * n,
        "unit": ["uV"] * n, "pmin": ["-9830"] * n, "pmax": ["9830"] * n,
        "dmin": ["-32768"] * n, "dmax": ["32767"] * n,
        "prefilter": ["HP:1.0Hz;LP:49.0Hz"] * n, "samples": [str(samples)] * n,
        "reserved": ["RESERVED"] * n,
    }
    widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
    header = fixed + b"".join(b"".join(v.ljust(width).encode() for v in values)
                               for values, width in zip(c.values(), widths))
    t = np.arange(samples) / (samples / duration)
    digital = np.rint(np.stack([amplitude * np.sin(2*np.pi*(8+ch*.07+index*.013)*t+index*.17)
                               for ch in range(n)])).astype("<i2")
    blob = header + digital.tobytes()
    path.write_bytes(blob)
    return {
        "relative_path": path.name, "file_sha256": hashlib.sha256(blob).hexdigest(),
        "expected_bytes": len(blob), "actual_bytes": len(blob),
        "header_sha256": hashlib.sha256(header).hexdigest(), "n_channels": n,
        "header_bytes": len(header), "n_records": 1, "record_duration_s": duration,
        "duration_s": duration, "channels": c, "payload_read": False,
    }, hashlib.md5(blob).hexdigest()


def frozen_fixture(root, amplitude=30, remove_holdouts=True):
    raw = root / "raw"
    raw.mkdir()
    metadata, evidence = fixture()
    for i, original in enumerate(evidence["rows"]):
        record, md5 = write_synthetic_edf(raw / original["relative_path"], i, amplitude)
        evidence["rows"][i] = record
        metadata["files"][i].update(size=record["actual_bytes"], computed_md5=md5, supplied_md5=md5)
    plan = f.make_plan(metadata, evidence)
    plan["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    plan["freeze_code_sha256"] = a.FREEZE_CODE_SHA256
    pp = root / "frozen_plan.json"
    pp.write_bytes(f.canonical(plan))
    if remove_holdouts:
        for r in plan["records"]:
            if r["split"] != "train":
                (raw / r["relative_path"]).unlink()
    return raw, pp, plan


def prep_fixture(root, amplitude=30, bci_base=True):
    raw, plan_path, plan = frozen_fixture(root, amplitude)
    base, base_hash, manifest = make_base(root)
    if bci_base:
        manifest[0]["source_recording_id"] = "zenodo:8089820:synthetic_prior"
        (base/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in manifest))
        completion = json.loads((base/"completion.json").read_text())
        completion["manifest_sha256"] = a.sha(base/"manifest.jsonl")
        a.dump(base/"completion.json", completion)
        base_hash = a.sha(base/"completion.json")
    deny = root/"deny.json"
    a.dump(deny, {"excluded_source_sha256": ["f" * 64]})
    args = SimpleNamespace(raw_root=str(raw), frozen_plan=str(plan_path),
                           base_corpus=str(base), expected_base_completion_sha256=base_hash,
                           output=str(root/"prepared"), deny_json=[str(deny)], reserve_gib=1)
    return args, plan, manifest


class DesignEDFPreparationTests(unittest.TestCase):
    def test_mne_three_intervals_match_physical_calibration_and_hash_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, pp, plan = frozen_fixture(root)
            row = next(r for r in plan["records"] if r["split"] == "train")
            x, proof = a.decode_train_edf(raw/row["relative_path"], row)
            self.assertEqual(x.shape, (32, 12288))
            self.assertTrue(proof["MNE_three_interval_physical_V_passed"])
            self.assertEqual(len(proof["MNE_three_interval_difference_V"]), 3)
            self.assertLess(max(proof["MNE_three_interval_difference_V"]), 1e-11)
            self.assertLess(float(np.max(np.abs(x))), 1.1e-5)
            self.assertFalse(proof["label_or_questionnaire_used"])
            path = raw/row["relative_path"]
            with path.open("r+b") as stream:
                stream.seek(-2, 2)
                stream.write(b"\x00\x00")
            with self.assertRaisesRegex(ValueError, "SHA changed"):
                a.decode_train_edf(path, row)

    def test_holdout_rejected_before_any_file_open(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, pp, plan = frozen_fixture(Path(temp))
            row = next(r for r in plan["records"] if r["split"] == "test")
            with mock.patch.object(Path, "open", side_effect=AssertionError("must not open")):
                with self.assertRaisesRegex(ValueError, "Only frozen train"):
                    a.decode_train_edf(raw/row["relative_path"], row)

    def test_unit_and_header_mismatch_fail_without_guessing_correction(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, pp, plan = frozen_fixture(Path(temp))
            row = next(r for r in plan["records"] if r["split"] == "train")
            path = raw/row["relative_path"]
            with path.open("r+b") as stream:
                stream.seek(256 + 32*(16+80))
                stream.write(b"mV      ")
            with self.assertRaisesRegex(ValueError, "header SHA"):
                a.decode_train_edf(path, row)

    def test_fractional_rate_future_perturbation_and_fixed_amplitude_QC(self):
        rate_num, rate_den = 12288000000, 11999999
        fs = rate_num / rate_den
        t = np.arange(12288) / fs
        x = np.stack([8e-6*np.sin(2*np.pi*(8+ch*.07)*t) for ch in range(32)])
        before, js, cutoff = a.causal_transform(x, rate_num, rate_den)
        altered = x.copy()
        altered[:, 8192:] += 20e-6
        after, _, _ = a.causal_transform(altered, rate_num, rate_den)
        np.testing.assert_array_equal(before[:, cutoff < 8192], after[:, cutoff < 8192])
        self.assertTrue(np.all(cutoff.astype(float) / fs <= js / 200))
        array, windows, _ = a.causal_windows(x, rate_num, rate_den)
        self.assertEqual(array.shape, (3, 32, 3, 200))
        self.assertEqual(sum(w["window_end_s"]-w["window_start_s"] for w in windows), 9.)
        self.assertLessEqual(windows[-1]["window_end_s"], x.shape[1]/fs)
        array, _, reject = a.causal_windows(x*100, rate_num, rate_den)
        self.assertEqual(len(array), 0)
        self.assertGreater(reject["peak_above_fixed_100uV"], 0)
        array, _, reject = a.causal_windows(np.zeros_like(x), rate_num, rate_den)
        self.assertEqual(len(array), 0)
        self.assertGreater(reject["flat_below_fixed_0_01uV"], 0)
        with self.assertRaisesRegex(ValueError, "fixed_0_1V"):
            a.causal_windows(x+1., rate_num, rate_den)

    def test_plan_seal_and_post_BCI_parent_are_mandatory(self):
        with tempfile.TemporaryDirectory() as temp:
            args, plan, base_manifest = prep_fixture(Path(temp), bci_base=False)
            with mock.patch.object(a, "PLAN_SHA256", a.sha(args.frozen_plan)):
                with self.assertRaisesRegex(ValueError, "post-BCI"):
                    a.prepare(args)
            with self.assertRaisesRegex(ValueError, "plan seal"):
                a.load_plan(args.frozen_plan)
            self.assertFalse(Path(args.output).exists())

    def test_all17_train_prepared_old_val_unchanged_and_eight_holdout_files_absent(self):
        with tempfile.TemporaryDirectory() as temp:
            args, plan, base_manifest = prep_fixture(Path(temp))
            with mock.patch.object(a, "PLAN_SHA256", a.sha(args.frozen_plan)):
                result = a.prepare(args)
            self.assertEqual(result["new_source_records"], 17)
            self.assertEqual(result["new_source_windows"], 51)
            self.assertEqual(result["windows"], {"train": 52, "val": 1})
            self.assertEqual(result["newsource_val_or_test_signals_opened"], 0)
            self.assertEqual(result["questionnaire_or_label_tables_opened"], 0)
            out = Path(args.output)
            manifest = a.rows(out/"manifest.jsonl")
            self.assertEqual([r for r in manifest if r["split"] == "val"],
                             [{**r, "eeg_path": "base/"+r["eeg_path"]}
                              for r in base_manifest if r["split"] == "val"])
            certs = json.loads((out/"source_certificates.json").read_text())
            self.assertEqual(sum(c["signal_opened"] for c in certs), 17)
            self.assertEqual(sum(not c["signal_opened"] for c in certs), 8)
            admission = json.loads((out/"source_admission.json").read_text())
            for r in plan["records"]:
                if r["split"] != "train":
                    self.assertIn(r["source_sha256"], admission["excluded_source_sha256"])
                    self.assertIn(r["subject_group"], admission["excluded_subject_groups"])
            a.trainer.load_corpus(out/"manifest.jsonl", out/"artifacts.json", out/"source_admission.json")

    def test_prior_source_deny_skips_signal_before_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            args, plan, base_manifest = prep_fixture(Path(temp))
            row = next(r for r in plan["records"] if r["split"] == "train")
            a.dump(Path(args.deny_json[0]), {"excluded_source_sha256": [row["source_sha256"]]})
            (Path(args.raw_root)/row["relative_path"]).unlink()
            with mock.patch.object(a, "PLAN_SHA256", a.sha(args.frozen_plan)):
                result = a.prepare(args)
            self.assertEqual(result["new_source_records"], 16)
            certs = json.loads((Path(args.output)/"source_certificates.json").read_text())
            c = next(c for c in certs if c["source_sha256"] == row["source_sha256"])
            self.assertFalse(c["signal_opened"])
            self.assertEqual(c["status"], "excluded_prior_identity")

    def test_old_group_denies_and_sealed_parent_metadata_survive_merge(self):
        with tempfile.TemporaryDirectory() as temp:
            args, plan, _ = prep_fixture(Path(temp))
            row = next(r for r in plan["records"] if r["split"] == "train")
            base = Path(args.base_corpus)
            admission = json.loads((base/"source_admission.json").read_text())
            admission["excluded_source_groups"] = [row["subject_group"], "old-source-group"]
            admission["excluded_subject_groups"] = ["old-subject-group"]
            a.dump(base/"source_admission.json", admission)
            optional = {"source_certificates.json": [{"status": "held_review", "subject_group": "old-certificate-group"}],
                        "split_manifest.json": {"members": [{"split": "test", "subject_group": "old-test-group"}]},
                        "parent_provenance.json": {"source_admission": {"excluded_subject_groups": ["nested-old-group"]}}}
            completion = json.loads((base/"completion.json").read_text())
            completion["source_admission_sha256"] = a.sha(base/"source_admission.json")
            for filename, key in (("source_certificates.json", "certificates_sha256"),
                                  ("split_manifest.json", "split_manifest_sha256"),
                                  ("parent_provenance.json", "parent_provenance_sha256")):
                a.dump(base/filename, optional[filename])
                completion[key] = a.sha(base/filename)
            a.dump(base/"completion.json", completion)
            args.expected_base_completion_sha256 = a.sha(base/"completion.json")
            (Path(args.raw_root)/row["relative_path"]).unlink()
            with mock.patch.object(a, "PLAN_SHA256", a.sha(args.frozen_plan)):
                result = a.prepare(args)
            self.assertEqual(result["new_source_records"], 16)
            out = Path(args.output)
            merged = json.loads((out/"source_admission.json").read_text())
            expected = {row["subject_group"], "old-source-group", "old-subject-group", "nested-old-group",
                        "old-certificate-group", "old-test-group"}
            self.assertTrue(expected <= set(merged["excluded_subject_groups"]))
            self.assertEqual(merged["excluded_subject_groups"], merged["excluded_source_groups"])
            parent = json.loads((out/"parent_provenance.json").read_text())
            self.assertEqual(parent["optional_metadata"], optional)
            self.assertEqual(parent["source_admission"], admission)
            self.assertEqual(result["parent_provenance_sha256"], a.sha(out/"parent_provenance.json"))
            self.assertEqual(result["split_manifest_sha256"], a.sha(out/"split_manifest.json"))

    def test_train_alias_or_QC_hold_does_not_promote_group_to_holdout(self):
        proof = {"records": [
            {"split": "train", "status": "excluded_exact_duplicate_train_alias", "subject_group": "train-alias"},
            {"split": "train", "status": "held_no_window_meets_fixed_QC", "subject_group": "train-QC"},
            {"split": "val", "status": "admitted_offline_SSL", "subject_group": "prior-val"},
            {"split": "review", "subject_group": "prior-review"},
            {"status": "held_review", "subject_group": "unknown-split-review"}],
            "excluded_source_groups": ["explicit-deny"],
            "nested": {"status": "held_review", "rows": [
                {"split": "train", "subject_group": "explicit-train"},
                {"split": "test", "subject_group": "prior-test"}]}}
        self.assertEqual(a.explicit_group_denies(proof),
                         {"prior-val", "prior-review", "unknown-split-review", "explicit-deny", "prior-test"})

    def test_parent_certificate_seal_tamper_stops_before_signal_read(self):
        with tempfile.TemporaryDirectory() as temp:
            args, plan, _ = prep_fixture(Path(temp))
            base = Path(args.base_corpus)
            a.dump(base/"source_certificates.json", [{"status": "admitted_offline_SSL"}])
            completion = json.loads((base/"completion.json").read_text())
            completion["certificates_sha256"] = a.sha(base/"source_certificates.json")
            a.dump(base/"completion.json", completion)
            args.expected_base_completion_sha256 = a.sha(base/"completion.json")
            a.dump(base/"source_certificates.json", [{"status": "changed"}])
            with mock.patch.object(a, "PLAN_SHA256", a.sha(args.frozen_plan)), \
                 mock.patch.object(a, "decode_train_edf", side_effect=AssertionError("must not decode")):
                with self.assertRaisesRegex(ValueError, "optional metadata seal changed"):
                    a.prepare(args)
            self.assertFalse(Path(args.output).exists())

    def test_original_scalar_identity_is_retained_and_not_added_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            args, plan, _ = prep_fixture(Path(temp))
            row = next(r for r in plan["records"] if r["split"] == "train")
            _, proof = a.decode_train_edf(Path(args.raw_root)/row["relative_path"], row)
            base = Path(args.base_corpus)
            admission = json.loads((base/"source_admission.json").read_text())
            source = next(k for k, v in admission["sources"].items() if v["split"] == "train")
            alias = proof["original_integer_EEG_payload_sha256"]
            admission["sources"][source]["original_EEG_scalar_payload_sha256"] = alias
            a.dump(base/"source_admission.json", admission)
            completion = json.loads((base/"completion.json").read_text())
            completion["source_admission_sha256"] = a.sha(base/"source_admission.json")
            a.dump(base/"completion.json", completion)
            args.expected_base_completion_sha256 = a.sha(base/"completion.json")
            with mock.patch.object(a, "PLAN_SHA256", a.sha(args.frozen_plan)):
                result = a.prepare(args)
            self.assertEqual(result["new_source_records"], 16)
            out = Path(args.output)
            merged = json.loads((out/"source_admission.json").read_text())
            self.assertEqual(merged["sources"][source]["original_EEG_scalar_payload_sha256"], alias)
            certs = json.loads((out/"source_certificates.json").read_text())
            certificate = next(c for c in certs if c["source_sha256"] == row["source_sha256"])
            self.assertEqual(certificate["status"], "excluded_base_identity")

    def test_all_fixed_QC_rejected_keeps_receipt_and_does_not_create_corpus(self):
        with tempfile.TemporaryDirectory() as temp:
            args, plan, base_manifest = prep_fixture(Path(temp), amplitude=3000)
            with mock.patch.object(a, "PLAN_SHA256", a.sha(args.frozen_plan)):
                result = a.prepare(args)
            self.assertEqual(result["status"], "blocked_no_eligible_new_source_windows")
            self.assertEqual(result["new_source_windows"], 0)
            self.assertFalse((Path(args.output)/"completion.json").exists())
            self.assertFalse((Path(args.output)/"base").exists())
            self.assertTrue((Path(args.output)/"preparation_result.json").exists())


if __name__ == "__main__":
    unittest.main()
