"""CPU tests for unit integrity, fixed splits, numeric masking, and token limits."""
import argparse
import collections
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

spec = importlib.util.spec_from_file_location("numeric_prepare", Path(__file__).with_name("evlm_prepare_numeric.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def prompt(observed, indices, schema, geometry=None):
    # A compact deterministic stand-in for the production shared prompt builder.
    return json.dumps({"observed": observed, "masked": indices, "schema": schema,
                       "geometry": geometry}, sort_keys=True, separators=(",", ":"))


def window(channels=16):
    x = np.arange(600, dtype=np.float64)
    values_uV = np.stack([7 * np.sin(2 * np.pi * x / (19 + c)) + .3 * c for c in range(channels)])
    return (values_uV / 100).reshape(channels, 3, 200).astype(np.float32)


def row(index=0, split="train", subject="synthetic-train", channels=16, path="train.npy"):
    return {"id": f"synthetic-{split}-{index}", "split": split, "subject_group": subject,
            "source_recording_id": f"zenodo:806023:synthetic-{subject}",
            "source_sha256": m.canonical_sha(["source", subject]),
            "eeg_payload_sha256": m.canonical_sha(["payload", subject]),
            "eeg_path": path, "eeg_index": index, "channels": [f"C{i}" for i in range(channels)],
            "sample_rate_hz": 200, "native_sample_rate_hz": 512, "unit": "V_x_1e4",
            "physical_unit_verified": True, "montage_key": "synthetic-montage",
            "reference": "synthetic-reference", "window_start_s": index * 3., "window_end_s": index * 3. + 3.}


class ToyTokenizer:
    def __init__(self, broken=False):
        self.broken = broken

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False and kwargs["return_dict"] is False
        prefix = [10001, 10002]
        if len(messages) == 1:
            return prefix if not self.broken else [10001, 10003]
        return prefix + [ord(c) for c in messages[-1]["content"]] + [10004]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(t) for t in tokens if t < 10000)


def certified_fixture(root, transform=None):
    root.mkdir()
    train = np.stack([window(), window()])
    val = np.stack([window()])
    np.save(root / "train.npy", train)
    np.save(root / "val.npy", val)
    rows = [row(0), row(1), row(0, "val", "synthetic-val", path="val.npy")]
    sources = {}
    for r in rows:
        sources[r["source_sha256"]] = {"split": r["split"], "subject_group": r["subject_group"],
             "montage_key": r["montage_key"], "eeg_payload_sha256": r["eeg_payload_sha256"],
             "physical_unit_verified": True, "fit_allowed": r["split"] == "train",
             "validation_allowed": r["split"] == "val"}
    admission = {"sources": sources, "excluded_source_sha256": [], "excluded_payload_sha256": [],
                 "excluded_source_groups": []}
    artifacts = {name: {"sha256": m.sha(root / name), "bytes": (root / name).stat().st_size,
                       "dtype": "float32", "shape": list(array.shape)}
                 for name, array in [("train.npy", train), ("val.npy", val)]}
    if transform:
        transform(rows, admission, artifacts)
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    values = {"protocol.json": {}, "source_admission.json": admission, "source_certificates.json": [],
              "split_manifest.json": {}, "parent_provenance.json": {}, "artifacts.json": artifacts}
    for name, value in values.items():
        (root / name).write_text(json.dumps(value))
    completion = {"status": "prepared_verified_new_offline_SSL_corpus", "windows": {"train": 2, "val": 1}}
    completion.update({key: m.sha(root / name) for key, name in m.IDENTITY_FILES.items()})
    (root / "completion.json").write_text(json.dumps(completion))
    return m.sha(root / "completion.json")


class NumericPreparationTests(unittest.TestCase):
    def test_physical_scale_is_times_100_not_times_one_million(self):
        source = window()
        actual = m.physical_microvolts(source)
        np.testing.assert_allclose(actual, source.reshape(16, 600).astype(np.float64) * 100,
                                   rtol=0, atol=0)
        self.assertLess(float(np.abs(actual).max()), 20)
        with self.assertRaisesRegex(ValueError, "100uV"):
            m.physical_microvolts(source * np.float32(100))

    def test_finite_flatline_and_dtype_are_checked_without_repair(self):
        for bad, reason in [(window().astype(np.float64), "float32"),
                            (np.zeros((16, 3, 200), np.float32), "0_01uV")]:
            with self.assertRaisesRegex(ValueError, reason):
                m.physical_microvolts(bad)
        bad = window()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            m.physical_microvolts(bad)

    def test_nineteen_cycles_cover_all_samples_once_without_fake_padding(self):
        intervals = [m.crop_bounds("synthetic", 23, c)[:2] for c in range(19)]
        self.assertEqual(sorted(v for a, b in intervals for v in range(a, b)), list(range(600)))
        self.assertEqual(sorted(b - a for a, b in intervals), [24] + [32] * 18)
        self.assertEqual(intervals, [m.crop_bounds("synthetic", 23, c)[:2] for c in range(19)])
        with self.assertRaisesRegex(ValueError, "cycle"):
            m.crop_bounds("synthetic", 23, 19)

    def test_actual_channel_names_and_partial_block_are_preserved(self):
        r = row(channels=14)
        examples = m.make_examples(r, window(14), seed=5, cycle=0, prompt_builder=prompt)
        self.assertEqual(len(examples), 2)
        self.assertEqual([x["reconstruction"]["geometry"]["shape"][0] for x in examples], [8, 6])
        self.assertEqual([c for e in examples for c in e["reconstruction"]["geometry"]["channel_names"]], r["channels"])
        for e in examples:
            rec = e["reconstruction"]
            self.assertEqual(len(rec["observed"]), math_product(rec["geometry"]["shape"]))
            self.assertEqual([i for i, v in enumerate(rec["observed"]) if v is None], rec["masked_indices"])
            self.assertEqual(len(json.loads(e["messages"][1]["content"])), 4)

    def test_masked_truth_changes_neither_prompt_nor_baseline(self):
        r = row()
        original_window = window()
        original = m.make_examples(r, original_window, seed=12, cycle=0, prompt_builder=prompt)[0]
        changed = original_window.copy().reshape(16, 600)
        rec = original["reconstruction"]
        times = rec["geometry"]["shape"][1]
        first = rec["geometry"]["crop_start_sample"]
        for i in rec["masked_indices"]:
            channel, t = divmod(i, times)
            changed[channel, first + t] += np.float32(.04)  # four microvolts, within original QC
        edited = m.make_examples(r, changed.reshape(16, 3, 200), seed=12, cycle=0, prompt_builder=prompt)[0]
        self.assertNotEqual(original["messages"][1]["content"], edited["messages"][1]["content"])
        self.assertEqual(original["messages"][0]["content"], edited["messages"][0]["content"])
        self.assertEqual(original["reconstruction"], edited["reconstruction"])
        self.assertEqual(original["baselines"], edited["baselines"])

    def test_interpolation_stays_inside_channel_and_uses_only_visible_values(self):
        observed = [0, None, None, None, None, 10, 12, 14] + [99] * 8
        self.assertEqual(m.interpolation_baseline(observed, [1, 2, 3, 4], [2, 8]), [2, 4, 6, 8])
        observed[8:] = [-99] * 8
        self.assertEqual(m.interpolation_baseline(observed, [1, 2, 3, 4], [2, 8]), [2, 4, 6, 8])
        with self.assertRaisesRegex(ValueError, "crosses_channel"):
            m.interpolation_baseline([0] * 16, [6, 7, 8, 9], [2, 8])
        with self.assertRaisesRegex(ValueError, "masked_truth"):
            m.interpolation_baseline([0] * 16, [1, 2, 3, 4], [2, 8])

    def test_coverage_does_not_multiply_time_by_channel_blocks(self):
        rows = [row(channels=27)]
        result = m.coverage(rows, 7, 0)
        self.assertEqual(result["source_windows"], 1)
        self.assertEqual(result["source_window_seconds"], 3)
        self.assertEqual(result["derived_examples"], 4)
        self.assertIn(result["unique_exposed_time_seconds"], [.12, .16])

    def test_token_limit_and_prefix_mismatch_fail_without_truncation(self):
        record = m.make_examples(row(), window(), seed=1, cycle=0, prompt_builder=prompt)[0]
        self.assertGreater(m.check_tokens(ToyTokenizer(), record)[0], 2)
        with self.assertRaisesRegex(ValueError, "3_no_truncation"):
            m.check_tokens(ToyTokenizer(), record, max_length=3)
        with self.assertRaisesRegex(ValueError, "prefix_mismatch"):
            m.check_tokens(ToyTokenizer(broken=True), record)

    def test_positive_certificate_and_partition_roles_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "corpus"
            digest = certified_fixture(root)
            data = m.validate_corpus(root, digest)
            self.assertEqual(collections.Counter(r["split"] for r in data["rows"]), {"train": 2, "val": 1})
            self.assertEqual(len(data["metadata_sha256"]), 8)

    def test_denied_source_cannot_be_relabelled_as_train(self):
        def denied(rows, admission, artifacts):
            admission["excluded_source_sha256"].append(rows[0]["source_sha256"])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "corpus"
            digest = certified_fixture(root, denied)
            with self.assertRaisesRegex(ValueError, "denied_source"):
                m.validate_corpus(root, digest)

    def test_fit_false_certificate_cannot_enter_training(self):
        def forbidden(rows, admission, artifacts):
            admission["sources"][rows[0]["source_sha256"]]["fit_allowed"] = False
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "corpus"
            digest = certified_fixture(root, forbidden)
            with self.assertRaisesRegex(ValueError, "fit_role"):
                m.validate_corpus(root, digest)

    def test_metadata_hashes_detect_changed_source_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "corpus"
            digest = certified_fixture(root)
            with (root / "manifest.jsonl").open("a") as f:
                f.write("\n")
            with self.assertRaisesRegex(ValueError, "metadata_changed"):
                m.validate_corpus(root, digest)

    def test_unverified_unit_is_not_guessed(self):
        def changed(rows, admission, artifacts):
            rows[0]["unit"] = "unknown"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "corpus"
            digest = certified_fixture(root, changed)
            with self.assertRaisesRegex(ValueError, "units_or_rate"):
                m.validate_corpus(root, digest)

    def test_probe_then_complete_source_manifest_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            corpus = root / "corpus"
            digest = certified_fixture(corpus)
            output = root / "new_numeric"
            args = argparse.Namespace(command="probe", corpus=str(corpus), output=str(output),
                     expected_completion_sha256=digest, tokenizer=str(root / "tokenizer"), seed=11, cycle=0)
            with mock.patch.object(m, "load_tokenizer", return_value=(ToyTokenizer(), {"test": "fixed"})):
                self.assertEqual(m.run(args), 0)
                args.command = "prepare"
                self.assertEqual(m.run(args), 0)
                with self.assertRaisesRegex(ValueError, "already_exists"):
                    m.run(args)
            manifest_path = output / "data/zenodo_806023/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["inputs"]["train"]["count"], 4)
            self.assertEqual(manifest["inputs"]["val"]["count"], 2)
            self.assertEqual(m.sha(manifest["qc_receipt"]["path"]), manifest["qc_receipt"]["sha256"])
            receipt = json.loads((output / "qc_receipt.json").read_text())
            self.assertEqual(receipt["train_coverage"]["source_window_seconds"], 6)
            self.assertFalse(receipt["model_training_complete"])
            self.assertEqual(receipt["test_review_signals_read"], 0)

    def test_changed_crop_seed_cannot_reuse_probe(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            corpus = root / "corpus"
            digest = certified_fixture(corpus)
            output = root / "new_numeric"
            args = argparse.Namespace(command="probe", corpus=str(corpus), output=str(output),
                     expected_completion_sha256=digest, tokenizer=str(root / "tokenizer"), seed=11, cycle=0)
            with mock.patch.object(m, "load_tokenizer", return_value=(ToyTokenizer(), {"test": "fixed"})):
                self.assertEqual(m.run(args), 0)
                args.command = "prepare"
                args.seed = 12
                with self.assertRaisesRegex(ValueError, "probe_identity_mismatch"):
                    m.run(args)
            self.assertFalse((output / "data").exists())

    def test_six_channel_blocks_preserve_every_channel_sample_and_crop(self):
        for channels in (14, 16, 27, 32):
            with self.subTest(channels=channels):
                r, values = row(channels=channels), window(channels)
                new = m.make_examples(r, values, seed=20261124, cycle=0,
                                      prompt_builder=prompt, block_channels=6)
                old = m.make_examples(r, values, seed=20261124, cycle=0, prompt_builder=prompt)
                self.assertEqual([c for e in new for c in e["reconstruction"]["geometry"]["channel_names"]], r["channels"])
                self.assertEqual([c for e in new for c in e["provenance"]["channel_indices"]], list(range(channels)))
                self.assertTrue(all(e["reconstruction"]["geometry"]["shape"][0] <= 6 for e in new))
                start, end, _ = m.crop_bounds(r["id"], 20261124, 0)
                reconstructed = []
                for e in new:
                    rec = e["reconstruction"]
                    self.assertEqual(rec["geometry"]["crop_start_sample"], old[0]["reconstruction"]["geometry"]["crop_start_sample"])
                    restored = list(rec["observed"])
                    for index, target in zip(rec["masked_indices"], json.loads(e["messages"][1]["content"])):
                        restored[index] = target
                    reconstructed.extend(restored)
                expected = np.rint(m.physical_microvolts(values)[:, start:end]).astype(int).reshape(-1).tolist()
                self.assertEqual(reconstructed, expected)
                old_coverage, new_coverage = m.coverage([r], 20261124, 0), m.coverage([r], 20261124, 0, 6)
                self.assertEqual(old_coverage["unique_exposed_time_seconds"], new_coverage["unique_exposed_time_seconds"])
                self.assertEqual(new_coverage["derived_examples"], (channels + 5) // 6)
                self.assertTrue(set(e["id"] for e in new).isdisjoint(e["id"] for e in old))

    def test_801_limit_is_enforced_without_changing_1024_default(self):
        class SizedTokenizer(ToyTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                tokens = super().apply_chat_template(messages, **kwargs)
                return tokens if len(messages) == 1 else tokens + [10004] * (802 - len(tokens))
        record = m.make_examples(row(), window(), seed=1, cycle=0, prompt_builder=prompt, block_channels=6)[0]
        self.assertEqual(m.check_tokens(SizedTokenizer(), record)[0], 802)
        with self.assertRaisesRegex(ValueError, "801_no_truncation"):
            m.check_tokens(SizedTokenizer(), record, max_length=801)

    def test_six_channel_manifest_binds_limits_and_helper(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            corpus, output = root / "corpus", root / "new_six_channel"
            digest = certified_fixture(corpus)
            args = argparse.Namespace(command="probe", corpus=str(corpus), output=str(output),
                     expected_completion_sha256=digest, tokenizer=str(root / "tokenizer"), seed=11, cycle=0,
                     block_channels=6, max_tokens=801)
            with mock.patch.object(m, "load_tokenizer", return_value=(ToyTokenizer(), {"test": "fixed"})):
                self.assertEqual(m.run(args), 0)
                args.command = "prepare"
                self.assertEqual(m.run(args), 0)
            manifest = json.loads((output / "data/zenodo_806023/manifest.json").read_text())
            self.assertEqual(manifest["max_channels_per_training_block"], 6)
            self.assertEqual(manifest["max_seq_length"], 801)
            self.assertEqual(manifest["inputs"]["train"]["count"], 6)
            self.assertEqual(manifest["inputs"]["val"]["count"], 3)
            self.assertEqual(len(manifest["preparation_identity"]["trainer_source_sha256"]), 64)
            state = json.loads((output / "prepare_state.json").read_text())
            self.assertEqual(state["completed_source_windows"], 2)
            self.assertEqual(state["generated_examples"], 6)

    def test_probe_cannot_be_reused_after_block_geometry_change(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            corpus, output = root / "corpus", root / "new_numeric"
            digest = certified_fixture(corpus)
            args = argparse.Namespace(command="probe", corpus=str(corpus), output=str(output),
                     expected_completion_sha256=digest, tokenizer=str(root / "tokenizer"), seed=11, cycle=0,
                     block_channels=8, max_tokens=1024)
            with mock.patch.object(m, "load_tokenizer", return_value=(ToyTokenizer(), {"test": "fixed"})):
                self.assertEqual(m.run(args), 0)
                args.command, args.block_channels = "prepare", 6
                with self.assertRaisesRegex(ValueError, "probe_identity_mismatch"):
                    m.run(args)
            self.assertFalse((output / "data").exists())


def math_product(values):
    out = 1
    for value in values:
        out *= value
    return out


if __name__ == "__main__":
    unittest.main()
