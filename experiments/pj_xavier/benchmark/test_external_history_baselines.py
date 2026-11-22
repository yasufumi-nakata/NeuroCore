import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import external_history_baselines as adapter
import history_baselines as history


COMMON_PATH = Path(__file__).resolve().parents[1] / "baselines" / "cpu_baselines.py"
common = adapter.load_module(COMMON_PATH, adapter.COMMON_SHA, "external_history_test_common")


def row_fixture():
    features = {}
    for kind in history.KINDS:
        features[kind + "_source_available"] = True
        for horizon in history.HORIZONS:
            features[f"{kind}_count_{horizon}s"] = 0
        features[kind + "_seconds_since_last_capped_60"] = None
    pack = {"cutoff_s": 9.9, "max_source_onset_used_s": None,
            "source_fields": ["onset", "action only for key-down/up classification"],
            "target_or_future_events_included": False, "features": features}
    return {"id": "r", "label": "keyboard", "window_start_s": 10., "window_end_s": 11.75,
            "decision_time_s": 16., "event_time_s": 20., "clock_uncertainty_guard_s": .1,
            "history_pre_eeg": pack, "history_pre_decision_guarded": {**copy.deepcopy(pack), "cutoff_s": 15.9},
            "history_evidence": [{"acquisition": "keypress", "sha256": "a" * 64, "uri": "source#events.tsv"}],
            "environment_text": "Java programming experiment. Last recorded experiment page or event: showTask. AI availability recorded at that event: True."}


def model_fixture(mode="history_pre_eeg", classes=None):
    names = history.history_vector(row_fixture(), "history_pre_eeg")[1]
    if mode.endswith("_plus_context"):
        names += ["context:ai_availability=True", "context:page=showTask", "context:page_ai_pair=showTask|True"]
    return {"names": names, "class_order": classes or [adapter.AI, adapter.OTHER],
            "coefficients": np.arange(len(names), dtype=float).reshape(1, -1) / 100,
            "intercept": np.asarray([.7]), "scaler_mean": np.zeros(len(names)), "scaler_scale": np.ones(len(names)),
            "sha256": "a" * 64}


def dump_fixture(path, model, **overrides):
    contents = {"classes": np.asarray(model["class_order"]), "feature_names": np.asarray(model["names"]),
                "coefficients": model["coefficients"], "intercept": model["intercept"],
                "scaler_mean": model["scaler_mean"], "scaler_scale": model["scaler_scale"],
                "solver": np.asarray("multinomial logistic, lbfgs, balanced class weights")}
    contents.update(overrides)
    np.savez_compressed(path, **contents)
    return adapter.sha(path)


class ExternalHistoryTests(unittest.TestCase):
    def test_portable_inference_matches_fitted_sklearn_binary(self):
        rng = np.random.default_rng(88)
        train = rng.normal(size=(100, 4))
        labels = np.where(train[:, 0] + train[:, 1] > .1, adapter.AI, adapter.OTHER)
        scaler = StandardScaler().fit(train)
        fitted = LogisticRegression(solver="lbfgs", class_weight="balanced", C=.1).fit(scaler.transform(train), labels)
        query = rng.normal(size=(35, 4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "portable.npz"
            common.save_portable_model(path, scaler, fitted, ["h0", "h1", "h2", "h3"])
            model = adapter.load_portable_binary(path, adapter.sha(path), ["h0", "h1", "h2", "h3"])
            actual = adapter.predict_portable(model, query)
        np.testing.assert_allclose(actual, fitted.predict_proba(scaler.transform(query)), atol=1e-12, rtol=0)

    def test_class_order_and_extreme_sigmoid_are_explicit(self):
        model = {"names": ["x"], "class_order": [adapter.OTHER, adapter.AI], "coefficients": np.asarray([[1.]]),
                 "intercept": np.asarray([0.]), "scaler_mean": np.asarray([0.]), "scaler_scale": np.asarray([1.])}
        values = adapter.predict_portable(model, [[-1000.], [0.], [1000.]])
        np.testing.assert_array_equal(values, [[1., 0.], [.5, .5], [0., 1.]])
        self.assertEqual(values[2, model["class_order"].index(adapter.AI)], 1.)
        model["class_order"] = [adapter.AI, adapter.OTHER]
        self.assertEqual(values[2, model["class_order"].index(adapter.AI)], 0.)

    def test_unknown_context_uses_only_saved_vocabulary_without_fit(self):
        row = row_fixture()
        row["environment_text"] = "Java programming experiment. Last recorded experiment page or event: novelPage. AI availability recorded at that event: False."
        model = model_fixture("history_pre_eeg_plus_context")
        with mock.patch.object(common, "fit_context", side_effect=AssertionError("fit forbidden")), mock.patch.object(common, "select_model", side_effect=AssertionError("selection forbidden")):
            x = adapter.feature_matrix([row], "history_pre_eeg_plus_context", model, history, common)
        np.testing.assert_array_equal(x[0, -3:], [0., 0., 0.])

    def test_label_and_identity_are_not_predictive_features(self):
        one, two = row_fixture(), row_fixture()
        two.update(label=adapter.AI, id="changed", subject="other", label_evidence_row=999)
        model = model_fixture("history_pre_eeg_plus_context")
        x = adapter.feature_matrix([one, two], "history_pre_eeg_plus_context", model, history, common)
        np.testing.assert_array_equal(x[0], x[1])
        np.testing.assert_array_equal(*adapter.predict_portable(model, x))

    def test_future_history_and_reordered_columns_fail_closed(self):
        row = row_fixture()
        row["history_pre_eeg"]["max_source_onset_used_s"] = 9.9
        with self.assertRaisesRegex(ValueError, "future or boundary"):
            adapter.feature_matrix([row], "history_pre_eeg", model_fixture(), history, common)
        model = model_fixture()
        model["names"][0], model["names"][1] = model["names"][1], model["names"][0]
        with self.assertRaisesRegex(ValueError, "prefix/order"):
            adapter.feature_matrix([row_fixture()], "history_pre_eeg", model, history, common)

    def test_portable_tamper_and_invalid_scale_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model = model_fixture()
            digest = dump_fixture(path, model)
            loaded = adapter.load_portable_binary(path, digest, model["names"])
            self.assertEqual(loaded["class_order"], [adapter.AI, adapter.OTHER])
            with path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                adapter.load_portable_binary(path, digest, model["names"])
            for invalid in (np.zeros(len(model["names"])), np.full(len(model["names"]), np.nan)):
                digest = dump_fixture(path, model, scaler_scale=invalid)
                with self.assertRaises(ValueError):
                    adapter.load_portable_binary(path, digest, model["names"])

    def test_pickle_parameters_are_never_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model = model_fixture()
            digest = dump_fixture(path, model, coefficients=np.asarray([[object()]], dtype=object))
            with self.assertRaises(ValueError):
                adapter.load_portable_binary(path, digest, model["names"])

    def test_saved_predictions_exact_ids_hash_order_and_tolerance(self):
        row, model = row_fixture(), model_fixture()
        values = np.asarray([[.8, .2]])
        saved = [{"id": row["id"], "variant": adapter.MODELS[0], "true_label": row["label"], "action_executed": False,
                  "model_artifact_sha256": model["sha256"], "predicted_label": adapter.AI,
                  "probabilities": {adapter.OTHER: .2, adapter.AI: .8}}]
        self.assertEqual(adapter.compare_saved_predictions([row], values, model, saved, adapter.MODELS[0])["max_absolute_difference"], 0)
        for changed in ([], saved + saved, [{**saved[0], "model_artifact_sha256": "b" * 64}],
                        [{**saved[0], "probabilities": {adapter.OTHER: .3, adapter.AI: .7}}]):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    adapter.compare_saved_predictions([row], values, model, changed, adapter.MODELS[0])
        with self.assertRaisesRegex(ValueError, "cannot be relaxed"):
            adapter.compare_saved_predictions([row], values, model, saved, adapter.MODELS[0], tolerance=.1)

    def test_external_authorization_is_source_specific(self):
        good = {"history_inference_authorized": True, "history_adapter_source_sha256": "a" * 64,
                "history_models": list(adapter.MODELS), "history_threshold": .5}
        adapter.validate_authorization(good, "a" * 64)
        for change in ({"history_inference_authorized": False}, {"history_adapter_source_sha256": "b" * 64},
                       {"history_models": list(adapter.MODELS)[1:]}, {"history_threshold": .4}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                adapter.validate_authorization({**good, **change}, "a" * 64)

    def test_unique_output_never_overwrites_previous_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "completion.json"
            adapter.save_new(path, {"original": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                adapter.save_new(path, {"original": False})
            self.assertEqual(path.read_bytes(), before)

    def test_external_partition_no_tuning_or_identity_claim(self):
        rows = []
        for index in range(630):
            row = row_fixture()
            row.update(id=f"external-{index}", site="itu", subject=f"itu:{index % 25}", recording_id=f"itu:r{index % 25}",
                       split="external_test", human_identity_disjointness_verified=False, cohort_protocol_sha256=adapter.EXTERNAL_FILES["protocol.json"],
                       sampling_design="case_control_enriched", label_source="recorded_event", target_kind="observed_action", data_origin="real_recording",
                       sample_rate_hz=256, channels=list(common.CHANNELS), clock_sync={"verified": True},
                       environment_source="predecision_harness", environment_contains_target_label=False, environment_time_s=9.,
                       feature_availability_verified=False, assumed_filter_lookahead_s=4.125, feature_available_time_s=15.875,
                       eeg_path=f"waveforms/{index % 25}.npy", eeg_index=index // 25,
                       eeg_source_sha256=hashlib.sha256(str(index % 25).encode()).hexdigest(),
                       label=adapter.AI if index < 126 else "keyboard" if index < 378 else "mouse_click")
            rows.append(row)
        membership = {"partition": "external_test", "training_ids": [], "validation_ids": [], "model_selection_allowed": False,
                      "human_identity_disjointness_verified": False, "manifest_sha256": adapter.EXTERNAL_FILES["manifest.jsonl"],
                      "external_test_ids": [row["id"] for row in rows]}
        self.assertFalse(adapter.validate_external_rows(rows, membership, history, common, [])["human_identity_disjointness_verified"])
        with self.assertRaisesRegex(ValueError, "membership"):
            adapter.validate_external_rows(rows, {**membership, "training_ids": [rows[0]["id"]]}, history, common, [])
        with self.assertRaisesRegex(ValueError, "raw source identity"):
            adapter.validate_external_rows(rows, membership, history, common, [{"id": "uniba", "eeg_source_sha256": rows[0]["eeg_source_sha256"]}])
        for field, value in (("feature_availability_verified", True), ("assumed_filter_lookahead_s", 0.), ("feature_available_time_s", 14.)):
            old = rows[0][field]
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "assumed filter-availability"):
                adapter.validate_external_rows(rows, membership, history, common, [])
            rows[0][field] = old
        rows[0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "site/split"):
            adapter.validate_external_rows(rows, membership, history, common, [])


if __name__ == "__main__":
    unittest.main()
