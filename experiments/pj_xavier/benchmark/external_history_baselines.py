#!/usr/bin/env python3
"""Inference-only transfer of four frozen UniBa history models to ITU.

This module never calls fit, fits a vocabulary, selects C, or changes a threshold.
First reproduce the original UniBa probabilities from portable coefficients;
only a separately pinned, authorized external protocol permits ITU prediction.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

AI = "ai_prompt_submit"
OTHER = "other_observed_action"
MODES = ("history_pre_eeg", "history_pre_eeg_plus_context", "history_pre_decision_guarded", "history_pre_decision_guarded_plus_context")
MODELS = tuple("ai_vs_other:" + mode for mode in MODES)
TOLERANCE = 1e-10
THRESHOLD = .5
HISTORY_SHA = "5be7b7c4e91f6776895e2454adbec2ba12bfba1548e572e69b61ce7b9704ea13"
COMMON_SHA = "1b74445e9a85c34c86eaa435805b427ba8dbee4d675f6d3e0a4b301b1c10ff08"
UNIBA_SHA = "030e8550bd6f4c79005722eb617e9b64f5f80e87b77866b7f17bec4e7e2a1093"
UNIBA_PROTOCOL_SHA = "0324dbbbc10706738723265b36d3478c0982714fdf3ab4597feb20eefcaa7cf5"
BASELINE_FILES = {
    "protocol.json": "102593d8c1e5fd0c69600950e93f37aae098f90af55194cdc7fdd314ab06f25a",
    "completion.json": "e7ff81f57cc6286e7942cbb1aa1119814dbd98fb3edfd398ef31356b06c79232",
    "selection.json": "0576e0d46e2add279e8a1145431a51c00f8d978772acd37c95f2bb786af83631",
    "history_report.json": "8444307baa019ca4de01987962d7473b095ec4b943be8620bf3759af711fd795",
    "test_predictions.jsonl": "884bd99377c45103502a260aa778ddd760f54ccb60b88a3976107b6ec9eb3f75",
    "ai_vs_other__history_pre_eeg.npz": "eefc3dbe70c123a532d792ed622cdbf63720833993baa7ee17592830afc5ec63",
    "ai_vs_other__history_pre_eeg_plus_context.npz": "255eecc1b295861d325101da7b1f415a001b22b8e603213510ad3ea0e4968701",
    "ai_vs_other__history_pre_decision_guarded.npz": "dbf67b450c1c57cabdc44896e575b960172a21ddc0174bddc379c5f14ab9e6be",
    "ai_vs_other__history_pre_decision_guarded_plus_context.npz": "4161c5f87730bc2a07984541e89c81a6a0c045d419dec564e9a18068a70875c0",
}
EXTERNAL_FILES = {
    "manifest.jsonl": "cfd93863b8f8e263174e6ab59395496efbd397e821542fa3a1f0c1e5faa503b0",
    "history_features.jsonl": "1326aef4ea7b0a3010f21c13cff56f7e476ff6da948673dac99dd8974ab853c9",
    "labels.jsonl": "59933900a4998579bcf3d0484d0b07dc128859b2a8cbb023e9362db3ddab5d41",
    "external_membership.json": "66351f4e59cf84f6e51eff0032139ab685327a2afbf6916ec5dcb9bdb52f3762",
    "eeg_artifacts.json": "87d5e0b944df33ac27f4101308022a9aa769c1080f043332a2d06feb0d2e3f02",
    "protocol.json": "6dae42f26833eab36028b886907a4cfcf9047f3ad76af182c497129cd885eca0",
    "independent_validation.json": "4c9d066e99a8db6e72d192565b3c6d7406a7253cb7b84ab9a66212cc262ce7a5",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_load(path):
    return json.loads(Path(path).read_text())


def json_lines(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_new(path, value):
    """Publish atomically without replacing another attempt or evidence file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def verify_files(root, expected):
    root = Path(root)
    for name, digest in expected.items():
        path = root / name
        require(path.is_file() and not path.is_symlink() and sha(path) == digest, "immutable file mismatch: " + name)


def load_module(path, expected, name):
    require(sha(path) == expected, "frozen feature source SHA mismatch")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_portable_binary(path, expected_sha, selected_names):
    require(sha(path) == expected_sha, "portable model SHA mismatch")
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == {"classes", "coefficients", "intercept", "scaler_mean", "scaler_scale", "feature_names", "solver"}, "portable model schema mismatch")
        model = {key: data[key].copy() for key in data.files}
    classes = model["classes"].tolist()
    names = model["feature_names"].tolist()
    require(isinstance(classes, list) and len(classes) == 2 and set(classes) == {AI, OTHER}, "binary class identity/order invalid")
    require(isinstance(names, list) and names == selected_names and len(names) == len(set(names)), "feature names/order differ from frozen selection")
    n = len(names)
    require(n > 0 and model["coefficients"].shape == (1, n) and model["intercept"].shape == (1,), "binary logistic coefficient shape invalid")
    require(model["scaler_mean"].shape == (n,) and model["scaler_scale"].shape == (n,), "scaler shape invalid")
    for key in ("coefficients", "intercept", "scaler_mean", "scaler_scale"):
        require(model[key].dtype.kind == "f" and np.isfinite(model[key]).all(), "nonfinite/nonnumeric portable parameter")
    require((model["scaler_scale"] > 0).all(), "nonpositive portable scale")
    require(model["solver"].shape == () and model["solver"].item() == "multinomial logistic, lbfgs, balanced class weights", "unrecognized solver metadata")
    model["names"] = names
    model["class_order"] = classes
    model["sha256"] = expected_sha
    return model


def feature_matrix(rows, mode, model, history, common):
    """Exactly the saved vocabulary; unknown categories have no column."""
    require(mode in MODES and bool(rows), "unknown history mode or empty rows")
    family = mode.removesuffix("_plus_context")
    vectors = [history.history_vector(row, family) for row in rows]
    history_names = vectors[0][1]
    require(all(names == history_names for _, names in vectors), "history feature order changed")
    names = model["names"]
    require(names[:len(history_names)] == history_names, "saved history prefix/order mismatch")
    matrix = np.asarray([values for values, _ in vectors], dtype=np.float64)
    context_names = names[len(history_names):]
    if mode.endswith("_plus_context"):
        require(bool(context_names), "saved context vocabulary missing")
        allowed_keys = {"page", "ai_availability", "page_ai_pair"}
        for name in context_names:
            require(name.startswith("context:") and "=" in name, "invalid saved context column")
            key, value = name[len("context:"):].split("=", 1)
            require(key in allowed_keys and bool(value), "nonallowlisted saved context column")
        onehot = []
        for row in rows:
            context = common.parse_context(row)
            require(set(context) == allowed_keys and all(isinstance(value, str) for value in context.values()), "context mapper contract changed")
            active = {"context:" + key + "=" + value for key, value in context.items()}
            onehot.append([float(name in active) for name in context_names])
        matrix = np.column_stack((matrix, np.asarray(onehot, dtype=np.float64)))
    else:
        require(not context_names, "context columns in history-only model")
    require(matrix.shape == (len(rows), len(names)) and np.isfinite(matrix).all(), "feature matrix invalid")
    return matrix


def predict_portable(model, features):
    """sklearn binary convention: sigmoid(z) is probability of classes_[1]."""
    x = np.asarray(features, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] == len(model["names"]) and np.isfinite(x).all(), "prediction feature shape or finiteness invalid")
    scaled = (x - model["scaler_mean"]) / model["scaler_scale"]
    scores = (scaled @ model["coefficients"].T + model["intercept"])[:, 0]
    require(np.isfinite(scores).all(), "nonfinite logistic score")
    second = expit(scores)
    result = np.column_stack((1.0 - second, second))
    require(np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all(), "invalid logistic probabilities")
    return result


def compare_saved_predictions(rows, predicted, model, saved, model_key, tolerance=TOLERANCE):
    require(0 < tolerance <= TOLERANCE, "parity tolerance cannot be relaxed")
    ids = [row["id"] for row in rows]
    reference = {}
    for record in saved:
        if record.get("variant") != model_key:
            continue
        require(record.get("id") not in reference, "duplicate saved prediction ID")
        reference[record["id"]] = record
    require(len(ids) == len(set(ids)) and set(reference) == set(ids), "saved prediction IDs differ from exact UniBa test partition")
    require(predicted.shape == (len(rows), 2), "parity probability shape mismatch")
    maximum = 0.0
    for row, values in zip(rows, predicted):
        old = reference[row["id"]]
        require(old.get("true_label") == row["label"] and old.get("action_executed") is False and old.get("model_artifact_sha256") == model["sha256"], "saved prediction provenance mismatch")
        require(set(old.get("probabilities", {})) == set(model["class_order"]), "saved probability class set mismatch")
        expected = np.asarray([old["probabilities"][label] for label in model["class_order"]], dtype=np.float64)
        require(np.isfinite(expected).all() and np.all((expected >= 0) & (expected <= 1)) and abs(float(expected.sum()) - 1) <= tolerance, "saved probability invalid")
        difference = float(np.max(np.abs(values - expected)))
        require(difference <= tolerance, "portable binary probabilities do not reproduce UniBa saved predictions")
        require(old.get("predicted_label") == model["class_order"][int(np.argmax(values))], "portable predicted class differs from saved prediction")
        maximum = max(maximum, difference)
    return {"status": "passed", "n": len(rows), "probability_values_compared": len(rows) * 2,
            "absolute_tolerance": tolerance, "max_absolute_difference": maximum,
            "class_order": model["class_order"], "selected_model_unchanged": True}


def verify_uniba(args):
    started = time.monotonic()
    source_sha = sha(__file__)
    require(source_sha == args.expected_source_sha256, "adapter source SHA mismatch")
    root = Path(args.baseline_run)
    verify_files(root, BASELINE_FILES)
    history = load_module(args.history_module, HISTORY_SHA, "external_frozen_history")
    common = load_module(args.common_module, COMMON_SHA, "external_frozen_cpu")
    protocol = json_load(root / "protocol.json")
    old_report, completion, selection = (json_load(root / name) for name in ("history_report.json", "completion.json", "selection.json"))
    require(old_report.get("status") == "completed" and old_report.get("source_sha256") == HISTORY_SHA
            and old_report.get("common_module_sha256") == COMMON_SHA and old_report.get("protocol_sha256") == UNIBA_PROTOCOL_SHA
            and old_report.get("manifest_sha256") == UNIBA_SHA and old_report.get("model_selection_partition") == "validation"
            and old_report.get("test_used_for_tuning") is False and selection.get("test_used_for_selection") is False,
            "original baseline completion/selection contract mismatch")
    require(completion.get("status") == "completed" and completion.get("exit_code") == 0
            and completion.get("report_sha256") == BASELINE_FILES["history_report.json"]
            and completion.get("test_prediction_sha256") == BASELINE_FILES["test_predictions.jsonl"]
            and completion.get("manifest_sha256") == UNIBA_SHA, "original baseline receipt mismatch")
    require(old_report.get("selection_sha256") == BASELINE_FILES["selection.json"]
            and old_report.get("prediction_sha256") == BASELINE_FILES["test_predictions.jsonl"], "original baseline chained hashes mismatch")
    rows, verification = common.verify_inputs(args.uniba_manifest, args.reference_run, protocol)
    test = [row for row in rows if row["split"] == "test"]
    require(len(test) == 203, "original UniBa test partition size mismatch")
    saved = json_lines(root / "test_predictions.jsonl")
    models, checks = {}, {}
    for key in MODELS:
        declared = selection["models"][key]
        filename = key.replace(":", "__") + ".npz"
        require(declared["artifact_sha256"] == BASELINE_FILES[filename], "selection model SHA mismatch")
        model = load_portable_binary(root / filename, BASELINE_FILES[filename], declared["feature_names"])
        mode = key.split(":", 1)[1]
        x = feature_matrix(test, mode, model, history, common)
        predicted = predict_portable(model, x)
        checks[key] = compare_saved_predictions(test, predicted, model, saved, key)
        models[key] = model
    verify_files(root, BASELINE_FILES)
    require(sha(__file__) == source_sha and sha(args.history_module) == HISTORY_SHA and sha(args.common_module) == COMMON_SHA, "source changed during parity verification")
    result = {"schema": "xavier-external-history-uniba-parity-v1", "status": "passed", "source_sha256": source_sha,
              "history_source_sha256": HISTORY_SHA, "common_source_sha256": COMMON_SHA, "uniba_manifest_sha256": UNIBA_SHA,
              "frozen_baseline_files": BASELINE_FILES, "models": checks, "cohort": verification,
              "model_fit_calls": 0, "vocabulary_fit_calls": 0, "C_selection_calls": 0,
              "probability_convention": "column 1 = sigmoid(standardized_x @ coefficient + intercept); column 0 = 1 - column 1; classes order read from each saved artifact",
              "artifact_solver_description_note": "The original portable solver field says multinomial logistic; the original two-class sklearn estimator uses binary logistic. Exact saved-prediction parity, not that descriptive string, authenticates the probability mapping.",
              "external_predictions_created": False, "elapsed_s": time.monotonic() - started}
    return models, history, common, rows, result


def validate_external_rows(rows, membership, history, common, uniba_rows):
    ids = [row.get("id") for row in rows]
    require(len(rows) == 630 and len(set(ids)) == 630 and all(isinstance(item, str) for item in ids), "external cohort IDs/count invalid")
    require(membership.get("partition") == "external_test" and membership.get("training_ids") == [] and membership.get("validation_ids") == []
            and membership.get("model_selection_allowed") is False and membership.get("human_identity_disjointness_verified") is False
            and membership.get("manifest_sha256") == EXTERNAL_FILES["manifest.jsonl"]
            and sorted(membership.get("external_test_ids", [])) == sorted(ids), "external membership contract mismatch")
    require(Counter(row["label"] for row in rows) == {AI: 126, "keyboard": 252, "mouse_click": 252}, "fixed external label counts changed")
    require(not (set(ids) & {row["id"] for row in uniba_rows}), "external ID overlaps UniBa")
    require(not ({row["eeg_source_sha256"] for row in rows} & {row["eeg_source_sha256"] for row in uniba_rows}), "raw source identity overlaps UniBa")
    sources, observed, windows = set(), set(), set()
    for row in rows:
        require(row.get("site") == "itu" and row.get("split") == "external_test" and row.get("subject", "").startswith("itu:")
                and row.get("recording_id", "").startswith("itu:") and row.get("human_identity_disjointness_verified") is False,
                "external site/split/identity declaration mismatch")
        require(row.get("cohort_protocol_sha256") == EXTERNAL_FILES["protocol.json"] and row.get("sampling_design") == "case_control_enriched"
                and row.get("label_source") == "recorded_event" and row.get("target_kind") == "observed_action"
                and row.get("data_origin") == "real_recording", "external observed-action protocol mismatch")
        require(row.get("sample_rate_hz") == 256 and tuple(row.get("channels", [])) == common.CHANNELS, "external montage/sample rate changed")
        start, end, decision, event = (float(row[k]) for k in ("window_start_s", "window_end_s", "decision_time_s", "event_time_s"))
        require(all(math.isfinite(v) for v in (start, end, decision, event)) and 0 <= start < end <= decision < event
                and abs(end - start - 1.75) < 1e-8 and row.get("clock_sync", {}).get("verified") is True,
                "external timing/clock mismatch")
        require(row.get("environment_source") == "predecision_harness" and row.get("environment_contains_target_label") is False
                and math.isfinite(float(row.get("environment_time_s", math.inf))) and float(row["environment_time_s"]) <= start,
                "external future context")
        # The frozen cohorts conservatively *assume* the 4.125 s device-filter
        # support; they explicitly do not claim measured online availability.
        # Authenticating that limitation is different from asserting it True.
        availability = float(row.get("feature_available_time_s", math.inf))
        require(row.get("feature_availability_verified") is False and row.get("assumed_filter_lookahead_s") == 4.125
                and math.isfinite(availability) and abs(availability - (end + 4.125)) <= 1e-8
                and availability <= decision + 1e-8, "external assumed filter-availability contract changed")
        key = (row["eeg_path"], row["eeg_index"])
        require(key not in windows, "external repeated array window")
        windows.add(key)
        common.parse_context(row)
        for family in history.FAMILIES:
            history.history_vector(row, family)
        sources.add(row["eeg_source_sha256"])
        observed.add((row["site"], row["subject"]))
    require(len(observed) == 25 and len(sources) == 25, "external observation/source membership changed")
    return {"n": len(rows), "label_counts": dict(Counter(row["label"] for row in rows)), "site_participant_observation_ids": len(observed),
            "raw_sources": len(sources), "train_or_validation_rows": 0, "cross_site_raw_sha_matches": 0,
            "human_identity_disjointness_verified": False,
            "identity_limit": "Site-qualified participant labels distinguish observations; matching short labels neither prove nor exclude the same human across sites."}


def verify_external(root, history, common, uniba_rows):
    root = Path(root)
    verify_files(root, EXTERNAL_FILES)
    validation = json_load(root / "independent_validation.json")
    require(validation.get("status") == "passed" and validation.get("manifest_sha256") == EXTERNAL_FILES["manifest.jsonl"]
            and validation.get("actual_waveform_windows_compared_sample_for_sample") == 630
            and validation.get("recorded_event_labels_rederived") == 630 and validation.get("past_history_feature_sets_rederived") == 1260
            and validation.get("train_or_validation_rows") == 0 and validation.get("failed_clock_records_used") == 0,
            "independent external raw/clock/history validation missing")
    rows = json_lines(root / "manifest.jsonl")
    result = validate_external_rows(rows, json_load(root / "external_membership.json"), history, common, uniba_rows)
    artifacts = json_load(root / "eeg_artifacts.json")
    require(set(artifacts) == {row["eeg_path"] for row in rows} and len(artifacts) == 25, "external artifact set mismatch")
    total_bytes = 0
    for name, record in artifacts.items():
        path = common.safe_child(root, name)
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "external EEG artifact SHA/bytes changed")
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        subset = [row for row in rows if row["eeg_path"] == name]
        require(data.dtype == np.dtype("float32") and data.shape == (len(subset), 8, 448) and np.isfinite(data).all()
                and sorted(row["eeg_index"] for row in subset) == list(range(len(subset))), "external array/index shape invalid")
        total_bytes += record["bytes"]
    require(total_bytes == 9034880, "external prepared bytes differ")
    result.update(prepared_artifact_files=25, prepared_artifact_bytes=total_bytes,
                  independent_raw_validation_sha256=EXTERNAL_FILES["independent_validation.json"],
                  current_run_raw_rederivation=False, current_run_prepared_hash_reverification=True)
    return rows, result


def validate_authorization(protocol, source_sha):
    require(protocol.get("history_inference_authorized") is True and protocol.get("history_adapter_source_sha256") == source_sha,
            "external history inference not authorized for this immutable source")
    # The shared neural protocol may contain extra sections. Any history-specific
    # overrides must exactly match the fixed estimator list and .5 endpoint.
    if "history_models" in protocol:
        require(protocol["history_models"] == list(MODELS), "external history model list changed")
    if "history_threshold" in protocol:
        require(type(protocol["history_threshold"]) in (int, float) and protocol["history_threshold"] == THRESHOLD, "external threshold retuning forbidden")


def runtime_versions():
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")}}


def run(args):
    started = time.monotonic()
    protocol_path = Path(args.protocol)
    protocol_sha, source_sha = sha(protocol_path), sha(__file__)
    require(protocol_sha == args.expected_protocol_sha256 and source_sha == args.expected_source_sha256, "external protocol/source SHA mismatch")
    protocol = json_load(protocol_path)
    validate_authorization(protocol, source_sha)
    # Creating this unique output root is the one-attempt claim. Failure retains
    # evidence; an existing root is never reused, even if it lacks completion.
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "attempt_started.json", {"started_epoch": time.time(), "protocol_sha256": protocol_sha,
             "source_sha256": source_sha, "models": list(MODELS), "fit_permitted": False, "external_predict_calls_per_model_max": 1})
    save_new(output / "protocol.json", protocol)
    models, history, common, uniba_rows, parity = verify_uniba(args)
    save_new(output / "uniba_parity.json", parity)
    rows, cohort = verify_external(args.external_prepared, history, common, uniba_rows)
    save_new(output / "external_inputs_verified.json", cohort)
    save_new(output / "external_test_consumed.json", {"started_epoch": time.time(), "protocol_sha256": protocol_sha,
             "manifest_sha256": EXTERNAL_FILES["manifest.jsonl"], "models": list(MODELS), "threshold": THRESHOLD})
    # Every prediction is produced before any external performance is calculated.
    all_probabilities = {}
    for key in MODELS:
        all_probabilities[key] = predict_portable(models[key], feature_matrix(rows, key.split(":", 1)[1], models[key], history, common))
    prediction_path = output / "external_predictions.jsonl"
    with prediction_path.open("x") as stream:
        for key in MODELS:
            model = models[key]
            for row, values in zip(rows, all_probabilities[key]):
                result = {"id": row["id"], "split": "external_test", "variant": key, "true_label": row["label"],
                          "predicted_label": model["class_order"][int(np.argmax(values))],
                          "probabilities": dict(zip(model["class_order"], map(float, values))),
                          "model_artifact_sha256": model["sha256"], "manifest_sha256": EXTERNAL_FILES["manifest.jsonl"],
                          "action_executed": False}
                stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    truth = np.asarray([row["label"] == AI for row in rows], dtype=int)
    scores = {key: values[:, models[key]["class_order"].index(AI)] for key, values in all_probabilities.items()}
    measured = {key: {"ai_vs_other": history.binary_metrics(truth, values, THRESHOLD),
                     "model_artifact_sha256": models[key]["sha256"], "external_predict_calls": 1} for key, values in scores.items()}
    subjects = [row["site"] + ":" + row["subject"] for row in rows]
    uncertainty = history.clustered_comparison(truth, subjects, scores, repetitions=1000, seed=20261122)
    uncertainty["limits"] = "Conditional paired participant-observation-cluster uncertainty for 25 ITU site-qualified observations in an enriched case-control sample. Not repeated-fit uncertainty, natural prevalence, multiplicity-adjusted proof, online false-action rate, or verified cross-site human disjointness."
    verify_files(args.baseline_run, BASELINE_FILES)
    verify_files(args.external_prepared, EXTERNAL_FILES)
    _, final_cohort = verify_external(args.external_prepared, history, common, uniba_rows)
    require(final_cohort == cohort and sha(__file__) == source_sha and sha(protocol_path) == protocol_sha
            and sha(args.history_module) == HISTORY_SHA and sha(args.common_module) == COMMON_SHA, "immutable inputs changed during external inference")
    report = {"schema": "xavier-itu-external-history-inference-v1", "status": "completed", "source_sha256": source_sha,
              "protocol_sha256": protocol_sha, "manifest_sha256": EXTERNAL_FILES["manifest.jsonl"], "external_inputs": EXTERNAL_FILES,
              "frozen_baseline_files": BASELINE_FILES, "cohort": cohort, "models": measured,
              "subject_cluster_uncertainty": uncertainty, "prediction_sha256": sha(prediction_path),
              "uniba_parity_sha256": sha(output / "uniba_parity.json"), "versions": runtime_versions(),
              "fit_calls": 0, "vocabulary_fit_calls": 0, "C_selection_calls": 0, "threshold": THRESHOLD,
              "external_used_for_tuning": False, "selection_preceded_external_inference": True,
              "unknown_context_categories": "All zero for categories absent from the saved UniBa feature_names vocabulary; no test-derived vocabulary.",
              "history_feature_scope": "Frozen past-event counts, recency and source/missing indicators, optionally saved coarse context only. Labels, absolute times, IDs and future events are never estimator inputs.",
              "precision_scope": "Observed AI request transmission versus other recorded actions, enriched case-control sample; not prompt content or latent intention.",
              "natural_population_precision_estimated": False, "false_actions_per_minute": None, "task_completion": None,
              "online_causality_verified": False, "autonomous_agent_validated": False, "cpu_only": True,
              "native_thread_limit": 2, "elapsed_s": time.monotonic() - started}
    save_new(output / "external_history_report.json", report)
    save_new(output / "completion.json", {"status": "completed", "exit_code": 0, "report_sha256": sha(output / "external_history_report.json"),
             "prediction_sha256": sha(prediction_path), "manifest_sha256": EXTERNAL_FILES["manifest.jsonl"], "protocol_sha256": protocol_sha,
             "source_sha256": source_sha, "models": list(MODELS), "n_per_model": len(rows)})
    print(json.dumps({"status": "completed", "models": measured, "report_sha256": sha(output / "external_history_report.json"), "elapsed_s": report["elapsed_s"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-uniba", "run"):
        sub = subs.add_parser(name)
        for arg in ("baseline-run", "uniba-manifest", "reference-run", "history-module", "common-module", "output", "expected-source-sha256"):
            sub.add_argument("--" + arg, required=True)
        if name == "run":
            for arg in ("protocol", "expected-protocol-sha256", "external-prepared"):
                sub.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.command == "run":
            run(args)
        else:
            _, _, _, _, report = verify_uniba(args)
            report["versions"] = runtime_versions()
            save_new(args.output, report)
            print(json.dumps({"status": report["status"], "models": report["models"], "report_sha256": sha(args.output), "external_predictions_created": False}), flush=True)


if __name__ == "__main__":
    main()
