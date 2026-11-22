#!/usr/bin/env python3
"""Frozen-split CPU logistic baselines for the existing UniBa action cohort.

No LLM, GPU, raw-file edits, relabeling, or test-based model selection. Run once
per predeclared output directory. The same held-out subjects as the LLM are used.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import resource
import time
import warnings

import numpy as np
from scipy.signal import welch
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits


VERSION = "pj-xavier-cpu-baselines-v1"
MODES = ("eeg_spectrum_only", "coarse_context_only", "eeg_plus_coarse_context")
SPLITS = ("train", "val", "test")
CHANNELS = ("CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4")
BANDS = (("theta", 4., 8.), ("alpha", 8., 13.), ("beta", 13., 30.), ("low_gamma", 30., 40.))
CONTEXT_PATTERN = re.compile(r"Java programming experiment\. Last recorded experiment page or event: ([A-Za-z][A-Za-z0-9_]{0,63})\. AI availability recorded at that event: (True|False|None|unknown)\.")
SHA = re.compile(r"[0-9a-f]{64}")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def safe_child(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError("prepared EEG paths must be relative and contained")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()) or result.is_symlink():
        raise ValueError("prepared EEG path escapes root or is a symlink")
    return result


def parse_context(row):
    """Only the two predeclared coarse variables, never IDs/times/target text."""
    match = CONTEXT_PATTERN.fullmatch(row.get("environment_text", ""))
    if not match:
        raise ValueError("coarse context does not match the upstream declared template")
    page, available = match.groups()
    return {"page": page, "ai_availability": available, "page_ai_pair": page + "|" + available}


def validate_rows(rows, membership, expected_counts=None):
    ids, prepared_examples = set(), set()
    subjects, physical_sources, prepared_files = defaultdict(set), defaultdict(set), defaultdict(set)
    counts = {s: Counter() for s in SPLITS}
    for row in rows:
        required = ("id", "dataset", "subject", "session", "recording_id", "split", "label", "label_source", "eeg_path", "eeg_index",
                    "eeg_source_sha256", "label_evidence_sha256", "environment_sha256", "window_start_s", "window_end_s", "decision_time_s", "event_time_s")
        if any(key not in row for key in required):
            raise ValueError("required upstream record fields missing")
        if row["id"] in ids or row["split"] not in SPLITS:
            raise ValueError("duplicate ID or invalid split")
        ids.add(row["id"])
        sample_key = (row["eeg_path"], row["eeg_index"])
        if sample_key in prepared_examples:
            raise ValueError("same prepared EEG window appears more than once")
        prepared_examples.add(sample_key)
        if row.get("data_origin") != "real_recording" or row["label_source"] != "recorded_event" or row.get("target_kind") != "observed_action":
            raise ValueError("baseline requires real, recorded observed-action labels")
        if tuple(row.get("channels", [])) != CHANNELS or row.get("sample_rate_hz") != 256.0:
            raise ValueError("montage or sample-rate differs from this frozen protocol")
        if any(not SHA.fullmatch(str(row[key])) for key in ("eeg_source_sha256", "label_evidence_sha256", "environment_sha256")):
            raise ValueError("source provenance SHA-256 missing")
        start, end, decision, event = (float(row[k]) for k in ("window_start_s", "window_end_s", "decision_time_s", "event_time_s"))
        if not all(math.isfinite(x) for x in (start, end, decision, event)) or not 0 <= start < end <= decision < event:
            raise ValueError("invalid pre-action time ordering")
        if decision > event - float(row.get("minimum_lead_s", 0)) + 1e-8:
            raise ValueError("pre-action guard violated")
        if row.get("feature_available_time_s", math.inf) > decision + 1e-8:
            raise ValueError("feature availability is after decision")
        if (row.get("environment_source") != "predecision_harness" or row.get("environment_contains_target_label") is not False
                or not math.isfinite(float(row.get("environment_time_s", float("nan"))))
                or float(row["environment_time_s"]) > start or row.get("clock_sync", {}).get("verified") is not True):
            raise ValueError("context or clock is not predecision verified metadata")
        parse_context(row)
        dataset = row.get("canonical_dataset_id", row["dataset"])
        subject = row.get("canonical_subject_id", row["subject"])
        subjects[(dataset, subject)].add(row["split"])
        physical_sources[row["eeg_source_sha256"]].add(row["split"])
        prepared_files[row["eeg_path"]].add(row["split"])
        counts[row["split"]][row["label"]] += 1
    if any(len(s) != 1 for mapping in (subjects, physical_sources, prepared_files) for s in mapping.values()):
        raise ValueError("subject or physical EEG source crosses frozen subject-held-out splits")
    if expected_counts and {s: sum(counts[s].values()) for s in SPLITS} != expected_counts:
        raise ValueError("split sample counts changed from the frozen protocol")
    for split, key in (("train", "training_ids"), ("val", "validation_ids"), ("test", "test_ids")):
        actual = sorted(row["id"] for row in rows if row["split"] == split)
        declared = membership.get(key, [])
        if not actual or actual != sorted(declared) or len(declared) != len(set(declared)):
            raise ValueError("split IDs differ from the LLM training membership:" + split)
    classes = set(counts["train"])
    if len(classes) < 2 or any(set(counts[s]) - classes for s in ("val", "test")):
        raise ValueError("unseen evaluation class or fewer than two training classes")
    return {"label_counts": {s: dict(sorted(counts[s].items())) for s in SPLITS},
            "samples": {s: sum(counts[s].values()) for s in SPLITS},
            "subjects": {s: sum(s in v for v in subjects.values()) for s in SPLITS},
            "subject_disjoint": True, "physical_source_disjoint": True,
            "split_id_hashes": {s: hashlib.sha256("\n".join(sorted(r["id"] for r in rows if r["split"] == s)).encode()).hexdigest() for s in SPLITS}}


def verify_inputs(manifest, reference_run, protocol):
    manifest, reference_run = Path(manifest), Path(reference_run)
    if sha256(manifest) != protocol["manifest_sha256"]:
        raise ValueError("frozen manifest SHA-256 mismatch")
    reference = {}
    for name, expected in protocol["reference_file_sha256"].items():
        path = reference_run / name
        if sha256(path) != expected:
            raise ValueError("immutable LLM reference changed:" + name)
        reference[name] = json.loads(path.read_text())
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    membership = reference["training_membership.json"]
    if membership.get("manifest_sha256") != protocol["manifest_sha256"] or membership.get("test_used_for_tuning") is not False:
        raise ValueError("reference membership manifest or tuning declaration mismatch")
    validation = validate_rows(rows, membership, protocol["split_counts"])
    declared = reference["eeg_artifacts.json"]
    keys = {r["eeg_path"] for r in rows}
    if keys != set(declared):
        raise ValueError("prepared EEG artifact membership differs from LLM reference")
    total_bytes = 0
    for key in sorted(keys):
        path = safe_child(manifest.parent, key)
        size = path.stat().st_size
        if size != declared[key]["bytes"] or sha256(path) != declared[key]["sha256"]:
            raise ValueError("prepared EEG bytes differ from the LLM training inputs:" + key)
        total_bytes += size
    validation.update(prepared_artifact_files=len(keys), prepared_artifact_bytes=total_bytes,
                      prepared_artifact_manifest_sha256=canonical_sha(declared),
                      raw_archive_reverification="not performed here; prepared waveform bytes matched the running LLM's frozen artifact ledger")
    return rows, validation


def spectrum_features(eeg, sample_rate=256.):
    values = np.asarray(eeg, dtype=np.float64)
    if values.shape != (8, 448) or not np.isfinite(values).all():
        raise ValueError("expected finite common-montage EEG [8,448]")
    frequencies, psd = welch(values, fs=sample_rate, window="hann", nperseg=256, noverlap=128,
                             nfft=256, detrend="constant", return_onesided=True, scaling="density", axis=-1, average="mean")
    df = float(frequencies[1] - frequencies[0])
    total = np.sum(psd[:, (frequencies >= 3) & (frequencies < 40)], axis=1) * df
    total = np.maximum(total, 1e-30)
    powers = np.stack([np.sum(psd[:, (frequencies >= lo) & (frequencies < hi)], axis=1) * df for _, lo, hi in BANDS], axis=1)
    features = np.concatenate((np.log10(np.maximum(powers, 1e-30)), np.log10(np.maximum(powers / total[:, None], 1e-30)),
                               np.log10(total[:, None])), axis=1)
    return features.reshape(-1)


def eeg_feature_names():
    return [channel + ":" + feature for channel in CHANNELS for feature in
            (["log10_abs_" + name for name, _, _ in BANDS] + ["log10_relative_" + name for name, _, _ in BANDS] + ["log10_total_3_40Hz"])]


def extract_eeg(rows, manifest_root):
    arrays, features = {}, []
    for row in rows:
        key = row["eeg_path"]
        if key not in arrays:
            arrays[key] = np.load(safe_child(manifest_root, key), allow_pickle=False, mmap_mode="r")
        array = arrays[key]
        index = row["eeg_index"]
        if not isinstance(index, int) or not 0 <= index < len(array):
            raise ValueError("EEG index outside prepared array")
        features.append(spectrum_features(array[index], row["sample_rate_hz"]))
    return np.asarray(features, dtype=np.float64)


def fit_context(train_rows, val_rows):
    vectorizer = DictVectorizer(sparse=False, sort=True)
    train = vectorizer.fit_transform([parse_context(row) for row in train_rows])
    val = vectorizer.transform([parse_context(row) for row in val_rows])
    return vectorizer, train, val


def metrics(y_true, y_pred, classes):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    confusion = confusion_matrix(y_true, y_pred, labels=classes)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
            "samples": len(y_true), "class_order": list(classes), "confusion_rows_true_columns_predicted": confusion.tolist(),
            "per_class": {label: {"support": int(support[i]), "predicted": int(confusion[:, i].sum()),
                                  "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                          for i, label in enumerate(classes)}}


def select_model(x_train, y_train, x_val, y_val, classes, c_grid, *, max_iter=2000):
    """This API intentionally has no test argument; scaling and fit use train only."""
    scaler = StandardScaler().fit(x_train)
    train_scaled, val_scaled = scaler.transform(x_train), scaler.transform(x_val)
    trials, candidates = [], []
    for strength in sorted(set(c_grid)):
        if not math.isfinite(strength) or strength <= 0:
            raise ValueError("regularization C must be positive and finite")
        model = LogisticRegression(C=strength, penalty="l2", solver="lbfgs", class_weight="balanced",
                                   max_iter=max_iter, tol=1e-6, random_state=20261122)
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_scaled, y_train)
        convergence = [str(w.message) for w in observed if issubclass(w.category, ConvergenceWarning)]
        score = metrics(y_val, model.predict(val_scaled), classes)
        trials.append({"C": strength, "val": score, "converged": not convergence, "iterations": model.n_iter_.tolist(),
                       "fit_seconds": time.monotonic() - started, "convergence_warnings": convergence})
        if not convergence:
            candidates.append(((score["balanced_accuracy"], score["macro_f1"], -strength), model, trials[-1]))
    if not candidates:
        raise ValueError("no C candidate converged; no test prediction may be made")
    _, selected, trial = max(candidates, key=lambda item: item[0])
    return scaler, selected, {"selected_C": trial["C"], "selected_val": trial["val"], "selection": "maximum validation balanced accuracy; macro-F1 then smaller C break ties",
                              "trials": trials, "train_examples": len(y_train), "val_examples": len(y_val), "feature_count": x_train.shape[1]}


def claim_test_once(output, selection_sha):
    marker = Path(output) / "test_consumed.json"
    with marker.open("x") as f:
        json.dump({"selection_sha256": selection_sha, "started_epoch": time.time(),
                   "policy": "one prediction call per predeclared model; no refit or retuning after this marker"}, f, indent=2)


def save_portable_model(path, scaler, model, feature_names):
    np.savez_compressed(path, classes=np.asarray(model.classes_), coefficients=model.coef_, intercept=model.intercept_,
                        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, feature_names=np.asarray(feature_names),
                        solver=np.asarray("multinomial logistic, lbfgs, balanced class weights"))


def run(protocol_path, manifest, reference_run, output, expected_source_sha256):
    started = time.monotonic()
    source_path, protocol_path = Path(__file__).resolve(), Path(protocol_path).resolve()
    if sha256(source_path) != expected_source_sha256:
        raise ValueError("immutable baseline source hash mismatch")
    source_sha, protocol_sha = sha256(source_path), sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output already exists; do not repeat or overwrite a frozen test experiment")
    output.mkdir(parents=True)
    save_json(output / "protocol.json", protocol)
    rows, validation = verify_inputs(manifest, reference_run, protocol)
    partitions = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
    classes = sorted({row["label"] for row in partitions["train"]})
    provenance = {"source_sha256": source_sha, "protocol_sha256": protocol_sha, "manifest_sha256": protocol["manifest_sha256"],
                  "reference_file_sha256": protocol["reference_file_sha256"], "validation": validation,
                  "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
                  "platform": platform.platform(), "cpu_only": True, "gpu_imports_or_calls": 0,
                  "thread_environment": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}
    save_json(output / "provenance.json", provenance)
    vectorizer, context_train, context_val = fit_context(partitions["train"], partitions["val"])
    eeg_train = extract_eeg(partitions["train"], Path(manifest).parent)
    eeg_val = extract_eeg(partitions["val"], Path(manifest).parent)
    context_names = ["context:" + name for name in vectorizer.get_feature_names_out()]
    inputs = {"eeg_spectrum_only": (eeg_train, eeg_val, eeg_feature_names()),
              "coarse_context_only": (context_train, context_val, context_names),
              "eeg_plus_coarse_context": (np.column_stack((eeg_train, context_train)), np.column_stack((eeg_val, context_val)), eeg_feature_names() + context_names)}
    y_train = np.asarray([row["label"] for row in partitions["train"]])
    y_val = np.asarray([row["label"] for row in partitions["val"]])
    models, selection = {}, {}
    for mode in MODES:
        x_train, x_val, names = inputs[mode]
        scaler, model, chosen = select_model(x_train, y_train, x_val, y_val, classes, protocol["C_grid"])
        models[mode] = scaler, model
        save_portable_model(output / (mode + ".npz"), scaler, model, names)
        selection[mode] = chosen
        print(json.dumps({"stage": "validation_selected", "mode": mode, "C": chosen["selected_C"], "val_balanced_accuracy": chosen["selected_val"]["balanced_accuracy"]}), flush=True)
    save_json(output / "selection.json", {"models": selection, "test_used_for_selection": False, "selection_frozen_epoch": time.time()})
    # Recheck source, manifest, reference splits, and every prepared EEG file before
    # the one held-out prediction pass. A concurrent input change fails closed.
    if sha256(source_path) != source_sha or sha256(protocol_path) != protocol_sha:
        raise ValueError("baseline code or protocol changed during fitting")
    _, second_validation = verify_inputs(manifest, reference_run, protocol)
    if validation != second_validation:
        raise ValueError("input provenance changed during fitting")
    selection_sha = sha256(output / "selection.json")
    claim_test_once(output, selection_sha)
    eeg_test = extract_eeg(partitions["test"], Path(manifest).parent)
    context_test = vectorizer.transform([parse_context(row) for row in partitions["test"]])
    test_inputs = {"eeg_spectrum_only": eeg_test, "coarse_context_only": context_test,
                   "eeg_plus_coarse_context": np.column_stack((eeg_test, context_test))}
    y_test = np.asarray([row["label"] for row in partitions["test"]])
    results = {}
    for mode in MODES:
        scaler, model = models[mode]
        tick = time.monotonic()
        predictions = model.predict(scaler.transform(test_inputs[mode]))
        results[mode] = {"selected_C": selection[mode]["selected_C"], "feature_count": test_inputs[mode].shape[1],
                         "test": metrics(y_test, predictions, classes), "test_predict_calls": 1,
                         "test_batch_predict_seconds": time.monotonic() - tick,
                         "model_artifact_sha256": sha256(output / (mode + ".npz"))}
    majority = sorted(Counter(y_train).items(), key=lambda item: (-item[1], item[0]))[0][0]
    majority_reference = metrics(y_test, np.full(len(y_test), majority), classes)
    # Confirm immutable inputs again after reading test data. No source/upstream
    # file is ever opened for writing by this program.
    _, final_validation = verify_inputs(manifest, reference_run, protocol)
    if final_validation != validation or sha256(source_path) != source_sha or sha256(protocol_path) != protocol_sha:
        raise ValueError("input/source changed during held-out evaluation")
    ai_support = {split: validation["label_counts"][split].get("ai_prompt_submit", 0) for split in SPLITS}
    report = {"schema_version": VERSION, "status": "completed", "manifest_sha256": protocol["manifest_sha256"],
              "source_sha256": source_sha, "protocol_sha256": protocol_sha, "selection_sha256": selection_sha,
              "cohort": validation, "classes": classes, "models": results,
              "train_majority_reference": {"selected_class_from_train": str(majority), "test": majority_reference},
              "regularization": {"penalty": "L2", "class_weight": "balanced", "C_grid": protocol["C_grid"],
                                  "fit_partition": "train", "selection_partition": "val", "test_used_for_tuning": False,
                                  "refit_train_plus_val": False, "selection": "val balanced accuracy, then val macro-F1, then smaller C"},
              "coarse_context_features": {"variables": ["previous_experiment_page_or_event", "AI_availability", "their_joint_category"],
                                         "encoding": "train-only DictVectorizer; unknown categories ignored", "features": context_names,
                                         "excluded": ["subject/session/record IDs", "absolute or relative timestamps", "target event and evidence-row number", "future actions", "prompt/key content"]},
              "spectral_features": {"sample_rate_hz": 256, "samples_per_window": 448, "channels": list(CHANNELS), "feature_count": 72,
                                    "welch": {"window": "hann", "nperseg": 256, "noverlap": 128, "nfft": 256, "detrend": "constant", "scaling": "density"},
                                    "bands_hz_lower_inclusive_upper_exclusive": [[name, lo, hi] for name, lo, hi in BANDS],
                                    "features": "per-channel log10 absolute and relative band power, plus log10 total power 3<=f<40 Hz; train-only feature standardization"},
              "ai_prompt_submit_support": ai_support,
              "ai_prompt_submit_performance_estimable": False,
              "ai_prompt_submit_interpretation": f"Train={ai_support['train']}, validation={ai_support['val']}, test={ai_support['test']} in this frozen cohort. Point metrics are descriptive only; this cannot support a useful generalization estimate of AI submission intent or operation.",
              "latent_intent_goal_validated": False, "autonomous_agent_validated": False,
              "limitations": ["Recorded event-conditioned action windows, not latent intent labels or continuous autonomous operation.",
                              "Published EEG is offline filtered; original XDF and binary-to-recipe equivalence are unverified; 4.125s lookahead is an assumption.",
                              "Mouse movement/EMG and phase or subject-state confounding remain possible.",
                              "32 subjects with 8 held-out subjects; windows are not independent participants; point metrics are not a population confidence claim.",
                              "Only 5 AI submission examples across all splits; three-class balanced accuracy and macro-F1 are highly sensitive to this rare class.",
                              "A difference from the LLM cannot be attributed only to parameter count: representation, optimization and EEG amplitude normalization differ.",
                              "This comparison cannot prove the EEG contains no useful information when a baseline fails."],
              "input_write_operations": 0, "gpu_imports_or_calls": 0, "elapsed_seconds": time.monotonic() - started,
              "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)),
              "native_threadpools": threadpool_info(), "provenance": provenance}
    save_json(output / "comparison_report.json", report)
    with (output / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "selected_C", "feature_count", "test_samples", "accuracy", "balanced_accuracy", "macro_f1"])
        writer.writeheader()
        for mode in MODES:
            value = results[mode]
            writer.writerow({"model": mode, "selected_C": value["selected_C"], "feature_count": value["feature_count"],
                             "test_samples": value["test"]["samples"], **{key: value["test"][key] for key in ("accuracy", "balanced_accuracy", "macro_f1")}})
    save_json(output / "completion.json", {"status": "completed", "exit_code": 0, "report_sha256": sha256(output / "comparison_report.json"),
                                          "manifest_sha256": protocol["manifest_sha256"], "source_sha256": source_sha,
                                          "test_predict_calls_per_model": 1, "elapsed_seconds": report["elapsed_seconds"]})
    print(json.dumps({"stage": "complete", "report_path": str(output / "comparison_report.json"),
                      "models": {mode: results[mode]["test"] for mode in MODES}, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.protocol, args.manifest, args.reference_run, args.output, args.expected_source_sha256)


if __name__ == "__main__":
    main()
