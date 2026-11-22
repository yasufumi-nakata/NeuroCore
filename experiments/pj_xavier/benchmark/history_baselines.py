#!/usr/bin/env python3
"""Frozen v3 CPU history baselines and subject-cluster observed-action scoring.

Only the declared past-event feature mappings enter fitting. The optional score
subcommand consumes already recorded predictions; it never runs an LLM again.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

KINDS = ("key_down", "key_up", "mouse_click", "mouse_move")
HORIZONS = (1, 5, 15, 60)
FAMILIES = ("history_pre_eeg", "history_pre_decision_guarded")
MODES = tuple(f + suffix for f in FAMILIES for suffix in ("", "_plus_context"))
ENDPOINTS = ("ai_vs_other", "three_class")
AI = "ai_prompt_submit"
SCHEMA = "xavier-v3-history-baselines-v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def common_module(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError("frozen cpu_baselines module SHA mismatch")
    spec = importlib.util.spec_from_file_location("frozen_cpu_baselines", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def history_vector(row, family):
    if family not in FAMILIES:
        raise ValueError("unknown history family")
    pack = row[family]
    guard = row["clock_uncertainty_guard_s"]
    base = row["window_start_s"] if family == "history_pre_eeg" else row["decision_time_s"]
    cutoff, maximum = pack["cutoff_s"], pack["max_source_onset_used_s"]
    if (not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (guard, base, cutoff))
            or guard < 0 or abs(cutoff - (base - guard)) > 1e-8 or cutoff >= row["event_time_s"]):
        raise ValueError("history cutoff does not match the preregistered guarded boundary")
    if maximum is not None and (not math.isfinite(maximum) or maximum >= cutoff):
        raise ValueError("future or boundary history onset")
    if (pack.get("target_or_future_events_included") is not False
            or pack.get("source_fields") != ["onset", "action only for key-down/up classification"]):
        raise ValueError("unverified history field scope")
    features = pack["features"]
    allowed = {f"{kind}_{suffix}" for kind in KINDS for suffix in
               (["source_available", "seconds_since_last_capped_60"] + [f"count_{h}s" for h in HORIZONS])}
    if set(features) != allowed:
        raise ValueError("extra or missing history features; IDs, target text and timing fields are forbidden")
    values, names = [], []
    for kind in KINDS:
        available = features[kind + "_source_available"]
        counts = [features[f"{kind}_count_{h}s"] for h in HORIZONS]
        lag = features[kind + "_seconds_since_last_capped_60"]
        if not isinstance(available, bool) or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in counts):
            raise ValueError("invalid history availability or event count")
        if counts != sorted(counts):
            raise ValueError("event counts decrease with increasing time horizon")
        if lag is not None and (isinstance(lag, bool) or not isinstance(lag, (int, float)) or not math.isfinite(lag) or not 0 < lag <= 60):
            raise ValueError("nonpast or nonfinite lag")
        if (not available and (any(counts) or lag is not None)) or (lag is None and any(counts)):
            raise ValueError("missing-source or missing-history indicators contradict counts")
        values.extend([float(available), *map(float, counts), 60. if lag is None else float(lag), float(lag is None)])
        names.extend([kind + "_source_available", *[f"{kind}_count_{h}s" for h in HORIZONS], kind + "_lag_capped_60", kind + "_lag_missing"])
    evidence = row.get("history_evidence", [])
    if not evidence or any(e.get("acquisition") not in ("keypress", "mouseclick", "mousemove") or not isinstance(e.get("sha256"), str)
                           or len(e["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in e["sha256"])
                           or not e.get("uri") for e in evidence):
        raise ValueError("history evidence is missing, includes GPT, or has no source hash")
    return values, names


def binary_metrics(truth, scores, threshold=.5):
    y = np.asarray(truth, dtype=int)
    p = np.asarray(scores, dtype=float)
    if y.shape != p.shape or not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid binary truth or probabilities")
    pred = p >= threshold
    tp, fn = int(np.sum((y == 1) & pred)), int(np.sum((y == 1) & ~pred))
    tn, fp = int(np.sum((y == 0) & ~pred)), int(np.sum((y == 0) & pred))
    recall, specificity = tp / (tp + fn) if tp + fn else None, tn / (tn + fp) if tn + fp else None
    return {"n": len(y), "positive_support": tp + fn, "negative_support": tn + fp,
            "sample_prevalence": float(np.mean(y)), "threshold": threshold,
            "recall": recall, "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "precision_enriched_sample_only": tp / (tp + fp) if tp + fp else None,
            "auroc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            "auprc_average_precision_enriched_sample_only": float(average_precision_score(y, p)) if len(set(y)) == 2 else None,
            "brier_enriched_sample_only": float(np.mean((p - y)**2)),
            "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp}}


CI_METRICS = ("recall", "specificity", "balanced_accuracy", "auroc", "auprc_average_precision_enriched_sample_only")


def cluster_indices(subjects, repetitions, seed):
    groups = sorted(set(subjects))
    rng = np.random.default_rng(seed)
    members = {group: np.flatnonzero(np.asarray(subjects) == group) for group in groups}
    for _ in range(repetitions):
        yield np.concatenate([members[group] for group in rng.choice(groups, size=len(groups), replace=True)])


def clustered_comparison(truth, subject_ids, probability_map, repetitions=1000, seed=20261122):
    """Paired resampling of whole participants, conditional on the frozen fits."""
    if repetitions < 20 or len(set(subject_ids)) < 2:
        return {"status": "blocked", "reason": "at least two subjects and 20 bootstrap repetitions required"}
    truth = np.asarray(truth)
    arrays = {k: np.asarray(v) for k, v in probability_map.items()}
    names = sorted(arrays)
    values = {k: {metric: [] for metric in CI_METRICS} for k in names}
    pairs = {(a, b): {metric: [] for metric in CI_METRICS} for i, a in enumerate(names) for b in names[i + 1:]}
    for indices in cluster_indices(subject_ids, repetitions, seed):
        measured = {k: binary_metrics(truth[indices], p[indices]) for k, p in arrays.items()}
        for name, result in measured.items():
            for metric in CI_METRICS:
                if result[metric] is not None:
                    values[name][metric].append(result[metric])
        for (a, b), metrics in pairs.items():
            for metric in CI_METRICS:
                if measured[a][metric] is not None and measured[b][metric] is not None:
                    metrics[metric].append(measured[a][metric] - measured[b][metric])
    def intervals(items):
        return {metric: {"lower_95": float(np.quantile(v, .025)) if v else None,
                         "upper_95": float(np.quantile(v, .975)) if v else None,
                         "valid_resamples": len(v), "invalid_resamples": repetitions - len(v)} for metric, v in items.items()}
    return {"status": "measured", "method": "paired participant-cluster percentile bootstrap; equal participant resampling, all sampled participant windows retained",
            "n_subjects": len(set(subject_ids)), "repetitions": repetitions, "seed": seed,
            "models": {k: intervals(v) for k, v in values.items()},
            "paired_differences_first_minus_second": [{"first": a, "second": b, "intervals": intervals(v)} for (a, b), v in pairs.items()],
            "limits": "Conditional uncertainty of these fitted models on an enriched held-out sample with only 8 subjects; not natural-population precision, training-repeat uncertainty, multiplicity-adjusted evidence, or a continuous-use false-action rate."}


def features_for(rows, family, common, vectorizer=None):
    vectors = [history_vector(row, family) for row in rows]
    x = np.asarray([v[0] for v in vectors], dtype=float)
    names = vectors[0][1]
    if vectorizer is not None:
        x = np.column_stack([x, vectorizer.transform([common.parse_context(row) for row in rows])])
        names += ["context:" + name for name in vectorizer.get_feature_names_out()]
    return x, names


def run(args):
    started = time.monotonic()
    protocol = json.loads(Path(args.protocol).read_text())
    protocol_sha, source_sha = sha(args.protocol), sha(__file__)
    if source_sha != args.expected_source_sha256 or protocol_sha != args.expected_protocol_sha256:
        raise ValueError("history source/protocol hash mismatch")
    if protocol.get("predeclared_before_test_prediction") is not True or protocol["models"] != list(MODES):
        raise ValueError("invalid frozen model protocol")
    common = common_module(args.common_module, protocol["common_module_sha256"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "protocol.json", protocol)
    rows, verification = common.verify_inputs(args.manifest, args.reference_run, protocol)
    for row in rows:
        if row.get("sampling_design") != "case_control_enriched" or row.get("cohort_protocol_sha256") != protocol["cohort_protocol_sha256"]:
            raise ValueError("wrong sampling design or preregistered cohort")
        for family in FAMILIES:
            history_vector(row, family)
    partitions = {part: [row for row in rows if row["split"] == part] for part in ("train", "val", "test")}
    vectorizer, _, _ = common.fit_context(partitions["train"], partitions["val"])
    train_classes = sorted({row["label"] for row in partitions["train"]})
    fitted, selected = {}, {}
    for endpoint in ENDPOINTS:
        classes = ["other_observed_action", AI] if endpoint == "ai_vs_other" else train_classes
        transform = lambda r: AI if r["label"] == AI else "other_observed_action"
        y = {part: np.asarray([transform(r) if endpoint == "ai_vs_other" else r["label"] for r in partitions[part]]) for part in ("train", "val")}
        for mode in MODES:
            family = mode.replace("_plus_context", "")
            encoder = vectorizer if mode.endswith("_plus_context") else None
            xtrain, names = features_for(partitions["train"], family, common, encoder)
            xval, _ = features_for(partitions["val"], family, common, encoder)
            scaler, model, choice = common.select_model(xtrain, y["train"], xval, y["val"], classes, protocol["C_grid"])
            key = endpoint + ":" + mode
            artifact = output / (key.replace(":", "__") + ".npz")
            common.save_portable_model(artifact, scaler, model, names)
            selected[key] = {**choice, "artifact_sha256": sha(artifact), "feature_names": names}
            fitted[key] = scaler, model, family, encoder
    save(output / "selection.json", {"models": selected, "test_used_for_selection": False, "selected_at_epoch": time.time()})
    if sha(__file__) != source_sha or sha(args.protocol) != protocol_sha or sha(args.common_module) != protocol["common_module_sha256"]:
        raise ValueError("code/protocol changed during fitting")
    _, checked_again = common.verify_inputs(args.manifest, args.reference_run, protocol)
    if checked_again != verification:
        raise ValueError("manifest/EEG/membership changed during fitting")
    common.claim_test_once(output, sha(output / "selection.json"))
    test = partitions["test"]
    binary_truth = np.asarray([row["label"] == AI for row in test], dtype=int)
    subjects = [row.get("canonical_dataset_id", row["dataset"]) + ":" + row.get("canonical_subject_id", row["subject"]) for row in test]
    results, probability_map, prediction_rows = {}, {}, []
    for key, (scaler, model, family, encoder) in fitted.items():
        x, _ = features_for(test, family, common, encoder)
        probabilities = model.predict_proba(scaler.transform(x))  # exactly one held-out inference call
        inferred = model.classes_[probabilities.argmax(axis=1)]
        ai_scores = probabilities[:, list(model.classes_).index(AI)]
        probability_map[key] = ai_scores
        metrics = {"ai_vs_other": binary_metrics(binary_truth, ai_scores), "test_predict_proba_calls": 1,
                   "selected_C": selected[key]["selected_C"], "model_artifact_sha256": selected[key]["artifact_sha256"]}
        if key.startswith("three_class:"):
            metrics["three_class"] = common.metrics(np.asarray([r["label"] for r in test]), inferred, train_classes)
        results[key] = metrics
        for row, values, prediction in zip(test, probabilities, inferred):
            prediction_rows.append({"id": row["id"], "true_label": row["label"], "predicted_label": str(prediction), "variant": key,
                                    "probabilities": dict(zip(map(str, model.classes_), map(float, values))),
                                    "model_artifact_sha256": selected[key]["artifact_sha256"], "action_executed": False})
    prediction_path = output / "test_predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in prediction_rows))
    uncertainty = clustered_comparison(binary_truth, subjects, probability_map, protocol["bootstrap_repetitions"], protocol["seed"])
    _, final = common.verify_inputs(args.manifest, args.reference_run, protocol)
    if final != verification or sha(__file__) != source_sha or sha(args.protocol) != protocol_sha or sha(args.common_module) != protocol["common_module_sha256"]:
        raise ValueError("frozen inputs changed during final test scoring")
    report = {"schema": SCHEMA, "status": "completed", "manifest_sha256": protocol["manifest_sha256"],
              "source_sha256": source_sha, "protocol_sha256": protocol_sha, "common_module_sha256": protocol["common_module_sha256"],
              "cohort": verification, "models": results, "subject_cluster_uncertainty": uncertainty,
              "prediction_sha256": sha(prediction_path), "selection_sha256": sha(output / "selection.json"),
              "model_selection_partition": "validation", "test_used_for_tuning": False, "test_calls_per_model": 1,
              "history_feature_scope": "Only frozen past-event counts, capped recency, source/missing indicators; source manifest SHA authenticated, source TSV histories not independently reconstructed by this baseline.",
              "imputation": "Deterministic capped lag=60 plus missing indicator; no test-derived statistics; train-only scaling and context vocabulary.",
              "cohort_scope": "Enriched case-control, observed agent request transmission versus other observed actions. No idle labels. No latent-intent gold.",
              "natural_population_precision_estimated": False, "false_actions_per_minute": None, "task_completion": None,
              "online_causality_verified": False, "autonomous_agent_validated": False,
              "cpu_only": True, "native_thread_limit": 2, "elapsed_seconds": time.monotonic() - started}
    save(output / "history_report.json", report)
    save(output / "completion.json", {"status": "completed", "exit_code": 0, "report_sha256": sha(output / "history_report.json"),
                                       "test_prediction_sha256": sha(prediction_path), "manifest_sha256": protocol["manifest_sha256"]})
    print(json.dumps({"status": "completed", "output": str(output), "models": results, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


def score_predictions(args):
    """Join already recorded model outputs to the same frozen test cohort."""
    if sha(args.manifest) != args.expected_manifest_sha256:
        raise ValueError("score manifest SHA mismatch")
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    test = [row for row in rows if row["split"] == "test"]
    ids = [row["id"] for row in test]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate test IDs")
    truth = [row["label"] == AI for row in test]
    subjects = [row.get("canonical_dataset_id", row["dataset"]) + ":" + row.get("canonical_subject_id", row["subject"]) for row in test]
    mapping, source_hashes, checksums = {}, {}, {}
    for filename in args.predictions:
        source_hashes[str(Path(filename).name)] = sha(filename)
        for line in Path(filename).read_text().splitlines():
            p = json.loads(line)
            name = p.get("variant", p.get("condition"))
            if not name:
                raise ValueError("prediction condition missing")
            condition = mapping.setdefault(name, {})
            if p["id"] in condition or p["id"] not in ids:
                raise ValueError("duplicate or non-test prediction ID")
            if p.get("true_label") not in (None, next(row["label"] for row in test if row["id"] == p["id"])):
                raise ValueError("reported true label differs from frozen cohort")
            probabilities = p["probabilities"]
            if not isinstance(probabilities, dict) or AI not in probabilities or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1 for x in probabilities.values()) or abs(sum(probabilities.values()) - 1) > 1e-5:
                raise ValueError("invalid recorded categorical probabilities")
            checkpoint = p.get("checkpoint_sha256", p.get("model_artifact_sha256"))
            if not isinstance(checkpoint, str) or len(checkpoint) != 64:
                raise ValueError("recorded model artifact SHA required")
            if name in checksums and checksums[name] != checkpoint:
                raise ValueError("condition mixes checkpoints")
            checksums[name] = checkpoint
            condition[p["id"]] = probabilities[AI]
    if any(set(condition) != set(ids) for condition in mapping.values()):
        raise ValueError("conditions do not cover the exact frozen test IDs")
    ordered = {name: [condition[rid] for rid in ids] for name, condition in mapping.items()}
    report = {"schema": SCHEMA + "-recorded-prediction-score", "status": "completed", "manifest_sha256": args.expected_manifest_sha256,
              "prediction_source_sha256": source_hashes, "checkpoint_sha256_per_condition": checksums,
              "models": {name: binary_metrics(truth, values) for name, values in ordered.items()},
              "subject_cluster_uncertainty": clustered_comparison(truth, subjects, ordered, args.bootstrap_repetitions),
              "new_inference_calls": 0, "natural_population_precision_estimated": False, "false_actions_per_minute": None,
              "task_completion": None, "autonomous_agent_validated": False,
              "scope": "AI probability threshold 0.5 specified before scoring; no test threshold tuning. Sample-enriched observed transmissions only; predictive uncertainty conditional on the fixed selected models."}
    output = Path(args.output)
    if output.exists():
        raise ValueError("recorded score output already exists")
    save(output, report)
    print(json.dumps({"status": "completed", "models": report["models"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("run")
    for name in ("protocol", "manifest", "reference-run", "common-module", "output", "expected-source-sha256", "expected-protocol-sha256"):
        fit.add_argument("--" + name, required=True)
    score = sub.add_parser("score-predictions")
    for name in ("manifest", "expected-manifest-sha256", "output"):
        score.add_argument("--" + name, required=True)
    score.add_argument("--predictions", action="append", required=True)
    score.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args) if args.command == "run" else score_predictions(args)


if __name__ == "__main__":
    main()
