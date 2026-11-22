"""Independently recompute saved binary EEG predictions without model inference.

Uses sklearn's confusion-independent metrics and participant grouping, then
compares point estimates to the frozen evaluator report. Does not refit models,
rerun EEG, recompute bootstrap intervals, or validate biological intent.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score

TASKS = {"left_right": ("left_fist", "right_fist"),
         "fists_feet": ("both_fists", "both_feet")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(directory):
    root = Path(directory)
    completion = json.loads((root / "completion.json").read_text())
    assert completion["status"] == "completed" and completion["exit_code"] == 0
    report_path = root / "evaluation_report.json"
    report = json.loads(report_path.read_text())
    assert sha(report_path) == completion["report_sha256"]
    prediction_path = root / "predictions.jsonl"
    before = sha(prediction_path)
    assert before == report["predictions_sha256"]
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    assert len({(r["condition"], r["id"]) for r in rows}) == len(rows)
    by_condition = defaultdict(list)
    for r in rows:
        assert r["checkpoint_sha256"] == report["checkpoint_sha256"]
        by_condition[r["condition"] + "::" + r["task_family"]].append(r)
    assert set(by_condition) == set(report["full_test_metrics"])
    checks = {}
    for key, group in sorted(by_condition.items()):
        task = key.split("::", 1)[1]
        labels = TASKS[task]
        probability = np.asarray([[r["probabilities"][label] for label in labels] for r in group], dtype=float)
        assert np.isfinite(probability).all()
        assert (probability >= 0).all() and (probability <= 1).all()
        assert np.allclose(probability.sum(1), 1, atol=1e-6, rtol=0)
        truth = np.asarray([labels.index(r["true_label"]) for r in group])
        predicted = np.asarray([labels.index(r["predicted_label"]) for r in group])
        assert np.array_equal(predicted, probability.argmax(1))
        accepted = probability.max(1) >= 0.8
        subjects = defaultdict(list)
        for i, row in enumerate(group):
            subjects[(row["dataset"], row["subject"])].append(i)
        subject_scores = []
        for index in subjects.values():
            if len(set(truth[index])) == 2:
                subject_scores.append(balanced_accuracy_score(truth[index], predicted[index]))
        actual = {
            "n": len(group),
            "balanced_accuracy": balanced_accuracy_score(truth, predicted),
            "mean_subject_balanced_accuracy": float(np.mean(subject_scores)),
            "macro_f1": f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0),
            "auroc": roc_auc_score(truth, probability[:, 1]),
            "brier": brier_score_loss(truth, probability[:, 1]),
            "negative_log_likelihood": math.fsum(-math.log(max(1e-12, min(1, p[y]))) for p, y in zip(probability, truth)) / len(group),
            "coverage": float(accepted.mean()),
            "accepted": int(accepted.sum()),
            "selective_accuracy": float((truth[accepted] == predicted[accepted]).mean()) if accepted.any() else None,
            "evaluated_subject_groups": len(subject_scores),
        }
        published = report["full_test_metrics"][key]
        differences = {}
        for name, value in actual.items():
            expected = published[name]
            if value is None:
                assert expected is None, (key, name)
            else:
                delta = abs(float(value) - float(expected))
                assert delta <= 2e-10, (key, name, delta)
                differences[name] = delta
        checks[key] = {"recomputed": actual, "max_absolute_difference": max(differences.values())}
    assert sha(prediction_path) == before
    return {
        "evaluation_name": root.name,
        "status": "passed",
        "checkpoint_sha256": report["checkpoint_sha256"],
        "report_sha256": sha(report_path),
        "predictions_sha256": before,
        "records_across_conditions": len(rows),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), "Existing audit output must be preserved"
    result = {
        "schema": "xavier-independent-saved-binary-prediction-check-v1",
        "source_sha256": sha(__file__),
        "created_epoch": time.time(),
        "scope": "Saved point metrics only; no EEG/model forward, refit, bootstrap recomputation, or free-intent validation",
        "model_forward_calls": 0,
        "evaluations": [check(path) for path in args.evaluation],
    }
    result["status"] = "passed"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "evaluations": len(result["evaluations"]),
                      "output_sha256": sha(output)}))


if __name__ == "__main__":
    main()
