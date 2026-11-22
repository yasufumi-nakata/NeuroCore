#!/usr/bin/env python3
"""Descriptive keyboard/mouse diagnostic from already-frozen confusion tables.

No model loading, training, test feature reads, or new predictions occur here.
"""
import argparse
import hashlib
import json
from pathlib import Path


def binary_subset(metric):
    names = metric["class_order"]
    if set(names) != {"ai_prompt_submit", "keyboard", "mouse_click"} or len(names) != 3:
        raise ValueError("unexpected source class order")
    full = metric["confusion_rows_true_columns_predicted"]
    if len(full) != 3 or any(len(row) != 3 or any(not isinstance(x, int) or x < 0 for x in row) for row in full):
        raise ValueError("invalid frozen confusion matrix")
    common = ["keyboard", "mouse_click"]
    matrix = [list(full[names.index(name)]) for name in common]
    samples = sum(map(sum, matrix))
    per_class = {}
    for index, name in enumerate(common):
        col = names.index(name)
        support = sum(matrix[index])
        tp = matrix[index][col]
        fp = sum(row[col] for row in matrix) - tp
        fn = support - tp  # Includes predictions of the third AI class.
        per_class[name] = {"support": support, "precision": tp / (tp + fp) if tp + fp else 0.,
                           "recall": tp / support if support else 0., "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                           "true_positive": tp, "false_positive": fp, "false_negative_including_AI_prediction": fn}
    correct = sum(matrix[i][names.index(name)] for i, name in enumerate(common))
    return {"true_classes": common, "prediction_columns": names, "confusion_2_true_rows_3_prediction_columns": matrix,
            "samples": samples, "excluded_true_AI_examples": sum(full[names.index("ai_prompt_submit")]),
            "AI_predictions_on_common_action_examples_counted_as_errors": sum(row[names.index("ai_prompt_submit")] for row in matrix),
            "accuracy": correct / samples if samples else None,
            "balanced_accuracy": sum(v["recall"] for v in per_class.values()) / 2,
            "macro_f1": sum(v["f1"] for v in per_class.values()) / 2, "per_class": per_class,
            "policy": "Condition on true keyboard/mouse labels; retain every such example. Predicted AI remains an error, never discarded or reassigned. No binary classifier was refit."}


def summarize(report_path, output, expected_sha256):
    body = Path(report_path).read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError("immutable primary report checksum mismatch")
    report = json.loads(body)
    result = {"schema_version": "pj-xavier-frozen-confusion-diagnostic-v1", "primary_report_sha256": expected_sha256,
              "primary_source_sha256": report["source_sha256"], "manifest_sha256": report["manifest_sha256"],
              "posthoc_descriptive_analysis_only": True, "new_model_fits": 0, "new_test_predictions": 0,
              "selection_unchanged": True, "latent_intent_goal_validated": False,
              "models": {name: binary_subset(value["test"]) for name, value in report["models"].items()},
              "train_majority_reference": binary_subset(report["train_majority_reference"]["test"]),
              "ai_class": {"support": report["ai_prompt_submit_support"], "performance_estimable": False,
                           "per_model": {name: value["test"]["per_class"]["ai_prompt_submit"] for name, value in report["models"].items()}},
              "interpretation": "The two common actions have 1955 held-out windows; AI-submission test support is one. Three-class balanced accuracy gives that single AI example one third of the mean recall. These descriptive diagnostics do not establish population performance or latent intent."}
    output = Path(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--expected-sha256", required=True)
    args = p.parse_args()
    print(json.dumps(summarize(args.report, args.output, args.expected_sha256), ensure_ascii=False))
