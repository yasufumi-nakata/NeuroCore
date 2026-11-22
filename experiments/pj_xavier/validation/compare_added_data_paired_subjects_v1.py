"""Supplementary paired participant BA comparison from saved predictions only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_evaluation(directory):
    directory = Path(directory)
    completion = json.loads((directory / "completion.json").read_text())
    report = json.loads((directory / "evaluation_report.json").read_text())
    assert completion["status"] == "completed" and completion["exit_code"] == 0
    assert sha(directory / "evaluation_report.json") == completion["report_sha256"]
    assert sha(directory / "predictions.jsonl") == report["predictions_sha256"]
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["condition"] == "fused"]
    assert len(rows) == 1489 and len({row["id"] for row in rows}) == 1489
    assert all(row["checkpoint_sha256"] == report["checkpoint_sha256"] for row in rows)
    return rows, report, {
        "evaluation_name": directory.name,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "report_sha256": completion["report_sha256"],
        "predictions_sha256": report["predictions_sha256"],
    }


def participant_ba(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["subject"]), []).append(row)
    scores = {}
    for group, entries in groups.items():
        labels = sorted(entries[0]["probabilities"])
        assert len(labels) == 2
        assert all(sorted(row["probabilities"]) == labels for row in entries)
        recalls = []
        for label in labels:
            selected = [row for row in entries if row["true_label"] == label]
            assert selected, "Participant lacks one class"
            recalls.append(sum(row["predicted_label"] == label for row in selected) / len(selected))
        scores[group] = sum(recalls) / 2
    assert len(scores) == 17
    return scores


def compare(name, old_dir, new_dir, plan):
    old, old_report, old_pins = load_evaluation(old_dir)
    new, new_report, new_pins = load_evaluation(new_dir)
    assert old_pins["checkpoint_sha256"] == plan["baseline_checkpoints"][name]
    assert old_report["manifest_sha256"] == new_report["manifest_sha256"]
    old_by_id = {row["id"]: row for row in old}
    new_by_id = {row["id"]: row for row in new}
    assert set(old_by_id) == set(new_by_id)
    identity = ("dataset", "subject", "session", "recording_id", "task_family", "true_label")
    assert all(all(old_by_id[key][field] == new_by_id[key][field] for field in identity)
               for key in old_by_id)
    result = {}
    for task, expected_n in (("left_right", 751), ("fists_feet", 738)):
        old_task = [row for row in old if row["task_family"] == task]
        new_task = [row for row in new if row["task_family"] == task]
        assert len(old_task) == len(new_task) == expected_n
        old_scores = participant_ba(old_task)
        new_scores = participant_ba(new_task)
        assert set(old_scores) == set(new_scores)
        keys = sorted(old_scores)
        a = np.asarray([old_scores[key] for key in keys], dtype=np.float64)
        b = np.asarray([new_scores[key] for key in keys], dtype=np.float64)
        delta = b - a
        for report, scores in ((old_report, a), (new_report, b)):
            assert abs(scores.mean() - report["full_test_metrics"]["fused::" + task]
                       ["mean_subject_balanced_accuracy"]) < 2e-10
        rng = np.random.default_rng(plan["bootstrap_seed"])
        indices = rng.integers(0, len(keys), size=(plan["bootstrap_replicates"], len(keys)))
        samples = delta[indices].mean(axis=1)
        ci = np.quantile(samples, [0.025, 0.975], method="linear")
        result[task] = {
            "records": expected_n,
            "paired_participant_groups": len(keys),
            "baseline_mean_subject_BA": float(a.mean()),
            "new_mean_subject_BA": float(b.mean()),
            "new_minus_baseline_BA": float(delta.mean()),
            "new_minus_baseline_percentage_points": float(100 * delta.mean()),
            "paired_participant_percentile_95CI_BA": [float(x) for x in ci],
            "paired_participant_percentile_95CI_percentage_points": [float(100 * x) for x in ci],
            "paired_IDs_sha256": canonical_sha(sorted(row["id"] for row in old_task)),
            "paired_delta_vector_sha256": canonical_sha(delta.tolist()),
        }
    return {"comparison": name, "baseline": old_pins, "new": new_pins, "tasks": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--pair", nargs=3, action="append", required=True,
                        metavar=("NAME", "BASELINE_DIRECTORY", "NEW_DIRECTORY"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    plan = json.loads(Path(args.plan).read_text())
    assert plan["analysis_source_sha256"] == sha(__file__)
    assert plan["bootstrap_seed"] == 20261122 and plan["bootstrap_replicates"] == 1000
    assert [pair[0] for pair in args.pair] == plan["comparisons"]
    result = {
        "schema": "xavier-added-data-paired-subject-comparison-v1",
        "status": "passed",
        "created_epoch": time.time(),
        "source_sha256": sha(__file__),
        "plan_sha256": sha(args.plan),
        "model_forward_calls": 0,
        "model_or_threshold_selection_changed": False,
        "scope": plan["scope"],
        "interpretation_limits": plan["interpretation_limits"],
        "comparisons": [compare(*pair, plan) for pair in args.pair],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "comparisons": len(result["comparisons"]),
                      "output_sha256": sha(output)}))


if __name__ == "__main__":
    main()
