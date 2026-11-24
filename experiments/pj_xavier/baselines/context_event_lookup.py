"""Train-only exact-context majority baseline for small UniBa Qwen comparison.

This command consumes only subject-held-out train and validation JSONL.
It never opens test data, EEG waveforms, or individual Qwen predictions.
Outputs are aggregate statistics only, without subjects or record IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

LABELS = ("mouse_click", "keyboard", "ai_prompt_submit")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_parts(row: dict) -> tuple[str, str, str]:
    messages = row["messages"]
    context = messages[0]["content"]
    target = messages[-1]["content"]
    subject = row["canonical_subject_id"]
    if not isinstance(context, str) or "EEG_FIXED_Q16" in context:
        raise ValueError("Baseline accepts context-only text, not EEG features")
    if not isinstance(subject, str) or not subject or target not in LABELS:
        raise ValueError("Invalid subject or label")
    return context, target, subject


def fit(train: list[dict]) -> tuple[dict[str, str], str, Counter]:
    counts = Counter()
    event_counts: dict[str, Counter] = defaultdict(Counter)
    for row in train:
        context, target, _ = row_parts(row)
        counts[target] += 1
        event_counts[context][target] += 1
    if not counts:
        raise ValueError("Empty training split")

    def mode(counter: Counter) -> str:
        return min(LABELS, key=lambda k: (-counter[k], -counts[k], LABELS.index(k)))

    return {key: mode(c) for key, c in event_counts.items()}, mode(counts), counts


def run(train: list[dict], validation: list[dict]) -> dict:
    tr_subjects = {row_parts(r)[2] for r in train}
    val_subjects = {row_parts(r)[2] for r in validation}
    if tr_subjects & val_subjects:
        raise ValueError("Subject leakage across splits")
    if any(r.get("split") != "train" for r in train):
        raise ValueError("Training input contains a non-train split")
    if any(r.get("split") != "val" for r in validation):
        raise ValueError("Validation input contains a non-val split")
    lookup, fallback, counts = fit(train)
    truth = [row_parts(r)[1] for r in validation]
    predictions = [lookup.get(row_parts(r)[0], fallback) for r in validation]
    if not truth:
        raise ValueError("Empty validation split")
    supports = Counter(truth)
    if set(supports) != set(LABELS):
        raise ValueError("Validation has missing target classes")
    recalls = {k: sum(y == k and p == k for y, p in zip(truth, predictions)) / supports[k] for k in LABELS}
    return {
        "method": "exact predecision context -> majority training label; deterministic train-prior fallback",
        "train_rows": len(train), "validation_rows": len(validation),
        "train_subjects": len(tr_subjects), "validation_subjects": len(val_subjects),
        "subject_disjoint": True, "train_unique_contexts": len(lookup),
        "validation_unseen_contexts": len({row_parts(r)[0] for r in validation} - set(lookup)),
        "training_class_counts": dict(counts), "validation_class_counts": dict(supports),
        "accuracy": sum(y == p for y, p in zip(truth, predictions)) / len(truth),
        "correct": sum(y == p for y, p in zip(truth, predictions)),
        "macro_balanced_accuracy": sum(recalls.values()) / len(LABELS),
        "class_recall": recalls, "prediction_counts": dict(Counter(predictions)),
        "no_eeg": True, "test_read": False,
        "not_final_test": True, "not_latent_intent_evaluation": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; will not overwrite")
    result = run(read_rows(args.train), read_rows(args.validation))
    result["input_sha256"] = {
        "train": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "validation": hashlib.sha256(args.validation.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "correct": result["correct"], "n": result["validation_rows"], "accuracy": result["accuracy"]}))


if __name__ == "__main__":
    main()
