#!/usr/bin/env python3
"""Append fixed SSL-model recorded predictions to the prior UniBa comparison.

No inference, fitting, threshold selection, or replacement of prior results.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import compare_frozen_v3_models as prior

HELPER_SHA = "fa8db13dccfa2558fcf143efbab90a884aac6da53e51ee6a32e212b9bd9dad86"
OLD_AGGREGATE_SHA = "46543d49fd751b5a6e3b026c0a7190c15bd7e09201484944efaa96ec0b9089ba"
CHECKPOINT = "d2ef4ba0fa7e2d01aa30b73faf977ab0edca0baa9ee12470ce6065cf5bdcfa16"
REPORT = "1d15108458fd59e74d549b0d5857c923d0f9a00082944ab705589bdf6b95da9a"
ENCODER = "0132c9372cbcff28ad36965de043f7b8190233e9faba8ce3ec02c2ae8c93f3e7"
RECOVERY = "17a0ccc353b7cfa2109dc4628b5cc5848fb5d8be785b1865897737896e44bf38"
PREFIX = "cbramod_ssl_smol17b_30epochs"


def run(root):
    root = Path(root).resolve(); private = root/"private/xavier"; public = root/"experiments/pj_xavier/benchmark"
    assert prior.sha(prior.__file__) == HELPER_SHA
    assert prior.sha(public/"history_baselines.py") == prior.SCORER_SHA
    manifest = private/"prepared/ai_coding_uniba_v3_natural_casecontrol_verified/manifest.jsonl"
    assert prior.sha(manifest) == prior.MANIFEST_SHA
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    ids = {row["id"] for row in records if row["split"] == "test"}; assert len(ids) == 203
    old = private/"comparisons/uniba_v3_with_cbramod_recorded_v1"
    previous = prior.read(old/"aggregate.json"); assert prior.sha(old/"aggregate.json") == OLD_AGGREGATE_SHA
    provenance = prior.read(old/"derivation_provenance.json")
    assert prior.sha(old/"derivation_provenance.json") == previous["derivation_provenance_sha256"]
    inputs = []
    locations = {
        "smol17b_3epochs": private/"comparisons/uniba_v3_fixed_models_v2/smol17b_3epochs_derived_predictions.jsonl",
        "qwen8b_30epochs": private/"comparisons/uniba_v3_fixed_models_v2/qwen8b_30epochs_derived_predictions.jsonl",
        "history": private/"comparisons/uniba_v3_fixed_models_v2/history_binary_derived_predictions.jsonl",
        "cbramod_smol17b_30epochs": old/"cbramod_derived_predictions.jsonl",
    }
    for name, path in locations.items():
        assert prior.sha(path) == provenance["sources"][name]["derived_prediction_sha256"]
        inputs.append(path)
    recovery = private/"exports/cbramod_ssl_v1_recovery_receipt.json"
    assert prior.sha(recovery) == RECOVERY
    receipt = prior.read(recovery); assert receipt["status"] == "completed" and receipt["all_member_hashes_verified"] is True
    run_dir = private/"runs/cbramod_ssl_smollm2_uniba_v3_symbol_v1"
    evaluation = private/"evaluations/cbramod_ssl_smollm2_uniba_v3_fiveconditions_v1"
    assert prior.sha(run_dir/"best.pt") == CHECKPOINT
    assert prior.sha(private/"runs/cbramod_ssl_portable_uniba_v1/best_encoder_weights.pth") == ENCODER
    assert prior.sha(evaluation/"variant_evaluation_report.json") == REPORT
    completed = prior.read(run_dir/"variant_evaluation_completion.pending.json")
    assert completed["status"] == "completed" and completed["exit_code"] == 0
    assert completed["checkpoint_sha256"] == CHECKPOINT and completed["report_sha256"] == REPORT
    new_rows, evidence = prior.derived_rows(PREFIX, evaluation, CHECKPOINT, ids)
    out = private/"comparisons/uniba_v3_with_ssl_recorded_v1"; out.mkdir(parents=True, exist_ok=False)
    path = out/"ssl_derived_predictions.jsonl"
    with path.open("x") as stream:
        for row in new_rows: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    inputs.append(path); evidence["derived_prediction_sha256"] = prior.sha(path)
    sources = {**provenance["sources"], PREFIX: evidence}
    prior.save(out/"derivation_provenance.json", {"schema": "xavier-v3-fourth-SSL-recorded-comparison-v1",
                "manifest_sha256": prior.MANIFEST_SHA, "previous_19_condition_aggregate_sha256": OLD_AGGREGATE_SHA,
                "scorer_sha256": prior.SCORER_SHA, "recovery_receipt_sha256": RECOVERY, "sources": sources,
                "probabilities_truth_and_ids_unchanged": True, "new_model_inference_calls": 0})
    score_path = out/"paired_ai_comparison.json"
    command = [sys.executable, str(public/"history_baselines.py"), "score-predictions", "--manifest", str(manifest),
               "--expected-manifest-sha256", prior.MANIFEST_SHA, "--output", str(score_path)]
    for source in inputs: command += ["--predictions", str(source)]
    with (out/"scoring.log").open("x") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"))
    score = prior.read(score_path)
    names = ("smol17b_3epochs", "qwen8b_30epochs", "cbramod_smol17b_30epochs", PREFIX)
    expected = set(prior.HISTORY_VARIANTS) | {prefix+"::"+variant for prefix in names for variant in prior.VARIANTS}
    assert set(score["models"]) == expected and len(expected) == 24 and all(m["n"] == 203 for m in score["models"].values())
    result = {"schema": "xavier-exploratory-v3-fourth-SSL-recorded-comparison-v1", "status": "completed",
              "conditions": 24, "records_per_condition": 203, "comparison": score, "sources": sources,
              "derivation_provenance_sha256": prior.sha(out/"derivation_provenance.json"),
              "recorded_comparison_sha256": prior.sha(score_path), "previous_19_condition_aggregate_sha256": OLD_AGGREGATE_SHA,
              "new_model_inference_calls": 0, "new_parameter_fits": 0, "goal_validated": False,
              "analysis_scope": "Exploratory descriptive comparison of already recorded test outputs, after earlier test results. Fixed threshold0.5, 1000 paired observation-participant bootstrap draws, eight UniBa groups. No repeated-training or multiplicity correction.",
              "matched_comparison": "New SSL-adapted CBraMod versus original CBraMod keeps alignment data, batch32, 30epochs/900s, seed, loss, token mapping, frozen Smol backbone and projection structure unchanged; only encoder weights change. One training run does not establish a general causal benefit.",
              "natural_population_precision_estimated": False}
    prior.save(out/"aggregate.json", result)
    target = public/"v3_ssl_recorded_comparison_aggregate.json"; assert not target.exists(); target.write_bytes((out/"aggregate.json").read_bytes())
    prior.save(out/"completion.json", {"status": "completed", "aggregate_sha256": prior.sha(target),
               "comparison_sha256": prior.sha(score_path), "conditions": 24, "records_per_condition": 203, "new_model_inference_calls": 0})
    print(json.dumps({"status": "completed", "output": str(out), "aggregate_sha256": prior.sha(target)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--worktree", required=True)
    run(parser.parse_args().worktree)
