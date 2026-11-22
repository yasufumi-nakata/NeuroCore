#!/usr/bin/env python3
"""Independent final QA for five new SSL conditions plus 19 unchanged recordings."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import audit_external_completion as prior
import ssl_external_contract as ssl

AUDIT_HELPER_SHA = "6e8fa0a20fa823b33a06abc66076abaf8901cfc9a8d3bf72a0d5b1421960c171"
ORIGINAL_AUDIT_SHA = "524a00074112fd696b1ea9e8a2a405d16f4f562b26135d6126495c81831aa94d"
sha, read, rows, require = prior.sha, prior.read, prior.rows, prior.require


def required_condition_keys(protocol):
    require(protocol["model_order"] == [ssl.MODEL_KEY] and set(protocol["models"]) == {ssl.MODEL_KEY}, "Only the fourth model may be newly evaluated")
    require(protocol["conditions"] == prior.VARIANTS, "Registered five conditions changed")
    require(set(protocol["recorded_comparators"]["neural"]) == {"smollm17b_v3", "qwen8b_v3_30e", "cbramod_smollm_v3"}, "Original three comparator identities changed")
    require(len(protocol["history_models"]) == 4 and protocol["history_inference_authorized"] is False, "Only four recorded history comparators are allowed")
    neural = set(protocol["models"]) | set(protocol["recorded_comparators"]["neural"])
    return {key+"::"+variant for key in neural for variant in prior.VARIANTS} | {"history::"+name for name in protocol["history_models"]}


def verify_preflight(preflight, protocol_sha, checkpoint, guard_sha):
    require(preflight.get("status") == "passed" and preflight.get("protocol_sha256") == protocol_sha
            and preflight.get("aligned_checkpoint_sha256") == checkpoint and preflight.get("guard_sha256") == guard_sha,
            "SSL external preflight identity does not match this model/contract")
    require(preflight["membership"]["all_ITU_raw_sources_explicitly_excluded"] == 26
            and preflight["membership"]["original_UniBa_val_test_raw_sources_explicitly_excluded"] == 14
            and preflight["membership"]["actual_SSL_raw_source_overlap_with_heldout"] == 0,
            "Preflight did not establish full SSL raw-source exclusions")


def run(args):
    require(sha(prior.__file__) == AUDIT_HELPER_SHA, "Frozen audit utility changed")
    require(sha(args.protocol) == args.expected_protocol_sha256, "Fourth-model protocol changed")
    p = read(args.protocol)
    require(sha(__file__) == p["independent_final_audit_source_sha256"], "Fourth-model final audit source changed")
    expected = required_condition_keys(p)
    require(len(expected) == 24, "Expected one new five-condition model plus nineteen recorded comparators")
    require(sha(args.original_audit) == ORIGINAL_AUDIT_SHA, "Original nineteen-condition QA changed")
    original_audit = read(args.original_audit)
    require(original_audit["status"] == "passed" and original_audit["conditions"] == 19
            and original_audit["protocol_sha256"] == p["prior_external_protocol_sha256"], "Prior comparison was not independently completed")
    manifest_sha = sha(args.manifest)
    require(manifest_sha == p["external_cohort"]["cohort_files"]["manifest.jsonl"], "External cohort changed")
    external = rows(args.manifest); truth = {r["id"]: r for r in external}
    require(len(truth) == len(external) == 630 and all(r["site"] == "itu" and r["split"] == "external_test" for r in external), "External test identity changed")
    inputs = read(args.inputs)
    require(set(inputs) == set(p["models"]) | set(p["recorded_comparators"]["neural"]) | {"history"}, "Missing or extra recorded model")
    score = read(args.score)
    require(score["status"] == "completed" and score["protocol_sha256"] == args.expected_protocol_sha256
            and score["manifest_sha256"] == manifest_sha and score["inputs_sha256"] == sha(args.inputs), "CPU score is not bound to this exact input set")
    require(set(score["binary_ai_vs_other"]) == expected and all(v["n"] == 630 for v in score["binary_ai_vs_other"].values()), "CPU score condition/row coverage differs")
    require(score["thresholds"] == {"ai_vs_other": .5, "action_acceptance": .8}
            and score["new_model_inference_calls"] == score["new_fit_calls"] == 0, "Unregistered fitting or threshold in summary")
    results = {}
    for key, source in inputs.items():
        require(sha(source["predictions_path"]) == source["predictions_sha256"], "Recorded predictions changed")
        grouped = defaultdict(list)
        for row in rows(source["predictions_path"]):
            grouped[row["variant"]].append(row)
        if key == "history":
            require(source["predictions_sha256"] == p["recorded_comparators"]["history_predictions_sha256"], "History predictions were rerun or changed")
            require(set(grouped) == set(p["history_models"]), "History comparator conditions changed")
            for variant, values in grouped.items():
                artifact = original_audit["condition_identity_and_arithmetic_checks"]["history::"+variant]["artifact_sha256"]
                results[key+"::"+variant] = prior.verify_predictions(values, truth, artifact, variant, neural=False)
            continue
        require(set(grouped) == set(prior.VARIANTS), "Neural condition coverage changed")
        report = read(source["report_path"])
        require(sha(source["report_path"]) == source["report_sha256"], "Source report changed")
        if key != ssl.MODEL_KEY:
            spec = p["recorded_comparators"]["neural"][key]
            require(spec["predictions_sha256"] == source["predictions_sha256"]
                    and spec["report_sha256"] == source["report_sha256"] == original_audit["source_reports"][key], "Original comparator was changed or rerun")
            require(report["external_protocol_sha256"] == p["prior_external_protocol_sha256"], "Original comparator was relabelled under new protocol")
        else:
            spec = p["models"][key]
            output = Path(source["report_path"]).parent
            receipt = read(output/"variant_evaluation_completion.pending.json")
            require(receipt["status"] == "completed" and receipt["exit_code"] == 0
                    and receipt["report_sha256"] == source["report_sha256"]
                    and receipt["checkpoint_sha256"] == spec["checkpoint_sha256"], "New inference is not complete")
            require(report["external_protocol_sha256"] == args.expected_protocol_sha256
                    and report["model_module_sha256"] == spec["source_sha256"], "New model source or contract changed")
            require(report["resource_feasibility"]["status"] == "passed"
                    and report["memory"]["cuda_peak_reserved_bytes"] <= 21*1024**3, "New measured GPU memory gate failed")
            for path in (args.preflight_before, args.preflight_after):
                verify_preflight(read(path), args.expected_protocol_sha256, spec["checkpoint_sha256"], p["ssl_continuation_provenance"]["guard_sha256"])
        require(report["status"] == "completed" and report["evaluation_scope"] == "external_only"
                and report["checkpoint_sha256"] == source["checkpoint_sha256"] == spec["checkpoint_sha256"], "Neural checkpoint identity changed")
        for variant, values in grouped.items():
            result = prior.verify_predictions(values, truth, spec["checkpoint_sha256"], variant)
            metric = report["conditions"][variant]
            require(metric["status"] == "measured" and abs(result["forced_balanced_accuracy"]-metric["balanced_accuracy"]) < 1e-7
                    and abs(result["coverage"]-metric["coverage"]) < 1e-7, "Direct neural arithmetic differs from report")
            if key == ssl.MODEL_KEY:
                path = output/(variant+"_predictions.jsonl")
                require(sha(path) == metric["prediction_sha256"] and rows(path) == values, "Combined new predictions differ from condition source")
                result["negative_control"] = prior.verify_controls(variant, values, truth, output, Path(args.recovered_root),
                                                                   args.expected_protocol_sha256, manifest_sha, spec["checkpoint_sha256"])
            results[key+"::"+variant] = result
    require(set(results) == expected and len(results) == 24, "Incomplete final condition coverage")
    require(score["site_cluster_uncertainty"]["status"] == "blocked" and score["human_identity_disjointness_verified"] is False
            and score["autonomous_agent_validated"] is False and score["online_causality_verified"] is False,
            "An unproved goal/site/causality gate was promoted")
    result = {"schema": "xavier-ssl-external-24-condition-independent-audit-v1", "status": "passed",
              "protocol_sha256": args.expected_protocol_sha256, "manifest_sha256": manifest_sha,
              "score_sha256": sha(args.score), "inputs_sha256": sha(args.inputs),
              "new_model_conditions": 5, "unchanged_recorded_comparator_conditions": 19,
              "total_conditions": 24, "records_per_condition": 630, "recorded_prediction_rows": 15120,
              "original_19_condition_audit_sha256": ORIGINAL_AUDIT_SHA,
              "preflight_before_sha256": sha(args.preflight_before), "preflight_after_sha256": sha(args.preflight_after),
              "condition_identity_and_arithmetic_checks": results, "new_inference_calls_in_audit": 0,
              "original_models_rerun": False, "goal_validated": False,
              "scope": "Exploratory fourth-model comparison and evidence custody, not autonomous-agent acceptance"}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": "passed", "conditions": 24, "records_per_condition": 630, "audit_sha256": sha(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "expected-protocol-sha256", "manifest", "inputs", "score", "original-audit", "recovered-root", "preflight-before", "preflight-after", "output"):
        parser.add_argument("--"+name, required=True)
    run(parser.parse_args())
