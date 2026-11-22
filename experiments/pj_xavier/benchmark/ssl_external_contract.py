#!/usr/bin/env python3
"""Additional immutable SSL-provenance contract for one exploratory external model.

The original external evaluator, three-model protocol, and predictions are not
modified. Registration/verification is CPU-only and never runs model inference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import time

PREVIOUS_PROTOCOL_SHA = "1acd857958d034125c4a69d222b7564c15531f17cdf89a509d6319beb625eb02"
TRAINER_PROTOCOL_SHA = "9f264097ae6699fa4320e89afc26a5eac999bee0113788325c6001f6d249a8b0"
TRAINER_SHA = "47f24f9429bdc9af9495b121c4bf2cbc94ce62dbeb6b1033eeff6a4a88fe60fd"
ALIGNMENT_SHA = "f246128e909d3581168fbda7a5d062dc31491f27e96b6cae999579795a04fe10"
INITIAL_WEIGHTS_SHA = "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"
SESSION_DEADLINE = 1795321234.65
MODEL_KEY = "cbramod_ssl_smollm_v3"
SSL_RUN = "runs/cbramod_ssl_portable_uniba_v1"
ALIGN_RUN = "runs/cbramod_ssl_smollm2_uniba_v3_symbol_v1"
CORPUS = "prepared/ssl_portable_uniba_v1"
FILES = {
    "trainer": "code/model/cbramod_ssl.py",
    "alignment": "code/model/cbramod_ssl_token_llm.py",
    "trainer_protocol": "code/model/cbramod_ssl_training_protocol.json",
    "manifest": CORPUS + "/manifest.jsonl",
    "artifacts": CORPUS + "/artifacts.json",
    "corpus_summary": CORPUS + "/corpus_summary.json",
    "source_allowlist": CORPUS + "/source_allowlist.json",
    "independent_input_validation": CORPUS + "/independent_model_input_validation.json",
    "ssl_config": SSL_RUN + "/run_config.json",
    "ssl_summary": SSL_RUN + "/run_summary.json",
    "ssl_membership": SSL_RUN + "/training_membership.json",
    "ssl_history": SSL_RUN + "/history.jsonl",
    "ssl_best": SSL_RUN + "/best.pt",
    "encoder_weights": SSL_RUN + "/best_encoder_weights.pth",
    "encoder_seal": SSL_RUN + "/selected_encoder_seal.json",
    "alignment_protocol": "workflows/cbramod_ssl_v1/alignment_protocol.json",
    "alignment_best": ALIGN_RUN + "/best.pt",
    "alignment_summary": ALIGN_RUN + "/run_summary.json",
    "alignment_membership": ALIGN_RUN + "/training_membership.json",
    "alignment_arrays": ALIGN_RUN + "/eeg_artifacts.json",
}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def inside(root, value):
    root = Path(root).resolve()
    p = (root / value).resolve()
    require(p.is_relative_to(root), "Evidence path escapes its artifact root")
    return p


def validate_source_membership(corpus_rows, membership, allowlist, original, external, itu_members):
    require(bool(corpus_rows), "Empty SSL corpus")
    index = {r["id"]: r for r in corpus_rows}
    require(len(index) == len(corpus_rows), "Duplicate SSL IDs")
    require(set(r["split"] for r in corpus_rows) == {"train", "val"}, "Unexpected SSL partition")
    groups, sources, intervals = {}, {}, defaultdict(list)
    for row in corpus_rows:
        require(not ({"label", "target", "action_label", "environment_text"} & row.keys()), "Behavioral labels reached SSL")
        source = allowlist["sources"].get(row["source_sha256"])
        require(source is not None and row["source_sha256"] not in allowlist["excluded_source_sha256"], "Source is not admitted for SSL")
        require(all(source[field] == row[field] for field in ("split", "subject_group", "source_recording_id")), "Original subject/session partition was altered")
        for field, mapping in (("subject_group", groups), ("source_sha256", sources)):
            require(mapping.setdefault(row[field], row["split"]) == row["split"], "SSL subject or raw source crosses partitions")
        start, end = row["window_start_s"], row["window_end_s"]
        require(all(type(v) in (float, int) and math.isfinite(v) for v in (start, end)) and start >= 0 and abs(end-start-30) < 1e-8, "Invalid SSL source interval")
        intervals[row["source_sha256"]].append((start, end))
    for group in intervals.values():
        ordered = sorted(group)
        require(all(a[1] <= b[0]+1e-8 for a, b in zip(ordered, ordered[1:])), "SSL raw intervals overlap")
    for split in ("train", "val"):
        selected = [r for r in corpus_rows if r["split"] == split]
        require(membership[split + "_ids"] == sorted(r["id"] for r in selected), "Recorded SSL membership differs from actual corpus")
        require(membership["source_sha256_by_partition"][split] == sorted({r["source_sha256"] for r in selected}), "SSL source membership was rewritten")
        require(membership["subject_groups_by_partition"][split] == sorted({r["subject_group"] for r in selected}), "SSL participant membership was rewritten")
    require(membership.get("test_source_used") is False and membership.get("behavior_labels_used") == 0, "SSL membership does not exclude test/behavior targets")
    exclusions = set(allowlist["excluded_source_sha256"])
    all_itu_hashes = {value["sha256"] for name, value in itu_members.items() if name.endswith(".eeg")}
    heldout = {r["eeg_source_sha256"] for r in original if r["split"] in ("val", "test")}
    external_hashes = {r["eeg_source_sha256"] for r in external}
    require(bool(all_itu_hashes) and external_hashes <= all_itu_hashes, "ITU physical source membership incomplete")
    require(all_itu_hashes | heldout <= exclusions, "ITU or original UniBa heldout raw source not explicitly excluded")
    used = set(sources)
    require(not used & (all_itu_hashes | heldout | exclusions), "Heldout raw EEG reached SSL")
    return {"status": "passed", "train_windows": len(membership["train_ids"]),
            "SSL_val_windows": len(membership["val_ids"]),
            "training_unique_seconds": 30*len(membership["train_ids"]),
            "all_ITU_raw_sources_explicitly_excluded": len(all_itu_hashes),
            "original_UniBa_val_test_raw_sources_explicitly_excluded": len(heldout),
            "actual_SSL_raw_source_overlap_with_heldout": 0,
            "scope": "This new SSL continuation only",
            "original_upstream_pretraining_membership_reaudited": False,
            "real_person_identity_disjointness_verified": False}


def validate_selected_tensors(state, weights, seal, membership, history, summary, torch):
    require(state.get("schema") == "xavier-cbramod-masked-ssl-v1", "Unexpected SSL state schema")
    require(state["model"].keys() == weights.keys(), "Exported SSL tensor names differ from best checkpoint")
    require(all(torch.equal(tensor, state["model"][name]) for name, tensor in weights.items()), "Exported encoder tensors differ from selected best")
    seen = state["seen_training_ids"]
    require(bool(seen) and len(set(seen)) == len(seen) and set(seen) <= set(membership["train_ids"]), "Selected checkpoint saw nontraining or duplicate IDs, or no training data")
    require(not set(seen) & set(membership["val_ids"]), "Validation samples were used for gradient training")
    candidates = [row for row in history if "masked_mse" in row.get("val", {})]
    require(bool(candidates) and all(math.isfinite(row["val"]["masked_mse"]) for row in candidates), "No finite registered validation loss")
    best = min(candidates, key=lambda row: row["val"]["masked_mse"])
    require(best["global_step"] == state["global_step"] == seal["selected_global_step"], "Encoder was not selected at earliest minimum validation loss")
    require(best["val"]["masked_mse"] == summary["best_validation_masked_mse"], "Selected validation value changed")
    require(seal["unique_training_windows_before_selection"] == len(seen)
            and seal["unique_training_seconds_before_selection"] == 30*len(seen), "Selected unique EEG exposure was overstated")
    require(seal["training_examples_presented_before_selection"] == state["training_examples_presented"], "Repeated exposure count mismatch")
    require(seal.get("downstream_test_source_used") is False and seal.get("itu_sources_used") == 0
            and seal.get("behavior_labels_used") == 0 and seal.get("updated_tensor_count", 0) > 0, "Invalid encoder selection/exclusion seal")
    return {"status": "passed", "selected_global_step": state["global_step"],
            "selected_unique_training_windows": len(seen),
            "selected_unique_training_seconds": 30*len(seen),
            "weights_equal_validation_selected_checkpoint": True}


def verify(root, contract):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    root = Path(root).resolve()
    require(contract["model_order"] == [MODEL_KEY] and set(contract["models"]) == {MODEL_KEY}, "This contract permits only the fourth model")
    require(contract["action_acceptance_threshold"] == .8 and contract["AI_binary_threshold"] == .5
            and contract["history_inference_authorized"] is False, "Unregistered external threshold or repeated history inference")
    extra = contract["ssl_continuation_provenance"]
    require(extra["guard_sha256"] == sha(__file__), "SSL guard source changed")
    actual = {}
    for key, file in extra["files"].items():
        path = inside(root, file["path"])
        actual[key] = path
        require(sha(path) == file["sha256"], "SSL contract evidence changed: " + key)
    require(sha(actual["trainer_protocol"]) == TRAINER_PROTOCOL_SHA and sha(actual["trainer"]) == TRAINER_SHA
            and sha(actual["alignment"]) == ALIGNMENT_SHA, "Unregistered SSL implementation/protocol")
    trainer = read(actual["trainer_protocol"])
    for name in ("manifest", "artifacts", "corpus_summary", "source_allowlist"):
        require(sha(actual[name]) == trainer["corpus"][name+"_sha256"], "Corpus differs from frozen training protocol")
    require(sha(actual["independent_input_validation"]) == trainer["corpus"]["independent_model_input_validation_sha256"], "Independent input validation seal changed")
    ssl_rows = rows(actual["manifest"])
    membership, allowlist = read(actual["ssl_membership"]), read(actual["source_allowlist"])
    require(membership["manifest_sha256"] == sha(actual["manifest"]), "SSL membership has another manifest")
    original = rows(inside(root, contract["training_manifest_path"]))
    prepared = inside(root, contract["external_cohort"]["prepared_path"])
    require(sha(inside(root, contract["training_manifest_path"])) == contract["training_manifest_sha256"], "Original cohort changed")
    for name in ("manifest.jsonl", "source_members_verified.json"):
        require(sha(prepared/name) == contract["external_cohort"]["cohort_files"][name], "External source seal changed")
    membership_check = validate_source_membership(ssl_rows, membership, allowlist, original,
                                                rows(prepared/"manifest.jsonl"), read(prepared/"source_members_verified.json"))
    require(membership_check["all_ITU_raw_sources_explicitly_excluded"] == 26
            and membership_check["original_UniBa_val_test_raw_sources_explicitly_excluded"] == 14, "Full recorded raw exclusions not established")
    artifacts = read(actual["artifacts"])
    artifacts = artifacts.get("artifacts", artifacts)
    require(set(artifacts) == {r["eeg_path"] for r in ssl_rows}, "SSL array membership differs from corpus")
    for name, value in artifacts.items():
        path = inside(actual["manifest"].parent, name)
        require(sha(path) == value["sha256"] and path.stat().st_size == value["bytes"], "Actual SSL array bytes differ from frozen inputs")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        require(arr.dtype == np.float32 and list(arr.shape) == value["shape"], "SSL array dtype/shape changed")
    summary, config, seal = (read(actual[name]) for name in ("ssl_summary", "ssl_config", "encoder_seal"))
    require(summary["status"] in ("completed_epochs", "stopped_time_budget") and summary["global_step"] > 0,
            "SSL failed or performed no gradient steps")
    require(config["command"] == "train" and config["protocol_sha256"] == TRAINER_PROTOCOL_SHA
            and config["model_source_sha256"] == TRAINER_SHA, "Capacity probe or unregistered training run")
    for key in ("manifest", "artifacts", "source_allowlist"):
        require(config[key+"_sha256"] == sha(actual[key]), "Selected SSL config belongs to another corpus")
    for key, evidence in (("ssl_best_checkpoint_sha256", "ssl_best"), ("weights_sha256", "encoder_weights"),
                          ("ssl_training_membership_sha256", "ssl_membership"), ("ssl_run_summary_sha256", "ssl_summary"),
                          ("ssl_source_allowlist_sha256", "source_allowlist"), ("ssl_manifest_sha256", "manifest"),
                          ("ssl_protocol_sha256", "trainer_protocol"), ("ssl_source_sha256", "trainer")):
        require(seal[key] == sha(actual[evidence]), "Selected encoder provenance mismatch: " + key)
    require(seal["initial_weights_sha256"] == INITIAL_WEIGHTS_SHA and seal["weights_sha256"] != INITIAL_WEIGHTS_SHA,
            "Encoder did not continue from the registered initialization")
    require(seal["selection_partition"] == "portable_SSL_validation", "Unregistered SSL selection partition")
    require(summary["best_checkpoint_sha256"] == sha(actual["ssl_best"])
            and summary["best_encoder_weights_sha256"] == sha(actual["encoder_weights"]), "Final SSL summary changed selected artifacts")
    state = torch.load(actual["ssl_best"], map_location="cpu", weights_only=True)
    weights = torch.load(actual["encoder_weights"], map_location="cpu", weights_only=True)
    for key in ("protocol_sha256", "model_source_sha256", "manifest_sha256", "artifacts_sha256",
                "source_allowlist_sha256", "batch_size", "seed", "lr", "precision"):
        require(state["config"][key] == config[key], "SSL checkpoint/run config mismatch")
    tensor_check = validate_selected_tensors(state, weights, seal, membership,
                                             rows(actual["ssl_history"]), summary, torch)
    aligned = torch.load(actual["alignment_best"], map_location="cpu", weights_only=True)
    require(aligned["config"]["ssl_encoder_seal"] == seal, "Aligned checkpoint uses another SSL encoder")
    aligned_summary = read(actual["alignment_summary"])
    require(aligned_summary.get("checkpoint_sha256") == sha(actual["alignment_best"])
            and aligned_summary.get("model_selection_partition") == "validation"
            and aligned_summary.get("test_used_for_tuning") is False
            and aligned_summary.get("status") in ("completed_epochs", "stopped_time_budget"), "Alignment checkpoint selection is not sealed")
    require(aligned["config"]["abstain_threshold"] == .8 and aligned["config"]["mode"] == "fused", "Aligned threshold/input condition changed")
    return {"schema": "xavier-ssl-external-preflight-v1", "status": "passed",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "guard_sha256": sha(__file__), "membership": membership_check, "selected_encoder": tensor_check,
            "sealed_SSL_arrays_verified": len(artifacts), "aligned_checkpoint_sha256": sha(actual["alignment_best"]),
            "registered_before_this_model_external_inference": True,
            "upstream_original_pretraining_overlap_excluded": False,
            "new_model_is_exploratory_after_prior_results": True,
            "new_training_or_inference_in_this_guard": False}


def register(args):
    root = Path(args.root).resolve()
    previous = Path(args.previous_protocol)
    require(sha(previous) == PREVIOUS_PROTOCOL_SHA, "Original external protocol changed")
    require(time.time() < SESSION_DEADLINE, "Original session deadline reached")
    protocol = copy.deepcopy(read(previous))
    spec = copy.deepcopy(protocol["models"]["cbramod_smollm_v3"])
    spec.update(source_sha256=ALIGNMENT_SHA, source_path=FILES["alignment"],
                checkpoint_path=FILES["alignment_best"], checkpoint_sha256=sha(root/FILES["alignment_best"]),
                gate_id="itu_external_cbramod_ssl_smollm_v3_five_variants",
                output_path="external_evaluations/itu_ssl_v1/"+MODEL_KEY,
                receipt_path="external_evaluations/itu_ssl_v1/"+MODEL_KEY+"/variant_evaluation_completion.json")
    require(not (root/spec["output_path"]).exists(), "This external model already has an evaluation attempt")
    protocol.update(registered_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    registered_before_any_ITU_predictions=False,
                    registered_before_this_model_ITU_predictions=True,
                    exploratory_after_prior_ITU_results=True,
                    prior_external_protocol_sha256=PREVIOUS_PROTOCOL_SHA,
                    prior_three_models_rerun=False, model_order=[MODEL_KEY], models={MODEL_KEY: spec},
                    absolute_deadline_epoch=SESSION_DEADLINE,
                    GPU_order="Only after the SSL/alignment/UniBa evaluation workflow fully exits and nvidia-smi is empty; one external model process.",
                    history_inference_authorized=False,
                    history_rule="Reuse previously recorded external predictions only; no refit, new inference or threshold selection.")
    protocol.pop("supersedes_protocol_sha256", None)
    protocol.pop("amendment", None)
    protocol["ssl_continuation_provenance"] = {
        "guard_sha256": sha(__file__),
        "files": {key: {"path": path, "sha256": sha(root/path)} for key, path in FILES.items()},
        "checks": ["All26 ITU raw hashes and original UniBa val/test14 explicitly excluded",
                   "Actual SSL source membership and array bytes match sealed corpus",
                   "Exported encoder tensors equal validation-selected SSL checkpoint",
                   "Aligned checkpoint binds the same selected SSL encoder seal"],
        "upstream_original_pretraining_membership_reaudited": False,
    }
    protocol["independent_final_audit_source_sha256"] = sha(Path(__file__).with_name("audit_ssl_external_completion.py"))
    protocol["recorded_comparators"] = {
        "neural": {key: {"output_path": old["output_path"],
                          "checkpoint_sha256": old["checkpoint_sha256"],
                          "predictions_sha256": sha(root/old["output_path"]/"predictions.jsonl"),
                          "report_sha256": sha(root/old["output_path"]/"external_evaluation_report.json")}
                   for key, old in read(previous)["models"].items()},
        "history_predictions_sha256": "378eedeecb519fe50a9f4b6f0ba4a94667a1b82ed37236b0b65ea57b99ce26c8",
        "history_original_report_sha256": "bdb15c7ba90888347e173be95e70b9e9ca460139d556576ea96cf13a2b1aeab6",
        "previous_19_condition_aggregate_sha256": "058d8a6255b57057520262206ffae1663cf2aba9baa978195249defee4258e86",
        "new_inference_for_comparators": False,
    }
    protocol["interpretation"] = [
        "Fourth model is a separate exploratory continuation after earlier external results; original three results remain unchanged.",
        "The new SSL corpus, alignment parameters and quality criteria were fixed before this model's external inference.",
        "Compare its five conditions and paired fixed-model differences to already recorded comparators; no model selection on ITU.",
        "The added SSL corpus is overwhelmingly portable EEG: only240 seconds of UniBa passed the fixed amplitude QC.",
        "Participant CIs are conditional on a single trained model and this enriched sample; no unseen-site or training-repeat uncertainty.",
        "Anonymous cross-site identity, upstream pretraining overlap, online causality, latent intent and actual task completion remain unverified.",
    ]
    preflight = verify(root, protocol)
    out = Path(args.output)
    require(not out.exists(), "Refusing to overwrite external protocol")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    receipt = out.with_name(out.stem+".registration.json")
    require(not receipt.exists(), "Registration receipt already exists")
    receipt.write_text(json.dumps({"protocol_sha256": sha(out), "preflight": preflight,
                                   "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2)+"\n")
    print(json.dumps({"status": "registered", "protocol": str(out), "protocol_sha256": sha(out),
                      "registration_receipt": str(receipt), "checkpoint_sha256": spec["checkpoint_sha256"]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register")
    for name in ("root", "previous-protocol", "output"):
        reg.add_argument("--"+name, required=True)
    check = sub.add_parser("verify")
    for name in ("root", "protocol", "expected-protocol-sha256", "output"):
        check.add_argument("--"+name, required=True)
    args = p.parse_args()
    if args.command == "register":
        register(args)
    else:
        require(sha(args.protocol) == args.expected_protocol_sha256, "Registered external protocol changed")
        result = verify(args.root, read(args.protocol))
        result["protocol_sha256"] = args.expected_protocol_sha256
        output = Path(args.output)
        require(not output.exists(), "Refusing to overwrite prior preflight")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
