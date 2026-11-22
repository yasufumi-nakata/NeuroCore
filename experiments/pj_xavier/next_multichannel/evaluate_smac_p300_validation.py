#!/usr/bin/env python3
"""Evaluate two frozen CBraMod SSL checkpoints on *P300 validation only*.

This is an independent, immutable, offline waveform-reconstruction diagnostic.
It never selects a new checkpoint, opens the frozen P300 test subjects, or
treats stimulus/oracle/decoder/future text as model inputs. The one held-out
P300 validation subject does not establish population-level generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE.parent / "model"))
import cbramod_ssl_smac as trainer
import prepare_smac_p300 as p300


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(p)


def frozen_preflight(cfg):
    node = platform.node().split(".")[0].lower()
    require(node.startswith("smac") and node != "smac19", "Only approved S Mac nodes")
    require(cfg["schema"] == "xavier-p300-heldout-checkpoint-validation-v1",
            "Unrecognized frozen evaluation protocol")
    require(cfg["holdout_group"] == "openneuro:ds005028:sub-11",
            "P300 validation subject must not change")
    require(cfg["excluded_test_groups"] ==
            ["openneuro:ds005028:sub-01", "openneuro:ds005028:sub-10"],
            "P300 test subjects must remain sealed")
    require(cfg["fixed_seed"] == 20261122 and cfg["fixed_mask"] is True,
            "Unregistered validation mask")
    require(cfg["no_checkpoint_selection"] is True and cfg["no_test_read"] is True,
            "Diagnostic must not select checkpoints or read P300 test")
    require(cfg["expected_preprocessing_source_sha256"] == sha(p300.__file__),
            "Frozen P300 preprocessing implementation changed")
    require(cfg["expected_trainer_sha256"] == sha(trainer.__file__),
            "Frozen SSL trainer changed")
    for key, sha_key in (("source_ledger", "source_ledger_sha256"),
                         ("metadata_admission", "metadata_admission_sha256"),
                         ("audit_protocol", "audit_protocol_sha256"),
                         ("training_manifest", "training_manifest_sha256"),
                         ("training_completion", "training_completion_sha256")):
        require(sha(cfg[key]) == cfg[sha_key], "Sealed file mismatch: " + key)
    require(cfg["source_raw_root"].startswith(str(Path.home() / "pj-xavier" / node / "raw") + "/"),
            "Validation raw must remain on assigned smac node")
    require(cfg["evaluation_output"].startswith(str(Path.home() / "pj-xavier" / node / "evaluations") + "/"),
            "Evaluation output must remain on assigned S Mac")
    require(Path(cfg["training_checkpoint_after"]).is_file(), "Frozen after checkpoint absent")
    require(Path(cfg["training_checkpoint_before"]).is_file(), "Frozen before checkpoint absent")
    for key in ("training_checkpoint_before", "training_checkpoint_after"):
        require(sha(cfg[key]) == cfg[key + "_sha256"], "Checkpoint checksum changed")
    require(Path(cfg["evaluation_output"]).exists() is False,
            "Immutable output path already exists")


def prepare_validation(cfg, out):
    records = p300.rows(cfg["source_ledger"])
    metadata = p300.rows(cfg["metadata_admission"])
    plan = p300.plan_records(records, metadata, set(), set(), set())
    selected = [e for e in plan if e["split"] == "val"]
    require(len(selected) == 9 and all(e["status"] == "excluded_frozen_holdout"
                                       and e["record"]["subject_group"] == cfg["holdout_group"]
                                       for e in selected), "Unexpected validation split or leakage")
    require(sum(e["split"] == "test" for e in plan) == 19,
            "Previously sealed P300 test count changed")
    trained = [json.loads(line) for line in Path(cfg["training_manifest"]).read_text().splitlines()
               if line.strip()]
    train_sources = {r["source_sha256"] for r in trained if r["split"] == "train"}
    train_groups = {r["subject_group"] for r in trained if r["split"] == "train"}
    require(cfg["holdout_group"] not in train_groups, "P300 validation group contaminated")
    require(not train_sources.intersection(e["record"]["source_file_sha256"] for e in selected),
            "P300 validation raw source was in fitting")
    source_root = Path(cfg["source_raw_root"])
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    (out / "arrays").mkdir()
    rows, certs = [], []
    for item in selected:
        rec = item["record"]
        relative = Path(rec["source_path"])
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.suffix == ".dat", "Unsafe original P300 DAT name")
        f = (source_root / relative).resolve()
        require(source_root.resolve() in f.parents and f.is_file()
                and not f.is_symlink(), "Validation original DAT missing")
        x, proof = p300.read_original_eeg_volts(f, rec)
        try:
            array, intervals, rejected = p300.windows32(x)
        except ValueError as ex:
            certs.append({"source_sha256": rec["source_file_sha256"],
                          "status": "held_native_QC", "reason": str(ex),
                          "raw_payload_verified": True})
            continue
        certs.append({"source_sha256": rec["source_file_sha256"],
                      "status": "validated_offline_signal_only",
                      "raw_payload_verified": True,
                      "windows": len(intervals), "rejected": rejected,
                      "decoded_calibrated_V_sha256": proof["decoded_calibrated_V_sha256"]})
        if not intervals:
            continue
        path = out / "arrays" / (rec["source_file_sha256"] + ".npy")
        np.save(path, array, allow_pickle=False)
        for i, interval in enumerate(intervals):
            rows.append({"id": rec["source_file_sha256"] + ":" + str(i),
                         "split": "val", "subject_group": cfg["holdout_group"],
                         "source_sha256": rec["source_file_sha256"],
                         "channels": p300.CHANNELS, "patches": p300.PATCHES,
                         "montage_key": "p300-original-dat32-heldout-same-processing",
                         "reference": p300.REFERENCE,
                         "native_sample_rate_hz": p300.NATIVE_HZ,
                         "eeg_index": i, "_path": str(path),
                         **interval})
    require(len(certs) == 9, "Not all nine validation originals audited")
    require(rows, "No fixed-QC-valid P300 validation windows")
    require(all(x["split"] == "val" for x in rows), "Test or train entered evaluation")
    atomic_json(out / "certificates.json", certs)
    atomic_json(out / "validation_window_provenance.json",
                {"schema": "xavier-p300-validation-window-provenance-v1",
                 "windows": len(rows), "subject_groups": 1,
                 "records_with_accepted_windows": len({r["source_sha256"] for r in rows}),
                 "input_signal_only": True, "stimulus_oracle_decoded": False,
                 "test_recordings_opened": 0,
                 "input_array_hashes": {str(p.relative_to(out)): sha(p)
                                        for p in sorted((out / "arrays").glob("*.npy"))}})
    return rows


def evaluate(cfg):
    frozen_preflight(cfg)
    out = Path(cfg["evaluation_output"])
    start = time.time()
    # Protocol exists before touching the validation DAT signal.
    out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    rows = prepare_validation(cfg, out)
    device = trainer.legacy.frozen.select_device("mps")
    require(device.type == "mps", "Actual Apple MPS device required")
    torch.set_num_threads(4)
    trainer.seed_all(cfg["fixed_seed"], device)
    model = trainer.legacy.load_full_encoder(cfg["upstream"], cfg["official_weights"])
    model = model.to(device=device, dtype=torch.float32)
    store = trainer.Store()
    scores = {}
    for key in ("before", "after"):
        path = Path(cfg["training_checkpoint_" + key])
        state = torch.load(path, map_location="cpu", weights_only=True)
        require(state.get("kind") == "train" and "model" in state,
                "Only real trained checkpoints eligible")
        model.load_state_dict(state["model"], strict=True)
        report = trainer.validate(model, rows, store, device, cfg["batch_size"],
                                  cfg["fixed_seed"], cfg["memory_budget_gib"])
        scores[key] = {"checkpoint_sha256": cfg["training_checkpoint_" + key + "_sha256"],
                       **report}
    result = {
        "schema": "xavier-p300-heldout-checkpoint-validation-result-v1",
        "status": "completed",
        "scope": "P300 heldout subject11 validation-only offline masked waveform reconstruction",
        "validation_subject_groups": 1, "validation_windows": len(rows),
        "test_groups_opened": 0, "test_windows": 0, "fit_steps": 0,
        "behavior_gold_read": False, "new_action_or_LLM_success_evaluation": False,
        "does_not_select_checkpoint": True, "fixed_mask_seed": cfg["fixed_seed"],
        "source_ledger_sha256": cfg["source_ledger_sha256"],
        "config_sha256": sha(cfg["_config_path"]),
        "elapsed_seconds": time.time() - start,
        "scores": scores,
        "difference_after_minus_before": (scores["after"]["mean_subject_masked_mse"]
                                           - scores["before"]["mean_subject_masked_mse"]),
        "limitations": ["one held-out P300 subject only", "no online causality",
                        "not a population generalization estimate",
                        "no free-intent or real-OS task evidence"]
    }
    atomic_json(out / "evaluation.json", result)
    print(json.dumps({k:v for k,v in result.items()
                      if k in ["status", "validation_subject_groups", "validation_windows",
                               "scores", "difference_after_minus_before", "elapsed_seconds"]},
                     sort_keys=True))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_json(args.config)
    cfg["_config_path"] = args.config
    evaluate(cfg)


if __name__ == "__main__":
    main()
