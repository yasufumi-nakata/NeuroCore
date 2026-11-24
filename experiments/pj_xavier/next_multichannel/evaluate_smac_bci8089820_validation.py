#!/usr/bin/env python3
"""Independent frozen BCI validation, after completed training and verification.

Only the metadata-frozen 96 validation GDFs may be extracted. This diagnostic
cannot fit a model, select a checkpoint, read test/review signals, or change QC.
Raw stays on smac05. The scientific protocol was sealed before new model scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np
import torch

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE.parent / "model"))
import prepare_smac_bci8089820 as bci
import cbramod_ssl_smac as trainer

PROTOCOL_SHA256 = "65e33394e56f05ce03fb5f79c41486d0e781af81d555ff18780178ec6458f9ab"
SCIENTIFIC_SPEC_SHA256 = "07f02d0b31f85127a41a084acb2d488e1a08a19f06f57e29167a8356a7326596"
PREPARATION_SHA256 = "85d8a12467deea65c3cecc0dd568455636def142e8f1b1a6cb969f7d68c60975"
TRAINER_SHA256 = "ad0d8c27352411db52640c077af4b4bb38c6b1652ea20f3e6e965af2c172ca8a"
LEGACY_SHA256 = "47f24f9429bdc9af9495b121c4bf2cbc94ce62dbeb6b1033eeff6a4a88fe60fd"
FROZEN_MODEL_SHA256 = "b0c397176f8cb9e6dd759ed7169c0029c870f756c501ce317c2e6c7c41f32505"
SEED, BATCH_SIZE, MEMORY_GIB, RESERVE_GIB = 20261123, 32, 40, 100
FOLDS = {"train": 446, "val": 96, "test": 96, "review": 56}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_protocol(path):
    require(sha(path) == PROTOCOL_SHA256, "Preregistered scientific protocol changed")
    p = read_json(path)
    require(p["schema"] == "xavier-bci8089820-independent-validation-preregistration-v1",
            "Unknown scientific protocol")
    require(canonical(p["scientific_spec"]) == SCIENTIFIC_SPEC_SHA256,
            "Scientific selection, masking or QC changed")
    require(p["frozen_before_new_BCI_checkpoint_scores_observed"] is True,
            "Scientific plan was not preregistered")
    return p


def protocol_matches_preparation(protocol, preparation_protocol):
    """Do not silently substitute a filter, source reader, or model implementation."""
    spec = protocol["scientific_spec"]
    pp = preparation_protocol
    require(pp["preparation_sha256"] == PREPARATION_SHA256 == sha(bci.__file__),
            "Training/evaluation GDF preparation implementations differ")
    require(pp["trainer_sha256"] == TRAINER_SHA256 == sha(trainer.__file__),
            "Training/evaluation trainer implementations differ")
    require(sha(trainer.legacy.__file__) == LEGACY_SHA256
            and sha(trainer.legacy.frozen.__file__) == FROZEN_MODEL_SHA256,
            "Frozen masking, loss or model dependency changed")
    require(pp["audit_helper_sha256"] == sha(bci.original.__file__),
            "Original GDF audit parser differs")
    require(pp["frozen_split_sha256"] == spec["frozen_metadata_sha256"]
            and pp["source_ledger_sha256"] == spec["recording_ledger_sha256"]
            and pp["source_constraints_sha256"] == spec["duplicate_constraints_sha256"],
            "Source metadata or original duplicate constraints differ")
    actual = pp["preprocessing"]
    expected = spec["preprocessing"]
    require(actual["channels"] == bci.CHANNELS
            and actual["native_sample_rate_hz"] == 512
            and actual["output_sample_rate_hz"] == 200
            and actual["warmup_seconds"] == 2
            and actual["nonoverlap_window_seconds"] == 3
            and actual["output_unit"] == "V_x_1e4"
            and actual["additional_LP80_used"] is False,
            "Frozen preprocessing geometry differs")
    require(np.array_equal(np.asarray(actual["sos_coefficients"], dtype=np.float64),
                           np.asarray(expected["sos_coefficients"], dtype=np.float64)),
            "Training/evaluation filter coefficients differ")
    q = pp["QC"]
    require(q["native_max_abs_V"] == .1 and q["postprocess_max_abs_V"] == 1e-4
            and q["minimum_channel_std_V"] == 1e-8 and q["unit_correction_or_clipping"] is False,
            "Fixed source QC changed")
    require(pp["event_timebase_verified"] is False and pp["behavior_labels_used"] is False
            and pp["online_harness_allowed"] is False, "Unexpected label/causality promotion")


def select_validation(plan, records, existing_rows, existing_admission):
    require(dict(Counter(m["split"] for m in plan["members"])) == FOLDS,
            "Frozen source folds changed")
    selected = sorted((m for m in plan["members"] if m["split"] == "val"),
                      key=lambda m: m["source_member"])
    require(len(selected) == 96 and len({m["subject_group"] for m in selected}) == 12,
            "Validation must remain the frozen 96 records / 12 candidate groups")
    require(len({m["source_member"] for m in selected}) == 96,
            "Repeated validation source member")

    # Include old checkpoint-selection validation too, to keep this diagnostic
    # independent of both fitting and previously used model-selection inputs.
    protected_sources = {r["source_sha256"] for r in existing_rows}
    protected_groups = {r["subject_group"] for r in existing_rows}
    protected_payloads = {r["eeg_payload_sha256"] for r in existing_rows}
    for info in existing_admission["sources"].values():
        for key in ("eeg_payload_sha256", "original_EEG_scalar_payload_sha256",
                    "original_integer_EEG_payload_sha256"):
            if info.get(key):
                protected_payloads.add(info[key])
    nonval = [m for m in plan["members"] if m["split"] != "val"]
    nonval_sources = {m["source_file_sha256"] for m in nonval}
    nonval_scalars = {m["eeg_payload_sha256"] for m in nonval}
    for member in selected:
        rec = records[member["source_member"]]
        require(rec["source_file_sha256"] == member["source_file_sha256"]
                and rec["eeg_payload_sha256"] == member["eeg_payload_sha256"]
                and rec["EEG_calibration_sha256"] == member["calibration_sha256"],
                "Frozen validation identity changed")
        require(member["subject_group"] not in protected_groups
                and member["source_file_sha256"] not in protected_sources
                and member["eeg_payload_sha256"] not in protected_payloads,
                "Validation overlaps earlier fit or checkpoint-selection data")
        require(member["source_file_sha256"] not in nonval_sources
                and member["eeg_payload_sha256"] not in nonval_scalars,
                "Validation overlaps a frozen train/test/review identity")
    return selected, {"sources": protected_sources,
                      "payloads": protected_payloads | nonval_scalars,
                      "groups": protected_groups}


def extract_validation_member(zf, member, record, target):
    require(member["split"] == "val", "Only frozen validation GDF extraction is permitted")
    name = member["source_member"]
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and path.suffix.lower() == ".gdf",
            "Unsafe validation ZIP member")
    info = zf.getinfo(name)
    require(not info.is_dir() and info.file_size == record["source_file_bytes"]
            and info.file_size <= bci.MAX_MEMBER_BYTES,
            "Validation source size differs from original ledger")
    require((info.external_attr >> 16) & 0o170000 != 0o120000, "ZIP symlink disallowed")
    h = hashlib.sha256()
    count = 0
    with zf.open(info) as source, Path(target).open("xb") as output:
        for chunk in iter(lambda: source.read(1024**2), b""):
            h.update(chunk)
            count += len(chunk)
            output.write(chunk)
    require(count == info.file_size and h.hexdigest() == record["source_file_sha256"],
            "Validation ZIP CRC/source SHA mismatch")


def prepare_validation(cfg, out, selected, records, protected, deadline):
    out = Path(out)
    (out / "arrays").mkdir()
    manifest, certificates, artifacts = [], [], {}
    seen_source, seen_scalar, seen_physical = {}, {}, {}
    opened = []
    montage = bci.canonical([bci.CHANNELS, bci.REFERENCE, 512, bci.SOURCE])
    with zipfile.ZipFile(cfg["archive"], "r") as zf:
        with tempfile.TemporaryDirectory(prefix="temporary_validation_GDF_", dir=out) as temp:
            for member in selected:
                require(time.monotonic() < deadline, "Independent evaluation time budget expired")
                require(shutil.disk_usage(out).free > cfg.get("reserve_gib", RESERVE_GIB) * 1024**3
                        + 4 * bci.MAX_MEMBER_BYTES, "Evaluation storage reserve exhausted")
                record = records[member["source_member"]]
                target = Path(temp) / "current.gdf"
                cert = {"source_member": member["source_member"], "split": "val",
                        "subject_group": member["subject_group"],
                        "source_sha256": member["source_file_sha256"],
                        "original_EEG_scalar_payload_sha256": member["eeg_payload_sha256"],
                        "EEG_calibration_sha256": member["calibration_sha256"],
                        "signal_opened": False, "fit_allowed": False}
                x = None
                try:
                    extract_validation_member(zf, member, record, target)
                    cert["signal_opened"] = True
                    opened.append(member["source_member"])
                    try:
                        x, proof = bci.verify_gdf(target, record)
                    except bci.PhysicalCalibrationHold as hold:
                        cert.update(hold.evidence, status=hold.reason)
                        certificates.append(cert)
                        continue
                    cert.update(proof)
                    require(proof["physical_unit_verified"] is True,
                            "Independent physical-V calibration missing")
                    physical = proof["decoded_EEG_V_sha256"]
                    require(physical not in protected["payloads"],
                            "Validation physical payload overlaps protected fit/selection/folds")
                    scalar_identity = (member["eeg_payload_sha256"], member["calibration_sha256"],
                                       tuple(bci.CHANNELS), 512, record["samples_per_EEG_channel"])
                    aliases = [seen_source.get(member["source_file_sha256"]),
                               seen_scalar.get(scalar_identity), seen_physical.get(physical)]
                    aliases = [a for a in aliases if a is not None]
                    if aliases:
                        require(all(a["group"] == member["subject_group"] for a in aliases),
                                "Unexpected cross-group validation duplicate needs review")
                        cert.update(status="excluded_exact_validation_alias",
                                    same_as_source_sha256=aliases[0]["source"])
                    elif proof["observed_native_peak_V"] > .1:
                        cert["status"] = "held_native_peak_above_fixed_0_1V"
                    else:
                        array, intervals, rejected = bci.causal_windows(x)
                        cert.update(accepted_windows=len(intervals), rejected_windows=rejected)
                        if not intervals:
                            cert["status"] = "held_no_window_meets_fixed_QC"
                        else:
                            require(array.dtype == np.float32
                                    and array.shape == (len(intervals), 27, 3, 200),
                                    "Validation array geometry differs")
                            source = member["source_file_sha256"]
                            rel = "arrays/" + source + ".npy"
                            np.save(out / rel, array, allow_pickle=False)
                            artifacts[rel] = {"sha256": sha(out / rel),
                                              "shape": list(array.shape), "dtype": "float32",
                                              "bytes": (out / rel).stat().st_size}
                            for index, interval in enumerate(intervals):
                                manifest.append({
                                    "id": source + ":" + str(index), "split": "val",
                                    "subject_group": member["subject_group"],
                                    "source_sha256": source, "eeg_payload_sha256": physical,
                                    "source_recording_id": bci.SOURCE + ":" + member["source_member"],
                                    "channels": bci.CHANNELS, "patches": 3, "eeg_index": index,
                                    "_path": str(out / rel), "eeg_path": rel,
                                    "montage_key": montage, "reference": bci.REFERENCE,
                                    "native_sample_rate_hz": 512, "sample_rate_hz": 200,
                                    "unit": "V_x_1e4", "fit_allowed": False, **interval,
                                })
                            ident = {"group": member["subject_group"], "source": source}
                            seen_source[source] = ident
                            seen_scalar[scalar_identity] = ident
                            seen_physical[physical] = ident
                            cert.update(status="accepted_independent_validation",
                                        accepted_unique_seconds=len(intervals) * 3)
                    certificates.append(cert)
                finally:
                    del x
                    target.unlink(missing_ok=True)
    require(len(certificates) == len(selected) == 96 and len(set(opened)) == 96,
            "Not all frozen validation originals were checked")
    require(set(opened) == {m["source_member"] for m in selected},
            "A non-validation source was opened")
    candidate_groups = sorted({m["subject_group"] for m in selected})
    achieved_groups = sorted({r["subject_group"] for r in manifest})
    missing = sorted(set(candidate_groups) - set(achieved_groups))
    summary = {
        "schema": "xavier-bci8089820-independent-validation-provenance-v1",
        "candidate_records": 96, "candidate_groups": 12,
        "records_with_accepted_windows": len({r["source_sha256"] for r in manifest}),
        "groups_with_accepted_windows": len(achieved_groups),
        "complete_12_group_coverage": len(achieved_groups) == 12,
        "missing_groups": missing,
        "missing_group_reasons": {
            g: dict(Counter(c["status"] for c in certificates if c["subject_group"] == g))
            for g in missing
        },
        "validation_windows": len(manifest), "unique_seconds": len(manifest) * 3,
        "source_status_counts": dict(Counter(c["status"] for c in certificates)),
        "train_GDF_opened": 0, "test_GDF_opened": 0, "review_GDF_opened": 0,
        "validation_GDF_opened": len(opened), "fit_steps": 0,
        "behavior_labels_used": False, "event_timebase_verified": False,
        "checkpoint_selection": False,
    }
    write_json(out / "source_certificates.json", certificates)
    write_json(out / "validation_provenance.json", summary)
    write_json(out / "artifacts.json", artifacts)
    write_json(out / "source_access.json", {
        "validation_members_opened": opened, "train_members_opened": [],
        "test_members_opened": [], "review_members_opened": [],
    })
    (out / "validation_manifest.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in manifest))
    return manifest, summary


def aggregate_group_scores(group_scores):
    require(group_scores, "No eligible validation groups")
    require(all(r["windows"] > 0 and math.isfinite(r["mean_masked_mse"])
                for r in group_scores), "Invalid group reconstruction score")
    return {
        "mean_subject_masked_mse": float(np.mean([r["mean_masked_mse"] for r in group_scores])),
        "groups": len(group_scores), "windows": sum(r["windows"] for r in group_scores),
        "per_group": group_scores, "test_windows": 0,
    }


def score_model(model, manifest, store, device, deadline=None):
    groups = defaultdict(list)
    for row in manifest:
        groups[row["subject_group"]].append(row)
    scores = []
    with torch.inference_mode():
        for group in sorted(groups):
            require(deadline is None or time.monotonic() < deadline, "Evaluation group scoring deadline reached")
            report = trainer.validate(model, groups[group], store, device,
                                      BATCH_SIZE, SEED, MEMORY_GIB)
            require(report["groups"] == 1 and report["windows"] == len(groups[group]),
                    "Validation group accounting changed")
            scores.append({"subject_group": group, "windows": report["windows"],
                           "mean_masked_mse": report["mean_subject_masked_mse"]})
    return aggregate_group_scores(scores)


@contextmanager
def exclusive_gpu(path):
    path = Path(path)
    require(path.is_file(), "Shared training GPU lock is missing")
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def preflight(cfg, config_path, expected_config_sha):
    require(sha(config_path) == expected_config_sha, "Immutable evaluation run config changed")
    require(cfg["schema"] == "xavier-bci8089820-independent-validation-run-v1",
            "Unknown evaluation run config")
    node = platform.node().split(".")[0].lower()
    require(node == "smac05", "Real BCI validation may run only on smac05")
    root = (Path.home() / "pj-xavier" / node).resolve()
    out = Path(cfg["output"]).expanduser().absolute()
    require(out.parent.resolve() == root / "evaluations" and not out.exists(),
            "Use a new immutable evaluation directory on smac05")
    for key in ("archive", "corpus", "before", "after", "upstream", "official_weights",
                "frozen_manifest", "recordings", "constraints", "verification_receipt"):
        path = Path(cfg[key]).expanduser().resolve()
        require(root in path.parents, "Evaluation input must stay on the S Mac: " + key)
    require(Path(cfg["archive"]).resolve() ==
            root / "raw" / bci.preflight.SOURCE_KEY / "BCI Database.zip",
            "Reuse only the existing pinned source archive")
    require(Path(cfg["gpu_lock"]).absolute() == root / "train_gpu.lock",
            "Evaluation must use the same training GPU lock")
    require(cfg.get("reserve_gib", RESERVE_GIB) == RESERVE_GIB
            and cfg.get("max_seconds", 1800) == 1800, "Frozen resource limits changed")
    for key in ("before", "after", "official_weights", "verification_receipt"):
        require(sha(cfg[key]) == cfg[key + "_sha256"], "Sealed input differs: " + key)
    require(Path(cfg["before"]).name == "best.pt" and Path(cfg["after"]).name == "best.pt",
            "Only the preregistered best checkpoints may be evaluated")
    require(cfg["before_sha256"] != cfg["after_sha256"], "Before/after model bytes are identical")
    proof = read_json(cfg["verification_receipt"])
    require(proof["status"] in ("passed", "completed_verified")
            and proof.get("checks") and all(v is True for v in proof["checks"].values()),
            "Independent preparation/model verification must pass first")
    require(proof["bindings"]["before_sha256"] == cfg["before_sha256"]
            and proof["bindings"]["after_sha256"] == cfg["after_sha256"]
            and proof["bindings"]["corpus_completion_sha256"] == cfg["completion_sha256"],
            "Independent verification belongs to another corpus/checkpoint pair")
    protocol = load_protocol(cfg["protocol"])
    base, manifest, admission, _, _ = bci.validate_base(cfg["corpus"], cfg["completion_sha256"])
    prep_protocol = read_json(base / "protocol.json")
    protocol_matches_preparation(protocol, prep_protocol)
    require(sha(__file__) == cfg["evaluator_sha256"], "Evaluator source differs from bound config")
    plan, records = bci.load_metadata(cfg["frozen_manifest"], cfg["recordings"], cfg["constraints"])
    selected, protected = select_validation(plan, records, manifest, admission)
    source = bci.preflight.run(root, Path(cfg["frozen_manifest"]))
    require(source["status"] == "ready_for_train_only_physical_qc",
            "Pinned original ZIP is not complete and verified")
    require(shutil.disk_usage(root).free >= RESERVE_GIB * 1024**3 + 2 * 1024**3,
            "Validation reserve or temporary capacity insufficient")
    return out, protocol, selected, records, protected


def evaluate(cfg, config_path, expected_config_sha):
    out, protocol, selected, records, protected = preflight(cfg, config_path, expected_config_sha)
    started = time.monotonic()
    source_stat = Path(cfg["archive"]).stat()
    with exclusive_gpu(cfg["gpu_lock"]):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.mkdir(mode=0o700)
        shutil.copyfile(cfg["protocol"], out / "preregistered_protocol.json")
        shutil.copyfile(config_path, out / "run_config.json")
        write_json(out / "evaluation_state.json", {
            "phase": "validation_preparation", "fit_steps": 0,
            "scientific_protocol_sha256": PROTOCOL_SHA256,
            "new_checkpoint_scores_observed_before_protocol_freeze": False,
        })
        try:
            manifest, provenance = prepare_validation(
                cfg, out, selected, records, protected, started + 1800)
            if not manifest:
                result = {
                    **provenance,
                    "schema": "xavier-bci8089820-independent-validation-result-v1",
                    "status": "blocked_no_eligible_validation_windows",
                    "scores": {}, "fit_steps": 0,
                    "scientific_protocol_sha256": PROTOCOL_SHA256,
                    "checkpoint_selection": False,
                }
                write_json(out / "results.json", result)
                return result
            require(time.monotonic() < started + 1800, "Evaluation deadline reached before scoring")
            device = trainer.legacy.frozen.select_device("mps")
            require(device.type == "mps", "Actual Apple MPS device required")
            torch.set_num_threads(4)
            trainer.seed_all(SEED, device)
            model = trainer.legacy.load_full_encoder(
                cfg["upstream"], cfg["official_weights"]).to(device=device, dtype=torch.float32)
            store = trainer.Store()
            scores = {}
            for label in ("before", "after"):
                require(time.monotonic() < started + 1800, "Evaluation scoring deadline reached")
                require(sha(cfg[label]) == cfg[label + "_sha256"], "Checkpoint changed before scoring")
                checkpoint = torch.load(cfg[label], map_location="cpu", weights_only=True)
                require(checkpoint.get("kind") == "train" and "model" in checkpoint,
                        "Only completed real trained checkpoints are eligible")
                model.load_state_dict(checkpoint["model"], strict=True)
                scores[label] = {**score_model(model, manifest, store, device, started + 1800),
                                 "checkpoint_sha256": cfg[label + "_sha256"]}
                del checkpoint
            now = Path(cfg["archive"]).stat()
            require((source_stat.st_ino, source_stat.st_size, source_stat.st_mtime_ns) ==
                    (now.st_ino, now.st_size, now.st_mtime_ns), "Pinned source archive changed")
            delta = scores["after"]["mean_subject_masked_mse"] - scores["before"]["mean_subject_masked_mse"]
            before_score = scores["before"]["mean_subject_masked_mse"]
            paired = []
            for old, new in zip(scores["before"]["per_group"], scores["after"]["per_group"]):
                require(old["subject_group"] == new["subject_group"] and old["windows"] == new["windows"],
                        "Before/after evaluation populations differ")
                paired.append({"subject_group": old["subject_group"], "windows": old["windows"],
                               "after_minus_before": new["mean_masked_mse"] - old["mean_masked_mse"]})
            result = {
                **provenance,
                "schema": "xavier-bci8089820-independent-validation-result-v1", "status": "completed",
                "scores": scores, "paired_group_changes": paired,
                "after_minus_before": delta,
                "relative_change_percent": 100 * delta / before_score if before_score else None,
                "lower_loss_group_count": sum(r["after_minus_before"] < 0 for r in paired),
                "scientific_protocol_sha256": PROTOCOL_SHA256,
                "run_config_sha256": expected_config_sha,
                "preparation_sha256": PREPARATION_SHA256, "trainer_sha256": TRAINER_SHA256,
                "evaluator_sha256": sha(__file__),
                "independent_verification_sha256": cfg["verification_receipt_sha256"],
                "provenance_sha256": sha(out / "validation_provenance.json"),
                "certificates_sha256": sha(out / "source_certificates.json"),
                "artifacts_sha256": sha(out / "artifacts.json"),
                "manifest_sha256": sha(out / "validation_manifest.jsonl"),
                "elapsed_seconds": time.monotonic() - started,
                "limitations": [
                    "Offline masked EEG reconstruction on eligible frozen validation groups only.",
                    "Missing groups and rejected recordings are reported, never replaced or zero-imputed.",
                    "No test/review signal access, fitting, or checkpoint selection by this diagnostic.",
                    "Original event timebase, amplifier calibration, and online causality are unverified.",
                    "Known exact identities only; transformed or undocumented cross-release equivalence remains unverified.",
                    "No behavioral, free-intent, real-OS task, or population-level generalization claim.",
                ],
            }
            write_json(out / "results.json", result)
            write_json(out / "evaluation_state.json", {"phase": "completed", "fit_steps": 0})
            return result
        except Exception as exc:
            write_json(out / "failure.json", {
                "status": "error", "error_type": type(exc).__name__, "reason": str(exc),
                "fit_steps": 0, "checkpoint_selection": False,
                "scientific_protocol_sha256": PROTOCOL_SHA256,
                "elapsed_seconds": time.monotonic() - started,
            })
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--expected-config-sha256", required=True)
    args = ap.parse_args()
    cfg = read_json(args.config)
    result = evaluate(cfg, args.config, args.expected_config_sha256)
    print(json.dumps({k: result.get(k) for k in (
        "status", "candidate_records", "candidate_groups", "groups_with_accepted_windows",
        "validation_windows", "after_minus_before", "relative_change_percent", "fit_steps"
    )}, sort_keys=True, allow_nan=False))
    return 2 if result["status"].startswith("blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
