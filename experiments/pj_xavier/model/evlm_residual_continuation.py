"""Strict one-stage residual continuation admission, using immutable parent receipts."""
import json
import math
from pathlib import Path
import hashlib

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def need(ok, why):
    if not ok:
        raise ValueError(why)

def bound(entry):
    p = Path(entry["path"])
    need(p.is_file() and sha(p) == entry["sha256"], "continuation_receipt_hash_mismatch")
    return p

def admit(cfg):
    parent = cfg["warm_start_adapter"]
    state_path = bound(cfg["residual_parent_state"])
    state = json.loads(state_path.read_text())
    need(state["status"] == "paused_update_budget" and state["test_opened"] is False
         and state["latest_candidate"] == parent and state["completed_stages"] == 1,
         "parent_not_accepted_single_stage")
    prior_plan = json.loads(bound(cfg["residual_parent_plan"]).read_text())
    need(cfg["residual_parent_plan"]["sha256"] == state["plan_sha256"], "parent_plan_mismatch")
    stage = state["last_completed_training_stage"]
    need(cfg["data_manifest"] == stage["data_manifest"] and
         cfg["order_offset"] == stage["order_offset"] + stage["max_updates"],
         "parent_offset_or_source_mismatch")
    need(cfg["residual_candidate"] == prior_plan["stage_template"]["residual_candidate"]
         and all(cfg[k] == prior_plan["stage_template"][k] for k in
                 ("seed", "num_layers", "rank", "learning_rate", "max_seq_length",
                  "memory_limit_gb", "eval_max_records", "residual_protocol")),
         "parent_protocol_changed")
    receipt = json.loads(bound(cfg["residual_parent_completed"]).read_text())
    need(receipt["checkpoint"] == parent and receipt["stage"] == stage and
         receipt["test_opened"] is False and receipt["quality_gates_passed"] is True,
         "parent_completion_mismatch")
    nr = json.loads(bound({"path": state["last_numeric_retention_receipt"],
                           "sha256": state["last_numeric_retention_sha256"]}).read_text())
    need({k: v for k, v in nr.items() if k not in ("stage", "checkpoint")} == receipt["numeric_retention"]
         and nr["stage"] == stage and nr["passed"] is True and nr["new_best"] is True
         and nr["checkpoint"] == parent and nr["valid_rate"] == 1.0
         and nr["test_opened"] is False and nr["policy"]["relative_tolerance"] == 0.05,
         "parent_numeric_not_accepted")
    vision = receipt["vision_retention"]
    bundle = Path(vision["bundle"])
    need(str(bundle) == state["latest_vlm_bundle"], "parent_bundle_mismatch")
    vr = json.loads(bound({"path": vision["receipt"], "sha256": vision["receipt_sha256"]}).read_text())
    need(vr["status"] == "passed" and vr["synthetic_only"] is True and
         vr["base_correct"] == vr["adapter_correct"] == vr["cases_per_mode"] == 2
         and vr["test_opened"] is False and vr["executed_actions"] == 0
         and sha(bundle / "manifest.json") == vr["manifest_sha256"] and
         sha(bundle / "adapter/adapters.safetensors") == parent["adapter_sha256"],
         "parent_vision_not_accepted")
    evp = Path(parent["checkpoint_dir"]) / "evaluation.json"
    need(sha(evp) == receipt["evaluation_sha256"] == nr["evaluation_sha256"], "parent_eval_changed")
    ev = json.loads(evp.read_text())
    need(ev["metrics"]["residual_protocol"] == cfg["residual_protocol"]
         and ev["metrics"]["valid_count"] == 36 and
         ev["metrics"]["mse_on_valid_predictions"] == nr["ordinary_mse_on_valid_predictions"],
         "parent_residual_metrics_mismatch")
    return ev["metrics"]

def verify_baseline(expected, current):
    need(current["n"] == expected["n"] == 36 and
         current["valid_count"] == expected["valid_count"] == 36 and
         math.isclose(current["mse_on_valid_predictions"], expected["mse_on_valid_predictions"],
                      rel_tol=0, abs_tol=1e-12),
         "parent_development_baseline_not_reproduced")
