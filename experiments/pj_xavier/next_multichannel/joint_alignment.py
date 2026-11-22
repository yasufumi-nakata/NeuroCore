"""Pre-registered synthetic environment-mapping proxy for EEG->action alignment.

Only source cue A/B is gold; synthetic X/Y is a deterministic transformation,
not a measurement of spontaneous intent or evidence of OS action success.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

FORBIDDEN = frozenset({
    "cue", "cue_label", "target", "gold", "answer", "intent", "subject",
    "participant_id", "recording_id", "trial_id", "runnumber", "future",
    "texttospell", "stimulustype", "textresult", "selectedtarget",
    "selectedrow", "selectedcolumn", "task_correct",
})
CONTEXT_FIELDS = frozenset({"candidate_ids", "slot_to_id", "cue_to_slot",
                            "layout_id", "context_version"})
CLASSES = frozenset({"A","B"})
ACTIONS = frozenset({"X","Y"})
LAYOUTS = ("normal","swap_candidates","swap_cues","swap_both")


def validate_input_context(ctx: dict) -> None:
    if set(ctx) != CONTEXT_FIELDS:
        raise ValueError("environment input not on exact allowlist")
    if any(k.lower() in FORBIDDEN for k in ctx):
        raise ValueError("oracle feature in input")
    if ctx["candidate_ids"] != ["X","Y"]:
        raise ValueError("candidate whitelist")
    if set(ctx["slot_to_id"]) != {"left","right"} or set(ctx["slot_to_id"].values())!=ACTIONS:
        raise ValueError("invalid slot assignment")
    if set(ctx["cue_to_slot"]) != CLASSES or set(ctx["cue_to_slot"].values())!={"left","right"}:
        raise ValueError("invalid cue mapping")
    if ctx["layout_id"] not in LAYOUTS or ctx["context_version"]!="next-joint-v1":
        raise ValueError("unregistered layout")


def mapping(trial_key: str, layout: str) -> dict:
    if not trial_key or layout not in LAYOUTS:
        raise ValueError("missing fixed mapping key")
    # Shuffle order is deterministic and balanced across four layouts.
    swap_candidates = layout in ("swap_candidates","swap_both")
    swap_cues = layout in ("swap_cues","swap_both")
    ctx = {
        "context_version": "next-joint-v1",
        "candidate_ids": ["X","Y"],
        "layout_id": layout,
        "slot_to_id": {"left":"Y","right":"X"} if swap_candidates else {"left":"X","right":"Y"},
        "cue_to_slot": {"A":"right","B":"left"} if swap_cues else {"A":"left","B":"right"},
    }
    validate_input_context(ctx)
    return ctx


def synthetic_action_gold(cue_class: str, environment: dict) -> str:
    if cue_class not in CLASSES:
        raise ValueError("cue not independently observed as A/B")
    validate_input_context(environment)
    return environment["slot_to_id"][environment["cue_to_slot"][cue_class]]


def make_counterbalanced_trials(trial_key: str, cue_class: str, *,
                                allowed_split: str) -> list[dict]:
    if allowed_split not in ("train","val","test"):
        raise ValueError("sealed split required")
    out=[]
    for layout in LAYOUTS:
        env=mapping(trial_key,layout)
        out.append({
            "signal_trial_id_sha256":hashlib.sha256(trial_key.encode()).hexdigest(),
            "split": allowed_split,
            "input_environment":env,
            "original_source_cue_gold":cue_class,
            "synthetic_mapping_action_gold":synthetic_action_gold(cue_class,env),
            "label_provenance":"cue_original_plus_synthetic_environment_proxy",
            "free_intent_gold":None,
            "heldout_group_required":True,
        })
    return out


def verify_train_counterbalance(rows: list[dict]) -> dict:
    if any(r["split"]!="train" for r in rows):
        raise ValueError("counterbalance training uses train split only")
    counts=defaultdict(Counter)
    for r in rows:
        validate_input_context(r["input_environment"])
        counts[r["signal_trial_id_sha256"]][r["input_environment"]["layout_id"]]+=1
        if synthetic_action_gold(r["original_source_cue_gold"],r["input_environment"])!=r["synthetic_mapping_action_gold"]:
            raise ValueError("mismatched oracle mapping")
    if not counts or any(c!=Counter(dict.fromkeys(LAYOUTS,1)) for c in counts.values()):
        raise ValueError("not all four environments exactly once per original EEG trial")
    return {"unique_EEG_trials":len(counts),"environment_views":len(rows),
            "independent_signal_multiplier":1}


BENCHMARK_SPEC = {
    "primary": "mean_subject_balanced_accuracy",
    "secondary": ["macro_F1","AUROC","NLL","Brier","ECE","coverage_threshold_0.8",
                  "selective_accuracy","mapping_equivariance",
                  "routing_episode_success","abstention"],
    "negative_controls": [
        "context_only","EEG_removed","within_record_EEG_shuffle",
        "real_EEG_minus_10_seconds_common_ids","correct_class_routing_upper_bound",
        "T0_idle_diagnostic_not_natural_idle_false_actions_per_min",
    ],
    "bootstrap": {"unit":"heldout_subject_or_must_link_component","repetitions":1000,
                  "seed":20261122},
    "generalization": "fresh_unused_subject_holdout_must_be_reserved_before_fit",
    "old_17test_usage": "exploratory_only_not_for_new_model_selection",
    "action_truth": "A/B source cue transformed under 4 synthetic layout contexts; no free-intent gold",
    "harness": "fresh_state_per_trial; same cache-vs-live generation accounting; no OS actions",
    "new_model_performance": "NOT_EVALUATED",
}
