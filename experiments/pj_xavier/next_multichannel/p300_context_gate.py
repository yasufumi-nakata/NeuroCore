"""Fail-closed P300 context partition for a future, untrained PJ-Xavier run.

This module is a metadata/timing contract, NOT an EEG decoder, a clock
calibration, a signal loader or evidence of free intent. Labels and historical
decoder outputs never become model input.
"""
from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping
from typing import Any

_ALLOWED_RECORD = frozenset({
    "raw_dat_sha256", "decision_sample", "sampling_rate_hz", "display",
    "presentations", "protected_labels", "decoder_outputs",
})
_ALLOWED_DISPLAY = frozenset({
    "candidate_ids", "slot_to_id", "known_at_sample", "origin", "evidence_sha256",
})
_ALLOWED_EVENT = frozenset({
    "candidate_id", "onset_sample", "available_at_sample",
    "clock_evidence", "evidence_sha256",
})
_ALLOWED_ORACLE = frozenset({
    "TextToSpell", "StimulusType", "target_character", "target_row", "target_column",
})
_ALLOWED_OUTPUT = frozenset({
    "SelectedTarget", "SelectedRow", "SelectedColumn", "TextResult",
})
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class P300ContractError(ValueError):
    """A field, timing, or provenance violates the P300 input boundary."""


def _keys(value: Any, allowed: frozenset[str], required: set[str], name: str) -> dict:
    if not isinstance(value, Mapping):
        raise P300ContractError(f"{name} must be a mapping")
    keys = set(value)
    if not all(isinstance(x, str) for x in keys):
        raise P300ContractError(f"{name} contains non-string keys")
    if not required.issubset(keys) or not keys.issubset(allowed):
        raise P300ContractError(f"{name} has missing or prohibited keys")
    return dict(value)


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise P300ContractError(f"{name} requires lower-case SHA256")
    return value


def _sample(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise P300ContractError(f"{name} must be a nonnegative integer sample index")
    return value


def _safe_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,32}", value):
        raise P300ContractError("candidate/slot ID must be a bounded simple token")
    return value


def partition_p300_record(record: Mapping[str, Any]) -> dict:
    """Partition a *declared* record into inputs, protected labels and audit.

    For an input presentation to be included, its external display clock and
    both onset and availability must precede the decision sample. This does not
    validate external evidence and NEVER authorizes any action/test gold.
    """
    r = _keys(record, _ALLOWED_RECORD, set(_ALLOWED_RECORD), "record")
    raw_sha = _sha(r["raw_dat_sha256"], "raw_dat_sha256")
    cutoff = _sample(r["decision_sample"], "decision_sample")
    hz = r["sampling_rate_hz"]
    if type(hz) not in (int, float) or not (0 < hz <= 100000):
        raise P300ContractError("invalid declared sampling rate")
    display = _keys(r["display"], _ALLOWED_DISPLAY, set(_ALLOWED_DISPLAY), "display")
    candidates = display["candidate_ids"]
    if (not isinstance(candidates, list) or not 2 <= len(candidates) <= 64
            or len({_safe_id(v) for v in candidates}) != len(candidates)):
        raise P300ContractError("candidate set missing, too large or duplicated")
    slots = display["slot_to_id"]
    if not isinstance(slots, dict) or not slots or not all(
        isinstance(k, str) and _safe_id(k) and _safe_id(v)
        for k, v in slots.items()
    ) or set(slots.values()) != set(candidates) or len(slots) != len(candidates):
        raise P300ContractError("candidate-position mapping not one-to-one")
    known = _sample(display["known_at_sample"], "known_at_sample")
    if known > cutoff:
        raise P300ContractError("future display mapping")
    _sha(display["evidence_sha256"], "display.evidence_sha256")
    if display["origin"] != "verified_prior_screen_log":
        raise P300ContractError("mapping provenance not verified prior display")

    oracle = _keys(r["protected_labels"], _ALLOWED_ORACLE, set(), "protected_labels")
    decoder = _keys(r["decoder_outputs"], _ALLOWED_OUTPUT, set(), "decoder_outputs")
    if not isinstance(r["presentations"], list):
        raise P300ContractError("presentations must be a list")

    safe_presentations = []
    rejected = []
    for index, obj in enumerate(r["presentations"]):
        ev = _keys(obj, _ALLOWED_EVENT, set(_ALLOWED_EVENT), f"event[{index}]")
        candidate = _safe_id(ev["candidate_id"])
        if candidate not in candidates:
            raise P300ContractError("event references unknown candidate")
        onset = _sample(ev["onset_sample"], "onset_sample")
        available = _sample(ev["available_at_sample"], "available_at_sample")
        _sha(ev["evidence_sha256"], "event.evidence_sha256")
        if onset > available or available > cutoff:
            raise P300ContractError("future onset or late-available display event")
        if ev["clock_evidence"] != "externally_calibrated_display":
            rejected.append({"index": index, "reason": "clock_not_externally_calibrated"})
            continue
        safe_presentations.append({
            "candidate_id": candidate, "onset_sample": onset,
            "available_at_sample": available,
        })

    # The digest is audit metadata, never a learned feature.
    return {
        "schema": "xavier-p300-input-separation-v1",
        "model_input": {
            "candidate_ids": copy.deepcopy(candidates),
            "slot_to_id": copy.deepcopy(slots),
            "presentations": safe_presentations,
        },
        "oracle_only": copy.deepcopy(oracle),
        "decoder_output_only": copy.deepcopy(decoder),
        "audit_only": {
            "raw_dat_sha256": raw_sha,
            "decision_sample": cutoff,
            "sampling_rate_hz_declared_not_verified": hz,
            "display_evidence_sha256": display["evidence_sha256"],
            "display_known_at_sample": known,
            "total_presentations": len(r["presentations"]),
            "excluded_unverified_presentations": rejected,
        },
        "action_benchmark_authorized": False,
        "supervised_gold_authorized": False,
        "free_intent_gold": None,
        "onset_timing_ready_for_future_review": (
            bool(safe_presentations) and not rejected
        ),
        "blocking_conditions": [
            "external display-clock claim must be independently audited",
            "actual-person must-link and new holdout remain unverified",
            "no independent target gold or free-intent truth established",
        ],
    }


def proof_of_input_boundary(partition: dict) -> str:
    """Digest of input-only content for comparison; no protected data."""
    import json
    payload = json.dumps(partition["model_input"], sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
