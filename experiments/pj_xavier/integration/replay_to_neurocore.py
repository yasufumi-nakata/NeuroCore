#!/usr/bin/env python3
"""Exercise the real NeuroCore router with model predictions, without OS actions.

Recorded action classes are proxy targets. They do not contain a keyboard key,
scroll direction, UI target, or a high-level user intention. Only an explicitly
requested mouse-click proxy can be mapped to the existing select binding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from neurocore.audit import DryRunActionSink
from neurocore.control import ControlAction, ControlRouter, IntentCommand
from neurocore.settings import NeuroCoreSettings


def replay(predictions, *, allow_mouse_click_proxy=False, human_armed=False):
    settings = NeuroCoreSettings.default().to_dict()
    settings["safety"]["human_armed"] = bool(human_armed)
    router = ControlRouter(NeuroCoreSettings.from_dict(settings))
    sink = DryRunActionSink()
    counts = {"records": 0, "dry_run_candidates": 0, "blocked": 0,
              "os_actions_executed": 0, "cloud_agent_calls": 0}
    for row in predictions:
        counts["records"] += 1
        label = row.get("predicted_label")
        probability = row.get("probabilities", {}).get(label)
        timestamp = row.get("decision_time_s")
        valid = (isinstance(probability, (int, float)) and not isinstance(probability, bool)
                 and math.isfinite(probability) and 0 <= probability <= 1
                 and isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool)
                 and math.isfinite(timestamp))
        command = None
        reason = None
        if not valid:
            reason = "invalid_probability_or_recording_timestamp"
        elif row.get("action_executed") is not False:
            reason = "prediction_not_verified_as_nonexecuting"
        elif row.get("condition") != "normal":
            reason = "perturbed_control_is_not_a_deployment_prediction"
        elif row.get("abstain") is not False:
            reason = "model_abstained_or_abstention_flag_missing"
        elif label != "mouse_click" or not allow_mouse_click_proxy:
            reason = "proxy_label_has_no_explicit_intent_mapping"
        if reason:
            action = ControlAction("blocked", str(label), None, True, reason,
                                   {"record_id": row.get("id"), "observed_action_proxy": True})
        else:
            command = IntentCommand("select", confidence=probability,
                                    source="pj-xavier-experimental-observed-action-proxy",
                                    payload={"record_id": row.get("id"), "label": label},
                                    timestamp=timestamp)
            action = router.route(command)
        sink.submit(action, command)
        counts["blocked" if action.blocked else "dry_run_candidates"] += 1
    return sink, {**counts, "mode": "dry_run_only", "autonomous_agent_validated": False,
                  "interpretation": "Router integration only. Class probabilities are not calibrated intent confidence; no task completion or EEG decoding accuracy is inferred."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-mouse-click-proxy", action="store_true")
    p.add_argument("--human-armed", action="store_true", help="Arm only this in-memory dry-run router")
    args = p.parse_args()
    rows = [json.loads(x) for x in args.predictions.read_text().splitlines() if x.strip()]
    sink, report = replay(rows, allow_mouse_click_proxy=args.allow_mouse_click_proxy,
                          human_armed=args.human_armed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sink.audit_log.write_jsonl(args.output)
    report.update(predictions_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
                  audit_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
