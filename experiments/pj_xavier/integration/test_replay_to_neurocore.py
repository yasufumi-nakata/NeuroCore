from replay_to_neurocore import replay


def prediction(label="mouse_click", **changes):
    return {"id": "fixture", "predicted_label": label, "probabilities": {label: .95},
            "decision_time_s": 10., "condition": "normal", "abstain": False,
            "action_executed": False, **changes}


def test_real_router_requires_arming_and_records_only_dry_run():
    blocked, report = replay([prediction()], allow_mouse_click_proxy=True)
    assert blocked.audit_log.events[0].action["reason"] == "human_arm_required"
    sink, report = replay([prediction()], allow_mouse_click_proxy=True, human_armed=True)
    assert report["dry_run_candidates"] == 1
    assert sink.audit_log.events[0].note == "dry_run_only"
    assert report["os_actions_executed"] == report["cloud_agent_calls"] == 0


def test_ambiguous_keyboard_and_scroll_never_invent_payloads():
    sink, report = replay([prediction("keyboard"), prediction("mouse_scroll")],
                          allow_mouse_click_proxy=True, human_armed=True)
    assert report["blocked"] == 2
    assert all(e.action["value"] is None for e in sink.audit_log.events)


def test_perturbed_nan_and_abstaining_predictions_cannot_route():
    _, report = replay([prediction(condition="shuffled_eeg"), prediction(abstain=True),
                        prediction(probabilities={"mouse_click": float("nan")})],
                       allow_mouse_click_proxy=True, human_armed=True)
    assert report["blocked"] == 3
    assert report["autonomous_agent_validated"] is False
