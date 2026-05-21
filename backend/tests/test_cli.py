from __future__ import annotations

from neurocore.cli import main


def test_cli_settings_json(capsys) -> None:
    exit_code = main(["settings", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"settings"' in captured.out
    assert '"generic-eeg"' in captured.out


def test_cli_self_test_json(capsys) -> None:
    exit_code = main(["self-test", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"status": "passed"' in captured.out


def test_cli_stream_demo_json(capsys) -> None:
    exit_code = main(["stream-demo", "--seconds", "2", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"window_count"' in captured.out


def test_cli_simulate_intents_json(capsys) -> None:
    exit_code = main(["simulate-intents", "samples/intent_commands.json", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"events"' in captured.out
    assert '"agent_prompt_like_payload"' in captured.out
