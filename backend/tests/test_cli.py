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


def test_cli_dataset_exercise_json(tmp_path, capsys) -> None:
    csv_path = tmp_path / "inventory.csv"
    csv_path.write_text(
        "id,dataset_name,url,doi,source_domain,access_status,score,description\n"
        "1,sample,https://zenodo.org/records/123456,10.5281/zenodo.123456,zenodo.org,すぐに使える,5,raw EEG .edf\n",
        encoding="utf-8",
    )

    exit_code = main(["dataset-exercise", str(csv_path), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"record_count": 1' in captured.out
    assert '"remote_not_exercised": 1' in captured.out
