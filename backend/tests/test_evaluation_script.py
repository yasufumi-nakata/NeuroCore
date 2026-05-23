from __future__ import annotations

import json
import subprocess
import sys


def test_runtime_accuracy_evaluator_writes_private_report(tmp_path) -> None:
    report = tmp_path / "report.json"
    summary = tmp_path / "report.md"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_runtime_accuracy.py",
            "--skip-package",
            "--output",
            str(report),
            "--markdown-output",
            str(summary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["confidentiality"].startswith("private research metrics")
    assert payload["metrics"]["routing"]["case_count"] >= 10
    assert payload["overall"]["failed_checks"] == 0
    assert "Do not copy this report into public docs" in summary.read_text(encoding="utf-8")
