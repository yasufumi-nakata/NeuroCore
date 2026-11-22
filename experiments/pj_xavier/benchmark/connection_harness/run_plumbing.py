#!/usr/bin/env python3
"""Execute only bundled, hand-authored Java fixtures against a new test oracle.

This is not an evaluator for arbitrary/untrusted submissions and is not an EEG test.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent
FIXTURES = {
    "correct": "all_passed",
    "wrong_priority": "test_failure",
    "always_success": "test_failure",
    "accidental_npe": "test_failure",
    "syntax_error": "compile_failure",
}
CASE_IDS = {
    "wifi_priority_despite_list_order", "wifi_only_success", "mobile_only_success",
    "failed_wifi_falls_back_to_mobile", "no_available_types_throws",
    "all_available_attempts_fail", "unavailable_mobile_is_not_fallback",
    "unavailable_wifi_is_not_fallback",
}
EXPECTED_FAILED_CASES = {
    "correct": set(),
    "wrong_priority": {"wifi_priority_despite_list_order"},
    "always_success": CASE_IDS - {"wifi_priority_despite_list_order", "wifi_only_success"},
    "accidental_npe": CASE_IDS,
    "syntax_error": set(),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(argv: list[str], directory: Path, timeout: int) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in {"CLASSPATH", "JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS"}}
    start = time.monotonic()
    try:
        result = subprocess.run(argv, cwd=directory, env=env, capture_output=True,
                                text=True, timeout=timeout, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout[:16384],
                "stderr": result.stderr[:4096], "timeout": False,
                "elapsed_seconds": time.monotonic() - start}
    except subprocess.TimeoutExpired:
        return {"returncode": None, "stdout": "", "stderr": "", "timeout": True,
                "elapsed_seconds": time.monotonic() - start}


def parse_cases(stdout: str) -> list[dict]:
    rows = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    if len(rows) != len(CASE_IDS) or {x.get("case") for x in rows} != CASE_IDS:
        raise ValueError("Oracle did not report each required case exactly once")
    if any(type(x.get("passed")) is not bool for x in rows):
        raise ValueError("Nonboolean outcome")
    return rows


def run_fixture(name: str, java: str, javac: str, temporary_root: Path) -> dict:
    if name not in FIXTURES:
        raise ValueError("Only reviewed bundled synthetic fixtures may be executed")
    result = {"fixture": name, "expected_status": FIXTURES[name],
              "source_sha256": sha(ROOT / "fixtures" / (name + ".java")),
              "source_origin": "hand_authored_synthetic_fixture", "cases": []}
    with tempfile.TemporaryDirectory(prefix=name + "-", dir=temporary_root) as work:
        directory = Path(work)
        source, classes, oracle = (directory / n for n in ("candidate", "classes", "oracle"))
        for path in (source, classes, oracle):
            path.mkdir()
        # Only public API stubs and the selected fixture enter the candidate tree.
        for api in ("ConnectionAPI.java", "DeviceAPI.java"):
            shutil.copyfile(ROOT / "public_input" / api, source / api)
        shutil.copyfile(ROOT / "fixtures" / (name + ".java"), source / "Device.java")
        result["candidate_compile"] = execute(
            [javac, "-J-Xmx128m", "-encoding", "UTF-8", "-d", str(classes)]
            + [str(p) for p in sorted(source.glob("*.java"))], directory, 15)
        if result["candidate_compile"]["timeout"]:
            result["status"] = "compile_timeout"
        elif result["candidate_compile"]["returncode"] != 0:
            result["status"] = "compile_failure"
        else:
            # Oracle is a separate input and is never included in task.json.
            shutil.copyfile(ROOT / "oracle" / "ConnectionCases.java", oracle / "ConnectionCases.java")
            result["oracle_compile"] = execute(
                [javac, "-J-Xmx128m", "-encoding", "UTF-8", "-cp", str(classes),
                 "-d", str(classes), str(oracle / "ConnectionCases.java")], directory, 15)
            if result["oracle_compile"]["returncode"] != 0:
                result["status"] = "harness_compile_error"
            else:
                evaluation = execute(
                    [java, "-Xms16m", "-Xmx128m", "-XX:ActiveProcessorCount=1",
                     "-Djava.awt.headless=true", "-Duser.home=" + str(directory),
                     "-Djava.io.tmpdir=" + str(directory), "-cp", str(classes),
                     "task.ConnectionCases"], directory, 5)
                result["execution"] = evaluation
                if evaluation["timeout"]:
                    result["status"] = "execution_timeout"
                else:
                    try:
                        result["cases"] = parse_cases(evaluation["stdout"])
                        passed = all(x["passed"] for x in result["cases"])
                        result["status"] = "all_passed" if passed and evaluation["returncode"] == 0 else "test_failure"
                    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                        result["status"] = "invalid_oracle_output"
                        result["reason"] = str(error)
        result["passed_cases"] = sum(x["passed"] for x in result["cases"])
        result["evaluated_cases"] = len(result["cases"])
        result["expected_failed_case_ids"] = sorted(EXPECTED_FAILED_CASES[name])
        result["actual_failed_case_ids"] = sorted(x["case"] for x in result["cases"] if not x["passed"])
        result["matches_expected_outcome"] = matches_expected(name, result)
    result["temporary_workspace_removed"] = not directory.exists()
    return result


def matches_expected(name: str, result: dict) -> bool:
    expected_count = 0 if name == "syntax_error" else len(CASE_IDS)
    return (result["status"] == FIXTURES[name]
            and len(result["cases"]) == expected_count
            and {x["case"] for x in result["cases"] if not x["passed"]} == EXPECTED_FAILED_CASES[name])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a previous plumbing result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    java, javac = shutil.which("java"), shutil.which("javac")
    sources = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix in {".java", ".py", ".json"}
               and "__pycache__" not in p.parts]
    report = {
        "schema": "xavier.synthetic_connection_plumbing_result.v2",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "domain": "synthetic_non_eeg_engineering_fixture",
        "claim": "Independent plumbing and test-sensitivity check only",
        "original_publisher_oracle_recovered": False,
        "independent_failure_contract": "Expected unsuccessful connection must deliberately throw exact RuntimeException; accidental subclasses are not credited",
        "v1_correction": "Earlier v1 credited any RuntimeException subclass; v1 source/receipt preserved and this v2 uses a new execution output",
        "empirical_eeg_task_completion": {"status": "not_evaluated", "passed": False},
        "autonomous_agent_goal": {"status": "unproven", "passed": False},
        "participant_solutions_loaded": 0, "eeg_records_used": 0,
        "model_inference_calls": 0, "network_calls": 0, "desktop_actions": 0,
        "execution_scope": "Only bundled hand-authored fixtures; no untrusted-code sandbox claim",
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(sources)},
        "fixtures": [],
    }
    if not java or not javac:
        report.update(status="blocked", reason="Existing Java runtime and compiler required")
    else:
        report["java_version"] = execute([java, "-version"], ROOT, 10)
        report["javac_version"] = execute([javac, "-version"], ROOT, 10)
        if report["java_version"]["returncode"] != 0 or report["javac_version"]["returncode"] != 0:
            report.update(status="blocked", reason="Existing Java runtime/compiler unavailable")
        else:
            with tempfile.TemporaryDirectory(prefix="connection-plumbing-", dir=args.output.parent) as temp:
                report["fixtures"] = [run_fixture(name, java, javac, Path(temp)) for name in FIXTURES]
            report["status"] = "completed" if all(x["matches_expected_outcome"] and x["temporary_workspace_removed"]
                                                      for x in report["fixtures"]) else "failed"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output),
                      "report_sha256": sha(args.output),
                      "fixtures": [{k: x[k] for k in ("fixture", "status", "passed_cases", "evaluated_cases",
                                                      "matches_expected_outcome")} for x in report["fixtures"]]}))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
