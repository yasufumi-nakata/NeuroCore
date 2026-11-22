#!/usr/bin/env python3
"""Read-only, aggregate audit of acquired coding-task resources. Never reads answers.

Only public task members and behavioral header schemas are read. Participant Java
and prompt/key content are not opened. No tar member is extracted or executed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import xml.etree.ElementTree as ET

PINNED = {
    "resources.tar.gz": "2b91b0bee3636c7fcd791e5e614033a6e44e9bb3aedb9af23db3d8899bc2bc1d",
    "scripts.tar.gz": "2b213f7aeab7d78f7bf5552b9d67781f7b6a946c942d44ec9f1a26a6a166d89b",
    "bids_code-evaluations_uniba.tar.gz": "1a13819e0e12c0225668a09268ca9fc13b19ae357ff17320506472e1f7d4442b",
    "bids_code-submissions_itu.tar.gz": "13b79cd222f6e03f3feee27e038786b7cb5168fd37b249f14b4884fe8ba04fd6",
    "bids_behavioral_uniba.tar.gz": "fcdeeb7a2ee46b7f8b0a31459c919fc5b4050133ff699c76b7e722da87d6e848",
    "bids_behavioral_itu.tar.gz": "fb501e02f74120f665ce860e004bf214bd17381068e497e4986c2a56923167c9",
    "bids_metadata_uniba.tar.gz": "26dc99e988661b902d5f2a73d1b314ee00c976fd5a21579e434678f7524cf9d9",
    "bids_metadata_itu.tar.gz": "490f7f2fdbb5d4e57b7243fdbc42bbad3175bbde73b5625f5b991f772ba349fa",
    "README.md": "5f9a76bc6fe4866e7280f142a1a6d4226a7e1dd65e26ab83ebd2d41a26446e07",
}

SELECTED = [
    "resources/[eng] experiment-app/exp-server/run-tests.sh",
    "resources/[eng] experiment-app/exp-server/setup-exp.sh",
    "resources/[eng] experiment-app/exp-server/server.py",
    "resources/[eng] experiment-app/docker-compose.yml",
    "resources/[eng] experiment-app/otree/experiment/templates/TaskDescription1.html",
    "resources/[eng] experiment-app/otree/experiment/templates/TaskDescription2.html",
    "resources/[eng] experiment-app/otree/experiment/templates/Task.html",
    "resources/[eng] experiment-app/java-src/Experiment1/src/task/ConnectionAPI.java",
    "resources/[eng] experiment-app/java-src/Experiment1/src/task/DeviceAPI.java",
    "resources/tasks/Connection/[eng]/img/UML-diagrams.drawio",
    "resources/rubric/code_evaluation_template.xlsx",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(archive: tarfile.TarFile) -> dict:
    names = [member.name for member in archive if member.isfile()]
    test_sources = [name for name in names if name.endswith(".java") and
                    ("/tests/" in name.lower() or "/test/" in name.lower()
                     or Path(name).stem.lower().endswith("test"))]
    manifests = [name for name in names if Path(name).name in
                 {"pom.xml", "build.gradle", "build.gradle.kts", "gradlew"}]
    return {"regular_file_count": len(names), "java_test_source_count": len(test_sources),
            "java_build_manifest_count": len(manifests),
            "extensions": dict(sorted(Counter(Path(name).suffix for name in names).items()))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-root", type=Path, required=True)
    p.add_argument("--controller-state", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite prior source audit")
    state = json.loads(args.controller_state.read_text())
    receipts = state["raw_preservation_receipt"]["files"]
    hashes = {}
    for name, expected in PINNED.items():
        path = args.archive_root / name
        actual = digest(path)
        matches = [v for k, v in receipts.items() if k.endswith("/" + name)]
        if actual != expected or len(matches) != 1 or matches[0]["sha256"] != actual:
            raise ValueError("Archive or raw-preservation hash mismatch: " + name)
        if matches[0]["size"] != path.stat().st_size:
            raise ValueError("Archive size mismatch: " + name)
        hashes[name] = {"sha256": actual, "bytes": path.stat().st_size,
                        "raw_preservation_receipt_match": True}
    result = {
        "schema": "xavier.coding_harness_source_audit.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_release": "Zenodo 20801101",
        "source_url": "https://zenodo.org/records/20801101",
        "archives": hashes, "selected_public_members": {}, "nested_apps": {},
        "participant_solution_contents_read": 0, "prompt_or_key_contents_read": 0,
        "archive_members_extracted": 0, "publisher_scripts_executed": 0,
    }
    with tarfile.open(args.archive_root / "resources.tar.gz") as archive:
        result["resources_inventory"] = inventory(archive)
        for name in SELECTED:
            b = archive.extractfile(name).read()
            result["selected_public_members"][name] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
        for member in archive.getmembers():
            if member.isfile() and member.name.endswith("experiment-app.tar.gz"):
                b = archive.extractfile(member).read()
                with tarfile.open(fileobj=io.BytesIO(b)) as nested:
                    result["nested_apps"][member.name] = {"sha256": hashlib.sha256(b).hexdigest(), **inventory(nested)}
        drawio = archive.extractfile("resources/tasks/Connection/[eng]/img/UML-diagrams.drawio").read()
        result["connection_public_diagram_pages"] = [d.get("name") for d in ET.fromstring(drawio).findall("diagram")]
    result["code_archive_inventory"] = {}
    for name in ("bids_code-evaluations_uniba.tar.gz", "bids_code-submissions_itu.tar.gz"):
        with tarfile.open(args.archive_root / name) as archive:
            result["code_archive_inventory"][name] = inventory(archive)
    result["behavioral_schemas"] = {}
    for name in ("bids_behavioral_uniba.tar.gz", "bids_behavioral_itu.tar.gz"):
        schemas = defaultdict(set)
        counts = Counter()
        with tarfile.open(args.archive_root / name) as archive:
            for member in archive:
                match = re.search(r"_acq-([^_]+)_", member.name)
                if member.isfile() and member.name.endswith(".tsv") and match:
                    header = archive.extractfile(member).readline().decode("utf-8-sig").strip().split("\t")
                    schemas[match.group(1)].add(tuple(header))
                    counts[match.group(1)] += 1
        result["behavioral_schemas"][name] = {key: {"tsv_files": counts[key],
            "header_variants": [list(columns) for columns in sorted(values)]}
            for key, values in sorted(schemas.items())}
    with tarfile.open(args.archive_root / "scripts.tar.gz") as archive:
        result["script_members"] = [m.name for m in archive if m.isfile()]
    result["findings"] = {
        "public_task_instructions_and_api_stubs": "present",
        "publisher_automatic_test_implementation": "missing_from_inspected_release_archives",
        "publisher_test_shell_hooks": "present_but_referenced_tests_Main_and_tests_java_absent",
        "recorded_code_quality_scores": "manual_rubric_not_an_executable_oracle",
        "structured_predecision_editor_state": "not_present_in_inspected_behavioral_schemas",
        "final_code_as_predecision_input": "forbidden_future_information",
        "historical_agent_counterfactual_outcomes": "unavailable",
        "new_independent_specification_tests": "feasible_for_engineering_plumbing_only",
        "empirical_eeg_linked_task_completion": "blocked",
        "latent_intent_ground_truth": "unavailable_observed_events_are_not_intent",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "completed", "output": str(args.output), "sha256": digest(args.output),
                      "resources": result["resources_inventory"], "nested_apps": result["nested_apps"],
                      "findings": result["findings"]}))


if __name__ == "__main__":
    main()
