"""Verify provenance-only dictionary reuse with local fake teacher outputs."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import ollama_label_worker as worker
import propagate_weak_labels as propagation
from test_ollama_label_worker import fake_ollama, config, fixture_job


def dump_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_fixture(directory, endpoint):
    root = Path(directory)
    source, jobs = [], []
    for i in range(6):
        partition = "train" if i < 4 else "val" if i == 4 else "test"
        row = {"id": f"r{i}", "dataset": "fixture", "subject": "s0", "session": "one", "split": partition,
               "label": "keyboard", "label_source": "recorded_event", "target_kind": "observed_action", "event_time_s": float(i),
               "original_event_description": "Keyboard key " + ("A" if i % 2 == 0 else "B") + " pressed",
               "label_evidence_uri": "fixture://events", "eeg_set_sha256": "b" * 64}
        source.append(row)
        if partition == "train":
            job = fixture_job(f"j{i}")
            job["record_id"] = row["id"]
            job["evidence"][0].update(id=f"event-{i}", text=row["original_event_description"], timestamp_s=float(i))
            job["original_gold"] = [{"value": "keyboard", "origin": "recorded_event"}]
            jobs.append(job)
    dump_rows(root / "source.jsonl", source)
    checksum = worker.file_digest(root / "source.jsonl")
    for job in jobs:
        job["source_manifest_sha256"] = checksum
    dump_rows(root / "jobs.jsonl", jobs)
    dump_rows(root / "dictionary.jsonl", jobs[:2])
    membership = {"manifest_sha256": checksum, "training_ids": ["r0", "r1", "r2", "r3"], "validation_ids": ["r4"], "test_ids": ["r5"],
                  "training_ids_sha256": hashlib.sha256(b"r0\nr1\nr2\nr3").hexdigest()}
    (root / "membership.json").write_text(json.dumps(membership))
    cfg = config(endpoint, source_records_path=str(root / "source.jsonl"))
    worker.run_worker(root / "dictionary.jsonl", root / "results", cfg)
    return root


def run(root):
    return propagation.build_propagation(root / "source.jsonl", root / "membership.json", root / "jobs.jsonl", root / "dictionary.jsonl",
                                        root / "results/shard-000-of-001.jsonl", root / "results/shard-000-of-001.manifest.json")


class PropagationTests(unittest.TestCase):
    def test_exact_dictionary_reuse_preserves_gold_and_all_original_citations(self):
        with tempfile.TemporaryDirectory() as directory, fake_ollama() as (endpoint, calls):
            root = make_fixture(directory, endpoint)
            before = {p.name: worker.file_digest(p) for p in root.glob("*.json*")}
            summary, records, mapping = run(root)
            self.assertEqual(summary["independent_teacher_jobs"], 2)
            self.assertEqual(summary["training_record_references"], 4)
            self.assertEqual(sorted(rid for entry in mapping for rid in entry["source_record_ids"]), ["r0", "r1", "r2", "r3"])
            target = next(row for row in records if row["record_id"] == "r2")
            self.assertEqual(target["pseudo_labels"][0]["evidence"][0]["evidence_id"], "event-0")
            self.assertEqual(target["target_event_evidence"][0]["id"], "event-2")
            self.assertFalse(target["propagation"]["teacher_inference_performed_for_this_record"])
            self.assertFalse(target["ground_truth"] or target["gold_labels_modified"])
            self.assertEqual(summary["new_waveform_records_created"], 0)
            self.assertEqual(len(calls["chat"]), 2)
            self.assertEqual(before, {p.name: worker.file_digest(p) for p in root.glob("*.json*")})

    def test_heldout_record_cannot_be_introduced_as_a_propagation_target(self):
        with tempfile.TemporaryDirectory() as directory, fake_ollama() as (endpoint, calls):
            root = make_fixture(directory, endpoint)
            jobs = propagation.rows(root / "jobs.jsonl")
            jobs[2]["record_id"] = "r5"
            dump_rows(root / "jobs.jsonl", jobs)
            with self.assertRaisesRegex(ValueError, "exactly equal"):
                run(root)

    def test_changing_frozen_subject_or_session_cannot_preserve_membership(self):
        with tempfile.TemporaryDirectory() as directory, fake_ollama() as (endpoint, calls):
            root = make_fixture(directory, endpoint)
            source = propagation.rows(root / "source.jsonl")
            source[0]["session"] = "changed"
            dump_rows(root / "source.jsonl", source)
            with self.assertRaisesRegex(ValueError, "original frozen manifest"):
                run(root)

    def test_changed_event_text_is_not_invented_into_an_exact_dictionary_match(self):
        with tempfile.TemporaryDirectory() as directory, fake_ollama() as (endpoint, calls):
            root = make_fixture(directory, endpoint)
            jobs = propagation.rows(root / "jobs.jsonl")
            jobs[2]["evidence"][0]["text"] = "Unobserved high-level goal"
            dump_rows(root / "jobs.jsonl", jobs)
            with self.assertRaisesRegex(ValueError, "Event text differs"):
                run(root)

    def test_tampered_teacher_citation_fails_even_after_outer_file_rehash(self):
        with tempfile.TemporaryDirectory() as directory, fake_ollama() as (endpoint, calls):
            root = make_fixture(directory, endpoint)
            path = root / "results/shard-000-of-001.jsonl"
            results = propagation.rows(path)
            results[0]["pseudo_labels"][0]["evidence"][0]["evidence_id"] = "invented"
            dump_rows(path, results)
            manifest_path = root / "results/shard-000-of-001.manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["results_file_sha256"] = worker.file_digest(path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "evidence validation"):
                run(root)


if __name__ == "__main__":
    unittest.main()
