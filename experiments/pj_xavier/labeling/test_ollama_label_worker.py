"""Fake loopback HTTP only. Fixture labels are never reported as EEG findings."""
import copy
import hashlib
import json
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import ollama_label_worker as worker


def fixture_job(identifier="job1"):
    return {"job_id": identifier, "record_id": identifier + "/record", "dataset_id": "fixture", "split": "train", "source_manifest_sha256": "a" * 64,
            "source_assets": [{"uri": "fixture://events", "sha256": "b" * 64}],
            "evidence": [{"id": "event1", "kind": "recorded_event", "text": "Keyboard key A pressed", "source_uri": "fixture://events#1", "timestamp_s": 1.}],
            "allowed_values": {"observed_action": ["keyboard", "mouse_click"]},
            "original_gold": [{"value": "source-label", "origin": "original_annotation"}]}


def valid_answer(request):
    prompt = json.loads(request["messages"][-1]["content"])
    evidence = prompt["evidence"][0]
    return {"status": "labeled", "labels": [{"namespace": "observed_action", "value": "keyboard", "evidence": [{"evidence_id": evidence["id"], "quote": evidence["text"]}]}], "abstain_reason": "none"}


@contextmanager
def fake_ollama(answer=valid_answer):
    requests = {"tags": [], "chat": []}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            requests["tags"].append(self.path)
            self.reply({"models": [{"name": "fixture:1b", "digest": "c" * 64, "details": {"parameter_size": "1B", "quantization_level": "Q4"}}]})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests["chat"].append(request)
            content = answer(request) if callable(answer) else answer
            self.reply({"model": "fixture:1b", "done": True, "done_reason": "stop", "message": {"role": "assistant", "content": json.dumps(content)}, "prompt_eval_count": 100, "eval_count": 20})
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def write_jobs(root, jobs):
    source = Path(root) / "source_records.jsonl"
    source_rows = [{"id": job["record_id"], "dataset": job["dataset_id"], "split": job["split"]} for job in jobs]
    source.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    source_hash = worker.file_digest(source)
    for job in jobs:
        job["source_manifest_sha256"] = source_hash
    path = Path(root) / "jobs.jsonl"
    path.write_text("".join(json.dumps(job) + "\n" for job in jobs))
    return path


def config(endpoint, **kwargs):
    return worker.WorkerConfig(model="fixture:1b", endpoint=endpoint, min_free_bytes=0, **kwargs)


def run_fixture_worker(path, output, configuration):
    configuration.source_records_path = str(path.parent / "source_records.jsonl")
    return worker.run_worker(path, output, configuration)


class WorkerTests(unittest.TestCase):
    def test_train_declaration_cannot_relabel_an_actual_heldout_record(self):
        with tempfile.TemporaryDirectory() as directory, fake_ollama() as (endpoint, calls):
            job = fixture_job()
            job["split"] = "test"
            path = write_jobs(directory, [job])
            job["split"] = "train"
            path.write_text(json.dumps(job) + "\n")
            result = run_fixture_worker(path, Path(directory) / "output", config(endpoint))
            self.assertEqual(result["counts_this_run"]["rejected"], 1)
            self.assertFalse(calls["tags"] or calls["chat"])

    def test_changed_source_subject_or_session_invalidates_frozen_bytes_hash(self):
        with tempfile.TemporaryDirectory() as directory, fake_ollama() as (endpoint, calls):
            path = write_jobs(directory, [fixture_job()])
            source = path.parent / "source_records.jsonl"
            row = json.loads(source.read_text())
            row.update(subject="different-subject", session="different-session")
            source.write_text(json.dumps(row) + "\n")
            result = run_fixture_worker(path, Path(directory) / "output", config(endpoint))
            output = worker.read_results(Path(result["output_uri"]))[0]
            self.assertIn("frozen_source_manifest_hash_mismatch", output["errors"])
            self.assertFalse(calls["chat"])

    def test_representative_cannot_be_hidden_by_a_safe_source_id_list(self):
        job = fixture_job()
        job["source_record_ids"] = ["safe-train"]
        index = {"safe-train": {"split": "train", "dataset": "fixture"}, job["record_id"]: {"split": "test", "dataset": "fixture"}}
        self.assertTrue(worker.check_job_membership(job, job["source_manifest_sha256"], index))
        job["source_record_ids"].append(job["record_id"])
        self.assertTrue(any("held out" in e for e in worker.check_job_membership(job, job["source_manifest_sha256"], index)))

    def test_original_gold_is_not_in_teacher_prompt(self):
        job = fixture_job()
        job["original_gold"] = [{"value": "unique-gold-sentinel", "origin": "original_annotation"}]
        self.assertNotIn("unique-gold-sentinel", json.dumps(worker.messages_for(job)))

    def test_valid_weak_label_preserves_source_gold_and_bounds_request(self):
        with tempfile.TemporaryDirectory() as directory, fake_ollama() as (endpoint, calls):
            job = fixture_job()
            path = write_jobs(directory, [job])
            original_hash = worker.file_digest(path)
            summary = run_fixture_worker(path, Path(directory) / "output", config(endpoint))
            result = worker.read_results(Path(summary["output_uri"]))[0]
            self.assertEqual(result["status"], "labeled")
            self.assertFalse(result["ground_truth"])
            self.assertEqual(result["label_origin"], "ai_pseudo")
            self.assertEqual(result["original_gold_snapshot"], job["original_gold"])
            self.assertFalse(result["gold_labels_modified"])
            self.assertEqual(original_hash, worker.file_digest(path))
            self.assertEqual(summary["counts_this_run"]["pseudo_labels_created"], 1)
            request = calls["chat"][0]
            self.assertFalse(request["stream"])
            self.assertEqual(request["format"], worker.OUTPUT_SCHEMA)
            self.assertEqual(request["options"]["num_predict"], 768)
            self.assertEqual(request["keep_alive"], "30s")
            self.assertEqual(result["teacher"]["model_digest_sha256"], "c" * 64)

    def test_all_heldout_partitions_rejected_without_http(self):
        with tempfile.TemporaryDirectory() as directory, fake_ollama() as (endpoint, calls):
            jobs = []
            for partition in ("val", "validation", "test", "holdout"):
                job = fixture_job(partition)
                job["split"] = partition
                jobs.append(job)
            summary = run_fixture_worker(write_jobs(directory, jobs), Path(directory) / "output", config(endpoint))
            self.assertEqual(summary["counts_this_run"]["rejected"], 4)
            self.assertEqual(calls, {"tags": [], "chat": []})

    def test_fabricated_evidence_id_fails_and_produces_no_pseudo_label(self):
        answer = valid_answer({"messages": [{"content": json.dumps({"evidence": fixture_job()["evidence"]})}]})
        answer["labels"][0]["evidence"][0]["evidence_id"] = "invented"
        with tempfile.TemporaryDirectory() as directory, fake_ollama(answer) as (endpoint, calls):
            summary = run_fixture_worker(write_jobs(directory, [fixture_job()]), Path(directory) / "output", config(endpoint))
            result = worker.read_results(Path(summary["output_uri"]))[0]
            self.assertEqual(result["status"], "failed")
            self.assertIn("fabricated_or_unknown_evidence_id", result["errors"])
            self.assertEqual(result["pseudo_labels"], [])

    def test_fabricated_quote_and_new_goal_fields_fail_closed_schema(self):
        job = fixture_job()
        answer = valid_answer({"messages": [{"content": json.dumps({"evidence": job["evidence"]})}]})
        answer["labels"][0]["evidence"][0]["quote"] = "Participant wants to send a message"
        self.assertIn("fabricated_or_nonverbatim_evidence_quote", worker.validate_output(answer, job))
        answer["goal"] = "send message"
        self.assertTrue(worker.validate_output(answer, job))

    def test_task_protocol_cannot_be_promoted_to_observed_action(self):
        job = fixture_job()
        job["evidence"][0]["kind"] = "task_protocol"
        job["evidence"][0]["text"] = "Please press the A key"
        answer = valid_answer({"messages": [{"content": json.dumps({"evidence": job["evidence"]})}]})
        self.assertIn("evidence kind cannot support requested label namespace", worker.validate_output(answer, job))

    def test_raw_eeg_and_future_context_never_reach_teacher(self):
        jobs = [fixture_job("raw"), fixture_job("future")]
        jobs[0]["evidence"][0].update(kind="raw_eeg", text=[1., 2., 3.])
        jobs[1]["evidence"][0].update(kind="predecision_context", timestamp_s=11.)
        jobs[1]["decision_time_s"] = 10.
        with tempfile.TemporaryDirectory() as directory, fake_ollama() as (endpoint, calls):
            summary = run_fixture_worker(write_jobs(directory, jobs), Path(directory) / "output", config(endpoint))
            self.assertEqual(summary["counts_this_run"]["rejected"], 2)
            self.assertEqual(calls["chat"], [])

    def test_resume_skips_same_job_source_but_changed_source_is_reprocessed(self):
        with tempfile.TemporaryDirectory() as directory, fake_ollama() as (endpoint, calls):
            job = fixture_job()
            path = write_jobs(directory, [job])
            output = Path(directory) / "output"
            run_fixture_worker(path, output, config(endpoint))
            summary = run_fixture_worker(path, output, config(endpoint))
            self.assertEqual(summary["counts_this_run"]["resume_skipped"], 1)
            self.assertEqual(len(calls["chat"]), 1)
            job["evidence"][0]["text"] = "Keyboard key B pressed"
            write_jobs(directory, [job])
            run_fixture_worker(path, output, config(endpoint))
            self.assertEqual(len(calls["chat"]), 2)

    def test_shards_are_disjoint_and_cover_every_job_once(self):
        with tempfile.TemporaryDirectory() as directory, fake_ollama() as (endpoint, calls):
            jobs = [fixture_job(f"job-{index}") for index in range(25)]
            path = write_jobs(directory, jobs)
            all_ids = []
            for shard in range(4):
                summary = run_fixture_worker(path, Path(directory) / "output", config(endpoint, shard_index=shard, num_shards=4))
                results = worker.read_results(Path(summary["output_uri"]))
                all_ids.extend(result["job_id"] for result in results)
            self.assertEqual(sorted(all_ids), sorted(job["job_id"] for job in jobs))
            self.assertEqual(len(all_ids), len(set(all_ids)))
            self.assertEqual(len(calls["chat"]), 25)

    def test_unknown_is_persisted_without_fabricated_label(self):
        answer = {"status": "unknown", "labels": [], "abstain_reason": "insufficient_evidence"}
        with tempfile.TemporaryDirectory() as directory, fake_ollama(answer) as (endpoint, calls):
            path = write_jobs(directory, [fixture_job()])
            output = Path(directory) / "output"
            summary = run_fixture_worker(path, output, config(endpoint))
            result = worker.read_results(Path(summary["output_uri"]))[0]
            self.assertEqual(result["status"], "unknown")
            self.assertEqual(result["pseudo_labels"], [])
            run_fixture_worker(path, output, config(endpoint))
            self.assertEqual(len(calls["chat"]), 1)

    def test_failure_attempts_are_bounded_and_maxjobs_resumes(self):
        bad = {"status": "labeled", "labels": [], "abstain_reason": "none"}
        with tempfile.TemporaryDirectory() as directory, fake_ollama(bad) as (endpoint, calls):
            path = write_jobs(directory, [fixture_job()])
            output = Path(directory) / "output"
            for _ in range(3):
                summary = run_fixture_worker(path, output, config(endpoint, max_attempts=2))
            self.assertEqual(len(calls["chat"]), 2)
            self.assertEqual(summary["counts_this_run"]["attempt_limit_skipped"], 1)
        with tempfile.TemporaryDirectory() as directory, fake_ollama() as (endpoint, calls):
            path = write_jobs(directory, [fixture_job("j1"), fixture_job("j2")])
            output = Path(directory) / "output"
            first = run_fixture_worker(path, output, config(endpoint, max_jobs=1))
            second = run_fixture_worker(path, output, config(endpoint, max_jobs=1))
            self.assertEqual(first["status"], "max_jobs_reached")
            self.assertEqual(second["counts_this_run"]["labeled"], 1)
            self.assertEqual(len(calls["chat"]), 2)

    def test_disk_guard_prevents_inference_and_external_endpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, fake_ollama() as (endpoint, calls):
            with patch.object(worker.shutil, "disk_usage", return_value=type("Usage", (), {"free": 0})()):
                cfg = config(endpoint)
                cfg.min_free_bytes = 1
                result = run_fixture_worker(write_jobs(directory, [fixture_job()]), Path(directory) / "output", cfg)
            self.assertEqual(result["status"], "blocked_disk_reserve")
            self.assertEqual(calls["chat"], [])
        for endpoint in ("http://example.com:11434", "http://localhost.evil.example:11434", "http://user:pass@127.0.0.1:11434"):
            with self.assertRaises(ValueError):
                worker.LocalOllama(endpoint)


if __name__ == "__main__":
    unittest.main()
