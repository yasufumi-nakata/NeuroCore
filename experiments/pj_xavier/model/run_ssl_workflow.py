#!/usr/bin/env python3
"""Sequential bounded Colab workflow, invoked only through workstation CLI.

Scientific sources/protocols are separately sealed. This orchestrator never
repeats a prior output directory and keeps canonical evaluation receipts pending
until independent recovery, strict scoring, and external-site gates are ready.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def utc():
    return datetime.now(timezone.utc).isoformat()


class Workflow:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.root).resolve()
        self.directory = self.root / "workflows/cbramod_ssl_v1"
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "execution_started.json").open("x") as file:
            json.dump({"started_at": utc(), "supervisor_pid": os.getpid(), "arguments": vars(args),
                       "original_deadline_epoch": args.deadline_epoch, "deadline_extended": False}, file)
        self.protocol_path = Path(args.protocol).resolve()
        self.protocol = json.loads(self.protocol_path.read_text())
        self.plan_path = Path(args.stage_plan).resolve()
        self.plan = json.loads(self.plan_path.read_text())
        self.model = self.root / "code/model"
        self.corpus = self.root / "prepared/ssl_portable_uniba_v1"
        self.supervised = self.root / "prepared/ai_coding_uniba_v3_natural_casecontrol_verified/manifest.jsonl"
        self.upstream = self.root / "upstream/CBraMod"
        self.initial = self.root / "upstream/cbramod_pretrained_weights.pth"
        self.ssl_run = self.root / "runs/cbramod_ssl_portable_uniba_v1"
        self.align_run = self.root / "runs/cbramod_ssl_smollm2_uniba_v3_symbol_v1"
        self.env = os.environ.copy()
        if self.env.get("CUDA_VISIBLE_DEVICES") == "":
            self.env.pop("CUDA_VISIBLE_DEVICES")
        for key in ("XAVIER_CBRAMOD_UPSTREAM", "XAVIER_CBRAMOD_WEIGHTS", "XAVIER_CBRAMOD_SSL_WEIGHTS"):
            self.env.pop(key, None)
        self.env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                        MKL_NUM_THREADS="2", PYTORCH_ALLOC_CONF="expandable_segments:True")

    def remaining(self):
        return self.args.deadline_epoch - time.time()

    def stage(self, name, command, *, maximum_seconds, allow_failure=False, cpu_only=False):
        if (self.directory / (name + ".launch.json")).exists():
            raise ValueError("A recorded stage must never be silently repeated: " + name)
        remaining = self.remaining()
        if remaining < 15:
            raise TimeoutError("Preserved session deadline has arrived")
        env = self.env.copy()
        if cpu_only:
            env["CUDA_VISIBLE_DEVICES"] = ""
        started = time.time()
        log = self.directory / (name + ".log")
        with log.open("xb") as handle:
            process = subprocess.Popen(command, cwd=self.root, stdout=handle, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            launch = {"stage": name, "pid": process.pid, "command": command, "started_at": utc(),
                      "started_epoch": started, "maximum_seconds": min(maximum_seconds, remaining),
                      "session_deadline_epoch": self.args.deadline_epoch, "cpu_only": cpu_only,
                      "allocator": self.env["PYTORCH_ALLOC_CONF"]}
            write(self.directory / (name + ".launch.json"), launch)
            write(self.directory / "current_stage.json", launch)
            timed_out = False
            try:
                code = process.wait(timeout=min(maximum_seconds, remaining))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait(timeout=10)
        receipt = {**launch, "exit_code": code, "timed_out": timed_out, "completed_at": utc(),
                   "process_wall_seconds": time.time() - started, "log_sha256": digest(log)}
        write(self.directory / (name + ".exit.json"), receipt)
        if code != 0 and not allow_failure:
            raise RuntimeError(name + " failed; preserve logs and do not rerun prior stages")
        return receipt

    def preflight(self):
        if digest(self.plan_path) != self.protocol["stage_plan_sha256"]:
            raise ValueError("Stage plan bytes changed")
        for name, expected in self.protocol["implementation_sha256"].items():
            if digest(self.model / name) != expected:
                raise ValueError("Sealed implementation changed: " + name)
        if digest(self.args.benchmark_runner) != "bd92732e088ff633d2ff299161cf47a8a5630435c4c425dbae7992f563bc0447":
            raise ValueError("Frozen five-condition evaluator changed")
        if not Path(self.args.gate_evidence).is_file():
            raise ValueError("Recorded live controller gate evidence is missing")
        # The caller checked the live controller. This local receipt binds the
        # exact evidence rather than pretending a monitoring gate grants access.
        if self.remaining() < 4200:
            raise TimeoutError("Less than the predeclared 70-minute admission reserve remains before the session deadline")
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if processes:
            raise RuntimeError("Another GPU process is active; no concurrent ML stage is admitted")
        write(self.directory / "preflight.json", {"checked_at": utc(), "GPU_processes": [],
              "protocol_sha256": digest(self.protocol_path), "stage_plan_sha256": digest(self.plan_path),
              "gate_evidence_sha256": digest(self.args.gate_evidence), "remaining_session_seconds": self.remaining(),
              "authorization_source": "existing user and root research task; monitoring gate is not authorization"})

    def ssl_command(self, command, output, batch):
        return [sys.executable, str(self.model / "cbramod_ssl.py"), command,
                "--manifest", str(self.corpus / "manifest.jsonl"), "--artifacts", str(self.corpus / "artifacts.json"),
                "--source-allowlist", str(self.corpus / "source_allowlist.json"),
                "--corpus-summary", str(self.corpus / "corpus_summary.json"),
                "--protocol", str(self.protocol_path), "--output", str(output),
                "--cbramod-upstream", str(self.upstream), "--cbramod-weights", str(self.initial),
                "--device", "cuda", "--batch-size", str(batch), "--epochs", "10", "--max-seconds", "1800",
                "--lr", ".0001", "--seed", "20261122", "--precision", "bfloat16", "--memory-budget-gib", "21",
                "--cpu-threads", "2", "--probe-steps", "50"]

    def alignment_protocol(self, seal_path):
        old_path = self.model / "cbramod_symbol_protocol.json"
        value = json.loads(old_path.read_text())
        value.update(schema="xavier-cbramod-ssl-symbol-experiment-v1", run_name=self.align_run.name,
                     registered_at_utc=utc(), model_source_sha256=digest(self.model / "cbramod_ssl_token_llm.py"),
                     cpu_test_source_sha256=digest(self.model / "test_cbramod_ssl_token_llm.py"),
                     ssl_encoder_seal_sha256=digest(seal_path), stage_plan_sha256=digest(self.plan_path),
                     parent_alignment_protocol_sha256=digest(old_path),
                     question="Does the predeclared additional unlabeled EEG continuation change the same fixed observed-action decoder?")
        value["interpretation"].update(simultaneous_changes=["Frozen EEG encoder weights after registered additional masked EEG SSL"],
                                       matched_alignment_hyperparameters=True, original_v3_test_reused=True,
                                       ITU_used_for_training_or_selection=False)
        value["interpretation"].pop("ITU_used", None)
        selected = json.loads(Path(seal_path).read_text())
        value["eeg_encoder"].update(weights_repo=None, weights_revision=None, weights_sha256=selected["weights_sha256"],
                                     ssl_encoder_seal=selected, initial_weights_sha256=self.plan["model"]["initial_weights_sha256"])
        value["capacity_selection"].update(candidate_batches=[32], maximum_probe_seconds=120,
                                            rule="The same fixed batch32 must pass its sustained50-step train-only check; no batch tuning")
        value["training"]["batch_size"] = 32
        value["evaluation"].update(gate_id="cbramod_ssl_smollm2_uniba_v3_symbol_evaluation",
                                    receipt=str(self.align_run.relative_to(self.root) / "variant_evaluation_completion.json"),
                                    checkpoint=str(self.align_run.relative_to(self.root) / "best.pt"))
        value["execution_preconditions"] = ["Live new alignment gate and original session deadline retained",
             "SSL raw-source exclusions and actual corpus arrays independently verified",
             "Selected SSL weights are sealed from reconstruction validation only",
             "No overlapping GPU workflow", "Original baselines and held-out protocols unchanged"]
        path = self.directory / "alignment_protocol.json"
        if path.exists():
            raise ValueError("Alignment protocol already registered")
        write(path, value)
        return path

    def align_command(self, command, output, protocol_path, seal_path):
        return [sys.executable, str(self.model / "cbramod_ssl_token_llm.py"), command,
                "--manifest", str(self.supervised), "--output", str(output), "--protocol", str(protocol_path),
                "--ssl-encoder-seal", str(seal_path), "--cbramod-upstream", str(self.upstream),
                "--cbramod-weights", str(self.ssl_run / "best_encoder_weights.pth"),
                "--device", "cuda", "--dtype", "bfloat16", "--batch-size", "32", "--epochs", "30",
                "--max-seconds", "900" if command == "train" else "120", "--memory-budget-gib", "21",
                "--lr", ".001", "--seed", "20261122", "--max-prompt-tokens", "128", "--abstain-threshold", ".8",
                "--cpu-threads", "2", "--probe-steps", "50"]

    def run(self):
        self.preflight()
        reports = []
        for batch in self.protocol["capacity"]["candidate_batches"]:
            target = self.root / f"probes/cbramod_ssl_v1_b{batch}"
            receipt = self.stage(f"ssl_probe_b{batch}", self.ssl_command("probe", target, batch), maximum_seconds=600, allow_failure=True)
            report = target / "run_summary.json"
            if not report.exists():
                report = self.directory / f"failed_probe_b{batch}.report.json"
                write(report, {"status": "failed_train_only_probe", "batch_size": batch,
                       "protocol_sha256": digest(self.protocol_path), "model_source_sha256": self.protocol["model_source_sha256"],
                       "failure_exit_receipt": receipt, "memory": {}, "probe": {"completed_steps": False}})
            reports.append(str(report))
        selection = self.directory / "ssl_capacity_selection.json"
        self.stage("ssl_capacity_selection", [sys.executable, str(self.model / "cbramod_ssl.py"), "select-capacity",
                   "--protocol", str(self.protocol_path), "--reports", *reports, "--output", str(selection)], maximum_seconds=120, cpu_only=True)
        batch = json.loads(selection.read_text())["selected_batch"]
        if self.remaining() < 3600:
            raise TimeoutError("Insufficient session time for the registered main SSL and following fixed alignment/evaluation")
        self.stage("ssl_training", self.ssl_command("train", self.ssl_run, batch) + ["--capacity-selection", str(selection)], maximum_seconds=2000)
        seal = self.ssl_run / "selected_encoder_seal.json"
        self.stage("ssl_encoder_seal", [sys.executable, str(self.model / "cbramod_ssl.py"), "seal-encoder",
                   "--ssl-run", str(self.ssl_run), "--output", str(seal)], maximum_seconds=120, cpu_only=True)
        alignment_protocol = self.alignment_protocol(seal)
        probe = self.root / "probes/cbramod_ssl_alignment_v1_b32"
        self.stage("alignment_probe_b32", self.align_command("probe", probe, alignment_protocol, seal), maximum_seconds=180)
        alignment_selection = self.directory / "alignment_capacity_selection.json"
        self.stage("alignment_capacity_selection", [sys.executable, str(self.model / "cbramod_ssl_token_llm.py"), "select-capacity",
                   "--protocol", str(alignment_protocol), "--reports", str(probe / "run_summary.json"),
                   "--output", str(alignment_selection)], maximum_seconds=120, cpu_only=True)
        self.stage("alignment_training", self.align_command("train", self.align_run, alignment_protocol, seal) +
                   ["--capacity-selection", str(alignment_selection)], maximum_seconds=1050)
        evaluation = self.root / "evaluations/cbramod_ssl_smollm2_uniba_v3_fiveconditions_v1"
        self.stage("five_conditions", [sys.executable, str(self.args.benchmark_runner),
                   "--model-module", str(self.model / "cbramod_ssl_token_llm.py"), "--manifest", str(self.supervised),
                   "--checkpoint", str(self.align_run / "best.pt"), "--raw-bids-root", str(self.root / "verified_ai_coding_uniba_v2"),
                   "--artifact-root", str(self.root), "--output-dir", str(evaluation),
                   "--completion-receipt", str(self.align_run / "variant_evaluation_completion.pending.json"),
                   "--expected-manifest-sha256", self.plan["alignment"]["manifest_sha256"],
                   "--expected-model-sha256", digest(self.model / "cbramod_ssl_token_llm.py"),
                   "--expected-test-records", "203", "--device", "cuda", "--memory-budget-gib", "21"], maximum_seconds=600)
        write(self.directory / "completion.json", {"status": "SSL_alignment_and_five_conditions_completed_pending_strict_recovery_external",
              "completed_at": utc(), "ssl_encoder_seal_sha256": digest(seal),
              "alignment_checkpoint_sha256": digest(self.align_run / "best.pt"),
              "alignment_protocol_sha256": digest(alignment_protocol),
              "evaluation_report_sha256": digest(evaluation / "variant_evaluation_report.json"),
              "canonical_published": False, "remaining_session_seconds": self.remaining(), "os_actions_executed": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "protocol", "stage-plan", "benchmark-runner", "gate-evidence"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    args = parser.parse_args()
    workflow = Workflow(args)
    try:
        workflow.run()
    except Exception as exc:
        write(workflow.directory / "failure.json", {"status": "stopped_preserving_prior_stages", "at": utc(),
              "type": type(exc).__name__, "message": str(exc), "remaining_session_seconds": workflow.remaining(),
              "automatic_repeat": False})
        raise


if __name__ == "__main__":
    main()
