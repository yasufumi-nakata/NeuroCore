#!/usr/bin/env python3
"""Workstation controller for the existing PJ-Xavier Colab session.

Uses only the installed Colab CLI. It never purchases units, creates/replaces a
runtime, restarts a completed training run, or extends original job deadlines.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time
import zipfile

from remote_core import db_status, digest_file, json_save, raw_chunk_key, safe_child, RAW_CHUNK_LIMIT

SESSION = "xavier-eeg-20261122"
CLI = "/Users/yasufumi/.local/bin/colab"
MARKER = "XAVIER_CONTROLLER_RESULT_JSON:"


class ColabCLI:
    def __init__(self, config, *, runner=subprocess.run):
        if config.get("cli") != CLI or config.get("session") != SESSION or config.get("remote_root") != "/content/xavier":
            raise ValueError("CLI and session must match the explicitly owned workstation Colab environment")
        self.config, self.runner = config, runner
        self.remote_source = Path(__file__).with_name("remote_core.py").read_text()

    def call(self, args, *, timeout=60):
        if args[0] not in {"exec", "download", "usage", "status", "stop"}:
            raise ValueError("command outside controller CLI allowlist")
        if args[0] != "usage":
            if (args.count("--session") != 1 or args.index("--session") + 1 >= len(args)
                    or args[args.index("--session") + 1] != SESSION):
                raise ValueError("controller command must target its exact owned session")
        absolute_deadline = self.config.get("controller_deadline_epoch")
        if absolute_deadline is not None:
            timeout = min(timeout, absolute_deadline - time.time())
        if timeout <= 0:
            raise RuntimeError("controller_absolute_deadline_reached_before_CLI_action")
        result = self.runner([CLI, *args], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            # Avoid persisting provider redirects/credentials in exception logs.
            raise RuntimeError(f"Colab CLI {args[0]} failed with exit {result.returncode}")
        return result.stdout

    def usage(self):
        text = re.sub(r"\x1b\[[0-9;]*m", "", self.call(["usage"], timeout=45))
        balance = re.search(r"Current balance:\s*([\d,.]+)\s*compute units", text)
        rate = re.search(r"Usage rate:\s*([\d,.]+)\s*/hr", text)
        if not balance or not rate:
            return {"verified": False, "reason": "unrecognized_usage_output", "observed_epoch": time.time()}
        return {"verified": True, "balance_units": float(balance.group(1).replace(",", "")),
                "usage_units_per_hour": float(rate.group(1).replace(",", "")), "observed_epoch": time.time()}

    def remote(self, action, config, request=None, *, timeout=120):
        timeout = int(timeout)
        if timeout <= 0:
            raise RuntimeError("controller_deadline_too_close_for_remote_action")
        script = self.remote_source + "\n" + (
            "_config=json.loads(" + repr(json.dumps(config)) + ")\n"
            "_request=json.loads(" + repr(json.dumps(request or {})) + ")\n"
            "print(" + repr(MARKER) + "+json.dumps(dispatch(" + repr(action) + ",_config,_request),ensure_ascii=False))\n")
        with tempfile.TemporaryDirectory(prefix="xavier_controller_") as tmp:
            path = Path(tmp) / "action.py"
            path.write_text(script)
            output = self.call(["exec", "--session", SESSION, "--file", str(path), "--timeout", str(timeout)], timeout=timeout + 30)
        for line in reversed(output.splitlines()):
            if MARKER in line:
                return json.loads(line.split(MARKER, 1)[1])
        raise RuntimeError("Colab execution did not return the controller result marker")

    def download(self, remote_path, local_path, *, reserve_bytes=0, timeout=3600):
        """Supervise only this CLI child; preserve partial bytes on interruption."""
        if not remote_path.startswith("/content/xavier/"):
            raise ValueError("download path outside owned Xavier project")
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_deadline = self.config.get("controller_deadline_epoch")
        if absolute_deadline is not None:
            timeout = min(timeout, absolute_deadline - time.time())
        if timeout <= 0:
            raise RuntimeError("controller_absolute_deadline_reached_before_download")
        with tempfile.TemporaryFile() as log:
            child = subprocess.Popen([CLI, "download", remote_path, str(local_path), "--session", SESSION],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + timeout
            try:
                while child.poll() is None:
                    if shutil.disk_usage(local_path.parent).free <= reserve_bytes:
                        raise RuntimeError("local_raw_recovery_paused_at_disk_reserve")
                    if time.monotonic() >= deadline:
                        raise RuntimeError("local_raw_recovery_timeout_partial_preserved")
                    time.sleep(1)
                if child.returncode:
                    raise RuntimeError(f"Colab CLI download failed with exit {child.returncode}")
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=5)
                raise

    def stop_owned_session(self):
        return self.call(["stop", "--session", SESSION], timeout=60)


def mac_process(pid, *, runner=subprocess.run):
    if not pid:
        return {"pid": None, "alive": False, "state": "missing"}
    env = {**os.environ, "LC_ALL": "C", "TZ": "UTC"}
    result = runner(["ps", "-p", str(int(pid)), "-o", "pid=,state=,lstart=,command="],
                    capture_output=True, text=True, timeout=10, env=env)
    if result.returncode or not result.stdout.strip():
        return {"pid": int(pid), "alive": False, "state": "missing"}
    try:
        parts = result.stdout.strip().split(None, 7)
        start = dt.datetime.strptime(" ".join(parts[2:7]), "%a %b %d %H:%M:%S %Y").replace(tzinfo=dt.timezone.utc).timestamp()
        argv = shlex.split(parts[7])
        return {"pid": int(parts[0]), "state": parts[1], "alive": not parts[1].startswith(("Z", "X")),
                "start_epoch": start, "argv": argv,
                "cmd_sha256": hashlib.sha256(json.dumps(argv, ensure_ascii=False).encode()).hexdigest()}
    except (ValueError, IndexError):
        return {"pid": int(pid), "alive": None, "state": "unreadable"}


def local_job_status(job):
    candidates = job.get("launch_receipts", [job.get("launch_receipt")])
    receipt_path = next((Path(p) for p in candidates if p and Path(p).exists()), None)
    if receipt_path is None:
        return {"state": "blocked", "reason": "original_launch_receipt_missing"}
    receipt = json.loads(receipt_path.read_text())
    process = mac_process(receipt["pid"])
    matches = (process.get("alive") is True and process.get("argv") == receipt["command"]
               and abs(process.get("start_epoch", 0) - receipt["started_epoch"]) < 2)
    deadline = receipt.get("deadline_epoch")
    if not deadline:
        return {"state": "blocked", "reason": "original_absolute_deadline_missing", "process": process}
    ledger = db_status(Path(job["db"])) if job.get("db") else {}
    state = "complete" if ledger.get("all_complete") else ("running" if matches else ("identity_mismatch" if process.get("alive") else "stopped"))
    if time.time() >= deadline and state not in {"complete", "running"}:
        state = "deadline_reached"
    return {"state": state, "process": process, "pinned_identity_matches": matches, "ledger": ledger,
            "started_epoch": receipt["started_epoch"], "deadline_epoch": deadline,
            "launch_receipt": str(receipt_path), "automatic_restart": False,
            "note": "Original absolute launch deadline retained; local jobs are monitored, not relaunched"}


def budget_allows(usage, seconds, config, *, now_epoch=None):
    now_epoch = time.time() if now_epoch is None else now_epoch
    if not usage.get("verified") or now_epoch - usage.get("observed_epoch", 0) > 600:
        return False
    balance, rate = usage["balance_units"], usage["usage_units_per_hour"]
    if balance < 0 or rate < 0:
        return False
    required = max(0, seconds) / 3600 * rate + float(config.get("compute_unit_reserve", 2))
    return required <= balance


def update_job_receipts(config, state, status):
    records = state.setdefault("remote_jobs", {})
    for job in config["remote_jobs"]:
        current = status["jobs"][job["job_id"]]
        process = current["process"]
        receipt = records.setdefault(job["job_id"], {"kind": job["kind"]})
        if not receipt.get("original_identity") and process.get("alive") and current["command_matches"]:
            receipt["original_identity"] = {k: process[k] for k in ("pid", "start_ticks", "start_epoch", "cmd_sha256")}
            receipt["current_identity"] = dict(receipt["original_identity"])
            receipt["original_argv"] = process["argv"]
            receipt["deadline_epoch"] = process["start_epoch"] + int(job["max_runtime_seconds"])
            if current.get("verified_launch_receipt"):
                receipt["deadline_epoch"] = min(receipt["deadline_epoch"], current["verified_launch_receipt"]["deadline_epoch"])
                receipt["launch_receipt"] = current["verified_launch_receipt"]
            receipt["identity_adoption"] = "pinned_on_first_matching_live_observation"
        if receipt.get("original_identity"):
            expected = receipt.get("current_identity", receipt["original_identity"])
            receipt["identity_matches"] = (process.get("pid") == expected.get("pid") and process.get("start_ticks") == expected.get("start_ticks")
                                           and process.get("cmd_sha256") == expected.get("cmd_sha256")) if process.get("alive") else False
        if current.get("ledger", {}).get("all_complete"):
            receipt.setdefault("completed_epoch", status["now_epoch"])
        if job.get("run_id"):
            run = status.get("runs", {}).get(job["run_id"], {})
            if run.get("terminal_artifacts"):
                receipt.setdefault("completed_epoch", status["now_epoch"])
                receipt["result_type"] = run["result_type"]
        receipt["last_observation"] = current
    origins = [r["original_identity"]["start_epoch"] for r in records.values() if r.get("original_identity")]
    if origins:
        candidate = min(origins) + int(config.get("max_session_work_seconds", 18000))
        state["session_deadline_epoch"] = min(candidate, state.get("session_deadline_epoch", candidate))


def restart_request(job, receipt, status, usage, state, config):
    if job["kind"] != "download" or not job.get("allow_resume", False) or receipt.get("completed_epoch"):
        return None
    process = status["jobs"][job["job_id"]]["process"]
    if process.get("alive") is not False or not receipt.get("original_identity"):
        return None
    deadline = min(receipt["deadline_epoch"], state.get("session_deadline_epoch", receipt["deadline_epoch"]))
    remaining = deadline - status["now_epoch"]
    if remaining <= 0 or status["free_bytes"] <= config["remote_reserve_bytes"] + 64 * 1024**2:
        return None
    if not budget_allows(usage, remaining, config, now_epoch=status["now_epoch"]):
        return None
    ledger = status["jobs"][job["job_id"]].get("ledger", {})
    if ledger.get("all_complete") or not ledger.get("pending_transfer_files"):
        return None
    request = {"job_id": job["job_id"], "deadline_epoch": deadline,
               "original_identity": receipt["original_identity"], "original_argv": receipt["original_argv"]}
    if "--max-bytes" in receipt["original_argv"]:
        traffic = receipt.get("verified_traffic_receipt")
        if not traffic or not traffic.get("verified") or traffic.get("process_identity") != receipt.get("current_identity"):
            receipt["resume_hold_reason"] = "cumulative_transfer_budget_unknown_for_capped_original_job"
            return None
        request["verified_traffic_receipt"] = traffic
    return request


def verify_and_extract(archive, receipt, destination, *, reserve_bytes=0):
    archive, destination = Path(archive), Path(destination)
    if archive.stat().st_size != receipt["size"] or digest_file(archive) != receipt["sha256"]:
        raise ValueError("snapshot archive size or SHA-256 mismatch")
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("snapshot_manifest.json"))
        expected = {r["path"]: r for r in manifest["files"]}
        actual = {item.filename for item in z.infolist()}
        if len(expected) != len(manifest["files"]) or len(actual) != len(z.infolist()):
            raise ValueError("duplicate archive members are forbidden")
        if actual != set(expected) | {"snapshot_manifest.json"}:
            raise ValueError("archive membership differs from snapshot manifest")
        for name, record in expected.items():
            safe_child(destination, name)
            h, size = hashlib.sha256(), 0
            with z.open(name) as stream:
                while block := stream.read(1024**2):
                    h.update(block)
                    size += len(block)
            if size != record["size"] or h.hexdigest() != record["sha256"]:
                raise ValueError("snapshot member checksum mismatch:" + name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        expanded_bytes = sum(record["size"] for record in expected.values())
        if shutil.disk_usage(destination.parent).free < reserve_bytes + expanded_bytes + 64 * 1024**2:
            raise ValueError("snapshot extraction would cross local disk reserve")
        for name in sorted(actual):
            target = safe_child(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as source, target.open("wb") as output:
                while chunk := source.read(1024**2):
                    if shutil.disk_usage(destination.parent).free <= reserve_bytes + len(chunk):
                        raise ValueError("snapshot extraction paused at local disk reserve")
                    output.write(chunk)
    return manifest


def local_fingerprint(path):
    s = path.stat()
    return {"device": s.st_dev, "inode": s.st_ino, "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def validate_chunk_descriptor(config, item, descriptor, offset, length, source_identity=None):
    expected = {"schema_version": "xavier-raw-chunk-v1", "source_path": item["remote_path"],
                "source_sha256": item["sha256"], "source_size": item["size"], "offset": offset, "size": length}
    if any(descriptor.get(k) != value for k, value in expected.items()):
        raise ValueError("raw_chunk_descriptor_source_offset_or_size_mismatch")
    identity = descriptor.get("source_identity")
    if not isinstance(identity, dict) or identity.get("size") != item["size"]:
        raise ValueError("raw_chunk_descriptor_source_identity_missing")
    if source_identity is not None and identity != source_identity:
        raise ValueError("raw_chunk_descriptor_source_identity_changed")
    sha = descriptor.get("chunk_sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("raw_chunk_descriptor_checksum_missing")
    key = raw_chunk_key(item["remote_path"], item["sha256"], identity)
    expected_path = str(safe_child(Path(config["remote_root"]), "controller_chunks/" + key + "/" + f"{offset:020d}-{length}.bin"))
    if descriptor.get("remote_path") != expected_path:
        raise ValueError("raw_chunk_descriptor_export_path_mismatch")


def download_raw_chunks(client, config, item, partial, *, reserve_bytes, progress_callback=None):
    """Persist bounded chunks and resume verified bytes, including interrupted appends."""
    partial = Path(partial)
    chunk_size = int(config.get("raw_chunk_bytes", RAW_CHUNK_LIMIT))
    if not 0 < chunk_size <= RAW_CHUNK_LIMIT:
        raise ValueError("raw_chunk_size_exceeds_64_MiB_limit")
    sidecar = partial.with_name(partial.name + ".chunks.json")
    timeout_deadline = time.time() + config.get("raw_download_timeout_seconds", 3600)
    deadline = min(timeout_deadline, config.get("controller_deadline_epoch", timeout_deadline))

    def remaining():
        seconds = deadline - time.time()
        if seconds <= 0:
            raise RuntimeError("raw_chunk_deadline_reached_partial_preserved")
        return max(.1, seconds)

    def save(receipt):
        receipt["updated_epoch"] = time.time()
        json_save(sidecar, receipt)
        if progress_callback:
            progress_callback({"source_path": item["remote_path"], "source_sha256": item["sha256"],
                               "confirmed_bytes": receipt["confirmed_bytes"], "source_size": item["size"],
                               "resume_receipt": str(sidecar), "updated_epoch": receipt["updated_epoch"]})

    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.with_name(partial.name + ".chunk.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if partial.is_symlink() or sidecar.is_symlink():
            raise ValueError("raw_chunk_local_symlink_rejected")
        if sidecar.exists():
            receipt = json.loads(sidecar.read_text())
            if any(receipt.get(k) != v for k, v in {"schema_version": "xavier-raw-recovery-chunks-v1",
                    "source_path": item["remote_path"], "source_sha256": item["sha256"],
                    "source_size": item["size"], "chunk_bytes": chunk_size}.items()):
                raise ValueError("raw_chunk_resume_source_or_protocol_changed")
        else:
            if partial.exists() and partial.stat().st_size:
                raise ValueError("untracked_raw_partial_preserved_without_overwrite")
            receipt = {"schema_version": "xavier-raw-recovery-chunks-v1", "source_path": item["remote_path"],
                       "source_sha256": item["sha256"], "source_size": item["size"], "chunk_bytes": chunk_size,
                       "confirmed_bytes": 0, "chunks": [], "status": "transferring"}
            if not partial.exists():
                partial.touch(exist_ok=False)
            save(receipt)
        if not partial.is_file() or partial.stat().st_size < receipt["confirmed_bytes"]:
            raise ValueError("raw_chunk_confirmed_prefix_missing")
        confirmed = 0
        # A restart rehashes the committed prefix using a bounded read buffer.
        with partial.open("rb") as stream:
            for descriptor in receipt["chunks"]:
                length = min(chunk_size, item["size"] - confirmed)
                validate_chunk_descriptor(config, item, descriptor, confirmed, length, receipt.get("source_identity"))
                digest, left = hashlib.sha256(), length
                while left:
                    block = stream.read(min(1024**2, left))
                    if not block:
                        raise ValueError("raw_chunk_committed_prefix_short")
                    digest.update(block)
                    left -= len(block)
                if digest.hexdigest() != descriptor["chunk_sha256"]:
                    raise ValueError("raw_chunk_committed_prefix_checksum_mismatch")
                confirmed += length
        if confirmed != receipt["confirmed_bytes"]:
            raise ValueError("raw_chunk_receipt_offset_inconsistent")
        if partial.stat().st_size > confirmed and not receipt.get("pending_chunk"):
            raise ValueError("unrecorded_raw_append_preserved_for_review")
        while confirmed < item["size"]:
            remaining()
            if receipt.get("pending_ack"):
                client.remote("ack_raw_chunk", config, receipt["pending_ack"], timeout=min(120, remaining()))
                receipt.pop("pending_ack")
                save(receipt)
            length = min(chunk_size, item["size"] - confirmed)
            if shutil.disk_usage(partial.parent).free < reserve_bytes + 2 * length + 1024**2:
                raise RuntimeError("raw_chunk_local_disk_reserve")
            request = {"source_path": item["remote_path"], "source_sha256": item["sha256"], "source_size": item["size"],
                       "offset": confirmed, "length": length, "source_identity": receipt.get("source_identity")}
            descriptor = client.remote("raw_chunk", config, request, timeout=min(120, remaining()))
            validate_chunk_descriptor(config, item, descriptor, confirmed, length, receipt.get("source_identity"))
            if receipt.get("pending_chunk") and receipt["pending_chunk"] != descriptor:
                raise ValueError("raw_chunk_pending_descriptor_changed")
            receipt["source_identity"] = descriptor["source_identity"]
            receipt["pending_chunk"] = descriptor
            save(receipt)
            staged = partial.with_name(partial.name + f".chunk-{confirmed:020d}.part")
            if staged.is_symlink():
                raise ValueError("raw_chunk_staging_symlink_rejected")
            if not staged.exists() or staged.stat().st_size != length or digest_file(staged) != descriptor["chunk_sha256"]:
                client.download(descriptor["remote_path"], staged, reserve_bytes=reserve_bytes,
                                timeout=min(300, remaining()))
            if staged.stat().st_size != length or digest_file(staged) != descriptor["chunk_sha256"]:
                raise ValueError("raw_chunk_download_checksum_mismatch")
            tail = partial.stat().st_size - confirmed
            if not 0 <= tail <= length:
                raise ValueError("raw_chunk_pending_append_size_inconsistent")
            # A process interruption can occur between append and receipt commit.
            # Verify any such tail against the staged piece and append only its remainder.
            with partial.open("r+b") as output, staged.open("rb") as source:
                output.seek(confirmed)
                left = tail
                while left:
                    count = min(1024**2, left)
                    if output.read(count) != source.read(count):
                        raise ValueError("raw_chunk_pending_append_bytes_mismatch")
                    left -= count
                output.seek(0, os.SEEK_END)
                while body := source.read(1024**2):
                    remaining()
                    if shutil.disk_usage(partial.parent).free <= reserve_bytes + len(body):
                        raise RuntimeError("raw_chunk_append_paused_at_disk_reserve")
                    output.write(body)
                output.flush()
                os.fsync(output.fileno())
            if partial.stat().st_size != confirmed + length:
                raise ValueError("raw_chunk_append_length_mismatch")
            receipt["chunks"].append(descriptor)
            confirmed += length
            receipt["confirmed_bytes"] = confirmed
            receipt.pop("pending_chunk")
            receipt["pending_ack"] = descriptor
            save(receipt)
            staged.unlink()
        if receipt.get("pending_ack"):
            client.remote("ack_raw_chunk", config, receipt["pending_ack"], timeout=min(120, remaining()))
            receipt.pop("pending_ack")
        if partial.stat().st_size != item["size"] or digest_file(partial) != item["sha256"]:
            raise ValueError("raw_chunk_final_source_checksum_mismatch")
        receipt["status"] = "complete_sha256_verified"
        save(receipt)
        return receipt


def recover_raw(config, state, inventory, client, *, progress_callback=None):
    """Recover all completed raw payloads with SHA verification, never overwrite gold files."""
    archive_root = Path(config["raw_archive_root"]).expanduser()
    archive_root.mkdir(parents=True, exist_ok=True)
    verified, held = {}, list(inventory.get("held", []))
    previous = state.get("raw_preservation_receipt", {}).get("files", {})
    basename_counts = {}
    for item in inventory["files"]:
        name = PurePosixPath(item["relative_path"]).name
        basename_counts[name] = basename_counts.get(name, 0) + 1
    transferred_files, transferred_bytes = 0, 0
    for item in inventory["files"]:
        relative = item["relative_path"]
        target = safe_child(archive_root, relative)
        try:
            cached = previous.get(relative, {})
            candidates = [target]
            if cached.get("local_path"):
                p = Path(cached["local_path"])
                if p.resolve().is_relative_to(archive_root.resolve()):
                    candidates.append(p)
            # Parent may already be recovering a unique basename outside dataset subdirs.
            basename = PurePosixPath(relative).name
            if basename_counts[basename] == 1:
                candidates.append(archive_root / basename)
            existing = next((p for p in candidates if p.exists()), None)
            if existing:
                before = local_fingerprint(existing)
                if existing.is_symlink() or not existing.is_file() or before["size"] != item["size"]:
                    raise ValueError("existing_raw_recovery_incomplete_or_conflicting; preserved_without_overwrite")
                if not (cached.get("local_path") == str(existing) and cached.get("local_stat") == before
                        and cached.get("sha256") == item["sha256"] and cached.get("size") == item["size"]):
                    if digest_file(existing) != item["sha256"] or local_fingerprint(existing) != before:
                        raise ValueError("existing_raw_recovery_sha256_mismatch_or_file_changed")
                verified[relative] = {"local_path": str(existing), "size": item["size"], "sha256": item["sha256"],
                                      "local_stat": before, "verified_epoch": time.time()}
                continue
            flat_partial = [archive_root / (basename + suffix) for suffix in (".part", ".download.part")]
            if basename_counts[basename] == 1 and any(p.exists() for p in flat_partial):
                raise ValueError("external_raw_recovery_partial_present; awaiting_owner")
            if transferred_files >= config.get("raw_recovery_max_files_per_cycle", 2):
                raise ValueError("raw_recovery_deferred_to_next_cycle")
            cycle_cap = config.get("raw_recovery_max_bytes_per_cycle", 32 * 1024**3)
            if transferred_files and transferred_bytes + item["size"] > cycle_cap:
                raise ValueError("raw_recovery_cycle_byte_limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".controller.download.part")
            if partial.is_symlink():
                raise ValueError("raw_recovery_partial_symlink_rejected")
            # Existing own partial may be replaced by this CLI, never a verified payload.
            needed = max(0, item["size"] - (partial.stat().st_size if partial.exists() else 0))
            if shutil.disk_usage(archive_root).free < config["local_reserve_bytes"] + needed + 64 * 1024**2:
                raise ValueError("raw_recovery_local_disk_reserve")
            if not partial.exists() or partial.stat().st_size != item["size"]:
                transferred_files += 1
                transferred_bytes += item["size"]
                if item["size"] > int(config.get("raw_chunk_bytes", RAW_CHUNK_LIMIT)):
                    chunk_config = dict(config, expected_boot_id=state.get("boot_id"))
                    download_raw_chunks(client, chunk_config, item, partial, reserve_bytes=config["local_reserve_bytes"],
                                        progress_callback=progress_callback)
                else:
                    client.download(item["remote_path"], partial, reserve_bytes=config["local_reserve_bytes"],
                                    timeout=config.get("raw_download_timeout_seconds", 3600))
            before = local_fingerprint(partial)
            if partial.is_symlink() or before["size"] != item["size"] or digest_file(partial) != item["sha256"]:
                raise ValueError("raw_recovery_download_size_or_sha256_mismatch")
            if local_fingerprint(partial) != before:
                raise ValueError("raw_recovery_download_changed_during_verification")
            if target.exists():
                raise ValueError("raw_recovery_target_created_concurrently; no_overwrite")
            os.link(partial, target)  # atomic no-overwrite promotion on this filesystem
            partial.unlink()
            verified[relative] = {"local_path": str(target), "size": item["size"], "sha256": item["sha256"],
                                  "local_stat": local_fingerprint(target), "verified_epoch": time.time()}
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as e:
            held.append({"relative_path": relative, "reason": str(e)[:300]})
    receipt = {"inventory_sha256": inventory["inventory_sha256"], "remote_boot_id": state.get("boot_id"),
               "file_count": inventory["file_count"], "verified_file_count": len(verified),
               "all_complete": bool(inventory.get("ok")) and not held and len(verified) == inventory["file_count"],
               "files": verified, "held": held, "verified_epoch": time.time(),
               "raw_payload_bytes_verified": sum(r["size"] for r in verified.values())}
    state["raw_preservation_receipt"] = receipt
    return {k: v for k, v in receipt.items() if k != "files"}


def can_stop(status, state, config):
    if not config.get("allow_idle_stop"):
        return False, "idle_stop_disabled_in_config"
    if not status.get("ok") or not status.get("process_scan_complete"):
        return False, "process_inventory_incomplete"
    if status.get("project_processes") or any(j["process"].get("alive") is not False for j in status.get("jobs", {}).values()):
        return False, "project_process_still_active_including_downloads"
    gpu = status.get("gpu_utilization_percent")
    if gpu is None or any(value > 1 for value in gpu):
        return False, "GPU_idle_not_verified"
    if not state.get("idle_since_epoch") or status["now_epoch"] - state["idle_since_epoch"] < config.get("idle_grace_seconds", 120):
        return False, "idle_grace_not_elapsed"
    if not state.get("last_snapshot_verified"):
        return False, "artifact_snapshot_not_verified_locally"
    snapshot_receipt = state["last_snapshot_verified"]
    try:
        if local_fingerprint(Path(snapshot_receipt["path"])) != snapshot_receipt.get("local_stat"):
            return False, "verified_artifact_snapshot_changed_or_missing"
    except (OSError, KeyError):
        return False, "verified_artifact_snapshot_changed_or_missing"
    if status.get("boot_id") != state.get("boot_id"):
        return False, "runtime_identity_changed"
    if status.get("raw_partial_file_count", 0) and not config.get("allow_stop_with_unpreserved_partials", False):
        return False, "unfinished_raw_partials_still_need_preservation_or_explicit_policy"
    inventory, custody = status.get("raw_inventory", {}), state.get("raw_preservation_receipt", {})
    if (not inventory.get("ok") or not custody.get("all_complete")
            or custody.get("inventory_sha256") != inventory.get("inventory_sha256")
            or custody.get("verified_file_count") != inventory.get("file_count")
            or custody.get("remote_boot_id") != status.get("boot_id")):
        return False, "all_completed_raw_payloads_not_verified_on_workstation_or_inventory_changed"
    for entry in custody.get("files", {}).values():
        try:
            if local_fingerprint(Path(entry["local_path"])) != entry.get("local_stat"):
                return False, "recovered_raw_payload_changed_after_verification"
        except OSError:
            return False, "recovered_raw_payload_missing_after_verification"
    for name, run in status.get("runs", {}).items():
        if run.get("terminal_artifacts") and state.get("exported_runs", {}).get(name) != run["fingerprint"]:
            return False, "completed_run_not_yet_exported"
        if run.get("terminal_artifacts") and not retained_archive_valid(state.get("run_archive_receipts", {}).get(name)):
            return False, "completed_run_archive_missing_or_changed"
    for gate in config.get("completion_gates", []):
        result = status.get("completion_gates", {}).get(gate["gate_id"], {})
        if not result.get("passed"):
            return False, "required_evaluation_not_completed:" + gate["gate_id"]
        if state.get("exported_completion_gates", {}).get(gate["gate_id"]) != result.get("report_sha256"):
            return False, "evaluation_report_not_yet_exported:" + gate["gate_id"]
        if not retained_archive_valid(state.get("gate_archive_receipts", {}).get(gate["gate_id"])):
            return False, "evaluation_report_archive_missing_or_changed:" + gate["gate_id"]
    return True, "owned_session_idle_and_required_artifacts_verified"


def retained_archive_valid(receipt):
    try:
        return bool(receipt) and local_fingerprint(Path(receipt["path"])) == receipt["local_stat"]
    except (OSError, KeyError, TypeError):
        return False


def tick(config, state_path, *, apply=False, client=None):
    state_path = Path(state_path).expanduser()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix(".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"state": "controller_already_running"}
        state = json.loads(state_path.read_text()) if state_path.exists() else {"schema_version": "xavier-controller-v1", "session": SESSION, "exported_runs": {}}
        if state.get("session") != SESSION:
            raise ValueError("state belongs to a different session")
        if state.get("session_stop_receipt"):
            return {"state": "owned_session_already_stopped", "session_stop_receipt": state["session_stop_receipt"]}
        cli = client or ColabCLI(config)
        events = []
        try:
            usage = cli.usage()
            state["usage"] = usage
            remote_config = copy.deepcopy(config)
            for job in remote_config["remote_jobs"]:
                expected = state.get("remote_jobs", {}).get(job["job_id"], {}).get("current_identity")
                if expected:
                    job["expected_identity"] = expected
            status = cli.remote("probe", remote_config)
            if not status.get("ok"):
                raise RuntimeError("remote_project_not_available")
            boot_id = status.get("boot_id")
            if not boot_id:
                raise RuntimeError("runtime_boot_identity_unavailable")
            if state.get("boot_id") and state["boot_id"] != boot_id:
                raise RuntimeError("runtime_replaced; never_act_on_reused_session_name")
            state["boot_id"] = boot_id
            remote_config["expected_boot_id"] = boot_id
            if apply:
                events.append({"reconciliation": cli.remote("reconcile", remote_config)})
                status = cli.remote("probe", remote_config)
                if status.get("boot_id") != boot_id:
                    raise RuntimeError("runtime_changed_during_reconciliation")
            update_job_receipts(config, state, status)
            state["local_jobs"] = {j["job_id"]: local_job_status(j) for j in config.get("local_jobs", [])}
            busy = bool(status["project_processes"]) or any(j["process"].get("alive") is not False for j in status["jobs"].values())
            if busy:
                state.pop("idle_since_epoch", None)
            else:
                state.setdefault("idle_since_epoch", status["now_epoch"])
            state["last_observed_status"] = status
            # Persist original identities/deadlines before any restart or export.
            json_save(state_path, state)
            if apply:
                new_runs = [name for name, run in status["runs"].items() if run["terminal_artifacts"] and state["exported_runs"].get(name) != run["fingerprint"]]
                new_gate_reports = [name for name, gate in status.get("completion_gates", {}).items() if gate.get("passed")
                                    and state.get("exported_completion_gates", {}).get(name) != gate.get("report_sha256")]
                elapsed = status["now_epoch"] - state.get("last_snapshot_epoch", 0)
                if new_runs or new_gate_reports or elapsed >= config.get("snapshot_interval_seconds", 900):
                    local_root = Path(config["local_root"]).expanduser()
                    local_root.mkdir(parents=True, exist_ok=True)
                    free = shutil.disk_usage(local_root).free
                    cap = min(config.get("max_snapshot_bytes", 8 * 1024**3), max(0, (free - config["local_reserve_bytes"]) // 2))
                    if cap > 0:
                        exported = cli.remote("snapshot", remote_config,
                                              {"run_ids": new_runs, "gate_ids": new_gate_reports,
                                               "include_prepared": not state.get("prepared_exported"), "max_bytes": cap}, timeout=300)
                        if exported.get("ok"):
                            target = local_root / "snapshots" / (exported["sha256"][:20] + ".zip")
                            target.parent.mkdir(parents=True, exist_ok=True)
                            if not target.exists():
                                cli.download(exported["archive"], target, reserve_bytes=config["local_reserve_bytes"])
                            manifest = verify_and_extract(target, exported, target.with_suffix(""), reserve_bytes=config["local_reserve_bytes"])
                            state["exported_runs"].update(exported["run_fingerprints"])
                            state.setdefault("exported_completion_gates", {}).update(exported.get("completion_gate_reports", {}))
                            state["prepared_exported"] = state.get("prepared_exported") or exported["prepared_included"]
                            state["last_snapshot_epoch"] = time.time()
                            state["last_snapshot_verified"] = {"path": str(target), "sha256": exported["sha256"], "size": exported["size"],
                                                                "local_stat": local_fingerprint(target), "included_runs": manifest["included_runs"],
                                                                "raw_waveforms_included": False}
                            for name in exported["run_fingerprints"]:
                                state.setdefault("run_archive_receipts", {})[name] = dict(state["last_snapshot_verified"])
                            for name in exported.get("completion_gate_reports", {}):
                                state.setdefault("gate_archive_receipts", {})[name] = dict(state["last_snapshot_verified"])
                            events.append({"snapshot_verified": state["last_snapshot_verified"]})
                        else:
                            events.append({"snapshot_blocked": exported})
                inventory = cli.remote("raw_inventory", remote_config)
                def recovery_progress(progress):
                    state["raw_recovery_progress"] = progress
                    json_save(state_path, state)
                events.append({"raw_recovery": recover_raw(config, state, inventory, cli, progress_callback=recovery_progress)})
                json_save(state_path, state)
                for job in config["remote_jobs"]:
                    receipt = state["remote_jobs"][job["job_id"]]
                    request = restart_request(job, receipt, status, usage, state, config)
                    if request:
                        # A verified old traffic receipt is single-use; if the RPC
                        # fails ambiguously we must not reuse its allowance.
                        receipt.pop("verified_traffic_receipt", None)
                        json_save(state_path, state)
                        launched = cli.remote("launch_download", remote_config, request)
                        events.append({"resume": launched})
                        if launched.get("started"):
                            p = launched["process"]
                            if all(k in p for k in ("pid", "start_ticks", "start_epoch", "cmd_sha256")):
                                receipt["current_identity"] = {k: p[k] for k in ("pid", "start_ticks", "start_epoch", "cmd_sha256")}
                            else:
                                receipt["identity_needs_review"] = True
                            receipt["resume_count"] = receipt.get("resume_count", 0) + 1
                            state.pop("idle_since_epoch", None)
                # Fresh probe after possible launch; an active download is never idle.
                status = cli.remote("probe", remote_config)
                if status.get("boot_id") != boot_id:
                    raise RuntimeError("runtime_changed_during_cycle")
                status["raw_inventory"] = cli.remote("raw_inventory", remote_config)
                state["last_observed_status"] = status
                stop, reason = can_stop(status, state, config)
                state["idle_stop_gate"] = {"allowed": stop, "reason": reason}
                if stop and status.get("boot_id") == state["boot_id"]:
                    cli.stop_owned_session()
                    state["session_stop_receipt"] = {"session": SESSION, "epoch": time.time(), "reason": reason}
                    events.append({"stopped_owned_session": SESSION})
            state["controller_status"] = "observed" if not apply else "cycle_completed"
            state.pop("last_error", None)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, sqlite3.Error) as e:
            state["controller_status"] = "blocked"
            state["last_error"] = {"type": type(e).__name__, "reason": str(e)[:500], "epoch": time.time()}
        state["last_cycle_epoch"] = time.time()
        state["last_cycle_apply"] = apply
        state["last_events"] = events
        json_save(state_path, state)
        return {"state": state["controller_status"], "events": events, "error": state.get("last_error"),
                "session_stop_receipt": state.get("session_stop_receipt")}


def default_config():
    private = "/Users/yasufumi/Documents/GitHub/NeuroCore-xavier-training/private/xavier"
    return {"cli": CLI, "session": SESSION, "remote_root": "/content/xavier", "local_root": private + "/controller",
            "raw_archive_root": private + "/colab_raw_archive",
            "remote_reserve_bytes": 30 * 1024**3, "local_reserve_bytes": 35 * 1024**3,
            "compute_unit_reserve": 2, "max_session_work_seconds": 18000, "snapshot_interval_seconds": 900,
            "allow_idle_stop": False, "allow_stop_with_unpreserved_partials": False, "idle_grace_seconds": 120,
            "raw_chunk_bytes": RAW_CHUNK_LIMIT,
            "raw_recovery_max_files_per_cycle": 2, "raw_recovery_max_bytes_per_cycle": 32 * 1024**3,
            "raw_download_timeout_seconds": 3600,
            "remote_jobs": [
                {"job_id": "download", "kind": "download", "pid_hint": 2312, "pid_file": "download.pid", "log_file": "download.log",
                 "db": "downloads.sqlite", "max_runtime_seconds": 14400, "allow_resume": True,
                 "expected_tokens": ["/content/xavier/code/acquire.py", "download", "/content/xavier/downloads.sqlite"]},
                {"job_id": "priority_eeg", "kind": "download", "pid_hint": 6406, "pid_file": "priority_eeg.pid", "log_file": "priority_eeg.log",
                 "db": "priority_eeg.sqlite", "max_runtime_seconds": 14400, "allow_resume": True,
                 "expected_tokens": ["/content/xavier/code/acquire.py", "download", "/content/xavier/priority_eeg.sqlite"]},
                {"job_id": "train", "kind": "train", "pid_file": "train.pid", "max_runtime_seconds": 3600, "allow_resume": False,
                 "run_id": "smollm2_1_7b_eeglass", "expected_tokens": ["/content/xavier/code/model/eeg_llm.py", "train", "HuggingFaceTB/SmolLM2-1.7B-Instruct"]},
                {"job_id": "train_8b", "kind": "train", "pid_hint": 4471, "pid_file": "train_8b.pid", "max_runtime_seconds": 3600, "allow_resume": False,
                 "run_id": "qwen3_8b_eeglass", "expected_tokens": ["/content/xavier/code/model/eeg_llm.py", "train", "Qwen/Qwen3-8B"]},
                {"job_id": "train_uniba", "kind": "train", "pid_hint": 15167, "pid_file": "train_uniba.pid", "max_runtime_seconds": 4800,
                 "allow_resume": False, "launch_receipt": "train_uniba_launch.json", "receipt_command_field": "supervisor_command",
                 "run_id": "qwen3_8b_aicoding_uniba_fused_v1", "expected_tokens": ["/content/xavier/run_uniba_training.py"]}],
            "completion_gates": [{"gate_id": "uniba_five_variant_evaluation",
                                  "receipt_path": "runs/qwen3_8b_aicoding_uniba_fused_v1/variant_evaluation_completion.json",
                                  "checkpoint_path": "runs/qwen3_8b_aicoding_uniba_fused_v1/best.pt"},
                                 {"gate_id": "uniba_natural_v3_five_variant_evaluation",
                                  "receipt_path": "runs/qwen3_8b_aicoding_uniba_natural_v3_fused_30e_v1/variant_evaluation_completion.json",
                                  "checkpoint_path": "runs/qwen3_8b_aicoding_uniba_natural_v3_fused_30e_v1/best.pt"}],
            "local_jobs": [
                {"job_id": "bulk", "kind": "download", "launch_receipts": [private + "/bulk_launch_v3.json", private + "/bulk_launch_v2.json"],
                 "db": private + "/bulk_downloads.sqlite"},
                {"job_id": "metadata_refresh", "kind": "metadata", "launch_receipt": private + "/metadata_refresh_launch.json"}]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("config")
    init.add_argument("--output", required=True)
    for name in ("tick", "run"):
        parser = sub.add_parser(name)
        parser.add_argument("--config", required=True)
        parser.add_argument("--state", required=True)
        parser.add_argument("--apply", action="store_true")
        if name == "run":
            parser.add_argument("--interval-seconds", type=int, default=300)
            parser.add_argument("--max-seconds", type=int, default=18000)
    args = p.parse_args()
    if args.command == "config":
        json_save(Path(args.output), default_config())
        return
    config = json.loads(Path(args.config).read_text())
    if args.command == "tick":
        print(json.dumps(tick(config, args.state, apply=args.apply), ensure_ascii=False))
        return
    state_path = Path(args.state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix(".daemon.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"state": "controller_daemon_already_running"}))
            return
        state_path.with_suffix(".pid").write_text(str(os.getpid()))
        remaining = min(18000, max(0, args.max_seconds))
        if config.get("controller_deadline_epoch") is not None:
            remaining = min(remaining, max(0, config["controller_deadline_epoch"] - time.time()))
        end = time.monotonic() + remaining
        while time.monotonic() < end:
            result = tick(config, args.state, apply=args.apply)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result.get("session_stop_receipt"):
                break
            until = min(end, time.monotonic() + max(60, args.interval_seconds))
            while time.monotonic() < until:
                time.sleep(min(30, max(0, until - time.monotonic())))


if __name__ == "__main__":
    main()
