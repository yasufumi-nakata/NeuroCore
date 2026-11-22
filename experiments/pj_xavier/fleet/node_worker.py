#!/usr/bin/env python3
"""Execute one configured node-local job with a kind-specific exclusive lock.

This runner does not invent label/train commands or claim that exit 0 means all
downloads completed. Controller owns retries and selecting subsequent jobs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import time


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def completed(job):
    check = job.get("completion_check", {})
    if check.get("kind") == "download_sqlite":
        path = Path(check["path"]).expanduser()
        if not path.exists():
            return False, {"reason": "download_ledger_missing"}
        with sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True, timeout=10) as db:
            counts = dict(db.execute("SELECT status,count(*) FROM downloads GROUP BY status"))
            has_quarantine = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='quarantined_releases'").fetchone()
            quarantined = db.execute("SELECT count(*) FROM quarantined_releases").fetchone()[0] if has_quarantine else 0
        active = {k: v for k, v in counts.items() if k not in {"legacy_preserved", "superseded_manifest_path"}}
        return bool(active) and set(active) == {"complete"} and quarantined == 0, {"download_status_counts": counts,
                                                                               "quarantined_release_count": quarantined}
    if check.get("kind") == "artifacts":
        paths = [Path(p).expanduser() for p in check.get("paths", [])]
        return bool(paths) and all(p.is_file() and p.stat().st_size > 0 for p in paths), {"artifact_count": len(paths)}
    return False, {"reason": "completion_evidence_contract_missing"}


def run_job(path):
    path = Path(path).expanduser().resolve()
    job = json.loads(path.read_text())
    kind = job.get("kind")
    if kind not in {"download", "label", "train"}:
        raise ValueError("job kind must be download, label, or train")
    root = path.parent
    state = root / f"{kind}_state.json"
    lock_dir = root / "locks"
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f"{kind}.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"kind": kind, "state": "already_running", "reason": "node_local_kind_lock_held"}
        base = {"kind": kind, "job_id": job.get("job_id"), "worker_pid": os.getpid(), "started_at": stamp(),
                "command_sha256": hashlib.sha256(json.dumps(job.get("command", [])).encode()).hexdigest()}
        if job.get("state") != "ready" or not job.get("command"):
            result = {**base, "state": "blocked", "reason": job.get("reason", "job_not_configured_ready")}
            save(state, result)
            return result
        for dependency in job.get("dependencies", []):
            p = Path(dependency).expanduser()
            if not p.exists() or json.loads(p.read_text()).get("state") != "complete":
                result = {**base, "state": "dependency_blocked", "dependency": str(p)}
                save(state, result)
                return result
        data_root = Path(job["data_root"]).expanduser()
        data_root.mkdir(parents=True, exist_ok=True)
        reserve = int(job["reserve_bytes"])
        if reserve < 1024**3:
            raise ValueError("reserve must be at least 1 GiB")
        if shutil.disk_usage(data_root).free < reserve:
            result = {**base, "state": "capacity_blocked", "free_bytes": shutil.disk_usage(data_root).free}
            save(state, result)
            return result
        command = [str(Path(a).expanduser()) if isinstance(a, str) and a.startswith("~/") else a for a in job["command"]]
        if not all(isinstance(a, str) for a in command):
            raise ValueError("command must be argv strings")
        workdir = Path(job.get("workdir", root)).expanduser()
        workdir.mkdir(parents=True, exist_ok=True)
        log_path = root / f"{kind}.log"
        with log_path.open("ab") as log:
            child = subprocess.Popen(command, cwd=workdir, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            running = {**base, "state": "running", "child_pid": child.pid, "log": str(log_path)}
            save(state, running)
            capacity_stop = False
            try:
                while child.poll() is None:
                    if shutil.disk_usage(data_root).free < reserve:
                        capacity_stop = True
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                        break
                    time.sleep(1)
                code = child.wait()
            except BaseException:
                # Only the process group created above is ours to terminate.
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                save(state, {**running, "state": "interrupted", "ended_at": stamp()})
                raise
        try:
            done, evidence = completed(job) if code == 0 else (False, {})
        except (OSError, ValueError, sqlite3.Error) as e:
            done, evidence = False, {"completion_evidence_error": f"{type(e).__name__}:{str(e)[:300]}"}
        final_state = "capacity_blocked" if capacity_stop else ("complete" if done else ("paused" if code == 0 else "failed"))
        result = {**running, "state": final_state, "exit_code": code, "ended_at": stamp(), **evidence}
        save(state, result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    print(json.dumps(run_job(parser.parse_args().job), ensure_ascii=False))
