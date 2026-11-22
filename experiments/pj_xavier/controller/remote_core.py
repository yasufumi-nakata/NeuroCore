"""Code executed only through workstation's `colab exec --file` command.

`probe` is read-only. Other actions are explicit controller mutations scoped to
/content/xavier. No session creation, billing, password, or external messaging.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import subprocess
import time
import zipfile


def json_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024**2):
            h.update(chunk)
    return h.hexdigest()


def safe_child(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or "\\" in relative or any(s in {"", ".", ".."} for s in relative.split("/")):
        raise ValueError("unsafe relative path")
    result = root.joinpath(*p.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("path escapes project root")
    return result


def proc_identity(pid, *, proc_root=Path("/proc"), clock_ticks=None):
    if not pid:
        return {"pid": None, "alive": False, "state": "missing"}
    directory = proc_root / str(int(pid))
    try:
        stat = (directory / "stat").read_text()
        fields = stat.rsplit(") ", 1)[1].split()
        state, start_ticks = fields[0], int(fields[19])
        argv = [s.decode("utf-8", errors="replace") for s in (directory / "cmdline").read_bytes().split(b"\0") if s]
        boot_time = next(int(s.split()[1]) for s in (proc_root / "stat").read_text().splitlines() if s.startswith("btime "))
        hz = clock_ticks or os.sysconf("SC_CLK_TCK")
        return {"pid": int(pid), "state": state, "alive": state not in {"Z", "X", "x"},
                "start_ticks": start_ticks, "start_epoch": boot_time + start_ticks / hz,
                "argv": argv, "cmd_sha256": hashlib.sha256(json.dumps(argv, ensure_ascii=False).encode()).hexdigest()}
    except FileNotFoundError:
        return {"pid": int(pid), "alive": False, "state": "missing"}
    except (OSError, ValueError, IndexError, StopIteration) as e:
        return {"pid": int(pid), "alive": None, "state": "unreadable", "error": type(e).__name__}


def command_matches(process, job):
    argv = process.get("argv", [])
    return bool(argv) and all(token in argv for token in job.get("expected_tokens", []))


def identity_matches(process, expected):
    return (process.get("alive") is True and process.get("pid") == expected.get("pid")
            and process.get("start_ticks") == expected.get("start_ticks")
            and process.get("cmd_sha256") == expected.get("cmd_sha256"))


def read_pid(root, job):
    path = safe_child(root, job["pid_file"])
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return job.get("pid_hint")


def db_status(path):
    if not path.exists():
        return {"exists": False, "all_complete": False}
    try:
        with sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True, timeout=10) as db:
            rows = list(db.execute("SELECT status,count(*),sum(bytes_present),sum(coalesce(expected_bytes,0)) FROM downloads GROUP BY status"))
            qtable = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='quarantined_releases'").fetchone()
            quarantined = db.execute("SELECT count(*) FROM quarantined_releases").fetchone()[0] if qtable else 0
        statuses = {r[0]: {"files": r[1], "bytes_present": r[2] or 0, "expected_bytes": r[3] or 0} for r in rows}
        active = set(statuses) - {"legacy_preserved", "superseded_manifest_path"}
        return {"exists": True, "statuses": statuses, "quarantined_releases": quarantined,
                "all_complete": bool(active) and active == {"complete"} and quarantined == 0,
                "pending_transfer_files": sum(v["files"] for k, v in statuses.items() if k in {"queued", "paused", "retry", "capacity_blocked", "running"}),
                "completed_file_bytes": statuses.get("complete", {}).get("bytes_present", 0)}
    except sqlite3.Error as e:
        return {"exists": True, "all_complete": False, "error": type(e).__name__}


def project_processes(root, *, proc_root=Path("/proc")):
    found, complete = [], True
    own_ancestors, pid = set(), os.getpid()
    for _ in range(20):
        if pid <= 1 or pid in own_ancestors:
            break
        own_ancestors.add(pid)
        try:
            pid = int((proc_root / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        if int(directory.name) in own_ancestors:
            continue  # this short-lived controller probe is not a workload
        try:
            command = (directory / "cmdline").read_bytes()
        except FileNotFoundError:
            continue
        except PermissionError:
            continue  # system processes outside the current user's ownership
        except OSError:
            complete = False
            continue
        try:
            cwd_is_project = (directory / "cwd").resolve(strict=True).is_relative_to(root.resolve())
        except (OSError, RuntimeError):
            cwd_is_project = False
        if str(root).encode() in command or cwd_is_project:
            process = proc_identity(int(directory.name), proc_root=proc_root)
            if process.get("alive") is not False:
                found.append(process)
    return found, complete


def run_fingerprint(path):
    entries = []
    for file in sorted(path.rglob("*")):
        if file.is_file() and not file.is_symlink() and not file.name.endswith((".tmp", ".part")):
            s = file.stat()
            entries.append((str(file.relative_to(path)), s.st_size, s.st_mtime_ns))
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


VARIANT_NAMES = ("fused", "context_only", "eeg_only", "eeg_shuffled", "eeg_time_shifted")


def project_file(root, value):
    path = Path(value)
    if path.is_absolute():
        path = path.relative_to(root)
    return safe_child(root, path.as_posix())


def verify_variant_completion(root, gate):
    """Research evaluation completion, deliberately independent of goal success."""
    result = {"passed": False, "receipt_path": gate["receipt_path"]}
    try:
        receipt_file = safe_child(root, gate["receipt_path"])
        receipt = json.loads(receipt_file.read_text())
        if receipt.get("status") != "completed" or receipt.get("exit_code") != 0:
            raise ValueError("variant_runner_not_completed_successfully")
        names = receipt.get("variant_names", [])
        if len(names) != 5 or set(names) != set(VARIANT_NAMES):
            raise ValueError("completion_receipt_does_not_cover_all_five_variants")
        checkpoint = safe_child(root, gate["checkpoint_path"])
        report_file = project_file(root, receipt["report_path"])
        if checkpoint.is_symlink() or report_file.is_symlink() or not checkpoint.is_file() or not report_file.is_file():
            raise ValueError("checkpoint_or_report_missing_or_symlink")
        checkpoint_before, report_before = checkpoint.stat(), report_file.stat()
        checkpoint_sha, report_sha = digest_file(checkpoint), digest_file(report_file)
        if checkpoint_sha != receipt.get("checkpoint_sha256") or report_sha != receipt.get("report_sha256"):
            raise ValueError("checkpoint_or_report_content_sha256_mismatch")
        report = json.loads(report_file.read_text())
        if report.get("schema_version") != "pj-xavier-variant-evaluation-v1" or report.get("status") != "completed":
            raise ValueError("variant_report_schema_or_status_mismatch")
        if report.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("report_refers_to_different_checkpoint")
        if len(report.get("variant_names", [])) != 5 or set(report["variant_names"]) != set(VARIANT_NAMES):
            raise ValueError("report_variant_names_mismatch")
        conditions = report.get("conditions", {})
        for name in VARIANT_NAMES:
            item = conditions.get(name, {})
            if item.get("status") not in {"measured", "blocked"}:
                raise ValueError("variant_outcome_not_finalized:" + name)
            if item["status"] == "blocked" and not item.get("reason"):
                raise ValueError("blocked_variant_missing_reason:" + name)
        for file, before in ((checkpoint, checkpoint_before), (report_file, report_before)):
            after = file.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("evaluation_artifact_changed_during_verification")
        result.update(passed=True, reason="five_variant_research_evaluation_finalized",
                      checkpoint_path=str(checkpoint.relative_to(root)), checkpoint_sha256=checkpoint_sha,
                      report_path=str(report_file.relative_to(root)), report_sha256=report_sha,
                      variant_statuses={name: conditions[name]["status"] for name in VARIANT_NAMES},
                      model_goal_validated=report.get("autonomous_agent_validated") is True)
    except (OSError, ValueError, KeyError, TypeError) as e:
        result["reason"] = str(e)[:300]
    return result


def probe(config, *, root_override=None, proc_root=Path("/proc")):
    root = Path(root_override or config["remote_root"])
    if not root.exists():
        return {"ok": False, "reason": "remote_project_root_missing", "now_epoch": time.time()}
    processes, scan_complete = project_processes(root, proc_root=proc_root)
    jobs = {}
    for job in config["remote_jobs"]:
        process = proc_identity(read_pid(root, job), proc_root=proc_root)
        expected = job.get("expected_identity")
        jobs[job["job_id"]] = {"kind": job["kind"], "process": process,
                               "command_matches": command_matches(process, job),
                               "pinned_identity_matches": identity_matches(process, expected) if expected else None}
        if job.get("launch_receipt"):
            try:
                receipt = json.loads(safe_child(root, job["launch_receipt"]).read_text())
                argv = receipt[job.get("receipt_command_field", "command")]
                if (argv == process.get("argv") and abs(receipt["started_epoch"] - process.get("start_epoch", 0)) < 2
                        and receipt["deadline_epoch"] >= receipt["started_epoch"]):
                    jobs[job["job_id"]]["verified_launch_receipt"] = {
                        "started_epoch": receipt["started_epoch"], "deadline_epoch": receipt["deadline_epoch"],
                        "command": argv, "path": job["launch_receipt"]}
                else:
                    jobs[job["job_id"]]["launch_receipt_error"] = "launch_receipt_process_identity_mismatch"
            except (OSError, ValueError, KeyError, TypeError):
                jobs[job["job_id"]]["launch_receipt_error"] = "launch_receipt_missing_or_invalid"
        if job.get("db"):
            jobs[job["job_id"]]["ledger"] = db_status(safe_child(root, job["db"]))
    runs = {}
    for path in sorted((root / "runs").glob("*")) if (root / "runs").exists() else []:
        if not path.is_dir() or path.is_symlink():
            continue
        active = any(str(path) in p.get("argv", []) for p in processes)
        summary, failure = path / "run_summary.json", path / "failure.json"
        checkpoints = [p.name for p in path.glob("*.pt") if p.is_file()]
        terminal = scan_complete and not active and (summary.exists() or failure.exists() or bool(checkpoints))
        runs[path.name] = {"active": active, "terminal_artifacts": terminal,
                           "result_type": "summary" if summary.exists() else ("failure" if failure.exists() else ("interrupted_checkpoint" if checkpoints else "unfinished")),
                           "fingerprint": run_fingerprint(path) if terminal else None,
                           "checkpoint_files": checkpoints}
    try:
        boot_id = (proc_root / "sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot_id = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        gpu = [int(x.strip()) for x in r.stdout.splitlines() if x.strip()] if r.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        gpu = None
    return {"ok": True, "now_epoch": time.time(), "boot_id": boot_id, "free_bytes": shutil.disk_usage(root).free,
            "jobs": jobs, "runs": runs, "project_processes": processes, "process_scan_complete": scan_complete,
            "completion_gates": {gate["gate_id"]: verify_variant_completion(root, gate) for gate in config.get("completion_gates", [])},
            "gpu_utilization_percent": gpu,
            "raw_partial_file_count": sum(1 for _ in (root / "raw").rglob("*.part")) if (root / "raw").exists() else 0}


def reconcile_priority(root, *, max_files=10):
    root = Path(root)
    primary, priority = root / "downloads.sqlite", root / "priority_eeg.sqlite"
    if not primary.exists() or not priority.exists():
        return {"reconciled": [], "held": [], "reason": "one_or_both_ledgers_missing"}
    result = {"reconciled": [], "held": []}
    with sqlite3.connect(primary, timeout=20) as main, sqlite3.connect("file:" + str(priority) + "?mode=ro", uri=True, timeout=20) as other:
        main.row_factory = other.row_factory = sqlite3.Row
        candidates = main.execute("SELECT * FROM downloads WHERE status='delegated_priority' LIMIT ?", (max_files,)).fetchall()
        for row in candidates:
            delegated = other.execute("SELECT * FROM downloads WHERE id=?", (row["id"],)).fetchone()
            try:
                if not delegated or delegated["status"] != "complete":
                    raise ValueError("priority_file_not_complete")
                for field in ("dataset_key", "relative_path", "url", "expected_bytes", "checksum_json"):
                    if row[field] != delegated[field]:
                        raise ValueError("ledger_identity_mismatch:" + field)
                size = row["expected_bytes"]
                if not isinstance(size, int) or size < 0 or delegated["bytes_present"] != size:
                    raise ValueError("expected_size_unverified_or_mismatched")
                file = safe_child(root / "raw", row["dataset_key"] + "/" + row["relative_path"])
                before = file.stat()
                if before.st_size != size or file.is_symlink():
                    raise ValueError("actual_file_size_or_symlink_mismatch")
                sha = digest_file(file)
                after = file.stat()
                if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                    raise ValueError("file_changed_during_verification")
                if not delegated["sha256"] or sha != delegated["sha256"] or (row["sha256"] and row["sha256"] != sha):
                    raise ValueError("actual_body_sha256_mismatch")
                checksum = json.loads(row["checksum_json"] or "null")
                if checksum and checksum.get("algorithm") in {"md5", "sha1", "sha256", "sha512"}:
                    h = hashlib.new(checksum["algorithm"])
                    with file.open("rb") as f:
                        while chunk := f.read(1024**2):
                            h.update(chunk)
                    if h.hexdigest() != checksum["value"]:
                        raise ValueError("published_checksum_mismatch")
                final = file.stat()
                if (before.st_ino, before.st_size, before.st_mtime_ns) != (final.st_ino, final.st_size, final.st_mtime_ns):
                    raise ValueError("file_changed_during_verification")
                main.execute("CREATE TABLE IF NOT EXISTS priority_reconciliation_ledger (id TEXT PRIMARY KEY, path TEXT, bytes INTEGER, sha256 TEXT, verified_at REAL)")
                changed = main.execute("UPDATE downloads SET status='complete',bytes_present=?,sha256=?,error=NULL,updated_at=? WHERE id=? AND status='delegated_priority'",
                                       (size, sha, dt.datetime.now(dt.timezone.utc).isoformat(), row["id"])).rowcount
                if changed:
                    main.execute("INSERT OR REPLACE INTO priority_reconciliation_ledger VALUES(?,?,?,?,?)", (row["id"], str(file), size, sha, time.time()))
                    main.commit()
                    result["reconciled"].append({"id": row["id"], "path": str(file), "size": size, "sha256": sha})
            except (OSError, ValueError, KeyError) as e:
                result["held"].append({"id": row["id"], "reason": str(e)})
    return result


def raw_inventory(config, *, root_override=None):
    """Inventory every completed payload; content is verified on local recovery.

    A DB 'complete' flag alone is insufficient: require the expected size, saved
    SHA-256, and current regular file. Duplicate ledger entries are counted once.
    No unverified file is silently omitted from the preservation gate.
    """
    root = Path(root_override or config["remote_root"])
    records, held = {}, []
    databases = sorted({j["db"] for j in config["remote_jobs"] if j.get("kind") == "download" and j.get("db")})
    if not databases:
        held.append({"reason": "no_download_ledgers_configured"})
    for name in databases:
        path = safe_child(root, name)
        if not path.is_file():
            held.append({"ledger": name, "reason": "ledger_missing"})
            continue
        try:
            with sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True, timeout=15) as db:
                db.row_factory = sqlite3.Row
                rows = db.execute("SELECT id,dataset_key,relative_path,expected_bytes,bytes_present,sha256 FROM downloads WHERE status='complete'").fetchall()
            for row in rows:
                relative = row["dataset_key"] + "/" + row["relative_path"]
                try:
                    file = safe_child(root / "raw", relative)
                    size, sha = row["expected_bytes"], row["sha256"]
                    if not isinstance(size, int) or size < 0 or row["bytes_present"] != size:
                        raise ValueError("completed_row_size_not_verified")
                    if not sha or not re.fullmatch(r"[0-9a-f]{64}", sha):
                        raise ValueError("completed_row_sha256_missing")
                    if not file.is_file() or file.is_symlink() or file.stat().st_size != size:
                        raise ValueError("completed_raw_missing_or_size_mismatch")
                    item = {"relative_path": relative, "remote_path": str(file), "size": size, "sha256": sha}
                    if relative in records and records[relative] != item:
                        raise ValueError("completed_ledgers_disagree_on_file")
                    records[relative] = item
                except (OSError, ValueError) as e:
                    held.append({"ledger": name, "id": row["id"], "relative_path": relative, "reason": str(e)})
        except sqlite3.Error as e:
            held.append({"ledger": name, "reason": type(e).__name__})
    files = [records[key] for key in sorted(records)]
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"ok": not held, "files": files, "file_count": len(files), "total_bytes": sum(r["size"] for r in files),
            "inventory_sha256": identity, "held": held, "observed_epoch": time.time()}


RAW_CHUNK_LIMIT = 64 * 1024**2


def source_file_identity(path):
    s = path.stat()
    return {"device": s.st_dev, "inode": s.st_ino, "size": s.st_size,
            "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}


def raw_chunk_key(source_path, source_sha256, source_identity):
    payload = {"source_path": source_path, "source_sha256": source_sha256, "source_identity": source_identity}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def raw_chunk(config, request, *, root_override=None):
    """Export one bounded piece of a completed raw file; leave its source unchanged."""
    root = Path(root_override or config["remote_root"])
    source_path = request["source_path"]
    relative = str(Path(source_path).relative_to(root / "raw"))
    source = safe_child(root / "raw", relative)
    if source.is_symlink() or not source.is_file():
        raise ValueError("raw_chunk_source_missing_or_symlink")
    inventory = raw_inventory(config, root_override=root)
    entry = next((f for f in inventory["files"] if f["remote_path"] == str(source)), None)
    if not entry or entry["size"] != request["source_size"] or entry["sha256"] != request["source_sha256"]:
        raise ValueError("raw_chunk_source_not_in_verified_completed_ledger")
    offset, length = request["offset"], request["length"]
    if (type(offset) is not int or type(length) is not int or not 0 <= offset < entry["size"]
            or not 0 < length <= RAW_CHUNK_LIMIT or offset + length > entry["size"]):
        raise ValueError("raw_chunk_offset_or_size_out_of_range")
    identity = source_file_identity(source)
    if request.get("source_identity") and request["source_identity"] != identity:
        raise ValueError("raw_chunk_source_identity_changed")
    key = raw_chunk_key(str(source), entry["sha256"], identity)
    directory = safe_child(root, "controller_chunks/" + key)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (f"{offset:020d}-{length}.bin")
    sidecar = target.with_suffix(".json")
    descriptor = {"schema_version": "xavier-raw-chunk-v1", "source_path": str(source),
                  "source_sha256": entry["sha256"], "source_size": entry["size"], "source_identity": identity,
                  "offset": offset, "size": length, "remote_path": str(target)}
    if target.exists() and sidecar.exists():
        saved = json.loads(sidecar.read_text())
        if (any(saved.get(k) != v for k, v in descriptor.items()) or target.is_symlink()
                or target.stat().st_size != length or digest_file(target) != saved.get("chunk_sha256")):
            raise ValueError("existing_raw_chunk_export_conflicts")
        return saved
    if target.exists() or sidecar.exists():
        raise ValueError("incomplete_raw_chunk_export_metadata")
    if shutil.disk_usage(root).free < int(config["remote_reserve_bytes"]) + length + 1024**2:
        raise ValueError("raw_chunk_remote_disk_reserve")
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    sha, remaining = hashlib.sha256(), length
    try:
        with source.open("rb") as src, temporary.open("xb") as out:
            src.seek(offset)
            while remaining:
                body = src.read(min(1024**2, remaining))
                if not body:
                    raise ValueError("raw_chunk_source_short_read")
                sha.update(body)
                out.write(body)
                remaining -= len(body)
            out.flush()
            os.fsync(out.fileno())
        if source_file_identity(source) != identity:
            raise ValueError("raw_chunk_source_changed_during_read")
        descriptor["chunk_sha256"] = sha.hexdigest()
        os.link(temporary, target)
        json_save(sidecar, descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return descriptor


def acknowledge_raw_chunk(config, request, *, root_override=None):
    """Remove only an identified, verified controller chunk after local commit."""
    root = Path(root_override or config["remote_root"])
    key = raw_chunk_key(request["source_path"], request["source_sha256"], request["source_identity"])
    target = safe_child(root, "controller_chunks/" + key + "/" + f"{request['offset']:020d}-{request['size']}.bin")
    if str(target) != request["remote_path"]:
        raise ValueError("raw_chunk_ack_path_identity_mismatch")
    sidecar = target.with_suffix(".json")
    if target.is_symlink() or sidecar.is_symlink():
        raise ValueError("raw_chunk_ack_missing_or_symlink")
    if not target.exists() and not sidecar.exists():
        return {"acknowledged": True, "already_absent": True}
    # A crash after deleting the binary but before deleting its sidecar is an
    # acknowledged, locally committed chunk. Only its exact retained descriptor
    # authorizes finishing this cleanup; unrelated files remain untouched.
    if not target.exists() and sidecar.is_file():
        if json.loads(sidecar.read_text()) != request:
            raise ValueError("raw_chunk_ack_content_mismatch")
        sidecar.unlink()
        return {"acknowledged": True, "recovered_interrupted_ack": True}
    if not target.is_file() or not sidecar.is_file():
        raise ValueError("raw_chunk_ack_missing_or_symlink")
    if json.loads(sidecar.read_text()) != request or target.stat().st_size != request["size"] or digest_file(target) != request["chunk_sha256"]:
        raise ValueError("raw_chunk_ack_content_mismatch")
    target.unlink()
    sidecar.unlink()
    return {"acknowledged": True}


def snapshot(config, request, *, root_override=None):
    root = Path(root_override or config["remote_root"])
    current = probe(config, root_override=root)
    selected = request.get("run_ids", [])
    for name in selected:
        if not current.get("runs", {}).get(name, {}).get("terminal_artifacts"):
            raise ValueError("requested run is still active or lacks terminal evidence:" + name)
    export = root / "controller_exports" / (str(time.time_ns()))
    export.mkdir(parents=True)
    paths = []
    for name in ("downloads.sqlite", "priority_eeg.sqlite"):
        source = root / name
        if source.exists():
            dest = export / name
            with sqlite3.connect("file:" + str(source) + "?mode=ro", uri=True, timeout=20) as src, sqlite3.connect(dest) as dst:
                src.backup(dst)
            paths.append((dest, "ledgers/" + name))
    for name in selected:
        base = safe_child(root, "runs/" + name)
        paths.extend((p, str(p.relative_to(root))) for p in base.rglob("*") if p.is_file() and not p.is_symlink() and not p.name.endswith((".tmp", ".part")))
    selected_gate_ids = set(request.get("gate_ids", current.get("completion_gates", {})))
    for gate_id, gate in current.get("completion_gates", {}).items():
        if gate_id in selected_gate_ids and gate.get("passed"):
            for key in ("receipt_path", "report_path", "checkpoint_path"):
                p = safe_child(root, gate[key])
                paths.append((p, gate[key]))
    if request.get("include_prepared") and (root / "prepared").exists():
        paths.extend((p, str(p.relative_to(root))) for p in (root / "prepared").rglob("*") if p.is_file() and not p.is_symlink() and not p.name.endswith((".tmp", ".part")))
    for name in ("priority_seed_manifest.jsonl", "priority_eeg_manifest.jsonl", "ai_coding_behavior_inventory.json", "exit_receipts.json"):
        p = root / name
        if p.exists() and p.is_file():
            paths.append((p, name))
    json_save(export / "status.json", current)
    paths.append((export / "status.json", "controller_status.json"))
    unique = {arcname: p for p, arcname in paths}
    total = sum(p.stat().st_size for p in unique.values())
    cap = int(request.get("max_bytes", 10 * 1024**3))
    if total > cap or shutil.disk_usage(root).free < total + int(config["remote_reserve_bytes"]):
        return {"ok": False, "reason": "snapshot_capacity_blocked", "source_bytes": total}
    records = []
    archive = export / "artifacts.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
        for arcname, p in sorted(unique.items()):
            safe_child(root, arcname)  # validate archive relative path
            before = p.stat()
            sha = digest_file(p)
            z.write(p, arcname)
            after = p.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("artifact changed during snapshot:" + arcname)
            records.append({"path": arcname, "size": before.st_size, "sha256": sha})
        manifest = {"files": records, "included_runs": selected,
                    "run_fingerprints": {name: current["runs"][name]["fingerprint"] for name in selected},
                    "created_epoch": time.time(), "raw_waveforms_included": False}
        manifest["completion_gate_reports"] = {key: value["report_sha256"] for key, value in current.get("completion_gates", {}).items()
                                                if key in selected_gate_ids and value.get("passed")}
        z.writestr("snapshot_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"ok": True, "archive": str(archive), "size": archive.stat().st_size, "sha256": digest_file(archive),
            "included_runs": selected, "run_fingerprints": manifest["run_fingerprints"],
            "completion_gate_reports": manifest["completion_gate_reports"],
            "prepared_included": bool(request.get("include_prepared")), "raw_waveforms_included": False}


def launch_download(config, request, *, root_override=None, now_epoch=None):
    """Resume an existing owned download within its original absolute deadline."""
    root = Path(root_override or config["remote_root"])
    now_epoch = time.time() if now_epoch is None else now_epoch
    job = next(j for j in config["remote_jobs"] if j["job_id"] == request["job_id"])
    if job["kind"] != "download":
        return {"started": False, "reason": "training_runs_are_never_relaunched_by_this_controller"}
    deadline = request.get("deadline_epoch")
    if not deadline or now_epoch >= deadline:
        return {"started": False, "reason": "original_absolute_deadline_reached_or_missing"}
    if not request.get("original_identity"):
        return {"started": False, "reason": "original_process_identity_not_pinned"}
    if shutil.disk_usage(root).free <= int(config["remote_reserve_bytes"]) + 64 * 1024**2:
        return {"started": False, "reason": "reserve_capacity_gate"}
    command = list(request["original_argv"])
    expected_script = str(root / "code/acquire.py")
    if expected_script not in command or "download" not in command or str(safe_child(root, job["db"])) not in command:
        return {"started": False, "reason": "restart_command_outside_pinned_download_scope"}
    if "--seconds" not in command:
        return {"started": False, "reason": "original_download_command_lacks_time_limit"}
    # The original acquisition ledger does not persist cumulative network traffic.
    # Retained file bytes cannot substitute for traffic after failed/repeated reads.
    # Never silently reset an existing --max-bytes budget on resume.
    if "--max-bytes" in command:
        traffic = request.get("verified_traffic_receipt")
        original_limit = int(command[command.index("--max-bytes") + 1])
        if not traffic or not traffic.get("verified") or not isinstance(traffic.get("bytes_consumed"), int):
            return {"started": False, "reason": "cumulative_transfer_budget_unknown_for_capped_original_job"}
        allowance = original_limit - traffic["bytes_consumed"]
        if allowance <= 0:
            return {"started": False, "reason": "original_transfer_budget_exhausted"}
        command[command.index("--max-bytes") + 1] = str(allowance)
    locks = root / "controller_locks"
    locks.mkdir(exist_ok=True)
    with (locks / (job["job_id"] + ".lock")).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"started": False, "reason": "another_controller_is_launching_this_job"}
        process = proc_identity(read_pid(root, job))
        if process.get("alive") is not False:
            return {"started": False, "reason": "pid_still_active_or_identity_unknown"}
        db_path = safe_child(root, job["db"])
        ledger = db_status(db_path)
        if ledger.get("all_complete"):
            return {"started": False, "reason": "download_already_complete"}
        if not ledger.get("pending_transfer_files"):
            return {"started": False, "reason": "no_transfer_ready_files"}
        processes, complete = project_processes(root)
        if not complete or any(str(db_path) in p.get("argv", []) for p in processes):
            return {"started": False, "reason": "another_process_may_own_download_ledger"}
        # The known sole worker is gone; its unfinished lease can be requeued.
        with sqlite3.connect(db_path, timeout=10) as db:
            db.execute("UPDATE downloads SET status='retry',lease_until=0,retry_at=0 WHERE status='running'")
        remaining = max(1, min(int(deadline - now_epoch), int(job.get("restart_slice_seconds", 600))))
        command[command.index("--seconds") + 1] = str(remaining)
        with safe_child(root, job["log_file"]).open("ab") as log:
            child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
        safe_child(root, job["pid_file"]).write_text(str(child.pid))
        identity = proc_identity(child.pid)
        receipt = {"started": True, "job_id": job["job_id"], "process": identity, "deadline_epoch": deadline,
                   "original_identity": request["original_identity"], "started_epoch": now_epoch,
                   "requires_new_traffic_receipt_for_further_resume": "--max-bytes" in command}
        json_save(root / "controller_receipts" / (job["job_id"] + ".json"), receipt)
        return receipt


def dispatch(action, config, request=None):
    if config.get("session") != "xavier-eeg-20261122" or config.get("remote_root") != "/content/xavier":
        raise ValueError("controller scope must be the explicitly owned Xavier session and root")
    if action != "probe" and config.get("expected_boot_id"):
        current = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if current != config["expected_boot_id"]:
            raise ValueError("runtime boot identity changed; action rejected")
    if action == "probe":
        return probe(config)
    if action == "reconcile":
        return reconcile_priority(Path(config["remote_root"]))
    if action == "raw_inventory":
        return raw_inventory(config)
    if action == "raw_chunk":
        return raw_chunk(config, request or {})
    if action == "ack_raw_chunk":
        return acknowledge_raw_chunk(config, request or {})
    if action == "snapshot":
        return snapshot(config, request or {})
    if action == "launch_download":
        return launch_download(config, request or {})
    raise ValueError("unknown controller action")
