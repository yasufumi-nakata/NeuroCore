#!/usr/bin/env python3
"""Small, deterministic PJ-Xavier data fleet planner and SSH preflight.

No authentication prompts, password handling, cluster-wide RAM pooling, or
automatic assumptions that historical nodes are currently usable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys

EEG_SCOPES = {"include_provisional", "component_only_provisional", "頭皮EEG候補", "EEG成分のみ候補"}
KNOWN_NODES = ("smac31", "smac01", "smac06", "smac11")
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10", "-o", "ConnectionAttempts=1"]


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("node ID must be a plain identifier")
    return value


def safe_file_path(value):
    p = PurePosixPath(value)
    if p.is_absolute() or "\\" in value or any(x in {"", ".", ".."} for x in value.split("/")):
        raise ValueError("unsafe manifest path")
    return str(p)


def ssh_prefix(node, binary="ssh"):
    if node.get("transport") != "ssh":
        raise ValueError("SSH dispatch is only for SSH nodes; Colab requires workstation CLI")
    target = node["ssh_target"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9@._-]*", target):
        raise ValueError("invalid SSH target")
    args = [binary, *SSH_OPTIONS]
    if node.get("control_path"):
        args.extend(["-o", "ControlPath=" + node["control_path"]])
    return args


PROBE = r'''import json,os,pathlib,platform,shutil,subprocess,datetime,sys
def sysctl(key):
    try:
        r=subprocess.run(['sysctl','-n',key],capture_output=True,text=True,timeout=5)
        return r.stdout.strip() if r.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):return None
root=pathlib.Path(sys.argv[1]).expanduser()
while not root.exists() and root.parent != root:root=root.parent
disk=shutil.disk_usage(root)
mem=sysctl('hw.memsize')
print(json.dumps({'status':'ready','preflight_success':True,'ssh_auth':'batch_authenticated','hostkey_verified':True,
 'measured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'platform':platform.platform(),
 'machine':platform.machine(),'hardware_model':sysctl('hw.model'),'cpu_label':sysctl('machdep.cpu.brand_string'),
 'cpu_cores':os.cpu_count(),'ram_bytes':int(mem) if mem and mem.isdigit() else None,
 'free_bytes':disk.free,'total_disk_bytes':disk.total,'capacity_path':str(root),
 'python_version':platform.python_version(),'gpu_training_verified':False,
 'measurement_scope':'read-only SSH and host/disk metadata; no training benchmark; no files created'}))
'''


def preflight_node(node, run=subprocess.run):
    out = dict(node)
    if node.get("transport") != "ssh":
        return {**out, "status": "blocked", "reason": "Colab_inventory_must_be_collected_via_workstation_CLI"}
    args = [*ssh_prefix(node), node["ssh_target"], shlex.join(["python3", "-", node.get("remote_root", "~/pj-xavier")])]
    try:
        r = run(args, input=PROBE, capture_output=True, text=True, timeout=45)
        if r.returncode:
            return {**out, "status": "blocked", "preflight_success": False,
                    "reason": "SSH_BatchMode_or_known_host_or_remote_probe_failed",
                    "ssh_exit_code": r.returncode, "stderr_tail": r.stderr[-1500:], "measured_at": utc_now()}
        measurement = json.loads(r.stdout.strip().splitlines()[-1])
        return {**out, **measurement, "reserve_bytes": int(node.get("reserve_bytes", 30 * 1024**3)),
                "reserved_pending_bytes": int(node.get("reserved_pending_bytes", 0))}
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as e:
        return {**out, "status": "blocked", "preflight_success": False, "reason": f"{type(e).__name__}:{str(e)[:300]}"}


def node_capacity(node, *, at, max_age_hours):
    if node.get("status") != "ready" or not node.get("preflight_success"):
        return 0, "current_preflight_not_successful"
    if node.get("transport") == "ssh" and (node.get("ssh_auth") != "batch_authenticated" or not node.get("hostkey_verified")):
        return 0, "SSH_authentication_or_known_host_not_verified"
    if node.get("transport") not in {"ssh", "workstation_colab_cli", "local_workstation"}:
        return 0, "unsupported_transport"
    if node.get("transport") == "workstation_colab_cli" and not node.get("runtime_id"):
        return 0, "Colab_runtime_ID_not_recorded"
    try:
        measured = dt.datetime.fromisoformat(node["measured_at"].replace("Z", "+00:00"))
        if measured.tzinfo is None or not 0 <= (at - measured).total_seconds() <= max_age_hours * 3600:
            return 0, "capacity_measurement_stale_or_future"
        free = int(node["free_bytes"])
        reserve = int(node["reserve_bytes"])
        pending = int(node.get("reserved_pending_bytes", 0))
        workspace = int(node.get("preprocessing_headroom_bytes", 0))
        if min(free, reserve, pending, workspace) < 0 or reserve < 1024**3:
            return 0, "invalid_capacity_or_missing_reserve"
    except (ValueError, KeyError, TypeError):
        return 0, "capacity_fields_unverified"
    return max(0, free - reserve - pending - workspace), "measured_disk_capacity_only_not_RAM_pool"


def file_group_key(path):
    """Keep required BrainVision/EEGLAB companions and split archives together."""
    low = path.lower()
    match = re.search(r"\.(?:zip|7z|rar|tar\.gz)\.\d{3}$", low)
    if match:
        return low[:match.end() - 4] + ":split_archive"
    if low.endswith((".vhdr", ".vmrk", ".eeg")):
        return str(PurePosixPath(low).with_suffix("")) + ":brainvision"
    if low.endswith((".set", ".fdt")):
        return str(PurePosixPath(low).with_suffix("")) + ":eeglab"
    return low


def make_plan(records, inventory, *, at=None, max_age_hours=24):
    at = at or dt.datetime.now(dt.timezone.utc)
    nodes = sorted(inventory["nodes"], key=lambda n: n["node_id"])
    if len({n["node_id"] for n in nodes}) != len(nodes):
        raise ValueError("node IDs must be unique")
    budget, node_summary = {}, {}
    for node in nodes:
        ident = safe_name(node["node_id"])
        cap, reason = node_capacity(node, at=at, max_age_hours=max_age_hours)
        budget[ident] = cap
        node_summary[ident] = {"allocatable_bytes": cap, "reason": reason, "measured_at": node.get("measured_at"),
                               "ram_bytes": node.get("ram_bytes"), "gpu_training_verified": node.get("gpu_training_verified", False)}
    assignments = {n["node_id"]: [] for n in nodes}
    excluded, review, blocked, datasets = [], [], [], []
    for row in sorted(records, key=lambda r: r["catalog_id"]):
        ident, scope = row["catalog_id"], row.get("scope_decision", "")
        if row.get("status") in {"excluded", "duplicate"} or scope.startswith("exclude_") or scope.startswith("対象外"):
            excluded.append({"catalog_id": ident, "reason": row.get("status") or scope})
            continue
        if scope not in EEG_SCOPES:
            review.append({"catalog_id": ident, "reason": "EEG_scope_not_explicitly_supported; filename_extensions_are_not_proof",
                           "scope_decision": scope, "status": row.get("status")})
            continue
        if row.get("status") != "resolved":
            blocked.append({"catalog_id": ident, "reason": "public_manifest_not_resolved", "status": row.get("status")})
            continue
        grouped, seen = defaultdict(list), {}
        conflict = False
        for f in row.get("files", []):
            path = safe_file_path(f["path"])
            identity = (f["url"], f.get("size_bytes"), json.dumps(f.get("published_checksum"), sort_keys=True))
            if path in seen:
                if identity != seen[path]:
                    conflict = True
                continue
            seen[path] = identity
            grouped[file_group_key(path)].append(f)
        if conflict:
            blocked.append({"catalog_id": ident, "reason": "conflicting_files_for_same_provider_path"})
            continue
        valid_groups = []
        for key, files in sorted(grouped.items()):
            if any(not f.get("declared_public") for f in files):
                blocked.append({"catalog_id": ident, "group": key, "reason": "file_access_not_declared_public"})
            elif any(isinstance(f.get("size_bytes"), bool) or not isinstance(f.get("size_bytes"), int) or f["size_bytes"] < 0 for f in files):
                blocked.append({"catalog_id": ident, "group": key, "reason": "size_unknown_or_invalid; measure_before_capacity_allocation"})
            else:
                valid_groups.append((key, files, sum(f["size_bytes"] for f in files)))
        datasets.append((row, valid_groups, sum(g[2] for g in valid_groups)))
    # Larger dataset packages first gives deterministic packing without treating
    # all 38 historical Macs as measured capacity. Equal-size ties use catalog ID.
    datasets.sort(key=lambda x: (int(x[0].get("priority", 100)), -x[2], x[0]["catalog_id"]))

    def fit(size):
        choices = [(remaining - size, node_id) for node_id, remaining in budget.items() if remaining >= size and node_summary[node_id]["allocatable_bytes"] > 0]
        return min(choices)[1] if choices else None

    def assign(node, row, files, size, basis):
        assignments[node].append({**row, "files": files, "assignment_basis": basis, "assigned_bytes": size})
        budget[node] -= size

    for row, groups, total in datasets:
        if not groups:
            continue
        node = fit(total)
        if node:
            assign(node, row, [f for _, fs, _ in groups for f in fs], total, "whole_dataset_available_file_groups")
            continue
        for key, files, size in sorted(groups, key=lambda g: (-g[2], g[0])):
            node = fit(size)
            if node:
                assign(node, row, files, size, "dataset_split_at_companion_or_archive_group_boundary")
            else:
                blocked.append({"catalog_id": row["catalog_id"], "group": key, "bytes": size,
                                "reason": "no_verified_node_with_remaining_capacity_for_atomic_file_group"})
    for ident in node_summary:
        node_summary[ident]["assigned_bytes"] = node_summary[ident]["allocatable_bytes"] - budget[ident]
        node_summary[ident]["remaining_allocatable_bytes"] = budget[ident]
        node_summary[ident]["file_count"] = sum(len(r["files"]) for r in assignments[ident])
    identity = {"assignments": assignments, "nodes": node_summary, "excluded": excluded, "review": review, "blocked": blocked}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"schema_version": "xavier-fleet-v1", "plan_sha256": digest, "created_at": at.isoformat(),
            "inventory": inventory, **identity,
            "summary": {"catalog_rows": len(records), "scope_review_records": len(review), "excluded_or_duplicate_records": len(excluded),
                        "blocked_records_or_file_groups": len(blocked), "assigned_files": sum(n["file_count"] for n in node_summary.values()),
                        "assigned_file_bytes": sum(n["assigned_bytes"] for n in node_summary.values()),
                        "byte_scope": "download_file_bytes_all_modalities_not_EEG_only_not_unique_recording_hours"}}


def write_bundle(plan, output, acquire_code, worker_code):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    commands = []
    for node in plan["inventory"]["nodes"]:
        ident = safe_name(node["node_id"])
        rows = plan["assignments"][ident]
        if not rows:
            continue
        local = root / ident
        local.mkdir(exist_ok=True)
        (local / "assigned.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        shutil.copyfile(acquire_code, local / "acquire.py")
        shutil.copyfile(worker_code, local / "node_worker.py")
        remote = node.get("remote_root", "~/pj-xavier") + "/" + ident
        if not re.fullmatch(r"(?:~/|/)[A-Za-z0-9._/-]+", remote) or ".." in PurePosixPath(remote).parts:
            raise ValueError("remote_root must be a dedicated simple absolute or home-relative path")
        jobs = {
            "download": {"job_id": plan["plan_sha256"][:16] + "-download", "kind": "download", "state": "ready",
                         "workdir": remote, "data_root": remote + "/raw", "reserve_bytes": node["reserve_bytes"],
                         "command": ["python3", remote + "/acquire.py", "download", "--manifest", remote + "/assigned.jsonl",
                                     "--db", remote + "/downloads.sqlite", "--root", remote + "/raw", "--reserve-gib",
                                     str(node["reserve_bytes"] / 1024**3), "--seconds", "600"],
                         "completion_check": {"kind": "download_sqlite", "path": remote + "/downloads.sqlite"}},
            "label": {"kind": "label", "state": "blocked", "reason": "label_worker_and_gold_vs_AI_provenance_contract_not_yet_configured"},
            "train": {"kind": "train", "state": "blocked", "reason": "model_runtime_memory_gate_and_benchmark_contract_not_yet_configured"},
        }
        for kind, job in jobs.items():
            save_json(local / f"{kind}.json", job)
        if node["transport"] != "ssh":
            commands.append({"node_id": ident, "transport": node["transport"], "status": "workstation_CLI_or_local_dispatch_required",
                             "bundle_path": str(local), "note": "No direct Colab connector or secret-handling command generated"})
            continue
        target = node["ssh_target"]
        mkdir_code = f"from pathlib import Path; Path({remote!r}).expanduser().mkdir(parents=True,exist_ok=True)"
        mkdir_argv = [*ssh_prefix(node), target, shlex.join(["python3", "-c", mkdir_code])]
        uploads = [str(local / name) for name in ("assigned.jsonl", "acquire.py", "node_worker.py", "download.json", "label.json", "train.json")]
        upload_argv = [*ssh_prefix(node, "scp"), *uploads, target + ":" + remote + "/"]
        launch_code = ("import pathlib,subprocess,json; "
                       f"r=pathlib.Path({remote!r}).expanduser(); "
                       "log=(r/'launcher.log').open('ab'); "
                       "p=subprocess.Popen(['python3',str(r/'node_worker.py'),'--job',str(r/'download.json')],"
                       "stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True); "
                       "print(json.dumps({'launcher_pid':p.pid,'status':'started_not_completion_verified'}))")
        launch_argv = [*ssh_prefix(node), target, shlex.join(["python3", "-c", launch_code])]
        commands.append({"node_id": ident, "transport": "ssh", "status": "commands_prepared_not_executed",
                         "steps": [{"purpose": label, "argv": argv, "shell_preview": shlex.join(argv)}
                                   for label, argv in (("create_dedicated_directory", mkdir_argv), ("upload_assigned_manifest_and_code", upload_argv), ("launch_download_worker", launch_argv))]})
    save_json(root / "dispatch_commands.json", commands)
    save_json(root / "plan.json", plan)
    return commands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("template")
    n.add_argument("--output", required=True)
    q = sub.add_parser("preflight")
    q.add_argument("--inventory", required=True)
    q.add_argument("--output", required=True)
    s = sub.add_parser("plan")
    s.add_argument("--manifest", required=True)
    s.add_argument("--inventory", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--max-age-hours", type=float, default=24)
    b = sub.add_parser("bundle")
    b.add_argument("--plan", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--acquire-code", required=True)
    b.add_argument("--worker-code", default=str(Path(__file__).with_name("node_worker.py")))
    args = p.parse_args()
    if args.cmd == "template":
        value = {"nodes": [{"node_id": name, "transport": "ssh", "ssh_target": "smac" if name == "smac31" else name,
                            "status": "unverified", "preflight_success": False, "reserve_bytes": 30 * 1024**3,
                            "remote_root": "~/pj-xavier", "note": "Historical node only; current capacity/authentication must be measured"}
                           for name in KNOWN_NODES], "historical_fleet_count": 38, "current_fleet_count_verified": False}
        save_json(args.output, value)
    elif args.cmd == "preflight":
        inv = json.loads(Path(args.inventory).read_text())
        inv["nodes"] = [preflight_node(node) for node in inv["nodes"]]
        save_json(args.output, inv)
        print(json.dumps({n["node_id"]: n["status"] for n in inv["nodes"]}))
    elif args.cmd == "plan":
        plan = make_plan(load_jsonl(args.manifest), json.loads(Path(args.inventory).read_text()), max_age_hours=args.max_age_hours)
        save_json(args.output, plan)
        print(json.dumps(plan["summary"], ensure_ascii=False))
    else:
        print(json.dumps(write_bundle(json.loads(Path(args.plan).read_text()), args.output, args.acquire_code, args.worker_code), ensure_ascii=False))


if __name__ == "__main__":
    main()
