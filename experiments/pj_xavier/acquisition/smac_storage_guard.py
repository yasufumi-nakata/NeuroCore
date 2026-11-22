"""Fail-closed runtime guard for PJ-Xavier large public dataset downloads.

Only a physically verified smac node and its node-specific raw root may acquire
new public data. Historical download ledgers are not proof of locally held bytes.
This is an operational guard on CLI entrypoints, not an OS sandbox for imports.
"""
from __future__ import annotations
import json
from pathlib import Path
import socket

REPOSITORY = Path(__file__).absolute().parents[3]
DEFAULT_POLICY_PATH = REPOSITORY / "private/xavier/storage_cleanup_20261122/storage_policy.json"
DEFAULT_NODES_PATH = REPOSITORY / "private/xavier/storage_cleanup_20261122/nodes.smac_only.json"


def require_smac_raw_root(raw_root, *, policy_path=None, nodes_path=None, hostname=None, home=None):
    """Reject all workstation roots before creating any download or ledger file.

    A remote node requires its own copied, checked policy and nodes registry.
    `hostname` and `home` are injectable only for deterministic CPU tests.
    """
    policy_path = Path(policy_path) if policy_path is not None else DEFAULT_POLICY_PATH
    nodes_path = Path(nodes_path) if nodes_path is not None else DEFAULT_NODES_PATH
    if not policy_path.is_file() or not nodes_path.is_file():
        raise ValueError("smac-only acquisition policy/registry absent: downloads blocked")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    registry = json.loads(nodes_path.read_text(encoding="utf-8"))
    if policy.get("download_location") != "smac_only" or policy.get("local_bulk_downloads_allowed") is not False or policy.get("local_fallback_allowed") is not False:
        raise ValueError("smac-only guard requires explicit deny-local policy")
    node = (hostname if hostname is not None else socket.gethostname()).split(".", 1)[0].lower()
    allowed = {row["node_id"] for row in registry.get("nodes", []) if row.get("transport") == "ssh" and row.get("node_id", "").startswith("smac")}
    excluded = set(policy.get("excluded_nodes", [])) | {"smac19"}
    if node in excluded or node not in allowed:
        raise ValueError("download host is not an authorized smac storage node")
    expected = Path(home if home is not None else Path.home()).expanduser() / "pj-xavier" / node / "raw"
    actual = Path(raw_root).expanduser()
    if not actual.is_absolute():
        raise ValueError("raw root must be absolute (or expand from ~)")
    expected = expected.resolve(strict=False)
    actual = actual.resolve(strict=False)
    if actual != expected and expected not in actual.parents:
        raise ValueError("raw root must remain under this smac node's dedicated raw directory")
    return {"status":"smac_only_verified_locally", "node_id": node, "raw_root":str(actual), "policy_path":str(policy_path)}
