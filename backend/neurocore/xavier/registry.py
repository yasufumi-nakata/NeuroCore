"""Trusted startup catalog; HTTP callers can select IDs, never artifact paths."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")


@dataclass(frozen=True)
class RegisteredModel:
    model_id: str
    title: str
    package_dir: Path
    base_model_dir: Path
    manifest_sha256: str

    def public(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "title": self.title,
                "manifest_sha256": self.manifest_sha256,
                "scientific_validation": "not_established"}


class ModelRegistry:
    """Immutable in-memory catalog loaded once from operator-owned configuration."""

    def __init__(self, entries: list[RegisteredModel] | None = None) -> None:
        entries = entries or []
        self._models: dict[str, RegisteredModel] = {}
        for entry in entries:
            if not _ID.fullmatch(entry.model_id) or entry.model_id in self._models:
                raise ValueError("Invalid or duplicate registered model ID")
            if not _SHA.fullmatch(entry.manifest_sha256):
                raise ValueError("Every model requires a pinned manifest SHA256")
            if not entry.package_dir.is_absolute() or not entry.base_model_dir.is_absolute():
                raise ValueError("Registered artifact roots must be absolute")
            self._models[entry.model_id] = entry

    @classmethod
    def from_file(cls, path: str | Path) -> ModelRegistry:
        path = Path(path).expanduser().resolve(strict=True)
        if not path.is_file() or path.stat().st_size > 128 * 1024:
            raise ValueError("Invalid registry configuration file")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"schema", "models"}:
            raise ValueError("Invalid registry schema")
        if payload["schema"] != "neurocore-xavier-registry-v1":
            raise ValueError("Unsupported registry schema")
        rows = payload["models"]
        if not isinstance(rows, list) or len(rows) > 32:
            raise ValueError("Registry must contain at most 32 models")
        entries = []
        for row in rows:
            required = {"id", "title", "package_dir", "base_model_dir", "manifest_sha256"}
            if not isinstance(row, dict) or set(row) != required:
                raise ValueError("Invalid registry model entry")
            if not all(isinstance(row[key], str) for key in required):
                raise ValueError("Registry fields must be strings")
            if not row["title"] or len(row["title"]) > 160:
                raise ValueError("Invalid registered title")
            roots = []
            for key in ("package_dir", "base_model_dir"):
                root = Path(row[key]).expanduser()
                if not root.is_absolute():
                    root = path.parent / root
                roots.append(root.resolve())
            entries.append(RegisteredModel(row["id"], row["title"], *roots, row["manifest_sha256"]))
        return cls(entries)

    @classmethod
    def from_environment(cls) -> ModelRegistry:
        # Called by API startup only. Changing env/files later cannot redirect requests.
        configured = os.environ.get("XAVIER_MODEL_REGISTRY")
        if configured:
            return cls.from_file(configured)
        packaged = Path(__file__).with_name("packagedcatalog.json")
        return cls.from_file(packaged) if packaged.is_file() else cls()

    def list(self) -> list[dict[str, Any]]:
        return [entry.public() for entry in self._models.values()]

    def get(self, model_id: str) -> RegisteredModel:
        if not isinstance(model_id, str) or not _ID.fullmatch(model_id):
            raise KeyError("Unknown model ID")
        try:
            return self._models[model_id]
        except KeyError:
            raise KeyError("Unknown model ID") from None
