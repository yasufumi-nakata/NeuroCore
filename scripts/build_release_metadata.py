from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build checksums and minimal SPDX metadata for release artifacts.")
    parser.add_argument("--dist", default="dist")
    parser.add_argument("--out", default="dist")
    args = parser.parse_args()
    dist = Path(args.dist)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = [
        path
        for path in sorted(dist.iterdir())
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    ]
    (out / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in artifacts), encoding="utf-8")
    spdx = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "neurocore-release-artifacts",
        "creationInfo": {"creators": ["Tool: NeuroCore release metadata builder"]},
        "packages": [
            {"fileName": path.name, "checksums": [{"algorithm": "SHA256", "checksumValue": sha256(path)}]}
            for path in artifacts
        ],
    }
    (out / "neurocore-release.spdx.json").write_text(json.dumps(spdx, indent=2), encoding="utf-8")
    print("NeuroCore release metadata built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
