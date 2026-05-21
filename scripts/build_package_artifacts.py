from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DOCS_PACKAGES = ROOT / "docs" / "packages"


def read_version() -> str:
    for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            return line.split('"', 2)[1]
    raise RuntimeError("pyproject.toml version is missing")


def run(command: list[str], cwd: Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, text=True, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_index(version: str, entries: list[dict[str, object]]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(entry['kind']))}</td>"
        f"<td><a href=\"{html.escape(str(entry['path']))}\">{html.escape(str(entry['name']))}</a></td>"
        f"<td><code>{html.escape(str(entry['sha256'])[:16])}</code></td>"
        f"<td>{entry['size']}</td>"
        "</tr>"
        for entry in entries
    )
    return f"""<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>NeuroCore Packages</title>
    <meta name="description" content="NeuroCore の Python 配布物と checksum です。">
    <link rel="stylesheet" href="../assets/neurocore-pages.css">
  </head>
  <body>
    <div class="site-shell">
      <header class="site-header"><nav class="nav" aria-label="Primary navigation"><a class="brand" href="../"><span class="brand-mark">NC</span><span><strong>NeuroCore</strong><span>EEG Control Runtime</span></span></a><div class="nav-links"><a href="../tutorial/">Tutorial</a><a href="../distributions/">Packages</a><a class="button" href="https://github.com/yasufumi-nakata/NeuroCore">GitHub</a></div></nav></header>
      <main>
        <article class="article section">
          <p class="breadcrumb"><a href="../">NeuroCore</a> / Packages</p>
          <h1>NeuroCore package registry</h1>
          <p>Version <code>{html.escape(version)}</code> の wheel、source distribution、checksum、SPDX metadata です。</p>
          <pre class="terminal"><code>pip install https://www.yasufumi.net/NeuroCore/packages/python/neurocore-{html.escape(version)}-py3-none-any.whl</code></pre>
          <table class="artifact-table">
            <thead><tr><th>Kind</th><th>File</th><th>SHA256 prefix</th><th>Bytes</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
          <p><a href="manifest.json">manifest.json</a> contains the full hashes and sizes.</p>
        </article>
      </main>
      <footer class="site-footer"><div class="section">NeuroCore keeps decoder weights outside the core package.</div></footer>
    </div>
  </body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NeuroCore package artifacts for Pages.")
    parser.parse_args()
    version = read_version()
    if DIST.exists():
        shutil.rmtree(DIST)
    if DOCS_PACKAGES.exists():
        shutil.rmtree(DOCS_PACKAGES)
    run([sys.executable, "scripts/build_pages.py"])
    run([sys.executable, "-m", "build"])

    target = DOCS_PACKAGES / "python"
    target.mkdir(parents=True)
    entries: list[dict[str, object]] = []
    for source in sorted(DIST.iterdir()):
        if source.suffix not in {".whl", ".gz"}:
            continue
        destination = target / source.name
        shutil.copy2(source, destination)
        entries.append(
            {
                "kind": "python",
                "name": destination.name,
                "path": f"python/{destination.name}",
                "sha256": sha256(destination),
                "size": destination.stat().st_size,
            }
        )

    (target / "SHA256SUMS").write_text(
        "".join(f"{entry['sha256']}  {entry['name']}\n" for entry in entries),
        encoding="utf-8",
    )
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "neurocore-pages-python-artifacts",
        "documentNamespace": "https://www.yasufumi.net/NeuroCore/packages/python",
        "creationInfo": {"creators": ["Tool: NeuroCore package artifact builder"]},
        "packages": [
            {"fileName": entry["name"], "checksums": [{"algorithm": "SHA256", "checksumValue": entry["sha256"]}]}
            for entry in entries
        ],
    }
    (target / "neurocore-release.spdx.json").write_text(json.dumps(sbom, indent=2), encoding="utf-8")
    entries.extend(
        {
            "kind": "metadata",
            "name": path.name,
            "path": f"python/{path.name}",
            "sha256": sha256(path),
            "size": path.stat().st_size,
        }
        for path in [target / "SHA256SUMS", target / "neurocore-release.spdx.json"]
    )
    DOCS_PACKAGES.mkdir(exist_ok=True)
    (DOCS_PACKAGES / "manifest.json").write_text(json.dumps({"version": version, "artifacts": entries}, indent=2), encoding="utf-8")
    (DOCS_PACKAGES / "index.html").write_text(render_index(version, entries), encoding="utf-8")
    print("NeuroCore package artifacts built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
