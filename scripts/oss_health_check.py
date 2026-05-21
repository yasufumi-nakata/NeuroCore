from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "SUPPORT.md",
    "GOVERNANCE.md",
    "ROADMAP.md",
    "AGENTS.md",
    "CLAUDE.md",
    ".github/CODEOWNERS",
    ".github/dependabot.yml",
    ".github/pull_request_template.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/ISSUE_TEMPLATE/safety_case.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/pages.yml",
    ".github/workflows/release.yml",
    ".github/workflows/codeql.yml",
    ".github/workflows/dependency-review.yml",
    ".github/workflows/scorecard.yml",
    ".github/workflows/oss-health.yml",
    "docs/assets/neurocore-pages.css",
    "docs/index.html",
    "docs/tutorial/index.html",
    "docs/distributions.md",
    "docs/deployment.md",
    "docs/github-action.md",
    "docs/maintainer-guide.md",
    "scripts/build_pages.py",
    "scripts/build_package_artifacts.py",
    "scripts/build_release_metadata.py",
    "scripts/validate_pages.py",
]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def project_version() -> str:
    match = re.search(r'^version = "([^"]+)"$', read("pyproject.toml"), re.MULTILINE)
    if not match:
        raise AssertionError("pyproject.toml version is missing")
    return match.group(1)


def package_version() -> str:
    match = re.search(r'^__version__ = "([^"]+)"$', read("backend/neurocore/__init__.py"), re.MULTILINE)
    if not match:
        raise AssertionError("backend/neurocore/__init__.py __version__ is missing")
    return match.group(1)


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED_FILES:
        if not (ROOT / path).is_file():
            errors.append(f"missing required OSS file: {path}")
    version = project_version()
    if package_version() != version:
        errors.append("pyproject.toml and neurocore.__version__ disagree")
    citation = read("CITATION.cff")
    if f'version: "{version}"' not in citation:
        errors.append("CITATION.cff does not match project version")
    readme = read("README.md")
    for marker in ["actions/workflows/ci.yml/badge.svg", "CONTRIBUTING.md", "SECURITY.md", "ROADMAP.md"]:
        if marker not in readme:
            errors.append(f"README.md missing marker: {marker}")
    for workflow in ["ci.yml", "pages.yml", "oss-health.yml"]:
        text = read(f".github/workflows/{workflow}")
        if "permissions:" not in text or "contents: read" not in text:
            errors.append(f"{workflow} missing least-privilege permissions")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"NeuroCore OSS health check passed for version {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
