from __future__ import annotations

import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PAGES = [
    Path("index.html"),
    Path("tutorial/index.html"),
    Path("architecture/index.html"),
    Path("testing-strategy/index.html"),
    Path("settings-ui/index.html"),
    Path("distributions/index.html"),
    Path("deployment/index.html"),
    Path("github-action/index.html"),
    Path("maintainer-guide/index.html"),
    Path("packages/index.html"),
]


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_name = "src" if tag in {"img", "script"} else "href"
        for name, value in attrs:
            if name == attr_name and value:
                self.links.append((tag, value))


def local_target(page: Path, value: str) -> Path | None:
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc or value.startswith("#"):
        return None
    if not parsed.path:
        return None
    target = (page.parent / parsed.path).resolve()
    if parsed.path.endswith("/") or target.is_dir():
        target = target / "index.html"
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated NeuroCore pages.")
    parser.add_argument("--root", default=str(ROOT / "docs"), help="Site root to validate")
    args = parser.parse_args()
    site_root = Path(args.root).resolve()
    errors: list[str] = []

    for required in REQUIRED_PAGES:
        if not (site_root / required).is_file():
            errors.append(f"missing page: {required}")

    pages = sorted(site_root.rglob("*.html"))
    if not pages:
        errors.append(f"no HTML pages found in {site_root}")

    for page in pages:
        html = page.read_text(encoding="utf-8")
        for marker in ["NeuroCore", "decoder weights"]:
            if marker not in html:
                errors.append(f"{page.relative_to(site_root)} missing marker: {marker}")
        parser = LinkParser()
        parser.feed(html)
        for tag, value in parser.links:
            parsed = urlparse(value)
            if not parsed.scheme and not parsed.netloc and parsed.path.endswith(".md"):
                errors.append(f"{page.relative_to(site_root)} raw markdown {tag} link: {value}")
            target = local_target(page, value)
            if target and not target.exists():
                errors.append(f"{page.relative_to(site_root)} broken {tag} link: {value}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("NeuroCore pages validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
