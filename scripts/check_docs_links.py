#!/usr/bin/env python3
"""Validate relative links and doc layout across the repository.

Checks:
  1. every relative .md/.html link in a tracked file resolves on disk
  2. tracked .md/.html basenames are unique where the docs generator relies on it
  3. every GitHub blob URL emitted by the course site points at a real file

Exits non-zero on failure so CI can gate on it.

Usage:
    python3 scripts/check_docs_links.py
    python3 scripts/check_docs_links.py --html-dir benchmark_reports/docs
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJ = ROOT

# Excluded from the walk: generated course output and the vendored virtualenv.
GENERATED_PREFIX = "docs/topics/html/"

# Links whose targets are produced by CI at deploy time, not committed here.
ALLOWLIST = [
    re.compile(r"^(\.\./)+(benchmark_reports)/"),
    re.compile(r"^(\.\./)+(dashboard|history|tests|rv32_bench)\.html$"),
]

# Basenames that legitimately repeat; the docs generator disambiguates these by
# a path prefix rather than by basename.
BASENAME_EXEMPT = {"INDEX.md", "__init__.py", "README.md"}

LINK_RE = re.compile(
    r"""
      \]\(\s*<?(?P<md>[^)\s<>]+\.(?:md|html))(?P<md_a>\#[^)\s>]*)?>?\s*\)
    | href="(?P<href>[^"]+\.(?:md|html))(?P<href_a>\#[^"]*)?"
    | ^\[[^\]]+\]:\s*(?P<ref>\S+\.(?:md|html))(?P<ref_a>\#\S*)?
    """,
    re.X | re.M,
)

# Only this repository's blob URLs -- other GitHub links (e.g. onnx/onnx) are
# external references, not paths we can validate.
SELF_REPO = "https://github.com/ScratchV-Compiler/ScratchV/blob/"
BLOB_RE = re.compile(re.escape(SELF_REPO) + r"[^/]+/([^\"'>\s]+)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "data:", "/", "#")


def git_ls_files(*patterns: str) -> list[str]:
    """Tracked paths. -z keeps non-ASCII names intact (core.quotepath escapes them)."""
    out = subprocess.run(
        ["git", "ls-files", "-z", *patterns],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [p for p in out.split("\0") if p]


def tracked_docs() -> list[str]:
    return [
        p for p in git_ls_files("*.md", "*.html")
        if not p.startswith(GENERATED_PREFIX)
    ]


def check_links(paths: list[str]) -> tuple[int, list[str]]:
    broken, checked = [], 0
    for p in paths:
        text = (ROOT / p).read_bytes().decode("utf-8", "replace")
        for m in LINK_RE.finditer(text):
            dst = m.group("md") or m.group("href") or m.group("ref")
            if dst is None or dst.startswith(SKIP_PREFIXES):
                continue
            checked += 1
            if any(a.match(dst) for a in ALLOWLIST):
                continue
            target = (ROOT / p).parent / urllib.parse.unquote(dst)
            if not target.resolve().exists():
                broken.append(f"{p}: {dst}")
    return checked, broken


def check_basenames(paths: list[str]) -> list[str]:
    counts = Counter(Path(p).name for p in paths)
    return sorted(n for n, c in counts.items() if c > 1 and n not in BASENAME_EXEMPT)


def check_blob_urls(html_dir: Path | None) -> list[str]:
    """Every blob/main/... path the site emits must exist in the working tree."""
    if html_dir is None or not html_dir.is_dir():
        return []
    tracked = set(git_ls_files())
    dead = set()
    for page in sorted(html_dir.glob("*.html")):
        text = page.read_bytes().decode("utf-8", "replace")
        for m in BLOB_RE.finditer(text):
            rel = urllib.parse.unquote(m.group(1)).rstrip(".,)")
            if rel and rel not in tracked:
                dead.add(rel)
    return sorted(dead)


def main() -> int:
    ap = argparse.ArgumentParser(description="Check docs links and layout")
    ap.add_argument("--html-dir", default=None,
                    help="generated course dir, for blob-URL validation")
    args = ap.parse_args()

    paths = tracked_docs()
    failed = False

    checked, broken = check_links(paths)
    if broken:
        failed = True
        print(f"BROKEN: {len(broken)} dead link(s) across {len(paths)} files", file=sys.stderr)
        for b in broken:
            print(f"  {b}", file=sys.stderr)
    else:
        print(f"OK: {checked} links checked across {len(paths)} files, 0 broken")

    dups = check_basenames(paths)
    if dups:
        failed = True
        print(f"BASENAME: not unique: {dups}", file=sys.stderr)
    else:
        print("OK: tracked md/html basenames unique")

    html_dir = Path(args.html_dir) if args.html_dir else None
    dead = check_blob_urls(html_dir)
    if dead:
        failed = True
        print(f"DEAD BLOB: {len(dead)} generated link(s) point at missing files",
              file=sys.stderr)
        for d in dead:
            print(f"  {d}", file=sys.stderr)
    elif html_dir:
        print(f"OK: no dead GitHub blob URLs in {html_dir}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
