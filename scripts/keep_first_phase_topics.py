#!/usr/bin/env python3
"""One-off: leave only the first-phase topics in docs/topics/.

14 topics were set up in the project's first phase (2026-05-21 task specs,
implemented in PR #5).  The other 16 topics were added later.  This moves the
later ones to docs/archive/后续课题/ so docs/topics/ holds the first phase only.

    python3 scripts/keep_first_phase_topics.py --phase move
    python3 scripts/keep_first_phase_topics.py --phase links
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = "docs/archive/后续课题"
SIDECAR = Path("/tmp/keep_moves.json")

# Topics from the project's first phase (2026-05-21 task specs / PR #5).
FIRST_PHASE = {1, 5, 6, 7, 9, 11, 12, 13, 14, 17, 18, 20, 21, 28}


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, check=check,
                          capture_output=True, text=True)


def git_ls_files(*patterns: str) -> list[str]:
    out = git("ls-files", "-z", *patterns).stdout
    return [p for p in out.split("\0") if p]


def build_moves() -> list[tuple[str, str]]:
    moves = []
    for p in git_ls_files("docs/topics/*"):
        rel = p[len("docs/topics/"):]
        m = re.match(r"^(\d{2})-", rel)
        if not m:
            continue                                   # INDEX.md etc.
        if int(m.group(1)) in FIRST_PHASE:
            continue
        moves.append((p, f"{DEST}/{rel}"))
    return moves


MOVES = build_moves()
PATH_MAP = {s: d for s, d in MOVES}


def save() -> None:
    SIDECAR.write_text(json.dumps(MOVES, ensure_ascii=False), encoding="utf-8")


def load() -> None:
    global MOVES, PATH_MAP
    MOVES = [tuple(x) for x in json.loads(SIDECAR.read_text(encoding="utf-8"))]
    PATH_MAP = {s: d for s, d in MOVES}


def preflight() -> None:
    srcs = [s for s, _ in MOVES]
    dsts = [d for _, d in MOVES]
    assert len(set(srcs)) == len(srcs), f"dup source: {_dups(srcs)}"
    assert len(set(dsts)) == len(dsts), f"target conflict: {_dups(dsts)}"
    for s in srcs:
        assert (ROOT / s).is_file(), f"missing: {s}"
        assert git("ls-files", "--error-unmatch", "--", s, check=False).returncode == 0, \
            f"not tracked: {s}"
    for d in dsts:
        assert not (ROOT / d).exists(), f"target exists: {d}"
    # no first-phase topic may be dragged along
    for s in srcs:
        rel = s[len("docs/topics/"):]
        assert int(rel[:2]) not in FIRST_PHASE, f"would move a first-phase topic: {s}"
    print(f"preflight OK: {len(MOVES)} files from "
          f"{len({s.split('/')[2] for s in srcs})} later topics -> {DEST}")


def _dups(xs: list[str]) -> list[str]:
    return sorted(n for n, c in Counter(xs).items() if c > 1)


def do_move() -> None:
    for src, dst in MOVES:
        (ROOT / dst).parent.mkdir(parents=True, exist_ok=True)
        git("mv", "--", src, dst)
    # git does not track directories; prune emptied topic folders
    topics = ROOT / "docs/topics"
    for d in sorted(topics.iterdir()):
        if d.is_dir() and d.name != "html" and not any(d.iterdir()):
            d.rmdir()
            print(f"  pruned empty folder: {d.name}")
    print(f"moved {len(MOVES)} files")


LINK_RE = re.compile(
    r"""
      \[(?P<label>[^\]]*)\]\(\s*<?(?P<dst>[^)\s<>]+\.(?:md|html))(?P<anchor>\#[^)\s>]*)?>?\s*\)
    | href="(?P<href>[^"]+\.(?:md|html))(?P<href_a>\#[^"]*)?"
    | ^\[[^\]]+\]:\s*(?P<ref>\S+\.(?:md|html))(?P<ref_a>\#\S*)?
    """,
    re.X | re.M,
)
SKIP = ("http://", "https://", "mailto:", "data:", "/", "#")


def rewrite(path: str) -> tuple[int, int]:
    old_src = next((s for s, d in MOVES if d == path), path)
    raw = (ROOT / path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return 0, 0
    n_dst = n_label = 0

    def sub(m: re.Match) -> str:
        nonlocal n_dst, n_label
        dst = m.group("dst") or m.group("href") or m.group("ref")
        if dst is None or dst.startswith(SKIP):
            return m.group(0)
        old_abs = os.path.normpath(os.path.join(os.path.dirname(old_src), dst))
        if old_abs in PATH_MAP:
            new_abs = PATH_MAP[old_abs]
        elif (ROOT / old_abs).exists():
            new_abs = old_abs
        else:
            return m.group(0)
        new_rel = os.path.relpath(new_abs, os.path.dirname(path)).replace(os.sep, "/")
        if new_rel == dst:
            return m.group(0)
        n_dst += 1
        # Replace only inside the (...) so a label that looks like the old path
        # is not clobbered; fix such a label explicitly instead.
        out = m.group(0)
        open_paren = out.index("](") + 2
        out = out[:open_paren] + out[open_paren:].replace(dst, new_rel, 1)
        label = m.group("label")
        if label == dst:
            out = out.replace(f"[{label}]", f"[{new_rel}]", 1)
            n_label += 1
        return out

    new_text = LINK_RE.sub(sub, text)
    if n_dst:
        (ROOT / path).write_bytes(new_text.encode("utf-8"))
    return n_dst, n_label


def do_links() -> None:
    targets = [p for p in git_ls_files("*.md", "*.html")
               if not p.startswith("docs/topics/html/")]
    total = labels = touched = 0
    for p in targets:
        a, b = rewrite(p)
        if a:
            touched += 1
            total += a
            labels += b
    print(f"rewrote {total} links ({labels} labels) across {touched} files")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["move", "links"], required=True)
    args = ap.parse_args()
    if args.phase == "move":
        preflight()
        save()
        do_move()
    else:
        load()
        do_links()
    return 0


if __name__ == "__main__":
    sys.exit(main())
