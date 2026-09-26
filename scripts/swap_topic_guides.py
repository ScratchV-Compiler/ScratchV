#!/usr/bin/env python3
"""One-off: make the archived topic proposals the guides under docs/topics/.

For every topic that has a proposal in docs/archive/提案/, the proposal becomes
the topic's main document and the previous (much longer) guide is renamed to
NN-X-详细教程.md and kept alongside it.

    python3 scripts/swap_topic_guides.py --phase move
    python3 scripts/swap_topic_guides.py --phase links

Topics with no proposal (3, 8, 10, 15, 19, 23-27, 29, 30) are left alone.
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
DOCS = ROOT / "docs"
SIDECAR = Path("/tmp/swap_moves.json")

# English-named proposals that are short topic specs, keyed by topic number.
# The remaining English files (topicNN_*_guide.md, backend_const_merge.md,
# backend_inst_select_ext.md, backend_regalloc_linear.md) are longer module or
# implementation docs and stay in the archive.
ENGLISH_SPECS = {
    2: "frontend_onnx_parser.md",
    4: "optimizer_framework.md",
    16: "backend_llvm_codegen.md",
    22: "standalone_llvm_compiler.md",
}


def topic_dirs() -> dict[int, str]:
    """{topic number: 'docs/topics/NN-名字'}"""
    out = {}
    for d in sorted((DOCS / "topics").iterdir()):
        if d.is_dir() and re.match(r"^\d{2}-", d.name):
            out[int(d.name[:2])] = f"docs/topics/{d.name}"
    return out


def build_moves() -> list[tuple[str, str]]:
    """[(old path, new path)] covering both halves of every swap."""
    moves: list[tuple[str, str]] = []
    proposals = DOCS / "archive/提案"
    for num, tdir in topic_dirs().items():
        stem = Path(tdir).name                       # "13-窥孔优化器"
        main = f"{tdir}/{stem}.md"                   # current guide
        if not (ROOT / main).is_file():
            continue                                 # already swapped

        zh = [p for p in proposals.glob(f"课题{num}：*.md")]
        if zh:
            src = f"docs/archive/提案/{zh[0].name}"
        elif num in ENGLISH_SPECS:
            src = f"docs/archive/提案/{ENGLISH_SPECS[num]}"
        else:
            continue                                 # no proposal -> leave as-is

        if not (ROOT / src).is_file():
            continue
        moves.append((main, f"{tdir}/{stem}-详细教程.md"))
        moves.append((src, main))
    return moves


MOVES = build_moves()
PATH_MAP = {src: dst for src, dst in MOVES}


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, check=check,
                          capture_output=True, text=True)


def git_ls_files(*patterns: str) -> list[str]:
    out = git("ls-files", "-z", *patterns).stdout
    return [p for p in out.split("\0") if p]


def save() -> None:
    SIDECAR.write_text(json.dumps(MOVES, ensure_ascii=False), encoding="utf-8")


def load() -> None:
    global MOVES, PATH_MAP
    if not SIDECAR.is_file():
        raise SystemExit("run --phase move first")
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
    # A target may legitimately be another move's source -- this is a swap, and
    # that source vacates first.  Only reject targets nothing else frees up.
    for d in dsts:
        if d in set(srcs):
            continue
        assert not (ROOT / d).exists(), f"target exists and is not vacated: {d}"
    print(f"preflight OK: {len(MOVES)} moves / {len(MOVES)//2} topics swapped")


def _dups(xs: list[str]) -> list[str]:
    return sorted(n for n, c in Counter(xs).items() if c > 1)


def do_move() -> None:
    # Order matters: a swap is A->B plus C->A, so the move that vacates A has to
    # run before the one that lands on it.  Resolve that here rather than relying
    # on the table's ordering.
    pending = list(MOVES)
    while pending:
        progressed = False
        for m in list(pending):
            src, dst = m
            if (ROOT / dst).exists():
                continue                       # still occupied; try after its vacator
            (ROOT / dst).parent.mkdir(parents=True, exist_ok=True)
            git("mv", "--", src, dst)
            pending.remove(m)
            progressed = True
        if not progressed:
            raise SystemExit(f"unstuck mid-swap: {[m[0] for m in pending]}")
    # Promote the leading '##' of the Chinese proposals to '#' -- they are now
    # the topic's top-level guide, and the site's progress JS keys off h1.
    promoted = 0
    for _, dst in MOVES:
        p = ROOT / dst
        if p.name.endswith("-详细教程.md"):
            continue
        raw = p.read_bytes().decode("utf-8")
        if raw.startswith("## ") and not raw.startswith("### "):
            p.write_bytes(raw.replace("## ", "# ", 1).encode("utf-8"))
            promoted += 1
    print(f"moved {len(MOVES)} files, promoted {promoted} headings to h1")


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

        out = m.group(0).replace(dst, new_rel, 1)
        n_dst += 1
        # A label that is itself the old path would now be misleading.
        label = m.group("label")
        if label is not None and label == dst:
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
            print(f"  {a:>3}  {p}")
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


if __name__ == "__main__":
    sys.exit(main())
