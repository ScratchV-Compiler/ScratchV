#!/usr/bin/env python3
"""One-off migration: reorganise docs/ into per-topic folders.

Two phases, run in order:

    python3 scripts/migrate_docs_layout.py --phase move    # git mv only
    python3 scripts/migrate_docs_layout.py --phase links   # rewrite md links

The MOVES table below is the single source of truth: it drives the git mv
operations, the old->new path map used for link rewriting, and the preflight
checks.  Kept in-tree only for the duration of the migration; delete after.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# ── Topic stems: the 30 course guides under docs/topics/ ─────────────────────
TOPIC_STEMS = [
    "01-DSL前端增强器", "02-ONNX解析器", "03-IR系统", "04-IR优化器框架",
    "05-汇编代码美化器", "06-性能基准套件", "07-编译器日志增强器", "08-指令选择",
    "09-DSL错误提示美化器", "10-循环展开优化", "11-控制流图生成器", "12-指令计数统计器",
    "13-窥孔优化器", "14-常量加载合并", "15-函数内联", "16-LLVM代码生成",
    "17-寄存器分配", "18-指令调度器", "19-Standalone-RISC-V编译器", "20-代码规范",
    "21-IR验证器", "22-Standalone-LLVM编译器", "23-Cache模型", "24-Spike仿真",
    "25-LLVM对比工具", "26-TinyFive对比", "27-RV32全量Benchmark", "28-扩展指令选择",
    "29-SIMD向量化", "30-CI-Dashboard",
]

FOUNDATION_STEMS = [
    "00-环境搭建指南", "01-编译器概念入门", "02-快速上手教程",
    "03-指标解读指南", "04-故障排除FAQ",
]

# Topic folder for a guide stem, e.g. "13-窥孔优化器" -> "docs/topics/13-窥孔优化器"
def _topic_dir(stem: str) -> str:
    return f"docs/topics/{stem}"


def _build_moves() -> list[tuple[str, str]]:
    moves: list[tuple[str, str]] = []

    # (1) 30 course guides -> one folder each
    for stem in TOPIC_STEMS:
        moves.append((f"docs/topics/{stem}.md", f"{_topic_dir(stem)}/{stem}.md"))

    # (2) 5 foundation guides -> docs/guide/
    for stem in FOUNDATION_STEMS:
        moves.append((f"docs/{stem}.md", f"docs/guide/{stem}.md"))

    # (3) archive proposals -> docs/archive/提案/
    # Tolerate either side of the move: pre-move we glob the old directory,
    # post-move the old one is gone and we read the new one instead.
    old_arc = DOCS / "topics/archive"
    new_arc = DOCS / "archive/提案"
    if old_arc.is_dir():
        for p in sorted(old_arc.iterdir()):
            if p.is_file():
                moves.append((f"docs/topics/archive/{p.name}", f"docs/archive/提案/{p.name}"))
    elif new_arc.is_dir():
        for p in sorted(new_arc.iterdir()):
            if p.is_file():
                moves.append((f"docs/topics/archive/{p.name}", f"docs/archive/提案/{p.name}"))
    else:
        raise SystemExit("cannot locate the archive directory on either side of the move")

    # (4) topic attachments still sitting in docs/topics/ -> their topic folder
    moves += [
        ("docs/topics/01-DSL前端增强器-设计文档.md",
         f"{_topic_dir('01-DSL前端增强器')}/01-DSL前端增强器-设计文档.md"),
        ("docs/topics/01-DSL前端增强器-开发文档.md",
         f"{_topic_dir('01-DSL前端增强器')}/01-DSL前端增强器-开发文档.md"),
        ("docs/topics/06-性能测试套件使用说明.md",
         f"{_topic_dir('06-性能基准套件')}/06-性能测试套件使用说明.md"),
        ("docs/topics/06-性能测试套件设计文档.md",
         f"{_topic_dir('06-性能基准套件')}/06-性能测试套件设计文档.md"),
        ("docs/topics/09-DSL错误提示美化器-设计文档.md",
         f"{_topic_dir('09-DSL错误提示美化器')}/09-DSL错误提示美化器-设计文档.md"),
        ("docs/topics/09-DSL错误提示美化器-开发文档.md",
         f"{_topic_dir('09-DSL错误提示美化器')}/09-DSL错误提示美化器-开发文档.md"),
        ("docs/topics/09-DSL诊断-CI与Benchmark.md",
         f"{_topic_dir('09-DSL错误提示美化器')}/09-DSL错误提示美化器-CI与Benchmark.md"),
        ("docs/topics/13-窥孔优化器-设计文档.md",
         f"{_topic_dir('13-窥孔优化器')}/13-窥孔优化器-设计文档.md"),
        ("docs/topics/06-双后端未通过用例分析.md",
         "docs/reports/06-双后端未通过用例分析.md"),
    ]

    # (5) scattered attachments from the repo root and docs/ root
    moves += [
        # topic 13 — root dropped files
        ("窥孔优化器设计文档.md",
         f"{_topic_dir('13-窥孔优化器')}/13-窥孔优化器-技术设计文档.md"),
        ("窥孔优化器开发文档.md",
         f"{_topic_dir('13-窥孔优化器')}/13-窥孔优化器-开发文档.md"),
        ("topic13/README.md",
         f"{_topic_dir('13-窥孔优化器')}/13-窥孔优化器-完成目录.md"),
        ("docs/feat13#suai/设计文档.md",
         f"{_topic_dir('13-窥孔优化器')}/13-窥孔优化器-Benchmark设计文档.md"),
        ("docs/feat13#suai/开发文档.md",
         f"{_topic_dir('13-窥孔优化器')}/13-窥孔优化器-Benchmark开发文档.md"),
        # topic 05
        ("docs/topic5汇编代码美化器设计文档.md",
         f"{_topic_dir('05-汇编代码美化器')}/05-汇编代码美化器-设计文档.md"),
        ("docs/topic5汇编代码美化器开发文档.md",
         f"{_topic_dir('05-汇编代码美化器')}/05-汇编代码美化器-开发文档.md"),
        # topic 11 — CFG
        ("docs/CFG_Design.md",
         f"{_topic_dir('11-控制流图生成器')}/11-控制流图生成器-设计文档.md"),
        ("docs/CFG_Dev.md",
         f"{_topic_dir('11-控制流图生成器')}/11-控制流图生成器-开发文档.md"),
        # topic 14 — drafts
        ("docs/课题14-常量加载合并优化-技术设计文档初稿.md",
         f"{_topic_dir('14-常量加载合并')}/14-常量加载合并-设计文档-初稿.md"),
        ("docs/课题14-常量加载合并优化-开发文档初稿.md",
         f"{_topic_dir('14-常量加载合并')}/14-常量加载合并-开发文档-初稿.md"),
        # topic 17 — current versions stay with the topic, superseded ones archive
        ("docs/topic17_v1.5设计文档.md",
         f"{_topic_dir('17-寄存器分配')}/17-寄存器分配-设计文档-v1.5.md"),
        ("docs/topic17_v1.5开发文档.md",
         f"{_topic_dir('17-寄存器分配')}/17-寄存器分配-开发文档-v1.5.md"),
        ("docs/topic17_benchmark文档.md",
         f"{_topic_dir('17-寄存器分配')}/17-寄存器分配-Benchmark设计文档.md"),
        ("docs/topic17_v1.3设计文档.md",
         "docs/archive/旧版/17-寄存器分配-设计文档-v1.3.md"),
        ("docs/topic17_v1.3开发文档.md",
         "docs/archive/旧版/17-寄存器分配-开发文档-v1.3.md"),
        # topic 17 — reports
        ("docs/topic17_AI自审报告.md", "docs/reports/17-寄存器分配-AI自审报告.md"),
        ("docs/topic17_P1实现报告.md", "docs/reports/17-寄存器分配-P1实现报告.md"),
        ("docs/topic17_伪指令溢出统计排期.md",
         "docs/reports/17-寄存器分配-伪指令溢出统计排期.md"),
        ("benchmark_reports/topic17_fix_review.md",
         "docs/reports/17-寄存器分配-修改测试自查报告.md"),
        # reference material
        ("docs/ARCHITECTURE.md", "docs/reference/ARCHITECTURE.md"),
        ("docs/developer_guide.md", "docs/reference/developer_guide.md"),
        ("docs/optimization_guide.md", "docs/reference/optimization_guide.md"),
        ("docs/verification.md", "docs/reference/verification.md"),
        ("docs/CODING_STANDARDS.md", "docs/reference/CODING_STANDARDS.md"),
        # plans
        ("docs/superpowers/plans/2026-08-12-dsl-diagnostics-implementation.md",
         "docs/plans/2026-08-12-dsl-diagnostics-implementation.md"),
        ("docs/superpowers/plans/2026-09-19-topic01-ci-benchmark.md",
         "docs/plans/2026-09-19-topic01-ci-benchmark.md"),
        # promo + archived landing page
        ("docs/ScratchV.md", "docs/promo/ScratchV.md"),
        ("docs/ScratchV.html", "docs/promo/ScratchV.html"),
        ("docs/index.html", "docs/archive/旧版/课程首页-index.html"),
    ]
    return moves


MOVES = _build_moves()
PATH_MAP = {src: dst for src, dst in MOVES}

# The move phase writes the table here so the links phase can reuse it even
# though several sources no longer exist on disk by then.
SIDECAR = Path("/tmp/docs_moves.json")


def save_moves() -> None:
    import json
    SIDECAR.write_text(json.dumps(MOVES, ensure_ascii=False), encoding="utf-8")


def load_moves() -> None:
    """Prefer the recorded table — it is the authoritative post-move mapping."""
    global MOVES, PATH_MAP
    if not SIDECAR.is_file():
        return
    import json
    recorded = [tuple(x) for x in json.loads(SIDECAR.read_text(encoding="utf-8"))]
    if len(recorded) == len(MOVES) and set(recorded) == set(MOVES):
        return  # rebuilt table agrees; nothing to do
    MOVES = recorded
    PATH_MAP = {src: dst for src, dst in MOVES}

# docs/INDEX.md deliberately keeps its name: renaming it to README.md would
# collide with the repo-root README.md and the topic13 index under the
# generator's basename-based link matching.
KEEP_INDEX_MD = "docs/INDEX.md"


# ── git helpers ─────────────────────────────────────────────────────────────

def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check,
        capture_output=True, text=True,
    )


def git_ls_files(*patterns: str) -> list[str]:
    """Tracked paths, NUL-separated so non-ASCII names survive core.quotepath."""
    out = git("ls-files", "-z", *patterns).stdout
    return [p for p in out.split("\0") if p]


# ── preflight ───────────────────────────────────────────────────────────────

def preflight() -> None:
    srcs = [s for s, _ in MOVES]
    dsts = [d for _, d in MOVES]

    assert len(set(srcs)) == len(srcs), f"duplicate sources: {_dups(srcs)}"
    assert len(set(dsts)) == len(dsts), f"target path conflict: {_dups(dsts)}"

    for s in srcs:
        assert (ROOT / s).is_file(), f"source missing: {s}"
        r = git("ls-files", "--error-unmatch", "--", s, check=False)
        assert r.returncode == 0, f"source not tracked by git: {s}"
    for d in dsts:
        assert not (ROOT / d).exists(), f"target already exists: {d}"

    # file-count conservation
    before = len(git_ls_files())
    assert before > 0

    # Final-state basename uniqueness, scoped to the md/html files the
    # generator resolves links through (_md_link_to_html matches on basename).
    # INDEX.md is the one known pair: docs/INDEX.md + docs/topics/INDEX.md,
    # which the generator's "topics/INDEX" special case already disambiguates.
    final = set(dsts) | (set(git_ls_files()) - set(srcs))
    final = {
        p for p in final
        if p.endswith((".md", ".html"))
        and (p.startswith("docs/") or "/" not in p)
        and not p.startswith("docs/topics/html/")
    }
    dup_names = {n for n, c in Counter(Path(p).name for p in final).items() if c > 1}
    dup_names.discard("INDEX.md")
    assert not dup_names, f"basename not unique in final state: {sorted(dup_names)}"

    print(f"preflight OK: {len(MOVES)} moves, {before} tracked files before")


def _dups(items: list[str]) -> list[str]:
    return sorted(n for n, c in Counter(items).items() if c > 1)


# ── phase: move ─────────────────────────────────────────────────────────────

def do_move() -> None:
    for src, dst in MOVES:
        (ROOT / dst).parent.mkdir(parents=True, exist_ok=True)
        # list-form argv: '#' and non-ASCII names never touch a shell
        git("mv", "--", src, dst)

    # git does not track directories; prune the ones we emptied
    for src, _ in MOVES:
        d = (ROOT / src).parent
        while d != DOCS and d != ROOT and d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            d = d.parent

    print(f"moved {len(MOVES)} files")


# ── phase: links ────────────────────────────────────────────────────────────

LINK_RE = re.compile(
    r"""
      \]\(\s*<?(?P<md>[^)\s<>]+\.(?:md|html))(?P<md_a>\#[^)\s>]*)?>?\s*\)
    | href="(?P<href>[^"]+\.(?:md|html))(?P<href_a>\#[^"]*)?"
    | ^\[[^\]]+\]:\s*(?P<ref>\S+\.(?:md|html))(?P<ref_a>\#\S*)?
    """,
    re.X | re.M,
)

SKIP_PREFIXES = ("http://", "https://", "mailto:", "data:", "/", "#")


def rewrite_targets() -> list[str]:
    """Every tracked md/html file that may contain links to moved files."""
    targets = [
        p for p in git_ls_files("*.md", "*.html")
        if not p.startswith("docs/topics/html/")
    ]
    return targets


def rewrite_file(path: str) -> int:
    """Rewrite relative links in `path` (its post-move location)."""
    new_src = path
    old_src = next((s for s, d in MOVES if d == path), path)

    raw = (ROOT / new_src).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return 0

    changed = 0

    def sub(m: re.Match) -> str:
        nonlocal changed
        dst = m.group("md") or m.group("href") or m.group("ref")
        anchor = m.group("md_a") or m.group("href_a") or m.group("ref_a") or ""
        if dst is None or dst.startswith(SKIP_PREFIXES):
            return m.group(0)

        old_abs = os.path.normpath(os.path.join(os.path.dirname(old_src), dst))
        if old_abs in PATH_MAP:
            new_abs = PATH_MAP[old_abs]
        elif (ROOT / old_abs).exists():
            new_abs = old_abs
        else:
            return m.group(0)  # unresolvable (e.g. CI-generated) — leave alone

        new_rel = os.path.relpath(new_abs, os.path.dirname(new_src)).replace(os.sep, "/")
        if new_rel == dst:
            return m.group(0)
        changed += 1
        return m.group(0).replace(dst, new_rel, 1)

    out = LINK_RE.sub(sub, text)
    if changed:
        (ROOT / new_src).write_bytes(out.encode("utf-8"))
    return changed


def do_links() -> None:
    targets = rewrite_targets()
    total = 0
    touched = 0
    for p in targets:
        n = rewrite_file(p)
        if n:
            touched += 1
            total += n
            print(f"  {n:>3}  {p}")
    print(f"rewrote {total} links across {touched} files")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reorganise docs/ into per-topic folders")
    ap.add_argument("--phase", choices=["move", "links"], required=True)
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    if args.phase == "move":
        if not args.skip_preflight:
            preflight()
        save_moves()
        do_move()
    else:
        load_moves()
        assert len(MOVES) == 103, f"expected 103 moves, loaded {len(MOVES)}"
        do_links()


if __name__ == "__main__":
    main()
