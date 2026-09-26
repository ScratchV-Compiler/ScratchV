#!/usr/bin/env python3
"""Validate the llm-deploy document deliverables.

Backs the ``docs:interfaces`` and ``docs:risks`` CI gates. Kept as a script
rather than inline shell so it can be run locally before pushing.

    python3 scripts/check_llm_deploy_docs.py
    python3 scripts/check_llm_deploy_docs.py --week W2 \
        --interfaces docs/llm-deploy-v1.0/W2/interfaces.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The four interface contracts the plan requires interfaces.md to cover.
INTERFACE_MARKERS = [
    ("接口一", "前端 (ONNX -> IR)"),
    ("接口二", "IR 数据结构"),
    ("接口三", "后端 (IR -> RISC-V)"),
    ("接口四", "运行时 FFI"),
]

MIN_RISKS = 15


def read(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_bytes().decode("utf-8", "replace")


def display(path: Path) -> str:
    """Repo-relative when possible; absolute otherwise (paths outside the repo
    are legal input, so relative_to must not be allowed to raise)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def check_interfaces(path: Path) -> list[str]:
    problems: list[str] = []
    text = read(path)
    if text is None:
        return [f"接口文档不存在: {display(path)}"]

    for marker, label in INTERFACE_MARKERS:
        if marker not in text:
            problems.append(f"缺少「{marker}」（{label}）")
    if "变更记录" not in text:
        problems.append("缺少「变更记录」小节（冻结规则要求）")
    if not re.search(r"v\d+\.\d+", text):
        problems.append("未标注版本号（如 v1.0）")

    return problems


def _table_cells(line: str) -> list[str]:
    """Split a markdown table row into cells (leading/trailing pipes dropped)."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    """True for the |---|---| row that follows a markdown table header."""
    t = line.strip()
    return bool(t) and set(t) <= set("|-: ")


def check_risks(path: Path) -> tuple[list[str], int]:
    problems: list[str] = []
    text = read(path)
    if text is None:
        return [f"风险清单不存在: {display(path)}"], 0

    lines = text.split("\n")

    # Walk the document tracking which table we are inside, so a risk row is
    # only judged against its own table's Plan B column. The document also has
    # an evidence table keyed by the same R<n> ids; those rows have no Plan B
    # column and must not be reported as missing their mitigation.
    seen: set[str] = set()
    thin: list[str] = []
    planb_col: int | None = None          # None => current table has no Plan B

    for line_no, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            planb_col = None              # left the table entirely
            continue

        nxt = lines[line_no + 1] if line_no + 1 < len(lines) else ""
        if _is_separator(nxt):            # this line is a table header
            planb_col = None
            if "Plan B" in line:
                for idx, cell in enumerate(_table_cells(line)):
                    if "Plan B" in cell:
                        planb_col = idx
                        break
            continue

        m = re.match(r"^\|\s*(R\d+)\s*\|", line)
        if not m or planb_col is None:
            continue

        rid = m.group(1)
        seen.add(rid)
        cells = _table_cells(line)
        if planb_col >= len(cells) or not cells[planb_col]:
            thin.append(rid)

    count = len(seen)
    if count < MIN_RISKS:
        problems.append(f"风险条数 {count} < 要求 {MIN_RISKS}")
    if thin:
        problems.append(f"以下风险缺少 Plan B 内容: {', '.join(sorted(set(thin)))}")

    return problems, count


def main() -> int:
    ap = argparse.ArgumentParser(description="Check llm-deploy document gates")
    ap.add_argument("--week", default="W1", help="week directory name")
    ap.add_argument("--interfaces", default=None)
    ap.add_argument("--risks", default=None)
    args = ap.parse_args()

    base = ROOT / "docs/llm-deploy-v1.0" / args.week
    iface = Path(args.interfaces) if args.interfaces else base / "interfaces.md"
    risks = Path(args.risks) if args.risks else base / "risks.md"

    failed = False

    problems = check_interfaces(iface)
    if problems:
        failed = True
        print(f"FAIL docs:interfaces ({display(iface)})", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
    else:
        print(f"OK   docs:interfaces  {display(iface)} — 四类接口齐全，有版本号与变更记录")

    problems, count = check_risks(risks)
    if problems:
        failed = True
        print(f"FAIL docs:risks ({display(risks)})", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
    else:
        print(f"OK   docs:risks       {display(risks)} — {count} 条风险，均有 Plan B")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
