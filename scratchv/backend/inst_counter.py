"""RISC-V Instruction Counter.

Parses RISC-V assembly text and counts instructions by category,
producing text tables, matplotlib charts, and HTML reports.

Usage as module::

    from scratchv.backend.inst_counter import count_instructions
    stats = count_instructions(asm_text)
    print(stats)  # dict: {category: count}

Usage as CLI::

    python -m scratchv.backend.inst_counter file1.s file2.s --chart output.png
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Instruction categories
# ---------------------------------------------------------------------------

# Map each RISC-V mnemonic to a category
_OPCODE_CATEGORIES: dict[str, str] = {
    # ALU (integer arithmetic and logic)
    "add": "ALU", "sub": "ALU", "sll": "ALU", "srl": "ALU",
    "sra": "ALU", "xor": "ALU", "or": "ALU", "and": "ALU",
    "slt": "ALU", "sltu": "ALU",
    "addi": "ALU", "slli": "ALU", "srli": "ALU", "srai": "ALU",
    "xori": "ALU", "ori": "ALU", "andi": "ALU",
    "slti": "ALU", "sltiu": "ALU",
    "mul": "ALU", "mulh": "ALU", "mulhsu": "ALU", "mulhu": "ALU",
    "div": "ALU", "divu": "ALU", "rem": "ALU", "remu": "ALU",
    "lui": "ALU", "auipc": "ALU",
    # Memory
    "lw": "MEM", "lh": "MEM", "lb": "MEM",
    "lbu": "MEM", "lhu": "MEM", "lwu": "MEM", "ld": "MEM",
    "sw": "MEM", "sh": "MEM", "sb": "MEM", "sd": "MEM",
    "flw": "MEM", "fsw": "MEM", "fld": "MEM", "fsd": "MEM",
    # Branches
    "beq": "BRANCH", "bne": "BRANCH", "blt": "BRANCH",
    "bge": "BRANCH", "bltu": "BRANCH", "bgeu": "BRANCH",
    "beqz": "BRANCH", "bnez": "BRANCH", "blez": "BRANCH",
    "bgtz": "BRANCH", "bltz": "BRANCH", "bgez": "BRANCH",
    # Jumps
    "j": "JUMP", "jal": "JUMP", "jalr": "JUMP", "ret": "JUMP",
    "jr": "JUMP",
    # Pseudo-instructions
    "li": "PSEUDO", "mv": "PSEUDO", "not": "PSEUDO",
    "neg": "PSEUDO", "seqz": "PSEUDO", "snez": "PSEUDO",
    "call": "PSEUDO", "nop": "PSEUDO", "la": "PSEUDO",
    "tail": "PSEUDO",
    # MISC (everything else: directives, custom, environment)
}

# Default set of category names in display order
_CATEGORY_ORDER = ["ALU", "MEM", "BRANCH", "JUMP", "PSEUDO", "MISC"]


# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------

_INST_RE = re.compile(
    r'^\s*'
    r'(?:[A-Za-z_.][A-Za-z0-9_.]*:\s*)?'   # optional label
    r'(?P<opcode>[a-zA-Z][a-zA-Z0-9.]*)'     # required opcode
    r'\b'
)

# Assembler directives that are not instructions (keep original dot prefix)
_DIRECTIVES = {
    ".text", ".data", ".bss", ".rodata", ".section",
    ".globl", ".global", ".type", ".size", ".align",
    ".file", ".loc", ".cfi_startproc", ".cfi_endproc",
    ".cfi_def_cfa", ".cfi_offset", ".cfi_restore",
    ".byte", ".word", ".dword", ".half", ".quad",
    ".string", ".asciz", ".ascii", ".zero", ".skip",
    ".balign", ".p2align", ".option", ".set",
}


def _extract_opcode(line: str) -> Optional[str]:
    """Extract the opcode mnemonic from an assembly line.

    Returns None for empty lines, pure comments, labels, and directives.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    # Remove comment portion
    code_part = stripped.split("#")[0].strip()
    if not code_part:
        return None

    # Check if the whole line is a directive
    if any(code_part.lower().startswith(d) for d in _DIRECTIVES):
        return None

    # Split into tokens
    tokens = code_part.replace(",", " ").split()
    if not tokens:
        return None

    first = tokens[0].lower()

    # Skip labels (line ends with ':')
    if first.endswith(":"):
        return None

    # Skip assembler directives that are not instructions
    if first.startswith(".") or first.startswith("cfi_"):
        return None

    return first


def _classify_opcode(opcode: str) -> str:
    """Return the category name for an opcode mnemonic."""
    return _OPCODE_CATEGORIES.get(opcode, "MISC")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_instructions(asm_text: str) -> dict[str, int]:
    """Count RISC-V instructions by category.

    Parameters
    ----------
    asm_text:
        Raw RISC-V assembly text.

    Returns
    -------
    Dictionary mapping category name to instruction count.
    All six standard categories (ALU, MEM, BRANCH, JUMP, PSEUDO, MISC)
    are guaranteed to be present, even if their count is 0.
    """
    counts: dict[str, int] = {cat: 0 for cat in _CATEGORY_ORDER}
    detailed: Counter[str] = Counter()

    for line in asm_text.split("\n"):
        opcode = _extract_opcode(line)
        if opcode is None:
            continue
        category = _classify_opcode(opcode)
        if category in counts:
            counts[category] += 1
        else:
            counts["MISC"] += 1
        detailed[opcode] += 1

    # Attach detailed counts as a separate attribute-like key
    counts["_detailed"] = detailed  # type: ignore[assignment]

    return counts


def count_instructions_file(filepath: str) -> dict[str, int]:
    """Count instructions from an assembly file on disk."""
    with open(filepath, "r") as f:
        return count_instructions(f.read())


# ---------------------------------------------------------------------------
# Text table output
# ---------------------------------------------------------------------------

def format_table(counts: dict[str, int]) -> str:
    """Format instruction count results as a text table.

    Parameters
    ----------
    counts:
        Result dict from ``count_instructions``.

    Returns
    -------
    Formatted multi-line string suitable for printing.
    """
    total = sum(v for k, v in counts.items()
                if not k.startswith("_") and isinstance(v, int))
    detailed = counts.get("_detailed", Counter())

    lines = []
    lines.append("=" * 60)
    lines.append("RISC-V Instruction Statistics")
    lines.append("=" * 60)
    lines.append(f"{'Category':<10} {'Count':>8} {'Percent':>10} {'Bar'}")
    lines.append("-" * 60)

    for cat in _CATEGORY_ORDER:
        cnt = counts.get(cat, 0)
        pct = (cnt / total * 100) if total > 0 else 0.0
        bar = "#" * max(1, int(pct / 2))
        lines.append(f"{cat:<10} {cnt:>8} {pct:>9.1f}% {bar}")

    lines.append("-" * 60)
    lines.append(f"{'TOTAL':<10} {total:>8}")
    lines.append("")

    # Detailed per-instruction breakdown
    if detailed:
        lines.append("-" * 60)
        lines.append("Per-instruction Breakdown")
        lines.append("-" * 60)
        for op, cnt in detailed.most_common():
            cat = _classify_opcode(op)
            lines.append(f"  {op:<12} {cnt:>6}  ({cat})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_chart(counts: dict[str, int], output_path: str,
                   title: str = "RISC-V Instruction Distribution") -> None:
    """Generate a bar chart of instruction categories using matplotlib.

    Parameters
    ----------
    counts:
        Result dict from ``count_instructions``.
    output_path:
        File path for the output image (e.g. 'chart.png').
    title:
        Chart title.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for chart generation. "
            "Install it with: pip install matplotlib"
        )

    categories = [c for c in _CATEGORY_ORDER if counts.get(c, 0) > 0]
    values = [counts.get(c, 0) for c in categories]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    colors = [
        "#4C72B0", "#55A868", "#C44E52",
        "#8172B2", "#CCB974", "#64B5CD",
    ]
    bar_colors = [colors[i % len(colors)] for i in range(len(categories))]
    ax1.bar(
        categories, values, color=bar_colors,
        edgecolor="white", linewidth=0.8,
    )
    ax1.set_title(title)
    ax1.set_ylabel("Instruction Count")
    for i, v in enumerate(values):
        ax1.text(
            i, v + max(values) * 0.01, str(v),
            ha="center", fontsize=9,
        )

    # Pie chart
    wedges, texts, autotexts = ax2.pie(
        values, labels=categories, autopct="%1.1f%%",
        colors=bar_colors, startangle=90,
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title("Instruction Distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_multi_chart(
        multi_counts: dict[str, dict[str, int]],
        output_path: str,
        title: str = "Multi-file Instruction Comparison",
) -> None:
    """Generate a grouped bar chart comparing multiple files.

    Parameters
    ----------
    multi_counts:
        Dict mapping file label to counts dict.
    output_path:
        Output image path.
    title:
        Chart title.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for chart generation.")

    files = list(multi_counts.keys())
    categories = [c for c in _CATEGORY_ORDER
                  if any(multi_counts[f].get(c, 0) > 0 for f in files)]

    x = range(len(categories))
    width = 0.8 / len(files)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [
        "#4C72B0", "#55A868", "#C44E52",
        "#8172B2", "#CCB974", "#64B5CD",
    ]

    for i, fname in enumerate(files):
        values = [multi_counts[fname].get(c, 0) for c in categories]
        offset = [xi + i * width for xi in x]
        ax.bar(
            offset, values, width, label=fname,
            color=colors[i % len(colors)],
            edgecolor="white", linewidth=0.5,
        )

    ax.set_title(title)
    ax.set_ylabel("Instruction Count")
    ax.set_xticks([xi + width * (len(files) - 1) / 2 for xi in x])
    ax.set_xticklabels(categories)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         max-width: 900px; margin: 2em auto; padding: 0 1em; }}
  h1 {{ color: #333;
        border-bottom: 2px solid #4C72B0; padding-bottom: 0.3em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{
    padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
  th {{ background: #f5f5f5; }}
  .bar {{ display: inline-block; height: 1em; background: #4C72B0;
          border-radius: 2px; vertical-align: middle; }}
  .section {{ margin: 2em 0; }}
  .summary {{ font-size: 1.2em; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="summary"><strong>Total instructions:</strong> {total}</div>
<div class="section">
<table>
<tr><th>Category</th><th>Count</th><th>Percent</th><th>Distribution</th></tr>
{table_rows}
</table>
</div>
<div class="section">
<h2>Per-instruction Breakdown</h2>
<table>
<tr><th>Instruction</th><th>Count</th><th>Category</th></tr>
{detail_rows}
</table>
</div>
</body>
</html>
"""


def generate_html_report(
        counts: dict[str, int],
        output_path: str,
        title: str = "RISC-V Instruction Statistics",
) -> None:
    """Generate an HTML report file from instruction counts.

    Parameters
    ----------
    counts:
        Result dict from ``count_instructions``.
    output_path:
        Path to write the HTML file.
    title:
        Page title.
    """
    total = sum(v for k, v in counts.items()
                if not k.startswith("_") and isinstance(v, int))
    detailed = counts.get("_detailed", Counter())

    table_rows = []
    for cat in _CATEGORY_ORDER:
        cnt = counts.get(cat, 0)
        pct = (cnt / total * 100) if total > 0 else 0.0
        bar_w = int(pct * 3)
        table_rows.append(
            f"<tr><td>{cat}</td><td>{cnt}</td><td>{pct:.1f}%</td>"
            f"<td><span class='bar' style='width:{bar_w}px'></span></td></tr>"
        )

    detail_rows = []
    for op, cnt in detailed.most_common():
        cat = _classify_opcode(op)
        detail_rows.append(
            f"<tr><td><code>{html.escape(op)}</code></td>"
            f"<td>{cnt}</td><td>{cat}</td></tr>"
        )

    html_content = _HTML_TEMPLATE.format(
        title=html.escape(title),
        total=total,
        table_rows="\n".join(table_rows),
        detail_rows="\n".join(detail_rows),
    )

    with open(output_path, "w") as f:
        f.write(html_content)


# ---------------------------------------------------------------------------
# Multi-file comparison
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Result of comparing instruction counts across files."""
    files: list[str] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    diffs: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dataframe(self):  # type: ignore[name-defined]  # noqa: F821
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd
            rows = []
            for fname in self.files:
                row = {"File": fname}
                row.update({
                    c: self.counts[fname].get(c, 0)
                    for c in _CATEGORY_ORDER
                })
                rows.append(row)
            return pd.DataFrame(rows)
        except ImportError:
            raise ImportError(
                "pandas is required for DataFrame conversion."
            )


def compare_files(filepaths: list[str]) -> ComparisonResult:
    """Compare instruction counts across multiple assembly files.

    Parameters
    ----------
    filepaths:
        List of assembly file paths.

    Returns
    -------
    ComparisonResult with per-file counts and per-category differences.
    """
    files = [os.path.basename(p) for p in filepaths]
    counts = {}
    for label, path in zip(files, filepaths):
        counts[label] = count_instructions_file(path)

    # Compute diffs relative to first file
    diffs: dict[str, dict[str, int]] = {}
    if len(files) > 1:
        baseline = counts[files[0]]
        for f in files[1:]:
            diffs[f] = {}
            for cat in _CATEGORY_ORDER:
                diffs[f][cat] = counts[f].get(cat, 0) - baseline.get(cat, 0)

    return ComparisonResult(files=files, counts=counts, diffs=diffs)


def print_comparison(result: ComparisonResult) -> None:
    """Pretty-print a comparison result to stdout."""
    print("=" * 80)
    print("RISC-V Instruction Count Comparison")
    print("=" * 80)

    header = f"{'Category':<10}"
    for f in result.files:
        header += f" {f:>12}"
    if result.diffs:
        header += " " * 13 + "".join(
            f" {f:>12}" for f in result.diffs.keys()
        )
    print(header)
    print("-" * 80)

    for cat in _CATEGORY_ORDER:
        line = f"{cat:<10}"
        for f in result.files:
            line += f" {result.counts[f].get(cat, 0):>12}"
        for f, d in result.diffs.items():
            val = d.get(cat, 0)
            sign = "+" if val > 0 else ""
            line += f" {sign}{val:>11}"
        print(line)

    print("-" * 80)
    for f in result.files:
        total = sum(v for k, v in result.counts[f].items()
                    if not k.startswith("_") and isinstance(v, int))
        print(f"{f:>10} total: {total}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "RISC-V Instruction Counter - count and visualize "
            "instructions in assembly files"
        ),
    )
    parser.add_argument(
        "files", nargs="+",
        help="Assembly file(s) to analyze (.s)",
    )
    parser.add_argument(
        "--chart", type=str, default=None,
        help="Generate bar/pie chart and save to this path",
    )
    parser.add_argument(
        "--html", type=str, default=None,
        help="Generate HTML report and save to this path",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare multiple files side-by-side",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-instruction breakdown",
    )

    args = parser.parse_args(argv)

    if args.compare and len(args.files) > 1:
        result = compare_files(args.files)
        print_comparison(result)
        return

    # Single-file or sequential mode
    multi: dict[str, dict[str, int]] = {}
    for fpath in args.files:
        if not os.path.exists(fpath):
            print(f"Error: file not found: {fpath}", file=sys.stderr)
            sys.exit(1)
        counts = count_instructions_file(fpath)
        label = os.path.basename(fpath)
        multi[label] = counts
        print(format_table(counts))

    # Charts
    if args.chart and len(multi) == 1:
        counts = list(multi.values())[0]
        generate_chart(counts, args.chart)
        print(f"Chart saved to {args.chart}")
    elif args.chart and len(multi) > 1:
        generate_multi_chart(multi, args.chart)
        print(f"Comparison chart saved to {args.chart}")

    # HTML report
    if args.html:
        if len(multi) == 1:
            counts = list(multi.values())[0]
            generate_html_report(counts, args.html)
        else:
            # For multi-file, use the first file's data
            counts = multi[list(multi.keys())[0]]
            generate_html_report(counts, args.html)
        print(f"HTML report saved to {args.html}")


if __name__ == "__main__":
    main()
