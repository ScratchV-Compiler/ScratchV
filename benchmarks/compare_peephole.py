#!/usr/bin/env python3
"""Compare assembly output with and without peephole optimization.

Usage:
    python benchmarks/compare_peephole.py
    python benchmarks/compare_peephole.py --json peephole_compare.json
    python benchmarks/compare_peephole.py --markdown peephole_compare.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scratchv.backend.asm_peephole import AsmPeepholeOptimizer  # noqa: E402
from scratchv.backend.inst_counter import count_instructions  # noqa: E402
from scratchv.compiler import CompilerConfig, CompilerDriver  # noqa: E402


def _total_static(counts: dict) -> int:
    return sum(v for k, v in counts.items() if not k.startswith("_"))


def _opcode_total(counts: dict, opcode: str) -> int:
    detailed = counts.get("_detailed", {})
    return detailed.get(opcode, 0)


@dataclass
class CaseCompare:
    name: str
    before_total: int
    after_total: int
    saved: int
    saved_pct: float
    peephole_changes: int
    rule_hits: dict[str, int] = field(default_factory=dict)
    before_opcodes: dict[str, int] = field(default_factory=dict)
    after_opcodes: dict[str, int] = field(default_factory=dict)


@dataclass
class CompareReport:
    cases: list[CaseCompare] = field(default_factory=list)

    @property
    def total_before(self) -> int:
        return sum(c.before_total for c in self.cases)

    @property
    def total_after(self) -> int:
        return sum(c.after_total for c in self.cases)

    @property
    def total_saved(self) -> int:
        return self.total_before - self.total_after

    @property
    def cases_with_savings(self) -> int:
        return sum(1 for c in self.cases if c.saved > 0)


def compile_dsl(path: Path, *, peephole: bool) -> str:
    driver = CompilerDriver(
        CompilerConfig(
            peephole_asm=peephole,
            optimize_level="none",
        )
    )
    out = path.with_suffix(".s")
    result = driver.compile(str(path), str(out))
    if not result.success:
        raise RuntimeError(f"compile failed: {path}: {result.errors}")
    return result.output_text


def compare_asm(name: str, asm_before: str) -> CaseCompare:
    opt = AsmPeepholeOptimizer()
    asm_after, changes = opt.optimize(asm_before)
    before_counts = count_instructions(asm_before)
    after_counts = count_instructions(asm_after)
    before_total = _total_static(before_counts)
    after_total = _total_static(after_counts)
    saved = before_total - after_total
    saved_pct = (saved / before_total * 100) if before_total else 0.0

    opcodes = ("addi", "li", "mv", "beq", "j", "jal", "ret")
    return CaseCompare(
        name=name,
        before_total=before_total,
        after_total=after_total,
        saved=saved,
        saved_pct=round(saved_pct, 2),
        peephole_changes=changes,
        rule_hits=dict(opt.total_matches),
        before_opcodes={op: _opcode_total(before_counts, op) for op in opcodes},
        after_opcodes={op: _opcode_total(after_counts, op) for op in opcodes},
    )


def run_dsl_suite(cases_dir: Path) -> CompareReport:
    report = CompareReport()
    for dsl in sorted(cases_dir.glob("*.dsl")):
        asm = compile_dsl(dsl, peephole=False)
        report.cases.append(compare_asm(dsl.stem, asm))
    return report


def run_synthetic_cases() -> list[CaseCompare]:
    from benchmarks.bench_asm_peephole import _gen_synthetic_asm

    results = []
    for size in (100, 500, 1000, 2000):
        asm = _gen_synthetic_asm(size, fusion_ratio=0.3)
        results.append(compare_asm(f"synthetic_{size}", asm))
    return results


def to_markdown(report: CompareReport, synthetic: list[CaseCompare]) -> str:
    lines = [
        "# 窥孔优化器 — 前后对比报告",
        "",
        "## 汇总",
        "",
        "| 指标 | 优化前 | 优化后 | 变化 |",
        "|------|--------|--------|------|",
        (
            f"| DSL 基准静态指令合计 | {report.total_before} | {report.total_after} | **-{report.total_saved}** ({report.total_saved/report.total_before*100:.2f}%) |"
            if report.total_before
            else ""
        ),
        f"| 有节省的用例 | {report.cases_with_savings} / {len(report.cases)} | — | — |",
        "",
        "## DSL 基准（23 个用例）",
        "",
        "| 用例 | 优化前 | 优化后 | 节省 | 节省% | 规则命中次数 |",
        "|------|--------|--------|------|-------|--------------|",
    ]
    for c in sorted(report.cases, key=lambda x: -x.saved):
        hits = sum(c.rule_hits.values())
        lines.append(
            f"| {c.name} | {c.before_total} | {c.after_total} | {c.saved} | {c.saved_pct}% | {hits} |"
        )

    lines.extend(
        [
            "",
            "## 合成汇编（高 fusion 密度）",
            "",
            "| 规模 | 优化前 | 优化后 | 节省 | 节省% | addi 前→后 |",
            "|------|--------|--------|------|-------|------------|",
        ]
    )
    for c in synthetic:
        addi_b = c.before_opcodes.get("addi", 0)
        addi_a = c.after_opcodes.get("addi", 0)
        lines.append(
            f"| {c.name} | {c.before_total} | {c.after_total} | {c.saved} | {c.saved_pct}% | {addi_b}→{addi_a} |"
        )

    # Aggregate rule hits across DSL
    agg: dict[str, int] = {}
    for c in report.cases:
        for k, v in c.rule_hits.items():
            agg[k] = agg.get(k, 0) + v
    if agg:
        lines.extend(["", "## 规则命中汇总（DSL）", ""])
        for name, cnt in sorted(agg.items(), key=lambda x: -x[1]):
            if cnt:
                lines.append(f"- **{name}**: {cnt}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Peephole before/after comparison")
    parser.add_argument("--cases", default="benchmarks/cases", help="DSL cases dir")
    parser.add_argument("--json", help="Write JSON report")
    parser.add_argument("--markdown", help="Write Markdown report")
    parser.add_argument("--html", help="Write DSL HTML report")
    args = parser.parse_args()

    cases_dir = ROOT / args.cases
    report = run_dsl_suite(cases_dir)
    synthetic = run_synthetic_cases()

    print("=" * 72)
    print("Peephole Optimizer — Before / After Comparison")
    print("=" * 72)
    print(f"\nDSL suite ({len(report.cases)} cases):")
    print(f"  Before: {report.total_before} static instructions")
    print(f"  After:  {report.total_after} static instructions")
    print(
        f"  Saved:  {report.total_saved} ({report.total_saved/report.total_before*100:.2f}%)"
        if report.total_before
        else ""
    )
    print(f"  Cases with savings: {report.cases_with_savings}/{len(report.cases)}")

    print(f"\n{'Case':<28} {'Before':>8} {'After':>8} {'Saved':>8} {'%':>7}")
    print("-" * 72)
    for c in sorted(report.cases, key=lambda x: -x.saved):
        if c.saved > 0:
            print(
                f"{c.name:<28} {c.before_total:>8} {c.after_total:>8} {c.saved:>8} {c.saved_pct:>6.1f}%"
            )

    print("\nSynthetic (fusion_ratio=0.3):")
    print(f"{'Case':<20} {'Before':>8} {'After':>8} {'Saved':>8} {'%':>7}")
    print("-" * 56)
    for c in synthetic:
        print(
            f"{c.name:<20} {c.before_total:>8} {c.after_total:>8} {c.saved:>8} {c.saved_pct:>6.1f}%"
        )

    payload = {
        "dsl_suite": {
            "total_before": report.total_before,
            "total_after": report.total_after,
            "total_saved": report.total_saved,
            "cases_with_savings": report.cases_with_savings,
            "cases": [asdict(c) for c in report.cases],
        },
        "synthetic": [asdict(c) for c in synthetic],
    }

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nJSON: {args.json}")

    md = to_markdown(report, synthetic)
    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
        print(f"Markdown: {args.markdown}")

    if args.html:
        from benchmarks.compare_peephole_html import generate_dsl_html_report

        Path(args.html).write_text(generate_dsl_html_report(report), encoding="utf-8")
        print(f"HTML: {args.html}")


if __name__ == "__main__":
    main()
