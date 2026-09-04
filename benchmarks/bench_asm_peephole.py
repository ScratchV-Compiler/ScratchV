# flake8: noqa
"""Reproducible micro-benchmarks for the assembly peephole optimizer.

The module deliberately separates data collection from presentation.  It
provides a small, deterministic case suite and emits JSON that can be consumed
by `compare_peephole.py` or other tooling.

Usage:
    python benchmarks/bench_asm_peephole.py
    python benchmarks/bench_asm_peephole.py --repeats 20
    python benchmarks/bench_asm_peephole.py --output benchmark_reports/peephole_raw.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

# Direct script execution omits the repository root from sys.path.
# Put this worktree first so the benchmark measures the checked-out code.
if __package__ is None:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from scratchv.backend._asm_parser import parse_asm
from scratchv.backend.asm_peephole import AsmPeepholeOptimizer


PR39_RULES = (
    "addi+addi fusion",
    "li+addi fusion",
    "beq zero-zero to jump",
    "redundant mv elimination",
    "addi-zero self elimination",
    "addi-zero to mv",
    "nop elimination",
    "mv-self elimination",
)


@dataclass(frozen=True)
class BenchmarkCase:
    """One assembly input used by the Benchmark suite."""

    case_id: str
    assembly: str
    expected_rule: Optional[str] = None
    category: str = "synthetic"
    description: str = ""


def count_instructions(asm_text: str) -> int:
    """Count effective instructions, excluding directives, labels and blanks."""

    return sum(
        1
        for line in parse_asm(asm_text)
        if line.opcode is not None and not line.is_directive
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_repeats(repeats: int) -> int:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    return repeats


def default_cases() -> list[BenchmarkCase]:
    """Return deterministic positive, negative and representative cases.

    The positive cases cover each default rule introduced or retained by PR
    #39.  Negative cases are intentionally kept in the suite so that a
    benchmark report also shows patterns the optimizer correctly leaves alone.
    """

    return [
        BenchmarkCase(
            "addi_addi_fusion",
            "addi t0, t0, 3\naddi t0, t0, 5\n",
            "addi+addi fusion",
            description="Two addi increments with a legal signed-12-bit sum.",
        ),
        BenchmarkCase(
            "li_addi_fusion",
            "li t0, 10\naddi t0, t0, 5\n",
            "li+addi fusion",
            description="li followed by an increment of the same register.",
        ),
        BenchmarkCase(
            "beq_zero_jump",
            "beq zero, x0, target\ntarget:\nret\n",
            "beq zero-zero to jump",
            description="Unconditional branch using x0/zero aliases.",
        ),
        BenchmarkCase(
            "redundant_mv_chain",
            "mv t0, t1\nmv t2, t0\n",
            "redundant mv elimination",
            description="Move chain with a redundant intermediate register.",
        ),
        BenchmarkCase(
            "addi_zero_self",
            "addi t0, t0, 0\nret\n",
            "addi-zero self elimination",
            description="Self-add with zero immediate is a no-op.",
        ),
        BenchmarkCase(
            "addi_zero_to_mv",
            "addi t0, t1, 0\nret\n",
            "addi-zero to mv",
            description="Zero-add between different registers becomes mv.",
        ),
        BenchmarkCase(
            "nop_elimination",
            "nop\nadd t0, t1, t2\n",
            "nop elimination",
            description="Standalone nop is removed.",
        ),
        BenchmarkCase(
            "mv_self",
            "mv t0, t0\nret\n",
            "mv-self elimination",
            description="Self move is a no-op.",
        ),
        BenchmarkCase(
            "addi_overflow_negative",
            "addi t0, t0, 2000\naddi t0, t0, 100\n",
            category="negative",
            description="The addi sum exceeds the signed-12-bit range.",
        ),
        BenchmarkCase(
            "mv_swap_negative",
            "mv t0, t1\nmv t1, t0\n",
            category="negative",
            description="A swap-shaped move pair must not be deleted.",
        ),
        BenchmarkCase(
            "label_barrier_negative",
            "addi t0, t0, 1\nL1:\naddi t0, t0, 2\n",
            category="negative",
            description="A jump target between instructions blocks fusion.",
        ),
        BenchmarkCase(
            "beq_nonzero_negative",
            "beq t0, t1, target\n",
            category="negative",
            description="Conditional branch is not an unconditional jump.",
        ),
        BenchmarkCase(
            "representative_codegen",
            ".text\n.globl main\nmain:\n"
            "  li t0, 4\n"
            "  addi t0, t0, 6\n"
            "  addi t1, t0, 0\n"
            "  mv t2, t2\n"
            "  nop\n"
            "  ret\n",
            category="representative",
            description="Small codegen-shaped sequence with labels and directives.",
        ),
        BenchmarkCase(
            "clean_assembly",
            ".text\nmain:\n  add t0, t1, t2\n  ret\n",
            category="negative",
            description="Already-clean assembly with no local rewrite.",
        ),
    ]


def _gen_synthetic_asm(
    num_instrs: int,
    seed: int = 42,
    fusion_ratio: float = 0.3,
) -> str:
    """Generate deterministic assembly with a controllable addi ratio."""

    if num_instrs < 1:
        raise ValueError("num_instrs must be at least 1")
    if not 0.0 <= fusion_ratio <= 1.0:
        raise ValueError("fusion_ratio must be between 0 and 1")

    rng = random.Random(seed)
    lines = [".text", "synthetic_func:"]
    regs = ["t0", "t1", "t2", "s0", "s1", "a0", "a1"]
    other_regs = ["t0", "t1", "t2", "t3", "t4", "s0", "s1", "a0", "a1"]
    i = 0
    while i < num_instrs:
        if i + 1 < num_instrs and rng.random() < fusion_ratio:
            register = rng.choice(regs)
            lines.append(f"  addi {register}, {register}, {rng.randint(1, 5)}")
            lines.append(f"  addi {register}, {register}, {rng.randint(1, 5)}")
            i += 2
            continue

        opcode = rng.choice(["add", "sub", "lw", "sw", "li", "mv", "mul", "xor"])
        rd = rng.choice(other_regs)
        rs1 = rng.choice(other_regs)
        rs2 = rng.choice(other_regs)
        if opcode == "li":
            lines.append(f"  li {rd}, {rng.randint(0, 100)}")
        elif opcode == "mv":
            lines.append(f"  mv {rd}, {rs1}")
        elif opcode in ("lw", "sw"):
            lines.append(f"  {opcode} {rd}, {rng.randint(0, 16)}(sp)")
        else:
            lines.append(f"  {opcode} {rd}, {rs1}, {rs2}")
        i += 1

    lines.append("  ret")
    return "\n".join(lines) + "\n"


def _run_once(asm_text: str) -> tuple[str, int, float, dict[str, int]]:
    optimizer = AsmPeepholeOptimizer()
    started = time.perf_counter()
    output, changes = optimizer.optimize(asm_text)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return output, changes, elapsed_ms, optimizer.total_matches


def measure_case(case: BenchmarkCase, repeats: int = 5) -> dict:
    """Measure one case and return a JSON-serializable result dictionary."""

    repeats = _validate_repeats(repeats)
    timings: list[float] = []
    output = case.assembly
    changes = 0
    rule_matches: dict[str, int] = {}

    for _ in range(repeats):
        output, changes, elapsed_ms, rule_matches = _run_once(case.assembly)
        timings.append(elapsed_ms)

    before = count_instructions(case.assembly)
    after = count_instructions(output)
    reduced = before - after
    reduction_percent = (100.0 * reduced / before) if before else 0.0
    expected_hit = (
        case.expected_rule is not None
        and rule_matches.get(case.expected_rule, 0) > 0
    )

    return {
        "case_id": case.case_id,
        "category": case.category,
        "description": case.description,
        "expected_rule": case.expected_rule,
        "expected_rule_hit": expected_hit,
        "input_sha256": _sha256(case.assembly),
        "before_instructions": before,
        "after_instructions": after,
        "reduced_instructions": reduced,
        "reduction_percent": round(reduction_percent, 3),
        "changes": changes,
        "rule_matches": dict(rule_matches),
        "elapsed_ms_median": round(statistics.median(timings), 6),
        "elapsed_ms_min": round(min(timings), 6),
        "elapsed_ms_max": round(max(timings), 6),
        "repeats": repeats,
    }


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def run_benchmark(
    cases: Optional[Sequence[BenchmarkCase]] = None,
    repeats: int = 5,
) -> dict:
    """Run all cases and aggregate static savings and rule matches."""

    repeats = _validate_repeats(repeats)
    selected = list(cases if cases is not None else default_cases())
    case_results = [measure_case(case, repeats=repeats) for case in selected]

    before = sum(result["before_instructions"] for result in case_results)
    after = sum(result["after_instructions"] for result in case_results)
    reduced = before - after
    rule_matches: dict[str, int] = {name: 0 for name in PR39_RULES}
    for result in case_results:
        for name, count in result["rule_matches"].items():
            rule_matches[name] = rule_matches.get(name, 0) + count

    return {
        "schema_version": 1,
        "benchmark": "ScratchV assembly peephole optimizer",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "repeats": repeats,
        },
        "summary": {
            "case_count": len(case_results),
            "before_instructions": before,
            "after_instructions": after,
            "reduced_instructions": reduced,
            "reduction_percent": round(
                100.0 * reduced / before if before else 0.0,
                3,
            ),
            "changes": sum(result["changes"] for result in case_results),
            "elapsed_ms_median_sum": round(
                sum(result["elapsed_ms_median"] for result in case_results),
                6,
            ),
            "rule_matches": rule_matches,
        },
        "cases": case_results,
    }


def save_json(report: dict, path: str | Path) -> None:
    """Write a benchmark report to *path*, creating its parent directory."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def bench_optimize(asm_text: str, repeats: int = 20) -> dict:
    """Benchmark one assembly input while preserving the legacy return fields."""

    repeats = _validate_repeats(repeats)
    timings: list[float] = []
    changes_list: list[int] = []
    output = asm_text
    for _ in range(repeats):
        output, changes, elapsed_ms, _ = _run_once(asm_text)
        timings.append(elapsed_ms / 1000.0)
        changes_list.append(changes)

    input_lines = len(asm_text.splitlines())
    output_lines = len(output.splitlines())
    input_instructions = count_instructions(asm_text)
    output_instructions = count_instructions(output)
    return {
        "input_lines": input_lines,
        "output_lines": output_lines,
        "line_reduction": input_lines - output_lines,
        "input_instructions": input_instructions,
        "output_instructions": output_instructions,
        "instruction_reduction": input_instructions - output_instructions,
        "changes_mean": statistics.mean(changes_list),
        "changes_stdev": (
            statistics.stdev(changes_list) if len(changes_list) > 1 else 0.0
        ),
        "repeats": repeats,
        "min_s": min(timings),
        "max_s": max(timings),
        "mean_s": statistics.mean(timings),
        "median_s": statistics.median(timings),
        "stdev_s": statistics.stdev(timings) if len(timings) > 1 else 0.0,
    }


def _print_summary(report: dict) -> None:
    summary = report["summary"]
    print("=" * 96)
    print("ScratchV RISC-V Peephole Optimizer Benchmark")
    print("=" * 96)
    print(
        f"Cases: {summary['case_count']} | "
        f"Instructions: {summary['before_instructions']} -> "
        f"{summary['after_instructions']} | "
        f"Saved: {summary['reduced_instructions']} "
        f"({summary['reduction_percent']:.1f}%)"
    )
    print()
    print(
        f"{'Case':<28} {'Category':<14} {'Before':>8} {'After':>8} "
        f"{'Saved':>8} {'Changes':>8} {'Median(ms)':>12}"
    )
    print("-" * 96)
    for result in report["cases"]:
        print(
            f"{result['case_id']:<28} {result['category']:<14} "
            f"{result['before_instructions']:>8} "
            f"{result['after_instructions']:>8} "
            f"{result['reduced_instructions']:>8} "
            f"{result['changes']:>8} "
            f"{result['elapsed_ms_median']:>12.3f}"
        )
    print()
    print("Rule matches:")
    for name, count in summary["rule_matches"].items():
        print(f"  {name}: {count}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the ScratchV assembly peephole optimizer",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timing repetitions per case (default: 5)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 2000, 5000],
        help="Synthetic instruction sizes to benchmark",
    )
    parser.add_argument(
        "--fusion-ratio",
        type=float,
        default=0.3,
        help="Fraction of synthetic instructions formed into addi pairs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the structured case-suite JSON report",
    )
    args = parser.parse_args(argv)

    print("=" * 96)
    print("ScratchV RISC-V Peephole Optimizer Benchmark")
    print("=" * 96)
    print(
        f"\n{'Size':>8} {'Mean(ms)':>10} {'Stdev(ms)':>10} "
        f"{'Changes':>8} {'InpLines':>10} {'OutLines':>10} {'Reduc':>8}"
    )
    print("-" * 96)
    for size in args.sizes:
        asm = _gen_synthetic_asm(size, fusion_ratio=args.fusion_ratio)
        stats = bench_optimize(asm, repeats=args.repeats)
        print(
            f"{size:>8} {stats['mean_s'] * 1000:>10.3f} "
            f"{stats['stdev_s'] * 1000:>10.3f} "
            f"{stats['changes_mean']:>8.1f} "
            f"{stats['input_lines']:>10} {stats['output_lines']:>10} "
            f"{stats['line_reduction']:>8}"
        )

    print("\nFusion Ratio Impact (2000 instructions):")
    print("-" * 60)
    for ratio in [0.0, 0.1, 0.3, 0.5]:
        asm = _gen_synthetic_asm(2000, fusion_ratio=ratio)
        stats = bench_optimize(asm, repeats=args.repeats)
        print(
            f"  ratio={ratio:.1f}  {stats['mean_s'] * 1000:.3f} ms  "
            f"changes: {stats['changes_mean']:.1f}  "
            f"reduction: {stats['instruction_reduction']}"
        )

    print("\nRule Coverage Suite (effective instruction counts):")
    report = run_benchmark(repeats=args.repeats)
    _print_summary(report)
    if args.output is not None:
        save_json(report, args.output)
        print(f"Structured JSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
