# flake8: noqa
"""Benchmark for RISC-V Assembly Beautifier.

Measures beautification time and output size for assembly files
of varying complexity.

Usage:
    python benchmarks/bench_asm_beautifier.py
    python benchmarks/bench_asm_beautifier.py --repeats 100
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scratchv.backend.asm_beautifier import beautify_asm


# ---------------------------------------------------------------------------
# Test programs of varying complexity
# ---------------------------------------------------------------------------

_SIMPLE_ASM = """.text
main:
  addi sp, sp, -16
  sw ra, 12(sp)
  li a0, 42
  lw ra, 12(sp)
  addi sp, sp, 16
  ret
"""

_MODERATE_ASM = """.text
main:
  addi sp, sp, -32
  sw ra, 28(sp)
  sw s0, 24(sp)
  addi s0, sp, 32
  li a0, 1
  li a1, 10
loop:
  beq a0, a1, exit
  addi a0, a0, 1
  mv t0, a0
  slli t1, t0, 2
  add t2, s0, t1
  lw t3, 0(t2)
  add t4, t4, t3
  j loop
exit:
  mv a0, t4
  lw s0, 24(sp)
  lw ra, 28(sp)
  addi sp, sp, 32
  ret
"""

_LARGE_ASM = _MODERATE_ASM * 20


def _gen_random_asm(num_instrs: int, seed: int = 42) -> str:
    """Generate deterministic synthetic RISC-V assembly of a given size."""

    generator = random.Random(seed)
    arithmetic_ops = [
        "add",
        "sub",
        "mul",
        "xor",
        "or",
        "and",
        "sll",
        "slt",
        "div",
    ]
    immediate_ops = ["addi", "slli", "srli"]
    registers = [
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
        "t6",
        "a0",
        "a1",
        "a2",
        "a3",
        "s0",
        "s1",
        "s2",
    ]
    ops = [
        *arithmetic_ops,
        *immediate_ops,
        "lw",
        "sw",
        "beq",
        "j",
        "li",
        "mv",
        "ret",
    ]

    lines = [".text", ".globl synthetic_func", "synthetic_func:"]
    for index in range(num_instrs):
        opcode = generator.choice(ops)
        label = f"label_{index % 10}"
        if opcode == "j":
            lines.append(f"  {opcode} {label}")
        elif opcode == "beq":
            lines.append(
                f"  {opcode} {generator.choice(registers)}, "
                f"{generator.choice(registers)}, {label}"
            )
        elif opcode == "li":
            lines.append(
                f"  {opcode} {generator.choice(registers)}, "
                f"{generator.randint(0, 4096)}"
            )
        elif opcode == "mv":
            lines.append(
                f"  {opcode} {generator.choice(registers)}, "
                f"{generator.choice(registers)}"
            )
        elif opcode in {"lw", "sw"}:
            lines.append(
                f"  {opcode} {generator.choice(registers)}, "
                f"{generator.randint(0, 16)}(sp)"
            )
        elif opcode in immediate_ops:
            lines.append(
                f"  {opcode} {generator.choice(registers)}, "
                f"{generator.choice(registers)}, {generator.randint(0, 31)}"
            )
        elif opcode == "ret":
            lines.append("  ret")
        else:
            lines.append(
                f"  {opcode} {generator.choice(registers)}, "
                f"{generator.choice(registers)}, {generator.choice(registers)}"
            )

        if index % 15 == 0:
            lines.append(f"label_{index % 10}:")
    lines.append("  ret")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_beautify(
    asm_text: str,
    align: bool = True,
    add_comments: bool = True,
    abi_register_names: bool = False,
    repeats: int = 50,
) -> dict[str, Any]:
    """Run beautification repeatedly and return timing statistics."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    times: list[float] = []
    output_size = 0
    expected_result: str | None = None

    for _ in range(repeats):
        start = time.perf_counter()
        result = beautify_asm(
            asm_text,
            align=align,
            add_comments=add_comments,
            abi_register_names=abi_register_names,
        )
        times.append(time.perf_counter() - start)
        output_size = len(result)
        if expected_result is None:
            expected_result = result
        elif result != expected_result:
            raise RuntimeError("beautifier output changed between runs")

    return {
        "input_lines": len(asm_text.splitlines()),
        "input_chars": len(asm_text),
        "output_chars": output_size,
        "ratio": output_size / max(len(asm_text), 1),
        "repeats": repeats,
        "min_s": min(times),
        "max_s": max(times),
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def run_all_benchmarks(repeats: int = 50) -> None:
    """Run fixed-size and synthetic beautifier benchmarks."""

    benchmarks = {
        "simple": _SIMPLE_ASM,
        "moderate": _MODERATE_ASM,
        "large": _LARGE_ASM,
        "synthetic_1k": _gen_random_asm(1000),
        "synthetic_5k": _gen_random_asm(5000),
    }

    print("=" * 80)
    print("RISC-V Assembly Beautifier Benchmark")
    print("=" * 80)
    print(
        f"{'Test':<20} {'Input':>8} {'Output':>8} {'Ratio':>7} "
        f"{'Mean(ms)':>10} {'Stdev(ms)':>10}"
    )
    print("-" * 80)

    for name, asm_text in benchmarks.items():
        stats = bench_beautify(asm_text, repeats=repeats)
        print(
            f"{name:<20} {stats['input_chars']:>8} "
            f"{stats['output_chars']:>8} {stats['ratio']:>6.2f}x "
            f"{stats['mean_s'] * 1000:>10.3f} "
            f"{stats['stdev_s'] * 1000:>10.3f}"
        )

    print()
    print("Feature Impact (on synthetic_1k):")
    print("-" * 60)

    synthetic_1k = benchmarks["synthetic_1k"]
    for align in (True, False):
        for comments in (True, False):
            stats = bench_beautify(
                synthetic_1k,
                align=align,
                add_comments=comments,
                repeats=repeats,
            )
            label = f"align={align}, comments={comments}"
            print(
                f"  {label:<30} {stats['mean_s'] * 1000:>8.3f} ms  "
                f"output: {stats['output_chars']} chars"
            )


def _positive_int(value: str) -> int:
    repeats = int(value)
    if repeats < 1:
        raise argparse.ArgumentTypeError("repeats must be at least 1")
    return repeats


def main() -> None:
    parser = argparse.ArgumentParser(description="Beautifier Benchmark")
    parser.add_argument(
        "--repeats",
        type=_positive_int,
        default=50,
        help="number of repeat measurements",
    )
    args = parser.parse_args()
    run_all_benchmarks(args.repeats)


if __name__ == "__main__":
    main()
