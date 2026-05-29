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
import time
import statistics
from typing import Optional

from scratchv.backend.asm_beautifier import beautify_asm


# ---------------------------------------------------------------------------
# Test programs of varying complexity
# ---------------------------------------------------------------------------

_SIMPLE_ASM = """
.text
main:
  addi sp, sp, -16
  sw ra, 12(sp)
  li a0, 42
  lw ra, 12(sp)
  addi sp, sp, 16
  ret
"""

_MODERATE_ASM = """
.text
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

_LARGE_ASM = _MODERATE_ASM * 20  # Duplicate for size


def _gen_random_asm(num_instrs: int, seed: int = 42) -> str:
    """Generate synthetic RISC-V assembly of a given size."""
    import random
    random.seed(seed)

    ops = ["add", "sub", "addi", "lw", "sw", "beq", "j", "li", "mv", "mul",
           "xor", "or", "and", "slli", "srli", "slt", "div", "jal", "ret"]
    regs = ["t0", "t1", "t2", "t3", "t4", "t5", "t6",
            "a0", "a1", "a2", "a3", "s0", "s1", "s2"]

    lines = [".text", "synthetic_func:"]
    for i in range(num_instrs):
        op = random.choice(ops)
        if op in ("j", "jal"):
            lines.append(f"  {op} label_{i % 10}")
        elif op in ("beq", "bne", "blt", "bge"):
            lines.append(f"  {op} {random.choice(regs)}, {random.choice(regs)}, label_{i % 10}")
        elif op == "li":
            lines.append(f"  {op} {random.choice(regs)}, {random.randint(0, 4096)}")
        elif op == "mv":
            lines.append(f"  {op} {random.choice(regs)}, {random.choice(regs)}")
        elif op in ("lw", "sw"):
            lines.append(f"  {op} {random.choice(regs)}, {random.randint(0, 16)}(sp)")
        elif op in ("addi", "slli", "srli"):
            lines.append(f"  {op} {random.choice(regs)}, {random.choice(regs)}, {random.randint(0, 31)}")
        else:
            lines.append(f"  {op} {random.choice(regs)}, {random.choice(regs)}, {random.choice(regs)}")
        # Occasionally add labels
        if i % 15 == 0:
            lines.append(f"label_{i % 10}:")
    lines.append("  ret\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_beautify(asm_text: str, align: bool = True,
                   add_comments: bool = True,
                   repeats: int = 50) -> dict:
    """Run beautify benchmark and return timing statistics."""
    times = []
    output_size = 0

    for _ in range(repeats):
        t0 = time.perf_counter()
        result = beautify_asm(asm_text, align=align, add_comments=add_comments)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        output_size = len(result)

    return {
        "input_lines": asm_text.count("\n"),
        "input_chars": len(asm_text),
        "output_chars": output_size,
        "ratio": output_size / max(len(asm_text), 1),
        "repeats": repeats,
        "min_s": min(times),
        "max_s": max(times),
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0,
    }


def run_all_benchmarks(repeats: int = 50):
    """Run all beautifier benchmarks."""
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
    print(f"{'Test':<20} {'Input':>8} {'Output':>8} {'Ratio':>7} {'Mean(ms)':>10} {'Stdev(ms)':>10}")
    print("-" * 80)

    for name, asm in benchmarks.items():
        stats = bench_beautify(asm, repeats=repeats)
        print(
            f"{name:<20} {stats['input_chars']:>8} "
            f"{stats['output_chars']:>8} {stats['ratio']:>6.2f}x "
            f"{stats['mean_s'] * 1000:>10.3f} {stats['stdev_s'] * 1000:>10.3f}"
        )

    # Compare with and without features
    print()
    print("Feature Impact (on synthetic_1k):")
    print("-" * 60)

    asm = _gen_random_asm(1000)
    for align in (True, False):
        for comments in (True, False):
            stats = bench_beautify(asm, align=align, add_comments=comments,
                                   repeats=repeats)
            label = f"align={align}, comments={comments}"
            print(f"  {label:<30} {stats['mean_s'] * 1000:>8.3f} ms  "
                  f"output: {stats['output_chars']} chars")


def main():
    parser = argparse.ArgumentParser(description="Beautifier Benchmark")
    parser.add_argument("--repeats", type=int, default=50,
                        help="Number of repeat measurements")
    args = parser.parse_args()
    run_all_benchmarks(args.repeats)


if __name__ == "__main__":
    main()
