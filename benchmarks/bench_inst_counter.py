# flake8: noqa
"""Benchmark for RISC-V Instruction Counter.

Measures counting and reporting time for assembly files of varying size.

Usage:
    python benchmarks/bench_inst_counter.py
    python benchmarks/bench_inst_counter.py --repeats 100
"""

from __future__ import annotations

import argparse
import os
import statistics
import tempfile
import time
from typing import Optional

from scratchv.backend.inst_counter import (
    count_instructions, format_table, generate_html_report,
)


def _gen_synthetic_asm(num_instrs: int, seed: int = 42) -> str:
    """Generate synthetic RISC-V assembly of a given size."""
    import random
    random.seed(seed)

    ops = ["add", "sub", "addi", "lw", "sw", "beq", "j", "li", "mv", "mul",
           "xor", "or", "and", "slli", "srli", "div", "ret"]
    regs = ["t0", "t1", "t2", "t3", "t4", "t5",
            "a0", "a1", "a2", "a3", "s0", "s1", "s2", "s3"]

    lines = [".text", "synthetic_func:"]
    for i in range(num_instrs):
        op = random.choice(ops)
        if op in ("j",):
            lines.append(f"  {op} label_{i % 10}")
        elif op in ("beq", "bne", "blt", "bge"):
            r1, r2 = random.choice(regs), random.choice(regs)
            lines.append(f"  {op} {r1}, {r2}, label_{i % 10}")
        elif op == "li":
            lines.append(f"  {op} {random.choice(regs)}, {random.randint(0, 4096)}")
        elif op == "mv":
            lines.append(f"  {op} {random.choice(regs)}, {random.choice(regs)}")
        elif op in ("lw", "sw"):
            r = random.choice(regs)
            offset = random.randint(0, 16)
            lines.append(f"  {op} {r}, {offset}(sp)")
        elif op in ("addi", "slli", "srli"):
            r = random.choice(regs)
            imm = random.randint(0, 31)
            lines.append(f"  {op} {r}, {r}, {imm}")
        elif op == "ret":
            lines.append(f"  ret")
        else:
            r1, r2, r3 = random.choice(regs), random.choice(regs), random.choice(regs)
            lines.append(f"  {op} {r1}, {r2}, {r3}")
    lines.append("")
    return "\n".join(lines)


def bench_count(asm_text: str, repeats: int = 100) -> dict:
    """Benchmark instruction counting."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        counts = count_instructions(asm_text)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "num_instrs": sum(v for k, v in counts.items()
                          if not k.startswith("_") and isinstance(v, int)),
        "repeats": repeats,
        "min_s": min(times),
        "max_s": max(times),
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0,
    }


def bench_format(counts: dict, repeats: int = 100) -> dict:
    """Benchmark table formatting."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        table = format_table(counts)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return {
        "mean_s": statistics.mean(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0,
    }


def bench_html(counts: dict, output_path: str, repeats: int = 50) -> dict:
    """Benchmark HTML report generation."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        generate_html_report(counts, output_path)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return {
        "mean_s": statistics.mean(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Instruction Counter Benchmark")
    parser.add_argument("--repeats", type=int, default=100,
                        help="Number of repeat measurements")
    args = parser.parse_args()

    sizes = [100, 500, 1000, 5000, 10000]
    print("=" * 80)
    print("RISC-V Instruction Counter Benchmark")
    print("=" * 80)

    print(f"\n{'Size':>8} {'Count(s) mean':>14} {'Count(s) stdev':>14} "
          f"{'Instrs':>8}")
    print("-" * 60)

    for size in sizes:
        asm = _gen_synthetic_asm(size)
        stats = bench_count(asm, repeats=args.repeats)
        print(f"{size:>8} {stats['mean_s'] * 1000:>14.3f}ms "
              f"{stats['stdev_s'] * 1000:>14.3f}ms "
              f"{stats['num_instrs']:>8}")

    # Format and HTML benchmarks on a moderate size
    print(f"\nOutput Format Benchmarks (on 5000 instructions):")
    print("-" * 60)

    asm = _gen_synthetic_asm(5000)
    counts = count_instructions(asm)

    fmt = bench_format(counts, repeats=args.repeats // 2)
    print(f"  format_table:   {fmt['mean_s'] * 1000:.3f} ms ± "
          f"{fmt['stdev_s'] * 1000:.3f}")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        html_path = f.name
    try:
        html = bench_html(counts, html_path, repeats=args.repeats // 4)
        print(f"  HTML report:    {html['mean_s'] * 1000:.3f} ms ± "
              f"{html['stdev_s'] * 1000:.3f}")
    finally:
        if os.path.exists(html_path):
            os.unlink(html_path)


if __name__ == "__main__":
    main()
