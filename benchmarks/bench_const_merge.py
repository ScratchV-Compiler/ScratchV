# flake8: noqa
"""Benchmark for Constant Load Merge Optimizer.

Measures optimization time and instruction reduction for code
with varying density of lui+addi pairs.

Usage:
    python benchmarks/bench_const_merge.py
    python benchmarks/bench_const_merge.py --repeats 100
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Optional

from scratchv.backend.const_merge import merge_constants


def _gen_synthetic_asm(num_instrs: int, seed: int = 42,
                        lui_ratio: float = 0.3) -> str:
    """Generate synthetic assembly with lui+addi patterns.

    Parameters
    ----------
    num_instrs:
        Target number of instructions.
    seed:
        Random seed for reproducibility.
    lui_ratio:
        Fraction of instructions that form lui+addi pairs.
    """
    import random
    random.seed(seed)

    lines = [".text", "synthetic_func:"]
    i = 0
    while i < num_instrs:
        use_lui = random.random() < lui_ratio

        if use_lui and i + 1 < num_instrs:
            regs = ["t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3"]
            r = random.choice(regs)
            imm_hi = random.choice([0x10000, 0x20000, 0x12345, 0xABCDE, 0xFFFFF])
            imm_lo = random.choice([0x000, 0x100, 0x678, 0xFFF, 0x800])
            lines.append(f"  lui {r}, {hex(imm_hi)}")
            lines.append(f"  addi {r}, {r}, {hex(imm_lo)}")
            i += 2
        else:
            op = random.choice(["add", "sub", "lw", "sw", "mv", "mul", "xor",
                                "li", "addi", "beq", "j", "ret"])
            regs = ["t0", "t1", "t2", "t3", "t4", "s0", "s1",
                    "a0", "a1", "a2", "a3", "sp", "ra"]
            r1 = random.choice(regs)
            r2 = random.choice(regs)
            r3 = random.choice(regs)
            if op == "li":
                lines.append(f"  {op} {r1}, {random.randint(0, 4096)}")
            elif op == "addi":
                lines.append(f"  {op} {r1}, {r2}, {random.randint(-2048, 2047)}")
            elif op in ("lw", "sw"):
                lines.append(f"  {op} {r1}, {random.randint(0, 16)}(sp)")
            elif op in ("beq", "bne", "blt", "bge"):
                lines.append(f"  {op} {r1}, {r2}, label_{i}")
            elif op == "j":
                lines.append(f"  {op} label_{i}")
            elif op == "ret":
                lines.append(f"  ret")
            else:
                lines.append(f"  {op} {r1}, {r2}, {r3}")
            i += 1

    lines.append("")
    return "\n".join(lines)


def bench_merge(asm_text: str, repeats: int = 50) -> dict:
    """Benchmark the constant merge optimizer."""
    times = []
    results = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        result, changes = merge_constants(asm_text)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        results.append((result, changes))

    changes_list = [r[1] for r in results]
    input_lines = asm_text.count("\n")
    output_lines = results[0][0].count("\n") if results else 0

    return {
        "input_lines": input_lines,
        "output_lines": output_lines,
        "line_reduction": input_lines - output_lines,
        "changes_mean": statistics.mean(changes_list),
        "changes_stdev": statistics.stdev(changes_list) if len(changes_list) > 1 else 0,
        "repeats": repeats,
        "min_s": min(times),
        "max_s": max(times),
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Constant Merge Benchmark")
    parser.add_argument("--repeats", type=int, default=50,
                        help="Number of repeat measurements")
    args = parser.parse_args()

    sizes = [100, 500, 1000, 2000, 5000]
    print("=" * 80)
    print("RISC-V Constant Load Merge Optimizer Benchmark")
    print("=" * 80)

    print(f"\n{'Size':>8} {'Mean(ms)':>10} {'Stdev(ms)':>10} "
          f"{'Changes':>8} {'InpLines':>10} {'OutLines':>10} {'Reduc':>8}")
    print("-" * 80)

    for size in sizes:
        asm = _gen_synthetic_asm(size, lui_ratio=0.3)
        stats = bench_merge(asm, repeats=args.repeats)
        print(f"{size:>8} {stats['mean_s'] * 1000:>10.3f} "
              f"{stats['stdev_s'] * 1000:>10.3f} "
              f"{stats['changes_mean']:>8.1f} "
              f"{stats['input_lines']:>10} {stats['output_lines']:>10} "
              f"{stats['line_reduction']:>8}")

    # Test different lui densities
    print(f"\nLUI Density Impact (2000 instructions):")
    print("-" * 60)
    for ratio in [0.0, 0.1, 0.3, 0.5]:
        asm = _gen_synthetic_asm(2000, lui_ratio=ratio)
        stats = bench_merge(asm, repeats=args.repeats)
        print(f"  ratio={ratio:.1f}  {stats['mean_s'] * 1000:.3f} ms  "
              f"changes: {stats['changes_mean']:.1f}  "
              f"reduction: {stats['line_reduction']}")


if __name__ == "__main__":
    main()
