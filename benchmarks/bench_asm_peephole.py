# flake8: noqa
"""Benchmark for Assembly-level Peephole Optimizer.

Measures optimization time, match counts, and instruction reduction
for assembly files of varying sizes.

Usage:
    python benchmarks/bench_asm_peephole.py
    python benchmarks/bench_asm_peephole.py --repeats 100
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Optional

from scratchv.backend.asm_peephole import AsmPeepholeOptimizer


def _gen_synthetic_asm(num_instrs: int, seed: int = 42,
                        fusion_ratio: float = 0.3) -> str:
    """Generate synthetic assembly with peephole optimization opportunities.

    Parameters
    ----------
    num_instrs:
        Target number of instructions.
    seed:
        Random seed for reproducibility.
    fusion_ratio:
        Fraction of instructions that form fusible patterns.
    """
    import random
    random.seed(seed)

    lines = [".text", "synthetic_func:"]
    i = 0
    while i < num_instrs:
        use_fusion = random.random() < fusion_ratio

        if use_fusion:
            # Generate a fusible pattern: addi x, x, a; addi x, x, b
            regs = ["t0", "t1", "t2", "s0", "s1", "a0", "a1"]
            r = random.choice(regs)
            imm1 = random.randint(1, 5)
            imm2 = random.randint(1, 5)
            lines.append(f"  addi {r}, {r}, {imm1}")
            lines.append(f"  addi {r}, {r}, {imm2}")
            i += 2
        else:
            op = random.choice(["add", "sub", "lw", "sw", "li", "mv", "mul", "xor"])
            regs = ["t0", "t1", "t2", "t3", "t4", "s0", "s1",
                    "a0", "a1", "a2", "a3"]
            r1 = random.choice(regs)
            r2 = random.choice(regs)
            r3 = random.choice(regs)
            if op == "li":
                lines.append(f"  {op} {r1}, {random.randint(0, 100)}")
            elif op == "mv":
                lines.append(f"  {op} {r1}, {r2}")
            elif op in ("lw", "sw"):
                lines.append(f"  {op} {r1}, {random.randint(0, 16)}(sp)")
            else:
                lines.append(f"  {op} {r1}, {r2}, {r3}")
            i += 1

    lines.append("  ret\n")
    return "\n".join(lines)


def bench_optimize(asm_text: str, repeats: int = 20) -> dict:
    """Benchmark the peephole optimizer."""
    times = []
    results = []

    for _ in range(repeats):
        optimizer = AsmPeepholeOptimizer()
        t0 = time.perf_counter()
        result, changes = optimizer.optimize(asm_text)
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
    parser = argparse.ArgumentParser(description="Peephole Optimizer Benchmark")
    parser.add_argument("--repeats", type=int, default=20,
                        help="Number of repeat measurements")
    args = parser.parse_args()

    sizes = [100, 500, 1000, 2000, 5000]
    print("=" * 80)
    print("RISC-V Peephole Optimizer Benchmark")
    print("=" * 80)

    print(f"\n{'Size':>8} {'Mean(ms)':>10} {'Stdev(ms)':>10} "
          f"{'Changes':>8} {'InpLines':>10} {'OutLines':>10} {'Reduc':>8}")
    print("-" * 80)

    for size in sizes:
        asm = _gen_synthetic_asm(size, fusion_ratio=0.3)
        stats = bench_optimize(asm, repeats=args.repeats)
        print(f"{size:>8} {stats['mean_s'] * 1000:>10.3f} "
              f"{stats['stdev_s'] * 1000:>10.3f} "
              f"{stats['changes_mean']:>8.1f} "
              f"{stats['input_lines']:>10} {stats['output_lines']:>10} "
              f"{stats['line_reduction']:>8}")

    # Test different fusion ratios
    print(f"\nFusion Ratio Impact (2000 instructions):")
    print("-" * 60)
    for ratio in [0.0, 0.1, 0.3, 0.5]:
        asm = _gen_synthetic_asm(2000, fusion_ratio=ratio)
        stats = bench_optimize(asm, repeats=args.repeats)
        print(f"  ratio={ratio:.1f}  {stats['mean_s'] * 1000:.3f} ms  "
              f"changes: {stats['changes_mean']:.1f}  "
              f"reduction: {stats['line_reduction']}")


if __name__ == "__main__":
    main()
