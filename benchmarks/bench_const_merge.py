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
import os
import statistics
import sys
import time

BENCH_DIR = os.path.dirname(__file__)
PROJ_DIR = os.path.dirname(BENCH_DIR)
sys.path.insert(0, PROJ_DIR)

from scratchv.backend._asm_parser import parse_asm
from scratchv.backend.const_merge import merge_constants_detailed


def _gen_synthetic_asm(
    num_instructions: int,
    seed: int = 42,
    pair_density: float = 0.3,
    redundant_lui_density: float = 0.1,
) -> str:
    """Generate controlled synthetic assembly covering both optimization rules.

    Parameters
    ----------
    num_instructions:
        Target number of instructions.
    seed:
        Random seed for reproducibility.
    pair_density:
        Fraction of instructions that form lui+addi pairs.
    redundant_lui_density:
        Fraction of generated groups that contain a redundant LUI pattern.
    """
    import random

    if num_instructions < 0:
        raise ValueError("num_instructions must be non-negative")
    if not 0.0 <= pair_density <= 1.0:
        raise ValueError("pair_density must be between 0 and 1")
    if not 0.0 <= redundant_lui_density <= 1.0:
        raise ValueError("redundant_lui_density must be between 0 and 1")
    if pair_density + redundant_lui_density > 1.0:
        raise ValueError(
            "pair_density + redundant_lui_density must not exceed 1",
        )

    rng = random.Random(seed)

    lines = [".text", "synthetic_func:"]
    i = 0
    while i < num_instructions:
        choice = rng.random()

        if choice < redundant_lui_density and i + 2 < num_instructions:
            regs = ["t0", "t1", "t2", "s0", "s1", "a0", "a1"]
            reg = rng.choice(regs)
            imm_hi = rng.choice([0x10000, 0x20000, 0x12345])
            lines.append(f"  lui {reg}, {hex(imm_hi)}")
            lines.append("  add a4, a5, a6")
            lines.append(f"  lui {reg}, {hex(imm_hi)}")
            i += 3
        elif (
            choice < redundant_lui_density + pair_density
            and i + 1 < num_instructions
        ):
            regs = ["t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3"]
            r = rng.choice(regs)
            imm_hi = rng.choice([0x10000, 0x20000, 0x12345, 0xABCDE, 0xFFFFF])
            imm_lo = rng.choice([0x000, 0x100, 0x678, 0xFFF, 0x800])
            lines.append(f"  lui {r}, {hex(imm_hi)}")
            lines.append(f"  addi {r}, {r}, {hex(imm_lo)}")
            i += 2
        else:
            op = rng.choice(["add", "sub", "lw", "sw", "mv", "mul", "xor",
                             "li", "addi", "beq", "j", "ret"])
            regs = ["t0", "t1", "t2", "t3", "t4", "s0", "s1",
                    "a0", "a1", "a2", "a3", "sp", "ra"]
            r1 = rng.choice(regs)
            r2 = rng.choice(regs)
            r3 = rng.choice(regs)
            if op == "li":
                lines.append(f"  {op} {r1}, {rng.randint(0, 4096)}")
            elif op == "addi":
                lines.append(f"  {op} {r1}, {r2}, {rng.randint(-2048, 2047)}")
            elif op in ("lw", "sw"):
                lines.append(f"  {op} {r1}, {rng.randint(0, 16)}(sp)")
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
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    times = []
    results = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        result, stats = merge_constants_detailed(asm_text)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        results.append((result, stats))

    changes_list = [r[1].total_changes for r in results]
    first_stats = results[0][1]
    parsed_input = parse_asm(asm_text)
    parsed_output = parse_asm(results[0][0])
    input_instructions = sum(
        line.opcode is not None and not line.is_directive
        for line in parsed_input
    )
    output_instructions = sum(
        line.opcode is not None and not line.is_directive
        for line in parsed_output
    )

    return {
        "benchmark_type": "synthetic",
        "input_instructions": input_instructions,
        "output_instructions": output_instructions,
        "instruction_reduction": input_instructions - output_instructions,
        "candidate_pairs": first_stats.candidate_pairs,
        "merged_pairs": first_stats.merged_pairs,
        "redundant_lui_removed": first_stats.redundant_lui_removed,
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair-density", type=float, default=0.3)
    parser.add_argument("--redundant-lui-density", type=float, default=0.1)
    args = parser.parse_args()

    sizes = [100, 500, 1000, 2000, 5000]
    print("=" * 80)
    print("RISC-V Constant Load Merge Optimizer Benchmark")
    print("benchmark_type=synthetic")
    print("=" * 80)

    print(f"\n{'Size':>8} {'Mean(ms)':>10} {'Stdev(ms)':>10} "
          f"{'Pairs':>8} {'RedLUI':>8} {'InpInst':>10} {'OutInst':>10}")
    print("-" * 80)

    for size in sizes:
        asm = _gen_synthetic_asm(
            size,
            seed=args.seed,
            pair_density=args.pair_density,
            redundant_lui_density=args.redundant_lui_density,
        )
        stats = bench_merge(asm, repeats=args.repeats)
        print(f"{size:>8} {stats['mean_s'] * 1000:>10.3f} "
              f"{stats['stdev_s'] * 1000:>10.3f} "
              f"{stats['merged_pairs']:>8} "
              f"{stats['redundant_lui_removed']:>8} "
              f"{stats['input_instructions']:>10} "
              f"{stats['output_instructions']:>10}")

    # Test different lui densities
    print(f"\nLUI Density Impact (2000 instructions):")
    print("-" * 60)
    for ratio in [0.0, 0.1, 0.3, 0.5]:
        asm = _gen_synthetic_asm(
            2000,
            seed=args.seed,
            pair_density=ratio,
            redundant_lui_density=args.redundant_lui_density,
        )
        stats = bench_merge(asm, repeats=args.repeats)
        print(f"  ratio={ratio:.1f}  {stats['mean_s'] * 1000:.3f} ms  "
              f"changes: {stats['changes_mean']:.1f}  "
              f"reduction: {stats['instruction_reduction']}")


if __name__ == "__main__":
    main()
