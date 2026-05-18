# flake8: noqa
"""Benchmark for Linear Scan Register Allocator.

Measures live interval computation, allocation time, and spill count
for basic blocks with varying register pressure.

Usage:
    python benchmarks/bench_regalloc_linear.py
    python benchmarks/bench_regalloc_linear.py --repeats 100
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Optional

from scratchv.backend.regalloc_linear import (
    LinearScanAllocator, LsInstruction,
)


def _gen_block(num_insts: int, num_vregs: int, seed: int = 42) -> list[LsInstruction]:
    """Generate a synthetic basic block with given instruction and vreg count.

    Parameters
    ----------
    num_insts:
        Number of instructions in the block.
    num_vregs:
        Number of distinct virtual registers (higher = more register pressure).
    seed:
        Random seed for reproducibility.
    """
    import random
    random.seed(seed)

    ops = ["add", "sub", "mul", "div", "xor", "or", "and", "addi"]
    vreg_names = [f"v{i}" for i in range(num_vregs)]
    insts = []

    for i in range(num_insts):
        if i < num_vregs:
            # Define a new vreg
            vreg = vreg_names[i]
            src1 = random.choice(vreg_names[:i]) if i > 0 else vreg_names[0]
            src2 = random.choice(vreg_names[:i]) if i > 0 else vreg_names[0]
            insts.append(LsInstruction(
                id=i,
                opcode=random.choice(ops),
                operands=[vreg, src1, src2],
                defines={vreg},
                uses={src1, src2},
            ))
        else:
            # Use existing vregs
            vreg = random.choice(vreg_names)
            src1 = random.choice(vreg_names)
            src2 = random.choice(vreg_names)
            insts.append(LsInstruction(
                id=i,
                opcode=random.choice(ops),
                operands=[vreg, src1, src2],
                defines={vreg},
                uses={src1, src2},
            ))

    return insts


def bench_live_intervals(block: list[LsInstruction], repeats: int = 50) -> dict:
    """Benchmark live interval computation."""
    alloc = LinearScanAllocator()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        intervals = alloc.compute_live_intervals(block)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return {
        "num_intervals": len(intervals),
        "mean_s": statistics.mean(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0,
    }


def bench_allocate(block: list[LsInstruction], phys_regs: list[str],
                   repeats: int = 50) -> dict:
    """Benchmark full allocation pipeline."""
    times = []
    alloc_map_sizes = []
    spill_counts = []

    for _ in range(repeats):
        alloc = LinearScanAllocator(phys_regs=phys_regs)
        t0 = time.perf_counter()
        intervals = alloc.compute_live_intervals(block)
        mapping = alloc.allocate(intervals)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        alloc_map_sizes.append(len(mapping))
        spill_counts.append(len(alloc.spill_code))

    return {
        "mean_s": statistics.mean(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0,
        "allocated_mean": statistics.mean(alloc_map_sizes),
        "spills_mean": statistics.mean(spill_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Register Allocator Benchmark")
    parser.add_argument("--repeats", type=int, default=50,
                        help="Number of repeat measurements")
    args = parser.parse_args()

    print("=" * 80)
    print("Linear Scan Register Allocator Benchmark")
    print("=" * 80)

    # Test: varying instruction count with fixed vreg count
    print(f"\nVarying Instruction Count (24 vregs, 16 phys regs):")
    print("-" * 70)
    print(f"{'Instrs':>8} {'Mean(ms)':>10} {'Stdev(ms)':>10} "
          f"{'Alloc':>6} {'Spills':>7}")
    print("-" * 70)

    phys16 = [f"r{i}" for i in range(16)]  # 16 physical registers

    for num_insts in [10, 50, 100, 200, 500]:
        block = _gen_block(num_insts, num_vregs=24)
        stats = bench_allocate(block, phys16, repeats=args.repeats)
        print(f"{num_insts:>8} {stats['mean_s'] * 1000:>10.3f} "
              f"{stats['stdev_s'] * 1000:>10.3f} "
              f"{stats['allocated_mean']:>6.0f} {stats['spills_mean']:>7.0f}")

    # Test: varying register pressure (fixed instruction count)
    print(f"\nVarying Register Pressure (200 instrs, 16 phys regs):")
    print("-" * 70)
    print(f"{'VRegs':>8} {'Mean(ms)':>10} {'Stdev(ms)':>10} "
          f"{'Alloc':>6} {'Spills':>7} {'Intervals':>10}")
    print("-" * 70)

    for num_vregs in [8, 16, 32, 64, 128]:
        block = _gen_block(200, num_vregs=num_vregs)
        # Live interval benchmark
        liv = bench_live_intervals(block, repeats=args.repeats)
        # Allocation benchmark
        stats = bench_allocate(block, phys16, repeats=args.repeats)
        print(f"{num_vregs:>8} {stats['mean_s'] * 1000:>10.3f} "
              f"{stats['stdev_s'] * 1000:>10.3f} "
              f"{stats['allocated_mean']:>6.0f} {stats['spills_mean']:>7.0f} "
              f"{liv['num_intervals']:>10}")

    # Test: large block stress test
    print(f"\nStress Test (2000 instrs, 64 vregs, 16 phys regs):")
    print("-" * 60)
    block = _gen_block(2000, num_vregs=64, seed=123)
    t0 = time.perf_counter()
    alloc = LinearScanAllocator(phys_regs=phys16)
    intervals = alloc.compute_live_intervals(block)
    mapping = alloc.allocate(intervals)
    elapsed = time.perf_counter() - t0
    print(f"  Live intervals: {len(intervals)}")
    print(f"  Allocated: {len(mapping)}")
    print(f"  Spills: {len(alloc.spill_code)}")
    print(f"  Time: {elapsed * 1000:.3f} ms")


if __name__ == "__main__":
    main()
