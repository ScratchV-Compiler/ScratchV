# flake8: noqa
"""Benchmark for Extended Instruction Selector.

Measures instruction selection time for programs of varying complexity,
comparing base selector vs extended selector performance.

Usage:
    python benchmarks/bench_inst_select_ext.py
    python benchmarks/bench_inst_select_ext.py --repeats 100
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Optional

from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Program
from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.inst_select_ext import ExtendedInstructionSelector


def _build_program_simple() -> Program:
    """Build a simple program: add, sub, mul, relu."""
    builder = IRBuilder()
    builder.new_function("simple")
    builder.new_block("entry")
    a = builder.make_value(name="a")
    b = builder.make_value(name="b")
    c = builder.add(a, b)
    d = builder.sub(c, a)
    e = builder.mul(d, b)
    f = builder.relu(e)
    builder.ret(f)
    return builder.program


def _build_program_moderate() -> Program:
    """Build a moderate program with loops and multiple ops."""
    builder = IRBuilder()
    builder.new_function("moderate")
    builder.new_block("entry")
    # Multiple arithmetic chains
    x = builder.make_value(name="x")
    y = builder.make_value(name="y")
    a = builder.add(x, y)
    b = builder.mul(a, y)
    c = builder.sub(b, x)
    d = builder.neg(c)
    e = builder.load_const(42)
    f = builder.add(d, e)
    g = builder.relu(f)
    h = builder.mul(g, a)
    builder.ret(h)
    return builder.program


def _build_program_large(num_chains: int = 10) -> Program:
    """Build a large program with many independent computation chains."""
    builder = IRBuilder()
    builder.new_function("large")
    builder.new_block("entry")

    inputs = [builder.make_value(name=f"in_{i}") for i in range(3)]

    prev = inputs[0]
    for i in range(num_chains):
        op = i % 6
        if op == 0:
            prev = builder.add(prev, inputs[i % 3])
        elif op == 1:
            prev = builder.sub(prev, inputs[i % 3])
        elif op == 2:
            prev = builder.mul(prev, inputs[i % 3])
        elif op == 3:
            prev = builder.relu(prev)
        elif op == 4:
            prev = builder.neg(prev)
        else:
            c = builder.load_const(i * 10)
            prev = builder.add(prev, c)

    builder.ret(prev)
    return builder.program


def bench_selector(program: Program, use_extended: bool = False,
                   repeats: int = 50) -> dict:
    """Benchmark instruction selection time."""
    times = []

    for _ in range(repeats):
        if use_extended:
            selector = ExtendedInstructionSelector(program)
        else:
            selector = InstructionSelector(program)
        t0 = time.perf_counter()
        instrs = selector.run()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "num_instrs": len(instrs) if 'instrs' in dir() else sum(
            1 for f in program.functions for bb in f.blocks for _ in bb.instructions),
        "repeats": repeats,
        "min_s": min(times),
        "max_s": max(times),
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0,
    }


def bench_with_count(program: Program, use_extended: bool,
                     repeats: int) -> tuple:
    """Return (mean_time, num_instrs, std)."""
    times = []
    num_instrs = 0

    for _ in range(repeats):
        if use_extended:
            selector = ExtendedInstructionSelector(program)
        else:
            selector = InstructionSelector(program)
        t0 = time.perf_counter()
        instrs = selector.run()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        num_instrs = len(instrs)

    return (statistics.mean(times), num_instrs,
            statistics.stdev(times) if len(times) > 1 else 0)


def main():
    parser = argparse.ArgumentParser(description="Extended Instruction Selector Benchmark")
    parser.add_argument("--repeats", type=int, default=100,
                        help="Number of repeat measurements")
    args = parser.parse_args()

    print("=" * 80)
    print("Extended Instruction Selector Benchmark")
    print("=" * 80)

    print(f"\nBase vs Extended Selector Performance:")
    print("-" * 80)
    print(f"{'Program':<18} {'Selector':<12} {'Mean(ms)':>10} "
          f"{'Stdev(ms)':>10} {'MI Instrs':>10}")
    print("-" * 80)

    for name, build_fn in [
        ("simple", _build_program_simple),
        ("moderate", _build_program_moderate),
        ("large(10)", lambda: _build_program_large(10)),
        ("large(50)", lambda: _build_program_large(50)),
        ("large(200)", lambda: _build_program_large(200)),
    ]:
        prog = build_fn()
        ir_count = sum(1 for f in prog.functions
                       for bb in f.blocks for _ in bb.instructions)

        for label, extended in [("base", False), ("extended", True)]:
            mean_t, mi_count, std_t = bench_with_count(prog, extended, args.repeats)
            print(f"{name:<18} {label:<12} {mean_t * 1000:>10.3f} "
                  f"{std_t * 1000:>10.3f} {mi_count:>10}")

    # Overhead analysis
    print(f"\nOverhead Analysis (large(50), 200 repeats):")
    print("-" * 60)
    prog = _build_program_large(50)
    base_mean, _, base_std = bench_with_count(prog, False, 200)
    ext_mean, _, ext_std = bench_with_count(prog, True, 200)
    overhead = ext_mean - base_mean
    overhead_pct = (overhead / base_mean * 100) if base_mean > 0 else 0
    print(f"  Base:     {base_mean * 1000:.4f} ms")
    print(f"  Extended: {ext_mean * 1000:.4f} ms")
    print(f"  Overhead: {overhead * 1000:.4f} ms ({overhead_pct:.1f}%)")


if __name__ == "__main__":
    main()
