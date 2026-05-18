# flake8: noqa
"""Benchmark for Instruction Scheduler (List Scheduling).

Measures DAG construction time, scheduling time, and cycle count
improvement for basic blocks of varying size and dependency depth.

Usage:
    python benchmarks/bench_inst_scheduler.py
    python benchmarks/bench_inst_scheduler.py --repeats 100
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Optional

from scratchv.backend.inst_scheduler import (
    InstructionScheduler, SchedInst,
)


def _gen_instructions(num_insts: int, seed: int = 42,
                       dep_chains: int = 3) -> list[SchedInst]:
    """Generate synthetic instructions with controlled dependency patterns.

    Parameters
    ----------
    num_insts:
        Number of instructions to generate.
    seed:
        Random seed.
    dep_chains:
        Number of dependency chains to create.
    """
    import random
    random.seed(seed)

    ops = ["add", "sub", "mul", "lw", "sw", "xor", "or", "and",
           "addi", "slli", "srli", "div", "li", "mv"]

    # Create register groups (each chain uses its own group to avoid cross-chain deps)
    reg_prefixes = [f"r{c}_" for c in range(dep_chains)]
    all_regs = []
    for prefix in reg_prefixes:
        all_regs.extend([f"{prefix}{i}" for i in range(max(2, num_insts // dep_chains))])

    insts = []
    prev_rd: list[Optional[str]] = [None] * dep_chains

    for i in range(num_insts):
        chain = i % dep_chains
        prefix = reg_prefixes[chain]
        op = random.choice(ops)

        if op in ("li", "mv"):
            rd = f"{prefix}{i}"
            src = prev_rd[chain] or f"{prefix}0"
            insts.append(SchedInst(
                id=i, opcode=op, operands=[rd, src],
                defines={rd}, uses={src},
            ))
            prev_rd[chain] = rd
        elif op in ("lw", "sw"):
            rd = f"{prefix}{i}"
            addr_reg = f"{prefix}addr_{chain}"
            insts.append(SchedInst(
                id=i, opcode=op, operands=[rd, f"0({addr_reg})"],
                defines={rd}, uses={addr_reg},
            ))
            prev_rd[chain] = rd
        else:
            rd = f"{prefix}{i}"
            src1 = prev_rd[chain] or f"{prefix}0"
            src2 = f"{prefix}{random.randint(0, max(1, i-1))}"
            insts.append(SchedInst(
                id=i, opcode=op, operands=[rd, src1, src2],
                defines={rd}, uses={src1, src2},
            ))
            prev_rd[chain] = rd

    return insts


def bench_build_dag(insts: list[SchedInst], repeats: int = 20) -> dict:
    """Benchmark DAG construction."""
    scheduler = InstructionScheduler()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        dag = scheduler.build_dag(insts)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return {
        "num_nodes": len(dag),
        "mean_s": statistics.mean(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0,
    }


def bench_schedule(insts: list[SchedInst], repeats: int = 20) -> dict:
    """Benchmark the full scheduling pipeline."""
    times = []
    for _ in range(repeats):
        scheduler = InstructionScheduler()
        t0 = time.perf_counter()
        dag = scheduler.build_dag(insts)
        scheduled = scheduler.schedule(dag)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    orig_cycles = scheduler.estimate_cycles(insts)
    sched_cycles = scheduler.estimate_cycles(scheduled)
    return {
        "num_insts": len(insts),
        "orig_cycles": orig_cycles,
        "sched_cycles": sched_cycles,
        "improvement": orig_cycles - sched_cycles,
        "mean_s": statistics.mean(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Instruction Scheduler Benchmark")
    parser.add_argument("--repeats", type=int, default=20,
                        help="Number of repeat measurements")
    args = parser.parse_args()

    print("=" * 80)
    print("RISC-V Instruction Scheduler (List Scheduling) Benchmark")
    print("=" * 80)

    # Varying instruction count
    print(f"\nVarying Instruction Count (3 dependency chains):")
    print("-" * 80)
    print(f"{'Size':>8} {'Build(ms)':>10} {'Sched(ms)':>10} "
          f"{'OrigCyc':>8} {'SchedCyc':>8} {'Improv':>8} {'%Impr':>7}")
    print("-" * 80)

    sizes = [10, 50, 100, 200, 500, 1000]
    for size in sizes:
        insts = _gen_instructions(size, dep_chains=3)
        dag_stats = bench_build_dag(insts, repeats=args.repeats)
        sched_stats = bench_schedule(insts, repeats=args.repeats)
        pct = (sched_stats["improvement"] / max(sched_stats["orig_cycles"], 1)) * 100
        print(f"{size:>8} {dag_stats['mean_s'] * 1000:>10.3f} "
              f"{sched_stats['mean_s'] * 1000:>10.3f} "
              f"{sched_stats['orig_cycles']:>8} {sched_stats['sched_cycles']:>8} "
              f"{sched_stats['improvement']:>8} {pct:>6.1f}%")

    # Varying dependency depth
    print(f"\nVarying Dependency Depth (200 instructions):")
    print("-" * 70)
    print(f"{'Chains':>8} {'Sched(ms)':>10} {'OrigCyc':>8} "
          f"{'SchedCyc':>8} {'Improv':>8} {'%Impr':>7}")
    print("-" * 70)

    for chains in [1, 2, 5, 10, 20]:
        insts = _gen_instructions(200, dep_chains=chains)
        sched_stats = bench_schedule(insts, repeats=args.repeats)
        pct = (sched_stats["improvement"] / max(sched_stats["orig_cycles"], 1)) * 100
        print(f"{chains:>8} {sched_stats['mean_s'] * 1000:>10.3f} "
              f"{sched_stats['orig_cycles']:>8} {sched_stats['sched_cycles']:>8} "
              f"{sched_stats['improvement']:>8} {pct:>6.1f}%")

    # Large benchmark
    print(f"\nLarge Block Stress Test (5000 instructions):")
    print("-" * 60)
    insts = _gen_instructions(5000, dep_chains=10, seed=99)
    t0 = time.perf_counter()
    scheduler = InstructionScheduler()
    dag = scheduler.build_dag(insts)
    scheduled = scheduler.schedule(dag)
    elapsed = time.perf_counter() - t0
    print(f"  DAG nodes: {len(dag)}")
    print(f"  Scheduled: {len(scheduled)}")
    print(f"  Time: {elapsed * 1000:.3f} ms")
    print(scheduler.report(insts, scheduled))


if __name__ == "__main__":
    main()
