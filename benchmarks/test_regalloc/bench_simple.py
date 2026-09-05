# flake8: noqa
"""Benchmark 1 — Simple arithmetic (3-5 vregs, no spilling).

Verifies that the linear scan allocator produces zero spills when
physical registers are plentiful.
"""

import argparse
import os
import random
import statistics
import sys
import time

from scratchv.backend.regalloc_linear_v1_5 import LinearScanAllocator, LsInstruction


def _gen_block(
    num_insts: int = 10, num_vregs: int = 5, seed: int = 42
) -> list[LsInstruction]:
    """Generate a basic block with simple arithmetic using few vregs."""
    random.seed(seed)
    ops = ["add", "sub", "mul", "and", "or"]
    vreg_names = [f"v{i}" for i in range(num_vregs)]
    insts = []

    for i in range(num_insts):
        if i < num_vregs:
            dst = vreg_names[i]
            pool = vreg_names[: max(i, 1)]
            src1 = random.choice(pool)
            src2 = random.choice(pool)
            insts.append(
                LsInstruction(
                    id=i,
                    opcode=random.choice(ops),
                    operands=[dst, src1, src2],
                    defines={dst},
                    uses={src1, src2},
                )
            )
        else:
            dst = random.choice(vreg_names)
            src1 = random.choice(vreg_names)
            src2 = random.choice(vreg_names)
            insts.append(
                LsInstruction(
                    id=i,
                    opcode=random.choice(ops),
                    operands=[dst, src1, src2],
                    defines={dst},
                    uses={src1, src2} - {dst},
                )
            )
    return insts


def bench_allocate(
    block: list[LsInstruction], phys_regs: list[str], repeats: int = 50
) -> dict:
    """Benchmark the full allocation pipeline."""
    times = []
    for _ in range(repeats):
        alloc = LinearScanAllocator(phys_regs=phys_regs)
        t0 = time.perf_counter()
        alloc.allocate(alloc.compute_live_intervals(block))
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # One final run for stable stats
    alloc = LinearScanAllocator(phys_regs=phys_regs)
    alloc.allocate(alloc.compute_live_intervals(block))
    code = alloc.get_allocated_code(block)

    return {
        "mean_s": statistics.mean(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0,
        "vreg_count": len(alloc.alloc_map),
        "spills": alloc.spill_store_count,
        "spill_slots": len(alloc._spill_slots),
        "spill_stores": alloc.spill_store_count,
        "reg_spill_count": alloc.spill_store_count,
        "reloads": alloc.reload_load_count,
        "peak_active": alloc.peak_active,
        "pressure_peak": alloc.pressure_peak,
        "pressure_excess_peak": alloc.pressure_excess_peak,
        "asm_lines": len(code.splitlines()),
        "_report": alloc.report(),
        "_alloc": alloc,
    }


def run_bench(phys_regs: list[str] | None = None, repeats: int = 50) -> dict:
    """Entry point for the test suite runner."""
    if phys_regs is None:
        phys_regs = [f"r{i}" for i in range(8)]
    block = _gen_block(num_insts=10, num_vregs=5)
    stats = bench_allocate(block, phys_regs, repeats=repeats)
    stats["valid"] = stats["spills"] == 0
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 1 — Simple Arithmetic (no-spill)"
    )
    parser.add_argument(
        "--repeats", type=int, default=50, help="Number of repeat measurements"
    )
    args = parser.parse_args()

    phys_regs = [f"r{i}" for i in range(8)]

    print("=" * 60)
    print("Benchmark 1 — Simple Arithmetic (5 vregs / 8 phys regs)")
    print("=" * 60)

    stats = run_bench(phys_regs=phys_regs, repeats=args.repeats)

    print(f"\n{'':>8} {'Mean(ms)':>10} {'Stdev(ms)':>10} {'Vregs':>6} {'Spills':>7}")
    print("-" * 50)
    print(
        f"{'simple':>8} {stats['mean_s'] * 1000:>10.3f} "
        f"{stats['stdev_s'] * 1000:>10.3f} "
        f"{stats['vreg_count']:>6} {stats['spills']:>7}"
    )

    print()
    print(stats["_report"])

    spills = stats["spills"]
    ok = "PASS" if spills == 0 else "FAIL"
    print(f"\n  reg_spill_count={spills}  [{ok}]")
    return 0 if spills == 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
