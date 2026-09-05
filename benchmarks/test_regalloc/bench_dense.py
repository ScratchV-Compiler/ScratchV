# flake8: noqa
"""Benchmark 2 — Dense computation (30 vregs, triggers spilling).

Forces the linear scan allocator to spill by providing more virtual
registers than physical registers.
"""

import argparse
import os
import random
import statistics
import sys
import time

from scratchv.backend.regalloc_linear_v1_5 import LinearScanAllocator, LsInstruction


def _gen_block(
    num_insts: int = 80, num_vregs: int = 30, seed: int = 42
) -> list[LsInstruction]:
    """Generate a high-register-pressure block."""
    random.seed(seed)
    ops = ["add", "sub", "mul", "and", "or", "xor", "sll", "srl"]
    vreg_names = [f"v{i}" for i in range(num_vregs)]
    insts = []

    # Phase 1: define each vreg — creates long live ranges
    for i in range(num_vregs):
        insts.append(
            LsInstruction(
                id=i,
                opcode="addi",
                operands=[vreg_names[i], "zero", str(random.randint(1, 100))],
                defines={vreg_names[i]},
                uses=set(),
                comment=f"def {vreg_names[i]}",
            )
        )

    # Phase 2: cross-reference dense ops — keeps many vregs live
    for i in range(num_vregs, num_insts):
        src1 = random.choice(vreg_names)
        src2 = random.choice(vreg_names)
        dst = random.choice(vreg_names)
        insts.append(
            LsInstruction(
                id=i,
                opcode=random.choice(ops),
                operands=[dst, src1, src2],
                defines={dst},
                uses={src1, src2},
                comment=f"dense op {i}",
            )
        )
    return insts


def bench_allocate(
    block: list[LsInstruction], phys_regs: list[str], repeats: int = 30
) -> dict:
    """Benchmark the full allocation pipeline under register pressure."""
    times = []
    for _ in range(repeats):
        alloc = LinearScanAllocator(phys_regs=phys_regs)
        t0 = time.perf_counter()
        alloc.allocate(alloc.compute_live_intervals(block))
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # Final run for stable stats
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
        "peak_active": alloc.peak_active,
        "pressure_peak": alloc.pressure_peak,
        "pressure_excess_peak": alloc.pressure_excess_peak,
        "asm_lines": len(code.splitlines()),
        "reloads": alloc.reload_load_count,
        "_report": alloc.report(),
        "_alloc": alloc,
    }


def run_bench(phys_regs: list[str] | None = None, repeats: int = 30) -> dict:
    """Entry point for the test suite runner."""
    if phys_regs is None:
        phys_regs = [f"r{i}" for i in range(5)]
    block = _gen_block(num_insts=80, num_vregs=30)
    stats = bench_allocate(block, phys_regs, repeats=repeats)
    stats["valid"] = stats["spills"] > 0
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 2 — Dense Computation (spills)"
    )
    parser.add_argument(
        "--repeats", type=int, default=30, help="Number of repeat measurements"
    )
    args = parser.parse_args()

    phys_regs = [f"r{i}" for i in range(5)]

    print("=" * 60)
    print("Benchmark 2 — Dense Computation (30 vregs / 5 phys regs)")
    print("=" * 60)

    stats = run_bench(phys_regs=phys_regs, repeats=args.repeats)

    print(
        f"\n{'':>8} {'Mean(ms)':>10} {'Stdev(ms)':>10} "
        f"{'Vregs':>6} {'Spills':>7} {'Peak':>6} {'Reloads':>8} "
        f"{'Asm':>5}"
    )
    print("-" * 65)
    print(
        f"{'dense':>8} {stats['mean_s'] * 1000:>10.3f} "
        f"{stats['stdev_s'] * 1000:>10.3f} "
        f"{stats['vreg_count']:>6} {stats['spills']:>7} "
        f"{stats['peak_active']:>6} {stats['reloads']:>8} "
        f"{stats['asm_lines']:>5}"
    )

    print()
    print(stats["_report"])

    spills = stats["spills"]
    ok = "PASS" if spills > 0 else "FAIL (expected spills)"
    print(f"\n  reg_spill_count={spills}  [{ok}]")
    return 0 if spills > 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
