# flake8: noqa
"""Benchmark 3 — CNN model integration with emulator verification.

Compiles the CNN model through the full ScratchV pipeline, runs the
linear scan allocator, validates the output assembly, and optionally
verifies execution via the RV32 emulator.
"""

import argparse
import os
import re
import statistics
import sys
import time

from scratchv.backend.regalloc_linear import (
    LinearScanAllocator,
    block_from_machine_instrs,
)
from scratchv.backend.machine_semantics import virtual_register_defs_uses
from scratchv.backend.machine_types import (
    ALL_REGS,
    MachineInstr,
    MachineOp,
    MachineOperand,
)
from scratchv.backend.riscv_encoder import RISCVAEncoder
from scratchv.backend.abi_frame import apply_abi_frames
from scratchv.backend.register_alloc import RegisterAllocator
from scratchv.backend.regalloc_metrics import count_spill_reload_sites
from scratchv.backend.asm_emit import AsmEmitter
from scratchv.standalone.compare_codegen import count_riscv_instrs


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------


def _compile_onnx(onnx_path: str) -> tuple:
    """ONNX → IR → MachineInstr (with virtual registers).

    Returns ``(machine_instrs, ir_inst_count, vreg_count)``.
    """
    from scratchv.frontend.onnx_parser import ONNXParser
    from scratchv.backend.instruction_select import InstructionSelector
    from scratchv.optimizer.constant_folding import ConstantFolder
    from scratchv.optimizer.dead_code import DeadCodeEliminator

    program = ONNXParser().parse(onnx_path)
    ir_count = sum(
        1 for f in program.functions for bb in f.blocks for _ in bb.instructions
    )
    ConstantFolder(program).run()
    DeadCodeEliminator(program).run()
    machine = InstructionSelector(program).run()

    # The scalarized CNN Machine IR names model inputs and initializers as
    # live-ins.  Materialize deterministic values so the allocator output is
    # an executable benchmark program instead of assembly with undefined
    # entry-register contents.
    defined: set[str] = set()
    used: set[str] = set()
    for instruction in machine:
        defs, uses = virtual_register_defs_uses(instruction)
        defined.update(defs)
        used.update(uses)
    live_ins = sorted(used - defined)
    initializers = [
        MachineInstr(
            MachineOp.LI,
            MachineOperand.vreg(name),
            MachineOperand.immediate(1 + sum(map(ord, name)) % 5),
            comment=f"benchmark live-in {name}",
        )
        for name in live_ins
    ]
    insertion = 0
    while insertion < len(machine) and machine[insertion].op == MachineOp.LABEL:
        insertion += 1
    machine[insertion:insertion] = initializers

    vregs: set[str] = set()
    for mi in machine:
        for op in (mi.dst, mi.src1, mi.src2):
            if op and getattr(op, "kind", None) == "vreg":
                vregs.add(str(op.value))
    return machine, ir_count, len(vregs)


# ---------------------------------------------------------------------------
# Assembly validation
# ---------------------------------------------------------------------------
def _validate_asm(asm: str) -> list[str]:
    """Run the emitted program through the real RV32IM encoder."""
    try:
        RISCVAEncoder().assemble(asm)
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return [f"RISCVAEncoder: {type(exc).__name__}: {exc}"]
    return []


# ---------------------------------------------------------------------------
# Emulator verification
# ---------------------------------------------------------------------------


def _interpret_machine(machine: list[MachineInstr]) -> int:
    """Independently interpret the integer Machine IR and return ``a0``."""
    labels = {
        instruction.comment: index
        for index, instruction in enumerate(machine)
        if instruction.op == MachineOp.LABEL
    }
    values: dict[str, int] = {"zero": 0, "x0": 0, "ra": len(machine)}

    def unsigned(value: int) -> int:
        return value & 0xFFFFFFFF

    def signed(value: int) -> int:
        value = unsigned(value)
        return value if value < 0x80000000 else value - 0x100000000

    def read(operand: MachineOperand | None) -> int:
        if operand is None:
            return 0
        if operand.kind == "imm":
            return int(operand.value)
        name = str(operand.value)
        if name not in values:
            raise ValueError(f"undefined Machine IR value: {name}")
        return values[name]

    def write(operand: MachineOperand | None, value: int) -> None:
        if operand is None or str(operand.value) in {"zero", "x0"}:
            return
        values[str(operand.value)] = unsigned(value)

    pc = 0
    steps = 0
    while 0 <= pc < len(machine) and steps < 10_000:
        instruction = machine[pc]
        steps += 1
        next_pc = pc + 1
        left = read(instruction.src1)
        right = read(instruction.src2)
        op = instruction.op
        if op == MachineOp.LABEL:
            pass
        elif op in {MachineOp.LI, MachineOp.MV}:
            write(instruction.dst, left)
        elif op in {MachineOp.ADD, MachineOp.ADDI}:
            write(instruction.dst, left + right)
        elif op == MachineOp.SUB:
            write(instruction.dst, left - right)
        elif op == MachineOp.MUL:
            write(instruction.dst, signed(left) * signed(right))
        elif op == MachineOp.DIV:
            divisor = signed(right)
            dividend = signed(left)
            write(instruction.dst, -1 if divisor == 0 else int(dividend / divisor))
        elif op == MachineOp.REM:
            divisor = signed(right)
            dividend = signed(left)
            quotient = 0 if divisor == 0 else int(dividend / divisor)
            result = (
                dividend if divisor == 0
                else dividend - quotient * divisor
            )
            write(instruction.dst, result)
        elif op == MachineOp.MAX:
            write(instruction.dst, max(signed(left), signed(right)))
        elif op == MachineOp.SLT:
            write(instruction.dst, int(signed(left) < signed(right)))
        elif op == MachineOp.XOR:
            write(instruction.dst, left ^ right)
        elif op == MachineOp.AND:
            write(instruction.dst, left & right)
        elif op == MachineOp.SRAI:
            write(instruction.dst, signed(left) >> (right & 31))
        elif op == MachineOp.BNEZ:
            if read(instruction.dst) != 0:
                next_pc = labels[instruction.comment]
        elif op in {MachineOp.BEQ, MachineOp.BNE, MachineOp.BLT, MachineOp.BGE}:
            comparisons = {
                MachineOp.BEQ: left == right,
                MachineOp.BNE: left != right,
                MachineOp.BLT: signed(left) < signed(right),
                MachineOp.BGE: signed(left) >= signed(right),
            }
            if comparisons[op]:
                next_pc = labels[instruction.comment]
        elif op == MachineOp.J:
            next_pc = labels[instruction.comment]
        elif op in {MachineOp.JAL, MachineOp.CALL}:
            write(instruction.dst or MachineOperand.reg("ra"), pc + 1)
            next_pc = labels[instruction.comment]
        elif op == MachineOp.JALR:
            if str(instruction.dst.value) in {"zero", "x0"}:
                break
            next_pc = read(instruction.src1) + read(instruction.src2)
        else:
            raise ValueError(f"unsupported CNN benchmark opcode: {op.value}")
        pc = next_pc
    else:
        if steps >= 10_000:
            raise RuntimeError("Machine IR reference interpreter did not terminate")
    return values.get("a0", 0) & 0xFFFFFFFF


def _run_emulator(assembly: str, expected_a0: int) -> dict:
    """Execute the allocated assembly itself and compare its return value."""
    try:
        from scratchv.simulator.tinyfive import ProfiledMachine
    except ImportError as e:
        return {"passed": False, "error": f"import error: {e}"}

    try:
        harness = (
            "li sp, 8192\n"
            "jal ra, main_graph\n"
            "li a7, 0x5a5\n"
            "j .bench_done\n"
            + assembly
            + "\n.bench_done:\nj .bench_done"
        )
        binary = bytes(RISCVAEncoder().assemble(harness))
        words = [
            int.from_bytes(binary[offset:offset + 4], "little")
            for offset in range(0, len(binary), 4)
        ]
        profile = ProfiledMachine(mem_size=16384)
        if not profile.available:
            raise RuntimeError("TinyFive is unavailable")
        profile.load_binary(words, origin=0)
        profile.run(instructions=len(words) + 16, start=0, strict=True)
        actual = profile.get_reg(10) & 0xFFFFFFFF
        returned = profile.get_reg(17) == 0x5A5
        return {
            "passed": returned and actual == expected_a0,
            "error": "" if returned and actual == expected_a0 else (
                f"allocated assembly returned={returned}, "
                f"a0={actual}, expected={expected_a0}"
            ),
            "actual_a0": actual,
            "expected_a0": expected_a0,
        }
    except Exception as exc:
        return {"passed": False, "error": str(exc)[:120]}


# ---------------------------------------------------------------------------
# LLVM comparison
# ---------------------------------------------------------------------------

# Instruction category buckets used to compare opcode mixes between the
# ScratchV backend and the LLVM backend. `sd`/`ld` are ABI stack
# save/restore pairs — the classic LLVM frame-management cost.
from benchmarks.test_regalloc.bench_utils import (
    _CAT_ALU,
    _CAT_LOAD,
    _CAT_STORE,
    _CAT_BRANCH,
    _CAT_MUL,
    _CAT_STACK,
)
from benchmarks.test_regalloc.bench_utils import _op_categories

# RV64 ABI callee-saved registers — `sd`/`ld` to these at sp offsets are
# prologue/epilogue frame save/restore, not spills.
from benchmarks.test_regalloc.bench_utils import _CALLEE_SAVED


def _llvm_spill_stats(asm: str) -> dict:
    """Approximate LLVM spill/frame stats from RISC-V assembly.

    libLLVM codegen does not expose regalloc pass statistics, so spill
    counts are inferred from sp-based memory accesses:
      - ``sd``/``ld`` to callee-saved regs → ABI frame save/restore
      - 4-byte ``sw``/``lw``/``fsw``/``flw`` → spilled values
        (each spill site emits a store + reload pair)

    Returns ``{llvm_spill_slots, llvm_frame_save, llvm_frame_restore}``.
    """
    frame_save = frame_restore = spill4 = 0
    for line in asm.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(sd|ld|sw|lw|fsw|flw)\s+([^,]+),\s*(-?\d+)\(sp\)", line)
        if not m:
            continue
        op, reg, _ = m.group(1), m.group(2).strip(), int(m.group(3))
        if op in ("sd", "ld") and reg in _CALLEE_SAVED:
            if op == "sd":
                frame_save += 1
            else:
                frame_restore += 1
        elif op in ("sw", "lw", "fsw", "flw"):
            spill4 += 1
    return {
        "llvm_spill_slots": spill4 // 2,
        "llvm_frame_save": frame_save,
        "llvm_frame_restore": frame_restore,
    }


def _llvm_compare(cnn_path: str) -> dict:
    """Compile *cnn_path* via LLVM (O2) and return comparison stats.

    Reuses the libLLVM pipeline from ``compare_codegen.py``: ONNX →
    LLVM IR → RISC-V assembly at both RV64IM and RV64FD feature sets.
    """
    from scratchv.standalone.compare_codegen import _load_llvm, llvm_ir_to_riscv
    from scratchv.standalone.onnx_to_llvm_standalone import convert_onnx_to_llvm
    from benchmarks.test_regalloc.bench_utils import llvmlite_ir_to_riscv

    # lib = _load_llvm()
    ir = convert_onnx_to_llvm(cnn_path)
    im_cnt, im_asm, _ = llvmlite_ir_to_riscv(ir, "+m", 2)
    fd_cnt, fd_asm, fd_cats = llvmlite_ir_to_riscv(ir, "+m,+f,+d", 2)

    result = {
        "llvm_im_instrs": im_cnt,
        "llvm_fd_instrs": fd_cnt,
        "llvm_fd_cats": fd_cats,
        "llvm_fd_cat_buckets": _op_categories(fd_cats),
        "_llvm_fd_asm": fd_asm,
        "_llvm_im_asm": im_asm,
    }
    result.update(_llvm_spill_stats(fd_asm))
    return result


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def bench_allocate(cnn_path: str, phys_regs: list[str], repeats: int = 30) -> dict:
    """Full CNN compilation pipeline with linear scan allocator."""
    machine, ir_count, vreg_total = _compile_onnx(cnn_path)
    block = block_from_machine_instrs(machine)

    times = []
    # Warm up
    for _ in range(repeats):
        alloc = LinearScanAllocator(phys_regs=phys_regs)
        t0 = time.perf_counter()
        alloc.allocate(alloc.compute_live_intervals(block))
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # Final run for stable stats + assembly validation
    alloc = LinearScanAllocator(phys_regs=phys_regs)
    intervals = alloc.compute_live_intervals(block)
    alloc.allocate(intervals)
    code = apply_abi_frames(
        alloc.get_allocated_code(block), alloc.spill_slot_count
    )
    asm_errors = _validate_asm(code)
    expected_a0 = _interpret_machine(machine)
    sv_cnt, sv_cats = count_riscv_instrs(code)

    # Greedy allocator baseline.  Measure it with the same Machine IR and the
    # same repeat count as LinearScan so the before/after time is comparable.
    greedy_times = []
    for _ in range(repeats):
        greedy = RegisterAllocator(machine, mode="greedy")
        t0 = time.perf_counter()
        greedy.run()
        greedy_times.append(time.perf_counter() - t0)

    # Final run for stable assembly, spill, and execution metrics.
    greedy = RegisterAllocator(machine, mode="greedy")
    greedy_out = greedy.run()
    greedy_code = apply_abi_frames(
        AsmEmitter(greedy_out).emit(), greedy.spill_slot_count
    )
    greedy_static_instrs, _ = count_riscv_instrs(greedy_code)
    greedy_spill_stores, greedy_reloads = count_spill_reload_sites(greedy_code)
    greedy_errors = _validate_asm(greedy_code)
    greedy_emu = _run_emulator(greedy_code, expected_a0)

    return {
        "mean_s": statistics.mean(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0,
        "vreg_total": vreg_total,
        "ir_inst_count": ir_count,
        "machine_instrs": len(machine),
        "vreg_count": len(intervals),
        "phys_reg_count": len(phys_regs),
        "spill_slots": alloc.spill_slot_count,
        "spill_stores": alloc.spill_store_count,
        "reg_spill_count": alloc.spill_store_count,
        "reloads": alloc.reload_load_count,
        "peak_active": alloc.peak_active,
        "pressure_peak": alloc.pressure_peak,
        "pressure_excess_peak": alloc.pressure_excess_peak,
        "asm_lines": len(code.splitlines()),
        "sv_static_instrs": sv_cnt,
        "sv_cats": sv_cats,
        "sv_cat_buckets": _op_categories(sv_cats),
        "asm_errors": asm_errors,
        "asm_valid": len(asm_errors) == 0,
        "greedy_time_s": statistics.mean(greedy_times),
        "greedy_stdev_s": (
            statistics.stdev(greedy_times) if len(greedy_times) > 1 else 0
        ),
        "greedy_out_instrs": len(greedy_out),
        "greedy_static_instrs": greedy_static_instrs,
        "greedy_spill_slots": greedy.spill_slot_count,
        "greedy_spill_stores": greedy_spill_stores,
        "greedy_reloads": greedy_reloads,
        "greedy_asm_valid": not greedy_errors,
        "greedy_asm_errors": greedy_errors,
        "greedy_emu_passed": greedy_emu["passed"],
        "greedy_emu_error": greedy_emu.get("error", ""),
        "_report": alloc.report(),
        "_alloc": alloc,
        "_assembly": code,
        "expected_a0": expected_a0,
    }


def _improvement_pct(before: float, after: float) -> float | None:
    """Return the percentage reduction; positive values are improvements."""

    if before == 0:
        return 0.0 if after == 0 else None
    return round((before - after) / before * 100, 2)


def _build_optimization_comparison(stats: dict) -> dict | None:
    """Build a machine-readable Greedy -> LinearScan comparison."""

    required = {
        "greedy_time_s",
        "mean_s",
        "greedy_static_instrs",
        "sv_static_instrs",
        "greedy_spill_slots",
        "spill_slots",
        "greedy_spill_stores",
        "spill_stores",
        "greedy_reloads",
        "reloads",
    }
    if not required.issubset(stats):
        return None

    def metric(label: str, unit: str, before: float, after: float) -> dict:
        return {
            "label": label,
            "unit": unit,
            "before": before,
            "after": after,
            "improvement_pct": _improvement_pct(before, after),
        }

    return {
        "baseline": "Greedy allocator",
        "optimized": "Topic17 LinearScan",
        "same_machine_ir": True,
        "metrics": {
            "allocation_time": metric(
                "Allocation mean",
                "ms",
                stats["greedy_time_s"] * 1000,
                stats["mean_s"] * 1000,
            ),
            "static_instructions": metric(
                "Static instructions",
                "instructions",
                stats["greedy_static_instrs"],
                stats["sv_static_instrs"],
            ),
            "spill_slots": metric(
                "Spill slots",
                "slots",
                stats["greedy_spill_slots"],
                stats["spill_slots"],
            ),
            "spill_stores": metric(
                "Spill stores",
                "instructions",
                stats["greedy_spill_stores"],
                stats["spill_stores"],
            ),
            "reloads": metric(
                "Reloads",
                "instructions",
                stats["greedy_reloads"],
                stats["reloads"],
            ),
        },
        "correctness": {
            "before": bool(
                stats.get("greedy_asm_valid")
                and stats.get("greedy_emu_passed")
            ),
            "after": bool(stats.get("asm_valid") and stats.get("emu_passed")),
        },
    }


def run_bench(
    cnn_path: str, phys_regs: list[str] | None = None, repeats: int = 30
) -> dict:
    """Entry point for the test suite runner."""
    if phys_regs is None:
        phys_regs = list(ALL_REGS)
    stats = bench_allocate(cnn_path, phys_regs, repeats=repeats)

    # Emulator verification is part of the end-to-end validity contract.
    emu = _run_emulator(
        stats.get("_assembly", ""), stats.get("expected_a0", 0)
    )
    stats["emu_passed"] = emu["passed"]
    stats["emu_error"] = emu.get("error", "")
    stats["actual_a0"] = emu.get("actual_a0")
    stats["valid"] = (
        stats["asm_valid"]
        and stats["emu_passed"]
        and stats.get("greedy_asm_valid", True)
        and stats.get("greedy_emu_passed", True)
    )

    # LLVM comparison (non-fatal)
    try:
        stats.update(_llvm_compare(cnn_path))
        stats["llvm_available"] = True
    except Exception as exc:
        stats["llvm_available"] = False
        stats["llvm_error"] = str(exc)[:120]
    if stats["llvm_available"]:
        stats["instr_ratio_fd"] = round(
            stats["llvm_fd_instrs"] / max(stats["sv_static_instrs"], 1), 2
        )
    comparison = _build_optimization_comparison(stats)
    if comparison is not None:
        stats["optimization_comparison"] = comparison
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark 3 — CNN Model Integration")
    parser.add_argument(
        "--repeats", type=int, default=30, help="Number of repeat measurements"
    )
    parser.add_argument("--cnn-path", default="", help="Path to ONNX model")

    args = parser.parse_args()

    if not args.cnn_path:
        args.cnn_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "models",
            "graph",
            "cnn.onnx",
        )

    phys_regs = list(ALL_REGS)

    print("=" * 60)
    print("Benchmark 3 - CNN Model Integration And Comparison With LLVM Backend")
    print(f"  Model: {os.path.basename(args.cnn_path)}")
    print("=" * 60)

    stats = run_bench(args.cnn_path, phys_regs=phys_regs, repeats=args.repeats)

    print(
        f"\n{'':>8} {'Mean(ms)':>10} {'Stdev(ms)':>10} "
        f"{'Vregs':>6} {'Spills':>7} {'Peak':>6} {'Asm':>5}"
    )
    print("-" * 55)
    print(
        f"{'cnn':>8} {stats['mean_s'] * 1000:>10.3f} "
        f"{stats['stdev_s'] * 1000:>10.3f} "
        f"{stats['vreg_count']:>6} {stats['reg_spill_count']:>7} "
        f"{stats['peak_active']:>6} {stats['asm_lines']:>5}"
    )

    print()
    print(stats["_report"])
    print(
        f"  Greedy baseline: {stats['greedy_time_s'] * 1000:.3f}ms, "
        f"{stats['greedy_out_instrs']} instrs"
    )

    if not stats["asm_valid"]:
        for e in stats["asm_errors"][:3]:
            print(f"  FAIL {e}")
    if not stats["emu_passed"]:
        print(f"  Emulator: FAIL {stats['emu_error']}")
    else:
        print("  Emulator: PASS")

    # LLVM comparison output
    print()
    print("-" * 55)
    print("  LLVM comparison (O2, same ONNX model)")
    print("-" * 55)

    print(
        f"  ScratchV LinearScan: {stats['sv_static_instrs']} instrs "
        f"{stats['sv_cat_buckets']}"
    )
    if stats["llvm_available"]:
        print(f"  LLVM RV64IM:         {stats['llvm_im_instrs']} instrs")
        print(
            f"  LLVM RV64FD:         {stats['llvm_fd_instrs']} instrs "
            f"({stats['instr_ratio_fd']}x vs ScratchV) "
            f"{stats['llvm_fd_cat_buckets']}"
        )
        print(
            f"  Spill (LLVM approx): {stats['llvm_spill_slots']} slots "
            f"(frame save/restore {stats['llvm_frame_save']}/"
            f"{stats['llvm_frame_restore']})"
        )
    else:
        print(f"  LLVM: unavailable ({stats['llvm_error']})")
    print(
        "  ScratchV regalloc: "
        f"spill_slots={stats['spill_slots']}, "
        f"spill_stores={stats['spill_stores']}, "
        f"reloads={stats['reloads']}"
    )

    benchmark_ok = "PASS" if stats["valid"] else "FAIL"
    print(
        f"\n  asm_valid={stats['asm_valid']}, "
        f"reg_spill_count={stats['reg_spill_count']}  [{benchmark_ok}]"
    )
    return 0 if stats["valid"] else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
