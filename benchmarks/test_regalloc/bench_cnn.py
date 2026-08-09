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
    _INT_REGS,
)
from scratchv.backend.register_alloc import RegisterAllocator
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

    vregs: set[str] = set()
    for mi in machine:
        for op in (mi.dst, mi.src1, mi.src2):
            if op and getattr(op, "kind", None) == "vreg":
                vregs.add(str(op.value))
    return machine, ir_count, len(vregs)


# ---------------------------------------------------------------------------
# Assembly validation
# ---------------------------------------------------------------------------
from .bench_utils import _KNOWN_OPS


def _validate_asm(asm: str) -> list[str]:
    """Check no unresolved vregs, valid opcodes."""
    errors: list[str] = []
    for lineno, line in enumerate(asm.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        content = stripped.lstrip()
        if content.endswith(":") or not content:
            continue
        if "#" in content:
            content = content[: content.index("#")].strip()
        parts = content.split()
        if not parts:
            continue
        if parts[0] not in _KNOWN_OPS:
            errors.append(f"Line {lineno}: unknown opcode '{parts[0]}'")
        for token in parts:
            if token.startswith("v") and token[1:].isdigit():
                errors.append(f"Line {lineno}: unresolved vreg '{token}'")
    return errors


# ---------------------------------------------------------------------------
# Emulator verification
# ---------------------------------------------------------------------------


def _run_emulator(cnn_path: str) -> dict:
    """Compile *cnn_path* via standalone pipeline, run through RV32Emulator."""
    try:
        from scratchv.standalone.onnx_to_riscv_standalone import (
            ONNXModel,
            MemoryPlan,
            CNNRISCVGenerator,
        )
        from scratchv.simulator.rv32_emulator import RV32Emulator
    except ImportError as e:
        return {"passed": False, "error": f"import error: {e}"}

    try:
        model = ONNXModel.from_file(cnn_path)
        memory = MemoryPlan()
        memory.layout_weights(model.initializers)
        if model.inputs:
            inp = model.inputs[0]
            el = 1
            for d in model.get_shape(inp.name):
                el *= d
            memory.alloc_workspace(inp.name, el)

        generator = CNNRISCVGenerator(model, memory)
        code_bytes = generator.generate()

        emu = RV32Emulator()
        emu.load_code(code_bytes)
        emu.run(max_instr=100_000)
        return {"passed": True, "error": ""}
    except Exception as exc:
        return {"passed": False, "error": str(exc)[:120]}


# ---------------------------------------------------------------------------
# LLVM comparison
# ---------------------------------------------------------------------------

# Instruction category buckets used to compare opcode mixes between the
# ScratchV backend and the LLVM backend. `sd`/`ld` are ABI stack
# save/restore pairs — the classic LLVM frame-management cost.
from .bench_utils import (
    _CAT_ALU,
    _CAT_LOAD,
    _CAT_STORE,
    _CAT_BRANCH,
    _CAT_MUL,
    _CAT_STACK,
)
from .bench_utils import _op_categories

# RV64 ABI callee-saved registers — `sd`/`ld` to these at sp offsets are
# prologue/epilogue frame save/restore, not spills.
from .bench_utils import _CALLEE_SAVED


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
    from .bench_utils import llvmlite_ir_to_riscv

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
    spill_counts = []

    # Warm up
    for _ in range(repeats):
        alloc = LinearScanAllocator(phys_regs=phys_regs)
        t0 = time.perf_counter()
        alloc.allocate(alloc.compute_live_intervals(block))
        t1 = time.perf_counter()
        times.append(t1 - t0)
        spill_counts.append(len(alloc._spill_slots))

    # Final run for stable stats + assembly validation
    alloc = LinearScanAllocator(phys_regs=phys_regs)
    alloc.allocate(alloc.compute_live_intervals(block))
    code = alloc.get_allocated_code(block)
    asm_errors = _validate_asm(code)
    sv_cnt, sv_cats = count_riscv_instrs(code)

    # Greedy allocator baseline
    t0 = time.perf_counter()
    greedy = RegisterAllocator(machine, mode="greedy")
    greedy_out = greedy.run()
    greedy_time = time.perf_counter() - t0

    return {
        "mean_s": statistics.mean(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0,
        "vreg_total": vreg_total,
        "ir_inst_count": ir_count,
        "machine_instrs": len(machine),
        "vreg_count": len(alloc.alloc_map),
        "reg_spill_count": spill_counts[-1],
        "peak_active": alloc.peak_active,
        "asm_lines": len(code.splitlines()),
        "sv_static_instrs": sv_cnt,
        "sv_cats": sv_cats,
        "sv_cat_buckets": _op_categories(sv_cats),
        "asm_errors": asm_errors,
        "asm_valid": len(asm_errors) == 0,
        "greedy_time_s": greedy_time,
        "greedy_out_instrs": len(greedy_out),
        "_report": alloc.report(),
        "_alloc": alloc,
    }


def run_bench(
    cnn_path: str, phys_regs: list[str] | None = None, repeats: int = 30
) -> dict:
    """Entry point for the test suite runner."""
    if phys_regs is None:
        phys_regs = list(_INT_REGS)
    stats = bench_allocate(cnn_path, phys_regs, repeats=repeats)

    # Emulator verification (non-fatal)
    emu = _run_emulator(cnn_path)
    stats["emu_passed"] = emu["passed"]
    stats["emu_error"] = emu.get("error", "")
    stats["valid"] = stats["asm_valid"]

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

    phys_regs = list(_INT_REGS)

    print("=" * 60)
    print("Benchmark 3 — CNN Model Integration And Comparation With LLVM Backend")
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
            print(f"  ✗ {e}")
    if not stats["emu_passed"]:
        print(f"  Emulator: ✗ {stats['emu_error']}")
    else:
        print(f"  Emulator: ✓ passed")

    # LLVM comparison output
    print()
    print("-" * 55)
    print("  LLVM comparison (O2, same ONNX model)")
    print("-" * 55)

    print(
        f"  ScratchV LinearScan: {stats['sv_static_instrs']} instrs "
        f"{stats['sv_cat_buckets']}"
    )
    print(f"  LLVM RV64IM:         {stats['llvm_im_instrs']} instrs")
    print(
        f"  LLVM RV64FD:         {stats['llvm_fd_instrs']} instrs "
        f"({stats['instr_ratio_fd']}x vs ScratchV) "
        f"{stats['llvm_fd_cat_buckets']}"
    )
    print(
        f"  Spill (LLVM approx): {stats['llvm_spill_slots']} slots "
        f"(frame save/restore {stats['llvm_frame_save']}/"
        f"{stats['llvm_frame_restore']}); "
        f"ScratchV (exact): reg_spill_count={stats['reg_spill_count']}"
    )

    asm_ok = "PASS" if stats["asm_valid"] else "FAIL"
    print(
        f"\n  asm_valid={stats['asm_valid']}, "
        f"reg_spill_count={stats['reg_spill_count']}  [{asm_ok}]"
    )
    return 0 if stats["asm_valid"] else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
