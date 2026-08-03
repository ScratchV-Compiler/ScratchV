# flake8: noqa
"""Benchmark 3 — CNN model integration with emulator verification.

Compiles the CNN model through the full ScratchV pipeline, runs the
linear scan allocator, validates the output assembly, and optionally
verifies execution via the RV32 emulator.
"""

import argparse
import os
import statistics
import sys
import time

from scratchv.backend.regalloc_linear import (
    LinearScanAllocator,
    block_from_machine_instrs,
    _INT_REGS,
)
from scratchv.backend.register_alloc import RegisterAllocator


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

_KNOWN_OPS = {
    "add",
    "addi",
    "sub",
    "mul",
    "div",
    "rem",
    "and",
    "andi",
    "or",
    "ori",
    "xor",
    "xori",
    "sll",
    "slli",
    "srl",
    "srli",
    "sra",
    "srai",
    "slt",
    "slti",
    "sltu",
    "sltiu",
    "lw",
    "lh",
    "lb",
    "lbu",
    "lhu",
    "sw",
    "sh",
    "sb",
    "beq",
    "bne",
    "blt",
    "bge",
    "bltu",
    "bgeu",
    "jal",
    "jalr",
    "auipc",
    "lui",
    "li",
    "mv",
    "nop",
    "ret",
    "bnez",
    "j",
    "max",
    ".label",
    ".text",
    ".data",
    ".global",
    ".type",
    "flw",
    "fsw",
    "fadd.s",
    "fsub.s",
    "fmul.s",
    "fdiv.s",
    "feq.s",
    "flt.s",
    "fle.s",
    "fcvt.w.s",
    "fcvt.s.w",
}


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
# Benchmark
# ---------------------------------------------------------------------------


def bench_allocate(cnn_path: str, phys_regs: list[str], repeats: int = 30) -> dict:
    """Full CNN compilation pipeline with linear scan allocator."""
    machine, ir_count, vreg_total = _compile_onnx(cnn_path)
    block = block_from_machine_instrs(machine)

    times = []
    spill_counts = []

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
        "spills": spill_counts[-1],
        "reg_spill_count": spill_counts[-1],
        "peak_active": alloc.peak_active,
        "asm_lines": len(code.splitlines()),
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
    print("Benchmark 3 — CNN Model Integration")
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
        f"{stats['vreg_count']:>6} {stats['spills']:>7} "
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

    asm_ok = "PASS" if stats["asm_valid"] else "FAIL"
    print(
        f"\n  asm_valid={stats['asm_valid']}, "
        f"reg_spill_count={stats['spills']}  [{asm_ok}]"
    )
    return 0 if stats["asm_valid"] else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
