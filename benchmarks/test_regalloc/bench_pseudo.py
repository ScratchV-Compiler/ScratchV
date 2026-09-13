"""Benchmark every RV32IM pseudo supported by the register-allocation path."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import asdict, dataclass

from scratchv.backend.machine_types import (
    ALL_REGS,
    MachineInstr,
    MachineOp,
    MachineOperand,
)
from scratchv.backend.regalloc_linear_v1_5 import (
    LinearScanAllocator,
    block_from_machine_instrs,
)
from scratchv.backend.riscv_encoder import RISCVAEncoder
from scratchv.simulator.tinyfive import ProfiledMachine


BENCHMARKED_MACHINE_PSEUDOS = frozenset(
    {
        MachineOp.MV,
        MachineOp.LI,
        MachineOp.MAX,
        MachineOp.BNEZ,
        MachineOp.J,
        MachineOp.CALL,
        MachineOp.LABEL,
    }
)
BENCHMARKED_ASSEMBLER_PSEUDOS = frozenset({"nop", "ret"})


@dataclass(frozen=True)
class MachinePseudoCase:
    """One allocation, encoding, and execution case for a machine pseudo."""

    name: str
    opcode: MachineOp
    instructions: tuple[MachineInstr, ...]
    expected_a0: int
    instruction_limit: int = 32


@dataclass(frozen=True)
class AssemblerPseudoCase:
    """One encoding and execution case for an assembler-only pseudo."""

    name: str
    assembly: str
    expected_a0: int
    instruction_limit: int = 32


@dataclass(frozen=True)
class PseudoCaseResult:
    """Serializable result for one pseudo benchmark case."""

    name: str
    opcode: str
    layer: str
    mean_s: float
    stdev_s: float
    encoded_instructions: int
    virtual_registers: int
    spill_slots: int
    spill_stores: int
    reloads: int
    pressure_peak: int
    expected_a0: int
    actual_a0: int
    assembly: str

    @property
    def valid(self) -> bool:
        """Return whether execution and allocation invariants hold."""

        return (
            self.actual_a0 == self.expected_a0
            and self.spill_slots == 0
            and self.spill_stores == 0
            and self.reloads == 0
            and "%" not in self.assembly
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe metrics without embedding the full assembly."""

        result = asdict(self)
        result.pop("assembly")
        result["valid"] = self.valid
        return result


def _done_loop() -> tuple[MachineInstr, ...]:
    return (
        MachineInstr(MachineOp.LABEL, comment=".done"),
        MachineInstr(MachineOp.J, comment=".done"),
    )


def machine_pseudo_cases() -> tuple[MachinePseudoCase, ...]:
    """Return one executable Machine IR case for every supported pseudo."""

    v = MachineOperand.vreg
    reg = MachineOperand.reg
    imm = MachineOperand.immediate
    return (
        MachinePseudoCase(
            "mv",
            MachineOp.MV,
            (
                MachineInstr(MachineOp.LI, v("source"), imm(42)),
                MachineInstr(MachineOp.MV, v("copy"), v("source")),
                MachineInstr(MachineOp.MV, reg("a0"), v("copy")),
                *_done_loop(),
            ),
            42,
        ),
        MachinePseudoCase(
            "li",
            MachineOp.LI,
            (
                MachineInstr(MachineOp.LI, v("constant"), imm(0x12345)),
                MachineInstr(MachineOp.MV, reg("a0"), v("constant")),
                *_done_loop(),
            ),
            0x12345,
        ),
        MachinePseudoCase(
            "max",
            MachineOp.MAX,
            (
                MachineInstr(MachineOp.LI, v("left"), imm(-4)),
                MachineInstr(MachineOp.LI, v("right"), imm(-2)),
                MachineInstr(MachineOp.MAX, v("result"), v("left"), v("right")),
                MachineInstr(MachineOp.MV, reg("a0"), v("result")),
                *_done_loop(),
            ),
            -2,
        ),
        MachinePseudoCase(
            "bnez",
            MachineOp.BNEZ,
            (
                MachineInstr(MachineOp.LI, v("condition"), imm(1)),
                MachineInstr(MachineOp.BNEZ, v("condition"), comment=".taken"),
                MachineInstr(MachineOp.LI, reg("a0"), imm(1)),
                MachineInstr(MachineOp.J, comment=".done"),
                MachineInstr(MachineOp.LABEL, comment=".taken"),
                MachineInstr(MachineOp.LI, reg("a0"), imm(2)),
                *_done_loop(),
            ),
            2,
        ),
        MachinePseudoCase(
            "j",
            MachineOp.J,
            (
                MachineInstr(MachineOp.LI, reg("a0"), imm(0)),
                MachineInstr(MachineOp.J, comment=".target"),
                MachineInstr(MachineOp.LI, reg("a0"), imm(1)),
                MachineInstr(MachineOp.LABEL, comment=".target"),
                MachineInstr(MachineOp.LI, reg("a0"), imm(2)),
                *_done_loop(),
            ),
            2,
        ),
        MachinePseudoCase(
            "call",
            MachineOp.CALL,
            (
                MachineInstr(MachineOp.LI, reg("a0"), imm(1)),
                MachineInstr(MachineOp.CALL, comment=".callee"),
                MachineInstr(MachineOp.ADDI, reg("a0"), reg("a0"), imm(10)),
                MachineInstr(MachineOp.J, comment=".done"),
                MachineInstr(MachineOp.LABEL, comment=".callee"),
                MachineInstr(MachineOp.ADDI, reg("a0"), reg("a0"), imm(2)),
                MachineInstr(MachineOp.JALR, reg("zero"), reg("ra"), imm(0)),
                *_done_loop(),
            ),
            13,
        ),
        MachinePseudoCase(
            "label",
            MachineOp.LABEL,
            (
                MachineInstr(MachineOp.LABEL, comment=".entry"),
                MachineInstr(MachineOp.LI, reg("a0"), imm(17)),
                *_done_loop(),
            ),
            17,
        ),
    )


def assembler_pseudo_cases() -> tuple[AssemblerPseudoCase, ...]:
    """Return executable cases for pseudos accepted only by the encoder."""

    return (
        AssemblerPseudoCase(
            "nop",
            "li a0, 41\nnop\naddi a0, a0, 1\n.done:\nj .done",
            42,
        ),
        AssemblerPseudoCase(
            "ret",
            (
                "li a0, 1\n"
                "jal ra, .callee\n"
                "addi a0, a0, 10\n"
                "j .done\n"
                ".callee:\n"
                "addi a0, a0, 2\n"
                "ret\n"
                ".done:\n"
                "j .done"
            ),
            13,
        ),
    )


def _execute(binary: bytes, instruction_limit: int) -> int:
    words = [
        int.from_bytes(binary[offset:offset + 4], "little")
        for offset in range(0, len(binary), 4)
    ]
    machine = ProfiledMachine(mem_size=4096)
    if not machine.available:
        raise RuntimeError("TinyFive is required for pseudo benchmark execution")
    machine.load_binary(words, origin=0)
    machine.run(instructions=instruction_limit, start=0, strict=True)
    return machine.get_reg(10)  # a0


def _run_machine_case(case: MachinePseudoCase, repeats: int) -> PseudoCaseResult:
    times: list[float] = []
    final: tuple[LinearScanAllocator, str, bytes, int] | None = None
    for _ in range(repeats):
        allocator = LinearScanAllocator(phys_regs=list(ALL_REGS))
        block = block_from_machine_instrs(list(case.instructions))
        start = time.perf_counter()
        intervals = allocator.compute_live_intervals(block)
        allocator.allocate(intervals)
        assembly = allocator.get_allocated_code(block)
        binary = bytes(RISCVAEncoder().assemble(assembly))
        times.append(time.perf_counter() - start)
        final = allocator, assembly, binary, len(intervals)

    assert final is not None
    allocator, assembly, binary, virtual_registers = final
    return PseudoCaseResult(
        name=case.name,
        opcode=case.opcode.value,
        layer="machine",
        mean_s=statistics.mean(times),
        stdev_s=statistics.stdev(times) if len(times) > 1 else 0.0,
        encoded_instructions=len(binary) // 4,
        virtual_registers=virtual_registers,
        spill_slots=allocator.spill_slot_count,
        spill_stores=allocator.spill_store_count,
        reloads=allocator.reload_load_count,
        pressure_peak=allocator.pressure_peak,
        expected_a0=case.expected_a0,
        actual_a0=_execute(binary, case.instruction_limit),
        assembly=assembly,
    )


def _run_assembler_case(
    case: AssemblerPseudoCase,
    repeats: int,
) -> PseudoCaseResult:
    times: list[float] = []
    binary = b""
    for _ in range(repeats):
        start = time.perf_counter()
        binary = bytes(RISCVAEncoder().assemble(case.assembly))
        times.append(time.perf_counter() - start)

    return PseudoCaseResult(
        name=case.name,
        opcode=case.name,
        layer="assembler",
        mean_s=statistics.mean(times),
        stdev_s=statistics.stdev(times) if len(times) > 1 else 0.0,
        encoded_instructions=len(binary) // 4,
        virtual_registers=0,
        spill_slots=0,
        spill_stores=0,
        reloads=0,
        pressure_peak=0,
        expected_a0=case.expected_a0,
        actual_a0=_execute(binary, case.instruction_limit),
        assembly=case.assembly,
    )


def run_bench(repeats: int = 30) -> dict[str, object]:
    """Benchmark and execute every supported RV32IM pseudo case."""

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    results = [
        *(_run_machine_case(case, repeats) for case in machine_pseudo_cases()),
        *(_run_assembler_case(case, repeats) for case in assembler_pseudo_cases()),
    ]
    all_times = [result.mean_s for result in results]
    return {
        "mean_s": statistics.mean(all_times),
        "stdev_s": statistics.stdev(all_times) if len(all_times) > 1 else 0.0,
        "vreg_count": sum(result.virtual_registers for result in results),
        "spills": sum(result.spill_stores for result in results),
        "spill_slots": sum(result.spill_slots for result in results),
        "spill_stores": sum(result.spill_stores for result in results),
        "reg_spill_count": sum(result.spill_stores for result in results),
        "reloads": sum(result.reloads for result in results),
        "peak_active": max(result.pressure_peak for result in results),
        "pressure_peak": max(result.pressure_peak for result in results),
        "pressure_excess_peak": 0,
        "asm_lines": sum(result.encoded_instructions for result in results),
        "case_count": len(results),
        "machine_pseudos": sorted(op.value for op in BENCHMARKED_MACHINE_PSEUDOS),
        "assembler_pseudos": sorted(BENCHMARKED_ASSEMBLER_PSEUDOS),
        "cases": [result.to_dict() for result in results],
        "valid": all(result.valid for result in results),
        "_case_results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RV32IM pseudo benchmark")
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    stats = run_bench(repeats=args.repeats)

    print("Pseudo  Layer       Mean(ms)  Encoded  Spill  Result")
    print("-" * 58)
    for result in stats["_case_results"]:
        status = "PASS" if result.valid else "FAIL"
        print(
            f"{result.name:<7} {result.layer:<11} "
            f"{result.mean_s * 1000:>8.3f} "
            f"{result.encoded_instructions:>8} "
            f"{result.spill_stores:>6}  {status}"
        )
    return 0 if stats["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
