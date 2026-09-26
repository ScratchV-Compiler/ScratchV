"""Detailed register-allocation comparison helpers for Topic 17.

The baseline in this module intentionally reproduces only the historical
``dst = def, src1/src2 = use`` operand inference.  Control-flow targets and
the allocator implementation stay identical in both runs, so any reported
difference is attributable to the machine-semantics table rather than to a
different allocator or compiler revision.
"""

from __future__ import annotations

from collections import Counter

from scratchv.backend.abi_frame import apply_abi_frames
from scratchv.backend.machine_semantics import (
    get_machine_semantics,
    linear_scan_operands,
    virtual_register_defs_uses,
)
from scratchv.backend.machine_types import MachineInstr
from scratchv.backend.regalloc_linear import (
    LinearScanAllocator,
    LsInstruction,
    block_from_machine_instrs,
)
from scratchv.backend.riscv_encoder import RISCVAEncoder
from scratchv.standalone.compare_codegen import count_riscv_instrs


DEFAULT_PRESSURE_REGS = (2, 3, 5, 8, 12, 19)


def legacy_positional_block(instrs: list[MachineInstr]) -> list[LsInstruction]:
    """Build a block with the pre-Topic17 positional def/use inference.

    Branch targets still use the current operand representation.  This keeps
    CFG construction and assembly emission comparable while changing only the
    register semantics under test.
    """

    result: list[LsInstruction] = []
    for index, instruction in enumerate(instrs):
        defines: set[str] = set()
        uses: set[str] = set()
        for position, operand in enumerate(
            (instruction.dst, instruction.src1, instruction.src2)
        ):
            if operand is None or operand.kind != "vreg":
                continue
            name = str(operand.value)
            if position == 0:
                defines.add(name)
            else:
                uses.add(name)

        operands, comment = linear_scan_operands(instruction)
        if instruction.op.value == ".label":
            result.append(
                LsInstruction(
                    id=index,
                    opcode=".label",
                    operands=[instruction.comment],
                    comment=instruction.comment,
                )
            )
        else:
            result.append(
                LsInstruction(
                    id=index,
                    opcode=instruction.op.value,
                    operands=operands,
                    defines=defines,
                    uses=uses,
                    comment=comment,
                )
            )
    return result


def _public_intervals(intervals) -> dict[str, dict[str, object]]:
    return {
        interval.vreg: {
            "start": interval.start,
            "end": interval.end,
            "uses": sorted(interval.uses),
        }
        for interval in intervals
    }


def allocation_snapshot(
    instrs: list[MachineInstr],
    phys_regs: list[str],
    *,
    legacy_semantics: bool = False,
) -> dict[str, object]:
    """Allocate one Machine IR stream and return stable structural metrics."""

    block = (
        legacy_positional_block(instrs)
        if legacy_semantics
        else block_from_machine_instrs(instrs)
    )
    allocator = LinearScanAllocator(phys_regs=phys_regs)
    intervals = []
    try:
        intervals = allocator.compute_live_intervals(block)
        allocator.allocate(intervals)
        assembly = apply_abi_frames(
            allocator.get_allocated_code(block), allocator.spill_slot_count
        )
        binary = bytes(RISCVAEncoder().assemble(assembly))
        static_instructions, opcode_counts = count_riscv_instrs(assembly)
        return {
            "valid": True,
            "error": "",
            "virtual_registers": len(intervals),
            "pressure_peak": allocator.pressure_peak,
            "pressure_excess_peak": allocator.pressure_excess_peak,
            "spill_slots": allocator.spill_slot_count,
            "spill_stores": allocator.spill_store_count,
            "reloads": allocator.reload_load_count,
            "static_instructions": static_instructions,
            "encoded_instructions": len(binary) // 4,
            "opcode_counts": opcode_counts,
            "intervals": _public_intervals(intervals),
        }
    except Exception as exc:
        return {
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
            "virtual_registers": len(intervals),
            "pressure_peak": allocator.pressure_peak,
            "pressure_excess_peak": allocator.pressure_excess_peak,
            "spill_slots": allocator.spill_slot_count,
            "spill_stores": allocator.spill_store_count,
            "reloads": allocator.reload_load_count,
            "static_instructions": 0,
            "encoded_instructions": 0,
            "opcode_counts": {},
            "intervals": _public_intervals(intervals),
        }


def compare_semantics(
    instrs: list[MachineInstr],
    physical_registers: list[str],
) -> dict[str, object]:
    """Compare legacy positional inference with the semantics table."""

    return {
        "baseline": "legacy positional dst/src inference",
        "current": "machine_semantics.py",
        "physical_register_count": len(physical_registers),
        "before": allocation_snapshot(
            instrs, physical_registers, legacy_semantics=True
        ),
        "after": allocation_snapshot(
            instrs, physical_registers, legacy_semantics=False
        ),
    }


def machine_instruction_summary(instrs: list[MachineInstr]) -> dict[str, object]:
    """Return exact opcode and pseudo counts for selected Machine IR."""

    opcode_counts = Counter(instruction.op.value for instruction in instrs)
    pseudo_counts = Counter(
        instruction.op.value
        for instruction in instrs
        if get_machine_semantics(instruction.op).is_pseudo
    )
    return {
        "machine_instructions": len(instrs),
        "opcode_counts": dict(sorted(opcode_counts.items())),
        "pseudo_counts": dict(sorted(pseudo_counts.items())),
    }


def operand_semantics_differences(
    instrs: list[MachineInstr],
) -> list[dict[str, object]]:
    """List instructions whose current def/use sets differ from the baseline."""

    differences: list[dict[str, object]] = []
    legacy = legacy_positional_block(instrs)
    for index, (instruction, legacy_instruction) in enumerate(zip(instrs, legacy)):
        current_defs, current_uses = virtual_register_defs_uses(instruction)
        if (
            current_defs == legacy_instruction.defines
            and current_uses == legacy_instruction.uses
        ):
            continue
        differences.append(
            {
                "instruction_index": index,
                "opcode": instruction.op.value,
                "before_defs": sorted(legacy_instruction.defines),
                "before_uses": sorted(legacy_instruction.uses),
                "after_defs": sorted(current_defs),
                "after_uses": sorted(current_uses),
            }
        )
    return differences
