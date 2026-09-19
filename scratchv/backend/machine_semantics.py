"""Central register semantics for machine instructions.

The linear-scan allocators must reason about the *meaning* of operands,
not about the historical ``dst/src1/src2`` field names.  This module is the
single source of truth for positional defs/uses and pseudo-instruction
metadata.  Every ``MachineOp`` has an explicit entry so a newly added opcode
cannot silently inherit incorrect register semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scratchv.backend.machine_types import ARG_REGS, TEMP_REGS, MachineOp

if TYPE_CHECKING:
    from scratchv.backend.machine_types import MachineInstr


@dataclass(frozen=True)
class MachineOpSemantics:
    """Register-allocation and emission metadata for one machine opcode.

    Operand positions use ``0=dst``, ``1=src1`` and ``2=src2``.  An entry in
    ``immediate_positions`` means that position may contain an immediate; a
    virtual register in the same position is still treated according to
    ``uses``.  ``n_phys`` is the number of register operands needed after
    pseudo expansion (not the number of emitted instructions), matching the
    Topic17 terminology.
    """

    defs: tuple[int, ...] = ()
    uses: tuple[int, ...] = ()
    immediate_positions: tuple[int, ...] = ()
    is_terminator: bool = False
    target_from_comment: bool = False
    target_required: bool = False
    implicit_defs: frozenset[str] = frozenset()
    implicit_uses: frozenset[str] = frozenset()
    clobbers: frozenset[str] = frozenset()
    is_call: bool = False
    n_phys: int = 0
    is_pseudo: bool = False
    is_label: bool = False


_DEF_USE_USE = MachineOpSemantics(defs=(0,), uses=(1, 2), n_phys=3)
_DEF_USE = MachineOpSemantics(defs=(0,), uses=(1,), n_phys=2)
_STORE = MachineOpSemantics(uses=(0, 1), n_phys=2)
_NO_REGISTERS = MachineOpSemantics()


OP_SEM: dict[MachineOp, MachineOpSemantics] = {
    MachineOp.ADD: _DEF_USE_USE,
    MachineOp.ADDI: MachineOpSemantics(
        defs=(0,), uses=(1,), immediate_positions=(2,), n_phys=2
    ),
    MachineOp.SUB: _DEF_USE_USE,
    MachineOp.MUL: _DEF_USE_USE,
    MachineOp.DIV: _DEF_USE_USE,
    MachineOp.SRAI: MachineOpSemantics(
        defs=(0,), uses=(1,), immediate_positions=(2,), n_phys=2
    ),
    MachineOp.XOR: _DEF_USE_USE,
    MachineOp.AND: _DEF_USE_USE,
    MachineOp.SLT: _DEF_USE_USE,
    MachineOp.REM: _DEF_USE_USE,
    MachineOp.LW: _DEF_USE,
    MachineOp.SW: _STORE,
    MachineOp.FLD: _DEF_USE,
    MachineOp.FSD: _STORE,
    # mv rd, rs -> addi rd, rs, 0
    MachineOp.MV: MachineOpSemantics(
        defs=(0,),
        uses=(1,),
        n_phys=2,
        is_pseudo=True,
    ),
    # li rd, imm -> addi rd, x0, imm or lui/addi for a large immediate
    MachineOp.LI: MachineOpSemantics(
        defs=(0,),
        immediate_positions=(1,),
        n_phys=1,
        is_pseudo=True,
    ),
    # ScratchV's RV32IM max pseudo accepts a register rhs or immediate zero.
    MachineOp.MAX: MachineOpSemantics(
        defs=(0,),
        uses=(1, 2),
        immediate_positions=(2,),
        n_phys=3,
        is_pseudo=True,
    ),
    MachineOp.LABEL: MachineOpSemantics(
        n_phys=0,
        is_pseudo=True,
        is_label=True,
    ),
    # bnez rs, label -> bne rs, x0, label
    MachineOp.BNEZ: MachineOpSemantics(
        uses=(0,),
        is_terminator=True,
        target_from_comment=True,
        target_required=True,
        n_phys=1,
        is_pseudo=True,
    ),
    # j label -> jal x0, label
    MachineOp.J: MachineOpSemantics(
        is_terminator=True,
        target_from_comment=True,
        target_required=True,
        n_phys=0,
        is_pseudo=True,
    ),
    MachineOp.JALR: MachineOpSemantics(
        defs=(0,),
        uses=(1,),
        immediate_positions=(2,),
        is_terminator=True,
        n_phys=2,
    ),
    MachineOp.JAL: MachineOpSemantics(
        defs=(0,),
        is_terminator=True,
        target_from_comment=True,
        target_required=True,
        n_phys=1,
    ),
    MachineOp.BEQ: MachineOpSemantics(
        uses=(0, 1),
        is_terminator=True,
        target_from_comment=True,
        target_required=True,
        n_phys=2,
    ),
    MachineOp.BNE: MachineOpSemantics(
        uses=(0, 1),
        is_terminator=True,
        target_from_comment=True,
        target_required=True,
        n_phys=2,
    ),
    MachineOp.BLT: MachineOpSemantics(
        uses=(0, 1),
        is_terminator=True,
        target_from_comment=True,
        target_required=True,
        n_phys=2,
    ),
    MachineOp.BGE: MachineOpSemantics(
        uses=(0, 1),
        is_terminator=True,
        target_from_comment=True,
        target_required=True,
        n_phys=2,
    ),
    # The CFG-aware spill rewriter and greedy allocator use these ABI clobbers
    # to preserve values that remain live across a call.
    MachineOp.CALL: MachineOpSemantics(
        target_from_comment=True,
        target_required=True,
        implicit_defs=frozenset({"ra"}),
        clobbers=frozenset({"ra", *ARG_REGS, *TEMP_REGS}),
        is_call=True,
        n_phys=1,
        is_pseudo=True,
    ),
    MachineOp.SECTION: _NO_REGISTERS,
    MachineOp.GLOBL: _NO_REGISTERS,
    MachineOp.SIZE: _NO_REGISTERS,
    MachineOp.TYPE: _NO_REGISTERS,
    MachineOp.SQRT_S: _DEF_USE,
    MachineOp.SQRT_D: _DEF_USE,
    MachineOp.FMIN_D: _DEF_USE_USE,
    MachineOp.FMAX_D: _DEF_USE_USE,
    MachineOp.FABS_D: MachineOpSemantics(
        defs=(0,), uses=(1,), n_phys=2, is_pseudo=True
    ),
    MachineOp.FNEG_D: MachineOpSemantics(
        defs=(0,), uses=(1,), n_phys=2, is_pseudo=True
    ),
    MachineOp.FADD_D: _DEF_USE_USE,
    MachineOp.FSUB_D: _DEF_USE_USE,
    MachineOp.FMUL_D: _DEF_USE_USE,
    MachineOp.FDIV_D: _DEF_USE_USE,
    MachineOp.FLT_D: _DEF_USE_USE,
    MachineOp.FEQ_D: _DEF_USE_USE,
    MachineOp.FCVT_S_D: _DEF_USE,
    MachineOp.FCVT_D_S: _DEF_USE,
    MachineOp.LI_D: MachineOpSemantics(
        defs=(0,), immediate_positions=(1,), n_phys=1, is_pseudo=True
    ),
    MachineOp.FADD_S: _DEF_USE_USE,
    MachineOp.FSUB_S: _DEF_USE_USE,
    MachineOp.FMUL_S: _DEF_USE_USE,
    MachineOp.FDIV_S: _DEF_USE_USE,
    MachineOp.FMAX_S: _DEF_USE_USE,
    MachineOp.FMIN_S: _DEF_USE_USE,
    MachineOp.FLE_S: _DEF_USE_USE,
    MachineOp.FLT_S: _DEF_USE_USE,
    MachineOp.FEQ_S: _DEF_USE_USE,
    MachineOp.FLW: _DEF_USE,
    MachineOp.FSW: _STORE,
    MachineOp.FMV_S: MachineOpSemantics(
        defs=(0,), uses=(1,), n_phys=2, is_pseudo=True
    ),
    MachineOp.FMV_S_X: _DEF_USE,
}


_MISSING_SEMANTICS = set(MachineOp) - set(OP_SEM)
if _MISSING_SEMANTICS:
    missing = ", ".join(sorted(op.value for op in _MISSING_SEMANTICS))
    raise RuntimeError(f"missing machine semantics for: {missing}")


def get_machine_semantics(op: MachineOp) -> MachineOpSemantics:
    """Return the explicit semantics for *op*."""

    return OP_SEM[op]


def virtual_register_defs_uses(
    instr: "MachineInstr",
) -> tuple[set[str], set[str]]:
    """Collect virtual-register defs and uses according to opcode semantics."""

    semantics = get_machine_semantics(instr.op)
    operands = (instr.dst, instr.src1, instr.src2)

    def _names_at(positions: tuple[int, ...]) -> set[str]:
        names: set[str] = set()
        for position in positions:
            operand = operands[position]
            if operand is not None and operand.kind == "vreg":
                names.add(str(operand.value))
        return names

    return _names_at(semantics.defs), _names_at(semantics.uses)


def linear_scan_operands(instr: "MachineInstr") -> tuple[list[str], str]:
    """Return emitted operands and any remaining non-semantic comment.

    Branch targets historically live in ``MachineInstr.comment``.  At the
    linear-scan boundary they become real assembly operands so later comment
    stripping cannot erase control-flow semantics.
    """

    operands = [
        str(operand).lstrip("%")
        for operand in (instr.dst, instr.src1, instr.src2)
        if operand is not None
    ]
    semantics = get_machine_semantics(instr.op)
    comment = instr.comment
    if semantics.target_from_comment:
        if semantics.target_required and not comment:
            raise ValueError(f"{instr.op.value} requires a target label")
        if comment:
            operands.append(comment)
            comment = ""
    return operands, comment
