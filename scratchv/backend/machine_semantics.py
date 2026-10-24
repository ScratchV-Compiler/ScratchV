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

from scratchv.backend.machine_types import (
    ARG_REGS,
    TEMP_REGS,
    FP_TEMP_REGS,
    FP_ARG_REGS,
    MachineOp,
    RegisterClass,
    physical_register_class,
)

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
    operand_classes: tuple[
        RegisterClass | None,
        RegisterClass | None,
        RegisterClass | None,
    ] = (None, None, None)


_GPR = RegisterClass.GPR
_FPR = RegisterClass.FPR
_DEF_USE_USE = MachineOpSemantics(
    defs=(0,), uses=(1, 2), n_phys=3,
    operand_classes=(_GPR, _GPR, _GPR),
)
_DEF_USE = MachineOpSemantics(
    defs=(0,), uses=(1,), n_phys=2,
    operand_classes=(_GPR, _GPR, None),
)
_STORE = MachineOpSemantics(
    uses=(0, 1), n_phys=2,
    operand_classes=(_GPR, _GPR, None),
)
_FP_DEF_USE_USE = MachineOpSemantics(
    defs=(0,), uses=(1, 2), n_phys=3,
    operand_classes=(_FPR, _FPR, _FPR),
)
_FP_DEF_USE = MachineOpSemantics(
    defs=(0,), uses=(1,), n_phys=2,
    operand_classes=(_FPR, _FPR, None),
)
_FP_COMPARE = MachineOpSemantics(
    defs=(0,), uses=(1, 2), n_phys=3,
    operand_classes=(_GPR, _FPR, _FPR),
)
_FP_LOAD = MachineOpSemantics(
    defs=(0,), uses=(1,), n_phys=2,
    operand_classes=(_FPR, _GPR, None),
)
_FP_STORE = MachineOpSemantics(
    uses=(0, 1), n_phys=2,
    operand_classes=(_FPR, _GPR, None),
)
_NO_REGISTERS = MachineOpSemantics()


OP_SEM: dict[MachineOp, MachineOpSemantics] = {
    MachineOp.ADD: _DEF_USE_USE,
    MachineOp.ADDI: MachineOpSemantics(
        defs=(0,), uses=(1,), immediate_positions=(2,), n_phys=2,
        operand_classes=(_GPR, _GPR, None),
    ),
    MachineOp.SUB: _DEF_USE_USE,
    MachineOp.MUL: _DEF_USE_USE,
    MachineOp.DIV: _DEF_USE_USE,
    MachineOp.SRAI: MachineOpSemantics(
        defs=(0,), uses=(1,), immediate_positions=(2,), n_phys=2,
        operand_classes=(_GPR, _GPR, None),
    ),
    MachineOp.XOR: _DEF_USE_USE,
    MachineOp.AND: _DEF_USE_USE,
    MachineOp.SLT: _DEF_USE_USE,
    MachineOp.SLTI: MachineOpSemantics(
        defs=(0,), uses=(1,), immediate_positions=(2,), n_phys=2,
        operand_classes=(_GPR, _GPR, None),
    ),
    MachineOp.REM: _DEF_USE_USE,
    MachineOp.LW: _DEF_USE,
    MachineOp.SW: _STORE,
    MachineOp.FLD: _FP_LOAD,
    MachineOp.FSD: _FP_STORE,
    # mv rd, rs -> addi rd, rs, 0
    MachineOp.MV: MachineOpSemantics(
        defs=(0,),
        uses=(1,),
        n_phys=2,
        is_pseudo=True,
        operand_classes=(_GPR, _GPR, None),
    ),
    # li rd, imm -> addi rd, x0, imm or lui/addi for a large immediate
    MachineOp.LI: MachineOpSemantics(
        defs=(0,),
        immediate_positions=(1,),
        n_phys=1,
        is_pseudo=True,
        operand_classes=(_GPR, None, None),
    ),
    MachineOp.LA: MachineOpSemantics(
        defs=(0,), n_phys=1, is_pseudo=True,
        operand_classes=(_GPR, None, None),
    ),
    # ScratchV's RV32IM max pseudo accepts a register rhs or immediate zero.
    MachineOp.MAX: MachineOpSemantics(
        defs=(0,),
        uses=(1, 2),
        immediate_positions=(2,),
        n_phys=3,
        is_pseudo=True,
        operand_classes=(_GPR, _GPR, _GPR),
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
        operand_classes=(_GPR, None, None),
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
        operand_classes=(_GPR, _GPR, None),
    ),
    MachineOp.JAL: MachineOpSemantics(
        defs=(0,),
        is_terminator=True,
        target_from_comment=True,
        target_required=True,
        n_phys=1,
        operand_classes=(_GPR, None, None),
    ),
    MachineOp.BEQ: MachineOpSemantics(
        uses=(0, 1),
        is_terminator=True,
        target_from_comment=True,
        target_required=True,
        n_phys=2,
        operand_classes=(_GPR, _GPR, None),
    ),
    MachineOp.BNE: MachineOpSemantics(
        uses=(0, 1),
        is_terminator=True,
        target_from_comment=True,
        target_required=True,
        n_phys=2,
        operand_classes=(_GPR, _GPR, None),
    ),
    MachineOp.BLT: MachineOpSemantics(
        uses=(0, 1),
        is_terminator=True,
        target_from_comment=True,
        target_required=True,
        n_phys=2,
        operand_classes=(_GPR, _GPR, None),
    ),
    MachineOp.BGE: MachineOpSemantics(
        uses=(0, 1),
        is_terminator=True,
        target_from_comment=True,
        target_required=True,
        n_phys=2,
        operand_classes=(_GPR, _GPR, None),
    ),
    # The CFG-aware spill rewriter and greedy allocator use these ABI clobbers
    # to preserve values that remain live across a call.
    MachineOp.CALL: MachineOpSemantics(
        target_from_comment=True,
        target_required=True,
        implicit_defs=frozenset({"ra"}),
        clobbers=frozenset({"ra", *ARG_REGS, *TEMP_REGS, *FP_TEMP_REGS, *FP_ARG_REGS}),
        is_call=True,
        n_phys=1,
        is_pseudo=True,
    ),
    MachineOp.SECTION: _NO_REGISTERS,
    MachineOp.GLOBL: _NO_REGISTERS,
    MachineOp.SIZE: _NO_REGISTERS,
    MachineOp.TYPE: _NO_REGISTERS,
    MachineOp.SQRT_S: _FP_DEF_USE,
    MachineOp.SQRT_D: _FP_DEF_USE,
    MachineOp.FMIN_D: _FP_DEF_USE_USE,
    MachineOp.FMAX_D: _FP_DEF_USE_USE,
    MachineOp.FABS_D: MachineOpSemantics(
        defs=(0,), uses=(1,), n_phys=2, is_pseudo=True,
        operand_classes=(_FPR, _FPR, None),
    ),
    MachineOp.FNEG_D: MachineOpSemantics(
        defs=(0,), uses=(1,), n_phys=2, is_pseudo=True,
        operand_classes=(_FPR, _FPR, None),
    ),
    MachineOp.FNEG_S: MachineOpSemantics(
        defs=(0,), uses=(1,), n_phys=2, is_pseudo=True,
        operand_classes=(_FPR, _FPR, None),
    ),
    MachineOp.FADD_D: _FP_DEF_USE_USE,
    MachineOp.FSUB_D: _FP_DEF_USE_USE,
    MachineOp.FMUL_D: _FP_DEF_USE_USE,
    MachineOp.FDIV_D: _FP_DEF_USE_USE,
    MachineOp.FLT_D: _FP_COMPARE,
    MachineOp.FLE_D: _FP_COMPARE,
    MachineOp.FEQ_D: _FP_COMPARE,
    MachineOp.FCVT_S_D: _FP_DEF_USE,
    MachineOp.FCVT_D_S: _FP_DEF_USE,
    MachineOp.LI_D: MachineOpSemantics(
        defs=(0,), immediate_positions=(1,), n_phys=1, is_pseudo=True,
        operand_classes=(_FPR, None, None),
    ),
    MachineOp.FADD_S: _FP_DEF_USE_USE,
    MachineOp.FSUB_S: _FP_DEF_USE_USE,
    MachineOp.FMUL_S: _FP_DEF_USE_USE,
    MachineOp.FDIV_S: _FP_DEF_USE_USE,
    MachineOp.FMAX_S: _FP_DEF_USE_USE,
    MachineOp.FMIN_S: _FP_DEF_USE_USE,
    MachineOp.FABS_S: MachineOpSemantics(
        defs=(0,), uses=(1,), n_phys=2, is_pseudo=True,
        operand_classes=(_FPR, _FPR, None),
    ),
    MachineOp.FLE_S: _FP_COMPARE,
    MachineOp.FLT_S: _FP_COMPARE,
    MachineOp.FEQ_S: _FP_COMPARE,
    MachineOp.FLW: _FP_LOAD,
    MachineOp.FSW: _FP_STORE,
    MachineOp.FMV_S: MachineOpSemantics(
        defs=(0,), uses=(1,), n_phys=2, is_pseudo=True,
        operand_classes=(_FPR, _FPR, None),
    ),
    MachineOp.FMV_D: MachineOpSemantics(
        defs=(0,), uses=(1,), n_phys=2, is_pseudo=True,
        operand_classes=(_FPR, _FPR, None),
    ),
    MachineOp.FMV_S_X: MachineOpSemantics(
        defs=(0,), uses=(1,), n_phys=2,
        operand_classes=(_FPR, _GPR, None),
    ),
}


_MISSING_SEMANTICS = set(MachineOp) - set(OP_SEM)
if _MISSING_SEMANTICS:
    missing = ", ".join(sorted(op.value for op in _MISSING_SEMANTICS))
    raise RuntimeError(f"missing machine semantics for: {missing}")


def get_machine_semantics(op: MachineOp) -> MachineOpSemantics:
    """Return the explicit semantics for *op*."""

    return OP_SEM[op]


def validate_register_classes(instructions: list["MachineInstr"]) -> None:
    """Validate positional constraints and whole-stream vreg consistency."""
    vreg_classes: dict[str, RegisterClass] = {}
    for instr in instructions:
        semantics = get_machine_semantics(instr.op)
        operands = (instr.dst, instr.src1, instr.src2)
        for position, expected in enumerate(semantics.operand_classes):
            operand = operands[position]
            if operand is None or operand.kind not in {"reg", "vreg"}:
                continue
            actual = operand.reg_class
            if operand.kind == "reg" and physical_register_class(str(operand.value)) is not actual:
                raise ValueError(f"physical register {operand.value} has wrong register class")
            if expected is not None and actual is not expected:
                raise ValueError(
                    f"{instr.op.value} operand {position} requires "
                    f"{expected.value}, got {actual.value if actual else 'none'}"
                )
            if operand.kind == "vreg" and actual is not None:
                name = str(operand.value)
                previous = vreg_classes.setdefault(name, actual)
                if previous is not actual:
                    raise ValueError(
                        f"virtual register {name} requires both "
                        f"{previous.value} and {actual.value}"
                    )


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
        target = instr.target or comment
        if semantics.target_required and not target:
            raise ValueError(f"{instr.op.value} requires a target label")
        if target:
            operands.append(target)
            comment = ""
    return operands, comment
