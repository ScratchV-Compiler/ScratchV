"""Use/def providers for ScratchV IR and machine instructions.

The generic liveness solver operates on :class:`ValueId` strings.  These
providers translate concrete operands into those ids.  Labels, immediates, and
jump targets are deliberately excluded from variable use/def sets.
"""

from __future__ import annotations

from typing import Any, AbstractSet

from scratchv.analysis.cfg import BlockId, ValueId
from scratchv.backend.machine_types import MachineInstr, MachineOp


def ir_value_id(value: Any) -> ValueId | None:
    """Return a stable value id for an IR ``Value``.

    Constants are not variables: they are embedded in the instruction stream
    and must not enter liveness sets.
    """

    if value is None:
        return None
    if getattr(value, "is_constant", False):
        return None
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else None


class IRUseDefProvider:
    """Use/def semantics for ``scratchv.ir.types.Instruction``."""

    def uses(self, instr: Any) -> AbstractSet[ValueId]:
        result: set[ValueId] = set()
        for operand in getattr(instr, "operands", []) or []:
            value_id = ir_value_id(operand)
            if value_id is not None:
                result.add(value_id)
        return frozenset(result)

    def defs(self, instr: Any) -> AbstractSet[ValueId]:
        dest = getattr(instr, "dest", None)
        value_id = ir_value_id(dest)
        return frozenset({value_id}) if value_id is not None else frozenset()

    def edge_uses(
        self,
        pred: BlockId,
        succ: BlockId,
    ) -> AbstractSet[ValueId]:
        # ScratchV IR currently has no SSA Phi nodes.  If Phi support is
        # added, edge operands will be reported here.
        return frozenset()

    def phi_defs(self, block: BlockId) -> AbstractSet[ValueId]:
        # ScratchV IR currently has no SSA Phi nodes.
        return frozenset()


class MachineUseDefProvider:
    """Use/def semantics for ``MachineInstr``.

    Only virtual registers participate in value liveness.  Physical registers,
    immediates, labels, and jump targets are not virtual values.  Caller-saved
    clobber information is exposed separately by :meth:`clobbers`.
    """

    def uses(self, instr: MachineInstr) -> AbstractSet[ValueId]:
        result: set[ValueId] = set()
        for operand in (instr.src1, instr.src2):
            if operand is not None and operand.kind == "vreg":
                value = operand.value
                if isinstance(value, str):
                    result.add(value)
        return frozenset(result)

    def defs(self, instr: MachineInstr) -> AbstractSet[ValueId]:
        if instr.dst is not None and instr.dst.kind == "vreg":
            value = instr.dst.value
            if isinstance(value, str):
                return frozenset({value})
        return frozenset()

    def edge_uses(
        self,
        pred: BlockId,
        succ: BlockId,
    ) -> AbstractSet[ValueId]:
        # Machine-level Phi is represented as copies on predecessor edges when
        # SSA is explicitly used; ScratchV currently uses non-SSA vregs.
        return frozenset()

    def phi_defs(self, block: BlockId) -> AbstractSet[ValueId]:
        return frozenset()

    def clobbers(self, instr: MachineInstr) -> AbstractSet[ValueId]:
        """Return physical register names clobbered by ``instr``.

        This is kept out of :meth:`defs` so CALL clobbers cannot be mixed into
        virtual-register liveness.
        """

        if instr.op is not MachineOp.CALL:
            return frozenset()

        # RISC-V caller-saved registers used by ScratchV's backend.
        return frozenset(
            {
                "ra", "t0", "t1", "t2", "t3", "t4", "t5", "t6",
                "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
            }
        )


__all__ = [
    "IRUseDefProvider",
    "MachineUseDefProvider",
    "ir_value_id",
]