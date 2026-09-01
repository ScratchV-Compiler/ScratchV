"""Constant folding optimization pass.

Evaluates arithmetic operations with constant operands at compile time,
replacing them with load_const instructions.
"""

from __future__ import annotations

from scratchv.ir.types import (
    OpCode, Instruction, BasicBlock, Function, Program,
)
from scratchv.pass_interface import OptimizationPass


class ConstantFolder(OptimizationPass):
    """Fold constant expressions in an IR Program."""

    name = "constant-folding"

    def optimize(self, program: Program) -> int:
        """Run constant folding on all functions. Returns number of folds."""
        return sum(self._fold_function(func) for func in program.functions)

    def _fold_function(self, func: Function) -> int:
        return sum(self._fold_block(block) for block in func.blocks)

    def _fold_block(self, block: BasicBlock) -> int:
        changes = 0
        new_instrs: list[Instruction] = []
        for instr in block.instructions:
            folded = self._try_fold(instr)
            if folded is not None:
                new_instrs.append(folded)
                changes += 1
            else:
                new_instrs.append(instr)
        block.instructions = new_instrs
        return changes

    def _try_fold(self, instr: Instruction) -> Instruction | None:
        """Try to fold an instruction. Returns a replacement or None."""
        if instr.opcode not in (
                OpCode.ADD, OpCode.SUB,
                OpCode.MUL, OpCode.DIV):
            return None
        if len(instr.operands) != 2:
            return None

        lhs, rhs = instr.operands
        if not lhs.is_constant or not rhs.is_constant:
            return None
        if lhs.const_value is None or rhs.const_value is None:
            return None

        a, b = float(lhs.const_value), float(rhs.const_value)
        result = self._compute(instr.opcode, a, b)
        if result is None:
            return None

        dest = instr.dest
        if dest is not None:
            dest.is_constant = True
            dest.const_value = result

        return Instruction(
            opcode=OpCode.LOAD_CONST,
            dest=dest,
            attrs={"value": result},
        )

    @staticmethod
    def _compute(opcode: OpCode, a: float, b: float) -> float | None:
        mapping = {
            OpCode.ADD: a + b,
            OpCode.SUB: a - b,
            OpCode.MUL: a * b,
            OpCode.DIV: a / b if b != 0 else None,
        }
        return mapping.get(opcode)
