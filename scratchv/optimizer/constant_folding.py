"""Constant folding optimization pass.

Evaluates arithmetic operations with constant operands at compile time,
replacing them with load_const instructions.
"""

from __future__ import annotations

from scratchv.ir.types import (
    OpCode, Instruction, BasicBlock, Function, Program,
)
from scratchv.optimizer.hoist_safety import HoistSafety
from scratchv.pass_interface import OptimizationPass
from scratchv.verification.ir_numpy_ops import OpError, check_instruction, compute


class ConstantFolder(OptimizationPass):
    """Fold constant expressions in an IR Program."""

    name = "constant-folding"

    def optimize(self, program: Program) -> int:
        """Run constant folding on all functions. Returns number of folds."""
        return sum(self._fold_function(program, func) for func in program.functions)

    def _fold_function(self, program: Program, func: Function) -> int:
        safety = HoistSafety(program, func)
        return sum(self._fold_block(block, safety) for block in func.blocks)

    def _fold_block(self, block: BasicBlock, safety: HoistSafety) -> int:
        changes = 0
        new_instrs: list[Instruction] = []
        for instr in block.instructions:
            folded = self._try_fold(instr, safety)
            if folded is not None:
                new_instrs.append(folded)
                changes += 1
            else:
                new_instrs.append(instr)
        block.instructions = new_instrs
        return changes

    def _try_fold(self, instr: Instruction, safety: HoistSafety) -> Instruction | None:
        """Try to fold an instruction. Returns a replacement or None."""
        if instr.opcode not in (
                OpCode.ADD, OpCode.SUB,
                OpCode.MUL, OpCode.DIV):
            return None
        if instr.dest is None or instr.dest.shape or len(instr.operands) != 2:
            return None
        if any(value.dtype != instr.dest.dtype or value.shape for value in instr.operands):
            return None
        operands = [safety.facts(value).constant for value in instr.operands]
        if any(value is None for value in operands):
            return None
        try:
            check_instruction(instr)
            # Reuse runtime rounding, integer wrapping and checked division.
            # Facts resolve definitions, not stale const_value hints or params.
            result = compute(instr, operands).item()
        except (OpError, ValueError, TypeError, OverflowError, FloatingPointError):
            # Keep failing operations at their original execution position.
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
