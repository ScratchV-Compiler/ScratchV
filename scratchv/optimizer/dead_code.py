"""Dead code elimination pass.

Removes instructions whose result is never used.

"""

from __future__ import annotations

from scratchv.ir.types import (
    OpCode, Instruction, BasicBlock, Function, Program,
)
from scratchv.pass_interface import OptimizationPass


class DeadCodeEliminator(OptimizationPass):
    """Remove unused instructions from an IR Program."""

    name = "dead-code-elim"

    def optimize(self, program: Program) -> int:
        """Run dead code elimination.

        Returns number of eliminated instructions.
        """
        return sum(self._eliminate_function(func) for func in program.functions)

    def _eliminate_function(self, func: Function) -> int:
        return sum(self._eliminate_block(block) for block in func.blocks)

    def _eliminate_block(self, block: BasicBlock) -> int:
        changes = 0
        # Collect all used value names
        used: set[str | None] = set()
        # Return values and branch targets are always live
        for instr in block.instructions:
            if instr.opcode in (
                    OpCode.RETURN, OpCode.BR,
                    OpCode.BR_IF, OpCode.STORE,
                    OpCode.ENDFOR, OpCode.FOR):
                used.add(instr.dest.name if instr.dest else None)
            for op in instr.operands:
                used.add(op.name)

        # Filter: keep instructions with side effects or whose dest is used
        new_instrs: list[Instruction] = []
        for instr in block.instructions:
            if self._is_side_effect(instr):
                new_instrs.append(instr)
            elif instr.dest is None or instr.dest.name in used:
                new_instrs.append(instr)
            else:
                changes += 1

        block.instructions = new_instrs
        return changes

    @staticmethod
    def _is_side_effect(instr: Instruction) -> bool:
        """Check if an instruction has side effects and must be kept."""
        return instr.opcode in (
            OpCode.STORE,
            OpCode.RETURN,
            OpCode.BR,
            OpCode.BR_IF,
            OpCode.FOR,
            OpCode.ENDFOR,
            OpCode.ALLOCA,
        )
