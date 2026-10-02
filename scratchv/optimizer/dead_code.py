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
        # SSA values can be consumed in another block, including PHI edges.
        # Collect uses across the whole function before filtering any block;
        # block-local liveness can erase a still-needed dominating definition.
        used = {value.name for value in func.returns}
        for block in func.blocks:
            for instr in [*block.phi_nodes, *block.instructions]:
                used.update(value.name for value in instr.operands)
        return sum(self._eliminate_block(block, used) for block in func.blocks)

    def _eliminate_block(self, block: BasicBlock, used: set[str]) -> int:
        changes = 0

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
