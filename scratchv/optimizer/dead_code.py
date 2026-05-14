"""Dead code elimination pass.

Removes instructions whose result is never used (no other instruction references it
and it's not a function return value).
"""

from __future__ import annotations

from scratchv.ir.types import OpCode, Instruction, BasicBlock, Function, Program


class DeadCodeEliminator:
    """Remove unused instructions from an IR Program."""

    def __init__(self, program: Program):
        self.program = program
        self._stats = {"eliminated": 0}

    def run(self) -> int:
        """Run dead code elimination. Returns number of eliminated instructions."""
        for func in self.program.functions:
            self._eliminate_function(func)
        return self._stats["eliminated"]

    def _eliminate_function(self, func: Function) -> None:
        for block in func.blocks:
            self._eliminate_block(block)

    def _eliminate_block(self, block: BasicBlock) -> None:
        # Collect all used value names
        used: set[str | None] = set()
        # Return values and branch targets are always live
        for instr in block.instructions:
            if instr.opcode in (OpCode.RETURN, OpCode.BR, OpCode.BR_IF, OpCode.STORE,
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
                self._stats["eliminated"] += 1

        block.instructions = new_instrs

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
