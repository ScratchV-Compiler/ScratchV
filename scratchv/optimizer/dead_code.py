"""Dead-code elimination for ScratchV IR.

The pass traces value definitions backwards from observable instructions, then
removes pure instructions that do not contribute to those roots.
"""

from __future__ import annotations

from scratchv.ir.types import BasicBlock, Function, Instruction, OpCode, Program
from scratchv.pass_interface import OptimizationPass


class DeadCodeEliminator(OptimizationPass):
    """Remove unused instructions from an IR Program."""

    name = "dead-code-elim"

    _EFFECT_ROOTS = frozenset(
        {
            OpCode.STORE,
            OpCode.RETURN,
            OpCode.BR,
            OpCode.BR_IF,
            OpCode.FOR,
            OpCode.ENDFOR,
            OpCode.ALLOCA,
        }
    )

    def optimize(self, program: Program) -> int:
        """Eliminate dead instructions and return this invocation's count."""
        return sum(self._eliminate_function(func) for func in program.functions)

    def _eliminate_function(self, func: Function) -> int:
        if len(func.blocks) != 1:
            return 0
        return self._eliminate_block(func.blocks[0])

    def _eliminate_block(self, block: BasicBlock) -> int:
        if block.phi_nodes:
            return 0
        definitions = self._build_definitions(block.instructions)
        if definitions is None:
            return 0
        live_ids: set[int] = set()

        for instr in block.instructions:
            if self._is_effect_root(instr):
                self._mark_used(instr, definitions, live_ids)

        original_count = len(block.instructions)
        block.instructions = [
            instr for instr in block.instructions if id(instr) in live_ids
        ]
        return original_count - len(block.instructions)

    @staticmethod
    def _build_definitions(
        instructions: list[Instruction],
    ) -> dict[str, Instruction] | None:
        definitions: dict[str, Instruction] = {}
        for instr in instructions:
            if instr.dest is None:
                continue
            if instr.dest.name in definitions:
                return None
            definitions[instr.dest.name] = instr
        return definitions

    def _mark_used(
        self,
        instr: Instruction,
        definitions: dict[str, Instruction],
        live_ids: set[int],
    ) -> None:
        instr_id = id(instr)
        if instr_id in live_ids:
            return
        live_ids.add(instr_id)

        for operand in instr.operands:
            producer = definitions.get(operand.name)
            if producer is not None:
                self._mark_used(producer, definitions, live_ids)

    @classmethod
    def _is_effect_root(cls, instr: Instruction) -> bool:
        """Return whether an instruction must remain observable."""
        return instr.dest is None or instr.opcode in cls._EFFECT_ROOTS
