"""Constant folding optimization pass.

Evaluates arithmetic operations with constant operands at compile time,
replacing them with load_const instructions.
"""

from __future__ import annotations

from scratchv.ir.types import OpCode, Instruction, BasicBlock, Function, Program


class ConstantFolder:
    """Fold constant expressions in an IR Program."""

    def __init__(self, program: Program):
        self.program = program
        self._stats = {"folded": 0}

    def run(self) -> int:
        """Run constant folding on all functions. Returns number of folds."""
        for func in self.program.functions:
            self._fold_function(func)
        return self._stats["folded"]

    def _fold_function(self, func: Function) -> None:
        for block in func.blocks:
            self._fold_block(block)

    def _fold_block(self, block: BasicBlock) -> None:
        new_instrs: list[Instruction] = []
        for instr in block.instructions:
            folded = self._try_fold(instr)
            if folded is not None:
                new_instrs.append(folded)
            else:
                new_instrs.append(instr)
        block.instructions = new_instrs

    def _try_fold(self, instr: Instruction) -> Instruction | None:
        """Try to fold an instruction. Returns a replacement or None."""
        if instr.opcode not in (OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV):
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

        self._stats["folded"] += 1
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
