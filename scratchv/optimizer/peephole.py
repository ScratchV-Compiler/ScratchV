"""Peephole optimization: eliminates redundant instruction patterns.

Scans basic blocks for common redundant patterns:
  - addi rd, rs, 0  →  (delete, no-op)
  - mul rd, rs, 1   →  mv rd, rs
  - mul rd, rs, 0   →  li rd, 0
  - j L immediately followed by L:  →  delete jump
"""

from __future__ import annotations

from scratchv.ir.types import OpCode, Instruction, BasicBlock, Program


class IRPeepholeOptimizer:
    """Eliminate redundant instruction patterns in IR."""

    def __init__(self, program: Program):
        self.program = program
        self._stats = {"eliminated": 0}

    def run(self) -> int:
        """Run peephole optimization.

        Returns number of eliminated instructions.
        """
        for func in self.program.functions:
            for block in func.blocks:
                self._optimize_block(block)
        return self._stats["eliminated"]

    def _optimize_block(self, block: BasicBlock) -> None:
        instrs = block.instructions
        i = 0
        while i < len(instrs):
            instr = instrs[i]

            # Pattern 1: addi rd, rs, 0 → delete (no-op)
            if self._is_addi_zero(instr):
                instrs.pop(i)
                self._stats["eliminated"] += 1
                continue

            # Pattern 2: mul rd, rs, 1 → mv rd, rs (use add rd, rs, x0)
            if self._is_mul_one(instr):
                dest = instr.dest
                src = instr.operands[0]
                instrs[i] = Instruction(
                    opcode=OpCode.ADD,
                    dest=dest,
                    operands=[src, self._make_zero_operand(instr)],
                )
                self._stats["eliminated"] += 1  # counts as optimization
                i += 1
                continue

            # Pattern 3: mul rd, rs, 0 → li rd, 0 (load_const 0)
            if self._is_mul_zero(instr):
                dest = instr.dest
                instrs[i] = Instruction(
                    opcode=OpCode.LOAD_CONST,
                    dest=dest,
                    attrs={"value": 0},
                )
                self._stats["eliminated"] += 1
                i += 1
                continue

            # Pattern 4: j L followed immediately by L:
            if self._is_jump_to_next(instrs, i):
                instrs.pop(i)
                self._stats["eliminated"] += 1
                continue

            i += 1

    def _is_addi_zero(self, instr: Instruction) -> bool:
        """Check for: add rd, rs, 0  (or  add rd, rs, const where const=0)."""
        if instr.opcode != OpCode.ADD:
            return False
        if len(instr.operands) < 2:
            return False
        rhs = instr.operands[1]
        return rhs.is_constant and rhs.const_value == 0

    def _is_mul_one(self, instr: Instruction) -> bool:
        if instr.opcode != OpCode.MUL:
            return False
        if len(instr.operands) < 2:
            return False
        rhs = instr.operands[1]
        return rhs.is_constant and rhs.const_value == 1

    def _is_mul_zero(self, instr: Instruction) -> bool:
        if instr.opcode != OpCode.MUL:
            return False
        if len(instr.operands) < 2:
            return False
        rhs = instr.operands[1]
        return rhs.is_constant and rhs.const_value == 0

    def _is_jump_to_next(self, instrs: list[Instruction], i: int) -> bool:
        """Check for: br L  followed by label L."""
        if i + 1 >= len(instrs):
            return False
        instr = instrs[i]
        if instr.opcode != OpCode.BR:
            return False
        next_instr = instrs[i + 1]
        return (next_instr.opcode == OpCode.LABEL
                and next_instr.target == instr.target)

    def _make_zero_operand(self, instr: Instruction):
        """Create a zero constant value."""
        from scratchv.ir.types import Value, DataType
        return Value(name="_zero", dtype=DataType.INT32,
                     is_constant=True, const_value=0)
