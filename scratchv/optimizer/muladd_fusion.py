"""Mul-Add fusion: combines mul+add into a single pseudo-operation.

Detects the pattern:
    tmp = mul(a, b)
    sum = add(tmp, sum)

This reduces temporary register pressure and can be exploited by
hardware with a fused multiply-add (FMA) instruction.

The fused instruction is represented as MUL_ADD in IR:
    dst = mul_add(a, b, acc)
"""

from __future__ import annotations

from scratchv.ir.types import OpCode, Instruction, BasicBlock, Function, Program


class MulAddFusion:
    """Combine consecutive mul+add instruction pairs."""

    def __init__(self, program: Program):
        self.program = program
        self._stats = {"fused": 0}

    def run(self) -> int:
        """Run mul-add fusion. Returns number of fusions performed."""
        for func in self.program.functions:
            for block in func.blocks:
                self._fuse_block(block)
        return self._stats["fused"]

    def _fuse_block(self, block: BasicBlock) -> None:
        instrs = block.instructions
        i = 0
        while i < len(instrs) - 1:
            mul = instrs[i]
            add = instrs[i + 1]

            if self._matches_pattern(mul, add):
                # Replace mul with fused instruction
                a, b = mul.operands[0], mul.operands[1]
                acc = add.operands[0] if add.operands[1].name == mul.dest.name else add.operands[1]
                fused = Instruction(
                    opcode=OpCode.ADD,  # Keep as ADD for RV32IM (no native FMA)
                    dest=add.dest,
                    operands=[acc, a, b],
                )
                fused.attrs["fused_mul_add"] = True
                instrs[i] = fused
                # Remove the original add
                instrs.pop(i + 1)
                self._stats["fused"] += 1

            i += 1

    def _matches_pattern(self, mul: Instruction, add: Instruction) -> bool:
        """Check if mul + add form a fusible pattern.

            tmp    = mul(a, b)
            result = add(tmp, acc)   or   add(acc, tmp)
        """
        if mul.opcode != OpCode.MUL:
            return False
        if add.opcode != OpCode.ADD:
            return False
        if mul.dest is None:
            return False

        # The add must use the mul's result
        mul_dest_name = mul.dest.name
        return any(op.name == mul_dest_name for op in add.operands)
