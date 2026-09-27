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

from scratchv.ir.types import OpCode, Instruction, BasicBlock, Program
from scratchv.pass_interface import OptimizationPass


class MulAddFusion(OptimizationPass):
    """Combine consecutive mul+add instruction pairs."""

    name = "muladd-fusion"

    def optimize(self, program: Program) -> int:
        """Run mul-add fusion. Returns number of fusions performed."""
        return sum(
            self._fuse_block(block)
            for func in program.functions
            for block in func.blocks
        )

    def _fuse_block(self, block: BasicBlock) -> int:
        changes = 0
        instrs = block.instructions
        i = 0
        while i < len(instrs) - 1:
            mul = instrs[i]
            add = instrs[i + 1]

            if self._matches_pattern(mul, add):
                # Replace mul with fused instruction
                a, b = mul.operands[0], mul.operands[1]
                assert mul.dest is not None
                if add.operands[1].name == mul.dest.name:
                    acc = add.operands[0]
                else:
                    acc = add.operands[1]
                fused = Instruction(
                    opcode=OpCode.ADD,  # RV32IM has no native FMA
                    dest=add.dest,
                    operands=[acc, a, b],
                )
                fused.attrs["fused_mul_add"] = True
                instrs[i] = fused
                # Remove the original add
                instrs.pop(i + 1)
                changes += 1

            i += 1

        return changes

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
