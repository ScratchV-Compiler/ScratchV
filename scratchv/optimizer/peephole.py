"""Eliminate proven identities while preserving SSA and tensor semantics."""

from __future__ import annotations

from scratchv.ir.types import DataType, OpCode, Program
from scratchv.optimizer.hoist_safety import HoistSafety
from scratchv.pass_interface import OptimizationPass


class IRPeepholeOptimizer(OptimizationPass):
    """Redirect uses of safe identities to their existing typed source.

    Floating addition by zero can change signed zero; multiplication by zero
    needs a tensor-shaped result and must retain numeric errors. Keep those
    operations. Floating multiplication by one is removable only when its
    source is already proven finite, since kernels reject nonfinite results.
    """

    name = "ir-peephole"

    def optimize(self, program: Program) -> int:
        changes = 0
        for function in program.functions:
            safety = HoistSafety(program, function)
            # FOR counters are updated implicitly by ENDFOR. An identity can
            # capture their last body value for use after the loop exits.
            counters = {instr.dest.name for block in function.blocks
                        for instr in block.instructions
                        if instr.opcode == OpCode.FOR and instr.dest is not None}
            aliases = {}

            def resolve(value):
                while value.name in aliases:
                    value = aliases[value.name]
                return value

            for block in function.blocks:
                retained = []
                for instr in block.instructions:
                    instr.operands = [resolve(value) for value in instr.operands]
                    if self._is_safe_identity(instr, safety, counters):
                        aliases[instr.dest.name] = instr.operands[0]
                        changes += 1
                    else:
                        retained.append(instr)
                block.instructions = retained
            # A use may be in another block, including one listed before its
            # dominating definition. Update all uses after collecting aliases.
            for block in function.blocks:
                for instr in [*block.instructions, *block.phi_nodes]:
                    instr.operands = [resolve(value) for value in instr.operands]
            function.returns = [resolve(value) for value in function.returns]
        return changes

    @staticmethod
    def _is_safe_identity(instr, safety, counters):
        if (instr.opcode not in (OpCode.ADD, OpCode.MUL) or instr.dest is None
                or len(instr.operands) != 2 or instr.attrs):
            return False
        source, constant = instr.operands
        if (source.name == instr.dest.name or source.name in counters
                or source.dtype != instr.dest.dtype or constant.dtype != source.dtype
                or (instr.dest.shape and instr.dest.shape != source.shape)):
            return False
        literal = safety.facts(constant).constant
        expected = 0 if instr.opcode == OpCode.ADD else 1
        if literal is None or literal.shape != () or literal.item() != expected:
            return False
        source_facts = safety.facts(source)
        if not source_facts.tensor:
            return False
        if source.dtype in (DataType.INT32, DataType.INT64):
            return True
        return instr.opcode == OpCode.MUL and source_facts.finite
