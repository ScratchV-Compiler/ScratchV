"""Loop Invariant Code Motion (LICM).

Moves instructions that produce the same result every iteration
from inside a loop to before the loop header.

This pass works on the IR FOR/ENDFOR structure. An instruction
is loop-invariant if all its operands are:
  1. Constants, or
  2. Defined outside the loop, or
  3. Already hoisted invariants.
"""

from __future__ import annotations

from __future__ import annotations

from scratchv.ir.types import OpCode, Instruction, BasicBlock, Function, Program


class LICM:
    """Hoist loop-invariant code out of loops."""

    def __init__(self, program: Program):
        self.program = program
        self._stats = {"hoisted": 0}

    def run(self) -> int:
        """Run LICM on all functions. Returns number of hoisted instructions."""
        for func in self.program.functions:
            self._process_function(func)
        return self._stats["hoisted"]

    def _process_function(self, func: Function) -> None:
        for block in func.blocks:
            self._process_block(block)

    def _process_block(self, block: BasicBlock) -> None:
        """Find FOR/ENDFOR pairs in a block and hoist invariants."""
        instrs = block.instructions
        i = 0
        while i < len(instrs):
            if instrs[i].opcode != OpCode.FOR:
                i += 1
                continue

            # Find matching ENDFOR
            loop_start = i
            loop_end = self._find_matching_endfor(instrs, loop_start)
            if loop_end is None:
                i += 1
                continue

            # Collect loop-variant names (loop variable induction var)
            iv_name = instrs[loop_start].dest.name if instrs[loop_start].dest else ""
            variant_names = {iv_name}

            # Find instructions defined within the loop (excluding FOR itself)
            loop_defs = set()
            for j in range(loop_start + 1, loop_end):
                instr = instrs[j]
                if instr.dest:
                    loop_defs.add(instr.dest.name)

            # Scan for invariant instructions
            hoisted = []
            j = loop_start + 1
            while j < loop_end:
                instr = instrs[j]
                if self._is_invariant(instr, variant_names, loop_defs):
                    hoisted.append((j, instr))
                    j += 1
                else:
                    # Add this instruction's dest to variant set
                    if instr.dest:
                        variant_names.add(instr.dest.name)
                    j += 1

            # Hoist: move invariant instructions before the FOR
            for idx, instr in reversed(hoisted):
                instrs.pop(idx)
                instrs.insert(loop_start, instr)
                loop_end += 1  # adjust for shift
                self._stats["hoisted"] += 1

            i = loop_end + 1

    def _find_matching_endfor(self, instrs: list[Instruction], start: int):
        """Find matching ENDFOR for a FOR at given index."""
        depth = 0
        for i in range(start, len(instrs)):
            if instrs[i].opcode == OpCode.FOR:
                depth += 1
            elif instrs[i].opcode == OpCode.ENDFOR:
                depth -= 1
                if depth == 0:
                    return i
        return None

    def _is_invariant(self, instr: Instruction, variant_names: set[str],
                      loop_defs: set[str]) -> bool:
        """Check if an instruction is loop-invariant."""
        # Control flow and store instructions are never invariant
        if instr.opcode in (OpCode.STORE, OpCode.BR, OpCode.BR_IF, OpCode.RETURN,
                            OpCode.FOR, OpCode.ENDFOR, OpCode.LABEL):
            return False
        # An instruction is invariant if all its operands are:
        # - constants, or
        # - defined outside the loop (not in loop_defs and not variant)
        if not instr.operands:
            return True
        for op in instr.operands:
            if op.is_constant:
                continue
            if op.name in variant_names:
                return False
            if op.name in loop_defs:
                return False
        return True
