"""Constant folding optimization pass.

Evaluates arithmetic operations with constant operands at compile time,
replacing them with load_const instructions.
"""

from __future__ import annotations

from scratchv.ir.types import (
    DataType, OpCode, Instruction, BasicBlock, Function, Program,
)
from scratchv.optimizer.hoist_safety import HoistSafety
from scratchv.pass_interface import OptimizationPass
from scratchv.verification.ir_numpy_ops import OpError, check_instruction, compute


class ConstantFolder(OptimizationPass):
    """Fold constant expressions in an IR Program.

    ``preserve_float_semantics=True`` is required for real typed hard-float
    targets: NumPy cannot model target rounding modes or fflags. By default
    folds use the checked IR kernels. ``rv32_integer_semantics=True`` selects
    hardware i32 wraparound for DIV/ABS overflow. CompilerDriver sets these
    policies for the target, including explicit pass selections.
    """

    name = "constant-folding"

    def __init__(self, *, preserve_float_semantics: bool = False,
                 rv32_integer_semantics: bool = False):
        self.preserve_float_semantics = preserve_float_semantics
        self.rv32_integer_semantics = rv32_integer_semantics

    def optimize(self, program: Program) -> int:
        """Run constant folding on all functions. Returns number of folds."""
        return sum(self._fold_function(program, func) for func in program.functions)

    def _fold_function(self, program: Program, func: Function) -> int:
        safety = HoistSafety(program, func)
        return sum(self._fold_block(block, safety) for block in func.blocks)

    def _fold_block(self, block: BasicBlock, safety: HoistSafety) -> int:
        changes = 0
        new_instrs: list[Instruction] = []
        for instr in block.instructions:
            folded = self._try_fold(instr, safety)
            if folded is not None:
                new_instrs.append(folded)
                safety.definitions[folded.dest.name] = folded
                safety.cache.clear()
                changes += 1
            else:
                new_instrs.append(instr)
        block.instructions = new_instrs
        return changes

    def _try_fold(self, instr: Instruction, safety: HoistSafety) -> Instruction | None:
        """Try to fold an instruction. Returns a replacement or None."""
        if instr.opcode not in (
                OpCode.ADD, OpCode.SUB,
                OpCode.MUL, OpCode.DIV, OpCode.REM,
                OpCode.MIN, OpCode.MAX, OpCode.ABS):
            return None
        expected_operands = 1 if instr.opcode is OpCode.ABS else 2
        if (instr.dest is None or instr.dest.shape
                or len(instr.operands) != expected_operands):
            return None
        if any(value.dtype != instr.dest.dtype or value.shape for value in instr.operands):
            return None
        if self.preserve_float_semantics and instr.dest.dtype in (
            DataType.FLOAT32, DataType.FLOAT64,
        ):
            return None
        operands = [safety.facts(value).constant for value in instr.operands]
        if any(value is None for value in operands):
            return None
        try:
            if instr.dest.dtype is DataType.INT32 and (
                self.rv32_integer_semantics
                or instr.opcode in (OpCode.REM, OpCode.MIN, OpCode.MAX)
            ):
                if instr.attrs:
                    return None
                values = [value.item() for value in operands]
                if instr.opcode is OpCode.ABS:
                    result = self._wrap_i32(abs(values[0]))
                else:
                    result = self._compute_i32(instr.opcode, *values)
                if result is None:
                    return None
            else:
                check_instruction(instr)
                # Facts resolve definitions, not stale const_value hints or params.
                result = compute(instr, operands).item()
        except (OpError, ValueError, TypeError, OverflowError, FloatingPointError):
            # Keep failing operations at their original execution position.
            return None

        dest = instr.dest
        if dest is not None:
            dest.is_constant = True
            dest.const_value = result
            dest.bit_pattern = None

        return Instruction(
            opcode=OpCode.LOAD_CONST,
            dest=dest,
            attrs={"value": result},
        )

    @staticmethod
    def _compute_i32(opcode: OpCode, a: int, b: int) -> int | None:
        """Evaluate an i32 operation with RV32 two's-complement semantics."""
        if opcode in (OpCode.DIV, OpCode.REM):
            if b == 0:
                return None
            quotient = abs(a) // abs(b)
            if (a < 0) != (b < 0):
                quotient = -quotient
            result = quotient if opcode is OpCode.DIV else a - quotient * b
        else:
            mapping = {
                OpCode.ADD: a + b,
                OpCode.SUB: a - b,
                OpCode.MUL: a * b,
                OpCode.MIN: min(a, b),
                OpCode.MAX: max(a, b),
            }
            result = mapping.get(opcode)
            if result is None:
                return None

        # RV32 integer arithmetic wraps modulo 2**32.
        return ConstantFolder._wrap_i32(result)

    @staticmethod
    def _wrap_i32(value: int) -> int:
        value &= 0xFFFF_FFFF
        return value if value < 0x8000_0000 else value - 0x1_0000_0000
