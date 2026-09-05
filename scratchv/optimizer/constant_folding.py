"""Constant folding optimization pass.

Evaluates arithmetic operations with constant operands at compile time,
replacing them with load_const instructions.
"""

from __future__ import annotations

import math
import struct

from scratchv.ir.types import (
    BasicBlock,
    DataType,
    Function,
    Instruction,
    OpCode,
    Program,
)
from scratchv.pass_interface import OptimizationPass


class ConstantFolder(OptimizationPass):
    """Fold constant expressions in an IR Program."""

    name = "constant-folding"

    def optimize(self, program: Program) -> int:
        """Fold constants in one forward scan and return this run's count."""
        return sum(self._fold_function(func) for func in program.functions)

    def _fold_function(self, func: Function) -> int:
        return sum(self._fold_block(block) for block in func.blocks)

    def _fold_block(self, block: BasicBlock) -> int:
        changes = 0
        new_instrs: list[Instruction] = []
        for instr in block.instructions:
            folded = self._try_fold(instr)
            if folded is not None:
                new_instrs.append(folded)
                changes += 1
            else:
                new_instrs.append(instr)
        block.instructions = new_instrs
        return changes

    def _try_fold(self, instr: Instruction) -> Instruction | None:
        """Try to fold an instruction. Returns a replacement or None."""
        if instr.opcode not in (
            OpCode.ADD,
            OpCode.SUB,
            OpCode.MUL,
            OpCode.DIV,
        ):
            return None
        if len(instr.operands) != 2:
            return None

        lhs, rhs = instr.operands
        if not lhs.is_constant or not rhs.is_constant:
            return None
        if lhs.const_value is None or rhs.const_value is None:
            return None

        dest = instr.dest
        if instr.opcode in (OpCode.ADD, OpCode.SUB):
            if dest is None or instr.target is not None or instr.attrs:
                return None
            if not (
                lhs.dtype is rhs.dtype
                and rhs.dtype is dest.dtype
                and dest.dtype in (DataType.FLOAT32, DataType.INT32)
            ):
                return None
            if lhs.shape != () or rhs.shape != () or dest.shape != ():
                return None

        if instr.opcode in (OpCode.ADD, OpCode.SUB):
            result = self._compute(
                instr.opcode,
                lhs.const_value,
                rhs.const_value,
                lhs.dtype,
            )
        else:
            try:
                legacy_lhs = float(lhs.const_value)
                legacy_rhs = float(rhs.const_value)
            except (TypeError, ValueError, OverflowError):
                return None
            result = self._compute(
                instr.opcode,
                legacy_lhs,
                legacy_rhs,
                lhs.dtype,
            )
        if result is None:
            return None

        replacement = Instruction(
            opcode=OpCode.LOAD_CONST,
            dest=dest,
            attrs={"value": result},
        )
        if dest is not None:
            dest.is_constant = True
            dest.const_value = result
        return replacement

    @staticmethod
    def _quantize_float32(value: float | int) -> float | None:
        """Return a finite IEEE-754 binary32 value as a Python float."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            result = struct.unpack("<f", struct.pack("<f", value))[0]
        except (OverflowError, struct.error, TypeError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _normalize_int32(value: float | int) -> int | None:
        """Normalize a valid INT32 scalar payload to a Python int."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if isinstance(value, float) and (
            not math.isfinite(value) or not value.is_integer()
        ):
            return None
        normalized = int(value)
        if not -(2**31) <= normalized <= 2**31 - 1:
            return None
        return normalized

    @staticmethod
    def _wrap_int32(value: int) -> int:
        """Interpret the low 32 bits of ``value`` as signed two's complement."""
        return (value + 2**31) % 2**32 - 2**31

    @staticmethod
    def _compute(
        opcode: OpCode,
        a: float | int,
        b: float | int,
        dtype: DataType,
    ) -> float | int | None:
        """Compute a typed scalar result, or return None when unsafe."""
        if dtype is DataType.FLOAT32 and opcode in (OpCode.ADD, OpCode.SUB):
            lhs = ConstantFolder._quantize_float32(a)
            rhs = ConstantFolder._quantize_float32(b)
            if lhs is None or rhs is None:
                return None
            raw_result = lhs + rhs if opcode is OpCode.ADD else lhs - rhs
            return ConstantFolder._quantize_float32(raw_result)

        if dtype is DataType.INT32 and opcode in (OpCode.ADD, OpCode.SUB):
            lhs = ConstantFolder._normalize_int32(a)
            rhs = ConstantFolder._normalize_int32(b)
            if lhs is None or rhs is None:
                return None
            raw_result = lhs + rhs if opcode is OpCode.ADD else lhs - rhs
            return ConstantFolder._wrap_int32(raw_result)

        # W3 preserves the reference implementation's simple MUL/DIV behavior;
        # W4 will bring these operations under the typed numeric contract.
        if opcode is OpCode.MUL:
            return a * b
        if opcode is OpCode.DIV:
            return a / b if b != 0 else None
        return None
