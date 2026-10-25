"""Target-semantic constant-folding tests for extended integer operations."""

import pytest

from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, OpCode
from scratchv.optimizer.constant_folding import ConstantFolder


def _fold_binary(op: str, lhs: int, rhs: int):
    builder = IRBuilder()
    builder.new_function("test")
    builder.new_block("entry")
    left = builder.load_const(lhs, DataType.INT32)
    right = builder.load_const(rhs, DataType.INT32)
    result = getattr(builder, op)(left, right)
    builder.ret(result)

    changes = ConstantFolder(rv32_integer_semantics=True).optimize(builder.program)
    instruction = builder.program.functions[0].blocks[0].instructions[2]
    return changes, instruction


def _fold_unary(op: str, value: int):
    builder = IRBuilder()
    builder.new_function("test")
    builder.new_block("entry")
    source = builder.load_const(value, DataType.INT32)
    result = getattr(builder, op)(source)
    builder.ret(result)

    changes = ConstantFolder(rv32_integer_semantics=True).optimize(builder.program)
    instruction = builder.program.functions[0].blocks[0].instructions[1]
    return changes, instruction


class TestIntegerConstantFolding:
    @pytest.mark.parametrize(
        ("lhs", "rhs", "expected"),
        [
            (-7, 3, -2),
            (7, -3, -2),
            (-7, -3, 2),
            (-(1 << 31), -1, -(1 << 31)),
        ],
    )
    def test_i32_division_folds_with_rv32_semantics(self, lhs, rhs, expected):
        changes, instruction = _fold_binary("div", lhs, rhs)

        assert changes == 1
        assert instruction.opcode is OpCode.LOAD_CONST
        assert instruction.attrs["value"] == expected

    @pytest.mark.parametrize(
        ("lhs", "rhs", "expected"),
        [
            (-7, 3, -1),
            (7, -3, 1),
            (-7, -3, -1),
            (-(1 << 31), -1, 0),
        ],
    )
    def test_i32_remainder_folds_with_rv32_semantics(self, lhs, rhs, expected):
        changes, instruction = _fold_binary("rem", lhs, rhs)

        assert changes == 1
        assert instruction.opcode is OpCode.LOAD_CONST
        assert instruction.attrs["value"] == expected

    @pytest.mark.parametrize("op", ["div", "rem"])
    def test_i32_division_by_zero_is_not_folded(self, op):
        changes, instruction = _fold_binary(op, 7, 0)

        assert changes == 0
        assert instruction.opcode is getattr(OpCode, op.upper())

    @pytest.mark.parametrize(
        ("op", "lhs", "rhs", "expected"),
        [
            ("min", -(1 << 31), 7, -(1 << 31)),
            ("min", 9, -4, -4),
            ("max", (1 << 31) - 1, -1, (1 << 31) - 1),
            ("max", -9, -4, -4),
        ],
    )
    def test_i32_min_max_are_constant_folded(self, op, lhs, rhs, expected):
        changes, instruction = _fold_binary(op, lhs, rhs)

        assert changes == 1
        assert instruction.opcode is OpCode.LOAD_CONST
        assert instruction.attrs["value"] == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, 0), (-19, 19), (-(1 << 31), -(1 << 31))],
    )
    def test_i32_abs_is_constant_folded_with_rv32_wrap(self, value, expected):
        changes, instruction = _fold_unary("abs", value)

        assert changes == 1
        assert instruction.opcode is OpCode.LOAD_CONST
        assert instruction.attrs["value"] == expected
