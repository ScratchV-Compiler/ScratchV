"""Tests for optimizer passes."""

from __future__ import annotations

import math

import pytest

from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import (
    BasicBlock,
    DataType,
    Function,
    Instruction,
    OpCode,
    Program,
    Value,
)
from scratchv.optimizer.constant_folding import ConstantFolder
from scratchv.optimizer.dead_code import DeadCodeEliminator


def _typed_binary_program(
    opcode: OpCode,
    lhs_value: float | int,
    rhs_value: float | int,
    dtype: DataType,
    *,
    dest_dtype: DataType | None = None,
    shape: tuple[int, ...] = (),
    candidate_attrs: dict[str, object] | None = None,
    target: str | None = None,
) -> tuple[Program, BasicBlock, Value, Instruction]:
    """Build minimal, explicitly typed scalar IR for folding tests."""
    lhs = Value(
        "lhs",
        dtype=dtype,
        is_constant=True,
        const_value=lhs_value,
        shape=shape,
    )
    rhs = Value(
        "rhs",
        dtype=dtype,
        is_constant=True,
        const_value=rhs_value,
        shape=shape,
    )
    dest = Value("result", dtype=dest_dtype or dtype, shape=shape)
    candidate = Instruction(
        opcode,
        dest=dest,
        operands=[lhs, rhs],
        attrs=candidate_attrs or {},
        target=target,
    )
    user = Instruction(OpCode.RETURN, operands=[dest])
    block = BasicBlock("entry")
    block.instructions = [
        Instruction(OpCode.LOAD_CONST, dest=lhs, attrs={"value": lhs_value}),
        Instruction(OpCode.LOAD_CONST, dest=rhs, attrs={"value": rhs_value}),
        candidate,
        user,
    ]
    program = Program()
    program.add_function(Function("test", blocks=[block]))
    return program, block, dest, user


class TestConstantFolder:
    def test_compute_rejects_unsupported_typed_operation(self):
        result = ConstantFolder._compute(
            OpCode.ADD,
            1.0,
            2.0,
            DataType.FLOAT64,
        )

        assert result is None

    def test_compute_rejects_unsupported_opcode(self):
        result = ConstantFolder._compute(
            OpCode.NEG,
            1.0,
            2.0,
            DataType.FLOAT32,
        )

        assert result is None

    def test_fold_float32_add_uses_binary32_rounding(self):
        program, block, _, _ = _typed_binary_program(
            OpCode.ADD,
            16_777_216.0,
            1.0,
            DataType.FLOAT32,
        )

        count = ConstantFolder().optimize(program)

        result = block.instructions[2].attrs["value"]
        assert count == 1
        assert result == 16_777_216.0
        assert type(result) is float

    def test_fold_float32_preserves_signed_zero(self):
        program, block, _, _ = _typed_binary_program(
            OpCode.SUB,
            -0.0,
            0.0,
            DataType.FLOAT32,
        )

        count = ConstantFolder().optimize(program)

        result = block.instructions[2].attrs["value"]
        assert count == 1
        assert result == 0.0
        assert math.copysign(1.0, result) == -1.0

    @pytest.mark.parametrize(
        ("opcode", "lhs", "rhs", "expected"),
        [
            (OpCode.ADD, 3.0, 4.0, 7.0),
            (OpCode.SUB, 3.5, 5.0, -1.5),
        ],
    )
    def test_fold_float32_add_and_sub(self, opcode, lhs, rhs, expected):
        program, block, _, _ = _typed_binary_program(
            opcode,
            lhs,
            rhs,
            DataType.FLOAT32,
        )

        count = ConstantFolder().optimize(program)

        result = block.instructions[2].attrs["value"]
        assert count == 1
        assert result == expected
        assert type(result) is float

    def test_fold_int32_add_wraps_on_overflow(self):
        program, block, dest, _ = _typed_binary_program(
            OpCode.ADD,
            2_147_483_647,
            1,
            DataType.INT32,
        )

        count = ConstantFolder().optimize(program)

        result = block.instructions[2].attrs["value"]
        assert count == 1
        assert result == -2_147_483_648
        assert type(result) is int
        assert dest.dtype is DataType.INT32

    @pytest.mark.parametrize(
        ("opcode", "lhs", "rhs", "expected"),
        [
            (OpCode.ADD, 20, 22, 42),
            (OpCode.SUB, 3, 5, -2),
            (OpCode.SUB, -2_147_483_648, 1, 2_147_483_647),
            (OpCode.ADD, 2.0, 3.0, 5),
        ],
    )
    def test_fold_int32_add_and_sub(self, opcode, lhs, rhs, expected):
        program, block, dest, _ = _typed_binary_program(
            opcode,
            lhs,
            rhs,
            DataType.INT32,
        )

        count = ConstantFolder().optimize(program)

        result = block.instructions[2].attrs["value"]
        assert count == 1
        assert result == expected
        assert type(result) is int
        assert dest.const_value == expected

    @pytest.mark.parametrize(
        "invalid_payload",
        [True, 1.5, 2_147_483_648, -2_147_483_649, math.inf],
    )
    def test_skip_invalid_int32_payload(self, invalid_payload):
        program, block, dest, _ = _typed_binary_program(
            OpCode.ADD,
            invalid_payload,
            1,
            DataType.INT32,
        )
        candidate = block.instructions[2]

        count = ConstantFolder().optimize(program)

        assert count == 0
        assert block.instructions[2] is candidate
        assert not dest.is_constant

    @pytest.mark.parametrize("invalid_payload", [math.nan, math.inf, -math.inf])
    def test_skip_non_finite_float32_payload(self, invalid_payload):
        program, block, dest, _ = _typed_binary_program(
            OpCode.ADD,
            invalid_payload,
            1.0,
            DataType.FLOAT32,
        )
        candidate = block.instructions[2]

        count = ConstantFolder().optimize(program)

        assert count == 0
        assert block.instructions[2] is candidate
        assert not dest.is_constant

    def test_skip_float32_result_overflow(self):
        program, block, dest, _ = _typed_binary_program(
            OpCode.ADD,
            3.4028234663852886e38,
            3.4028234663852886e38,
            DataType.FLOAT32,
        )
        candidate = block.instructions[2]

        count = ConstantFolder().optimize(program)

        assert count == 0
        assert block.instructions[2] is candidate
        assert not dest.is_constant

    def test_skip_add_when_destination_dtype_differs(self):
        program, block, dest, _ = _typed_binary_program(
            OpCode.ADD,
            20,
            22,
            DataType.INT32,
            dest_dtype=DataType.FLOAT32,
        )
        candidate = block.instructions[2]

        count = ConstantFolder().optimize(program)

        assert count == 0
        assert block.instructions[2] is candidate
        assert not dest.is_constant
        assert dest.const_value is None

    def test_skip_add_when_operand_dtypes_differ(self):
        program, block, dest, _ = _typed_binary_program(
            OpCode.ADD,
            20,
            22.0,
            DataType.INT32,
        )
        candidate = block.instructions[2]
        candidate.operands[1].dtype = DataType.FLOAT32

        count = ConstantFolder().optimize(program)

        assert count == 0
        assert block.instructions[2] is candidate
        assert not dest.is_constant

    @pytest.mark.parametrize("dtype", [DataType.FLOAT64, DataType.INT64])
    def test_skip_unsupported_dtype(self, dtype):
        program, block, dest, _ = _typed_binary_program(
            OpCode.ADD,
            1,
            2,
            dtype,
        )
        candidate = block.instructions[2]

        count = ConstantFolder().optimize(program)

        assert count == 0
        assert block.instructions[2] is candidate
        assert not dest.is_constant

    def test_skip_non_scalar_add(self):
        program, block, dest, _ = _typed_binary_program(
            OpCode.ADD,
            1.0,
            2.0,
            DataType.FLOAT32,
            shape=(2,),
        )
        candidate = block.instructions[2]

        count = ConstantFolder().optimize(program)

        assert count == 0
        assert block.instructions[2] is candidate
        assert not dest.is_constant

    def test_skip_add_with_unknown_instruction_semantics(self):
        for extra_fields in (
            {"candidate_attrs": {"saturating": True}},
            {"target": "other"},
        ):
            program, block, dest, _ = _typed_binary_program(
                OpCode.ADD,
                1.0,
                2.0,
                DataType.FLOAT32,
                **extra_fields,
            )
            candidate = block.instructions[2]

            count = ConstantFolder().optimize(program)

            assert count == 0
            assert block.instructions[2] is candidate
            assert not dest.is_constant

    def test_skip_float32_add_with_bool_payload(self):
        program, block, dest, _ = _typed_binary_program(
            OpCode.ADD,
            True,
            1.0,
            DataType.FLOAT32,
        )
        candidate = block.instructions[2]

        count = ConstantFolder().optimize(program)

        assert count == 0
        assert block.instructions[2] is candidate
        assert not dest.is_constant

    def test_skip_float32_add_with_non_numeric_payload(self):
        program, block, dest, _ = _typed_binary_program(
            OpCode.ADD,
            1.0,
            2.0,
            DataType.FLOAT32,
        )
        candidate = block.instructions[2]
        candidate.operands[0].const_value = "not-a-number"  # type: ignore[assignment]

        count = ConstantFolder().optimize(program)

        assert count == 0
        assert block.instructions[2] is candidate
        assert not dest.is_constant

    @pytest.mark.parametrize(
        "malformation",
        ["zero-operands", "one-operand", "three-operands", "no-dest"],
    )
    def test_skip_structurally_invalid_add(self, malformation):
        program, block, dest, _ = _typed_binary_program(
            OpCode.ADD,
            1.0,
            2.0,
            DataType.FLOAT32,
        )
        candidate = block.instructions[2]
        if malformation == "zero-operands":
            candidate.operands = []
        elif malformation == "one-operand":
            candidate.operands = candidate.operands[:1]
        elif malformation == "three-operands":
            candidate.operands.append(candidate.operands[0])
        else:
            candidate.dest = None

        count = ConstantFolder().optimize(program)

        assert count == 0
        assert block.instructions[2] is candidate
        assert not dest.is_constant

    @pytest.mark.parametrize("invalid_constant", ["not-constant", "no-value"])
    def test_skip_unavailable_constant_payload(self, invalid_constant):
        program, block, dest, _ = _typed_binary_program(
            OpCode.ADD,
            1.0,
            2.0,
            DataType.FLOAT32,
        )
        candidate = block.instructions[2]
        lhs = candidate.operands[0]
        if invalid_constant == "not-constant":
            lhs.is_constant = False
        else:
            lhs.const_value = None

        count = ConstantFolder().optimize(program)

        assert count == 0
        assert block.instructions[2] is candidate
        assert not dest.is_constant

    def test_fold_preserves_destination_and_uses(self):
        program, block, dest, user = _typed_binary_program(
            OpCode.SUB,
            9.0,
            2.0,
            DataType.FLOAT32,
        )
        original_shape = dest.shape

        count = ConstantFolder().optimize(program)

        replacement = block.instructions[2]
        assert count == 1
        assert replacement.opcode is OpCode.LOAD_CONST
        assert replacement.dest is dest
        assert replacement.operands == []
        assert replacement.attrs == {"value": 7.0}
        assert dest.dtype is DataType.FLOAT32
        assert dest.shape == original_shape
        assert dest.is_constant
        assert dest.const_value == replacement.attrs["value"]
        assert user.operands[0] is dest

    def test_repeated_run_reports_only_current_changes(self):
        program, _, _, _ = _typed_binary_program(
            OpCode.ADD,
            1.0,
            2.0,
            DataType.FLOAT32,
        )
        folder = ConstantFolder()

        first_count = folder.optimize(program)
        second_count = folder.optimize(program)

        assert first_count == 1
        assert second_count == 0

    def test_fold_counts_candidates_across_functions_and_blocks(self):
        program = Program()
        blocks = []
        for index, opcode in enumerate((OpCode.ADD, OpCode.SUB, OpCode.ADD)):
            candidate_program, block, _, _ = _typed_binary_program(
                opcode,
                float(index + 3),
                1.0,
                DataType.FLOAT32,
            )
            candidate_function = candidate_program.functions[0]
            if index == 0:
                candidate_function.name = "first"
                program.add_function(candidate_function)
            elif index == 1:
                block.name = "next"
                program.functions[0].blocks.extend(candidate_function.blocks)
            else:
                candidate_function.name = "second"
                program.add_function(candidate_function)
            blocks.append(block)

        count = ConstantFolder().optimize(program)

        assert count == 3
        assert all(
            block.instructions[2].opcode is OpCode.LOAD_CONST
            for block in blocks
        )

    def test_fold_add_constants(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        c1 = builder.load_const(3.0)
        c2 = builder.load_const(4.0)
        r = builder.add(c1, c2)
        builder.ret(r)

        folder = ConstantFolder()
        count = folder.optimize(builder.program)
        assert count == 1

        block = builder.program.functions[0].blocks[0]
        # The add (index 2) should be replaced by load_const 7.0
        assert block.instructions[2].opcode == OpCode.LOAD_CONST
        assert block.instructions[2].attrs["value"] == 7.0

    def test_fold_mul_constants(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        c1 = builder.load_const(2.0)
        c2 = builder.load_const(5.0)
        r = builder.mul(c1, c2)
        builder.ret(r)

        folder = ConstantFolder()
        count = folder.optimize(builder.program)
        assert count == 1
        # The mul (index 2) was replaced by load_const 10.0
        block = builder.program.functions[0].blocks[0]
        assert block.instructions[2].attrs["value"] == 10.0

    def test_no_fold_with_variable(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        a = builder.make_value(name="a")
        c = builder.load_const(5.0)
        r = builder.add(a, c)
        builder.ret(r)

        folder = ConstantFolder()
        count = folder.optimize(builder.program)
        assert count == 0  # cannot fold because 'a' is not constant


class TestDeadCodeEliminator:
    def test_eliminate_unused(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        a = builder.load_const(1.0)
        b = builder.load_const(2.0)
        builder.add(a, b)  # unused!
        d = builder.load_const(3.0)
        builder.ret(d)

        elim = DeadCodeEliminator()
        count = elim.optimize(builder.program)
        assert count == 1  # the add should be eliminated

        block = builder.program.functions[0].blocks[0]
        instrs = block.instructions
        assert all(i.opcode != OpCode.ADD for i in instrs)

    def test_keep_used_value(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        a = builder.load_const(1.0)
        b = builder.load_const(2.0)
        c = builder.add(a, b)  # used by ret
        builder.ret(c)

        elim = DeadCodeEliminator()
        count = elim.optimize(builder.program)
        assert count == 0  # nothing eliminated
