"""Extended instruction-selection IR and DSL coverage."""

import pytest

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.frontend.dsl_errors import DSLSyntaxError
from scratchv.frontend.dsl_parser import DSLParser
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


def _extension_program(
    opcode: OpCode,
    operand_types: tuple[DataType, ...],
    result_type: DataType,
) -> Program:
    operands = [
        Value(f"arg{index}", dtype=dtype) for index, dtype in enumerate(operand_types)
    ]
    result = Value("result", dtype=result_type)
    block = BasicBlock("entry")
    block.add(Instruction(opcode, result, operands))
    block.add(Instruction(OpCode.RETURN, operands=[result]))
    function = Function("main", params=operands, blocks=[block])
    program = Program()
    program.add_function(function)
    return program


class TestTypedIRFrontend:
    def test_builder_propagates_arithmetic_result_types(self):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        lhs = builder.make_value("lhs", dtype=DataType.FLOAT64)
        rhs = builder.make_value("rhs", dtype=DataType.FLOAT64)

        assert builder.add(lhs, rhs).dtype is DataType.FLOAT64
        assert builder.neg(lhs).dtype is DataType.FLOAT64

    @pytest.mark.parametrize(
        ("method", "opcode", "arity"),
        [
            ("rem", OpCode.REM, 2),
            ("min", OpCode.MIN, 2),
            ("max", OpCode.MAX, 2),
            ("abs", OpCode.ABS, 1),
        ],
    )
    def test_builder_constructs_integer_extension_ops(self, method, opcode, arity):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        values = [
            builder.make_value(f"arg{index}", dtype=DataType.INT32)
            for index in range(arity)
        ]

        result = getattr(builder, method)(*values)

        instruction = builder.current_block.instructions[-1]
        assert instruction.opcode is opcode
        assert instruction.dest is result
        assert result.dtype is DataType.INT32

    def test_builder_rejects_mixed_and_unsupported_types(self):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        i32 = builder.make_value("i32", dtype=DataType.INT32)
        f32 = builder.make_value("f32", dtype=DataType.FLOAT32)
        i64 = builder.make_value("i64", dtype=DataType.INT64)

        with pytest.raises(ValueError, match="add operands must have the same type"):
            builder.add(i32, f32)
        with pytest.raises(ValueError, match="rem does not support f32"):
            builder.rem(f32, f32)
        with pytest.raises(ValueError, match="min does not support i64"):
            builder.min(i64, i64)
        # Shared IR also serves the checked tensor interpreter; the extended
        # RV32 backend rejects i64 at its own type boundary.
        assert builder.abs(i64).dtype is DataType.INT64

    @pytest.mark.parametrize(
        "dtype", [DataType.INT32, DataType.FLOAT32, DataType.FLOAT64]
    )
    def test_builder_accepts_numeric_min_max_abs(self, dtype):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        lhs = builder.make_value("lhs", dtype=dtype)
        rhs = builder.make_value("rhs", dtype=dtype)

        assert builder.min(lhs, rhs).dtype is dtype
        assert builder.max(lhs, rhs).dtype is dtype
        assert builder.abs(lhs).dtype is dtype

    def test_dsl_const_makes_integer_extension_ops_reachable(self):
        program = DSLParser().parse(
            """
        a = const(7, dtype:i32)
        b = const(3, dtype:i32)
        r = rem(a, b)
        alias = mod(a, b)
        lo = min(r, alias)
        hi = max(r, alias)
        magnitude = abs(hi)
        return magnitude
        """
        )
        instructions = program.functions[0].blocks[0].instructions

        assert [instruction.opcode for instruction in instructions] == [
            OpCode.LOAD_CONST,
            OpCode.LOAD_CONST,
            OpCode.REM,
            OpCode.REM,
            OpCode.MIN,
            OpCode.MAX,
            OpCode.ABS,
            OpCode.RETURN,
        ]
        assert all(
            instruction.dest is None or instruction.dest.dtype is DataType.INT32
            for instruction in instructions
        )
        assert verify_ir(program) == (True, [])

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            ("i32", DataType.INT32),
            ("f32", DataType.FLOAT32),
            ("f64", DataType.FLOAT64),
        ],
    )
    def test_dsl_const_supports_extended_numeric_types(self, dtype, expected):
        program = DSLParser().parse(f"value = const(1, dtype:{dtype})\nreturn value")
        constant = program.functions[0].blocks[0].instructions[0]

        assert constant.opcode is OpCode.LOAD_CONST
        assert constant.dest.dtype is expected

    @pytest.mark.parametrize(
        "source",
        [
            "value = const(1, dtype:i64)",
            "value = const(1.5, dtype:i32)",
            "value = const(nope, dtype:f32)",
        ],
    )
    def test_dsl_const_rejects_unsupported_or_invalid_values(self, source):
        collector = DSLParser().validate(source)
        assert [error.error_code for error in collector.errors] == ["E203"]
        with pytest.raises(DSLSyntaxError):
            DSLParser().parse(source)

    @pytest.mark.parametrize(
        ("opcode", "arity", "dtypes"),
        [
            (OpCode.REM, 2, (DataType.INT32,)),
            (OpCode.MIN, 2, (DataType.INT32, DataType.FLOAT32, DataType.FLOAT64)),
            (OpCode.MAX, 2, (DataType.INT32, DataType.FLOAT32, DataType.FLOAT64)),
            (OpCode.ABS, 1, tuple(DataType)),
        ],
    )
    def test_ir_verifier_accepts_extension_signatures(self, opcode, arity, dtypes):
        for dtype in dtypes:
            program = _extension_program(opcode, (dtype,) * arity, dtype)
            assert verify_ir(program) == (True, [])

    @pytest.mark.parametrize(
        ("opcode", "operands", "result"),
        [
            (OpCode.REM, (DataType.FLOAT32, DataType.FLOAT32), DataType.FLOAT32),
            (OpCode.MIN, (DataType.INT32, DataType.FLOAT32), DataType.INT32),
            (OpCode.MAX, (DataType.INT64, DataType.INT64), DataType.INT64),
            (OpCode.ABS, (DataType.INT64,), DataType.INT32),
        ],
    )
    def test_ir_verifier_rejects_unsupported_extension_signatures(
        self,
        opcode,
        operands,
        result,
    ):
        passed, issues = verify_ir(_extension_program(opcode, operands, result))
        assert not passed
        assert any(
            issue.rule == "type-consistency"
            and issue.message.startswith("incompatible types:")
            for issue in issues
        )
