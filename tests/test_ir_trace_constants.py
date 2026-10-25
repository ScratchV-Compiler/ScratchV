"""R3: constant storage bits agree at load, operand and return boundaries."""

import sys

import numpy as np
import pytest

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as T, Instruction, OpCode, Program, Value
from scratchv.simulator.rv32_emulator import IRTraceExecutor


PRECISIONS = [pytest.param(T.FLOAT32, id="f32"), pytest.param(T.FLOAT64, id="f64")]
STORAGE = {T.FLOAT32: (np.float32, 4), T.FLOAT64: (np.float64, 8)}
ENCODINGS = {
    "one": (0x3F800000, 0x3FF0000000000000),
    "positive_zero": (0x00000000, 0x0000000000000000),
    "negative_zero": (0x80000000, 0x8000000000000000),
    "quiet_nan": (0x7FC01234, 0x7FF8000000001234),
    "signaling_nan": (0x7F801234, 0x7FF0000000001234),
}
TWO = {T.FLOAT32: 0x40000000, T.FLOAT64: 0x4000000000000000}
THREE = {T.FLOAT32: 0x40400000, T.FLOAT64: 0x4008000000000000}


def bits_for(case, dtype):
    return ENCODINGS[case][dtype is T.FLOAT64]


def assert_storage(result, dtype, bits):
    numpy_type, size = STORAGE[dtype]
    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result.dtype == numpy_type
    # Expected bytes come from integer literals, not the production decoder.
    assert result.tobytes() == bits.to_bytes(size, sys.byteorder)


def read_constant(executor, entry, value):
    if entry == "operand":
        return executor._resolve(value)
    return executor._op_load_const(
        Instruction(
            OpCode.LOAD_CONST,
            value,
            attrs={"value": value.const_value, "bit_pattern": value.bit_pattern},
        )
    )


def encoded_program(dtype, source):
    builder = IRBuilder()
    builder.new_function("main")
    builder.new_block("entry")
    value = builder.load_const(0.0, dtype, bit_pattern=bits_for("one", dtype))
    instruction = builder.current_block.instructions[-1]
    if source == "instruction":
        value.bit_pattern = None
    elif source == "value":
        instruction.attrs.pop("bit_pattern")
    return builder, value, instruction


class TestIRTraceConstantStorage:
    @pytest.mark.parametrize("dtype", PRECISIONS)
    @pytest.mark.parametrize("case", ENCODINGS)
    @pytest.mark.parametrize("entry", ["load", "operand"])
    def test_explicit_constant_encoding_preserves_every_bit(self, dtype, case, entry):
        host = (
            float("nan")
            if case.endswith("nan")
            else 1.0 if case == "positive_zero" else 0.0
        )
        bits = bits_for(case, dtype)
        value = Value(
            "encoded", dtype, is_constant=True, const_value=host, bit_pattern=bits
        )
        executor = IRTraceExecutor(Program())

        assert_storage(read_constant(executor, entry, value), dtype, bits)

    @pytest.mark.parametrize("dtype", PRECISIONS)
    @pytest.mark.parametrize("source", ["instruction", "value", "both"])
    def test_return_uses_the_encoding_from_the_constant_definition(self, dtype, source):
        builder, value, _ = encoded_program(dtype, source)
        builder.ret(value)
        assert verify_ir(builder.program) == (True, [])

        assert_storage(
            IRTraceExecutor(builder.program).run({}), dtype, bits_for("one", dtype)
        )

    @pytest.mark.parametrize("dtype", PRECISIONS)
    def test_arithmetic_reads_instruction_only_and_bare_value_encodings(self, dtype):
        builder, left, _ = encoded_program(dtype, "instruction")
        right = Value(
            "global", dtype, is_constant=True, const_value=0.0, bit_pattern=TWO[dtype]
        )
        builder.program.global_values.append(right)
        builder.ret(builder.add(left, right))
        assert verify_ir(builder.program) == (True, [])

        result = IRTraceExecutor(builder.program).run({})
        assert_storage(np.asarray(result), dtype, THREE[dtype])

    @pytest.mark.parametrize("dtype", PRECISIONS)
    def test_repeated_runs_reload_encodings_and_do_not_override_bare_constants(
        self, dtype
    ):
        builder, value, instruction = encoded_program(dtype, "instruction")
        builder.ret(value)
        executor = IRTraceExecutor(builder.program)
        assert_storage(executor.run({}), dtype, bits_for("one", dtype))
        instruction.attrs["bit_pattern"] = TWO[dtype]
        assert_storage(executor.run({}), dtype, TWO[dtype])

        # Reuse the name without an encoded LOAD_CONST in the next execution.
        builder.current_block.instructions.remove(instruction)
        builder.program.global_values.append(value)
        assert verify_ir(builder.program) == (True, [])
        numpy_type, _ = STORAGE[dtype]
        result = executor.run({value.name: np.array(9.0, dtype=numpy_type)})
        assert_storage(result, dtype, bits_for("positive_zero", dtype))

    @pytest.mark.parametrize("dtype", PRECISIONS)
    def test_constant_definitions_do_not_leak_between_functions(self, dtype):
        first, value, _ = encoded_program(dtype, "instruction")
        first.ret(value)
        second = IRBuilder()
        second.new_function("other")
        second.new_block("entry")
        constant = second.make_const(0.0, dtype)
        constant.bit_pattern = TWO[dtype]
        second.ret(constant)
        assert value.name == constant.name
        first.program.functions.extend(second.program.functions)
        assert verify_ir(first.program) == (True, [])

        assert_storage(IRTraceExecutor(first.program).run({}), dtype, TWO[dtype])

    @pytest.mark.parametrize("dtype", PRECISIONS)
    def test_conflicting_instruction_and_value_encodings_fail_before_storing(
        self, dtype
    ):
        _, value, instruction = encoded_program(dtype, "both")
        instruction.attrs["bit_pattern"] = TWO[dtype]
        executor = IRTraceExecutor(Program())
        previous = np.array(9.0)
        executor._values[value.name] = previous

        with pytest.raises(ValueError, match="bit_pattern differs"):
            executor._op_load_const(instruction)

        assert executor._values[value.name] is previous

    @pytest.mark.parametrize("dtype", PRECISIONS)
    @pytest.mark.parametrize("invalid", ["negative", "too_wide", "boolean", "fraction"])
    @pytest.mark.parametrize("entry", ["load", "operand"])
    def test_malformed_explicit_encodings_are_rejected(self, dtype, invalid, entry):
        _, size = STORAGE[dtype]
        bits = {
            "negative": -1,
            "too_wide": 1 << (size * 8),
            "boolean": True,
            "fraction": 1.5,
        }[invalid]
        value = Value(
            "invalid", dtype, is_constant=True, const_value=0.0, bit_pattern=bits
        )

        with pytest.raises(ValueError, match="bit_pattern"):
            read_constant(IRTraceExecutor(Program()), entry, value)

    @pytest.mark.parametrize("entry", ["load", "operand"])
    def test_float_encoding_is_not_accepted_for_an_integer_constant(self, entry):
        value = Value(
            "invalid", T.INT32, is_constant=True, const_value=0, bit_pattern=0
        )

        with pytest.raises(ValueError, match="bit_pattern"):
            read_constant(IRTraceExecutor(Program()), entry, value)

    @pytest.mark.parametrize(
        "dtype,host,numpy_type",
        [
            (T.INT32, -7, np.int32),
            (T.FLOAT32, 1.25, np.float32),
            (T.FLOAT64, 1.0000000000000002, np.float64),
        ],
    )
    def test_unencoded_constants_keep_existing_numeric_and_dtype_behavior(
        self, dtype, host, numpy_type
    ):
        value = Value("plain", dtype, is_constant=True, const_value=host)
        expected = np.array(host, dtype=numpy_type)
        executor = IRTraceExecutor(Program())
        executor._values[value.name] = np.array(99)
        for entry in ("operand", "load"):
            result = read_constant(executor, entry, value)
            assert result.shape == ()
            assert result.dtype == numpy_type
            assert result.tobytes() == expected.tobytes()
