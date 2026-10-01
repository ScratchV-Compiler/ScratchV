"""Numeric Qwen primitives: dtype contracts, exact integers and strict failures."""

import numpy as np
import pytest

from scratchv.analysis.ir_verifier import verify_ir
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as D
from scratchv.ir.types import Instruction, OpCode as O, Value
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter


DTYPES = {
    np.dtype("float32"): D.FLOAT32,
    np.dtype("float64"): D.FLOAT64,
    np.dtype("int32"): D.INT32,
    np.dtype("int64"): D.INT64,
}


def operation(op, arrays, dtype=None, attrs=None):
    params = [Value(f"x{i}", DTYPES[x.dtype], shape=x.shape)
              for i, x in enumerate(arrays)]
    builder = IRBuilder()
    builder.new_function("main", params)
    builder.new_block()
    dest = Value("result", dtype or params[0].dtype)
    builder.current_block.add(Instruction(op, dest, params, attrs or {}))
    builder.ret(dest)
    return builder.program, dict(zip((p.name for p in params), arrays))


def run(op, arrays, dtype=None, attrs=None):
    program, inputs = operation(op, arrays, dtype, attrs)
    return IRInterpreter(program).run(inputs).return_value


@pytest.mark.parametrize("name,op", [
    ("abs", O.ABS), ("cos", O.COS), ("sin", O.SIN),
    ("reciprocal", O.RECIPROCAL),
])
def test_unary_builders_preserve_type_and_shape(name, op):
    value = Value("x", D.FLOAT64, shape=(2, 3))
    builder = IRBuilder()
    builder.new_function("main", [value])
    builder.new_block()
    result = getattr(builder, name)(value)
    builder.ret(result)
    assert (result.dtype, result.shape) == (D.FLOAT64, (2, 3))
    assert builder.current_block.instructions[0].opcode is op
    assert verify_ir(builder.program)[0]


def test_cast_and_mixed_power_builders():
    value = Value("x", D.FLOAT32, shape=(3,))
    builder = IRBuilder()
    builder.new_function("main", [value])
    builder.new_block()
    cast = builder.cast(value, D.FLOAT64)
    result = builder.pow(cast, builder.make_const(2, D.INT64))
    builder.ret(result)
    assert cast.shape == value.shape
    assert result.dtype is D.FLOAT64
    assert builder.current_block.instructions[0].attrs == {}
    np.testing.assert_array_equal(
        IRInterpreter(builder.program).run({"x": np.array([1, 2, 3], "float32")})
        .return_value, [1, 4, 9]
    )
    with pytest.raises(ValueError, match="supported DataType"):
        builder.cast(value, "float16")


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_abs_tensor_and_scalar(dtype):
    for data in (np.array([[-3, 0], [2, -7]], dtype), np.array(-5, dtype)):
        result = run(O.ABS, [data])
        np.testing.assert_array_equal(result, np.abs(data))
        assert result.dtype == data.dtype and result.shape == data.shape


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_trigonometric_known_angles_and_reciprocal(dtype):
    data = np.array([0, np.pi / 2, -np.pi / 2, np.pi], dtype)
    tolerance = 2e-7 if dtype == "float32" else 1e-15
    np.testing.assert_allclose(run(O.SIN, [data]), [0, 1, -1, 0], atol=tolerance)
    np.testing.assert_allclose(run(O.COS, [data]), [1, 0, 0, -1], atol=tolerance)
    np.testing.assert_array_equal(
        run(O.RECIPROCAL, [np.array([[2, -4], [0.5, -0.25]], dtype)]),
        [[0.5, -0.25], [2, -4]],
    )


@pytest.mark.parametrize("source", ["float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("target", ["float32", "float64", "int32", "int64"])
def test_cast_supported_type_pairs(source, target):
    data = np.array([[-12.75, 0], [3.75, 42]], source)
    result = run(O.CAST, [data], DTYPES[np.dtype(target)])
    np.testing.assert_array_equal(result, data.astype(target))
    assert result.dtype == target and result.shape == data.shape


def test_cast_i64_precision_narrowing_and_truncation():
    wide = np.array([2**60 + 1, -(2**60 + 3), 2**31, -(2**31) - 1], "int64")
    np.testing.assert_array_equal(run(O.CAST, [wide], D.INT64), wide)
    np.testing.assert_array_equal(run(O.CAST, [wide], D.INT32), [1, -3, -2**31, 2**31 - 1])
    np.testing.assert_array_equal(
        run(O.CAST, [np.array([-3.9, -0.9, 0.9, 3.9], "float64")], D.INT64),
        [-3, 0, 0, 3],
    )


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_float_to_integer_endpoint_bounds(dtype):
    target = DTYPES[np.dtype(dtype)]
    bound = 2 ** (np.iinfo(dtype).bits - 1)
    values = np.array([-bound, np.nextafter(float(bound), -np.inf)], "float64")
    result = run(O.CAST, [values], target)
    np.testing.assert_array_equal(result, [int(v) for v in values])
    for invalid in (float(bound), np.nextafter(float(-bound), -np.inf) - 1):
        with pytest.raises(IRExecutionError, match="NumericError"):
            run(O.CAST, [np.array(invalid, "float64")], target)


@pytest.mark.parametrize("base_dtype", ["float32", "float64"])
@pytest.mark.parametrize("exponent_dtype", ["float32", "float64", "int32", "int64"])
def test_power_mixed_dtype_broadcast(base_dtype, exponent_dtype):
    result = run(O.POW, [np.array([[2], [3]], base_dtype),
                         np.array([0, 1, 3], exponent_dtype)])
    np.testing.assert_array_equal(result, [[1, 2, 8], [1, 3, 27]])
    assert result.dtype == base_dtype and result.shape == (2, 3)


def test_power_integer_scalar_exponent_keeps_f32_and_parity():
    result = run(O.POW, [np.array([-1, 1], "float32"), np.array(16_777_217, "int64")])
    np.testing.assert_array_equal(result, [-1, 1])
    assert result.dtype == "float32"
    np.testing.assert_array_equal(
        run(O.POW, [np.array([4, 16], "float32"), np.array(0.5, "float64")]), [2, 4]
    )


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_integer_power_exact_bounded_and_fractional(dtype):
    minimum = np.iinfo(dtype).min
    exact = np.array([minimum, np.iinfo(dtype).max, 0], dtype)
    np.testing.assert_array_equal(run(O.POW, [exact, np.array(1, "int64")]), exact)
    np.testing.assert_array_equal(
        run(O.POW, [np.array([-2, 2, -1], dtype), np.array([3, 10, 2**62 + 1], "int64")]),
        [-8, 1024, -1],
    )
    np.testing.assert_array_equal(
        run(O.POW, [np.array([9, 2, -1], dtype), np.array([0.5, -2, -3], "float64")]),
        [3, 0, -1],
    )


@pytest.mark.parametrize("op,arrays,dtype", [
    (O.ABS, [np.array(np.iinfo("int32").min, "int32")], None),
    (O.ABS, [np.array(np.iinfo("int64").min, "int64")], None),
    (O.RECIPROCAL, [np.array([0.0, -0.0], "float32")], None),
    (O.RECIPROCAL, [np.array(np.nextafter(np.float32(0), np.float32(1)))], None),
    (O.COS, [np.array(-np.inf, "float32")], None),
    (O.SIN, [np.array(-np.inf, "float64")], None),
    (O.CAST, [np.array(-np.inf, "float64")], D.INT64),
    (O.CAST, [np.array(np.finfo("float64").max, "float64")], D.FLOAT32),
    (O.POW, [np.array(-2, "float32"), np.array(0.5, "float64")], None),
    (O.POW, [np.array(0, "float32"), np.array(-1, "int64")], None),
    (O.POW, [np.array(1e20, "float32"), np.array(2, "int64")], None),
    (O.POW, [np.array(2, "int32"), np.array(31, "int64")], None),
    (O.POW, [np.array(2, "int64"), np.array(63, "int64")], None),
    (O.POW, [np.array(-1, "int64"), np.array(0.5, "float32")], None),
])
def test_numeric_failures_are_explicit(op, arrays, dtype):
    with pytest.raises(IRExecutionError, match="NumericError") as found:
        run(op, arrays, dtype)
    assert found.value.opcode is op
    assert found.value.instruction_index == 0


@pytest.mark.parametrize("op", [O.COS, O.SIN, O.RECIPROCAL])
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_floating_unary_rejects_integer_ir(op, dtype):
    with pytest.raises(IRExecutionError, match="InvalidProgram"):
        run(op, [np.array([1], dtype)])


@pytest.mark.parametrize("op", [O.ABS, O.COS, O.SIN, O.RECIPROCAL, O.CAST, O.POW])
def test_bad_arity_and_unknown_attributes(op):
    arrays = [np.array(2, "float32")] * (2 if op is O.POW else 1)
    program, inputs = operation(op, arrays)
    program.functions[0].blocks[0].instructions[0].operands.clear()
    with pytest.raises(IRExecutionError, match="InvalidProgram"):
        IRInterpreter(program).run(inputs)
    with pytest.raises(IRExecutionError, match="UnsupportedAttribute"):
        run(op, arrays, attrs={"unexpected": 1})


def test_pow_result_type_and_broadcast_failures():
    with pytest.raises(IRExecutionError, match="InvalidProgram"):
        run(O.POW, [np.array(2, "float32"), np.array(2, "int64")], D.FLOAT64)
    with pytest.raises(IRExecutionError, match="ShapeError"):
        run(O.POW, [np.ones((2, 3), "float32"), np.ones((4,), "int64")])


@pytest.mark.parametrize("op", [O.ABS, O.COS, O.SIN, O.RECIPROCAL, O.CAST, O.POW])
def test_empty_tensor_shape_preserved(op):
    data = np.empty((2, 0, 3), "float32")
    result = run(op, [data, np.array(2, "int64")] if op is O.POW else [data])
    assert result.shape == data.shape and result.dtype == data.dtype
