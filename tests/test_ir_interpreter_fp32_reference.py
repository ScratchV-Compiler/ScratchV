"""The opt-in FP32 profile preserves IR semantics and per-run isolation."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as D, Instruction, OpCode as O, Value
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter


DTYPES = {np.dtype("float32"): D.FLOAT32, np.dtype("float64"): D.FLOAT64,
          np.dtype("int32"): D.INT32, np.dtype("int64"): D.INT64}


def operation(op, arrays, attrs=None):
    params = [Value(f"x{i}", DTYPES[x.dtype], shape=x.shape)
              for i, x in enumerate(arrays)]
    builder = IRBuilder()
    builder.new_function("main", params)
    builder.new_block()
    dest = Value("result", params[0].dtype)
    builder.current_block.add(Instruction(op, dest, params, attrs or {}))
    builder.ret(dest)
    return builder.program, dict(zip((p.name for p in params), arrays))


def execute(op, arrays, attrs=None, **options):
    program, inputs = operation(op, arrays, attrs)
    return IRInterpreter(program).run(inputs, **options).return_value


@pytest.mark.parametrize("mode", [None, [], True, "", "Reference", "automatic"])
def test_unknown_profile_is_an_explicit_option_error(mode):
    program, inputs = operation(O.SIGMOID, [np.array([0], "float32")])
    with pytest.raises(IRExecutionError, match="fp32_mode") as caught:
        IRInterpreter(program).run(inputs, fp32_mode=mode)
    assert caught.value.code == "InvalidOptions"


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
def test_softmax_normalizes_the_requested_axis(axis):
    x = np.linspace(-4, 3, 30, dtype="float32").reshape(2, 3, 5)
    actual = execute(O.SOFTMAX, [x], {"axis": axis}, fp32_mode="reference")
    wide = x.astype(np.float64)
    weights = np.exp(wide - np.max(wide, axis=axis, keepdims=True))
    expected = weights / np.sum(weights, axis=axis, keepdims=True)
    assert actual.shape == x.shape and actual.dtype == x.dtype
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=3e-8)
    np.testing.assert_allclose(actual.sum(axis=axis), 1, rtol=0, atol=2e-7)


def test_softmax_finite_extremes_allow_only_negative_overflow_in_shift():
    maximum = np.finfo(np.float32).max
    x = np.array([[-maximum, maximum], [maximum, maximum],
                  [-maximum, -maximum]], "float32")
    actual = execute(O.SOFTMAX, [x], {"axis": 1}, fp32_mode="reference")
    np.testing.assert_array_equal(actual, [[0, 1], [0.5, 0.5], [0.5, 0.5]])


def test_softmax_partial_negative_infinity_mask_is_preserved():
    x = np.array([[0, -np.inf, 0], [-np.inf, 0, -np.inf]], "float32")
    actual = execute(O.SOFTMAX, [x], fp32_mode="reference")
    np.testing.assert_array_equal(actual, [[0.5, 0, 0.5], [0, 1, 0]])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_softmax_invalid_or_entirely_masked_rows_still_fail(bad):
    x = np.full((2, 3), bad, "float32")
    with pytest.raises(IRExecutionError):
        execute(O.SOFTMAX, [x], fp32_mode="reference")


@pytest.mark.parametrize("axis", [3, -4])
def test_softmax_invalid_axis_is_not_silently_reinterpreted(axis):
    with pytest.raises(IRExecutionError, match="ShapeError"):
        execute(O.SOFTMAX, [np.ones((2, 3, 4), "float32")],
                {"axis": axis}, fp32_mode="reference")


@pytest.mark.parametrize("shape,axis", [
    ((0, 7), 1), ((5, 0), 0), ((3, 0, 129), 2), ((3, 129, 0), 1),
])
def test_softmax_preserves_empty_dimensions_outside_normalization_axis(shape, axis):
    actual = execute(O.SOFTMAX, [np.empty(shape, dtype="float32")],
                     {"axis": axis}, fp32_mode="reference")
    assert actual.shape == shape and actual.dtype == np.float32


@pytest.mark.parametrize("left,right", [
    ((3,), (3,)), ((3,), (3, 4)), ((2, 3), (3,)),
    ((2, 1, 3, 4), (1, 5, 4, 2)),
    ((3, 257), (257, 7)), ((2, 1025), (1025, 3)),
    ((257,), (257,)), ((257,), (2, 257, 7)), ((2, 3, 257), (257,)),
    ((2, 1, 3, 257), (1, 5, 257, 2)),
])
def test_matmul_vector_batch_and_reduction_tail_semantics(left, right):
    # Integer-valued inputs give an exact, independent integer arithmetic oracle.
    rng = np.random.default_rng(1221)
    a = rng.integers(-3, 4, left, dtype=np.int64)
    b = rng.integers(-3, 4, right, dtype=np.int64)
    expected = a @ b
    actual = execute(O.MATMUL, [a.astype("float32"), b.astype("float32")],
                     fp32_mode="reference")
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == expected.shape and actual.dtype == np.float32


@pytest.mark.parametrize("left,right,shape", [
    ((2, 0), (0, 3), (2, 3)), ((0,), (0,), ()),
    ((2, 0), (0,), (2,)), ((0,), (0, 3), (3,)),
    ((0, 4), (4, 3), (0, 3)), ((2, 4), (4, 0), (2, 0)),
    ((0, 129), (129, 3), (0, 3)), ((2, 129), (129, 0), (2, 0)),
    ((0, 2, 129), (1, 129, 3), (0, 2, 3)),
])
def test_matmul_empty_dimensions_keep_numpy_shapes(left, right, shape):
    actual = execute(O.MATMUL, [np.zeros(left, "float32"),
                                np.zeros(right, "float32")],
                     fp32_mode="reference")
    np.testing.assert_array_equal(actual, np.zeros(shape, "float32"))


def test_matmul_legacy_flat_shapes_are_still_honored():
    a = np.arange(6, dtype="float32")
    b = np.arange(12, dtype="float32")
    result = execute(O.MATMUL, [a, b], {"m": 2, "n": 4, "k": 3},
                     fp32_mode="reference")
    np.testing.assert_array_equal(result, [[20, 23, 26, 29], [56, 68, 80, 92]])


def test_matmul_consumes_an_ir_transpose_view_without_mutation():
    a = np.arange(18, dtype="float32").reshape(6, 3)
    b = np.arange(24, dtype="float32").reshape(6, 4)
    x, w = Value("x", shape=a.shape), Value("w", shape=b.shape)
    builder = IRBuilder()
    builder.new_function("main", [x, w])
    builder.new_block()
    transposed = builder.transpose(x, (1, 0))
    builder.ret(builder.matmul(transposed, w))
    observed = []

    def observe(name, value):
        if name == transposed.name:
            observed.append(value.flags.c_contiguous)

    actual = IRInterpreter(builder.program).run(
        {"x": a, "w": b}, fp32_mode="reference", observer=observe).return_value
    assert observed == [False]
    np.testing.assert_array_equal(actual, a.astype(np.int64).T @ b.astype(np.int64))
    np.testing.assert_array_equal(a, np.arange(18).reshape(6, 3))
    np.testing.assert_array_equal(b, np.arange(24).reshape(6, 4))


def test_matmul_random_fp32_remains_accurate_beyond_one_reduction_block():
    rng = np.random.default_rng(128)
    a = rng.normal(size=(2, 3, 257)).astype("float32")
    b = rng.normal(size=(257, 129)).astype("float32")
    actual = execute(O.MATMUL, [a, b], fp32_mode="reference")
    expected = a.astype(np.float64) @ b.astype(np.float64)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("length", [1, 3, 7, 8, 9, 15, 17, 127, 129, 1025])
@pytest.mark.parametrize("keepdims", [True, False])
def test_mean_last_axis_tails_and_keepdims(length, keepdims):
    x = np.arange(2 * length, dtype="float32").reshape(2, length) / np.float32(8)
    actual = execute(O.REDUCE_MEAN, [x],
                     {"axes": (-1,), "keepdims": keepdims}, fp32_mode="reference")
    expected = np.mean(x.astype(np.float64), axis=-1, keepdims=keepdims)
    assert actual.shape == expected.shape and actual.dtype == x.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-7)


@pytest.mark.parametrize("axes", [None, (), (0,), (1,), (0, 2), (-1, 0)])
@pytest.mark.parametrize("keepdims", [True, False])
def test_mean_multiple_axes_and_empty_axes_contract(axes, keepdims):
    x = np.arange(30, dtype="float32").reshape(2, 3, 5)
    attrs = {"keepdims": keepdims}
    if axes is not None:
        attrs["axes"] = axes
    expected = np.mean(x.astype(np.float64), axis=axes or None, keepdims=keepdims)
    actual = execute(O.REDUCE_MEAN, [x], attrs, fp32_mode="reference")
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-7)
    assert actual.shape == expected.shape and actual.dtype == x.dtype


def test_mean_scalar_and_empty_dimension_semantics():
    scalar = np.array(7.25, "float32")
    actual = execute(O.REDUCE_MEAN, [scalar], fp32_mode="reference")
    assert actual.shape == () and actual.dtype == scalar.dtype and actual == scalar
    actual = execute(O.REDUCE_MEAN, [np.empty((0, 3), "float32")],
                     {"axes": (-1,)}, fp32_mode="reference")
    assert actual.shape == (0, 1)
    with pytest.raises(IRExecutionError, match="NumericError"):
        execute(O.REDUCE_MEAN, [np.empty((2, 0), "float32")],
                {"axes": (-1,)}, fp32_mode="reference")


def test_sigmoid_remains_bounded_monotone_and_close_to_mathematics():
    maximum = np.finfo(np.float32).max
    x = np.concatenate([np.array([-maximum], "float32"),
                        np.linspace(-30, 30, 1001, dtype="float32"),
                        np.array([maximum], "float32")])
    actual = execute(O.SIGMOID, [x], fp32_mode="reference")
    exp = np.exp(-np.abs(x.astype(np.float64)))
    expected = np.where(x >= 0, 1 / (1 + exp), exp / (1 + exp))
    assert np.isfinite(actual).all() and np.all((actual >= 0) & (actual <= 1))
    # Approximation roundoff can jitter saturated tails by a few ULP.
    assert np.all(np.diff(actual) >= -np.float32(2e-7))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
    assert execute(O.SIGMOID, [np.array(0, "float32")], fp32_mode="reference") == 0.5


def test_sigmoid_saturated_roundoff_never_leaves_probability_range():
    # This neighborhood includes a rational-approximation result above one.
    # Probe both signs and adjacent binary32 values against sigmoid itself.
    center = np.float32(17.868923).view(np.uint32)
    positive = np.arange(center - 16, center + 17, dtype=np.uint32).view(np.float32)
    x = np.concatenate((-positive[::-1], positive))
    actual = execute(O.SIGMOID, [x], fp32_mode="reference")
    expected = 1 / (1 + np.exp(-x.astype(np.float64)))
    assert np.all((actual >= 0) & (actual <= 1))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)


@pytest.mark.parametrize("opcode", [O.MATMUL, O.REDUCE_MEAN, O.SOFTMAX,
                                     O.SIGMOID, O.SIN, O.COS])
def test_float64_results_are_exactly_the_existing_native_results(opcode):
    x = np.linspace(-2, 2, 15, dtype="float64").reshape(3, 5)
    arrays = [x, x.T.copy()] if opcode is O.MATMUL else [x]
    expected = execute(opcode, arrays)
    result = execute(opcode, arrays, fp32_mode="reference")
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float64


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_integer_matmul_preserves_native_wrap_and_dtype(dtype):
    x = np.array([[np.iinfo(dtype).max, 2], [3, -4]], dtype)
    y = np.array([[2, -1], [4, 7]], dtype)
    expected = execute(O.MATMUL, [x, y])
    actual = execute(O.MATMUL, [x, y], fp32_mode="reference")
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == x.dtype


@pytest.mark.parametrize("opcode", [O.SIGMOID, O.REDUCE_MEAN, O.SOFTMAX, O.SIN, O.COS])
def test_profile_does_not_enable_unsupported_integer_float_operations(opcode):
    with pytest.raises(IRExecutionError) as native:
        execute(opcode, [np.array([1, 2], "int64")])
    with pytest.raises(IRExecutionError) as reference:
        execute(opcode, [np.array([1, 2], "int64")], fp32_mode="reference")
    assert reference.value.code == native.value.code


@pytest.mark.parametrize("opcode", [O.MATMUL, O.REDUCE_MEAN, O.SOFTMAX,
                                     O.SIGMOID, O.SIN, O.COS])
def test_reference_profile_never_mutates_registry_or_another_run(opcode):
    from scratchv.verification import ir_numpy_ops

    x = np.random.default_rng(48).normal(size=(2, 257)).astype("float32")
    arrays = [x, x.T.copy()] if opcode is O.MATMUL else [x]
    program, inputs = operation(opcode, arrays)
    interpreter = IRInterpreter(program)
    kernels = dict(ir_numpy_ops.KERNELS)
    original_matmul = np.matmul
    baseline = interpreter.run(inputs).return_value
    explicit = interpreter.run(inputs, fp32_mode="native").return_value
    np.testing.assert_array_equal(explicit, baseline)
    observations = []

    def observe(_name, _value):
        assert ir_numpy_ops.KERNELS == kernels
        assert np.matmul is original_matmul
        nested = interpreter.run(inputs).return_value
        np.testing.assert_array_equal(nested, baseline)
        observations.append(True)

    interpreter.run(inputs, fp32_mode="reference", observer=observe)
    assert observations == [True]
    np.testing.assert_array_equal(interpreter.run(inputs).return_value, baseline)
    assert ir_numpy_ops.KERNELS == kernels


def test_reference_profile_imports_and_executes_without_onnxruntime():
    # A clean interpreter catches eager imports as well as hidden runtime calls.
    script = '''
import sys
class BlockORT:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "onnxruntime":
            raise AssertionError("Reference IR must not import ONNX Runtime")
sys.meta_path.insert(0, BlockORT())
import numpy as np
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value
from scratchv.verification.ir_interpreter import IRInterpreter
x, w = Value("x", shape=(2,3)), Value("w", shape=(3,3))
b = IRBuilder(); b.new_function("main", [x,w]); b.new_block()
v = b.matmul(x,w); v = b.sin(v); v = b.cos(v); v = b.sigmoid(v)
v = b.softmax(v, axis=1); v = b.reduce_mean(v, axes=(1,), keepdims=False); b.ret(v)
out = IRInterpreter(b.program).run(
    {"x":np.arange(6,dtype="float32").reshape(2,3),"w":np.eye(3,dtype="float32")},
    fp32_mode="reference").return_value
np.testing.assert_allclose(out, [1/3,1/3], rtol=3e-7)
assert out.dtype == np.float32
assert not any(name.split(".")[0] == "onnxruntime" for name in sys.modules)
'''
    completed = subprocess.run([sys.executable, "-B", "-c", script],
                               cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
