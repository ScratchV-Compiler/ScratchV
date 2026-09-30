"""Kernel semantics with hand computed data, precision and attribute boundaries."""

import math

import numpy as np
import pytest

from benchmarks.ir_interpreter_cases import builder
from scratchv.analysis.ir_verifier import verify_ir
from scratchv.ir.types import DataType as D
from scratchv.ir.types import Instruction, Value
from scratchv.ir.types import OpCode as O
from scratchv.verification.ir_interpreter import IRExecutionError, IRInterpreter


def operation(op, arrays, attrs=None, result_dtype=None):
    dtype_map = {
        "float32": D.FLOAT32,
        "float64": D.FLOAT64,
        "int32": D.INT32,
        "int64": D.INT64,
    }
    params = [Value(f"x{i}", dtype_map[str(x.dtype)]) for i, x in enumerate(arrays)]
    b = builder(*params)
    result = Value("result", result_dtype or params[0].dtype)
    b.current_block.add(Instruction(op, result, params, attrs or {}))
    b.ret(result)
    return b.program, dict(zip((v.name for v in params), arrays))


def run(op, arrays, attrs=None, result_dtype=None):
    p, inputs = operation(op, arrays, attrs, result_dtype)
    return IRInterpreter(p).run(inputs).return_value


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "op,expected",
    [
        (O.ADD, [5, 1]),
        (O.SUB, [1, -5]),
        (O.MUL, [6, -6]),
        (O.DIV, [1.5, -2 / 3]),
    ],
)
def test_arithmetic_dtype_and_no_epsilon(dtype, op, expected):
    result = run(op, [np.array([3, -2], dtype=dtype), np.array([2, 3], dtype=dtype)])
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    if op == O.DIV:
        exact = run(op, [np.array([1], dtype=dtype), np.array([1e-8], dtype=dtype)])
        np.testing.assert_allclose(exact, [1e8], rtol=1e-6)


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_integer_wrap_and_truncated_division(dtype):
    limits = np.iinfo(dtype)

    def arr(value):
        return np.asarray(value, dtype=dtype)

    assert run(O.ADD, [arr(limits.max), arr(1)]) == limits.min
    assert run(O.SUB, [arr(limits.min), arr(1)]) == limits.max
    assert run(O.MUL, [arr(limits.max), arr(2)]) == -2
    assert run(O.NEG, [arr(limits.min)]) == limits.min
    numerators = [7, -7, 7, -7, limits.max]
    denominators = [3, 3, -3, -3, 3]
    actual = run(O.DIV, [arr(numerators), arr(denominators)])
    np.testing.assert_array_equal(actual, [2, -2, -2, 2, limits.max // 3])
    with pytest.raises(IRExecutionError, match="NumericError"):
        run(O.DIV, [arr(limits.min), arr(-1)])
    with pytest.raises(IRExecutionError, match="NumericError"):
        run(O.DIV, [arr(1), arr(0)])


def test_broadcast_and_batched_matmul():
    result = run(
        O.ADD,
        [np.array([[1], [3]], dtype="float64"), np.array([2, 4, 6], dtype="float64")],
    )
    np.testing.assert_array_equal(result, [[3, 5, 7], [5, 7, 9]])
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype="float32")
    weights = np.array([[1, 0], [0, 2]], dtype="float32")
    np.testing.assert_array_equal(
        run(O.MATMUL, [data, weights]), [[[1, 4], [3, 8]], [[5, 12], [7, 16]]]
    )
    with pytest.raises(IRExecutionError, match="ShapeError"):
        run(O.MATMUL, [data, weights], {"m": 2, "n": 2, "k": 2})
    np.testing.assert_array_equal(
        run(O.MATMUL, [data[0].ravel(), weights.ravel()], {"m": 2, "n": 2, "k": 2}),
        [[1, 4], [3, 8]],
    )


def test_reshape_and_transpose_rank_three():
    x = np.arange(6, dtype="float32").reshape(2, 3)
    np.testing.assert_array_equal(run(O.RESHAPE, [x], {"shape": (0, -1)}), x)
    y = x.reshape(1, 2, 3)
    expected = [[[0], [3]], [[1], [4]], [[2], [5]]]
    np.testing.assert_array_equal(run(O.TRANSPOSE, [y]), expected)
    np.testing.assert_array_equal(
        run(O.TRANSPOSE, [y], {"perm": (0, 2, 1)}), [[[0, 3], [1, 4], [2, 5]]]
    )
    assert run(O.RESHAPE, [np.array([7], dtype="float32")], {"shape": ()}).shape == ()


def test_new_shape_and_index_boundaries():
    data = np.array([[10, 20, 30], [40, 50, 60]], dtype="float64")
    indices = np.array([[-1], [0]], dtype="int32")
    result = run(O.GATHER, [data, indices], {"axis": -1})
    assert result.dtype == np.dtype("float64")
    np.testing.assert_array_equal(result, [[[30], [10]], [[60], [40]]])
    np.testing.assert_array_equal(
        run(
            O.SLICE,
            [np.array([1, 2, 3, 4, 5], dtype="int64")],
            {"starts": (4,), "ends": (-10,), "steps": (-1,)},
        ),
        [5, 4, 3, 2, 1],
    )
    assert run(O.UNSQUEEZE, [data], {"axes": (-1, 0)}).shape == (1, 2, 3, 1)
    np.testing.assert_array_equal(
        run(O.EXPAND, [np.array([1, 2, 3], dtype="int32")], {"shape": (1,)}), [1, 2, 3]
    )
    assert run(O.EXPAND, [np.array([1], dtype="float32")], {"shape": (0,)}).shape == (
        0,
    )


@pytest.mark.parametrize("keepdims,expected", [(True, [[2], [6]]), (False, [2, 6])])
def test_mean_axes_and_dtype(keepdims, expected):
    data = np.array([[1, 3], [5, 7]], dtype="float64")
    result = run(O.REDUCE_MEAN, [data], {"axes": (-1,), "keepdims": keepdims})
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == data.dtype
    assert run(O.REDUCE_MEAN, [data], {"axes": (), "keepdims": False}) == 4


@pytest.mark.parametrize(
    "op,attrs,arrays,code",
    [
        (O.SQRT, {}, [np.array([-1], dtype="float32")], "NumericError"),
        (
            O.REDUCE_MEAN,
            {"axes": (0,)},
            [np.empty((0, 2), dtype="float32")],
            "NumericError",
        ),
        (
            O.GATHER,
            {},
            [np.ones(2, dtype="float32"), np.array([2], dtype="int64")],
            "IndexError",
        ),
        (
            O.SLICE,
            {"starts": (0,), "ends": (2,), "steps": (0,)},
            [np.ones(2, dtype="float32")],
            "AttributeError",
        ),
        (
            O.UNSQUEEZE,
            {"axes": (0, -3)},
            [np.ones(2, dtype="float32")],
            "AttributeError",
        ),
        (O.EXPAND, {"shape": (3,)}, [np.ones(2, dtype="float32")], "ShapeError"),
        (O.RESHAPE, {}, [np.ones(2, dtype="float32")], "AttributeError"),
        (
            O.RESHAPE,
            {"shape": (-1, -1)},
            [np.ones(2, dtype="float32")],
            "AttributeError",
        ),
        (
            O.TRANSPOSE,
            {"perm": (0, 0)},
            [np.ones((2, 2), dtype="float32")],
            "AttributeError",
        ),
        (O.CONCAT, {"axis": 3}, [np.ones(2, dtype="float32")], "ShapeError"),
        (
            O.SOFTMAX,
            {},
            [np.array([-np.inf, -np.inf], dtype="float32")],
            "NumericError",
        ),
        (O.EXP, {}, [np.array([1000], dtype="float32")], "NumericError"),
        (
            O.DIV,
            {},
            [np.ones(1, dtype="float32"), np.zeros(1, dtype="float32")],
            "NumericError",
        ),
        (O.MATMUL, {"m": 2}, [np.ones((2, 2), dtype="float32")] * 2, "AttributeError"),
    ],
)
def test_failures(op, attrs, arrays, code):
    with pytest.raises(IRExecutionError) as found:
        run(op, arrays, attrs)
    assert found.value.code == code


def test_softmax_sigmoid_and_remaining_activations():
    np.testing.assert_array_equal(
        run(O.SOFTMAX, [np.array([-np.inf, 3], dtype="float32")]), [0, 1]
    )
    np.testing.assert_allclose(
        run(O.SOFTMAX, [np.array([1000, 1001], dtype="float64")]),
        [1 / (1 + math.e), math.e / (1 + math.e)],
    )
    np.testing.assert_array_equal(
        run(O.SOFTMAX, [np.zeros((2, 3), dtype="float32")], {"axis": 0}),
        np.full((2, 3), 0.5),
    )
    np.testing.assert_array_equal(
        run(O.SIGMOID, [np.array([-10000, 0, 10000], dtype="float32")]), [0, 0.5, 1]
    )
    assert run(O.SIGMOID, [np.array(0, dtype="float64")]) == 0.5
    np.testing.assert_array_equal(
        run(O.RELU, [np.array([-2, 0, 3], dtype="float32")]), [0, 0, 3]
    )
    assert run(O.EXP, [np.array(0, dtype="float64")]) == 1
    assert run(O.SQRT, [np.array(9, dtype="float64")]) == 3
    assert run(O.GELU, [np.array(0, dtype="float64")]) == 0
    assert (
        run(
            O.DOT,
            [np.array([1, 2], dtype="float64"), np.array([3, 4], dtype="float64")],
            {"length": 2},
        )
        == 11
    )
    np.testing.assert_array_equal(
        run(
            O.GEMM,
            [
                np.array([[1, 3], [2, 4]], dtype="float32"),
                np.eye(2, dtype="float32"),
                np.array([1, 2], dtype="float32"),
            ],
            {"trans_a": True, "alpha": 2, "beta": 3},
        ),
        [[5, 10], [9, 14]],
    )


def test_conv_pool_with_batches_and_nonsquare_spatial_dimensions():
    x = np.arange(24, dtype="float32").reshape(2, 1, 3, 4) - 20
    w = np.ones((1, 1, 2, 2), dtype="float32")
    result = run(
        O.CONV,
        [x, w, np.array([2], dtype="float32")],
        {"kernel_size": 2, "stride": 1, "padding": 0, "out_channels": 1},
    )
    expected = [
        [
            [
                [
                    sum(
                        float(x[n, 0, i + di, j + dj])
                        for di in range(2)
                        for dj in range(2)
                    )
                    + 2
                    for j in range(3)
                ]
                for i in range(2)
            ]
        ]
        for n in range(2)
    ]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(
        run(O.MAXPOOL, [x], {"kernel": 2, "stride": 2}), [[[[-15, -13]]], [[[-3, -1]]]]
    )
    with pytest.raises(IRExecutionError, match="UnsupportedAttribute"):
        run(O.CONV, [x, w, np.array([2], dtype="float32")], {"group": 2})


def test_gather_static_types_and_builder_precision():
    p, _inputs = operation(
        O.GATHER, [np.ones(2, dtype="float32"), np.array([0], dtype="float32")]
    )
    assert not verify_ir(p)[0]
    p, _inputs = operation(
        O.GATHER,
        [np.ones(2, dtype="float32"), np.array([0], dtype="int64")],
        result_dtype=D.FLOAT64,
    )
    assert not verify_ir(p)[0]
    x = Value("x", D.FLOAT64)
    b = builder(x)
    result = b.sqrt(b.reduce_mean(b.mul(x, x), (0,), False))
    b.ret(result)
    assert result.dtype == D.FLOAT64
    assert (
        IRInterpreter(b.program)
        .run({"x": np.array([3, 3], dtype="float64")})
        .return_value
        == 3
    )
