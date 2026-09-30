"""Direct Program fixtures shared by demos, tests and interpreter benchmarks.

References use small hand-computed arrays or independent scalar Python math.
No frontend, legacy executor or interpreter kernels are used for references.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType as D
from scratchv.ir.types import Program, Value


@dataclass
class InterpreterCase:
    name: str
    program: Program
    inputs: dict[str, np.ndarray]
    initializers: dict[str, np.ndarray]
    expected: np.ndarray
    atol: float = 1e-6
    rtol: float = 1e-6
    seed: int = 0


def builder(*params):
    result = IRBuilder()
    result.new_function("main", list(params))
    result.new_block("entry")
    return result


def matmul_add_softmax():
    x, w, bias = (
        Value("x", shape=(2, 3)),
        Value("W", shape=(3, 2)),
        Value("bias", shape=(2,)),
    )
    b = builder(x)
    b.program.global_values.extend([w, bias])
    b.ret(b.softmax(b.add(b.matmul(x, w), bias)))
    expected = []
    for left, right in ((6, 1), (15, 1)):
        denominator = 1 + math.exp(right - left)
        expected.append([1 / denominator, math.exp(right - left) / denominator])
    return InterpreterCase(
        "matmul_add_softmax",
        b.program,
        {"x": np.array([[1, 2, 3], [4, 5, 6]], dtype="float32")},
        {
            "W": np.array([[1, -1], [2, 0], [0, 1]], dtype="float32"),
            "bias": np.array([1, -1], dtype="float32"),
        },
        np.array(expected, dtype="float32"),
    )


def rmsnorm():
    x, gamma = Value("x", shape=(2, 2)), Value("gamma", shape=(2,))
    b = builder(x)
    b.program.global_values.append(gamma)
    variance = b.reduce_mean(b.mul(x, x), (-1,), True)
    denominator = b.sqrt(b.add(variance, b.make_const(1e-5)))
    b.ret(b.mul(b.div(x, denominator), gamma))
    expected = [
        [
            row[0] * 2 / math.sqrt(sum(v * v for v in row) / 2 + 1e-5),
            row[1] * 0.5 / math.sqrt(sum(v * v for v in row) / 2 + 1e-5),
        ]
        for row in ([1, 2], [3, 4])
    ]
    return InterpreterCase(
        "rmsnorm",
        b.program,
        {"x": np.array([[1, 2], [3, 4]], dtype="float32")},
        {"gamma": np.array([2, 0.5], dtype="float32")},
        np.array(expected, dtype="float32"),
    )


def gather_embedding():
    indices = Value("indices", D.INT64, shape=(2, 2))
    data = Value("embedding", shape=(4, 2))
    b = builder(indices)
    b.program.global_values.append(data)
    b.ret(b.gather(data, indices))
    return InterpreterCase(
        "gather_embedding",
        b.program,
        {"indices": np.array([[2, 0], [-1, 1]], dtype="int64")},
        {"embedding": np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype="float32")},
        np.array([[[5, 6], [1, 2]], [[7, 8], [3, 4]]], dtype="float32"),
    )


def shape_chain():
    x = Value("x", shape=(2, 3))
    b = builder(x)
    selected = b.slice(x, (0,), (3,), (1,), (2,))
    expanded = b.expand(b.unsqueeze(selected, (1,)), (2, 3, 2))
    b.ret(b.concat([expanded, expanded], -1))
    return InterpreterCase(
        "shape_chain",
        b.program,
        {"x": np.array([[1, 2, 3], [4, 5, 6]], dtype="float32")},
        {},
        np.array([[[1, 3, 1, 3]] * 3, [[4, 6, 4, 6]] * 3], dtype="float32"),
    )


def branch():
    condition, x = Value("condition", D.INT32), Value("x", shape=(2,))
    b = builder(condition, x)
    b.br_if(condition, "positive", "negative")
    b.new_block("positive")
    b.ret(b.add(x, b.make_const(1.0)))
    b.new_block("negative")
    b.ret(b.neg(x))
    return InterpreterCase(
        "branch",
        b.program,
        {
            "condition": np.array(1, dtype="int32"),
            "x": np.array([2, -3], dtype="float32"),
        },
        {},
        np.array([3, -2], dtype="float32"),
    )


def loop_sum(start=0, end=5, step=1, nested=False):
    b = builder()
    slot = b.alloca(8, D.INT64)
    b.store(slot, b.make_const(0, D.INT64))
    b.for_loop(start, end, step)
    if nested:
        b.for_loop(0, 2)
    # Independent i64 counter: loop iv stays i32 as required by shared IR.
    previous = b.load(slot)
    b.store(slot, b.add(previous, b.make_const(1, D.INT64)))
    if nested:
        b.endfor()
    b.endfor()
    b.ret(b.load(slot))
    iterations = len(range(start, end, step)) * (2 if nested else 1)
    return InterpreterCase(
        "loop_count", b.program, {}, {}, np.array(iterations, dtype="int64")
    )


def loop_accumulate():
    b = builder()
    slot = b.alloca(4, D.INT32)
    b.store(slot, b.make_const(0, D.INT32))
    iv = b.for_loop(0, 5)
    b.store(slot, b.add(b.load(slot), iv))
    b.endfor()
    b.ret(b.load(slot))
    return InterpreterCase("loop_sum", b.program, {}, {}, np.array(10, dtype="int32"))


def matmul_scaled(size, seed=0):
    """Larger benchmark with an independently computable diagonal product."""
    x, w = Value("x", shape=(size, size)), Value("W", shape=(size, size))
    b = builder(x)
    b.program.global_values.append(w)
    b.ret(b.matmul(x, w))
    rng = np.random.default_rng(seed)
    data = rng.integers(-4, 5, size=(size, size)).astype("float32")
    weights = np.eye(size, dtype="float32") * np.float32(2)
    expected = np.array([[float(v) * 2 for v in row] for row in data], dtype="float32")
    return InterpreterCase(
        f"matmul_{size}", b.program, {"x": data}, {"W": weights}, expected, seed=seed
    )


CASE_FACTORIES = {
    "matmul_add_softmax": matmul_add_softmax,
    "rmsnorm": rmsnorm,
    "gather_embedding": gather_embedding,
    "shape_chain": shape_chain,
    "branch": branch,
    "loop_sum": loop_accumulate,
    "loop_zero": lambda: loop_sum(5, 5),
    "loop_step": lambda: loop_sum(1, 8, 3),
    "loop_nested": lambda: loop_sum(nested=True),
    "matmul_32": lambda: matmul_scaled(32),
    "matmul_128": lambda: matmul_scaled(128),
}

# Tensor workloads are the default benchmark selection. Control-flow fixtures
# remain available to dedicated execution tests and explicit --case selection.
TENSOR_BENCHMARK_CASES = (
    "matmul_add_softmax",
    "rmsnorm",
    "gather_embedding",
    "shape_chain",
    "matmul_32",
    "matmul_128",
)

REFERENCE_METHODS = {
    "matmul_add_softmax": "手算 logits + Python math.exp",
    "rmsnorm": "Python 标量平方求和 + math.sqrt",
    "gather_embedding": "手写索引结果",
    "shape_chain": "手写形状变换结果",
    "branch": "手算分支输出",
    "loop_sum": "手算 0+1+2+3+4",
    "loop_zero": "Python range 计数",
    "loop_step": "Python range 计数",
    "loop_nested": "Python range 计数 × 内层次数",
    "matmul_32": "W=2I，Python 标量逐元素 ×2",
    "matmul_128": "W=2I，Python 标量逐元素 ×2",
}


def make_case(name):
    case = CASE_FACTORIES[name]()
    case.name = name
    return case


def compare(actual, case):
    if (
        actual is None
        or actual.shape != case.expected.shape
        or actual.dtype != case.expected.dtype
    ):
        raise AssertionError(f"{case.name}: output shape/dtype mismatch")
    if not np.isfinite(actual).all() or not np.isfinite(case.expected).all():
        raise AssertionError(f"{case.name}: nonfinite output")
    if np.issubdtype(actual.dtype, np.integer):
        matches = np.array_equal(actual, case.expected)
        differences = np.abs(actual.astype(object) - case.expected.astype(object))
    else:
        differences = np.abs(actual.astype("float64") - case.expected.astype("float64"))
        matches = np.all(
            differences
            <= case.atol + case.rtol * np.abs(case.expected.astype("float64"))
        )
    maximum = float(np.max(differences)) if differences.size else 0.0
    if not matches:
        location = np.unravel_index(int(np.argmax(differences)), differences.shape)
        raise AssertionError(f"{case.name}: max_abs_error={maximum} at {location}")
    return maximum
