"""Strict NumPy kernels for shared IR, independent of frontend and execution state.

Axes use Python-style negative indexing. REDUCE_MEAN with omitted/empty axes
reduces all dimensions. RESHAPE zeros copy input dimensions (allowzero=False).
SLICE uses clamped Python slice bounds; UNSQUEEZE axes refer to the output rank.
EXPAND computes the broadcast shape of the input and the requested shape.
GELU uses the tanh approximation. CONV is NCHW, group=1, dilation=1; MAXPOOL
accepts NCHW/CHW with no padding. Unsupported semantic attributes are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from scratchv.ir.types import DataType, Instruction, OpCode

DTYPES = {
    DataType.FLOAT32: np.dtype("float32"),
    DataType.FLOAT64: np.dtype("float64"),
    DataType.INT32: np.dtype("int32"),
    DataType.INT64: np.dtype("int64"),
}


class OpError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Kernel:
    handler: Callable
    attributes: frozenset[str]


KERNELS: dict[OpCode, Kernel] = {}


def kernel(opcode, *attributes):
    def register(handler):
        KERNELS[opcode] = Kernel(handler, frozenset(attributes))
        return handler

    return register


def integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise OpError("AttributeError", f"{name} must be an integer")
    return int(value)


def integers(value, name):
    if not isinstance(value, (tuple, list)):
        raise OpError("AttributeError", f"{name} must be an integer sequence")
    return tuple(integer(v, name) for v in value)


def axis(value, rank):
    value = integer(value, "axis")
    if not -rank <= value < rank:
        raise OpError("ShapeError", f"axis {value} outside rank {rank}")
    return value % rank


def axes(values, rank):
    normalized = tuple(axis(a, rank) for a in values)
    if len(set(normalized)) != len(normalized):
        raise OpError("AttributeError", "axes must not contain duplicates")
    return normalized


def flag(value, name):
    if not isinstance(value, (bool, int, np.bool_, np.integer)) or value not in (0, 1):
        raise OpError("AttributeError", f"{name} must be boolean or 0/1")
    return bool(value)


def check_instruction(instr: Instruction):
    """Preflight attributes without evaluating tensors, including dead branches."""
    spec = KERNELS.get(instr.opcode)
    if spec is None:
        raise OpError("UnsupportedOpcode", f"unsupported opcode: {instr.opcode.value}")
    unknown = set(instr.attrs) - spec.attributes
    if unknown:
        raise OpError(
            "UnsupportedAttribute", f"unsupported attributes: {sorted(unknown)}"
        )
    required = {
        OpCode.RESHAPE: ("shape",),
        OpCode.CONCAT: ("axis",),
        OpCode.SLICE: ("starts", "ends"),
        OpCode.UNSQUEEZE: ("axes",),
        OpCode.EXPAND: ("shape",),
        OpCode.LOAD_CONST: ("value",),
        OpCode.DOT: ("length",),
        OpCode.MAXPOOL: ("kernel", "stride"),
    }.get(instr.opcode, ())
    for key in required:
        if key not in instr.attrs:
            raise OpError("AttributeError", f"missing attribute: {key}")
    for key in ("shape", "axes", "perm", "starts", "ends", "steps"):
        if key in instr.attrs:
            integers(instr.attrs[key], key)
    for key in (
        "axis",
        "m",
        "n",
        "k",
        "length",
        "kernel",
        "kernel_size",
        "stride",
        "padding",
        "out_channels",
    ):
        if key in instr.attrs:
            integer(instr.attrs[key], key)
    for key in ("keepdims", "trans_a", "trans_b"):
        if key in instr.attrs:
            flag(instr.attrs[key], key)
    if instr.opcode == OpCode.SLICE:
        count = len(instr.attrs["starts"])
        if len(instr.attrs["ends"]) != count or any(
            key in instr.attrs and len(instr.attrs[key]) != count
            for key in ("axes", "steps")
        ):
            raise OpError("AttributeError", "SLICE parameter lengths must match")
        if "steps" in instr.attrs and 0 in instr.attrs["steps"]:
            raise OpError("AttributeError", "SLICE step cannot be zero")
    if instr.opcode == OpCode.MATMUL:
        supplied = set(instr.attrs) & {"m", "n", "k"}
        if supplied and supplied != {"m", "n", "k"}:
            raise OpError("AttributeError", "MATMUL m/n/k must be supplied together")
        if any(instr.attrs[key] < 1 for key in supplied):
            raise OpError("AttributeError", "MATMUL m/n/k must be positive")
    if (
        instr.opcode in (OpCode.UNSQUEEZE, OpCode.REDUCE_MEAN)
        and "axes" in instr.attrs
        and len(set(instr.attrs["axes"])) != len(instr.attrs["axes"])
    ):
        raise OpError("AttributeError", "duplicate axes")
    if instr.opcode in (OpCode.EXPAND, OpCode.RESHAPE):
        shape = instr.attrs["shape"]
        if instr.opcode == OpCode.EXPAND and any(d < 0 for d in shape):
            raise OpError("AttributeError", "EXPAND dimensions must be nonnegative")
        if instr.opcode == OpCode.RESHAPE and (
            any(d < -1 for d in shape) or shape.count(-1) > 1
        ):
            raise OpError(
                "AttributeError",
                "RESHAPE permits at most one -1 and no smaller dimensions",
            )
    if instr.opcode in (
        OpCode.CONV,
        OpCode.GEMM,
        OpCode.MAXPOOL,
        OpCode.GELU,
        OpCode.SIGMOID,
        OpCode.SOFTMAX,
        OpCode.EXP,
        OpCode.SQRT,
        OpCode.REDUCE_MEAN,
    ) and instr.dest.dtype not in (DataType.FLOAT32, DataType.FLOAT64):
        raise OpError(
            "UnsupportedDType", f"{instr.opcode.value} requires floating data"
        )
    for key in ("alpha", "beta"):
        if key in instr.attrs and (
            isinstance(instr.attrs[key], bool)
            or not isinstance(instr.attrs[key], (int, float))
            or not np.isfinite(instr.attrs[key])
        ):
            raise OpError("AttributeError", f"{key} must be finite numeric")


def compute(instr, operands):
    with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
        result = np.asarray(
            KERNELS[instr.opcode].handler(
                operands, instr.attrs, DTYPES[instr.dest.dtype]
            )
        )
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise OpError("NumericError", "operation produced nonfinite output")
    return result


@kernel(OpCode.LOAD_CONST, "value")
def load_const(xs, attrs, dtype):
    return np.asarray(attrs["value"], dtype=dtype)


def binary(opcode, function):
    @kernel(opcode)
    def run(xs, attrs, dtype):
        return function(xs[0], xs[1])

    return run


binary(OpCode.ADD, np.add)
binary(OpCode.SUB, np.subtract)
binary(OpCode.MUL, np.multiply)


@kernel(OpCode.DIV)
def divide(xs, attrs, dtype):
    a, b = np.broadcast_arrays(*xs)
    if np.any(b == 0):
        raise OpError("NumericError", "division by zero")
    if np.issubdtype(dtype, np.integer):
        minimum = np.iinfo(dtype).min
        if np.any((a == minimum) & (b == -1)):
            raise OpError("NumericError", "integer division overflow")
        # Object integers retain all i64 bits; // on signed arrays rounds down.
        magnitude = np.abs(a.astype(object)) // np.abs(b.astype(object))
        return np.asarray(
            np.where((a < 0) ^ (b < 0), -magnitude, magnitude), dtype=dtype
        )
    return np.divide(a, b)


@kernel(OpCode.NEG)
def negate(xs, attrs, dtype):
    return np.negative(xs[0])


@kernel(OpCode.EXP)
def exponent(xs, attrs, dtype):
    return np.exp(xs[0])


@kernel(OpCode.SQRT)
def square_root(xs, attrs, dtype):
    if np.any(xs[0] < 0):
        raise OpError("NumericError", "SQRT requires nonnegative data")
    return np.sqrt(xs[0])


@kernel(OpCode.REDUCE_MEAN, "axes", "keepdims")
def reduce_mean(xs, attrs, dtype):
    x = xs[0]
    selected = attrs.get("axes")
    selected = tuple(range(x.ndim)) if not selected else axes(selected, x.ndim)
    if any(x.shape[a] == 0 for a in selected):
        raise OpError("NumericError", "cannot reduce an empty dimension")
    return np.mean(
        x,
        axis=selected,
        keepdims=flag(attrs.get("keepdims", True), "keepdims"),
        dtype=dtype,
    )


@kernel(OpCode.MATMUL, "m", "n", "k")
def matmul(xs, attrs, dtype):
    a, b = xs
    if "m" in attrs:
        m, n, k = (attrs[key] for key in ("m", "n", "k"))
        if a.ndim == b.ndim == 1:
            a, b = a.reshape(m, k), b.reshape(k, n)
        elif a.shape != (m, k) or b.shape != (k, n):
            raise OpError("ShapeError", "MATMUL m/n/k disagree with tensor shapes")
    return np.matmul(a, b)


@kernel(OpCode.DOT, "length")
def dot(xs, attrs, dtype):
    if any(x.ndim != 1 or x.size != attrs["length"] for x in xs):
        raise OpError("ShapeError", "DOT requires vectors matching length")
    return np.dot(*xs)


@kernel(OpCode.RESHAPE, "shape")
def reshape(xs, attrs, dtype):
    x = xs[0]
    shape = []
    for i, dimension in enumerate(attrs["shape"]):
        if dimension == 0:
            if i >= x.ndim:
                raise OpError(
                    "ShapeError", "RESHAPE zero has no corresponding input axis"
                )
            dimension = x.shape[i]
        shape.append(dimension)
    return np.reshape(x, tuple(shape))


@kernel(OpCode.TRANSPOSE, "perm")
def transpose(xs, attrs, dtype):
    x = xs[0]
    perm = attrs.get("perm", tuple(reversed(range(x.ndim))))
    if len(perm) != x.ndim or set(perm) != set(range(x.ndim)):
        raise OpError(
            "AttributeError", "perm must be a complete nonnegative axis permutation"
        )
    return np.transpose(x, perm)


@kernel(OpCode.CONCAT, "axis")
def concat(xs, attrs, dtype):
    return np.concatenate(xs, axis=axis(attrs["axis"], xs[0].ndim))


@kernel(OpCode.GATHER, "axis")
def gather(xs, attrs, dtype):
    data, indices = xs
    selected = axis(attrs.get("axis", 0), data.ndim)
    dimension = data.shape[selected]
    if np.any(indices < -dimension) or np.any(indices >= dimension):
        raise OpError("IndexError", "GATHER index out of bounds")
    return np.take(data, indices, axis=selected)


@kernel(OpCode.SLICE, "starts", "ends", "axes", "steps")
def slice_tensor(xs, attrs, dtype):
    x = xs[0]
    count = len(attrs["starts"])
    selected = axes(attrs.get("axes", tuple(range(count))), x.ndim)
    steps = attrs.get("steps", (1,) * count)
    slices = [slice(None)] * x.ndim
    for a, start, end, step in zip(selected, attrs["starts"], attrs["ends"], steps):
        slices[a] = slice(start, end, step)
    return x[tuple(slices)]


@kernel(OpCode.UNSQUEEZE, "axes")
def unsqueeze(xs, attrs, dtype):
    selected = axes(attrs["axes"], xs[0].ndim + len(attrs["axes"]))
    return np.expand_dims(xs[0], axis=selected)


@kernel(OpCode.EXPAND, "shape")
def expand(xs, attrs, dtype):
    shape = np.broadcast_shapes(xs[0].shape, tuple(attrs["shape"]))
    return np.broadcast_to(xs[0], shape)


@kernel(OpCode.RELU)
def relu(xs, attrs, dtype):
    return np.maximum(xs[0], np.asarray(0, dtype=dtype))


@kernel(OpCode.SIGMOID)
def sigmoid(xs, attrs, dtype):
    x = xs[0]
    result = np.empty_like(x)
    positive = x >= 0
    one = np.asarray(1, dtype=dtype)
    result[positive] = one / (one + np.exp(-x[positive]))
    z = np.exp(x[~positive])
    result[~positive] = z / (one + z)
    return result


@kernel(OpCode.GELU)
def gelu(xs, attrs, dtype):
    x = xs[0]
    half, one, factor, cubic = (
        np.asarray(v, dtype=dtype) for v in (0.5, 1, np.sqrt(2 / np.pi), 0.044715)
    )
    return half * x * (one + np.tanh(factor * (x + cubic * x**3)))


@kernel(OpCode.SOFTMAX, "axis")
def softmax(xs, attrs, dtype):
    x = xs[0]
    selected = axis(attrs.get("axis", -1), x.ndim)
    if x.shape[selected] == 0:
        raise OpError("NumericError", "SOFTMAX cannot normalize an empty axis")
    maximum = np.max(x, axis=selected, keepdims=True)
    if not np.isfinite(maximum).all():
        raise OpError("NumericError", "SOFTMAX row is entirely masked")
    weights = np.exp(x - maximum)
    return weights / np.sum(weights, axis=selected, keepdims=True, dtype=dtype)


@kernel(OpCode.GEMM, "trans_a", "trans_b", "alpha", "beta")
def gemm(xs, attrs, dtype):
    a, b, bias = xs
    if a.ndim != 2 or b.ndim != 2:
        raise OpError("ShapeError", "GEMM inputs A/B must be matrices")
    if attrs.get("trans_a", False):
        a = a.T
    if attrs.get("trans_b", False):
        b = b.T
    product = np.matmul(a, b)
    bias = np.broadcast_to(bias, product.shape)
    return (
        np.asarray(attrs.get("alpha", 1), dtype=dtype) * product
        + np.asarray(attrs.get("beta", 1), dtype=dtype) * bias
    )


@kernel(OpCode.CONV, "out_channels", "kernel_size", "stride", "padding")
def conv(xs, attrs, dtype):
    x, w, bias = xs
    if x.ndim != 4 or w.ndim != 4 or x.shape[1] != w.shape[1]:
        raise OpError(
            "ShapeError", "CONV requires compatible NCHW data and OIHW weights"
        )
    channels, _, height, width = w.shape
    kernel_size = attrs.get("kernel_size", 3)
    stride, padding = attrs.get("stride", 1), attrs.get("padding", 1)
    if height != width or height != kernel_size or stride < 1 or padding < 0:
        raise OpError("AttributeError", "invalid CONV square kernel/stride/padding")
    if attrs.get("out_channels", channels) != channels or bias.shape != (channels,):
        raise OpError(
            "ShapeError", "CONV output channels or bias disagree with weights"
        )
    out_h = (x.shape[2] + 2 * padding - height) // stride + 1
    out_w = (x.shape[3] + 2 * padding - width) // stride + 1
    if out_h < 1 or out_w < 1:
        raise OpError("ShapeError", "CONV kernel exceeds input")
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    output = np.empty((x.shape[0], channels, out_h, out_w), dtype=dtype)
    for i in range(out_h):
        for j in range(out_w):
            patch = padded[
                :, :, i * stride : i * stride + height, j * stride : j * stride + width
            ]
            output[:, :, i, j] = np.einsum("nchw,ochw->no", patch, w) + bias
    return output


@kernel(OpCode.MAXPOOL, "kernel", "stride")
def maxpool(xs, attrs, dtype):
    x = xs[0]
    if x.ndim not in (3, 4):
        raise OpError("ShapeError", "MAXPOOL requires CHW or NCHW")
    size, stride = attrs["kernel"], attrs["stride"]
    if size < 1 or stride < 1:
        raise OpError("AttributeError", "MAXPOOL kernel/stride must be positive")
    height, width = ((dimension - size) // stride + 1 for dimension in x.shape[-2:])
    if height < 1 or width < 1:
        raise OpError("ShapeError", "MAXPOOL kernel exceeds input")
    output = np.empty(x.shape[:-2] + (height, width), dtype=dtype)
    for i in range(height):
        for j in range(width):
            output[..., i, j] = np.max(
                x[..., i * stride : i * stride + size, j * stride : j * stride + size],
                axis=(-2, -1),
            )
    return output
