"""Opt-in, fixed-order NumPy FP32 reference evaluation.

This profile specifies evaluation order and polynomial approximations, not
universally greater mathematical accuracy or bitwise equality to every ORT
build/CPU. No model names, positions, weights, or expected outputs are used.
Default interpreter execution continues to use the native NumPy kernels.

MLAS polynomial constants/evaluation are adapted from ONNX Runtime v1.22.1:
core/mlas/lib/logistic.cpp, compute.cpp, amd64/TransKernelAvx512F.asm.
Copyright (c) Microsoft Corporation. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import numpy as np

from scratchv.ir.types import OpCode
from scratchv.verification.fp32_eigen import mean_last, trig
from scratchv.verification.fp32_fma import fma


def profile():
    """Versioned arithmetic policy, recorded separately from graph optimization."""
    return {
        "name": "numpy-fp32-reference-v2",
        "dtype": "float32",
        "matmul_k_block": 128,
        "mean": "two-four-lane-last-axis",
        "trig": "non-fused-range-reduction-polynomial",
        "softmax": "exp-polynomial-16-lane-sum-reciprocal-multiply",
        "sigmoid": "bounded-rational-polynomial-clamped-to-[0,1]",
        "multiply_add": "binary32-round-to-nearest-even",
        "reference_basis": "ORT 1.22.1 CPU arithmetic; no ORT runtime dependency",
        "scope": "Explicit FP32 reference evaluation; other dtypes unchanged. "
                 "Not a promise of bitwise matching every CPU or ORT version.",
    }


def matmul(xs, attrs, dtype):
    from scratchv.verification.ir_numpy_ops import OpError
    a, b = xs
    if "m" in attrs:
        m, n, k = (attrs[key] for key in ("m", "n", "k"))
        if a.ndim == b.ndim == 1:
            a, b = a.reshape(m, k), b.reshape(k, n)
        elif a.shape != (m, k) or b.shape != (k, n):
            raise OpError("ShapeError", "MATMUL m/n/k disagree with tensor shapes")
    # Preserve NumPy's vector promotion, empty reductions and shape errors.
    if a.ndim < 2 or b.ndim < 2 or a.shape[-1] <= 128:
        return np.matmul(a, b)
    if a.shape[-1] != b.shape[-2]:
        raise OpError("ShapeError", "MATMUL reduction dimensions differ")
    shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-1])
    result = np.zeros(shape, dtype=np.float32)
    for start in range(0, a.shape[-1], 128):
        result += np.matmul(a[..., start:start + 128], b[..., start:start + 128, :])
    return result


def reduce_mean(xs, attrs, dtype):
    from scratchv.verification.ir_numpy_ops import OpError, axes, flag, reduce_mean as native
    x = xs[0]
    requested = attrs.get("axes")
    selected = tuple(range(x.ndim)) if not requested else axes(requested, x.ndim)
    if any(x.shape[a] == 0 for a in selected):
        raise OpError("NumericError", "cannot reduce an empty dimension")
    if x.ndim and selected == (x.ndim - 1,):
        return mean_last(x, keepdims=flag(attrs.get("keepdims", True), "keepdims"))
    return native(xs, attrs, dtype)


_SIGMOID_P = tuple(np.float32(v) for v in (
    4.37031012579801e-11, 1.15627324459942e-7, 6.08574864600143e-5,
    8.51377133304701e-3, 2.48287947061529e-1))
_SIGMOID_Q = tuple(np.float32(v) for v in (
    6.10247389755681e-13, 5.76102136993427e-9, 6.29106785017040e-6,
    1.70198817374094e-3, 1.16817656904453e-1, 9.93151921023180e-1))
_EXP = tuple(np.float32(float.fromhex(v)) for v in (
    "0x1.694000p-10", "0x1.125edcp-7", "0x1.555b5ap-5", "0x1.555450p-3",
    "0x1.fffff6p-2", "0x1.000000p+0", "0x1.000000p+0"))


def sigmoid(xs, attrs, dtype):
    x = np.clip(xs[0], np.float32(-18), np.float32(18))
    squared = x * x
    numerator = np.full_like(x, _SIGMOID_P[0])
    denominator = np.full_like(x, _SIGMOID_Q[0])
    for coefficient in _SIGMOID_P[1:]:
        numerator = fma(numerator, squared, coefficient)
    for coefficient in _SIGMOID_Q[1:]:
        denominator = fma(denominator, squared, coefficient)
    # Rational-approximation roundoff can cross either saturation endpoint.
    return np.clip((numerator * x) / denominator + np.float32(0.5),
                   np.float32(0), np.float32(1))


def _exp_nonpositive(x):
    # The bounded polynomial's smallest result rounds to zero; masked -inf
    # from stable subtraction is consequently normalized to probability zero.
    x = np.maximum(x, np.float32(-103.9720840454))
    bias = np.float32(12582912)
    rounded = fma(x, np.float32(1.44269504088896341), bias)
    exponent = rounded - bias
    x = fma(exponent, np.float32(-6.93145752e-1), x)
    x = fma(exponent, np.float32(-1.42860677e-6), x)
    polynomial = np.full_like(x, _EXP[0])
    for coefficient in _EXP[1:]:
        polynomial = fma(polynomial, x, coefficient)
    return np.ldexp(polynomial, exponent.astype(np.int32))


def _sum_sixteen(values):
    sums = np.zeros(values.shape[:-1] + (16,), dtype=np.float32)
    for start in range(0, values.shape[-1], 16):
        block = values[..., start:start + 16]
        sums[..., :block.shape[-1]] += block
    sums = sums[..., :8] + sums[..., 8:]
    first = (sums[..., 0] + sums[..., 1]) + (sums[..., 2] + sums[..., 3])
    last = (sums[..., 4] + sums[..., 5]) + (sums[..., 6] + sums[..., 7])
    return (first + last)[..., None]


def softmax(xs, attrs, dtype):
    from scratchv.verification.ir_numpy_ops import OpError, axis
    x = xs[0]
    selected = axis(attrs.get("axis", -1), x.ndim)
    if x.shape[selected] == 0:
        raise OpError("NumericError", "SOFTMAX cannot normalize an empty axis")
    x = np.moveaxis(x, selected, -1)
    maximum = np.max(x, axis=-1, keepdims=True)
    if not np.isfinite(maximum).all():
        raise OpError("NumericError", "SOFTMAX row is entirely masked or nonfinite")
    with np.errstate(over="ignore"):
        reduced = x - maximum
    weights = _exp_nonpositive(reduced)
    result = weights * (np.float32(1) / _sum_sixteen(weights))
    return np.moveaxis(result, -1, selected)


HANDLERS = {
    OpCode.MATMUL: matmul,
    OpCode.REDUCE_MEAN: reduce_mean,
    OpCode.SOFTMAX: softmax,
    OpCode.SIGMOID: sigmoid,
    OpCode.COS: lambda xs, attrs, dtype: trig(xs[0], sine=False),
    OpCode.SIN: lambda xs, attrs, dtype: trig(xs[0], sine=True),
}
