"""Opt-in, fixed-order NumPy FP32 reference evaluation.

This profile specifies evaluation order and polynomial approximations, not
universally greater mathematical accuracy or bitwise equality to every ORT
build/CPU. No model names, positions, weights, or expected outputs are used.
Default interpreter execution continues to use the native NumPy kernels.

MLAS polynomial constants/evaluation are adapted from ONNX Runtime v1.22.1:
core/mlas/lib/logistic.cpp, compute.cpp, x86_64/TransKernelFma3.S and
x86_64/TransKernelAvx512F.S (including their amd64 Windows equivalents).
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

import os
import re

import numpy as np

from scratchv.ir.types import OpCode
from scratchv.verification.fp32_eigen import mean_last, trig
from scratchv.verification.fp32_fma import fma


_CPU_ENV = "SCRATCHV_FP32_REFERENCE_CPU"
_CPU_STRATEGIES = ("avx2-fma3", "avx512")


def _cpu_features():
    # NumPy checks both hardware capabilities and usable OS vector state.
    # This private diagnostic is optional: an explicit strategy works without it.
    try:
        from numpy._core._multiarray_umath import __cpu_features__
    except ImportError:
        try:
            from numpy.core._multiarray_umath import __cpu_features__
        except ImportError:
            return {}
    return __cpu_features__


def _cpu_strategy(requested=None):
    """Resolve a reference arithmetic policy; never change ORT dispatch.

    Explicit policies also allow reproducing saved arithmetic on another CPU.
    NumPy's disable list does not constrain MLAS, so automatic inference must
    not treat deliberately hidden NumPy capabilities as hardware evidence.
    """
    requested = os.environ.get(_CPU_ENV, "auto") if requested is None else requested
    if requested in _CPU_STRATEGIES:
        return requested
    if requested != "auto":
        raise ValueError(f"{_CPU_ENV} must be auto, avx2-fma3, or avx512")
    disabled = re.split(r"[\s,]+", os.environ.get("NPY_DISABLE_CPU_FEATURES", "").upper())
    if any(feature in ("AVX", "AVX2", "FMA3") or feature.startswith("AVX512")
           for feature in disabled):
        raise ValueError(f"NPY_DISABLE_CPU_FEATURES obscures MLAS capabilities; set {_CPU_ENV} explicitly")
    features = _cpu_features()
    if features.get("AVX2") and features.get("FMA3"):
        return "avx512" if features.get("AVX512F") else "avx2-fma3"
    raise ValueError(f"Cannot infer a supported FP32 CPU strategy; set {_CPU_ENV} explicitly")


def profile(cpu_strategy=None):
    """Return a canonical arithmetic contract, independent of selection method.

    Passing a supported explicit cpu_strategy validates a saved profile on a
    different host without claiming that the local ORT uses that strategy.
    """
    strategy = _cpu_strategy(cpu_strategy)
    lanes = 8 if strategy == "avx2-fma3" else 16
    return {
        "name": "numpy-fp32-reference-v3",
        "dtype": "float32",
        "cpu_strategy": strategy,
        "matmul_k_block": 128,
        "mean": "two-four-lane-last-axis",
        "trig": "non-fused-range-reduction-polynomial",
        "softmax": f"exp-polynomial-{lanes}-lane-sum-reciprocal-multiply",
        "softmax_lanes": lanes,
        "softmax_exp_lower_bound": float(np.float32(
            -88.3762626647949 if lanes == 8 else -103.9720840454)),
        "softmax_exp_scaling": "binary32-exponent-bits" if lanes == 8 else "ldexp",
        "sigmoid": "bounded-rational-polynomial-clamped-to-[0,1]",
        "multiply_add": "binary32-round-to-nearest-even",
        "reference_basis": "ORT 1.22.1 CPU arithmetic; no ORT runtime dependency",
        "scope": "Explicit FP32 reference evaluation; other dtypes unchanged. "
                 "CPU strategy specifies reference arithmetic, not ORT dispatch. "
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


def _exp_nonpositive(x, *, cpu_strategy="avx512"):
    # The bounded polynomial's smallest result rounds to zero; masked -inf
    # from stable subtraction is consequently normalized to probability zero.
    x = np.maximum(x, np.float32(
        -88.3762626647949 if cpu_strategy == "avx2-fma3" else -103.9720840454))
    bias = np.float32(12582912)
    rounded = fma(x, np.float32(1.44269504088896341), bias)
    exponent = rounded - bias
    x = fma(exponent, np.float32(-6.93145752e-1), x)
    x = fma(exponent, np.float32(-1.42860677e-6), x)
    polynomial = np.full_like(x, _EXP[0])
    for coefficient in _EXP[1:]:
        polynomial = fma(polynomial, x, coefficient)
    if cpu_strategy == "avx2-fma3":
        # FMA3 reconstructs 2**m directly in the binary32 exponent field.
        # In particular m=-127 becomes zero, unlike AVX512's VSCALEFPS.
        bits = (rounded.view(np.uint32) << np.uint32(23)) + np.uint32(0x3f800000)
        return polynomial * bits.view(np.float32)
    return np.ldexp(polynomial, exponent.astype(np.int32))


def _sum_lanes(values, lanes):
    sums = np.zeros(values.shape[:-1] + (lanes,), dtype=np.float32)
    for start in range(0, values.shape[-1], lanes):
        block = values[..., start:start + lanes]
        sums[..., :block.shape[-1]] += block
    if lanes == 16:
        sums = sums[..., :8] + sums[..., 8:]
    first = (sums[..., 0] + sums[..., 1]) + (sums[..., 2] + sums[..., 3])
    last = (sums[..., 4] + sums[..., 5]) + (sums[..., 6] + sums[..., 7])
    return (first + last)[..., None]


def _sum_sixteen(values):
    # Retain the named primitive for existing standalone arithmetic diagnostics.
    return _sum_lanes(values, 16)


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
    strategy = _cpu_strategy()
    weights = _exp_nonpositive(reduced, cpu_strategy=strategy)
    result = weights * (np.float32(1) / _sum_lanes(
        weights, 8 if strategy == "avx2-fma3" else 16))
    return np.moveaxis(result, -1, selected)


HANDLERS = {
    OpCode.MATMUL: matmul,
    OpCode.REDUCE_MEAN: reduce_mean,
    OpCode.SOFTMAX: softmax,
    OpCode.SIGMOID: sigmoid,
    OpCode.COS: lambda xs, attrs, dtype: trig(xs[0], sine=False),
    OpCode.SIN: lambda xs, attrs, dtype: trig(xs[0], sine=True),
}
