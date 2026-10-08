"""Descriptive tensor metrics; these never decide numerical acceptance.

Norms are accumulated in float64 relative to a power-of-two scale. Iteration
is bounded even for noncontiguous arrays and memory-mapped model outputs.
"""
from __future__ import annotations

import math

import numpy as np


def undefined_numeric_metrics(reason):
    """JSON-safe fields for a comparison that has no numerical meaning."""
    return dict(relative_l2=None, cosine_similarity=None,
                relative_l2_reason=reason, cosine_similarity_reason=reason)


class _ScaledSquares:
    def __init__(self):
        self.exponent = 0
        self.squares = 0.0

    def add(self, values, exponent_shift=0):
        maximum = float(np.max(np.abs(values), initial=0.0))
        if maximum == 0:
            return
        _, exponent = math.frexp(maximum)
        normalized = np.ldexp(values, -exponent)
        squares = float(np.sum(normalized * normalized, dtype=np.float64))
        exponent += exponent_shift
        if self.squares == 0:
            self.exponent, self.squares = exponent, squares
        elif exponent > self.exponent:
            self.squares = math.ldexp(self.squares, 2 * (self.exponent - exponent)) + squares
            self.exponent = exponent
        else:
            self.squares += math.ldexp(squares, 2 * (exponent - self.exponent))


def _chunks(actual, expected, block_elements):
    return np.nditer([actual, expected], flags=["external_loop", "buffered"],
                     op_flags=[["readonly"], ["readonly"]], order="C",
                     buffersize=block_elements)


def numeric_metrics(actual, expected, *, block_elements=262144):
    """Return cosine and ||actual-expected||_2 / ||expected||_2.

    Shapes and dtypes must match, with real numeric (at most FP64) data. Empty
    and nonfinite tensors have undefined metrics. Cosine is undefined if either
    norm is zero. A zero reference gives relative L2 zero only when actual is
    also zero; otherwise it is undefined. Unrepresentable relative L2 values
    are reported as null with a reason, never as infinity or an invented zero.
    """
    if type(block_elements) is not int or block_elements < 1:
        raise ValueError("block_elements must be a positive integer")
    if not isinstance(actual, np.ndarray) or not isinstance(expected, np.ndarray):
        return undefined_numeric_metrics("outputs must be NumPy arrays")
    if actual.shape != expected.shape:
        return undefined_numeric_metrics("shape mismatch")
    if actual.dtype != expected.dtype:
        return undefined_numeric_metrics("dtype mismatch")
    if actual.dtype.kind not in "biuf" or (actual.dtype.kind == "f" and actual.dtype.itemsize > 8):
        return undefined_numeric_metrics("metrics require real numeric dtypes up to float64")
    if not actual.size:
        return undefined_numeric_metrics("empty tensor")

    actual_norm, expected_norm, difference_norm = (_ScaledSquares() for _ in range(3))
    identical = True
    for a, b in _chunks(actual, expected, block_elements):
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return undefined_numeric_metrics("nonfinite output or reference")
        equal = bool(np.array_equal(a, b))
        identical = identical and equal
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        actual_norm.add(a64)
        expected_norm.add(b64)
        if equal:
            continue
        if actual.dtype.kind in "biu":
            # Subtract before rounding to FP64: int64/uint64 neighbours above
            # 2**53 must retain their nonzero difference. Copies stay bounded.
            difference = (a.astype(object) - b.astype(object)).astype(np.float64)
            difference_norm.add(difference)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                difference = a64 - b64
            overflow = ~np.isfinite(difference)
            if overflow.any():
                difference_norm.add(difference[~overflow])
                # Only overflowing differences are halved, so unrelated tiny
                # values are not lost to a premature division by two.
                difference_norm.add(a64[overflow] * 0.5 - b64[overflow] * 0.5, 1)
            else:
                difference_norm.add(difference)

    result = dict(relative_l2=None, cosine_similarity=None,
                  relative_l2_reason=None, cosine_similarity_reason=None)
    if expected_norm.squares == 0:
        if actual_norm.squares == 0:
            result["relative_l2"] = 0.0
        else:
            result["relative_l2_reason"] = "zero reference norm with nonzero actual norm"
    elif difference_norm.squares == 0:
        result["relative_l2"] = 0.0
    else:
        factor = math.sqrt(difference_norm.squares / expected_norm.squares)
        try:
            relative = math.ldexp(factor, difference_norm.exponent - expected_norm.exponent)
        except OverflowError:
            relative = math.inf
        if not math.isfinite(relative):
            result["relative_l2_reason"] = "relative L2 exceeds float64 range"
        elif relative == 0:
            result["relative_l2_reason"] = "nonzero relative L2 is below float64 range"
        else:
            result["relative_l2"] = relative

    if actual_norm.squares == 0 or expected_norm.squares == 0:
        result["cosine_similarity_reason"] = "cosine is undefined for a zero norm"
    elif identical:
        result["cosine_similarity"] = 1.0
    else:
        # Scale each vector independently; squaring unscaled FP64 values can
        # overflow, and using one shared scale can erase the smaller vector.
        dot, correction = 0.0, 0.0
        for a, b in _chunks(actual, expected, block_elements):
            a64 = np.ldexp(a.astype(np.float64), -actual_norm.exponent)
            b64 = np.ldexp(b.astype(np.float64), -expected_norm.exponent)
            term = float(np.sum(a64 * b64, dtype=np.float64))
            total = dot + term
            correction += ((dot - total) + term if abs(dot) >= abs(term)
                           else (term - total) + dot)
            dot = total
        cosine = ((dot + correction) / math.sqrt(actual_norm.squares)
                  / math.sqrt(expected_norm.squares))
        # Cauchy-Schwarz bounds the exact result; trim only FP64 roundoff.
        result["cosine_similarity"] = min(1.0, max(-1.0, cosine))
    return result
