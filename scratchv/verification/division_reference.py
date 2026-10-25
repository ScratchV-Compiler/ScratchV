"""Typed value-only division references; not an fflags/rounding oracle."""

from __future__ import annotations

import numpy as np


def divide(lhs, rhs) -> np.ndarray:
    """Preserve i32 DIV semantics or the operands' numpy floating precision.

    RV32 DIV truncates toward zero, returns -1 for a zero divisor and wraps
    INT32_MIN/-1 to INT32_MIN. Do not use float conversion to compute its
    quotient. Floating division uses the original denominator, canonicalizes
    NaN values, and does not claim to model target exception flags or rm.
    """
    left, right = np.broadcast_arrays(np.asarray(lhs), np.asarray(rhs))
    if left.dtype == np.int32 and right.dtype == np.int32:
        def scalar_div(a, b):
            dividend, divisor = int(a), int(b)
            if divisor == 0:
                return -1
            if dividend == -(1 << 31) and divisor == -1:
                return -(1 << 31)
            quotient = abs(dividend) // abs(divisor)
            return -quotient if (dividend < 0) != (divisor < 0) else quotient

        result = np.vectorize(scalar_div, otypes=[np.int64])(left, right)
        return np.asarray(result, dtype=np.int32)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        result = np.divide(left, right)
    if np.issubdtype(result.dtype, np.floating):
        result = np.where(np.isnan(result), np.asarray(np.nan, dtype=result.dtype), result)
    return np.asarray(result)
