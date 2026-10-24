"""Value-only minimumNumber/maximumNumber references for Topic 28.

These numpy helpers preserve binary32/binary64 dtype, select the number when
only one input is NaN, canonicalize two NaNs and order -0 below +0. They do
not model signaling-NaN exceptions, fflags or non-default rounding modes;
Clang/LLD/QEMU execution remains the binary/exception oracle.
"""

from __future__ import annotations

import numpy as np


def _min_max_number(lhs, rhs, *, maximum: bool) -> np.ndarray:
    left, right = np.broadcast_arrays(np.asarray(lhs), np.asarray(rhs))
    dtype = np.result_type(left.dtype, right.dtype)
    if not np.issubdtype(dtype, np.floating):
        operation = np.maximum if maximum else np.minimum
        return np.asarray(operation(left, right))
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"Unsupported minimumNumber/maximumNumber dtype: {dtype}")
    left, right = left.astype(dtype, copy=False), right.astype(dtype, copy=False)
    operation = np.fmax if maximum else np.fmin
    result = operation(left, right)
    both_nan = np.isnan(left) & np.isnan(right)
    result = np.where(both_nan, np.asarray(np.nan, dtype=dtype), result)
    both_zero = (left == 0) & (right == 0)
    negative_zero = (
        np.signbit(left) & np.signbit(right) if maximum
        else np.signbit(left) | np.signbit(right)
    )
    zero = np.copysign(np.zeros_like(result), np.where(negative_zero, -1, 1))
    return np.asarray(np.where(both_zero, zero, result), dtype=dtype)


def minimum_number(lhs, rhs) -> np.ndarray:
    """Return the smaller number, with IEEE/RISC-V NaN and signed-zero rules."""
    return _min_max_number(lhs, rhs, maximum=False)


def maximum_number(lhs, rhs) -> np.ndarray:
    """Return the larger number, with IEEE/RISC-V NaN and signed-zero rules."""
    return _min_max_number(lhs, rhs, maximum=True)
