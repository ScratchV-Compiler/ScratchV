"""Bounded-memory absolute-error comparisons, including padded query rows."""
from __future__ import annotations

import math
import numpy as np

from scratchv.verification.numeric_metrics import numeric_metrics, undefined_numeric_metrics

ATOL = 1e-4


def compare_tensor(actual, reference, *, atol=ATOL, chunk_elements=262144):
    if not math.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive")
    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError("chunk_elements must be a positive integer")
    row = {"passed": False, "atol": atol, "rtol": 0,
           "shape": list(actual.shape), "reference_shape": list(reference.shape),
           "dtype": str(actual.dtype), "reference_dtype": str(reference.dtype),
           "elements": int(actual.size), "max_abs": None, "finite": False}
    row.update(undefined_numeric_metrics("comparison is invalid"))
    if actual.shape != reference.shape or actual.dtype != np.float32 or reference.dtype != np.float32:
        row["reason"] = "shape or FP32 dtype mismatch"
        row.update(undefined_numeric_metrics(row["reason"]))
        return row
    if not actual.size:
        row["reason"] = "empty comparison is not numerical evidence"
        row.update(undefined_numeric_metrics(row["reason"]))
        return row
    # nditer handles strided slices without a full-size ravel/copy allocation.
    chunks = np.nditer([actual, reference], flags=["external_loop", "buffered"],
                       op_flags=[["readonly"], ["readonly"]], order="C", buffersize=chunk_elements)
    maximum, offset, worst = -1.0, 0, None
    for a, b in chunks:
        finite = np.isfinite(a) & np.isfinite(b)
        if not finite.all():
            index = offset + int(np.flatnonzero(~finite)[0])
            row.update(reason="nonfinite value", worst_index=list(np.unravel_index(index, actual.shape)))
            row["worst_index"] = [int(x) for x in row["worst_index"]]
            row.update(undefined_numeric_metrics(row["reason"]))
            return row
        difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
        local = int(np.argmax(difference))
        if float(difference[local]) > maximum:
            maximum = float(difference[local])
            worst = (offset + local, float(a[local]), float(b[local]))
        offset += a.size
    row.update(passed=maximum < atol, finite=True, max_abs=maximum,
               worst_index=[int(x) for x in np.unravel_index(worst[0], actual.shape)],
               actual_value=worst[1], reference_value=worst[2])
    row.update(numeric_metrics(actual, reference, block_elements=chunk_elements))
    return row


def compare_positions(actual, reference, valid_length, *, atol=ATOL):
    if actual.ndim != 3 or reference.ndim != 3 or not 1 <= valid_length <= actual.shape[1]:
        raise ValueError("expected [batch, sequence, width] and a valid query length")
    total = compare_tensor(actual, reference, atol=atol)
    total["valid_queries"] = compare_tensor(actual[:, :valid_length], reference[:, :valid_length], atol=atol)
    total["padding_queries"] = (compare_tensor(actual[:, valid_length:], reference[:, valid_length:], atol=atol)
                                if valid_length < actual.shape[1] else None)
    return total
