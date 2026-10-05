# SPDX-License-Identifier: MPL-2.0
# Adapted from Eigen (https://gitlab.com/libeigen/eigen), commit
# 1d8b82b0740839c0de7f1242a3585e3390ff5f33:
#   Eigen/src/Core/Redux.h
#   Eigen/src/Core/arch/SSE/PacketMath.h
#   Eigen/src/Core/arch/Default/GenericPacketMathFunctions.h
# Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
# Copyright (C) 2007 Julien Pommier
# Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
# Copyright (C) 2009-2019 Gael Guennebaud <gael.guennebaud@inria.fr>
# Copyright (C) 2014 Pedro Gonnet (pedro.gonnet@gmail.com)
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# The full license is also distributed in LICENSES/Eigen-MPL-2.0.txt.
"""Deterministic FP32 reductions and trigonometric approximations.

This NumPy adaptation preserves Eigen's SSE reduction order and its non-FMA
``psincos_float`` polynomial. All floating-point computation remains FP32.
It makes no promise of bitwise agreement with every platform's math library.
The reduction uses canonical contiguous row alignment, independent of the
input's memory address or strides. Trigonometry uses NumPy outside a shared
conservative interval for the non-FMA approximation.
"""

import numpy as np


def _fp32_array(values):
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float32):
        raise TypeError("FP32 math requires a float32 array")
    return array


def _sum_last(array, aligned_start=0):
    """Eigen's two Packet4f accumulators, scalar prefix/tail, and predux."""
    width = array.shape[-1]
    if width - aligned_start < 4:
        result = array[..., 0].copy()
        for index in range(1, width):
            result = result + array[..., index]
        return result

    packet_end = aligned_start + ((width - aligned_start) // 4) * 4
    pair_end = aligned_start + ((width - aligned_start) // 8) * 8
    first = array[..., aligned_start : aligned_start + 4].copy()
    if packet_end - aligned_start > 4:
        second = array[..., aligned_start + 4 : aligned_start + 8].copy()
        for index in range(aligned_start + 8, pair_end, 8):
            first = first + array[..., index : index + 4]
            second = second + array[..., index + 4 : index + 8]
        first = first + second
        if packet_end > pair_end:
            first = first + array[..., pair_end : pair_end + 4]

    result = (first[..., 0] + first[..., 2]) + (first[..., 1] + first[..., 3])
    for index in range(aligned_start):
        result = result + array[..., index]
    for index in range(packet_end, width):
        result = result + array[..., index]
    return result


def mean_last(values, keepdims=True):
    """Reduce a nonempty last dimension using an explicit FP32 sum order.

    Leading dimensions may be empty. Scalar inputs and an empty reduction
    dimension are rejected rather than inventing a mean for an empty set.
    Inputs, including read-only and strided arrays, are never modified.
    """
    array = _fp32_array(values)
    if array.ndim == 0 or array.shape[-1] == 0:
        raise ValueError("mean_last requires a nonempty last dimension")
    if not isinstance(keepdims, (bool, np.bool_)):
        raise TypeError("keepdims must be boolean")

    width = array.shape[-1]
    if width % 4 == 0 or width < 4:
        sums = _sum_last(array)
    else:
        # Model a 16-byte-aligned first row and contiguous successive rows.
        # Shape alone determines the prefix; array pointer alignment does not.
        rows = array.reshape(-1, width)
        row_starts = (-np.arange(rows.shape[0]) * width) % 4
        sums = np.empty(rows.shape[0], dtype=np.float32)
        for start in range(4):
            selected = np.flatnonzero(row_starts == start)
            if selected.size:
                sums[selected] = _sum_last(rows[selected], start)

    result = np.asarray(sums / np.float32(width)).reshape(array.shape[:-1] + (1,))
    return result if keepdims else result.squeeze(axis=-1)


def trig(values, sine: bool):
    """FP32 sine or cosine with Eigen's non-FMA range reduction polynomial.

    The shared interval ``abs(x) < 18000`` is below both upstream non-FMA
    cutoffs. Larger arguments and nonfinite values retain NumPy semantics.
    This approximation can differ from NumPy by a few ULP; it is not a claim
    of universally improved accuracy. Scalar and empty arrays are supported.
    """
    array = _fp32_array(values)
    if not isinstance(sine, (bool, np.bool_)):
        raise TypeError("sine must be boolean")
    selected = np.isfinite(array) & (np.abs(array) < np.float32(18000))
    result = np.array(np.sin(array) if sine else np.cos(array), dtype=np.float32, copy=True)
    reduced = np.abs(array[selected])
    rounded = reduced * np.float32(0.636619746685028076171875)
    rounded = rounded + np.float32(12582912)
    quadrant = rounded.view(np.uint32)
    multiple = rounded - np.float32(12582912)
    for coefficient in (
        -1.5703125,
        -0.000483989715576171875,
        1.62865035235881805419921875e-7,
        5.5644315544167710640977020375430583953857421875e-11,
    ):
        reduced = multiple * np.float32(coefficient) + reduced

    squared = reduced * reduced
    cosine = np.full_like(reduced, np.float32(2.4372266125283204019069671630859375e-5))
    for coefficient in (
        -0.00138865201734006404876708984375,
        0.041666619479656219482421875,
        -0.5,
        1.0,
    ):
        cosine = cosine * squared + np.float32(coefficient)

    sinpoly = np.full_like(
        reduced,
        np.float32(-0.0001959234114083702898469196984621021329076029360294342041015625),
    )
    sinpoly = sinpoly * squared + np.float32(
        0.00833268736556168516937947998712843400426208972930908203125
    )
    sinpoly = sinpoly * squared + np.float32(
        -0.166666620398229825550373561782180331647396087646484375
    )
    sinpoly = sinpoly * squared
    sinpoly = sinpoly * reduced + reduced

    even = (quadrant & np.uint32(1)) == 0
    if sine:
        output = np.where(even, sinpoly, cosine)
        sign = (array[selected].view(np.uint32) ^ (quadrant << np.uint32(30))) & np.uint32(
            0x80000000
        )
    else:
        output = np.where(even, cosine, sinpoly)
        sign = ((quadrant + np.uint32(1)) << np.uint32(30)) & np.uint32(0x80000000)
    result[selected] = (output.view(np.uint32) ^ sign).view(np.float32)
    return result
