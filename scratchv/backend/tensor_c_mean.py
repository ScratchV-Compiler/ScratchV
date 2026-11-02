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
"""C source for the explicitly selected FP32 last-axis Mean order.

This portable scalar C follows the existing FP32 Eigen reference's two four-lane
accumulators, canonical row prefix, pairwise lane fold and tail. It changes
floating-point summation order; it is not a host SIMD or hardware requirement.
Finite input/output checks remain the tensor backend's responsibility.
"""

TWO_FOUR_LANE_LAST_AXIS_C = r'''
/*
 * SPDX-License-Identifier: MPL-2.0
 * Adapted from Eigen (https://gitlab.com/libeigen/eigen), commit
 * 1d8b82b0740839c0de7f1242a3585e3390ff5f33:
 *   Eigen/src/Core/Redux.h
 *   Eigen/src/Core/arch/SSE/PacketMath.h
 *   Eigen/src/Core/arch/Default/GenericPacketMathFunctions.h
 * Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
 * Copyright (C) 2007 Julien Pommier
 * Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
 * Copyright (C) 2009-2019 Gael Guennebaud <gael.guennebaud@inria.fr>
 * Copyright (C) 2014 Pedro Gonnet (pedro.gonnet@gmail.com)
 *
 * This Source Code Form is subject to the terms of the Mozilla
 * Public License, v. 2.0. If a copy of the MPL was not distributed
 * with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
 * The full license is also distributed in LICENSES/Eigen-MPL-2.0.txt.
 */

/* Nonempty contiguous FP32 row. Canonical row alignment is independent of the
 * caller's actual address. Compile without fast-math or contraction. */
static float sv_two_four_lane_mean_last(const float *x, size_t width, size_t row_index) {
    const size_t start = (width < 4 || width % 4 == 0)
        ? 0 : (4 - ((row_index % 4) * (width % 4)) % 4) % 4;
    float sum;
    if (width - start < 4) {
        sum = x[0];
        for (size_t k=1; k<width; ++k) sum = sum + x[k];
    } else {
        const size_t packet_end = start + ((width-start)/4)*4;
        const size_t pair_end = start + ((width-start)/8)*8;
        float a0=x[start], a1=x[start+1], a2=x[start+2], a3=x[start+3];
        if (packet_end-start > 4) {
            float b0=x[start+4], b1=x[start+5], b2=x[start+6], b3=x[start+7];
            for (size_t k=start+8; k<pair_end; k+=8) {
                a0=a0+x[k]; a1=a1+x[k+1]; a2=a2+x[k+2]; a3=a3+x[k+3];
                b0=b0+x[k+4]; b1=b1+x[k+5]; b2=b2+x[k+6]; b3=b3+x[k+7];
            }
            a0=a0+b0; a1=a1+b1; a2=a2+b2; a3=a3+b3;
            if (packet_end > pair_end) {
                a0=a0+x[pair_end]; a1=a1+x[pair_end+1];
                a2=a2+x[pair_end+2]; a3=a3+x[pair_end+3];
            }
        }
        sum = (a0+a2)+(a1+a3);
        for (size_t k=0; k<start; ++k) sum=sum+x[k];
        for (size_t k=packet_end; k<width; ++k) sum=sum+x[k];
    }
    return sum/(float)width;
}

'''
