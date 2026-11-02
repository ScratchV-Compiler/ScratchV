# SPDX-License-Identifier: MPL-2.0
"""Portable scalar C for the explicit FP32 nonlinear arithmetic policy.

The trigonometric portion derives from Eigen under MPL-2.0; its emitted source
carries the copyright notice and license reference. Sigmoid and Softmax
polynomials derive from ONNX Runtime MLAS under MIT and retain their full
notice below. See THIRD_PARTY_NOTICES.txt and LICENSES/Eigen-MPL-2.0.txt.

This fixes arithmetic order, independently of the host CPU: non-fused trig
range reduction (libm fallback for abs(x) >= 18000), fused bounded-rational
Sigmoid and polynomial Softmax with sixteen accumulators, pairwise lane fold,
ldexpf scaling and reciprocal multiplication. It requires no SIMD ISA.
Input/output finiteness and error propagation remain the backend's contract.
"""

TRIG_REFERENCE_C = r'''
/*
 * SPDX-License-Identifier: MPL-2.0
 * Adapted from Eigen (https://gitlab.com/libeigen/eigen), commit
 * 1d8b82b0740839c0de7f1242a3585e3390ff5f33:
 * Eigen/src/Core/Redux.h
 * Eigen/src/Core/arch/SSE/PacketMath.h
 * Eigen/src/Core/arch/Default/GenericPacketMathFunctions.h
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

/* Explicit FP32 non-FMA trig order from fp32_eigen.trig.
 * Outside the conservative interval, retain the original platform libm.
 * Portable scalar C; no host SIMD or RVV requirement. */
static float sv_fp32_reference_trig(float value, int sine) {
    union {float f; uint32_t u;} input={.f=value}, absolute, rounded_bits, output;
    absolute.u=input.u & 0x7fffffffU;
    if (!(absolute.f < 18000.0f)) return sine ? sinf(value) : cosf(value);
    float reduced=absolute.f;
    float rounded=reduced * 0x1.45f3060000000p-1f;
    rounded=rounded + 12582912.0f;
    rounded_bits.f=rounded;
    uint32_t quadrant=rounded_bits.u;
    float multiple=rounded-12582912.0f;
    reduced=multiple*-0x1.9200000000000p+0f+reduced;
    reduced=multiple*-0x1.fb80000000000p-12f+reduced;
    reduced=multiple*0x1.5dc0000000000p-23f+reduced;
    reduced=multiple*0x1.e973dc0000000p-35f+reduced;
    float squared=reduced*reduced;
    float cosine=0x1.98e6160000000p-16f;
    cosine=cosine*squared+-0x1.6c06dc0000000p-10f;
    cosine=cosine*squared+0x1.55553c0000000p-5f;
    cosine=cosine*squared-0.5f;
    cosine=cosine*squared+1.0f;
    float sinpoly=-0x1.9ae1940000000p-13f;
    sinpoly=sinpoly*squared+0x1.110ba60000000p-7f;
    sinpoly=sinpoly*squared+-0x1.5555500000000p-3f;
    sinpoly=sinpoly*squared;
    sinpoly=sinpoly*reduced+reduced;
    int even=(quadrant & 1U)==0;
    output.f=sine ? (even ? sinpoly : cosine) : (even ? cosine : sinpoly);
    uint32_t sign=sine ? (input.u ^ (quadrant<<30)) & 0x80000000U
                       : ((quadrant+1U)<<30) & 0x80000000U;
    output.u ^= sign;
    return output.f;
}
'''

SIGMOID_REFERENCE_C = r'''
/*
MLAS polynomial constants/evaluation adapted from ONNX Runtime v1.22.1
core/mlas/lib/logistic.cpp, via scratchv.verification.fp32_reference.sigmoid.
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
*/

static float sv_fp32_reference_sigmoid(float x) {
  if (x < -18.0f) x=-18.0f; if (x > 18.0f) x=18.0f;
  float squared=x*x;
  float numerator=0x1.806aa20000000p-35f;
  float denominator=0x1.5789ea0000000p-41f;
  numerator=__builtin_fmaf(numerator,squared,0x1.f09d960000000p-24f);
  numerator=__builtin_fmaf(numerator,squared,0x1.fe82760000000p-15f);
  numerator=__builtin_fmaf(numerator,squared,0x1.16fab00000000p-7f);
  numerator=__builtin_fmaf(numerator,squared,0x1.fc7e640000000p-3f);
  denominator=__builtin_fmaf(denominator,squared,0x1.8be4f60000000p-28f);
  denominator=__builtin_fmaf(denominator,squared,0x1.a62fba0000000p-18f);
  denominator=__builtin_fmaf(denominator,squared,0x1.be2a7e0000000p-10f);
  denominator=__builtin_fmaf(denominator,squared,0x1.de7c300000000p-4f);
  denominator=__builtin_fmaf(denominator,squared,0x1.fc7e680000000p-1f);
  float value=(numerator*x)/denominator+0.5f;
  if (value < 0.0f) value=0.0f; if (value > 1.0f) value=1.0f;
  return value;
}
'''

SOFTMAX_REFERENCE_C = r'''
/*
 * MLAS polynomial constants/evaluation are adapted from ONNX Runtime v1.22.1:
 * core/mlas/lib/logistic.cpp, compute.cpp, x86_64/TransKernelFma3.S and
 * x86_64/TransKernelAvx512F.S (including their amd64 Windows equivalents).
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* Fixed sixteen-lane reference arithmetic, scalar FP32 implementation.
 * This chooses numerical order, not the runtime host ISA. */
extern float fmaf(float,float,float);
extern float ldexpf(float,int);
static float sv_fp32_reference_exp_nonpositive(float x) {
    const float lower=-103.9720840454f, bias=12582912.0f;
    if (x < lower) x=lower;
    float rounded=fmaf(x,1.44269504088896341f,bias);
    float exponent=rounded-bias;
    x=fmaf(exponent,-6.93145752e-1f,x);
    x=fmaf(exponent,-1.42860677e-6f,x);
    float p=0x1.694000p-10f;
    p=fmaf(p,x,0x1.125edcp-7f);
    p=fmaf(p,x,0x1.555b5ap-5f);
    p=fmaf(p,x,0x1.555450p-3f);
    p=fmaf(p,x,0x1.fffff6p-2f);
    p=fmaf(p,x,0x1.000000p+0f);
    p=fmaf(p,x,0x1.000000p+0f);
    return ldexpf(p,(int)exponent);
}
static float sv_fp32_reference_sum_sixteen(const float *s) {
    float a0=s[0]+s[8],a1=s[1]+s[9],a2=s[2]+s[10],a3=s[3]+s[11];
    float a4=s[4]+s[12],a5=s[5]+s[13],a6=s[6]+s[14],a7=s[7]+s[15];
    float first=(a0+a1)+(a2+a3);
    float last=(a4+a5)+(a6+a7);
    return first+last;
}
'''
