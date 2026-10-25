/* Thin NMSIS-DSP adapter for ScratchV's freestanding RV64GC/LP64D guest.
 *
 * A and B are read-only row-major matrices; C is a separate row-major buffer.
 * The caller owns storage and lifetimes. No allocation or retained pointers.
 * Build the official riscv_mat_mult_f32.c with the same strict FP flags.
 */
#include <stddef.h>
#include <stdint.h>

#if !defined(__riscv) || !defined(__riscv_xlen) || __riscv_xlen != 64
#error "ScratchV NMSIS adapter requires RV64"
#endif
#if !defined(__riscv_flen) || __riscv_flen != 64 \
    || !defined(__riscv_float_abi_double)
#error "ScratchV NMSIS adapter requires RV64GC/LP64D floating point"
#endif
#if defined(__riscv_vector) || defined(__riscv_dsp) || defined(__riscv_xxldsp)
#error "ScratchV NMSIS adapter supports the scalar RV64GC target only"
#endif

#include "dsp/matrix_functions.h"

#if defined(RISCV_MATH_VECTOR) || defined(RISCV_MATH_VECTOR_ZVE32F) \
    || defined(RISCV_MATH_DSP)
#error "ScratchV NMSIS adapter supports scalar NMSIS-DSP only"
#endif

int sv_nmsis_matmul_f32_v1(
    const float *a, const float *b, float *c,
    uint64_t m, uint64_t n, uint64_t k)
{
    if (m > UINT16_MAX || n > UINT16_MAX || k > UINT16_MAX) {
        return 1;
    }
    /* The upstream scalar kernel uses do/while loops for rows and columns. */
    if (m == 0 || n == 0) {
        return 0;
    }
    if (c == NULL || m > SIZE_MAX / sizeof(float) / n) {
        return 1;
    }
    if (k == 0) {
        const size_t count = (size_t)m * (size_t)n;
        for (size_t i = 0; i < count; ++i) {
            c[i] = 0.0f;
        }
        return 0;
    }
    if (a == NULL || b == NULL
        || m > SIZE_MAX / sizeof(float) / k
        || k > SIZE_MAX / sizeof(float) / n) {
        return 1;
    }

    /* NMSIS's shared matrix type has mutable pData even for input matrices.
     * riscv_mat_mult_f32 reads both inputs without modifying their storage.
     */
    const riscv_matrix_instance_f32 left = {
        (uint16_t)m, (uint16_t)k, (float32_t *)a
    };
    const riscv_matrix_instance_f32 right = {
        (uint16_t)k, (uint16_t)n, (float32_t *)b
    };
    riscv_matrix_instance_f32 result = {(uint16_t)m, (uint16_t)n, c};
    return (int)riscv_mat_mult_f32(&left, &right, &result);
}
