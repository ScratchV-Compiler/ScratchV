/* Thin host-only adapter for the standard LP64 OpenBLAS CBLAS interface.
 *
 * A and B are read-only row-major matrices; C is a separate row-major buffer.
 * All buffer sizes and lifetimes are owned by the caller. The adapter neither
 * retains pointers nor changes global OpenBLAS settings. A zero return means
 * this call returned successfully; it cannot intercept a library abort.
 */
#include <cblas.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

_Static_assert(sizeof(blasint) == 4,
               "ScratchV OpenBLAS adapter requires LP64 (32-bit blasint)");

int sv_openblas_matmul_f32_v1(
    const float *a, const float *b, float *c,
    uint64_t m, uint64_t n, uint64_t k)
{
    if (m > INT_MAX || n > INT_MAX || k > INT_MAX) {
        return 1;
    }
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

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (blasint)m, (blasint)n, (blasint)k,
                1.0f, a, (blasint)k, b, (blasint)n,
                0.0f, c, (blasint)n);
    return 0;
}
