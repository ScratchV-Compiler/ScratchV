/* Controlled test provider, not a production operator library.
 * Mutable counters and injected failures exist only to observe test calls.
 */
#include <stdint.h>
#include <math.h>

#if defined(_WIN32)
#define TEST_EXPORT __declspec(dllexport)
#else
#define TEST_EXPORT
#endif

static uint64_t test_calls = 0;
static int test_failure = 0;

TEST_EXPORT uint64_t scratchv_test_matmul_calls(void) { return test_calls; }
TEST_EXPORT void scratchv_test_matmul_failure(int status) { test_failure = status; }

int sv_matmul_f32_v1(const float *a, const float *b, float *c,
                     uint64_t m, uint64_t n, uint64_t k) {
    ++test_calls;
    if (test_failure > 0) return test_failure;
    for (uint64_t i = 0; i < m; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (uint64_t p = 0; p < k; ++p) {
                float product = a[i * k + p] * b[p * n + j];
                sum = sum + product;
            }
            c[i * n + j] = sum;
        }
    }
    if (test_failure < 0 && m && n) c[0] = NAN;
    return 0;
}

int scratchv_test_matmul_fail(const float *a, const float *b, float *c,
                              uint64_t m, uint64_t n, uint64_t k) {
    (void)a; (void)b; (void)c; (void)m; (void)n; (void)k;
    return 17;
}
