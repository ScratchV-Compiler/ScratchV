"""Specialize 5×4 score writeback when input magnitude bounds rule out rounding to −0."""
from dataclasses import replace

from probes.w6_performance.attention_bounded_output import with_bounded_output
from probes.w6_performance.attention_panel import _score_kernel, _shared_blocks
from probes.w6_performance.attention_profile import replace_once
from probes.w6_performance.attention_scalar_epilogue import _BIT_STORE, _FLOAT_STORE
from scratchv.backend.tensor_c_codegen import TensorCArtifact


_ROUNDING = """
#if defined(__riscv) && defined(__riscv_flen)
static int w6_normal_rne(void) {
    unsigned long frm;
    __asm__ volatile("csrr %0, frm" : "=r"(frm));
    return frm == 0;
}
#else
#include <fenv.h>
static int w6_normal_rne(void) { return fegetround() == FE_TONEAREST; }
#endif
"""


def with_normal_output(artifact: TensorCArtifact) -> TensorCArtifact:
    """Store ordered accumulators directly only for nonzero operands in [2**-30, 2**32]."""
    # Such operands are multiples of 2**-53. Each product and rounded partial
    # sum is a multiple of 2**-106, too large for a nonzero result to round to
    # zero. Exact cancellation under RNE produces +0. After j products the
    # magnitude is at most the exactly representable j*2**64 (j <= 128).
    artifact = with_bounded_output(artifact, minimum_magnitude_bits=0x30800000)
    source = artifact.source

    checked = _score_kernel(check_output=True, row_tile=5, cache_output=True)
    if checked.count(_FLOAT_STORE) != 2:
        raise ValueError("Expected a cached five-row score kernel and its one-row tail")
    checked = checked.replace(_FLOAT_STORE, _BIT_STORE)
    claims = _shared_blocks(8)
    if source.count(checked + claims) != 1:
        raise ValueError("Expected the original bitwise 5×4 score kernel and eight-block claims")

    # Share the checked kernel's struct definitions, packing, FMA loop and
    # scheduler. Only the range callback's checked writeback changes.
    score_range = checked[checked.index("static void w6_score_range"):]
    score_range = score_range.replace(_BIT_STORE, "                    out_row[c] = acc[r][c];")
    score_range = replace_once(score_range, "    unsigned invalid = 0;\n", "")
    score_range = replace_once(score_range, "    if (invalid) __atomic_fetch_or(&ctx->error, 2u, __ATOMIC_RELAXED);\n", "")
    workers = claims[claims.index("#if SCRATCHV_CORES > 1"):].replace("w6_claim_scores", "w6_normal_claim_scores")
    start = source.index("static __attribute__((noinline)) int sv_kernel_0(")
    stop = source.index("\n}\n", start) + 3
    dispatch = source[start:stop]
    fast_dispatch = dispatch.replace("sv_kernel_0", "w6_normal_kernel").replace("w6_claim_scores", "w6_normal_claim_scores")
    fast_dispatch = replace_once(fast_dispatch, "  if (__atomic_load_n(&ctx.score.error, __ATOMIC_RELAXED)) return 2;\n", "")
    specialized = (score_range + workers + fast_dispatch).replace("w6_score_range", "w6_normal_range")
    source = replace_once(source, dispatch, dispatch + _ROUNDING + specialized)
    source = replace_once(source, "float *sv_v2 = bounded ?", "bounded = bounded && w6_normal_rne();\n  float *sv_v2 = bounded ?")
    source = replace_once(source, "int sv_status_0 = sv_kernel_0(sv_v2, sv_v0, sv_v1);",
                          "int sv_status_0 = bounded ? w6_normal_kernel(sv_v2, sv_v0, sv_v1)\n"
                          "                            : sv_kernel_0(sv_v2, sv_v0, sv_v1);")
    return replace(artifact, source=source)
