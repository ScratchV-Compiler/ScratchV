"""Bitwise FP32 writeback for the W6 cached 5×4 score kernel on an RNE guest."""
from dataclasses import replace

from probes.w6_performance.attention_panel import _score_kernel
from scratchv.backend.tensor_c_codegen import TensorCArtifact
from scratchv.ir.types import DataType


_FLOAT_STORE = """                    float value = 0.0f + acc[r][c];
                    invalid |= !sv_finite(value);
                    out_row[c] = value;"""
_BIT_STORE = """                    uint32_t bits;
                    __builtin_memcpy(&bits, &acc[r][c], sizeof(bits));
                    uint32_t magnitude = bits & 0x7fffffffU;
                    invalid |= magnitude >= 0x7f800000U;
                    if (magnitude == 0) bits = 0;
                    __builtin_memcpy(out_row + c, &bits, sizeof(bits));"""


def with_bitwise_writeback(artifact: TensorCArtifact) -> TensorCArtifact:
    """Under round-to-nearest-even, adding +0 preserves finite nonzero values and yields +0 for either zero."""
    inputs = [(spec.name, spec.shape, spec.dtype) for spec in artifact.inputs]
    if (artifact.constant_storage != "external" or artifact.function_name != "scratchv_run_external"
            or artifact.qemu_workspace_layout != "none" or artifact.external_weights
            or artifact.workspace_bytes != 4194304 or artifact.matmul_policy != "blocked_fma"
            or inputs != [("x", (16, 256, 128), DataType.FLOAT32), ("weight", (16, 128, 256), DataType.FLOAT32)]
            or artifact.output.shape != (16, 256, 256) or artifact.output.dtype != DataType.FLOAT32):
        raise ValueError("Bitwise writeback requires the fixed dynamic FP32 Attention graph")

    source = artifact.source
    kernel = _score_kernel(check_output=True, row_tile=5, cache_output=True)
    markers = (kernel, "__atomic_fetch_add(&ctx->next, 8, __ATOMIC_RELAXED)",
               "w6_score_range(&ctx->score, first, first + 8);",
               "if (__atomic_load_n(&ctx.score.error, __ATOMIC_RELAXED)) return 2;",
               "if (!sv_finite(sv_v0[i])) return 2;", "if (!sv_finite(sv_v1[i])) return 2;",
               "int sv_status_0 = sv_kernel_0(sv_v2, sv_v0, sv_v1);\n  if (sv_status_0) return sv_status_0;",
               "w6_publish_parallel(output, sv_v2, 4194304ULL);")
    # Replace the full tiles and row 255 together. The FMA loop, error check and
    # final copy to the caller all belong to the original checked scalar kernel.
    if any(source.count(marker) != 1 for marker in markers) or kernel.count(_FLOAT_STORE) != 2:
        raise ValueError("Expected the cached 5×4 checked score kernel with eight-block claims")
    updated = kernel.replace(_FLOAT_STORE, _BIT_STORE)
    return replace(artifact, source=source.replace(kernel, updated, 1))
