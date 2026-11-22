"""Compare K load strategies for the 5×4 RV64GC score kernel."""
from dataclasses import replace

from probes.w6_performance.attention_profile import replace_once
from scratchv.backend.tensor_c_codegen import TensorCArtifact
from scratchv.ir.types import DataType


_LOADS = """
static inline uint64_t w6_box_fp32(const float *src) {
    uint32_t bits;
    __builtin_memcpy(&bits, src, sizeof(bits));
    return UINT64_C(0xffffffff00000000) | bits;
}
static inline float w6_load_boxed_k(const uint64_t *src) {
    float value;
#if defined(__riscv) && defined(__riscv_flen) && __riscv_flen == 64
    // FLD transfers the already boxed bits. FMADD.S consumes their FP32 payload.
    __asm__("fld %0, %1" : "=f"(value) : "m"(*src));
#else
    uint32_t bits = (uint32_t)*src;
    __builtin_memcpy(&value, &bits, sizeof(value));
#endif
    return value;
}
"""


_PAIR_LOAD = """
static inline void w6_load_k_pair(const float *src, float *low, float *high) {
#if defined(__riscv) && __riscv_xlen == 64 && defined(__riscv_flen) && \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    // Each pair starts on an eight-byte boundary. List both float memory
    // operands so the compiler sees the full LD without any type punning.
    uint64_t word;
    __asm__("ld %0, %1" : "=r"(word) : "m"(src[0]), "m"(src[1]));
    __asm__("fmv.w.x %0, %1" : "=f"(*low) : "r"(word));
    __asm__("fmv.w.x %0, %1" : "=f"(*high) : "r"(word >> 32));
#else
    __builtin_memcpy(low, src, sizeof(*low));
    __builtin_memcpy(high, src + 1, sizeof(*high));
#endif
}
"""


def _checked_source(artifact):
    inputs = [(spec.name, spec.shape, spec.dtype) for spec in artifact.inputs]
    if (inputs != [("x", (16, 256, 128), DataType.FLOAT32),
                   ("weight", (16, 128, 256), DataType.FLOAT32)]
            or artifact.constant_storage != "external" or artifact.external_weights
            or artifact.qemu_workspace_layout != "none"
            or artifact.output.shape != (16, 256, 256) or artifact.output.dtype != DataType.FLOAT32):
        raise ValueError("K load variants require the fixed dynamic FP32 score graph")
    source = artifact.source
    for marker in ("float *const output = ctx->output;", "row + 5 <= 256", "const size_t row = 255;",
                   "__atomic_fetch_add(&ctx->next, 8, __ATOMIC_RELAXED)",
                   "w6_publish_parallel(output, sv_v2, 4194304ULL);"):
        if source.count(marker) != 1:
            raise ValueError("K load variants require 5×4 with cached output and eight-block claims")
    return source


def _replace_k_loads(source, load_key):
    for rows in (5, 1):
        original = f"""for (size_t k = 0; k < 128; ++k)
                for (size_t r = 0; r < {rows}; ++r)
                    for (size_t c = 0; c < 4; ++c)
                        acc[r][c] = __builtin_fmaf(x[r * 128 + k], panel[k][c], acc[r][c]);"""
        modified = f"""for (size_t k = 0; k < 128; ++k) {{
                float key[4];
{load_key}
                for (size_t r = 0; r < {rows}; ++r)
                    for (size_t c = 0; c < 4; ++c)
                        acc[r][c] = __builtin_fmaf(x[r * 128 + k], key[c], acc[r][c]);
            }}"""
        source = replace_once(source, original, modified)
    return source


def with_preboxed_k(artifact: TensorCArtifact) -> TensorCArtifact:
    """Box each of the four K columns once while packing its 128-element panel."""
    source = _checked_source(artifact)
    source = replace_once(source, "struct w6_scores {", _LOADS + "\nstruct w6_scores {")
    source = replace_once(source, "float panel[128][4];", "uint64_t panel[128][4];")
    source = replace_once(source, "panel[k][lane] = rhs[k * 256 + lane];",
                          "panel[k][lane] = w6_box_fp32(&rhs[k * 256 + lane]);")
    load_key = """                for (size_t c = 0; c < 4; ++c)
                    key[c] = w6_load_boxed_k(&panel[k][c]);"""
    return replace(artifact, source=_replace_k_loads(source, load_key))


def with_paired_k(artifact: TensorCArtifact) -> TensorCArtifact:
    """Use two aligned integer loads for four FP32 K values in each K step."""
    source = _checked_source(artifact)
    source = replace_once(source, "struct w6_scores {", _PAIR_LOAD + "\nstruct w6_scores {")
    source = replace_once(source, "float panel[128][4];", "_Alignas(8) float panel[128][4];")
    load_key = """                w6_load_k_pair(&panel[k][0], &key[0], &key[1]);
                w6_load_k_pair(&panel[k][2], &key[2], &key[3]);"""
    return replace(artifact, source=_replace_k_loads(source, load_key))
