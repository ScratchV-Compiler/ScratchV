"""Compare two ordered K steps with separate or paired loads from each Q row."""
from dataclasses import replace
from textwrap import indent

from probes.w6_performance.attention_panel import _score_kernel
from probes.w6_performance.attention_profile import replace_once
from probes.w6_performance.attention_scalar_epilogue import _BIT_STORE, _FLOAT_STORE
from scratchv.backend.tensor_c_codegen import TensorCArtifact


_PAIR_HELPERS = """
// The aligned path is entered only for rows starting on an 8-byte boundary.
// Each subsequent pair begins two floats later. Express both input objects
// to the compiler without reinterpreting a float pointer as a uint64_t*.
static inline uint64_t w6_q_pair(const float *src) {
#if defined(__riscv) && __riscv_xlen == 64 && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    uint64_t pair;
    __asm__("ld %0, %1" : "=r"(pair) : "m"(src[0]), "m"(src[1]));
    return pair;
#else
    uint32_t lo, hi;
    __builtin_memcpy(&lo, src, sizeof(lo));
    __builtin_memcpy(&hi, src + 1, sizeof(hi));
    return (uint64_t)hi << 32 | lo;
#endif
}
static inline float w6_q_value(uint32_t bits) {
#if defined(__riscv) && defined(__riscv_flen) && __riscv_flen >= 32
    float value;
    __asm__("fmv.w.x %0, %1" : "=f"(value) : "r"(bits));
    return value;
#else
    float value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}
"""


def _two_steps(rows, paired):
    lines = ["for (size_t k = 0; k < 128; k += 2) {"]
    if paired:
        lines += [f"    uint64_t pairs[{rows}];", f"    for (size_t r = 0; r < {rows}; ++r)",
                  "        pairs[r] = w6_q_pair(x + r * 128 + k);"]
    for step in (0, 1):
        query = f"x[r * 128 + k + {step}]"
        if paired:
            bits = "pairs[r]" if step == 0 else "pairs[r] >> 32"
            query = f"w6_q_value((uint32_t)({bits}))"
        lines += [f"    for (size_t r = 0; r < {rows}; ++r) {{", f"        float q = {query};",
                  "        for (size_t c = 0; c < 4; ++c)",
                  f"            acc[r][c] = __builtin_fmaf(q, panel[k + {step}][c], acc[r][c]);", "    }"]
    return "\n".join([*lines, "}"])


def with_paired_q(artifact: TensorCArtifact, *, packed_load: bool) -> TensorCArtifact:
    """Preserve the checked 5×4 and tail kernels, including four-byte aligned Q inputs."""
    original = _score_kernel(check_output=True, row_tile=5, cache_output=True).replace(_FLOAT_STORE, _BIT_STORE)
    if (type(packed_load) is not bool or artifact.qemu_workspace_layout != "none" or artifact.external_weights
            or artifact.source.count(original) != 1):
        raise ValueError("Paired Q requires the fixed checked 5×4 bitwise score kernel")
    updated = original
    for rows in (5, 1):
        loop = f"""for (size_t k = 0; k < 128; ++k)
                for (size_t r = 0; r < {rows}; ++r)
                    for (size_t c = 0; c < 4; ++c)
                        acc[r][c] = __builtin_fmaf(x[r * 128 + k], panel[k][c], acc[r][c]);"""
        replacement = _two_steps(rows, paired=False)
        if packed_load:
            replacement = ("if (((uintptr_t)x & 7u) == 0) {\n" + indent(_two_steps(rows, paired=True), "    ")
                           + "\n} else {\n" + indent(replacement, "    ") + "\n}")
        updated = replace_once(updated, "            " + loop, indent(replacement, "            "))
    if packed_load:
        updated = _PAIR_HELPERS + updated
    return replace(artifact, source=replace_once(artifact.source, original, updated))
