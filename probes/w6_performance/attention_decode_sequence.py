"""Two successive one-query score operations over the same fixed-stride key cache."""

from dataclasses import replace
import re

from probes.w6_performance.attention_decode_native import native_artifact
from probes.w6_performance.attention_profile import replace_once
from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorSpec
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Value


_LENGTHS = (255, 256)
_CAPACITY = 256
_GROUPS = 8
_WIDTH = 128
_INPUTS = (
    TensorSpec("history", DataType.FLOAT32, (8, 254, 128)),
    TensorSpec("queries", DataType.FLOAT32, (2, 8, 2, 128)),
    TensorSpec("new_keys", DataType.FLOAT32, (2, 8, 128)),
)
_OUTPUT = TensorSpec("scores", DataType.FLOAT32, (16 * sum(_LENGTHS),))
_TICKS = """
static volatile uint64_t w6_decode_sequence_ticks[8];
static inline uint64_t w6_read_time(void) {
    uint64_t t;
    __asm__ volatile("fence iorw,iorw\\n\\trdtime %0\\n\\tfence iorw,iorw"
                     : "=r"(t) :: "memory");
    return t;
}
"""


def _score_source(artifact, length, variant):
    """Keep the original operators, changing only the physical head stride of K."""
    direct = variant == "native-paired-dot"
    start = ("struct w6_decode_group {" if direct
             else "static __attribute__((noinline)) int sv_kernel_0(")
    prelude, first, rest = artifact.source.partition(start)
    kernels, export, entry = rest.partition("#if defined(_WIN32)")
    if not first or not export:
        raise ValueError("Expected the native score kernels and external entry")
    kernels = first + kernels
    if length != _CAPACITY:
        if direct:
            old = f"ctx->keys + group * {length} * 128"
            new = f"ctx->keys + group * {_CAPACITY} * 128"
        else:
            old = f"((int64_t)((i / {length * _WIDTH}ULL) % 8ULL) * ({length * _WIDTH}LL))"
            new = old.replace(f"({length * _WIDTH}LL)", f"({_CAPACITY * _WIDTH}LL)")
        kernels = replace_once(kernels, old, new)
    operation = entry.split("/* w6_decode_native_begin */")[1].split("/* w6_decode_native_end */")[0]
    score = f"""
static int w6_scores_{length}(const float *sv_v0, const float *sv_v1,
                              unsigned char *sv_workspace, const float **result) {{
    {operation}
    *result = sv_v3;
    return 0;
}}
"""
    # Both static shapes live in one ELF. Give their generated C helpers distinct names.
    names = r"\b(sv_kernel_[01]|sv_matmul_sv_v3(?:_range)?|w6_decode_group(?:_range)?|w6_paired_dot_tile)\b"
    return prelude, re.sub(names, lambda m: f"{m[0]}_{length}", kernels + score)


def _checks(workspace_bytes, inputs=_INPUTS, output=_OUTPUT):
    # Reuse the normal external ABI validation for the complete sequence entry.
    graph = IRBuilder()
    args = [Value(spec.name, shape=spec.shape) for spec in inputs]
    graph.new_function("main", args)
    graph.new_block("entry")
    graph.ret(args[0])
    generator = TensorCCodegen(graph.program, {}, constant_storage="external")
    generator.generate()
    return "\n    ".join(generator._external_prologue(inputs, output, workspace_bytes))


def sequence_artifact(variant, *, instrumented=False):
    """Emit both steps as one guest entry. Publish one flat vector only after both succeed."""
    if variant not in ("native-transpose-2x4", "native-paired-dot"):
        raise ValueError(f"Unknown sequence variant: {variant}")
    steps = [native_artifact(length, variant) for length in _LENGTHS]
    sources = [_score_source(artifact, length, variant) for artifact, length in zip(steps, _LENGTHS)]
    if sources[0][0] != sources[1][0]:
        raise ValueError("Expected both native score shapes to share a C prelude")
    cache_bytes = _GROUPS * _CAPACITY * _WIDTH * 4
    scratch_offset = cache_bytes + _OUTPUT.nbytes
    workspace_bytes = scratch_offset + max(a.workspace_bytes for a in steps)

    def tick(i):
        return f"w6_decode_sequence_ticks[4 * step + {i}] = w6_read_time();" if instrumented else ""

    entry = f"""
int scratchv_run_external(const void *const inputs[], const void *const weights[],
                          void *workspace, size_t workspace_bytes, void *output) {{
    {_checks(workspace_bytes)}
    const float *history = inputs[0], *queries = inputs[1], *new_keys = inputs[2];
    float *cache = (float *)sv_workspace;
    float *staged = (float *)(sv_workspace + {cache_bytes}ULL);
    unsigned char *scratch = sv_workspace + {scratch_offset}ULL;
    // Initialise only once. NaNs in unused slots help detect accidental reads.
    for (size_t h = 0; h < 8; ++h) {{
        float *head = cache + h * 256 * 128;
        for (size_t i = 0; i < 254 * 128; ++i) {{
            float value = history[h * 254 * 128 + i];
            if (!sv_finite(value)) return 2;
            head[i] = value;
        }}
        for (size_t i = 254 * 128; i < 256 * 128; ++i) head[i] = __builtin_nanf("");
    }}
    for (size_t step = 0; step < 2; ++step) {{
        {tick(0)}
        const size_t valid = 255 + step;
        const float *q = queries + step * 16 * 128;
        for (size_t i = 0; i < 16 * 128; ++i) if (!sv_finite(q[i])) return 2;
        for (size_t h = 0; h < 8; ++h) {{
            float *dest = cache + (h * 256 + valid - 1) * 128;
            const float *next = new_keys + (step * 8 + h) * 128;
            for (size_t d = 0; d < 128; ++d) {{
                if (!sv_finite(next[d])) return 2;
                dest[d] = next[d];
            }}
        }}
        {tick(1)}
        const float *scores;
        int status = step == 0 ? w6_scores_255(q, cache, scratch, &scores)
                               : w6_scores_256(q, cache, scratch, &scores);
        if (status) return status;
        {tick(2)}
        for (size_t i = 0; i < 16 * valid; ++i) staged[step * 16 * 255 + i] = scores[i];
        {tick(3)}
    }}
    for (size_t i = 0; i < {_OUTPUT.size}; ++i) ((float *)output)[i] = staged[i];
    return 0;
}}
"""
    source = sources[0][0] + (_TICKS if instrumented else "") + "".join(s[1] for s in sources) + entry
    return replace(steps[0], source=source, inputs=_INPUTS, output=_OUTPUT, workspace_bytes=workspace_bytes)
