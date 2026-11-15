"""Two successive single-token Attention outputs with a fixed-capacity K/V cache."""

from dataclasses import replace
import math
import re

from probes.w6_performance.attention_decode_native import native_artifact
from probes.w6_performance.attention_decode_sequence import (
    _CAPACITY, _GROUPS, _INPUTS, _LENGTHS, _TICKS, _WIDTH, _checks, _score_source,
)
from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorSpec
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Value


_FULL_INPUTS = (*_INPUTS, TensorSpec("value_history", DataType.FLOAT32, (8, 254, 128)),
                TensorSpec("new_values", DataType.FLOAT32, (2, 8, 128)))
_CONTEXT = TensorSpec("context", DataType.FLOAT32, (2, 8, 2, 128))


def _softmax_and_value_source(length):
    """Use the backend Softmax and P @ V kernels with a 256-position V head stride."""
    scores = Value("scores", shape=(_GROUPS, 2, length))
    values = Value("v", shape=(_GROUPS, length, _WIDTH))
    graph = IRBuilder()
    graph.new_function("main", [scores, values])
    graph.new_block("entry")
    graph.ret(graph.matmul(graph.softmax(scores, axis=-1), values))
    generator = TensorCCodegen(graph.program, {}, constant_storage="external", kernel_calls=True,
                               matmul_policy="blocked_fma", matmul_tile="2x4")
    generator.generate()
    source = "".join(generator.kernels)
    if length != _CAPACITY:
        # The existing MatMul traverses only valid positions. Adjacent KV heads
        # are separated by the reserved capacity, also when position 256 is empty.
        name = generator.names[values.name]
        rhs = f"{name}+(batch)*{length * _WIDTH}ULL+k*{_WIDTH}ULL+column"
        if source.count(rhs) != 2:  # One four-core callback and its scalar fallback.
            raise ValueError("Unexpected generated P @ V head stride")
        source = source.replace(rhs, rhs.replace(f"*{length * _WIDTH}ULL", f"*{_CAPACITY * _WIDTH}ULL"))
    # These two generated operators and their callback coexist with both QK shapes.
    symbols = {"sv_kernel_0": "w6_softmax", "sv_kernel_1": "w6_weight_values",
               "sv_matmul_sv_v3": "w6_pv", "sv_matmul_sv_v3_range": "w6_pv_range"}
    return re.sub(r"\b(" + "|".join(symbols) + r")\b",
                   lambda match: f"{symbols[match[0]]}_{length}", source)


_CACHE_OPS = """
static int w6_init_cache(float *cache, const float *history) {
    for (size_t h = 0; h < 8; ++h) {
        float *head = cache + h * 256 * 128;
        for (size_t i = 0; i < 254 * 128; ++i) {
            float value = history[h * 254 * 128 + i];
            if (!sv_finite(value)) return 2;
            head[i] = value;
        }
        for (size_t i = 254 * 128; i < 256 * 128; ++i) head[i] = __builtin_nanf("");
    }
    return 0;
}

static int w6_append(float *cache, const float *next, size_t pos) {
    for (size_t h = 0; h < 8; ++h) {
        float *row = cache + (h * 256 + pos) * 128;
        for (size_t d = 0; d < 128; ++d) {
            float value = next[h * 128 + d];
            if (!sv_finite(value)) return 2;
            row[d] = value;
        }
    }
    return 0;
}
"""


def full_sequence_artifact(variant, *, instrumented=False):
    """Emit two complete Attention steps, publishing both results together."""
    if variant not in ("native-transpose-2x4", "native-paired-dot"):
        raise ValueError(f"Unknown K score variant: {variant}")
    scores = [native_artifact(length, variant) for length in _LENGTHS]
    kernels = [_score_source(a, length, variant) for a, length in zip(scores, _LENGTHS)]
    if kernels[0][0] != kernels[1][0]:
        raise ValueError("Score kernels must share one C prelude")
    cache_bytes = _GROUPS * _CAPACITY * _WIDTH * 4
    staged_offset = 2 * cache_bytes
    prob_offset = staged_offset + _CONTEXT.nbytes
    context_offset = prob_offset + 16 * _CAPACITY * 4
    scratch_offset = context_offset + 16 * _WIDTH * 4
    workspace_bytes = scratch_offset + max(a.workspace_bytes for a in scores)

    def tick(i):
        return f"w6_attention_sequence_ticks[step * 6 + {i}] = w6_read_time();" if instrumented else ""

    entry = f"""
int scratchv_run_external(const void *const inputs[], const void *const weights[],
                          void *workspace, size_t workspace_bytes, void *output) {{
    {_checks(workspace_bytes, _FULL_INPUTS, _CONTEXT)}
    const float *queries = inputs[1], *new_keys = inputs[2], *new_values = inputs[4];
    float *keys = (float *)sv_workspace, *values = (float *)(sv_workspace + {cache_bytes}ULL);
    float *staged = (float *)(sv_workspace + {staged_offset}ULL);
    float *prob = (float *)(sv_workspace + {prob_offset}ULL);
    float *context = (float *)(sv_workspace + {context_offset}ULL);
    unsigned char *scratch = sv_workspace + {scratch_offset}ULL;
    if (w6_init_cache(keys, inputs[0]) || w6_init_cache(values, inputs[3])) return 2;
    for (size_t step = 0; step < 2; ++step) {{
        {tick(0)}
        size_t length = 255 + step;
        const float *q = queries + step * 16 * 128;
        for (size_t i = 0; i < 16 * 128; ++i) if (!sv_finite(q[i])) return 2;
        if (w6_append(keys, new_keys + step * 8 * 128, length - 1) ||
            w6_append(values, new_values + step * 8 * 128, length - 1)) return 2;
        {tick(1)}
        const float *scores;
        int status = step == 0 ? w6_scores_255(q, keys, scratch, &scores)
                               : w6_scores_256(q, keys, scratch, &scores);
        if (status) return status;
        {tick(2)}
        for (size_t i = 0; i < 16 * length; ++i) prob[i] = scores[i] * {(1 / math.sqrt(_WIDTH)).hex()}f;
        // Each row's maximum is read before the backend writes its probabilities.
        status = step == 0 ? w6_softmax_255(prob, prob) : w6_softmax_256(prob, prob);
        if (status) return status;
        {tick(3)}
        status = step == 0 ? w6_weight_values_255(context, prob, values)
                           : w6_weight_values_256(context, prob, values);
        if (status) return status;
        {tick(4)}
        for (size_t i = 0; i < 16 * 128; ++i) staged[step * 16 * 128 + i] = context[i];
        {tick(5)}
    }}
    for (size_t i = 0; i < {_CONTEXT.size}; ++i) ((float *)output)[i] = staged[i];
    return 0;
}}
"""
    timer = _TICKS.replace("w6_decode_sequence_ticks[8]", "w6_attention_sequence_ticks[12]") if instrumented else ""
    source = (kernels[0][0] + timer + "".join(k[1] for k in kernels)
              + "".join(_softmax_and_value_source(n) for n in _LENGTHS) + _CACHE_OPS + entry)
    return replace(scores[0], source=source, inputs=_FULL_INPUTS, output=_CONTEXT,
                   workspace_bytes=workspace_bytes)
