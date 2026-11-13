"""Schedule complete single-token Attention independently for each K/V head in QEMU."""
from dataclasses import replace
import math

from probes.w6_performance.attention_decode_sequence_full import (
    _softmax_and_value_source, full_sequence_artifact,
)


def _head_kernel(length):
    # Use the backend's complete Softmax row body, with the two query rows in one group.
    softmax = _softmax_and_value_source(length).split("#if defined(SCRATCHV_CORES)", 1)[0]
    if softmax.count("o<16ULL") != 1:
        raise ValueError("Expected the generated Softmax loop over 16 rows")
    softmax = softmax.replace(f"w6_softmax_{length}", f"w6_group_softmax_{length}").replace("o<16ULL", "o<2ULL")
    return softmax + f"""
struct w6_heads_{length} {{
    struct w6_decode_group_{length} score;
    struct w6_pv_{length} value;
    float *prob;
    int status[8];
}};

static void w6_head_range_{length}(void *opaque, size_t begin, size_t end) {{
    struct w6_heads_{length} *ctx = opaque;
    for (size_t group = begin; group < end; ++group) {{
        w6_decode_group_range_{length}(&ctx->score, group, group + 1);
        size_t offset = group * 2 * {length};
        float *prob = ctx->prob + offset;
        const float *scores = ctx->score.output + offset;
        for (size_t j = 0; j < 2 * {length}; ++j) {{
            if (!sv_finite(scores[j])) {{
                ctx->status[group] = 2;
                return;
            }}
            prob[j] = scores[j] * {(1 / math.sqrt(128)).hex()}f;
        }}
        int status = w6_group_softmax_{length}(prob, prob);
        if (status) {{
            ctx->status[group] = status;
            return;
        }}
        // A K/V head covers 32 tiles, each with two query rows and four V columns.
        w6_pv_range_{length}(&ctx->value, group * 32, (group + 1) * 32);
        for (size_t j = group * 256; j < (group + 1) * 256; ++j) {{
            if (!sv_finite(ctx->value.output[j])) {{
                ctx->status[group] = 2;
                return;
            }}
        }}
    }}
}}

static int w6_head_attention_{length}(const float *q, const float *keys,
        const float *values, float *scores, float *prob, float *context) {{
    struct w6_heads_{length} ctx = {{{{q, keys, scores}}, {{prob, values, context}}, prob, {{0}}}};
    sv_parallel_for(8, w6_head_range_{length}, &ctx);
    for (size_t group = 0; group < 8; ++group)
        if (ctx.status[group]) return ctx.status[group];
    return 0;
}}
"""


def head_groups_artifact(*, instrumented=False):
    artifact = full_sequence_artifact("native-paired-dot", instrumented=instrumented)
    prelude, marker, entry = artifact.source.partition("int scratchv_run_external(")
    original_start = "        const float *scores;"
    original_end = "        if (status) return status;"
    if not marker or entry.count(original_start) != 1 or entry.count(original_end) != 3:
        raise ValueError("Expected the two-step entry with separate QK, Softmax and V calls")
    start = entry.index(original_start)
    end = entry.rindex(original_end) + len(original_end)
    four_cores = "#if defined(SCRATCHV_CORES) && SCRATCHV_CORES == 4"
    kernels = "\n".join(_head_kernel(n) for n in (255, 256))
    dispatch = """
        int status = step == 0 ? w6_head_attention_255(q, keys, values, (float *)scratch, prob, context)
                               : w6_head_attention_256(q, keys, values, (float *)scratch, prob, context);
        if (status) return status;
"""
    source = (prelude + four_cores + "\n" + kernels + "\n#endif\n" + marker + entry[:start]
              + four_cores + dispatch + "#else\n" + entry[start:end] + "\n#endif" + entry[end:])
    return replace(artifact, source=source)
