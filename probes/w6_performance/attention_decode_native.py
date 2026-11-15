"""Score one query per head using the eight native, token-major KV heads."""

from dataclasses import replace

from probes.w6_performance.attention_chain_kernel import _replace_part
from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value


def _paired_dot_kernel(prefix):
    # Each of the two query heads uses the same K vector. Compute up to four
    # positions per tile and reuse the two query loads across those positions.
    width = 4
    full = prefix - prefix % width
    tiles = []
    if full:
        tiles.append(f"""
        for (size_t pos = 0; pos < {full}; pos += {width})
            w6_paired_dot_tile(q, keys + pos * 128, scores + pos, {width});""")
    if full != prefix:
        tiles.append(f"""
        w6_paired_dot_tile(q, keys + {full} * 128, scores + {full}, {prefix - full});""")
    return f"""
struct w6_decode_group {{ const float *query, *keys; float *output; }};

static inline __attribute__((always_inline)) void w6_paired_dot_tile(
        const float *q, const float *keys, float *out, size_t positions) {{
    float a[{width}] = {{0}}, b[{width}] = {{0}};
    for (size_t d = 0; d < 128; ++d) {{
        const float q0 = q[d], q1 = q[128 + d];
        for (size_t i = 0; i < positions; ++i) {{
            const float k = keys[i * 128 + d];
            a[i] = __builtin_fmaf(q0, k, a[i]);
            b[i] = __builtin_fmaf(q1, k, b[i]);
        }}
    }}
    for (size_t i = 0; i < positions; ++i) {{
        out[i] = 0.0f + a[i];
        out[{prefix} + i] = 0.0f + b[i];
    }}
}}

static void w6_decode_group_range(void *opaque, size_t begin, size_t end) {{
    const struct w6_decode_group *ctx = opaque;
    for (size_t group = begin; group < end; ++group) {{
        const float *q = ctx->query + group * 2 * 128;
        const float *keys = ctx->keys + group * {prefix} * 128;
        float *scores = ctx->output + group * 2 * {prefix};
        {"".join(tiles)}
    }}
}}
"""


def native_artifact(prefix, variant):
    """Return a validated Transpose -> MatMul graph or the equivalent fused artifact."""
    if type(prefix) is not int or not 1 <= prefix <= 256:
        raise ValueError("native decode prefix length must be in [1,256]")
    if variant not in ("native-transpose-2x4", "native-paired-dot"):
        raise ValueError(f"Unknown native decode variant: {variant}")

    q, k = Value("q", shape=(8, 2, 128)), Value("k", shape=(8, prefix, 128))
    graph = IRBuilder()
    graph.new_function("main", [q, k])
    graph.new_block("entry")
    graph.ret(graph.matmul(q, graph.transpose(k, (0, 2, 1))))
    generator = TensorCCodegen(graph.program, {}, constant_storage="external", kernel_calls=True,
                               matmul_policy="blocked_fma", matmul_tile="2x4")
    plain = generator.generate()
    source = plain.source
    begin = generator.body.index("/* 0: transpose */")
    copy = generator.body.index("if (!output) return 1;")
    operation = generator.body[begin:copy]
    workspace_bytes = plain.workspace_bytes
    if variant == "native-paired-dot":
        # Only the final scores occupy the guest arena, until the generated
        # external entry validates and publishes them to the caller's buffer.
        query, key = (generator.names[s.name] for s in plain.inputs)
        scores = generator.names[plain.output.name]
        workspace_bytes = plain.output.nbytes
        source = _replace_part(source, "".join(generator.kernels), _paired_dot_kernel(prefix))
        operation = [f"""
float *{scores} = (float *)sv_workspace;
struct w6_decode_group ctx = {{{query}, {key}, {scores}}};
#if SCRATCHV_CORES > 1 && 16 * 128 * {prefix} >= 32768
sv_parallel_for(8, w6_decode_group_range, &ctx);
#else
w6_decode_group_range(&ctx, 0, 8);
#endif
for (size_t i = 0; i < {plain.output.size}; ++i)
    if (!sv_finite({scores}[i])) return 2;"""]

    old_body = generator._external_prologue(plain.inputs, plain.output, plain.workspace_bytes) + generator.body
    new_body = (generator._external_prologue(plain.inputs, plain.output, workspace_bytes)
                + generator.body[:begin] + ["/* w6_decode_native_begin */"]
                + operation + ["/* w6_decode_native_end */"] + generator.body[copy:])
    return replace(plain, source=_replace_part(source, "\n  ".join(old_body), "\n  ".join(new_body)),
                   workspace_bytes=workspace_bytes)
