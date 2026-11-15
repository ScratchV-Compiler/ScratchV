"""Fixed W6 Q @ transpose(K) -> Softmax comparison on two runtime FP32 inputs."""
from dataclasses import replace

from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorCCodegenError
from scratchv.ir.types import DataType, OpCode


_SHAPE = (16, 256, 128)
_OUTPUT = (16, 256, 256)


def _check_chain(generator, artifact):
    fn = generator.program.functions[0]
    ops = fn.blocks[0].instructions
    if ([op.opcode for op in ops] != [OpCode.TRANSPOSE, OpCode.MATMUL, OpCode.SOFTMAX, OpCode.RETURN]
            or len(artifact.inputs) != 2 or generator.program.global_values
            or any(v.dtype != DataType.FLOAT32 or v.shape != _SHAPE for v in artifact.inputs)
            or artifact.output.shape != _OUTPUT):
        raise TensorCCodegenError("W6 expects exactly Q,K[16,256,128] -> transpose/MatMul/Softmax")
    query, key = artifact.inputs
    transpose, product, softmax, ret = ops
    if (tuple(transpose.attrs.get("perm", ())) != (0, 2, 1)
            or [v.name for v in transpose.operands] != [key.name]
            or [v.name for v in product.operands] != [query.name, transpose.dest.name]
            or product.attrs
            or softmax.attrs.get("axis", -1) not in (-1, 2)
            or [v.name for v in softmax.operands] != [product.dest.name]
            or [v.name for v in ret.operands] != [softmax.dest.name]):
        raise TensorCCodegenError("W6 expects Softmax(Q @ K.T, axis=-1) with no other consumer")


_FUSED_RANGE = """
struct w6_attention {
    const float *query;
    const float *key;
    struct W6_SOFTMAX probabilities;
};

static void w6_attention_range(void *opaque, size_t begin, size_t end) {
    struct w6_attention *ctx = opaque;
    for (size_t task = begin; task < end; ++task) {
        const size_t first_row = 2 * task;
        const size_t head = first_row / 256;
        const float *q = ctx->query + first_row * 128;
        const float *keys = ctx->key + head * 256 * 128;
        float *scores = ctx->probabilities.output + first_row * 256;
        for (size_t column = 0; column < 256; column += 4) {
            const float *k = keys + column * 128;
            float acc[2][4] = {{0}};
            for (size_t d = 0; d < 128; ++d)
                for (size_t c = 0; c < 4; ++c) {
                    const float key_value = k[c * 128 + d];
                    acc[0][c] = __builtin_fmaf(q[d], key_value, acc[0][c]);
                    acc[1][c] = __builtin_fmaf(q[128 + d], key_value, acc[1][c]);
                }
            for (size_t r = 0; r < 2; ++r)
                for (size_t c = 0; c < 4; ++c) {
                    const float value = 0.0f + acc[r][c];
                    if (!sv_finite(value)) {
                        __atomic_store_n(&ctx->probabilities.status, 2, __ATOMIC_RELAXED);
                        return;
                    }
                    scores[r * 256 + column + c] = value;
                }
        }
        // The generated Softmax scans a whole row for its maximum before replacing
        // that row's scores with exponentials and then normalized probabilities.
        W6_SOFTMAX_range(&ctx->probabilities, first_row, first_row + 2);
    }
}
"""


def _replace_part(source, original, replacement):
    if not original or source.count(original) != 1:
        raise TensorCCodegenError("Unexpected generated Attention kernel or external entry point")
    return source.replace(original, replacement, 1)


def attention_chain_artifact(program, *, fused, nonlinear_policy="fp32-reference"):
    """Lower the same validated graph to the ordinary kernels or a single row dispatch."""
    generator = TensorCCodegen(program, {}, constant_storage="external", kernel_calls=True,
                               matmul_policy="blocked_fma", matmul_tile="2x4",
                               nonlinear_policy=nonlinear_policy, qemu_workspace_layout="none")
    plain = generator.generate()
    _check_chain(generator, plain)
    if not fused:
        return plain

    # Keep the generated Softmax callback verbatim. This probe also calls it in the
    # one-core fallback, so remove just the code generator's outer compile-time guard.
    softmax = generator._softmax_callbacks[(DataType.FLOAT32, _OUTPUT, 2)]
    callbacks = [part for part in generator.kernels if f"static void {softmax}_range(" in part]
    if len(callbacks) != 1:
        raise TensorCCodegenError("Missing generated last-axis Softmax callback")
    lines = callbacks[0].strip().splitlines()
    if not lines[0].startswith("#if defined(SCRATCHV_CORES)") or lines[-1] != "#endif":
        raise TensorCCodegenError("Unexpected Softmax callback guard")
    kernel = "\n" + "\n".join(lines[1:-1]) + "\n" + _FUSED_RANGE.replace("W6_SOFTMAX", softmax)

    # The only arena tensor is the final output; its rows double as score storage
    # until the owning worker normalizes them. Preserve the generated input checks
    # and the successful return copy, and generate the ABI guards for the new size.
    arena_bytes = plain.output.nbytes
    query, key = (generator.names[spec.name] for spec in plain.inputs)
    result = generator.names[plain.output.name]
    dispatch = f"""
  float *{result} = (float *)sv_workspace;
  struct w6_attention chain = {{{query}, {key}, {{{result}, {result}, 0}}}};
#if defined(SCRATCHV_CORES) && SCRATCHV_CORES > 1
  sv_parallel_for(2048ULL, w6_attention_range, &chain);
#else
  w6_attention_range(&chain, 0, 2048ULL);
#endif
  if (__atomic_load_n(&chain.probabilities.status, __ATOMIC_RELAXED)) return 2;"""
    first_op = generator.body.index("/* 0: transpose */")
    before_copy = generator.body.index("if (!output) return 1;")
    old_body = generator._external_prologue(plain.inputs, plain.output, plain.workspace_bytes) + generator.body
    new_body = (generator._external_prologue(plain.inputs, plain.output, arena_bytes)
                + generator.body[:first_op] + [dispatch] + generator.body[before_copy:])
    source = _replace_part(plain.source, "".join(generator.kernels), kernel)
    source = _replace_part(source, "\n  ".join(old_body), "\n  ".join(new_body))
    return replace(plain, source=source, workspace_bytes=arena_bytes)
