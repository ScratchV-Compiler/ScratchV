"""Optional fixed-shape GQA decode lowering; cache ownership stays with the graph."""
from dataclasses import dataclass

from scratchv.ir.types import DataType, OpCode, Value


@dataclass(frozen=True)
class DecodeAttention:
    query: Value
    keys: Value
    values: Value
    scale: Value
    mask: Value
    removed: frozenset[str]

    @property
    def inputs(self):
        return (self.query, self.keys, self.values, self.scale, self.mask)


def match_decode_attention(instructions):
    """Match only private repeat-K/V -> QK -> scale/mask -> Softmax -> PV chains.

    All original instructions are still type/shape checked by TensorCCodegen.
    A shared intermediate or a different layout retains the ordinary lowering.
    """
    producers = {i.dest.name: i for i in instructions if i.dest is not None}
    consumers = {}
    for i in instructions:
        for operand in i.operands:
            consumers.setdefault(operand.name, []).append(i)

    def node(value, opcode, count=None):
        candidate = producers.get(value.name)
        if (candidate is None or candidate.opcode != opcode
                or (count is not None and len(candidate.operands) != count)):
            raise ValueError
        return candidate

    def shape(value, expected):
        if value.dtype != DataType.FLOAT32 or tuple(value.shape) != expected:
            raise ValueError

    def repeat(value, removed):
        shape(value, (1, 16, 256, 128))
        reshape = node(value, OpCode.RESHAPE, 1)
        concat = node(reshape.operands[0], OpCode.CONCAT, 2)
        if concat.attrs.get('axis') != 2 or concat.operands[0].name != concat.operands[1].name:
            raise ValueError
        unsqueeze = node(concat.operands[0], OpCode.UNSQUEEZE, 1)
        if tuple(unsqueeze.attrs.get('axes', ())) != (2,):
            raise ValueError
        original = unsqueeze.operands[0]
        shape(original, (1, 8, 256, 128))
        shape(unsqueeze.dest, (1, 8, 1, 256, 128))
        shape(concat.dest, (1, 8, 2, 256, 128))
        removed.update(i.dest.name for i in (reshape, concat, unsqueeze))
        return original

    matches = {}
    for final in instructions:
        if final.opcode != OpCode.MATMUL or len(final.operands) != 2 or final.dest is None:
            continue
        try:
            shape(final.dest, (1, 16, 1, 128))
            removed = set()
            values = repeat(final.operands[1], removed)
            probability = final.operands[0]
            cast = producers.get(probability.name)
            if cast is not None and cast.opcode == OpCode.CAST:
                shape(cast.dest, (1, 16, 1, 256))
                if len(cast.operands) != 1 or cast.operands[0].dtype != DataType.FLOAT32:
                    raise ValueError
                removed.add(cast.dest.name)
                probability = cast.operands[0]
            softmax = node(probability, OpCode.SOFTMAX, 1)
            if softmax.attrs.get('axis', -1) not in (-1, 3):
                raise ValueError
            add = node(softmax.operands[0], OpCode.ADD, 2)
            mul = node(add.operands[0], OpCode.MUL, 2)
            score = node(mul.operands[0], OpCode.MATMUL, 2)
            transpose = node(score.operands[1], OpCode.TRANSPOSE, 1)
            if tuple(transpose.attrs.get('perm', ())) != (0, 1, 3, 2):
                raise ValueError
            keys = repeat(transpose.operands[0], removed)
            query, scale, mask = score.operands[0], mul.operands[1], add.operands[1]
            shape(query, (1, 16, 1, 128))
            shape(mask, (1, 1, 1, 256))
            if scale.dtype != DataType.FLOAT32 or tuple(scale.shape) not in ((), (1,)):
                raise ValueError
            for item in (score, mul, add, softmax):
                shape(item.dest, (1, 16, 1, 256))
            removed.update(i.dest.name for i in (transpose, score, mul, add, softmax))
            permitted = removed | {final.dest.name}
            if any(c.dest is None or c.dest.name not in permitted
                   for name in removed for c in consumers.get(name, ())):
                raise ValueError
            matches[final.dest.name] = DecodeAttention(query, keys, values, scale, mask,
                                                       frozenset(removed))
        except (ValueError, TypeError):
            continue
    return matches


def decode_attention_kernel(reference):
    """Same FP32 per-output order as blocked_fma/2x4, one task per KV head."""
    initialization = 'float lanes[16] = {0};' if reference else 'float sum = 0.0f;'
    exponential = 'sv_fp32_reference_exp_nonpositive' if reference else 'expf'
    accumulate = 'lanes[j % 16] += value;' if reference else 'sum += value;'
    finish = ('float sum = sv_fp32_reference_sum_sixteen(lanes); float reciprocal = 1.0f / sum;'
              if reference else '')
    normalize = '*= reciprocal' if reference else '/= sum'
    return r'''
struct sv_decode_attention {
    const float *query, *keys, *values, *scale, *mask;
    float *output, *scratch;
    int status[8];
};

static int sv_decode_attention_head(struct sv_decode_attention *ctx, size_t group) {
    const float *q = ctx->query + group * 256;
    const float *keys = ctx->keys + group * 256 * 128;
    const float *values = ctx->values + group * 256 * 128;
    float *prob = ctx->scratch + group * 512;
    float *output = ctx->output + group * 256;
    for (size_t pos = 0; pos < 256; pos += 4) {
        float a[4] = {0}, b[4] = {0};
        for (size_t d = 0; d < 128; ++d) {
            float q0 = q[d], q1 = q[128 + d];
            for (size_t lane = 0; lane < 4; ++lane) {
                float key = keys[(pos + lane) * 128 + d];
                a[lane] = __builtin_fmaf(q0, key, a[lane]);
                b[lane] = __builtin_fmaf(q1, key, b[lane]);
            }
        }
        for (size_t lane = 0; lane < 4; ++lane) {
            float scores[2] = {0.0f + a[lane], 0.0f + b[lane]};
            for (size_t row = 0; row < 2; ++row) {
                float value = scores[row];
                if (!sv_finite(value)) return 2;
                value *= *ctx->scale;
                if (!sv_finite(value)) return 2;
                value += ctx->mask[pos + lane];
                if (!sv_finite(value)) return 2;
                prob[row * 256 + pos + lane] = value;
            }
        }
    }
    for (size_t row = 0; row < 2; ++row) {
        float *p = prob + row * 256;
        float maximum = -0x1.fffffep+127f;
        @INITIALIZATION@
        for (size_t j = 0; j < 256; ++j) if (p[j] > maximum) maximum = p[j];
        for (size_t j = 0; j < 256; ++j) {
            float shifted = p[j] - maximum;
            if (!sv_finite(shifted)) return 2;
            float value = @EXPONENTIAL@(shifted);
            p[j] = value;
            @ACCUMULATE@
        }
        @FINISH@
        for (size_t j = 0; j < 256; ++j) {
            p[j] @NORMALIZE@;
            if (!sv_finite(p[j])) return 2;
        }
    }
    for (size_t column = 0; column < 128; column += 4) {
        float sums[8] = {0};
        for (size_t kb = 0; kb < 256; kb += 128) {
            float acc[8] = {0};
            for (size_t k = kb; k < kb + 128; ++k) {
                float p0 = prob[k], p1 = prob[256 + k];
                for (size_t lane = 0; lane < 4; ++lane) {
                    float value = values[k * 128 + column + lane];
                    acc[lane] = __builtin_fmaf(p0, value, acc[lane]);
                    acc[4 + lane] = __builtin_fmaf(p1, value, acc[4 + lane]);
                }
            }
            for (size_t lane = 0; lane < 8; ++lane) sums[lane] += acc[lane];
        }
        for (size_t row = 0; row < 2; ++row) for (size_t lane = 0; lane < 4; ++lane) {
            float value = sums[row * 4 + lane];
            if (!sv_finite(value)) return 2;
            output[row * 128 + column + lane] = value;
        }
    }
    return 0;
}

static void sv_decode_attention_range(void *opaque, size_t begin, size_t end) {
    struct sv_decode_attention *ctx = opaque;
    for (size_t group = begin; group < end; ++group)
        ctx->status[group] = sv_decode_attention_head(ctx, group);
}

static __attribute__((noinline)) int sv_decode_attention_run(float *output, float *scratch,
        const float *query, const float *keys, const float *values,
        const float *scale, const float *mask) {
    struct sv_decode_attention ctx = {query, keys, values, scale, mask, output, scratch, {0}};
#if defined(SCRATCHV_CORES) && SCRATCHV_CORES > 1
    sv_parallel_for(8, sv_decode_attention_range, &ctx);
#else
    sv_decode_attention_range(&ctx, 0, 8);
#endif
    for (size_t group = 0; group < 8; ++group) if (ctx.status[group]) return ctx.status[group];
    return 0;
}
'''.replace('@INITIALIZATION@', initialization).replace('@EXPONENTIAL@', exponential).replace(
        '@ACCUMULATE@', accumulate).replace('@FINISH@', finish).replace('@NORMALIZE@', normalize)
