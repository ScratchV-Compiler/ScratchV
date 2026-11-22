"""Four-core-capable FP32 microkernel for a tied, original-layout LM head."""

VOCAB_BLOCK = 32
MAX_HIDDEN = 1024  # The private [K,4] stack panel is at most 16 KiB.


def vocab_blocked_kernel(rows, vocabulary, hidden):
    return f"""
struct w6_head {{ const float *x, *weight; float *out; }};
static void w6_head_range(void *opaque, size_t begin, size_t end) {{
    const struct w6_head *ctx = opaque;
    const size_t rows = {rows}ULL, vocab = {vocabulary}ULL, hidden = {hidden}ULL;
    float panel[{hidden}][4];
    for (size_t block = begin; block < end; ++block) {{
        size_t first = block * {VOCAB_BLOCK}ULL;
        size_t stop = first + {VOCAB_BLOCK}ULL < vocab ? first + {VOCAB_BLOCK}ULL : vocab;
        for (size_t col = first; col < stop; col += 4) {{
            /* Contiguous reads from four unmodified vocabulary rows. */
            for (size_t k = 0; k < hidden; ++k) {{
                for (size_t lane = 0; lane < 4; ++lane) {{
                    panel[k][lane] = col + lane < vocab
                        ? ctx->weight[(col+lane)*hidden+k] : 0.0f;
                }}
            }}
            for (size_t row = 0; row < rows; row += 2) {{
                float sum0[4] = {{0}}, sum1[4] = {{0}};
                for (size_t k0 = 0; k0 < hidden; k0 += 128) {{
                    float acc0[4] = {{0}}, acc1[4] = {{0}};
                    size_t last = k0+128 < hidden ? k0+128 : hidden;
                    for (size_t k = k0; k < last; ++k) {{
                        float x0 = ctx->x[row*hidden+k];
                        float x1 = row+1 < rows ? ctx->x[(row+1)*hidden+k] : 0.0f;
                        for (size_t lane = 0; lane < 4; ++lane) {{
                            acc0[lane] = __builtin_fmaf(x0, panel[k][lane], acc0[lane]);
                            if (row+1 < rows)
                                acc1[lane] = __builtin_fmaf(x1, panel[k][lane], acc1[lane]);
                        }}
                    }}
                    for (size_t lane = 0; lane < 4; ++lane) {{
                        sum0[lane] += acc0[lane];
                        if (row+1 < rows) sum1[lane] += acc1[lane];
                    }}
                }}
                for (size_t lane = 0; lane < 4 && col+lane < vocab; ++lane) {{
                    ctx->out[row*vocab+col+lane] = sum0[lane];
                    if (row+1 < rows) ctx->out[(row+1)*vocab+col+lane] = sum1[lane];
                }}
            }}
        }}
    }}
}}
"""
