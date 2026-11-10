"""Pack one eight-block claim for the fixed W6 Attention score comparison."""
from string import Template
from textwrap import indent

from probes.w6_performance.attention_panel import _score_tile


_SOURCE = Template("""
struct w6_scores { const float *x, *rhs; float *output; unsigned error; };
static void w6_score_range(void *opaque, size_t begin, size_t end) {
    struct w6_scores *ctx = opaque;$output_base
    float panels[8][128][4];
    unsigned invalid = 0;
    // A head has 64 four-column blocks. Every aligned eight-block claim fits in one head.
    for (size_t first = begin; first < end; first += 8) {
        size_t head = first / 64, $claim_col = (first % 64) * 4;
        const float *rhs = ctx->rhs + head * 128 * 256 + $claim_col;
        for (size_t k = 0; k < 128; ++k)
            for (size_t block = 0; block < 8; ++block)
                for (size_t c = 0; c < 4; ++c)
                    panels[block][k][c] = rhs[k * 256 + block * 4 + c];
$compute
    }
    if (invalid) __atomic_fetch_or(&ctx->error, 2u, __ATOMIC_RELAXED);
}
""")

_FOUR_ROW_LOOP = """        for (size_t row = 0; row < 256; row += 4) {
            const float *x = ctx->x + (head * 256 + row) * 128;
            for (size_t block = 0; block < 8; ++block) {
                float acc[4][4] = {{0}};
                for (size_t k = 0; k < 128; ++k)
                    for (size_t r = 0; r < 4; ++r)
                        for (size_t c = 0; c < 4; ++c)
                            acc[r][c] = __builtin_fmaf(x[r * 128 + k], panels[block][k][c], acc[r][c]);
                for (size_t r = 0; r < 4; ++r)
                    for (size_t c = 0; c < 4; ++c) {
                        float value = 0.0f + acc[r][c];
                        invalid |= !sv_finite(value);
                        ctx->output[((head * 256 + row + r) * 256) + col + block * 4 + c] = value;
                    }
            }
        }"""


def _cached_panels(rows):
    tile = _score_tile(rows, "invalid |= !sv_finite(value);", cache_output=True)
    return f"""            for (size_t block = 0; block < 8; ++block) {{
                const size_t col = claim_col + block * 4;
                const float (*panel)[4] = panels[block];
{indent(tile, "    ")}
            }}"""


def checked_claim_panel_kernel(row_tile=4, cache_output=False):
    """Pack each claim once and visit all eight panels before advancing the Q rows."""
    if row_tile == 4 and not cache_output:
        return _SOURCE.substitute(claim_col="col", output_base="", compute=_FOUR_ROW_LOOP)
    if row_tile != 5 or not cache_output:
        raise ValueError("W6 claim panels support plain 4x4 or 5x4 with cached output")
    compute = f"""        for (size_t row = 0; row + 5 <= 256; row += 5) {{
{_cached_panels(5)}
        }}
        {{
            const size_t row = 255;
{_cached_panels(1)}
        }}"""
    return _SOURCE.substitute(claim_col="claim_col", output_base="\n    float *const output = ctx->output;",
                              compute=compute)
