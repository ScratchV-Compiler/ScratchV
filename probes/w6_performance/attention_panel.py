"""Alternative kernels for the W6 fixed-shape, dynamic Attention score MatMul."""
from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorCCodegenError
from scratchv.ir.types import DataType, OpCode


class RuntimeScorePanel(TensorCCodegen):
    """Generate a probe-only kernel, preserving the regular guest ABI and output checks."""
    def __init__(self, *args, loop_variant="original", claim_blocks=8, row_tile=4, cache_output=False, **kwargs):
        if loop_variant not in ("original", "k8", "head-panels", "shared-blocks", "shared-checked",
                                "shared-claim-panels", "rvv"):
            raise ValueError(f"Unknown W6 score loop: {loop_variant}")
        if type(claim_blocks) is not int or claim_blocks not in (4, 8, 16):
            raise ValueError("W6 score claims must contain 4, 8 or 16 blocks")
        if claim_blocks != 8 and loop_variant not in ("shared-blocks", "shared-checked"):
            raise ValueError("Custom block claims require a shared score loop")
        if type(row_tile) is not int or row_tile not in (4, 5, 6) or type(cache_output) is not bool:
            raise ValueError("W6 score kernels accept 4, 5 or 6 rows per tile and a boolean cache_output")
        if row_tile == 6 and (loop_variant != "shared-checked" or not cache_output):
            raise ValueError("The six-row score kernel requires cached output addresses and shared checked blocks")
        if loop_variant == "shared-claim-panels":
            if (row_tile, cache_output) not in ((4, False), (5, True)):
                raise ValueError("Claim panels support 4 rows or 5 rows with cached output addresses")
        elif (row_tile != 4 or cache_output) and loop_variant != "shared-checked":
            raise ValueError("Row tile and output address variants require the shared checked kernel")
        super().__init__(*args, **kwargs)
        self.loop_variant = loop_variant
        self.claim_blocks = claim_blocks
        self.row_tile = row_tile
        self.cache_output = cache_output

    def _emit(self, instruction, output, details):
        if instruction.opcode != OpCode.MATMUL:
            return super()._emit(instruction, output, details)
        left, right = instruction.operands
        if (tuple(left.shape) != (16, 256, 128) or tuple(right.shape) != (16, 128, 256)
                or left.dtype != DataType.FLOAT32 or right.dtype != DataType.FLOAT32
                or left.name not in {"x"} or right.name != "weight"
                or {p.name for p in self.program.functions[0].params} != {left.name, right.name}):
            raise TensorCCodegenError("W6 score panel expects the two fixed runtime FP32 inputs")
        a, b = (self.names[v.name] for v in instruction.operands)
        fields = f"{a}, {b}, {self.names[output.name]}"
        checked = self.loop_variant in ("shared-checked", "shared-claim-panels")
        if checked or self.loop_variant == "shared-blocks":
            if self.loop_variant == "shared-claim-panels":
                from probes.w6_performance.attention_claim_panel import checked_claim_panel_kernel
                kernel = checked_claim_panel_kernel(row_tile=self.row_tile, cache_output=self.cache_output)
            else:
                kernel = _score_kernel(check_output=checked, row_tile=self.row_tile, cache_output=self.cache_output)
            self.kernels.append(kernel + _shared_blocks(self.claim_blocks))
            score_fields = fields + ", 0" if checked else fields
            context = f"struct w6_shared_scores ctx = {{{{{score_fields}}}, 0}};"
            parallel = "sv_parallel_for(4ULL, w6_claim_scores, &ctx);"
            serial = "w6_score_range(&ctx.score, 0, 1024ULL);"
        else:
            if self.loop_variant == "rvv":
                from probes.w6_performance.attention_rvv import rvv_score_kernel
                kernel = rvv_score_kernel()
            else:
                kernel = _score_kernel() if self.loop_variant == "original" else _candidate_kernel(self.loop_variant)
            self.kernels.append(kernel)
            tasks = 16 if self.loop_variant == "head-panels" else 1024
            context = f"struct w6_scores ctx = {{{fields}}};"
            parallel = f"sv_parallel_for({tasks}ULL, w6_score_range, &ctx);"
            serial = f"w6_score_range(&ctx, 0, {tasks}ULL);"
        self.body.append(f"{context}\n#if SCRATCHV_CORES > 1\n{parallel}\n#else\n{serial}\n#endif\n")
        if checked:
            # sv_parallel_for joins all writers before the caller can publish the output.
            self.body.append("if (__atomic_load_n(&ctx.score.error, __ATOMIC_RELAXED)) return 2;")
        return checked  # Other kernels retain the normal full-output finite check.


def _score_tile(rows, record_error, cache_output):
    address = "ctx->output[((head * 256 + row + r) * 256) + col + c]"
    row_pointer = row_end = ""
    if cache_output:
        row_pointer = " {\n                float *const out_row = output + ((head * 256 + row + r) * 256) + col;"
        address = "out_row[c]"
        row_end = "\n            }"
    return f"""            float acc[{rows}][4] = {{{{0}}}};
            const float *x = ctx->x + (head * 256 + row) * 128;
            for (size_t k = 0; k < 128; ++k)
                for (size_t r = 0; r < {rows}; ++r)
                    for (size_t c = 0; c < 4; ++c)
                        acc[r][c] = __builtin_fmaf(x[r * 128 + k], panel[k][c], acc[r][c]);
            for (size_t r = 0; r < {rows}; ++r){row_pointer}
                for (size_t c = 0; c < 4; ++c) {{
                    float value = 0.0f + acc[r][c];
                    {record_error}
                    {address} = value;
                }}{row_end}"""


def _score_kernel(*, check_output=False, row_tile=4, cache_output=False):
    """Per-worker 2 KiB panel; optionally check each score before it leaves registers."""
    error_field = " unsigned error;" if check_output else ""
    context_type = "struct w6_scores" if check_output else "const struct w6_scores"
    local_error = "unsigned invalid = 0;" if check_output else ""
    record_error = "invalid |= !sv_finite(value);" if check_output else ""
    publish_error = ("if (invalid) __atomic_fetch_or(&ctx->error, 2u, __ATOMIC_RELAXED);"
                     if check_output else "")
    output_base = "\n    float *const output = ctx->output;" if cache_output else ""
    row_condition = "row < 256" if row_tile == 4 else f"row + {row_tile} <= 256"
    tile_loop = f"""        for (size_t row = 0; {row_condition}; row += {row_tile}) {{
{_score_tile(row_tile, record_error, cache_output)}
        }}"""
    tail_rows = 256 % row_tile
    if tail_rows:
        tile_loop += f"""
        {{
            const size_t row = {256 - tail_rows};
{_score_tile(tail_rows, record_error, cache_output)}
        }}"""
    return f"""
struct w6_scores {{ const float *x, *rhs; float *output;{error_field} }};
static void w6_score_range(void *opaque, size_t begin, size_t end) {{
    {context_type} *ctx = opaque;{output_base}
    float panel[128][4];
    {local_error}
    for (size_t task = begin; task < end; ++task) {{
        size_t head = task / 64, col = (task % 64) * 4;
        const float *rhs = ctx->rhs + head * 128 * 256 + col;
        for (size_t k = 0; k < 128; ++k)
            for (size_t lane = 0; lane < 4; ++lane)
                panel[k][lane] = rhs[k * 256 + lane];
{tile_loop}
    }}
    {publish_error}
}}
"""


_SHARED_BLOCKS = """
struct w6_shared_scores { struct w6_scores score; size_t next; };
#if SCRATCHV_CORES > 1
static void w6_claim_scores(void *opaque, size_t begin, size_t end) {
    struct w6_shared_scores *ctx = opaque;
    (void)begin;
    (void)end;
    for (;;) {
        // Eight neighboring four-column blocks cover 128 bytes in each output row.
        size_t first = __atomic_fetch_add(&ctx->next, 8, __ATOMIC_RELAXED);
        if (first >= 1024) return;
        w6_score_range(&ctx->score, first, first + 8);
    }
}
#endif
"""


def _shared_blocks(claim_blocks):
    if claim_blocks == 8:
        return _SHARED_BLOCKS
    # Every supported claim divides the fixed graph's 1,024 four-column blocks.
    return (_SHARED_BLOCKS
            .replace("Eight neighboring four-column blocks cover 128 bytes",
                     f"{claim_blocks} neighboring four-column blocks cover {claim_blocks * 16} bytes")
            .replace("&ctx->next, 8,", f"&ctx->next, {claim_blocks},")
            .replace("first + 8", f"first + {claim_blocks}"))


def _candidate_kernel(variant):
    """Both candidates keep a separate FP32 accumulator for each output score."""
    unroll = ""
    if variant == "k8":
        unroll = """#if defined(__clang__)
#pragma clang loop unroll_count(8)
#elif defined(__GNUC__)
#pragma GCC unroll 8
#endif"""
    query_index = "k*4 + r" if variant == "head-panels" else "r*128 + k"
    tile = f"""
struct w6_scores {{ const float *x, *rhs; float *output; }};
static inline __attribute__((always_inline)) void w6_score_tile(
        const float *query, const float *key_panel, float *out) {{
    float sums[4][4] = {{{{0}}}};
{unroll}
    for (size_t k = 0; k < 128; ++k) {{
        for (size_t r = 0; r < 4; ++r) {{
            float q = query[{query_index}];
            for (size_t c = 0; c < 4; ++c)
                sums[r][c] = __builtin_fmaf(q, key_panel[k*4 + c], sums[r][c]);
        }}
    }}
    for (size_t r = 0; r < 4; ++r)
        for (size_t c = 0; c < 4; ++c)
            out[r*256 + c] = 0.0f + sums[r][c];
}}
"""
    if variant == "k8":
        return tile + """
static void w6_score_range(void *opaque, size_t begin, size_t end) {
    const struct w6_scores *ctx = opaque;
    const float *queries = ctx->x, *keys = ctx->rhs;
    float *output = ctx->output;
    float panel[128*4];
    for (size_t task = begin; task < end; ++task) {
        size_t head = task / 64, col = (task % 64) * 4;
        const float *rhs = keys + head*128*256 + col;
        for (size_t k = 0; k < 128; ++k)
            for (size_t c = 0; c < 4; ++c)
                panel[k*4 + c] = rhs[k*256 + c];
        for (size_t row = 0; row < 256; row += 4)
            w6_score_tile(queries + (head*256 + row)*128, panel,
                          output + (head*256 + row)*256 + col);
    }
}
"""
    return tile + """
static void w6_score_range(void *opaque, size_t begin, size_t end) {
    const struct w6_scores *ctx = opaque;
    const float *queries = ctx->x, *keys = ctx->rhs;
    float *output = ctx->output;
    // Each worker owns its heads and reuses 130 KiB of its 1 MiB stack.
    float key_panels[64][128*4], query_panel[128*4];
    for (size_t head = begin; head < end; ++head) {
        const float *q = queries + head*256*128;
        const float *b = keys + head*128*256;
        float *y = output + head*256*256;
        for (size_t g = 0; g < 64; ++g)
            for (size_t k = 0; k < 128; ++k)
                for (size_t c = 0; c < 4; ++c)
                    key_panels[g][k*4 + c] = b[k*256 + g*4 + c];
        for (size_t row = 0; row < 256; row += 4) {
            for (size_t k = 0; k < 128; ++k)
                for (size_t r = 0; r < 4; ++r)
                    query_panel[k*4 + r] = q[(row+r)*128 + k];
            for (size_t g = 0; g < 64; ++g)
                w6_score_tile(query_panel, key_panels[g], y + row*256 + g*4);
        }
    }
}
"""
