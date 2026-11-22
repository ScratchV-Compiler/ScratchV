"""RVV 1.0 FP32 score tile for the fixed W6 dynamic Attention probe."""
import json


# RISC-V psABI: a0 = Q row, a1 = first K column, a2 = output row, a3 = columns left.
# All vector registers and ft0..ft3 are caller-saved under the ordinary scalar ABI.
# Scope V to this leaf: the remainder of the guest is compiled for the scalar target.
_TILE = """
.pushsection .text.w6_rvv_score_tile,"ax",@progbits
.option push
.option arch,+v
.p2align 2
.local w6_rvv_score_tile
.type w6_rvv_score_tile,@function
w6_rvv_score_tile:
    vsetvli a3, a3, e32, m4, ta, ma
    vmv.v.i v4, 0
    vmv.v.i v8, 0
    vmv.v.i v12, 0
    vmv.v.i v16, 0
    li t0, 128
.Lw6_rvv_score_k:
    vle32.v v0, (a1)
    flw ft0, 0(a0)
    flw ft1, 512(a0)
    flw ft2, 1024(a0)
    flw ft3, 1536(a0)
    vfmacc.vf v4, ft0, v0
    vfmacc.vf v8, ft1, v0
    vfmacc.vf v12, ft2, v0
    vfmacc.vf v16, ft3, v0
    addi a0, a0, 4
    addi a1, a1, 1024
    addi t0, t0, -1
    bnez t0, .Lw6_rvv_score_k
    fmv.w.x ft0, zero
    vfadd.vf v4, v4, ft0
    vfadd.vf v8, v8, ft0
    vfadd.vf v12, v12, ft0
    vfadd.vf v16, v16, ft0
    vse32.v v4, (a2)
    addi a2, a2, 1024
    vse32.v v8, (a2)
    addi a2, a2, 1024
    vse32.v v12, (a2)
    addi a2, a2, 1024
    vse32.v v16, (a2)
    mv a0, a3
    ret
.size w6_rvv_score_tile, .-w6_rvv_score_tile
.option pop
.popsection
"""

_RANGE = """
extern size_t w6_rvv_score_tile(const float *q, const float *key, float *out,
                                size_t columns_left);
struct w6_scores { const float *x, *rhs; float *output; };
static void w6_score_range(void *opaque, size_t begin, size_t end) {
    const struct w6_scores *ctx = opaque;
    // This bare-metal probe runs in M-mode. Enable VS on each core that executes it.
    // The normal boot path already initializes the FP rounding mode to nearest-even.
    __asm__ volatile ("csrs mstatus, %0" :: "r"(3u << 9) : "memory");
    for (size_t task = begin; task < end; ++task) {
        size_t head = task / 64;
        size_t row = (task % 64) * 4;
        const float *q = ctx->x + (head * 256 + row) * 128;
        const float *key = ctx->rhs + head * 128 * 256;
        float *out = ctx->output + (head * 256 + row) * 256;
        for (size_t col = 0; col < 256;) {
            // Each vector lane follows one output column through K=0..127.
            // The leaf returns the actual vl, including the final partial group.
            col += w6_rvv_score_tile(q, key + col, out + col, 256 - col);
        }
    }
}
"""


def rvv_score_kernel():
    """Fuse FP32 multiplies across columns, retaining scalar K order and final +0."""
    return f"__asm__({json.dumps(_TILE)});\n" + _RANGE
