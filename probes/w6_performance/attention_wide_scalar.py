"""Compare 2×8 and 3×8 scalar tiles in the fixed four-core Attention score graph."""
from dataclasses import replace

from probes.w6_performance.attention_panel import _score_kernel, _score_tile, _shared_blocks
from probes.w6_performance.attention_profile import replace_once
from scratchv.backend.tensor_c_codegen import TensorCArtifact


def _wide_tile(rows):
    # Hold all Q values for this k while each successive K feeds every row.
    # With three rows only 24 independent sums need to survive between k steps.
    old = _score_tile(rows, "invalid |= !sv_finite(value);", True)
    original_loop = f"""            for (size_t k = 0; k < 128; ++k)
                for (size_t r = 0; r < {rows}; ++r)
                    for (size_t c = 0; c < 4; ++c)
                        acc[r][c] = __builtin_fmaf(x[r * 128 + k], panel[k][c], acc[r][c]);"""
    loop = f"""            for (size_t k = 0; k < 128; ++k) {{
                float query[{rows}];
                for (size_t r = 0; r < {rows}; ++r)
                    query[r] = x[r * 128 + k];
                for (size_t c = 0; c < 8; ++c) {{
                    float key = panel[k][c];
                    for (size_t r = 0; r < {rows}; ++r)
                        acc[r][c] = __builtin_fmaf(query[r], key, acc[r][c]);
                }}
            }}"""
    tile = replace_once(old, original_loop, loop)
    tile = replace_once(tile, f"acc[{rows}][4]", f"acc[{rows}][8]")
    return replace_once(tile, "c < 4", "c < 8")


def with_wide_scalar(artifact: TensorCArtifact, rows=3) -> TensorCArtifact:
    """Widen the checked 4×4 probe's K panel while preserving error handling and final copy."""
    if type(rows) is not int or rows not in (2, 3):
        raise ValueError("The wide Attention probe supports two or three rows")
    if (artifact.qemu_workspace_layout != "none" or artifact.external_weights
            or artifact.matmul_policy != "blocked_fma"
            or artifact.source.count("w6_publish_parallel(output, sv_v2, 4194304ULL);") != 1):
        raise ValueError("Expected the checked dynamic Attention probe with parallel output copy")
    wide = _score_kernel(check_output=True, row_tile=rows, cache_output=True)
    # Row 255 is a one-row tile for 3×8. The two-row kernel divides 256 exactly.
    heights = (3, 1) if rows == 3 else (2,)
    for height in heights:
        old_tile = _score_tile(height, "invalid |= !sv_finite(value);", True)
        wide = replace_once(wide, old_tile, _wide_tile(height))
    wide = replace_once(wide, "float panel[128][4];", "float panel[128][8];")
    wide = replace_once(wide, "task / 64, col = (task % 64) * 4", "task / 32, col = (task % 32) * 8")
    wide = replace_once(wide, "lane < 4", "lane < 8")
    claims = _shared_blocks(8)
    wide_claims = claims.replace("four-column blocks cover 128 bytes", "eight-column blocks cover 256 bytes")
    wide_claims = replace_once(wide_claims, "first >= 1024", "first >= 512")
    original = _score_kernel(check_output=True) + claims
    source = replace_once(artifact.source, original, wide + wide_claims)
    source = replace_once(source, "w6_score_range(&ctx.score, 0, 1024ULL);",
                          "w6_score_range(&ctx.score, 0, 512ULL);")
    return replace(artifact, source=source)
