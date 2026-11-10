"""Fuse the first five output rows with packing a four-column runtime K panel."""
from dataclasses import replace

from probes.w6_performance.attention_panel import _score_kernel, _score_tile
from probes.w6_performance.attention_profile import replace_once


def with_first_pass_packing(artifact):
    """The first tile consumes each K value from a local while also saving it for later rows."""
    kernel = _score_kernel(check_output=True, row_tile=5, cache_output=True)
    pack_and_start = """        for (size_t k = 0; k < 128; ++k)
            for (size_t lane = 0; lane < 4; ++lane)
                panel[k][lane] = rhs[k * 256 + lane];
        for (size_t row = 0; row + 5 <= 256; row += 5) {"""
    tile = _score_tile(5, "invalid |= !sv_finite(value);", True)
    k_start = "            for (size_t k = 0; k < 128; ++k)"
    fma = "                        acc[r][c] = __builtin_fmaf(x[r * 128 + k], panel[k][c], acc[r][c]);"
    first_tile = replace_once(tile, k_start, k_start + """ {
                float key[4];
                for (size_t c = 0; c < 4; ++c) {
                    key[c] = rhs[k * 256 + c];
                    panel[k][c] = key[c];
                }""")
    first_tile = replace_once(first_tile, fma, fma.replace("panel[k][c]", "key[c]") + "\n            }")
    replacement = ("        {\n            const size_t row = 0;\n" + first_tile
                   + "\n        }\n        for (size_t row = 5; row + 5 <= 256; row += 5) {")
    fused = replace_once(kernel, pack_and_start, replacement)
    return replace(artifact, source=replace_once(artifact.source, kernel, fused))
