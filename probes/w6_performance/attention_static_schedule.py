"""Assign contiguous heads to each core of the W6 checked score kernel."""
from dataclasses import replace

from probes.w6_performance.attention_panel import _shared_blocks
from scratchv.backend.tensor_c_codegen import TensorCArtifact


def with_static_schedule(artifact: TensorCArtifact) -> TensorCArtifact:
    """Use the runtime's fixed partition of the same 1,024 four-column tasks."""
    dynamic_dispatch = """struct w6_shared_scores ctx = {{sv_v0, sv_v1, sv_v2, 0}, 0};
#if SCRATCHV_CORES > 1
sv_parallel_for(4ULL, w6_claim_scores, &ctx);
#else
w6_score_range(&ctx.score, 0, 1024ULL);
#endif

  if (__atomic_load_n(&ctx.score.error, __ATOMIC_RELAXED)) return 2;"""
    static_dispatch = """struct w6_scores ctx = {sv_v0, sv_v1, sv_v2, 0};
#if SCRATCHV_CORES > 1
sv_parallel_for(1024ULL, w6_score_range, &ctx);
#else
w6_score_range(&ctx, 0, 1024ULL);
#endif

  if (__atomic_load_n(&ctx.error, __ATOMIC_RELAXED)) return 2;"""
    shared = _shared_blocks(8)
    source = artifact.source
    if source.count(shared) != 1 or source.count(dynamic_dispatch) != 1:
        raise ValueError("Expected the checked W6 Attention graph with eight-block claims")
    source = source.replace(shared, "", 1).replace(dynamic_dispatch, static_dispatch, 1)
    return replace(artifact, source=source)
