"""Input-check alternatives for the fixed W6 four-core Attention score experiment."""
from dataclasses import replace

from probes.w6_performance.attention_profile import ARRAY, TIMER, replace_once
from scratchv.backend.tensor_c_codegen import TensorCArtifact
from scratchv.ir.types import DataType


_Q_SCAN = "  for (size_t i=0; i<524288ULL; ++i) if (!sv_finite(sv_v0[i])) return 2;"
_K_BINDING = "  if (!inputs || !inputs[1]) return 1;\n  const float *sv_v1 = (const float *)inputs[1];"
_K_SCAN = "  for (size_t i=0; i<524288ULL; ++i) if (!sv_finite(sv_v1[i])) return 2;"
_ORIGINAL = "\n".join((_Q_SCAN, _K_BINDING, _K_SCAN))
_ENTRY = "#if defined(_WIN32)\n__declspec(dllexport)\n#endif\nint scratchv_run_external("


def _parallel_helper(timed):
    start = "uint64_t start = w6_now();" if timed else ""
    q_done = "uint64_t q_done = w6_now();" if timed else ""
    k_done = "uint64_t k_done = w6_now();" if timed else ""
    record = "" if not timed else """
    size_t core;
    __asm__ volatile("csrr %0, mhartid" : "=r"(core));
    w6_ticks[core][0] = start;
    w6_ticks[core][1] = q_done;
    w6_ticks[core][2] = k_done;
    w6_ticks[core][3] = w6_now();
    w6_ticks[core][4] = k_ok ? end - begin : 0;"""
    return f"""
struct w6_input_check {{ const float *q, *k; unsigned error; }};
static int w6_input_slice_finite(const float *input, size_t begin, size_t end) {{
    for (size_t i = begin; i < end; ++i)
        if (!sv_finite(input[i])) return 0;
    return 1;
}}
static void w6_check_input_range(void *opaque, size_t begin, size_t end) {{
    struct w6_input_check *ctx = opaque;
    {start}
    int q_ok = w6_input_slice_finite(ctx->q, begin, end);
    {q_done}
    int k_ok = q_ok && w6_input_slice_finite(ctx->k, begin, end);
    {k_done}
    if (!k_ok) __atomic_store_n(&ctx->error, 2u, __ATOMIC_RELAXED);
    {record}
}}
"""


def with_input_checks(artifact: TensorCArtifact, *, parallel: bool, timed: bool = False) -> TensorCArtifact:
    """Change just the two pre-MatMul finite scans. All four workers join before MatMul."""
    inputs = [(s.name, s.shape, s.dtype) for s in artifact.inputs]
    if (artifact.constant_storage != "external" or artifact.function_name != "scratchv_run_external"
            or artifact.external_weights or artifact.qemu_workspace_layout != "none"
            or inputs != [("x", (16, 256, 128), DataType.FLOAT32), ("weight", (16, 128, 256), DataType.FLOAT32)]
            or artifact.output.shape != (16, 256, 256) or artifact.output.dtype != DataType.FLOAT32):
        raise ValueError("Expected the fixed dynamic FP32 Attention score graph with padding off")
    source = artifact.source
    for marker in (_ORIGINAL + "\n  /* 0: matmul */", "__atomic_fetch_add(&ctx->next, 8, __ATOMIC_RELAXED)",
                   "w6_score_range(&ctx->score, first, first + 8);", "invalid |= !sv_finite(value);",
                   "w6_publish_parallel(output, sv_v2, 4194304ULL);", _ENTRY):
        if source.count(marker) != 1:
            raise ValueError("Expected one checked eight-block Attention kernel with parallel output copy")
    if not parallel and not timed:
        return artifact

    if timed:
        source = replace_once(source, "#include <stdint.h>", "#include <stdint.h>\n" + ARRAY + TIMER)
    if parallel:
        source = replace_once(source, _ENTRY, _parallel_helper(timed) + _ENTRY)
        start = "uint64_t start = w6_now();\n  " if timed else ""
        stop = "w6_ticks[4][1] = w6_now();\n  w6_ticks[4][0] = start;\n  " if timed else ""
        scans = f"""  {start}{_K_BINDING.lstrip()}
  struct w6_input_check input_check = {{sv_v0, sv_v1, 0}};
#if SCRATCHV_CORES > 1
  sv_parallel_for(524288ULL, w6_check_input_range, &input_check);
#else
  w6_check_input_range(&input_check, 0, 524288ULL);
#endif
  unsigned status = __atomic_load_n(&input_check.error, __ATOMIC_RELAXED);
  {stop}if (status) return 2;"""
    else:
        scans = "\n".join(("  uint64_t start = w6_now();", _Q_SCAN,
                           "  uint64_t q_done = w6_now();", _K_BINDING,
                           "  uint64_t k_start = w6_now();", _K_SCAN,
                           "  uint64_t end = w6_now();",
                           "  w6_ticks[4][0] = start; w6_ticks[4][1] = end;",
                           "  w6_ticks[4][2] = q_done - start; w6_ticks[4][3] = end - k_start;"))
    return replace(artifact, source=replace_once(source, _ORIGINAL, scans))
