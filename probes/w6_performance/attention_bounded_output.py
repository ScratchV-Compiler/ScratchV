"""Write fixed Attention scores straight to the caller when all partial sums must be finite."""
from dataclasses import replace

from probes.w6_performance.attention_profile import replace_once
from scratchv.backend.tensor_c_codegen import TensorCArtifact
from scratchv.ir.types import DataType


def with_bounded_output(artifact: TensorCArtifact, *, minimum_magnitude_bits=0) -> TensorCArtifact:
    """Keep the workspace path outside the chosen lower bound and the fixed upper bound of 2**32."""
    if type(minimum_magnitude_bits) is not int or not 0 <= minimum_magnitude_bits <= 0x4f800000:
        raise ValueError("Invalid minimum FP32 magnitude for bounded output")
    shapes = [(spec.name, spec.shape, spec.dtype) for spec in artifact.inputs]
    if (shapes != [("x", (16, 256, 128), DataType.FLOAT32), ("weight", (16, 128, 256), DataType.FLOAT32)]
            or artifact.constant_storage != "external" or artifact.external_weights
            or artifact.qemu_workspace_layout != "none" or artifact.workspace_bytes != 4194304
            or artifact.output.shape != (16, 256, 256) or artifact.output.dtype != DataType.FLOAT32):
        raise ValueError("Bounded output is scoped to the fixed FP32 score MatMul")
    source = artifact.source
    # The generated ABI must rule out invalid or overlapping caller buffers before computing.
    entry = source.index("int scratchv_run_external(")
    scans = source.index("const float *sv_v0 =", entry)
    guards = source[entry:scans]
    for marker in ("if (!sv_buffer(output, 4194304ULL, 4)) return 1;",
                   "sv_overlap(workspace, 4194304ULL, output, 4194304ULL)",
                   "sv_overlap(output, 4194304ULL, inputs, 2ULL * sizeof(void *))",
                   "sv_overlap(output, 4194304ULL, inputs[0], 2097152ULL)",
                   "sv_overlap(output, 4194304ULL, inputs[1], 2097152ULL)"):
        if guards.count(marker) != 1:
            raise ValueError("Expected output alignment and alias checks before input scans")
    source = replace_once(source, "const float *sv_v0 =", "int bounded = 1;\n  const float *sv_v0 =")
    # Each product is at most 2**64. The bounds ±j*2**64, j=1..128, are
    # exactly representable in FP32. Ordered FMA and monotone rounding keep
    # every partial sum within its interval, including finite signed zeros.
    outside = "magnitude > 0x4f800000U"
    if minimum_magnitude_bits:
        outside = f"magnitude < 0x{minimum_magnitude_bits:08x}U || {outside}"
    for operand in ("sv_v0", "sv_v1"):
        original = f"for (size_t i=0; i<524288ULL; ++i) if (!sv_finite({operand}[i])) return 2;"
        scan = f"""for (size_t i = 0; i < 524288ULL; ++i) {{
    uint32_t magnitude = sv_float_bits({operand}[i]) & 0x7fffffffU;
    if ({outside}) {{
      if (magnitude >= 0x7f800000U) return 2;
      bounded = 0;
    }}
  }}"""
        source = replace_once(source, original, scan)
    source = replace_once(source, "float *sv_v2 = (float *)(sv_workspace + 0ULL);",
                          "float *sv_v2 = bounded ? (float *)output : (float *)(sv_workspace + 0ULL);")
    source = replace_once(source, "w6_publish_parallel(output, sv_v2, 4194304ULL);",
                          "if (!bounded) w6_publish_parallel(output, sv_v2, 4194304ULL);")
    return replace(artifact, source=source)
