"""Optional W6 comparison of how the verified output is copied to its caller."""
from dataclasses import replace
import re

from scratchv.backend.tensor_c_codegen import TensorCArtifact


_COPY = r"""
#if defined(__clang__)
#define W6_COPY_BLOCK __builtin_memcpy_inline
#else
#define W6_COPY_BLOCK __builtin_memcpy
#endif
static void w6_publish_output(void *destination, const void *source, size_t bytes) {
    unsigned char *out = destination;
    const unsigned char *in = source;
    if (!(((uintptr_t)out | (uintptr_t)in) & 7)) {
        for (; bytes >= 64; out += 64, in += 64, bytes -= 64)
            W6_COPY_BLOCK(__builtin_assume_aligned(out, 8), __builtin_assume_aligned(in, 8), 64);
    } else if (!(((uintptr_t)out | (uintptr_t)in) & 3)) {
        for (; bytes >= 64; out += 64, in += 64, bytes -= 64)
            W6_COPY_BLOCK(__builtin_assume_aligned(out, 4), __builtin_assume_aligned(in, 4), 64);
    }
    for (; bytes; --bytes)
        *out++ = *in++;
}
#undef W6_COPY_BLOCK
"""


_PARALLEL_COPY = r"""
#if defined(SCRATCHV_CORES) && SCRATCHV_CORES > 1
struct w6_publish_context {
    unsigned char *destination;
    const unsigned char *source;
};
static void w6_publish_range(void *opaque, size_t begin, size_t end) {
    const struct w6_publish_context *ctx = opaque;
    w6_publish_output(ctx->destination + begin * 64, ctx->source + begin * 64,
                      (end - begin) * 64);
}
#endif
static void w6_publish_parallel(void *destination, const void *source, size_t bytes) {
#if defined(SCRATCHV_CORES) && SCRATCHV_CORES > 1
    size_t blocks = bytes / 64;
    struct w6_publish_context ctx = {destination, source};
    sv_parallel_for(blocks, w6_publish_range, &ctx);
    size_t copied = blocks * 64;
    w6_publish_output((unsigned char *)destination + copied,
                      (const unsigned char *)source + copied, bytes - copied);
#else
    w6_publish_output(destination, source, bytes);
#endif
}
"""


def with_block_output(artifact: TensorCArtifact, *, parallel=False) -> TensorCArtifact:
    """Replace only the final, nonoverlapping copy after successful execution."""
    if artifact.constant_storage != "external" or artifact.function_name != "scratchv_run_external":
        raise ValueError("Block output requires the checked external tensor ABI")
    spec = artifact.output
    ctype = {"float32": "float", "int32": "int32_t", "int64": "int64_t"}[spec.numpy_dtype.name]
    entry = "#if defined(_WIN32)\n__declspec(dllexport)\n#endif\nint scratchv_run_external("
    if artifact.source.count(entry) != 1:
        raise ValueError("Missing or ambiguous external entry point")
    start = artifact.source.index(entry)
    assignment = rf"for \(size_t i=0; i<{spec.size}ULL; \+\+i\) \(\({ctype} \*\)output\)\[i\] = (\w+)\[i\];"
    tail = re.compile(r"  if \(!output\) return 1;\n  " + assignment + r"\n  return 0;\n}\n\Z")
    match = tail.search(artifact.source, start)
    if match is None:
        raise ValueError("Expected a checked final output copy matching the tensor shape and dtype")
    helper = _COPY + (_PARALLEL_COPY if parallel else "")
    function = "w6_publish_parallel" if parallel else "w6_publish_output"
    source = (artifact.source[:start] + helper + artifact.source[start:match.start()]
              + f"  if (!output) return 1;\n  {function}(output, {match[1]}, {spec.nbytes}ULL);\n"
              + "  return 0;\n}\n")
    return replace(artifact, source=source)
