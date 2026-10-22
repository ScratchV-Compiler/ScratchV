"""External weights and caller workspace: execute the generated host C ABI.

Set SCRATCHV_ZIG to a Zig executable, or install a host clang/gcc. Executable
checks skip explicitly if no compiler is available; graph checks always run.
"""

from contextlib import contextmanager
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import ctypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorCCodegenError
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Value


ROOT = Path(__file__).resolve().parents[1]
POINTERS = ctypes.POINTER(ctypes.c_void_p)


def graph(shape=(2, 3)):
    x = Value("x", shape=shape)
    w, bias = Value("weight", shape=(3, 4)), Value("bias", shape=(4,))
    builder = IRBuilder()
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.program.global_values.extend([w, bias])
    literal = builder.make_const(0.5)
    builder.ret(builder.mul(builder.add(builder.matmul(x, w), bias), literal))
    bindings = {
        "weight": np.arange(12, dtype=np.float32).reshape(3, 4) / 8,
        "bias": np.array([0.25, -0.5, 0.75, -1.0], dtype=np.float32),
    }
    return builder.program, bindings, literal


@pytest.fixture(scope="module")
def host_cc():
    zig = os.environ.get("SCRATCHV_ZIG")
    if zig:
        return [zig, "cc"]
    for name in ("clang", "gcc", "cc"):
        found = shutil.which(name)
        if found:
            return [found]
    pytest.skip("Host C execution requires clang/gcc or SCRATCHV_ZIG")


@contextmanager
def library(tmp_path, artifact, compiler):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "model.c"
    source.write_text(artifact.source, encoding="utf-8")
    binary = tmp_path / ("model.dll" if sys.platform == "win32" else "model.so")
    environment = dict(os.environ)
    environment.setdefault("ZIG_GLOBAL_CACHE_DIR", str(ROOT / "output/zig-global-cache"))
    environment.setdefault("ZIG_LOCAL_CACHE_DIR", str(tmp_path / "zig-cache"))
    command = [*compiler, "-shared", "-O2", "-std=c11", *artifact.compile_flags]
    if sys.platform != "win32":
        command.append("-fPIC")
    command += [str(source), "-o", str(binary), "-lm"]
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=180, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    loaded = ctypes.CDLL(str(binary))
    function = getattr(loaded, artifact.function_name)
    function.restype = ctypes.c_int
    function.argtypes = ([POINTERS, POINTERS, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
                         if artifact.constant_storage == "external"
                         else [POINTERS, ctypes.c_void_p])
    try:
        yield function
    finally:
        if sys.platform == "win32":
            import _ctypes
            _ctypes.FreeLibrary(loaded._handle)


def pointers(arrays):
    return (ctypes.c_void_p * len(arrays))(*(array.ctypes.data for array in arrays))


def arguments(artifact, x):
    # Model packaging may receive strided initializers: materialize C-order only
    # here at the ABI boundary, never silently reinterpret their underlying bytes.
    weights = [np.array(array, copy=True, order="C") for array in artifact.external_initializers]
    workspace = np.empty(max(1, (artifact.workspace_bytes + 7) // 8), dtype=np.uint64)
    output = np.full(artifact.output.shape, -999, dtype=artifact.output.numpy_dtype)
    inputs = [x[spec.name] for spec in artifact.inputs] if isinstance(x, dict) else [x]
    return weights, workspace, output, [pointers(inputs), pointers(weights),
                                       workspace.ctypes.data, artifact.workspace_bytes,
                                       output.ctypes.data]


@pytest.mark.parametrize("shape", [(2, 3), (2, 1, 3)])
@pytest.mark.parametrize("kernel_calls", [False, True])
def test_external_matches_inline_and_numpy(tmp_path, host_cc, shape, kernel_calls):
    program, bindings, _ = graph(shape)
    inline = TensorCCodegen(program, bindings).generate()
    external = TensorCCodegen(program, bindings, constant_storage="external", kernel_calls=kernel_calls).generate()
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 4
    weights, workspace, output, args = arguments(external, x)
    expected = (x @ bindings["weight"] + bindings["bias"]) * np.float32(0.5)
    inline_output = np.empty_like(output)
    with library(tmp_path / "inline", inline, host_cc) as run:
        assert run(pointers([x]), inline_output.ctypes.data) == 0
    with library(tmp_path / "external", external, host_cc) as run:
        assert run(*args) == 0
        np.testing.assert_array_equal(output, inline_output)
        np.testing.assert_allclose(output, expected, rtol=0, atol=1e-6)
        # Runtime data is not baked into the generated function or cached.
        weights[0] *= np.float32(2)
        assert run(*args) == 0
        np.testing.assert_allclose(output, (x @ weights[0] + weights[1]) * weights[2],
                                   rtol=0, atol=1e-6)
    assert external.workspace_bytes == inline.workspace_bytes


@pytest.mark.parametrize("kernel_calls", [False, True])
def test_external_abi_rejects_malformed_buffers_before_writing(tmp_path, host_cc, kernel_calls):
    program, bindings, _ = graph()
    artifact = TensorCCodegen(program, bindings, constant_storage="external", kernel_calls=kernel_calls).generate()
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    weights, workspace, output, valid = arguments(artifact, x)
    bad_weight = (ctypes.c_void_p * 3)(0, weights[1].ctypes.data, weights[2].ctypes.data)
    unaligned_weight = (ctypes.c_void_p * 3)(weights[0].ctypes.data + 1,
                                            weights[1].ctypes.data, weights[2].ctypes.data)
    bad_input = (ctypes.c_void_p * 1)(x.ctypes.data + 1)
    overflowing_input = (ctypes.c_void_p * 1)((1 << 64) - 4)
    cases = [
        (0, None), (1, None), (2, None), (3, artifact.workspace_bytes - 1),
        (2, workspace.ctypes.data + 1), (4, None), (4, output.ctypes.data + 1),
        (0, bad_input), (0, overflowing_input), (1, bad_weight), (1, unaligned_weight),
        (4, x.ctypes.data), (4, weights[0].ctypes.data), (4, workspace.ctypes.data),
        (2, weights[0].ctypes.data), (2, x.ctypes.data),
        (4, ctypes.addressof(valid[1])),
    ]
    original_input = x.copy()
    with library(tmp_path, artifact, host_cc) as run:
        for argument_index, replacement in cases:
            args = list(valid)
            args[argument_index] = replacement
            assert run(*args) == 1, (argument_index, replacement)
            assert np.all(output == -999)
            np.testing.assert_array_equal(x, original_input)
        assert run(*valid) == 0


@pytest.mark.parametrize("kernel_calls", [False, True])
def test_external_runtime_checks_finite_weights_and_scalar_literals(tmp_path, host_cc, kernel_calls):
    program, bindings, _ = graph()
    artifact = TensorCCodegen(program, bindings, constant_storage="external", kernel_calls=kernel_calls).generate()
    x = np.ones((2, 3), np.float32)
    weights, workspace, output, args = arguments(artifact, x)
    with library(tmp_path, artifact, host_cc) as run:
        weights[0][0, 0] = np.nan
        assert run(*args) == 2
        weights[0][0, 0] = bindings["weight"][0, 0]
        weights[2][()] = 0.75
        assert run(*args) == 2
        weights[2][()] = 0.5
        assert run(*args) == 0


def test_external_runs_with_independent_workspaces(tmp_path, host_cc):
    program, bindings, _ = graph()
    artifact = TensorCCodegen(program, bindings, constant_storage="external").generate()
    with library(tmp_path, artifact, host_cc) as run:
        def execute(index):
            x = np.full((2, 3), index / 8, np.float32)
            weights, workspace, output, args = arguments(artifact, x)
            assert run(*args) == 0
            np.testing.assert_array_equal(output, (x @ bindings["weight"] + bindings["bias"]) * 0.5)
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(execute, range(32)))


def test_external_return_input_needs_no_weights_or_workspace(tmp_path, host_cc):
    x = Value("x", shape=(3,))
    builder = IRBuilder()
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.ret(x)
    artifact = TensorCCodegen(builder.program, constant_storage="external").generate()
    assert not artifact.external_weights and artifact.workspace_bytes == 0
    data, output = np.array([1, 2, 3], np.float32), np.zeros(3, np.float32)
    with library(tmp_path, artifact, host_cc) as run:
        assert run(pointers([data]), None, None, 0, output.ctypes.data) == 0
    np.testing.assert_array_equal(output, data)


@pytest.mark.parametrize("dtype,numpy_dtype", [(DataType.FLOAT32, np.float32),
                                              (DataType.INT32, np.int32),
                                              (DataType.INT64, np.int64)])
def test_return_only_external_literal_and_zero_workspace(tmp_path, host_cc, dtype, numpy_dtype):
    builder = IRBuilder()
    builder.new_function("main", [])
    builder.new_block("entry")
    builder.ret(builder.make_const(3, dtype))
    artifact = TensorCCodegen(builder.program, constant_storage="external").generate()
    assert artifact.workspace_bytes == 0
    weight = np.array(3, dtype=numpy_dtype)
    output = np.array(0, dtype=numpy_dtype)
    with library(tmp_path, artifact, host_cc) as run:
        assert run(None, pointers([weight]), None, 0, output.ctypes.data) == 0
        assert output.item() == 3
        weight[()] = 4
        assert run(None, pointers([weight]), None, 0, output.ctypes.data) == 2


def test_external_preserves_initializer_order_and_storage_identity():
    program, bindings, literal = graph()
    bindings["weight"] = np.arange(24, dtype=np.float32).reshape(3, 8)[:, ::2]
    before = program.dump()
    codegen = TensorCCodegen(program, bindings, constant_storage="external")
    first, second = codegen.generate(), codegen.generate()
    assert first == second
    assert program.dump() == before
    assert [spec.name for spec in first.external_weights] == ["weight", "bias", literal.name]
    assert first.external_initializers[0] is bindings["weight"]
    assert first.external_initializers[1] is bindings["bias"]
    assert first.external_initializers[2].shape == ()
    assert "array(" not in repr(first)
    assert "sv_arena" not in first.source
    assert "static const float" not in first.source
    assert "scratchv_run_external" in first.source


def test_inline_default_is_unchanged_by_opt_in():
    program, bindings, _ = graph()
    default = TensorCCodegen(program, bindings).generate()
    explicit = TensorCCodegen(program, bindings, constant_storage="inline").generate()
    assert default == explicit
    assert default.function_name == "scratchv_run"
    assert default.constant_storage == "inline"
    assert not default.external_weights and not default.external_initializers
    assert "static const float" in default.source
    assert "sv_arena" in default.source
    assert "scratchv_run_external" not in default.source
    assert default.kernel_calls is False


def test_kernel_calls_deduplicate_operands_and_keep_error_status(tmp_path, host_cc):
    x = Value("x", shape=(2,))
    builder = IRBuilder()
    builder.new_function("main", [x])
    builder.new_block("entry")
    squared = builder.mul(x, x)
    builder.ret(builder.sqrt(builder.sub(squared, builder.make_const(1.0))))
    model = TensorCCodegen(builder.program, constant_storage="external", kernel_calls=True).generate()
    assert model.source.count("static __attribute__((noinline)) int sv_kernel_") == 3
    input_data = np.array([2, 3], np.float32)
    weights, workspace, output, args = arguments(model, input_data)
    with library(tmp_path, model, host_cc) as run:
        assert run(*args) == 0
        np.testing.assert_array_equal(output, np.sqrt(input_data * input_data - np.float32(1)))
        input_data[0] = 0
        assert run(*args) == 2


def test_kernel_calls_propagate_gather_bounds(tmp_path, host_cc):
    x, indexes = Value("x", shape=(2,)), Value("indexes", DataType.INT64, shape=(1,))
    builder = IRBuilder()
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.program.global_values.append(indexes)
    builder.ret(builder.gather(x, indexes))
    model = TensorCCodegen(builder.program, {"indexes": np.array([0], np.int64)},
                          constant_storage="external", kernel_calls=True).generate()
    input_data = np.array([2, 3], np.float32)
    weights, workspace, output, args = arguments(model, input_data)
    with library(tmp_path, model, host_cc) as run:
        assert run(*args) == 0 and output[0] == 2
        weights[0][0] = 2
        assert run(*args) == 3


def test_kernel_calls_are_explicit_external_opt_in():
    program, bindings, _ = graph()
    with pytest.raises(TensorCCodegenError, match="requires external"):
        TensorCCodegen(program, bindings, kernel_calls=True).generate()
    with pytest.raises(TensorCCodegenError, match="boolean"):
        TensorCCodegen(program, bindings, constant_storage="external", kernel_calls=1).generate()
    default = TensorCCodegen(program, bindings, constant_storage="external").generate()
    explicit = TensorCCodegen(program, bindings, constant_storage="external", kernel_calls=False).generate()
    assert default.source == explicit.source and not default.kernel_calls
    assert default.matmul_policy == "sequential"


def matmul_experiment(left, right):
    """Use identical seeded FP32 operands for both C policies and ORT."""
    x, weight = Value("x", shape=left), Value("weight", shape=right)
    builder = IRBuilder()
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.program.global_values.append(weight)
    builder.ret(builder.matmul(x, weight))
    generator = np.random.default_rng(4108)
    data = generator.standard_normal(left).astype(np.float32)
    weights = generator.standard_normal(right).astype(np.float32)
    return builder.program, {"weight": weights}, data


def ort_matmul(data, weights):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    output_shape = np.matmul(data, weights).shape
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["x", "weight"], ["output"])], "matmul-policy",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, data.shape)],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, output_shape)],
        [onnx.numpy_helper.from_array(weights, "weight")],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)], ir_version=10)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    return session.run(None, {"x": data})[0]


MATMUL_POLICY_CASES = [
    ((3, 127), (127, 5)), ((3, 128), (128, 7)), ((3, 129), (129, 5)),
    ((3, 1024), (1024, 9)), ((2, 1, 3, 129), (1, 2, 129, 5)),
    ((129,), (129,)), ((129,), (129, 5)), ((3, 129), (129,)), ((2, 0), (0, 3)),
]


@pytest.mark.parametrize("left,right", MATMUL_POLICY_CASES)
def test_blocked_fma_host_against_ort(tmp_path, host_cc, left, right):
    program, bindings, data = matmul_experiment(left, right)
    expected = ort_matmul(data, bindings["weight"])
    for policy in ("sequential", "blocked_fma"):
        model = TensorCCodegen(program, bindings, constant_storage="external", kernel_calls=True,
                              matmul_policy=policy).generate()
        weights, workspace, output, args = arguments(model, data)
        with library(tmp_path / policy, model, host_cc) as run:
            assert run(*args) == 0
        assert model.matmul_policy == policy
        assert output.shape == expected.shape and output.dtype == expected.dtype
        assert np.isfinite(output).all()
        assert np.max(np.abs(output.astype(np.float64) - expected.astype(np.float64)), initial=0) < 1e-3


@pytest.mark.parametrize("left,right,kernel_calls", [
    ((3, 127), (127, 9), True),
    ((3, 128), (128, 7), False),
    ((3, 129), (129, 9), True),
    ((2, 1, 3, 129), (1, 2, 129, 9), True),
    ((129,), (129, 9), False),
    ((3, 129), (129,), True),
    ((1, 3), (3, 3), True),
    ((2, 0), (0, 3), True),
])
def test_blocked_fma_microtiles_preserve_baseline_bits(
        tmp_path, host_cc, left, right, kernel_calls):
    # Compare to the existing K=128 grouped 1x4 kernel, not NumPy's reduction
    # order. Odd M/N and vectors exercise both row and column cleanup paths.
    program, bindings, data = matmul_experiment(left, right)
    results = {}
    for tile in ("1x4", "2x4", "1x8"):
        model = TensorCCodegen(program, bindings, constant_storage="external",
                              kernel_calls=kernel_calls, matmul_policy="blocked_fma",
                              matmul_tile=tile).generate()
        weights, workspace, output, args = arguments(model, data)
        with library(tmp_path / tile, model, host_cc) as run:
            assert run(*args) == 0
        assert model.matmul_tile == tile
        assert np.isfinite(output).all()
        results[tile] = output.copy()
    for tile in ("2x4", "1x8"):
        np.testing.assert_array_equal(results[tile].view(np.uint32),
                                      results["1x4"].view(np.uint32))


@pytest.mark.parametrize("scenario", ["finite_edges", "overflow"])
def test_blocked_fma_microtiles_preserve_numeric_edges(tmp_path, host_cc, scenario):
    from scratchv.verification.fp32_fma import fma

    program, bindings, _ = matmul_experiment((3, 129), (129, 9))
    data = np.zeros((3, 129), np.float32)
    matrix = np.zeros((129, 9), np.float32)
    if scenario == "overflow":
        data[:, 0] = np.finfo(np.float32).max
        matrix[0, :] = np.float32(2)
        expected = None
    else:
        subnormal = np.array(1, np.uint32).view(np.float32)[()]
        tiny = np.float32(np.finfo(np.float32).tiny)
        above_one = np.float32(1 + 2**-23)
        below_one = np.float32(1 - 2**-23)
        data[0, :] = np.float32(1)
        data[0, 127:129] = above_one
        data[1, 1::2] = np.float32(-0.0)
        data[1, [0, 1, 127, 128]] = [tiny, subnormal, -subnormal, -0.0]
        data[2, :] = -data[0, :]
        # Column 0 loses the unit at its original K position; summing the
        # opposite large values first would instead produce one.
        matrix[[0, 1, 2], 0] = [2**24, 1, -(2**24)]
        # Columns 1 and 6 put the same fused cancellation across and within
        # the K=128 boundary, respectively. Their results must differ.
        matrix[0, 1], matrix[128, 1] = -1, below_one
        matrix[126, 6], matrix[127, 6] = -1, below_one
        matrix[0, [2, 3, 4, 5]] = [subnormal, tiny, -subnormal, -0.0]
        matrix[[0, 1], 7] = [2**100, -(2**100)]
        matrix[127, 8], matrix[128, 8] = subnormal, -subnormal
        # Reuse the independent exact FMA oracle, including subnormal inputs;
        # only the specified FP32 addition joins the two K blocks.
        expected = np.zeros((3, 9), np.float32)
        for block in range(0, 129, 128):
            accumulator = np.zeros_like(expected)
            for k in range(block, min(block + 128, 129)):
                accumulator = fma(data[:, k, None], matrix[k, :], accumulator)
            expected = np.add(expected, accumulator, dtype=np.float32)
        bits = expected.view(np.uint32)
        assert bits[0, 0] == 0 and bits[0, 1] == 0
        assert bits[0, 6] == 0xA8800000  # -2**-46: a fused, unrounded product.
        assert bits[0, 2] == 1 and bits[0, 3] == 0x00800000
        assert bits[0, 4] == 0x80000001 and bits[0, 5] == 0
        assert np.float32(np.float32(2**24) + np.float32(-(2**24))) + np.float32(1) == 1
        assert int(fma(above_one, below_one, np.float32(-1)).view(np.uint32)) == 0xA8800000
    bindings["weight"] = matrix
    assert np.isfinite(data).all() and np.isfinite(matrix).all()
    results = {}
    for tile in ("1x4", "2x4", "1x8"):
        model = TensorCCodegen(program, bindings, constant_storage="external",
                              kernel_calls=True, matmul_policy="blocked_fma",
                              matmul_tile=tile).generate()
        weights, workspace, output, args = arguments(model, data)
        with library(tmp_path / tile, model, host_cc) as run:
            status = run(*args)
        if scenario == "overflow":
            assert status == 2
            assert np.all(output == -999)  # Reject before copying to the caller.
        else:
            assert status == 0
            np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))
            results[tile] = output.copy()
    if scenario == "finite_edges":
        for tile in ("2x4", "1x8"):
            np.testing.assert_array_equal(results[tile].view(np.uint32),
                                          results["1x4"].view(np.uint32))


def test_matmul_tile_is_explicit_external_blocked_fma_experiment():
    program, bindings, _ = graph()
    default = TensorCCodegen(program, bindings).generate()
    assert default.matmul_tile == "1x4"
    with pytest.raises(TensorCCodegenError, match="matmul_tile"):
        TensorCCodegen(program, bindings, constant_storage="external",
                       matmul_policy="blocked_fma", matmul_tile="2x8").generate()
    for tile in ("2x4", "1x8"):
        with pytest.raises(TensorCCodegenError):
            TensorCCodegen(program, bindings, matmul_policy="blocked_fma",
                           matmul_tile=tile).generate()
        with pytest.raises(TensorCCodegenError):
            TensorCCodegen(program, bindings, constant_storage="external",
                           matmul_tile=tile).generate()



# Exercise the generated callback ABI with real host threads. The target runtime
# has its own hart/barrier tests; this executor checks disjoint output ownership
# and the synchronous boundary without running a QEMU model.
_PARALLEL_HOST_RUNTIME = r"""
#include <pthread.h>
#include <stdlib.h>
size_t sv_test_dispatches;
struct sv_test_job {
    void (*fn)(void *, size_t, size_t);
    void *context;
    size_t begin, end;
};
static void *sv_test_worker(void *opaque) {
    struct sv_test_job *job=opaque;
    job->fn(job->context, job->begin, job->end);
    return NULL;
}
void sv_parallel_for(size_t count, void (*fn)(void *, size_t, size_t), void *context) {
    struct sv_test_job jobs[SCRATCHV_HARTS];
    pthread_t threads[SCRATCHV_HARTS];
    ++sv_test_dispatches;
    for (size_t id=0; id<SCRATCHV_HARTS; ++id)
        jobs[id]=(struct sv_test_job){fn,context,count*id/SCRATCHV_HARTS,
                                     count*(id+1)/SCRATCHV_HARTS};
    for (size_t id=SCRATCHV_HARTS-1; id>0; --id)
        if (pthread_create(&threads[id],NULL,sv_test_worker,&jobs[id])) abort();
    sv_test_worker(&jobs[0]);
    for (size_t id=1; id<SCRATCHV_HARTS; ++id)
        if (pthread_join(threads[id],NULL)) abort();
}
"""


def _host_parallel_result(directory, model, compiler, data, harts):
    if harts:
        model = replace(model, source=model.source + _PARALLEL_HOST_RUNTIME)
        compiler = [*compiler, f"-DSCRATCHV_HARTS={harts}", "-pthread"]
    weights, workspace, output, args = arguments(model, data)
    with library(directory, model, compiler) as run:
        assert run(*args) == 0
        calls = (ctypes.c_size_t.in_dll(ctypes.CDLL(str(directory / "model.so")),
                                      "sv_test_dispatches").value if harts else 0)
    return output.copy(), calls


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("kernel_calls", [False, True])
@pytest.mark.parametrize("left,right", [
    ((17, 127), (127, 17)), ((17, 128), (128, 19)),
    ((2, 1, 17, 129), (1, 2, 129, 17)),
    ((129,), (2, 129, 257)), ((2, 257, 129), (129,)),
])
def test_parallel_blocked_fma_keeps_exact_k_order(tmp_path, host_cc, kernel_calls, left, right):
    program, bindings, data = matmul_experiment(left, right)
    model = TensorCCodegen(program, bindings, constant_storage="external",
                          kernel_calls=kernel_calls, matmul_policy="blocked_fma").generate()
    expected, _ = _host_parallel_result(tmp_path / "default", model, host_cc, data, 0)
    for harts in (1, 2, 4):
        actual, calls = _host_parallel_result(tmp_path / str(harts), model, host_cc, data, harts)
        assert calls == (0 if harts == 1 else 1)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("kernel_calls", [False, True])
def test_parallel_matmuls_finish_before_workspace_reuse(tmp_path, host_cc, kernel_calls):
    x = Value("x", shape=(33, 129))
    w, back = Value("w", shape=(129, 33)), Value("back", shape=(33, 129))
    builder = IRBuilder()
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.program.global_values.extend([w, back])
    first = builder.matmul(x, w)
    # The middle callback receives the same tensor twice; its generated pointer
    # aliases must be deduplicated in both monolithic and split-kernel modes.
    second = builder.matmul(first, first)
    shifted = builder.add(second, builder.make_const(0.125))
    builder.ret(builder.matmul(shifted, back))
    rng = np.random.default_rng(371)
    data = rng.standard_normal(x.shape).astype(np.float32) / 8
    bindings = {v.name: rng.standard_normal(v.shape).astype(np.float32) / 8 for v in (w, back)}
    model = TensorCCodegen(builder.program, bindings, constant_storage="external",
                          kernel_calls=kernel_calls, matmul_policy="blocked_fma").generate()
    expected, _ = _host_parallel_result(tmp_path / "default", model, host_cc, data, 0)
    for harts in (2, 4):
        actual, calls = _host_parallel_result(tmp_path / str(harts), model, host_cc, data, harts)
        assert calls == 3
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("left,right", [
    ((3, 129), (129, 5)), ((2, 0), (0, 3)),
    ((0, 129), (129, 17)), ((17, 129), (129, 0)),
    ((0, 17, 129), (129, 17)), ((1, 8192), (8192, 5)),
])
def test_parallel_matmul_small_empty_or_few_blocks_stays_serial(tmp_path, host_cc, left, right):
    program, bindings, data = matmul_experiment(left, right)
    model = TensorCCodegen(program, bindings, constant_storage="external",
                          kernel_calls=True, matmul_policy="blocked_fma").generate()
    expected, _ = _host_parallel_result(tmp_path / "default", model, host_cc, data, 0)
    actual, calls = _host_parallel_result(tmp_path / "four", model, host_cc, data, 4)
    assert calls == 0
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))



def _softmax_experiment(shape, axis=-1, passes=1):
    x = Value("x", shape=shape)
    builder = IRBuilder()
    builder.new_function("main", [x])
    builder.new_block("entry")
    value = x
    for _ in range(passes):
        value = builder.softmax(value, axis=axis)
    builder.ret(value)
    normalized_axis = axis % len(shape)
    row_shape = shape[:normalized_axis] + shape[normalized_axis + 1:]
    rows = int(np.prod(row_shape))
    dimension = shape[normalized_axis]
    rng = np.random.default_rng(683)
    values = rng.standard_normal((rows, dimension)).astype(np.float32)
    if rows >= 4 and dimension >= 8:
        # Finite rows exercise signed zero, extreme but valid subtraction,
        # exp underflow, and the original ordered sum of unequal magnitudes.
        values[0, :] = 0.0
        values[0, 1::2] = -0.0
        values[1, :] = np.finfo(np.float32).max
        values[2, :] = -np.finfo(np.float32).max
        values[3, :] = 0.0
        values[3, :8] = [-104, -100, -90, -80, 0, -0.0,
                          np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny]
    data = np.ascontiguousarray(np.moveaxis(
        values.reshape(row_shape + (dimension,)), -1, normalized_axis))
    return builder.program, data


def _softmax_pair_experiment(shape=(17, 2049), axis=-1):
    # Keep both results live until Concat so stale callback pointers cannot be
    # hidden by a long Softmax chain converging toward uniform rows.
    _, first = _softmax_experiment(shape, axis)
    second = np.random.default_rng(719).standard_normal(shape).astype(np.float32)
    x, y = Value("x", shape=shape), Value("y", shape=shape)
    builder = IRBuilder()
    builder.new_function("main", [x, y])
    builder.new_block("entry")
    left = builder.softmax(x, axis=axis)
    right = builder.softmax(y, axis=axis % len(shape))
    builder.ret(builder.concat([left, right], axis=0))
    return builder.program, {"x": first, "y": second}


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("kernel_calls", [False, True])
def test_parallel_softmax_shared_callback_keeps_independent_contexts(tmp_path, host_cc, kernel_calls):
    program, data = _softmax_pair_experiment()
    generator = TensorCCodegen(program, constant_storage="external", kernel_calls=kernel_calls)
    model = generator.generate()
    # Equal normalized axes share one code body, but generate() must not retain
    # callback names from its previous artifact.
    assert len(re.findall(r"static void sv_softmax_\w+_range\(", model.source)) == 1
    assert generator.generate().source == model.source
    expected, _ = _host_parallel_result(tmp_path / "serial", model, host_cc, data, 0)
    split = data["x"].shape[0]
    assert not np.array_equal(expected[:split], expected[split:])
    swapped_expected = np.concatenate([expected[split:], expected[:split]], axis=0)
    for harts in (1, 4):
        weights, workspace, output, args = arguments(model, data)
        parallel = replace(model, source=model.source + _PARALLEL_HOST_RUNTIME)
        with library(tmp_path / str(harts), parallel,
                     [*host_cc, f"-DSCRATCHV_HARTS={harts}", "-pthread"]) as run:
            for inputs, wanted in ((data, expected),
                                   ({"x": data["y"], "y": data["x"]}, swapped_expected),
                                   (data, expected)):
                args[0] = pointers([inputs[spec.name] for spec in model.inputs])
                output.fill(-999)
                assert run(*args) == 0
                np.testing.assert_array_equal(output.view(np.uint32), wanted.view(np.uint32))


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("second_shape,second_axis", [
    pytest.param((128, 256), 0, id="different_axis"),
    pytest.param((64, 512), 1, id="same_elements_different_shape"),
])
def test_parallel_softmax_callback_key_keeps_shape_and_axis(tmp_path, host_cc, second_shape, second_axis):
    x, y = Value("x", shape=(128, 256)), Value("y", shape=second_shape)
    builder = IRBuilder()
    builder.new_function("main", [x, y])
    builder.new_block("entry")
    left = builder.reshape(builder.softmax(x, axis=1), (32768,))
    right = builder.reshape(builder.softmax(y, axis=second_axis), (32768,))
    builder.ret(builder.concat([left, right], axis=0))
    rng = np.random.default_rng(720)
    data = {value.name: rng.standard_normal(value.shape).astype(np.float32) for value in (x, y)}
    model = TensorCCodegen(builder.program, constant_storage="external", kernel_calls=True).generate()
    assert len(re.findall(r"static void sv_softmax_\w+_range\(", model.source)) == 2
    expected, _ = _host_parallel_result(tmp_path / "serial", model, host_cc, data, 0)
    actual, calls = _host_parallel_result(tmp_path / "parallel", model, host_cc, data, 4)
    assert calls == 2
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("kernel_calls", [False, True])
@pytest.mark.parametrize("shape,axis", [
    ((17, 2049), -1), ((3, 2049, 7), 1), ((3, 2049, 7), -2),
    pytest.param((4, 8192), -1, id="at_threshold"),
])
def test_parallel_softmax_preserves_each_row_order(tmp_path, host_cc, kernel_calls, shape, axis):
    # Seventeen/21 independent rows leave uneven work for four harts. The
    # middle-axis cases also require strided element access within each row.
    program, data = _softmax_experiment(shape, axis)
    model = TensorCCodegen(program, constant_storage="external",
                          kernel_calls=kernel_calls).generate()
    assert model.matmul_policy == "sequential"  # Softmax does not depend on it.
    expected, _ = _host_parallel_result(tmp_path / "default", model, host_cc, data, 0)
    for harts in (1, 4):
        actual, calls = _host_parallel_result(tmp_path / str(harts), model, host_cc, data, harts)
        assert calls == (1 if harts == 4 else 0)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("kernel_calls", [False, True])
def test_parallel_softmax_finishes_before_workspace_reuse(tmp_path, host_cc, kernel_calls):
    program, data = _softmax_experiment((17, 2049), passes=4)
    model = TensorCCodegen(program, constant_storage="external",
                          kernel_calls=kernel_calls).generate()
    assert model.workspace_bytes < 4 * data.nbytes  # Intermediates really reuse storage.
    assert len(re.findall(r"static void sv_softmax_\w+_range\(", model.source)) == 1
    expected, _ = _host_parallel_result(tmp_path / "serial", model, host_cc, data, 0)
    actual, calls = _host_parallel_result(tmp_path / "parallel", model, host_cc, data, 4)
    assert calls == 4
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("shape", [
    (17, 17), (3, 16384), (0, 2049),
    pytest.param((4, 8191), id="below_threshold"),
    pytest.param((1, 32768), id="single_row"),
])
def test_parallel_softmax_small_empty_or_few_rows_stays_serial(tmp_path, host_cc, shape):
    program, data = _softmax_experiment(shape)
    model = TensorCCodegen(program, constant_storage="external", kernel_calls=True).generate()
    expected, _ = _host_parallel_result(tmp_path / "serial", model, host_cc, data, 0)
    actual, calls = _host_parallel_result(tmp_path / "parallel", model, host_cc, data, 4)
    assert calls == 0
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("kernel_calls", [False, True])
def test_parallel_softmax_worker_overflow_returns_status_and_recovers(tmp_path, host_cc, kernel_calls):
    program, data = _softmax_pair_experiment()
    model = TensorCCodegen(program, constant_storage="external",
                          kernel_calls=kernel_calls).generate()
    expected, _ = _host_parallel_result(tmp_path / "serial", model, host_cc, data, 0)
    valid_last_row = data["y"][-1].copy()
    for harts in (1, 4):
        # Only the second invocation of the shared callback fails, on a row
        # owned by a worker. The first branch has already completed normally.
        data["y"][-1, 0] = np.finfo(np.float32).max
        data["y"][-1, 1] = -np.finfo(np.float32).max
        weights, workspace, output, args = arguments(model, data)
        parallel = replace(model, source=model.source + _PARALLEL_HOST_RUNTIME)
        with library(tmp_path / str(harts), parallel,
                     [*host_cc, f"-DSCRATCHV_HARTS={harts}", "-pthread"]) as run:
            assert run(*args) == 2
            assert np.all(output == -999)  # No successful caller-output copy.
            data["y"][-1, :] = valid_last_row
            assert run(*args) == 0  # Both operation contexts reset their status.
            np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("kernel_calls", [False, True])
def test_parallel_softmax_normalization_failure_returns_status_and_recovers(
        tmp_path, host_cc, kernel_calls):
    shape = (17, 2049)
    x, y = Value("x", shape=shape), Value("y", shape=shape)
    builder = IRBuilder()
    builder.new_function("main", [x, y])
    builder.new_block("entry")
    builder.softmax(x)
    # Return the second result directly: a downstream operation's finite
    # check must not hide a missing check inside either Softmax context.
    builder.ret(builder.softmax(y))
    program = builder.program
    data = {name: np.zeros(shape, dtype=np.float32) for name in ("x", "y")}
    # Only the second context has this shifted value, on its last row. With
    # four harts that row belongs to a worker, not the caller's own partition.
    data["y"][-1, 0] = -13.0
    assert all(np.isfinite(value).all() for value in data.values())
    model = TensorCCodegen(program, constant_storage="external",
                          kernel_calls=kernel_calls).generate()
    expected, _ = _host_parallel_result(tmp_path / "reference", model, host_cc, data, 0)
    # A stable Softmax of finite inputs normally cannot produce an invalid
    # normalized value. Redirect only this test artifact's expf calls so the
    # new output check is exercised independently of the shifted-input check.
    fault_source = r"""
#undef expf
extern float expf(float);
int sv_test_softmax_fault, sv_test_softmax_fault_hits;
float sv_test_softmax_expf(float value) {
    if (sv_test_softmax_fault && value == -13.0f) {
        __atomic_fetch_add(&sv_test_softmax_fault_hits, 1, __ATOMIC_RELAXED);
        if (sv_test_softmax_fault == 1) return -2048.0f;
        union { uint32_t bits; float value; } invalid = {0x7fc00000U};
        return invalid.value;
    }
    return expf(value);
}
"""
    injected = replace(model, source="#define expf sv_test_softmax_expf\n"
                       + model.source + _PARALLEL_HOST_RUNTIME + fault_source)
    for harts in (1, 4):
        weights, workspace, output, args = arguments(model, data)
        directory = tmp_path / str(harts)
        with library(directory, injected,
                     [*host_cc, f"-DSCRATCHV_HARTS={harts}", "-pthread"]) as run:
            loaded = ctypes.CDLL(str(directory / "model.so"))
            fault = ctypes.c_int.in_dll(loaded, "sv_test_softmax_fault")
            hits = ctypes.c_int.in_dll(loaded, "sv_test_softmax_fault_hits")
            dispatches = ctypes.c_size_t.in_dll(loaded, "sv_test_dispatches")
            for mode in (1, 2):
                # Mode 1 has finite exp outputs whose ordered sum is exactly
                # zero: -2048 followed by 2048 ones, making division infinite.
                # Mode 2 poisons the sum with NaN; all input shifts stay finite.
                fault.value, hits.value = mode, 0
                output.fill(-999)
                before = dispatches.value
                assert run(*args) == 2
                assert hits.value == 1
                assert dispatches.value - before == (2 if harts == 4 else 0)
                assert np.all(output == -999)
                # The join completed even on failure. Reuse these exact input,
                # workspace and output buffers, with fresh operation statuses.
                fault.value, hits.value = 0, 0
                assert run(*args) == 0
                assert hits.value == 0
                np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))


def test_inline_softmax_keeps_independent_serial_abi(tmp_path, host_cc):
    program, data = _softmax_experiment((17, 2049))
    inline = TensorCCodegen(program).generate()
    assert "sv_parallel_for" not in inline.source
    assert "SCRATCHV_HARTS" not in inline.source
    external = TensorCCodegen(program, constant_storage="external").generate()
    expected, _ = _host_parallel_result(tmp_path / "external", external, host_cc, data, 0)
    actual = np.empty_like(expected)
    with library(tmp_path / "inline", inline, host_cc) as run:
        assert run(pointers([data]), actual.ctypes.data) == 0
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires explicit SCRATCHV_QEMU/CC")
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("shape,axis,passes,kernel_calls", [
    ((17, 2049), -1, 3, False), ((3, 2049, 7), 1, 1, True),
])
def test_parallel_softmax_rv64_matches_serial_bits(
        tmp_path, host_cc, shape, axis, passes, kernel_calls, paired):
    from scratchv.runtime.riscv_external import build_external, run_external
    from scratchv.runtime.riscv_tensor import discover_toolchain
    from scratchv.runtime.weight_bundle import write_weight_bundle

    if paired:
        program, data = _softmax_pair_experiment(shape, axis)
    else:
        program, values = _softmax_experiment(shape, axis, passes)
        data = {"x": values}
    # Move the expf-underflow row from hart 0 to a worker (row 4 for the last
    # axis, row 10 for the middle axis), preserving all within-row values.
    data["x"] = np.roll(data["x"], 1, axis=0)
    model = TensorCCodegen(program, constant_storage="external",
                          kernel_calls=kernel_calls).generate()
    expected, _ = _host_parallel_result(tmp_path / "host", model, host_cc, data, 0)
    write_weight_bundle(model.external_weights, model.external_initializers, tmp_path / "weights")
    serial_output = None
    for harts in (1, 4):
        executable = build_external(model, tmp_path / "weights", tmp_path / f"build_{harts}",
                                    discover_toolchain(), harts=harts)
        actual, report = run_external(executable, data, tmp_path / f"run_{harts}", timeout=30)
        assert report["passed"] and report["guest_completed"] and report["exit_code"] == 0
        assert report["harts"] == harts
        # Host and guest libm implementations may round expf differently.
        # Check semantics against Host C, but require exact bits between the
        # original serial and row-parallel RV64 paths using the same libm.
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
        if harts == 1:
            serial_output = actual.copy()
        else:
            np.testing.assert_array_equal(actual.view(np.uint32), serial_output.view(np.uint32))


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires explicit SCRATCHV_QEMU/CC")
def test_parallel_softmax_rv64_propagates_worker_overflow(tmp_path):
    from scratchv.runtime.riscv_external import build_external, run_external
    from scratchv.runtime.riscv_tensor import discover_toolchain
    from scratchv.runtime.weight_bundle import write_weight_bundle

    program, data = _softmax_pair_experiment()
    valid_last_row = data["y"][-1].copy()
    model = TensorCCodegen(program, constant_storage="external", kernel_calls=True).generate()
    write_weight_bundle(model.external_weights, model.external_initializers, tmp_path / "weights")
    serial_output = None
    for harts in (1, 4):
        executable = build_external(model, tmp_path / "weights", tmp_path / f"build_{harts}",
                                    discover_toolchain(), harts=harts)
        # The second shared callback receives an independent input/context.
        # Its final row runs on a worker when harts=4.
        data["y"][-1, 0] = np.finfo(np.float32).max
        data["y"][-1, 1] = -np.finfo(np.float32).max
        with pytest.raises(RuntimeError, match="guest failed with status 0x00000002"):
            run_external(executable, data, tmp_path / f"failed_{harts}", timeout=15)
        data["y"][-1, :] = valid_last_row
        actual, report = run_external(executable, data, tmp_path / f"recovered_{harts}", timeout=15)
        assert report["passed"] and report["guest_completed"] and report["exit_code"] == 0
        if harts == 1:
            serial_output = actual.copy()
        else:
            np.testing.assert_array_equal(actual.view(np.uint32), serial_output.view(np.uint32))


def test_matmul_policy_is_explicit_external_experiment():
    program, bindings, _ = graph()
    with pytest.raises(TensorCCodegenError, match="matmul_policy"):
        TensorCCodegen(program, bindings, constant_storage="external", matmul_policy="unknown").generate()
    with pytest.raises(TensorCCodegenError, match="require external"):
        TensorCCodegen(program, bindings, matmul_policy="blocked_fma").generate()


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires explicit SCRATCHV_QEMU/CC")
@pytest.mark.parametrize("left,right", [((2, 127), (127, 5)), ((2, 128), (128, 7)),
                                      ((2, 129), (129, 5)), ((2, 1024), (1024, 9))])
def test_blocked_fma_rv64_against_ort(tmp_path, left, right):
    from scratchv.runtime.riscv_external import build_external, run_external
    from scratchv.runtime.riscv_tensor import discover_toolchain
    from scratchv.runtime.weight_bundle import write_weight_bundle
    program, bindings, data = matmul_experiment(left, right)
    expected = ort_matmul(data, bindings["weight"])
    for policy in ("sequential", "blocked_fma"):
        model = TensorCCodegen(program, bindings, constant_storage="external", kernel_calls=True,
                              matmul_policy=policy).generate()
        root = tmp_path / policy
        root.mkdir()
        write_weight_bundle(model.external_weights, model.external_initializers, root / "weights")
        executable = build_external(model, root / "weights", root / "build", discover_toolchain())
        output, report = run_external(executable, {"x": data}, root / "run", timeout=30)
        assert report["passed"]
        assert np.max(np.abs(output.astype(np.float64) - expected.astype(np.float64)), initial=0) < 1e-3


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires explicit SCRATCHV_QEMU/CC")
def test_kernel_calls_rv64_matches_monolithic(tmp_path):
    from scratchv.runtime.riscv_external import build_external, run_external
    from scratchv.runtime.riscv_tensor import discover_toolchain
    from scratchv.runtime.weight_bundle import write_weight_bundle
    program, bindings, _ = graph()
    x = np.arange(6, dtype=np.float32).reshape(2, 3) / 4
    actuals = []
    for split in (False, True):
        artifact = TensorCCodegen(program, bindings, constant_storage="external", kernel_calls=split).generate()
        root = tmp_path / str(split)
        root.mkdir()
        write_weight_bundle(artifact.external_weights, artifact.external_initializers, root / "weights")
        executable = build_external(artifact, root / "weights", root / "build", discover_toolchain())
        output, report = run_external(executable, {"x": x}, root / "run", timeout=30)
        assert report["passed"]
        actuals.append(np.array(output))
    np.testing.assert_array_equal(actuals[0].view(np.uint32), actuals[1].view(np.uint32))
    np.testing.assert_array_equal(actuals[0], (x @ bindings["weight"] + bindings["bias"]) * 0.5)


@pytest.mark.parametrize("failure", ["missing", "shape", "dtype", "nan", "infinity", "limit"])
def test_external_preserves_initializer_validation(failure):
    program, bindings, _ = graph()
    kwargs = {}
    if failure == "missing":
        del bindings["weight"]
    elif failure == "shape":
        bindings["weight"] = bindings["weight"].reshape(4, 3)
    elif failure == "dtype":
        bindings["weight"] = bindings["weight"].astype(np.float64)
    elif failure in ("nan", "infinity"):
        bindings["weight"][2, 3] = np.nan if failure == "nan" else np.inf
    else:
        kwargs["max_constant_bytes"] = 1
    with pytest.raises(TensorCCodegenError):
        TensorCCodegen(program, bindings, constant_storage="external", **kwargs).generate()


def test_external_large_strided_weight_is_scanned_including_final_chunk():
    count = 262144 + 17
    weight = Value("weight", shape=(count,))
    builder = IRBuilder()
    builder.new_function("main", [])
    builder.new_block("entry")
    builder.program.global_values.append(weight)
    builder.ret(weight)
    data = np.ones(count * 2, np.float32)[::2]
    codegen = TensorCCodegen(builder.program, {"weight": data}, constant_storage="external")
    artifact = codegen.generate()
    assert artifact.external_initializers[0] is data
    assert len(artifact.source) < 10000
    data[-1] = np.nan
    with pytest.raises(TensorCCodegenError, match="Nonfinite"):
        codegen.generate()


def test_external_scalar_binding_preserves_signed_zero(tmp_path, host_cc):
    value = Value("zero", is_constant=True, const_value=0.0)
    builder = IRBuilder()
    builder.new_function("main", [])
    builder.new_block("entry")
    builder.program.global_values.append(value)
    builder.ret(value)
    with pytest.raises(TensorCCodegenError, match="constant bits"):
        TensorCCodegen(builder.program, {"zero": np.array(-0.0, np.float32)},
                       constant_storage="external").generate()
    artifact = TensorCCodegen(builder.program, constant_storage="external").generate()
    negative_zero = np.array(-0.0, np.float32)
    output = np.array(1.0, np.float32)
    with library(tmp_path, artifact, host_cc) as run:
        assert run(None, pointers([negative_zero]), None, 0, output.ctypes.data) == 2
        assert output == 1


@pytest.mark.parametrize("storage", ["unknown", None, 1])
def test_invalid_storage_mode(storage):
    program, bindings, _ = graph()
    with pytest.raises(TensorCCodegenError, match="constant_storage"):
        TensorCCodegen(program, bindings, constant_storage=storage).generate()


@pytest.mark.parametrize("field", ["max_workspace_bytes", "max_constant_bytes"])
@pytest.mark.parametrize("bound", [-1, True, 1 << 64])
def test_memory_limits_fit_u64(field, bound):
    program, bindings, _ = graph()
    with pytest.raises(TensorCCodegenError, match="64-bit"):
        TensorCCodegen(program, bindings, constant_storage="external", **{field: bound}).generate()


def test_qemu_two_buffer_padding_is_explicit_bounded_and_repeatable():
    program, data = _softmax_experiment((4, 8), passes=4)
    ordinary = TensorCCodegen(program, constant_storage="external").generate()
    explicit = TensorCCodegen(program, constant_storage="external",
                             qemu_workspace_layout="none").generate()
    assert explicit.source == ordinary.source
    assert explicit.workspace_bytes == ordinary.workspace_bytes == 2 * data.nbytes
    assert ordinary.qemu_workspace_layout == "none"
    assert ordinary.qemu_workspace_buffers == ()
    capacity = ordinary.workspace_bytes + 4096
    generator = TensorCCodegen(program, constant_storage="external",
                              qemu_workspace_layout="two-buffer-pad-4k",
                              max_workspace_bytes=capacity)
    padded = generator.generate()
    assert padded.workspace_bytes == capacity
    assert padded.qemu_workspace_layout == "two-buffer-pad-4k"
    assert padded.qemu_workspace_buffers == ((0, data.nbytes),
                                            (data.nbytes + 4096, data.nbytes))
    again = generator.generate()
    assert again.source == padded.source
    assert again.qemu_workspace_buffers == padded.qemu_workspace_buffers
    for limit in (ordinary.workspace_bytes, capacity - 1):
        with pytest.raises(TensorCCodegenError):
            TensorCCodegen(program, constant_storage="external",
                           qemu_workspace_layout="two-buffer-pad-4k",
                           max_workspace_bytes=limit).generate()
    for invalid in ("padding", None, True):
        with pytest.raises(TensorCCodegenError):
            TensorCCodegen(program, constant_storage="external",
                           qemu_workspace_layout=invalid).generate()
    with pytest.raises(TensorCCodegenError):
        TensorCCodegen(program, qemu_workspace_layout="two-buffer-pad-4k").generate()


@pytest.mark.parametrize("layout", ["one_slot", "three_slots", "resized_reuse"])
def test_qemu_two_buffer_padding_rejects_other_allocation_plans(layout):
    if layout == "one_slot":
        program, _ = _softmax_experiment((4, 8))
    elif layout == "three_slots":
        program, _ = _softmax_pair_experiment((4, 8))
    else:
        # The third destination reuses the first offset with a smaller size.
        # A count of distinct starting addresses would wrongly accept this.
        x = Value("x", shape=(4, 8))
        builder = IRBuilder()
        builder.new_function("main", [x])
        builder.new_block("entry")
        reduced = builder.reduce_mean(builder.softmax(x), axes=(1,), keepdims=True)
        builder.ret(builder.softmax(reduced))
        program = builder.program
    assert TensorCCodegen(program, constant_storage="external").generate().workspace_bytes > 0
    with pytest.raises(TensorCCodegenError):
        TensorCCodegen(program, constant_storage="external",
                       qemu_workspace_layout="two-buffer-pad-4k").generate()


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("harts", [1, 4])
def test_qemu_two_buffer_padding_preserves_values_bounds_and_recovery(tmp_path, host_cc, harts):
    # Four operations reuse both slots and the shared callback; 32768 elements
    # exercise the existing parallel threshold without a large model.
    program, data = _softmax_experiment((32, 1024), passes=4)
    ordinary = TensorCCodegen(program, constant_storage="external", kernel_calls=True).generate()
    expected, _ = _host_parallel_result(tmp_path / "reference", ordinary, host_cc, data, 0)
    padded = TensorCCodegen(program, constant_storage="external", kernel_calls=True,
                           qemu_workspace_layout="two-buffer-pad-4k").generate()
    capacity, guard = padded.workspace_bytes, 16
    storage = np.full((capacity + 2 * guard) // 8, 0x5A5A5A5A5A5A5A5A, dtype=np.uint64)
    raw = storage.view(np.uint8)
    weights, unused_workspace, output, args = arguments(padded, data)
    args[2] = storage.ctypes.data + guard
    parallel = replace(padded, source=padded.source + _PARALLEL_HOST_RUNTIME)
    valid_last_row = data[-1].copy()
    with library(tmp_path / "padded", parallel,
                 [*host_cc, f"-DSCRATCHV_HARTS={harts}", "-pthread"]) as run:
        # Rejection must precede touching caller output or any workspace byte.
        args[3] = capacity - 1
        assert run(*args) == 1
        assert np.all(output == -999)
        assert np.all(raw == 0x5A)
        args[3] = capacity
        assert run(*args) == 0
        np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))
        output.fill(-999)
        # A finite input whose subtraction overflows on a worker-owned row
        # must propagate status without copying partially computed output.
        data[-1, 0] = np.finfo(np.float32).max
        data[-1, 1] = -np.finfo(np.float32).max
        assert run(*args) == 2
        assert np.all(output == -999)
        data[-1] = valid_last_row
        assert run(*args) == 0
        np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))
        dispatches = ctypes.c_size_t.in_dll(
            ctypes.CDLL(str(tmp_path / "padded" / "model.so")), "sv_test_dispatches").value
        assert dispatches == (9 if harts == 4 else 0)
    assert np.all(raw[:guard] == 0x5A)
    assert np.all(raw[guard + capacity:] == 0x5A)
    assert np.all(raw[guard + data.nbytes:guard + data.nbytes + 4096] == 0x5A)


# Column panels change external weight storage, never the ordered FP reduction.
def _panel_graph(k=129, n=17, m=17, transpose=False, lhs_batch=(), shared=False):
    x = Value("panel_x", shape=(*lhs_batch, m, k))
    weight = Value("panel_weight", shape=(n, k) if transpose else (k, n))
    builder = IRBuilder()
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.program.global_values.append(weight)
    rhs = builder.transpose(weight, (1, 0)) if transpose else weight
    left = builder.matmul(x, rhs)
    right = builder.matmul(x, rhs)
    # Both MatMuls share a single physical RHS, but retain distinct contexts.
    result = builder.add(left, right)
    if shared:
        # Keeping even an unused non-MatMul consumer is intentional: the
        # generator validates and executes it rather than silently deleting it.
        builder.gather(weight, builder.make_const(0, DataType.INT64), axis=0)
    builder.ret(builder.relu(result))
    rng = np.random.default_rng(417)
    bindings = {weight.name: rng.standard_normal(weight.shape).astype(np.float32) / 8}
    data = rng.standard_normal(x.shape).astype(np.float32) / 8
    return builder.program, bindings, data


def _panel_artifact(program, bindings, layout="column-panel", **kwargs):
    return TensorCCodegen(program, bindings, constant_storage="external",
                          matmul_policy="blocked_fma", matmul_rhs_layout=layout,
                          **kwargs).generate()


def test_column_panel_policy_is_opt_in_and_default_source_unchanged():
    program, bindings, _ = _panel_graph()
    baseline = TensorCCodegen(program, bindings, constant_storage="external",
                             matmul_policy="blocked_fma").generate()
    explicit = _panel_artifact(program, bindings, "original")
    assert baseline.source == explicit.source
    assert baseline.matmul_rhs_layout == "original"
    assert baseline.matmul_rhs_panels == ()
    for invalid in (None, True, "packed"):
        with pytest.raises(TensorCCodegenError, match="matmul_rhs_layout"):
            _panel_artifact(program, bindings, invalid)
    for options in ({}, {"constant_storage": "external"},
                    {"matmul_policy": "blocked_fma"}):
        with pytest.raises(TensorCCodegenError):
            TensorCCodegen(program, bindings, matmul_rhs_layout="column-panel",
                           **options).generate()


@pytest.mark.parametrize("tile", ["1x4", "2x4", "1x8"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("storage", ["c", "fortran", "strided"])
def test_column_panel_view_metadata_and_original_ir_unchanged(tile, transpose, storage):
    program, bindings, _ = _panel_graph(k=9, n=16, transpose=transpose)
    source = bindings["panel_weight"]
    if storage == "fortran":
        source = np.asfortranarray(source)
    elif storage == "strided":
        backing = np.empty((source.shape[0]*2, source.shape[1]*2), np.float32)
        view = backing[::-2, ::2]
        view[:] = source
        source = view
    bindings = {"panel_weight": source}
    before = repr(program)
    instruction_ids = [id(i) for i in program.functions[0].blocks[0].instructions]
    generator = TensorCCodegen(program, bindings, constant_storage="external",
                              matmul_policy="blocked_fma", matmul_tile=tile,
                              matmul_rhs_layout="column-panel")
    model = generator.generate()
    assert generator.generate().source == model.source
    assert repr(program) == before
    assert instruction_ids == [id(i) for i in program.functions[0].blocks[0].instructions]
    assert list(bindings) == ["panel_weight"] and bindings["panel_weight"] is source
    assert len(model.matmul_rhs_panels) == 1
    panel = model.matmul_rhs_panels[0]
    columns = int(tile.split("x")[1])
    assert panel["logical_shape"] == [9, 16]
    assert panel["physical_shape"] == [16//columns, 9, columns]
    assert panel["panel_columns"] == columns
    assert panel["source_weight"] == "panel_weight"
    assert bool(panel["folded_transpose"]) == transpose
    assert panel["materialization"] == "view"
    packed = model.external_initializers[0]
    assert np.shares_memory(packed, source)
    assert tuple(panel["physical_shape"]) == model.external_weights[0].shape
    logical = packed.transpose(1, 0, 2).reshape(9, 16)
    np.testing.assert_array_equal(logical.view(np.uint32),
                                  (source.T if transpose else source).view(np.uint32))
    assert "transpose */" not in model.source


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("tile,transpose,kernel_calls,batch", [
    ("1x4", False, False, ()), ("2x4", True, True, ()),
    ("1x8", False, True, ()), ("2x4", True, False, (2,)),
])
def test_column_panel_bitwise_multiconsumer_tail_and_lhs_batch(
        tmp_path, host_cc, tile, transpose, kernel_calls, batch):
    program, bindings, data = _panel_graph(transpose=transpose, lhs_batch=batch)
    common = dict(matmul_tile=tile, kernel_calls=kernel_calls)
    original = _panel_artifact(program, bindings, "original", **common)
    packed = _panel_artifact(program, bindings, **common)
    assert len(packed.matmul_rhs_panels) == 1
    assert packed.matmul_rhs_panels[0]["materialization"] == "padded-copy"
    if transpose:
        assert packed.workspace_bytes < original.workspace_bytes
    expected, _ = _host_parallel_result(tmp_path/"reference", original, host_cc, data, 1)
    for harts in (1, 4):
        actual, calls = _host_parallel_result(tmp_path/str(harts), packed, host_cc, data, harts)
        assert calls == (2 if harts == 4 else 0)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("transpose", [False, True])
def test_column_panel_shared_non_rhs_consumer_keeps_original(transpose):
    program, bindings, _ = _panel_graph(transpose=transpose, shared=True)
    original = _panel_artifact(program, bindings, "original")
    packed = _panel_artifact(program, bindings)
    assert packed.matmul_rhs_panels == ()
    assert packed.source == original.source
    assert packed.external_weights == original.external_weights
    assert packed.external_initializers[0] is bindings["panel_weight"]


@pytest.mark.parametrize("right", [(2, 129, 17), (129,), (0, 17), (129, 0)])
def test_column_panel_batched_vector_or_empty_rhs_keeps_original(right):
    left = (17, right[-2]) if len(right) > 1 else (17, right[0])
    program, bindings, _ = matmul_experiment(left, right)
    original = _panel_artifact(program, bindings, "original")
    packed = _panel_artifact(program, bindings)
    assert packed.matmul_rhs_panels == ()
    assert packed.source == original.source


@pytest.mark.parametrize("malformed", [
    "attrs", "perm", "dest_dtype", "dest_shape", "duplicate_dest",
    "binding_shape", "binding_dtype", "extra_binding", "use_before_transpose",
    "operand_dtype", "target",
])
def test_column_panel_fold_preserves_original_rejection(malformed):
    from copy import copy
    program, bindings, _ = _panel_graph(transpose=True)
    instructions = program.functions[0].blocks[0].instructions
    transpose = instructions[0]
    if malformed == "attrs":
        transpose.attrs["unexpected"] = 1
    elif malformed == "perm":
        transpose.attrs["perm"] = (0, 0)
    elif malformed == "dest_dtype":
        transpose.dest.dtype = DataType.INT32
    elif malformed == "dest_shape":
        transpose.dest.shape = (1, 1)
    elif malformed == "duplicate_dest":
        instructions.insert(1, copy(transpose))
    elif malformed == "binding_shape":
        bindings["panel_weight"] = bindings["panel_weight"].reshape(-1)
    elif malformed == "binding_dtype":
        bindings["panel_weight"] = bindings["panel_weight"].astype(np.float64)
    elif malformed == "extra_binding":
        bindings[transpose.dest.name] = bindings["panel_weight"].T
    elif malformed == "use_before_transpose":
        instructions[0], instructions[1] = instructions[1], instructions[0]
    elif malformed == "operand_dtype":
        transpose.operands = [replace(transpose.operands[0], dtype=DataType.INT32)]
    elif malformed == "target":
        transpose.target = "other_block"
    for layout in ("original", "column-panel"):
        with pytest.raises(TensorCCodegenError):
            _panel_artifact(program, bindings, layout)


def test_column_panel_physical_capacity_and_tail_zero():
    program, bindings, _ = _panel_graph(k=9, n=5)
    physical_bytes = 2 * 9 * 4 * 4
    with pytest.raises(TensorCCodegenError, match="storage exceeds"):
        _panel_artifact(program, bindings, max_constant_bytes=physical_bytes-1)
    model = _panel_artifact(program, bindings, max_constant_bytes=physical_bytes)
    assert model.constant_bytes == physical_bytes
    assert model.external_weights[0].nbytes == physical_bytes
    packed = model.external_initializers[0]
    assert packed.shape == (2, 9, 4)
    assert not np.shares_memory(packed, bindings["panel_weight"])
    np.testing.assert_array_equal(packed[1, :, 1:], np.zeros((9, 3), np.float32))
    bindings["panel_weight"][0, 0] = np.nan
    with pytest.raises(TensorCCodegenError, match="Nonfinite"):
        _panel_artifact(program, bindings)


def test_column_panel_strided_view_streams_through_existing_weight_bundle(tmp_path):
    from scratchv.runtime.weight_bundle import write_weight_bundle
    program, bindings, _ = _panel_graph(k=9, n=16, transpose=True)
    model = _panel_artifact(program, bindings)
    packed = model.external_initializers[0]
    assert np.shares_memory(packed, bindings["panel_weight"])
    manifest = write_weight_bundle(model.external_weights, model.external_initializers,
                                   tmp_path/"bundle")
    assert manifest["tensors"][0]["shape"] == list(packed.shape)
    restored = np.fromfile(tmp_path/"bundle/weights.bin", dtype="<f4").reshape(packed.shape)
    np.testing.assert_array_equal(restored.view(np.uint32), packed.view(np.uint32))


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
def test_column_panel_errors_physical_overlap_and_workspace_recovery(tmp_path, host_cc):
    program, bindings, data = _panel_graph(transpose=True)
    model = _panel_artifact(program, bindings, kernel_calls=True, matmul_tile="2x4")
    original = _panel_artifact(program, bindings, "original", kernel_calls=True, matmul_tile="2x4")
    expected, _ = _host_parallel_result(tmp_path/"reference", original, host_cc, data, 1)
    for harts in (1, 4):
        weights, workspace, output, args = arguments(model, data)
        parallel = replace(model, source=model.source + _PARALLEL_HOST_RUNTIME)
        with library(tmp_path/str(harts), parallel,
                     [*host_cc, f"-DSCRATCHV_HARTS={harts}", "-pthread"]) as run:
            # Nonfinite tail padding is still part of the declared physical
            # weight buffer and must fail the existing complete-buffer check.
            weights[0][-1, -1, -1] = np.inf
            assert run(*args) == 2
            assert np.all(output == -999)
            weights[0][-1, -1, -1] = 0.0
            bad_capacity = list(args)
            bad_capacity[3] -= 1
            assert run(*bad_capacity) == 1
            bad_overlap = list(args)
            logical_bytes = np.prod(model.matmul_rhs_panels[0]["logical_shape"])*4
            bad_overlap[2] = weights[0].ctypes.data + (int(logical_bytes)+7)//8*8
            assert run(*bad_overlap) == 1
            saved_weights = weights[0].copy()
            saved_input = data.copy()
            weights[0].fill(np.finfo(np.float32).max)
            data.fill(2)
            assert run(*args) == 2
            assert np.all(output == -999)
            weights[0][:] = saved_weights
            data[:] = saved_input
            assert run(*args) == 0
            np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))
            # Reuse the same workspace with a second input context.
            data[:] = -saved_input
            second_input = data.copy()  # Keep the ctypes input buffer alive.
            second = arguments(model, second_input)
            assert run(*second[3]) == 0
            assert run(*args) == 0
            np.testing.assert_array_equal(output.view(np.uint32), second[2].view(np.uint32))
            data[:] = saved_input


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"),
                    reason="requires explicit SCRATCHV_QEMU/CC")
def test_column_panel_rv64_errors_recover_in_same_guest_workspace(tmp_path):
    from scratchv.runtime.riscv_external import build_external, run_external
    from scratchv.runtime.riscv_tensor import discover_toolchain
    from scratchv.runtime.weight_bundle import write_weight_bundle

    # Only the final row is poisoned, owned by a worker in the 4-hart 2x4 partition.
    program, bindings, data = _panel_graph(k=129, n=17, m=17, transpose=True)
    bindings["panel_weight"].fill(2.0)
    data.fill(0.25)
    model = _panel_artifact(program, bindings, kernel_calls=True, matmul_tile="2x4")
    assert len(model.matmul_rhs_panels) == 1
    panel = model.matmul_rhs_panels[0]
    assert panel["folded_transpose"] and panel["physical_shape"] == [5, 129, 4]
    assert panel["materialization"] == "padded-copy"
    # Call the unmodified generated graph repeatedly inside one guest.
    # Weight lengths have no C argument: bundle validation enforces that contract.
    w = model.workspace_bytes
    input_capacity = (data.nbytes + 7)//8*8
    output_bytes = model.output.nbytes
    input_offset = w + 32
    other_offset = input_offset + input_capacity
    output_guard_offset = other_offset + input_capacity
    output_offset = output_guard_offset + 16
    total = (output_offset + output_bytes + 16 + 7)//8*8
    wrapper = r"""
#undef scratchv_run_external
static int sv_panel_test_guards(const unsigned char *base) {
    for (size_t i=0; i<16; ++i)
        if (base[i] != 0x5a || base[@W@+16+i] != 0x5a ||
            base[@OUTPUT_GUARD@+i] != 0x5a ||
            base[@OUTPUT@+@OUT_BYTES@+i] != 0x5a) return 0;
    return 1;
}
static int sv_panel_test_values(const float *values, float expected) {
    for (size_t i=0; i<289; ++i) if (values[i] != expected) return 0;
    return 1;
}
int scratchv_run_external(const void *const inputs[], const void *const weights[],
                         void *workspace, size_t bytes, void *output) {
    if (bytes < @TOTAL@ULL) return 90;
    unsigned char *base=workspace;
    for (size_t i=0; i<@TOTAL@ULL; ++i) base[i]=0x5a;
    void *core=base+16;
    float *first=(float *)(base+@INPUT@);
    float *other=(float *)(base+@OTHER@);
    float *result=(float *)(base+@OUTPUT@);
    const float *given=inputs[0];
    for (size_t i=0; i<2193; ++i) {
        first[i]=given[i];
        other[i]=0.5f;
    }
    const void *first_inputs[]={first}, *other_inputs[]={other};
    int status=sv_panel_graph(first_inputs,weights,core,@W@ULL,result);
    if (status || !sv_panel_test_values(result,129.0f) ||
        !sv_panel_test_guards(base)) return 91;

    /* A one-byte-short workspace fails before any output or arena write. */
    for (size_t i=0; i<289; ++i) result[i]=-999.0f;
    for (size_t i=0; i<@W@ULL; ++i) ((unsigned char *)core)[i]=0xa5;
    status=sv_panel_graph(first_inputs,weights,core,@W@ULL-1,result);
    if (status != 1 || !sv_panel_test_values(result,-999.0f) ||
        !sv_panel_test_guards(base)) return 92;
    for (size_t i=0; i<@W@ULL; ++i)
        if (((unsigned char *)core)[i] != 0xa5) return 93;

    /* Physical, not just logical, weight range must reject writable overlap. */
    status=sv_panel_graph(first_inputs,weights,core,@W@ULL,
                         (unsigned char *)weights[0]+8772);
    if (status != 1 || !sv_panel_test_guards(base)) return 94;
    const float *packed=weights[0];
    for (size_t p=0; p<5; ++p)
        for (size_t k=0; k<129; ++k)
            for (size_t lane=0; lane<4; ++lane)
                if (packed[p*129*4+k*4+lane] != (p*4+lane<17 ? 2.0f : 0.0f))
                    return 95;

    /* Finite input, overflowing worker-owned FMA; output remains unpublished. */
    first[16*129]=0x1.fffffep+127f;
    status=sv_panel_graph(first_inputs,weights,core,@W@ULL,result);
    if (status != 2 || !sv_panel_test_values(result,-999.0f) ||
        !sv_panel_test_guards(base)) return 96;

    first[16*129]=given[16*129];
    status=sv_panel_graph(first_inputs,weights,core,@W@ULL,result);
    if (status || !sv_panel_test_values(result,129.0f) ||
        !sv_panel_test_guards(base)) return 97;

    /* New input pointer/context, exact same workspace and result storage. */
    status=sv_panel_graph(other_inputs,weights,core,@W@ULL,result);
    if (status || !sv_panel_test_values(result,258.0f) ||
        !sv_panel_test_guards(base)) return 98;
    for (size_t i=0; i<289; ++i) ((float *)output)[i]=result[i];
    return 0;
}
"""
    for token, value in {
        "@W@": w, "@INPUT@": input_offset, "@OTHER@": other_offset,
        "@OUTPUT_GUARD@": output_guard_offset, "@OUTPUT@": output_offset,
        "@OUT_BYTES@": output_bytes, "@TOTAL@": total,
    }.items():
        wrapper = wrapper.replace(token, str(value))
    wrapped = replace(
        model, source="#define scratchv_run_external sv_panel_graph\n" +
        model.source + wrapper, workspace_bytes=total)
    write_weight_bundle(wrapped.external_weights, wrapped.external_initializers,
                        tmp_path/"weights")
    outputs = []
    for harts in (1, 4):
        executable = build_external(wrapped, tmp_path/"weights", tmp_path/f"build_{harts}",
                                    discover_toolchain(), harts=harts)
        actual, report = run_external(executable, {"panel_x": data},
                                      tmp_path/f"run_{harts}", timeout=20)
        assert report["passed"] and report["guest_completed"] and report["exit_code"] == 0
        assert report["harts"] == harts
        assert executable.evidence["matmul_rhs_layout"] == "column-panel"
        np.testing.assert_array_equal(actual, np.full((17, 17), 258.0, np.float32))
        outputs.append(actual.view(np.uint32))
    np.testing.assert_array_equal(outputs[0], outputs[1])
