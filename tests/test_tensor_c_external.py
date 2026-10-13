"""External weights and caller workspace: execute the generated host C ABI.

Set SCRATCHV_ZIG to a Zig executable, or install a host clang/gcc. Executable
checks skip explicitly if no compiler is available; graph checks always run.
"""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import ctypes
import os
from pathlib import Path
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
    return weights, workspace, output, [pointers([x]), pointers(weights),
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
