"""Real Host OpenBLAS integration with an independent scalar numerical oracle.

Set SCRATCHV_OPENBLAS_INCLUDE_DIR and SCRATCHV_OPENBLAS_LIBRARY_DIR explicitly.
Start pytest with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Missing optional
dependency configuration skips execution; an explicitly broken setup fails.

Normal seeded operands use a fixed abs(error) <= 1e-4 + 2e-5*abs(reference).
Cancellation cases use gamma_K * sum(abs(products)), gamma_K=K*u/(1-K*u),
u=2**-24. These are provider-specific checks, not changes to builtin tolerances.
"""
import ctypes
import math
import os
from pathlib import Path

import numpy as np
import pytest

from probes.w4_qwen3_full.host_diagnostic import build_library
from scratchv.backend.openblas import openblas_matmul
from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorCCodegenError
from tests.test_tensor_c_external import (
    MATMUL_POLICY_CASES, POINTERS, arguments, host_cc, library, matmul_experiment, pointers,
)


@pytest.fixture(scope="module")
def openblas_provider():
    include = os.environ.get("SCRATCHV_OPENBLAS_INCLUDE_DIR")
    libdir = os.environ.get("SCRATCHV_OPENBLAS_LIBRARY_DIR")
    if not include and not libdir:
        pytest.skip("Set both SCRATCHV_OPENBLAS_INCLUDE_DIR and SCRATCHV_OPENBLAS_LIBRARY_DIR")
    assert include and libdir, "Both explicit OpenBLAS dependency directories are required"
    assert Path(include).is_dir(), f"OpenBLAS include directory does not exist: {include}"
    assert Path(libdir).is_dir(), f"OpenBLAS library directory does not exist: {libdir}"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1", "Start pytest with OPENBLAS_NUM_THREADS=1"
    assert os.environ.get("OMP_NUM_THREADS") == "1", "Start pytest with OMP_NUM_THREADS=1"
    return openblas_matmul(include_dirs=(include,), library_dirs=(libdir,))


def scalar_reference(data, weights):
    """Evaluate each dot product without NumPy matmul/dot or any BLAS call.

    Each finite FP32 product in these cases is exact in Python's binary64;
    math.fsum then avoids imposing the provider's FP32 accumulation order.
    """
    left_vector, right_vector = data.ndim == 1, weights.ndim == 1
    left = data.reshape(1, -1) if left_vector else data
    right = weights.reshape(-1, 1) if right_vector else weights
    batches = np.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    left = np.broadcast_to(left, batches + left.shape[-2:])
    right = np.broadcast_to(right, batches + right.shape[-2:])
    m, k, n = left.shape[-2], left.shape[-1], right.shape[-1]
    result = np.empty(batches + (m, n), dtype=np.float64)
    scale = np.empty_like(result)
    for batch in np.ndindex(batches):
        for i in range(m):
            for j in range(n):
                products = [float(left[batch + (i, p)]) * float(right[batch + (p, j)])
                            for p in range(k)]
                result[batch + (i, j)] = math.fsum(products)
                scale[batch + (i, j)] = math.fsum(abs(value) for value in products)
    if left_vector:
        result, scale = result.squeeze(axis=-2), scale.squeeze(axis=-2)
    if right_vector:
        result, scale = result.squeeze(axis=-1), scale.squeeze(axis=-1)
    return result, scale


def assert_provider_accuracy(actual, expected, budget, label):
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    error = np.abs(actual.astype(np.float64) - expected)
    limit = np.broadcast_to(budget, error.shape)
    assert np.all(error <= limit), (label, error, limit)
    ratio = np.divide(error, limit, out=np.zeros_like(error), where=limit > 0)
    print(f"OpenBLAS {label}: max_abs={np.max(error, initial=0):.8g}, "
          f"max_budget_ratio={np.max(ratio, initial=0):.8g}")


def generate(program, bindings, provider, **options):
    return TensorCCodegen(
        program, bindings, matmul_implementation="require_external",
        external_matmul=provider, external_matmul_numerics="allow_provider_fp32",
        **options,
    ).generate()


def test_openblas_requires_explicit_provider_numerics():
    program, bindings, _ = matmul_experiment((3, 5), (5, 7))
    selected = openblas_matmul()
    assert selected.numerics == "provider_fp32"
    assert selected.targets == ("host",)
    assert selected.max_dimension == 2**31 - 1
    fallback = TensorCCodegen(
        program, bindings, matmul_implementation="prefer_external",
        external_matmul=selected,
    ).generate()
    assert fallback.source == TensorCCodegen(program, bindings).generate().source
    assert fallback.external_matmul is None
    assert fallback.matmul_selections[0].implementation == "builtin"
    assert fallback.matmul_selections[0].reason
    with pytest.raises(TensorCCodegenError):
        TensorCCodegen(program, bindings, matmul_implementation="require_external",
                       external_matmul=selected).generate()
    for policy in ("sequential", "blocked_fma"):
        model = generate(program, bindings, selected, constant_storage="external",
                         matmul_policy=policy)
        assert model.external_matmul == selected
        assert model.matmul_selections[0].implementation == "external"
        assert selected.symbol in model.source
    with pytest.raises(TensorCCodegenError):
        generate(program, bindings, selected, matmul_target="riscv64")
    with pytest.raises(TensorCCodegenError, match="original row-major storage"):
        generate(program, bindings, selected, constant_storage="external",
                 matmul_policy="blocked_fma", matmul_rhs_layout="column-panel")


@pytest.mark.parametrize("left,right", MATMUL_POLICY_CASES[:-1])
def test_openblas_normal_values_against_scalar_fsum(
        tmp_path, host_cc, openblas_provider, left, right):
    program, bindings, data = matmul_experiment(left, right)
    original_data, original_weights = data.copy(), bindings["weight"].copy()
    model = generate(program, bindings, openblas_provider, constant_storage="external",
                     kernel_calls=True)
    weights, workspace, output, args = arguments(model, data)
    expected, _ = scalar_reference(data, weights[0])
    with library(tmp_path, model, host_cc) as run:
        assert run(*args) == 0
        assert_provider_accuracy(output, expected, 1e-4 + 2e-5 * np.abs(expected),
                                 f"{left} x {right}")
        # C is overwritten on every call; it is never an implicit accumulator.
        output.fill(12345)
        assert run(*args) == 0
        assert_provider_accuracy(output, expected, 1e-4 + 2e-5 * np.abs(expected),
                                 "repeat overwrite")
    np.testing.assert_array_equal(data.view(np.uint32), original_data.view(np.uint32))
    np.testing.assert_array_equal(weights[0].view(np.uint32), original_weights.view(np.uint32))


def test_openblas_inline_weights(tmp_path, host_cc, openblas_provider):
    program, bindings, data = matmul_experiment((3, 17), (17, 7))
    model = generate(program, bindings, openblas_provider)
    output = np.full(model.output.shape, -999, np.float32)
    expected, _ = scalar_reference(data, bindings["weight"])
    with library(tmp_path, model, host_cc) as run:
        assert run(pointers([data]), output.ctypes.data) == 0
    assert_provider_accuracy(output, expected, 1e-4 + 2e-5 * np.abs(expected), "inline")


def test_openblas_cancellation_uses_scale_aware_bound(tmp_path, host_cc, openblas_provider):
    program, bindings, data = matmul_experiment((3, 129), (129, 5))
    data[0, :] = 1
    data[1, :] = np.where(np.arange(129) % 2, -1, 1)
    data[2, :] = np.float32(2**-12)
    matrix = bindings["weight"]
    matrix[:, :] = 0
    matrix[:3, 0] = [2**24, 1, -(2**24)]
    matrix[:128, 1] = np.tile([2**12, -(2**12)], 64)
    matrix[128, 1] = 1
    matrix[:, 2] = np.ldexp(np.ones(129, np.float32), np.arange(129) % 21 - 10)
    matrix[1::2, 2] *= -1
    matrix[:, 3] = np.float32(2**-12)
    matrix[[0, 64, 128], 4] = [2**20, -(2**20), 3]
    model = generate(program, bindings, openblas_provider, constant_storage="external")
    weights, workspace, output, args = arguments(model, data)
    expected, scale = scalar_reference(data, weights[0])
    k, unit_roundoff = data.shape[-1], 2**-24
    gamma_k = k * unit_roundoff / (1 - k * unit_roundoff)
    with library(tmp_path, model, host_cc) as run:
        assert run(*args) == 0
    assert_provider_accuracy(output, expected, gamma_k * scale, "cancellation / mixed scales")


@pytest.mark.parametrize("left,right", [
    ((0, 5), (5, 7)), ((3, 5), (5, 0)), ((0, 3, 5), (1, 5, 7)),
    ((2, 0), (0, 3)), ((0,), (0,)),
])
def test_openblas_degenerate_shapes_need_no_library(tmp_path, host_cc, left, right):
    program, bindings, data = matmul_experiment(left, right)
    # Even an unavailable provider is unused for empty shapes and zero K.
    model = generate(program, bindings, openblas_matmul(), constant_storage="external")
    assert model.external_matmul is None
    weights, workspace, output, args = arguments(model, data)
    expected, _ = scalar_reference(data, weights[0])
    with library(tmp_path, model, host_cc) as run:
        assert run(*args) == 0
    assert output.shape == expected.shape
    np.testing.assert_array_equal(output.view(np.uint32),
                                  np.zeros_like(output).view(np.uint32))


def test_openblas_real_link_runtime_guards_and_adapter_dimensions(
        tmp_path, host_cc, openblas_provider):
    program, bindings, data = matmul_experiment((3, 5), (5, 7))
    model = generate(program, bindings, openblas_provider, constant_storage="external")
    binary, evidence = build_library(model, tmp_path / "build", host_cc)
    assert "-lopenblas" in evidence["command"]
    assert evidence["external_matmul"] == {
        "name": "openblas", "numerics": "provider_fp32",
        "permission": "allow_provider_fp32",
    }
    loaded = ctypes.CDLL(str(binary))
    config = threads = run = adapter = None
    try:
        config = loaded.openblas_get_config
        config.restype = ctypes.c_char_p
        threads = loaded.openblas_get_num_threads
        threads.restype = ctypes.c_int
        description = config().decode("ascii")
        assert "OpenBLAS" in description
        assert "USE64BITINT" not in description
        assert threads() == 1
        assert ctypes.cast(loaded.cblas_sgemm, ctypes.c_void_p).value
        print(f"OpenBLAS real dependency: {description}; threads={threads()}")

        run = getattr(loaded, model.function_name)
        run.restype = ctypes.c_int
        run.argtypes = [POINTERS, POINTERS, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        weights, workspace, output, args = arguments(model, data)
        expected, _ = scalar_reference(data, weights[0])
        assert run(*args) == 0
        assert_provider_accuracy(output, expected, 1e-4 + 2e-5 * np.abs(expected),
                                 "production Host builder")

        saved = data.copy()
        output.fill(-999)
        data.flat[0] = np.nan
        assert run(*args) == 2
        assert np.all(output == -999)
        data[:] = saved
        data.fill(np.finfo(np.float32).max)
        weights[0].fill(2)
        assert run(*args) == 2
        assert np.all(output == -999)

        adapter = getattr(loaded, openblas_provider.symbol)
        adapter.restype = ctypes.c_int
        adapter.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint64] * 3
        cell = np.array([123], dtype=np.float32)
        for dimensions in ((2**31, 1, 1), (1, 2**31, 1), (1, 1, 2**31),
                           (2**64 - 1, 1, 1)):
            assert adapter(cell.ctypes.data, cell.ctypes.data, cell.ctypes.data,
                           *dimensions) != 0
            assert cell[0] == 123
        assert adapter(None, None, None, 0, 7, 5) == 0
        assert adapter(None, None, None, 3, 0, 5) == 0
        zeros = np.full((2, 3), -999, np.float32)
        assert adapter(None, None, zeros.ctypes.data, 2, 3, 0) == 0
        np.testing.assert_array_equal(zeros.view(np.uint32),
                                      np.zeros_like(zeros).view(np.uint32))
    finally:
        del config, threads, run, adapter
        import _ctypes
        if os.name == "nt":
            _ctypes.FreeLibrary(loaded._handle)
        else:
            _ctypes.dlclose(loaded._handle)
        loaded._handle = 0


@pytest.mark.parametrize("value", ["anything", None, 1])
def test_invalid_external_numerics_is_rejected(value):
    program, bindings, _ = matmul_experiment((3, 5), (5, 7))
    with pytest.raises(TensorCCodegenError, match="external_matmul_numerics"):
        TensorCCodegen(program, bindings, external_matmul_numerics=value).generate()


def test_compiler_config_preserves_openblas_numerics_opt_in(tmp_path):
    from scratchv.compiler import CompilerConfig, CompilerDriver
    from tests.test_tensor_compiler import model_file

    model_path, _ = model_file(tmp_path)
    selected = openblas_matmul()
    driver = CompilerDriver(CompilerConfig(
        backend="tensor-c", tensor_matmul_implementation="require_external",
        tensor_external_matmul=selected,
        tensor_external_matmul_numerics="allow_provider_fp32",
    ))
    result = driver.compile(str(model_path), str(tmp_path / "model.c"))
    assert result.success, result.errors
    artifact = driver.tensor_artifact
    assert artifact.external_matmul == selected
    assert artifact.external_matmul_numerics == "allow_provider_fp32"
    assert artifact.matmul_selections[0].implementation == "external"
    assert selected.symbol in result.output_text



def test_openblas_measured_choice_coexists_with_builtin_panels(
        tmp_path, host_cc, openblas_provider):
    """A policy fixture drives a real library call; durations are not a speed claim."""
    from scratchv.backend.external_matmul import MatMulMeasurement, MatMulPerformance
    from scratchv.ir.types import Value
    from tests.test_tensor_c_codegen import builder

    x, first, last = (Value("x", shape=(13, 17)), Value("first", shape=(17, 19)),
                      Value("last", shape=(19, 23)))
    graph = builder(x)
    graph.program.global_values.extend((first, last))
    graph.ret(graph.matmul(graph.softmax(graph.matmul(x, first), axis=-1), last))
    rng = np.random.default_rng(722)
    data = rng.standard_normal(x.shape).astype(np.float32) / 4
    bindings = {v.name: rng.standard_normal(v.shape).astype(np.float32) / 4
                for v in (first, last)}
    original = {name: value.copy() for name, value in bindings.items()}
    before = repr(graph.program)
    profile = MatMulPerformance(
        provider=openblas_provider, target="host",
        environment="test fixture; installed LP64 OpenBLAS, queried one thread",
        harts=1, measurements=(MatMulMeasurement(
            left_shape=x.shape, right_shape=first.shape, rhs_kind="constant",
            builtin_rhs_layout="column-panel", builtin_seconds=1.0, external_seconds=0.5),),
        matmul_policy="blocked_fma", matmul_tile="2x4",
        matmul_rhs_layout="column-panel", kernel_calls=True,
    )
    model = TensorCCodegen(
        graph.program, bindings, constant_storage="external", kernel_calls=True,
        matmul_policy="blocked_fma", matmul_tile="2x4", matmul_rhs_layout="column-panel",
        matmul_implementation="measured", external_matmul=openblas_provider,
        matmul_performance=profile, external_matmul_numerics="allow_provider_fp32",
        matmul_target="host",
    ).generate()
    assert [s.implementation for s in model.matmul_selections] == ["external", "builtin"]
    assert len(model.matmul_rhs_panels) == 1
    assert model.matmul_rhs_panels[0]["source_weight"] == "last"
    assert repr(graph.program) == before
    projected, _ = scalar_reference(data, original["first"])
    exponential = np.exp(projected - projected.max(axis=-1, keepdims=True))
    probabilities = exponential / exponential.sum(axis=-1, keepdims=True)
    expected, _ = scalar_reference(probabilities, original["last"])
    binary, evidence = build_library(model, tmp_path / "build", host_cc)
    assert "-lopenblas" in evidence["command"]
    loaded = ctypes.CDLL(str(binary))
    run = threads = None
    try:
        threads = loaded.openblas_get_num_threads
        threads.restype = ctypes.c_int
        assert threads() == 1
        run = getattr(loaded, model.function_name)
        run.restype = ctypes.c_int
        run.argtypes = [POINTERS, POINTERS, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        weights, workspace, output, args = arguments(model, data)
        for _ in range(2):
            output.fill(-999)
            assert run(*args) == 0
            assert_provider_accuracy(output, expected, 1e-4 + 2e-5 * np.abs(expected),
                                     "measured external + Softmax + builtin panel")
        for name, value in original.items():
            np.testing.assert_array_equal(bindings[name].view(np.uint32), value.view(np.uint32))
    finally:
        del run, threads
        import _ctypes
        if os.name == "nt":
            _ctypes.FreeLibrary(loaded._handle)
        else:
            _ctypes.dlclose(loaded._handle)
        loaded._handle = 0
