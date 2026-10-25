"""Small external MatMul calls through the existing Tensor-C build paths."""
from dataclasses import replace
import ctypes
from pathlib import Path
import os

import numpy as np
import pytest

from probes.w4_qwen3_full.host_diagnostic import build_library
from scratchv.backend.external_matmul import ExternalMatMul
from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorCCodegenError
from scratchv.compiler import CompilerConfig, CompilerDriver
from tests.test_tensor_c_external import (
    MATMUL_POLICY_CASES, POINTERS, _PARALLEL_HOST_RUNTIME,
    arguments, host_cc, library, matmul_experiment, pointers,
)
from tests.test_tensor_compiler import model_file

FIXTURE = Path(__file__).parent / "fixtures" / "external_matmul.c"


def provider(**options):
    return replace(ExternalMatMul(name="test-c", sources=(str(FIXTURE),)), **options)


def generate(*, left=(3, 5), right=(5, 7), **options):
    program, bindings, data = matmul_experiment(left, right)
    model = TensorCCodegen(program, bindings, constant_storage="external", **options).generate()
    return model, data


def test_default_and_explicit_builtin_do_not_add_a_library_dependency():
    default, _ = generate()
    explicit, _ = generate(matmul_implementation="builtin", external_matmul=provider())
    assert default.source == explicit.source
    assert default.external_matmul is explicit.external_matmul is None
    assert all(selection.implementation == "builtin" for selection in explicit.matmul_selections)
    assert "sv_matmul_f32_v1" not in explicit.source


def test_prefer_without_provider_records_builtin_reason():
    model, _ = generate(matmul_implementation="prefer_external")
    assert model.external_matmul is None
    assert len(model.matmul_selections) == 1
    selection = model.matmul_selections[0]
    assert selection.implementation == "builtin" and selection.reason


def test_require_without_provider_is_an_error():
    with pytest.raises(TensorCCodegenError, match="external|provider"):
        generate(matmul_implementation="require_external")


@pytest.mark.parametrize("options", [
    {"external_matmul": provider(targets=("riscv64",))},
    {"external_matmul": provider(max_dimension=4)},
    {"external_matmul": provider(alignment=16)},
    {"external_matmul": provider(), "matmul_policy": "blocked_fma"},
    {"external_matmul": provider(), "matmul_policy": "blocked_fma",
     "matmul_rhs_layout": "column-panel"},
])
def test_unsupported_combination_falls_back_or_fails_explicitly(options):
    fallback, _ = generate(matmul_implementation="prefer_external", **options)
    assert fallback.external_matmul is None
    assert fallback.matmul_selections[0].implementation == "builtin"
    assert fallback.matmul_selections[0].reason
    assert "sv_matmul_f32_v1" not in fallback.source
    with pytest.raises(TensorCCodegenError):
        generate(matmul_implementation="require_external", **options)


@pytest.mark.parametrize("mode", ["prefer_external", "require_external"])
def test_supported_modes_select_external_with_dependency(mode):
    selected = provider()
    model, _ = generate(matmul_implementation=mode, external_matmul=selected)
    assert model.external_matmul == selected
    assert model.matmul_selections[0].implementation == "external"
    assert model.matmul_target == "host"
    assert "sv_matmul_f32_v1" in model.source


@pytest.mark.parametrize("left,right", MATMUL_POLICY_CASES[:-1])
def test_external_numerics_match_existing_sequential_bits(tmp_path, host_cc, left, right):
    program, bindings, data = matmul_experiment(left, right)
    original = data.copy(), bindings["weight"].copy()
    results = []
    for mode in ("builtin", "require_external"):
        model = TensorCCodegen(
            program, bindings, constant_storage="external", kernel_calls=True,
            matmul_implementation=mode, external_matmul=provider(),
        ).generate()
        weights, workspace, output, args = arguments(model, data)
        with library(tmp_path / mode, model, host_cc) as run:
            assert run(*args) == 0
        np.testing.assert_array_equal(weights[0], original[1])
        results.append(output.copy())
    np.testing.assert_array_equal(data, original[0])
    assert results[0].shape == np.matmul(data, bindings["weight"]).shape
    np.testing.assert_array_equal(results[1].view(np.uint32), results[0].view(np.uint32))


@pytest.mark.parametrize("left,right", [
    ((0, 5), (5, 7)),
    ((3, 5), (5, 0)),
    ((0, 3, 5), (1, 5, 7)),
    ((2, 0), (0, 3)),
    ((0,), (0,)),
])
def test_degenerate_matmul_is_local_even_when_external_is_required(
        tmp_path, host_cc, left, right):
    model, data = generate(left=left, right=right, matmul_implementation="require_external")
    assert model.external_matmul is None
    assert model.matmul_selections[0].implementation == "builtin"
    assert model.matmul_selections[0].reason
    weights, workspace, output, args = arguments(model, data)
    with library(tmp_path, model, host_cc) as run:
        assert run(*args) == 0
    expected = data @ model.external_initializers[0]
    assert output.shape == expected.shape
    np.testing.assert_array_equal(output.view(np.uint32), np.zeros_like(expected).view(np.uint32))


@pytest.mark.parametrize("constant_storage", ["inline", "external"])
def test_external_failure_is_not_replaced_by_builtin(tmp_path, host_cc, constant_storage):
    program, bindings, data = matmul_experiment((3, 5), (5, 7))
    model = TensorCCodegen(
        program, bindings, constant_storage=constant_storage,
        matmul_implementation="prefer_external",
        external_matmul=provider(symbol="scratchv_test_matmul_fail"),
    ).generate()
    output = np.full(model.output.shape, -999, np.float32)
    if constant_storage == "external":
        weights, workspace, output, args = arguments(model, data)
    else:
        args = [pointers([data]), output.ctypes.data]
    with library(tmp_path, model, host_cc) as run:
        assert run(*args) == 4
    assert np.all(output == -999)


def test_host_build_calls_external_function_and_rejects_bad_results(tmp_path, host_cc):
    model, data = generate(
        left=(2, 1, 3, 5), right=(1, 2, 5, 7),
        matmul_implementation="require_external", external_matmul=provider(),
    )
    binary, evidence = build_library(model, tmp_path / "build", host_cc)
    assert str(FIXTURE.resolve()) in evidence["command"]
    loaded = ctypes.CDLL(str(binary))
    run = getattr(loaded, model.function_name)
    run.restype = ctypes.c_int
    run.argtypes = [POINTERS, POINTERS, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    calls = loaded.scratchv_test_matmul_calls
    calls.restype = ctypes.c_uint64
    fail = loaded.scratchv_test_matmul_failure
    fail.argtypes = [ctypes.c_int]
    fail.restype = None
    weights, workspace, output, args = arguments(model, data)
    try:
        assert calls() == 0
        assert run(*args) == 0
        assert calls() == 4  # One 2-D library call per broadcast batch.
        np.testing.assert_allclose(output, data @ weights[0], rtol=1e-6, atol=1e-6)

        for injected, expected_status in ((23, 4), (-1, 2)):
            fail(injected)
            output.fill(-999)
            previous = calls()
            assert run(*args) == expected_status
            assert calls() == previous + (1 if injected > 0 else 4)
            assert np.all(output == -999)

        fail(0)
        previous = calls()
        saved = data.flat[0]
        data.flat[0] = np.nan
        assert run(*args) == 2
        assert calls() == previous
        data.flat[0] = saved
        assert run(*args) == 0
        assert calls() == previous + 4
        np.testing.assert_allclose(output, data @ weights[0], rtol=1e-6, atol=1e-6)
    finally:
        del run, calls, fail
        import _ctypes
        if os.name == "nt":
            _ctypes.FreeLibrary(loaded._handle)
        else:
            _ctypes.dlclose(loaded._handle)
        loaded._handle = 0


def test_compiler_config_carries_external_matmul_to_artifact(tmp_path):
    model_path, _ = model_file(tmp_path)
    selected = provider()
    driver = CompilerDriver(CompilerConfig(
        backend="tensor-c", tensor_matmul_implementation="require_external",
        tensor_external_matmul=selected, tensor_matmul_target="host",
    ))
    result = driver.compile(str(model_path), str(tmp_path / "model.c"))
    assert result.success, result.errors
    assert driver.tensor_artifact.external_matmul == selected
    assert driver.tensor_artifact.matmul_selections[0].implementation == "external"
    assert "sv_matmul_f32_v1" in result.output_text


def test_inline_external_call_preserves_matmul_values(tmp_path, host_cc):
    program, bindings, data = matmul_experiment((3, 5), (5, 7))
    model = TensorCCodegen(
        program, bindings, matmul_implementation="require_external", external_matmul=provider(),
    ).generate()
    output = np.full(model.output.shape, -999, np.float32)
    with library(tmp_path, model, host_cc) as run:
        assert run(pointers([data]), output.ctypes.data) == 0
    np.testing.assert_allclose(output, data @ bindings["weight"], rtol=1e-6, atol=1e-6)


def test_prefer_selects_each_matmul_and_keeps_needed_dependency(tmp_path, host_cc):
    from tests.test_tensor_c_codegen import builder
    from scratchv.ir.types import Value

    x, w, v = (Value("x", shape=(2, 2)), Value("w", shape=(2, 2)),
               Value("v", shape=(2, 3)))
    graph = builder(x, w, v)
    graph.ret(graph.matmul(graph.matmul(x, w), v))
    selected = provider(max_dimension=2)
    model = TensorCCodegen(
        graph.program, constant_storage="external",
        matmul_implementation="prefer_external", external_matmul=selected,
    ).generate()
    assert [choice.implementation for choice in model.matmul_selections] == ["external", "builtin"]
    assert model.matmul_selections[1].reason
    assert model.external_matmul == selected
    data = {"x": np.arange(4, dtype=np.float32).reshape(2, 2) / 4,
            "w": np.arange(4, dtype=np.float32).reshape(2, 2) / 2,
            "v": np.arange(6, dtype=np.float32).reshape(2, 3) / 4}
    weights, workspace, output, args = arguments(model, data)
    with library(tmp_path, model, host_cc) as run:
        assert run(*args) == 0
    np.testing.assert_array_equal(output, (data["x"] @ data["w"]) @ data["v"])


def test_compiler_required_library_failure_preserves_existing_output(tmp_path):
    model_path, _ = model_file(tmp_path)
    output = tmp_path / "model.c"
    output.write_text("keep this output", encoding="utf-8")
    result = CompilerDriver(CompilerConfig(
        backend="tensor-c", tensor_matmul_implementation="require_external",
    )).compile(str(model_path), str(output))
    assert not result.success
    assert "external" in " ".join(result.errors).lower()
    assert output.read_text(encoding="utf-8") == "keep this output"


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("kernel_calls", [False, True])
def test_external_call_stays_on_caller_with_parallel_builtin_kernels(tmp_path, host_cc, kernel_calls):
    from scratchv.ir.builder import IRBuilder
    from scratchv.ir.types import Value

    x, first, last = (Value("x", shape=(128, 5)), Value("first", shape=(5, 257)),
                      Value("last", shape=(257, 259)))
    graph = IRBuilder()
    graph.new_function("main", [x])
    graph.new_block("entry")
    graph.program.global_values.extend([first, last])
    graph.ret(graph.softmax(graph.matmul(graph.matmul(x, first), last), axis=-1))
    rng = np.random.default_rng(432)
    data = rng.standard_normal(x.shape).astype(np.float32) / 8
    bindings = {v.name: rng.standard_normal(v.shape).astype(np.float32) / 8 for v in (first, last)}
    selected = provider(symbol="sv_test_serial_matmul", numerics="provider_fp32", max_dimension=257)
    model = TensorCCodegen(
        graph.program, bindings, constant_storage="external", kernel_calls=kernel_calls,
        matmul_policy="blocked_fma", matmul_implementation="prefer_external",
        external_matmul=selected, external_matmul_numerics="allow_provider_fp32",
    ).generate()
    assert [item.implementation for item in model.matmul_selections] == ["external", "builtin"]
    wrapper = r"""
static pthread_t sv_test_caller;
void sv_test_set_caller(void) { sv_test_caller=pthread_self(); }
extern int sv_matmul_f32_v1(const float *, const float *, float *, uint64_t, uint64_t, uint64_t);
int sv_test_serial_matmul(const float *a, const float *b, float *c,
                          uint64_t m, uint64_t n, uint64_t k) {
    if (!pthread_equal(pthread_self(), sv_test_caller)) return 91;
    return sv_matmul_f32_v1(a, b, c, m, n, k);
}
"""
    model = replace(model, source=model.source + _PARALLEL_HOST_RUNTIME + wrapper)
    expected = None
    for harts in (1, 2, 4):
        directory = tmp_path / str(harts)
        weights, workspace, output, args = arguments(model, data)
        with library(directory, model, [*host_cc, f"-DSCRATCHV_HARTS={harts}", "-pthread"]) as run:
            loaded = ctypes.CDLL(str(directory / "model.so"))
            set_caller = loaded.sv_test_set_caller
            set_caller.restype = None
            set_caller()
            dispatches = ctypes.c_size_t.in_dll(loaded, "sv_test_dispatches")
            calls = loaded.scratchv_test_matmul_calls
            calls.restype = ctypes.c_uint64
            failure = loaded.scratchv_test_matmul_failure
            failure.argtypes = [ctypes.c_int]
            failure.restype = None
            assert run(*args) == 0
            assert calls() == 1
            assert dispatches.value == (0 if harts == 1 else 2)
            if expected is None:
                expected = output.copy()
            np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))
            assert np.all(np.isfinite(output))
            # Provider failure returns before dispatching or copying output.
            before = dispatches.value
            failure(23)
            output.fill(-999)
            assert run(*args) == 4
            assert calls() == 2
            assert dispatches.value == before
            assert np.all(output == -999)
            failure(0)
            assert run(*args) == 0
            np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("value", [None, 0, 1, "rows", np.bool_(True)])
def test_parallel_rows_requires_explicit_boolean(value):
    with pytest.raises(ValueError, match="parallel_rows"):
        provider(parallel_rows=value)


def test_parallel_rows_default_keeps_serial_provider_contract():
    assert provider().parallel_rows is False
    default, _ = generate(
        left=(17, 129), right=(129, 19),
        matmul_implementation="require_external", external_matmul=provider())
    explicit, _ = generate(
        left=(17, 129), right=(129, 19),
        matmul_implementation="require_external", external_matmul=provider(parallel_rows=False))
    assert default.source == explicit.source


# Reuse the fixture's ordered scalar arithmetic. Its original call counter is
# deliberately serial; only this test copy makes it safe for worker callbacks.
_ROW_PARALLEL_PROVIDER = FIXTURE.read_text(encoding="utf-8").replace(
    "return test_calls;", "return __atomic_load_n(&test_calls, __ATOMIC_RELAXED);"
).replace(
    "++test_calls;", "__atomic_fetch_add(&test_calls, 1, __ATOMIC_RELAXED);"
) + r"""
static pthread_t sv_rows_caller;
int sv_rows_worker_calls, sv_rows_active, sv_rows_completed, sv_rows_fail_workers;
void sv_rows_reset(void) {
    sv_rows_caller = pthread_self();
    sv_rows_worker_calls = sv_rows_active = sv_rows_completed = 0;
    sv_rows_fail_workers = 0;
}
int sv_test_rows(const float *a, const float *b, float *c,
                 uint64_t m, uint64_t n, uint64_t k) {
    __atomic_fetch_add(&sv_rows_active, 1, __ATOMIC_RELAXED);
    int worker = !pthread_equal(pthread_self(), sv_rows_caller);
    if (worker) __atomic_fetch_add(&sv_rows_worker_calls, 1, __ATOMIC_RELAXED);
    int status = (worker && sv_rows_fail_workers) ? 37
                 : sv_matmul_f32_v1(a, b, c, m, n, k);
    __atomic_fetch_add(&sv_rows_completed, 1, __ATOMIC_RELAXED);
    __atomic_fetch_sub(&sv_rows_active, 1, __ATOMIC_RELAXED);
    return status;
}
"""


def _row_test_artifact(program, bindings, *, enabled=True, storage="external", **options):
    selected = provider(symbol="sv_test_rows", sources=(), parallel_rows=enabled)
    model = TensorCCodegen(
        program, bindings, constant_storage=storage,
        matmul_implementation="require_external", external_matmul=selected, **options,
    ).generate()
    return replace(model, source=model.source + _PARALLEL_HOST_RUNTIME + _ROW_PARALLEL_PROVIDER)


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("kernel_calls", [False, True])
@pytest.mark.parametrize("case", ["tail", "broadcast", "same-operand"])
def test_parallel_external_rows_preserve_values_and_broadcast_offsets(
        tmp_path, host_cc, kernel_calls, case):
    from scratchv.ir.types import Value
    from tests.test_tensor_c_codegen import builder

    if case == "same-operand":
        x = Value("x", shape=(33, 33))
        graph = builder(x)
        graph.ret(graph.matmul(x, x))
        program, bindings = graph.program, {}
        data = np.random.default_rng(844).standard_normal(x.shape).astype(np.float32) / 8
        expected = data @ data
        batches = 1
    else:
        left, right = (((17, 129), (129, 19)) if case == "tail"
                       else ((2, 1, 17, 129), (1, 2, 129, 19)))
        program, bindings, data = matmul_experiment(left, right)
        expected = data @ bindings["weight"]
        batches = 1 if case == "tail" else 4
    saved_data = data.copy()
    model = _row_test_artifact(program, bindings, kernel_calls=kernel_calls)
    serial = None
    for harts in (1, 2, 4):
        directory = tmp_path / str(harts)
        weights, workspace, output, args = arguments(model, data)
        original_weights = [value.copy() for value in weights]
        with library(directory, model, [*host_cc, f"-DSCRATCHV_HARTS={harts}", "-pthread"]) as run:
            loaded = ctypes.CDLL(str(directory / "model.so"))
            loaded.sv_rows_reset()
            assert run(*args) == 0
            assert ctypes.c_int.in_dll(loaded, "sv_rows_active").value == 0
            assert ctypes.c_int.in_dll(loaded, "sv_rows_completed").value == batches * harts
            assert ctypes.c_int.in_dll(loaded, "sv_rows_worker_calls").value == batches * (harts - 1)
            assert ctypes.c_size_t.in_dll(loaded, "sv_test_dispatches").value == (batches if harts > 1 else 0)
            if serial is None:
                serial = output.copy()
            np.testing.assert_array_equal(output.view(np.uint32), serial.view(np.uint32))
            np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-5)
            for value, original in zip(weights, original_weights, strict=True):
                np.testing.assert_array_equal(value.view(np.uint32), original.view(np.uint32))
    np.testing.assert_array_equal(data.view(np.uint32), saved_data.view(np.uint32))


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("enabled,storage,left,right,parallel_harts,batches", [
    (False, "external", (17, 129), (129, 19), (), 1),
    (True, "inline", (17, 129), (129, 19), (), 1),
    (True, "external", (3, 1024), (1024, 17), (2,), 1),
    (True, "external", (3, 5), (5, 7), (), 1),
    (True, "external", (129,), (2, 129, 257), (), 2),
])
def test_parallel_rows_keep_serial_path_when_not_eligible(
        tmp_path, host_cc, enabled, storage, left, right, parallel_harts, batches):
    program, bindings, data = matmul_experiment(left, right)
    model = _row_test_artifact(program, bindings, enabled=enabled, storage=storage)
    serial = None
    for harts in (1, 2, 4):
        directory = tmp_path / str(harts)
        if storage == "external":
            weights, workspace, output, args = arguments(model, data)
        else:
            output = np.full(model.output.shape, -999, dtype=np.float32)
            args = [pointers([data]), output.ctypes.data]
        with library(directory, model, [*host_cc, f"-DSCRATCHV_HARTS={harts}", "-pthread"]) as run:
            loaded = ctypes.CDLL(str(directory / "model.so"))
            loaded.sv_rows_reset()
            assert run(*args) == 0
            parallel = harts in parallel_harts
            assert ctypes.c_size_t.in_dll(loaded, "sv_test_dispatches").value == (batches if parallel else 0)
            assert ctypes.c_int.in_dll(loaded, "sv_rows_worker_calls").value == (batches * (harts - 1) if parallel else 0)
            assert ctypes.c_int.in_dll(loaded, "sv_rows_completed").value == batches * (harts if parallel else 1)
            if serial is None:
                serial = output.copy()
            np.testing.assert_array_equal(output.view(np.uint32), serial.view(np.uint32))


@pytest.mark.skipif(os.name == "nt", reason="pthread callback execution is a POSIX check")
@pytest.mark.parametrize("kernel_calls", [False, True])
def test_parallel_external_failure_joins_before_return_and_recovers(tmp_path, host_cc, kernel_calls):
    from scratchv.ir.types import Value
    from tests.test_tensor_c_codegen import builder

    x, weight = Value("x", shape=(17, 129)), Value("weight", shape=(129, 129))
    graph = builder(x)
    graph.program.global_values.append(weight)
    graph.ret(graph.matmul(graph.matmul(x, weight), weight))
    rng = np.random.default_rng(845)
    data = rng.standard_normal(x.shape).astype(np.float32) / 8
    bindings = {"weight": rng.standard_normal(weight.shape).astype(np.float32) / 8}
    model = _row_test_artifact(graph.program, bindings, kernel_calls=kernel_calls)
    serial = None
    for harts in (1, 2, 4):
        directory = tmp_path / str(harts)
        weights, workspace, output, args = arguments(model, data)
        with library(directory, model, [*host_cc, f"-DSCRATCHV_HARTS={harts}", "-pthread"]) as run:
            loaded = ctypes.CDLL(str(directory / "model.so"))
            reset = loaded.sv_rows_reset
            fail = loaded.scratchv_test_matmul_failure
            fail.argtypes = [ctypes.c_int]
            active = ctypes.c_int.in_dll(loaded, "sv_rows_active")
            completed = ctypes.c_int.in_dll(loaded, "sv_rows_completed")
            worker_failure = ctypes.c_int.in_dll(loaded, "sv_rows_fail_workers")
            reset()
            assert run(*args) == 0
            assert active.value == 0 and completed.value == 2 * harts
            if serial is None:
                serial = output.copy()
            np.testing.assert_array_equal(output.view(np.uint32), serial.view(np.uint32))
            if harts > 1:
                reset()
                worker_failure.value = 1
                output.fill(-999)
                assert run(*args) == 4
                # Every callback from the failed first operation has returned;
                # neither the second operation nor a builtin retry is executed.
                assert active.value == 0 and completed.value == harts
                assert np.all(output == -999)
            reset()
            fail(-1)  # Successful provider status but invalid returned values.
            output.fill(-999)
            assert run(*args) == 2
            assert active.value == 0 and completed.value == harts
            assert np.all(output == -999)
            reset()
            fail(0)
            assert run(*args) == 0
            assert active.value == 0 and completed.value == 2 * harts
            np.testing.assert_array_equal(output.view(np.uint32), serial.view(np.uint32))


# Measured selection uses only an explicit, matching record. Fixture durations
# test the selection policy; they are not benchmark claims.
def _measurement(**changes):
    from scratchv.backend.external_matmul import MatMulMeasurement
    values = dict(left_shape=(3, 5), right_shape=(5, 7), rhs_kind="constant",
                  builtin_rhs_layout="column-panel", builtin_seconds=1.0,
                  external_seconds=0.5)
    values.update(changes)
    return MatMulMeasurement(**values)


def _performance(selected, measurements=None, **changes):
    from scratchv.backend.external_matmul import MatMulPerformance
    values = dict(provider=selected, target="host", environment="fixture host; one thread",
                  harts=1, measurements=(_measurement(),) if measurements is None else measurements,
                  matmul_policy="blocked_fma", matmul_tile="2x4",
                  matmul_rhs_layout="column-panel", kernel_calls=True,
                  qemu_workspace_layout="none", minimum_gain=0.1)
    values.update(changes)
    return MatMulPerformance(**values)


def _measured_model(program, bindings, selected, performance, **changes):
    options = dict(constant_storage="external", kernel_calls=True,
                   matmul_policy="blocked_fma", matmul_tile="2x4",
                   matmul_rhs_layout="column-panel", matmul_implementation="measured",
                   external_matmul=selected, matmul_performance=performance,
                   external_matmul_numerics="allow_provider_fp32", matmul_target="host")
    options.update(changes)
    return TensorCCodegen(program, bindings, **options).generate()


@pytest.mark.parametrize("field,value", [
    ("builtin_seconds", 0), ("external_seconds", -1),
    ("external_seconds", float("nan")), ("builtin_seconds", float("inf")),
    ("external_seconds", True), ("builtin_seconds", "1.0"),
    ("left_shape", (-1, 5)), ("right_shape", (True, 7)),
    ("rhs_kind", "guessed"), ("builtin_rhs_layout", "packed"),
])
def test_measured_records_reject_invalid_data(field, value):
    with pytest.raises(ValueError):
        _measurement(**{field: value})


def test_measured_profile_rejects_duplicate_keys_and_invalid_contracts():
    selected = provider(numerics="provider_fp32")
    first = _measurement()
    duplicate = replace(first, external_seconds=0.2)
    with pytest.raises(ValueError):
        _performance(selected, (first, duplicate))
    for changes in ({"provider": None}, {"environment": ""},
                    {"measurements": (object(),)}, {"harts": True},
                    {"minimum_gain": float("nan")}, {"minimum_gain": -0.1},
                    {"minimum_gain": 1.1}):
        with pytest.raises(ValueError):
            _performance(selected, **changes)


@pytest.mark.parametrize("external_seconds,wanted", [
    (1.5, "builtin"), (0.95, "builtin"),
    (0.9, "external"), (0.5, "external"),
])
def test_measured_selection_requires_minimum_gain(external_seconds, wanted):
    selected = provider(numerics="provider_fp32")
    program, bindings, _ = matmul_experiment((3, 5), (5, 7))
    profile = _performance(selected, (_measurement(external_seconds=external_seconds),))
    artifact = _measured_model(program, bindings, selected, profile)
    assert artifact.matmul_selections[0].implementation == wanted
    assert artifact.matmul_selections[0].reason
    assert artifact.matmul_performance == profile
    assert (artifact.external_matmul is not None) == (wanted == "external")
    if wanted == "builtin":
        builtin = TensorCCodegen(
            program, bindings, constant_storage="external", kernel_calls=True,
            matmul_policy="blocked_fma", matmul_tile="2x4",
            matmul_rhs_layout="column-panel").generate()
        # A matching profile retains its hart contract even on builtin;
        # compare math/layout, not the additional C compilation guard.
        assert selected.symbol not in artifact.source
        assert artifact.matmul_rhs_panels == builtin.matmul_rhs_panels


@pytest.mark.parametrize("record", [
    None,
    {"left_shape": (4, 5)},
    {"left_shape": (1, 3, 5)},
    {"right_shape": (1, 5, 7), "rhs_kind": "batched_constant", "builtin_rhs_layout": "original"},
    {"rhs_kind": "dynamic", "builtin_rhs_layout": "original"},
    {"builtin_rhs_layout": "original"},
])
def test_measured_selection_does_not_infer_unmeasured_shapes_batches_or_storage(record):
    selected = provider(numerics="provider_fp32")
    program, bindings, _ = matmul_experiment((3, 5), (5, 7))
    measurements = () if record is None else (_measurement(**record),)
    artifact = _measured_model(program, bindings, selected, _performance(selected, measurements))
    assert artifact.external_matmul is None
    assert artifact.matmul_selections[0].implementation == "builtin"
    assert artifact.matmul_selections[0].reason
    assert artifact.matmul_rhs_panels


@pytest.mark.parametrize("profile_change", [
    {"target": "riscv64"}, {"matmul_tile": "1x4"},
    {"matmul_rhs_layout": "original"}, {"kernel_calls": False},
    {"provider": provider(name="other-provider", numerics="provider_fp32")},
])
def test_measured_execution_contract_mismatch_stays_builtin(profile_change):
    selected = provider(numerics="provider_fp32", targets=("host", "riscv64"))
    program, bindings, _ = matmul_experiment((3, 5), (5, 7))
    artifact = _measured_model(
        program, bindings, selected, _performance(selected, **profile_change))
    assert artifact.external_matmul is None
    assert artifact.matmul_performance is None
    assert artifact.matmul_selections[0].implementation == "builtin"
    assert artifact.matmul_selections[0].reason


@pytest.mark.parametrize("constraint", ["dimensions", "numerical_permission", "target"])
def test_measured_speed_cannot_override_provider_safety(constraint):
    selected = provider(numerics="provider_fp32")
    if constraint == "dimensions":
        selected = replace(selected, max_dimension=4)
    elif constraint == "target":
        selected = replace(selected, targets=("riscv64",))
    program, bindings, _ = matmul_experiment((3, 5), (5, 7))
    options = ({"external_matmul_numerics": "strict"}
               if constraint == "numerical_permission" else {})
    artifact = _measured_model(program, bindings, selected, _performance(selected), **options)
    assert artifact.external_matmul is None
    assert artifact.matmul_selections[0].implementation == "builtin"


@pytest.mark.parametrize("transpose", [False, True])
def test_measured_shared_rhs_keeps_packed_builtin_and_original_graph(
        tmp_path, host_cc, transpose):
    from tests.test_tensor_c_external import _panel_graph

    program, bindings, data = _panel_graph(m=3, k=5, n=7, transpose=transpose)
    selected = provider(numerics="provider_fp32")
    measurement = _measurement(rhs_kind="transposed_constant" if transpose else "constant")
    before = repr(program)
    original = bindings["panel_weight"].copy()
    profile = _performance(selected, (measurement,))
    artifact = _measured_model(program, bindings, selected, profile)
    assert [s.implementation for s in artifact.matmul_selections] == ["builtin", "builtin"]
    assert len(artifact.matmul_rhs_panels) == 1
    assert artifact.external_matmul is None
    assert repr(program) == before
    np.testing.assert_array_equal(bindings["panel_weight"].view(np.uint32), original.view(np.uint32))
    weights, workspace, output, args = arguments(artifact, data)
    with library(tmp_path, artifact, host_cc) as run:
        assert run(*args) == 0
    rhs = original.T if transpose else original
    expected = np.maximum(2 * (data.astype(np.float64) @ rhs.astype(np.float64)), 0)
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=1e-6)


def test_measured_independent_rhs_mix_original_external_and_builtin_panels(
        tmp_path, host_cc):
    from scratchv.ir.types import Value
    from tests.test_tensor_c_codegen import builder

    x, first, last = (Value("x", shape=(3, 5)), Value("first", shape=(5, 7)),
                      Value("last", shape=(7, 11)))
    graph = builder(x)
    graph.program.global_values.extend((first, last))
    graph.ret(graph.matmul(graph.relu(graph.matmul(x, first)), last))
    rng = np.random.default_rng(721)
    data = rng.standard_normal(x.shape).astype(np.float32) / 8
    bindings = {v.name: rng.standard_normal(v.shape).astype(np.float32) / 8
                for v in (first, last)}
    originals = {name: value.copy() for name, value in bindings.items()}
    before = repr(graph.program)
    selected = provider(numerics="provider_fp32")
    profile = _performance(selected)
    artifact = _measured_model(graph.program, bindings, selected, profile)
    assert [s.implementation for s in artifact.matmul_selections] == ["external", "builtin"]
    assert artifact.external_matmul == selected
    assert len(artifact.matmul_rhs_panels) == 1
    assert artifact.matmul_rhs_panels[0]["source_weight"] == "last"
    first_spec = next(spec for spec in artifact.external_weights if spec.name == "first")
    assert first_spec.shape == first.shape
    assert repr(graph.program) == before
    for name, original in originals.items():
        np.testing.assert_array_equal(bindings[name].view(np.uint32), original.view(np.uint32))
    weights, workspace, output, args = arguments(artifact, data)
    with library(tmp_path, artifact, host_cc) as run:
        assert run(*args) == 0
        np.testing.assert_allclose(
            output, np.maximum(data.astype(np.float64) @ originals["first"], 0) @ originals["last"],
            rtol=2e-5, atol=1e-6)
        output.fill(-999)
        assert run(*args) == 0
        assert np.isfinite(output).all()




def test_measured_missing_profile_and_zero_gain_tie_keep_builtin():
    selected = provider(numerics="provider_fp32")
    program, bindings, _ = matmul_experiment((3, 5), (5, 7))
    for profile in (None, _performance(
            selected, (_measurement(external_seconds=1.0),), minimum_gain=0)):
        artifact = _measured_model(program, bindings, selected, profile)
        assert artifact.external_matmul is None
        assert artifact.matmul_selections[0].implementation == "builtin"
        assert artifact.matmul_rhs_panels


def test_measured_profile_is_explicit_and_requires_external_storage():
    selected = provider(numerics="provider_fp32")
    program, bindings, _ = matmul_experiment((3, 5), (5, 7))
    profile = _performance(selected)
    with pytest.raises(TensorCCodegenError, match="measured"):
        TensorCCodegen(program, bindings, matmul_performance=profile).generate()
    with pytest.raises(TensorCCodegenError, match="external"):
        _measured_model(program, bindings, selected, profile, constant_storage="inline")
    with pytest.raises(TensorCCodegenError, match="profile|MatMulPerformance"):
        _measured_model(program, bindings, selected, {"measurements": []})




def test_compiler_config_carries_measured_profile_and_external_abi(tmp_path):
    model_path, weights = model_file(tmp_path)
    selected = provider(numerics="provider_fp32")
    record = _measurement(left_shape=(3, 2), right_shape=(2, 2))
    profile = _performance(selected, (record,))
    driver = CompilerDriver(CompilerConfig(
        backend="tensor-c", tensor_constant_storage="external", tensor_kernel_calls=True,
        tensor_matmul_policy="blocked_fma", tensor_matmul_tile="2x4",
        tensor_matmul_rhs_layout="column-panel", tensor_qemu_workspace_layout="none",
        tensor_matmul_implementation="measured", tensor_external_matmul=selected,
        tensor_matmul_target="host", tensor_external_matmul_numerics="allow_provider_fp32",
        tensor_matmul_performance=profile,
    ))
    result = driver.compile(str(model_path), str(tmp_path / "measured.c"))
    assert result.success, result.errors
    artifact = driver.tensor_artifact
    assert artifact.function_name == "scratchv_run_external"
    assert artifact.constant_storage == "external"
    assert artifact.matmul_performance == profile
    assert artifact.external_matmul == selected
    assert artifact.matmul_selections[0].implementation == "external"
    assert artifact.matmul_rhs_panels == ()
    assert artifact.external_weights[0].shape == (2, 2)
    np.testing.assert_array_equal(artifact.external_initializers[0], weights)
    assert result.output_text == artifact.source




@pytest.mark.parametrize("m,n,k", [(2, 2, 3), (3, 3, 2)])
def test_measured_flat_explicit_dimensions_never_reuse_shape_only_timings(m, n, k):
    from scratchv.ir.types import Value
    from tests.test_tensor_c_codegen import builder

    left, right = Value("x", shape=(6,)), Value("weight", shape=(6,))
    graph = builder(left)
    graph.program.global_values.append(right)
    graph.ret(graph.matmul(left, right, m=m, n=n, k=k))
    selected = provider(numerics="provider_fp32")
    record = _measurement(left_shape=(6,), right_shape=(6,), builtin_rhs_layout="original")
    artifact = _measured_model(
        graph.program, {"weight": np.arange(6, dtype=np.float32)},
        selected, _performance(selected, (record,)))
    assert artifact.output.shape == (m, n)
    assert artifact.external_matmul is None
    assert artifact.matmul_selections[0].implementation == "builtin"
    assert "flattened" in artifact.matmul_selections[0].reason


def test_measured_mismatched_runtime_profile_builds_and_executes_builtin(tmp_path, host_cc):
    from probes.w4_qwen3_full.host_diagnostic import invoke_library, prepare_buffers

    selected = provider(numerics="provider_fp32")
    program, bindings, data = matmul_experiment((3, 5), (5, 7))
    stale_profile = _performance(selected, target="riscv64", harts=4)
    artifact = _measured_model(program, bindings, selected, stale_profile)
    assert artifact.external_matmul is None
    assert artifact.matmul_performance is None
    assert artifact.matmul_selections[0].implementation == "builtin"
    assert "mismatch" in artifact.matmul_selections[0].reason
    binary, evidence = build_library(artifact, tmp_path / "build", host_cc)
    assert str(FIXTURE.resolve()) not in evidence["command"]
    buffers = prepare_buffers(artifact, {"x": data})
    actual = invoke_library(artifact, buffers, binary)
    expected = data.astype(np.float64) @ bindings["weight"].astype(np.float64)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("kind", ["transposed_constant_3d", "dynamic_4d"])
def test_measured_exact_model_rank_keeps_transpose_and_batch_contract(tmp_path, host_cc, kind):
    from probes.w4_qwen3_full.host_diagnostic import invoke_library, prepare_buffers
    from scratchv.ir.types import OpCode, Value
    from tests.test_tensor_c_codegen import builder

    rng = np.random.default_rng(812)
    if kind == "transposed_constant_3d":
        x, weight = Value("x", shape=(1, 3, 5)), Value("weight", shape=(7, 5))
        graph = builder(x)
        graph.program.global_values.append(weight)
        rhs = graph.transpose(weight, perm=(1, 0))
        graph.ret(graph.matmul(x, rhs))
        bindings = {weight.name: rng.standard_normal(weight.shape).astype(np.float32) / 8}
        feed = {x.name: rng.standard_normal(x.shape).astype(np.float32) / 8}
        expected = feed[x.name].astype(np.float64) @ bindings[weight.name].T.astype(np.float64)
        record = _measurement(left_shape=x.shape, right_shape=(5, 7), rhs_kind="transposed_constant")
        stale = replace(record, left_shape=(3, 5))
    else:
        x = Value("x", shape=(1, 2, 3, 5))
        rhs = Value("rhs", shape=(1, 2, 5, 4))
        graph = builder(x, rhs)
        graph.ret(graph.matmul(x, rhs))
        bindings = {}
        feed = {v.name: rng.standard_normal(v.shape).astype(np.float32) / 8 for v in (x, rhs)}
        expected = feed[x.name].astype(np.float64) @ feed[rhs.name].astype(np.float64)
        record = _measurement(left_shape=x.shape, right_shape=rhs.shape,
                              rhs_kind="dynamic", builtin_rhs_layout="original")
        stale = replace(record, left_shape=(3, 5), right_shape=(5, 4))
    selected = provider(numerics="provider_fp32")
    before = repr(graph.program)
    fallback = _measured_model(graph.program, bindings, selected, _performance(selected, (stale,)))
    assert fallback.matmul_selections[0].implementation == "builtin"
    assert "no measured match" in fallback.matmul_selections[0].reason
    artifact = _measured_model(graph.program, bindings, selected, _performance(selected, (record,)))
    assert artifact.matmul_selections[0].implementation == "external"
    assert artifact.matmul_rhs_panels == ()
    assert artifact.output.shape == expected.shape
    assert repr(graph.program) == before
    if kind == "transposed_constant_3d":
        # Original NxK weight is retained, and the generated function still
        # computes its transpose before calling the row-major external ABI.
        assert [(v.name, v.shape) for v in artifact.external_weights] == [("weight", (7, 5))]
        assert f"/* 0: {OpCode.TRANSPOSE.value} */" in artifact.source
        assert len(fallback.matmul_rhs_panels) == 1
        assert fallback.matmul_rhs_panels[0]["folded_transpose"] == rhs.name
    else:
        assert artifact.external_weights == ()
        assert len(artifact.inputs) == 2
        assert "batch<2ULL" in artifact.source
    binary, _ = build_library(artifact, tmp_path / "build", host_cc)
    actual = invoke_library(artifact, prepare_buffers(artifact, feed), binary)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)
