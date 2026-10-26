"""Targeted NMSIS-DSP FP32 MatMul execution on the existing RV64/QEMU guest.

Set SCRATCHV_NMSIS_ROOT to an official NMSIS checkout and SCRATCHV_QEMU to opt in.
Missing optional configuration skips real-library tests; broken explicit paths
fail. The independent scalar oracle and error limits match the Host experiment:
abs(error) <= 1e-4 + 2e-5*abs(reference) for ordinary values, and
gamma_K*sum(abs(products)) for cancellation, with u=2**-24.
"""
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from scratchv.backend.nmsis import nmsis_matmul
from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorCCodegenError
from scratchv.runtime import riscv_external, riscv_tensor
from scratchv.runtime.weight_bundle import write_weight_bundle
from tests.test_tensor_c_external import matmul_experiment
from tests.test_tensor_c_openblas import scalar_reference


@pytest.fixture
def descriptor(tmp_path):
    # Configuration/selection tests do not need a downloaded third-party library.
    root = tmp_path / "uncompiled-nmsis"
    for relative in (
        "NMSIS/DSP/Source/MatrixFunctions/riscv_mat_mult_f32.c",
        "NMSIS/DSP/Include/dsp/matrix_functions.h",
        "NMSIS/Core/Include/nmsis_core.h",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return nmsis_matmul(root)


@pytest.fixture(scope="module")
def real_provider():
    root = os.environ.get("SCRATCHV_NMSIS_ROOT")
    if not root:
        pytest.skip("Set SCRATCHV_NMSIS_ROOT to an official NMSIS checkout")
    # Explicitly invalid dependencies are errors, not successful skips.
    return nmsis_matmul(root)


@pytest.fixture(scope="module")
def toolchain():
    if not os.environ.get("SCRATCHV_QEMU"):
        pytest.skip("Set SCRATCHV_QEMU to opt in to RV64 execution")
    # Let invalid explicit CC/QEMU overrides fail through normal discovery/build.
    return riscv_tensor.discover_toolchain()


def generate(program, bindings, provider, **options):
    return TensorCCodegen(
        program, bindings, matmul_implementation="require_external",
        external_matmul=provider, matmul_target="riscv64",
        external_matmul_numerics="allow_provider_fp32", **options,
    ).generate()


def build(model, directory, toolchain, **options):
    if model.constant_storage == "inline":
        return riscv_tensor.build_riscv_tensor(model, directory / "build", toolchain, **options)
    bundle = directory / "weights"
    write_weight_bundle(model.external_weights, model.external_initializers, bundle)
    return riscv_external.build_external(model, bundle, directory / "build", toolchain, **options)


def run(model, executable, data, directory):
    if model.constant_storage == "inline":
        result = riscv_tensor.run_riscv_tensor(
            executable, {"x": data}, directory, timeout=30)
        assert result.command[result.command.index("-cpu") + 1] == "rv64"
        return result.output
    output, report = riscv_external.run_external(
        executable, {"x": data}, directory, timeout=30)
    assert report["passed"] and report["guest_completed"] and report["harts"] == 1
    return np.asarray(output)


def assert_real_nmsis_link(executable, provider):
    # Check the actual linked RV64 ELF, not just a generated call or a mock.
    nm = shutil.which("llvm-nm") or shutil.which("nm")
    assert nm, "An ELF symbol reader (nm or llvm-nm) is required"
    symbols = subprocess.run(
        [nm, "--defined-only", str(executable.elf_path)], check=True,
        capture_output=True, text=True,
    ).stdout
    names = {line.split()[-1] for line in symbols.splitlines() if line.split()}
    assert "riscv_mat_mult_f32" in names
    assert provider.symbol in names


def assert_accuracy(actual, expected, budget, label):
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32 and np.isfinite(actual).all()
    error = np.abs(actual.astype(np.float64) - expected)
    limit = np.broadcast_to(budget, error.shape)
    assert np.all(error <= limit), (label, error, limit)
    ratio = np.divide(error, limit, out=np.zeros_like(error), where=limit > 0)
    print(f"NMSIS {label}: max_abs={np.max(error, initial=0):.8g}, "
          f"max_budget_ratio={np.max(ratio, initial=0):.8g}")


def test_nmsis_configuration_and_explicit_selection(descriptor):
    assert descriptor.targets == ("riscv64",)
    assert descriptor.max_dimension == 65535
    assert descriptor.alignment == 4
    assert descriptor.numerics == "provider_fp32"
    program, bindings, _ = matmul_experiment((3, 5), (5, 7))
    fallback = TensorCCodegen(
        program, bindings, matmul_implementation="prefer_external",
        external_matmul=descriptor, matmul_target="riscv64",
    ).generate()
    assert fallback.external_matmul is None
    assert fallback.source == TensorCCodegen(program, bindings).generate().source
    assert "permission" in fallback.matmul_selections[0].reason
    with pytest.raises(TensorCCodegenError, match="permission"):
        TensorCCodegen(
            program, bindings, matmul_implementation="require_external",
            external_matmul=descriptor, matmul_target="riscv64",
        ).generate()
    with pytest.raises(TensorCCodegenError, match="target host"):
        TensorCCodegen(
            program, bindings, matmul_implementation="require_external",
            external_matmul=descriptor, external_matmul_numerics="allow_provider_fp32",
        ).generate()
    selected = generate(program, bindings, descriptor)
    assert selected.external_matmul == descriptor
    assert selected.matmul_selections[0].implementation == "external"
    with pytest.raises(TensorCCodegenError, match="original row-major storage"):
        generate(program, bindings, descriptor, constant_storage="external",
                 matmul_policy="blocked_fma", matmul_rhs_layout="column-panel")


@pytest.mark.parametrize("left,right", [
    ((65536, 1), (1, 1)), ((1, 1), (1, 65536)), ((1, 65536), (65536, 1)),
])
def test_nmsis_uint16_dimensions_do_not_wrap(descriptor, left, right):
    program, bindings, _ = matmul_experiment(left, right)
    options = dict(constant_storage="external", external_matmul=descriptor,
                   matmul_target="riscv64", external_matmul_numerics="allow_provider_fp32")
    fallback = TensorCCodegen(
        program, bindings, matmul_implementation="prefer_external", **options,
    ).generate()
    assert fallback.external_matmul is None
    assert "provider limit" in fallback.matmul_selections[0].reason
    with pytest.raises(TensorCCodegenError, match="provider limit"):
        TensorCCodegen(
            program, bindings, matmul_implementation="require_external", **options,
        ).generate()


def test_nmsis_missing_checkout_fails(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        nmsis_matmul(tmp_path / "missing")


@pytest.mark.parametrize("storage,left,right", [
    ("external", (3, 127), (127, 5)),
    ("inline", (3, 128), (128, 7)),
    ("external", (3, 129), (129, 5)),
    ("external", (3, 1024), (1024, 9)),
    ("external", (2, 1, 3, 129), (1, 2, 129, 5)),
    ("inline", (129,), (129,)),
])
def test_nmsis_normal_values_on_qemu(
        tmp_path, real_provider, toolchain, storage, left, right):
    program, bindings, data = matmul_experiment(left, right)
    model = generate(program, bindings, real_provider, constant_storage=storage)
    executable = build(model, tmp_path, toolchain)
    assert_real_nmsis_link(executable, real_provider)
    expected, _ = scalar_reference(data, bindings["weight"])
    actual = run(model, executable, data, tmp_path / "run")
    assert_accuracy(actual, expected, 1e-4 + 2e-5 * np.abs(expected),
                    f"{storage} {left} x {right}")
    if left == (3, 127):
        # Reuse the already built ELF with a second input, without recompiling.
        second = -data
        actual = run(model, executable, second, tmp_path / "run-again")
        expected, _ = scalar_reference(second, bindings["weight"])
        assert_accuracy(actual, expected, 1e-4 + 2e-5 * np.abs(expected), "ELF reuse")



def test_compiler_config_executes_nmsis_on_qemu(tmp_path, real_provider, toolchain):
    from scratchv.compiler import CompilerConfig, CompilerDriver
    from tests.test_tensor_compiler import model_file

    model_path, weights = model_file(tmp_path)
    driver = CompilerDriver(CompilerConfig(
        backend="tensor-c", tensor_matmul_implementation="require_external",
        tensor_external_matmul=real_provider, tensor_matmul_target="riscv64",
        tensor_external_matmul_numerics="allow_provider_fp32",
    ))
    result = driver.compile(str(model_path), str(tmp_path / "compiled.c"))
    assert result.success, result.errors
    artifact = driver.tensor_artifact
    assert artifact.constant_storage == "inline"
    assert artifact.external_matmul == real_provider
    assert [choice.implementation for choice in artifact.matmul_selections] == ["external"]
    executable = build(artifact, tmp_path, toolchain)
    assert_real_nmsis_link(executable, real_provider)
    data = np.arange(6, dtype=np.float32).reshape(3, 2) / 4
    actual = run(artifact, executable, data, tmp_path / "run")
    expected, _ = scalar_reference(data, weights)
    assert_accuracy(actual, expected, 1e-4 + 2e-5 * np.abs(expected),
                    "CompilerDriver ONNX to NMSIS")



def test_nmsis_mixed_graph_keeps_blocked_fma_fallback(tmp_path, real_provider, toolchain):
    from scratchv.ir.types import Value
    from tests.test_tensor_c_codegen import builder

    x = Value("x", shape=(3, 5))
    first = Value("first", shape=(5, 129))
    last = Value("last", shape=(129, 131))
    graph = builder(x)
    graph.program.global_values.extend((first, last))
    graph.ret(graph.matmul(graph.matmul(x, first), last))
    rng = np.random.default_rng(4108)
    data = rng.standard_normal(x.shape).astype(np.float32) / 4
    bindings = {value.name: rng.standard_normal(value.shape).astype(np.float32) / 4
                for value in (first, last)}
    # A smaller declared bound exercises the full-model selection pattern
    # without allocating the real 151936-column LM head in a unit test.
    selected = replace(real_provider, max_dimension=129)
    model = TensorCCodegen(
        graph.program, bindings, constant_storage="external", kernel_calls=True,
        matmul_policy="blocked_fma", matmul_implementation="prefer_external",
        external_matmul=selected, matmul_target="riscv64",
        external_matmul_numerics="allow_provider_fp32",
    ).generate()
    assert [choice.implementation for choice in model.matmul_selections] == ["external", "builtin"]
    assert "provider limit" in model.matmul_selections[-1].reason
    assert model.external_matmul == selected
    assert model.matmul_policy == "blocked_fma" and model.matmul_tile == "1x4"
    assert model.matmul_rhs_layout == "original"
    assert "__builtin_fmaf" in model.source
    executable = build(model, tmp_path, toolchain)
    assert_real_nmsis_link(executable, real_provider)
    actual = run(model, executable, data, tmp_path / "run")
    intermediate, _ = scalar_reference(data, bindings["first"])
    expected, _ = scalar_reference(intermediate, bindings["last"])
    assert_accuracy(actual, expected, 1e-4 + 2e-5 * np.abs(expected),
                    "NMSIS then blocked fallback, K=129 / column tail")


_MAIN_HART_NMSIS = r"""
#include <stdint.h>
extern int sv_nmsis_matmul_f32_v1(const float *, const float *, float *,
                                 uint64_t, uint64_t, uint64_t);
int sv_main_hart_nmsis(const float *a, const float *b, float *c,
                       uint64_t m, uint64_t n, uint64_t k) {
    uintptr_t hart;
    static unsigned calls, fail_after_parallel;
    __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
    if (hart != 0) return 91;
    ++calls;
    if (calls == 1 && m == 17 && n == 129 && k == 5)
        fail_after_parallel = a[0] < -1000.0f;
    else if (calls == 2 && m == 51 && n == 7 && k == 683) {
        if (fail_after_parallel) return 92;
    } else return 93;
    return sv_nmsis_matmul_f32_v1(a, b, c, m, n, k);
}
"""


def test_nmsis_main_hart_between_parallel_kernels(tmp_path, real_provider, toolchain):
    from scratchv.ir.types import Value
    from tests.test_tensor_c_codegen import builder

    x = Value("x", shape=(17, 5))
    first = Value("first", shape=(5, 129))
    fallback = Value("fallback", shape=(129, 2049))
    last = Value("last", shape=(683, 7))
    graph = builder(x)
    graph.program.global_values.extend((first, fallback, last))
    value = graph.matmul(graph.matmul(x, first), fallback)
    value = graph.reshape(graph.softmax(value), (51, 683))
    graph.ret(graph.matmul(value, last))
    rng = np.random.default_rng(4110)
    data = rng.standard_normal(x.shape).astype(np.float32) / 4
    bindings = {value.name: rng.standard_normal(value.shape).astype(np.float32) / 4
                for value in (first, fallback, last)}
    # Force the middle MatMul to use the builtin fallback while keeping real
    # NMSIS calls both before and after it. Softmax has 17 uneven rows and
    # 34833 elements, so the existing parallel-work threshold is reached.
    selected = replace(real_provider, max_dimension=2048, symbol="sv_main_hart_nmsis")
    model = TensorCCodegen(
        graph.program, bindings, constant_storage="external", kernel_calls=True,
        matmul_policy="blocked_fma", matmul_implementation="prefer_external",
        external_matmul=selected, matmul_target="riscv64",
        external_matmul_numerics="allow_provider_fp32",
    ).generate()
    assert [choice.implementation for choice in model.matmul_selections] == [
        "external", "builtin", "external"]
    assert "provider limit" in model.matmul_selections[1].reason
    assert "__builtin_fmaf" in model.source and "sv_softmax_" in model.source
    # One declaration and the two builtin dispatch sites; external calls
    # stay on the caller, checked independently by mhartid in the real guest.
    assert model.source.count("sv_parallel_for(") == 3
    checked_source = tmp_path / "main_hart_nmsis.c"
    checked_source.write_text(_MAIN_HART_NMSIS, encoding="utf-8")
    intermediate, _ = scalar_reference(data, bindings["first"])
    logits, _ = scalar_reference(intermediate, bindings["fallback"])
    exponentials = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities = exponentials / exponentials.sum(axis=-1, keepdims=True)
    expected, _ = scalar_reference(probabilities.reshape(51, 683), bindings["last"])
    serial = None
    for harts in (1, 2, 4):
        directory = tmp_path / f"harts-{harts}"
        directory.mkdir()
        executable = build(model, directory, toolchain, harts=harts,
                           extra_sources=(checked_source,))
        assert_real_nmsis_link(executable, real_provider)
        actual, report = riscv_external.run_external(
            executable, {"x": data}, directory / "run", timeout=30)
        assert report["passed"] and report["guest_completed"] and report["exit_code"] == 0
        assert report["harts"] == harts
        assert_accuracy(actual, expected, 1e-4 + 2e-5 * np.abs(expected),
                        f"main-hart NMSIS / blocked / Softmax / NMSIS, harts={harts}")
        if serial is None:
            serial = actual.copy()
        else:
            np.testing.assert_array_equal(actual.view(np.uint32), serial.view(np.uint32))
        if harts == 4:
            # Fail the second provider call after both worker teams have
            # joined. Propagate status 4 without retry/fallback or a hang,
            # then verify the same ELF remains usable in a fresh run.
            failing = data.copy()
            failing[0, 0] = -2048.0
            with pytest.raises(RuntimeError, match="guest failed with status 0x00000004"):
                riscv_external.run_external(
                    executable, {"x": failing}, directory / "provider-failure", timeout=15)
            recovered, report = riscv_external.run_external(
                executable, {"x": data}, directory / "recovered", timeout=30)
            assert report["passed"] and report["exit_code"] == 0
            np.testing.assert_array_equal(recovered.view(np.uint32), serial.view(np.uint32))



_ROW_PARALLEL_NMSIS = r"""
#include <stdint.h>
extern int sv_nmsis_matmul_f32_v1(const float *, const float *, float *,
                                 uint64_t, uint64_t, uint64_t);
static unsigned started[2], finished[2], calls[2], fail_on_worker;
void sv_rows_begin(unsigned fail) {
    fail_on_worker = fail;
    for (unsigned op = 0; op < 2; ++op)
        started[op] = finished[op] = calls[op] = 0;
}
int sv_rows_complete(void) {
    const unsigned all = (1U << SCRATCHV_HARTS) - 1;
    for (unsigned op = 0; op < 2; ++op)
        if (__atomic_load_n(&started[op], __ATOMIC_ACQUIRE) != all ||
            __atomic_load_n(&finished[op], __ATOMIC_ACQUIRE) != all ||
            __atomic_load_n(&calls[op], __ATOMIC_ACQUIRE) != SCRATCHV_HARTS)
            return 97;
    return 0;
}
int sv_checked_row_nmsis(const float *a, const float *b, float *c,
                         uint64_t m, uint64_t n, uint64_t k) {
    uintptr_t hart;
    __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
    if (hart >= SCRATCHV_HARTS) return 91;
    unsigned op;
    uint64_t rows;
    if (n == 257 && k == 129) { op = 0; rows = 17; }
    else if (n == 7 && k == 683) { op = 1; rows = 51; }
    else return 92;
    /* The real runtime assigns a contiguous, nonempty range to each hart.
     * Reject a whole-matrix serial call when the row-parallel path is enabled. */
    if (m != rows / SCRATCHV_HARTS + (hart < rows % SCRATCHV_HARTS))
        return 93;
    if (__atomic_fetch_or(&started[op], 1U << hart, __ATOMIC_RELAXED) & (1U << hart))
        return 94;
    __atomic_fetch_add(&calls[op], 1, __ATOMIC_RELAXED);
    int status = sv_nmsis_matmul_f32_v1(a, b, c, m, n, k);
    /* Make a successful worker finish late, so the injected error on hart 1
     * must still be joined before the graph wrapper checks completion. */
    if (op == 1 && fail_on_worker && hart == SCRATCHV_HARTS - 1)
        for (volatile unsigned delay = 0; delay < 100000; ++delay) {}
    __atomic_fetch_or(&finished[op], 1U << hart, __ATOMIC_RELEASE);
    if (op == 1 && fail_on_worker && hart == 1) return 95;
    return status;
}
"""

_ROW_GRAPH_CHECK = r"""
extern void sv_rows_begin(unsigned);
extern int sv_rows_complete(void);
int sv_checked_rows_run(const void *const inputs[], const void *const weights[],
                        void *workspace, size_t workspace_bytes, void *output) {
    sv_rows_begin(((const float *)inputs[0])[0] < -1000.0f);
    int status = scratchv_run_external(inputs, weights, workspace, workspace_bytes, output);
    int incomplete = sv_rows_complete();
    return incomplete ? incomplete : status;
}
"""


def test_nmsis_parallel_rows_join_mixed_graph(tmp_path, real_provider, toolchain):
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper, TensorProto
    from scratchv.ir.types import Value
    from tests.test_tensor_c_codegen import builder

    x = Value("x", shape=(17, 129))
    first = Value("first", shape=(129, 257))
    fallback = Value("fallback", shape=(257, 2049))
    last = Value("last", shape=(683, 7))
    graph = builder(x)
    graph.program.global_values.extend((first, fallback, last))
    value = graph.matmul(graph.matmul(x, first), fallback)
    value = graph.reshape(graph.softmax(value), (51, 683))
    graph.ret(graph.matmul(value, last))
    rng = np.random.default_rng(4111)
    data = rng.standard_normal(x.shape).astype(np.float32) / 8
    bindings = {value.name: rng.standard_normal(value.shape).astype(np.float32) / 8
                for value in (first, fallback, last)}
    selected = replace(
        nmsis_matmul(os.environ["SCRATCHV_NMSIS_ROOT"], parallel_rows=True),
        max_dimension=2048, symbol="sv_checked_row_nmsis",
    )
    model = TensorCCodegen(
        graph.program, bindings, constant_storage="external", kernel_calls=True,
        matmul_policy="blocked_fma", matmul_implementation="prefer_external",
        external_matmul=selected, matmul_target="riscv64",
        external_matmul_numerics="allow_provider_fp32",
    ).generate()
    assert [choice.implementation for choice in model.matmul_selections] == [
        "external", "builtin", "external"]
    # The graph wrapper verifies that both external worker teams finished,
    # even when the second provider fails. No instrumentation enters the library.
    model = replace(model, source=model.source + _ROW_GRAPH_CHECK,
                    function_name="sv_checked_rows_run")
    checked_source = tmp_path / "row_parallel_nmsis.c"
    checked_source.write_text(_ROW_PARALLEL_NMSIS, encoding="utf-8")

    intermediate, _ = scalar_reference(data, bindings["first"])
    logits, _ = scalar_reference(intermediate, bindings["fallback"])
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities = exp / exp.sum(axis=-1, keepdims=True)
    expected, _ = scalar_reference(probabilities.reshape(51, 683), bindings["last"])
    nodes = [
        helper.make_node("MatMul", ["x", "first"], ["a"]),
        helper.make_node("MatMul", ["a", "fallback"], ["b"]),
        helper.make_node("Softmax", ["b"], ["p"], axis=-1),
        helper.make_node("Reshape", ["p", "shape"], ["r"]),
        helper.make_node("MatMul", ["r", "last"], ["y"]),
    ]
    reference_model = helper.make_model(helper.make_graph(
        nodes, "row-parallel-mixed",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [51, 7])],
        [numpy_helper.from_array(array, name) for name, array in bindings.items()]
        + [numpy_helper.from_array(np.array([51, 683], np.int64), "shape")],
    ), opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    reference_path = tmp_path / "reference.onnx"
    onnx.save(reference_model, reference_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    ort_expected = ort.InferenceSession(str(reference_path), options,
        providers=["CPUExecutionProvider"]).run(None, {"x": data})[0]
    np.save(tmp_path / "input.npy", data)
    np.save(tmp_path / "fp64.npy", expected)
    np.save(tmp_path / "ort.npy", ort_expected)

    serial = None
    for harts in (1, 2, 4):
        directory = tmp_path / f"harts-{harts}"
        directory.mkdir()
        executable = build(model, directory, toolchain, harts=harts,
                           extra_sources=(checked_source,))
        assert_real_nmsis_link(executable, real_provider)
        actual, report = riscv_external.run_external(
            executable, {"x": data}, directory / "run", timeout=30)
        assert report["passed"] and report["harts"] == harts and report["exit_code"] == 0
        assert_accuracy(actual, expected, 1e-4 + 2e-5 * np.abs(expected),
                        f"parallel NMSIS / blocked / Softmax / NMSIS, harts={harts}")
        ort_error = np.abs(actual.astype(np.float64) - ort_expected.astype(np.float64))
        assert float(ort_error.max()) < 0.001
        print(f"NMSIS row-parallel mixed harts={harts}: ORT max_abs={ort_error.max():.8g}")
        if serial is None:
            serial = actual.copy()
        else:
            np.testing.assert_array_equal(actual.view(np.uint32), serial.view(np.uint32))
        if harts == 4:
            failing = data.copy()
            failing[0, 0] = -2048.0
            with pytest.raises(RuntimeError, match="guest failed with status 0x00000004"):
                riscv_external.run_external(
                    executable, {"x": failing}, directory / "worker-failure", timeout=15)
            recovered, report = riscv_external.run_external(
                executable, {"x": data}, directory / "recovered", timeout=30)
            assert report["passed"] and report["exit_code"] == 0
            np.testing.assert_array_equal(recovered.view(np.uint32), serial.view(np.uint32))


def test_nmsis_parallel_rows_broadcast_on_qemu(tmp_path, real_provider, toolchain):
    from tests.test_tensor_c_external import ort_matmul

    program, bindings, data = matmul_experiment((2, 1, 17, 129), (1, 2, 129, 17))
    provider = nmsis_matmul(os.environ["SCRATCHV_NMSIS_ROOT"], parallel_rows=True)
    model = generate(program, bindings, provider, constant_storage="external",
                     kernel_calls=True)
    assert model.matmul_selections[0].implementation == "external"
    expected, _ = scalar_reference(data, bindings["weight"])
    ort_expected = ort_matmul(data, bindings["weight"])
    np.save(tmp_path / "input.npy", data)
    np.save(tmp_path / "fp64.npy", expected)
    np.save(tmp_path / "ort.npy", ort_expected)
    serial = None
    for harts in (1, 2, 4):
        directory = tmp_path / f"harts-{harts}"
        directory.mkdir()
        executable = build(model, directory, toolchain, harts=harts)
        assert_real_nmsis_link(executable, real_provider)
        actual, report = riscv_external.run_external(
            executable, {"x": data}, directory / "run", timeout=30)
        assert report["passed"] and report["harts"] == harts and report["exit_code"] == 0
        assert_accuracy(actual, expected, 1e-4 + 2e-5 * np.abs(expected),
                        f"parallel broadcast rows 17 / K129 / columns17, harts={harts}")
        error = np.abs(actual.astype(np.float64) - ort_expected.astype(np.float64))
        assert float(error.max()) < 0.001
        print(f"NMSIS row-parallel broadcast harts={harts}: ORT max_abs={error.max():.8g}")
        if serial is None:
            serial = actual.copy()
        else:
            np.testing.assert_array_equal(actual.view(np.uint32), serial.view(np.uint32))



def test_nmsis_measured_mixes_original_and_panel_on_qemu(tmp_path, real_provider, toolchain):
    import json
    from dataclasses import asdict
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper, TensorProto
    from scratchv.backend.external_matmul import MatMulMeasurement, MatMulPerformance
    from scratchv.ir.types import Value
    from tests.test_tensor_c_codegen import builder

    # All three operations reach their existing worker thresholds. Uneven
    # rows and a padded final panel exercise the 2x4 boundary paths cheaply.
    x = Value("x", shape=(129, 17))
    first = Value("first", shape=(17, 19))
    fallback = Value("fallback", shape=(19, 257))
    graph = builder(x)
    graph.program.global_values.extend((first, fallback))
    graph.ret(graph.softmax(graph.matmul(graph.matmul(x, first), fallback)))
    rng = np.random.default_rng(4112)
    data = rng.standard_normal(x.shape).astype(np.float32) / 8
    bindings = {value.name: rng.standard_normal(value.shape).astype(np.float32) / 8
                for value in (first, fallback)}
    selected = replace(real_provider, parallel_rows=True)
    # Deliberately invented durations test selection and layout correctness
    # only. They are not benchmark data and are never exported as measurements.
    profile = MatMulPerformance(
        provider=selected, target="riscv64", harts=4,
        environment="test fixture durations only; not measured performance",
        measurements=(
            MatMulMeasurement((129, 17), (17, 19), "constant", "column-panel", 1.0, 0.5),
            MatMulMeasurement((129, 19), (19, 257), "constant", "column-panel", 0.5, 1.0),
        ),
    )
    model = TensorCCodegen(
        graph.program, bindings, constant_storage="external", kernel_calls=True,
        matmul_policy="blocked_fma", matmul_tile="2x4", matmul_rhs_layout="column-panel",
        matmul_implementation="measured", matmul_performance=profile,
        external_matmul=selected, matmul_target="riscv64",
        external_matmul_numerics="allow_provider_fp32",
    ).generate()
    assert [choice.implementation for choice in model.matmul_selections] == ["external", "builtin"]
    assert model.external_matmul == selected and model.matmul_performance.harts == 4
    assert len(model.matmul_rhs_panels) == 1
    panel = model.matmul_rhs_panels[0]
    assert panel["name"] == "fallback" and panel["materialization"] == "padded-copy"
    assert panel["logical_shape"] == [19, 257] and panel["physical_shape"] == [65, 19, 4]
    assert "__builtin_fmaf" in model.source and "sv_softmax_" in model.source
    # One runtime declaration plus external, builtin and Softmax dispatches.
    assert model.source.count("sv_parallel_for(") == 4
    (tmp_path / "selection-fixture.json").write_text(json.dumps({
        "purpose": "Correctness-only selection fixture, not performance evidence",
        "measurements": [asdict(row) for row in profile.measurements],
        "selections": [asdict(choice) for choice in model.matmul_selections],
        "panels": list(model.matmul_rhs_panels),
    }, indent=2) + "\n")

    executable = build(model, tmp_path, toolchain, harts=4)
    assert_real_nmsis_link(executable, selected)
    # Inspect the actual serialized bundle consumed by this ELF. The first
    # provider gets its original row-major RHS; only the fallback is packed.
    restored = {}
    for entry in executable.manifest["tensors"]:
        restored[entry["name"]] = np.fromfile(
            tmp_path / "weights/weights.bin", dtype="<f4",
            count=entry["nbytes"] // 4, offset=entry["offset"],
        ).reshape(entry["shape"])
    np.testing.assert_array_equal(restored["first"].view(np.uint32),
                                  bindings["first"].view(np.uint32))
    packed = restored["fallback"]
    assert packed.shape == (65, 19, 4)
    logical = packed.transpose(1, 0, 2).reshape(19, 260)
    np.testing.assert_array_equal(logical[:, :257].view(np.uint32),
                                  bindings["fallback"].view(np.uint32))
    np.testing.assert_array_equal(logical[:, 257:], np.zeros((19, 3), np.float32))

    intermediate, _ = scalar_reference(data, bindings["first"])
    logits, _ = scalar_reference(intermediate, bindings["fallback"])
    exponentials = np.exp(logits - logits.max(axis=-1, keepdims=True))
    expected = exponentials / exponentials.sum(axis=-1, keepdims=True)
    nodes = [
        helper.make_node("MatMul", ["x", "first"], ["a"]),
        helper.make_node("MatMul", ["a", "fallback"], ["b"]),
        helper.make_node("Softmax", ["b"], ["y"], axis=-1),
    ]
    reference_model = helper.make_model(helper.make_graph(
        nodes, "measured-mixed-layout-correctness",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [129, 257])],
        [numpy_helper.from_array(array, name) for name, array in bindings.items()],
    ), opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    reference_path = tmp_path / "reference.onnx"
    onnx.save(reference_model, reference_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    ort_expected = ort.InferenceSession(str(reference_path), options,
        providers=["CPUExecutionProvider"]).run(None, {"x": data})[0]
    np.save(tmp_path / "input.npy", data)
    np.save(tmp_path / "fp64.npy", expected)
    np.save(tmp_path / "ort.npy", ort_expected)
    actual, report = riscv_external.run_external(
        executable, {"x": data}, tmp_path / "run", timeout=30)
    assert report["passed"] and report["guest_completed"] and report["exit_code"] == 0
    assert report["harts"] == 4
    assert_accuracy(actual, expected, 1e-4 + 2e-5 * np.abs(expected),
                    "measured NMSIS original / builtin 2x4 panel / Softmax, harts=4")
    ort_error = np.abs(actual.astype(np.float64) - ort_expected.astype(np.float64))
    assert float(ort_error.max()) < 0.001
    print(f"NMSIS measured mixed layout: ORT max_abs={ort_error.max():.8g}")


def test_nmsis_cancellation_on_qemu(tmp_path, real_provider, toolchain):
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
    model = generate(program, bindings, real_provider, constant_storage="external")
    executable = build(model, tmp_path, toolchain)
    actual = run(model, executable, data, tmp_path / "run")
    expected, scale = scalar_reference(data, matrix)
    k, u = data.shape[-1], 2**-24
    assert_accuracy(actual, expected, k*u/(1-k*u)*scale, "cancellation / mixed scales")


@pytest.mark.parametrize("storage,left,right", [
    ("inline", (0, 5), (5, 7)),
    ("external", (3, 5), (5, 0)),
    ("external", (2, 0), (0, 3)),
])
def test_nmsis_degenerate_shapes_do_not_link_library(
        tmp_path, descriptor, toolchain, storage, left, right):
    program, bindings, data = matmul_experiment(left, right)
    # Descriptor points at empty placeholder sources: a selected library would
    # fail to compile. Zero-size operations must not require it.
    model = generate(program, bindings, descriptor, constant_storage=storage)
    assert model.external_matmul is None
    executable = build(model, tmp_path, toolchain)
    actual = run(model, executable, data, tmp_path / "run")
    expected, _ = scalar_reference(data, bindings["weight"])
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint32), np.zeros_like(actual).view(np.uint32))


@pytest.mark.parametrize("storage", ["inline", "external"])
def test_nmsis_nonfinite_values_keep_runtime_failure(tmp_path, real_provider, toolchain, storage):
    program, bindings, data = matmul_experiment((1, 1), (1, 1))
    bindings["weight"].fill(2)
    model = generate(program, bindings, real_provider, constant_storage=storage)
    executable = build(model, tmp_path, toolchain)
    error = "failure status 2" if storage == "inline" else "status 0x00000002"
    for label, value in (("nan-input", np.nan), ("overflow", np.finfo(np.float32).max)):
        data.fill(value)
        with pytest.raises(RuntimeError, match=error):
            run(model, executable, data, tmp_path / label)


ADAPTER_CHECKS = """
#include <stddef.h>
#include <stdint.h>
extern int sv_nmsis_matmul_f32_v1(const float *, const float *, float *,
                                 uint64_t, uint64_t, uint64_t);
int sv_checked_nmsis_matmul(const float *a, const float *b, float *c,
                           uint64_t m, uint64_t n, uint64_t k) {
    float aa[2] = {1.0f, 2.0f}, bb[2] = {3.0f, 4.0f};
    float cc[3] = {-1.0f, -2.0f, 99.0f};
    union { float f[3]; uint32_t u[3]; } z = {{-1.0f, -2.0f, 99.0f}};
    if (sv_nmsis_matmul_f32_v1(NULL, NULL, NULL, 65536, 1, 1) == 0 ||
        sv_nmsis_matmul_f32_v1(NULL, NULL, NULL, 1, 65536, 1) == 0 ||
        sv_nmsis_matmul_f32_v1(NULL, NULL, NULL, 1, 1, 65536) == 0)
        return 91;
    if (sv_nmsis_matmul_f32_v1(NULL, NULL, NULL, 0, 2, 3) != 0 ||
        sv_nmsis_matmul_f32_v1(NULL, NULL, NULL, 2, 0, 3) != 0 ||
        sv_nmsis_matmul_f32_v1(NULL, NULL, z.f, 1, 2, 0) != 0 ||
        z.u[0] != 0 || z.u[1] != 0 || z.f[2] != 99.0f)
        return 92;
    if (sv_nmsis_matmul_f32_v1(NULL, bb, cc, 1, 1, 2) == 0 ||
        sv_nmsis_matmul_f32_v1(aa, NULL, cc, 1, 1, 2) == 0 ||
        sv_nmsis_matmul_f32_v1(aa, bb, NULL, 1, 1, 2) == 0 ||
        sv_nmsis_matmul_f32_v1(NULL, NULL, NULL, 1, 1, 0) == 0)
        return 93;
    for (int repeat = 0; repeat < 2; ++repeat)
        if (sv_nmsis_matmul_f32_v1(aa, bb, cc, 1, 1, 2) != 0 ||
            cc[0] != 11.0f || cc[1] != -2.0f || cc[2] != 99.0f ||
            aa[0] != 1.0f || aa[1] != 2.0f || bb[0] != 3.0f || bb[1] != 4.0f)
            return 94;
    return sv_nmsis_matmul_f32_v1(a, b, c, m, n, k);
}
"""


def test_nmsis_adapter_boundaries_in_guest(tmp_path, real_provider, toolchain):
    source = tmp_path / "adapter_checks.c"
    source.write_text(ADAPTER_CHECKS, encoding="utf-8")
    checked = replace(real_provider, symbol="sv_checked_nmsis_matmul")
    program, bindings, data = matmul_experiment((3, 5), (5, 7))
    model = generate(program, bindings, checked, constant_storage="external")
    executable = build(model, tmp_path, toolchain, extra_sources=(source,))
    assert_real_nmsis_link(executable, real_provider)
    actual = run(model, executable, data, tmp_path / "run")
    expected, _ = scalar_reference(data, bindings["weight"])
    assert_accuracy(actual, expected, 1e-4 + 2e-5 * np.abs(expected), "adapter boundary checks")
