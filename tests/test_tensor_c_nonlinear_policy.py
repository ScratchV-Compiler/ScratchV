"""Explicit portable nonlinear arithmetic, using the existing FP32 oracle/ABI."""
from dataclasses import replace
import ctypes
import re

import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorCCodegenError
from scratchv.ir.types import Value
from scratchv.verification import fp32_reference
from scratchv.verification.fp32_eigen import trig
from tests.test_tensor_c_codegen import builder
from tests.test_tensor_c_external import (
    arguments, host_cc, library, pointers, _PARALLEL_HOST_RUNTIME,
    _softmax_pair_experiment, _softmax_experiment,
)

POLICY = "fp32-reference"
MODES = [("inline", False), ("external", False), ("external", True)]


def oracle_softmax(x, axis=-1):
    # Explicit reference arithmetic, independent of local/hidden CPU features.
    moved = np.moveaxis(x, axis, -1)
    weights = fp32_reference._exp_nonpositive(
        moved - moved.max(axis=-1, keepdims=True), cpu_strategy="avx512")
    result = weights * (np.float32(1) / fp32_reference._sum_lanes(weights, 16))
    return np.moveaxis(result, -1, axis)


def abi(model, feed):
    if model.constant_storage == "external":
        weights, workspace, output, args = arguments(model, feed)
        return output, args, (weights, workspace)
    output = np.full(model.output.shape, -999, np.float32)
    return output, [pointers([feed[x.name] for x in model.inputs]), output.ctypes.data], ()


@pytest.mark.parametrize("invalid", [None, "", "auto", "avx512", True, 1, []])
def test_nonlinear_policy_rejects_invalid_values(invalid):
    x = Value("x", shape=(4, 8))
    graph = builder(x)
    graph.ret(graph.softmax(x))
    with pytest.raises(TensorCCodegenError, match="nonlinear_policy"):
        TensorCCodegen(graph.program, nonlinear_policy=invalid).generate()


@pytest.mark.parametrize("storage,kernels", MODES)
def test_default_source_and_unused_operations_stay_unchanged(storage, kernels):
    x = Value("x", shape=(4, 17))
    graph = builder(x)
    graph.ret(graph.softmax(graph.sigmoid(graph.add(graph.sin(x), graph.cos(x)))))
    options = dict(constant_storage=storage, kernel_calls=kernels)
    old = TensorCCodegen(graph.program, **options).generate()
    assert old == TensorCCodegen(graph.program, nonlinear_policy="libm", **options).generate()
    assert old.nonlinear_policy == "libm" and old.mean_policy == "sequential"
    assert "sv_fp32_reference_" not in old.source
    generator = TensorCCodegen(graph.program, nonlinear_policy=POLICY, **options)
    selected = generator.generate()
    assert selected == generator.generate()
    assert selected.workspace_bytes == old.workspace_bytes
    for name in ("trig", "sigmoid", "exp_nonpositive", "sum_sixteen"):
        assert selected.source.count(f"static float sv_fp32_reference_{name}(") == 1
    assert "SPDX-License-Identifier: MPL-2.0" in selected.source
    assert "Copyright (c) Microsoft Corporation" in selected.source
    generator.nonlinear_policy = "libm"
    assert generator.generate() == old  # A prior reference build leaves no state.
    plain = builder(x)
    plain.ret(plain.add(plain.exp(x), plain.pow(x, plain.make_const(2.0))))
    implicit = TensorCCodegen(plain.program, **options).generate()
    explicit = TensorCCodegen(plain.program, nonlinear_policy=POLICY, **options).generate()
    assert explicit.source == implicit.source  # Exp/Pow are outside this policy.
    assert explicit.mean_policy == "sequential"


@pytest.mark.parametrize("storage,kernels", MODES)
def test_trig_and_sigmoid_match_fixed_reference_and_preserve_libm_fallback(
        tmp_path, host_cc, storage, kernels):
    threshold = np.float32(18000)
    small = np.nextafter(np.float32(0), np.float32(1))
    values = np.concatenate([
        np.linspace(-17999, 17999, 513, dtype=np.float32),
        np.array([0, -0.0, small, -small, 18, -18,
                  np.nextafter(threshold, np.float32(0)), threshold,
                  np.nextafter(threshold, np.float32(np.inf)), 1e20,
                  np.finfo(np.float32).max], np.float32),
        np.array([-threshold, -np.nextafter(threshold, np.float32(0)),
                  -np.nextafter(threshold, np.float32(np.inf)), -1e20,
                  -np.finfo(np.float32).max], np.float32),
    ])
    x = Value("x", shape=values.shape)
    graph = builder(x)
    graph.ret(graph.concat([graph.sin(x), graph.cos(x), graph.sigmoid(x)], axis=0))
    opts = dict(constant_storage=storage, kernel_calls=kernels)
    baseline = TensorCCodegen(graph.program, **opts).generate()
    selected = TensorCCodegen(graph.program, nonlinear_policy=POLICY, **opts).generate()
    feed = {"x": values}
    with library(tmp_path / "libm", baseline, host_cc) as run:
        original, args, keep = abi(baseline, feed)
        assert run(*args) == 0
    with library(tmp_path / "reference", selected, host_cc) as run:
        actual, args, keep = abi(selected, feed)
        assert run(*args) == 0
        expected = np.concatenate([trig(values, sine=True), trig(values, sine=False),
                                   fp32_reference.sigmoid([values], {}, None)])
        fallback = np.abs(values) >= threshold
        for block in (0, 1):
            # The explicit fallback is the original platform libm, not NumPy.
            expected[block*values.size:(block+1)*values.size][fallback] = (
                original[block*values.size:(block+1)*values.size][fallback])
        np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
        assert np.signbit(actual[np.flatnonzero(np.signbit(values) & (values == 0))]).all()
        for invalid in (np.nan, np.inf, -np.inf):
            bad = values.copy()
            bad[0] = invalid
            args[0] = pointers([bad])
            actual.fill(-999)
            assert run(*args) == 2 and np.all(actual == -999)
            args[0] = pointers([values])
            assert run(*args) == 0
            np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))


@pytest.mark.parametrize("storage,kernels", MODES)
def test_softmax_axes_tails_empty_rows_and_fixed_cpu_independent_order(
        tmp_path, host_cc, monkeypatch, storage, kernels):
    specifications = [((3, n), -1) for n in (1, 7, 16, 17, 257)]
    specifications += [((2, 17, 3), 1), ((2, 17, 3), 0), ((0, 17), -1)]
    params = [Value(f"x{i}", shape=shape) for i, (shape, _) in enumerate(specifications)]
    graph = builder(*params)
    graph.ret(graph.concat([graph.reshape(graph.softmax(x, axis=axis), (int(np.prod(shape)),))
                            for x, (shape, axis) in zip(params, specifications)], axis=0))
    feed = {}
    rng = np.random.default_rng(217)
    for x in params:
        feed[x.name] = rng.normal(size=x.shape).astype("f4") * np.float32(20)
    expected = np.concatenate([oracle_softmax(feed[x.name], axis).ravel()
                               for x, (_, axis) in zip(params, specifications)])
    # Backend selection must not consult host CPU dispatch or reference env.
    monkeypatch.setenv("SCRATCHV_FP32_REFERENCE_CPU", "avx2-fma3")
    model = TensorCCodegen(graph.program, constant_storage=storage, kernel_calls=kernels,
                          nonlinear_policy=POLICY).generate()
    assert "ldexpf" in model.source and "sv_reference_lanes[16]" in model.source
    actual, args, keep = abi(model, feed)
    with library(tmp_path, model, host_cc) as run:
        assert run(*args) == 0
    np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))


@pytest.mark.parametrize("kernels", [False, True])
def test_reference_softmax_reuses_callback_with_separate_contexts_and_recovers(
        tmp_path, host_cc, kernels):
    program, feed = _softmax_pair_experiment()
    model = TensorCCodegen(program, constant_storage="external", kernel_calls=kernels,
                          nonlinear_policy=POLICY).generate()
    assert len(re.findall(r"static void sv_softmax_\w+_range\(", model.source)) == 1
    expected = np.concatenate([oracle_softmax(feed["x"]), oracle_softmax(feed["y"])], axis=0)
    parallel = replace(model, source=model.source + _PARALLEL_HOST_RUNTIME)
    for harts in (1, 4):
        actual, args, keep = abi(model, feed)
        with library(tmp_path / str(harts), parallel,
                     [*host_cc, f"-DSCRATCHV_HARTS={harts}", "-pthread"]) as run:
            for inputs, wanted in [(feed, expected),
                                   ({"x": feed["y"], "y": feed["x"]},
                                    np.concatenate(np.split(expected, 2)[::-1], axis=0)),
                                   (feed, expected)]:
                args[0] = pointers([inputs["x"], inputs["y"]])
                assert run(*args) == 0
                np.testing.assert_array_equal(actual.view("u4"), wanted.view("u4"))
            bad = feed["y"].copy()
            bad[-1, :2] = [np.finfo("f4").max, -np.finfo("f4").max]
            args[0] = pointers([feed["x"], bad])
            actual.fill(-999)
            assert run(*args) == 2 and np.all(actual == -999)
            args[0] = pointers([feed["x"], feed["y"]])
            assert run(*args) == 0
            np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))


def test_reference_softmax_chain_reuses_workspace(tmp_path, host_cc):
    program, x = _softmax_experiment((32, 1024), passes=4)
    model = TensorCCodegen(program, constant_storage="external", kernel_calls=True,
                          nonlinear_policy=POLICY).generate()
    assert model.workspace_bytes < 4*x.nbytes
    expected = x
    for _ in range(4):
        expected = oracle_softmax(expected)
    parallel = replace(model, source=model.source + _PARALLEL_HOST_RUNTIME)
    actual, args, keep = abi(model, {"x": x})
    with library(tmp_path, parallel, [*host_cc, "-DSCRATCHV_HARTS=4", "-pthread"]) as run:
        assert run(*args) == 0
    np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))


@pytest.mark.parametrize("kernels", [False, True])
def test_reference_softmax_output_finite_checks_before_workspace_recovery(
        tmp_path, host_cc, kernels):
    shape = (17, 2049)
    x, y = Value("x", shape=shape), Value("y", shape=shape)
    graph = builder(x, y)
    graph.softmax(x)
    graph.ret(graph.softmax(y))  # No downstream finite check can hide failure.
    model = TensorCCodegen(graph.program, constant_storage="external", kernel_calls=kernels,
                          nonlinear_policy=POLICY).generate()
    feed = {"x": np.zeros(shape, "f4"), "y": np.zeros(shape, "f4")}
    feed["y"][-1, 0] = -13
    expected = oracle_softmax(feed["y"])
    source = model.source.replace("float v=sv_fp32_reference_exp_nonpositive(shifted);",
                                  "float v=sv_test_reference_exp(shifted);")
    fault = r"""
int sv_test_fault, sv_test_hits;
float sv_test_reference_exp(float x) {
    if (sv_test_fault && x == -13.0f) {
        __atomic_fetch_add(&sv_test_hits,1,__ATOMIC_RELAXED);
        if (sv_test_fault == 1) return -2048.0f;
        union {uint32_t u; float f;} bad={0x7fc00000U}; return bad.f;
    }
    return sv_fp32_reference_exp_nonpositive(x);
}
"""
    injected = replace(model, source="extern float sv_test_reference_exp(float);\n"
                       + source + _PARALLEL_HOST_RUNTIME + fault)
    for harts in (1, 4):
        actual, args, keep = abi(model, feed)
        directory = tmp_path / str(harts)
        with library(directory, injected,
                     [*host_cc, f"-DSCRATCHV_HARTS={harts}", "-pthread"]) as run:
            loaded = ctypes.CDLL(str(directory / "model.so"))
            mode = ctypes.c_int.in_dll(loaded, "sv_test_fault")
            hits = ctypes.c_int.in_dll(loaded, "sv_test_hits")
            for number in (1, 2):
                mode.value, hits.value = number, 0
                actual.fill(-999)
                assert run(*args) == 2 and hits.value == 1
                assert np.all(actual == -999)
                mode.value, hits.value = 0, 0
                assert run(*args) == 0 and hits.value == 0
                np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
