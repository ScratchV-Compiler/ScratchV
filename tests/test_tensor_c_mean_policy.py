"""Explicit FP32 Mean order; use the existing oracle and Host ABI helpers."""

import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorCCodegenError
from scratchv.ir.types import DataType, Value
from scratchv.verification.fp32_eigen import mean_last
from tests.test_tensor_c_codegen import builder
from tests.test_tensor_c_external import arguments, host_cc, library, pointers


POLICY = "two-four-lane-last-axis"


def make_mean(shape, *, axes=(-1,), keepdims=True, **options):
    x = Value("x", shape=shape)
    graph = builder(x)
    graph.ret(graph.reduce_mean(x, axes, keepdims))
    return TensorCCodegen(graph.program, **options).generate()


@pytest.mark.parametrize("invalid", [None, "", "eigen", True, 1, []])
def test_mean_policy_rejects_unknown_values(invalid):
    with pytest.raises(TensorCCodegenError, match="mean_policy"):
        make_mean((2, 3), mean_policy=invalid)


@pytest.mark.parametrize("storage,kernels", [
    ("inline", False), ("external", False), ("external", True),
])
def test_mean_policy_default_and_non_last_axes_are_unchanged(storage, kernels):
    options = dict(constant_storage=storage, kernel_calls=kernels)
    implicit = make_mean((2, 3, 5), **options)
    explicit = make_mean((2, 3, 5), mean_policy="sequential", **options)
    assert implicit == explicit
    assert implicit.mean_policy == "sequential"
    assert "sv_two_four_lane_mean_last" not in implicit.source
    for axes in ((0,), (1,), (0, 2), None):
        old = make_mean((2, 3, 5), axes=axes, **options)
        selected = make_mean((2, 3, 5), axes=axes, mean_policy=POLICY, **options)
        assert selected.mean_policy == POLICY
        assert selected.source == old.source
        assert selected.workspace_bytes == old.workspace_bytes
    selected = make_mean((2, 3, 5), mean_policy=POLICY, **options)
    assert "sv_two_four_lane_mean_last" in selected.source
    assert "SPDX-License-Identifier: MPL-2.0" in selected.source
    assert selected.workspace_bytes == implicit.workspace_bytes
    # Canonical positive/negative axis spellings choose identical code.
    assert selected.source == make_mean((2, 3, 5), axes=(2,),
                                        mean_policy=POLICY, **options).source


@pytest.mark.parametrize("storage,kernels", [
    ("inline", False), ("external", False), ("external", True),
])
def test_mean_policy_matches_existing_fp32_oracle_at_prefix_tail_boundaries(
        tmp_path, host_cc, storage, kernels):
    widths = (1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 31, 127, 128, 1024)
    params = [Value(f"x{width}", shape=(5, width)) for width in widths]
    graph = builder(*params)
    outputs = []
    for x in params:
        for keepdims in (False, True):
            y = graph.reduce_mean(x, (-1,), keepdims)
            outputs.append(graph.reshape(y, (5,)))
    graph.ret(graph.concat(outputs, axis=0))
    codegen = TensorCCodegen(graph.program, constant_storage=storage,
                            kernel_calls=kernels, mean_policy=POLICY)
    artifact = codegen.generate()
    assert artifact == codegen.generate()  # Helper state resets on regeneration.
    assert artifact.source.count("static float sv_two_four_lane_mean_last(") == 1
    rng = np.random.default_rng(53)
    with library(tmp_path, artifact, host_cc) as run:
        for kind in ("random", "cancellation", "negative-zero", "subnormal"):
            feed = {}
            for x in params:
                # Legal 4-byte ABI alignment, deliberately shifted from 16-byte
                # alignment; canonical row order must not depend on its address.
                backing = np.empty(5 * x.shape[-1] + 1, np.float32)
                values = backing[1:].reshape(x.shape)
                if kind == "random":
                    values[:] = rng.normal(size=x.shape).astype(np.float32)
                elif kind == "cancellation":
                    values[:] = np.resize(np.array([2**24, 1, -2**24, 1, -3, 2, 5],
                                                    np.float32), x.shape)
                elif kind == "negative-zero":
                    values[:] = np.float32(-0.0)
                else:
                    values[:] = np.resize(np.array([1, -2, 3, -4], np.float32)
                                          * np.nextafter(np.float32(0), np.float32(1)),
                                          x.shape)
                values.flags.writeable = False
                feed[x.name] = values
            expected = np.concatenate([mean_last(feed[x.name], keepdims=False)
                                       for x in params for _ in range(2)])
            saved = [values.copy() for values in feed.values()]
            if storage == "external":
                weights, workspace, actual, args = arguments(artifact, feed)
            else:
                actual = np.empty(artifact.output.shape, np.float32)
                args = [pointers(list(feed.values())), actual.ctypes.data]
            assert run(*args) == 0
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            for original, values in zip(saved, feed.values()):
                np.testing.assert_array_equal(values.view(np.uint32), original.view(np.uint32))


@pytest.mark.parametrize("shape,axes", [((2, 0), (-1,)), ((0,), None)])
def test_mean_policy_empty_reduction_keeps_existing_rejection(shape, axes):
    with pytest.raises(TensorCCodegenError, match="empty axis"):
        make_mean(shape, axes=axes, mean_policy=POLICY)


@pytest.mark.parametrize("storage,kernels", [
    ("inline", False), ("external", False), ("external", True),
])
def test_mean_policy_empty_leading_dimensions(tmp_path, host_cc, storage, kernels):
    model = make_mean((2, 0, 7), keepdims=False, constant_storage=storage,
                      kernel_calls=kernels, mean_policy=POLICY)
    x = np.empty((2, 0, 7), np.float32)
    with library(tmp_path, model, host_cc) as run:
        if storage == "external":
            weights, workspace, output, args = arguments(model, x)
        else:
            output = np.empty(model.output.shape, np.float32)
            args = [pointers([x]), output.ctypes.data]
        assert run(*args) == 0
        assert output.shape == (2, 0)
        np.testing.assert_array_equal(output, mean_last(x, keepdims=False))


@pytest.mark.parametrize("kernels", [False, True])
def test_mean_policy_numeric_error_propagation_and_workspace_recovery(tmp_path, host_cc, kernels):
    x = Value("x", shape=(5, 13))
    graph = builder(x)
    first = graph.reduce_mean(x, (-1,), True)
    shifted = graph.add(x, first)
    second = graph.reduce_mean(shifted, (-1,), True)
    graph.ret(graph.add(second, second))
    model = TensorCCodegen(graph.program, constant_storage="external",
                          kernel_calls=kernels, mean_policy=POLICY).generate()
    assert model.workspace_bytes < (5 * 13 + 5 * 3) * 4
    x0 = np.arange(65, dtype=np.float32).reshape(5, 13) / np.float32(10)
    weights, workspace, output, args = arguments(model, x0)
    with library(tmp_path, model, host_cc) as run:
        for bad in (np.nan, np.inf, np.finfo(np.float32).max):
            x0[:] = bad
            output[:] = -123
            assert run(*args) == 2
            assert np.all(output == -123)  # No final result copied on failure.
            # New input pointer/context, exact same caller workspace.
            x1 = np.linspace(-2, 3, 65, dtype=np.float32).reshape(5, 13)
            args[0] = pointers([x1])
            assert run(*args) == 0
            expected = mean_last(x1 + mean_last(x1))
            expected = expected + expected
            np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))
            args[0] = pointers([x0])
        # Capacity and overlap validation must still precede touching workspace.
        workspace[:] = np.uint64(0x1122334455667788)
        output[:] = -123
        args[0] = pointers([x1])
        args[3] = model.workspace_bytes - 1
        assert run(*args) == 1
        assert np.all(workspace == np.uint64(0x1122334455667788))
        assert np.all(output == -123)


def test_mean_policy_does_not_bypass_fp32_type_contract():
    x = Value("x", DataType.INT32, shape=(2, 4))
    graph = builder(x)
    graph.ret(graph.reduce_mean(x, (-1,), True))
    with pytest.raises(TensorCCodegenError):
        TensorCCodegen(graph.program, mean_policy=POLICY).generate()
