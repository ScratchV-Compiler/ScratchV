"""Actual host/RV64 agreement for the same generated blocked-FMA MatMul.

This is toolchain conformance on small independent graphs, not full-model W4
acceptance. A configured QEMU with defective FMA must fail, not xfail/skip.
"""
import os

import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.runtime.riscv_external import build_external, run_external
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.weight_bundle import write_weight_bundle
from tests.test_tensor_c_external import arguments, host_cc, library, matmul_experiment


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires explicit SCRATCHV_QEMU/CC")
@pytest.mark.parametrize("harts", [1, 2, 4, "auto"])
@pytest.mark.parametrize("left,right", [
    # Each case exceeds the parallel threshold, including K-block and tail boundaries.
    ((12, 127), (127, 25)), ((12, 128), (128, 25)), ((12, 129), (129, 25)),
    ((4, 1024), (1024, 9)), ((2, 1, 8, 129), (1, 2, 129, 9)),
])
def test_blocked_fma_host_and_rv64_are_bitwise_equal(tmp_path, host_cc, left, right, harts):
    program, bindings, data = matmul_experiment(left, right)
    artifact = TensorCCodegen(program, bindings, constant_storage="external",
                             kernel_calls=True, matmul_policy="blocked_fma").generate()
    weights, workspace, expected, args = arguments(artifact, data)
    with library(tmp_path / "host", artifact, host_cc) as host:
        assert host(*args) == 0
    write_weight_bundle(artifact.external_weights, artifact.external_initializers, tmp_path / "weights")
    executable = build_external(artifact, tmp_path / "weights", tmp_path / "build", discover_toolchain(), harts=harts)
    actual, report = run_external(executable, {"x": data}, tmp_path / "qemu", timeout=30)
    assert report["passed"] and report["guest_completed"] and report["exit_code"] == 0
    assert report["harts"] == executable.harts
    assert type(executable.harts) is int and executable.harts in (1, 2, 4)
    if harts != "auto":
        assert executable.harts == harts
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="requires explicit SCRATCHV_QEMU/CC")
@pytest.mark.parametrize("harts", [1, 4])
@pytest.mark.parametrize("tile", ["2x4", "1x8"])
@pytest.mark.parametrize("left,right", [
    # The first three exceed the parallel threshold and include tile tails,
    # each side of K=128, and broadcast batches. The last stays serial.
    ((17, 127), (127, 17)),
    ((17, 128), (128, 19)),
    ((2, 1, 9, 129), (1, 2, 129, 9)),
    ((1, 3), (3, 3)),
])
def test_microtiles_rv64_match_host_baseline_bits(tmp_path, host_cc, left, right, tile, harts):
    program, bindings, data = matmul_experiment(left, right)
    baseline = TensorCCodegen(program, bindings, constant_storage="external",
                             kernel_calls=True, matmul_policy="blocked_fma").generate()
    weights, workspace, expected, args = arguments(baseline, data)
    with library(tmp_path / "host_baseline", baseline, host_cc) as host:
        assert host(*args) == 0
    artifact = TensorCCodegen(program, bindings, constant_storage="external",
                             kernel_calls=True, matmul_policy="blocked_fma",
                             matmul_tile=tile).generate()
    write_weight_bundle(artifact.external_weights, artifact.external_initializers,
                        tmp_path / "weights")
    executable = build_external(artifact, tmp_path / "weights", tmp_path / "build",
                                discover_toolchain(), harts=harts)
    actual, report = run_external(executable, {"x": data}, tmp_path / "qemu", timeout=30)
    assert report["passed"] and report["guest_completed"] and report["exit_code"] == 0
    assert report["harts"] == executable.harts == harts
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    # This ABI returns tensors, not per-hart FCSR. Independent FMA conformance
    # checks cover floating-point state; this assertion covers model output bits.
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
