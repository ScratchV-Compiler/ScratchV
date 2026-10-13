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
@pytest.mark.parametrize("left,right", [
    ((3, 127), (127, 5)), ((3, 128), (128, 7)), ((3, 129), (129, 5)),
    ((3, 1024), (1024, 9)), ((2, 1, 3, 129), (1, 2, 129, 5)),
])
def test_blocked_fma_host_and_rv64_are_bitwise_equal(tmp_path, host_cc, left, right):
    program, bindings, data = matmul_experiment(left, right)
    artifact = TensorCCodegen(program, bindings, constant_storage="external",
                             kernel_calls=True, matmul_policy="blocked_fma").generate()
    weights, workspace, expected, args = arguments(artifact, data)
    with library(tmp_path / "host", artifact, host_cc) as host:
        assert host(*args) == 0
    write_weight_bundle(artifact.external_weights, artifact.external_initializers, tmp_path / "weights")
    executable = build_external(artifact, tmp_path / "weights", tmp_path / "build", discover_toolchain())
    actual, report = run_external(executable, {"x": data}, tmp_path / "qemu", timeout=30)
    assert report["passed"] and report["guest_completed"] and report["exit_code"] == 0
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
