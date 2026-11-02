"""Small real RV64 checks for the explicitly selected last-axis mean order."""
import os
import json

import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value
from scratchv.runtime.riscv_external import build_external, run_external
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.weight_bundle import write_weight_bundle
from scratchv.verification.fp32_eigen import mean_last


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="explicit RV64 tools required")
@pytest.mark.parametrize("harts", [1, 4])
@pytest.mark.parametrize("width", [13, 128, 1024])
def test_explicit_mean_order_rv64_and_context_reuse(tmp_path, harts, width):
    builder = IRBuilder()
    x = Value("x", shape=(7, width))
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.ret(builder.reduce_mean(x, axes=(-1,), keepdims=True))
    artifact = TensorCCodegen(builder.program, constant_storage="external", kernel_calls=True,
                             mean_policy="two-four-lane-last-axis").generate()
    write_weight_bundle(artifact.external_weights, artifact.external_initializers, tmp_path / "weights")
    executable = build_external(artifact, tmp_path / "weights", tmp_path / "build",
                                discover_toolchain(), harts=harts)
    assert executable.evidence["mean_policy"] == "two-four-lane-last-axis"
    assert executable.evidence["qemu_workspace_layout"] == "none"
    rng = np.random.default_rng(1200 + width)
    for index in range(2):
        data = (rng.normal(size=(7, width)) * (1 if index == 0 else 256)).astype(np.float32)
        # Exact signs and cancellation exercise more than a positive-only mean.
        data[0] = np.float32(-0.0)
        data[1, ::2] *= np.float32(-1)
        expected = mean_last(data)
        before = data.copy()
        actual, report = run_external(executable, {"x": data}, tmp_path / f"run{index}", timeout=30)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(data.view(np.uint32), before.view(np.uint32))
        assert report["passed"] and report["qemu_reaped"] and report["harts"] == harts
        assert report["mean_policy"] == "two-four-lane-last-axis"


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="explicit RV64 tools required")
@pytest.mark.parametrize("harts", [1, 4])
def test_explicit_mean_overflow_propagates_and_build_remains_reusable(tmp_path, harts):
    builder = IRBuilder()
    x = Value("x", shape=(2, 16))
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.ret(builder.reduce_mean(x, axes=(-1,), keepdims=False))
    artifact = TensorCCodegen(builder.program, constant_storage="external", kernel_calls=True,
                             mean_policy="two-four-lane-last-axis").generate()
    write_weight_bundle(artifact.external_weights, artifact.external_initializers, tmp_path / "weights")
    executable = build_external(artifact, tmp_path / "weights", tmp_path / "build",
                                discover_toolchain(), harts=harts)
    with pytest.raises(RuntimeError, match="status 0x00000002"):
        run_external(executable, {"x": np.full((2, 16), np.finfo(np.float32).max, np.float32)},
                     tmp_path / "overflow", timeout=30)
    failure = json.loads((tmp_path / "overflow/run.json").read_text())
    assert not failure["passed"] and failure["qemu_reaped"]
    with pytest.raises(ProcessLookupError):
        os.killpg(failure["qemu_pid"], 0)
    # A valid new context uses the exact same executable after a guest error.
    data = np.arange(32, dtype=np.float32).reshape(2, 16) / np.float32(8)
    actual, report = run_external(executable, {"x": data}, tmp_path / "reused", timeout=30)
    np.testing.assert_array_equal(actual.view(np.uint32), mean_last(data, False).view(np.uint32))
    assert report["passed"] and report["qemu_reaped"]
