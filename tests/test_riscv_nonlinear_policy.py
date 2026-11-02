"""Real RV64 integration of the explicit portable nonlinear/Mean policies."""
import json
import os

import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value
from scratchv.runtime.riscv_external import build_external, run_external
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.weight_bundle import write_weight_bundle
from scratchv.verification.fp32_eigen import mean_last, trig
from scratchv.verification.fp32_reference import sigmoid, softmax


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="explicit RV64 tools required")
@pytest.mark.parametrize("harts", [1, 4])
def test_reference_nonlinear_rv64_contexts_and_error_recovery(tmp_path, monkeypatch, harts):
    # Select the saved arithmetic order, independent of the host's SIMD support.
    monkeypatch.setenv("SCRATCHV_FP32_REFERENCE_CPU", "avx512")
    graph = IRBuilder()
    x = Value("x", shape=(128, 256))
    graph.new_function("main", [x])
    graph.new_block("entry")
    graph.ret(graph.concat([
        graph.sin(x), graph.cos(x), graph.sigmoid(x),
        graph.softmax(x, axis=-1), graph.softmax(x, axis=0),
        graph.reduce_mean(x, axes=(-1,), keepdims=True),
    ], axis=1))
    artifact = TensorCCodegen(
        graph.program, constant_storage="external", kernel_calls=True,
        mean_policy="two-four-lane-last-axis", nonlinear_policy="fp32-reference",
    ).generate()
    write_weight_bundle(artifact.external_weights, artifact.external_initializers, tmp_path / "weights")
    executable = build_external(artifact, tmp_path / "weights", tmp_path / "build",
                                discover_toolchain(), harts=harts)
    assert executable.evidence["nonlinear_policy"] == "fp32-reference"
    assert executable.evidence["mean_policy"] == "two-four-lane-last-axis"
    assert executable.evidence["qemu_workspace_layout"] == "none"
    rng = np.random.default_rng(2718)
    for context in range(2):
        data = rng.uniform(-255, 255, size=x.shape).astype(np.float32)
        data[0, :8] = np.array([0., -0., 1e-44, -1e-44, -18., 18., -104., 104.], np.float32)
        wanted = np.concatenate([
            trig(data, sine=True), trig(data, sine=False), sigmoid([data], {}, None),
            softmax([data], {"axis": -1}, None), softmax([data], {"axis": 0}, None),
            mean_last(data),
        ], axis=1)
        saved = data.copy()
        actual, record = run_external(executable, {"x": data}, tmp_path / f"context-{context}", timeout=60)
        np.testing.assert_array_equal(actual.view(np.uint32), wanted.view(np.uint32))
        np.testing.assert_array_equal(data.view(np.uint32), saved.view(np.uint32))
        assert record["passed"] and record["qemu_reaped"] and record["harts"] == harts
        assert record["nonlinear_policy"] == "fp32-reference"
        if context == 0:
            # Finite inputs whose stable Softmax subtraction overflows must
            # still fail, then the same ELF/bundle must accept a fresh context.
            invalid = np.zeros(x.shape, np.float32)
            invalid[-1, :2] = [np.finfo(np.float32).max, -np.finfo(np.float32).max]
            with pytest.raises(RuntimeError, match="status 0x00000002"):
                run_external(executable, {"x": invalid}, tmp_path / "overflow", timeout=60)
            failure = json.loads((tmp_path / "overflow/run.json").read_text())
            assert not failure["passed"] and failure["qemu_reaped"]
            with pytest.raises(ProcessLookupError):
                os.killpg(failure["qemu_pid"], 0)
