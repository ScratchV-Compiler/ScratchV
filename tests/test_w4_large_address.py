"""Small real arithmetic with external weights physically above 4 GiB.

The unused reserved workspace is deliberate address-space coverage, not a model
memory measurement. It does not allocate a 2 GiB NumPy host buffer.
"""
from dataclasses import replace
import os

import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value
from scratchv.runtime.riscv_external import build_external, run_external
from scratchv.runtime.riscv_tensor import discover_toolchain
from scratchv.runtime.weight_bundle import write_weight_bundle


@pytest.mark.skipif(not os.environ.get("SCRATCHV_QEMU"), reason="explicit RV64 toolchain required")
def test_weight_pointer_above_32bit_limit(tmp_path):
    builder = IRBuilder()
    x, weight = Value("x", shape=(2,)), Value("w", shape=(2,))
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.program.global_values.append(weight)
    builder.ret(builder.mul(x, weight))
    artifact = TensorCCodegen(builder.program, {"w": np.array([1.5, -2], dtype=np.float32)},
                             constant_storage="external", kernel_calls=True).generate()
    # Capacity may exceed the generated function's actual minimum requirement.
    artifact = replace(artifact, workspace_bytes=2 * 1024**3)
    write_weight_bundle(artifact.external_weights, artifact.external_initializers, tmp_path / "weights")
    exe = build_external(artifact, tmp_path / "weights", tmp_path / "build", discover_toolchain())
    assert exe.layout["weights_base"] > 0xFFFFFFFF
    actual, report = run_external(exe, {"x": np.array([2, 3], dtype=np.float32)},
                                  tmp_path / "run", timeout=30)
    np.testing.assert_array_equal(actual, np.array([3, -6], dtype=np.float32))
    assert report["guest_completed"] and report["exit_code"] == 0
