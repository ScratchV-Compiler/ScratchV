"""Exercise failures that must still reach the caller after fusing the three operators."""
from dataclasses import replace
import os

import numpy as np
import pytest

from probes.w6_performance.attention_chain import INPUT_SHAPE, chain_program
from probes.w6_performance.attention_chain_kernel import attention_chain_artifact
from scratchv.backend.tensor_c_codegen import TensorCCodegenError
from tests.test_tensor_c_external import _PARALLEL_HOST_RUNTIME, host_cc, library, pointers


@pytest.mark.skipif(os.name == "nt", reason="The four-core host fixture uses pthreads")
@pytest.mark.parametrize("fused", [False, True], ids=["separate", "fused"])
def test_overflow_does_not_publish_probabilities_and_workspace_can_be_reused(
        tmp_path, host_cc, fused):
    artifact = attention_chain_artifact(chain_program(), fused=fused, nonlinear_policy="fp32-reference")
    artifact = replace(artifact, source=artifact.source + _PARALLEL_HOST_RUNTIME)
    workspace = np.empty(artifact.workspace_bytes // 8, dtype=np.uint64)
    output = np.full(artifact.output.shape, -1.0, dtype=np.float32)
    q = np.zeros(INPUT_SHAPE, dtype=np.float32)
    k = np.zeros_like(q)
    inputs = pointers([q, k])
    with library(tmp_path, artifact, [*host_cc, "-DSCRATCHV_CORES=4", "-pthread"]) as execute:
        def call(capacity=workspace.nbytes):
            return execute(inputs, None, workspace.ctypes.data, capacity, output.ctypes.data)

        assert call(workspace.nbytes - 1) == 1
        limit = np.finfo(np.float32).max
        q[-1, -1, 0] = 1
        # Both scores are finite, but subtracting the largest score overflows.
        k[-1, 0, 0], k[-1, -1, 0] = limit, -limit
        assert call() == 2
        assert np.all(output == -1)
        # Overflow during MatMul has the same observable failure.
        q[-1, -1, 0] = 2
        assert call() == 2
        assert np.all(output == -1)
        q.fill(0)
        k.fill(0)
        assert call() == 0
        assert np.all(output == np.float32(1 / output.shape[-1]))


def test_probe_rejects_softmax_over_another_axis():
    program = chain_program()
    softmax = program.functions[0].blocks[0].instructions[-2]
    softmax.attrs["axis"] = 1
    with pytest.raises(TensorCCodegenError, match="axis=-1"):
        attention_chain_artifact(program, fused=True)
