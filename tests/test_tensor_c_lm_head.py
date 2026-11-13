"""Exercise selection and execution of the terminal shared-weight projection."""
import numpy as np
import pytest

from scratchv.backend.tensor_c_codegen import TensorCCodegen, TensorCCodegenError
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, Value
from tests.test_tensor_c_external import host_cc, library, pointers


def model(*, tied=True, terminal=True, hidden=129):
    weight = Value("word_embeddings", shape=(37, hidden))
    inner = Value("layer_weight", shape=(hidden, hidden))
    x = Value("x", shape=(1, 3, hidden))
    token = Value("token", dtype=DataType.INT64, shape=(1,))
    builder = IRBuilder()
    builder.new_function("model", [x, token])
    builder.new_block("entry")
    builder.program.global_values.extend((weight, inner))
    middle = builder.matmul(x, inner)
    if tied:
        middle = builder.add(middle, builder.gather(weight, token, axis=0))
    result = builder.matmul(middle, builder.transpose(weight, (1, 0)))
    builder.ret(result if terminal else builder.relu(result))
    rng = np.random.default_rng(903)
    inputs = [rng.standard_normal(x.shape).astype(np.float32) / 16, np.array([3], dtype=np.int64)]
    weights = {v.name: rng.standard_normal(v.shape).astype(np.float32) / 16 for v in (weight, inner)}
    return builder.program, weights, inputs


def generate(program, weights, **options):
    return TensorCCodegen(program, weights, constant_storage="external", kernel_calls=True,
                          **({"matmul_tile": "2x4", "matmul_policy": "blocked_fma",
                              "matmul_rhs_layout": "column-panel"} | options)).generate()


def test_vocab_head_preserves_shared_weight_other_panels_and_all_logits(tmp_path, host_cc):
    program, weights, inputs = model()
    baseline = generate(program, weights)
    assert baseline.lm_head_policy == "original"
    assert baseline.source == generate(program, weights, lm_head_policy="original").source
    candidate = generate(program, weights, lm_head_policy="vocab-blocked")
    assert candidate.lm_head_policy == "vocab-blocked"
    assert [row["source_weight"] for row in candidate.matmul_rhs_panels] == ["layer_weight"]
    assert candidate.matmul_rhs_panels == baseline.matmul_rhs_panels
    assert candidate.external_weights == baseline.external_weights
    assert candidate.workspace_bytes < baseline.workspace_bytes
    expected = (inputs[0] @ weights["layer_weight"] + weights["word_embeddings"][inputs[1]]) @ weights["word_embeddings"].T
    outputs = []
    for artifact in (baseline, candidate):
        stored = [np.ascontiguousarray(a) for a in artifact.external_initializers]
        index = next(i for i,w in enumerate(artifact.external_weights) if w.name == "word_embeddings")
        assert np.shares_memory(artifact.external_initializers[index], weights["word_embeddings"])
        workspace = np.empty(artifact.workspace_bytes, np.uint8)
        output = np.empty(artifact.output.shape, np.float32)
        args = (pointers(inputs), pointers(stored), workspace.ctypes.data, workspace.nbytes, output.ctypes.data)
        with library(tmp_path / artifact.lm_head_policy, artifact, host_cc) as run:
            assert run(*args) == 0
            np.testing.assert_allclose(output, expected, rtol=0, atol=1e-5)
            outputs.append(output.copy())
            output.fill(-99)
            inputs[1][0] = 37
            assert run(*args) == 3
            inputs[1][0] = 3
            # An unselected embedding row still participates in the projection.
            old = stored[index][35, 0]
            stored[index][35, 0] = np.nan
            assert run(*args) == 2
            stored[index][35, 0] = old
            assert np.all(output == -99)
    np.testing.assert_array_equal(outputs[0].view(np.uint32), outputs[1].view(np.uint32))


@pytest.mark.parametrize("graph_options,codegen_options", [
    ({"tied": False}, {}),
    ({"terminal": False}, {}),
    ({"hidden": 1025}, {}),
    ({}, {"matmul_policy": "sequential", "matmul_tile": "1x4", "matmul_rhs_layout": "original"}),
    ({}, {"lm_head_policy": "typo"}),
])
def test_vocab_head_rejects_unmatched_graph_or_policy(graph_options, codegen_options):
    program, weights, _ = model(**graph_options)
    with pytest.raises(TensorCCodegenError):
        generate(program, weights, **({"lm_head_policy": "vocab-blocked"} | codegen_options))
