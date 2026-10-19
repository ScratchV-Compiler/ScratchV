"""The seven original W1 MatMul shapes, executed on real RV64 QEMU."""
import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

SHAPES = (
    ("4x4_fractional", (4, 4), (4, 4)),
    ("projection", (1, 7, 32), (32, 64)),
    ("attention", (1, 4, 7, 16), (1, 4, 16, 7)),
    ("batch_broadcast", (2, 1, 3, 5), (1, 4, 5, 2)),
    ("vector_dot", (5,), (5,)),
    ("matrix_vector", (3, 5), (5,)),
    ("vector_matrix", (5,), (5, 4)),
)


@pytest.fixture(scope="module")
def matmul_inputs() -> dict:
    rng = np.random.default_rng(314159)
    return {
        name: {key: rng.normal(size=shape).astype(np.float32) / np.float32(3)
               for key, shape in (("a", left), ("b", right))}
        for name, left, right in SHAPES
    }


@pytest.mark.parametrize("name", [row[0] for row in SHAPES])
def test_matmul_on_riscv(name, matmul_inputs, tmp_path, ort_session, run_riscv, assert_tensor):
    feed = matmul_inputs[name]
    expected = np.asarray(np.matmul(feed["a"], feed["b"]))
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["a", "b"], ["y"])], name,
        [helper.make_tensor_value_info(key, TensorProto.FLOAT, value.shape)
         for key, value in feed.items()],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, expected.shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    reference = ort_session(path).run(None, feed)[0]
    assert_tensor(reference, expected)
    assert_tensor(run_riscv(path, feed).output, reference)
