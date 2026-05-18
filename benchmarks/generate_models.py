# flake8: noqa
"""Benchmark model generation — uses ScratchV's currently supported ONNX ops.

Supported ops: Add, Mul, Sub, Div, Relu, MatMul, MaxPool, GeLU, Softmax, Neg, Exp
"""

import os

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

BENCH_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BENCH_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def _make_model(nodes, inputs, outputs, initializers=None, value_info=None,
                graph_name="graph"):
    graph = helper.make_graph(
        nodes, graph_name, inputs, outputs,
        initializer=initializers or [], value_info=value_info or []
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    onnx.checker.check_model(model)
    return model


def make_add_model(path: str | None = None) -> str:
    """Element-wise Add: A + B → C, shapes [1, 128, 64]."""
    if path is None:
        path = os.path.join(MODEL_DIR, "add.onnx")
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 128, 64])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 128, 64])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 128, 64])
    model = _make_model([helper.make_node("Add", ["A", "B"], ["C"])], [A, B], [C])
    onnx.save(model, path)
    return path


def make_mixed_model(path: str | None = None) -> str:
    """Mixed ops: Add → Mul → Relu → Sub → Div.

    Input: [1, 256, 256] × 3,  Output: [1, 256, 256]
    """
    if path is None:
        path = os.path.join(MODEL_DIR, "mixed_ops.onnx")
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 256, 256])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 256, 256])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 256, 256])
    O = helper.make_tensor_value_info("O", TensorProto.FLOAT, [1, 256, 256])

    vi = [
        helper.make_tensor_value_info("add_out", TensorProto.FLOAT, [1, 256, 256]),
        helper.make_tensor_value_info("mul_out", TensorProto.FLOAT, [1, 256, 256]),
        helper.make_tensor_value_info("relu_out", TensorProto.FLOAT, [1, 256, 256]),
        helper.make_tensor_value_info("sub_out", TensorProto.FLOAT, [1, 256, 256]),
    ]
    nodes = [
        helper.make_node("Add", ["X", "Y"], ["add_out"]),
        helper.make_node("Mul", ["add_out", "Z"], ["mul_out"]),
        helper.make_node("Relu", ["mul_out"], ["relu_out"]),
        helper.make_node("Sub", ["relu_out", "X"], ["sub_out"]),
        helper.make_node("Div", ["sub_out", "Y"], ["O"]),
    ]
    model = _make_model(nodes, [X, Y, Z], [O], value_info=vi)
    onnx.save(model, path)
    return path


def make_deep_relu_chain(path: str | None = None, length: int = 50) -> str:
    """Long chain: Relu → Relu → ... → Relu (50×), stress-test deep graphs.

    Input: [1, 1024], Output: [1, 1024]
    """
    if path is None:
        path = os.path.join(MODEL_DIR, "deep_relu.onnx")
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1024])
    O = helper.make_tensor_value_info("O", TensorProto.FLOAT, [1, 1024])

    nodes = []
    vi = []
    prev = "X"
    for i in range(length):
        out = f"r{i}" if i < length - 1 else "O"
        nodes.append(helper.make_node("Relu", [prev], [out]))
        if i < length - 1:
            vi.append(helper.make_tensor_value_info(out, TensorProto.FLOAT, [1, 1024]))
        prev = out

    model = _make_model(nodes, [X], [O], value_info=vi, graph_name="deep_relu")
    onnx.save(model, path)
    return path


def make_matmul_model(path: str | None = None) -> str:
    """MatMul: A @ B → C, shapes [4, 128] × [128, 64] → [4, 64]."""
    if path is None:
        path = os.path.join(MODEL_DIR, "matmul.onnx")
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4, 128])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [128, 64])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [4, 64])
    model = _make_model([helper.make_node("MatMul", ["A", "B"], ["C"])], [A, B], [C])
    onnx.save(model, path)
    return path


def make_maxpool_relu_model(path: str | None = None) -> str:
    """MaxPool → Relu: input [1, 8, 32, 32] → MaxPool(2x2, stride 2) → Relu.

    Output: [1, 8, 16, 16]
    """
    if path is None:
        path = os.path.join(MODEL_DIR, "maxpool_relu.onnx")
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8, 32, 32])
    P = helper.make_tensor_value_info("P", TensorProto.FLOAT, [1, 8, 16, 16])
    O = helper.make_tensor_value_info("O", TensorProto.FLOAT, [1, 8, 16, 16])
    model = _make_model([
        helper.make_node("MaxPool", ["X"], ["pool_out"],
                         kernel_shape=[2, 2], strides=[2, 2]),
        helper.make_node("Relu", ["pool_out"], ["O"]),
    ], [X], [O], value_info=[P])
    onnx.save(model, path)
    return path


def ensure_all_models() -> dict[str, str]:
    """Generate all benchmark ONNX models. Returns {model_name: path}."""
    models: dict[str, str] = {}
    gens = [
        ("add", make_add_model),
        ("mixed_ops", make_mixed_model),
        ("deep_relu", make_deep_relu_chain),
        ("matmul", make_matmul_model),
        ("maxpool_relu", make_maxpool_relu_model),
    ]
    for name, gen_func in gens:
        path = gen_func()
        models[name] = path
    return models


if __name__ == "__main__":
    models = ensure_all_models()
    for name, path in models.items():
        size_kb = os.path.getsize(path) / 1024
        print(f"  {name}: {path} ({size_kb:.1f} KB)")
