#!/usr/bin/env python3
"""Generate simple ONNX models for testing ScratchV."""

import onnx
import numpy as np
from onnx import helper, TensorProto


def make_add_model(path: str = "models/add.onnx"):
    """Create a simple Add model: C = A + B (element-wise, 4-element vectors)."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [4])

    node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph([node], "add_graph", [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    onnx.save(model, path)
    print(f"Created {path}")


def make_mul_model(path: str = "models/mul.onnx"):
    """Create a simple Mul model."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [4])

    node = helper.make_node("Mul", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph([node], "mul_graph", [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    onnx.save(model, path)
    print(f"Created {path}")


def make_relu_model(path: str = "models/relu.onnx"):
    """Create a ReLU model: Y = ReLU(X)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])

    node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "relu_graph", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    onnx.save(model, path)
    print(f"Created {path}")


def make_matmul_model(path: str = "models/matmul.onnx"):
    """Create a MatMul model: C = A * B (2x3 * 3x2)."""
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 2])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 2])

    node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph([node], "matmul_graph", [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    onnx.save(model, path)
    print(f"Created {path}")


def make_add_relu_model(path: str = "models/add_relu.onnx"):
    """Create a two-op model: Z = ReLU(X + Y)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [4])

    add_node = helper.make_node("Add", inputs=["X", "Y"], outputs=["sum"])
    relu_node = helper.make_node("Relu", inputs=["sum"], outputs=["Z"])
    graph = helper.make_graph([add_node, relu_node], "add_relu_graph", [X, Y], [Z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    onnx.save(model, path)
    print(f"Created {path}")


if __name__ == "__main__":
    make_add_model()
    make_mul_model()
    make_relu_model()
    make_matmul_model()
    make_add_relu_model()
