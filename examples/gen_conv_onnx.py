#!/usr/bin/env python3
"""Generate a simple Conv2D ONNX model for testing onnx_to_dsl.py."""

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np


def make_conv_model(path: str = "models/conv_5x5_3x3.onnx"):
    """Create a Conv2D model: input 5x5, kernel 3x3, output 3x3."""
    # Tensor info
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 1, 3, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 3, 3])

    # Create kernel data (Sobel-like edge detection)
    kernel_data = np.array([1, 0, -1, 1, 0, -1, 1, 0, -1], dtype=np.float32).reshape(1, 1, 3, 3)
    W_init = numpy_helper.from_array(kernel_data, name="W")

    # Create input data
    input_data = np.arange(1, 26, dtype=np.float32).reshape(1, 1, 5, 5)
    X_init = numpy_helper.from_array(input_data, name="X")

    # Conv node
    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "W"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )

    graph = helper.make_graph(
        [conv_node],
        "conv_graph",
        [X, W],
        [Y],
        initializer=[W_init, X_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"Created {path}")
    print(f"  Input shape:  {input_data.shape}")
    print(f"  Kernel shape: {kernel_data.shape}")


if __name__ == "__main__":
    make_conv_model()
