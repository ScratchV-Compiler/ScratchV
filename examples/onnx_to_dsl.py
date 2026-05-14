#!/usr/bin/env python3
"""
ONNX to DSL JSON Converter

Reads an ONNX file containing a single Conv2D node and converts it to the JSON format
expected by the frontend used in the RISC-V AI compiler project.
"""

import json
import argparse

try:
    import onnx
    from onnx import numpy_helper
    import numpy as np
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    np = None


def extract_conv2d_info(model_path: str):
    """
    Extract Conv2D operator information and all related constants from an ONNX model.
    """
    if not HAS_ONNX:
        raise ImportError(
            "The 'onnx' Python package is required. Install it with:\n"
            "  pip install onnx numpy"
        )

    # 1. Load the ONNX model
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    graph = model.graph

    conv_info = {
        "op": "conv2d",
        "input_shape": None,
        "kernel_shape": None,
        "stride": [1, 1],
        "padding": [0, 0, 0, 0],
        "data": {
            "input": None,
            "kernel": None,
        },
    }

    # 2. Parse all initializers (constants), map name -> numpy array
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    # 3. Find the first Conv node
    conv_node = None
    for node in graph.node:
        if node.op_type == "Conv":
            conv_node = node
            break

    if conv_node is None:
        raise ValueError("No Conv node found in the ONNX graph.")

    # 4. Extract parameters from Conv node attributes
    for attr in conv_node.attribute:
        if attr.name == "kernel_shape":
            conv_info["kernel_shape"] = list(attr.ints)
        elif attr.name == "strides":
            conv_info["stride"] = list(attr.ints)
        elif attr.name == "pads":
            conv_info["padding"] = list(attr.ints)

    # 5. Get input tensor shape
    input_name = conv_node.input[0]
    kernel_name = conv_node.input[1]

    for input_info in graph.input:
        if input_info.name == input_name:
            shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
            conv_info["input_shape"] = shape[-2:]  # take [H, W]
            break

    # 6. Extract kernel weights from initializers
    if kernel_name in initializers:
        kernel_data = initializers[kernel_name]
        conv_info["data"]["kernel"] = kernel_data.flatten().tolist()

    # 7. Extract input data if stored as initializer
    if input_name in initializers:
        input_data = initializers[input_name]
        conv_info["data"]["input"] = input_data.flatten().tolist()

    return conv_info


def save_to_json(conv_info: dict, output_path: str):
    """Save conv info as a JSON file compatible with conv_dsl.json."""
    if conv_info["data"]["input"] is None:
        h, w = conv_info["input_shape"]
        random_input = np.random.randint(0, 10, size=(h * w)).tolist()
        conv_info["data"]["input"] = random_input
        print(f"[INFO] No input data found, generated random input: {random_input}")

    if conv_info["data"]["kernel"] is None:
        raise ValueError(
            "Kernel data missing. Make sure the ONNX model exports kernel initializers."
        )

    with open(output_path, "w") as f:
        json.dump(conv_info, f, indent=2)

    print(f"[SUCCESS] JSON saved to: {output_path}")
    print(json.dumps(conv_info, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="ONNX to DSL JSON Converter for Conv2D"
    )
    parser.add_argument("input_onnx", help="Input ONNX file path")
    parser.add_argument("-o", "--output", default="conv_dsl.json",
                        help="Output JSON file path (default: conv_dsl.json)")
    args = parser.parse_args()

    conv_info = extract_conv2d_info(args.input_onnx)
    save_to_json(conv_info, args.output)


if __name__ == "__main__":
    main()
