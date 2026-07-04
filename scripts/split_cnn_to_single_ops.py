#!/usr/bin/env python3
"""Split cnn.onnx into single-operator models for per-operator benchmarking.

Extracts each Conv/Relu/MaxPool/Gemm/Sigmoid operator as a standalone ONNX model,
preserving weights, attributes, and input/output shape information.

Output: models/single_op/<op_type>/cnn_<op_name>.onnx
"""

from __future__ import annotations
import os, sys
from pathlib import Path
from collections import OrderedDict

import onnx
from onnx import helper, numpy_helper, TensorProto, NodeProto


PROJ = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJ / "models" / "graph" / "cnn.onnx"
OUTPUT_BASE = PROJ / "models" / "single_op"

# Map ONNX op_type to output directory name
OP_DIR_MAP = {
    "Conv": "conv",
    "Relu": "relu",
    "MaxPool": "maxpool",
    "Gemm": "gemm",
    "Sigmoid": "sigmoid",
    "Reshape": "reshape",
}


def _get_tensor_shape(model: onnx.ModelProto, name: str):
    """Look up the shape of a tensor by name from inputs, outputs, or value_info."""
    # Check graph inputs
    for inp in model.graph.input:
        if inp.name == name:
            dims = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
            return dims, inp.type.tensor_type.elem_type
    # Check graph outputs
    for out in model.graph.output:
        if out.name == name:
            dims = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
            return dims, out.type.tensor_type.elem_type
    # Check value_info (intermediate tensors)
    for vi in model.graph.value_info:
        if vi.name == name:
            dims = [d.dim_value if d.dim_value else d.dim_param for d in vi.type.tensor_type.shape.dim]
            return dims, vi.type.tensor_type.elem_type
    # Check initializers (weights)
    for init in model.graph.initializer:
        if init.name == name:
            return list(init.dims), init.data_type
    return None, None


def _short_name(node_name: str) -> str:
    """Convert /layer1.0/Conv to layer1.0_Conv."""
    return node_name.strip("/").replace("/", "_")


def _make_value_info(name: str, shape: list, elem_type: int = TensorProto.FLOAT):
    """Create a ValueInfoProto with given name, shape, and element type."""
    dims = []
    for d in shape:
        if isinstance(d, int) and d > 0:
            dims.append(helper.make_tensor_value_info("", elem_type, []).type.tensor_type.shape.dim.add())
            dims[-1].dim_value = d
        elif isinstance(d, str):
            dims.append(helper.make_tensor_value_info("", elem_type, []).type.tensor_type.shape.dim.add())
            dims[-1].dim_param = d
        else:
            dims.append(helper.make_tensor_value_info("", elem_type, []).type.tensor_type.shape.dim.add())
            dims[-1].dim_value = 1
    return helper.make_tensor_value_info(name, elem_type, shape)


def _clone_node_with_attrs(node: NodeProto) -> NodeProto:
    """Deep-clone a node preserving all attributes."""
    attrs = []
    for attr in node.attribute:
        attrs.append(attr.SerializeToString())
    new_node = NodeProto()
    new_node.ParseFromString(node.SerializeToString())
    return new_node


def extract_node(
    model: onnx.ModelProto,
    node: NodeProto,
) -> onnx.ModelProto:
    """Extract a single node as a standalone ONNX model.

    The new model takes the node's non-initializer inputs as graph inputs
    and produces the node's outputs as graph outputs. Initializer inputs
    (weights/biases) are embedded as model initializers.
    """
    # Identify which inputs are initializers (weights) vs data inputs
    init_names = {i.name for i in model.graph.initializer}
    graph_input_names = {i.name for i in model.graph.input}

    data_inputs = []
    weight_inits = []
    for inp_name in node.input:
        if not inp_name:
            continue
        if inp_name in init_names:
            # Find the initializer
            for init in model.graph.initializer:
                if init.name == inp_name:
                    weight_inits.append(init)
                    break
        else:
            data_inputs.append(inp_name)

    # Build input value_infos
    new_inputs = []
    for inp_name in data_inputs:
        shape, elem_type = _get_tensor_shape(model, inp_name)
        if shape is None:
            print(f"  WARNING: cannot determine shape for input '{inp_name}', using [1]")
            shape = [1]
        if elem_type is None:
            elem_type = TensorProto.FLOAT
        new_inputs.append(helper.make_tensor_value_info(inp_name, elem_type, shape))

    # Build output value_infos
    new_outputs = []
    for out_name in node.output:
        shape, elem_type = _get_tensor_shape(model, out_name)
        if shape is None:
            shape = [1]
        if elem_type is None:
            elem_type = TensorProto.FLOAT
        new_outputs.append(helper.make_tensor_value_info(out_name, elem_type, shape))

    # Deep-clone the node (to preserve attributes exactly)
    new_node = helper.make_node(
        op_type=node.op_type,
        inputs=list(node.input),
        outputs=list(node.output),
        name=node.name,
        **{attr.name: helper.get_attribute_value(attr) for attr in node.attribute},
    )

    # Build graph and model (use same opset as source)
    graph = helper.make_graph(
        nodes=[new_node],
        name=f"{_short_name(node.name)}_graph",
        inputs=new_inputs,
        outputs=new_outputs,
        initializer=weight_inits,
    )

    new_model = helper.make_model(
        graph,
        opset_imports=model.opset_import,
        producer_name="ScratchV split_cnn_to_single_ops",
    )

    # Validate
    try:
        onnx.checker.check_model(new_model)
    except onnx.checker.ValidationError as e:
        print(f"  WARNING: validation error for {node.name}: {e}")

    return new_model


def main():
    print(f"Loading: {MODEL_PATH}")
    model = onnx.load(str(MODEL_PATH))
    network = "cnn"

    # Collect input/output/initializer sets for debugging
    graph_input_names = {i.name for i in model.graph.input}
    init_names = {i.name for i in model.graph.initializer}

    count = 0
    report = OrderedDict()  # op_type -> [model_names]

    for node in model.graph.node:
        op_dir_name = OP_DIR_MAP.get(node.op_type)
        if op_dir_name is None:
            print(f"Skip: {node.name} ({node.op_type}) — unknown op type, skipping")
            continue

        short = _short_name(node.name)
        model_name = f"{network}_{short}"
        out_dir = OUTPUT_BASE / op_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_name}.onnx"

        print(f"Extracting: {node.name} ({node.op_type}) → {out_path}")
        try:
            single_op = extract_node(model, node)
            onnx.save(single_op, str(out_path))
            print(f"  OK: {out_path} ({os.path.getsize(out_path):,} bytes)")

            if op_dir_name not in report:
                report[op_dir_name] = []
            report[op_dir_name].append(model_name)
            count += 1
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Extracted {count} single-op models:")
    for op_type, names in report.items():
        print(f"  {op_type}/ ({len(names)}): {', '.join(names)}")
    print(f"Output base: {OUTPUT_BASE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
