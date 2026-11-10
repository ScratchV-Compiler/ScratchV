"""Create the standalone W6 LM Head input from the pinned Qwen3 ONNX release."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

from probes.w1_qwen3_export.run import acquire_model
from probes.w6_performance.lm_head_case import HIDDEN, ROWS, VOCAB
from scratchv.frontend.onnx_parser import ONNXParser


def prepare(model_dir, output):
    """Use the same input/weight bytes as the retained seed-734 benchmark."""
    if output.exists():
        raise FileExistsError(output)
    acquisition = acquire_model(model_dir)
    parser = ONNXParser()
    parser.parse(str(model_dir / "model.onnx"), mmap_external_data=True)
    name = "model.lm_head.weight"
    weight = parser.initializers[name]
    if weight.shape != (VOCAB, HIDDEN) or weight.dtype != np.dtype("<f4"):
        raise ValueError(f"Unexpected {name} shape or dtype")
    output.mkdir(parents=True)
    weight.tofile(output / "weight.bin")
    del parser, weight

    # This standalone model stores only a descriptor for the shared real weight.
    # The W6 guest graph also exercises the tied embedding Gather before the head.
    weight = TensorProto(name="weight", data_type=TensorProto.FLOAT,
                         dims=[VOCAB, HIDDEN], data_location=TensorProto.EXTERNAL)
    weight.external_data.add(key="location", value="weight.bin")
    weight.external_data.add(key="length", value=str(4 * VOCAB * HIDDEN))
    shape = [1, ROWS, HIDDEN]
    logits_shape = [1, ROWS, VOCAB]
    model = helper.make_model(helper.make_graph([
        helper.make_node("Transpose", ["weight"], ["rhs"], perm=[1, 0]),
        helper.make_node("MatMul", ["x", "rhs"], ["y"]),
    ], "w6_lm_head", [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, logits_shape)], [weight]),
        opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.save(model, output / "reference.onnx")
    x = np.random.default_rng(734).standard_normal(shape).astype(np.float32)
    x /= np.float32(np.sqrt(HIDDEN))
    np.save(output / "input_x.npy", x, allow_pickle=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output / "reference.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    logits = session.run(["y"], {"x": x})[0]
    if logits.shape != tuple(logits_shape) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError("Invalid full-vocabulary ORT reference")
    np.save(output / "ort.npy", logits, allow_pickle=False)
    metadata = {"initializer_name": name, "initializer_shape": [VOCAB, HIDDEN],
                "left_shape": shape, "output_shape": logits_shape,
                "input_seed": 734, "reference": "ONNX Runtime CPU", "model": acquisition}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model-dir", type=Path, required=True,
                     help="Verified local model directory, or a new directory for the pinned release")
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()
    prepare(args.model_dir, args.output_dir)
