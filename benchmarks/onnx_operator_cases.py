"""Small, reproducible ONNX fixtures for before/after frontend comparisons.

These are correctness/coverage probes, not representative Qwen throughput
workloads. Every model has one output, uses opset 18 / IR version 9, and is
paired with an NPZ containing only its runtime graph inputs. In particular,
Constant and initializer-only regressions intentionally have empty feeds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper


def build_cases(output_dir: Path) -> list[dict]:
    """Write the 23 cases and return their absolute paths and comparison policy."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    cases: list[dict] = []

    def floats(shape):
        return rng.normal(size=shape).astype(np.float32)

    def node(op, inputs, output="y", **attrs):
        return helper.make_node(op, inputs, [output], **attrs)

    def add(name, group, description, nodes, feed, output_shape, *,
            output_type=T.FLOAT, output_name="y", initializers=None,
            atol=1e-6, rtol=1e-6):
        initializers = initializers or {}
        graph = helper.make_graph(
            nodes, name,
            [helper.make_tensor_value_info(
                key, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape)
             for key, value in feed.items()],
            [helper.make_tensor_value_info(output_name, output_type, output_shape)],
            [numpy_helper.from_array(value, key)
             for key, value in initializers.items()],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = 9
        model.producer_name = "ScratchV ONNX operator comparison"
        onnx.checker.check_model(model, full_check=True)
        model_path = output_dir / f"{name}.onnx"
        inputs_path = output_dir / f"{name}.inputs.npz"
        onnx.save(model, model_path)
        np.savez(inputs_path, **feed)
        cases.append({
            "name": name,
            "group": group,
            "description": description,
            "operators": list(dict.fromkeys(item.op_type for item in nodes)),
            "model": str(model_path),
            "inputs": str(inputs_path),
            "atol": float(atol),
            "rtol": float(rtol),
        })

    # The controls bind every parameter through graph inputs: they do not
    # depend on fixes to initializer data binding in the current frontend.
    add("add_float32", "control",
        "FP32 x[2,4] + bias[4] -> [2,4]; both tensors are runtime inputs.",
        [node("Add", ["x", "bias"])],
        {"x": floats((2, 4)), "bias": floats((4,))}, (2, 4))
    add("matmul_32", "control",
        "FP32 x[32,32] @ w[32,32] -> [32,32]; both matrices are runtime inputs.",
        [node("MatMul", ["x", "w"])],
        {"x": floats((32, 32)), "w": floats((32, 32))}, (32, 32),
        atol=1e-5, rtol=1e-5)

    # The first eight ONNX operators were needed by PR #89's tiny model.
    add("gather", "operators",
        "FP32 table[8,4], INT64 indices[2,3] including -1; axis=0 -> [2,3,4].",
        [node("Gather", ["table", "indices"])],
        {"table": floats((8, 4)),
         "indices": np.array([[0, 3, -1], [5, 1, 6]], dtype=np.int64)}, (2, 3, 4))
    positive = np.abs(floats((2, 4))) + np.float32(0.01)
    add("sqrt", "operators", "Positive FP32 x[2,4] -> Sqrt [2,4].",
        [node("Sqrt", ["x"])], {"x": positive}, (2, 4))
    add("reduce_mean", "operators",
        "FP32 x[2,3,4], constant INT64 axes[1]=[-1], keepdims=1 -> [2,3,1].",
        [node("ReduceMean", ["x", "axes"], keepdims=1)],
        {"x": floats((2, 3, 4))}, (2, 3, 1),
        initializers={"axes": np.array([-1], dtype=np.int64)})
    add("transpose", "operators",
        "FP32 x[1,2,3,4], perm=[0,2,1,3] -> [1,3,2,4].",
        [node("Transpose", ["x"], perm=[0, 2, 1, 3])],
        {"x": floats((1, 2, 3, 4))}, (1, 3, 2, 4))
    add("concat", "operators",
        "FP32 left[2,1,4] and right[2,2,4], axis=-2 -> [2,3,4].",
        [node("Concat", ["left", "right"], axis=-2)],
        {"left": floats((2, 1, 4)), "right": floats((2, 2, 4))}, (2, 3, 4))
    add("slice", "operators",
        "FP32 x[2,8], INT64 vectors: start=-1,end=INT64_MIN,axis=-1,step=-2 -> [2,4].",
        [node("Slice", ["x", "starts", "ends", "axes", "steps"])],
        {"x": floats((2, 8))}, (2, 4), initializers={
            "starts": np.array([-1], dtype=np.int64),
            "ends": np.array([np.iinfo(np.int64).min], dtype=np.int64),
            "axes": np.array([-1], dtype=np.int64),
            "steps": np.array([-2], dtype=np.int64),
        })
    add("unsqueeze", "operators",
        "INT64 x[2,3], constant INT64 axes[2]=[0,-1] -> [1,2,3,1].",
        [node("Unsqueeze", ["x", "axes"])],
        {"x": rng.integers(-10, 10, (2, 3), dtype=np.int64)}, (1, 2, 3, 1),
        output_type=T.INT64, initializers={"axes": np.array([0, -1], dtype=np.int64)},
        atol=0, rtol=0)
    add("expand", "operators",
        "FP32 x[2,1,4], constant INT64 shape[3]=[1,3,4], bidirectional broadcast -> [2,3,4].",
        [node("Expand", ["x", "shape"])], {"x": floats((2, 1, 4))}, (2, 3, 4),
        initializers={"shape": np.array([1, 3, 4], dtype=np.int64)})

    # Eight additional ONNX operators from the full Qwen3 export. Constant
    # and Identity are frontend bindings/aliases, not additional IR opcodes.
    add("abs", "operators", "Signed INT64 x[2,3] -> Abs [2,3], preserving integer dtype.",
        [node("Abs", ["x"])],
        {"x": np.array([[-17, 0, 1], [123, -8, -3]], dtype=np.int64)}, (2, 3),
        output_type=T.INT64, atol=0, rtol=0)
    add("cast", "operators", "INT64 position IDs[1,8] -> FP32 [1,8].",
        [node("Cast", ["positions"], to=T.FLOAT)],
        {"positions": np.array([[0, 1, 2, 3, 7, 31, 127, 255]], dtype=np.int64)}, (1, 8))
    add("constant", "operators", "Constant tensor attribute FP32 [2,3]; no runtime inputs.",
        [node("Constant", [], value=numpy_helper.from_array(floats((2, 3))))], {}, (2, 3))
    add("cos", "operators", "FP32 angles[2,4] spanning positive/negative radians -> Cos [2,4].",
        [node("Cos", ["angles"])], {"angles": floats((2, 4)) * np.float32(3)}, (2, 4))
    add("identity", "operators", "INT64 indices[2,3] -> Identity [2,3]; frontend alias.",
        [node("Identity", ["indices"])],
        {"indices": rng.integers(-20, 20, (2, 3), dtype=np.int64)}, (2, 3),
        output_type=T.INT64, atol=0, rtol=0)
    add("pow", "operators", "Signed FP32 x[2,3] ** INT64 scalar exponent 2 -> FP32 [2,3].",
        [node("Pow", ["x", "exponent"])],
        {"x": floats((2, 3)), "exponent": np.array(2, dtype=np.int64)}, (2, 3))
    denominator = rng.uniform(0.25, 3.0, size=(2, 4)).astype(np.float32)
    denominator[:, ::2] *= -1
    add("reciprocal", "operators", "Nonzero signed FP32 x[2,4] -> Reciprocal [2,4].",
        [node("Reciprocal", ["x"])], {"x": denominator}, (2, 4))
    add("sin", "operators", "FP32 angles[2,4] spanning positive/negative radians -> Sin [2,4].",
        [node("Sin", ["angles"])], {"angles": floats((2, 4)) * np.float32(3)}, (2, 4))

    # Exact comparisons are essential here: rtol=1e-6 would hide the FP32
    # error (2) relative to a magnitude of 16777216.
    add("scalar_fp32_add_chain", "regressions",
        "FP32 scalar initializers: (16777216 + 1) + 1 must round after each Add; exact scalar output.",
        [node("Add", ["large", "one"], "partial"), node("Add", ["partial", "one"])],
        {}, (), initializers={"large": np.array(16777216, dtype=np.float32),
                              "one": np.array(1, dtype=np.float32)}, atol=0, rtol=0)
    add("singleton_initializer", "regressions",
        "FP32 initializer weight[1]=[7] returned directly: retain tensor data and shape [1], not scalar.",
        [], {}, (1,), output_name="weight",
        initializers={"weight": np.array([7], dtype=np.float32)}, atol=0, rtol=0)

    add("constant_cast_abs_expand", "composite",
        "Constant INT32 [-2,3] -> Cast INT64 -> Abs target[2]=[2,3]; Expand FP32 x[1,3] -> [2,3].",
        [node("Constant", [], "raw_shape",
              value=numpy_helper.from_array(np.array([-2, 3], dtype=np.int32))),
         node("Cast", ["raw_shape"], "shape_i64", to=T.INT64),
         node("Abs", ["shape_i64"], "shape"),
         node("Expand", ["x", "shape"])],
        {"x": floats((1, 3))}, (2, 3))
    add("qwen_rmsnorm", "composite",
        "FP32 x[1,4,8], gamma[8]; Pow(INT64 scalar 2)/ReduceMean(-1)/Add(eps=1e-6)/Sqrt/Reciprocal/Mul -> [1,4,8].",
        [node("Pow", ["x", "two"], "squared"),
         node("ReduceMean", ["squared", "axes"], "variance", keepdims=1),
         node("Add", ["variance", "epsilon"], "stabilized"),
         node("Sqrt", ["stabilized"], "denominator"),
         node("Reciprocal", ["denominator"], "scale"),
         node("Mul", ["x", "scale"], "normalized"),
         node("Mul", ["normalized", "gamma"])],
        {"x": floats((1, 4, 8)), "gamma": floats((8,))}, (1, 4, 8), initializers={
            "two": np.array(2, dtype=np.int64),
            "axes": np.array([-1], dtype=np.int64),
            "epsilon": np.array(1e-6, dtype=np.float32),
        })
    add("rope_numeric", "composite",
        "INT64 positions[1,4,1] -> Cast FP32; frequencies[1,1,4], Sin/Cos rotate FP32 left/right[1,4,4] -> [1,4,8].",
        [node("Cast", ["positions"], "positions_fp32", to=T.FLOAT),
         node("Mul", ["positions_fp32", "frequencies"], "angles"),
         node("Cos", ["angles"], "cosine"), node("Sin", ["angles"], "sine"),
         node("Mul", ["left", "cosine"], "left_cos"),
         node("Mul", ["right", "sine"], "right_sin"),
         node("Sub", ["left_cos", "right_sin"], "rotated_left"),
         node("Mul", ["right", "cosine"], "right_cos"),
         node("Mul", ["left", "sine"], "left_sin"),
         node("Add", ["right_cos", "left_sin"], "rotated_right"),
         node("Concat", ["rotated_left", "rotated_right"], axis=-1)],
        {"positions": np.array([0, 1, 7, 31], dtype=np.int64).reshape(1, 4, 1),
         "frequencies": np.array([1, 0.1, 0.01, 0.001], dtype=np.float32).reshape(1, 1, 4),
         "left": floats((1, 4, 4)), "right": floats((1, 4, 4))}, (1, 4, 8))

    print(f"Generated {len(cases)} ONNX operator comparison cases in {output_dir}")
    return cases
