"""Small, deterministic ONNX graphs with independent NumPy formula references.

These are semantic pattern tests, not fused RMSNorm/RoPE/SwiGLU/GQA operators.
The graph uses the same decomposed primitives accepted by the Qwen3 frontend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    model: onnx.ModelProto
    feed: dict[str, np.ndarray]
    expected: np.ndarray
    purpose: str


def graph(name, family, nodes, feed, expected, purpose, constants=None):
    expected = np.asarray(expected, dtype=np.float32)
    model = helper.make_model(helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info(key, T.FLOAT, value.shape) for key, value in feed.items()],
        [helper.make_tensor_value_info("y", T.FLOAT, expected.shape)],
        [numpy_helper.from_array(np.asarray(value), key) for key, value in (constants or {}).items()],
    ), opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model)
    return Case(name, family, model, feed, expected, purpose)


def rmsnorm_case(kind="hidden", epsilon=1e-6):
    shape = (1, 3, 16) if kind == "hidden" else ((1, 4, 3, 8) if kind == "q" else (1, 2, 3, 8))
    x = np.linspace(-0.003, 0.004, np.prod(shape), dtype=np.float32).reshape(shape)
    x[..., 0] = 0
    x[..., 0, :] = 0
    weight = np.linspace(0.4, 1.6, shape[-1], dtype=np.float32)
    # Accumulate independently in double precision, then round only the result.
    expected = (x.astype(np.float64) / np.sqrt(np.mean(x.astype(np.float64) ** 2,
                axis=-1, keepdims=True) + np.float32(epsilon))) * weight
    nodes = [helper.make_node("Pow", ["x", "two"], ["square"]),
             helper.make_node("ReduceMean", ["square", "last"], ["variance"], keepdims=1),
             helper.make_node("Add", ["variance", "epsilon"], ["stabilized"]),
             helper.make_node("Sqrt", ["stabilized"], ["denominator"]),
             helper.make_node("Div", ["x", "denominator"], ["normalized"]),
             helper.make_node("Mul", ["normalized", "weight"], ["y"])]
    return graph(f"rmsnorm_{kind}_eps_{epsilon:g}", "rmsnorm", nodes, {"x": x}, expected,
                 "Normalize the last dimension with learned weights; small inputs expose epsilon changes",
                 {"two": np.array(2, np.int64), "last": np.array([-1], np.int64),
                  "epsilon": np.array(epsilon, np.float32), "weight": weight})


def rope_case(positions=(0, 1, 255), heads=4):
    dim = 8
    x = np.linspace(-1.1, 1.2, heads * len(positions) * dim, dtype=np.float32).reshape(1, heads, -1, dim)
    position = np.array(positions, np.float32).reshape(1, -1, 1)
    inv = (1.0 / 1e6 ** (np.arange(0, dim, 2, dtype=np.float64) / dim)).astype(np.float32)
    # Direct complex rotation of each pair of halves, not graph Slice/Concat.
    angle = np.asarray(positions, np.float64)[:, None] * inv.astype(np.float64)[None, :]
    first, second = x[..., :dim // 2].astype(np.float64), x[..., dim // 2:].astype(np.float64)
    expected = np.empty_like(x)
    expected[..., :dim // 2] = first * np.cos(angle) - second * np.sin(angle)
    expected[..., dim // 2:] = second * np.cos(angle) + first * np.sin(angle)
    nodes = [helper.make_node("Mul", ["position", "inv_freq"], ["angle"]),
             helper.make_node("Concat", ["angle", "angle"], ["full_angle"], axis=-1),
             helper.make_node("Cos", ["full_angle"], ["cos_3d"]),
             helper.make_node("Sin", ["full_angle"], ["sin_3d"]),
             helper.make_node("Unsqueeze", ["cos_3d", "head_axis"], ["cos"]),
             helper.make_node("Unsqueeze", ["sin_3d", "head_axis"], ["sin"]),
             helper.make_node("Slice", ["x", "zero", "half", "last"], ["first"]),
             helper.make_node("Slice", ["x", "half", "end", "last"], ["second"]),
             helper.make_node("Neg", ["second"], ["negative_second"]),
             helper.make_node("Concat", ["negative_second", "first"], ["rotated"], axis=-1),
             helper.make_node("Mul", ["x", "cos"], ["real"]),
             helper.make_node("Mul", ["rotated", "sin"], ["imaginary"]),
             helper.make_node("Add", ["real", "imaginary"], ["y"])]
    return graph(f"rope_h{heads}_p{'_'.join(map(str, positions))}", "rope", nodes,
                 {"x": x, "position": position}, expected,
                 "Full head dimension rotate_half with broadcast positions, including nonzero offsets",
                 {"inv_freq": inv.reshape(1, 1, -1), "head_axis": np.array([1], np.int64),
                  "zero": np.array([0], np.int64), "half": np.array([4], np.int64),
                  "end": np.array([8], np.int64), "last": np.array([-1], np.int64)})


def swiglu_case(seed=11):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(1, 3, 4)).astype(np.float32)
    weights = {name: (rng.normal(size=shape) / 4).astype(np.float32) for name, shape in
               (("gate_weight", (4, 7)), ("up_weight", (4, 7)), ("down_weight", (7, 4)))}
    gate = x.astype(np.float64) @ weights["gate_weight"].astype(np.float64)
    up = x.astype(np.float64) @ weights["up_weight"].astype(np.float64)
    expected = ((gate / (1 + np.exp(-gate))) * up) @ weights["down_weight"].astype(np.float64)
    nodes = [helper.make_node("MatMul", ["x", "gate_weight"], ["gate"]),
             helper.make_node("MatMul", ["x", "up_weight"], ["up"]),
             helper.make_node("Sigmoid", ["gate"], ["activation"]),
             helper.make_node("Mul", ["gate", "activation"], ["silu"]),
             helper.make_node("Mul", ["silu", "up"], ["hidden"]),
             helper.make_node("MatMul", ["hidden", "down_weight"], ["y"])]
    return graph(f"swiglu_seed_{seed}", "swiglu", nodes, {"x": x}, expected,
                 "Different gate/up projections, SiLU on gate only, and down projection", weights)


def gqa_case(repeats=2):
    rng = np.random.default_rng(131)
    kv_heads, length, dim = 2, 3, 4
    q = (rng.normal(size=(1, kv_heads * repeats, length, dim)) / 2).astype(np.float32)
    k = (rng.normal(size=(1, kv_heads, length, dim)) / 2).astype(np.float32)
    v = (np.arange(kv_heads * length * dim).reshape(1, kv_heads, length, dim) / 13).astype(np.float32)
    mask = np.where(np.tril(np.ones((length, length), dtype=bool)),
                    0, np.finfo(np.float32).min).astype(np.float32)
    expected = np.empty_like(q)
    # Explicit head mapping detects tile-vs-repeat mistakes; masked softmax is
    # calculated independently in double precision for every query head.
    for head in range(q.shape[1]):
        kv = head // repeats
        scores = q[0, head].astype(np.float64) @ k[0, kv].astype(np.float64).T / np.sqrt(dim) + mask
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        expected[0, head] = probabilities @ v[0, kv].astype(np.float64)
    nodes = []
    for name in ("k", "v"):
        nodes.extend([helper.make_node("Unsqueeze", [name, "repeat_axis"], [f"{name}_5d"]),
                      helper.make_node("Expand", [f"{name}_5d", "expanded_shape"], [f"{name}_expanded"]),
                      helper.make_node("Reshape", [f"{name}_expanded", "repeated_shape"], [f"{name}_repeated"])])
    nodes.extend([helper.make_node("Transpose", ["k_repeated"], ["kt"], perm=[0, 1, 3, 2]),
                  helper.make_node("MatMul", ["q", "kt"], ["unscaled"]),
                  helper.make_node("Mul", ["unscaled", "scale"], ["scores"]),
                  helper.make_node("Add", ["scores", "mask"], ["masked"]),
                  helper.make_node("Softmax", ["masked"], ["probabilities"], axis=-1),
                  helper.make_node("MatMul", ["probabilities", "v_repeated"], ["y"])])
    return graph(f"gqa_repeat_{repeats}", "gqa", nodes, {"q": q, "k": k, "v": v}, expected,
                 "Contiguous KV head repetition, causal mask and attention context, including MHA ratio 1",
                 {"repeat_axis": np.array([2], np.int64), "scale": np.array(1 / np.sqrt(dim), np.float32),
                  "mask": mask, "expanded_shape": np.array([1, kv_heads, repeats, length, dim], np.int64),
                  "repeated_shape": np.array([1, kv_heads * repeats, length, dim], np.int64)})


def build_cases():
    rng = np.random.default_rng(314159)
    cases = []
    for name, left, right in (("matrix", (4, 4), (4, 4)),
                              ("batch_broadcast", (2, 1, 3, 5), (1, 4, 5, 2)),
                              ("vector_dot", (5,), (5,))):
        a, b = [(rng.normal(size=shape) / 3).astype(np.float32) for shape in (left, right)]
        cases.append(graph(f"matmul_{name}", "matmul", [helper.make_node("MatMul", ["a", "b"], ["y"])],
                           {"a": a, "b": b}, a.astype(np.float64) @ b.astype(np.float64),
                           "Matrix, batched broadcasting and scalar dot-product outputs"))
    x = np.linspace(-2, 2, 24, dtype=np.float32).reshape(2, 3, 4)
    bias = np.array([0.1, -0.3, 0.7, 1.1], np.float32)
    denominator = np.array(0.75, np.float32)
    cases.append(graph("elementwise_broadcast", "elementwise", [
        helper.make_node("Add", ["x", "bias"], ["sum"]),
        helper.make_node("Sub", ["sum", "denominator"], ["difference"]),
        helper.make_node("Mul", ["difference", "bias"], ["product"]),
        helper.make_node("Div", ["product", "denominator"], ["y"])],
        {"x": x, "bias": bias, "denominator": denominator},
        (x.astype(np.float64) + bias - denominator) * bias / denominator,
        "Scalar and trailing-dimension broadcast through Add/Sub/Mul/Div"))
    for axis in (-1, 0, 1):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1000
        expected = np.exp(data.astype(np.float64) - data.max(axis=axis, keepdims=True))
        expected /= expected.sum(axis=axis, keepdims=True)
        cases.append(graph(f"softmax_axis_{axis}", "softmax", [
            helper.make_node("Softmax", ["x"], ["y"], axis=axis)], {"x": data}, expected,
            "Numerically stable softmax with large logits and non-last dimensions"))
    cases.extend([rmsnorm_case("hidden"), rmsnorm_case("q"), rmsnorm_case("k", 1e-3),
                  rope_case(), rope_case((7, 19, 255), heads=2),
                  swiglu_case(), swiglu_case(23), gqa_case(), gqa_case(1)])
    return cases
