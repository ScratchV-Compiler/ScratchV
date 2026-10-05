"""Real dimensions/weights, explicit synthetic activations, independent Torch oracles.

ONNX primitives are assembled directly. Their Torch oracle uses F.linear,
rsqrt, trigonometric pair rotation, and an explicit per-head attention loop;
it neither exports nor executes the same ONNX graph to define correctness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper
import torch
import torch.nn.functional as F

from .assets import DIMENSIONS


@dataclass
class Case:
    name: str
    family: str
    model: onnx.ModelProto
    feed: dict
    expected: dict
    metadata: dict


def node(op, inputs, output, **attrs):
    return helper.make_node(op, inputs, [output], name=output, **attrs)


def graph(name, family, nodes, feed, expected, constants=None, **metadata):
    model = helper.make_model(helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info(key, T.FLOAT, value.shape) for key, value in feed.items()],
        [helper.make_tensor_value_info(key, T.FLOAT, value.shape) for key, value in expected.items()],
        [numpy_helper.from_array(np.asarray(value), key) for key, value in (constants or {}).items()],
    ), opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model, full_check=True)
    return Case(name, family, model, feed, expected, metadata)


def arr(tensor):
    return tensor.detach().numpy().copy()


def rms_reference(x, weight):
    value = torch.from_numpy(x)
    return arr(value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)
               * torch.from_numpy(weight))


def rope_reference(x, position):
    # Pair formula is deliberately separate from ONNX Slice/Neg/Concat.
    value = torch.from_numpy(x)
    inv = 1.0 / (1000000 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
    angle = torch.from_numpy(position).reshape(-1, 1) * inv.reshape(1, -1)
    first, second = value[..., :64], value[..., 64:]
    return arr(torch.cat((first * angle.cos() - second * angle.sin(),
                          second * angle.cos() + first * angle.sin()), dim=-1))


def projection_case(kind, x, weight):
    expected = arr(F.linear(torch.from_numpy(x), torch.from_numpy(weight)))
    return graph(f"projection_{kind}", "projection", [node("MatMul", ["x", "weight"], "y")],
                 {"x": x}, {"y": expected}, {"weight": weight.T.copy()},
                 source_weights=[f"model.layers.0.self_attn.{kind}_proj.weight"],
                 purpose="FP32 projection using authentic first-layer checkpoint weight")


def norm_case(kind, x, weight, suffix):
    return graph(f"rmsnorm_{kind}", "rmsnorm", [
        node("Mul", ["x", "x"], "square"),
        node("ReduceMean", ["square", "last"], "variance", keepdims=1),
        node("Add", ["variance", "epsilon"], "stabilized"),
        node("Sqrt", ["stabilized"], "denominator"),
        node("Div", ["x", "denominator"], "normalized"),
        node("Mul", ["normalized", "weight"], "y"),
    ], {"x": x}, {"y": rms_reference(x, weight)},
        {"last": np.array([-1], np.int64), "epsilon": np.array(1e-6, np.float32), "weight": weight},
        source_weights=["model.layers.0." + suffix],
        purpose="Learned RMSNorm across the final dimension; official epsilon=1e-6")


def rope_case(kind, x):
    length = x.shape[-2]
    positions = np.arange(length, dtype=np.float32).reshape(1, length, 1)
    inv = (1.0 / np.power(np.float32(1000000), np.arange(0, 128, 2, dtype=np.float32) / 128))
    return graph(f"rope_{kind}", "rope", [
        node("Mul", ["position", "inv_freq"], "angle"),
        node("Concat", ["angle", "angle"], "full_angle", axis=-1),
        node("Cos", ["full_angle"], "cos_3d"),
        node("Sin", ["full_angle"], "sin_3d"),
        node("Unsqueeze", ["cos_3d", "head_axis"], "cos"),
        node("Unsqueeze", ["sin_3d", "head_axis"], "sin"),
        node("Slice", ["x", "zero", "half", "last"], "first"),
        node("Slice", ["x", "half", "end", "last"], "second"),
        node("Neg", ["second"], "negative_second"),
        node("Concat", ["negative_second", "first"], "rotated", axis=-1),
        node("Mul", ["x", "cos"], "real"),
        node("Mul", ["rotated", "sin"], "imaginary"),
        node("Add", ["real", "imaginary"], "y"),
    ], {"x": x, "position": positions}, {"y": rope_reference(x, positions)},
        {"inv_freq": inv.reshape(1, 1, 64), "head_axis": np.array([1], np.int64),
         "zero": np.array([0], np.int64), "half": np.array([64], np.int64),
         "end": np.array([128], np.int64), "last": np.array([-1], np.int64)},
        source_weights=[f"model.layers.0.self_attn.{kind}_proj.weight",
                        f"model.layers.0.self_attn.{kind}_norm.weight"],
        purpose="Full head_dim=128 rotate-half, every position 0..L-1, theta=1e6",
        activation_origin="Synthetic hidden -> authentic projection and Q/K RMSNorm, computed in Torch")


def swiglu_case(x, gate, up, down):
    tx = torch.from_numpy(x)
    gt = F.linear(tx, torch.from_numpy(gate))
    ut = F.linear(tx, torch.from_numpy(up))
    hidden = F.silu(gt) * ut
    expected = {"gate": arr(gt), "up": arr(ut), "hidden": arr(hidden),
                "y": arr(F.linear(hidden, torch.from_numpy(down)))}
    return graph("swiglu", "swiglu", [
        node("MatMul", ["x", "gate_weight"], "gate"),
        node("MatMul", ["x", "up_weight"], "up"),
        node("Sigmoid", ["gate"], "activation"),
        node("Mul", ["gate", "activation"], "silu"),
        node("Mul", ["silu", "up"], "hidden"),
        node("MatMul", ["hidden", "down_weight"], "y"),
    ], {"x": x}, expected,
        {"gate_weight": gate.T.copy(), "up_weight": up.T.copy(), "down_weight": down.T.copy()},
        source_weights=["model.layers.0.mlp." + name + "_proj.weight" for name in ("gate", "up", "down")],
        purpose="Authentic 1024 -> 3072 -> 1024 SwiGLU, distinct gate/up learned weights")


def attention_reference(feed):
    q, k, v, mask = (torch.from_numpy(feed[name]) for name in ("q", "k", "v", "mask"))
    probabilities, contexts = [], []
    # Explicit mapping, unlike ONNX Expand/Reshape: heads (0,1) use KV0, etc.
    for head in range(16):
        scores = torch.matmul(q[:, head], k[:, head // 2].transpose(-1, -2)) * (128 ** -0.5)
        weights = torch.softmax(scores + mask[:, 0], dim=-1)
        probabilities.append(weights)
        contexts.append(torch.matmul(weights, v[:, head // 2]))
    return {"probabilities": arr(torch.stack(probabilities, dim=1)),
            "y": arr(torch.stack(contexts, dim=1))}


def attention_case(name, feed, valid_length, **metadata):
    length = feed["q"].shape[2]
    nodes = []
    for value in ("k", "v"):
        nodes += [node("Unsqueeze", [value, "repeat_axis"], value + "_5d"),
                  node("Expand", [value + "_5d", "expanded_shape"], value + "_expanded"),
                  node("Reshape", [value + "_expanded", "repeated_shape"], value + "_repeated")]
    nodes += [node("Transpose", ["k_repeated"], "kt", perm=[0, 1, 3, 2]),
              node("MatMul", ["q", "kt"], "unscaled"),
              node("Mul", ["unscaled", "scale"], "scores"),
              node("Add", ["scores", "mask"], "masked"),
              node("Softmax", ["masked"], "probabilities", axis=-1),
              node("MatMul", ["probabilities", "v_repeated"], "y")]
    return graph(name, "gqa", nodes, feed, attention_reference(feed),
        {"repeat_axis": np.array([2], np.int64), "scale": np.array(128 ** -0.5, np.float32),
         "expanded_shape": np.array([1, 8, 2, length, 128], np.int64),
         "repeated_shape": np.array([1, 16, length, 128], np.int64)},
        valid_length=valid_length, source_weights=["model.layers.0.self_attn." + suffix + ".weight"
                for suffix in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm")],
        purpose="Real 16Q/8KV, 128-dimensional heads; causal and key-padding masks",
        activation_origin="Synthetic hidden -> authentic Q/K/V projections, Q/K norm and RoPE, in Torch",
        **metadata)


def causal_mask(length, valid_length):
    allowed = np.arange(length)[None, :] <= np.arange(length)[:, None]
    allowed &= np.arange(length)[None, :] < valid_length
    return np.where(allowed, np.float32(0), np.finfo(np.float32).min).reshape(1, 1, length, length)


def iter_cases(weights, length=256):
    if not 2 <= length <= 256:
        raise ValueError("Sequence length must be in [2, 256]; official gate requires 256")
    rng = np.random.default_rng(20261004)
    x = rng.standard_normal((1, length, 1024), dtype=np.float32)
    attention_inputs = {}
    for kind, width in (("q", 2048), ("k", 1024), ("v", 1024)):
        weight = weights.get(f"self_attn.{kind}_proj.weight", (width, 1024))
        case = projection_case(kind, x, weight)
        attention_inputs[kind] = case.expected["y"].reshape(1, length, width // 128, 128).transpose(0, 2, 1, 3).copy()
        yield case
        del weight, case
    context = rng.standard_normal((1, length, 2048), dtype=np.float32)
    yield projection_case("o", context, weights.get("self_attn.o_proj.weight", (1024, 2048)))
    yield norm_case("hidden", x, weights.get("input_layernorm.weight", (1024,)), "input_layernorm.weight")
    for kind in ("q", "k"):
        suffix = f"self_attn.{kind}_norm.weight"
        weight = weights.get(suffix, (128,))
        case = norm_case(kind, attention_inputs[kind], weight, suffix)
        attention_inputs[kind] = case.expected["y"]
        yield case
        case = rope_case(kind, attention_inputs[kind])
        attention_inputs[kind] = case.expected["y"]
        yield case
        del weight, case
    full = {**attention_inputs, "mask": causal_mask(length, length)}
    yield attention_case("gqa_full", full, length)
    valid = min(17, max(1, length // 2))
    padded = {**attention_inputs, "mask": causal_mask(length, valid)}
    yield attention_case("gqa_padding", padded, valid)
    changed_padding = {name: array.copy() for name, array in padded.items()}
    for name in ("k", "v"):
        changed_padding[name][:, :, valid:] += np.float32(3)
    yield attention_case("gqa_changed_padding", changed_padding, valid, changed_from=valid)
    changed_future = {name: array.copy() for name, array in full.items()}
    future = max(1, length // 4)
    for name in ("q", "k", "v"):
        changed_future[name][:, :, future:] += np.float32(2)
    yield attention_case("gqa_changed_future", changed_future, length, changed_from=future)
    del attention_inputs, full, padded, changed_padding, changed_future
    gate = weights.get("mlp.gate_proj.weight", (3072, 1024))
    up = weights.get("mlp.up_proj.weight", (3072, 1024))
    down = weights.get("mlp.down_proj.weight", (1024, 3072))
    yield swiglu_case(x, gate, up, down)
