"""Combined small Qwen-style Attention, with an independent NumPy reference."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from onnx import helper, numpy_helper, TensorProto

@dataclass
class Case:
    name: str
    model: object
    feed: dict
    expected: np.ndarray
    valid_length: int
    relation: tuple | None = None

def attention_case(name, length, valid, *, change=None, relation=None):
    qh, kvh, dim = 4, 2, 8
    rng = np.random.default_rng(137 + length)
    feed = {
        "q": rng.normal(0, .4, (1, qh, length, dim)).astype(np.float32),
        "k": rng.normal(0, .4, (1, kvh, length, dim)).astype(np.float32),
        "v": rng.normal(0, .3, (1, kvh, length, dim)).astype(np.float32),
    }
    if change:
        for key in (("q", "k", "v") if change == "future" else ("k", "v")):
            feed[key][:, :, 5:] += np.float32(.8)
    weights = {"qw": np.linspace(.7, 1.2, dim, dtype=np.float32),
               "kw": np.linspace(1.1, .8, dim, dtype=np.float32)}
    angle = np.arange(length, dtype=np.float32)[:, None] * (
        1.0 / (1e6 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    )[None, :]
    angle = np.concatenate([angle, angle], axis=-1)[None, None]
    cos, sin = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
    allowed = np.tril(np.ones((length, length), dtype=bool))
    allowed[:, valid:] = False
    mask = np.where(allowed, 0, np.finfo(np.float32).min).astype(np.float32)[None, None]
    constants = {**weights, "epsilon": np.array(1e-6, np.float32),
                 "axes": np.array([-1], np.int64), "scale": np.array(dim**-.5, np.float32),
                 "zero": np.array(0, np.float32), "cos": cos, "sin": sin, "mask": mask,
                 "start0": np.array([0], np.int64), "half": np.array([dim//2], np.int64),
                 "end": np.array([dim], np.int64), "steps": np.array([1], np.int64),
                 "repeat_axis": np.array([2], np.int64),
                 "expand_shape": np.array([1, kvh, qh//kvh, length, dim], np.int64),
                 "repeat_shape": np.array([1, qh, length, dim], np.int64)}
    nodes = []
    rotated = {}
    for key, w in (("q", "qw"), ("k", "kw")):
        def node(op, ins, out, **attrs):
            nodes.append(helper.make_node(op, ins, [out], **attrs))
        node("Mul", [key, key], key+"_sq")
        node("ReduceMean", [key+"_sq", "axes"], key+"_mean", keepdims=1)
        node("Add", [key+"_mean", "epsilon"], key+"_eps")
        node("Sqrt", [key+"_eps"], key+"_sqrt")
        node("Div", [key, key+"_sqrt"], key+"_unit")
        node("Mul", [key+"_unit", w], key+"_norm")
        node("Slice", [key+"_norm", "start0", "half", "axes", "steps"], key+"_first")
        node("Slice", [key+"_norm", "half", "end", "axes", "steps"], key+"_second")
        node("Sub", ["zero", key+"_second"], key+"_negative")
        node("Concat", [key+"_negative", key+"_first"], key+"_halfrot", axis=-1)
        node("Mul", [key+"_norm", "cos"], key+"_cos")
        node("Mul", [key+"_halfrot", "sin"], key+"_sin")
        node("Add", [key+"_cos", key+"_sin"], key+"_rope")
        x = feed[key].astype(np.float64)
        norm = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + 1e-6)
        norm *= weights[w].astype(np.float64)
        halfrot = np.concatenate([-norm[..., dim//2:], norm[..., :dim//2]], axis=-1)
        rotated[key] = norm * cos.astype(np.float64) + halfrot * sin.astype(np.float64)
    for key, source in (("k", "k_rope"), ("v", "v")):
        nodes.extend([
            helper.make_node("Unsqueeze", [source, "repeat_axis"], [key+"_unsqueezed"]),
            helper.make_node("Expand", [key+"_unsqueezed", "expand_shape"], [key+"_expanded"]),
            helper.make_node("Reshape", [key+"_expanded", "repeat_shape"], [key+"_repeated"]),
        ])
    nodes.extend([
        helper.make_node("Transpose", ["k_repeated"], ["kt"], perm=[0,1,3,2]),
        helper.make_node("MatMul", ["q_rope", "kt"], ["raw_scores"]),
        helper.make_node("Mul", ["raw_scores", "scale"], ["scores"]),
        helper.make_node("Add", ["scores", "mask"], ["masked"]),
        helper.make_node("Softmax", ["masked"], ["probabilities"], axis=-1),
        helper.make_node("MatMul", ["probabilities", "v_repeated"], ["context"]),
    ])
    expected = np.empty((1, qh, length, dim), np.float32)
    for head in range(qh):
        # Independently enumerate visible keys. Never reuse the exported mask:
        # an erroneous but still causal mask could otherwise self-confirm.
        for query in range(length):
            visible = min(query + 1, valid)
            scores = (rotated["q"][0,head,query] @ rotated["k"][0,head//2,:visible].T) * dim**-.5
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            expected[0,head,query] = probabilities @ feed["v"][0,head//2,:visible].astype(np.float64)
    graph = helper.make_graph(nodes, name,
        [helper.make_tensor_value_info(k, TensorProto.FLOAT, list(v.shape)) for k,v in feed.items()],
        [helper.make_tensor_value_info("context", TensorProto.FLOAT, list(expected.shape))],
        [numpy_helper.from_array(v, k) for k,v in constants.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",18)])
    model.ir_version = 10
    return Case(name, model, feed, expected, valid, relation)

def build_cases():
    return [
        attention_case("full17",17,17),
        attention_case("padding17",17,5),
        attention_case("future17",17,17,change="future",relation=("full17",5)),
        attention_case("padding_changed17",17,5,change="padding",relation=("padding17",17)),
        attention_case("single17",17,1),
        attention_case("boundary256",256,255),
    ]
