"""Structural acceptance of the pinned, decomposed Qwen3 ONNX graph.

This deliberately recognizes the exported architecture, not arbitrary graphs
whose node names contain ``layers.N``. It follows actual operands from the
weights through both residuals in every decoder layer to the returned logits.
No external tensor data is read and no full-model numerical claim is made.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np
import onnx


QWEN3_06B_CONFIG = {
    "num_hidden_layers": 28,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "sequence_length": 256,
    "rms_norm_eps": 1e-6,
}


class Qwen3StructureError(ValueError):
    """A missing or incorrectly connected part of the Qwen3 architecture."""


def _require(condition, message):
    if not condition:
        raise Qwen3StructureError(message)


def _attrs(node):
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _one(nodes, context):
    _require(len(nodes) == 1, f"{context}: expected exactly one node, found {len(nodes)}")
    return nodes[0]


class _Graph:
    def __init__(self, graph):
        self.graph = graph
        self.initializers = {t.name: t for t in graph.initializer}
        self.nodes = list(graph.node)
        self.producers = {}
        self.consumers = defaultdict(list)
        self.by_op = defaultdict(list)
        self.bits = {}
        self.ancestors = {}
        self.constants = {}
        self.indices = {}
        roots = list(dict.fromkeys([*(v.name for v in graph.input), *self.initializers]))
        for index, name in enumerate(roots):
            self.bits[name] = self.ancestors[name] = 1 << index
        for index, node in enumerate(self.nodes):
            label = node.name or f"node[{index}]"
            self.indices[id(node)] = index
            _require(node.domain in ("", "ai.onnx"), f"{label}: unsupported domain {node.domain!r}")
            _require(len(node.output) == 1 and node.output[0],
                     f"{label}: this pinned graph requires one nonempty output per node")
            output = node.output[0]
            _require(output not in self.ancestors, f"{label}: duplicate definition {output!r}")
            bit = 1 << (len(roots) + index)
            inherited = bit
            for value in node.input:
                if not value:
                    continue
                _require(value in self.ancestors,
                         f"{label}: input {value!r} has no preceding definition")
                inherited |= self.ancestors[value]
                self.consumers[value].append(node)
            self.bits[output] = bit
            self.ancestors[output] = inherited
            self.producers[output] = node
            self.by_op[node.op_type].append(node)

    def depends(self, output, value):
        return bool(self.ancestors.get(output, 0) & self.bits.get(value, 0))

    def strip(self, value, extra=()):
        while value in self.producers:
            node = self.producers[value]
            if node.op_type not in ("Identity", "Cast", *extra):
                break
            value = node.input[0]
        return value

    def producer(self, value, op, context):
        node = self.producers.get(self.strip(value))
        _require(node is not None and node.op_type == op,
                 f"{context}: expected {op} producer for {value!r}")
        return node

    def same(self, left, right, extra=()):
        return self.strip(left, extra) == self.strip(right, extra)

    def constant(self, value):
        """Resolve only tiny control constants; never open external weights."""
        if value in self.constants:
            return self.constants[value]
        if value in self.initializers:
            tensor = self.initializers[value]
            _require(not tensor.external_data and math.prod(tensor.dims) <= 1024,
                     f"constant {value!r}: external or large tensor is not a control constant")
            result = onnx.numpy_helper.to_array(tensor)
        else:
            node = self.producers.get(value)
            _require(node is not None, f"constant {value!r}: missing definition")
            attrs = _attrs(node)
            op = node.op_type
            if op == "Constant":
                if "value" in attrs:
                    tensor = attrs["value"]
                    _require(not tensor.external_data and math.prod(tensor.dims) <= 1024,
                             f"constant {value!r}: external or large tensor")
                    result = onnx.numpy_helper.to_array(tensor)
                elif "value_int" in attrs:
                    result = np.asarray(attrs["value_int"], dtype=np.int64)
                elif "value_ints" in attrs:
                    result = np.asarray(attrs["value_ints"], dtype=np.int64)
                elif "value_float" in attrs:
                    result = np.asarray(attrs["value_float"], dtype=np.float32)
                elif "value_floats" in attrs:
                    result = np.asarray(attrs["value_floats"], dtype=np.float32)
                else:
                    raise Qwen3StructureError(f"constant {value!r}: unsupported Constant encoding")
            elif op in ("Identity", "Cast", "Abs"):
                result = self.constant(node.input[0])
                if op == "Cast":
                    result = result.astype(onnx.helper.tensor_dtype_to_np_dtype(attrs["to"]))
                elif op == "Abs":
                    result = np.abs(result)
            elif op == "Reshape":
                data = self.constant(node.input[0])
                shape = self.constant(node.input[1]).reshape(-1).tolist()
                shape = [data.shape[i] if d == 0 else d for i, d in enumerate(shape)]
                result = data.reshape(shape)
            elif op == "Concat":
                result = np.concatenate([self.constant(v) for v in node.input], axis=attrs["axis"])
            elif op == "Unsqueeze":
                axes = self.constant(node.input[1]).reshape(-1).tolist()
                result = np.expand_dims(self.constant(node.input[0]), tuple(axes))
            else:
                raise Qwen3StructureError(f"constant {value!r}: unexpected {op} dependency")
        _require(result.size <= 1024, f"constant {value!r}: exceeds control-constant size limit")
        self.constants[value] = result
        return result

    def scalar(self, value, expected, context):
        data = self.constant(value)
        _require(data.size == 1 and np.isfinite(data).all()
                 and math.isclose(float(data.reshape(-1)[0]), expected, rel_tol=1e-6, abs_tol=0),
                 f"{context}: {value!r} must be scalar {expected}")

    def projection(self, weight, context):
        candidates = []
        for node in self.by_op["MatMul"]:
            if len(node.input) != 2:
                continue
            transposed = self.producers.get(self.strip(node.input[1]))
            if (transposed is not None and transposed.op_type == "Transpose"
                    and self.same(transposed.input[0], weight)):
                _require(tuple(_attrs(transposed).get("perm", (1, 0))) == (1, 0),
                         f"{context}: weight transpose must use perm=[1, 0]")
                candidates.append(node)
        return _one(candidates, f"{context}, weight {weight!r}")

    def norm(self, weight, epsilon, context):
        node = _one([n for n in self.by_op["Mul"]
                     if len(n.input) == 2 and any(self.same(v, weight) for v in n.input)],
                    f"{context}, weight {weight!r}")
        normalized = _one([v for v in node.input if not self.same(v, weight)],
                          f"{context}: normalized activation")
        multiply = self.producer(normalized, "Mul", context)
        reciprocals = [v for v in multiply.input
                       if self.producers.get(self.strip(v)) is not None
                       and self.producers[self.strip(v)].op_type == "Reciprocal"]
        reciprocal_value = _one(reciprocals, f"{context}: RMS inverse square root")
        reciprocal = self.producer(reciprocal_value, "Reciprocal", context)
        raw = next(v for v in multiply.input if v != reciprocal_value)
        sqrt = self.producer(reciprocal.input[0], "Sqrt", context)
        add = self.producer(sqrt.input[0], "Add", context)
        means = [v for v in add.input if self.producers.get(self.strip(v)) is not None
                 and self.producers[self.strip(v)].op_type == "ReduceMean"]
        mean_value = _one(means, f"{context}: RMS mean")
        self.scalar(next(v for v in add.input if v != mean_value), epsilon, context)
        mean = self.producer(mean_value, "ReduceMean", context)
        attrs = _attrs(mean)
        axes = (self.constant(mean.input[1]).reshape(-1).tolist()
                if len(mean.input) == 2 else list(attrs.get("axes", [])))
        _require(axes == [-1] and attrs.get("keepdims", 1) == 1,
                 f"{context}: RMS ReduceMean must use axes=[-1], keepdims=1")
        power = self.producer(mean.input[0], "Pow", context)
        _require(self.same(power.input[0], raw), f"{context}: RMS numerator/denominator input mismatch")
        self.scalar(power.input[1], 2.0, context)
        return node, self.strip(raw)

    def residual(self, branch, bypass, context):
        candidates = [n for n in self.by_op["Add"] if len(n.input) == 2
                      and ((self.same(n.input[0], branch) and self.same(n.input[1], bypass))
                           or (self.same(n.input[1], branch) and self.same(n.input[0], bypass)))]
        return _one(candidates, context)


def make_control_constant_reader(model):
    """Return ``reader(onnx_value_name) -> ndarray`` for small control values.

    Resolution uses the original ONNX graph, never the parser, its cached
    constants, or the emitted IR. Every intermediate is limited to 1,024
    elements and external control tensors are rejected. This is the bounded
    constant subset needed by the pinned export's axes, slices and shapes,
    not a general ONNX interpreter or a way to load model weights.
    """
    return _Graph(model.graph).constant


def _metadata(graph, config):
    hidden, head = config["hidden_size"], config["head_dim"]
    query, kv = config["num_attention_heads"] * head, config["num_key_value_heads"] * head
    middle = config["intermediate_size"]
    per_layer = {
        "self_attn.q_proj.weight": (query, hidden),
        "self_attn.k_proj.weight": (kv, hidden),
        "self_attn.v_proj.weight": (kv, hidden),
        "self_attn.o_proj.weight": (hidden, query),
        "self_attn.q_norm.weight": (head,),
        "self_attn.k_norm.weight": (head,),
        "mlp.gate_proj.weight": (middle, hidden),
        "mlp.up_proj.weight": (middle, hidden),
        "mlp.down_proj.weight": (hidden, middle),
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
    }
    expected = {f"model.model.layers.{i}.{name}": (shape, onnx.TensorProto.FLOAT)
                for i in range(config["num_hidden_layers"]) for name, shape in per_layer.items()}
    expected.update({
        "model.model.norm.weight": ((hidden,), onnx.TensorProto.FLOAT),
        "model.lm_head.weight": ((config["vocab_size"], hidden), onnx.TensorProto.FLOAT),
        "positions": ((1, config["sequence_length"]), onnx.TensorProto.INT64),
        "model.model.rotary_emb.inv_freq": ((head // 2,), onnx.TensorProto.FLOAT),
    })
    actual = {t.name: t for t in graph.initializer}
    _require(len(actual) == len(graph.initializer), "initializers: duplicate weight names")
    _require(set(actual) == set(expected),
             f"initializers: missing={sorted(set(expected) - set(actual))}, "
             f"unexpected={sorted(set(actual) - set(expected))}")
    for name, (shape, dtype) in expected.items():
        tensor = actual[name]
        _require(tuple(tensor.dims) == shape and tensor.data_type == dtype,
                 f"weight {name!r}: expected shape={shape}, dtype={dtype}; "
                 f"got shape={tuple(tensor.dims)}, dtype={tensor.data_type}")
    length = config["sequence_length"]
    signature = {"input_ids": ((1, length), onnx.TensorProto.INT64),
                 "attention_mask": ((1, 1, length, length), onnx.TensorProto.FLOAT),
                 "logits": ((1, length, config["vocab_size"]), onnx.TensorProto.FLOAT)}
    _require([v.name for v in graph.input] == ["input_ids", "attention_mask"],
             "graph inputs must be input_ids and attention_mask, in that order")
    _require([v.name for v in graph.output] == ["logits"], "graph output must be logits")
    for value in [*graph.input, *graph.output]:
        tensor = value.type.tensor_type
        shape, dtype = signature[value.name]
        _require(all(d.HasField("dim_value") for d in tensor.shape.dim)
                 and tuple(d.dim_value for d in tensor.shape.dim) == shape
                 and tensor.elem_type == dtype, f"{value.name}: incorrect static shape or dtype")


def _attention(index, q, k, v, out, q_norm, k_norm, config, context):
    qv, kv = q_norm.output[0], k_norm.output[0]
    softmax = _one([n for n in index.by_op["Softmax"]
                    if index.depends(out.input[0], n.output[0])
                    and index.depends(n.input[0], qv) and index.depends(n.input[0], kv)],
                   f"{context}: attention Softmax")
    _require(_attrs(softmax).get("axis", -1) == -1,
             f"{context}: attention Softmax must use axis=-1")
    scores = _one([n for n in index.by_op["MatMul"]
                  if index.depends(softmax.input[0], n.output[0])
                  and index.depends(n.input[0], qv) and index.depends(n.input[1], kv)],
                 f"{context}: QK attention MatMul")
    _require(not index.depends(scores.input[0], kv) and not index.depends(scores.input[1], qv),
             f"{context}: Q and K score operands must retain separate paths")
    weighted = _one([n for n in index.by_op["MatMul"]
                    if index.depends(out.input[0], n.output[0])
                    and index.same(n.input[0], softmax.output[0])
                    and index.depends(n.input[1], v.output[0])],
                   f"{context}: attention probabilities times V")
    mask_add = index.producer(softmax.input[0], "Add", context)
    masks = [value for value in mask_add.input
             if index.depends(value, "attention_mask") and not index.depends(value, scores.output[0])]
    mask = _one(masks, f"{context}: additive attention mask")
    scale = index.producer(next(value for value in mask_add.input if value != mask), "Mul", context)
    scale_inputs = [value for value in scale.input if not index.same(value, scores.output[0])]
    _require(len(scale_inputs) == 1, f"{context}: scores must be multiplied by a scalar scale")
    index.scalar(scale_inputs[0], config["head_dim"] ** -0.5, f"{context}: attention scale")
    for op in ("Cos", "Sin"):
        rope = _one([n for n in index.by_op[op]
                     if index.depends(scores.input[0], n.output[0])
                     and index.depends(scores.input[1], n.output[0])], f"{context}: shared RoPE {op}")
        _require(index.depends(rope.input[0], "positions")
                 and index.depends(rope.input[0], "model.model.rotary_emb.inv_freq"),
                 f"{context}: RoPE {op} must use positions and inv_freq")
    if config["num_attention_heads"] != config["num_key_value_heads"]:
        expected = (1, config["num_key_value_heads"],
                    config["num_attention_heads"] // config["num_key_value_heads"],
                    config["sequence_length"], config["head_dim"])
        key_transpose = index.producer(scores.input[1], "Transpose", f"{context}: K score transpose")
        _require(tuple(_attrs(key_transpose).get("perm", ())) == (0, 1, 3, 2),
                 f"{context}: K score transpose must use perm=[0, 1, 3, 2]")
        for value, source, label in ((key_transpose.input[0], kv, "K"),
                                     (weighted.input[1], v.output[0], "V")):
            reshape = index.producer(value, "Reshape", f"{context}: GQA {label} flattened heads")
            shape = tuple(index.constant(reshape.input[1]).reshape(-1).tolist())
            target = (1, config["num_attention_heads"], config["sequence_length"], config["head_dim"])
            _require(shape == target, f"{context}: GQA {label} head reshape {shape} != {target}")
            expand = index.producer(reshape.input[0], "Expand", f"{context}: GQA {label} expansion")
            _require(index.depends(expand.input[0], source),
                     f"{context}: GQA {label} expansion does not consume its projection")
            shape = tuple(index.constant(expand.input[1]).reshape(-1).tolist())
            _require(shape == expected, f"{context}: GQA {label} expansion shape {shape} != {expected}")
            source_value = index.strip(expand.input[0], ("Slice",))
            unsqueeze = index.producer(source_value, "Unsqueeze", f"{context}: GQA {label} group axis")
            axes = tuple(index.constant(unsqueeze.input[1]).reshape(-1).tolist())
            _require(axes == (2,), f"{context}: GQA {label} repeat group must be axis 2")
    return {"scores": scores.name, "softmax": softmax.name, "weighted_values": weighted.name}


def audit_graph_structure(model, config=None):
    """Check the complete decoder dataflow without loading external weights.

    ``config`` overrides architecture dimensions for small test fixtures. This
    gate recognizes the pinned export's decomposed RMSNorm/RoPE/GQA/SwiGLU;
    a different exporter representation needs a reviewed new pattern.
    """
    config = dict(QWEN3_06B_CONFIG) | dict(config or {})
    for name, value in config.items():
        if name == "rms_norm_eps":
            _require(isinstance(value, (float, int)) and not isinstance(value, bool)
                     and math.isfinite(value) and value > 0, f"config: invalid {name}")
        else:
            _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                     f"config: invalid {name}")
    _require(config["head_dim"] % 2 == 0, "config: head_dim must be even for RoPE")
    _require(config["num_attention_heads"] % config["num_key_value_heads"] == 0,
             "config: attention heads must be divisible by KV heads")
    _metadata(model.graph, config)
    index = _Graph(model.graph)
    shared_weight = "model.lm_head.weight"
    embedding = _one([n for n in index.by_op["Gather"]
                      if list(n.input) == [shared_weight, "input_ids"]],
                     "embedding: tied LM Head weight and input_ids Gather")
    _require(_attrs(embedding).get("axis", 0) == 0, "embedding: Gather axis must be zero")
    previous = embedding.output[0]
    rows = []
    for layer in range(config["num_hidden_layers"]):
        context = f"layer {layer}"
        prefix = f"model.model.layers.{layer}."
        projections = {name: index.projection(prefix + name + ".weight", f"{context} {name}")
                       for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
        norms = {name: index.norm(prefix + name + ".weight", config["rms_norm_eps"], f"{context} {name}")
                 for name in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm")}
        input_norm, raw = norms["input_layernorm"]
        _require(index.same(raw, previous), f"{context}: input norm must consume previous layer residual")
        q, k, v, out = (projections[f"self_attn.{name}_proj"] for name in ("q", "k", "v", "o"))
        _require(all(index.same(n.input[0], input_norm.output[0]) for n in (q, k, v)),
                 f"{context}: Q/K/V projections must share this layer's input norm")
        q_norm, q_raw = norms["self_attn.q_norm"]
        k_norm, k_raw = norms["self_attn.k_norm"]
        _require(index.same(q_raw, q.output[0], ("Reshape",))
                 and index.same(k_raw, k.output[0], ("Reshape",)),
                 f"{context}: Q/K norm must consume the matching projection")
        attention = _attention(index, q, k, v, out, q_norm, k_norm, config, context)
        first_residual = index.residual(out.output[0], previous, f"{context}: attention residual")
        post_norm, post_raw = norms["post_attention_layernorm"]
        _require(index.same(post_raw, first_residual.output[0]),
                 f"{context}: post-attention norm must consume attention residual")
        gate, up, down = (projections[f"mlp.{name}_proj"] for name in ("gate", "up", "down"))
        _require(all(index.same(n.input[0], post_norm.output[0]) for n in (gate, up)),
                 f"{context}: gate/up projections must share post-attention norm")
        product = index.producer(down.input[0], "Mul", f"{context}: SwiGLU")
        activation_values = [value for value in product.input if not index.same(value, up.output[0])]
        _require(len(activation_values) == 1, f"{context}: SwiGLU must multiply by up projection")
        activation = index.producer(activation_values[0], "Mul", f"{context}: SiLU")
        sigmoid_values = [value for value in activation.input if not index.same(value, gate.output[0])]
        _require(len(sigmoid_values) == 1, f"{context}: SiLU must multiply gate by sigmoid(gate)")
        sigmoid = index.producer(sigmoid_values[0], "Sigmoid", f"{context}: SiLU")
        _require(index.same(sigmoid.input[0], gate.output[0]), f"{context}: SiLU sigmoid input must be gate")
        second_residual = index.residual(down.output[0], first_residual.output[0],
                                         f"{context}: MLP residual")
        previous = second_residual.output[0]
        rows.append({"layer": layer, "passed": True, "input": raw, "output": previous,
                     "projections": {name: node.name for name, node in projections.items()},
                     "norms": {name: node.name for name, (node, _) in norms.items()},
                     "attention": attention,
                     "residuals": [first_residual.name, second_residual.name]})
    final_norm, raw = index.norm("model.model.norm.weight", config["rms_norm_eps"], "final norm")
    _require(index.same(raw, previous), "final norm must consume the last layer residual")
    head = index.projection(shared_weight, "LM Head")
    _require(index.same(head.input[0], final_norm.output[0], ("Slice",)),
             "LM Head must consume final norm (allowing the exporter's logits slices)")
    _require(index.same("logits", head.output[0]), "logits must return the tied LM Head projection")
    live = [node for node in index.nodes if index.depends("logits", node.output[0])]
    dead = [node.name or node.output[0] for node in index.nodes if not index.depends("logits", node.output[0])]
    _require(not dead, f"graph has nodes disconnected from logits: {dead[:8]}")
    for name in index.initializers:
        _require(index.depends("logits", name), f"initializer {name!r} is disconnected from logits")
    return {"passed": True, "config": config,
            "graph": {"node_count": len(index.nodes), "initializer_count": len(index.initializers),
                      "operators": dict(sorted(Counter(n.op_type for n in index.nodes).items())),
                      "inputs": [v.name for v in model.graph.input], "outputs": ["logits"]},
            "layers": rows, "layer_count": len(rows), "live_node_count": len(live),
            "embedding": embedding.name, "final_norm": final_norm.name, "lm_head": head.name,
            "tied_embeddings": True,
            "checks": ["exact weight inventory, shapes and dtypes", "static input/output signature",
                       "RMSNorm paths and constants", "Q/K/V and GQA attention dataflow",
                       "shared RoPE position/frequency dependencies", "SwiGLU dataflow",
                       "both residuals and ordered layer chain", "final norm and tied LM Head",
                       "every node and initializer reaches logits"],
            "scope": "Structural acceptance of the pinned export; no numerical execution or external weight reads."}
