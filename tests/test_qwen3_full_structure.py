"""Small complete decoder fixtures and deliberately corrupted structure gates."""

from copy import deepcopy

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")

from probes.w2_qwen3_parse.validation import (
    Qwen3StructureError, audit_graph_structure, make_control_constant_reader,
)


def qwen3_fixture(layers=2):
    """A tiny, valid ONNX decoder using every required architectural path."""
    config = {"num_hidden_layers": layers, "hidden_size": 4, "intermediate_size": 6,
              "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
              "vocab_size": 9, "sequence_length": 3, "rms_norm_eps": 1e-6}
    nodes, weights, constants = [], [], {}

    def node(op, inputs, output, **attrs):
        nodes.append(onnx.helper.make_node(op, inputs, [output], name=f"opaque_{len(nodes)}", **attrs))
        return output

    def constant(value, dtype=np.int64):
        array = np.asarray(value, dtype=dtype)
        key = (array.dtype.str, array.shape, array.tobytes())
        if key not in constants:
            constants[key] = node("Constant", [], f"constant_{len(constants)}",
                                  value=onnx.numpy_helper.from_array(array))
        return constants[key]

    def weight(name, shape):
        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100 + len(weights) / 1000
        weights.append(onnx.numpy_helper.from_array(data, name))

    shapes = {"self_attn.q_proj.weight": (4, 4), "self_attn.k_proj.weight": (2, 4),
              "self_attn.v_proj.weight": (2, 4), "self_attn.o_proj.weight": (4, 4),
              "self_attn.q_norm.weight": (2,), "self_attn.k_norm.weight": (2,),
              "mlp.gate_proj.weight": (6, 4), "mlp.up_proj.weight": (6, 4),
              "mlp.down_proj.weight": (4, 6), "input_layernorm.weight": (4,),
              "post_attention_layernorm.weight": (4,)}
    for layer in range(layers):
        for name, shape in shapes.items():
            weight(f"model.model.layers.{layer}.{name}", shape)
    weight("model.model.norm.weight", (4,))
    weight("model.lm_head.weight", (9, 4))
    weights.append(onnx.numpy_helper.from_array(np.arange(3, dtype=np.int64).reshape(1, 3), "positions"))
    weights.append(onnx.numpy_helper.from_array(np.array([1.0], dtype=np.float32), "model.model.rotary_emb.inv_freq"))

    def norm(raw, name, weight_name):
        square = node("Pow", [raw, constant(2, np.float32)], name + "_square")
        mean = node("ReduceMean", [square, constant([-1])], name + "_mean", keepdims=1)
        plus = node("Add", [mean, constant(config["rms_norm_eps"], np.float32)], name + "_eps")
        sqrt = node("Sqrt", [plus], name + "_sqrt")
        inv = node("Reciprocal", [sqrt], name + "_inv")
        scaled = node("Mul", [raw, inv], name + "_scaled")
        return node("Mul", [weight_name, scaled], name)

    def projection(raw, name, weight_name):
        transpose = node("Transpose", [weight_name], name + "_weight", perm=[1, 0])
        return node("MatMul", [raw, transpose], name)

    def rope(raw, name, cosine, sine):
        first = node("Slice", [raw, constant([0]), constant([1]), constant([-1])], name + "_first")
        second = node("Slice", [raw, constant([1]), constant([2]), constant([-1])], name + "_second")
        negative = node("Neg", [second], name + "_negative")
        rotated = node("Concat", [negative, first], name + "_rotated", axis=-1)
        original = node("Mul", [raw, cosine], name + "_cos")
        rotated = node("Mul", [rotated, sine], name + "_sin")
        return node("Add", [original, rotated], name)

    def repeat(raw, name):
        expanded = node("Unsqueeze", [raw, constant([2])], name + "_unsqueezed")
        expanded = node("Expand", [expanded, constant([1, 1, 2, 3, 2])], name + "_expanded")
        return node("Reshape", [expanded, constant([1, 2, 3, 2])], name)

    previous = node("Gather", ["model.lm_head.weight", "input_ids"], "embedding", axis=0)
    positions = node("Cast", ["positions"], "float_positions", to=onnx.TensorProto.FLOAT)
    positions = node("Unsqueeze", [positions, constant([1])], "positions_3d")
    freq = node("Unsqueeze", ["model.model.rotary_emb.inv_freq", constant([0, 2])], "freq_3d")
    freq = node("MatMul", [freq, positions], "freq_matmul")
    freq = node("Transpose", [freq], "freq_transpose", perm=[0, 2, 1])
    freq = node("Concat", [freq, freq], "freq_full_head", axis=-1)
    cosine = node("Cos", [freq], "cosine")
    sine = node("Sin", [freq], "sine")
    cosine = node("Unsqueeze", [cosine, constant([1])], "cosine_4d")
    sine = node("Unsqueeze", [sine, constant([1])], "sine_4d")
    for layer in range(layers):
        prefix = f"layer_{layer}"
        path = f"model.model.layers.{layer}."
        normalized = norm(previous, prefix + "_input_norm", path + "input_layernorm.weight")
        q = projection(normalized, prefix + "_q", path + "self_attn.q_proj.weight")
        k = projection(normalized, prefix + "_k", path + "self_attn.k_proj.weight")
        v = projection(normalized, prefix + "_v", path + "self_attn.v_proj.weight")
        q = node("Reshape", [q, constant([1, 3, 2, 2])], prefix + "_q_view")
        k = node("Reshape", [k, constant([1, 3, 1, 2])], prefix + "_k_view")
        v = node("Reshape", [v, constant([1, 3, 1, 2])], prefix + "_v_view")
        q = norm(q, prefix + "_q_norm", path + "self_attn.q_norm.weight")
        k = norm(k, prefix + "_k_norm", path + "self_attn.k_norm.weight")
        q = node("Transpose", [q], prefix + "_q_transpose", perm=[0, 2, 1, 3])
        k = node("Transpose", [k], prefix + "_k_transpose", perm=[0, 2, 1, 3])
        v = node("Transpose", [v], prefix + "_v_transpose", perm=[0, 2, 1, 3])
        q = rope(q, prefix + "_q_rope", cosine, sine)
        k = rope(k, prefix + "_k_rope", cosine, sine)
        k = repeat(k, prefix + "_k_repeat")
        v = repeat(v, prefix + "_v_repeat")
        k = node("Transpose", [k], prefix + "_k_scores", perm=[0, 1, 3, 2])
        scores = node("MatMul", [q, k], prefix + "_scores")
        scores = node("Mul", [scores, constant(2 ** -0.5, np.float32)], prefix + "_scaled_scores")
        scores = node("Add", [scores, "attention_mask"], prefix + "_masked_scores")
        probabilities = node("Softmax", [scores], prefix + "_probabilities", axis=-1)
        values = node("MatMul", [probabilities, v], prefix + "_weighted_values")
        values = node("Transpose", [values], prefix + "_attention_transpose", perm=[0, 2, 1, 3])
        values = node("Reshape", [values, constant([1, 3, 4])], prefix + "_attention_view")
        output = projection(values, prefix + "_o", path + "self_attn.o_proj.weight")
        residual = node("Add", [previous, output], prefix + "_attention_residual")
        normalized = norm(residual, prefix + "_post_norm", path + "post_attention_layernorm.weight")
        gate = projection(normalized, prefix + "_gate", path + "mlp.gate_proj.weight")
        up = projection(normalized, prefix + "_up", path + "mlp.up_proj.weight")
        sigmoid = node("Sigmoid", [gate], prefix + "_sigmoid")
        activated = node("Mul", [gate, sigmoid], prefix + "_silu")
        swiglu = node("Mul", [activated, up], prefix + "_swiglu")
        down = projection(swiglu, prefix + "_down", path + "mlp.down_proj.weight")
        previous = node("Add", [residual, down], prefix + "_mlp_residual")
    normalized = norm(previous, "final_norm", "model.model.norm.weight")
    projection(normalized, "logits", "model.lm_head.weight")
    helper = onnx.helper.make_tensor_value_info
    graph = onnx.helper.make_graph(nodes, "small_full_structure",
                                  [helper("input_ids", onnx.TensorProto.INT64, [1, 3]),
                                   helper("attention_mask", onnx.TensorProto.FLOAT, [1, 1, 3, 3])],
                                  [helper("logits", onnx.TensorProto.FLOAT, [1, 3, 9])], weights)
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10)
    return model, config


def output_node(model, name):
    return next(node for node in model.graph.node if name in node.output)


def replace_control(model, output, input_index, value, dtype=np.int64):
    target_index = next(i for i, node in enumerate(model.graph.node) if output in node.output)
    constant_name = output + "_corrupt_control"
    constant = onnx.helper.make_node("Constant", [], [constant_name],
                                    value=onnx.numpy_helper.from_array(np.asarray(value, dtype=dtype)))
    model.graph.node.insert(target_index, constant)
    output_node(model, output).input[input_index] = constant_name


def test_complete_small_graph_validates_actual_dataflow():
    model, config = qwen3_fixture()
    onnx.checker.check_model(model, full_check=True)
    result = audit_graph_structure(model, config)
    assert result["passed"] and result["tied_embeddings"]
    assert result["layer_count"] == 2
    assert result["graph"]["initializer_count"] == 26
    assert result["live_node_count"] == len(model.graph.node)
    assert result["layers"][1]["input"] == "layer_0_mlp_residual"


def test_layer_detection_does_not_depend_on_node_names_or_exporter_metadata():
    model, config = qwen3_fixture()
    for node in model.graph.node:
        node.name = ""
        del node.metadata_props[:]
    assert audit_graph_structure(model, config)["layer_count"] == 2


def test_external_weight_bytes_are_not_loaded(monkeypatch):
    model, config = qwen3_fixture()
    for tensor in model.graph.initializer:
        onnx.external_data_helper.set_external_data(tensor, "does-not-exist.data", offset=0,
                                                   length=len(tensor.raw_data))
        tensor.ClearField("raw_data")
        tensor.data_location = onnx.TensorProto.EXTERNAL
    def forbidden(*args, **kwargs):
        raise AssertionError("structure validation must not read external weight data")
    monkeypatch.setattr(onnx.external_data_helper, "load_external_data_for_tensor", forbidden)
    assert audit_graph_structure(model, config)["passed"]


def test_control_reader_uses_original_onnx_graph_without_parser_state():
    model, _ = qwen3_fixture()
    reader = make_control_constant_reader(model)
    shape_name = output_node(model, "layer_0_k_repeat_expanded").input[1]
    np.testing.assert_array_equal(reader(shape_name), [1, 1, 2, 3, 2])
    assert reader(shape_name).dtype == np.int64


def test_control_reader_rejects_external_and_large_tensors_without_loading():
    model, _ = qwen3_fixture()
    tensor = model.graph.initializer[0]
    onnx.external_data_helper.set_external_data(tensor, "missing.data", offset=0, length=len(tensor.raw_data))
    tensor.ClearField("raw_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    reader = make_control_constant_reader(model)
    with pytest.raises(Qwen3StructureError, match="external or large tensor"):
        reader(tensor.name)
    big = onnx.numpy_helper.from_array(np.zeros(1025, dtype=np.int64), "big_control")
    model.graph.initializer.append(big)
    with pytest.raises(Qwen3StructureError, match="external or large tensor"):
        make_control_constant_reader(model)("big_control")


@pytest.mark.parametrize("change,expected", [("shape", "expected shape"), ("dtype", "dtype"),
                                               ("missing", "missing="), ("extra", "unexpected="),
                                               ("duplicate", "duplicate weight")])
def test_weight_inventory_rejects_incorrect_metadata(change, expected):
    model, config = qwen3_fixture()
    tensor = model.graph.initializer[0]
    if change == "shape":
        tensor.dims[0] += 1
    elif change == "dtype":
        tensor.data_type = onnx.TensorProto.FLOAT16
    elif change == "missing":
        del model.graph.initializer[0]
    elif change == "extra":
        extra = deepcopy(tensor)
        extra.name = "model.model.layers.99.self_attn.q_proj.weight"
        model.graph.initializer.append(extra)
    else:
        model.graph.initializer.append(deepcopy(tensor))
    with pytest.raises(Qwen3StructureError, match=expected):
        audit_graph_structure(model, config)


@pytest.mark.parametrize("target", ["layer_1_input_norm_square", "layer_1_input_norm_scaled"])
def test_cross_layer_norm_must_use_same_previous_residual(target):
    model, config = qwen3_fixture()
    output_node(model, target).input[0] = "embedding"
    with pytest.raises(Qwen3StructureError, match="layer 1.*input mismatch"):
        audit_graph_structure(model, config)


def test_bypassed_layer_is_rejected_even_when_all_weights_remain():
    model, config = qwen3_fixture()
    for name in ("final_norm_square", "final_norm_scaled"):
        output_node(model, name).input[0] = "layer_0_mlp_residual"
    with pytest.raises(Qwen3StructureError, match="final norm must consume the last layer"):
        audit_graph_structure(model, config)


def test_wrong_residual_bypass_is_rejected():
    model, config = qwen3_fixture()
    output_node(model, "layer_1_attention_residual").input[0] = "embedding"
    with pytest.raises(Qwen3StructureError, match="layer 1: attention residual"):
        audit_graph_structure(model, config)


def test_dead_final_norm_is_rejected():
    model, config = qwen3_fixture()
    output_node(model, "logits").input[0] = "layer_1_mlp_residual"
    with pytest.raises(Qwen3StructureError, match="LM Head must consume final norm"):
        audit_graph_structure(model, config)


def test_shared_embedding_weight_cannot_be_replaced_by_another_matrix():
    model, config = qwen3_fixture()
    output_node(model, "embedding").input[0] = "model.model.layers.0.self_attn.q_proj.weight"
    with pytest.raises(Qwen3StructureError, match="embedding: tied LM Head"):
        audit_graph_structure(model, config)


def test_same_shape_layer_weights_cannot_be_substituted():
    model, config = qwen3_fixture()
    output_node(model, "layer_0_q_weight").input[0] = "model.model.layers.1.self_attn.q_proj.weight"
    with pytest.raises(Qwen3StructureError, match="layer 0 self_attn.q_proj"):
        audit_graph_structure(model, config)


@pytest.mark.parametrize("node_name,new_input,expected", [
    ("layer_0_q_norm_square", "layer_0_k_view", "input mismatch"),
    ("layer_0_sigmoid", "layer_0_up", "sigmoid input must be gate"),
    ("layer_0_masked_scores", "layer_0_scores", "additive attention mask"),
])
def test_key_computations_cannot_be_rewired(node_name, new_input, expected):
    model, config = qwen3_fixture()
    node = output_node(model, node_name)
    node.input[1 if node_name == "layer_0_masked_scores" else 0] = new_input
    with pytest.raises(Qwen3StructureError, match=expected):
        audit_graph_structure(model, config)


def test_disconnected_extra_compute_node_cannot_hide_in_a_complete_graph():
    model, config = qwen3_fixture()
    model.graph.node.append(onnx.helper.make_node("Identity", ["logits"], ["dead_output"], name="dead_node"))
    with pytest.raises(Qwen3StructureError, match="disconnected from logits.*dead_node"):
        audit_graph_structure(model, config)


@pytest.mark.parametrize("output,input_index,value,dtype,expected", [
    ("layer_0_input_norm_square", 1, 3, np.float32, "must be scalar 2"),
    ("layer_0_input_norm_mean", 1, [0], np.int64, "axes=\\[-1\\]"),
    ("layer_0_input_norm_eps", 1, 0.001, np.float32, "must be scalar 1e-06"),
    ("layer_0_scaled_scores", 1, 1, np.float32, "attention scale"),
    ("layer_0_k_repeat_expanded", 1, [1, 2, 1, 3, 2], np.int64, "GQA K expansion shape"),
    ("layer_0_v_repeat", 1, [1, 1, 3, 4], np.int64, "GQA V head reshape"),
    ("layer_0_v_repeat_unsqueezed", 1, [1], np.int64, "GQA V repeat group must be axis 2"),
])
def test_numerical_structure_constants_and_group_layout_are_checked(output, input_index, value, dtype, expected):
    model, config = qwen3_fixture()
    replace_control(model, output, input_index, value, dtype)
    with pytest.raises(Qwen3StructureError, match=expected):
        audit_graph_structure(model, config)


def test_norm_weight_used_as_both_operands_has_a_clear_validation_error():
    model, config = qwen3_fixture()
    node = output_node(model, "layer_0_input_norm")
    node.input[1] = node.input[0]
    with pytest.raises(Qwen3StructureError, match="normalized activation"):
        audit_graph_structure(model, config)


@pytest.mark.parametrize("change,expected", [("output", "graph output"), ("shape", "incorrect static shape"),
                                               ("dtype", "incorrect static shape")])
def test_static_interface_is_required(change, expected):
    model, config = qwen3_fixture()
    if change == "output":
        model.graph.output[0].name = "layer_1_mlp_residual"
    elif change == "shape":
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 2
    else:
        model.graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.INT32
    with pytest.raises(Qwen3StructureError, match=expected):
        audit_graph_structure(model, config)
