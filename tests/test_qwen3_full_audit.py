"""The full-model acceptance audit must reject plausible but incomplete parses."""

import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T, helper, numpy_helper

from probes.w2_qwen3_parse.audit import (
    AuditError, AuditedONNXParser, audit_parsed_model,
)
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.ir.types import DataType, OpCode, Value


def save(tmp_path, nodes, arrays=None, outputs=None, external=False):
    graph = helper.make_graph(
        nodes, "audit_fixture", [helper.make_tensor_value_info("x", T.FLOAT, [2, 2])],
        outputs or [helper.make_tensor_value_info("logits", T.FLOAT, [2, 2])],
        [numpy_helper.from_array(array, name) for name, array in (arrays or {}).items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
    onnx.checker.check_model(model)
    path = tmp_path / "model.onnx"
    if external:
        onnx.save_model(model, path, save_as_external_data=True,
                        all_tensors_to_one_file=True, location="weights.data", size_threshold=0)
    else:
        onnx.save_model(model, path)
    return path


@pytest.fixture
def parsed(tmp_path):
    nodes = [helper.make_node("MatMul", ["x", "w1"], ["a"], name="project"),
             helper.make_node("Add", ["a", "w2"], ["b"], name="add_weight"),
             helper.make_node("Identity", ["b"], ["alias"], name="alias"),
             helper.make_node("Constant", [], ["bias"], name="bias",
                              value=numpy_helper.from_array(np.full((2, 2), .5, np.float32))),
             helper.make_node("Add", ["alias", "bias"], ["logits"], name="final")]
    path = save(tmp_path, nodes, {"w1": np.arange(4, dtype=np.float32).reshape(2, 2),
                                  "w2": np.full((2, 2), 8, dtype=np.float32)}, external=True)
    parser = AuditedONNXParser()
    program = parser.parse(str(path))
    return onnx.load(path, load_external_data=False), parser, program, tmp_path


def test_complete_parse_has_serializable_mapping_and_binding_evidence(parsed):
    result = audit_parsed_model(*parsed)
    json.dumps(result, allow_nan=False)
    assert result["passed"] and result["node_count"] == 5
    assert result["value_count"] == 8
    assert result["initializer_count"] == 2
    assert result["constant_count"] == 1
    assert result["global_count"] == result["binding_count"] == 3
    assert result["binding_bytes"] == 48
    assert result["alias_count"] == 1
    assert result["return"]["onnx_name"] == "logits"
    assert all(len(row["sha256"]) == 64 for row in result["tensor_bindings"])
    assert parsed[0].graph.initializer[0].raw_data == b""


def test_parser_reuse_resets_observations_and_progress(parsed):
    _, parser, _, directory = parsed
    events = []
    parser.progress = events.append
    parser.parse(str(directory / "model.onnx"))
    assert len(parser.node_records) == 5
    assert events[0]["stage"] == "parse" and events[-1]["stage"] == "parsed"
    assert [row["node_index"] for row in events if row["stage"] == "translate_node"] == list(range(5))


@pytest.mark.parametrize("fault", ["missing_record", "missing_map", "missing_instruction",
                                    "shape", "dtype", "alias", "return", "global",
                                    "duplicate_global", "swapped_weights", "constant_content",
                                    "operand", "opcode", "attrs", "parameter"])
def test_incomplete_or_corrupt_translation_fails_closed(parsed, fault):
    _, parser, program, _ = parsed
    instructions = program.functions[0].blocks[0].instructions
    if fault == "missing_record":
        parser.node_records.pop()
    elif fault == "missing_map":
        del parser._value_map["a"]
    elif fault == "missing_instruction":
        instructions.pop(0)
    elif fault == "shape":
        parser._value_map["a"].shape = (4,)
    elif fault == "dtype":
        parser._value_map["a"].dtype = DataType.FLOAT64
    elif fault == "alias":
        parser._value_map["alias"] = parser._value_map["a"]
    elif fault == "return":
        instructions[-1].operands = [parser._value_map["a"]]
    elif fault == "global":
        program.global_values.pop()
    elif fault == "duplicate_global":
        program.global_values.append(program.global_values[0])
    elif fault == "swapped_weights":
        parser.initializers["w1"], parser.initializers["w2"] = (
            parser.initializers["w2"], parser.initializers["w1"])
    elif fault == "constant_content":
        parser.initializers["bias"] = np.zeros((2, 2), dtype=np.float32)
    elif fault == "operand":
        instructions[0].operands[1] = parser._value_map["w2"]
    elif fault == "opcode":
        instructions[0].opcode = OpCode.ADD
    elif fault == "attrs":
        instructions[0].attrs["k"] = 1
    elif fault == "parameter":
        program.functions[0].params = []
    with pytest.raises(AuditError):
        audit_parsed_model(*parsed)


def test_skip_handler_is_not_counted_as_success(parsed, monkeypatch):
    model, _, _, directory = parsed
    parser = AuditedONNXParser()
    monkeypatch.setattr(parser, "_handle_add", lambda node, inputs, outputs:
                        parser._define_outputs(outputs, inputs[0]))
    program = parser.parse(str(directory / "model.onnx"))
    with pytest.raises(AuditError, match="exactly one instruction"):
        audit_parsed_model(model, parser, program, directory)


def test_equal_shape_value_copy_does_not_prove_identity_alias(parsed):
    _, parser, _, _ = parsed
    source = parser._value_map["alias"]
    parser._value_map["alias"] = Value(source.name, source.dtype, shape=source.shape)
    with pytest.raises(AuditError, match="alias binding"):
        audit_parsed_model(*parsed)


@pytest.mark.parametrize("fault", ["missing", "truncated", "escape", "negative_offset",
                                    "bad_length", "duplicate_key", "changed_content"])
def test_external_data_errors_are_rejected_without_loading_second_weights(parsed, fault):
    model, _, _, directory = parsed
    tensor = model.graph.initializer[0]
    fields = {row.key: row for row in tensor.external_data}
    path = directory / "weights.data"
    if fault == "missing":
        path.unlink()
    elif fault == "truncated":
        path.write_bytes(b"\0" * 3)
    elif fault == "escape":
        fields["location"].value = "../outside.data"
    elif fault == "negative_offset":
        fields["offset"].value = "-1"
    elif fault == "bad_length":
        fields["length"].value = "15"
    elif fault == "duplicate_key":
        tensor.external_data.add(key="location", value="weights.data")
    elif fault == "changed_content":
        with path.open("r+b") as stream:
            stream.write(b"\xFF" * 4)
    with pytest.raises(AuditError):
        audit_parsed_model(*parsed)


def test_noncontiguous_equal_binding_hashes_logical_c_order(parsed):
    _, parser, _, _ = parsed
    parser.initializers["w1"] = np.asfortranarray(parser.initializers["w1"])
    assert not parser.initializers["w1"].flags.c_contiguous
    assert audit_parsed_model(*parsed)["passed"]


def test_multiple_graph_outputs_are_explicitly_rejected(tmp_path):
    outputs = [helper.make_tensor_value_info(name, T.FLOAT, [2, 2]) for name in ("x", "logits")]
    path = save(tmp_path, [helper.make_node("Identity", ["x"], ["logits"])], outputs=outputs)
    parser = AuditedONNXParser()
    program = parser.parse(str(path))
    with pytest.raises(AuditError, match="exactly one graph output"):
        audit_parsed_model(onnx.load(path), parser, program, tmp_path)


def test_scalar_constants_and_static_shape_cast_chain(tmp_path):
    nodes = [helper.make_node("Constant", [], ["dim"], value_int=2),
             helper.make_node("Cast", ["dim"], ["cast"], to=T.INT64),
             helper.make_node("Constant", [], ["axis"], value_ints=[0]),
             helper.make_node("Unsqueeze", ["cast", "axis"], ["vec"]),
             helper.make_node("Concat", ["vec", "vec"], ["shape"], axis=0),
             helper.make_node("Reshape", ["x", "shape"], ["reshaped"]),
             helper.make_node("Add", ["reshaped", "scalar"], ["logits"])]
    path = save(tmp_path, nodes, {"scalar": np.array(1, dtype=np.float32)})
    model = onnx.load(path)
    model.graph.value_info.append(helper.make_tensor_value_info("reshaped", T.FLOAT, [2, 2]))
    onnx.save(model, path)
    parser = AuditedONNXParser()
    program = parser.parse(str(path))
    result = audit_parsed_model(onnx.load(path), parser, program, tmp_path)
    assert result["binding_count"] == 3
    assert result["node_count"] == 7


def test_failed_translation_preserves_node_context(tmp_path, monkeypatch):
    path = save(tmp_path, [helper.make_node("Identity", ["x"], ["logits"], name="failed_node")])
    parser = AuditedONNXParser()

    def broken(*args):
        raise RuntimeError("injected")

    monkeypatch.setattr(parser, "_handle_identity", broken)
    with pytest.raises(RuntimeError, match="injected"):
        parser.parse(str(path))
    assert parser.current_context["node_name"] == "failed_node"
    assert parser.node_records[-1]["status"] == "failed"


def test_scalar_constant_signed_zero_is_preserved(tmp_path):
    path = save(tmp_path, [helper.make_node("Constant", [], ["zero"], value_float=-0.0),
                           helper.make_node("Add", ["x", "zero"], ["logits"])])
    parser = AuditedONNXParser()
    program = parser.parse(str(path))
    model = onnx.load(path)
    assert audit_parsed_model(model, parser, program, tmp_path)["passed"]
    program.functions[0].blocks[0].instructions[0].attrs["value"] = 0.0
    with pytest.raises(AuditError, match="Constant.*scalar instruction"):
        audit_parsed_model(model, parser, program, tmp_path)


def test_complete_decoder_fixture_uses_production_parser_and_audit(tmp_path):
    from tests.test_qwen3_full_structure import qwen3_fixture

    model, _ = qwen3_fixture()
    path = tmp_path / "decoder.onnx"
    onnx.save_model(model, path, save_as_external_data=True,
                    all_tensors_to_one_file=True, location="decoder.data", size_threshold=0)
    original = onnx.load(path, load_external_data=False)
    parser = AuditedONNXParser()
    program = parser.parse(str(path))
    result = audit_parsed_model(original, parser, program, tmp_path)
    assert result["passed"] and result["initializer_count"] == 26
    assert result["node_count"] == len(original.graph.node)
    assert result["return"]["shape"] == [1, 3, 9]
    shared = parser._value_map["model.lm_head.weight"]
    consumers = [inst for inst in program.functions[0].blocks[0].instructions
                 if any(operand is shared for operand in inst.operands)]
    assert sorted(inst.opcode.value for inst in consumers) == ["gather", "transpose"]


@pytest.mark.parametrize("operation", ["Reshape", "Slice"])
def test_parser_control_evaluation_cannot_certify_its_own_wrong_results(tmp_path, operation):
    from scratchv.analysis.ir_verifier import verify_ir

    if operation == "Reshape":
        nodes = [helper.make_node("Reshape", ["x", "shape"], ["logits"])]
        arrays = {"shape": np.array([2, 2], np.int64)}
        shape = [2, 2]
        wrong = {"shape": (1, 4)}
    else:
        nodes = [helper.make_node("Slice", ["x", "starts", "ends", "axes"], ["logits"])]
        arrays = {name: np.array(value, np.int64) for name, value in
                  {"starts": [0], "ends": [1], "axes": [0]}.items()}
        shape = [1, 2]
        # The wrong row has the SAME output shape: metadata cannot catch this.
        wrong = {"starts": (1,), "ends": (2,)}
    path = save(tmp_path, nodes, arrays,
                outputs=[helper.make_tensor_value_info("logits", T.FLOAT, shape)])
    parser = AuditedONNXParser()
    correct = parser._constant_ints
    parser._constant_ints = lambda value: wrong[value.name] if value.name in wrong else correct(value)
    program = parser.parse(str(path))
    assert verify_ir(program) == (True, [])
    with pytest.raises(AuditError, match="IR attributes mismatch"):
        audit_parsed_model(onnx.load(path), parser, program, tmp_path)


@pytest.fixture
def qwen_parsed(tmp_path):
    from tests.test_qwen3_full_structure import qwen3_fixture

    model, _ = qwen3_fixture()
    path = tmp_path / "qwen.onnx"
    onnx.save_model(model, path)
    parser = AuditedONNXParser()
    program = parser.parse(str(path))
    return model, parser, program, tmp_path


@pytest.mark.parametrize("fault", ["transpose", "epsilon"])
def test_parser_attribute_reader_cannot_certify_its_own_wrong_results(tmp_path, monkeypatch, fault):
    from scratchv.analysis.ir_verifier import verify_ir
    from tests.test_qwen3_full_structure import qwen3_fixture

    model, _ = qwen3_fixture()
    original_bytes = model.SerializeToString()
    epsilon_name = next(node.input[1] for node in model.graph.node
                        if list(node.output) == ["layer_0_input_norm_eps"])
    path = tmp_path / "qwen.onnx"
    onnx.save(model, path)
    correct = ONNXParser._attributes

    def wrong_reader(node):
        attrs = correct(node)
        if fault == "transpose" and list(node.output) == ["layer_0_q_weight"]:
            attrs["perm"] = [0, 1]  # Square Q projection: output shape stays unchanged.
        if fault == "epsilon" and list(node.output) == [epsilon_name]:
            attrs["value"] = numpy_helper.from_array(np.array(1e-3, dtype=np.float32))
        return attrs

    monkeypatch.setattr(ONNXParser, "_attributes", staticmethod(wrong_reader))
    parser = AuditedONNXParser()
    program = parser.parse(str(path))
    assert verify_ir(program) == (True, [])
    expected = "IR attributes mismatch" if fault == "transpose" else "content SHA256 mismatch"
    with pytest.raises(AuditError, match=expected):
        audit_parsed_model(model, parser, program, tmp_path)
    assert model.SerializeToString() == original_bytes


@pytest.mark.parametrize("parameter", [0, 1])
@pytest.mark.parametrize("fault", ["name", "constant"])
def test_qwen_parameter_binding_contract_is_checked(qwen_parsed, parameter, fault):
    from scratchv.analysis.ir_verifier import verify_ir

    _, _, program, _ = qwen_parsed
    value = program.functions[0].params[parameter]
    if fault == "name":
        value.name = "renamed_" + value.name
    else:
        value.is_constant = True
        value.const_value = 0
    assert verify_ir(program) == (True, [])
    expected = "parameter names" if fault == "name" else "must not be constants"
    with pytest.raises(AuditError, match=expected):
        audit_parsed_model(*qwen_parsed)


@pytest.mark.parametrize("source", ["initializer", "Constant"])
@pytest.mark.parametrize("fault", ["shape", "constant_flag", "constant_value"])
def test_auxiliary_scalar_load_result_metadata_is_checked(tmp_path, source, fault):
    nodes = [helper.make_node("Add", ["x", "scalar"], ["logits"])]
    arrays = {}
    if source == "Constant":
        nodes.insert(0, helper.make_node("Constant", [], ["scalar"], value_float=1.0))
    else:
        arrays["scalar"] = np.array(1.0, dtype=np.float32)
    path = save(tmp_path, nodes, arrays)
    parser = AuditedONNXParser()
    program = parser.parse(str(path))
    scalar_load = program.functions[0].blocks[0].instructions[0]
    assert scalar_load.opcode is OpCode.LOAD_CONST
    assert scalar_load.dest is not parser._value_map["scalar"]
    if fault == "shape":
        scalar_load.dest.shape = (2,)
    elif fault == "constant_flag":
        scalar_load.dest.is_constant = False
    else:
        scalar_load.dest.const_value = 2.0
    with pytest.raises(AuditError, match="scalar instruction metadata mismatch"):
        audit_parsed_model(onnx.load(path), parser, program, tmp_path)


@pytest.mark.parametrize("binding", ["initializer", "Constant"])
def test_qwen_bindings_are_not_silently_coerced_to_numpy(qwen_parsed, binding):
    _, parser, _, _ = qwen_parsed
    name = "positions" if binding == "initializer" else next(
        name for name, value in parser.initializers.items()
        if name.startswith("constant_") and value.dtype == np.dtype("int64") and value.ndim == 1)
    original = parser.initializers[name]
    parser.initializers[name] = original.tolist()
    # Coercion would give the exact same bytes, shape and dtype, but execution
    # requires the original binding object to already be a NumPy ndarray.
    converted = np.asarray(parser.initializers[name])
    assert converted.dtype == original.dtype and converted.shape == original.shape
    np.testing.assert_array_equal(converted, original)
    with pytest.raises(AuditError, match="binding must be a NumPy array"):
        audit_parsed_model(*qwen_parsed)


def test_qwen_read_only_ndarray_bindings_remain_valid(qwen_parsed):
    _, parser, _, _ = qwen_parsed
    for array in parser.initializers.values():
        array.setflags(write=False)
    assert audit_parsed_model(*qwen_parsed)["passed"]
