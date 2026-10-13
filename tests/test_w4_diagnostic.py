"""Dependency-slice correctness with real ONNX/ORT and borrowed tensor data."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from probes.w4_qwen3_full import diagnostic as diagnostic
from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, OpCode, Value
from scratchv.verification.ir_interpreter import IRInterpreter


def ir_graph():
    x, unused_input = Value("x", shape=(2, 3)), Value("unused_input", shape=(2, 3))
    weight, unused_weight = Value("weight", shape=(2, 3)), Value("unused_weight", shape=(2, 3))
    builder = IRBuilder()
    builder.new_function("main", [x, unused_input])
    builder.new_block("entry")
    builder.program.global_values.extend([weight, unused_weight])
    target = builder.add(x, weight)
    builder.neg(unused_input)
    builder.ret(builder.mul(target, unused_weight))
    arrays = {"weight": np.arange(6, dtype=np.float32).reshape(2, 3),
              "unused_weight": np.full((2, 3), 7, np.float32)}
    return builder, target, arrays


def test_ir_slice_keeps_exact_dependencies_and_borrowed_readonly_weights():
    builder, target, arrays = ir_graph()
    arrays["weight"].setflags(write=False)
    before = copy.deepcopy(builder.program)
    sliced, selected = diagnostic.slice_ir(builder.program, arrays, target.name)
    assert builder.program.dump() == before.dump()
    assert [v.name for v in sliced.functions[0].params] == ["x", "unused_input"]
    assert [v.name for v in sliced.global_values] == ["weight"]
    assert selected == {"weight": arrays["weight"]}
    assert selected["weight"] is arrays["weight"] and not selected["weight"].flags.writeable
    instructions = sliced.functions[0].blocks[0].instructions
    assert [i.opcode for i in instructions] == [OpCode.ADD, OpCode.RETURN]
    assert instructions[-1].operands[0].name == target.name
    assert instructions[0] is not builder.current_block.instructions[0]
    feed = {"x": np.ones((2, 3), np.float32), "unused_input": np.zeros((2, 3), np.float32)}
    result = IRInterpreter(sliced).run(feed, initializers=selected).return_value
    np.testing.assert_array_equal(result, feed["x"] + arrays["weight"])
    generated = TensorCCodegen(sliced, selected, constant_storage="external").generate()
    assert generated.output.shape == (2, 3)
    assert [w.name for w in generated.external_weights] == ["weight"]


def test_ir_slice_preserves_inline_scalar_literals():
    x = Value("x", shape=(2,))
    builder = IRBuilder()
    builder.new_function("main", [x])
    builder.new_block("entry")
    target = builder.mul(x, builder.make_const(0.5))
    builder.ret(builder.neg(target))
    sliced, bindings = diagnostic.slice_ir(builder.program, {}, target.name)
    result = IRInterpreter(sliced).run({"x": np.array([3, -4], np.float32)}).return_value
    np.testing.assert_array_equal(result, [1.5, -2])
    artifact = TensorCCodegen(sliced, bindings, constant_storage="external").generate()
    assert len(artifact.external_weights) == 1
    assert artifact.external_weights[0].shape == ()


@pytest.mark.parametrize("failure", ["missing", "integer", "duplicate", "undefined", "control_flow", "phi", "early_return"])
def test_ir_slice_rejects_ambiguous_or_unsupported_dependency_graphs(failure):
    builder, target, arrays = ir_graph()
    name = target.name
    if failure == "missing":
        name = "not_a_value"
    elif failure == "integer":
        target.dtype = DataType.INT64
    elif failure == "duplicate":
        builder.current_block.instructions[1].dest.name = target.name
    elif failure == "undefined":
        builder.current_block.instructions[0].operands[0] = Value("missing", shape=(2, 3))
    elif failure == "control_flow":
        builder.current_block.instructions[1].target = "other_block"
    elif failure == "phi":
        builder.current_block.phi_nodes.append(builder.current_block.instructions[0])
    elif failure == "early_return":
        builder.current_block.instructions.insert(1, copy.deepcopy(builder.current_block.instructions[-1]))
    with pytest.raises(ValueError):
        diagnostic.slice_ir(builder.program, arrays, name)


def _onnx_graph(tmp_path):
    onnx = pytest.importorskip("onnx")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    values = np.arange(8, dtype=np.float32).reshape(2, 1, 2, 2) / 4
    weight_path = model_dir / "weights.data"
    weight_path.write_bytes(values.tobytes())
    initializers = []
    for index, name in enumerate(("weight", "unused_weight")):
        tensor = onnx.numpy_helper.from_array(values[index], name)
        onnx.external_data_helper.set_external_data(tensor, location="weights.data", offset=index * 16, length=16)
        tensor.ClearField("raw_data")
        initializers.append(tensor)
    info = lambda name: onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 2, 2])
    graph = onnx.helper.make_graph([
        onnx.helper.make_node("Add", ["x", "weight"], ["checkpoint"], name="arbitrary-add"),
        onnx.helper.make_node("Neg", ["unused_input"], ["unneeded"], name="unrelated"),
        onnx.helper.make_node("Mul", ["checkpoint", "unused_weight"], ["tail"], name="later-layer"),
    ], "synthetic-unit-graph", [info("x"), info("unused_input")], [info("tail")],
        initializer=initializers, value_info=[info("checkpoint"), info("unneeded")])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    model.ir_version = 10
    row = {"name": "layer_0.output", "onnx_name": "checkpoint", "shape": [1, 2, 2], "dtype": "float32"}
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return onnx, model, model_dir, output_dir, row, values


def test_onnx_slice_prunes_nodes_weights_and_runs_same_checkpoint_in_ort(tmp_path):
    pytest.importorskip("onnxruntime")
    onnx, model, model_dir, output_dir, row, values = _onnx_graph(tmp_path)
    before = model.SerializeToString()
    weight_hash = hashlib.sha256((model_dir / "weights.data").read_bytes()).hexdigest()
    metadata = output_dir / "checkpoint.onnx"
    evidence = diagnostic.slice_onnx(model, row, model_dir, metadata)
    assert model.SerializeToString() == before
    assert hashlib.sha256((model_dir / "weights.data").read_bytes()).hexdigest() == weight_hash
    derived = onnx.load(metadata, load_external_data=False)
    assert [node.op_type for node in derived.graph.node] == ["Add"]
    assert [w.name for w in derived.graph.initializer] == ["weight"]
    assert [v.name for v in derived.graph.input] == ["x", "unused_input"]
    assert [v.name for v in derived.graph.output] == ["checkpoint"]
    assert not derived.graph.initializer[0].HasField("raw_data")
    location = next(field.value for field in derived.graph.initializer[0].external_data if field.key == "location")
    assert (output_dir / location).resolve() == model_dir / "weights.data"
    assert set(path.name for path in output_dir.iterdir()) == {"checkpoint.onnx"}
    feed = {"x": np.full((1, 2, 2), 2, np.float32), "unused_input": np.zeros((1, 2, 2), np.float32)}
    profile = diagnostic.execute_ort(metadata, row, feed, output_dir / "ort.npy")
    assert profile["provider"] == "CPUExecutionProvider" and profile["graph_optimization"] == "disabled"
    np.testing.assert_array_equal(np.load(output_dir / "ort.npy", allow_pickle=False), feed["x"] + values[0])
    assert evidence["nodes"] == 1 and evidence["initializers"] == 1
    assert evidence["sha256"] == hashlib.sha256(metadata.read_bytes()).hexdigest()


@pytest.mark.parametrize("failure", ["missing_output", "duplicate_producer", "missing_dependency", "external_escape"])
def test_onnx_slice_rejects_invalid_dependencies_before_publishing(tmp_path, failure):
    onnx, model, model_dir, output_dir, row, _ = _onnx_graph(tmp_path)
    if failure == "missing_output":
        row["onnx_name"] = "missing"
    elif failure == "duplicate_producer":
        model.graph.node[1].output[0] = "checkpoint"
    elif failure == "missing_dependency":
        model.graph.node[0].input[0] = "missing"
    elif failure == "external_escape":
        next(entry for entry in model.graph.initializer[0].external_data if entry.key == "location").value = "../outside.data"
        (tmp_path / "outside.data").write_bytes(bytes(16))
    output = output_dir / "bad.onnx"
    with pytest.raises(ValueError):
        diagnostic.slice_onnx(model, row, model_dir, output)
    assert not output.exists()


def test_checkpoint_selection_uses_audited_bindings_not_layer_name_substrings():
    onnx = pytest.importorskip("onnx")
    nodes = [onnx.helper.make_node("Identity", ["x"], ["embed_out"], name="unrelated-embedding-node")]
    nodes += [onnx.helper.make_node("Identity", ["x"], [f"data_{i}"], name=f"arbitrary-node-{i}") for i in range(28)]
    nodes += [onnx.helper.make_node("Identity", ["x"], ["norm_out"], name="not-a-layer-name")]
    model = onnx.helper.make_model(onnx.helper.make_graph(nodes, "mapping-unit", [], []))
    audit = {"passed": True, "layer_count": 28, "embedding": "unrelated-embedding-node",
             "final_norm": "not-a-layer-name", "layers": [{"layer": i, "output": f"data_{i}"} for i in range(28)]}
    row = diagnostic.select_checkpoint(model, "layer_0.output", audit)
    assert row["onnx_name"] == "data_0" and row["shape"] == [1, 256, 1024]
    bad = {**audit, "passed": False}
    with pytest.raises(ValueError, match="complete ordered 28-layer"):
        diagnostic.select_checkpoint(model, "layer_0.output", bad)
    with pytest.raises(ValueError, match="Unknown"):
        diagnostic.select_checkpoint(model, "layer_28.output", audit)


@pytest.mark.parametrize("flag", ["--timeout", "--build-timeout"])
@pytest.mark.parametrize("value", ["0", "nan", "inf", "-1"])
def test_diagnostic_cli_rejects_unbounded_timeout_before_creating_output(tmp_path, flag, value):
    target = tmp_path / "report"
    with pytest.raises(SystemExit) as exc:
        diagnostic.main(["--model-dir", str(tmp_path), "--output-dir", str(target), flag, value])
    assert exc.value.code == 2
    assert not target.exists()


@pytest.mark.parametrize("policy,kernel_calls,flags", [
    ("sequential", True, []),
    ("sequential", False, ["--no-kernel-calls"]),
    ("blocked_fma", True, ["--matmul-policy", "blocked_fma", "--kernel-calls"]),
    ("blocked_fma", False, ["--matmul-policy", "blocked_fma", "--no-kernel-calls"]),
])
def test_diagnostic_codegen_policy_reaches_build_and_failed_build_never_accepts_gate(
        tmp_path, monkeypatch, policy, kernel_calls, flags):
    """Exercise actual slicing/codegen/bundle writing while stopping at toolchain.

    A tiny graph substitutes only fixed-asset loading/audit, never pretends to be
    full model numerical evidence. The simulated build failure must stay FAIL.
    """
    onnx = pytest.importorskip("onnx")
    from scratchv.frontend import onnx_parser

    x, w = Value("x", shape=(2, 3)), Value("weight", shape=(3, 2))
    builder = IRBuilder()
    builder.new_function("main", [x])
    builder.new_block("entry")
    builder.program.global_values.append(w)
    target = builder.matmul(x, w)
    builder.ret(target)
    parser = SimpleNamespace(
        _value_map={"checkpoint": target},
        initializers={"weight": np.arange(6, dtype=np.float32).reshape(3, 2)},
        parse=lambda *a, **kw: builder.program,
    )
    monkeypatch.setattr(diagnostic, "source_evidence", lambda: {})
    monkeypatch.setattr(diagnostic.assets, "verify_files", lambda path: [])
    monkeypatch.setattr(onnx, "load", lambda *a, **kw: object())
    monkeypatch.setattr(diagnostic, "audit_graph_structure", lambda model: {})
    monkeypatch.setattr(diagnostic, "select_checkpoint", lambda *a: {
        "onnx_name": "checkpoint", "shape": [2, 2], "name": "layer_0.output"})
    monkeypatch.setattr(onnx_parser, "ONNXParser", lambda: parser)
    monkeypatch.setattr(diagnostic, "discover_toolchain", lambda *a: object())
    generated = []

    def stopped_build(artifact, *a, **kw):
        generated.append(artifact)
        raise RuntimeError("intentional unit-test build stop")

    monkeypatch.setattr(diagnostic, "build_external", stopped_build)
    out = tmp_path / "result"
    result = diagnostic.main(["--model-dir", str(tmp_path), "--output-dir", str(out), *flags])
    assert result == 1
    assert len(generated) == 1
    artifact = generated[0]
    assert artifact.matmul_policy == policy and artifact.kernel_calls is kernel_calls
    assert ("__builtin_fmaf(" in artifact.source) is (policy == "blocked_fma")
    report = json.loads((out / "report.json").read_text())
    assert report["matmul_policy"] == report["codegen"]["matmul_policy"] == policy
    assert report["kernel_calls"] is report["codegen"]["kernel_calls"] is kernel_calls
    assert report["codegen"]["compile_flags"] == list(artifact.compile_flags)
    assert report["status"] == "FAIL" and not report["passed"]
    assert "intentional unit-test build stop" in report["error"]
    assert not report["numeric_gate_passed"] and not report["qemu_checkpoint_executed"]
    assert not report["full_qemu_forward_executed"] and not report["w4_team_accepted"]


def test_diagnostic_rejects_unknown_policy_before_output_creation(tmp_path):
    target = tmp_path / "report"
    with pytest.raises(SystemExit) as exc:
        diagnostic.main(["--model-dir", str(tmp_path), "--output-dir", str(target),
                         "--matmul-policy", "unknown"])
    assert exc.value.code == 2
    assert not target.exists()
