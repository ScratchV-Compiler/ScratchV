"""ONNX -> shared IR -> interpreter comparisons against ONNX Runtime."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T, helper, numpy_helper

from scratchv.frontend.onnx_parser import ONNXParser, ONNXParseError
from scratchv.verification.ir_interpreter import IRInterpreter

ort = pytest.importorskip("onnxruntime")
ROOT = Path(__file__).resolve().parents[1]


def save(tmp_path, nodes, inputs, output, arrays, opset=17):
    graph = helper.make_graph(nodes, "test", inputs, [output],
                              [numpy_helper.from_array(a, n) for n, a in arrays.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path = tmp_path / "test.onnx"
    onnx.save(model, path)
    return path


def compare(path, feed):
    parser = ONNXParser()
    program = parser.parse(str(path))
    actual = IRInterpreter(program).run(feed, initializers=parser.initializers).return_value
    expected = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, feed)[0]
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
    return parser, program


@pytest.mark.parametrize("left,right", [((2, 3), (3, 4)), ((1, 5, 3), (3, 4)),
                                       ((1, 2, 5, 3), (1, 2, 3, 4))])
def test_matmul_preserves_batched_semantics(tmp_path, left, right):
    rng = np.random.default_rng(9)
    x = rng.normal(size=left).astype("float32")
    w = rng.normal(size=right).astype("float32")
    path = save(tmp_path, [helper.make_node("MatMul", ["x", "w"], ["y"])],
                [helper.make_tensor_value_info("x", T.FLOAT, left)],
                helper.make_tensor_value_info("y", T.FLOAT, np.matmul(x, w).shape), {"w": w})
    compare(path, {"x": x})


def test_reshape_reads_values_instead_of_shape_tensor_dimensions(tmp_path):
    x = np.arange(6, dtype="float32").reshape(2, 3)
    path = save(tmp_path, [helper.make_node("Reshape", ["x", "shape"], ["y"])],
                [helper.make_tensor_value_info("x", T.FLOAT, [2, 3])],
                helper.make_tensor_value_info("y", T.FLOAT, [3, 2]),
                {"shape": np.array([3, 2], dtype="int64")})
    compare(path, {"x": x})


def test_external_weights_are_bound_from_the_model_directory(tmp_path):
    x = np.arange(6, dtype="float32").reshape(2, 3)
    path = save(tmp_path, [helper.make_node("MatMul", ["x", "w"], ["y"])],
                [helper.make_tensor_value_info("x", T.FLOAT, [2, 3])],
                helper.make_tensor_value_info("y", T.FLOAT, [2, 2]),
                {"w": np.arange(6, dtype="float32").reshape(3, 2)})
    model = onnx.load(path)
    onnx.save_model(model, path, save_as_external_data=True,
                    all_tensors_to_one_file=True, location="weights.data", size_threshold=0)
    compare(path, {"x": x})


def test_single_element_vector_is_not_a_scalar(tmp_path):
    path = save(tmp_path, [], [], helper.make_tensor_value_info("weight", T.FLOAT, [1]),
                {"weight": np.array([7], dtype="float32")})
    parser, program = compare(path, {})
    assert program.global_values[0].shape == (1,)
    assert not program.global_values[0].is_constant
    assert parser.initializers["weight"].shape == (1,)
    # Reusing one parser must not accumulate functions or definitions.
    second = parser.parse(str(path))
    assert len(second.functions) == len(second.global_values) == 1


def test_slice_steps_without_axes_input(tmp_path):
    path = save(tmp_path, [helper.make_node("Slice", ["x", "start", "end", "", "step"], ["y"])],
                [helper.make_tensor_value_info("x", T.FLOAT, [10])],
                helper.make_tensor_value_info("y", T.FLOAT, [4]),
                {"start": np.array([1], dtype="int64"), "end": np.array([8], dtype="int64"),
                 "step": np.array([2], dtype="int64")})
    compare(path, {"x": np.arange(10, dtype="float32")})


@pytest.mark.parametrize("noop", [0, 1])
def test_reducemean_opset18_empty_axes(tmp_path, noop):
    shape = [2, 3] if noop else [1, 1]
    path = save(tmp_path, [helper.make_node("ReduceMean", ["x", "axes"], ["y"],
                                          noop_with_empty_axes=noop)],
                [helper.make_tensor_value_info("x", T.FLOAT, [2, 3])],
                helper.make_tensor_value_info("y", T.FLOAT, shape),
                {"axes": np.array([], dtype="int64")}, opset=18)
    compare(path, {"x": np.arange(6, dtype="float32").reshape(2, 3)})


def test_dynamic_shape_input_fails_explicitly(tmp_path):
    path = save(tmp_path, [helper.make_node("Reshape", ["x", "shape"], ["y"])],
                [helper.make_tensor_value_info("x", T.FLOAT, [2, 3]),
                 helper.make_tensor_value_info("shape", T.INT64, [2])],
                helper.make_tensor_value_info("y", T.FLOAT, [3, 2]), {})
    with pytest.raises(ONNXParseError, match="constant integer vector"):
        ONNXParser().parse(str(path))


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location(
        "w1_transformer_probe", ROOT / "probes/w1_tiny_transformer/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_two_layer_graph_reaches_interpreter(tmp_path, probe, seed):
    model, _ = probe.build(probe.Config())
    path = tmp_path / "tiny.onnx"
    onnx.save(model, path)
    parser, program = compare(path, {
        "input_ids": np.random.default_rng(seed).integers(0, 128, (1, 256), dtype="int64"),
        "attention_mask": probe.causal_mask(probe.Config()),
    })
    assert len(program.global_values) == len(parser.initializers)
    assert all(instr.dest.shape for block in program.functions[0].blocks
               for instr in block.instructions if instr.dest and not instr.dest.is_constant)


@pytest.mark.parametrize("failure", ["parse", "execute", "numeric"])
def test_probe_gate_rejects_broken_ir_path(tmp_path, probe, monkeypatch, failure):
    if failure == "parse":
        monkeypatch.delattr(ONNXParser, "_handle_gather")
    else:
        original = IRInterpreter.run
        def broken(self, *args, **kwargs):
            if failure == "execute":
                raise RuntimeError("injected interpreter failure")
            result = original(self, *args, **kwargs)
            result.return_value[...] += np.float32(0.1)
            return result
        monkeypatch.setattr(IRInterpreter, "run", broken)
    assert probe.main(["--output-dir", str(tmp_path)]) == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["model_ok"] and not report["passed"]


def test_probe_rejects_lfs_pointer(tmp_path, probe):
    path = tmp_path / "pointer.onnx"
    path.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(SystemExit) as error:
        probe.main(["--model", str(path), "--output-dir", str(tmp_path / "out")])
    assert error.value.code == 2
