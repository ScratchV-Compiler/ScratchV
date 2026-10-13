"""Full-model worker contracts and real tiny ordinary/diagnostic execution."""
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T, helper as h, numpy_helper

from probes.w1_qwen3_export.run import make_inputs
from probes.w3_qwen3_full import worker


@pytest.mark.parametrize("length", [1, 17, 255, 256])
def test_inputs_accept_fixed_full_onnx_contract(tmp_path, length):
    original = make_inputs(length, 42)
    path = tmp_path / "inputs.npz"
    np.savez(path, **original)
    actual, valid_length = worker.load_inputs(path)
    assert valid_length == length
    assert all(np.array_equal(actual[key], original[key]) for key in original)


@pytest.mark.parametrize("change", ["ids_negative", "ids_large", "dtype", "mask_future",
                                   "mask_hole", "mask_infinite", "empty", "extra"])
def test_inputs_reject_invalid_contract(tmp_path, change):
    feed = make_inputs(17, 42)
    if change == "ids_negative":
        feed["input_ids"][0, 0] = -1
    elif change == "ids_large":
        feed["input_ids"][0, 0] = 151936
    elif change == "dtype":
        feed["input_ids"] = feed["input_ids"].astype(np.int32)
    elif change == "mask_future":
        feed["attention_mask"][0, 0, 0, 1] = 0
    elif change == "mask_hole":
        feed["attention_mask"][0, 0, 8, 1] = np.finfo(np.float32).min
    elif change == "mask_infinite":
        feed["attention_mask"][0, 0, 0, 1] = -np.inf
    elif change == "empty":
        feed["attention_mask"].fill(np.finfo(np.float32).min)
    else:
        feed["unexpected"] = np.ones(1)
    path = tmp_path / "inputs.npz"
    np.savez(path, **feed)
    with pytest.raises(ValueError):
        worker.load_inputs(path)


def _checkpoint_fixture():
    nodes = [h.make_node("Identity", ["x"], [f"value_{i}"], name=f"misleading_{27-i}")
             for i in range(30)]
    model = h.make_model(h.make_graph(nodes, "checkpoints", [], []))
    audit = {"passed": True, "layer_count": 28,
             "layers": [{"layer": i, "output": f"value_{i+1}"} for i in range(28)],
             "embedding": nodes[0].name, "final_norm": nodes[-1].name}
    return model, audit


def test_checkpoint_selection_uses_audited_dataflow_in_order():
    model, audit = _checkpoint_fixture()
    schema = worker.checkpoint_schema(model, audit)
    assert len(schema) == 30
    assert [row["onnx_name"] for row in schema] == [f"value_{i}" for i in range(30)]
    assert [row["layer"] for row in schema if "layer" in row] == list(range(28))
    assert schema[1]["name"] == "layer_0.output"
    assert all(row["shape"] == [1, 256, 1024] for row in schema)


@pytest.mark.parametrize("change", ["missing_layer", "audit_failed", "reordered", "missing_value",
                                   "duplicate_value", "duplicate_node"])
def test_checkpoint_selection_rejects_incomplete_or_ambiguous_coverage(change):
    model, audit = _checkpoint_fixture()
    if change == "missing_layer":
        audit["layers"].pop()
    elif change == "audit_failed":
        audit["passed"] = False
    elif change == "reordered":
        audit["layers"].reverse()
    elif change == "missing_value":
        audit["layers"][0]["output"] = "does_not_exist"
    elif change == "duplicate_value":
        audit["layers"][0]["output"] = "value_0"
    else:
        model.graph.node[1].name = audit["embedding"]
    with pytest.raises(ValueError):
        worker.checkpoint_schema(model, audit)


def test_atomic_save_refuses_to_replace_existing_evidence(tmp_path):
    path = tmp_path / "logits.npy"
    worker.save_array(path, np.ones((1, 2), np.float32))
    with pytest.raises(FileExistsError):
        worker.save_array(path, np.zeros((1, 2), np.float32))
    np.testing.assert_array_equal(np.load(path), np.ones((1, 2), np.float32))
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_archive_requires_exact_coverage(tmp_path):
    with pytest.raises(ValueError, match="omitted/duplicated"):
        worker.save_checkpoints(tmp_path, {}, [{"name": "missing"}], {"artifacts": {}})


def _numeric_fixture(tmp_path):
    """Real ONNX external scalar, thirty additions, independently run by ORT/IR."""
    source = tmp_path / "source"
    source.mkdir()
    nodes, schema = [], []
    previous = "x"
    for i in range(30):
        name = f"checkpoint_{i}"
        nodes.append(h.make_node("Add", [previous, "shift"], [name], name=f"add_{i}"))
        schema.append({"name": name, "onnx_name": name, "shape": [1, 2, 4],
                       "dtype": "float32", "sequence_axis": 1})
        previous = name
    nodes.append(h.make_node("Identity", [previous], ["logits"]))
    graph = h.make_graph(nodes, "tiny_worker", [h.make_tensor_value_info("x", T.FLOAT, [1, 2, 4])],
                         [h.make_tensor_value_info("logits", T.FLOAT, [1, 2, 4])],
                         [numpy_helper.from_array(np.asarray(0.25, np.float32), "shift")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10)
    onnx.save_model(model, str(source / "model.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="weights.data", size_threshold=0)
    metadata = onnx.load(str(source / "model.onnx"), load_external_data=False)
    return source, metadata, schema, {"x": np.arange(8, dtype=np.float32).reshape(1, 2, 4)}


def test_real_ort_ordinary_and_diagnostic_external_weights(tmp_path, monkeypatch):
    source, model, schema, feed = _numeric_fixture(tmp_path)
    out = tmp_path / "result"
    out.mkdir()
    original = (source / "model.onnx").read_bytes()
    monkeypatch.setattr(worker, "LOGITS_SHAPE", (1, 2, 4))
    report = {"artifacts": {}}
    worker.execute_ort(source, out, feed, schema, model, report, lambda _, operation: operation())
    np.testing.assert_array_equal(np.load(out / "logits.npy"), feed["x"] + 7.5)
    assert worker.exact_ordinary_diagnostic(out)["passed"]
    with np.load(out / "checkpoints.npz") as arrays:
        assert len(arrays.files) == 30
        np.testing.assert_array_equal(arrays["checkpoint_0"], feed["x"] + 0.25)
        np.testing.assert_array_equal(arrays["checkpoint_29"], feed["x"] + 7.5)
    assert (source / "model.onnx").read_bytes() == original
    assert report["artifacts"]["diagnostic_model"]["sha256"]


def test_real_ir_ordinary_and_observer_same_math(tmp_path, monkeypatch):
    source, model, schema, feed = _numeric_fixture(tmp_path)
    out = tmp_path / "result"
    out.mkdir()
    monkeypatch.setattr(worker, "LOGITS_SHAPE", (1, 2, 4))
    report = {"artifacts": {}}
    worker.execute_ir(source, out, feed, schema, model, report, lambda _, operation: operation())
    np.testing.assert_array_equal(np.load(out / "logits.npy"), feed["x"] + 7.5)
    assert worker.exact_ordinary_diagnostic(out)["passed"]
    assert set(report["executions"]) == {"ordinary", "diagnostic"}
    assert all(value["memory_stats"] for value in report["executions"].values())
    with np.load(out / "checkpoints.npz") as arrays:
        np.testing.assert_array_equal(arrays["checkpoint_12"], feed["x"] + 3.25)


def test_exact_consistency_rejects_real_diagnostic_change(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "LOGITS_SHAPE", (1, 2, 4))
    ordinary = np.zeros((1, 2, 4), np.float32)
    diagnostic = ordinary.copy()
    diagnostic[0, 1, 3] = 1e-7
    np.save(tmp_path / "logits.npy", ordinary)
    np.save(tmp_path / "diagnostic_logits.npy", diagnostic)
    with pytest.raises(ValueError, match="differ"):
        worker.exact_ordinary_diagnostic(tmp_path)


@pytest.mark.parametrize("interrupted", [False, True])
def test_initial_evidence_failure_has_atomic_fail_report(tmp_path, monkeypatch, interrupted):
    def fail():
        raise KeyboardInterrupt() if interrupted else RuntimeError("fingerprint unavailable")
    monkeypatch.setattr(worker, "source_evidence", fail)
    out = tmp_path / "run"
    code = worker.main(["--backend", "ir", "--model-dir", str(tmp_path / "model"),
                        "--input-file", str(tmp_path / "inputs.npz"), "--output-dir", str(out)])
    report = json.loads((out / "report.json").read_text())
    assert code == (130 if interrupted else 1)
    assert report["status"] == "FAIL" and not report["passed"]
    assert not report["full_ir_executed"] and not report["w3_exit_accepted"]
    assert report["stages_seconds"]["source_evidence"] >= 0
    assert (out / "report.md").exists() and (out / "report.html").exists()


def test_existing_output_is_never_reused(tmp_path):
    with pytest.raises(FileExistsError):
        worker.main(["--backend", "ort", "--model-dir", str(tmp_path),
                     "--input-file", str(tmp_path / "input.npz"), "--output-dir", str(tmp_path)])
