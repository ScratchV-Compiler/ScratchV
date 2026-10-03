"""Negative gate tests; the real integration is exercised by the probe CLI."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from probes.w2_runtime_model import run
from scratchv.runtime.llm_inputs import prepare_inputs


def logits():
    return np.zeros((1, 256, 8), dtype=np.float32)


def test_compare_checks_all_positions_and_reports_valid_and_padding():
    expected, actual = logits(), logits()
    actual[0, 2, 4] = np.float32(2e-6)
    actual[0, 255, 7] = np.float32(3e-6)
    result = run.compare_logits(actual, expected, 3, vocab_size=8)
    assert result["passed"]
    assert result["elements"] == 2048
    assert result["worst_index"] == [0, 255, 7]
    assert result["valid_max_abs"] == pytest.approx(2e-6)
    assert result["padding_max_abs"] == pytest.approx(3e-6)


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "inf", "masked", "list"])
@pytest.mark.parametrize("side", ["actual", "reference"])
def test_bad_logits_never_pass(kind, side):
    array = logits()
    if kind == "shape":
        array = array[:, :-1]
    elif kind == "dtype":
        array = array.astype(np.float64)
    elif kind in ("nan", "inf"):
        array[0, 255, 0] = float(kind)
    elif kind == "masked":
        array = np.ma.array(array)
    else:
        array = array.tolist()
    actual, expected = (array, logits()) if side == "actual" else (logits(), array)
    with pytest.raises((ValueError, TypeError)):
        run.compare_logits(actual, expected, 17, vocab_size=8)


def test_strict_threshold_is_not_allclose_relative_tolerance():
    expected, actual = logits(), logits()
    expected[0, 0, 0] = 100.0
    actual[0, 0, 0] = 100.001
    assert not run.compare_logits(actual, expected, 1, vocab_size=8)["passed"]
    expected.fill(0)
    actual.fill(0)
    actual[0, 255, 0] = np.nextafter(np.float32(run.ATOL), np.float32(np.inf))
    assert not run.compare_logits(actual, expected, 1, vocab_size=8)["passed"]


def test_original_ids_and_pad_inside_prompt_remain_valid():
    ids = [151644, 123, 151643, 151645]
    prepared = prepare_inputs(ids, pad_token_id=151643)
    feed = run.checked_inputs(ids, prepared, 151643)
    assert prepared.valid_length == 4
    assert feed["input_ids"][0, :4].tolist() == ids
    assert feed["attention_mask"][0, 0, 3, 2] == 0
    assert feed["attention_mask"][0, 0, 255, 4] < 0


@pytest.mark.parametrize("change", ["ids", "future", "padding", "dtype", "length"])
def test_input_oracle_rejects_mapping_and_wrong_mask(change):
    ids = [151644, 19, 151643, 20]
    prepared = prepare_inputs(ids, pad_token_id=151643)
    if change == "ids":
        prepared.input_ids[0, 0] %= 128
    elif change == "future":
        prepared.attention_mask[0, 0, 0, 1] = 0
    elif change == "padding":
        prepared.attention_mask[0, 0, 255, 255] = 0
    elif change == "dtype":
        prepared = SimpleNamespace(valid_length=4, as_feed=lambda: {
            "input_ids": np.zeros((1, 256), dtype=np.float32),
            "attention_mask": np.zeros((1, 1, 256, 256), dtype=np.float32)})
    else:
        prepared = SimpleNamespace(valid_length=3, as_feed=prepared.as_feed)
    with pytest.raises(ValueError):
        run.checked_inputs(ids, prepared, 151643)


class Tokenizer:
    def validate_token_ids(self, ids):
        if any(i >= 7 for i in ids):
            raise ValueError("not in tokenizer vocabulary")

    def decode(self, ids, **kwargs):
        return f"token:{ids[0]}"


def test_sampling_uses_last_valid_position_and_official_decode():
    array = logits()
    array[0, 2, 4] = 2
    array[0, 255, 6] = 100
    sample = run.sample_checked(array, array, 3, Tokenizer(), [Tokenizer(), Tokenizer()], vocab_size=8)
    assert sample["id"] == 4 and sample["position"] == 2
    assert sample["decoded"] == "token:4" and sample["reference_top2_margin"] == 2


def test_sampling_rejects_wrong_host_selected_row(monkeypatch):
    array = logits()
    array[0, 2, 4] = 2
    monkeypatch.setattr(run, "last_valid_logits", lambda a, *args, **kw: a[0, -1].copy())
    with pytest.raises(ValueError, match="wrong logits row"):
        run.sample_checked(array, array, 3, Tokenizer(), [Tokenizer()], vocab_size=8)


def test_sampling_rejects_argmax_drift_even_with_tiny_numeric_error():
    actual, expected = logits(), logits()
    actual[0, 2, 5] = 1e-7
    expected[0, 2, 4] = 1e-7
    assert run.compare_logits(actual, expected, 3, vocab_size=8)["passed"]
    with pytest.raises(ValueError, match="Greedy ID differs"):
        run.sample_checked(actual, expected, 3, Tokenizer(), [Tokenizer()], vocab_size=8)


def test_sampling_does_not_mask_undefined_model_vocabulary_rows():
    array = logits()
    array[0, 0, 7] = 3
    with pytest.raises(ValueError, match="not in tokenizer vocabulary"):
        run.sample_checked(array, array, 1, Tokenizer(), [Tokenizer()], vocab_size=8)


def test_sampling_rejects_different_official_decode():
    reference = SimpleNamespace(decode=lambda *a, **k: "other")
    with pytest.raises(ValueError, match="decode differs"):
        run.sample_checked(logits(), logits(), 1, Tokenizer(), [reference], vocab_size=8)


@pytest.mark.parametrize("failure", ["report.md", "report.html", "close", "replace"])
@pytest.mark.parametrize("primary", [False, True])
def test_report_errors_cannot_leave_success(tmp_path, monkeypatch, failure, primary):
    original_write, original_replace = Path.write_text, Path.replace

    def write(path, content, *a, **kw):
        if path.name == failure:
            raise PermissionError("blocked")
        result = original_write(path, content, *a, **kw)
        if path.name == ".report.json.tmp" and failure == "close":
            raise OSError("close failed after all bytes written")
        return result

    def replace(path, dest):
        if failure == "replace" and Path(dest).name == "report.json":
            raise PermissionError("cannot publish")
        return original_replace(path, dest)

    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(Path, "replace", replace)
    report = {"passed": not primary, "stage": "model" if primary else "complete"}
    if primary:
        report["error"] = "primary model failure"
    run.save_reports(tmp_path, report)
    assert not report["passed"]
    assert report["stage"] == ("model" if primary else "report-write")
    if primary:
        assert report["error"] == "primary model failure"
    if (tmp_path / "report.json").exists():
        assert not json.loads((tmp_path / "report.json").read_text())["passed"]
    else:
        assert failure in ("close", "replace")
    assert not (tmp_path / ".report.json.tmp").exists()
    for name in ("report.md", "report.html"):
        if name != failure:
            assert "FAIL" in (tmp_path / name).read_text()


def test_main_preserves_existing_evidence(tmp_path):
    existing = tmp_path / "report.json"
    existing.write_text("existing evidence")
    with pytest.raises(SystemExit):
        run.main(["--tokenizer-dir", str(tmp_path), "--output-dir", str(tmp_path)])
    assert existing.read_text() == "existing evidence"


def test_main_fails_with_original_stage_when_dependencies_are_unavailable(tmp_path, monkeypatch):
    def unavailable(directory, out, report):
        report["stage"] = "environment"
        raise RuntimeError("pinned dependency unavailable")

    monkeypatch.setattr(run, "run_probe", unavailable)
    assert run.main(["--tokenizer-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")]) == 1
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert not report["passed"] and report["stage"] == "environment"
    assert "pinned dependency unavailable" in report["error"]
    assert "probes/w2_runtime_model/run.py" in report["source_sha256"]


def test_no_git_source_archive_does_not_block_probe_dispatch(tmp_path, monkeypatch):
    from probes.w2_runtime import run as runtime
    # ROOT has no .git entry. Even if this directory lives under another Git
    # checkout, the helper must not borrow that unrelated checkout's identity.
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    reached = []

    def stop_at_assets(directory, out, report):
        reached.append(True)
        assert report["checkout"]["head"] is None
        assert "no .git" in report["checkout"]["unavailable"]
        assert len(report["source_sha256"]) >= 32
        report["stage"] = "tokenizer"
        raise ValueError("intentional stop before the real model")

    monkeypatch.setattr(run, "run_probe", stop_at_assets)
    assert run.main(["--tokenizer-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")]) == 1
    assert reached == [True]
    saved = json.loads((tmp_path / "out/report.json").read_text())
    assert not saved["passed"] and saved["stage"] == "tokenizer"
    assert saved["checkout"]["head"] is None


def abi_model():
    import onnx
    from onnx import helper as h
    # Shape-only toy graph for schema tests; it is never an integration result.
    shape = h.make_tensor("shape", onnx.TensorProto.INT64, [3], [1, 256, 151936])
    graph = h.make_graph([h.make_node("ConstantOfShape", ["shape"], ["logits"])], "schema", [
        h.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, [1, 256]),
        h.make_tensor_value_info("attention_mask", onnx.TensorProto.FLOAT, [1, 1, 256, 256]),
    ], [h.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 256, 151936])], [shape])
    return h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=10)


def test_onnx_contract_accepts_static_full_vocabulary_schema():
    run.check_onnx_contract(abi_model())


@pytest.mark.parametrize("change", ["vocab", "sequence", "dtype", "input_name", "extra_output", "opset"])
def test_onnx_contract_rejects_silent_exporter_changes(change):
    import onnx
    model = abi_model()
    if change == "vocab":
        model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 128
    elif change == "sequence":
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 17
    elif change == "dtype":
        model.graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.INT32
    elif change == "input_name":
        model.graph.input[0].name = "remapped_tokens"
    elif change == "extra_output":
        model.graph.output.append(model.graph.input[0])
    else:
        model.opset_import[0].version = 19
    with pytest.raises(Exception):
        run.check_onnx_contract(model)
