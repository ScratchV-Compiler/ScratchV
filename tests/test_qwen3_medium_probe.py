"""Fail-closed numerical evidence and independent ordinary/diagnostic paths."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("onnx")

from probes.w3_qwen3_medium import run as probe
from scratchv.verification.ir_interpreter import IRInterpreter


@pytest.mark.parametrize("kind", probe.GRAPH_KINDS)
@pytest.mark.parametrize("level", probe.LEVELS)
def test_one_failed_graph_or_optimization_makes_case_fail(kind, level):
    case = {"capture_preserves_logits": {"passed": True}, "attention_checks": [{"passed": True}],
            "ort": {key: {"passed": True} for key in probe.GRAPH_KINDS},
            "ir": {opt: {key: {"passed": True} for key in probe.GRAPH_KINDS} for opt in probe.LEVELS}}
    assert probe.case_passed(case)
    case["ir"][level][kind]["passed"] = False
    assert not probe.case_passed(case)


@pytest.mark.parametrize("kind", probe.GRAPH_KINDS)
@pytest.mark.parametrize("fault", ["value", "nonfinite", "shape", "exception"])
def test_ir_fault_is_failed_and_actual_trace_is_preserved(tmp_path, monkeypatch, kind, fault):
    expected = np.ones((1, 2, 3), np.float32)
    actual = expected.copy()
    if fault == "value":
        actual.flat[1] += 0.25
    elif fault == "nonfinite":
        actual.flat[1] = np.nan
    elif fault == "shape":
        actual = np.ones((1, 2, 4), np.float32)

    def execute(self, *args, **kwargs):
        assert kwargs["collect_memory_stats"] is True
        if fault == "exception":
            raise RuntimeError("injected execution failure")
        return SimpleNamespace(return_value=actual.ravel() if kind == "diagnostic" else actual,
                               executed_steps=5, memory_stats={"retained_values": 2})

    monkeypatch.setattr(IRInterpreter, "run", execute)
    report = {"trace_artifacts": []}
    result = probe.run_ir_variant(None, {}, {}, kind=kind, level="all",
        schema=[{"name": "logits", "shape": [1, 2, 3], "offset": 0, "size": 6}],
        names=["logits"], metadata={"logits": {"sequence_axis": 1}}, valid_length=1,
        reference={"logits": expected}, ort_reference={"logits": expected},
        out=tmp_path, case_name="fault", report=report)
    assert result["passed"] is False
    assert result["status"] in ("runtime_error", "numeric_failed")
    if fault in ("value", "nonfinite") or (fault == "shape" and kind == "normal"):
        assert len(report["trace_artifacts"]) == 1
        with np.load(tmp_path / result["trace"]["path"]) as data:
            np.testing.assert_array_equal(data["logits"], actual)
        assert not result["pytorch_comparison"]["passed"]
    if fault == "exception":
        assert "injected execution failure" in result["error"]


def test_missing_ort_reference_fails_even_when_ir_matches_pytorch(tmp_path, monkeypatch):
    expected = np.ones((1, 2, 3), np.float32)
    monkeypatch.setattr(IRInterpreter, "run", lambda *args, **kwargs: SimpleNamespace(
        return_value=expected.copy(), executed_steps=1, memory_stats={}))
    result = probe.run_ir_variant(None, {}, {}, kind="normal", level="none", schema=[],
        names=["logits"], metadata={"logits": {"sequence_axis": 1}}, valid_length=2,
        reference={"logits": expected}, ort_reference=None, out=tmp_path,
        case_name="missing", report={"trace_artifacts": []})
    assert result["pytorch_comparison"]["passed"]
    assert not result["passed"]
    assert result["ort_comparison"]["error"] == "ORT reference unavailable"


@pytest.mark.parametrize("query_index,passed", [(12, False), (100, True)])
def test_causal_isolation_observes_nonlogit_checkpoint_on_its_sequence_axis(tmp_path, query_index, passed):
    reference = {"layer_0.rope_q": np.ones((1, 4, 256, 16), np.float32),
                 "logits": np.ones((1, 256, 128), np.float32)}
    changed = {name: array.copy() for name, array in reference.items()}
    changed["layer_0.rope_q"][0, 0, query_index, 0] += 0.5
    for case_name, data in (("full_seed_0", reference), ("changed_future", changed)):
        folder = tmp_path / "traces" / case_name
        folder.mkdir(parents=True)
        np.savez(folder / "reference.npz", **data)
    rows = probe.isolation_checks(tmp_path, {"layer_0.rope_q": {"sequence_axis": 2},
                                              "logits": {"sequence_axis": 1}})
    row = next(row for row in rows if row["name"] == "causality" and row["backend"] == "pytorch")
    assert row["passed"] is passed
    assert row["comparison"]["first_divergence"] == (None if passed else "layer_0.rope_q")


def test_cli_exception_writes_failure_reports_and_refuses_overwrite(tmp_path, monkeypatch):
    import json

    def fail(out, report, seed):
        report["stage"] = "injected"
        raise RuntimeError("injected gate failure")

    monkeypatch.setattr(probe, "run_probe", fail)
    out = tmp_path / "evidence"
    assert probe.main(["--output-dir", str(out)]) == 1
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["status"] == "FAIL"
    assert report["error"] == "RuntimeError: injected gate failure"
    assert report["stage"] == "injected"
    assert (out / "report.md").is_file() and (out / "report.html").is_file()
    with pytest.raises(SystemExit) as caught:
        probe.main(["--output-dir", str(out)])
    assert caught.value.code == 2


def complete_numeric_report():
    return {
        "cases": [
            {"name": name, "valid_length": length, "passed": True,
             "ort": {graph: {"passed": True} for graph in probe.GRAPH_KINDS},
             "ir": {level: {graph: {"passed": True} for graph in probe.GRAPH_KINDS}
                    for level in probe.LEVELS}}
            for name, length in probe.CASE_IDENTITIES
        ],
        "invariants": [{"name": name, "backend": backend, "graph": graph, "passed": True}
                       for name, backend, graph in sorted(probe.INVARIANT_IDENTITIES)],
    }


@pytest.mark.parametrize("omission", ["case", "all_cases", "invariant", "all_invariants",
                                     "duplicate_case", "duplicate_invariant", "case_length", "ir_level", "ort_graph"])
def test_missing_or_duplicated_coverage_cannot_pass(omission):
    report = complete_numeric_report()
    probe.finalize_numeric_report(report)
    assert report["coverage_complete"] and report["passed"]
    if omission == "case":
        report["cases"].pop()
    elif omission == "all_cases":
        report["cases"] = []
    elif omission == "invariant":
        report["invariants"].pop()
    elif omission == "all_invariants":
        report["invariants"] = []
    elif omission == "duplicate_case":
        report["cases"][-1] = report["cases"][0]
    elif omission == "duplicate_invariant":
        report["invariants"][-1] = report["invariants"][0]
    elif omission == "case_length":
        report["cases"][0]["valid_length"] = 255
    elif omission == "ir_level":
        del report["cases"][0]["ir"]["all"]
    else:
        del report["cases"][0]["ort"]["normal"]
    probe.finalize_numeric_report(report)
    assert report["coverage_complete"] is False
    assert report["passed"] is False


def test_success_status_and_existing_empty_directory_contract(tmp_path, monkeypatch):
    import json

    def succeed(out, report, seed):
        report.update(complete_numeric_report())
        probe.finalize_numeric_report(report)

    monkeypatch.setattr(probe, "run_probe", succeed)
    existing = tmp_path / "already-created"
    existing.mkdir()
    with pytest.raises(SystemExit) as caught:
        probe.main(["--output-dir", str(existing)])
    assert caught.value.code == 2
    assert not list(existing.iterdir())
    out = tmp_path / "new-evidence"
    assert probe.main(["--output-dir", str(out)]) == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS" and report["passed"] is True
    assert report["coverage_complete"] is True
