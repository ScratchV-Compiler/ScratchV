"""The comparison must expose old failures without manufacturing improvements."""

import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
from benchmarks import bench_onnx_operators as bench
from benchmarks import onnx_operator_worker as worker
from benchmarks.onnx_operator_cases import build_cases
from benchmarks.onnx_operator_report import render_html, render_markdown
from scratchv.analysis.ir_verifier import verify_ir
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.verification.ir_interpreter import IRInterpreter


@pytest.fixture
def prepared(tmp_path):
    case = next(case for case in build_cases(tmp_path) if case["name"] == "add_float32")
    bench.prepare_references([case], tmp_path)
    assert not case.get("reference_error")
    return case


def execute(case, **overrides):
    arguments = dict(warmup=1, repeats=3, parser_cls=ONNXParser,
                     interpreter_cls=IRInterpreter, verifier=verify_ir)
    arguments.update(overrides)
    return worker.run_case(case, **arguments)


def test_fixed_fixture_coverage_and_exact_rounding_policy(tmp_path):
    cases = build_cases(tmp_path)
    standalone = [case for case in cases if case["group"] == "operators"]
    assert len(standalone) == 16
    assert {op for case in standalone for op in case["operators"]} == {
        "Gather", "Sqrt", "ReduceMean", "Transpose", "Concat", "Slice", "Unsqueeze", "Expand",
        "Abs", "Cast", "Constant", "Cos", "Identity", "Pow", "Reciprocal", "Sin"}
    rounding = next(case for case in cases if case["name"] == "scalar_fp32_add_chain")
    assert rounding["atol"] == rounding["rtol"] == 0


@pytest.mark.parametrize("actual,expected", [
    (np.array(7, "float32"), np.array([7], "float32")),
    (np.array([7], "float64"), np.array([7], "float32")),
    (np.array([2**60 + 1], "int64"), np.array([2**60], "int64")),
    (np.array([np.nan], "float32"), np.array([1], "float32")),
    (np.array(16777218, "float32"), np.array(16777216, "float32")),
])
def test_comparison_cannot_hide_rank_dtype_integer_or_fp32_errors(actual, expected):
    with pytest.raises(worker.NumericMismatch):
        worker.compare_output(actual, expected, 0, 0)


@pytest.mark.parametrize("wrong_call,samples", [(2, 0), (4, 1)])
def test_every_warmup_and_timed_result_is_checked(prepared, wrong_call, samples):
    class Unstable(IRInterpreter):
        calls = 0

        def run(self, *args, **kwargs):
            result = super().run(*args, **kwargs)
            Unstable.calls += 1
            if Unstable.calls == wrong_call:
                result.return_value[...] += np.float32(1)
            return result

    row = execute(prepared, interpreter_cls=Unstable)
    assert row["status"] == "NUMERIC_ERROR"
    assert row["max_abs_error"] > 0.9
    assert row["run_median_s"] is None
    assert len(row["run_samples_s"]) == samples


def test_execution_cannot_mutate_future_inputs(prepared):
    class Mutating(IRInterpreter):
        def run(self, inputs, **kwargs):
            result = super().run(inputs, **kwargs)
            inputs["x"][...] += 1
            return result

    row = execute(prepared, interpreter_cls=Mutating)
    assert row["status"] == "NUMERIC_ERROR" and "mutated" in row["error"]


@pytest.mark.parametrize("target", ["model", "expected", "initializers", "inputs"])
def test_artifact_changes_during_execution_cannot_pass(prepared, target):
    good = ONNXParser()
    program = good.parse(prepared["model"])

    class Tampering:
        initializers = good.initializers

        def parse(self, path):
            Path(prepared[target]).write_bytes(b"changed")
            return program

    row = execute(prepared, parser_cls=Tampering)
    assert row["status"] == "REFERENCE_ERROR"
    assert "fixture changed" in row["error"]


def test_preexisting_fixture_corruption_is_rejected(prepared):
    Path(prepared["expected"]).write_bytes(b"corrupted")
    assert execute(prepared)["status"] == "REFERENCE_ERROR"


def test_ort_failure_never_uses_current_ir_as_reference(tmp_path, monkeypatch):
    import onnxruntime as ort
    case = build_cases(tmp_path)[0]

    def fail(*args, **kwargs):
        raise RuntimeError("injected ORT failure")

    monkeypatch.setattr(ort, "InferenceSession", fail)
    bench.prepare_references([case], tmp_path)
    assert execute(case)["status"] == "REFERENCE_ERROR"
    assert "injected ORT failure" in case["reference_error"]


def test_new_parser_initializer_mismatch_is_not_hidden_by_common_weights():
    class Parser:
        initializers = {"weight": np.array([8], "float32")}

    with pytest.raises(ValueError, match="differs"):
        worker.bindings(Parser(), None, {"weight": np.array([7], "float32")})


def test_all_imported_submodules_must_belong_to_selected_snapshot(tmp_path, monkeypatch):
    module = ModuleType("scratchv.injected")
    module.__file__ = str(tmp_path / "injected.py")
    monkeypatch.setitem(sys.modules, "scratchv.injected", module)
    with pytest.raises(RuntimeError, match="mixed scratchv imports"):
        worker.assert_import_source(bench.ROOT)


def test_failed_round_cannot_retain_success_median():
    good = dict(worker.empty_row(), status="PASS", run_samples_s=[1.0], parse_samples_s=[0.5],
                run_median_s=1.0, parse_median_s=0.5)
    bad = dict(worker.empty_row(), status="NUMERIC_ERROR", error="wrong")
    merged = bench.merge_rounds([good, bad])
    assert merged["status"] == "NUMERIC_ERROR" and merged["run_median_s"] is None


def test_actual_git_baseline_and_dirty_snapshot_are_isolated(tmp_path):
    original_hash = worker.source_hash(bench.ROOT)
    selected = ["add_float32", "gather", "scalar_fp32_add_chain", "singleton_initializer", "constant"]
    args = ["--output-dir", str(tmp_path), "--warmup", "0", "--repeats", "1"]
    for name in selected:
        args += ["--case", name]
    assert bench.main(args) == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["baseline"]["commit"] == bench.BASELINE
    assert report["baseline"]["source_sha256"] != report["current"]["source_sha256"]
    assert worker.source_hash(bench.ROOT) == original_hash
    rows = {row["name"]: row for row in report["results"]}
    assert rows["add_float32"]["before"]["status"] == "PASS"
    assert rows["gather"]["before"]["status"] == "UNSUPPORTED"
    assert rows["constant"]["before"]["status"] == "UNSUPPORTED"
    assert rows["singleton_initializer"]["before"]["status"] == "EXECUTION_ERROR"
    assert rows["scalar_fp32_add_chain"]["before"]["status"] == "NUMERIC_ERROR"
    assert rows["scalar_fp32_add_chain"]["before"]["max_abs_error"] == 2
    assert all(row["after"]["status"] == "PASS" for row in rows.values())
    assert all(len(row["after"]["run_samples_s"]) == 2 for row in rows.values())
    assert (tmp_path / "report.html").is_file() and (tmp_path / "report.md").is_file()


def test_worker_timeout_is_visible_in_new_report(tmp_path, monkeypatch):
    actual_run = subprocess.run

    def timeout(command, **kwargs):
        if command[0] == sys.executable:
            raise subprocess.TimeoutExpired(command, 60)
        return actual_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", timeout)
    assert bench.main(["--output-dir", str(tmp_path), "--case", "add_float32",
                       "--warmup", "0", "--repeats", "1"]) == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert not report["passed"]
    assert report["results"][0]["after"]["status"] == "WORKER_ERROR"
    assert "timed out" in report["results"][0]["after"]["error"]


def test_preparation_failure_replaces_stale_pass_report(tmp_path, monkeypatch):
    (tmp_path / "report.html").write_text("stale PASS", encoding="utf-8")

    def fail(args):
        raise ValueError("invalid baseline")

    monkeypatch.setattr(bench, "run_comparison", fail)
    assert bench.main(["--output-dir", str(tmp_path)]) == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert not report["passed"] and "invalid baseline" in report["runner_error"]
    assert "stale PASS" not in (tmp_path / "report.html").read_text(encoding="utf-8")


def test_reports_escape_errors_and_never_compare_failed_timings():
    before = dict(worker.empty_row(), status="UNSUPPORTED", error="<script>bad()</script>",
                  run_median_s=1.0, parse_median_s=1.0)
    after = dict(worker.empty_row(), status="PASS", run_median_s=0.5, parse_median_s=0.5)
    report = dict(baseline={}, current={}, passed=True, results=[dict(name="case", group="operators",
                  description="test", operators=["Abs"], before=before, after=after)])
    html = render_html(report)
    markdown = render_markdown(report)
    assert "<script>bad()</script>" not in html
    assert "2.000×" not in html and "2.000×" not in markdown
