"""Benchmark reports carry checked results and fail on incorrect execution."""

import json

import numpy as np
import pytest

from benchmarks import bench_ir_interpreter as bench
from benchmarks.ir_interpreter_cases import TENSOR_BENCHMARK_CASES, compare, make_case
from scratchv.verification.ir_interpreter import ExecutionResult, IRInterpreter


def test_cli_creates_reports_and_samples(tmp_path):
    json_path, md_path = tmp_path / "nested/report.json", tmp_path / "nested/report.md"
    assert (
        bench.main(
            [
                "--warmup",
                "0",
                "--repeats",
                "2",
                "--json-output",
                str(json_path),
                "--markdown",
                str(md_path),
            ]
        )
        == 0
    )
    report = json.loads(json_path.read_text())
    assert report["passed"]
    assert [row["case"] for row in report["results"]] == list(TENSOR_BENCHMARK_CASES)
    assert all(
        row["correct"] and len(row["samples_s"]) == 2 and row["executed_steps"] > 0
        for row in report["results"]
    )
    assert all(
        row["min_s"] <= row["median_s"] <= row["max_s"] for row in report["results"]
    )
    assert {"python", "numpy", "blas_config", "thread_environment"} <= set(
        report["environment"]
    )
    assert "包含验证、计划、绑定、计算和返回副本" in md_path.read_text()
    summary = md_path.read_text()
    assert "Summary: 6/6 PASS, 0 FAIL" in summary
    assert "Python 标量平方求和 + math.sqrt" in summary
    assert "abs(actual - expected)" in summary
    assert "预期输出 shape / dtype" in summary
    assert "无失败。" in summary
    assert "<details>" not in summary


def test_wrong_results_fail_but_preserve_report(monkeypatch, tmp_path):
    original = IRInterpreter.run

    def wrong(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        return ExecutionResult(
            result.return_value + np.asarray(1, dtype=result.return_value.dtype),
            result.executed_steps,
        )

    monkeypatch.setattr(IRInterpreter, "run", wrong)
    json_path, md_path = tmp_path / "report.json", tmp_path / "report.md"
    assert (
        bench.main(
            [
                "--case",
                "loop_sum",
                "--json-output",
                str(json_path),
                "--markdown",
                str(md_path),
            ]
        )
        == 1
    )
    report = json.loads(json_path.read_text())
    assert not report["passed"]
    assert "max_abs_error" in report["results"][0]["error"]
    assert report["results"][0]["samples_s"] == []
    assert "FAIL" in md_path.read_text()


def test_every_repeat_is_checked(monkeypatch):
    original = IRInterpreter.run
    calls = []

    def unstable(self, *args, **kwargs):
        calls.append(1)
        result = original(self, *args, **kwargs)
        if len(calls) == 3:
            return ExecutionResult(
                result.return_value + np.asarray(1, dtype=result.return_value.dtype),
                result.executed_steps,
            )
        return result

    monkeypatch.setattr(IRInterpreter, "run", unstable)
    report = bench.benchmark(cases=["loop_sum"], warmup=0, repeats=3)
    assert not report["passed"]
    assert len(calls) == 3
    assert len(report["results"][0]["samples_s"]) == 2


def test_report_writing_failure(tmp_path):
    obstruction = tmp_path / "file"
    obstruction.write_text("not a directory")
    assert (
        bench.main(
            [
                "--case",
                "branch",
                "--json-output",
                str(obstruction / "report.json"),
                "--markdown",
                str(tmp_path / "report.md"),
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    "arguments", [["--repeats", "0"], ["--warmup", "-1"], ["--case", "unknown"]]
)
def test_invalid_cli_arguments(arguments):
    with pytest.raises(SystemExit) as found:
        bench.main(arguments)
    assert found.value.code == 2


def test_comparison_rejects_shape_dtype_and_nonfinite():
    case = make_case("gather_embedding")
    for invalid in (
        case.expected.ravel(),
        case.expected.astype("float64"),
        np.full_like(case.expected, np.nan),
    ):
        with pytest.raises(AssertionError):
            compare(invalid, case)


def test_example_cli(capsys):
    from examples.run_ir_interpreter import main

    assert main(["--case", "loop_sum"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_summary_preserves_original_execution_error_position(monkeypatch):
    case = make_case("gather_embedding")
    case.inputs["indices"][0, 0] = 4
    monkeypatch.setattr(bench, "make_case", lambda name: case)
    report = bench.benchmark(cases=["gather_embedding"], warmup=0, repeats=1)
    assert not report["passed"]
    row = report["results"][0]
    assert row["error_code"] == "IndexError"
    assert row["error_location"]["instruction_index"] == 0
    assert row["error_location"]["opcode"] == "gather"
    summary = bench.markdown(report)
    assert "总体状态：FAIL" in summary
    assert "Summary: 0/1 PASS, 1 FAIL" in summary
    assert "function=main, block=entry, instruction_index=0, opcode=gather" in summary


def test_summary_does_not_claim_unavailable_failure_metrics():
    from benchmarks.ir_interpreter_summary import render_summary

    summary = render_summary(
        [
            {"case": "unbuilt", "correct": False, "error": "case construction failed"},
        ]
    )
    assert "总体状态：FAIL" in summary
    assert "未获得" in summary
    assert "case construction failed" in summary
    assert "max_abs_error=0" not in summary
