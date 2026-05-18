"""Tests for the benchmark runner module."""

import os
import tempfile
from pathlib import Path

import pytest
from benchmarks.bench_runner import (
    BenchmarkRunner, BenchmarkReport, CaseResult,
)


# Path to test cases
CASES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "benchmarks", "cases",
)


class TestCaseResult:
    """Tests for CaseResult dataclass."""

    def test_create_result(self):
        result = CaseResult(
            name="test",
            description="A test case",
            passed=True,
            output="1.0",
            expected="1.0",
            parse_time_s=0.001,
            compile_time_s=0.002,
            sim_time_s=0.003,
            instruction_count=5,
        )
        assert result.name == "test"
        assert result.passed is True
        assert result.instruction_count == 5

    def test_failed_result(self):
        result = CaseResult(
            name="fail_test",
            passed=False,
            error="DSLParseError: Cannot parse",
        )
        assert not result.passed
        assert "DSLParseError" in result.error


class TestBenchmarkReport:
    """Tests for BenchmarkReport dataclass."""

    def test_empty_report(self):
        report = BenchmarkReport()
        assert report.total_cases == 0
        assert report.pass_count == 0
        assert report.fail_count == 0
        assert report.pass_rate == 0.0

    def test_report_with_results(self):
        report = BenchmarkReport(
            results=[
                CaseResult(name="a", passed=True),
                CaseResult(name="b", passed=True),
                CaseResult(name="c", passed=False),
            ],
            total_time_s=1.5,
        )
        assert report.total_cases == 3
        assert report.pass_count == 2
        assert report.fail_count == 1
        assert report.pass_rate == pytest.approx(66.67, abs=0.1)
        assert report.total_time_s == 1.5

    def test_report_statistics(self):
        report = BenchmarkReport(
            results=[
                CaseResult(name="a", parse_time_s=0.1, compile_time_s=0.2,
                           sim_time_s=0.3, instruction_count=10),
                CaseResult(name="b", parse_time_s=0.2, compile_time_s=0.3,
                           sim_time_s=0.4, instruction_count=20),
            ]
        )
        assert report.total_parse_time == pytest.approx(0.3)
        assert report.total_compile_time == pytest.approx(0.5)
        assert report.total_sim_time == pytest.approx(0.7)
        assert report.total_instructions == 30

    def test_to_dict(self):
        report = BenchmarkReport(
            results=[CaseResult(name="test", passed=True)],
            total_time_s=0.1,
        )
        d = report.to_dict()
        assert "pass_count" in d
        assert "pass_rate" in d
        assert "results" in d

    def test_to_markdown(self):
        report = BenchmarkReport(
            results=[CaseResult(name="test", passed=True)],
            total_time_s=0.1,
        )
        md = report.to_markdown()
        assert "# ScratchV Benchmark Report" in md
        assert "test" in md
        assert "PASS" in md

    def test_to_html(self):
        report = BenchmarkReport(
            results=[CaseResult(name="test", passed=True)],
        )
        html = report.to_html()
        assert "<html" in html
        assert "ScratchV Benchmark" in html

    def test_save_json(self):
        report = BenchmarkReport(results=[CaseResult(name="t", passed=True)])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            report.save_json(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_save_markdown(self):
        report = BenchmarkReport(results=[CaseResult(name="t", passed=True)])
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            report.save_markdown(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_save_html(self):
        report = BenchmarkReport(results=[CaseResult(name="t", passed=True)])
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report.save_html(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_runner_creation(self):
        runner = BenchmarkRunner(test_dir="benchmarks/cases")
        assert runner.test_dir == Path("benchmarks/cases")

    def test_discover_cases(self):
        if Path(CASES_DIR).is_dir():
            runner = BenchmarkRunner(CASES_DIR)
            cases = runner.discover_cases()
            assert len(cases) >= 20, f"Expected >= 20 cases, got {len(cases)}"
            for case in cases:
                assert "name" in case
                assert "dsl_path" in case
                assert os.path.exists(case["dsl_path"])

    def test_discover_cases_missing_dir(self):
        runner = BenchmarkRunner("nonexistent_dir", verbose=False)
        cases = runner.discover_cases()
        assert len(cases) == 0

    def test_run_all(self):
        if Path(CASES_DIR).is_dir():
            runner = BenchmarkRunner(CASES_DIR, verbose=False)
            report = runner.run_all()
            assert isinstance(report, BenchmarkReport)
            assert report.total_cases >= 20
            assert report.pass_count + report.fail_count == report.total_cases

    def test_run_case_basic(self):
        dsl_path = os.path.join(CASES_DIR, "001_simple_add.dsl")
        if os.path.exists(dsl_path):
            runner = BenchmarkRunner(CASES_DIR, verbose=False)
            case = {
                "name": "001_simple_add",
                "dsl_path": dsl_path,
                "expected_path": "",
                "desc_path": "",
            }
            result = runner.run_case(case)
            assert isinstance(result, CaseResult)
            assert result.name == "001_simple_add"

    def test_run_all_timing(self):
        if Path(CASES_DIR).is_dir():
            runner = BenchmarkRunner(CASES_DIR, verbose=False)
            report = runner.run_all()
            assert report.total_time_s >= 0
            assert isinstance(report.total_parse_time, float)
            assert isinstance(report.total_compile_time, float)

    def test_runner_with_timeout(self):
        runner = BenchmarkRunner(CASES_DIR, timeout=5.0, verbose=False)
        assert runner.timeout == 5.0

    def test_empty_test_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(tmpdir, verbose=False)
            cases = runner.discover_cases()
            assert len(cases) == 0
            report = runner.run_all()
            assert report.total_cases == 0
