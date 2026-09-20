"""Exercise the Topic 9 report through its real subprocess interface."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "bench_dsl_diagnostics.py"


def run_report(tmp_path: Path, *extra: str) -> tuple[subprocess.CompletedProcess, dict]:
    cases = tmp_path / "cases"
    cases.mkdir(exist_ok=True)
    source = cases / "valid.dsl"
    if not source.exists():
        source.write_text("x = add(a, b)\nreturn x\n", encoding="utf-8")
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--baseline-root", str(ROOT),
            "--cases", str(cases), "--warmup", "0", "--groups", "1",
            "--iterations", "1", "--json-output", str(output),
            "--markdown", str(tmp_path / "report.md"),
            "--html", str(tmp_path / "report.html"), *extra,
        ],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    report = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    return result, report


def test_real_report_checks_diagnostics_and_compares_same_corpus(tmp_path):
    result, report = run_report(tmp_path, "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == report
    assert report["status"] == "passed"
    assert report["parsing"]["ir_equal"] is True
    assert report["parsing"]["ratio"] > 0
    assert report["parsing"]["baseline"]["case_hashes"] == report["parsing"]["current"]["case_hashes"]
    assert report["parsing"]["current"]["parser_file"].startswith(str(ROOT))
    assert report["parsing"]["current"]["samples_s"]
    assert report["diagnostics"]["passed"] is True
    checks = {case["name"]: case for case in report["diagnostics"]["cases"]}
    assert checks["three_errors"]["actual_codes"] == ["E100", "E200", "E201"]
    assert checks["error_limit"]["actual_count"] == 20
    assert checks["error_limit"]["limit_reached"] is True
    assert checks["three_errors"]["cli_exit_code"] == 1
    assert "\x1b[" not in checks["three_errors"]["rendered"]
    assert "DSL diagnostics" in (tmp_path / "report.md").read_text(encoding="utf-8")
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<!doctype html>" in html.lower()
    assert "<script" not in html.lower()


def test_missing_baseline_fails_and_still_writes_reports(tmp_path):
    result, report = run_report(tmp_path, "--baseline-root", str(tmp_path / "missing"))
    assert result.returncode == 1
    assert report["status"] == "failed"
    assert "baseline" in report["error"].lower()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.html").exists()


def test_empty_corpus_is_not_a_successful_benchmark(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result, report = run_report(tmp_path, "--cases", str(empty))
    assert result.returncode == 1
    assert report["status"] == "failed"
    assert "no DSL cases" in report["error"]


def test_malformed_valid_case_is_not_silently_skipped(tmp_path):
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "valid.dsl").write_text("x = ad(a, b)\n", encoding="utf-8")
    result, report = run_report(tmp_path)
    assert result.returncode == 1
    assert report["status"] == "failed"
    assert "valid.dsl" in report["error"]


@pytest.mark.parametrize("option", ["--groups", "--iterations", "--max-parse-ratio"])
def test_invalid_measurement_configuration_is_rejected(tmp_path, option):
    result, _ = run_report(tmp_path, option, "0")
    assert result.returncode == 2
    assert "positive" in result.stderr


def test_performance_target_can_be_enforced_without_faking_functional_failure(tmp_path):
    result, report = run_report(tmp_path, "--max-parse-ratio", "0.000001", "--enforce-performance")
    assert result.returncode == 1
    assert report["status"] == "failed"
    assert report["diagnostics"]["passed"] is True
    assert report["parsing"]["ir_equal"] is True
    assert report["parsing"]["target_met"] is False


def test_performance_target_is_visible_when_not_enforced(tmp_path):
    result, report = run_report(tmp_path, "--max-parse-ratio", "0.000001")
    assert result.returncode == 0, result.stderr
    assert report["parsing"]["target_met"] is False
    assert report["warnings"]


def test_baseline_must_import_its_own_parser(tmp_path):
    baseline = tmp_path / "baseline"
    frontend = baseline / "scratchv" / "frontend"
    frontend.mkdir(parents=True)
    (baseline / "scratchv" / "__init__.py").write_text("", encoding="utf-8")
    (frontend / "__init__.py").write_text("", encoding="utf-8")
    (frontend / "dsl_extended.py").write_text(
        "raise RuntimeError('baseline sentinel: do not import the current checkout')\n",
        encoding="utf-8",
    )
    result, report = run_report(tmp_path, "--baseline-root", str(baseline))
    assert result.returncode == 1
    assert "baseline sentinel" in report["error"]


def test_changed_ir_fails_the_report_even_if_both_parsers_succeed(tmp_path):
    baseline = tmp_path / "baseline"
    frontend = baseline / "scratchv" / "frontend"
    frontend.mkdir(parents=True)
    (baseline / "scratchv" / "__init__.py").write_text("", encoding="utf-8")
    (frontend / "__init__.py").write_text("", encoding="utf-8")
    (frontend / "dsl_extended.py").write_text(
        "class ExtendedDSLParser:\n"
        "    def parse(self, source):\n"
        "        return self\n"
        "    def dump(self):\n"
        "        return 'deliberately different IR'\n",
        encoding="utf-8",
    )
    result, report = run_report(tmp_path, "--baseline-root", str(baseline))
    assert result.returncode == 1
    assert report["status"] == "failed"
    assert report["parsing"]["same_cases"] is True
    assert report["parsing"]["ir_equal"] is False


def test_expected_ir_change_can_be_allowed_for_one_named_case(tmp_path):
    baseline = tmp_path / "baseline"
    frontend = baseline / "scratchv" / "frontend"
    frontend.mkdir(parents=True)
    (baseline / "scratchv" / "__init__.py").write_text("", encoding="utf-8")
    (frontend / "__init__.py").write_text("", encoding="utf-8")
    (frontend / "dsl_extended.py").write_text(
        "class ExtendedDSLParser:\n"
        "    def parse(self, source):\n"
        "        return self\n"
        "    def dump(self):\n"
        "        return 'deliberately different IR'\n",
        encoding="utf-8",
    )
    result, report = run_report(
        tmp_path,
        "--baseline-root", str(baseline),
        "--allow-ir-change", "valid.dsl",
    )
    assert result.returncode == 0, result.stderr
    assert report["status"] == "passed"
    assert report["parsing"]["ir_equal"] is False
    assert report["parsing"]["ir_changed_cases"] == ["valid.dsl"]
    assert report["parsing"]["unexpected_ir_changes"] == []


def test_html_escapes_diagnostic_source(tmp_path):
    result, _ = run_report(tmp_path)
    assert result.returncode == 0, result.stderr
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "if (a &gt; b):" in html
    assert "if (a > b):" not in html


def test_diagnostic_logs_are_collapsed_in_markdown_and_html(tmp_path):
    result, report = run_report(tmp_path)
    assert result.returncode == 0, result.stderr
    count = len(report["diagnostics"]["cases"])
    for extension in ("md", "html"):
        content = (tmp_path / f"report.{extension}").read_text(encoding="utf-8")
        assert content.count("<details>") == count
        assert content.count("</details>") == count
        assert "<details open" not in content
        assert "<summary>error_limit: 20 error(s)" in content
        for case in report["diagnostics"]["cases"]:
            import html

            expected = html.escape(case["rendered"]) if extension == "html" else case["rendered"]
            assert expected in content
        assert content.index("Diagnostic acceptance and timings") < content.index("<details>")
