"""Tests for peephole on/off comparison and HTML report generation."""

from __future__ import annotations

import json

from benchmarks.bench_asm_peephole import BenchmarkCase
from benchmarks.compare_peephole import (
    compare_cases,
    generate_html_report,
    save_comparison,
)


def _case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id="addi",
        assembly=".text\naddi t0, t0, 1\naddi t0, t0, 2\n",
        expected_rule="addi+addi fusion",
    )


def test_compare_cases_uses_same_input_for_off_and_on():
    report = compare_cases([_case()], repeats=1)

    result = report["cases"][0]
    assert result["peephole_off"]["instructions"] == 2
    assert result["peephole_on"]["instructions"] == 1
    assert result["input_sha256"]
    assert report["summary"]["reduced_instructions"] == 1


def test_compare_cases_handles_no_change_and_zero_baseline():
    case = BenchmarkCase(case_id="empty", assembly="", expected_rule=None)

    report = compare_cases([case], repeats=1)

    result = report["cases"][0]
    assert result["reduced_instructions"] == 0
    assert result["reduction_percent"] == 0.0


def test_html_report_contains_cards_sections_and_rule_rows():
    report = compare_cases([_case()], repeats=1)

    html = generate_html_report(report)

    assert "ScratchV 窥孔优化器 Benchmark" in html
    assert "peephole 开关对比" in html
    assert "规则命中与节省" in html
    assert "样例明细" in html
    assert "addi+addi fusion" in html
    assert "reduction_percent" not in html


def test_save_comparison_writes_json_and_html(tmp_path):
    report = compare_cases([_case()], repeats=1)

    json_path = tmp_path / "comparison.json"
    html_path = tmp_path / "comparison.html"
    save_comparison(report, json_path, html_path)

    assert json.loads(json_path.read_text())["cases"]
    assert "<html" in html_path.read_text()
