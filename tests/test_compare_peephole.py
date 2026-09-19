"""Tests for peephole on/off comparison and HTML report generation."""

from __future__ import annotations

import json

import pytest

import benchmarks.compare_peephole_html as compare_html
from benchmarks.bench_asm_peephole import BenchmarkCase
from benchmarks.compare_peephole_html import (
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


def test_unchanged_cases_count_only_cases_without_optimizer_changes():
    cases = [
        BenchmarkCase(
            case_id="equal_length_branch_rewrite",
            assembly="beq zero, x0, target\ntarget:\nret\n",
            expected_rule="beq zero-zero to jump",
        ),
        BenchmarkCase(
            case_id="equal_length_addi_rewrite",
            assembly="addi t0, t1, 0\nret\n",
            expected_rule="addi-zero to mv",
        ),
    ]

    report = compare_cases(cases, repeats=1)

    assert all(item["reduced_instructions"] == 0 for item in report["cases"])
    assert all(item["changes"] == 1 for item in report["cases"])
    assert report["summary"]["unchanged_cases"] == 0


def test_peephole_off_time_is_unmeasured_and_serializes_as_null(tmp_path):
    report = compare_cases([_case()], repeats=1)
    result = report["cases"][0]
    assert result["peephole_off"]["elapsed_ms_median"] is None

    json_path = tmp_path / "comparison.json"
    save_comparison(report, json_path, tmp_path / "comparison.html")

    assert (
        json.loads(json_path.read_text())["cases"][0]["peephole_off"][
            "elapsed_ms_median"
        ]
        is None
    )


def test_html_report_contains_cards_sections_and_rule_rows():
    report = compare_cases([_case()], repeats=1)

    html = generate_html_report(report)

    assert "ScratchV 窥孔优化器 Benchmark" in html
    assert "peephole 开关对比" in html
    assert "规则命中与节省" in html
    assert "样例明细" in html
    assert "addi+addi fusion" in html
    assert "reduction_percent" not in html


def test_html_count_bars_use_integer_display():
    report = compare_cases([_case()], repeats=1)

    rendered = generate_html_report(report)

    assert "1.000 条" not in rendered
    assert "2 条" in rendered
    assert "1 次" in rendered


def test_html_zero_rule_report_does_not_invent_a_match():
    case = BenchmarkCase(
        case_id="clean",
        assembly="add t0, t1, t2\nret\n",
    )
    rendered = generate_html_report(compare_cases([case], repeats=1))

    assert "1 次" not in rendered
    assert rendered.count("0 次") == 8
    assert 'style="width:0.0%"' in rendered


def test_html_marks_missing_expected_rule_as_not_applicable():
    case = BenchmarkCase(case_id="negative", assembly="ret\n")

    rendered = generate_html_report(compare_cases([case], repeats=1))

    assert ">—<" in rendered
    assert ">否<" not in rendered
    assert 'class="tag muted"' in rendered


def test_html_escapes_custom_case_text_once():
    case = BenchmarkCase(
        case_id='<script>&"中文',
        assembly="ret\n",
        description='<script>&"中文',
    )

    rendered = generate_html_report(compare_cases([case], repeats=1))

    assert "<script>" not in rendered
    assert "&lt;script&gt;&amp;&quot;中文" in rendered
    assert "&amp;lt;script&amp;gt;" not in rendered


def test_repeated_optimizer_runs_have_identical_results():
    assembly = _case().assembly
    first_output, first_changes, first_matches, _ = compare_html._optimize_with_timing(
        assembly,
        repeats=3,
    )
    (
        second_output,
        second_changes,
        second_matches,
        _,
    ) = compare_html._optimize_with_timing(
        assembly,
        repeats=3,
    )

    assert (first_output, first_changes, first_matches) == (
        second_output,
        second_changes,
        second_matches,
    )


def test_compare_rejects_default_rule_drift_with_rule_names(monkeypatch):
    original = compare_html.AsmPeepholeOptimizer

    class DriftedOptimizer(original):
        def __init__(self):
            super().__init__()
            self.rules.append(type(self.rules[0])("future rule", [], []))

    monkeypatch.setattr(compare_html, "AsmPeepholeOptimizer", DriftedOptimizer)

    with pytest.raises(ValueError, match="future rule"):
        compare_cases([_case()], repeats=1)


def test_save_comparison_writes_json_and_html(tmp_path):
    report = compare_cases([_case()], repeats=1)

    json_path = tmp_path / "comparison.json"
    html_path = tmp_path / "comparison.html"
    save_comparison(report, json_path, html_path)

    assert json.loads(json_path.read_text())["cases"]
    assert "<html" in html_path.read_text()
