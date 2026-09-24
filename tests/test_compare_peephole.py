"""Tests for peephole on/off comparison and HTML report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmarks.compare_peephole as compare_dsl
import benchmarks.compare_peephole_html as compare_html
from benchmarks.bench_asm_peephole import BenchmarkCase, default_cases

compare_cases = compare_html.compare_cases
generate_dsl_html_report = compare_html.generate_dsl_html_report
generate_html_report = compare_html.generate_html_report
save_comparison = compare_html.save_comparison


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


def test_compare_cases_keeps_fixed_default_suite():
    report = compare_cases(repeats=1)

    assert report["summary"]["case_count"] == 14
    assert report["summary"]["before_instructions"] == 31
    assert report["summary"]["after_instructions"] == 22
    assert report["summary"]["reduced_instructions"] == 9
    assert report["summary"]["changes"] == 12
    assert report["summary"]["rule_matches"] == {
        "addi+addi fusion": 1,
        "li+addi fusion": 2,
        "beq zero-zero to jump": 1,
        "redundant mv elimination": 1,
        "addi-zero self elimination": 1,
        "addi-zero to mv": 2,
        "nop elimination": 2,
        "mv-self elimination": 2,
    }


def test_html_hides_commit_but_json_keeps_commit_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_html, "_git_commit", lambda: "abc123-dirty")
    report = compare_cases([_case()], repeats=1)

    json_path = tmp_path / "comparison.json"
    html_path = tmp_path / "comparison.html"
    save_comparison(report, json_path, html_path)

    assert json.loads(json_path.read_text())["metadata"]["git_commit"] == (
        "abc123-dirty"
    )
    rendered = html_path.read_text()
    assert "Commit:" not in rendered
    assert "abc123-dirty" not in rendered


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

    rendered = generate_html_report(report)
    assert ">是<" not in rendered
    assert '<td class="delta">0 (0.0%)</td>' in rendered
    assert "beq zero-zero to jump" in rendered
    assert "addi-zero to mv" in rendered


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
    assert "窥孔优化规则" in html
    assert "PR39 默认规则" not in html
    assert "各规则应用次数" in html
    assert "样例明细" in html
    assert "重复次数" in html
    assert "优化器参考耗时" in html
    assert (
        '<table class="case-table"><tr><th>样例</th><th>关闭</th>'
        "<th>开启</th><th>节省</th><th>应用规则</th></tr>"
    ) in html
    assert "类别" not in html
    assert "输入摘要" not in html
    assert "规则命中与节省" not in html
    assert "预期规则" not in html
    assert "addi+addi fusion" in html
    assert "reduction_percent" not in html


def test_html_count_bars_use_integer_display():
    report = compare_cases([_case()], repeats=1)

    rendered = generate_html_report(report)

    assert "1.000 条" not in rendered
    assert "2 条" in rendered
    assert "1 次" in rendered


def test_html_zero_rule_report_preserves_zero_application_counts():
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
    assert ">不适用<" not in rendered
    assert ">否<" not in rendered
    assert 'class="rule-list">—<' in rendered


def test_html_lists_all_rules_applied_by_a_multi_rule_case():
    case = next(
        case for case in default_cases() if case.case_id == "representative_codegen"
    )
    report = compare_cases([case], repeats=1)
    expected_rules = [
        name for name, count in report["cases"][0]["rule_matches"].items() if count > 0
    ]

    assert len(expected_rules) > 1
    details = generate_html_report(report).split("<section><h2>样例明细</h2>", 1)[1]
    for rule in expected_rules:
        assert rule in details


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


def test_dsl_html_report_uses_all_suite_cases_and_actual_rules():
    report = compare_dsl.run_dsl_suite(
        Path(__file__).resolve().parents[1] / "benchmarks" / "cases"
    )

    rendered = generate_dsl_html_report(report)
    details = rendered.split("<section><h2>样例明细</h2>", 1)[1].split("</table>", 1)[0]
    actual_rules = {
        rule
        for case in report.cases
        for rule, count in case.rule_hits.items()
        if count > 0
    }

    assert len(report.cases) == 23
    assert details.count("<tr>") == 24
    assert all(f"synthetic_{size}" not in rendered for size in (100, 500, 1000, 2000))
    assert actual_rules
    assert all(rule in details for rule in actual_rules)
    assert f">{report.total_before:,}<" in rendered
    assert f">{report.total_after:,}<" in rendered
    assert f">{report.total_saved:,} (" in rendered
    assert "正向样例 / 负向样例" not in rendered
    assert "重复次数" not in rendered
    assert "优化器参考耗时" not in rendered


def test_compare_dsl_main_writes_html_from_suite_result(tmp_path, monkeypatch):
    html_path = tmp_path / "peephole_dsl_compare.html"

    monkeypatch.setattr(
        compare_dsl.sys,
        "argv",
        ["compare_peephole.py", "--html", str(html_path)],
    )
    compare_dsl.main()
    rendered = html_path.read_text(encoding="utf-8")

    details = rendered.split("<section><h2>样例明细</h2>", 1)[1].split("</table>", 1)[0]
    assert details.count("<tr>") == 24
    assert "synthetic_100" not in rendered
