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
generate_unified_html_report = compare_html.generate_unified_html_report
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


def test_compare_dsl_main_writes_json_without_standalone_html(tmp_path, monkeypatch):
    json_path = tmp_path / "peephole_compare.json"

    monkeypatch.setattr(
        compare_dsl.sys,
        "argv",
        ["compare_peephole.py", "--json", str(json_path)],
    )
    compare_dsl.main()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["dsl_suite"]["cases"]) == 23
    assert not (tmp_path / "peephole_dsl_compare.html").exists()


def test_unified_html_contains_three_independent_benchmark_sections():
    from benchmarks.compare_peephole_cnn import compare_cnn_model

    micro = compare_cases(repeats=1)
    dsl = compare_dsl.run_dsl_suite(
        Path(__file__).resolve().parents[1] / "benchmarks" / "cases"
    )
    cnn = compare_cnn_model(
        Path(__file__).resolve().parents[1] / "models" / "graph" / "cnn.onnx"
    )

    rendered = generate_unified_html_report(micro, dsl, cnn)

    assert rendered.count("14个微型规则案例") == 1
    assert rendered.count("23个 DSL 案例") == 1
    assert rendered.count("cnn.onnx standalone A/B") == 1
    for value in (31, 22, 306, 305, 889, 865):
        assert f">{value:,}<" in rendered
    assert "Commit:" not in rendered
    assert "类别" not in rendered
    assert "输入摘要" not in rendered
    assert "预期规则" not in rendered
    assert "case-table" in rendered
    assert "benchmark-section" in rendered


def test_make_and_ci_use_only_the_unified_peephole_html():
    root = Path(__file__).resolve().parents[1]
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "peephole_benchmark.html" in makefile
    assert "peephole_compare.html" not in makefile
    assert "peephole_dsl_compare.html" not in makefile
    assert "cnn_peephole_compare.html" not in makefile
    assert "make -s ci-peephole" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "name: peephole-benchmark-report" in workflow
    assert "benchmark_reports/peephole_benchmark.html" in workflow
    assert "if-no-files-found: error" in workflow


def test_unified_summary_contains_dynamic_sections_and_case_details():
    from benchmarks.compare_peephole_cnn import compare_cnn_model

    root = Path(__file__).resolve().parents[1]
    micro = compare_cases(repeats=1)
    dsl = compare_dsl.run_dsl_suite(root / "benchmarks" / "cases")
    cnn = compare_cnn_model(root / "models" / "graph" / "cnn.onnx")

    summary = compare_html.generate_summary_markdown(micro, dsl, cnn)

    assert summary.startswith("## Topic 13 Peephole Benchmark")
    assert "| 微型汇编案例 | 14 | 31 | 22 | 9 | 29.032% |" in summary
    assert "| DSL 案例 | 23 | 306 | 305 | 1 |" in summary
    assert "| CNN standalone | 1 | 889 | 865 | 24 |" in summary
    assert "| CNN 代码大小 | 3556 B | 3460 B | 96 B |" in summary
    assert summary.count("<details>") == 3
    micro_details = summary.split("<summary>微型案例明细</summary>", 1)[1].split(
        "</details>", 1
    )[0]
    dsl_details = summary.split("<summary>DSL 案例明细</summary>", 1)[1].split(
        "</details>", 1
    )[0]
    assert (
        sum(f"| {case['case_id']} |" in micro_details for case in micro["cases"]) == 14
    )
    assert sum(f"| {case.name} |" in dsl_details for case in dsl.cases) == 23
    assert "representative_codegen" in summary
    assert "addi+addi fusion (1)" in summary
    assert "beq zero-zero to jump (1)" in summary
    assert "beq_zero_jump | 2 | 2 | 0 | 0.000% | beq zero-zero to jump (1)" in summary
    assert " | — |" in summary
    assert "fixed-point 迭代次数" in summary
    assert "synthetic_100" not in summary
    assert "完整 HTML 和 JSON 报告请下载 Artifact：peephole-benchmark-report" in summary


def test_unified_summary_cli_writes_requested_markdown(tmp_path):
    root = Path(__file__).resolve().parents[1]
    micro = compare_cases(repeats=1)
    dsl = compare_dsl.run_dsl_suite(root / "benchmarks" / "cases")
    from benchmarks.compare_peephole_cnn import compare_cnn_model

    cnn = compare_cnn_model(root / "models" / "graph" / "cnn.onnx")
    micro_path = tmp_path / "micro.json"
    dsl_path = tmp_path / "dsl.json"
    cnn_path = tmp_path / "cnn.json"
    output_path = tmp_path / "peephole_summary.md"
    micro_path.write_text(json.dumps(micro), encoding="utf-8")
    dsl_path.write_text(
        json.dumps(
            {
                "dsl_suite": {
                    "total_before": dsl.total_before,
                    "total_after": dsl.total_after,
                    "total_saved": dsl.total_saved,
                    "cases_with_savings": dsl.cases_with_savings,
                    "cases": [compare_dsl.asdict(case) for case in dsl.cases],
                },
                "synthetic": [],
            }
        ),
        encoding="utf-8",
    )
    cnn_path.write_text(json.dumps(cnn), encoding="utf-8")

    exit_code = compare_html.main(
        [
            "--unified",
            "--micro-json",
            str(micro_path),
            "--dsl-json",
            str(dsl_path),
            "--cnn-json",
            str(cnn_path),
            "--output-html",
            str(tmp_path / "report.html"),
            "--summary-output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert "## Topic 13 Peephole Benchmark" in output_path.read_text(encoding="utf-8")


def test_summary_and_ci_write_job_summary_artifact():
    root = Path(__file__).resolve().parents[1]
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "--summary-output benchmark_reports/peephole_summary.md" in makefile
    assert "benchmark_reports/peephole_summary.md" in workflow
    assert (
        'cat benchmark_reports/peephole_summary.md >> "$GITHUB_STEP_SUMMARY"'
        in workflow
    )
    assert "head -40 benchmark_reports/peephole_compare.md" not in workflow
