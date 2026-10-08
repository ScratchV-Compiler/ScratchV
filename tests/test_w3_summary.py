"""Human summaries select final outputs from the real W3 report schemas."""
from copy import deepcopy
import json

import pytest

from probes.w3_common import write_reports
from probes.w3_summary import evidence_index_views, numerical_summary, summary_views


def metric(value=1e-6, *, name=None, passed=True):
    row = dict(passed=passed, max_abs=value, relative_l2=2e-7, cosine_similarity=0.999,
               atol=1e-4, rtol=0)
    if name is not None:
        row["name"] = name
    return row


def test_full_logits_are_not_replaced_by_large_checkpoint_or_position_errors():
    ordinary = metric(name="logits.npy")
    ordinary["valid_queries"] = metric(0.1, passed=False)
    diagnostic = metric(3e-6, name="diagnostic_logits.npy")
    report = {"gate": "numeric:ir-full-qwen3", "fp32_mode": "native", "cases": [
        {"name": "full_seed_0", "passed": True, "comparison": {
            "logits": [ordinary, diagnostic], "checkpoints": [metric(0.5, name="layer_0.output")],
            "worst": metric(0.5), "max_logits_abs": 0.5}}]}
    before = deepcopy(report)
    description, comparisons, notes = numerical_summary(report)
    assert comparisons[0][1] == [ordinary]
    assert comparisons[1][1] == [diagnostic]
    assert "native" in description and "reference" in " ".join(notes)
    markdown, _ = summary_views(report)
    assert format(0.5, ".16e") not in markdown
    assert format(0.1, ".16e") not in markdown
    assert report == before


def test_full_missing_case_comparison_remains_an_explicit_placeholder():
    report = {"gate": "numeric:ir-full-qwen3", "fp32_mode": "reference", "cases": [
        {"name": "okay", "comparison": {"logits": [metric(name="logits.npy"), metric(name="diagnostic_logits.npy")]}},
        {"name": "failed_worker", "passed": False, "workers": [{"backend": "ir", "passed": False}]}]}
    markdown, page = summary_views(report)
    assert "未记录 1/2" in markdown and "未记录 1/2" in page
    assert "门槛未通过" in markdown
    assert "failed_worker" in markdown


def test_medium_schema_uses_normal_final_logits_not_diagnostic_checkpoints():
    ort_logits = metric(name="logits")
    ir_logits = metric(2e-6, name="logits")
    irrelevant = metric(0.9, name="layer_5.output", passed=False)
    case = {"name": "short_17", "ort": {"normal": {"pytorch_comparison": {"checkpoints": [ort_logits]}}},
            "ir": {level: {"normal": {"ort_comparison": {"checkpoints": [ir_logits]}},
                            "diagnostic": {"ort_comparison": {"checkpoints": [irrelevant]}}}
                   for level in ("none", "basic", "all")}}
    _, comparisons, _ = numerical_summary({"gate": "w3-medium-host", "cases": [case]})
    assert comparisons[0][1] == [ort_logits] + [{}] * 6
    assert all(rows == [ir_logits] + [{}] * 6 for _, rows in comparisons[1:])
    markdown, page = summary_views({"gate": "w3-medium-host", "cases": [case]})
    assert "未记录 6/7" in markdown and "已记录 1/7" in page
    assert "门槛通过" not in markdown


def test_missing_medium_logit_is_not_replaced_by_passing_intermediate():
    case = {"ort": {"normal": {"pytorch_comparison": {"checkpoints": [metric(name="embedding")]}}}}
    _, comparisons, _ = numerical_summary({"gate": "w3-medium-host", "cases": [case]})
    assert comparisons[0][1] == [{}] * 7
    markdown, _ = summary_views({"gate": "w3-medium-host", "cases": [case]})
    assert "门槛通过" not in markdown


@pytest.mark.parametrize("length", [17, 256])
def test_subgraphs_use_actual_length_and_ordinary_outputs(length):
    def subgraph_metric(value):
        row = metric(value)
        row["max_abs_error"] = row.pop("max_abs")
        return row

    comparisons = {"ordinary_ort_vs_torch": subgraph_metric(4e-6),
                   **{f"ordinary_ir_{level}_vs_ort": subgraph_metric(5e-6) for level in ("none", "basic", "all")},
                   "ir_none_vs_ort": {"checkpoints": [metric(0.7)]}}
    report = {"gate": "w3_qwen3_subgraphs", "sequence_length": length,
              "cases": [{"name": "rmsnorm", "comparisons": comparisons}]}
    description, rows, notes = numerical_summary(report)
    assert f"L={length}" in description
    assert rows[0][1] == [{**comparisons["ordinary_ort_vs_torch"], "max_abs": 4e-6}]
    assert rows[1][1] == [{**comparisons["ordinary_ir_none_vs_ort"], "max_abs": 5e-6}]
    assert "max_abs" not in comparisons["ordinary_ort_vs_torch"]
    markdown, page = summary_views(report)
    assert format(4e-6, ".16e") in markdown and format(5e-6, ".16e") in page
    assert "指标不完整" not in markdown
    assert ("短序列调试" in " ".join(notes)) == (length != 256)


def test_attention_missing_optimization_does_not_vanish_from_table():
    execution = {"optimization": "none", "ir_vs_ort": metric(), "qemu_vs_ort": metric(2e-6),
                 "qemu_vs_numpy": metric(3e-6)}
    report = {"gate": "unit:attention-backend", "cases": [{"name": "full17", "executions": [execution]}]}
    _, comparisons, _ = numerical_summary(report)
    assert len(comparisons) == 3
    assert all(rows[1] == {} for _, rows in comparisons)
    markdown, _ = summary_views(report)
    assert "未记录 1/2" in markdown
    assert "门槛通过" not in markdown


def test_audit_is_distinguished_from_execution_and_language_quality():
    description, _, notes = numerical_summary({"gate": "audit:w3-full-saved-evidence", "fp32_mode": "reference"})
    assert "reference" in description
    text = " ".join(notes)
    assert "没有重新执行模型" in text and "不算第二人独立运行" in text
    assert "PPL / zero-shot：未评测" in text


def test_partial_coverage_and_failed_invariants_remain_visible():
    report = {"gate": "numeric:ir-full-qwen3", "coverage_complete": False,
              "required_cases": ["one", "two"], "selected_cases": ["one"], "cases": [],
              "invariants": [{"name": "changed_future", "passed": False}]}
    markdown, _ = summary_views(report)
    assert "覆盖不完整" in markdown
    assert "选择 1/2" in markdown
    assert "隔离检查未通过 1 项" in markdown


def test_duplicate_output_names_are_not_silently_reduced_to_first_pass():
    report = {"gate": "numeric:ir-full-qwen3", "cases": [{"comparison": {"logits": [
        metric(name="logits.npy"), metric(1.0, name="logits.npy", passed=False)]}}]}
    _, rows, _ = numerical_summary(report)
    assert rows[0][1][0]["passed"] is False
    assert "重复" in rows[0][1][0]["reason"]


def test_views_fold_full_evidence_without_changing_overall_failed_status(tmp_path):
    report = {"gate": "numeric:ir-full-qwen3", "passed": False, "status": "FAIL", "atol": 1e-4,
              "cases": [{"passed": True, "comparison": {"logits": [metric(name="logits.npy"),
                        metric(name="diagnostic_logits.npy")], "checkpoints": [metric(0.8, name="layer_3.output")]}}],
              "invariants": [{"passed": False}], "error": "<script>alert(1)</script>"}
    before = deepcopy(report)
    write_reports(tmp_path, report)
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == before
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    page = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Result: **FAIL**" in markdown
    assert "门槛通过" in markdown and "总体结论" in markdown
    assert "<details>" in markdown and "<details>" in page
    assert markdown.index("最大绝对误差") < markdown.index("<details>")
    assert '"checkpoints"' not in markdown
    assert "[完整机器报告](report.json)" in markdown
    assert 'href="report.json"' in page
    assert "<script>" not in page
    assert "<script>" not in markdown


def test_large_machine_evidence_is_retained_only_in_json(tmp_path):
    sentinel = "RAW_TRACE_ONLY_" * 100000
    report = {"gate": "w3-medium-host", "status": "PASS", "passed": True,
              "trace_artifacts": {"large": sentinel}, "source_sha256": {"source": sentinel},
              "cases": [{"name": "full_seed_0", "passed": True, "unrelated_evidence": sentinel}]}
    write_reports(tmp_path, report)
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved == report
    for name in ("report.md", "report.html"):
        path = tmp_path / name
        content = path.read_text(encoding="utf-8")
        assert "RAW_TRACE_ONLY" not in content
        assert path.stat().st_size < 16000
        assert "report.json" in content and "Artifacts" in content


def test_index_is_bounded_and_prioritizes_late_failures():
    report = {"cases": [{"name": f"okay_{i}", "passed": True} for i in range(50)] + [
        {"name": "late_failure", "passed": False, "error": "wrong shape"}]}
    markdown, page = evidence_index_views(report)
    assert "共 51 项，展示 20 项" in markdown
    assert "另 31 项" in page
    assert "late_failure" in markdown and "wrong shape" in page
    assert "okay_49" not in markdown
    assert page.count("<td>") == 20 * 3


def test_preparation_gate_failure_remains_visible_without_raw_json(tmp_path):
    report = {"gate": "preparation:w3", "status": "FAIL", "passed": False,
              "gates": [{"name": "medium", "passed": True, "status": "PASS"},
                        {"name": "attention", "passed": False, "status": "FAIL", "error": "QEMU timeout"}]}
    write_reports(tmp_path, report)
    for name in ("report.md", "report.html"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "FAIL" in text and "子门禁未通过/未确认 1 项" in text
        assert "attention" in text and "QEMU timeout" in text


@pytest.mark.parametrize("partial", [False, True])
def test_layer_diff_exposes_coverage_and_first_difference(partial):
    report = {"gate": "w3_layer_diff", "status": "PARTIAL" if partial else "FAIL", "passed": False,
              "partial": partial, "coverage": {"compared_checkpoints": 1, "total_checkpoints": 81,
                                                 "complete": not partial},
              "first_divergence": None if partial else "layer_3.output",
              "checkpoints": [metric(0.8, name="layer_3.output", passed=partial)]}
    markdown, _ = summary_views(report)
    index, _ = evidence_index_views(report)
    assert "检查点覆盖：1/81" in markdown
    assert "layer_3.output" in index
    if partial:
        assert "PARTIAL" in markdown and "不能作为完整覆盖通过" in markdown
    else:
        assert "首个差异检查点：layer_3.output" in markdown
        assert "FAIL" in index


@pytest.mark.parametrize("gate", ["numeric:w3-full-worker", "preflight:w3-full-assets"])
def test_worker_and_preflight_keep_failure_stage_and_scope(gate):
    report = {"gate": gate, "status": "FAIL", "passed": False, "stage": "asset_hashes",
              "error": "hash mismatch"}
    markdown, _ = summary_views(report)
    index, _ = evidence_index_views(report)
    assert "asset_hashes" in markdown and "hash mismatch" in markdown
    assert "asset_hashes" in index and "FAIL" in index
    assert "上层门禁核验" in markdown if "worker" in gate else "没有执行完整模型" in markdown


def test_localizer_complete_is_not_a_numerical_gate_pass(tmp_path):
    report = {"gate": "diagnostic:w3-full-localize", "status": "COMPLETE", "passed": True,
              "diagnostic_complete": True, "numeric_gate_passed": False, "w3_exit_accepted": False}
    write_reports(tmp_path, report)
    for name in ("report.md", "report.html"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "COMPLETE" in text and "不表示数值门槛通过" in text
        assert "PASS" not in text
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == report


def test_medium_nested_failure_is_reachable_in_short_index():
    report = {"cases": [{"name": "short_17", "passed": False, "ir": {"none": {"normal": {
        "passed": False, "ort_comparison": {"passed": False, "first_divergence": "logits"}}}}}]}
    markdown, page = evidence_index_views(report)
    assert "short_17" in markdown and "首个差异：logits" in page


def test_long_error_is_shortened_only_in_human_view(tmp_path):
    report = {"gate": "preparation:w3", "passed": False, "status": "FAIL",
              "error": "exception " * 10000}
    write_reports(tmp_path, report)
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == report
    for name in ("report.md", "report.html"):
        content = (tmp_path / name).read_text(encoding="utf-8")
        assert "完整内容见 report.json" in content
        assert len(content) < 6000
