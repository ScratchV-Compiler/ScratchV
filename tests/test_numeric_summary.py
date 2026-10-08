"""Keep concise reports truthful when evidence is missing or undefined."""
from html import unescape
import re

import pytest

from probes.numeric_summary import (
    METRIC_EXPLANATION,
    compact_metric_table,
    compact_metric_table_html,
)


def record(**updates):
    row = dict(passed=True, max_abs=1e-6, relative_l2=2e-7, cosine_similarity=0.999)
    row.update(updates)
    return row


def test_worst_metrics_are_aggregated_independently_without_mutating_evidence():
    rows = [record(), {"passed": True, "max_abs": 3e-6, "relative_l2": 1e-8, "cosine_similarity": 0.9}]
    original = [dict(row) for row in rows]
    result = compact_metric_table([("IR vs ORT", rows)])
    assert format(3e-6, ".16e") in result
    assert format(2e-7, ".16e") in result
    assert format(0.9, ".17g") in result
    assert "门槛通过（2/2）" in result
    assert rows == original


@pytest.mark.parametrize("render", [compact_metric_table, compact_metric_table_html])
def test_legacy_pass_does_not_claim_complete_metric_evidence(render):
    result = render([("legacy", [{"passed": True, "max_abs": 0.0}])])
    assert "门槛通过（1/1）；指标不完整" in result
    assert result.count("未记录 1/1") == 2


@pytest.mark.parametrize("rows", [[], [{}], [{"passed": "true"}], [{"passed": 1}]])
def test_empty_missing_and_nonboolean_status_never_pass(rows):
    result = compact_metric_table([("unexecuted", rows)])
    assert "门槛通过" not in result
    assert "未测" in result or "未确认 1" in result


def test_failures_are_not_hidden_by_passing_rows():
    failed = record()
    failed["passed"] = False
    result = compact_metric_table([("mixed", [record(), failed, {}])])
    assert "门槛未通过（通过 1/3，失败 1，未确认 1）" in result
    assert "已记录 2/3" in result
    assert "未记录 1/3" in result


def test_undefined_values_show_reason_and_coverage_not_zero():
    undefined = dict(passed=True, max_abs=0.0, relative_l2=None, cosine_similarity=None,
                     relative_l2_reason="zero reference norm", cosine_similarity_reason="zero norm")
    result = compact_metric_table([("partial", [record(), undefined, {}])])
    assert "未定义 1/3（zero reference norm ×1）" in result
    assert "未定义 1/3（zero norm ×1）" in result
    assert "已记录 1/3" in result
    assert "未记录 1/3" in result


def test_undefined_general_reason_is_preserved():
    result = compact_metric_table([("bad", [{"passed": False, "max_abs": None, "reason": "shape mismatch"}])])
    assert "未定义 1/1（shape mismatch ×1）" in result


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, True, "0"])
def test_invalid_error_values_cannot_be_reported_as_numeric_evidence(invalid):
    row = record()
    row["relative_l2"] = invalid
    result = compact_metric_table([("bad", [row])])
    assert "未定义 1/1（数值无效 ×1）" in result
    assert "指标不完整" in result


def test_small_nonzero_errors_and_near_one_cosines_remain_distinguishable():
    row = dict(passed=True, max_abs=5e-324, relative_l2=1e-100, cosine_similarity=0.9999999999999999)
    result = compact_metric_table([("tiny", [row])])
    assert format(5e-324, ".16e") in result
    assert format(1e-100, ".16e") in result
    assert "0.99999999999999989" in result


def test_cosine_domain_and_existing_gate_flags_are_not_reinterpreted():
    # Cosine observes directional similarity; it does not impose a new gate.
    result = compact_metric_table([("opposite", [record(cosine_similarity=-1.0)])])
    assert "门槛通过（1/1）" in result and "未定义" not in result
    invalid = compact_metric_table([("invalid", [record(cosine_similarity=1.1)])])
    assert "未定义 1/1（数值无效 ×1）" in invalid


def test_entirely_undefined_metrics_do_not_acquire_a_numeric_value():
    row = dict(passed=False, max_abs=None, relative_l2=None, cosine_similarity=None)
    result = compact_metric_table([("undefined", [row])])
    assert result.count("未定义 1/1（未提供原因 ×1）") == 3
    assert "已记录" not in result


def test_html_and_markdown_have_same_values_order_and_safe_untrusted_text():
    labels = ["second <script>\"|", "first"]
    comparisons = [(label, [record()]) for label in labels]
    markdown = compact_metric_table(comparisons)
    html = compact_metric_table_html(comparisons)
    assert "<script>" not in markdown
    assert "<script>" not in html
    assert "&#124;" in markdown
    assert markdown.index("second") < markdown.index("first")
    cells = [unescape(value) for value in re.findall(r"<td>(.*?)</td>", html)]
    assert cells[0] == labels[0] and cells[5] == labels[1]
    assert cells[1] in markdown and cells[2] in markdown and cells[3] in markdown


def test_no_comparison_groups_are_reported_as_absent():
    assert "暂无数值比较记录" in compact_metric_table([])
    assert "暂无数值比较记录" in compact_metric_table_html([])
    assert "原有通过门槛不变" in METRIC_EXPLANATION


def test_bad_comparison_rows_fail_explicitly():
    with pytest.raises(TypeError, match="dictionaries"):
        compact_metric_table([("bad", [None])])
