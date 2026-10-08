"""Compact numerical evidence tables; never invent metrics for older reports.

The ``passed`` flag remains the existing numerical gate's result. The two
additional metrics are observations, not new pass/fail thresholds. A table row
is an aggregation of tensor comparisons, not a comparison of concatenated data.
"""
from __future__ import annotations

from collections import Counter
from html import escape
import math
from numbers import Real


METRIC_EXPLANATION = (
    "最大绝对误差是单个元素最坏的偏差；相对 L2 = ||实际−参考||₂ / ||参考||₂，越小越好；"
    "余弦相似度衡量两个输出向量方向的一致性，越接近 1 越好。"
    "先逐次比较，再取最大绝对误差、最大相对 L2、最小余弦（可能来自不同用例，不拼接计算）。"
    "原有通过门槛不变，其他检查仍生效；新增两项仅供观察，缺失或未定义明确标注。"
)

_METRICS = ("max_abs", "relative_l2", "cosine_similarity")
_HEADERS = ("比较", "最大绝对误差", "相对 L2 误差", "余弦相似度", "结果")


def _metric_summary(rows, key):
    values, missing, reasons = [], 0, Counter()
    for row in rows:
        if key not in row:
            missing += 1
            continue
        value = row[key]
        reason = row.get(key + "_reason") or row.get("reason")
        valid = isinstance(value, Real) and not isinstance(value, bool)
        try:
            number = float(value) if valid else None
            valid = valid and math.isfinite(number)
        except (OverflowError, ValueError):
            valid = False
        if valid and (number < 0 if key != "cosine_similarity" else not -1 <= number <= 1):
            valid = False
        if valid:
            values.append(number)
        else:
            reasons[str(reason or ("未提供原因" if value is None else "数值无效"))] += 1
    worst = (min(values) if key == "cosine_similarity" else max(values)) if values else None
    return dict(value=worst, valid_count=len(values), total=len(rows),
                missing=missing, undefined=sum(reasons.values()), reasons=reasons)


def _metric_text(summary, key):
    parts = []
    if summary["value"] is not None:
        # 17 significant digits distinguish adjacent binary64 cosine values.
        # Scientific notation preserves even the smallest nonzero errors.
        number = summary["value"]
        parts.append(format(number, ".17g") if key == "cosine_similarity" else format(number, ".16e"))
    if summary["missing"]:
        parts.append(f"未记录 {summary['missing']}/{summary['total']}")
    if summary["undefined"]:
        reasons = "；".join(f"{reason} ×{count}" for reason, count in summary["reasons"].items())
        parts.append(f"未定义 {summary['undefined']}/{summary['total']}（{reasons}）")
    if summary["value"] is not None and (summary["missing"] or summary["undefined"]):
        parts[0] += f"（已记录 {summary['valid_count']}/{summary['total']}）"
    return "；".join(parts) or "未记录（无比较）"


def _table_rows(comparisons):
    """Preserve caller ordering and distinguish gate results from coverage."""
    result = []
    for label, records in comparisons:
        rows = list(records)
        if not all(isinstance(row, dict) for row in rows):
            raise TypeError("comparison rows must be dictionaries")
        metrics = [_metric_summary(rows, key) for key in _METRICS]
        passed = sum(row.get("passed") is True for row in rows)
        failed = sum(row.get("passed") is False for row in rows)
        unknown = len(rows) - passed - failed
        if not rows:
            status = "未测（0 条比较）"
        elif passed == len(rows):
            status = f"门槛通过（{passed}/{len(rows)}）"
        else:
            status = f"门槛未通过（通过 {passed}/{len(rows)}，失败 {failed}，未确认 {unknown}）"
        if rows and any(item["missing"] or item["undefined"] for item in metrics):
            status += "；指标不完整"
        result.append([str(label), *(_metric_text(item, key) for item, key in zip(metrics, _METRICS)), status])
    return result


def _markdown_cell(value):
    # Escape HTML as well: GitHub renders HTML inside Markdown table cells.
    return escape(value).replace("|", "&#124;").replace("\r", " ").replace("\n", " ")


def compact_metric_table(comparisons):
    """Render ``[(label, [tensor_comparison, ...]), ...]`` as Markdown.

    Only individual comparison flags determine this table's gate status. The
    caller must report its overall probe status (including execution failures)
    separately; this table is not a replacement for that status.
    """
    rows = _table_rows(comparisons)
    if not rows:
        return "暂无数值比较记录。"
    def render(row):
        return "| " + " | ".join(_markdown_cell(value) for value in row) + " |"

    return "\n".join([render(_HEADERS), render(["---"] * len(_HEADERS)), *(render(row) for row in rows)])


def compact_metric_table_html(comparisons):
    """HTML equivalent of :func:`compact_metric_table`, using identical rules."""
    rows = _table_rows(comparisons)
    if not rows:
        return "<p>暂无数值比较记录。</p>"
    header = "".join(f"<th>{escape(value)}</th>" for value in _HEADERS)
    body = "".join("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
