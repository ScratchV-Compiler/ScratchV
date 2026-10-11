"""Publish bounded W3 numerical reports to GitHub Actions without changing gates."""
from __future__ import annotations

import argparse
from html import escape
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.numeric_summary import compact_metric_table
from probes.w3_summary import evidence_index_views, numerical_summary

# The medium report currently exceeds 12 MiB because it retains checkpoints.
# Bound input allocation separately from the much smaller Actions page budget.
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_SUMMARY_BYTES = 64 * 1024

GATES = (
    ("medium", "w3-medium-host", "6 层随机 Qwen3 / 宿主 IR"),
    ("subgraphs", "w3_qwen3_subgraphs", "预训练 layer-0 子图 / 宿主 IR"),
    ("attention", "unit:attention-backend", "小规模 Attention / RISC-V QEMU"),
    ("full-preflight", "preflight:w3-full-assets", "完整资产 / 仅静态检查"),
    ("layer-diff", "w3_layer_diff", "仅已保存的 medium 模型检查点"),
)


def cell(value, limit=180):
    text = " ".join(str(value).split())
    if len(text) > limit:
        text = text[:limit] + "…"
    return escape(text).replace("|", "&#124;")


def load_report(path, gate):
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_REPORT_BYTES + 1)
        if len(payload) > MAX_REPORT_BYTES:
            return None, "INVALID REPORT (size limit)"
        report = json.loads(payload.decode("utf-8"))
    except FileNotFoundError:
        return None, "MISSING REPORT"
    except (OSError, ValueError, RecursionError):
        return None, "INVALID REPORT"
    if not isinstance(report, dict) or report.get("gate") != gate:
        return None, "INVALID REPORT (gate identity)"
    # Fail closed on malformed structures before calling the shared renderer.
    for key in ("cases", "gates", "invariants"):
        rows = report.get(key, [])
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            return None, "INVALID REPORT (rows)"
    return report, ""


def status(report):
    if report.get("status") == "PARTIAL":
        return "PARTIAL (not acceptance)"
    if report.get("status") == "PASS" and report.get("passed") is True:
        if report.get("gate") == "numeric:ir-full-qwen3" and report.get("coverage_complete") is not True:
            return "INCOMPLETE (not acceptance)"
        return "PASS"
    return "FAIL" if report.get("passed") is False else "UNCONFIRMED"


def seconds(report):
    value = report.get("elapsed_seconds", report.get("seconds"))
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return f"{value:.3f} s"
    return "未记录"


def coverage(report):
    gate = report.get("gate")
    cases = report.get("cases", [])
    if gate == "w3_layer_diff":
        values = report.get("coverage", {})
        return f"{values.get('compared_checkpoints', '?')}/{values.get('total_checkpoints', '?')} checkpoints"
    if gate == "preflight:w3-full-assets":
        return "不执行模型"
    if gate == "unit:attention-backend":
        count = sum(len(row.get("executions", [])) for row in cases)
        return f"{len(cases)} cases / {count} target executions 已记录"
    required = report.get("required_cases")
    expected = len(required) if isinstance(required, list) else {"w3-medium-host": 7, "w3_qwen3_subgraphs": 14}.get(gate, "?")
    return f"{len(cases)}/{expected} cases 已记录"


def details(report, title):
    """Reuse bounded semantic selectors, never dump arbitrary report dictionaries."""
    description, comparisons, notes = numerical_summary(report)
    index, _ = evidence_index_views(report)
    text = f"<details><summary>{cell(title)}</summary>\n\n{cell(description, 700)}\n\n"
    if comparisons:
        text += compact_metric_table(comparisons) + "\n\n"
    if report.get("error"):
        text += "问题: " + cell(report["error"], 400) + "\n\n"
    text += "\n\n".join(cell(note, 500) for note in notes) + "\n\n" + index + "</details>\n\n"
    return text


def render(kind, directory, outcome):
    artifact = "w3-preparation-reports" if kind == "preparation" else "w3-full-numeric-reports"
    evidence = (f"产物: **{artifact}**（`report.json`、`report.md`、`report.html`）。"
                "原始数组在保留时单独存放。本概要不能替代独立复现或团队验收。\n")
    lines = ["## W3 测试项\n\n"]
    if outcome in ("skipped", "", "cancelled"):
        label = "CANCELLED" if outcome == "cancelled" else "NOT RUN"
        lines.append(f"**{label}** — 探针步骤未完成，未声称数值验收通过。\n\n")
        lines.append("既有文件即使存在，也不作为本步骤的结果。\n")
        return "".join(lines) + evidence
    identity = "preparation:w3" if kind == "preparation" else "numeric:ir-full-qwen3"
    report, problem = load_report(directory / "report.json", identity)
    if report is None:
        lines.append(f"**{problem}** — 步骤结果: **{cell(outcome)}**。未声称验收通过，请检查步骤日志。\n")
        return "".join(lines) + evidence
    conclusion = status(report)
    if outcome != "success" and conclusion == "PASS":
        conclusion = "FAIL (step did not succeed)"
    lines.append(f"**{conclusion}** · 步骤: {cell(outcome)} · 流水线耗时: {seconds(report)}\n\n")
    comparisons, folded = [], []
    try:
        if kind == "preparation":
            lines.append("仅为准备；完整 28 层 IR 数值与完整模型 QEMU **在本 job 中不运行**。\n\n")
            lines.append("| 检查 / 范围 | 结果 | 记录覆盖 | 流水线耗时 |\n| --- | --- | --- | --- |\n")
            for name, identity, scope in GATES:
                rows = [row for row in report.get("gates", []) if row.get("name") == name]
                row = rows[0] if len(rows) == 1 else None
                child, problem = load_report(directory / name / "report.json", identity) if row else (None, "NOT RUN" if not rows else "INVALID REPORT (duplicate gate)")
                child_status = problem or status(child)
                if row and status(row) != "PASS" and child_status == "PASS":
                    child_status = "FAIL (parent validation)"
                if child_status != "PASS" and conclusion == "PASS":
                    lines[1] = f"**INCOMPLETE REPORTS (not acceptance)** · 步骤: {cell(outcome)} · 流水线耗时: {seconds(report)}\n\n"
                lines.append("| " + " | ".join(cell(x) for x in (scope, child_status, coverage(child) if child else "未记录", seconds(row) if row else "未记录")) + " |\n")
                if child:
                    _, numeric, _ = numerical_summary(child)
                    selected = numeric[1:] if name in ("medium", "subgraphs") else numeric[1:2] if name == "attention" else []
                    records = [item for _, items in selected for item in items]
                    if selected:
                        comparisons.append((scope + " vs ORT", records))
                    folded.append(details(child, scope + " — details"))
            folded.insert(0, details(report, "Preparation execution / failures"))
        else:
            lines.append("28 层预训练 Qwen3，FP32 / L=256 / 无 KV cache；IR vs ORT。**本 job 无完整模型 QEMU 结果。**\n\n")
            lines.append(f"覆盖: **{cell(coverage(report))}** · FP32 模式: **{cell(report.get('fp32_mode', '未记录'))}** · 最大绝对误差阈值: **{cell(report.get('atol', '未记录'))}**（严格 &lt;，rtol=0）。\n\n")
            display = dict(report)
            recorded = {row.get("name") for row in report.get("cases", [])}
            display["cases"] = list(report.get("cases", [])) + [
                {"name": name, "status": "NOT RUN"} for name in report.get("required_cases", [])
                if name not in recorded]
            _, comparisons, _ = numerical_summary(display)
            folded.append(details(display, "用例、诊断、不变量与失败"))
        if comparisons:
            lines.append("\n" + compact_metric_table(comparisons) + "\n\n")
            lines.append("指标只汇总最终输出（已记录比较中最差 MaxAbs / 相对 L2、最低余弦）；检查点仍按原门槛判定。缺失指标保持显式。\n\n")
        if report.get("error"):
            lines.append("问题: " + cell(report["error"], 300) + "\n\n")
        lines.extend(folded)
    except (TypeError, KeyError, ValueError, AttributeError, RecursionError):
        # Unexpected report schemas must not make an apparently passing page.
        return f"## W3 测试项\n\n**INVALID REPORT** — 无法渲染数值证据；步骤: {cell(outcome)}。未声称验收通过。\n\n" + evidence
    lines.append(evidence)
    rendered = "".join(lines)
    if len(rendered.encode("utf-8")) > MAX_SUMMARY_BYTES:
        # Do not slice Markdown or HTML: a metric's free-form reason can be
        # enormous even when report structure and all numeric values are valid.
        # Retain the authoritative, already bounded status (including parent
        # validation failures) and replace the entire detail body atomically.
        return ("".join(lines[:2]) +
                "数值详情超出 Actions 页面 64 KiB 的概要预算，此处省略。"
                "上方记录的门槛结果不变；完整指标与范围请查看产物。\n\n" + evidence)
    return rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("preparation", "full"), required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--outcome", default=os.environ.get("PROBE_OUTCOME", ""))
    parser.add_argument("--summary", type=Path, default=os.environ.get("GITHUB_STEP_SUMMARY"))
    args = parser.parse_args(argv)
    text = render(args.kind, args.report_dir, args.outcome)
    if args.summary:
        with args.summary.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    else:
        print(text, end="")
    return 0  # Reporting never substitutes for or changes the probe's exit code.


if __name__ == "__main__":
    raise SystemExit(main())
