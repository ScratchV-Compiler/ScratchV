"""Render bounded W4 Actions summaries from this run's evidence (stdlib only)."""
from __future__ import annotations

import argparse
from html import escape
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET


NOT_RUN = "NOT RUN（未运行）"
MISSING = "MISSING（报告缺失或无效）"


def cell(value):
    text = " ".join(str(value).split())[:240]
    return escape(text).replace("|", "&#124;").replace("`", "&#96;")


def table(headers, rows):
    return "\n".join("| " + " | ".join(cell(value) for value in row) + " |"
                     for row in [headers, ["---"] * len(headers), *rows])


def number(value, *, integer=False):
    if type(value) not in (int, float):
        return "未记录"
    try:
        if not math.isfinite(value) or value < 0 or (integer and int(value) != value):
            return "无效"
    except OverflowError:
        return "无效"
    return f"{int(value):,}" if integer else f"{value:.9g}"


def mapping(value):
    return value if isinstance(value, dict) else {}


def records(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def exit_code(value):
    # Negative POSIX return codes identify a terminating signal.
    return str(value) if type(value) is int else "未记录"


def read_report(path):
    try:
        if path.stat().st_size > 16 * 1024 * 1024:
            return {}
        return mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, RecursionError):
        return {}


def outcome_status(outcome, report=None):
    if outcome == "skipped":
        return NOT_RUN
    if outcome in ("failure", "cancelled"):
        return "FAIL" if outcome == "failure" else "CANCELLED（已取消）"
    if outcome != "success":
        return "UNKNOWN（执行状态缺失）"
    if not report:
        return MISSING
    return "PASS" if report.get("passed") is True and report.get("status") == "PASS" else "FAIL / 未完成"


def unit_summary(path, outcome):
    """Count actual JUnit cases, including skipped cases; never infer zero passes."""
    counts = dict(passed=0, failed=0, errors=0, skipped=0)
    status = outcome_status(outcome)
    available = False
    if outcome != "skipped":
        try:
            root = ET.parse(path).getroot()
            cases = list(root.iter("testcase"))
            available = True
            for case in cases:
                key = ("errors" if case.find("error") is not None else
                       "failed" if case.find("failure") is not None else
                       "skipped" if case.find("skipped") is not None else "passed")
                counts[key] += 1
            valid = bool(cases) and not counts["failed"] and not counts["errors"] and counts["passed"] > 0
            status = outcome_status(outcome, {"passed": valid, "status": "PASS" if valid else "FAIL"})
        except (OSError, ET.ParseError):
            pass
    values = [counts[key] for key in ("passed", "failed", "errors", "skipped")]
    if not available or status in (NOT_RUN, MISSING, "UNKNOWN（执行状态缺失）"):
        values = ["—"] * 4
    return (f"## W4 单测与真实小图 QEMU：{status}\n\n"
            + table(["通过", "失败", "错误", "跳过"], [values])
            + ("\n\nJUnit 报告缺失或无效，未取得测试计数。" if not available and outcome != "skipped" else "")
            + "\n\n范围：外置权重 ABI、装载和小图回归；不代表完整 28 层前向通过。"
              "详细用例见本次运行的 `w4-unit-*` artifact / `tests.xml`。\n")


def full_summary(report, audit, *, probe_outcome, gate_outcome, audit_outcome, commit, case):
    # A skipped probe must not display metrics from an old or unrelated report.
    if probe_outcome == "skipped":
        report, audit = {}, {}
    status = outcome_status(probe_outcome, report)
    identity = mapping(report.get("git"))
    identity_ok = bool(commit) and identity.get("head") == commit and identity.get("dirty") is False
    if status == "PASS" and not identity_ok:
        status = "INVALID（报告来源与本次提交不符）"
    if status == "PASS" and gate_outcome != "success":
        status = "未验收（完整门禁未通过）"
    audit_status = outcome_status(audit_outcome, audit)
    if status == "PASS" and audit_status != "PASS":
        status = "未验收（数组审计未通过）"
    gates = mapping(report.get("gates"))
    gate_rows = []
    for key in ("build:riscv-full", "smoke:qemu-full", "numeric:qemu-full-qwen3"):
        flag = gates.get(key)
        value = "PASS" if flag is True else "未完成" if flag is False else NOT_RUN
        gate_rows.append([key, value])
    fma = mapping(report.get("fma_conformance"))
    gate_rows += [["FMA 位模式 / FCSR", outcome_status("success", fma) if fma else NOT_RUN],
                  ["完整来源与数值门禁", "PASS" if gate_outcome == "success" else outcome_status(gate_outcome)],
                  ["保留数组审计", audit_status]]
    cases = records(report.get("cases"))
    # Prioritize failures and cap presentation only; machine evidence stays intact.
    selected = sorted(cases, key=lambda row: row.get("passed") is True)[:7]
    numeric_rows, run_rows = [], []
    for row in selected:
        numeric, run = mapping(row.get("numeric")), mapping(row.get("qemu"))
        numeric_rows.append([row.get("name", "未命名"), number(numeric.get("max_abs")),
            number(numeric.get("rmse")), number(numeric.get("elements_compared"), integer=True),
            number(numeric.get("elements_at_or_above_atol"), integer=True),
            "PASS" if numeric.get("passed") is True else "未通过 / 未运行"])
        run_rows.append([row.get("name", "未命名"), number(run.get("qemu_wall_s")),
                         number(run.get("forward_wall_s")), exit_code(run.get("exit_code"))])
    if not numeric_rows:
        numeric_rows = [[case, "未测", "未测", "未测", "未测", NOT_RUN]]
    build = mapping(report.get("build"))
    lines = [f"## W4 完整 Qwen3 RV64 前向：{status}", "",
        f"本次提交：`{cell(commit)}`；请求用例：`{cell(case)}`。", "",
        "完整输出门槛：所有 38,895,616 个 FP32 logits 的 MaxAbs < 1e-3，rtol=0。", "",
        table(["用例", "MaxAbs", "RMSE", "比较元素", "超限元素", "数值结果"], numeric_rows), "",
        f"不含权重 ELF：{number(build.get('bytes'), integer=True)} bytes（要求 < 100,000,000）；"
        f"流程墙钟：{number(report.get('elapsed_seconds'))} s。", "",
        "QEMU 进程墙钟包含启动、执行和输出转储，不是芯片性能指标。", "",
        table(["用例", "QEMU 墙钟 / s", "到 guest 完成 / s", "退出码"], run_rows or [[case, "未测", "未测", "未测"]]), "",
        "<details><summary>门禁、阶段耗时与证据入口</summary>", "", table(["门禁", "结果"], gate_rows), "",
        f"报告阶段：{cell(report.get('stage', '未记录'))}；报告提交：{cell(identity.get('head', '未记录'))}。"]
    stages = mapping(report.get("stages_seconds"))
    if stages:
        lines += ["", table(["阶段", "墙钟 / s"], [[name, number(value)] for name, value in list(stages.items())[:25]])]
    for label, item in (("探测", report), ("审计", audit)):
        if item.get("error") or item.get("report_error"):
            lines += ["", f"{label}错误：{cell(item.get('error') or item.get('report_error'))}"]
    lines += ["", "完整报告见本次运行的 `w4-full-reports-*` artifact；原始数组在 `w4-full-raw-*`。",
        "Summary 只展示已有报告，不重新计算误差；失败时已显示的数值是部分证据。",
        "", "</details>", "", "范围：请求的单次前向；不代表七组覆盖、团队验收或第二人独立复现。", ""]
    return "\n".join(lines)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--kind", choices=("unit", "full"), required=True)
    cli.add_argument("--report-dir", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args(argv)
    if args.kind == "unit":
        text = unit_summary(args.report_dir / "tests.xml", os.environ.get("UNIT_OUTCOME", ""))
    else:
        text = full_summary(read_report(args.report_dir / "report.json"),
            read_report(args.report_dir.with_name(args.report_dir.name + "-audit") / "report.json"),
            probe_outcome=os.environ.get("PROBE_OUTCOME", ""), gate_outcome=os.environ.get("GATE_OUTCOME", ""),
            audit_outcome=os.environ.get("AUDIT_OUTCOME", ""), commit=os.environ.get("GITHUB_SHA", ""),
            case=os.environ.get("SELECTED_CASE", "未记录"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
