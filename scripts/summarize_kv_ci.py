"""Render this run's small-model KV evidence using only the standard library.

Historical reports under docs/ are deliberately never read. Reporting does not
replace the workflow's test exit codes or turn missing evidence into success.
"""
from __future__ import annotations

from html import escape
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

MAX_REPORT_BYTES = 16 * 1024 * 1024
SCOPE = ("完整 Qwen3-0.6B KV：**NOT RUN / 本次 CI 未执行**。"
         "本节仅汇总本次小模型验证；原始报告见 `w6-kv-cache` artifact。\n")


def _cell(value):
    return escape(" ".join(str(value).split())[:240]).replace("|", "&#124;")


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _json(path):
    with path.open("rb") as stream:
        data = stream.read(MAX_REPORT_BYTES + 1)
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError("报告过大")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("JSON 报告结构不符")
    return value


def _junit(path):
    with path.open("rb") as stream:
        data = stream.read(MAX_REPORT_BYTES + 1)
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError("JUnit 报告过大")
    cases = list(ET.fromstring(data).iter("testcase"))
    if not cases:
        raise ValueError("JUnit 未记录测试")
    counts = {name: sum(case.find(tag) is not None for case in cases)
              for name, tag in (("failed", "failure"), ("errors", "error"), ("skipped", "skipped"))}
    counts["passed"] = sum(not any(case.find(tag) is not None
                                  for tag in ("failure", "error", "skipped")) for case in cases)
    return counts


def _comparisons(rows, output_count):
    if not isinstance(rows, list) or not rows:
        raise ValueError("数值比较记录为空")
    for row in rows:
        values = row.get("max_abs_by_output") if isinstance(row, dict) else None
        if (not isinstance(values, list) or len(values) != output_count
                or not all(_number(value) for value in values)):
            raise ValueError("数值误差记录不完整或非有限")
    passed = all(row.get("passed") is True and max(row["max_abs_by_output"]) <= 1e-5 for row in rows)
    return passed, max(row["max_abs_by_output"][0] for row in rows), max(
        value for row in rows for value in row["max_abs_by_output"][1:])


def render(directory):
    directory = Path(directory)
    heading = "## KV cache\n\n"
    try:
        status = _json(directory / "w6-kv-status.json")
        if status.get("schema") != "scratchv.kv-ci-outcome.v1":
            raise ValueError("步骤记录版本不符")
        for field, variable in (("commit", "GITHUB_SHA"), ("run_id", "GITHUB_RUN_ID"),
                                ("run_attempt", "GITHUB_RUN_ATTEMPT")):
            if os.environ.get(variable) and status.get(field) != os.environ[variable]:
                raise ValueError("步骤记录不属于本次提交或运行")
        outcome = status.get("outcome")
        if outcome not in ("success", "failure", "cancelled", "skipped"):
            raise ValueError("步骤结果缺失或无效")
    except (OSError, ValueError, RecursionError) as exc:
        return heading + "**INCOMPLETE / 未取得本次执行记录**\n\n" + _cell(exc) + "\n\n" + SCOPE
    if outcome == "skipped":
        return heading + "**NOT RUN / KV 步骤未运行**\n\n" + SCOPE

    reports, problems = {}, []
    for name, path in (("tests", "w6-kv-tests.xml"), ("validation", "w6-kv-cache/validation.json"),
                       ("session", "w6-kv-cache/qemu-1/session.json"),
                       ("build", "w6-kv-cache/build-1/build.json")):
        try:
            reports[name] = _junit(directory/path) if name == "tests" else _json(directory/path)
        except (OSError, ValueError, ET.ParseError, RecursionError):
            problems.append(f"{path} 未生成或无法解析")
    validation = reports.get("validation", {})
    session = reports.get("session", {})
    build = reports.get("build", {})
    tests = reports.get("tests", {})
    layout = validation.get("layout", {})
    metrics = []
    greedy_passed = False
    try:
        if (validation.get("schema") != "scratchv.kv-export.v1"
                or validation.get("checkpoint") != "seeded-small" or not isinstance(layout, dict)
                or layout.get("layers") != 2 or layout.get("capacity") != 256
                or session.get("schema") != "scratchv.kv-session.v1"
                or session.get("cores") != 1 or session.get("cache_layout") != layout
                or build.get("cores") != 1 or build.get("cache_layout") != layout):
            raise ValueError("小模型配置、核数或报告接口不一致")
        compiled = validation.get("compiled")
        if not isinstance(compiled, list) or not all(isinstance(row, dict) for row in compiled):
            raise ValueError("编译后检查记录无效")
        groups = [("参考链（PyTorch / ORT / 完整重算）", validation.get("reference"))]
        groups += [(label, [row for row in compiled if str(row.get("case", "")).startswith(prefix)
                            and "max_abs_by_output" in row])
                   for label, prefix in (("IR / ORT", "ir-"), ("Host C / ORT", "host-"),
                                         ("QEMU / ORT", "qemu-1-"))]
        for label, rows in groups:
            passed, logits, kv = _comparisons(rows, 1 + 2*layout["layers"])
            metrics.append((label, passed, len(rows), logits, kv))
        greedy = [row for row in compiled if row.get("case") == "qemu-1-greedy-generation"]
        greedy_passed = (len(greedy) == 1 and greedy[0].get("passed") is True
                         and isinstance(greedy[0].get("generated_token_ids"), list)
                         and len(greedy[0]["generated_token_ids"]) == 4)
        if not greedy_passed:
            raise ValueError("四 token 生成结果未通过或记录不完整")
    except (TypeError, KeyError, ValueError) as exc:
        problems.append(str(exc))

    requests = session.get("requests", [])
    if (not isinstance(requests, list) or not requests
            or not all(isinstance(row, dict) and row.get("passed") is True for row in requests)):
        problems.append("QEMU 请求记录缺失或未全部成功")
        requests = []
    success = (outcome == "success" and not problems and bool(tests.get("passed"))
               and not tests.get("failed") and not tests.get("errors")
               and validation.get("passed") is True and len(metrics) == 4
               and all(row[1] for row in metrics) and session.get("passed") is True
               and session.get("state") == "CLOSED" and session.get("qemu_reaped") is True)
    result = ("PASS" if success else "CANCELLED" if outcome == "cancelled" else
              "FAIL" if outcome == "failure" or not problems else "INCOMPLETE")
    lines = [heading, f"**{result}**（CI 步骤：{outcome}）\n\n", SCOPE, "\n"]
    if problems:
        lines.append("未完成证据：" + "；".join(_cell(problem) for problem in problems) + "。\n\n")
    if tests:
        lines.append(f"单元与生命周期测试：通过 {tests['passed']}，失败 {tests['failed']}，"
                     f"错误 {tests['errors']}，跳过 {tests['skipped']}。\n\n")
    if layout:
        lines.append("模型：两层随机权重 Qwen3，FP32，batch=1，容量 256；目标为单核 RV64 QEMU。\n\n")
    if metrics:
        lines += ["| 数值对照 | 结果 / 比较数 | logits 最大绝对误差 | K/V 最大绝对误差 |\n",
                  "| --- | --- | --- | --- |\n"]
        for label, passed, count, logits, kv in metrics:
            lines.append(f"| {label} | {'PASS' if passed else 'FAIL'} / {count} | {logits:.9e} | {kv:.9e} |\n")
        lines.append("\n小模型阈值为绝对误差 1e-5；参考链同时包含 PyTorch/ORT 和缓存/完整重算对照，"
                     "IR/Host C/QEMU 的 K/V 按每个 head 的有效前缀比较。\n\n")
    lines.append(f"四 token 生成一致性：{'通过' if greedy_passed else '未确认'}。\n\n")
    timings = [("KV probe 总耗时（含导出、参考、编译及审计）", validation.get("seconds")),
               ("C → RV64 编译", build.get("compile_seconds")),
               ("QEMU 启动与权重加载", session.get("startup_s"))]
    for kind, label in ((2, "Prefill"), (3, "Decode")):
        rows = [row for row in requests if row.get("kind") == kind]
        if rows and all(_number(row.get("forward_s")) for row in rows):
            total = sum(row["forward_s"] for row in rows)
            timings.append((f"{label} 合计（{len(rows)} 次，平均 {total/len(rows):.6f} s）", total))
    lines += ["| 耗时项 | 秒 |\n", "| --- | --- |\n"]
    lines += [f"| {label} | {value:.6f} |\n" if _number(value) else f"| {label} | 未记录 |\n"
              for label, value in timings]
    memory = validation.get("memory", {})
    memory = memory if isinstance(memory, dict) else {}
    bank = memory.get("cache_bank_bytes")
    sizes = [("Prefill 工作区", memory.get("prefill_workspace_bytes")),
             ("Decode 工作区", memory.get("decode_workspace_bytes")),
             ("两块缓存区", 2*bank if type(bank) is int and bank >= 0 else None),
             ("采样 QEMU 进程峰值 RSS", session.get("sampled_peak_qemu_rss_bytes"))]
    lines += ["\n| 内存项 | 字节 |\n", "| --- | --- |\n"]
    lines += [f"| {label} | {value:,} |\n" if type(value) is int and value >= 0
              else f"| {label} | 未记录 |\n" for label, value in sizes]
    cleanup = session.get("state") == "CLOSED" and session.get("qemu_reaped") is True
    lines.append(f"\nQEMU 进程回收：{'完成' if cleanup else '未确认'}。"
                 "Prefill/Decode 请求时间包含通信和 guest 检查；总耗时包含缓存读取审计。"
                 "进程 RSS 与 guest 内存布局不同，本节不计算加速比。\n")
    return "".join(lines)
