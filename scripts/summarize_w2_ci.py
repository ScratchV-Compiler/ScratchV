"""Publish the W2 operator, precision and timing tables to the Actions summary.

The W2 jobs run their own probes and publish raw evidence. This script is the
one place that answers what the W2 summary is for:

* which operators each layer supports and which were exercised this run;
* how far the numbers are from ORT and the independent NumPy references;
* how long the whole chain takes, per model and in total.

Everything else belongs in the step log, not on the Actions page. Reporting
never substitutes for, or changes, a probe's exit code.

The acceptance job installs nothing, so this module imports only the standard
library plus ``probes.numeric_summary``. The operator lists are read from the
compiler sources with ``ast`` rather than imported, so no ``scratchv`` or
``onnx`` import happens here.
"""
from __future__ import annotations

import argparse
import ast
from html import escape
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.numeric_summary import METRIC_EXPLANATION, compact_metric_table

# Inputs are bound separately from the much smaller Actions page budget.
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_SUMMARY_BYTES = 64 * 1024

FRONTEND_SOURCE = ROOT / "scratchv/frontend/onnx_parser.py"
BACKEND_SOURCE = ROOT / "scratchv/backend/tensor_c_codegen.py"

# Probe directories under --report-dir, in the order the chain runs.
DIRECTORIES = ("w2-backend-ops", "w2-runtime", "w2-runtime-model",
               "qwen3-small", "qwen3-riscv", "qwen3-parse")
BACKEND_OPS, RUNTIME, RUNTIME_MODEL, QWEN3_SMALL, QWEN3_RISCV, QWEN3_PARSE = DIRECTORIES

# probes/w2_backend_ops/cases.py builds these decomposed primitives, one graph
# per case, and names each case after its family. Only families with a passing
# execution may mark the assembly layer as measured this run.
FAMILY_OPS = {
    "matmul": ("matmul",),
    "elementwise": ("add", "sub", "mul", "div"),
    "softmax": ("softmax",),
    "rmsnorm": ("pow", "reduce_mean", "add", "sqrt", "div", "mul"),
    "rope": ("mul", "concat", "cos", "sin", "unsqueeze", "slice", "neg", "add"),
    "swiglu": ("matmul", "sigmoid", "mul"),
    "gqa": ("unsqueeze", "expand", "reshape", "transpose", "matmul", "mul", "add", "softmax"),
}


def cell(value, limit=180):
    """One Markdown table cell: single line, bounded, HTML escaped."""
    text = " ".join(str(value).split())
    if len(text) > limit:
        text = text[:limit] + "…"
    return escape(text).replace("|", "&#124;")


def opcode_name(op_type):
    """The IR opcode value the frontend lowers an ONNX op type to."""
    lowered = str(op_type).lower()
    return "reduce_mean" if lowered == "reducemean" else lowered


def _parse(path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError, RecursionError):
        return None


def frontend_operators():
    """ONNX op types the frontend accepts, from its ``_handle_<op>`` methods."""
    tree = _parse(FRONTEND_SOURCE)
    if tree is None:
        return []
    names = {node.name[len("_handle_"):] for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name.startswith("_handle_")}
    return sorted(names)


def backend_opcodes():
    """OpCode values the tensor C backend lowers, from ``TensorCCodegen._infer``."""
    tree = _parse(BACKEND_SOURCE)
    if tree is None:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_infer":
            found = {child.attr for child in ast.walk(node)
                     if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name)
                     and child.value.id == "OpCode"}
            return sorted(attr.lower() for attr in found)
    return []


def load_report(directory, name):
    """Return ``(report, problem)``; a missing or malformed report is a problem.

    Three of the W2 probes publish no ``gate`` field, so identity is the
    directory plus the ``passed`` key every one of them records.
    """
    path = directory / name / "report.json"
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_REPORT_BYTES + 1)
        if len(payload) > MAX_REPORT_BYTES:
            return None, "报告过大"
        report = json.loads(payload.decode("utf-8"))
    except FileNotFoundError:
        return None, "报告缺失"
    except (OSError, ValueError, UnicodeDecodeError, RecursionError):
        return None, "报告无法解析"
    if not isinstance(report, dict) or "passed" not in report:
        return None, "报告结构不符"
    return report, ""


def _cases(report):
    rows = report.get("cases", []) if isinstance(report, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _rows(record, key):
    value = record.get(key, []) if isinstance(record, dict) else []
    return [row for row in value if isinstance(row, dict)]


def _seconds(value):
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def _onnx_operators(report, name):
    """ONNX op types the report recorded for a parsed model."""
    if name == QWEN3_SMALL:
        return set(_mapping(_mapping(report.get("onnx")).get("operators")))
    graph = _mapping(report.get("graph"))
    inner = _mapping(graph.get("graph")) or graph
    return set(_mapping(inner.get("operators")))


def exercised_operators(reports):
    """``(frontend_op_types, backend_opcodes)`` actually exercised in this run."""
    onnx_seen = set()
    for name in (QWEN3_SMALL, QWEN3_PARSE):
        if reports.get(name) is not None:
            onnx_seen |= _onnx_operators(reports[name], name)

    opcodes = set()
    # The two-layer model is the only graph the RISC-V QEMU probe compiles, so
    # its opcodes count as measured only once a QEMU execution passed.
    riscv = reports.get(QWEN3_RISCV)
    if riscv is not None and any(row.get("passed") is True
                                 for case in _cases(riscv) for row in _rows(case, "qemu")):
        if reports.get(QWEN3_SMALL) is not None:
            opcodes |= {opcode_name(op) for op in _onnx_operators(reports[QWEN3_SMALL], QWEN3_SMALL)}
    # The operator probe records only its families; a passing execution proves them.
    for case in _cases(reports.get(BACKEND_OPS)):
        family = str(case.get("name", "")).split("_", 1)[0]
        if any(row.get("passed") is True for row in _rows(case, "executions")):
            opcodes |= set(FAMILY_OPS.get(family, ()))
    return {str(op).lower() for op in onnx_seen}, opcodes & set(backend_opcodes())


def operator_table(reports):
    supported_frontend, supported_backend = frontend_operators(), backend_opcodes()
    exercised_frontend, exercised_backend = exercised_operators(reports)

    def listing(names):
        return f"{len(names)} 个：" + (", ".join(names) if names else "无")

    def measured(names, exercised):
        if not names:
            return "无"
        hit = [name for name in names if name.lower() in exercised]
        return listing(hit) if hit else "无（本次未执行）"

    rows = [
        ("IR 层 (`ONNXParser._handle_*`)", listing(supported_frontend),
         measured(supported_frontend, exercised_frontend)),
        ("汇编层 (`tensor_c_codegen`)", listing(supported_backend),
         measured(supported_backend, exercised_backend)),
    ]
    lines = ["| 层 | 代码里支持的算子 | 本次实测过的 |\n", "| --- | --- | --- |\n"]
    lines.extend("| " + " | ".join(cell(value, 4000) for value in row) + " |\n" for row in rows)
    lines.append("\n旧的标量后端（`llvm_codegen`、`instruction_select`）不接受带 shape 的张量图"
                 "（`scratchv/compiler.py:527-530` 会拒绝），所以不计入汇编层。\n")
    return lines


def precision_legend():
    """One line per name, saying what actually produced the numbers."""
    return [
        "- **ORT**：ONNX Runtime 在 CPU 上跑同一个 ONNX 模型。\n",
        "- **IR**：前端解析出的 IR，由 IR 解释器在 host 上跑。\n",
        "- **QEMU**：同一张图编成 RV64 的 ELF，在 QEMU 里跑。\n",
        "- **NumPy**：用 NumPy 手写公式算出的期望值（只有 w2-backend-ops 有）。\n",
        "- **最终 logits**：模型最后一层的输出，报告里名为 `logits` 的那个。\n",
        "\n",
    ]


def precision_comparisons(reports):
    comparisons = []

    def add(label, records):
        comparisons.append((label, [row for row in records if isinstance(row, dict)]))

    backend = reports.get(BACKEND_OPS)
    if backend is not None:
        executions = [row for case in _cases(backend) for row in _rows(case, "executions")]
        for key, label in (("ir_vs_ort", "IR vs ORT"), ("ir_vs_numpy", "IR vs NumPy"),
                           ("qemu_vs_ort", "QEMU vs ORT"), ("qemu_vs_numpy", "QEMU vs NumPy")):
            add(f"w2-backend-ops / {label}", [row.get(key) for row in executions])

    if reports.get(QWEN3_SMALL) is not None:
        # The host IR comparison wraps one tensor_diff per checkpoint; report
        # the final output, which is what the probe's own gate compares.
        records = []
        for case in _cases(reports[QWEN3_SMALL]):
            checkpoints = _rows(_mapping(case.get("ir_vs_ort")), "checkpoints")
            final = [row for row in checkpoints if row.get("name") == "logits"]
            if final:
                records.extend(final)
            elif isinstance(case.get("ordinary_logits"), dict):
                records.append(case["ordinary_logits"].get("ir_vs_ort"))
        add("qwen3-small host IR / IR vs ORT（最终 logits）", records)

    if reports.get(QWEN3_RISCV) is not None:
        add("qwen3-small RV64 QEMU / QEMU vs ORT",
            [row for case in _cases(reports[QWEN3_RISCV]) for row in _rows(case, "qemu")])

    if reports.get(RUNTIME_MODEL) is not None:
        executions = [row for case in _cases(reports[RUNTIME_MODEL]) for row in _rows(case, "executions")]
        add("w2-runtime-model / vs torch", [row.get("vs_torch") for row in executions])
        add("w2-runtime-model / vs ORT", [row.get("vs_ort") for row in executions])

    # These two chains publish structural evidence only; say so instead of omitting them.
    add("w2-runtime（文本层，无数值比较）", [])
    add("qwen3-parse（完整 28 层前端，无数值比较）", [])
    return comparisons


def timing_rows(loaded):
    rows = []

    def add(label, name, pick):
        report, problem = loaded.get(name, (None, "报告缺失"))
        if report is None:
            rows.append({"label": label, "status": problem, "seconds": None, "note": ""})
            return
        note = ""
        if name == QWEN3_RISCV:
            timing = _mapping(report.get("timing"))
            parts = []
            if _seconds(timing.get("cross_compile_seconds_total")) is not None:
                parts.append(f"交叉编译 {timing['cross_compile_seconds_total']:.3f} s")
            if _seconds(timing.get("qemu_process_wall_seconds_total")) is not None:
                parts.append(f"QEMU 合计 {timing['qemu_process_wall_seconds_total']:.3f} s")
            note = "；".join(parts)
        rows.append({"label": label, "status": "PASS" if report.get("passed") is True else "FAIL",
                     "seconds": _seconds(pick(report)), "note": note})

    add("w2-backend-ops 算子内核", BACKEND_OPS, lambda report: report.get("seconds"))
    add("w2-runtime 文本层", RUNTIME, lambda report: report.get("seconds"))
    add("w2-runtime-model 文本层↔小模型", RUNTIME_MODEL, lambda report: report.get("seconds"))
    add("qwen3-small host IR", QWEN3_SMALL, lambda report: report.get("seconds"))
    add("qwen3-small RV64 QEMU", QWEN3_RISCV,
        lambda report: _mapping(report.get("timing")).get("pipeline_seconds", report.get("seconds")))
    add("qwen3-parse 完整 28 层前端", QWEN3_PARSE, lambda report: report.get("seconds"))
    return rows


def timing_table(rows):
    lines = ["| 链路 | 步骤结果 | 耗时 |\n", "| --- | --- | --- |\n"]
    for row in rows:
        seconds = "未记录" if row["seconds"] is None else f"{row['seconds']:.3f} s"
        if row["note"]:
            seconds += f"（{row['note']}）"
        lines.append("| " + " | ".join(cell(value) for value in (row["label"], row["status"], seconds)) + " |\n")
    measured = [row for row in rows if row["seconds"] is not None]
    missing = [row["label"] for row in rows if row["seconds"] is None]
    total = f"{sum(row['seconds'] for row in measured):.3f} s"
    if missing:
        total += f"；不完整（缺：{', '.join(missing)}）"
    lines.append("| **总计** | | " + cell(total, 400) + " |\n")
    lines.append("\nQEMU 秒数含进程启动、guest 计算、UART 传输和退出，不是纯计算耗时；"
                 "交叉编译只统计成功的 C→RV64 构建。总计是各链路相加，而两条流水线曾并行运行，"
                 "不等于挂钟时间。\n")
    return lines


def render(directory):
    loaded = {name: load_report(directory, name) for name in DIRECTORIES}
    reports = {name: report for name, (report, _) in loaded.items()}
    try:
        operators = operator_table(reports)
        precision = compact_metric_table(precision_comparisons(reports))
        timing = timing_table(timing_rows(loaded))
    except (TypeError, KeyError, ValueError, AttributeError, RecursionError):
        # A malformed report must not turn into an apparently complete page.
        return ("## W2 测试项\n\n**报告结构不符** — 无法渲染算子、精度与耗时；"
                "请检查各 W2 job 的日志与产物。\n")

    lines = ["## W2 测试项\n\n", "### 算子支持\n\n", *operators,
             "\n### 精度\n\n", *precision_legend(), precision, "\n\n", METRIC_EXPLANATION, "\n\n",
             "### 耗时\n\n", *timing]
    rendered = "".join(lines)
    if len(rendered.encode("utf-8")) > MAX_SUMMARY_BYTES:
        return ("".join(lines[:1]) +
                "内容超出 Actions 页面 64 KiB 的 summary budget，此处省略；"
                "完整证据见各 W2 job 的产物。\n")
    return rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("output"))
    parser.add_argument("--summary", type=Path, default=os.environ.get("GITHUB_STEP_SUMMARY"))
    args = parser.parse_args(argv)
    text = render(args.report_dir)
    if args.summary:
        with args.summary.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    else:
        print(text, end="")
    return 0  # Reporting never substitutes for or changes a probe's exit code.


if __name__ == "__main__":
    raise SystemExit(main())
