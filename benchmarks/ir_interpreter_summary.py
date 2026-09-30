"""Correctness summaries for the interpreter benchmark and CI report."""

import numpy as np

from benchmarks.ir_interpreter_cases import REFERENCE_METHODS
from scratchv.ir.types import OpCode
from scratchv.verification.ir_interpreter import IRExecutionError


def describe_case(case):
    return {
        "case": case.name,
        "opcodes": sorted(
            {
                i.opcode.value
                for f in case.program.functions
                for b in f.blocks
                for i in b.instructions
                if i.opcode != OpCode.RETURN
            }
        ),
        "expected_shape": list(case.expected.shape),
        "expected_dtype": str(case.expected.dtype),
        "reference_method": REFERENCE_METHODS.get(case.name, "调用者独立提供"),
        "comparison": (
            "exact" if np.issubdtype(case.expected.dtype, np.integer) else "tolerance"
        ),
        "atol": case.atol,
        "rtol": case.rtol,
    }


def describe_failure(exc):
    row = {"error": f"{type(exc).__name__}: {exc}"}
    if isinstance(exc, IRExecutionError):
        row["error_code"] = exc.code
        row["error_location"] = {
            "function": exc.function_name,
            "block": exc.block_name,
            "instruction_index": exc.instruction_index,
            "opcode": exc.opcode.value if exc.opcode else None,
            "stage": exc.stage,
        }
    return row


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_summary(rows, *, title="IR 解释器 Summary"):
    passed = sum(bool(row.get("correct")) for row in rows)
    failed = len(rows) - passed
    lines = [
        f"# {title}",
        "",
        f"总体状态：{'PASS' if rows and failed == 0 else 'FAIL'}",
        f"Summary: {passed}/{len(rows)} PASS, {failed} FAIL",
        "",
        "| 案例 | 操作 | 预期输出 shape / dtype | 参考依据 | 正确性 | 最大绝对误差 |",
        "|---|---|---|---|---|---:|",
    ]
    for row in rows:
        shape = tuple(row["expected_shape"]) if "expected_shape" in row else "未获得"
        values = (
            row["case"],
            ", ".join(row.get("opcodes", [])),
            f"{shape} / {row.get('expected_dtype', '未获得')}",
            row.get("reference_method", "未获得"),
            "PASS" if row.get("correct") else "FAIL",
            row.get("max_abs_error", "未获得"),
        )
        lines.append("| " + " | ".join(_cell(v) for v in values) + " |")
    lines.extend(
        [
            "",
            "## 比较规则",
            "",
            "先检查 shape、dtype 和有限值；整数逐元素完全一致。",
            "",
        ]
    )
    tolerances = {
        (r["atol"], r["rtol"]) for r in rows if r.get("comparison") == "tolerance"
    }
    for atol, rtol in sorted(tolerances):
        names = ", ".join(
            r["case"]
            for r in rows
            if r.get("comparison") == "tolerance"
            and (r["atol"], r["rtol"]) == (atol, rtol)
        )
        lines.append(
            f"- {names}：`abs(actual - expected) <= {atol} + {rtol} * abs(expected)`。"
        )
    lines.extend(["", "## 失败定位", ""])
    if not failed:
        lines.append("无失败。")
    for row in rows:
        if row.get("correct"):
            continue
        lines.append(
            f"- **{_cell(row['case'])}**：{_cell(row.get('error', '未获得失败诊断'))}"
        )
        location = row.get("error_location", {})
        details = ", ".join(
            f"{key}={value}" for key, value in location.items() if value is not None
        )
        if details:
            lines.append(
                f"  - 原始 IR 位置（instruction_index 从 0 开始）：{_cell(details)}"
            )
        else:
            lines.append(
                "  - 此错误没有单条指令位置；检查绑定、Program 验证或输出比较诊断。"
            )
    return "\n".join(lines) + "\n"
