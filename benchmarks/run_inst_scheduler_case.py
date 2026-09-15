"""Report same-input scheduling A/B results through CompilerDriver.

The default feature case must change order and pass real TinyFive execution.
Use --static-only for compiler-generated assembly: zero-hit inputs are valid,
and execution is explicitly not claimed. Model cycles are never CPU timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from scratchv.backend._asm_parser import parse_asm
from scratchv.backend.inst_scheduler import (
    ScheduleConfig,
    parse_instructions,
    schedule_assembly,
)
from scratchv.backend.riscv_encoder import assemble_to_binary
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.simulator.tinyfive import ProfiledMachine

DEFAULT_CASE = Path(__file__).parent / "cases" / "inst_scheduler_feature.asm"
DATA_ADDRESS = 1024
DATA_WORDS = [9] + [0] * 15
INITIAL_REGISTERS = {10: DATA_ADDRESS, 7: 7}  # a0, t2; other registers are zero.


def _instruction_count(source: str) -> int:
    # Same executable-section scope as scheduling.input_instructions, including
    # unsupported opcodes and excluding directives, data and display labels.
    return len(parse_instructions(source))


def _execute(source: str) -> dict[str, Any]:
    """Execute the bounded integer feature-case contract without a fallback.

    Only straight-line code can use the encoded word count as an execution
    limit. Keep a0 fixed and confine word accesses to the observed data area.
    Arbitrary real programs belong in the static report, not this harness.
    """
    for line in parse_asm(source):
        if line.is_directive and line.opcode not in {"text", "globl", "global"}:
            raise ValueError(
                "feature execution only accepts text and global directives"
            )
    instructions = parse_instructions(source)
    if not instructions:
        raise ValueError("feature execution requires nonempty straight-line RV32 code")
    for inst in instructions:
        effects = inst.effects
        if (
            effects.barrier_reason
            or effects.control != "none"
            or effects.fp_flags
            or any(not reg.startswith("x") for reg in inst.uses | inst.defines)
            or "x10" in inst.defines
        ):
            raise ValueError(
                "feature execution requires straight-line integer code with fixed a0"
            )
        if effects.memory != "none":
            address = effects.address
            if (
                inst.opcode not in {"lw", "sw"}
                or address is None
                or address.base != "x10"
                or not 0 <= address.offset < len(DATA_WORDS) * 4
                or address.offset % 4
            ):
                raise ValueError(
                    "feature memory accesses must be aligned words within the a0 data area"
                )

    binary = assemble_to_binary(source)
    if not binary or len(binary) % 4 or len(binary) > DATA_ADDRESS:
        raise ValueError("invalid feature binary length or overlapping code and data")
    words = [
        int.from_bytes(binary[i : i + 4], "little") for i in range(0, len(binary), 4)
    ]
    machine = ProfiledMachine(mem_size=4096)
    if not machine.available:
        raise RuntimeError("real TinyFive is unavailable; fallback is forbidden")
    machine.load_binary(words, origin=0)
    for index, value in enumerate(DATA_WORDS):
        machine.write_mem_i32(DATA_ADDRESS + index * 4, value)
    for index, value in INITIAL_REGISTERS.items():
        machine.set_reg(index, value)
    machine.run(instructions=len(words), start=0, strict=True)
    if (
        machine.last_error
        or machine.instr_count != len(words)
        or machine.pc != len(binary)
    ):
        raise RuntimeError(
            machine.last_error or "TinyFive did not execute the complete feature case"
        )
    return {
        "encoded_instructions": len(words),
        "code_size_bytes": len(binary),
        "executed_instructions": machine.instr_count,
        "all_registers": [machine.get_reg(i) for i in range(32)],
        "memory_words": [
            machine.read_mem_i32(DATA_ADDRESS + i * 4) for i in range(len(DATA_WORDS))
        ],
        "final_pc": machine.pc,
    }


def run_case(
    case_path: Path, *, static_only: bool = False, repeats: int = 3
) -> dict[str, Any]:
    """Measure only the scheduling toggle on identical input assembly."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    source = case_path.read_bytes().decode("utf-8")
    baseline = CompilerDriver(CompilerConfig(schedule=False))
    before = baseline._run_asm_passes(source, [])
    if before != source:
        raise AssertionError("disabled scheduling changed the baseline assembly")
    driver = CompilerDriver(CompilerConfig(schedule=True, schedule_strict=True))
    times = []
    for _ in range(repeats):
        stats: dict[str, Any] = {}
        warnings: list[str] = []
        started = time.perf_counter()
        after = driver._run_asm_passes(source, warnings, stats)
        times.append(time.perf_counter() - started)
    if "schedule" not in stats:
        raise AssertionError("CompilerDriver did not report scheduling statistics")
    direct = schedule_assembly(source, ScheduleConfig(strict=True))
    if after != direct.asm_text:
        raise AssertionError(
            "CompilerDriver output differs from the public scheduling API"
        )
    scheduling = stats["schedule"]
    source_before, source_after = _instruction_count(before), _instruction_count(after)
    if source_before != source_after:
        raise AssertionError("scheduling changed the number of source instructions")
    if scheduling["final_cycles"] > scheduling["original_cycles"]:
        raise AssertionError("scheduling regressed the local cycle model")
    changed = before != after
    simulation: dict[str, Any] = {
        "status": "not_run",
        "backend": None,
        "output_equal": None,
        "reason": "Static assembly A/B only; no whole-program execution or hardware timing.",
    }
    errors = []
    if not static_only:
        before_run, after_run = _execute(before), _execute(after)
        registers_equal = before_run["all_registers"] == after_run["all_registers"]
        memory_equal = before_run["memory_words"] == after_run["memory_words"]
        simulation = {
            "status": "passed" if registers_equal and memory_equal else "failed",
            "backend": "tinyfive",
            "fallback": False,
            "initial_registers": {f"x{k}": v for k, v in INITIAL_REGISTERS.items()},
            "data_address": DATA_ADDRESS,
            "initial_memory_words": DATA_WORDS,
            "before": before_run,
            "after": after_run,
            "registers_equal": registers_equal,
            "memory_equal": memory_equal,
            "output_equal": registers_equal and memory_equal,
        }
        if (
            not changed
            or scheduling["applied_regions"] == 0
            or scheduling["saved_cycles"] <= 0
        ):
            errors.append(
                "feature case did not exercise a beneficial scheduling change"
            )
        if not simulation["output_equal"]:
            errors.append("execution changed register or memory outputs")
    return {
        "schema_version": 1,
        "benchmark_type": "assembly-ab" if static_only else "feature-case",
        "case": str(case_path),
        "input_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "status": "failed" if errors else "passed",
        "comparison_status": (
            "not_modeled"
            if not scheduling["modeled_instructions"]
            else "changed"
            if changed
            else "no_improvement"
        ),
        "errors": errors,
        "feature": {
            "name": "RV32 local instruction scheduling",
            "baseline_schedule": baseline.config.schedule,
            "compiler_config_schedule": driver.config.schedule,
            "schedule_strict": driver.config.schedule_strict,
            "compiler_path": "CompilerDriver._run_asm_passes",
            "pipeline_matches_public_pass": True,
            "used": changed and scheduling["applied_regions"] > 0,
            "warnings": warnings,
        },
        "source_instructions": {"before": source_before, "after": source_after},
        "scheduling": scheduling,
        "timing": {
            "repeats": repeats,
            "mean_ms": statistics.mean(times) * 1000,
            "stdev_ms": statistics.stdev(times) * 1000 if repeats > 1 else 0,
            "scope": "Host elapsed time of CompilerDriver assembly post-pass; not target runtime",
        },
        "simulation": simulation,
        "assembly": {"before": before, "after": after},
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 课题 18：指令调度 A/B 报告",
        "",
        f"- 报告状态：{'PASS' if report['status'] == 'passed' else 'FAIL'}",
        f"- 类型：`{report['benchmark_type']}`",
        f"- 输入：`{report['case']}`",
    ]
    if "scheduling" not in report:
        return "\n".join(lines + [f"- 错误：{report['error']}", ""])
    s, sim = report["scheduling"], report["simulation"]
    source_count = report["source_instructions"]["before"]
    coverage = s["modeled_instructions"] / source_count if source_count else 0
    modeled = s["modeled_instructions"] > 0
    lines.extend(
        [
            f"- 输入 SHA-256：`{report['input_sha256']}`",
            "- 编译器开关：`schedule=False` → `schedule=True, schedule_strict=True`",
            f"- 编译器输出与独立调度 API 一致：`{report['feature']['pipeline_matches_public_pass']}`",
            f"- 实际换序：`{report['feature']['used']}`",
            f"- 比较结论：`{report['comparison_status']}`（changed=已换序；no_improvement=无收益；not_modeled=未建模）",
            f"- 模型：`{s['model']}`；覆盖 {s['modeled_instructions']}/{source_count} 条源指令（{coverage:.1%}）",
            f"- 应用区域：{s['applied_regions']}；跳过/恢复区域：{s['skipped_regions']}；移动指令：{s['moved_instructions']}",
            f"- 调度耗时：{report['timing']['mean_ms']:.3f} ± {report['timing']['stdev_ms']:.3f} ms（{report['timing']['repeats']} 次）",
            "",
            "## 指标",
            "",
            "| 指标 | 调度前 | 调度后 | 减少 |",
            "|---|---:|---:|---:|",
            f"| 源汇编指令数 | {report['source_instructions']['before']} | {report['source_instructions']['after']} | {report['source_instructions']['before'] - report['source_instructions']['after']} |",
            f"| 局部模型周期（估算） | {s['original_cycles'] if modeled else 'N/A'} | {s['final_cycles'] if modeled else 'N/A'} | {s['saved_cycles'] if modeled else 'N/A'} |",
            f"| 局部模型停顿（估算） | {s['original_stalls'] if modeled else 'N/A'} | {s['final_stalls'] if modeled else 'N/A'} | {s['original_stalls'] - s['final_stalls'] if modeled else 'N/A'} |",
        ]
    )
    if sim["status"] != "not_run":
        a, b = sim["before"], sim["after"]
        for label, key in [
            ("编码机器指令数", "encoded_instructions"),
            ("代码大小（字节）", "code_size_bytes"),
            ("TinyFive 实际执行指令数", "executed_instructions"),
        ]:
            lines.append(f"| {label} | {a[key]} | {b[key]} | {a[key] - b[key]} |")
        lines.extend(
            [
                "",
                "## 执行验证",
                "",
                "- 后端：`tinyfive`；`fallback=false`。",
                f"- 32 个整数寄存器一致：`{sim['registers_equal']}`。",
                f"- 数据区 16 个字一致：`{sim['memory_equal']}`；输出一致：`{sim['output_equal']}`。",
                "- 初始条件：其余寄存器为零，`a0=1024`、`t2=7`，数据区首字为 9、其余为零。",
                "",
                "| 观察值 | 调度前 | 调度后 |",
                "|---|---:|---:|",
            ]
        )
        for index in (5, 6, 7, 28):
            lines.append(
                f"| x{index} | {a['all_registers'][index]} | {b['all_registers'][index]} |"
            )
        lines.append(
            f"| memory[1028] | {a['memory_words'][1]} | {b['memory_words'][1]} |"
        )
    else:
        lines.extend(
            ["", "- 执行验证：未运行；机器码大小、动态指令数及硬件耗时均未测量。"]
        )
    if report["errors"]:
        lines.extend(
            ["", "## 失败原因", "", *[f"- {error}" for error in report["errors"]]]
        )
    lines.extend(
        [
            "",
            "> 周期是各局部区域在输入就绪假设下的静态估算，未覆盖部分不按零周期解释。",
            "> 调度耗时是运行优化器的主机时间。TinyFive 仅验证执行结果和动态指令数，不能证明硬件加速。",
            "",
            "<details>",
            "<summary>区域状态与诊断</summary>",
            "",
            "| 起止行 | 指令数 | 状态 | 模型周期（前 → 后） |",
            "|---|---:|---|---|",
        ]
    )
    for row in s["regions"]:
        before = row["original_cycles"] if row["original_cycles"] is not None else "N/A"
        after = row["final_cycles"] if row["final_cycles"] is not None else "N/A"
        lines.append(
            f"| {row['start_line']}–{row['end_line']} | {row['instructions']} | {row['status']} | {before} → {after} |"
        )
    lines.append("")
    for diagnostic in s["diagnostics"]:
        lines.append(f"- 第 {diagnostic['line']} 行：{diagnostic['reason']}")
    lines.extend(["", "</details>"])
    for key, label in [("before", "调度前汇编"), ("after", "调度后汇编")]:
        lines.extend(
            [
                "",
                "<details>",
                f"<summary>{label}</summary>",
                "",
                "```asm",
                report["assembly"][key].rstrip(),
                "```",
                "",
                "</details>",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", type=Path, default=DEFAULT_CASE)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Report supplied assembly without executing it; zero-hit cases are valid",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("benchmark_reports/inst_scheduler_report.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("benchmark_reports/inst_scheduler_report.md"),
    )
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("repeats must be positive")
    try:
        report = run_case(args.case, static_only=args.static_only, repeats=args.repeats)
    except Exception as exc:  # noqa: BLE001 - Persist a failed CI report for unexpected errors too.
        report = {
            "schema_version": 1,
            "benchmark_type": "assembly-ab" if args.static_only else "feature-case",
            "case": str(args.case),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    markdown = _markdown(report)
    for path, content in [
        (args.json, json.dumps(report, indent=2, ensure_ascii=False) + "\n"),
        (args.markdown, markdown),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(markdown)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
