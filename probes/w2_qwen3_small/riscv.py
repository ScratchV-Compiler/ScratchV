"""Compile both real-Qwen graphs and compare actual RV64 QEMU outputs with ORT.

First export with probes/w2_qwen3_small/run.py. This gate never substitutes host
execution for QEMU, skips missing tools, or accepts logits in place of the trace.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from html import escape
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import onnxruntime as ort

from probes.w2_qwen3_small.diagnostics import compare_outputs, tensor_diff, unpack_trace
from probes.w2_qwen3_small.run import ATOL, arrays_sha256, input_cases
from scratchv.analysis.ir_verifier import verify_ir
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.pass_manager import create_optimization_pass_manager
from scratchv.runtime.riscv_tensor import build_riscv_tensor, discover_toolchain, run_riscv_tensor
from scratchv.verification.ir_interpreter import IRInterpreter

QEMU_GROUPS = tuple((graph, level) for graph in ("normal", "diagnostic") for level in ("none", "all"))
QEMU_CASE_COUNT = 7
QEMU_STATUSES = ("success", "numeric_failed", "runtime_error", "timeout", "not_started")


def _require_valid(program, stage):
    passed, issues = verify_ir(program, stage=stage)
    if not passed:
        raise ValueError("; ".join(str(issue) for issue in issues))


def _optimize(program, level):
    manager = create_optimization_pass_manager(level)
    manager.before_pass = lambda p, data: _require_valid(data, f"before:{p.name}")
    manager.after_pass = lambda p, data: _require_valid(data, f"after:{p.name}")
    return manager.run_pipeline(copy.deepcopy(program)).data


def validated_model_artifacts(model_dir):
    """Bind the QEMU inputs to a successful official-model/PyTorch probe.

    ORT agreement alone cannot establish that an arbitrary supplied ONNX is
    the official two-layer model. Preserve the preceding gate's evidence and
    reject mixed, edited or failed exports before compiling any executable.
    """
    files = {"normal": model_dir / "model.onnx", "diagnostic": model_dir / "diagnostics.onnx"}
    source_path = model_dir / "report.json"
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    if (source.get("passed") is not True or source.get("stage") != "complete"
            or source.get("atol") != ATOL or source.get("rtol") != 0):
        raise ValueError("Model directory requires a successful run.py PyTorch/ORT/IR report")
    model_hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}
    recorded = source.get("onnx", {})
    for kind, field in (("normal", "model_sha256"), ("diagnostic", "diagnostics_sha256")):
        if recorded.get(field) != model_hashes[kind]:
            raise ValueError(f"Model hash disagrees with the successful export report: {kind}")
    schema_path = model_dir / "checkpoints.json"
    schema_bytes = schema_path.read_bytes()
    schema = json.loads(schema_bytes)
    if schema != source.get("checkpoints"):
        raise ValueError("Checkpoint schema disagrees with the successful export report")
    expected_cases = [(name, length, arrays_sha256(feed)) for name, length, feed in input_cases()]
    cases = source.get("cases", [])
    if ([(case.get("name"), case.get("valid_length"), case.get("input_sha256"))
         for case in cases] != expected_cases or any(case.get("passed") is not True for case in cases)):
        raise ValueError("Export report does not validate the current QEMU input cases")
    evidence = {"report_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "model_sha256": model_hashes,
                "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
                "model_seed": source.get("model_seed"), "weights_sha256": source.get("weights_sha256"),
                "config": source.get("config"), "provenance": source.get("provenance"),
                "environment": source.get("environment")}
    return files, schema, evidence


def execution_evidence():
    """Describe this execution, independently of the earlier export's environment."""
    sources = [ROOT / "scratchv/compiler.py", Path(__file__),
               ROOT / "probes/w2_qwen3_small/run.py", ROOT / "probes/w2_qwen3_small/diagnostics.py",
               ROOT / "scratchv/frontend/onnx_parser.py", ROOT / "scratchv/ir/types.py",
               ROOT / "scratchv/ir/builder.py", ROOT / "scratchv/pass_manager.py",
               ROOT / "scratchv/pass_interface.py", ROOT / "scratchv/verification/ir_interpreter.py",
               ROOT / "scratchv/verification/ir_numpy_ops.py",
               *sorted((ROOT / "scratchv/analysis").glob("*.py")),
               *sorted((ROOT / "scratchv/optimizer").glob("*.py")),
               ROOT / "scratchv/backend/tensor_c_codegen.py", ROOT / "scratchv/runtime/riscv_tensor.py"]
    return {
        "environment": {"scope": "Current RISC-V probe process, including its ORT and IR references",
                        "python": platform.python_version(), "executable": sys.executable,
                        "platform": platform.platform(),
                        "packages": {"numpy": np.__version__, "onnx": onnx.__version__,
                                     "onnxruntime": ort.__version__}},
        "source_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }


def execute_qemu_case(executable, feed, folder, expected, *, graph, level,
                      schema, names, reference, case, timeout):
    """Keep every attempted run, including timing carried by runtime errors."""
    row = {"graph": graph, "optimization": level, "status": "not_started", "passed": False,
           "qemu_process_wall_seconds": None, "seconds": None, "timeout_seconds": timeout,
           "command": None}
    case["qemu"].append(row)
    phase = "execution"
    try:
        execution = run_riscv_tensor(executable, feed, folder, timeout=timeout)
        row.update(qemu_process_wall_seconds=execution.elapsed_s, seconds=execution.elapsed_s,
                   command=execution.command)
        phase = "output-evidence"
        actual = execution.output
        np.save(folder / "qemu.npy", actual)
        np.save(folder / "ort.npy", expected)
        phase = "numeric"
        row.update(tensor_diff(actual, expected, ATOL))
        if graph == "diagnostic":
            unpacked = unpack_trace(actual, schema)
            row["checkpoints"] = compare_outputs(unpacked, reference, names, ATOL)
            row["passed"] &= row["checkpoints"]["passed"]
            logits = unpacked["logits"]
        else:
            logits = actual
        row["status"] = "success" if row["passed"] else "numeric_failed"
        return logits
    except Exception as exc:
        row["passed"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["error_stage"] = phase
        if phase == "execution":
            # Timing belongs to the runtime's process boundary, not the outer
            # function's input preparation or output decoding. Never use the
            # configured timeout as a substitute for an actual measurement.
            elapsed = getattr(exc, "elapsed_s", None)
            if elapsed is not None:
                row.update(qemu_process_wall_seconds=elapsed, seconds=elapsed,
                           status=getattr(exc, "status", "runtime_error"),
                           command=getattr(exc, "command", None))
                row["timeout_seconds"] = getattr(exc, "timeout_s", timeout)
        else:
            row["status"] = "numeric_failed" if phase == "numeric" else "runtime_error"
        raise
    finally:
        print(f"[{case['name']}/{graph}/{level}] {row['status']} "
              f"qemu_process_wall_seconds={format_seconds(row['qemu_process_wall_seconds'])} "
              f"max_abs={row.get('max_abs')} "
              f"first={row.get('checkpoints', {}).get('first_divergence')}", flush=True)


def duration_summary(values):
    """Empty or unmeasured samples are null, never a fabricated zero-second run."""
    measured = [float(value) for value in values if value is not None]
    if any(not math.isfinite(value) or value < 0 for value in measured):
        raise ValueError("Measured durations must be finite and nonnegative")
    return {"count": len(measured), "total_seconds": sum(measured) if measured else None,
            "mean_seconds": sum(measured) / len(measured) if measured else None,
            "max_seconds": max(measured) if measured else None}


def summarize_timing(report):
    rows = [row for case in report.get("cases", []) for row in case.get("qemu", [])]

    def group_summary(selected, planned):
        counts = {status: sum(row["status"] == status for row in selected) for status in QEMU_STATUSES}
        completed = counts["success"] + counts["numeric_failed"]
        return {"planned_count": planned, "attempted_count": len(selected),
                "completed_count": completed, "not_attempted_count": planned - len(selected),
                "status_counts": counts, "incomplete": completed != planned,
                "all_measured": duration_summary(row.get("qemu_process_wall_seconds") for row in selected),
                "successful": duration_summary(row.get("qemu_process_wall_seconds") for row in selected
                                               if row["status"] == "success"),
                "failed": duration_summary(row.get("qemu_process_wall_seconds") for row in selected
                                           if row["status"] != "success")}

    total = group_summary(rows, QEMU_CASE_COUNT * len(QEMU_GROUPS))
    groups = [{"graph": graph, "optimization": level,
               **group_summary([row for row in rows if (row["graph"], row["optimization"]) == (graph, level)],
                               QEMU_CASE_COUNT)} for graph, level in QEMU_GROUPS]
    builds = duration_summary(build.get("compile_seconds") for build in report.get("builds", []))
    return {"pipeline_seconds": report.get("seconds"),
            "qemu_process_wall_seconds_total": total["all_measured"]["total_seconds"],
            "cross_compile_seconds_total": builds["total_seconds"],
            "cross_compile_completed_count": builds["count"], "cross_compile_planned_count": len(QEMU_GROUPS),
            "cross_compile_incomplete": builds["count"] != len(QEMU_GROUPS),
            "scope": "Host wall time of QEMU launch, guest computation, output packing/UART and exit; "
                     "timeouts include process cleanup. Excludes host input preparation and output decoding. "
                     "Diagnostic graphs include checkpoint work. Not pure forward or target hardware performance.",
            "pipeline_scope": "Probe pipeline through compilation, references, executions, comparisons and array "
                              "evidence; excludes report rendering.",
            "cross_compile_scope": "Sum of recorded successful C-to-RV64 builds; excludes ScratchV parsing, "
                                   "IR optimization/code generation and unsuccessful build attempts.",
            **total, "groups": groups}


def format_seconds(value):
    return "未测得" if value is None else f"{value:.3f}"


def run_probe(model_dir, out, report, *, cc=None, qemu=None, timeout=180):
    report["stage"] = "environment"
    report.update(execution_evidence())
    report["stage"] = "model-artifacts"
    files, schema, evidence = validated_model_artifacts(model_dir)
    report["export_evidence"] = evidence
    report["stage"] = "toolchain"
    tools = discover_toolchain(cc=cc, qemu=qemu)
    report["toolchain"] = {"cc": tools.cc, "qemu": tools.qemu}
    names = [entry["name"] for entry in schema]
    if len(names) != 29 or names[-1] != "logits":
        raise ValueError("Expected the real two-layer Qwen3 schema with 29 checkpoints")
    report["checkpoints"] = schema
    report["model_sha256"] = evidence["model_sha256"]
    report["schema_sha256"] = evidence["schema_sha256"]
    report["stage"] = "compile"
    report["builds"] = []
    sessions, programs, parsers, executables = {}, {}, {}, {}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    for kind, path in files.items():
        sessions[kind] = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        parser = ONNXParser()
        original = parser.parse(str(path))
        _require_valid(original, "after-parse")
        parsers[kind] = parser
        programs[kind] = {level: _optimize(original, level) for level in ("none", "basic", "all")}
        for level in ("none", "all"):
            directory = out / "build" / f"{kind}-{level}"
            directory.mkdir(parents=True)
            driver = CompilerDriver(CompilerConfig(backend="tensor-c", optimize_level=level, verify_ir=True))
            result = driver.compile(str(path), str(directory / "model.c"))
            if not result.success:
                raise RuntimeError("; ".join(result.errors))
            artifact = driver.tensor_artifact
            executable = build_riscv_tensor(artifact, directory / "rv64", tools)
            executables[kind, level] = executable
            report["builds"].append({"graph": kind, "optimization": level,
                                     "workspace_bytes": artifact.workspace_bytes,
                                     "constant_bytes": artifact.constant_bytes,
                                     "elf_sha256": executable.elf_sha256,
                                     "compile_command": executable.compile_command,
                                     "tool_versions": executable.tool_versions,
                                     "compile_seconds": executable.compile_seconds,
                                     "optimization_stats": result.stats["optimization"]})
            print(f"[build {kind}/{level}] RV64 ELF ready; workspace={artifact.workspace_bytes}", flush=True)

    report["stage"] = "numeric"
    report["cases"] = []
    cached = {}
    for case_name, valid_length, feed in input_cases():
        report["current_case"] = case_name
        case = {"name": case_name, "valid_length": valid_length,
                "input_sha256": arrays_sha256(feed), "passed": False,
                "optimized_ir": [], "qemu": []}
        report["cases"].append(case)
        ordinary_ref = sessions["normal"].run(None, feed)[0]
        diagnostic_values = sessions["diagnostic"].run(None, feed)
        diagnostic_ref = diagnostic_values[0]
        named_ref = unpack_trace(diagnostic_ref, schema)
        # Check the supplied packed layout against independent ORT named outputs.
        ort_names = [value.name for value in sessions["diagnostic"].get_outputs()][1:]
        layout = compare_outputs(named_ref, dict(zip(ort_names, diagnostic_values[1:])), names, ATOL)
        case["ort_pack_layout"] = layout
        case["ort_normal_vs_trace"] = tensor_diff(ordinary_ref, named_ref["logits"], ATOL)
        cached[case_name] = {}
        for kind, expected in (("normal", ordinary_ref), ("diagnostic", diagnostic_ref)):
            for level in ("none", "basic", "all"):
                actual = IRInterpreter(programs[kind][level]).run(
                    feed, initializers=parsers[kind].initializers).return_value
                comparison = tensor_diff(actual, expected, ATOL)
                case["optimized_ir"].append({"graph": kind, "optimization": level, **comparison})
            for level in ("none", "all"):
                folder = out / "runs" / case_name / f"{kind}-{level}"
                logits = execute_qemu_case(executables[kind, level], feed, folder, expected,
                                          graph=kind, level=level, schema=schema, names=names,
                                          reference=named_ref, case=case, timeout=timeout)
                cached[case_name][kind, level] = logits
        case["normal_vs_trace"] = [
            {"optimization": level, **tensor_diff(cached[case_name]["normal", level],
                                                   cached[case_name]["diagnostic", level], ATOL)}
            for level in ("none", "all")]
        case["passed"] = all(check["passed"] for check in (
            layout, case["ort_normal_vs_trace"], *case["optimized_ir"],
            *case["qemu"], *case["normal_vs_trace"]))
    report["invariants"] = []
    for name, left, right, length in (
        ("causality", "full_seed_0", "changed_future", 64),
        ("padding_isolation", "short_17", "changed_padding", 17),
    ):
        for kind in files:
            for level in ("none", "all"):
                report["invariants"].append({"name": name, "graph": kind, "optimization": level,
                    **tensor_diff(cached[left][kind, level][:, :length],
                                  cached[right][kind, level][:, :length], ATOL)})
    report["passed"] = all(case["passed"] for case in report["cases"]) and all(
        check["passed"] for check in report["invariants"])
    report["stage"] = "complete"


def _report_views(report):
    timing = report["timing"]
    counts = timing["status_counts"]
    lines = ["# Two-layer Qwen3: ScratchV IR → C tensor kernels → RV64GC → QEMU", "",
             f"Result: **{'PASS' if report['passed'] else 'FAIL'}**; stage: {report['stage']}", "",
             "## 仿真耗时", "",
             f"- 探测流程总耗时：**{format_seconds(timing['pipeline_seconds'])} 秒**（不含报告渲染）。",
             f"- 已记录 QEMU 进程耗时合计：**{format_seconds(timing['qemu_process_wall_seconds_total'])} 秒**。",
             f"- 成功交叉编译记录合计：**{format_seconds(timing['cross_compile_seconds_total'])} 秒**；"
             f"已记录 {timing['cross_compile_completed_count']}/{timing['cross_compile_planned_count']} 次构建，"
             "不含 ScratchV 解析/IR 优化/代码生成或失败构建的耗时。",
             f"- 计划 {timing['planned_count']} 次，已尝试 {timing['attempted_count']} 次；"
             f"通过 {counts['success']}，数值失败 {counts['numeric_failed']}，"
             f"运行失败 {counts['runtime_error']}，超时 {counts['timeout']}，"
             f"未启动 {counts['not_started']}，未尝试 {timing['not_attempted_count']}。",
             f"- 记录范围：{'部分结果，尚未完成全部执行/数值核验' if timing['incomplete'] else '全部执行/数值核验已有记录'}；"
             "失败样本与通过样本分别统计，未测得的时长不按零计。", "",
             "QEMU 进程墙钟时长包含启动、guest 计算、输出打包/UART 和退出；超时还包括进程清理。"
             "不包含主机输入准备或输出解码。diagnostic 包含检查点开销，须与 normal 分开。"
             "这些数据仅用于监控，不代表纯前向时长或目标硬件性能，也不作为性能通过阈值。", "",
             "| 图/优化级别 | 已尝试/计划 | 通过/失败 | 未尝试 | 通过平均/最大(s) | 失败平均/最大(s) | 已测合计(s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    group_rows = []
    for group in timing["groups"]:
        successful, failed = group["successful"], group["failed"]
        failed_count = group["attempted_count"] - group["status_counts"]["success"]
        cells = [f"{group['graph']}/{group['optimization']}",
                 f"{group['attempted_count']}/{group['planned_count']}",
                 f"{group['status_counts']['success']}/{failed_count}", str(group['not_attempted_count']),
                 f"{format_seconds(successful['mean_seconds'])}/{format_seconds(successful['max_seconds'])}",
                 f"{format_seconds(failed['mean_seconds'])}/{format_seconds(failed['max_seconds'])}",
                 format_seconds(group['all_measured']['total_seconds'])]
        group_rows.append(cells)
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "平均/最大值只统计有实测时长的样本；失败数包含尚未启动但已尝试的条目。"
              "七种输入不是重复采样，不据此宣称稳定加速比。", "",
             "FP32, L=256, random weights, 29 checkpoints, strict max absolute error < 1e-5.",
             "The C cross-compiler is Zig/LLVM; this does not validate the legacy scalar assembly selector.",
             "IR optimization levels: none/basic/all. QEMU: none/all, normal and diagnostic graphs.", ""]
    if report.get("error"):
        lines += [f"Error: {report['error']}", ""]
    lines += ["| Case | Graph | IR optimization | QEMU进程(s) | 超时设定(s) | Max abs | First divergent checkpoint | Status |",
              "|---|---|---|---:|---:|---:|---|---|"]
    execution_rows = []
    for case in report.get("cases", []):
        for row in case.get("qemu", []):
            cells = [case['name'], row['graph'], row['optimization'],
                     format_seconds(row.get('qemu_process_wall_seconds')),
                     format_seconds(row.get('timeout_seconds')), str(row.get('max_abs')),
                     row.get('checkpoints', {}).get('first_divergence') or 'none', row['status']]
            execution_rows.append(cells)
            lines.append("| " + " | ".join(cells) + " |")
    markdown = "\n".join(lines) + "\n"
    details = []
    for case in report.get("cases", []):
        details.append(f"<details><summary>{escape(case['name'])}: "
                       f"{'PASS' if case['passed'] else 'FAIL'}</summary><pre>"
                       f"{escape(json.dumps(case, indent=2))}</pre></details>")

    def html_table(headers, rows):
        header = "".join(f"<th>{escape(value)}</th>" for value in headers)
        body = "".join("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>"
                       for row in rows)
        return f"<div class='table-wrap'><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"

    overview = ("<h2>仿真耗时</h2>"
                f"<p>探测流程总耗时：<strong>{format_seconds(timing['pipeline_seconds'])} 秒</strong>；"
                f"已记录 QEMU 进程合计：<strong>{format_seconds(timing['qemu_process_wall_seconds_total'])} 秒</strong>；"
                f"成功交叉编译记录合计：{format_seconds(timing['cross_compile_seconds_total'])} 秒"
                f"（{timing['cross_compile_completed_count']}/{timing['cross_compile_planned_count']} 次）。</p>"
                f"<p>计划 {timing['planned_count']} 次，已尝试 {timing['attempted_count']} 次；"
                f"通过 {counts['success']}，数值失败 {counts['numeric_failed']}，"
                f"运行失败 {counts['runtime_error']}，超时 {counts['timeout']}，"
                f"未启动 {counts['not_started']}，未尝试 {timing['not_attempted_count']}。"
                f"<strong>{'部分结果' if timing['incomplete'] else '完整执行记录'}</strong>。</p>"
                "<p>QEMU 时长包含进程启动、guest 计算、输出打包/UART 和退出，超时包含进程清理；"
                "不含主机输入准备或输出解码。diagnostic 含检查点开销，与 normal 分开统计。"
                "流程总耗时不含报告渲染；交叉编译仅统计成功记录，不含 ScratchV 解析/IR 优化/代码生成。"
                "未测得不按零计，失败与通过样本分别计算平均/最大值。七种输入不是重复采样，"
                "仅用于监控，不代表纯前向时长、稳定加速比或目标硬件性能，无性能通过阈值。</p>")
    overview += html_table(["图/优化", "已尝试/计划", "通过/失败", "未尝试", "通过平均/最大(s)",
                            "失败平均/最大(s)", "已测合计(s)"], group_rows)
    overview += "<h2>逐次执行</h2>" + html_table(
        ["Case", "Graph", "IR优化", "QEMU进程(s)", "超时设定(s)", "Max abs", "首次偏差", "Status"],
        execution_rows)
    if report.get("error"):
        overview += f"<p>Error: {escape(report['error'])}</p>"
    html = (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Qwen3 RV64 probe</title>"
        "<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:32px auto}pre{white-space:pre-wrap}"
        ".table-wrap{overflow:auto}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}"
        "details{padding:12px;border:1px solid #ddd;margin:10px 0}</style>"
        f"<h1>Qwen3 RV64 QEMU: {'PASS' if report['passed'] else 'FAIL'}</h1>"
        + overview + "".join(details) + "</html>")
    return {"report.md": markdown, "report.html": html}


def write_reports(out, report):
    """Publish JSON last and atomically; failed evidence is a failed gate."""
    report["timing"] = summarize_timing(report)
    failures = []

    def write(name, content):
        # A write can leave complete bytes and still raise (for example on
        # close). Never expose those bytes as the final machine-readable PASS.
        temporary = out / f".{name}.tmp"
        try:
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(out / name)
            return True
        except OSError as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                failures.append(f"{temporary.name}: cleanup failed: {cleanup_error}")
            return False

    def fail():
        if report["passed"]:
            report.update(passed=False, stage="report-write", error="Incomplete QEMU gate evidence")
        report["report_write_errors"] = list(failures)

    def recover_views():
        summary = ("# Two-layer Qwen3 RV64 gate\n\nResult: **FAIL**\n\n"
                   f"Stage: {report['stage']}\n\nError: {report.get('error', 'Gate failed')}\n\n"
                   + "Report write errors:\n" + "\n".join(f"- {item}" for item in failures) + "\n")
        for name, content in (("report.md", summary), ("report.html",
                              '<!doctype html><meta charset="utf-8"><pre>' + escape(summary) + "</pre>")):
            write(name, content)
        report["report_write_errors"] = list(failures)

    for name, content in _report_views(report).items():
        write(name, content)
    if failures:
        fail()
        recover_views()
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if not write("report.json", payload):
        fail()
        recover_views()
        # A transient publication error may still allow the failure evidence
        # to be saved; a permanent error leaves no final PASS JSON.
        write("report.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["report_write_errors"] = list(failures)
    if failures:
        print("[RV64 gate] Report write failed: " + "; ".join(failures), file=sys.stderr)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, default=ROOT / "output/qwen3-riscv")
    cli.add_argument("--cc")
    cli.add_argument("--qemu")
    cli.add_argument("--timeout", type=float, default=180)
    args = cli.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        cli.error("--timeout must be finite and positive")
    out = args.output_dir.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        cli.error("--output-dir must be empty; preserve earlier evidence")
    out.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "stage": "initialization", "atol": ATOL, "rtol": 0}
    started = time.perf_counter()
    try:
        run_probe(args.model_dir.resolve(), out, report, cc=args.cc, qemu=args.qemu, timeout=args.timeout)
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    report["seconds"] = time.perf_counter() - started
    write_reports(out, report)
    timing = report["timing"]
    print(f"[耗时] 探测流程={format_seconds(timing['pipeline_seconds'])}s; "
          f"QEMU进程实测合计={format_seconds(timing['qemu_process_wall_seconds_total'])}s; "
          f"成功交叉编译记录={format_seconds(timing['cross_compile_seconds_total'])}s; "
          f"尝试={timing['attempted_count']}/{timing['planned_count']}; "
          f"通过={timing['status_counts']['success']}; 未尝试={timing['not_attempted_count']}; "
          f"{'部分结果' if timing['incomplete'] else '完整执行记录'}", flush=True)
    print(f"[RV64 gate] {'PASS' if report['passed'] else 'FAIL'}: {out / 'report.json'}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
