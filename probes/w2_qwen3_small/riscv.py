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
from probes.w2_qwen3_small.run import (
    ATOL, _details_views, _page_html, _table_views, arrays_sha256, input_cases,
)
from probes.numeric_summary import METRIC_EXPLANATION, compact_metric_table, compact_metric_table_html
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
               ROOT / "scratchv/verification/numeric_metrics.py", ROOT / "probes/numeric_summary.py",
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
    cases = report.get("cases", [])
    comparisons = []
    for field, levels, label in (
        ("optimized_ir", ("none", "basic", "all"), "IR 原生 NumPy（优化回归）/ ORT"),
        ("qemu", ("none", "all"), "生成 C → RV64/QEMU / ORT"),
    ):
        rows = []
        for case in cases + [{}] * max(0, QEMU_CASE_COUNT - len(cases)):
            for level in levels:
                rows.append(next((row for row in case.get(field, [])
                                  if row.get("graph") == "normal" and row.get("optimization") == level), {}))
        comparisons.append((label, rows))
    title = "两层 Qwen3：编译到 RISC-V 的数值精度"
    status = "PASS" if report.get("passed") else "FAIL"
    description = ("真实 Qwen3 结构的两层缩小模型，固定随机权重，FP32、L=256、无 KV cache。"
                   "同一 ONNX 模型、权重和输入分别经 ORT、原生 NumPy IR，以及 ScratchV IR → C 张量代码 → "
                   "C 交叉编译器 → RV64GC → QEMU 执行。C 行测的是 RISC-V 上的实际输出，不是单独的主机 C 测试。")
    scope = ("核心表只汇总普通图最终 logits 的最坏指标（全部位置含 padding），比较按“被测 / 参考”排列。"
             "七个输入场景覆盖两份满长输入、1/17/255 token 边界、因果与补齐隔离。"
             "普通/诊断图、29 个检查点及隔离检查仍须全部通过严格 MaxAbs < 1e-5（rtol=0）。")
    timing_line = (f"仿真耗时：探测流程 {format_seconds(timing['pipeline_seconds'])} 秒；"
                   f"已记录 QEMU 进程累计 {format_seconds(timing['qemu_process_wall_seconds_total'])} 秒"
                   "（含启动和传输，不代表纯前向或目标硬件性能）。")
    coverage = (f"执行覆盖：{timing['attempted_count']}/{timing['planned_count']} 次已尝试，"
                f"{counts['success']} 次通过；"
                + ("部分结果，尚未完成全部执行。" if timing['incomplete'] else "全部执行完成。"))
    markdown = [f"# {title}", "", f"Result: **{status}**; stage: {report.get('stage')}", "",
                description, "", scope, "", compact_metric_table(comparisons), "", METRIC_EXPLANATION,
                "", coverage, "", timing_line]
    html = [f"<h1>{escape(title)}</h1><p><strong>Result: {status}</strong>; stage: {escape(str(report.get('stage')))}</p>",
            f"<p>{escape(description)}</p><p>{escape(scope)}</p>", compact_metric_table_html(comparisons),
            f"<p>{escape(METRIC_EXPLANATION)}</p><p>{escape(coverage)}</p><p>{escape(timing_line)}</p>"]
    failures = []
    if report.get("error"):
        failures.append(f"阶段 {report.get('stage')}: {report['error']}")
    for case in cases:
        if case.get("passed") is not True:
            failures.append(f"{case.get('name')}: FAIL；有检查失败或未完成")
        for row in case.get("qemu", []):
            if row.get("passed") is not True:
                reason = row.get("error") or row.get("reason") or "尚未完成数值核验"
                first = row.get("checkpoints", {}).get("first_divergence")
                failures.append(f"{case.get('name')} / {row.get('graph')}/{row.get('optimization')}: "
                                f"{row.get('status')}; {reason}" + (f"；首次超限检查点 {first}" if first else ""))
        for row in case.get("optimized_ir", []):
            if row.get("passed") is not True:
                failures.append(f"{case.get('name')} IR {row.get('graph')}/{row.get('optimization')}: "
                                f"{row.get('reason') or 'FAIL'}")
    for row in report.get("invariants", []):
        if row.get("passed") is not True:
            failures.append(f"{row.get('name')} / {row.get('graph')}/{row.get('optimization')}: 隔离检查 FAIL")
    if not report.get("passed") and not failures:
        failures.append("尚未完成全部验收，请检查 report.json。")
    if failures:
        markdown += ["", "**失败／未完成项**", "", *[f"- {escape(item)}" for item in failures]]
        html.append("<div class='failure'><p>失败／未完成项</p><ul>"
                    + "".join(f"<li>{escape(item)}</li>" for item in failures) + "</ul></div>")

    def append_detail(title, headers, rows, note):
        md, body = _table_views(headers, rows)
        md, body = _details_views(title, md + "\n\n" + note, body + f"<p>{escape(note)}</p>")
        markdown.extend(["", md])
        html.append(body)

    graph_note = ("图指模型计算步骤及其连接关系：normal 只输出最终 logits；diagnostic 增加中间检查点以定位误差，含额外打包与传输开销。"
                  "none/basic/all 指 ScratchV IR 优化：none 不优化；basic 提前计算常量并删除不影响输出的计算；"
                  "all 再加入局部指令简化、乘加融合和循环内重复计算外提。它们不是 C 编译优化等级，本探测的 C 编译固定使用 -O2。")
    time_note = (graph_note + " QEMU 墙钟时间包含启动、目标程序计算、输出打包/串口传输和退出；超时还含清理，"
                 "不含主机输入准备或输出解码。未测得的时长不按零计；平均/最大值仅统计有测量值的样本。"
                 "七种输入不是重复采样，不能据此推导稳定加速比，也不设置性能通过阈值。"
                 f"成功交叉编译记录累计 {format_seconds(timing['cross_compile_seconds_total'])} 秒"
                 f"（{timing['cross_compile_completed_count']}/{timing['cross_compile_planned_count']} 次），"
                 "不含 ScratchV 解析、IR 优化、代码生成及失败构建；流程时间不含报告渲染。")
    groups = []
    for group in timing["groups"]:
        successful, failed = group["successful"], group["failed"]
        groups.append([f"{group['graph']}/{group['optimization']}",
                       f"{group['attempted_count']}/{group['planned_count']}",
                       f"{group['status_counts']['success']}/{group['attempted_count'] - group['status_counts']['success']}",
                       group['not_attempted_count'],
                       f"{format_seconds(successful['mean_seconds'])}/{format_seconds(successful['max_seconds'])}",
                       f"{format_seconds(failed['mean_seconds'])}/{format_seconds(failed['max_seconds'])}",
                       format_seconds(group['all_measured']['total_seconds'])])
    append_detail("图与优化级别：仿真耗时明细",
                  ["图/优化级别", "已尝试/计划", "通过/失败", "未尝试", "通过平均/最大(s)", "失败平均/最大(s)", "已测合计(s)"],
                  groups, time_note)
    executions = []
    for case in cases:
        for row in case.get("qemu", []):
            executions.append([case.get('name'), row.get('graph'), row.get('optimization'),
                               format_seconds(row.get('qemu_process_wall_seconds')),
                               format_seconds(row.get('timeout_seconds')), row.get('max_abs', '未统计'),
                               row.get('checkpoints', {}).get('first_divergence') or 'none', row.get('status')])
    append_detail("逐场景 QEMU 执行明细",
                  ["场景", "图", "IR 优化", "QEMU 进程(s)", "超时设定(s)", "本图输出 MaxAbs", "首次超限检查点", "结果"],
                  executions, graph_note + " 此明细中的 diagnostic MaxAbs 包含全部打包检查点；首页核心表只使用 normal logits。"
                  "超时设定是限制，QEMU 进程(s) 才是实测墙钟时间。")
    checks = []
    for case in cases:
        for row in case.get('optimized_ir', []):
            checks.append([case.get('name'), f"IR {row.get('graph')}/{row.get('optimization')}",
                           row.get('max_abs'), 'PASS' if row.get('passed') else row.get('reason') or 'FAIL'])
        for row in case.get('normal_vs_trace', []):
            checks.append([case.get('name'), f"normal/diagnostic logits {row.get('optimization')}",
                           row.get('max_abs'), 'PASS' if row.get('passed') else row.get('reason') or 'FAIL'])
    for row in report.get('invariants', []):
        checks.append([row.get('name'), f"{row.get('graph')}/{row.get('optimization')}",
                       row.get('max_abs'), 'PASS' if row.get('passed') else row.get('reason') or 'FAIL'])
    append_detail("优化 IR、普通/诊断图一致性与隔离检查", ["场景/检查", "路径", "MaxAbs", "结果"], checks,
                  "所有这些检查继续参与验收，完整逐检查点误差保存在 report.json。")
    footer = ("完整数值明细、命令、环境、源码指纹、C/ELF 与原始数组随 CI artifact 提供；结构化报告为 report.json。"
              "这条路径不验证旧的标量汇编选择器，也不声称完整预训练模型已在 QEMU 运行。")
    markdown.extend(["", footer])
    html.append(f"<p>{escape(footer)}</p>")
    return {"report.md": "\n".join(markdown) + "\n", "report.html": _page_html(title, "".join(html))}


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
