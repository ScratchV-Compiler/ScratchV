#!/usr/bin/env python3
"""Audit complete, pinned Qwen3 ONNX -> ScratchV IR parsing and weight binding.

This gate does not execute the full IR, generate RISC-V code, or run QEMU.
Large-model work runs in a bounded child process; failure preserves reports.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.w1_qwen3_export import run as artifacts

GATE = "frontend:parse-qwen3"
SCOPE = ("Complete pinned 28-layer Qwen3 ONNX parsing, graph connectivity, IR verification "
         "and actual weight/constant bindings. No full-model IR numerical comparison, "
         "code generation or QEMU execution.")


def write_json(path, value):
    path = Path(path)
    # Readers must not observe a half-written progress or result file.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def source_fingerprints():
    files = [*sorted(Path(__file__).parent.glob("*.py")),
             ROOT / "probes/w1_qwen3_export/run.py", ROOT / "probes/w1_qwen3_export/manifest.json",
             ROOT / "scratchv/frontend/onnx_parser.py", ROOT / "scratchv/ir/types.py",
             ROOT / "scratchv/ir/builder.py", ROOT / "scratchv/verification/ir_numpy_ops.py",
             ROOT / "scratchv/runtime/riscv_tensor.py",
             *sorted((ROOT / "scratchv/analysis").glob("*.py"))]
    return {p.relative_to(ROOT).as_posix(): artifacts.sha256(p) for p in files}


def checkout_evidence():
    result = {"head": None, "status": "unknown", "clean": None}
    if not (ROOT / ".git").exists():
        # Worktrees have a .git file; source archives have neither a file nor
        # a directory here and must not inherit another checkout's identity.
        result["reason"] = "Source directory has no .git metadata"
        return result
    try:
        for key, arguments in (("head", ["rev-parse", "HEAD"]),
                               ("status", ["status", "--porcelain=v1", "--untracked-files=all"])):
            process = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True,
                                     text=True, encoding="utf-8", errors="replace", timeout=10,
                                     check=True)
            result[key] = process.stdout.rstrip("\r\n")
        result["clean"] = not bool(result["status"])
    except (OSError, subprocess.SubprocessError) as exc:
        result.update(head=None, status="unknown", clean=None,
                      error=f"{type(exc).__name__}: {exc}",
                      reason="Checkout metadata unavailable")
    return result


def initial_report(mode):
    return {"schema_version": 1, "gate": GATE, "passed": False, "stage": "initialization",
            "created_at": datetime.now(timezone.utc).isoformat(), "mode": mode, "scope": SCOPE,
            "model_id": artifacts.MANIFEST["model_id"], "revision": artifacts.MANIFEST["revision"],
            "stages_seconds": {}}


def save_failure_evidence(path, value, report):
    """A secondary evidence failure must not replace the active validation error."""
    try:
        write_json(path, value)
    except (OSError, TypeError, ValueError) as exc:
        report.setdefault("evidence_errors", []).append(
            f"{path.name}: {type(exc).__name__}: {exc}")


def run_validation(model_dir, out, report, *, mode="verify", manifest=None, progress=None):
    """Run real parsing. Small test fixtures may explicitly supply a manifest.

    The CLI always uses the committed full-model manifest and architecture.
    This function never replaces parsing with a handler inventory or ORT run.
    """
    import onnx
    from scratchv.analysis.ir_verifier import verify_ir
    from probes.w2_qwen3_parse.audit import AuditedONNXParser, audit_parsed_model
    from probes.w2_qwen3_parse.validation import audit_graph_structure

    manifest = artifacts.MANIFEST if manifest is None else manifest
    model_dir, out = Path(model_dir).resolve(), Path(out).resolve()
    report["model_dir"] = str(model_dir)

    def stage(name, operation):
        report["stage"] = name
        if progress:
            progress({"stage": name})
        started = time.perf_counter()
        try:
            return operation()
        finally:
            report["stages_seconds"][name] = time.perf_counter() - started

    if mode == "download":
        report["acquisition"] = stage("acquire", lambda: artifacts.acquire_model(model_dir, manifest=manifest))
    report["hashes"] = stage("hashes", lambda: artifacts.verify_files(model_dir, manifest))
    report["onnx_structure"] = stage("onnx-structure", lambda: artifacts.inspect_model(model_dir, report["hashes"]))
    model = stage("graph-load", lambda: onnx.load(str(model_dir / "model.onnx"), load_external_data=False))
    report["graph"] = stage("graph-connectivity", lambda: audit_graph_structure(model))
    if report["graph"].get("passed") is not True:
        raise ValueError("Complete Qwen3 graph connectivity did not pass")
    parser = AuditedONNXParser(progress=progress)
    try:
        program = stage("scratchv-parse", lambda: parser.parse(str(model_dir / "model.onnx")))
    except Exception:
        report["failure_context"] = dict(parser.current_context)
        report["nodes_completed"] = sum(row.get("status") == "translated" for row in parser.node_records)
        save_failure_evidence(out / "nodes.partial.json", parser.node_records, report)
        raise
    report["ir"] = {"functions": len(program.functions), "globals": len(program.global_values),
                    "blocks": sum(len(f.blocks) for f in program.functions),
                    "instructions": sum(len(b.instructions) for f in program.functions for b in f.blocks),
                    "operators": dict(sorted(Counter(i.opcode.value for f in program.functions
                                                       for b in f.blocks for i in b.instructions).items()))}
    (out / "ir.txt").write_text(program.dump(), encoding="utf-8")
    write_json(out / "nodes.json", parser.node_records)
    verified, issues = stage("ir-verifier", lambda: verify_ir(program, stage="full-qwen3-after-parse"))
    report["ir_verifier"] = {"passed": verified, "issue_count": len(issues),
                             "issues": [str(issue) for issue in issues]}
    if not verified:
        raise ValueError("ScratchV IR verifier rejected the complete model; see ir_verifier.issues")
    try:
        audit = stage("ir-and-bindings", lambda: audit_parsed_model(model, parser, program, model_dir))
    except Exception:
        report["failure_context"] = dict(parser.current_context)
        save_failure_evidence(out / "nodes.json", parser.node_records, report)
        raise
    nodes = audit.pop("node_records", parser.node_records)
    bindings = audit.pop("tensor_bindings")
    write_json(out / "nodes.json", nodes)
    write_json(out / "bindings.json", bindings)
    report["audit"] = audit
    if audit.get("passed") is not True:
        raise ValueError("IR/weight binding audit did not pass")
    report["evidence"] = {name: {"sha256": artifacts.sha256(out / name),
                                 "bytes": (out / name).stat().st_size}
                          for name in ("nodes.json", "bindings.json", "ir.txt")}
    report["passed"] = True
    report["stage"] = "complete"


def worker_main(args):
    report = initial_report(args.mode)
    out = args.output_dir.resolve()
    started = time.perf_counter()
    last_checkpoint = -math.inf

    def checkpoint(context):
        nonlocal last_checkpoint
        now = time.perf_counter()
        # Stage boundaries are always recorded; frequent node progress is sampled.
        if set(context) == {"stage"} or now - last_checkpoint >= 0.5:
            write_json(out / "worker-progress.json", {"context": context,
                       "elapsed_seconds": now - started, "peak_memory": artifacts.peak_memory()})
            last_checkpoint = now

    try:
        report["stage"] = "environment"
        checkpoint({"stage": "environment"})
        report["environment"] = artifacts.environment()
        report["checkout"] = checkout_evidence()
        report["source_sha256"] = source_fingerprints()
        run_validation(args.model_dir, out, report, mode=args.mode, progress=checkpoint)
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    report["validation_seconds"] = time.perf_counter() - started
    report["peak_memory"] = artifacts.peak_memory()
    report["peak_memory"]["scope"] = "Validation worker process, including loading and real ScratchV parsing; parent excluded"
    write_json(out / "worker-report.json", report)
    return 0 if report["passed"] else 1


def worker_command(args):
    return [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--worker",
            "--mode", args.mode, "--model-dir", str(args.model_dir.resolve()),
            "--output-dir", str(args.output_dir.resolve())]


def launch_worker(args):
    from scratchv.runtime.riscv_tensor import _run_process
    return _run_process(worker_command(args), cwd=ROOT, timeout=args.timeout)


def number(value):
    return "未测得" if value is None else f"{value:.3f}"


def save_worker_logs(out, report, stdout, stderr):
    """Keep reporting the validation result even when a log file is locked."""
    for name, content in (("worker.stdout", stdout), ("worker.stderr", stderr)):
        try:
            (out / name).write_bytes(content or b"")
        except OSError as exc:
            report.setdefault("log_errors", []).append(f"{name}: {type(exc).__name__}: {exc}")
    if report.get("log_errors") and report.get("passed"):
        report.update(passed=False, stage="log-write", error="Could not preserve validation worker logs")


def _render_reports(report):
    graph, audit, ir = report.get("graph", {}), report.get("audit", {}), report.get("ir", {})
    layers = graph.get("layers", [])
    peak = report.get("peak_memory", {})
    lines = ["# 完整 Qwen3 ONNX → ScratchV IR 解析验收", "",
             f"Result: **{'PASS' if report['passed'] else 'FAIL'}**；阶段：`{report['stage']}`", "",
             "只验证完整解析、结构、IR 和实际权重绑定；不证明完整 IR 数值或 QEMU 前向通过。", "",
             f"- 模型：`{report['model_id']}`，revision `{report['revision']}`。",
             f"- 源码 HEAD：`{report.get('checkout', {}).get('head', 'unavailable')}`；"
             f"工作树 clean：`{report.get('checkout', {}).get('clean', 'unavailable')}`。",
             "- dirty 工作树的结果须结合 source_sha256 和本地 diff，不能只归属 HEAD。",
             f"- 完整层链：{len(layers) if isinstance(layers, list) else layers} 层；"
             f"ONNX 节点：{report.get('onnx_structure', {}).get('operators') and sum(report['onnx_structure']['operators'].values()) or '未完成'}。",
             f"- IR：{ir.get('instructions', '未完成')} 条指令、{ir.get('globals', '未完成')} 个 globals；"
             f"verifier：{report.get('ir_verifier', {}).get('passed', '未完成')}。",
             f"- Supervisor 总耗时：{number(report.get('seconds'))} 秒；"
             f"实际解析：{number(report.get('stages_seconds', {}).get('scratchv-parse'))} 秒。",
             f"- 验证进程峰值 RSS：{peak.get('bytes', '未测得')} bytes；{peak.get('method', '')}。", "",
             "| 阶段 | 耗时（秒） |", "|---|---:|"]
    lines += [f"| {name} | {number(seconds)} |" for name, seconds in report.get("stages_seconds", {}).items()]
    if report.get("error"):
        lines += ["", "## 失败信息", "", report["error"], "",
                  "```json", json.dumps(report.get("failure_context", report.get("last_progress", {})),
                                        ensure_ascii=False, indent=2), "```"]
    if report.get("log_errors"):
        lines += ["", "子进程日志落盘失败：", *report["log_errors"]]
    if report.get("evidence_errors"):
        lines += ["", "失败明细保存异常（原始验证错误保留）：", *report["evidence_errors"]]
    if report.get("report_write_errors"):
        lines += ["", "报告保存异常（原始验证错误保留）：", *report["report_write_errors"]]
    concise_audit = {key: value for key, value in audit.items() if key != "aliases"}
    lines += ["", "## 审计摘要", "", "```json", json.dumps(concise_audit, ensure_ascii=False, indent=2), "```", "",
              "`nodes.json` 保存节点翻译记录；`bindings.json` 保存逐张量绑定和内容哈希；"
              "`ir.txt` 保存 IR 文本，均不包含原始权重数组。", "",
              "缺文件、解析错误、审计失败、超时或进程异常均失败；未执行的检查不能视为通过。"
              "耗时和内存为观察数据，没有新增性能通过阈值。"]
    markdown = "\n".join(lines) + "\n"
    details = json.dumps({"graph": graph, "audit": audit, "ir_verifier": report.get("ir_verifier")},
                         ensure_ascii=False, indent=2)
    html = (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Qwen3 完整解析验收</title>"
        "<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:32px auto;padding:0 16px}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ddd;padding:12px}</style>"
        f"<h1>frontend:parse-qwen3 — {'PASS' if report['passed'] else 'FAIL'}</h1>"
        f"<pre>{escape(markdown)}</pre><details><summary>结构与绑定详情</summary>"
        f"<pre>{escape(details)}</pre></details></html>")
    return {"report.md": markdown, "report.html": html}


def write_reports(out, report):
    """Publish JSON last, so incomplete evidence cannot leave a stale PASS."""
    failures = []

    def record_failure(name, exc):
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
        report["report_write_errors"] = failures
        if report["passed"]:
            report.update(passed=False, stage="report-write",
                          error="Could not preserve complete validation reports")

    def write_views(*, record_errors):
        for name, content in _render_reports(report).items():
            try:
                (out / name).write_text(content, encoding="utf-8")
            except OSError as exc:
                if record_errors:
                    record_failure(name, exc)

    write_views(record_errors=True)
    if failures:
        # A view written before another failed may still contain PASS. Refresh
        # every writable view, preserving the original validation failure.
        write_views(record_errors=False)
    try:
        write_json(out / "report.json", report)
    except OSError as exc:
        record_failure("report.json", exc)
        write_views(record_errors=False)
    for error in failures:
        print(error, file=sys.stderr)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mode", choices=("verify", "download"), default="verify")
    cli.add_argument("--model-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, default=ROOT / "output/qwen3-parse")
    cli.add_argument("--timeout", type=float, default=300)
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = cli.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        cli.error("--timeout must be finite and positive")
    if args.worker:
        return worker_main(args)
    out = args.output_dir.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        cli.error("--output-dir must be absent or empty; preserve previous evidence")
    out.mkdir(parents=True, exist_ok=True)
    report = initial_report(args.mode)
    report.update(stage="worker-launch", timeout_seconds=args.timeout)
    started = time.perf_counter()
    result = None
    try:
        # Keep the attempted run identifiable even if the child never starts or
        # exits before it can validate dependencies and write its own report.
        report["attempt"] = {"command": worker_command(args), "cwd": str(ROOT),
                             "model_dir": str(args.model_dir.resolve()),
                             "python": sys.version, "checkout": checkout_evidence(),
                             "source_sha256": source_fingerprints(),
                             "scope": "Supervisor metadata; not proof of worker initialization or validation"}
        result = launch_worker(args)
        report["worker_returncode"] = result.returncode
        worker = json.loads((out / "worker-report.json").read_text(encoding="utf-8"))
        if worker.get("gate") != GATE or worker.get("schema_version") != 1:
            raise ValueError("Invalid validation worker report")
        report.update(worker)
        if result.returncode or report.get("passed") is not True or report.get("stage") != "complete":
            report["passed"] = False
            report.setdefault("error", f"Validation worker did not pass (exit {result.returncode})")
        elif any(not (out / name).is_file() for name in ("nodes.json", "bindings.json", "ir.txt")):
            raise ValueError("Validation worker omitted required evidence files")
    except subprocess.TimeoutExpired as exc:
        report.update(passed=False, stage="timeout", error=f"Validation worker exceeded {args.timeout} seconds")
        save_worker_logs(out, report, exc.stdout, exc.stderr)
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if result is not None:
            save_worker_logs(out, report, result.stdout, result.stderr)
    report["seconds"] = time.perf_counter() - started
    if not report["passed"] and (out / "worker-progress.json").is_file():
        try:
            report["last_progress"] = json.loads((out / "worker-progress.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    write_reports(out, report)
    print(f"[{GATE}] {'PASS' if report['passed'] else 'FAIL'}; stage={report['stage']}; "
          f"seconds={report['seconds']:.3f}; report={out / 'report.json'}", flush=True)
    if report.get("error"):
        print(report["error"], file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
