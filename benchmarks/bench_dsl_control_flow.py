"""Topic 01 semantic benchmark for DSL control flow."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import html
import json
import platform
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

from llvmlite import binding as llvm

from scratchv.backend.asm_emit import AsmEmitter
from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.llvm_codegen import LLVMCodegen
from scratchv.backend.register_alloc import RegisterAllocator
from scratchv.frontend.dsl_extended import ExtendedDSLParser


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "topic01"
CASES = (
    ("if_else", "if_else.dsl", 7.0),
    ("while_sum", "while_sum.dsl", 15.0),
    ("nested_loop", "nested_loop.dsl", 6.0),
)


def _revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _compile_and_execute(source: str) -> dict[str, Any]:
    program = ExtendedDSLParser().parse(source)
    llvm_ir = LLVMCodegen(program).emit()
    module = llvm.parse_assembly(llvm_ir)
    module.verify()
    target = llvm.Target.from_default_triple()
    engine = llvm.create_mcjit_compiler(module, target.create_target_machine())
    engine.finalize_object()
    address = engine.get_function_address("main")
    if not address:
        raise RuntimeError("LLVM did not emit main")
    actual = float(ctypes.CFUNCTYPE(ctypes.c_float)(address)())

    machine = InstructionSelector(program).run()
    allocated = RegisterAllocator(machine, mode="greedy").run()
    assembly = AsmEmitter(allocated).emit()
    if not assembly.strip():
        raise RuntimeError("RISC-V backend emitted empty assembly")

    function = program.functions[0]
    return {
        "actual": actual,
        "llvm_ir": llvm_ir,
        "assembly": assembly,
        "blocks": len(function.blocks),
        "ir_instructions": sum(len(block.instructions) for block in function.blocks),
        "machine_instructions": len(machine),
    }


def run_benchmark(repeats: int) -> dict[str, Any]:
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    cases = []
    for name, filename, expected in CASES:
        source = (EXAMPLES / filename).read_text(encoding="utf-8")
        samples = []
        result: dict[str, Any] = {}
        error = None
        try:
            for _ in range(repeats):
                started = time.perf_counter()
                result = _compile_and_execute(source)
                samples.append((time.perf_counter() - started) * 1000)
        except Exception as exc:  # Report failures instead of losing artifacts.
            error = f"{type(exc).__name__}: {exc}"
        actual = result.get("actual")
        passed = error is None and actual == expected
        cases.append({
            "name": name,
            "filename": filename,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "source": source,
            "expected": expected,
            "actual": actual,
            "passed": passed,
            "llvm_verified": error is None and bool(result.get("llvm_ir")),
            "riscv_generated": error is None and bool(result.get("assembly")),
            "blocks": result.get("blocks"),
            "ir_instructions": result.get("ir_instructions"),
            "machine_instructions": result.get("machine_instructions"),
            "samples_ms": samples,
            "median_ms": statistics.median(samples) if samples else 0.0,
            "error": error,
            "llvm_ir": result.get("llvm_ir", ""),
            "riscv_assembly": result.get("assembly", ""),
        })
    return {
        "schema_version": 1,
        "benchmark": "Topic 01 DSL frontend benchmark",
        "status": "passed" if all(case["passed"] for case in cases) else "failed",
        "revision": _revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repeats": repeats,
        "cases": cases,
    }


def markdown_report(report: dict[str, Any], *, include_logs: bool = True) -> str:
    lines = [
        "# Topic 01 DSL frontend benchmark",
        "",
        f"Status: **{report['status']}**; repeats: {report['repeats']}.",
        "",
        "| Case | Expected | Actual | LLVM verified | RISC-V generated | Median time (ms) |",
        "|---|---:|---:|---|---|---:|",
    ]
    for case in report["cases"]:
        lines.append(
            f"| {case['name']} | {case['expected']:.1f} | {case['actual']} | "
            f"{case['llvm_verified']} | {case['riscv_generated']} | {case['median_ms']:.3f} |"
        )
    if include_logs:
        lines += ["", "## Case logs", ""]
        for case in report["cases"]:
            label = f"{case['name']}: {'passed' if case['passed'] else 'failed'} - view logs"
            lines += [
                "<details>",
                f"<summary>{html.escape(label)}</summary>",
                "",
                "### Source",
                "```text",
                case["source"].rstrip(),
                "```",
                "",
                "### LLVM IR",
                "```llvm",
                case["llvm_ir"].rstrip(),
                "```",
                "",
                "### RISC-V assembly",
                "```asm",
                case["riscv_assembly"].rstrip(),
                "```",
                "",
                "</details>",
                "",
            ]
    return "\n".join(lines)


def html_report(report: dict[str, Any]) -> str:
    summary = html.escape(markdown_report(report, include_logs=False))
    details = []
    for case in report["cases"]:
        label = f"{case['name']}: {'passed' if case['passed'] else 'failed'} - view logs"
        log = (
            "Source\n" + case["source"]
            + "\nLLVM IR\n" + case["llvm_ir"]
            + "\nRISC-V assembly\n" + case["riscv_assembly"]
        )
        details.append(
            f"<details><summary>{html.escape(label)}</summary>"
            f"<pre>{html.escape(log)}</pre></details>"
        )
    return (
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
        "<title>Topic 01 DSL frontend benchmark</title><style>"
        "body{max-width:1100px;margin:32px auto;padding:0 20px;font:15px/1.6 system-ui}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:20px}"
        "details{margin:12px 0;border:1px solid #d8dee6;border-radius:6px}"
        "summary{padding:12px;cursor:pointer}details pre{margin:0}"
        "</style><body><pre>" + summary + "</pre>"
        "<h2>Case logs</h2>" + "".join(details) + "</body></html>\n"
    )


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=positive_int, default=20)
    parser.add_argument("--json", action="store_true", help="emit JSON to stdout")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--html", type=Path)
    args = parser.parse_args(argv)

    report = run_benchmark(args.repeats)
    json_text = json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    markdown = markdown_report(report)
    outputs = (
        (args.json_output, json_text),
        (args.markdown, markdown + "\n"),
        (args.html, html_report(report)),
    )
    for path, content in outputs:
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    print(json_text if args.json else markdown)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
