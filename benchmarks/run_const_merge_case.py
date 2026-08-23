#!/usr/bin/env python3
"""Run one constant-merge feature case and emit auditable CI reports.

The report proves three separate facts:

1. the compiler's configured assembly post-pass invokes constant merge;
2. the pass changes the selected feature case and reports categorized metrics;
3. real TinyFive execution produces identical architectural register state
   before and after the transformation.

This is a deterministic feature/integration case, not a real-workload speedup
claim.  Real ONNX zero-hit results remain separate in ``run_benchmark.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scratchv.backend._asm_parser import parse_asm
from scratchv.backend.const_merge import merge_constants_detailed
from scratchv.backend.riscv_encoder import assemble_to_binary
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.simulator.tinyfive import ProfiledMachine


DEFAULT_CASE = Path(__file__).parent / "cases" / "const_merge_feature.asm"
DEFAULT_JSON = Path("benchmark_reports/const_merge_report.json")
DEFAULT_MARKDOWN = Path("benchmark_reports/const_merge_report.md")
OBSERVED_REGISTERS = (5, 6, 7)


def _instruction_count(asm_text: str) -> int:
    return sum(
        line.opcode is not None and not line.is_directive
        for line in parse_asm(asm_text)
    )


def _binary_words(asm_text: str) -> tuple[bytearray, list[int]]:
    binary = assemble_to_binary(asm_text)
    if not binary or len(binary) % 4:
        raise ValueError(
            f"encoder returned invalid binary length: {len(binary)} bytes"
        )
    words = [
        int.from_bytes(binary[i:i + 4], "little")
        for i in range(0, len(binary), 4)
    ]
    return binary, words


def _simulate(words: list[int]) -> dict[str, Any]:
    machine = ProfiledMachine(mem_size=max(4096, len(words) * 4 + 64))
    if not machine.available:
        raise RuntimeError("real TinyFive is unavailable; fallback is forbidden")

    machine.load_binary(words, origin=0)
    machine.run(instructions=len(words), start=0, strict=True)
    if machine.last_error is not None:
        raise RuntimeError(machine.last_error)

    all_registers = [machine.get_reg(i) for i in range(32)]
    return {
        "backend": "tinyfive",
        "fallback": False,
        "executed_instructions": machine.instr_count,
        "registers": {
            f"x{i}": all_registers[i] for i in OBSERVED_REGISTERS
        },
        "all_registers": all_registers,
        "perf_counters": machine.get_perf(),
    }


def run_case(case_path: Path) -> dict[str, Any]:
    asm_before = case_path.read_text(encoding="utf-8")
    driver = CompilerDriver(CompilerConfig(const_merge=True))
    warnings: list[str] = []

    start = time.perf_counter()
    asm_after = driver._run_asm_passes(asm_before, warnings)
    pass_time_ms = (time.perf_counter() - start) * 1000

    direct_after, merge_stats = merge_constants_detailed(asm_before)
    pipeline_matches_public_pass = asm_after == direct_after
    if not pipeline_matches_public_pass:
        raise AssertionError(
            "CompilerDriver post-pass output differs from constant-merge API"
        )

    binary_before, words_before = _binary_words(asm_before)
    binary_after, words_after = _binary_words(asm_after)
    simulation_before = _simulate(words_before)
    simulation_after = _simulate(words_after)
    output_equal = (
        simulation_before["all_registers"]
        == simulation_after["all_registers"]
    )

    source_before = _instruction_count(asm_before)
    source_after = _instruction_count(asm_after)
    feature_used = (
        driver.config.const_merge
        and merge_stats.total_changes > 0
        and pipeline_matches_public_pass
        and any("Const merge:" in warning for warning in warnings)
    )
    success = feature_used and output_equal

    return {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_type": "feature-case",
        "case": str(case_path),
        "status": "passed" if success else "failed",
        "feature": {
            "name": "RV32 constant-load merge",
            "compiler_config_const_merge": driver.config.const_merge,
            "compiler_path": "CompilerDriver._run_asm_passes",
            "pipeline_matches_public_pass": pipeline_matches_public_pass,
            "used": feature_used,
            "warnings": warnings,
        },
        "optimization": {
            **asdict(merge_stats),
            "pass_time_ms": pass_time_ms,
            "source_instructions_before": source_before,
            "source_instructions_after": source_after,
            "source_instruction_reduction": source_before - source_after,
        },
        "machine_code": {
            "instructions_before": len(words_before),
            "instructions_after": len(words_after),
            "instruction_reduction": len(words_before) - len(words_after),
            "code_size_before": len(binary_before),
            "code_size_after": len(binary_after),
        },
        "simulation": {
            "backend": "tinyfive",
            "fallback": False,
            "before": simulation_before,
            "after": simulation_after,
            "output_equal": output_equal,
        },
        "assembly": {
            "before": asm_before,
            "after": asm_after,
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    opt = report["optimization"]
    machine = report["machine_code"]
    sim = report["simulation"]
    before_sim = sim["before"]
    after_sim = sim["after"]
    passed = report["status"] == "passed"

    lines = [
        "# Topic 14 Constant-Merge Case Report",
        "",
        f"- Status: {'PASS' if passed else 'FAIL'}",
        f"- Case type: `{report['benchmark_type']}`",
        f"- Case: `{report['case']}`",
        f"- Compiler const-merge enabled: `{report['feature']['compiler_config_const_merge']}`",
        f"- Feature used: `{report['feature']['used']}`",
        f"- Simulation backend: `{sim['backend']}`",
        f"- Static-analysis fallback: `{sim['fallback']}`",
        f"- Output equal: `{sim['output_equal']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Before | After | Reduction |",
        "|---|---:|---:|---:|",
        f"| Source assembly instructions | {opt['source_instructions_before']} | {opt['source_instructions_after']} | {opt['source_instruction_reduction']} |",
        f"| Encoded machine instructions | {machine['instructions_before']} | {machine['instructions_after']} | {machine['instruction_reduction']} |",
        f"| Code size (bytes) | {machine['code_size_before']} | {machine['code_size_after']} | {machine['code_size_before'] - machine['code_size_after']} |",
        f"| TinyFive executed instructions | {before_sim['executed_instructions']} | {after_sim['executed_instructions']} | {before_sim['executed_instructions'] - after_sim['executed_instructions']} |",
        "",
        "| Constant-merge metric | Value |",
        "|---|---:|",
        f"| Structural candidates | {opt['candidate_pairs']} |",
        f"| Merged `lui`/`addi` pairs | {opt['merged_pairs']} |",
        f"| Redundant `lui` removed | {opt['redundant_lui_removed']} |",
        f"| Fixed-point iterations | {opt['iterations']} |",
        f"| Pass time (ms) | {opt['pass_time_ms']:.3f} |",
        "",
        "## Observable registers",
        "",
        "| Register | Before | After |",
        "|---|---:|---:|",
    ]
    for register, before_value in before_sim["registers"].items():
        lines.append(
            f"| {register} | {before_value} | "
            f"{after_sim['registers'][register]} |"
        )

    lines.extend([
        "",
        "## Assembly before",
        "",
        "```asm",
        report["assembly"]["before"].rstrip(),
        "```",
        "",
        "## Assembly after",
        "",
        "```asm",
        report["assembly"]["after"].rstrip(),
        "```",
        "",
        "> This is a deterministic feature case. It proves integration and",
        "> execution equivalence; it is not presented as a real-workload speedup.",
        "",
    ])
    return "\n".join(lines)


def _failure_report(case_path: Path, exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_type": "feature-case",
        "case": str(case_path),
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one constant-merge case with real TinyFive A/B simulation",
    )
    parser.add_argument("case", nargs="?", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    try:
        report = run_case(args.case)
    except Exception as exc:
        report = _failure_report(args.case, exc)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if report["status"] == "passed":
        markdown = _markdown(report)
    else:
        markdown = (
            "# Topic 14 Constant-Merge Case Report\n\n"
            f"- Status: FAIL\n- Error: `{report.get('error', 'unknown')}`\n"
        )
    args.markdown.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"JSON report: {args.json}")
    print(f"Markdown report: {args.markdown}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
