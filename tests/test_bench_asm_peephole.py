"""Tests for the peephole benchmark data collection helpers."""

from __future__ import annotations

import json

from benchmarks.bench_asm_peephole import (
    BenchmarkCase,
    bench_optimize,
    count_instructions,
    default_cases,
    measure_case,
    run_benchmark,
    save_json,
)


def test_count_instructions_ignores_directives_labels_comments_and_blanks():
    asm = """.text
main:
  # comment-only line
  addi t0, t0, 1  # trailing comment
label: nop

  ret
"""

    assert count_instructions(asm) == 3


def test_default_cases_cover_all_pr39_rules():
    cases = default_cases()

    assert len(cases) >= 8
    assert {
        case.expected_rule
        for case in cases
        if case.expected_rule
    } >= {
        "addi+addi fusion",
        "li+addi fusion",
        "beq zero-zero to jump",
        "redundant mv elimination",
        "addi-zero self elimination",
        "addi-zero to mv",
        "nop elimination",
        "mv-self elimination",
    }


def test_measure_case_reports_static_reduction_and_rule_hits():
    case = BenchmarkCase(
        case_id="addi",
        assembly="addi t0, t0, 1\naddi t0, t0, 2\n",
        expected_rule="addi+addi fusion",
    )

    result = measure_case(case, repeats=2)

    assert result["before_instructions"] == 2
    assert result["after_instructions"] == 1
    assert result["reduced_instructions"] == 1
    assert result["reduction_percent"] == 50.0
    assert result["rule_matches"]["addi+addi fusion"] >= 1
    assert result["input_sha256"]


def test_run_benchmark_handles_zero_change_case_without_division_error():
    case = BenchmarkCase(
        case_id="clean",
        assembly="add t0, t1, t2\nret\n",
        expected_rule=None,
    )

    report = run_benchmark([case], repeats=1)

    assert report["summary"]["before_instructions"] == 2
    assert report["summary"]["after_instructions"] == 2
    assert report["summary"]["reduction_percent"] == 0.0
    assert report["cases"][0]["changes"] == 0


def test_save_json_writes_stable_machine_readable_fields(tmp_path):
    report = run_benchmark(default_cases()[:1], repeats=1)
    output = tmp_path / "raw.json"

    save_json(report, output)

    data = json.loads(output.read_text())
    assert data["schema_version"] == 1
    assert data["cases"]
    assert "before_instructions" in data["cases"][0]

def test_legacy_bench_helper_keeps_line_and_instruction_metrics():
    stats = bench_optimize("addi t0, t0, 1\naddi t0, t0, 2\n", repeats=2)

    assert stats["input_lines"] == 2
    assert stats["output_lines"] == 1
    assert stats["input_instructions"] == 2
    assert stats["output_instructions"] == 1
    assert stats["instruction_reduction"] == 1
    assert stats["changes_mean"] == 1.0
