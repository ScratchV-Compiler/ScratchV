"""Tests for the peephole benchmark data collection helpers."""

from __future__ import annotations

import json
import random
import subprocess

import pytest

import benchmarks.bench_asm_peephole as bench
from benchmarks.bench_asm_peephole import (
    PR39_RULES,
    BenchmarkCase,
    _gen_synthetic_asm,
    bench_optimize,
    count_instructions,
    default_cases,
    measure_case,
    run_benchmark,
    save_json,
)
from scratchv.backend.asm_peephole import AsmPeepholeOptimizer


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

    assert {case.expected_rule for case in cases if case.expected_rule} == set(
        PR39_RULES
    )
    assert {rule.name for rule in AsmPeepholeOptimizer().rules} == set(PR39_RULES)


def test_default_expected_cases_hit_and_negative_cases_do_not_change():
    results = [measure_case(case, repeats=1) for case in default_cases()]

    assert all(
        result["expected_rule_hit"]
        for result in results
        if result["expected_rule"] is not None
    )
    assert all(
        result["changes"] == 0 for result in results if result["category"] == "negative"
    )


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


def test_synthetic_size_includes_the_final_ret_instruction():
    for size in (1, 2, 100):
        assert count_instructions(_gen_synthetic_asm(size)) == size


def test_synthetic_fusion_ratio_controls_addi_pair_instructions_only():
    size = 2001
    body_size = size - 1

    for ratio in (0.0, 0.3, 1.0):
        assembly = _gen_synthetic_asm(size, seed=7, fusion_ratio=ratio)
        addi_count = sum(
            line.strip().startswith("addi ") for line in assembly.splitlines()
        )
        expected_addi = int(body_size * ratio) // 2 * 2
        assert count_instructions(assembly) == size
        assert addi_count == expected_addi

        optimizer = AsmPeepholeOptimizer()
        _, changes = optimizer.optimize(assembly)
        assert changes == expected_addi // 2
        assert sum(optimizer.total_matches.values()) == changes
        if ratio == 0.0:
            assert changes == 0
            assert all(value == 0 for value in optimizer.total_matches.values())


def test_synthetic_generation_is_seeded_without_global_random_state():
    random.seed(12345)
    expected_random = [random.random() for _ in range(3)]
    random.seed(12345)

    first = _gen_synthetic_asm(101, seed=99, fusion_ratio=0.3)
    actual_random = [random.random() for _ in range(3)]

    assert actual_random == expected_random
    assert _gen_synthetic_asm(101, seed=99, fusion_ratio=0.3) == first


def test_git_commit_is_worktree_head_even_when_cwd_is_elsewhere(tmp_path, monkeypatch):
    expected = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={bench._REPO_ROOT}",
            "rev-parse",
            "HEAD",
        ],
        cwd=bench._REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    monkeypatch.chdir(tmp_path)

    assert bench._git_commit() == expected


def test_bench_rejects_invalid_repeats_size_and_ratio():
    with pytest.raises(ValueError, match="repeats"):
        measure_case(default_cases()[0], repeats=0)
    with pytest.raises(ValueError, match="num_instrs"):
        _gen_synthetic_asm(0)
    with pytest.raises(ValueError, match="fusion_ratio"):
        _gen_synthetic_asm(2, fusion_ratio=1.1)
