"""Tests for real before/after const-merge metrics in the CNN pipeline."""

import json

from scratchv.standalone.bench_report import (
    generate_github_summary,
    generate_html_report,
    generate_json_report,
)
from scratchv.standalone.onnx_to_riscv_standalone import RISCVEmitter, rv_li


OPTIMIZATION = {
    "enabled": True,
    "used": True,
    "candidate_pairs": 70,
    "merged_pairs": 70,
    "redundant_lui_removed": 0,
    "iterations": 2,
    "source_instructions_before": 887,
    "source_instructions_after": 817,
    "source_instruction_reduction": 70,
    "machine_instructions_before": 887,
    "machine_instructions_after": 876,
    "machine_instruction_reduction": 11,
    "code_size_before": 3548,
    "code_size_after": 3504,
    "code_size_reduction": 44,
}

ESTIMATION = {
    "total_estimated": 2_203_799_060,
    "cm_ratio": 2.5,
    "compute_ratio": 62.8,
    "memory_ratio": 24.8,
    "branch_ratio": 7.4,
    "est_hw_time_50mhz": 44.1,
    "est_hw_time_100mhz": 22.0,
    "per_layer": {},
}


def test_compact_li32_uses_real_one_or_two_word_lowering():
    baseline = RISCVEmitter()
    compact = RISCVEmitter(compact_li32=True)

    baseline.emit_li32(5, 288)
    compact.emit_li32(5, 288)

    assert len(baseline.code) == 2
    assert compact.code == list(rv_li(5, 288))
    assert len(compact.code) == 1

    for immediate in (-2147483648, -196608, 4098, 65536, 196608):
        baseline_large = RISCVEmitter()
        compact_large = RISCVEmitter(compact_li32=True)
        baseline_large.emit_li32(6, immediate)
        compact_large.emit_li32(6, immediate)

        assert compact_large.code == baseline_large.code
        assert len(compact_large.code) == 2


def test_github_summary_displays_real_before_after_metrics():
    summary = generate_github_summary(
        code_size=3504,
        static_insns=876,
        est_data=ESTIMATION,
        optimization=OPTIMIZATION,
    )

    assert "3,548 B → **3,504 B** (-44 B, 1.2%)" in summary
    assert "887 → **876** (-11)" in summary
    assert "| Encoded machine instructions | 887 | 876 | 11 |" in summary
    assert "| Merged `lui`/`addi` pairs | 70 |" in summary


def test_json_and_html_reports_preserve_ab_data():
    json_report = json.loads(generate_json_report(
        code_size=3504,
        static_insns=876,
        est_data=ESTIMATION,
        optimization=OPTIMIZATION,
    ))
    html_report = generate_html_report(
        code_size=3504,
        static_insns=876,
        est_data=ESTIMATION,
        optimization=OPTIMIZATION,
    )

    assert json_report["code"] == {
        "size_bytes": 3504,
        "static_instructions": 876,
    }
    assert json_report["constant_merge"]["code_size_before"] == 3548
    assert json_report["constant_merge"]["code_size_after"] == 3504
    assert "Constant-Merge A/B Result" in html_report
    assert "3,548 → 3,504 B (887 → 876 static insns)" in html_report
