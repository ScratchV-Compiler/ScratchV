"""Report integration, execution evidence, and failed/zero-coverage cases."""

import hashlib
import json

import pytest

from benchmarks import bench_inst_scheduler as synthetic
from benchmarks import run_inst_scheduler_case as reports


def test_feature_report_uses_compiler_and_executes_equivalently():
    pytest.importorskip("tinyfive")
    report = reports.run_case(reports.DEFAULT_CASE, repeats=1)
    assert report["status"] == "passed"
    assert report["feature"]["baseline_schedule"] is False
    assert report["feature"]["compiler_config_schedule"] is True
    assert report["feature"]["pipeline_matches_public_pass"] is True
    assert report["feature"]["used"] is True
    assert (
        report["input_sha256"]
        == hashlib.sha256(reports.DEFAULT_CASE.read_bytes()).hexdigest()
    )
    assert report["source_instructions"] == {"before": 4, "after": 4}
    scheduling = report["scheduling"]
    assert (scheduling["original_cycles"], scheduling["final_cycles"]) == (5, 4)
    assert (scheduling["original_stalls"], scheduling["final_stalls"]) == (1, 0)
    assert scheduling["moved_instructions"] == 2
    sim = report["simulation"]
    assert sim["backend"] == "tinyfive" and sim["fallback"] is False
    assert sim["output_equal"] and sim["registers_equal"] and sim["memory_equal"]
    for result in (sim["before"], sim["after"]):
        assert result["encoded_instructions"] == result["executed_instructions"] == 4
        assert result["code_size_bytes"] == 16
        assert len(result["all_registers"]) == 32
        assert result["all_registers"][5:8] == [9, 16, 7]
        assert result["all_registers"][28] == 1
        assert result["memory_words"] == [9, 16] + [0] * 14
    markdown = reports._markdown(report)
    assert "| 局部模型周期（估算） | 5 | 4 | 1 |" in markdown
    assert "| TinyFive 实际执行指令数 | 4 | 4 | 0 |" in markdown
    assert "不能证明硬件加速" in markdown
    assert report["assembly"]["before"].rstrip() in markdown
    assert report["assembly"]["after"].rstrip() in markdown


def test_static_ab_retains_zero_gain_and_never_simulates(tmp_path, monkeypatch):
    source = b".text\r\nf:\r\n  add t0, t1, t2\r\n  add t3, t0, t4"
    path = tmp_path / "zero-hit.s"
    path.write_bytes(source)

    def forbidden(*args):
        pytest.fail("static reports must not pretend to execute a real program")

    monkeypatch.setattr(reports, "_execute", forbidden)
    report = reports.run_case(path, static_only=True, repeats=1)
    assert report["status"] == "passed"
    assert report["benchmark_type"] == "assembly-ab"
    assert report["feature"]["used"] is False
    assert report["scheduling"]["saved_cycles"] == 0
    assert report["scheduling"]["modeled_instructions"] == 2
    assert report["assembly"]["before"].encode() == source
    assert report["assembly"]["after"].encode() == source
    assert report["simulation"]["status"] == "not_run"
    assert report["simulation"]["output_equal"] is None
    assert "机器码大小、动态指令数及硬件耗时均未测量" in reports._markdown(report)


@pytest.mark.parametrize("source", ["custom t0, t1\n", "addi t0, t0, 1\n" * 1025])
def test_unmodeled_inputs_are_not_reported_as_zero_cycle_runs(tmp_path, source):
    path = tmp_path / "unmodeled.s"
    path.write_text(source)
    report = reports.run_case(path, static_only=True, repeats=1)
    assert report["status"] == "passed"
    assert report["scheduling"]["modeled_instructions"] == 0
    assert report["assembly"]["after"] == source
    assert "| 局部模型周期（估算） | N/A | N/A | N/A |" in reports._markdown(report)
    assert report["scheduling"]["diagnostics"]


def test_standalone_listing_labels_and_numeric_branches_are_reported_accurately(
    tmp_path,
):
    source = "_op_/layer1.0/Conv_5:\n  lw t0, 0(a0)\n  bne t1, zero, -4\n"
    path = tmp_path / "cnn-listing.s"
    path.write_text(source)
    report = reports.run_case(path, static_only=True, repeats=1)
    assert report["source_instructions"] == {"before": 2, "after": 2}
    assert report["comparison_status"] == "not_modeled"
    assert report["assembly"]["after"] == source
    assert (
        "numeric control-flow target"
        in report["scheduling"]["diagnostics"][0]["reason"]
    )
    assert "0/2 条源指令" in reports._markdown(report)


def test_missing_simulator_fails_and_replaces_stale_reports(tmp_path, monkeypatch):
    class UnavailableMachine:
        def __init__(self, **kwargs):
            self.available = False

    monkeypatch.setattr(reports, "ProfiledMachine", UnavailableMachine)
    json_path, md_path = tmp_path / "report.json", tmp_path / "nested/report.md"
    json_path.write_text('{"status":"passed"}')
    status = reports.main(
        ["--json", str(json_path), "--markdown", str(md_path), "--repeats", "1"]
    )
    assert status == 1
    report = json.loads(json_path.read_text())
    assert report["status"] == "failed"
    assert "fallback is forbidden" in report["error"]
    assert "FAIL" in md_path.read_text()


def test_execution_mismatch_is_failed_with_evidence(tmp_path, monkeypatch):
    pytest.importorskip("tinyfive")
    execute = reports._execute
    calls = 0

    def corrupt_after(source):
        nonlocal calls
        calls += 1
        result = execute(source)
        if calls == 2:
            result["memory_words"][1] += 1
        return result

    monkeypatch.setattr(reports, "_execute", corrupt_after)
    json_path, md_path = tmp_path / "report.json", tmp_path / "report.md"
    status = reports.main(
        ["--json", str(json_path), "--markdown", str(md_path), "--repeats", "1"]
    )
    assert status == 1
    report = json.loads(json_path.read_text())
    assert report["status"] == "failed"
    assert report["simulation"]["registers_equal"] is True
    assert report["simulation"]["memory_equal"] is False
    assert report["simulation"]["output_equal"] is False
    assert "execution changed register or memory outputs" in md_path.read_text()
    assert "| memory[1028] | 16 | 17 |" in md_path.read_text()


def test_feature_case_requires_actual_scheduling_gain(tmp_path):
    pytest.importorskip("tinyfive")
    path = tmp_path / "no-gain.s"
    path.write_text("add t1, t0, t2\n")
    report = reports.run_case(path, repeats=1)
    assert report["status"] == "failed"
    assert report["simulation"]["output_equal"] is True
    assert "did not exercise" in report["errors"][0]


@pytest.mark.parametrize(
    "source",
    [
        "j loop\nloop:\nadd t0, t1, t2\n",
        "fadd.s f0, f1, f2\n",
        "li a0, 32\n",
        "sw t0, 64(a0)\n",
        "sw t0, 0(t1)\n",
        ".data\n.word 1\n",
    ],
)
def test_execution_harness_rejects_unsupported_programs(source):
    with pytest.raises(ValueError):
        reports._execute(source)


def test_synthetic_skip_is_explicit_and_not_execution_evidence():
    record = synthetic.bench_schedule(
        synthetic._gen_instructions(5), repeats=1, max_region_size=4
    )
    assert record["benchmark_type"] == "synthetic"
    assert record["modeled"] == 0 and record["skipped"] == 1
    assert record["execution_verified"] is False
    markdown = synthetic._markdown([record])
    assert "N/A" in markdown and "未执行汇编" in markdown


def test_report_rejects_missing_compiler_integration(monkeypatch):
    monkeypatch.setattr(
        reports.CompilerDriver, "_run_asm_passes", lambda self, source, *args: source
    )
    with pytest.raises(AssertionError, match="did not report scheduling statistics"):
        reports.run_case(reports.DEFAULT_CASE, static_only=True, repeats=1)
