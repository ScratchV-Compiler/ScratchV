"""QEMU gate provenance and failure propagation, with no target execution."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from probes.w2_qwen3_small import riscv as probe
from scratchv.runtime.riscv_tensor import RiscVTensorExecutionError, RiscVTensorTimeoutError


@pytest.fixture
def model_evidence(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # These bytes exercise artifact binding only; no fake model is executed.
    (model_dir / "model.onnx").write_bytes(b"ordinary model")
    (model_dir / "diagnostics.onnx").write_bytes(b"diagnostic model")
    schema = [{"name": f"checkpoint_{index}", "offset": index, "size": 1,
               "shape": [1], "dtype": "float32"} for index in range(28)]
    schema.append({"name": "logits", "offset": 28, "size": 1,
                   "shape": [1], "dtype": "float32"})
    (model_dir / "checkpoints.json").write_text(json.dumps(schema), encoding="utf-8")
    report = {"passed": True, "stage": "complete", "atol": probe.ATOL, "rtol": 0,
              "model_seed": 0, "weights_sha256": "0" * 64,
              "config": {"model_class": "transformers.Qwen3ForCausalLM"},
              "provenance": {"git_commit": "fixture"}, "checkpoints": schema,
              "onnx": {key: hashlib.sha256((model_dir / filename).read_bytes()).hexdigest()
                       for key, filename in (("model_sha256", "model.onnx"),
                                             ("diagnostics_sha256", "diagnostics.onnx"))},
              "cases": [{"name": name, "valid_length": length, "passed": True,
                         "input_sha256": probe.arrays_sha256(feed)}
                        for name, length, feed in probe.input_cases()]}
    (model_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return model_dir, report


def test_model_artifacts_keep_export_provenance(model_evidence):
    model_dir, source = model_evidence
    files, schema, evidence = probe.validated_model_artifacts(model_dir)
    assert set(files) == {"normal", "diagnostic"}
    assert schema == source["checkpoints"]
    assert evidence["report_sha256"] == hashlib.sha256((model_dir / "report.json").read_bytes()).hexdigest()
    assert evidence["weights_sha256"] == source["weights_sha256"]
    assert evidence["provenance"] == source["provenance"]


@pytest.mark.parametrize("fault", ["missing_report", "failed", "unfinished", "normal", "diagnostic",
                                   "schema", "input", "case_failed", "missing_case", "tolerance"])
def test_mixed_or_failed_exports_cannot_enter_qemu_gate(model_evidence, tmp_path, monkeypatch, fault):
    model_dir, source = model_evidence
    if fault == "failed":
        source["passed"] = False
    elif fault == "unfinished":
        source["stage"] = "numeric"
    elif fault in ("normal", "diagnostic"):
        filename = "model.onnx" if fault == "normal" else "diagnostics.onnx"
        (model_dir / filename).write_bytes(b"changed graph")
    elif fault == "schema":
        changed = json.loads((model_dir / "checkpoints.json").read_text(encoding="utf-8"))
        changed[0]["shape"] = [2]
        (model_dir / "checkpoints.json").write_text(json.dumps(changed), encoding="utf-8")
    elif fault == "input":
        source["cases"][0]["input_sha256"] = "f" * 64
    elif fault == "case_failed":
        source["cases"][0]["passed"] = False
    elif fault == "missing_case":
        source["cases"].pop()
    elif fault == "tolerance":
        source["atol"] = 1e-3
    (model_dir / "report.json").write_text(json.dumps(source), encoding="utf-8")
    if fault == "missing_report":
        (model_dir / "report.json").unlink()

    def should_not_find_tools(**kwargs):
        pytest.fail("Bad source evidence must fail before discovering or executing QEMU")

    monkeypatch.setattr(probe, "discover_toolchain", should_not_find_tools)
    output = tmp_path / "qemu-report"
    assert probe.main(["--model-dir", str(model_dir), "--output-dir", str(output)]) == 1
    result = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert result["passed"] is False and result["stage"] == "model-artifacts"
    assert "error" in result
    assert (output / "report.md").exists() and (output / "report.html").exists()


def execute_one(case, folder, *, graph="normal", timeout=0.5):
    return probe.execute_qemu_case(object(), {}, folder, np.zeros((1, 2, 3), np.float32),
                                   graph=graph, level="none", schema=[], names=[], reference={},
                                   case=case, timeout=timeout)


@pytest.mark.parametrize("numeric_failure", [False, True])
def test_successful_execution_records_runtime_duration_before_numeric_result(tmp_path, monkeypatch, capsys,
                                                                           numeric_failure):
    case = {"name": "sample", "qemu": []}

    def run(executable, inputs, folder, timeout):
        assert len(case["qemu"]) == 1  # Even an exception would retain the attempt.
        folder.mkdir()
        return SimpleNamespace(output=np.full((1, 2, 3), 0.1 if numeric_failure else 0, np.float32),
                               elapsed_s=1.25, command=("qemu", "--fixture"))

    monkeypatch.setattr(probe, "run_riscv_tensor", run)
    execute_one(case, tmp_path / "run")
    row = case["qemu"][0]
    assert row["qemu_process_wall_seconds"] == row["seconds"] == 1.25
    assert row["timeout_seconds"] == 0.5  # Deadline is not used as the measurement.
    assert row["status"] == ("numeric_failed" if numeric_failure else "success")
    assert row["passed"] is not numeric_failure
    assert (tmp_path / "run/qemu.npy").exists()
    assert "qemu_process_wall_seconds=1.250" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["timeout", "runtime_error", "not_started"])
def test_failed_attempt_keeps_timing_and_error_in_cli_reports(tmp_path, monkeypatch, kind):
    if kind == "timeout":
        failure = RiscVTensorTimeoutError("deadline exceeded", elapsed_s=2.75,
                                         command=("qemu",), timeout_s=0.5)
        expected_time = 2.75
    elif kind == "runtime_error":
        failure = RiscVTensorExecutionError("bad UART checksum", elapsed_s=1.5, command=("qemu",))
        expected_time = 1.5
    else:
        failure = FileNotFoundError("QEMU is missing")
        expected_time = None

    def run(*args, **kwargs):
        raise failure

    def injected_probe(model_dir, out, report, **kwargs):
        report["stage"] = "numeric"
        case = {"name": "sample", "passed": False, "qemu": []}
        report["cases"] = [case]
        execute_one(case, out / "attempt")

    monkeypatch.setattr(probe, "run_riscv_tensor", run)
    monkeypatch.setattr(probe, "run_probe", injected_probe)
    output = tmp_path / "report"
    assert probe.main(["--model-dir", str(tmp_path), "--output-dir", str(output), "--timeout", "0.5"]) == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    row = report["cases"][0]["qemu"][0]
    assert row["status"] == kind and row["passed"] is False
    assert row["qemu_process_wall_seconds"] == row["seconds"] == expected_time
    assert row["timeout_seconds"] == 0.5 and str(failure) in row["error"]
    timing = report["timing"]
    assert timing["planned_count"] == 28 and timing["attempted_count"] == 1
    assert timing["not_attempted_count"] == 27 and timing["incomplete"] is True
    assert timing["status_counts"][kind] == 1
    assert timing["qemu_process_wall_seconds_total"] == expected_time
    assert timing["successful"]["count"] == 0 and timing["successful"]["mean_seconds"] is None
    for filename in ("report.md", "report.html"):
        text = (output / filename).read_text(encoding="utf-8")
        assert kind in text and "部分结果" in text
        assert probe.format_seconds(expected_time) in text
        assert "纯前向" in text and "目标硬件性能" in text


def timed_row(graph, level, elapsed, status="success"):
    return {"graph": graph, "optimization": level, "status": status, "passed": status == "success",
            "qemu_process_wall_seconds": elapsed, "seconds": elapsed, "timeout_seconds": 180}


def test_timing_summary_separates_graphs_statuses_and_unmeasured_samples():
    rows = [timed_row("normal", "none", 1), timed_row("normal", "none", 3, "numeric_failed"),
            timed_row("normal", "all", None, "not_started"),
            timed_row("diagnostic", "none", 20), timed_row("diagnostic", "all", 9, "timeout")]
    report = {"seconds": 100, "cases": [{"qemu": rows}],
              "builds": [{"compile_seconds": 0.25}, {"compile_seconds": 0.75}]}
    timing = probe.summarize_timing(report)
    assert timing["pipeline_seconds"] == 100
    assert timing["qemu_process_wall_seconds_total"] == 33
    assert timing["cross_compile_seconds_total"] == 1
    assert timing["cross_compile_completed_count"] == 2 and timing["cross_compile_incomplete"]
    assert timing["status_counts"] == {"success": 2, "numeric_failed": 1, "runtime_error": 0,
                                       "timeout": 1, "not_started": 1}
    assert timing["not_attempted_count"] == 23 and timing["incomplete"]
    groups = {(group["graph"], group["optimization"]): group for group in timing["groups"]}
    normal = groups["normal", "none"]
    assert normal["successful"] == {"count": 1, "total_seconds": 1, "mean_seconds": 1, "max_seconds": 1}
    assert normal["failed"]["mean_seconds"] == 3  # Numeric failures do not improve success timings.
    not_started = groups["normal", "all"]
    assert not_started["attempted_count"] == 1 and not_started["failed"]["count"] == 0
    assert not_started["all_measured"]["total_seconds"] is None
    assert groups["diagnostic", "none"]["successful"]["mean_seconds"] == 20
    assert groups["diagnostic", "all"]["failed"]["mean_seconds"] == 9


def test_no_executions_produces_null_measurements_not_zero_seconds():
    timing = probe.summarize_timing({"seconds": 0.1})
    assert timing["attempted_count"] == 0 and timing["not_attempted_count"] == 28
    assert timing["incomplete"] and timing["cross_compile_incomplete"]
    assert timing["qemu_process_wall_seconds_total"] is None
    assert timing["cross_compile_seconds_total"] is None
    assert all(group["successful"]["mean_seconds"] is None for group in timing["groups"])


def test_complete_timing_matrix_renders_in_markdown_and_html_without_performance_gate(tmp_path, monkeypatch,
                                                                                   capsys):
    def injected_probe(model_dir, out, report, **kwargs):
        report.update(passed=True, stage="complete", cases=[
            {"name": f"input_{index}", "passed": True,
             "qemu": [timed_row(graph, level, 10000 + group_index)
                      for group_index, (graph, level) in enumerate(probe.QEMU_GROUPS)]}
            for index in range(7)], builds=[{"compile_seconds": 0.25}] * 4)

    monkeypatch.setattr(probe, "run_probe", injected_probe)
    output = tmp_path / "report"
    assert probe.main(["--model-dir", str(tmp_path), "--output-dir", str(output)]) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    timing = report["timing"]
    assert not timing["incomplete"] and timing["status_counts"]["success"] == 28
    assert timing["completed_count"] == 28 and timing["not_attempted_count"] == 0
    assert timing["qemu_process_wall_seconds_total"] == 7 * sum(range(10000, 10004))
    assert timing["pipeline_seconds"] == report["seconds"]
    markdown = (output / "report.md").read_text(encoding="utf-8")
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "normal/none" in markdown and "diagnostic/all" in markdown
    assert "10000.000" in markdown and "10003.000" in html
    assert "<table>" in html and html.index("仿真耗时") < html.index("<details>")
    assert "不含 ScratchV" in markdown and "不代表纯前向" in html
    assert "QEMU进程实测合计=" in capsys.readouterr().out


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_timeout_requires_positive_finite_duration(tmp_path, timeout):
    with pytest.raises(SystemExit, match="2"):
        probe.main(["--model-dir", str(tmp_path), "--output-dir", str(tmp_path / "report"),
                    "--timeout", timeout])
    assert not (tmp_path / "report").exists()
