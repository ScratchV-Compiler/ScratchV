"""Full-gate evidence, mathematical scope, and resource review regressions."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from probes.w3_qwen3_full import resources
from probes.w3_qwen3_full import run as gate
from probes.w3_qwen3_full import worker


@pytest.mark.parametrize("status", ["Name:\tpython\nState:\tZ (zombie)\n",
                                    "VmRSS:\tinvalid kB\n", "VmRSS:\t12 bytes\n",
                                    "VmRSS:\t-1 kB\n", "VmRSS:\t\n"])
def test_linux_unavailable_or_malformed_rss_uses_observation_error(monkeypatch, status):
    monkeypatch.setattr(resources, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(resources, "Path", lambda _: SimpleNamespace(read_text=lambda **kwargs: status))
    with pytest.raises(OSError, match="Cannot observe worker"):
        resources.process_memory(42)


def test_linux_resident_unit_is_converted_to_bytes(monkeypatch):
    monkeypatch.setattr(resources, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(resources, "Path", lambda _: SimpleNamespace(read_text=lambda **kwargs: "VmRSS:\t123 kB\n"))
    assert resources.process_memory(42) == {"rss_bytes": 123 * 1024, "private_commit_bytes": None}


def test_successful_exit_during_memory_sample_is_not_failure(monkeypatch):
    class Process:
        pid = 42
        exited = False

        def poll(self):
            return 0 if self.exited else None

        def wait(self, timeout):
            assert self.exited
            return 0

    process = Process()

    def sample(pid):
        process.exited = True
        raise OSError("No VmRSS in zombie status")

    monkeypatch.setattr(resources, "process_memory", sample)
    row = {}
    assert resources.wait_bounded(process, timeout=2, max_memory_bytes=1024, row=row) == 0
    assert row["returncode"] == 0 and row["resource_monitor"]["samples"] == 0


def _worker_evidence(folder, backend, input_path, sources, assets):
    """Small logits but real hashed arrays/schema, to test supervisor contracts."""
    folder.mkdir(parents=True)
    schema = [{"name": name, "shape": [1, 256, 1024], "dtype": "float32", "sequence_axis": 1}
              for name in gate.CHECKPOINTS]
    zeros = np.zeros((1, 256, 1024), np.float32)
    np.savez_compressed(folder / "checkpoints.npz", **{name: zeros for name in gate.CHECKPOINTS})
    for filename in ("logits.npy", "diagnostic_logits.npy"):
        np.save(folder / filename, np.zeros((1, 2, 3), np.float32))
    (folder / "checkpoint_schema.json").write_text(json.dumps({"version": 1, "checkpoints": schema}))
    artifacts = {name: worker.evidence(folder / filename) for name, filename in {
        "logits": "logits.npy", "diagnostic_logits": "diagnostic_logits.npy",
        "checkpoints": "checkpoints.npz", "checkpoint_schema": "checkpoint_schema.json"}.items()}
    report = {"gate": worker.GATE, "schema_version": 1, "backend": backend,
              "optimization_level": "none", "stage": "complete", "passed": True, "status": "PASS",
              "full_model_executed": True, "full_ir_executed": backend == "ir", "w3_exit_accepted": False,
              "source_sha256": sources, "files": assets, "input": worker.evidence(input_path),
              "process_peak_rss_bytes": 4096,
              "stages_seconds": {f"{backend}_{mode}_execute": 0.1 for mode in ("ordinary", "diagnostic")},
              "ordinary_diagnostic": {"passed": True, "comparison": "exact", "max_abs": 0.0},
              "checkpoints": schema, "artifacts": artifacts,
              "executions": {mode: {"executed_steps": 31, "memory_stats": {"peak_numpy_storage_bytes": 4096}}
                             for mode in ("ordinary", "diagnostic")}}
    if backend == "ir":
        report["ir"] = {"instruction_count": 31}
    else:
        report["ort"] = {"provider": "CPUExecutionProvider", "graph_optimization": "disabled",
                         "intra_op_threads": 1, "inter_op_threads": 1, "cpu_mem_arena": False}
    for filename in ("report.md", "report.html"):
        (folder / filename).write_text("PASS")
    (folder / "report.json").write_text(json.dumps(report))
    return report


@pytest.mark.parametrize("backend", ["ort", "ir"])
def test_worker_validation_consumes_real_hashed_arrays(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(gate, "LOGIT_SHAPE", (1, 2, 3))
    feed = tmp_path / "inputs.npz"
    feed.write_bytes(b"fixture input")
    sources, assets = {"source.py": "sha"}, [{"name": "fixture.onnx", "sha256": "asset"}]
    folder = tmp_path / backend
    report = _worker_evidence(folder, backend, feed, sources, assets)
    assert gate.validate_worker(folder, backend, feed, sources, assets) == report


@pytest.fixture
def worker_record(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "LOGIT_SHAPE", (1, 2, 3))
    feed = tmp_path / "inputs.npz"
    feed.write_bytes(b"fixture input")
    sources, assets = {"source.py": "sha"}, [{"name": "fixture.onnx", "sha256": "asset"}]

    def create(backend):
        folder = tmp_path / backend
        report = _worker_evidence(folder, backend, feed, sources, assets)

        def validate():
            (folder / "report.json").write_text(json.dumps(report))
            return gate.validate_worker(folder, backend, feed, sources, assets)

        return report, validate

    return create


@pytest.mark.parametrize("backend", ["ort", "ir"])
@pytest.mark.parametrize("change", ["missing", "null", "ordinary_missing", "diagnostic_missing",
                                   "zero", "negative", "nan", "infinity", "string", "boolean"])
def test_worker_requires_measured_time_for_both_executions(worker_record, backend, change):
    report, validate = worker_record(backend)
    if change == "missing":
        report.pop("stages_seconds")
    elif change == "null":
        report["stages_seconds"] = None
    elif change.endswith("_missing"):
        report["stages_seconds"].pop(f"{backend}_{change.removesuffix('_missing')}_execute")
    else:
        report["stages_seconds"][f"{backend}_diagnostic_execute"] = {
            "zero": 0, "negative": -0.1, "nan": float("nan"), "infinity": float("inf"),
            "string": "0.1", "boolean": True}[change]
    with pytest.raises(ValueError, match="execution timing"):
        validate()


@pytest.mark.parametrize("field,value", [(None, None), ("provider", None),
    ("provider", "CUDAExecutionProvider"), ("graph_optimization", "enabled"),
    ("intra_op_threads", 2), ("inter_op_threads", 2), ("intra_op_threads", True),
    ("inter_op_threads", True), ("cpu_mem_arena", True), ("cpu_mem_arena", 0)])
def test_worker_requires_declared_ort_execution_profile(worker_record, field, value):
    report, validate = worker_record("ort")
    if field is None:
        report.pop("ort")
    else:
        report["ort"][field] = value
    with pytest.raises(ValueError, match="ORT execution profile"):
        validate()


@pytest.mark.parametrize("count,ordinary,diagnostic", [
    (None, 31, 31), (0, 31, 31), (-1, 31, 31), (True, 31, 31), (31.0, 31, 31),
    (31, 1, 31), (31, 31, 1), (31, 31, 32), (31, 31, True), (31, 31, 31.0),
])
def test_worker_requires_all_ir_instructions_in_both_executions(worker_record, count,
                                                              ordinary, diagnostic):
    report, validate = worker_record("ir")
    if count is None:
        report["ir"].pop("instruction_count")
    else:
        report["ir"]["instruction_count"] = count
    report["executions"]["ordinary"]["executed_steps"] = ordinary
    report["executions"]["diagnostic"]["executed_steps"] = diagnostic
    with pytest.raises(ValueError, match="instruction count"):
        validate()


def test_worker_arithmetic_profile_cannot_be_silently_substituted(tmp_path, monkeypatch):
    from scratchv.verification.fp32_reference import profile
    monkeypatch.setattr(gate, "LOGIT_SHAPE", (1, 2, 3))
    feed = tmp_path / "inputs.npz"
    feed.write_bytes(b"fixture input")
    sources, assets = {"source.py": "sha"}, [{"name": "fixture.onnx"}]
    folder = tmp_path / "ir"
    report = _worker_evidence(folder, "ir", feed, sources, assets)
    with pytest.raises(ValueError, match="FP32"):
        gate.validate_worker(folder, "ir", feed, sources, assets, expected_fp32_mode="reference")
    report.update(fp32_mode="reference")
    report["ir"].update(fp32_mode="reference", fp32_profile=profile())
    (folder / "report.json").write_text(json.dumps(report))
    gate.validate_worker(folder, "ir", feed, sources, assets, expected_fp32_mode="reference")
    with pytest.raises(ValueError, match="FP32"):
        gate.validate_worker(folder, "ir", feed, sources, assets, expected_fp32_mode="native")
    report["ir"]["fp32_profile"]["matmul_k_block"] = 64
    (folder / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="profile"):
        gate.validate_worker(folder, "ir", feed, sources, assets, expected_fp32_mode="reference")


def test_live_worker_rejects_other_cpu_profile_but_explicit_saved_audit_accepts_it(tmp_path, monkeypatch):
    from scratchv.verification.fp32_reference import profile
    monkeypatch.setattr(gate, "LOGIT_SHAPE", (1, 2, 3))
    monkeypatch.setenv("SCRATCHV_FP32_REFERENCE_CPU", "avx512")
    feed = tmp_path / "inputs.npz"
    feed.write_bytes(b"fixture input")
    sources, assets = {"source.py": "sha"}, [{"name": "fixture.onnx"}]
    folder = tmp_path / "ir"
    report = _worker_evidence(folder, "ir", feed, sources, assets)
    saved = profile(cpu_strategy="avx2-fma3")
    report["fp32_mode"] = "reference"
    report["ir"].update(fp32_mode="reference", fp32_profile=saved)
    (folder / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="profile"):
        gate.validate_worker(folder, "ir", feed, sources, assets, expected_fp32_mode="reference")
    assert gate.validate_worker(folder, "ir", feed, sources, assets,
                                expected_fp32_mode="reference", expected_fp32_profile=saved) == report


@pytest.mark.parametrize("change", ["source", "assets", "input", "backend", "optimization",
                                   "full_execution", "team_acceptance", "missing_artifact",
                                   "altered_array", "diagnostic_changed", "missing_view",
                                   "missing_checkpoint", "ir_steps", "ir_memory"])
def test_worker_validation_rejects_untrustworthy_evidence(tmp_path, monkeypatch, change):
    monkeypatch.setattr(gate, "LOGIT_SHAPE", (1, 2, 3))
    feed = tmp_path / "inputs.npz"
    feed.write_bytes(b"fixture input")
    sources, assets = {"source.py": "sha"}, [{"name": "fixture.onnx", "sha256": "asset"}]
    folder = tmp_path / "ir"
    report = _worker_evidence(folder, "ir", feed, sources, assets)
    if change == "source":
        report["source_sha256"] = {"source.py": "wrong"}
    elif change == "assets":
        report["files"] = []
    elif change == "input":
        report["input"]["sha256"] = "wrong"
    elif change == "backend":
        report["backend"] = "ort"
    elif change == "optimization":
        report["optimization_level"] = "all"
    elif change == "full_execution":
        report["full_ir_executed"] = False
    elif change == "team_acceptance":
        report["w3_exit_accepted"] = True
    elif change == "missing_artifact":
        (folder / "checkpoints.npz").unlink()
    elif change == "altered_array":
        with (folder / "logits.npy").open("ab") as stream:
            stream.write(b"changed")
    elif change == "diagnostic_changed":
        np.save(folder / "diagnostic_logits.npy", np.full((1, 2, 3), 1e-7, np.float32))
        # Even internally consistent replacement hashes cannot conceal a
        # changed diagnostic output behind the report's exact=true claim.
        report["artifacts"]["diagnostic_logits"] = worker.evidence(folder / "diagnostic_logits.npy")
    elif change == "missing_view":
        (folder / "report.html").unlink()
    elif change == "missing_checkpoint":
        report["checkpoints"].pop()
        (folder / "checkpoint_schema.json").write_text(json.dumps({"version": 1, "checkpoints": report["checkpoints"]}))
        report["artifacts"]["checkpoint_schema"] = worker.evidence(folder / "checkpoint_schema.json")
    elif change == "ir_steps":
        report["executions"]["diagnostic"]["executed_steps"] = 0
    elif change == "ir_memory":
        report["executions"]["ordinary"]["memory_stats"] = {}
    (folder / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        gate.validate_worker(folder, "ir", feed, sources, assets)


@pytest.mark.parametrize("diagnostic_only", [False, True])
def test_formal_gate_checks_padding_in_both_logits_outputs(tmp_path, monkeypatch, diagnostic_only):
    monkeypatch.setattr(gate, "LOGIT_SHAPE", (1, 2, 3))
    feed = tmp_path / "inputs.npz"
    feed.write_bytes(b"input")
    for backend in ("ort", "ir"):
        _worker_evidence(tmp_path / backend, backend, feed, {}, [])
    filenames = ["diagnostic_logits.npy"] if diagnostic_only else ["logits.npy", "diagnostic_logits.npy"]
    for filename in filenames:
        logits = np.zeros((1, 2, 3), np.float32)
        logits[0, 1, 2] = 2e-4
        np.save(tmp_path / "ir" / filename, logits)
    comparison = gate.compare_case(tmp_path, valid_length=1)
    assert not comparison["passed"]
    assert all(row["valid_queries"]["passed"] for row in comparison["logits"])
    assert any(not row["padding_queries"]["passed"] for row in comparison["logits"])


def test_large_hidden_ulp_is_diagnostic_without_relaxing_logit_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "LOGIT_SHAPE", (1, 2, 3))
    feed = tmp_path / "inputs.npz"
    feed.write_bytes(b"input")
    for backend in ("ort", "ir"):
        _worker_evidence(tmp_path / backend, backend, feed, {}, [])
    arrays = {name: np.full((1, 256, 1024), np.float32(8192), np.float32) for name in gate.CHECKPOINTS}
    np.savez_compressed(tmp_path / "ort/checkpoints.npz", **arrays)
    arrays["layer_0.output"][0, 0, 0] = np.nextafter(np.float32(8192), np.float32(np.inf))
    np.savez_compressed(tmp_path / "ir/checkpoints.npz", **arrays)
    result = gate.compare_case(tmp_path, valid_length=1)
    assert result["passed"] and result["max_logits_abs"] == 0
    assert result["first_divergence"]["name"] == "layer_0.output"
    assert result["first_divergence"]["max_abs"] > 1e-4
    assert "Diagnostic location only" in result["checkpoint_threshold_scope"]


@pytest.mark.parametrize("numerical_passed", [True, False])
def test_selected_case_can_only_be_partial_or_fail(tmp_path, monkeypatch, numerical_passed):
    monkeypatch.setattr(gate, "source_evidence", lambda: {"source_sha256": {"test": "same"}})
    monkeypatch.setattr(gate, "verify_files", lambda _: [])
    monkeypatch.setattr(gate, "input_cases", lambda: [("short_17", 17, {"input_ids": np.zeros((1, 256), np.int64)})])

    def executed(args, folder, backend, input_path, row):
        (folder / backend).mkdir()
        (folder / backend / "report.json").write_text("{}")
        row.update(passed=True)

    monkeypatch.setattr(gate, "run_worker", executed)
    monkeypatch.setattr(gate, "validate_worker", lambda *args, **kwargs: {"input": {"valid_length": 17},
                        "process_peak_rss_bytes": 4096, "stages_seconds": {"run": 1}})
    monkeypatch.setattr(gate, "compare_case", lambda *args: {"passed": numerical_passed,
                        "max_logits_abs": 0 if numerical_passed else 2e-4, "worst": {"max_abs": 0.01}})
    out = tmp_path / "report"
    code = gate.main(["--model-dir", str(tmp_path), "--output-dir", str(out), "--case", "short_17"])
    report = json.loads((out / "report.json").read_text())
    assert code == (2 if numerical_passed else 1)
    assert report["status"] == ("PARTIAL" if numerical_passed else "FAIL")
    assert report["passed"] is False and report["coverage_complete"] is False
    assert report["w3_exit_accepted"] is False
