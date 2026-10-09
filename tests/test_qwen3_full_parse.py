"""Full-model gate orchestration and failure evidence; no large checkpoint needed."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from probes.w2_qwen3_parse import run as gate


def completed_report(out, *, passed=True, stage="complete"):
    report = gate.initial_report("verify")
    report.update(passed=passed, stage=stage, validation_seconds=1.0,
                  graph={"passed": True, "layers": [{"layer": index} for index in range(28)]},
                  stages_seconds={"scratchv-parse": 0.25},
                  ir={"instructions": 12, "globals": 3}, ir_verifier={"passed": True},
                  audit={"passed": True, "binding_bytes": 16},
                  peak_memory={"bytes": 1024, "method": "test"})
    for name, content in (("nodes.json", "[]"), ("bindings.json", "[]"), ("ir.txt", "fixture IR")):
        (out / name).write_text(content, encoding="utf-8")
    gate.write_json(out / "worker-report.json", report)
    return report


def invoke(tmp_path):
    return ["--model-dir", str(tmp_path / "model"), "--output-dir", str(tmp_path / "report")]


def test_success_keeps_separate_parse_time_and_scope(tmp_path, monkeypatch):
    def launch(args):
        completed_report(args.output_dir)
        return subprocess.CompletedProcess([], 0, b"parsed", b"")
    monkeypatch.setattr(gate, "launch_worker", launch)
    assert gate.main(invoke(tmp_path)) == 0
    out = tmp_path / "report"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] and report["stage"] == "complete"
    assert report["stages_seconds"]["scratchv-parse"] == 0.25
    assert report["seconds"] >= 0 and report["validation_seconds"] == 1
    assert report["timeout_seconds"] == 300
    for name in ("report.md", "report.html"):
        content = (out / name).read_text(encoding="utf-8")
        assert "PASS" in content and "28 层" in content
        assert "不证明完整 IR 数值或 QEMU 前向通过" in content
    assert (out / "worker.stdout").read_bytes() == b"parsed"


@pytest.mark.parametrize("failure", ["nonzero", "failed", "unfinished", "missing_report",
                                    "bad_json", "wrong_gate", "missing_evidence", "spawn", "killed_missing_report"])
def test_worker_failure_cannot_be_success(tmp_path, monkeypatch, failure):
    def launch(args):
        if failure == "spawn":
            raise OSError("cannot start worker <unsafe>")
        report = completed_report(args.output_dir)
        path = args.output_dir / "worker-report.json"
        if failure == "failed":
            report.update(passed=False, stage="ir-verifier", error="invalid IR")
        elif failure == "unfinished":
            report["stage"] = "scratchv-parse"
        elif failure == "wrong_gate":
            report["gate"] = "ORT-only"
        gate.write_json(path, report)
        if failure in ("missing_report", "killed_missing_report"):
            path.unlink()
        elif failure == "bad_json":
            path.write_text("{broken", encoding="utf-8")
        elif failure == "missing_evidence":
            (args.output_dir / "bindings.json").unlink()
        code = -9 if failure == "killed_missing_report" else (4 if failure == "nonzero" else 0)
        return subprocess.CompletedProcess([], code, b"partial", b"failure")
    monkeypatch.setattr(gate, "launch_worker", launch)
    assert gate.main(invoke(tmp_path)) == 1
    out = tmp_path / "report"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False and report["error"]
    assert report["attempt"]["command"][0] == gate.sys.executable
    assert report["attempt"]["model_dir"] == str((tmp_path / "model").resolve())
    assert "probes/w2_qwen3_parse/run.py" in report["attempt"]["source_sha256"]
    if failure == "killed_missing_report":
        assert report["worker_returncode"] == -9
    assert "FAIL" in (out / "report.md").read_text(encoding="utf-8")
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "<unsafe>" not in html


def test_timeout_keeps_partial_progress_and_actual_elapsed_time(tmp_path, monkeypatch):
    clock = iter((10.0, 13.75))
    monkeypatch.setattr(gate.time, "perf_counter", lambda: next(clock))
    def launch(args):
        gate.write_json(args.output_dir / "worker-progress.json",
                        {"context": {"node_index": 17, "op": "MatMul"}, "elapsed_seconds": 2.0})
        raise subprocess.TimeoutExpired(["python"], 2, output=b"partial stdout", stderr=b"partial stderr")
    monkeypatch.setattr(gate, "launch_worker", launch)
    assert gate.main([*invoke(tmp_path), "--timeout", "2"]) == 1
    out = tmp_path / "report"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["stage"] == "timeout" and report["timeout_seconds"] == 2
    assert report["attempt"]["source_sha256"]
    assert report["seconds"] == 3.75
    assert report["last_progress"]["context"]["node_index"] == 17
    assert (out / "worker.stderr").read_bytes() == b"partial stderr"


def test_timeout_log_write_failure_still_produces_failure_report(tmp_path, monkeypatch):
    original = Path.write_bytes
    def locked_stderr(path, content):
        if path.name == "worker.stderr":
            raise PermissionError("log locked")
        return original(path, content)
    def timeout(args):
        raise subprocess.TimeoutExpired(["python"], 1, output=b"partial")
    monkeypatch.setattr(Path, "write_bytes", locked_stderr)
    monkeypatch.setattr(gate, "launch_worker", timeout)
    assert gate.main(invoke(tmp_path)) == 1
    out = tmp_path / "report"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["stage"] == "timeout" and not report["passed"]
    assert "log locked" in report["log_errors"][0]
    assert "log locked" in (out / "report.md").read_text(encoding="utf-8")
    assert (out / "worker.stdout").read_bytes() == b"partial"


@pytest.mark.parametrize("worker_passed", [False, True])
@pytest.mark.parametrize("locked", ["worker.stdout", "worker.stderr"])
def test_normal_log_failure_preserves_worker_result(tmp_path, monkeypatch, worker_passed, locked):
    original = Path.write_bytes
    def write(path, content):
        if path.name == locked:
            raise PermissionError("log locked")
        return original(path, content)
    def launch(args):
        report = completed_report(args.output_dir, passed=worker_passed,
                                  stage="complete" if worker_passed else "ir-and-bindings")
        report.update(checkout={"head": "review-sha"}, failure_context={"tensor_name": "positions"})
        if not worker_passed:
            report["error"] = "positions: bound tensor content SHA256 mismatch"
        gate.write_json(args.output_dir / "worker-report.json", report)
        return subprocess.CompletedProcess([], 0 if worker_passed else 1, b"out", b"err")
    monkeypatch.setattr(Path, "write_bytes", write)
    monkeypatch.setattr(gate, "launch_worker", launch)
    assert gate.main(invoke(tmp_path)) == 1
    out = tmp_path / "report"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["checkout"]["head"] == "review-sha"
    assert report["failure_context"] == {"tensor_name": "positions"}
    assert len(report["log_errors"]) == 1 and locked in report["log_errors"][0]
    assert report["worker_returncode"] == (0 if worker_passed else 1)
    if worker_passed:
        assert report["stage"] == "log-write" and "logs" in report["error"]
    else:
        assert report["stage"] == "ir-and-bindings" and "SHA256 mismatch" in report["error"]
        assert "SHA256 mismatch" in (out / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("worker_passed", [False, True])
@pytest.mark.parametrize("locked", ["report.md", "report.html", "report.json"])
def test_report_write_failure_cannot_publish_pass(tmp_path, monkeypatch, worker_passed, locked, capsys):
    write_text, replace = Path.write_text, Path.replace
    json_publications = []

    def write(path, content, **kwargs):
        if path.name == locked:
            raise PermissionError("injected report lock")
        return write_text(path, content, **kwargs)

    def publish(path, target):
        if Path(target).name == "report.json":
            if locked == "report.json":
                raise PermissionError("injected report lock")
            json_publications.append(json.loads(path.read_text(encoding="utf-8")))
        return replace(path, target)

    def launch(args):
        report = completed_report(args.output_dir, passed=worker_passed,
                                  stage="complete" if worker_passed else "ir-and-bindings")
        if not worker_passed:
            report["error"] = "positions: bound tensor content SHA256 mismatch"
            report["failure_context"] = {"tensor_name": "positions"}
        gate.write_json(args.output_dir / "worker-report.json", report)
        return subprocess.CompletedProcess([], 0 if worker_passed else 1, b"out", b"err")

    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(Path, "replace", publish)
    monkeypatch.setattr(gate, "launch_worker", launch)
    assert gate.main(invoke(tmp_path)) == 1
    out = tmp_path / "report"
    assert "injected report lock" in capsys.readouterr().err
    if locked == "report.json":
        assert not (out / "report.json").exists()
        assert not json_publications
    else:
        # JSON must never have been published as PASS, even temporarily.
        assert len(json_publications) == 1
        report = json_publications[0]
        assert report["passed"] is False
        assert locked in report["report_write_errors"][0]
        if worker_passed:
            assert report["stage"] == "report-write"
        else:
            assert report["stage"] == "ir-and-bindings"
            assert "SHA256 mismatch" in report["error"]
            assert report["failure_context"] == {"tensor_name": "positions"}
    for name in ("report.md", "report.html"):
        if name != locked:
            content = (out / name).read_text(encoding="utf-8")
            assert "FAIL" in content and "PASS" not in content
            assert "injected report lock" in content
            if not worker_passed:
                assert "SHA256 mismatch" in content


def test_w2_docs_links_resolve_without_local_output(monkeypatch):
    from scripts import check_docs_links
    exists = Path.exists
    def clean_checkout(path):
        if path.resolve().is_relative_to(gate.ROOT / "output"):
            return False
        return exists(path)
    monkeypatch.setattr(Path, "exists", clean_checkout)
    _, broken = check_docs_links.check_links([
        "docs/llm-deploy-v1.0/W2/README.md", "probes/w2_qwen3_parse/README.md"])
    assert not broken, broken


def test_checkout_evidence_preserves_porcelain_columns(monkeypatch, tmp_path):
    # A linked worktree stores .git as a file, not a directory.
    (tmp_path / ".git").write_text("gitdir: unused-test-path\n", encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    def run(command, **kwargs):
        assert kwargs["timeout"] == 10 and kwargs["cwd"] == tmp_path
        return SimpleNamespace(stdout="abcdef\n" if command[1] == "rev-parse" else " M file.py\n")
    monkeypatch.setattr(gate.subprocess, "run", run)
    report = gate.checkout_evidence()
    assert report["head"] == "abcdef" and report["status"] == " M file.py"
    assert report["clean"] is False


def test_source_archive_does_not_query_parent_checkout(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    archive = tmp_path / "source-archive"
    archive.mkdir()
    monkeypatch.setattr(gate, "ROOT", archive)
    monkeypatch.setattr(gate.subprocess, "run", lambda *a, **kw: pytest.fail("must not search parent Git"))
    report = gate.checkout_evidence()
    assert report["head"] is None and report["status"] == "unknown" and report["clean"] is None
    assert report["reason"] == "Source directory has no .git metadata"


@pytest.mark.parametrize("failure_at", ["rev-parse", "status"])
def test_git_timeout_is_unknown_and_does_not_leave_partial_identity(monkeypatch, tmp_path, failure_at):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    def query(command, **kwargs):
        assert kwargs["timeout"] == 10
        if command[1] == failure_at:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(stdout="partial-sha\n")
    monkeypatch.setattr(gate.subprocess, "run", query)
    report = gate.checkout_evidence()
    assert report["head"] is None and report["status"] == "unknown" and report["clean"] is None
    assert report["reason"] == "Checkout metadata unavailable"
    assert "TimeoutExpired" in report["error"]


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_invalid_timeout_does_not_start_worker(tmp_path, monkeypatch, timeout):
    monkeypatch.setattr(gate, "launch_worker", lambda args: pytest.fail("must not launch"))
    with pytest.raises(SystemExit, match="2"):
        gate.main([*invoke(tmp_path), "--timeout", timeout])
    assert not (tmp_path / "report").exists()


def test_existing_evidence_is_preserved(tmp_path):
    out = tmp_path / "report"
    out.mkdir()
    evidence = out / "report.json"
    evidence.write_text("prior evidence", encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        gate.main(invoke(tmp_path))
    assert evidence.read_text(encoding="utf-8") == "prior evidence"


def test_worker_uses_same_python_explicit_paths_and_timeout(tmp_path, monkeypatch):
    import scratchv.runtime.riscv_tensor as runtime
    captured = {}
    def process(command, **kwargs):
        captured.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, b"", b"")
    monkeypatch.setattr(runtime, "_run_process", process)
    args = SimpleNamespace(mode="verify", model_dir=tmp_path / "model with spaces",
                           output_dir=tmp_path / "output with spaces", timeout=12.5)
    gate.launch_worker(args)
    assert captured["command"][0] == gate.sys.executable
    assert str(args.model_dir.resolve()) in captured["command"]
    assert "--worker" in captured["command"] and captured["timeout"] == 12.5


def test_worker_dependency_failure_writes_report_before_parsing(tmp_path, monkeypatch):
    def bad_environment():
        raise RuntimeError("pinned environment mismatch")
    monkeypatch.setattr(gate.artifacts, "environment", bad_environment)
    monkeypatch.setattr(gate, "run_validation", lambda *args, **kwargs: pytest.fail("must not parse"))
    args = SimpleNamespace(mode="verify", model_dir=tmp_path / "model", output_dir=tmp_path)
    assert gate.worker_main(args) == 1
    result = json.loads((tmp_path / "worker-report.json").read_text(encoding="utf-8"))
    assert result["stage"] == "environment" and not result["passed"]
    assert "mismatch" in result["error"] and result["validation_seconds"] >= 0


def test_corrupt_hash_cannot_reach_scratchv_parser(tmp_path, monkeypatch):
    from probes.w2_qwen3_parse import audit
    def bad_files(*args):
        raise ValueError("Artifact size/SHA-256 mismatch: weights.data")
    monkeypatch.setattr(gate.artifacts, "verify_files", bad_files)
    monkeypatch.setattr(audit.AuditedONNXParser, "parse", lambda *args: pytest.fail("must not parse"))
    report = gate.initial_report("verify")
    with pytest.raises(ValueError, match="SHA-256"):
        gate.run_validation(tmp_path / "model", tmp_path, report)
    assert report["stage"] == "hashes" and not report["passed"]
    assert "hashes" in report["stages_seconds"]


@pytest.mark.parametrize("fault,evidence_locked", [
    (None, False), ("parse", False), ("verifier", False), ("binding", False),
    ("parse", True), ("binding", True), (None, True),
])
def test_real_decoder_pipeline_and_failure_context(tmp_path, monkeypatch, fault, evidence_locked):
    """Exercise the stage wiring with real files, parser and audits, cheaply."""
    import onnx
    from scratchv.analysis import ir_verifier
    from probes.w2_qwen3_parse import audit, validation
    from tests.test_qwen3_full_structure import qwen3_fixture

    model, config = qwen3_fixture()
    model_dir, out = tmp_path / "model", tmp_path / "out"
    model_dir.mkdir()
    out.mkdir()
    onnx.save_model(model, model_dir / "model.onnx", save_as_external_data=True,
                    all_tensors_to_one_file=True, location="weights.data", size_threshold=0)
    manifest = {"files": [{"name": path.name, "bytes": path.stat().st_size,
                           "sha256": gate.artifacts.sha256(path)} for path in model_dir.iterdir()]}
    # Only the full-model dimensions are adapted; loading, file integrity,
    # lowering, verifier and content audits still run against real fixtures.
    structure = validation.audit_graph_structure
    monkeypatch.setattr(validation, "audit_graph_structure", lambda graph: structure(graph, config))
    def inspect(directory, files):
        onnx.checker.check_model(str(directory / "model.onnx"), full_check=True)
        return {"operators": {"fixture": len(model.graph.node)}}
    monkeypatch.setattr(gate.artifacts, "inspect_model", inspect)
    if fault == "parse":
        def broken_gather(*args):
            raise RuntimeError("injected Gather failure")
        monkeypatch.setattr(audit.AuditedONNXParser, "_handle_gather", broken_gather)
    elif fault == "verifier":
        monkeypatch.setattr(ir_verifier, "verify_ir", lambda *args, **kwargs: (False, ["injected verifier failure"]))
    elif fault == "binding":
        original = audit.AuditedONNXParser.parse
        def misbind(self, path):
            program = original(self, path)
            self.initializers["model.lm_head.weight"] = self.initializers["model.lm_head.weight"].copy()
            self.initializers["model.lm_head.weight"][0, 0] += 1
            return program
        monkeypatch.setattr(audit.AuditedONNXParser, "parse", misbind)
    report = gate.initial_report("verify")
    if evidence_locked:
        original_write = gate.write_json
        def write(path, value):
            # The binding failure occurs after the first, successful nodes.json
            # save; lock only its diagnostic retry, not the earlier operation.
            locked = ((fault == "parse" and path.name == "nodes.partial.json")
                      or (fault == "binding" and path.name == "nodes.json"
                          and "failure_context" in report)
                      or (fault is None and path.name == "bindings.json"))
            if locked:
                raise PermissionError("injected evidence lock")
            return original_write(path, value)
        monkeypatch.setattr(gate, "write_json", write)
    if fault or evidence_locked:
        expected_error = ("injected Gather failure" if fault == "parse" else
                          "content SHA256 mismatch" if fault == "binding" else
                          "injected evidence lock" if evidence_locked else "IR verifier rejected")
        with pytest.raises((RuntimeError, ValueError, PermissionError), match=expected_error) as failure:
            gate.run_validation(model_dir, out, report, manifest=manifest)
        assert report["passed"] is False
        assert report["stage"] == {"parse": "scratchv-parse", "verifier": "ir-verifier",
                                   "binding": "ir-and-bindings", None: "ir-and-bindings"}[fault]
        if fault == "parse":
            assert report["failure_context"]["op_type"] == "Gather"
            if evidence_locked:
                assert not (out / "nodes.partial.json").exists()
                assert report["nodes_completed"] == 0
            else:
                partial = json.loads((out / "nodes.partial.json").read_text(encoding="utf-8"))
                assert partial[-1]["status"] == "failed"
                assert report["nodes_completed"] == len(partial) - 1
        elif fault == "verifier":
            assert report["ir_verifier"]["issues"] == ["injected verifier failure"]
            assert (out / "nodes.json").is_file()
        elif fault == "binding":
            assert "model.lm_head.weight" in json.dumps(report["failure_context"])
            assert (out / "nodes.json").is_file()
        if evidence_locked and fault:
            assert len(report["evidence_errors"]) == 1
            assert "injected evidence lock" in report["evidence_errors"][0]
            report["error"] = f"{type(failure.value).__name__}: {failure.value}"
            gate.write_reports(out, report)
            markdown = (out / "report.md").read_text(encoding="utf-8")
            assert expected_error in markdown and "injected evidence lock" in markdown
        elif evidence_locked:
            # Successful validation cannot pass if required evidence is lost.
            assert isinstance(failure.value, PermissionError)
            assert "evidence_errors" not in report
        assert not (out / "bindings.json").exists()
    else:
        gate.run_validation(model_dir, out, report, manifest=manifest)
        assert report["passed"] and report["stage"] == "complete"
        assert report["graph"]["layer_count"] == 2
        assert report["audit"]["node_count"] == len(model.graph.node)
        assert report["audit"]["return"]["shape"] == [1, 3, 9]
        for name, evidence in report["evidence"].items():
            assert evidence["sha256"] == gate.artifacts.sha256(out / name)
