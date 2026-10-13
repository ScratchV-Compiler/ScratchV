"""Exercise the real export and prove corrupt IR cannot produce a green gate.

The LLM workflow and full compiler CI run this file in the pinned CPU stack.
Minimal local compiler installations may omit torch/transformers and skip it.
"""

import json
from pathlib import Path
import subprocess

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from probes.w2_qwen3_small import run as probe
from scratchv.frontend.onnx_parser import ONNXParser
from scratchv.verification.ir_interpreter import IRInterpreter


def read_report(path):
    return json.loads((path / "report.json").read_text(encoding="utf-8"))


def test_official_two_layer_export_passes_all_checkpoints_and_invariants(tmp_path):
    assert probe.main(["--output-dir", str(tmp_path)]) == 0
    report = read_report(tmp_path)
    assert report["passed"] and report["stage"] == "complete"
    assert len(report["checkpoints"]) == 29
    assert len(report["cases"]) == 7
    assert len(report["invariants"]) == 6
    for case in report["cases"]:
        assert case["passed"]
        assert all(result["passed"] for result in case["ordinary_logits"].values())
        assert all(result["passed"] for result in case["attention_checks"])
    assert all(check["passed"] for check in report["invariants"])
    assert (tmp_path / "model.onnx").exists()
    assert (tmp_path / "diagnostics.onnx").exists()
    from probes.w2_qwen3_small.riscv import validated_model_artifacts

    _, schema, evidence = validated_model_artifacts(tmp_path)
    assert schema == report["checkpoints"]
    assert evidence["model_sha256"]["normal"] == report["onnx"]["model_sha256"]


@pytest.mark.parametrize("target", ["ordinary", "diagnostic"])
def test_corrupted_ir_fails_even_when_other_execution_path_passes(tmp_path, monkeypatch, target):
    original = IRInterpreter.run

    def corrupted(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        ordinary = result.return_value.ndim == 3
        if ordinary == (target == "ordinary"):
            result.return_value.flat[0] += 0.25
        return result

    monkeypatch.setattr(IRInterpreter, "run", corrupted)
    assert probe.main(["--output-dir", str(tmp_path)]) == 1
    report = read_report(tmp_path)
    assert not report["passed"]
    assert report["stage"] == "complete" and "error" not in report
    first = report["cases"][0]
    assert first["pytorch_vs_ort"]["passed"]
    assert first["ort_pack_layout"]["passed"]
    if target == "diagnostic":
        assert first["ir_vs_ort"]["first_divergence"] == "token_embedding"
        divergence = first["ir_vs_ort"]["checkpoints"][0]
        assert divergence["worst_index"] == [0, 0, 0]
        assert divergence["max_abs"] > 0.2
        assert first["ordinary_logits"]["ir_vs_ort"]["passed"]
    else:
        assert first["ir_vs_ort"]["passed"]
        assert not first["ordinary_logits"]["ir_vs_ort"]["passed"]


@pytest.mark.parametrize("stage", ["parse", "execute"])
def test_parser_and_interpreter_exceptions_fail_and_preserve_evidence(tmp_path, monkeypatch, stage):
    def broken(*args, **kwargs):
        raise RuntimeError(f"injected {stage} failure")

    monkeypatch.setattr(ONNXParser if stage == "parse" else IRInterpreter,
                        "parse" if stage == "parse" else "run", broken)
    assert probe.main(["--output-dir", str(tmp_path)]) == 1
    report = read_report(tmp_path)
    assert not report["passed"]
    assert report["error"] == f"RuntimeError: injected {stage} failure"
    assert report["stage"] == ("parse" if stage == "parse" else "numeric")
    if stage == "execute":
        assert report["current_case"] == "full_seed_0"
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "model.onnx").exists()


def _stub_numeric_probe(monkeypatch, *, numeric_passed):
    def run(out, report, model_seed):
        report.update(passed=numeric_passed, stage="complete" if numeric_passed else "numeric",
                      cases=[], invariants=[])
        if not numeric_passed:
            report["error"] = "RuntimeError: original numeric failure"
    monkeypatch.setattr(probe, "run_probe", run)


@pytest.mark.parametrize("name", ["report.md", "report.html", "report.json"])
@pytest.mark.parametrize("after_write", [False, True])
@pytest.mark.parametrize("numeric_passed", [False, True])
def test_required_report_write_failure_never_leaves_pass(
        tmp_path, monkeypatch, name, after_write, numeric_passed):
    _stub_numeric_probe(monkeypatch, numeric_passed=numeric_passed)
    original = Path.write_text

    def broken(path, content, *args, **kwargs):
        if path.name == f".{name}.tmp":
            if after_write:
                original(path, content, *args, **kwargs)
            raise OSError("injected report close failure" if after_write else "injected report write failure")
        return original(path, content, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", broken)
    assert probe.main(["--output-dir", str(tmp_path)]) == 1
    assert not (tmp_path / f".{name}.tmp").exists()
    if name == "report.json":
        assert not (tmp_path / "report.json").exists()
        assert "FAIL" in (tmp_path / "report.md").read_text(encoding="utf-8")
    else:
        report = read_report(tmp_path)
        assert not report["passed"]
        assert any(name in item for item in report["report_write_errors"])
        if numeric_passed:
            assert report["stage"] == "report-write"
        else:
            assert report["stage"] == "numeric"
            assert report["error"] == "RuntimeError: original numeric failure"


@pytest.mark.parametrize("name", ["report.md", "report.html", "report.json"])
@pytest.mark.parametrize("numeric_passed", [False, True])
def test_required_report_publish_failure_fails_gate(tmp_path, monkeypatch, name, numeric_passed):
    _stub_numeric_probe(monkeypatch, numeric_passed=numeric_passed)
    original = Path.replace

    def broken(path, target):
        if Path(target).name == name:
            raise OSError("injected atomic publish failure")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", broken)
    assert probe.main(["--output-dir", str(tmp_path)]) == 1
    if name == "report.json":
        assert not (tmp_path / "report.json").exists()
    else:
        report = read_report(tmp_path)
        assert not report["passed"] and report["report_write_errors"]
        if not numeric_passed:
            assert report["error"] == "RuntimeError: original numeric failure"
    assert not list(tmp_path.glob(".report.*.tmp"))


def test_json_retry_after_transient_publication_error_contains_only_failure(tmp_path, monkeypatch):
    _stub_numeric_probe(monkeypatch, numeric_passed=True)
    original = Path.replace
    attempts = []

    def broken_once(path, target):
        if Path(target).name == "report.json":
            attempts.append(json.loads(path.read_text(encoding="utf-8"))["passed"])
            if len(attempts) == 1:
                raise OSError("transient publish failure")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", broken_once)
    assert probe.main(["--output-dir", str(tmp_path)]) == 1
    assert attempts == [True, False]
    assert not read_report(tmp_path)["passed"]
    assert "FAIL" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "FAIL" in (tmp_path / "report.html").read_text(encoding="utf-8")


def test_json_is_published_only_after_required_views(tmp_path, monkeypatch):
    _stub_numeric_probe(monkeypatch, numeric_passed=True)
    original = Path.replace
    publications = []

    def observe(path, target):
        publications.append(Path(target).name)
        if Path(target).name == "report.json":
            assert (tmp_path / "report.md").is_file()
            assert (tmp_path / "report.html").is_file()
            assert not (tmp_path / "report.json").exists()
        return original(path, target)

    monkeypatch.setattr(Path, "replace", observe)
    assert probe.main(["--output-dir", str(tmp_path)]) == 0
    assert publications == ["report.md", "report.html", "report.json"]
    assert read_report(tmp_path)["passed"]


@pytest.mark.parametrize("error", [FileNotFoundError("git absent"),
                                  subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"]),
                                  subprocess.TimeoutExpired(["git", "rev-parse", "HEAD"], 10)])
def test_source_archive_without_git_preserves_hash_identity(monkeypatch, error):
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: True if path == probe.ROOT / ".git" else exists(path))
    def missing_git(*args, **kwargs):
        assert kwargs["timeout"] == 10
        raise error
    monkeypatch.setattr(probe.subprocess, "check_output", missing_git)
    result = probe.provenance()
    assert result["git_commit"] is None
    assert result["git_dirty"] is None and type(error).__name__ in result["git_reason"]
    assert result["source_sha256"]["probes/w2_qwen3_small/run.py"]
    assert result["source_sha256"]["scratchv/verification/ir_interpreter.py"]


def test_source_archive_never_queries_a_parent_checkout(monkeypatch):
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: False if path == probe.ROOT / ".git" else exists(path))
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *a, **kw: pytest.fail("must not search parent Git"))
    result = probe.provenance()
    assert result["git_commit"] is None and result["git_dirty"] is None
    assert result["git_reason"] == "Source directory has no .git metadata"
    assert result["source_sha256"]["probes/w2_qwen3_small/run.py"]


def test_checkout_metadata_commands_are_bounded_and_preserve_dirty_state(monkeypatch):
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: True if path == probe.ROOT / ".git" else exists(path))
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == probe.ROOT and kwargs["timeout"] == 10
        return "review-sha\n" if command[1] == "rev-parse" else " M file.py\n"
    monkeypatch.setattr(probe.subprocess, "check_output", query)
    result = probe.provenance()
    assert result["git_commit"] == "review-sha" and result["git_dirty"] is True
    assert len(calls) == 2
