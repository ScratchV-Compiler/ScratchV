"""Report failures cannot leave PASS evidence or hide a primary probe failure."""

import json
from pathlib import Path

import pytest

from probes.w2_runtime.run import save_report


@pytest.mark.parametrize("filename", ["report.json", "report.md", "report.html"])
def test_each_report_write_failure_invalidates_success_and_saves_other_formats(tmp_path, monkeypatch, capsys, filename):
    original = Path.write_text

    def fail_one(path, data, *args, **kwargs):
        target = ".report.json.tmp" if filename == "report.json" else filename
        if path.name == target:
            raise PermissionError("simulated blocked destination")
        return original(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_one)
    report = {"passed": True, "stage": "complete", "seconds": 0.1, "tokenizer_cases": [], "runtime_cases": []}
    save_report(tmp_path, report)
    assert report["passed"] is False
    assert report["stage"] == "report-write"
    assert "PermissionError" in report["report_write_errors"][0]
    for name in ("report.json", "report.md", "report.html"):
        if name != filename:
            text = (tmp_path / name).read_text(encoding="utf-8")
            if name == "report.json":
                assert json.loads(text)["passed"] is False
            else:
                assert "FAIL" in text and "PASS" not in text
    assert filename in capsys.readouterr().err


def test_report_failure_preserves_primary_error_and_stage(tmp_path, monkeypatch, capsys):
    original = Path.write_text

    def fail_markdown(path, data, *args, **kwargs):
        if path.suffix == ".md":
            raise OSError("disk full")
        return original(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_markdown)
    report = {
        "passed": False, "stage": "assets", "seconds": 0.1,
        "error": "ValueError: tokenizer SHA256 mismatch",
        "tokenizer_cases": [], "runtime_cases": [],
    }
    save_report(tmp_path, report)
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["stage"] == "assets"
    assert saved["error"] == "ValueError: tokenizer SHA256 mismatch"
    assert "disk full" in saved["report_write_errors"][0]
    assert "SHA256 mismatch" in capsys.readouterr().err


def test_json_is_first_written_as_fail_after_view_failure_without_recovery_write(tmp_path, monkeypatch):
    original = Path.write_text
    json_attempts = []

    def fail_view_and_any_second_json_write(path, data, *args, **kwargs):
        if path.name == "report.md":
            raise PermissionError("markdown blocked")
        if path.name == ".report.json.tmp":
            json_attempts.append(json.loads(data))
            if len(json_attempts) > 1:
                raise PermissionError("corrective JSON rewrite blocked")
        return original(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_view_and_any_second_json_write)
    report = {"passed": True, "stage": "complete", "seconds": 0.1, "tokenizer_cases": [], "runtime_cases": []}
    save_report(tmp_path, report)
    assert len(json_attempts) == 1
    assert json_attempts[0]["passed"] is False
    assert json_attempts[0]["stage"] == "report-write"
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["passed"] is False


@pytest.mark.parametrize("failure", ["close-after-full-write", "publish"])
@pytest.mark.parametrize("initially_passed", [True, False])
def test_json_write_must_close_and_publish_before_pass_evidence_is_visible(
    tmp_path, monkeypatch, failure, initially_passed,
):
    original_write = Path.write_text
    original_replace = Path.replace

    def fail_close(path, data, *args, **kwargs):
        result = original_write(path, data, *args, **kwargs)
        if failure == "close-after-full-write" and path.name == ".report.json.tmp":
            raise OSError("close failed after the complete JSON payload was written")
        return result

    def fail_publish(path, target):
        if failure == "publish" and Path(target).name == "report.json":
            # The temporary file is complete, but must not be gate evidence yet.
            assert json.loads(path.read_text(encoding="utf-8"))["passed"] is initially_passed
            raise PermissionError("atomic publication blocked")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "write_text", fail_close)
    monkeypatch.setattr(Path, "replace", fail_publish)
    report = {
        "passed": initially_passed, "stage": "complete" if initially_passed else "assets",
        "seconds": 0.1, "tokenizer_cases": [], "runtime_cases": [],
    }
    if not initially_passed:
        report["error"] = "ValueError: tokenizer SHA256 mismatch"
    save_report(tmp_path, report)
    assert report["passed"] is False
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / ".report.json.tmp").exists()
    assert report["stage"] == ("report-write" if initially_passed else "assets")
    if not initially_passed:
        assert report["error"] == "ValueError: tokenizer SHA256 mismatch"
    for name in ("report.md", "report.html"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "FAIL" in text and "PASS" not in text
