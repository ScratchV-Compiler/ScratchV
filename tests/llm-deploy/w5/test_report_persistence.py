"""Report I/O failures retain the primary outcome; no model or QEMU runs."""
import errno
import json
import signal

import pytest

from scratchv.runtime.generation import GenerationCancelledError


@pytest.mark.parametrize("error", [
    PermissionError(errno.EACCES, "report directory is read-only"),
    OSError(errno.ENOSPC, "report disk is full"),
    OSError(errno.EIO, "report device failed"),
], ids=["permission", "disk-full", "io"])
@pytest.mark.parametrize("build_only", [False, True], ids=["generation", "build-only"])
def test_final_write_failure_never_reports_success(
    cli, runtime, cli_helpers, tmp_path, monkeypatch, capsys, error, build_only,
):
    atomic = cli.atomic_text
    attempted = []
    collected = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def write(path, text):
        value = json.loads(text)
        attempted.append(value)
        if value["status"] in {"PASS", "BUILD_ONLY"}:
            raise error
        atomic(path, text)

    monkeypatch.setattr(cli, "atomic_text", write)
    monkeypatch.setattr(cli.gc, "collect", lambda: collected.append(True))
    options = ["--build-only"] if build_only else ["--prompt", "text", "--max-new-tokens", "1"]
    assert cli.main(cli_helpers.argv(tmp_path, *options)) == 1
    output = capsys.readouterr()
    assert "Report save failed:" in output.err and str(error) in output.err
    assert "missing or stale" in output.err
    assert "PASS:" not in output.out and "BUILD_ONLY:" not in output.out
    assert attempted[-1]["harts"] == 1 and attempted[-1]["qemu_workspace_layout"] == "none"
    assert collected == [True]
    assert all(signal.getsignal(sig) is handler for sig, handler in previous.items())
    if build_only:
        assert not runtime.fma and not runtime.adapters[0].runs
        assert not (tmp_path / "run/report.json").exists()
    else:
        # The older atomic checkpoint cannot be mistaken for final success.
        saved = cli_helpers.report(tmp_path)
        assert not saved["passed"] and saved["status"] != "PASS"


@pytest.mark.parametrize("failure,code", [
    (RuntimeError("primary generation failure"), 1),
    (TimeoutError("primary forward timed out"), 1),
    (GenerationCancelledError("primary cancellation"), 130),
    (KeyboardInterrupt("primary interrupt"), 130),
    (SystemExit(130), 130),
    (SystemExit(143), 143),
], ids=["generation", "timeout", "cancel", "interrupt", "runner-sigint", "runner-sigterm"])
@pytest.mark.parametrize("save_error", [
    PermissionError(errno.EACCES, "secondary save permission failure"),
    OSError(errno.ENOSPC, "secondary save disk full"),
], ids=["permission", "disk-full"])
def test_secondary_write_failure_keeps_primary_error_and_exit(
    cli, runtime, cli_helpers, tmp_path, monkeypatch, capsys, failure, code, save_error,
):
    runtime.failure = failure
    snapshots = []
    collected = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def write(path, text):
        snapshots.append(json.loads(text))
        raise save_error

    monkeypatch.setattr(cli, "atomic_text", write)
    monkeypatch.setattr(cli.gc, "collect", lambda: collected.append(True))
    assert cli.main(cli_helpers.argv(tmp_path, "--prompt", "text", "--max-new-tokens", "1")) == code
    output = capsys.readouterr()
    assert str(failure) in output.err and type(failure).__name__ in output.err
    assert str(save_error) in output.err and "Report save failed:" in output.err
    assert "report.json" not in output.out
    assert snapshots[-1]["error"] == f"{type(failure).__name__}: {failure}"
    assert snapshots[-1]["status"] == ("CANCELLED" if code in (130, 143) else "FAIL")
    assert snapshots[-1]["forward_attempts"] == 1 and not snapshots[-1]["passed"]
    assert collected == [True]
    assert all(signal.getsignal(sig) is handler for sig, handler in previous.items())


def test_intermediate_write_failure_is_saved_as_failure_when_storage_recovers(
    cli, runtime, cli_helpers, tmp_path, monkeypatch, capsys,
):
    atomic = cli.atomic_text
    attempts = []

    def write(path, text):
        attempts.append(json.loads(text))
        if len(attempts) == 1:
            raise OSError(errno.ENOSPC, "checkpoint disk full")
        atomic(path, text)

    monkeypatch.setattr(cli, "atomic_text", write)
    assert cli.main(cli_helpers.argv(tmp_path, "--prompt", "text", "--max-new-tokens", "1")) == 1
    saved = cli_helpers.report(tmp_path)
    assert len(attempts) == 2
    assert saved["status"] == "FAIL" and not saved["passed"]
    assert saved["qemu_forwards"] == 1 and len(saved["cases"]) == 1
    assert "checkpoint disk full" in saved["error"]
    assert "checkpoint disk full" in capsys.readouterr().err


def test_suite_initial_checkpoint_failure_prevents_generation(
    cli, runtime, reference_suite, cli_helpers, tmp_path, monkeypatch,
):
    atomic = cli.atomic_text
    attempts = []

    def write(path, text):
        attempts.append(json.loads(text))
        if len(attempts) == 1:
            raise PermissionError(errno.EACCES, "cannot save RUNNING")
        atomic(path, text)

    monkeypatch.setattr(cli, "atomic_text", write)
    assert cli.main(cli_helpers.argv(
        tmp_path, "--reference-suite", "--verify-onnx", "model.onnx",
    )) == 1
    saved = cli_helpers.report(tmp_path)
    assert saved["full_model_gate"] == "FAIL" and saved["qemu_forwards"] == 0
    assert saved["numeric_steps"] == [] and saved["cases"] == []
    assert "cannot save RUNNING" in saved["error"]


def test_numeric_failure_survives_report_write_failure(
    cli, runtime, reference_suite, cli_helpers, tmp_path, monkeypatch, capsys,
):
    runtime.oracle = cli_helpers.logits(4)
    atomic = cli.atomic_text
    snapshots = []

    def write(path, text):
        value = json.loads(text)
        snapshots.append(value)
        if value["full_model_gate"] == "FAIL":
            raise OSError(errno.ENOSPC, "numeric-report disk full")
        atomic(path, text)

    monkeypatch.setattr(cli, "atomic_text", write)
    assert cli.main(cli_helpers.argv(
        tmp_path, "--reference-suite", "--verify-onnx", "model.onnx",
    )) == 1
    output = capsys.readouterr()
    assert "QEMU/ORT full logits, selected row or frozen token differ" in output.err
    assert "numeric-report disk full" in output.err
    assert snapshots[-1]["full_model_gate"] == "FAIL"
    assert not snapshots[-1]["numeric_steps"][0]["passed"]
    assert "QEMU/ORT" in snapshots[-1]["error"]
    assert cli_helpers.report(tmp_path)["full_model_gate"] == "RUNNING"
    assert "PASS:" not in output.out


def test_reference_text_failure_survives_report_write_failure(
    cli, runtime, reference_suite, cli_helpers, tmp_path, monkeypatch, capsys,
):
    reference_suite.fixture["cases"][0]["generated_text"] = "not-the-result"
    fixture_path = cli.ROOT / "probes/w5_generation/references.json"
    fixture_path.write_text(json.dumps(reference_suite.fixture))
    atomic = cli.atomic_text
    snapshots = []

    def write(path, text):
        value = json.loads(text)
        snapshots.append(value)
        if value["cases"]:
            raise PermissionError(errno.EACCES, "reference-report denied")
        atomic(path, text)

    monkeypatch.setattr(cli, "atomic_text", write)
    assert cli.main(cli_helpers.argv(
        tmp_path, "--reference-suite", "--verify-onnx", "model.onnx",
    )) == 1
    output = capsys.readouterr()
    assert "Frozen reference mismatch: france" in output.err
    assert "reference-report denied" in output.err
    assert snapshots[-1]["error"] == "ValueError: Frozen reference mismatch: france"
    assert snapshots[-1]["full_model_gate"] == "FAIL"
    assert snapshots[-1]["numeric_steps"][0]["passed"]
    assert not snapshots[-1]["cases"][0]["reference_passed"]


def test_serialization_failure_returns_failure_without_false_saved_message(
    cli, runtime, cli_helpers, tmp_path, capsys,
):
    runtime.exe.evidence["invalid_measurement"] = float("nan")
    assert cli.main(cli_helpers.argv(tmp_path, "--build-only")) == 1
    output = capsys.readouterr()
    assert "Report save failed: ValueError:" in output.err
    assert "BUILD_ONLY:" not in output.out
    assert not (tmp_path / "run/report.json").exists()


def test_next_invocation_can_save_after_failed_final_report(
    cli, runtime, cli_helpers, tmp_path, monkeypatch,
):
    atomic = cli.atomic_text

    def deny(path, text):
        raise PermissionError(errno.EACCES, "first output unavailable")

    monkeypatch.setattr(cli, "atomic_text", deny)
    assert cli.main(cli_helpers.argv(tmp_path, "--build-only")) == 1
    monkeypatch.setattr(cli, "atomic_text", atomic)
    next_root = tmp_path / "next"
    assert cli.main(cli_helpers.argv(next_root, "--prompt", "text", "--max-new-tokens", "1")) == 0
    saved = cli_helpers.report(next_root)
    assert saved["status"] == "PASS" and saved["passed"]
    assert saved["qemu_forwards"] == 1 and "error" not in saved
    assert saved["harts"] == 1 and saved["qemu_workspace_layout"] == "none"


@pytest.mark.parametrize("code", [130, 143], ids=["sigint", "sigterm"])
@pytest.mark.parametrize("suite", [False, True], ids=["prompt", "reference-suite"])
def test_runner_signal_exit_is_saved_as_cancellation(
    cli, runtime, reference_suite, cli_helpers, tmp_path, monkeypatch, capsys, code, suite,
):
    runtime.failure = SystemExit(code)
    collected = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(cli.gc, "collect", lambda: collected.append(True))
    options = (["--reference-suite", "--verify-onnx", "model.onnx"] if suite else
               ["--prompt", "text", "--max-new-tokens", "1"])
    assert cli.main(cli_helpers.argv(tmp_path, *options)) == code
    saved = cli_helpers.report(tmp_path)
    assert saved["status"] == "CANCELLED" and not saved["passed"]
    assert saved["error"] == f"SystemExit: {code}"
    assert saved["full_model_gate"] == ("CANCELLED" if suite else "NOT_RUN")
    assert saved["forward_attempts"] == 1 and saved["qemu_forwards"] == 0
    assert f"SystemExit: {code}" in capsys.readouterr().err
    assert collected == [True]
    assert all(signal.getsignal(sig) is handler for sig, handler in previous.items())


@pytest.mark.parametrize("code", [0, 2, "143"], ids=["success", "other-error", "message"])
@pytest.mark.parametrize("save_fails", [False, True], ids=["saved", "save-fails"])
def test_non_signal_system_exit_propagates_after_cleanup(
    cli, runtime, cli_helpers, tmp_path, monkeypatch, capsys, code, save_fails,
):
    failure = SystemExit(code)
    runtime.failure = failure
    collected = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(cli.gc, "collect", lambda: collected.append(True))
    if save_fails:
        def deny(path, text):
            raise OSError(errno.ENOSPC, "unrelated exit report disk full")
        monkeypatch.setattr(cli, "atomic_text", deny)
    with pytest.raises(SystemExit) as caught:
        cli.main(cli_helpers.argv(tmp_path, "--prompt", "text", "--max-new-tokens", "1"))
    assert caught.value is failure and caught.value.code == code
    output = capsys.readouterr()
    assert "CANCELLED" not in output.out
    if save_fails:
        assert "Report save failed:" in output.err
        assert "unrelated exit report disk full" in output.err
    else:
        assert cli_helpers.report(tmp_path)["status"] == "FAIL"
    assert collected == [True]
    assert all(signal.getsignal(sig) is handler for sig, handler in previous.items())
