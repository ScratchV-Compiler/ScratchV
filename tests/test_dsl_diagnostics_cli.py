"""End-to-end tests for DSL diagnostics through driver and CLI."""

from pathlib import Path

from scratchv.compiler import CompilerDriver
from scratchv.main import main


def test_driver_preserves_structured_dsl_diagnostic(tmp_path: Path):
    source = tmp_path / "bad.dsl"
    source.write_text("x = ad(a, b)\n", encoding="utf-8")

    result = CompilerDriver().compile(str(source), str(tmp_path / "out.s"))

    assert not result.success
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.error_code == "E200"
    assert diagnostic.filename == str(source)
    assert result.errors == [str(diagnostic)]
    assert "Parse error:" not in result.errors[0]


def test_driver_does_not_mask_extended_dsl_error(tmp_path: Path):
    source = tmp_path / "bad.dsl"
    source.write_text("if (a > b):\nx = add(a, b)\n", encoding="utf-8")

    result = CompilerDriver().compile(str(source), str(tmp_path / "out.s"))

    assert not result.success
    assert result.diagnostics[0].error_code == "E111"
    assert "unterminated if block" in result.errors[0]


def test_driver_preserves_multiple_diagnostics_and_limit(tmp_path: Path):
    source = tmp_path / "many.dsl"
    source.write_text(
        "\n".join(f"bad statement {index}" for index in range(25)),
        encoding="utf-8",
    )
    result = CompilerDriver().compile(str(source), str(tmp_path / "out.s"))
    assert not result.success
    assert len(result.diagnostics) == 20
    assert len(result.errors) == 20
    assert result.diagnostic_limit_reached
    assert result.diagnostic_limit == 20


def test_cli_renders_multiple_diagnostics_once(tmp_path: Path, capsys):
    source = tmp_path / "many.dsl"
    source.write_text(
        "retrun x\ny = ad(a, b)\nz = relu(a, b)\n",
        encoding="utf-8",
    )
    assert main([str(source), "-o", str(tmp_path / "out.s")]) == 1
    captured = capsys.readouterr()
    assert captured.err.count(": error[") == 3


def test_cli_renders_diagnostic_once_without_ansi(tmp_path: Path, capsys):
    source = tmp_path / "bad.dsl"
    source.write_text("retrun x\n", encoding="utf-8")

    exit_code = main([str(source), "-o", str(tmp_path / "out.s")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.count("error[E100]") == 1
    assert "\033[" not in captured.err
    assert "Error: Parse error:" not in captured.err


def test_cli_reports_unexpected_dsl_internal_error_as_exit_2(
    monkeypatch, capsys,
):
    def fail(*args, **kwargs):
        raise AssertionError("broken invariant")

    monkeypatch.setattr(CompilerDriver, "compile", fail)
    exit_code = main(["broken.dsl"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "internal compiler error: broken invariant" in captured.err
    assert "Traceback" not in captured.err


def test_cli_does_not_change_non_dsl_internal_error_behavior(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("onnx invariant")

    monkeypatch.setattr(CompilerDriver, "compile", fail)
    try:
        main(["model.onnx"])
        assert False, "non-DSL exception should retain its previous behavior"
    except AssertionError as exc:
        assert str(exc) == "onnx invariant"
