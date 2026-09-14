"""Tests for --extended-isel CLI/config wiring (Topic 28)."""

from unittest import mock

from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.main import args_to_config, build_arg_parser

DSL_SOURCE = "y = add(a, b)\nreturn y\n"


def _write_dsl(tmp_path):
    path = tmp_path / "model.dsl"
    path.write_text(DSL_SOURCE)
    return str(path)


def test_cli_flags_map_to_config():
    args = build_arg_parser().parse_args([
        "-o", "x", "--dsl", "a",
        "--extended-isel", "--no-fp64", "--hardware-sqrt",
    ])
    config = args_to_config(args)
    assert config.extended_isel is True
    assert config.enable_fp64 is False
    assert config.use_hardware_sqrt is True


def test_cli_defaults_map_to_config():
    args = build_arg_parser().parse_args(["input.dsl"])
    config = args_to_config(args)
    assert config.extended_isel is False
    assert config.enable_fp64 is True
    assert config.use_hardware_sqrt is False


def test_driver_uses_extended_selector(tmp_path):
    inp = _write_dsl(tmp_path)
    out = str(tmp_path / "out.s")
    with mock.patch(
        "scratchv.backend.inst_select_ext.ExtendedInstructionSelector"
    ) as selector_cls:
        selector_cls.return_value.run.return_value = []
        result = CompilerDriver(
            CompilerConfig(extended_isel=True)).compile(inp, out)

    assert result.success, result.errors
    selector_cls.assert_called_once()
    assert selector_cls.call_args.kwargs == {
        "enable_fp64": True, "use_hardware_sqrt": False,
    }


def test_driver_passes_fp64_flags_to_selector(tmp_path):
    inp = _write_dsl(tmp_path)
    out = str(tmp_path / "out.s")
    with mock.patch(
        "scratchv.backend.inst_select_ext.ExtendedInstructionSelector"
    ) as selector_cls:
        selector_cls.return_value.run.return_value = []
        result = CompilerDriver(CompilerConfig(
            extended_isel=True, enable_fp64=False,
            use_hardware_sqrt=True)).compile(inp, out)

    assert result.success, result.errors
    assert selector_cls.call_args.kwargs == {
        "enable_fp64": False, "use_hardware_sqrt": True,
    }


def test_default_uses_base_selector(tmp_path):
    inp = _write_dsl(tmp_path)
    out = str(tmp_path / "out.s")
    with mock.patch(
        "scratchv.backend.instruction_select.InstructionSelector"
    ) as base_cls, mock.patch(
        "scratchv.backend.inst_select_ext.ExtendedInstructionSelector"
    ) as ext_cls:
        base_cls.return_value.run.return_value = []
        result = CompilerDriver(CompilerConfig()).compile(inp, out)

    assert result.success, result.errors
    base_cls.assert_called_once()
    ext_cls.assert_not_called()


def test_dag_isel_conflict_warns(tmp_path):
    inp = _write_dsl(tmp_path)
    out = str(tmp_path / "out.s")
    driver = CompilerDriver(
        CompilerConfig(extended_isel=True, use_dag_isel=True))
    with mock.patch.object(driver, "_generate_code", return_value=""):
        result = driver.compile(inp, out)

    assert result.success, result.errors
    assert any("precedence" in w for w in result.warnings)


def test_llvm_backend_warns(tmp_path):
    inp = _write_dsl(tmp_path)
    out = str(tmp_path / "out.ll")
    driver = CompilerDriver(
        CompilerConfig(extended_isel=True, backend="llvm"))
    with mock.patch.object(driver, "_generate_code", return_value=""):
        result = driver.compile(inp, out)

    assert result.success, result.errors
    assert any("RISC-V only" in w for w in result.warnings)


def test_fp64_flags_without_extended_warn(tmp_path):
    inp = _write_dsl(tmp_path)
    out = str(tmp_path / "out.s")
    result = CompilerDriver(
        CompilerConfig(enable_fp64=False)).compile(inp, out)

    assert result.success, result.errors
    assert any("no effect" in w for w in result.warnings)


def test_hardware_sqrt_without_extended_warns(tmp_path):
    inp = _write_dsl(tmp_path)
    out = str(tmp_path / "out.s")
    result = CompilerDriver(
        CompilerConfig(use_hardware_sqrt=True)).compile(inp, out)

    assert result.success, result.errors
    assert any("no effect" in w for w in result.warnings)
