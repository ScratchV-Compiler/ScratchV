"""Compiler-driver integration tests for extended instruction selection."""

from pathlib import Path

import pytest

from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.main import args_to_config, build_arg_parser
from scratchv.simulator.tinyfive import verify_assembly


class TestExtendedSelectorCLI:
    def test_extended_isel_cli_flag_reaches_compiler_config(self):
        args = build_arg_parser().parse_args(["input.dsl", "--extended-isel"])

        config = args_to_config(args)

        assert config.use_extended_isel is True

    def test_extended_isel_rejects_llvm_backend(self, tmp_path: Path):
        driver = CompilerDriver(
            CompilerConfig(backend="llvm", use_extended_isel=True),
        )

        result = driver.compile(
            "",
            str(tmp_path / "out.ll"),
            dsl_source="x = add(a, b)\nreturn x",
        )

        assert not result.success
        assert result.errors == ["--extended-isel requires the RISC-V backend"]

    def test_extended_isel_rejects_dag_selector(self, tmp_path: Path):
        driver = CompilerDriver(
            CompilerConfig(use_dag_isel=True, use_extended_isel=True),
        )

        result = driver.compile(
            "",
            str(tmp_path / "out.s"),
            dsl_source="x = add(a, b)\nreturn x",
        )

        assert not result.success
        assert result.errors == [
            "--extended-isel and --dag-isel are mutually exclusive",
        ]

    @pytest.mark.parametrize("reg_alloc", ["naive", "greedy", "linear"])
    def test_extended_isel_compiles_integer_remainder_end_to_end(
        self,
        tmp_path: Path,
        reg_alloc: str,
    ):
        source = """
a = const(7, dtype:i32)
b = const(3, dtype:i32)
r = rem(a, b)
return r
"""
        driver = CompilerDriver(
            CompilerConfig(
                use_extended_isel=True,
                verify_ir=True,
                reg_alloc=reg_alloc,
            ),
        )

        result = driver.compile(
            "",
            str(tmp_path / "out.s"),
            dsl_source=source,
        )

        assert result.success, result.errors
        rem_line = next(
            line
            for line in result.output_text.splitlines()
            if line.strip().startswith("rem ")
        )
        assert "rem t" in rem_line
        assert ", 7" not in rem_line
        assert ", 3" not in rem_line

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("rem(a, b)", -2),
            ("min(a, b)", -17),
            ("max(a, b)", 5),
            ("abs(a)", 17),
        ],
    )
    def test_extended_integer_ops_execute_in_tinyfive(
        self,
        tmp_path: Path,
        expression: str,
        expected: int,
    ):
        pytest.importorskip("tinyfive")
        source = """
a = const(-17, dtype:i32)
b = const(5, dtype:i32)
r = {expression}
return r
""".format(
            expression=expression
        )
        driver = CompilerDriver(
            CompilerConfig(use_extended_isel=True, verify_ir=True),
        )

        result = driver.compile(
            "",
            str(tmp_path / "out.s"),
            dsl_source=source,
        )
        execution = verify_assembly(result.output_text)

        assert result.success, result.errors
        assert execution["success"], execution["error"]
        assert execution["return_value"] == expected

    def test_extended_abs_int32_min_preserves_twos_complement_value(
        self,
        tmp_path: Path,
    ):
        pytest.importorskip("tinyfive")
        source = """
a = const(-2147483648, dtype:i32)
r = abs(a)
return r
"""
        result = CompilerDriver(
            CompilerConfig(use_extended_isel=True, verify_ir=True),
        ).compile("", str(tmp_path / "out.s"), dsl_source=source)
        execution = verify_assembly(result.output_text)

        assert result.success, result.errors
        assert execution["success"], execution["error"]
        assert execution["return_value"] == -(1 << 31)

    def test_extended_isel_reports_unsupported_integer_operation_type(
        self,
        tmp_path: Path,
    ):
        source = """
a = const(7.0, dtype:f32)
b = const(3.0, dtype:f32)
r = rem(a, b)
return r
"""
        driver = CompilerDriver(CompilerConfig(use_extended_isel=True))

        result = driver.compile(
            "",
            str(tmp_path / "out.s"),
            dsl_source=source,
        )

        assert not result.success
        assert "rem does not support f32" in result.errors[0]

    def test_integer_extension_ops_require_explicit_feature_flag(
        self,
        tmp_path: Path,
    ):
        source = """
a = const(-7, dtype:i32)
r = abs(a)
return r
"""

        result = CompilerDriver().compile(
            "",
            str(tmp_path / "out.s"),
            dsl_source=source,
        )

        assert not result.success
        assert result.errors == [
            "--extended-isel is required for operations: abs",
        ]
