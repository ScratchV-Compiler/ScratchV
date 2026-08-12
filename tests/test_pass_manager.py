"""Tests for the unified IR optimization pass framework."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import scratchv.compiler as compiler_module
from scratchv.compiler import (
    CompilerConfig,
    CompilerDriver,
    PassManager,
    create_optimization_pass_manager,
)
from scratchv.ir.types import Program
from scratchv.main import args_to_config, build_arg_parser
from scratchv.optimizer import (
    ConstantFolder,
    DeadCodeEliminator,
    IRPeepholeOptimizer,
    LICM,
    MulAddFusion,
)
from scratchv.pass_interface import (
    OptimizationPass,
    OptimizationPassError,
    OptimizationReport,
    PassExecutionStats,
)


class _RecordingPass(OptimizationPass):
    """Small public-contract fake used by PassManager tests."""

    def __init__(
        self,
        name: str,
        changes: int = 0,
        calls: list[str] | None = None,
    ):
        self._name = name
        self._changes = changes
        self._calls = calls

    @property
    def name(self) -> str:
        return self._name

    def optimize(self, program: Program) -> int:
        if self._calls is not None:
            self._calls.append(self.name)
        return self._changes


class TestOptimizationPass:
    def test_base_class_is_abstract(self):
        with pytest.raises(TypeError):
            OptimizationPass()

    @pytest.mark.parametrize("name", ["", "   ", None, 7])
    def test_register_rejects_invalid_name(self, name):
        manager = PassManager()

        with pytest.raises((TypeError, ValueError)):
            manager.register(_RecordingPass(name))

    @pytest.mark.parametrize(
        "pass_type",
        [
            ConstantFolder,
            DeadCodeEliminator,
            IRPeepholeOptimizer,
            MulAddFusion,
            LICM,
        ],
    )
    def test_production_passes_share_the_interface_and_local_counts(
        self, pass_type
    ):
        pass_ = pass_type()
        program = Program()

        assert isinstance(pass_, OptimizationPass)
        assert pass_.name
        assert pass_.optimize(program) == 0
        assert pass_.optimize(program) == 0


class TestPassManager:
    def test_runs_in_registration_order_and_reports_each_execution(self):
        calls: list[str] = []
        program = Program()
        manager = PassManager("ordered")
        manager.register(_RecordingPass("first", 2, calls))
        manager.add(_RecordingPass("second", 3, calls))

        report = manager.run(program)

        assert calls == ["first", "second"]
        assert report.pipeline_name == "ordered"
        assert report.total_changes == 5
        assert report.elapsed_seconds >= 0
        assert [execution.index for execution in report.executions] == [0, 1]
        assert [execution.name for execution in report.executions] == [
            "first",
            "second",
        ]
        assert [execution.changes for execution in report.executions] == [2, 3]
        assert all(
            execution.elapsed_seconds >= 0 for execution in report.executions
        )

    def test_allows_duplicate_pass_names(self):
        manager = PassManager()
        manager.register(_RecordingPass("repeat", 1))
        manager.register(_RecordingPass("repeat", 2))

        report = manager.run(Program())

        assert [(item.index, item.name) for item in report.executions] == [
            (0, "repeat"),
            (1, "repeat"),
        ]
        assert report.total_changes == 3

    def test_empty_pipeline_returns_an_empty_report(self):
        report = PassManager("empty").run(Program())

        assert report == OptimizationReport("empty", (), 0, 0.0)

    def test_reports_are_immutable(self):
        execution = PassExecutionStats(0, "pass", 0, 0.0)

        with pytest.raises(FrozenInstanceError):
            execution.changes = 1

    @pytest.mark.parametrize("result", [-1, True, 1.5, "1", None])
    def test_rejects_invalid_change_counts(self, result):
        manager = PassManager().register(_RecordingPass("invalid", result))

        with pytest.raises(OptimizationPassError) as caught:
            manager.run(Program())

        assert caught.value.pass_index == 0
        assert caught.value.pass_name == "invalid"
        assert caught.value.completed_report.executions == ()
        assert caught.value.__cause__ is caught.value.cause

    def test_wraps_failure_and_stops_before_later_passes(self):
        calls: list[str] = []

        class _FailingPass(_RecordingPass):
            def optimize(self, program: Program) -> int:
                calls.append(self.name)
                raise RuntimeError("boom")

        manager = PassManager("failure")
        manager.register(_RecordingPass("done", 4, calls))
        manager.register(_FailingPass("broken"))
        manager.register(_RecordingPass("never", 1, calls))

        with pytest.raises(OptimizationPassError) as caught:
            manager.run(Program())

        error = caught.value
        assert calls == ["done", "broken"]
        assert error.pass_index == 1
        assert error.pass_name == "broken"
        assert error.elapsed_seconds >= 0
        assert error.completed_report.total_changes == 4
        assert [item.name for item in error.completed_report.executions] == ["done"]
        assert isinstance(error.cause, RuntimeError)
        assert error.__cause__ is error.cause

    def test_rejects_non_program_before_running_passes(self):
        calls: list[str] = []
        manager = PassManager().register(_RecordingPass("pass", calls=calls))

        with pytest.raises(TypeError):
            manager.run(object())

        assert calls == []

    def test_does_not_catch_keyboard_interrupt(self):
        class _InterruptingPass(_RecordingPass):
            def optimize(self, program: Program) -> int:
                raise KeyboardInterrupt

        manager = PassManager().register(_InterruptingPass("interrupt"))

        with pytest.raises(KeyboardInterrupt):
            manager.run(Program())


class TestOptimizationLevels:
    @pytest.mark.parametrize(
        ("level", "names"),
        [
            ("none", []),
            ("basic", ["constant-folding", "dead-code-elim"]),
            (
                "all",
                [
                    "constant-folding",
                    "dead-code-elim",
                    "ir-peephole",
                    "muladd-fusion",
                    "licm",
                ],
            ),
        ],
    )
    def test_factory_builds_the_documented_pipeline(self, level, names):
        manager = create_optimization_pass_manager(level)

        assert [pass_.name for pass_ in manager.passes] == names

    @pytest.mark.parametrize("level", ["", "Basic", "fast", None, 1])
    def test_factory_rejects_unknown_level(self, level):
        with pytest.raises((TypeError, ValueError)):
            create_optimization_pass_manager(level)


class TestOptimizationCli:
    @pytest.mark.parametrize("flag", ["--opt-level", "--optimize"])
    @pytest.mark.parametrize("level", ["none", "basic", "all"])
    def test_accepts_canonical_and_compatibility_flags(self, flag, level):
        args = build_arg_parser().parse_args(["model.onnx", flag, level])

        assert args.optimize_level == level
        assert args_to_config(args).optimize_level == level

    def test_defaults_to_none(self):
        args = build_arg_parser().parse_args(["model.onnx"])

        assert args.optimize_level == "none"


class _ProgramDriver(CompilerDriver):
    """Compiler driver test seam that avoids parser and backend dependencies."""

    def __init__(self, config: CompilerConfig):
        super().__init__(config)
        self.codegen_called = False

    def _parse(self, input_path: str, dsl_source: str | None = None) -> Program:
        return Program()

    def _generate_code(self, program: Program) -> str:
        self.codegen_called = True
        return "generated"


class TestCompilerOptimizationIntegration:
    def test_exposes_structured_optimization_stats(self, tmp_path):
        output_path = tmp_path / "output.s"
        driver = _ProgramDriver(CompilerConfig(optimize_level="basic"))

        result = driver.compile("input.dsl", str(output_path))

        assert result.success
        assert result.stats["optimization"]["level"] == "basic"
        assert result.stats["optimization"]["total_changes"] == 0
        assert [
            item["name"] for item in result.stats["optimization"]["passes"]
        ] == ["constant-folding", "dead-code-elim"]
        assert result.stats["opt_message"]

    def test_pass_failure_stops_codegen_and_output(self, monkeypatch, tmp_path):
        class _FailingPass(_RecordingPass):
            def optimize(self, program: Program) -> int:
                raise RuntimeError("optimizer failed")

        manager = PassManager("optimizer").register(_FailingPass("broken"))
        monkeypatch.setattr(
            compiler_module,
            "create_optimization_pass_manager",
            lambda level: manager,
        )
        output_path = tmp_path / "must-not-exist.s"
        driver = _ProgramDriver(CompilerConfig(optimize_level="all"))

        result = driver.compile("input.dsl", str(output_path))

        assert not result.success
        assert "broken" in result.errors[0]
        assert not driver.codegen_called
        assert not output_path.exists()
