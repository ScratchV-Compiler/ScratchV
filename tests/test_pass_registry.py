"""Behavioral coverage for named pipelines, switches and functional passes."""

from __future__ import annotations

import pytest

from scratchv.assembly_passes import AssemblyPass, create_assembly_registry
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import OpCode, Program
from scratchv.main import args_to_config, build_arg_parser, main
from scratchv.optimizer.constant_folding import ConstantFolder
from scratchv.pass_interface import CompilerPass, OptimizationPassError, PassResult
from scratchv.pass_manager import (
    PassManager,
    PassRegistry,
    create_optimization_pass_manager,
)


def constant_program():
    builder = IRBuilder()
    builder.new_function("main")
    block = builder.new_block("entry")
    result = builder.add(builder.make_const(2), builder.make_const(3))
    builder.ret(result)
    return builder.program, block, result


class TestPassSwitches:
    @pytest.mark.parametrize("enabled", [False, True])
    def test_simple_if_and_registration_switch_control_real_folding(self, enabled):
        for style in ("if", "flag"):
            program, block, result = constant_program()
            manager = PassManager()
            if style == "if":
                if enabled:
                    manager.register(ConstantFolder())
            else:
                manager.register(ConstantFolder(), enabled=enabled)
            report = manager.run(program)
            assert block.instructions[0].opcode == (
                OpCode.LOAD_CONST if enabled else OpCode.ADD
            )
            assert result.const_value == (5 if enabled else None)
            assert report.total_changes == int(enabled)
            assert len(report.executions) == int(enabled)

    @pytest.mark.parametrize("level", ["basic", "all"])
    def test_disable_constant_folding_preserves_arithmetic(self, level):
        program, block, result = constant_program()
        report = create_optimization_pass_manager(
            level, disabled_passes=("constant-folding",)
        ).run(program)
        assert block.instructions[0].opcode == OpCode.ADD
        assert result.const_value is None
        assert "constant-folding" not in [item.name for item in report.executions]

    def test_explicit_pipeline_overrides_none_and_repeats_are_reported(self):
        program, _, result = constant_program()
        manager = create_optimization_pass_manager(
            "none", passes=("constant-folding", "constant-folding")
        )
        assert manager.passes[0] is not manager.passes[1]
        assert [item.changes for item in manager.run(program).executions] == [1, 0]
        assert result.const_value == 5
        assert create_optimization_pass_manager("all", passes=()).passes == []


class TestRegistry:
    def test_registration_is_lazy_and_disabled_factories_are_not_called(self):
        calls = []

        def factory():
            calls.append("constructed")
            return ConstantFolder()

        registry = PassRegistry()
        registry.register("constant-folding", factory)
        assert calls == []
        assert (
            registry.build(
                ["constant-folding"] * 2, disabled=["constant-folding"]
            ).passes
            == []
        )
        assert calls == []
        with pytest.raises(ValueError, match="unknown pass"):
            registry.build(["constant-folding", "missing"])
        assert calls == []
        with pytest.raises(ValueError, match="unknown pass"):
            registry.build([], disabled=["typo"])

    def test_duplicate_registration_and_invalid_factory_are_rejected(self):
        registry = PassRegistry()
        registry.register("constant-folding", ConstantFolder)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("constant-folding", ConstantFolder)
        registry.register("wrong-name", ConstantFolder)
        with pytest.raises(ValueError, match="returned pass named"):
            registry.build(["wrong-name"])
        registry.register("not-a-pass", lambda: object())
        with pytest.raises(TypeError, match="did not return"):
            registry.build(["not-a-pass"])

    def test_pipeline_instances_and_registries_are_isolated(self):
        registry = PassRegistry()
        registry.register("constant-folding", ConstantFolder)
        first = registry.build(["constant-folding"])
        second = registry.build(["constant-folding"])
        assert first.passes[0] is not second.passes[0]
        assert PassRegistry().names == ()


class TestFunctionalPasses:
    def test_transform_and_read_only_analysis_chain_outputs_and_warnings(self):
        manager = PassManager("text")
        manager.register(AssemblyPass("append", lambda text: PassResult(text + "!", 1)))
        manager.register(
            AssemblyPass(
                "count", lambda text: PassResult(text, warnings=[str(len(text))])
            )
        )
        result = manager.run_pipeline("hello")
        assert result.data == "hello!"
        assert result.warnings == ("6",)
        assert result.report.total_changes == 1
        assert [item.name for item in result.report.executions] == ["append", "count"]
        assert manager.run_pipeline("x").warnings == ("2",)

    def test_ir_analysis_can_share_an_optimization_pipeline(self):
        observed = []

        class Inspect(CompilerPass):
            name = "inspect"

            def run(self, program):
                observed.append(program.functions[0].blocks[0].instructions[0].opcode)
                return PassResult(program)

        program, _, _ = constant_program()
        manager = PassManager().register(ConstantFolder()).register(Inspect())
        assert manager.run(program).total_changes == 1
        assert observed == [OpCode.LOAD_CONST]

    @pytest.mark.parametrize(
        "result", [None, PassResult(None), PassResult("ok", -1), PassResult("ok", True)]
    )
    def test_failure_stops_pipeline_and_preserves_completed_stats(self, result):
        called = []
        manager = PassManager().register(
            AssemblyPass("first", lambda text: PassResult(text, 2))
        )
        manager.register(AssemblyPass("broken", lambda text: result))
        manager.register(AssemblyPass("never", lambda text: called.append(text)))
        with pytest.raises(OptimizationPassError) as caught:
            manager.run_pipeline("input")
        assert caught.value.pass_name == "broken"
        assert caught.value.completed_report.total_changes == 2
        assert called == []

    def test_wrong_stage_and_replacement_ir_are_rejected(self):
        with pytest.raises(OptimizationPassError, match="requires a Program"):
            PassManager().register(ConstantFolder()).run_pipeline("assembly")

        class Replace(CompilerPass):
            name = "replace"

            def run(self, program):
                return PassResult(Program())

        manager = PassManager().register(Replace())
        with pytest.raises(OptimizationPassError, match="in-place"):
            manager.run(Program())
        assert isinstance(manager.run_pipeline(Program()).data, Program)


class TestCliAndDriver:
    def test_cli_selection_and_disable_reach_config(self):
        config = args_to_config(
            build_arg_parser().parse_args(
                [
                    "--dsl",
                    "return 0",
                    "--opt-level",
                    "all",
                    "--passes",
                    "constant-folding, dead-code-elim,constant-folding",
                    "--disable-pass",
                    "constant-folding",
                ]
            )
        )
        report = CompilerDriver(config)._run_optimizations(constant_program()[0])
        assert [item.name for item in report.executions] == ["dead-code-elim"]

    def test_invalid_name_does_not_overwrite_existing_output(self, tmp_path, capsys):
        output = tmp_path / "result.s"
        output.write_text("keep me")
        status = main(["--dsl", "return 0", "--passes", "typo", "-o", str(output)])
        assert status == 1
        assert "unknown pass" in capsys.readouterr().err
        assert output.read_text() == "keep me"

    def test_real_assembly_adapters_preserve_existing_results(self):
        from scratchv.backend.asm_beautifier import beautify_asm
        from scratchv.backend.const_merge import merge_constants_detailed

        assembly = "lui t0, 1\naddi t0, t0, 4\nret\n"
        merged, stats = merge_constants_detailed(assembly)
        result = (
            create_assembly_registry()
            .build(["const-merge", "beautify", "count-instr"])
            .run_pipeline(assembly)
        )
        assert result.data == beautify_asm(merged)
        assert result.report.executions[0].changes == stats.total_changes
        assert result.report.executions[-1].changes == 0
        assert any("Instruction count:" in warning for warning in result.warnings)

    def test_compiler_runs_only_enabled_functional_passes(self, tmp_path):
        config = CompilerConfig(count_instr=True, beautify_asm=True)
        result = CompilerDriver(config).compile("", str(tmp_path / "out.s"), "return 0")
        assert result.success, result.errors
        assert [item["name"] for item in result.stats["assembly"]["passes"]] == [
            "beautify",
            "count-instr",
        ]
        assert result.stats["optimization"]["passes"] == []

    def test_assembly_failure_does_not_emit_output(self, monkeypatch, tmp_path):
        import scratchv.assembly_passes as adapters

        registry = PassRegistry()
        registry.register(
            "beautify", lambda: AssemblyPass("beautify", lambda text: PassResult(None))
        )
        monkeypatch.setattr(adapters, "create_assembly_registry", lambda: registry)
        output = tmp_path / "out.s"
        result = CompilerDriver(CompilerConfig(beautify_asm=True)).compile(
            "", str(output), "return 0"
        )
        assert not result.success
        assert "beautify" in result.errors[0]
        assert not output.exists()
