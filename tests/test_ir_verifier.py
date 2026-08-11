"""Tests for the IR verifier module."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from scratchv.frontend.dsl_parser import DSLParser
from scratchv.frontend.dsl_extended import ExtendedDSLParser
from scratchv.analysis.ir_verifier import (
    IRVerifier, VerificationError, ErrorLevel, verify_ir,
)
from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.ir.types import (
    Program, Function, Instruction, BasicBlock, OpCode, Value, DataType,
)
from scratchv.main import args_to_config, build_arg_parser
from scratchv.pass_interface import PassResult


class StubCompilerDriver(CompilerDriver):
    """Compiler driver with deterministic parsing and code generation."""

    def __init__(self, config: CompilerConfig, program: Program) -> None:
        super().__init__(config)
        self.program = program
        self.codegen_called = False

    def _parse(
        self, input_path: str, dsl_source: Optional[str] = None,
    ) -> Program:
        return self.program

    def _generate_code(self, program: Program) -> str:
        self.codegen_called = True
        return "stub assembly\n"


def make_valid_program() -> Program:
    program = Program()
    param = Value(name="x")
    func = Function(name="main", params=[param])
    program.add_function(func)
    block = func.new_block("entry")
    block.add(Instruction(opcode=OpCode.RETURN, operands=[param]))
    return program


def make_undefined_program() -> Program:
    program = Program()
    func = Function(name="main")
    program.add_function(func)
    block = func.new_block("entry")
    block.add(Instruction(
        opcode=OpCode.RETURN,
        operands=[Value(name="missing")],
    ))
    return program


def make_warning_only_program() -> Program:
    program = Program()
    func = Function(name="main")
    program.add_function(func)
    block = func.new_block("entry")
    left = Value(
        name="left", dtype=DataType.FLOAT32,
        is_constant=True, const_value=1.0,
    )
    right = Value(
        name="right", dtype=DataType.INT32,
        is_constant=True, const_value=2,
    )
    result = Value(name="result")
    block.add(Instruction(
        opcode=OpCode.ADD,
        dest=result,
        operands=[left, right],
    ))
    block.add(Instruction(opcode=OpCode.RETURN, operands=[result]))
    return program


class TestVerificationError:
    """Tests for VerificationError dataclass."""

    def test_create_error(self) -> None:
        err = VerificationError(
            level=ErrorLevel.ERROR,
            message="value used before definition",
            function_name="main",
            block_name="entry",
            instruction_index=2,
            value_name="x",
            rule="def-before-use",
        )
        assert err.level == ErrorLevel.ERROR
        assert "main" in err.function_name or err.function_name == "main"
        assert err.rule == "def-before-use"

    def test_create_warning(self) -> None:
        err = VerificationError(
            level=ErrorLevel.WARNING,
            message="type mismatch",
            rule="type-consistency",
        )
        assert err.level == ErrorLevel.WARNING

    def test_str_representation(self) -> None:
        err = VerificationError(
            level=ErrorLevel.ERROR,
            message="test message",
            function_name="main",
            rule="test-rule",
        )
        s = str(err)
        assert "ERROR" in s
        assert "test-rule" in s
        assert "test message" in s


class TestIRVerifier:
    """Tests for the IRVerifier class."""

    # ------------------------------------------------------------------
    # Simple valid programs
    # ------------------------------------------------------------------

    def test_valid_simple_program(self) -> None:
        dsl = """
        c = add(a, b)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        verifier = IRVerifier(program)
        errors = verifier.verify()
        assert len(errors) == 0  # Should be valid
        assert {value.name for value in program.functions[0].params} == {
            "a", "b",
        }

    def test_valid_nn_pipeline(self) -> None:
        dsl = """
        t1 = relu(x)
        t2 = softmax(t1, axis:-1)
        return t2
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        verifier = IRVerifier(program)
        errors = verifier.verify()
        assert len(errors) == 0

    # ------------------------------------------------------------------
    # def-before-use
    # ------------------------------------------------------------------

    def test_function_parameter_is_defined_at_entry(self) -> None:
        """Function parameters are available before the first instruction."""
        program = Program()
        v_x = Value(name="x", dtype=DataType.FLOAT32)
        func = Function(name="main", params=[v_x])
        program.add_function(func)
        block = func.new_block("entry")

        use_instr = Instruction(
            opcode=OpCode.ADD,
            dest=Value(name="c"),
            operands=[v_x, v_x],
        )
        block.add(use_instr)
        # Add return at end
        block.add(Instruction(
            opcode=OpCode.RETURN,
            operands=[Value(name="c")],
        ))

        verifier = IRVerifier(program)
        errors = verifier.verify()
        def_errors = [e for e in errors if e.rule == "def-before-use"]
        assert def_errors == []

    def test_undefined_value_is_rejected(self) -> None:
        """A non-constant operand must be declared or produced."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        block = func.new_block("entry")
        missing = Value(name="missing")
        result = Value(name="result")
        block.add(Instruction(
            opcode=OpCode.ADD,
            dest=result,
            operands=[missing, missing],
        ))
        block.add(Instruction(opcode=OpCode.RETURN, operands=[result]))

        errors = IRVerifier(program).verify()
        def_errors = [e for e in errors if e.rule == "def-before-use"]

        assert len(def_errors) == 2
        assert all(e.value_name == "missing" for e in def_errors)

    def test_use_before_later_definition_is_rejected(self) -> None:
        """A definition later in the block cannot satisfy an earlier use."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        block = func.new_block("entry")
        value = Value(name="later")
        result = Value(name="result")
        one = Value(name="one", is_constant=True, const_value=1.0)
        block.add(Instruction(
            opcode=OpCode.ADD,
            dest=result,
            operands=[value, one],
        ))
        block.add(Instruction(
            opcode=OpCode.ADD,
            dest=value,
            operands=[one, one],
        ))
        block.add(Instruction(opcode=OpCode.RETURN, operands=[result]))

        errors = IRVerifier(program).verify()
        def_errors = [e for e in errors if e.rule == "def-before-use"]

        assert len(def_errors) == 1
        assert def_errors[0].instruction_index == 0
        assert def_errors[0].value_name == "later"

    def test_definition_must_dominate_cross_block_use(self) -> None:
        """A value defined on one branch is unavailable at the merge."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        entry = func.new_block("entry")
        left = func.new_block("left")
        right = func.new_block("right")
        merge = func.new_block("merge")
        cond = Value(name="cond", is_constant=True, const_value=1)
        one = Value(name="one", is_constant=True, const_value=1.0)
        branch_value = Value(name="branch_value")

        entry.add(Instruction(
            opcode=OpCode.BR_IF,
            operands=[cond],
            target="left,right",
        ))
        left.add(Instruction(
            opcode=OpCode.ADD,
            dest=branch_value,
            operands=[one, one],
        ))
        left.add(Instruction(opcode=OpCode.BR, target="merge"))
        right.add(Instruction(opcode=OpCode.BR, target="merge"))
        merge.add(Instruction(
            opcode=OpCode.RETURN,
            operands=[branch_value],
        ))

        errors = IRVerifier(program).verify()
        def_errors = [e for e in errors if e.rule == "def-before-use"]

        assert len(def_errors) == 1
        assert def_errors[0].block_name == "merge"
        assert def_errors[0].value_name == "branch_value"

    def test_entry_definition_dominates_successor_use(self) -> None:
        """A value defined in entry is available in a successor block."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        entry = func.new_block("entry")
        exit_block = func.new_block("exit")
        one = Value(name="one", is_constant=True, const_value=1.0)
        value = Value(name="value")
        entry.add(Instruction(
            opcode=OpCode.ADD,
            dest=value,
            operands=[one, one],
        ))
        entry.add(Instruction(opcode=OpCode.BR, target="exit"))
        exit_block.add(Instruction(opcode=OpCode.RETURN, operands=[value]))

        errors = IRVerifier(program).verify()
        def_errors = [e for e in errors if e.rule == "def-before-use"]

        assert def_errors == []

    def test_empty_block_falls_through_for_dominance(self) -> None:
        """Empty warning-only blocks preserve sequential CFG flow."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        entry = func.new_block("entry")
        func.new_block("gap")
        exit_block = func.new_block("exit")
        one = Value(name="one", is_constant=True, const_value=1.0)
        value = Value(name="value")
        entry.add(Instruction(
            opcode=OpCode.ADD,
            dest=value,
            operands=[one, one],
        ))
        entry.add(Instruction(opcode=OpCode.BR, target="gap"))
        exit_block.add(Instruction(opcode=OpCode.RETURN, operands=[value]))

        passed, errors = verify_ir(program)

        assert passed is True
        assert [e for e in errors if e.rule == "def-before-use"] == []

    def test_program_global_is_defined_at_entry(self) -> None:
        """Program globals are available to every function."""
        program = Program()
        weight = Value(name="weight")
        program.global_values.append(weight)
        func = Function(name="main")
        program.add_function(func)
        block = func.new_block("entry")
        result = Value(name="result")
        block.add(Instruction(
            opcode=OpCode.ADD,
            dest=result,
            operands=[weight, weight],
        ))
        block.add(Instruction(opcode=OpCode.RETURN, operands=[result]))

        errors = IRVerifier(program).verify()

        assert [e for e in errors if e.rule == "def-before-use"] == []

    # ------------------------------------------------------------------
    # Block termination
    # ------------------------------------------------------------------

    def test_block_termination_missing(self) -> None:
        """Block without terminator should error."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        block = func.new_block("entry")
        # No terminator
        block.add(Instruction(
            opcode=OpCode.ADD,
            dest=Value(name="c"),
            operands=[Value(name="a", is_constant=True, const_value=1.0),
                      Value(name="b", is_constant=True, const_value=2.0)],
        ))

        verifier = IRVerifier(program)
        errors = verifier.verify()
        term_errors = [e for e in errors if e.rule == "block-termination"]
        assert len(term_errors) >= 1

    def test_empty_block_warning(self) -> None:
        """Empty block should be a warning."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        func.new_block("entry")  # empty block, no instructions

        verifier = IRVerifier(program)
        errors = verifier.verify()
        empty_errors = [e for e in errors if e.rule == "block-termination"]
        assert len(empty_errors) >= 1

    # ------------------------------------------------------------------
    # Label existence
    # ------------------------------------------------------------------

    def test_label_existence(self) -> None:
        """Branch to nonexistent label should error."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        block = func.new_block("entry")

        # Branch to non-existent label
        block.add(Instruction(
            opcode=OpCode.BR,
            target="nonexistent_label",
        ))

        verifier = IRVerifier(program)
        errors = verifier.verify()
        label_errors = [e for e in errors if e.rule == "label-existence"]
        assert len(label_errors) >= 1

    def test_branch_requires_a_target(self) -> None:
        """A branch without a target is not a valid label reference."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        block = func.new_block("entry")
        block.add(Instruction(opcode=OpCode.BR))

        errors = IRVerifier(program).verify()
        label_errors = [e for e in errors if e.rule == "label-existence"]

        assert len(label_errors) == 1

    def test_label_check_ignores_non_branch_target_metadata(self) -> None:
        """Only control-flow instructions carry block label targets."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        block = func.new_block("entry")
        one = Value(name="one", is_constant=True, const_value=1.0)
        block.add(Instruction(
            opcode=OpCode.ADD,
            dest=Value(name="result"),
            operands=[one, one],
            target="not-a-block",
        ))
        block.add(Instruction(opcode=OpCode.RETURN))

        errors = IRVerifier(program).verify()

        assert [e for e in errors if e.rule == "label-existence"] == []

    def test_duplicate_block_labels_are_rejected(self) -> None:
        """A branch label must identify exactly one basic block."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        first = BasicBlock("duplicate")
        second = BasicBlock("duplicate")
        first.add(Instruction(opcode=OpCode.RETURN))
        second.add(Instruction(opcode=OpCode.RETURN))
        func.blocks.extend([first, second])

        errors = IRVerifier(program).verify()
        label_errors = [e for e in errors if e.rule == "label-existence"]

        assert len(label_errors) == 1

    # ------------------------------------------------------------------
    # Type consistency
    # ------------------------------------------------------------------

    def test_type_consistency_warning(self) -> None:
        """Operands with different types should be a warning."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        block = func.new_block("entry")

        v_a = Value(
            name="a",
            dtype=DataType.FLOAT32,
            is_constant=True,
            const_value=1.0)
        v_b = Value(
            name="b",
            dtype=DataType.INT32,
            is_constant=True,
            const_value=2)
        block.add(Instruction(
            opcode=OpCode.ADD,
            dest=Value(name="c"),
            operands=[v_a, v_b],
        ))
        block.add(Instruction(opcode=OpCode.RETURN))

        verifier = IRVerifier(program)
        errors = verifier.verify()
        type_errors = [e for e in errors if e.rule == "type-consistency"]
        # WARNING, not ERROR
        assert all(e.level == ErrorLevel.WARNING for e in type_errors)

    def test_nn_op_checks_every_operand_type(self) -> None:
        """A mismatched third NN operand also produces a warning."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        block = func.new_block("entry")
        inputs = [
            Value(name="x", is_constant=True, const_value=1.0),
            Value(name="w", is_constant=True, const_value=2.0),
            Value(
                name="bias", dtype=DataType.INT32,
                is_constant=True, const_value=3,
            ),
        ]
        block.add(Instruction(
            opcode=OpCode.CONV,
            dest=Value(name="result"),
            operands=inputs,
        ))
        block.add(Instruction(opcode=OpCode.RETURN))

        errors = IRVerifier(program).verify()
        type_errors = [e for e in errors if e.rule == "type-consistency"]

        assert len(type_errors) == 1
        assert type_errors[0].level == ErrorLevel.WARNING

    # ------------------------------------------------------------------
    # Control flow integrity
    # ------------------------------------------------------------------

    def test_control_flow_unreachable_after_br(self) -> None:
        """Instructions after unconditional branch should error."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        block = func.new_block("entry")

        block2 = func.new_block("target")
        block2.add(Instruction(opcode=OpCode.RETURN))

        # BR followed by another instruction
        block.add(Instruction(opcode=OpCode.BR, target="target"))
        block.add(Instruction(
            opcode=OpCode.ADD,
            dest=Value(name="c"),
            operands=[Value(name="a", is_constant=True, const_value=1.0),
                      Value(name="b", is_constant=True, const_value=2.0)],
        ))

        verifier = IRVerifier(program)
        errors = verifier.verify()
        cf_errors = [e for e in errors if e.rule == "control-flow-integrity"]
        assert len(cf_errors) >= 1

    def test_br_if_target_count(self) -> None:
        """BR_IF must have exactly 2 targets."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        block = func.new_block("entry")

        # Only one target
        block.add(Instruction(
            opcode=OpCode.BR_IF,
            operands=[Value(name="cond", is_constant=True, const_value=1.0)],
            target="only_one",
        ))

        verifier = IRVerifier(program)
        errors = verifier.verify()
        cf_errors = [e for e in errors if e.rule == "control-flow-integrity"]
        assert len(cf_errors) >= 1

    def test_control_flow_unreachable_after_br_if(self) -> None:
        """A conditional branch is also a block terminator."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        entry = func.new_block("entry")
        true_block = func.new_block("true")
        false_block = func.new_block("false")
        cond = Value(name="cond", is_constant=True, const_value=1)
        entry.add(Instruction(
            opcode=OpCode.BR_IF,
            operands=[cond],
            target="true,false",
        ))
        entry.add(Instruction(opcode=OpCode.RETURN))
        true_block.add(Instruction(opcode=OpCode.RETURN))
        false_block.add(Instruction(opcode=OpCode.RETURN))

        errors = IRVerifier(program).verify()
        cf_errors = [e for e in errors if e.rule == "control-flow-integrity"]

        assert len(cf_errors) == 1

    def test_br_if_rejects_an_empty_target(self) -> None:
        """Two comma slots are invalid when either label is empty."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        entry = func.new_block("entry")
        target = func.new_block("target")
        cond = Value(name="cond", is_constant=True, const_value=1)
        entry.add(Instruction(
            opcode=OpCode.BR_IF,
            operands=[cond],
            target="target,",
        ))
        target.add(Instruction(opcode=OpCode.RETURN))

        errors = IRVerifier(program).verify()

        assert [e for e in errors if e.rule == "label-existence"]
        assert [e for e in errors if e.rule == "control-flow-integrity"]

    def test_br_if_requires_a_condition_operand(self) -> None:
        """A conditional branch without a condition cannot be selected."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        entry = func.new_block("entry")
        left = func.new_block("left")
        right = func.new_block("right")
        entry.add(Instruction(
            opcode=OpCode.BR_IF,
            target="left,right",
        ))
        left.add(Instruction(opcode=OpCode.RETURN))
        right.add(Instruction(opcode=OpCode.RETURN))

        errors = IRVerifier(program).verify()
        cf_errors = [e for e in errors if e.rule == "control-flow-integrity"]

        assert len(cf_errors) == 1

    # ------------------------------------------------------------------
    # Entry existence
    # ------------------------------------------------------------------

    def test_entry_existence(self) -> None:
        """Function with no blocks should error."""
        program = Program()
        func = Function(name="main")  # no blocks
        program.add_function(func)

        verifier = IRVerifier(program)
        errors = verifier.verify()
        entry_errors = [e for e in errors if e.rule == "entry-existence"]
        assert len(entry_errors) >= 1

    # ------------------------------------------------------------------
    # SSA validity
    # ------------------------------------------------------------------

    def test_ssa_validity(self) -> None:
        """Multiple assignments to same value should error."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        block = func.new_block("entry")

        v_c = Value(name="c")
        block.add(Instruction(
            opcode=OpCode.ADD,
            dest=v_c,
            operands=[Value(name="a", is_constant=True, const_value=1.0),
                      Value(name="b", is_constant=True, const_value=2.0)],
        ))
        # Second assignment to same name
        block.add(Instruction(
            opcode=OpCode.ADD,
            dest=v_c,  # same Value object
            operands=[Value(name="a", is_constant=True, const_value=1.0),
                       Value(name="b", is_constant=True, const_value=2.0)],
        ))
        block.add(Instruction(opcode=OpCode.RETURN, operands=[v_c]))

        verifier = IRVerifier(program)
        errors = verifier.verify()
        ssa_errors = [e for e in errors if e.rule == "ssa-validity"]
        assert len(ssa_errors) >= 1

    def test_ssa_rejects_redefining_a_parameter(self) -> None:
        """Function parameters count as their value's SSA definition."""
        program = Program()
        param = Value(name="x")
        func = Function(name="main", params=[param])
        program.add_function(func)
        block = func.new_block("entry")
        one = Value(name="one", is_constant=True, const_value=1.0)
        block.add(Instruction(
            opcode=OpCode.ADD,
            dest=Value(name="x"),
            operands=[one, one],
        ))
        block.add(Instruction(opcode=OpCode.RETURN))

        errors = IRVerifier(program).verify()
        ssa_errors = [e for e in errors if e.rule == "ssa-validity"]

        assert len(ssa_errors) == 1

    def test_ssa_rejects_duplicate_parameters(self) -> None:
        """Two entry definitions cannot share the same SSA name."""
        program = Program()
        func = Function(
            name="main",
            params=[Value(name="x"), Value(name="x")],
        )
        program.add_function(func)
        block = func.new_block("entry")
        block.add(Instruction(opcode=OpCode.RETURN))

        errors = IRVerifier(program).verify()
        ssa_errors = [e for e in errors if e.rule == "ssa-validity"]

        assert len(ssa_errors) == 1
        assert ssa_errors[0].value_name == "x"

    def test_ssa_rejects_duplicate_globals_without_functions(self) -> None:
        """Program globals are checked even when the program has no function."""
        program = Program()
        program.global_values.extend([
            Value(name="weight"),
            Value(name="weight"),
        ])

        errors = IRVerifier(program).verify()
        ssa_errors = [e for e in errors if e.rule == "ssa-validity"]

        assert len(ssa_errors) == 1
        assert ssa_errors[0].value_name == "weight"

    # ------------------------------------------------------------------
    # Convenience function
    # ------------------------------------------------------------------

    def test_verify_ir_function(self) -> None:
        dsl = """
        c = add(a, b)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        passed, errors = verify_ir(program)
        assert passed is True
        assert len(errors) == 0

    def test_verify_ir_allows_warning_only_program(self) -> None:
        """Warnings are reported but do not fail the convenience API."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        block = func.new_block("entry")
        left = Value(
            name="left", dtype=DataType.FLOAT32,
            is_constant=True, const_value=1.0,
        )
        right = Value(
            name="right", dtype=DataType.INT32,
            is_constant=True, const_value=2,
        )
        block.add(Instruction(
            opcode=OpCode.ADD,
            dest=Value(name="result"),
            operands=[left, right],
        ))
        block.add(Instruction(opcode=OpCode.RETURN))

        passed, errors = verify_ir(program)

        assert passed is True
        assert len(errors) == 1
        assert errors[0].level == ErrorLevel.WARNING


class TestIRVerifierWithExtendedParser:
    """Verify IR generated by the extended parser."""

    def test_if_else_without_phi_is_rejected(self) -> None:
        dsl = """
        if (a > b):
            c = add(a, b)
        else:
            c = mul(a, b)
        endif
        return c
        """
        parser = ExtendedDSLParser()
        program = parser.parse(dsl)
        verifier = IRVerifier(program)
        errors = verifier.verify()
        def_errors = [e for e in errors if e.rule == "def-before-use"]

        assert len(def_errors) == 1
        assert def_errors[0].block_name == "if_end3"

    def test_while_without_phi_is_rejected(self) -> None:
        dsl = """
        while (i < 10):
            acc = add(acc, x)
        endwhile
        return acc
        """
        parser = ExtendedDSLParser()
        program = parser.parse(dsl)
        verifier = IRVerifier(program)
        errors = verifier.verify()
        def_errors = [e for e in errors if e.rule == "def-before-use"]

        assert len(def_errors) == 1
        assert def_errors[0].block_name == "while_exit3"


class TestIRVerifierCompilerPipeline:
    """Exercise IR verification through the public compiler entry point."""

    def test_cli_flag_is_mapped_to_compiler_config(self) -> None:
        parser = build_arg_parser()

        enabled = args_to_config(parser.parse_args([
            "input.dsl", "--verify-ir",
        ]))
        disabled = args_to_config(parser.parse_args(["input.dsl"]))

        assert enabled.verify_ir is True
        assert disabled.verify_ir is False

    def test_error_after_parse_stops_the_pipeline(
        self, tmp_path: Path,
    ) -> None:
        config = CompilerConfig(verify_ir=True, optimize_level="basic")
        driver = StubCompilerDriver(config, make_undefined_program())
        output = tmp_path / "output.s"

        result = driver.compile("input.dsl", str(output))

        assert result.success is False
        assert driver.codegen_called is False
        assert output.exists() is False
        assert "after parsing" in result.errors[0]
        assert "def-before-use" in result.errors[0]

    def test_logger_does_not_enable_ir_verification(
        self, tmp_path: Path,
    ) -> None:
        config = CompilerConfig(use_logger=True, verify_ir=False)
        driver = StubCompilerDriver(config, make_undefined_program())

        result = driver.compile(
            "input.dsl", str(tmp_path / "output.s"),
        )

        assert result.success is True
        assert driver.codegen_called is True

    @pytest.mark.parametrize(
        ("optimize_level", "expected_checks"),
        [("basic", 4), ("all", 7)],
    )
    def test_verifies_after_every_optimization_pass(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        optimize_level: str,
        expected_checks: int,
    ) -> None:
        original_verify = IRVerifier.verify
        checks = []

        def counted_verify(
            verifier: IRVerifier,
        ) -> list[VerificationError]:
            checks.append(verifier.program)
            return original_verify(verifier)

        monkeypatch.setattr(IRVerifier, "verify", counted_verify)
        config = CompilerConfig(
            verify_ir=True,
            optimize_level=optimize_level,
        )
        driver = StubCompilerDriver(config, make_valid_program())

        result = driver.compile(
            "input.dsl", str(tmp_path / "output.s"),
        )

        assert result.success is True
        assert len(checks) == expected_checks

    def test_error_introduced_by_a_pass_stops_immediately(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        from scratchv.optimizer.constant_folding import ConstantFolder
        from scratchv.optimizer.dead_code import DeadCodeEliminator

        dead_code_called = False

        def corrupt_ir(folder: ConstantFolder) -> int:
            return_instr = (
                folder.program.functions[0].blocks[0].instructions[-1]
            )
            return_instr.operands = [Value(name="missing")]
            return 1

        def track_dead_code(_eliminator: DeadCodeEliminator) -> int:
            nonlocal dead_code_called
            dead_code_called = True
            return 0

        monkeypatch.setattr(ConstantFolder, "run", corrupt_ir)
        monkeypatch.setattr(DeadCodeEliminator, "run", track_dead_code)
        config = CompilerConfig(verify_ir=True, optimize_level="basic")
        driver = StubCompilerDriver(config, make_valid_program())

        result = driver.compile(
            "input.dsl", str(tmp_path / "output.s"),
        )

        assert result.success is False
        assert dead_code_called is False
        assert driver.codegen_called is False
        assert "constant-folding" in result.errors[0]
        assert "def-before-use" in result.errors[0]

    def test_verifies_again_before_codegen(self, tmp_path: Path) -> None:
        class CorruptingDriver(StubCompilerDriver):
            def _run_optimizations(self, program: Program) -> PassResult:
                return_instr = program.functions[0].blocks[0].instructions[-1]
                return_instr.operands = [Value(name="missing")]
                return PassResult(data=program)

        config = CompilerConfig(verify_ir=True, optimize_level="basic")
        driver = CorruptingDriver(config, make_valid_program())

        result = driver.compile(
            "input.dsl", str(tmp_path / "output.s"),
        )

        assert result.success is False
        assert driver.codegen_called is False
        assert "before code generation" in result.errors[0]

    def test_warnings_are_reported_without_stopping_codegen(
        self, tmp_path: Path,
    ) -> None:
        config = CompilerConfig(verify_ir=True)
        driver = StubCompilerDriver(config, make_warning_only_program())

        result = driver.compile(
            "input.dsl", str(tmp_path / "output.s"),
        )

        assert result.success is True
        assert driver.codegen_called is True
        assert any(
            "type-consistency" in warning
            for warning in result.warnings
        )

    def test_codegen_failure_preserves_ir_warnings(
        self, tmp_path: Path,
    ) -> None:
        class FailingCodegenDriver(StubCompilerDriver):
            def _generate_code(self, program: Program) -> str:
                raise RuntimeError("boom")

        config = CompilerConfig(verify_ir=True)
        driver = FailingCodegenDriver(config, make_warning_only_program())

        result = driver.compile(
            "input.dsl", str(tmp_path / "output.s"),
        )

        assert result.success is False
        assert "Codegen error: boom" in result.errors
        assert any(
            "type-consistency" in warning
            for warning in result.warnings
        )


class TestIRVerifierFrontendInputs:
    """Verify that frontends declare external IR values explicitly."""

    def test_onnx_tensor_initializer_is_a_program_global(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tensor_type = SimpleNamespace(
            elem_type=1,
            shape=SimpleNamespace(dim=[]),
        )
        graph = SimpleNamespace(
            name="main",
            initializer=[SimpleNamespace(
                name="weight",
                data_type=1,
                array=SimpleNamespace(size=2, shape=(2,)),
            )],
            input=[SimpleNamespace(
                name="x",
                type=SimpleNamespace(tensor_type=tensor_type),
            )],
            output=[SimpleNamespace(name="y")],
            node=[SimpleNamespace(
                op_type="Add",
                input=["x", "weight"],
                output=["y"],
            )],
        )
        fake_onnx = SimpleNamespace(
            load=lambda _path: SimpleNamespace(graph=graph),
            numpy_helper=SimpleNamespace(
                to_array=lambda initializer: initializer.array,
            ),
        )
        monkeypatch.setitem(sys.modules, "onnx", fake_onnx)

        from scratchv.frontend.onnx_parser import ONNXParser
        program = ONNXParser().parse("model.onnx")
        passed, errors = verify_ir(program)

        assert [value.name for value in program.global_values] == ["weight"]
        assert passed is True
        assert errors == []
