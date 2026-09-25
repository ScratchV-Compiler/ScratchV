"""Tests for the IR verifier module."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from scratchv.frontend.dsl_parser import DSLParser
from scratchv.frontend.dsl_extended import ExtendedDSLParser
from scratchv.analysis.ir_verifier import (
    IRVerifier, VerificationError, ErrorLevel, verify_ir,
)
from scratchv.ir.types import (
    BasicBlock, Program, Function, Instruction, OpCode, Value, DataType,
)


def _program(*functions: Function) -> Program:
    program = Program()
    program.functions.extend(functions)
    return program


def _function(*blocks: BasicBlock, name: str = "main",
              params: list[Value] | None = None) -> Function:
    return Function(name=name, params=list(params or ()), blocks=list(blocks))


def _block(name: str, *instructions: Instruction) -> BasicBlock:
    block = BasicBlock(name)
    block.instructions.extend(instructions)
    return block


def _rules(issues: list[VerificationError]) -> set[str | None]:
    return {issue.rule for issue in issues}


class TestVerificationError:
    """Tests for VerificationError dataclass."""

    def test_create_error(self):
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

    def test_create_warning(self):
        err = VerificationError(
            level=ErrorLevel.WARNING,
            message="type mismatch",
            rule="type-consistency",
        )
        assert err.level == ErrorLevel.WARNING

    def test_str_representation(self):
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

    def test_valid_simple_program(self):
        dsl = """
        c = add(a, b)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        verifier = IRVerifier(program)
        errors = verifier.verify()
        assert len(errors) == 0  # Should be valid

    def test_valid_nn_pipeline(self):
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

    def test_def_before_use_implicit_input(self):
        """Value without definition is treated as implicit input."""
        program = Program()
        func = Function(name="main")
        program.add_function(func)
        block = func.new_block("entry")

        v_x = Value(name="x", dtype=DataType.FLOAT32)
        # Use x without defining it -- treated as implicit input by verifier
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
        # Verifier treats undefined values as implicit inputs (not errors)
        # Only block-termination check applies here
        def_errors = [e for e in errors if e.rule == "def-before-use"]
        assert len(def_errors) == 0  # implicit inputs are allowed

    # ------------------------------------------------------------------
    # Block termination
    # ------------------------------------------------------------------

    def test_block_termination_missing(self):
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

    def test_empty_block_warning(self):
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

    def test_label_existence(self):
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

    # ------------------------------------------------------------------
    # Type consistency
    # ------------------------------------------------------------------

    def test_type_consistency_warning(self):
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

    # ------------------------------------------------------------------
    # Control flow integrity
    # ------------------------------------------------------------------

    def test_control_flow_unreachable_after_br(self):
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

    def test_br_if_target_count(self):
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

    # ------------------------------------------------------------------
    # Entry existence
    # ------------------------------------------------------------------

    def test_entry_existence(self):
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

    def test_ssa_validity(self):
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

    # ------------------------------------------------------------------
    # Convenience function
    # ------------------------------------------------------------------

    def test_verify_ir_function(self):
        dsl = """
        c = add(a, b)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        passed, errors = verify_ir(program)
        assert passed is True
        assert len(errors) == 0


class TestIRVerifierWithExtendedParser:
    """Verify IR generated by the extended parser."""

    def test_if_else_ir_valid(self):
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
        # The extended parser should generate valid IR
        real_errors = [e for e in errors if e.level == ErrorLevel.ERROR]
        assert len(real_errors) == 0, f"IR verification failed: {real_errors}"

    def test_while_ir_valid(self):
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
        real_errors = [e for e in errors if e.level == ErrorLevel.ERROR]
        assert len(real_errors) == 0, f"IR verification failed: {real_errors}"


class TestIRVerifierHardening:
    """Regression tests for structural, CFG, and pipeline verification."""

    def test_verify_program_and_function_are_public_entry_points(self):
        program = _program(_function(
            _block("entry", Instruction(opcode=OpCode.RETURN))))
        verifier = IRVerifier(program)

        assert verifier.verify_program() == []
        assert verifier.verify_function(program.functions[0]) == []

    def test_use_before_later_definition_is_rejected(self):
        value = Value("later")
        program = _program(_function(_block(
            "entry",
            Instruction(opcode=OpCode.RETURN, operands=[value]),
            Instruction(opcode=OpCode.ADD, dest=value, operands=[
                Value("one", is_constant=True, const_value=1),
                Value("two", is_constant=True, const_value=2),
            ]),
        )))

        assert "def-before-use" in _rules(IRVerifier(program).verify())

    def test_branch_local_definition_does_not_dominate_merge(self):
        cond = Value("cond")
        branch_value = Value("branch_value")
        program = _program(_function(
            _block("entry", Instruction(
                opcode=OpCode.BR_IF, operands=[cond], target="yes,no")),
            _block(
                "yes",
                Instruction(
                    opcode=OpCode.ADD,
                    dest=branch_value,
                    operands=[
                        Value("one", is_constant=True, const_value=1),
                        Value("two", is_constant=True, const_value=2),
                    ],
                ),
                Instruction(opcode=OpCode.BR, target="done"),
            ),
            _block("no", Instruction(opcode=OpCode.BR, target="done")),
            _block("done", Instruction(
                opcode=OpCode.RETURN, operands=[branch_value])),
            params=[cond],
        ))

        assert "def-before-use" in _rules(IRVerifier(program).verify())

    def test_dominating_definition_is_independent_of_block_order(self):
        result = Value("result")
        program = _program(_function(
            _block("entry", Instruction(opcode=OpCode.BR, target="define")),
            _block("use", Instruction(opcode=OpCode.RETURN, operands=[result])),
            _block(
                "define",
                Instruction(
                    opcode=OpCode.ADD,
                    dest=result,
                    operands=[
                        Value("one", is_constant=True, const_value=1),
                        Value("two", is_constant=True, const_value=2),
                    ],
                ),
                Instruction(opcode=OpCode.BR, target="use"),
            ),
        ))

        assert "def-before-use" not in _rules(IRVerifier(program).verify())

    def test_definition_after_terminator_does_not_dominate_successor(self):
        result = Value("result")
        program = _program(_function(
            _block(
                "entry",
                Instruction(opcode=OpCode.BR, target="use"),
                Instruction(
                    opcode=OpCode.ADD,
                    dest=result,
                    operands=[
                        Value("one", is_constant=True, const_value=1),
                        Value("two", is_constant=True, const_value=2),
                    ],
                ),
            ),
            _block("use", Instruction(opcode=OpCode.RETURN, operands=[result])),
        ))

        assert "def-before-use" in _rules(IRVerifier(program).verify())

    def test_duplicate_block_names_are_rejected(self):
        program = _program(_function(
            _block("entry", Instruction(opcode=OpCode.RETURN)),
            _block("entry", Instruction(opcode=OpCode.RETURN)),
        ))

        assert "label-existence" in _rules(IRVerifier(program).verify())

    def test_conditional_branch_requires_a_condition_operand(self):
        program = _program(_function(
            _block("entry", Instruction(
                opcode=OpCode.BR_IF,
                target="yes,no",
            )),
            _block("yes", Instruction(opcode=OpCode.RETURN)),
            _block("no", Instruction(opcode=OpCode.RETURN)),
        ))

        assert "control-flow-integrity" in _rules(
            IRVerifier(program).verify()
        )

    def test_binary_operation_requires_exactly_two_operands(self):
        program = _program(_function(_block(
            "entry",
            Instruction(
                opcode=OpCode.ADD,
                dest=Value("result"),
                operands=[Value("one", is_constant=True, const_value=1)],
            ),
            Instruction(opcode=OpCode.RETURN),
        )))

        assert "type-consistency" in _rules(IRVerifier(program).verify())

    def test_convolution_accepts_input_weights_and_bias(self):
        operands = [Value("input"), Value("weights"), Value("bias")]
        program = _program(_function(_block(
            "entry",
            Instruction(
                opcode=OpCode.CONV,
                dest=Value("output"),
                operands=operands,
            ),
            Instruction(opcode=OpCode.RETURN),
        )))

        passed, issues = verify_ir(program)

        assert passed, issues

    def test_malformed_nested_ir_reports_structure_error(self):
        function = Function(name="main")
        function.blocks = [42]  # type: ignore[list-item]
        program = _program(function)

        issues = IRVerifier(program).verify()

        assert "program-structure" in _rules(issues)

    def test_operand_missing_constant_flag_reports_structure_error(self):
        class MalformedValue:
            name = "broken"
            dtype = DataType.FLOAT32

        program = _program(_function(_block(
            "entry",
            Instruction(
                opcode=OpCode.RETURN,
                operands=[MalformedValue()],  # type: ignore[list-item]
            ),
        )))

        issues = IRVerifier(program).verify()

        assert "program-structure" in _rules(issues)

    def test_unknown_opcode_reports_structure_error(self):
        program = _program(_function(_block(
            "entry",
            Instruction(opcode="mystery"),  # type: ignore[arg-type]
            Instruction(opcode=OpCode.RETURN),
        )))

        issues = IRVerifier(program).verify()

        assert "program-structure" in _rules(issues)

    def test_instruction_missing_destination_field_reports_structure_error(self):
        class MalformedInstruction:
            opcode = OpCode.RETURN
            operands: list[Value] = []

        block = _block("entry")
        block.instructions = [MalformedInstruction()]  # type: ignore[list-item]
        program = _program(_function(block))

        issues = IRVerifier(program).verify()

        assert "program-structure" in _rules(issues)

    def test_pass_pipeline_stops_before_running_on_invalid_ir(self):
        from scratchv.analysis import ir_verifier

        called: list[str] = []
        program = _program(_function(_block("entry")))

        def should_not_run(current: Program) -> None:
            called.append("ran")

        with __import__("pytest").raises(
            ir_verifier.IRVerificationFailure, match="before pass"
        ):
            ir_verifier.run_optimization_pipeline(
                program, [should_not_run], verify_ir_enabled=True)
        assert called == []

    def test_pass_pipeline_stops_after_pass_corrupts_ir(self):
        from scratchv.analysis import ir_verifier

        called: list[str] = []
        program = _program(_function(_block(
            "entry", Instruction(opcode=OpCode.RETURN))))

        def corrupt(current: Program) -> None:
            called.append("corrupt")
            current.functions[0].blocks[0].instructions.clear()

        def next_pass(current: Program) -> None:
            called.append("next")

        with __import__("pytest").raises(
            ir_verifier.IRVerificationFailure, match="after pass #1"
        ):
            ir_verifier.run_optimization_pipeline(
                program, [corrupt, next_pass], verify_ir_enabled=True)
        assert called == ["corrupt"]

    def test_pass_pipeline_supports_legacy_zero_argument_pass(self):
        from scratchv.analysis import ir_verifier

        program = _program(_function(_block(
            "entry", Instruction(opcode=OpCode.RETURN))))

        class LegacyPass:
            name = "legacy"

            def __init__(self) -> None:
                self.called = False

            def run(self) -> int:
                self.called = True
                return 1

        optimization_pass = LegacyPass()

        result = ir_verifier.run_optimization_pipeline(
            program, [optimization_pass], verify_ir_enabled=True
        )

        assert result is program
        assert optimization_pass.called

    def test_pass_pipeline_unwraps_compiler_pass_result(self):
        from scratchv.analysis import ir_verifier
        from scratchv.pass_interface import PassResult

        program = _program(_function(_block(
            "entry", Instruction(opcode=OpCode.RETURN))))

        class ModernPass:
            name = "modern"

            def run(self, current: Program) -> PassResult:
                return PassResult(data=current, changes=0)

        result = ir_verifier.run_optimization_pipeline(
            program, [ModernPass()], verify_ir_enabled=True
        )

        assert result is program

    def test_failed_pass_result_stops_pipeline_with_message(self):
        import pytest

        from scratchv.analysis import ir_verifier
        from scratchv.pass_interface import PassResult

        program = _program(_function(_block(
            "entry", Instruction(opcode=OpCode.RETURN))))
        called: list[str] = []

        class FailedPass:
            name = "failed"

            def run(self, current: Program) -> PassResult:
                return PassResult(data=None, changes=0, message="bad transform")

        def next_pass(current: Program) -> None:
            called.append("next")

        with pytest.raises(RuntimeError, match="bad transform"):
            ir_verifier.run_optimization_pipeline(
                program,
                [FailedPass(), next_pass],
                verify_ir_enabled=False,
            )

        assert called == []

    def test_failure_report_contains_rule_and_ir_location(self):
        from scratchv.analysis import ir_verifier

        missing = Value("missing")
        later = Value("missing")
        program = _program(_function(
            _block(
                "entry",
                Instruction(opcode=OpCode.RETURN, operands=[missing]),
                Instruction(
                    opcode=OpCode.ADD,
                    dest=later,
                    operands=[
                        Value("one", is_constant=True, const_value=1),
                        Value("two", is_constant=True, const_value=2),
                    ],
                ),
            ),
            name="worker",
        ))

        with __import__("pytest").raises(
            ir_verifier.IRVerificationFailure
        ) as raised:
            ir_verifier.verify_or_raise(program, "test phase")

        report = str(raised.value)
        assert "test phase" in report
        assert "def-before-use" in report
        assert "worker" in report
        assert "entry" in report
        assert "instruction #0" in report


class TestIRVerifierCommandLine:
    def run_cli(
        self, payload: object, *options: str
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "program.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return subprocess.run(
                [sys.executable, "-m", "scratchv.analysis.ir_verifier",
                 *options, str(path)],
                text=True,
                capture_output=True,
                check=False,
            )

    def test_valid_ir_returns_zero(self):
        payload = {"functions": [{"name": "main", "blocks": [{
            "name": "entry", "instructions": [{"opcode": "RETURN"}],
        }]}]}

        result = self.run_cli(payload, "--verify-ir")

        assert result.returncode == 0, result.stderr
        assert "passed" in result.stdout

    def test_invalid_ir_returns_one_with_rule(self):
        payload = {"functions": [{"name": "main", "blocks": [{
            "name": "entry", "instructions": [],
        }]}]}

        result = self.run_cli(payload, "--verify-ir")

        assert result.returncode == 1
        assert "block-termination" in result.stderr

    def test_malformed_json_shape_returns_load_error(self):
        result = self.run_cli([], "--verify-ir")

        assert result.returncode == 2
        assert "cannot load IR file" in result.stderr
        assert "Traceback" not in result.stderr
