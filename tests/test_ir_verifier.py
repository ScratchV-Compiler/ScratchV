"""Tests for the IR verifier module."""

from scratchv.frontend.dsl_parser import DSLParser
from scratchv.frontend.dsl_extended import ExtendedDSLParser
from scratchv.analysis.ir_verifier import (
    IRVerifier, VerificationError, ErrorLevel, verify_ir,
)
from scratchv.ir.types import (
    Program, Function, Instruction, OpCode, Value, DataType,
)


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
