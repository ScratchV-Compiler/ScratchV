"""Tests for the extended DSL parser with if/else and while support."""

from scratchv.frontend.dsl_extended import ExtendedDSLParser, CondExpr
from scratchv.frontend.dsl_parser import DSLParseError
from scratchv.ir.types import OpCode


class TestCondExpr:
    """Tests for the conditional expression parser."""

    def test_parse_condition_gt(self):
        parser = ExtendedDSLParser()
        cond = parser._parse_condition("if (a > b):")
        assert cond is not None
        assert cond.lhs == "a"
        assert cond.op == ">"
        assert cond.rhs == "b"

    def test_parse_condition_lt(self):
        parser = ExtendedDSLParser()
        cond = parser._parse_condition("if (i < 10):")
        assert cond is not None
        assert cond.op == "<"
        assert cond.rhs == "10"

    def test_parse_condition_eq(self):
        parser = ExtendedDSLParser()
        cond = parser._parse_condition("if (x == 0):")
        assert cond is not None
        assert cond.op == "=="

    def test_parse_condition_ne(self):
        parser = ExtendedDSLParser()
        cond = parser._parse_condition("if (x != y):")
        assert cond is not None
        assert cond.op == "!="

    def test_parse_condition_le(self):
        parser = ExtendedDSLParser()
        cond = parser._parse_condition("if (x <= 5):")
        assert cond is not None
        assert cond.op == "<="

    def test_parse_condition_ge(self):
        parser = ExtendedDSLParser()
        cond = parser._parse_condition("if (x >= 10):")
        assert cond is not None
        assert cond.op == ">="

    def test_while_condition(self):
        parser = ExtendedDSLParser()
        cond = parser._parse_condition("while (i < 10):")
        assert cond is not None
        assert cond.op == "<"

    def test_invalid_condition(self):
        parser = ExtendedDSLParser()
        cond = parser._parse_condition("if a > b:")  # missing parens
        assert cond is None

    def test_cond_repr(self):
        cond = CondExpr("a", ">", "b")
        assert "a" in repr(cond)
        assert ">" in repr(cond)


class TestExtendedDSLParser:
    """Tests for the ExtendedDSLParser with control flow constructs."""

    def test_if_simple_true_branch(self):
        dsl = """
        if (a > b):
            c = add(a, b)
        endif
        return c
        """
        parser = ExtendedDSLParser()
        program = parser.parse(dsl)
        assert len(program.functions) == 1
        func = program.functions[0]
        assert len(func.blocks) >= 3  # entry + blocks

    def test_if_else_branch(self):
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
        func = program.functions[0]
        # Should have entry, then-block, else-block, merge-block
        assert len(func.blocks) >= 4

    def test_if_with_else_colon(self):
        """Test that 'else:' (with colon) is handled."""
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
        assert program.functions[0] is not None

    def test_while_loop(self):
        dsl = """
        while (i < 10):
            acc = add(acc, x)
        endwhile
        return acc
        """
        parser = ExtendedDSLParser()
        program = parser.parse(dsl)
        func = program.functions[0]
        assert len(func.blocks) >= 3  # entry + header + body + exit

    def test_while_with_multiple_ops(self):
        dsl = """
        while (i < 5):
            t1 = mul(x, y)
            acc = add(acc, t1)
        endwhile
        return acc
        """
        parser = ExtendedDSLParser()
        program = parser.parse(dsl)
        # Should parse without error
        assert len(program.functions) == 1

    def test_nested_if(self):
        dsl = """
        if (a > 0):
            if (b > 0):
                c = add(a, b)
            else:
                c = mul(a, b)
            endif
        else:
            c = sub(a, b)
        endif
        return c
        """
        parser = ExtendedDSLParser()
        program = parser.parse(dsl)
        func = program.functions[0]
        assert len(func.blocks) >= 5

    def test_if_inside_while(self):
        dsl = """
        while (i < 10):
            if (a > b):
                c = add(a, b)
            endif
            acc = add(acc, c)
        endwhile
        return acc
        """
        parser = ExtendedDSLParser()
        program = parser.parse(dsl)
        func = program.functions[0]
        assert len(func.blocks) >= 5

    def test_while_inside_if(self):
        dsl = """
        if (a > 0):
            while (i < 10):
                acc = add(acc, x)
            endwhile
        endif
        return acc
        """
        parser = ExtendedDSLParser()
        program = parser.parse(dsl)
        func = program.functions[0]
        assert len(func.blocks) >= 4

    def test_mixed_if_while_for(self):
        dsl = """
        for i = 0, 4
            if (i > 1):
                c = add(a, b)
            else:
                c = mul(a, b)
            endif
        endfor
        return c
        """
        parser = ExtendedDSLParser()
        program = parser.parse(dsl)
        # Should parse without error -- combines for, if/else
        assert program is not None

    def test_while_label_uniqueness(self):
        """Ensure labels don't collide in nested whiles."""
        dsl = """
        while (i < 3):
            while (j < 2):
                t = mul(x, y)
            endwhile
        endwhile
        return t
        """
        parser = ExtendedDSLParser()
        program = parser.parse(dsl)
        # Check that block names are unique
        func = program.functions[0]
        block_names = [b.name for b in func.blocks]
        assert len(block_names) == len(set(block_names))

    def test_standalone_add_still_works(self):
        """Base parser functionality should still work."""
        dsl = """
        c = add(a, b)
        return c
        """
        parser = ExtendedDSLParser()
        program = parser.parse(dsl)
        block = program.functions[0].blocks[0]
        assert block.instructions[0].opcode == OpCode.ADD

    def test_invalid_if_missing_parens(self):
        """Invalid if condition should raise an error."""
        dsl = """
        if a > b:
            c = add(a, b)
        endif
        return c
        """
        parser = ExtendedDSLParser()
        try:
            parser.parse(dsl)
            assert False, "Should have raised DSLParseError"
        except DSLParseError:
            pass

    def test_empty_program(self):
        parser = ExtendedDSLParser()
        program = parser.parse("")
        assert len(program.functions) == 1
        assert len(program.functions[0].blocks) == 1

    def test_if_without_else_block(self):
        dsl = """
        if (a > b):
            c = add(a, b)
        endif
        return c
        """
        parser = ExtendedDSLParser()
        program = parser.parse(dsl)
        # Should create then-block and else-block (empty jump)
        func = program.functions[0]
        # entry, then, else, merge = 4 blocks
        assert len(func.blocks) >= 4

    def test_condition_with_numeric_literal(self):
        dsl = """
        if (x == 0):
            y = add(x, 1.0)
        endif
        return y
        """
        parser = ExtendedDSLParser()
        program = parser.parse(dsl)
        func = program.functions[0]
        assert len(func.blocks) >= 4
