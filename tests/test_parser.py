"""Tests for the DSL parser (no ONNX dependency needed)."""

from scratchv.frontend.dsl_parser import DSLParser
from scratchv.ir.types import OpCode


class TestDSLParser:
    def test_parse_simple_add(self):
        dsl = """
        c = add(a, b)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        assert len(program.functions) == 1
        func = program.functions[0]
        assert len(func.blocks) == 1
        block = func.blocks[0]
        assert len(block.instructions) == 2
        add_instr = block.instructions[0]
        assert add_instr.opcode == OpCode.ADD
        ret_instr = block.instructions[1]
        assert ret_instr.opcode == OpCode.RETURN

    def test_parse_relu(self):
        dsl = """
        y = relu(x)
        return y
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        func = program.functions[0]
        block = func.blocks[0]
        assert block.instructions[0].opcode == OpCode.RELU

    def test_parse_matmul(self):
        dsl = """
        c = matmul(A, B, m:2, n:2, k:2)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        block = program.functions[0].blocks[0]
        mm = block.instructions[0]
        assert mm.opcode == OpCode.MATMUL
        assert mm.attrs["m"] == 2
        assert mm.attrs["n"] == 2
        assert mm.attrs["k"] == 2

    def test_parse_dot(self):
        dsl = """
        d = dot(a, b, len:4)
        return d
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        block = program.functions[0].blocks[0]
        dot = block.instructions[0]
        assert dot.opcode == OpCode.DOT
        assert dot.attrs["length"] == 4

    def test_parse_multi_op(self):
        dsl = """
        t1 = add(x, y)
        t2 = mul(t1, z)
        t3 = relu(t2)
        return t3
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        block = program.functions[0].blocks[0]
        assert len(block.instructions) == 4  # add, mul, relu, ret

    def test_parse_gelu(self):
        dsl = """
        y = gelu(x)
        return y
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        block = program.functions[0].blocks[0]
        assert block.instructions[0].opcode == OpCode.GELU

    def test_parse_constants(self):
        dsl = """
        c = mul(a, 2.0)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        block = program.functions[0].blocks[0]
        # First instruction is load_const(2.0), second is mul
        mul_instr = block.instructions[1]
        assert mul_instr.opcode == OpCode.MUL
        # Second operand should be a load_const
        assert len(mul_instr.operands) == 2
