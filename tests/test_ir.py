"""Tests for the IR module."""

from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import OpCode


class TestIRBuilder:
    def test_build_simple_add(self):
        builder = IRBuilder()
        func = builder.new_function("test")
        builder.new_block("entry")

        a = builder.make_value(name="a")
        b = builder.make_value(name="b")
        c = builder.add(a, b)
        builder.ret(c)

        assert len(func.blocks) == 1
        assert len(func.blocks[0].instructions) == 2  # add + ret
        add_instr = func.blocks[0].instructions[0]
        assert add_instr.opcode == OpCode.ADD
        assert add_instr.dest is not None
        assert len(add_instr.operands) == 2
        assert add_instr.operands[0].name == "a"
        assert add_instr.operands[1].name == "b"

    def test_build_with_constants(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        c1 = builder.load_const(3.0)
        c2 = builder.load_const(4.0)
        r = builder.add(c1, c2)
        builder.ret(r)

        const_instrs = [i for i in builder.current_block.instructions
                        if i.opcode == OpCode.LOAD_CONST]
        add_instrs = [i for i in builder.current_block.instructions
                      if i.opcode == OpCode.ADD]
        assert len(const_instrs) == 2
        assert len(add_instrs) == 1

    def test_build_relu(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        x = builder.make_value(name="x")
        y = builder.relu(x)
        builder.ret(y)

        relu_instr = builder.current_block.instructions[0]
        assert relu_instr.opcode == OpCode.RELU
        assert relu_instr.operands[0].name == "x"

    def test_for_loop(self):
        builder = IRBuilder()
        builder.new_function("test")
        builder.new_block("entry")

        builder.for_loop(0, 10)
        builder.endfor()
        builder.ret()

        assert len(builder.current_block.instructions) >= 2
        for_instr = builder.current_block.instructions[0]
        assert for_instr.opcode == OpCode.FOR
        assert for_instr.attrs["start"] == 0
        assert for_instr.attrs["end"] == 10

    def test_program_dump(self):
        builder = IRBuilder()
        builder.new_function("main")
        builder.new_block("entry")
        a = builder.load_const(1.0)
        b = builder.load_const(2.0)
        c = builder.add(a, b)
        builder.ret(c)

        dump = builder.program.dump()
        assert "fun $main" in dump
        assert "load_const" in dump
        assert "add" in dump
        assert "return" in dump

    def test_multiple_blocks(self):
        builder = IRBuilder()
        func = builder.new_function("test")
        builder.new_block("entry")
        builder.br("other")
        builder.new_block("other")
        builder.ret()

        assert len(func.blocks) == 2
        assert func.blocks[0].name == "entry"
        assert func.blocks[1].name == "other"
