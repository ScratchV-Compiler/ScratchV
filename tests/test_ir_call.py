"""Tests for OpCode.CALL and IRBuilder.call (Topic 15)."""

from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import DataType, OpCode


class TestCallOpcode:
    def test_opcode_value(self):
        assert OpCode.CALL.value == "call"

    def test_is_call(self):
        assert OpCode.CALL.is_call() is True
        assert OpCode.ADD.is_call() is False
        assert OpCode.RETURN.is_call() is False

    def test_call_is_not_control_flow(self):
        assert OpCode.CALL.is_control_flow() is False
        assert OpCode.RETURN.is_control_flow() is True


class TestBuilderCall:
    def test_call_layout(self):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        x = b.make_value(name="x")
        y = b.make_value(name="y")
        r = b.call("inc", [x, y])

        instr = b.current_block.instructions[-1]
        assert instr.opcode is OpCode.CALL
        assert instr.dest is r
        assert instr.operands == [x, y]
        assert instr.target == "inc"
        assert instr.attrs["argc"] == 2
        assert instr.attrs["is_tail"] is False

    def test_call_no_ret(self):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        x = b.make_value(name="x")
        r = b.call("report", [x], has_ret=False)

        assert r is None
        instr = b.current_block.instructions[-1]
        assert instr.opcode is OpCode.CALL
        assert instr.dest is None
        assert instr.attrs["argc"] == 1

    def test_call_dtype(self):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        r = b.call("f", dtype=DataType.INT32)

        assert r is not None
        assert r.dtype is DataType.INT32
        assert b.current_block.instructions[-1].attrs["argc"] == 0

    def test_call_is_tail_attr(self):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        b.call("f", is_tail=True)
        assert b.current_block.instructions[-1].attrs["is_tail"] is True

    def test_call_dump(self):
        b = IRBuilder()
        b.new_function("main")
        b.new_block("entry")
        x = b.make_value(name="x")
        y = b.make_value(name="y")
        r = b.call("inc", [x, y])
        b.ret(r)

        dump = b.program.dump()
        expected = f"${r.name} = call $x $y -> inc [argc=2] [is_tail=False]"
        assert expected in dump
