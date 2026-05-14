"""Tests for the TinyFive simulator adapter."""

from scratchv.simulator.tinyfive import ProfiledMachine, StubProfiledMachine, verify_assembly


class TestStubProfiledMachine:
    def setup_method(self):
        self.m = StubProfiledMachine()

    def test_instruction_counting(self):
        asm = [
            "addi 10, 0, 42",
            "addi 11, 0, 7",
            "mul 12, 10, 11",
        ]
        self.m.load_asm(asm)
        self.m.run()
        assert self.m.instr_count == 3

    def test_empty_asm(self):
        self.m.load_asm([])
        self.m.run()
        assert self.m.instr_count == 0

    def test_register_access(self):
        self.m.regs[10] = 42
        assert self.m.get_reg(10) == 42

    def test_memory_access(self):
        self.m.write_mem_i32(100, 42)
        assert self.m.read_mem_i32(100) == 42
        assert self.m.read_mem_i32(200) == 0


class TestVerifyAssembly:
    def test_verify_without_tinyfive(self):
        """Should return error result when tinyfive is not installed."""
        result = verify_assembly("addi x10, x0, 42")
        # On CI without tinyfive, should return error but not crash
        assert "success" in result
        assert "instr_count" in result

    def test_empty_assembly(self):
        result = verify_assembly("")
        # Should handle empty input gracefully
        assert "success" in result
