"""Tests for the TinyFive simulator adapter."""

from unittest.mock import patch

from scratchv.simulator.tinyfive import StubProfiledMachine, verify_assembly


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

        self.m.run(instructions=0)
        assert self.m.instr_count == 0

    def test_empty_asm(self):
        self.m.load_asm([])
        self.m.run()
        assert self.m.instr_count == 0

    def test_asm_count_ignores_comments_labels_and_blank_lines(self):
        self.m.load_asm(["entry:", "", "# note", "addi 10, 0, 42 # value"])
        self.m.run()
        assert self.m.instr_count == 1
        assert self.m.pc == 0x200

    def test_register_access(self):
        self.m.regs[10] = 42
        assert self.m.get_reg(10) == 42
        self.m.set_reg(-1, 99)
        assert self.m.get_reg(-1) == 0
        assert self.m.regs[-1] == 0

    def test_memory_access(self):
        self.m.write_mem_i32(100, 42)
        assert self.m.read_mem_i32(100) == 42
        assert self.m.read_mem_i32(200) == 0


    def test_memory_access_preserves_full_signed_i32(self):
        self.m.write_mem_i32(100, 0x12345678)
        self.m.write_mem_i32(104, -2)
        assert self.m.read_mem_i32(100) == 0x12345678
        assert self.m.read_mem_i32(104) == -2


class TestVerifyAssembly:
    def test_verify_without_tinyfive(self):
        """Should return error result when tinyfive is not installed."""
        with patch("scratchv.simulator.tinyfive.ProfiledMachine") as mock_m:
            mock_m.return_value.available = False
            result = verify_assembly("addi x10, x0, 42")
        assert "success" in result
        assert "instr_count" in result

    def test_empty_assembly(self):
        with patch("scratchv.simulator.tinyfive.ProfiledMachine") as mock_m:
            mock_m.return_value.available = False
            result = verify_assembly("")
        # Should handle empty input gracefully
        assert "success" in result
