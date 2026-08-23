"""Tests for the TinyFive simulator adapter."""

from unittest.mock import patch

import pytest

from scratchv.backend.riscv_encoder import assemble_to_binary
from scratchv.simulator.tinyfive import (
    ProfiledMachine,
    StubProfiledMachine,
    verify_assembly,
)


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
        self.m.load_asm([
            ".text",
            "entry:",
            "",
            "# note",
            "entry2: addi 10, 0, 42 # value",
            '.ascii "not an instruction"',
        ])
        self.m.run()
        assert self.m.instr_count == 1
        assert self.m.pc == 0x200

    def test_register_access(self):
        self.m.regs[10] = 42
        assert self.m.get_reg(10) == 42
        self.m.set_reg(0, 99)
        assert self.m.get_reg(0) == 0
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


class TestRealProfiledMachine:
    def setup_method(self):
        pytest.importorskip("tinyfive")

    def test_executes_all_bytes_of_encoded_instruction_words(self):
        binary = assemble_to_binary("lui x5, 1\naddi x5, x5, 2\n")
        words = [
            int.from_bytes(binary[i:i + 4], "little")
            for i in range(0, len(binary), 4)
        ]
        machine = ProfiledMachine(mem_size=4096)
        assert machine.available
        machine.load_binary(words, origin=0)
        machine.run(instructions=len(words), start=0, strict=True)
        assert machine.get_reg(5) == 4098
        assert machine.instr_count == 2
        assert machine.last_error is None

    def test_large_li_expands_and_simulates_equivalently(self):
        before = assemble_to_binary("lui x5, 1\naddi x5, x5, 2\n")
        after = assemble_to_binary("li x5, 4098\n")
        assert len(before) == 8
        assert len(after) == 8

        outputs = []
        for binary in (before, after):
            words = [
                int.from_bytes(binary[i:i + 4], "little")
                for i in range(0, len(binary), 4)
            ]
            machine = ProfiledMachine(mem_size=4096)
            machine.load_binary(words, origin=0)
            machine.run(instructions=len(words), start=0, strict=True)
            outputs.append(machine.get_reg(5))
        assert outputs == [4098, 4098]

    @pytest.mark.parametrize("value", [
        -(1 << 31), -4097, -2049, -2048, -1,
        0, 2047, 2048, 4096, 4098, (1 << 31) - 1,
    ])
    def test_li_covers_signed_rv32_range(self, value):
        binary = assemble_to_binary(f"li x5, {value}\n")
        words = [
            int.from_bytes(binary[i:i + 4], "little")
            for i in range(0, len(binary), 4)
        ]
        machine = ProfiledMachine(mem_size=4096)
        machine.load_binary(words, origin=0)
        machine.run(instructions=len(words), start=0, strict=True)
        assert machine.get_reg(5) == value
