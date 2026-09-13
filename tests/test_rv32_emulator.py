"""Regression tests for RV32 integer memory bit-pattern semantics."""

from scratchv.backend.riscv_encoder import RISCVAEncoder
from scratchv.simulator.rv32_emulator import REG_ID, RV32Emulator


def test_write_i32_accepts_any_rv32_bit_pattern() -> None:
    emulator = RV32Emulator(mem_size=4096)

    emulator.write_i32(64, 0xFFFFFFFF)

    assert emulator.mem[64:68] == b"\xff\xff\xff\xff"
    assert emulator.read_i32(64) == -1


def test_sw_lw_round_trip_preserves_high_bit_values() -> None:
    assembly = """\
li sp, 1024
li t0, -1
sw t0, -4(sp)
lw t1, -4(sp)
ret
"""
    binary = bytes(RISCVAEncoder().assemble(assembly))
    emulator = RV32Emulator(mem_size=4096)
    emulator.load_code(binary)

    emulator.run(max_instr=16)

    assert emulator.regs[REG_ID["t1"]] == 0xFFFFFFFF
    assert emulator.mem[1020:1024] == b"\xff\xff\xff\xff"
