"""Production stack-frame finalization tests."""

import pytest

from scratchv.backend.abi_frame import apply_abi_frames
from scratchv.backend.riscv_encoder import RISCVAEncoder


def test_frame_rebases_spills_and_preserves_callee_saved_and_ra():
    assembly = """\
main:
  sw t0, -4(sp)  # spill value [regalloc:spill]
  lw t1, -4(sp)  # reload value [regalloc:reload]
  add s0, t0, t1
  call helper
  jalr zero, ra, 0
helper:
  jalr zero, ra, 0
"""

    framed = apply_abi_frames(assembly, spill_slot_count=1)

    assert "addi sp, sp, -16" in framed
    assert "sw ra, 0(sp)  # ABI save" in framed
    assert "sw s0, 4(sp)  # ABI save" in framed
    assert "sw t0, 12(sp)  # spill value" in framed
    assert "lw t1, 12(sp)  # reload value" in framed
    assert "lw ra, 0(sp)  # ABI restore" in framed
    assert "addi sp, sp, 16  # destroy stack frame" in framed
    RISCVAEncoder().assemble(framed)


def test_framed_nested_call_returns_and_restores_callee_saved_register():
    pytest.importorskip("tinyfive")
    from scratchv.simulator.tinyfive import ProfiledMachine

    framed = apply_abi_frames(
        "main:\n"
        "  addi s0, zero, 9\n"
        "  call helper\n"
        "  jalr zero, ra, 0\n"
        "helper:\n"
        "  addi s0, zero, 3\n"
        "  jalr zero, ra, 0\n",
        spill_slot_count=0,
    )
    program = (
        "li sp, 4096\n"
        "li s0, 77\n"
        "jal ra, main\n"
        "mv a0, s0\n"
        "j .done\n"
        + framed
        + "\n.done:\nj .done"
    )
    binary = bytes(RISCVAEncoder().assemble(program))
    words = [
        int.from_bytes(binary[offset:offset + 4], "little")
        for offset in range(0, len(binary), 4)
    ]
    machine = ProfiledMachine(mem_size=8192)
    machine.load_binary(words, origin=0)
    machine.run(instructions=len(words) + 8, start=0, strict=True)

    assert machine.get_reg(10) == 77


def test_oversized_frame_fails_instead_of_emitting_truncated_offsets():
    with pytest.raises(ValueError, match="stack frame exceeds"):
        apply_abi_frames(
            "main:\n  sw t0, -4(sp)  # spill x [regalloc:spill]\n"
            "  jalr zero, ra, 0\n",
            spill_slot_count=509,
        )
