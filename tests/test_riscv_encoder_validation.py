"""Validation tests for immediates that must not be silently truncated."""

import pytest

from scratchv.backend.riscv_encoder import RISCVAEncoder, _b_type, _j_type


@pytest.mark.parametrize(
    "assembly, message",
    [
        ("addi a0, zero, 2048", "I-type immediate out of range"),
        ("lw a0, 2048(sp)", "I-type immediate out of range"),
        ("sw a0, -2049(sp)", "S-type immediate out of range"),
        ("srai a0, a1, 32", "RV32 shift amount out of range"),
    ],
)
def test_out_of_range_immediate_is_rejected(assembly, message):
    with pytest.raises(ValueError, match=message):
        RISCVAEncoder().assemble(assembly)


def test_branch_hex_immediate_uses_the_common_integer_parser():
    assert RISCVAEncoder().assemble("beq a0, 0x10, .done\n.done:\nnop") == (
        RISCVAEncoder().assemble("beq a0, 16, .done\n.done:\nnop")
    )


def test_control_transfer_helpers_reject_unencodable_offsets():
    with pytest.raises(ValueError, match="branch offset out of range"):
        _b_type(10, 11, 4096, 0)
    with pytest.raises(ValueError, match="jump offset out of range"):
        _j_type(1, 1048576)
