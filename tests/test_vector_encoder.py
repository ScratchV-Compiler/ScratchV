"""Tests for the RV32IM encoder's vector-mnemonic rejection guard (Topic 29).

Phase 1 never emits vector instructions, but the guard converts a silent
mis-encode into an explicit failure should one ever leak through.
"""

import pytest

from scratchv.backend.riscv_encoder import (
    VectorEncodingError,
    assemble_to_binary,
)


@pytest.mark.parametrize("text", [
    "vadd.vv v1, v2, v3",
    "vsetvli t0, a0, e32, m1, ta, ma",
    "vsetivli t0, 4, e32, m1, ta, ma",
    "vle32.v v1, (a0)",
    "vse32.v v1, (a0)",
    "vmv.v.x v1, a0",
    "vsub.vv v1, v2, v3",
    "vmul.vv v1, v2, v3",
    "vdiv.vv v1, v2, v3",
    "vmax.vx v1, v2, x0",
    "vfmv.v.f v1, f0",
    "vredsum.vs v1, v2, v3",
])
def test_vector_mnemonic_rejected(text):
    with pytest.raises(VectorEncodingError) as exc:
        assemble_to_binary(text)
    message = str(exc.value)
    assert "phase 1 targets RV32IM only" in message
    assert text.split()[0] in message


def test_scalar_still_encodes():
    assert len(assemble_to_binary("add a0, a1, a2\n")) == 4


def test_scalar_program_still_encodes():
    binary = assemble_to_binary(
        "li t0, 4\n"
        "mul t1, t0, t0\n"
        "addi a1, t0, 0\n"
        "lw t2, a1(0)\n"
        "sw a1(0), t2\n"
        "jalr zero, ra\n"
    )
    assert len(binary) == 6 * 4
