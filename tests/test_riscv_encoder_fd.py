"""Tests for RV32IM encoder fail-loud on F/D instructions (Topic 28)."""

import pytest

from scratchv.backend.riscv_encoder import (
    UnsupportedInstructionError,
    _is_fd_mnemonic,
    assemble_to_binary,
)


FD_SAMPLES = [
    "fadd.d f0, f1, f2",
    "fsub.s f0, f1, f2",
    "fmul.d f0, f1, f2",
    "fdiv.d f0, f1, f2",
    "fsqrt.d f0, f1",
    "fmin.d f0, f1, f2",
    "fmax.d f0, f1, f2",
    "fabs.d f0, f1",
    "fneg.d f0, f1",
    "fld f0, 0(sp)",
    "fsd f0, 0(sp)",
    "flw f0, 0(sp)",
    "fsw f0, 0(sp)",
    "flt.d a0, f1, f2",
    "feq.d a0, f1, f2",
    "fcvt.s.d f0, f1",
    "fcvt.d.s f0, f1",
    "fmv.x.w a0, f1",
    "li.d f0, 123",
    # Fused / classified / Zfa families (Topic 28 review F6).
    "fmadd.d f0, f1, f2, f3",
    "fnmadd.d f0, f1, f2, f3",
    "fmsub.s f0, f1, f2, f3",
    "fnmsub.s f0, f1, f2, f3",
    "fclass.s a0, f1",
    "fli.s f0, 1",
    "fround.s f0, f1",
]


@pytest.mark.parametrize("asm", FD_SAMPLES)
def test_fd_mnemonics_raise(asm):
    mnemonic = asm.split()[0]
    with pytest.raises(UnsupportedInstructionError) as excinfo:
        assemble_to_binary(asm)
    assert mnemonic in str(excinfo.value)


def test_unsupported_is_value_error():
    assert issubclass(UnsupportedInstructionError, ValueError)


def test_unknown_non_fd_still_value_error():
    with pytest.raises(ValueError) as excinfo:
        assemble_to_binary("frobnicate x1, x2")
    assert not isinstance(excinfo.value, UnsupportedInstructionError)
    assert "Unknown instruction" in str(excinfo.value)


@pytest.mark.parametrize("mnemonic,expected", [
    ("add", False),
    ("fence", False),
    ("frobnicate", False),
    ("flw", True),
    ("fld", True),
    ("li.d", True),
    ("fadd.d", True),
    ("fsqrt.s", True),
    ("fmv.x.w", True),
    ("fsgnjx.d", True),
    ("fmadd.d", True),
    ("fnmadd.d", True),
    ("fmsub.s", True),
    ("fnmsub.s", True),
    ("fclass.s", True),
    ("fli.s", True),
    ("fround.s", True),
])
def test_is_fd_mnemonic_predicate(mnemonic, expected):
    assert _is_fd_mnemonic(mnemonic) is expected


def test_rv32im_regression_still_assembles():
    asm = "\n".join([
        "  addi a0, x0, 5",
        "  add a1, a0, a0",
        "  sw a1, 0(sp)",
        "  li t0, 4098",
        "  beq a0, a1, .Ldone",
        "  nop",
        ".Ldone:",
        "  ret",
    ])
    result = assemble_to_binary(asm)
    assert isinstance(result, bytearray)
    # 8 encoded words: addi, add, sw, lui+addi (li), beq, nop, ret
    assert len(result) == 8 * 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
