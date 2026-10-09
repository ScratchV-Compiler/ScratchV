"""Legacy scalar entry points must not discard preserved GEMM coefficients."""

import pytest

from scratchv.backend.instruction_select import InstructionSelector
from scratchv.backend.llvm_codegen import LLVMCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value


@pytest.mark.parametrize("backend", ["riscv", "llvm"])
@pytest.mark.parametrize("scales", [{"alpha": 2.0}, {"beta": 0.0}, {"alpha": -0.5, "beta": 3.0}])
def test_direct_legacy_codegen_rejects_nondefault_gemm_scaling(backend, scales):
    # The normal CompilerDriver already refuses tensor buffers for these
    # backends. Exercise the public direct APIs too, without that protection.
    a = Value("a", shape=(2, 3))
    b = Value("b", shape=(3, 2))
    bias = Value("bias", shape=(2,))
    builder = IRBuilder()
    builder.new_function("main", [a, b, bias])
    builder.new_block("entry")
    builder.ret(builder.gemm(a, b, bias, **scales))
    with pytest.raises(ValueError, match="GEMM.*alpha/beta"):
        if backend == "riscv":
            InstructionSelector(builder.program).run()
        else:
            LLVMCodegen(builder.program).emit()
