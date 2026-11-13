"""Use the production opt-in LM head implementation in the isolated W6 probe."""

from scratchv.backend.tensor_c_codegen import TensorCCodegen


def generate_vocab_blocked(program, bindings):
    return TensorCCodegen(
        program, bindings, constant_storage="external", kernel_calls=True,
        max_constant_bytes=2**31, max_workspace_bytes=2**31, matmul_policy="blocked_fma",
        matmul_tile="2x4", matmul_rhs_layout="original", qemu_workspace_layout="none",
        lm_head_policy="vocab-blocked",
    ).generate()
