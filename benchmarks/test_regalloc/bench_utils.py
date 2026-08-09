# flake8: noqa
"""Common used benchmark utils for all benchmarks"""

from tabulate import tabulate


# Common used constants
_CALLEE_SAVED = {
    "ra",
    "fp",
    "s0",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
    "s8",
    "s9",
    "s10",
    "s11",
    "fs0",
    "fs1",
    "fs2",
    "fs3",
    "fs4",
    "fs5",
    "fs6",
    "fs7",
    "fs8",
    "fs9",
    "fs10",
    "fs11",
}

_KNOWN_OPS = {
    "add",
    "addi",
    "sub",
    "mul",
    "div",
    "rem",
    "and",
    "andi",
    "or",
    "ori",
    "xor",
    "xori",
    "sll",
    "slli",
    "srl",
    "srli",
    "sra",
    "srai",
    "slt",
    "slti",
    "sltu",
    "sltiu",
    "lw",
    "lh",
    "lb",
    "lbu",
    "lhu",
    "sw",
    "sh",
    "sb",
    "beq",
    "bne",
    "blt",
    "bge",
    "bltu",
    "bgeu",
    "jal",
    "jalr",
    "auipc",
    "lui",
    "li",
    "mv",
    "nop",
    "ret",
    "bnez",
    "j",
    "max",
    ".label",
    ".text",
    ".data",
    ".global",
    ".type",
    "flw",
    "fsw",
    "fadd.s",
    "fsub.s",
    "fmul.s",
    "fdiv.s",
    "feq.s",
    "flt.s",
    "fle.s",
    "fcvt.w.s",
    "fcvt.s.w",
}

# Instruction category buckets used to compare opcode mixes between the
# ScratchV backend and the LLVM backend. `sd`/`ld` are ABI stack
# save/restore pairs — the classic LLVM frame-management cost.
_CAT_ALU = {
    "add",
    "addi",
    "sub",
    "sll",
    "slli",
    "srl",
    "srli",
    "sra",
    "srai",
    "and",
    "andi",
    "or",
    "ori",
    "xor",
    "xori",
    "slt",
    "slti",
    "sltu",
    "sltiu",
    "lui",
    "auipc",
}
_CAT_LOAD = {"lw", "lh", "lb", "lbu", "lhu", "flw"}
_CAT_STORE = {"sw", "sh", "sb", "fsw"}
_CAT_BRANCH = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "jal", "jalr", "j"}
_CAT_MUL = {"mul", "mulh", "mulhu", "mulhsu"}
_CAT_STACK = {"sd", "ld"}


def _op_categories(cats: dict[str, int]) -> dict[str, int]:
    """Group raw opcode counts into instruction categories."""
    out = {
        "ALU": 0,
        "Load": 0,
        "Store": 0,
        "Branch": 0,
        "Mul": 0,
        "Stack": 0,
        "Other": 0,
    }
    for op, n in cats.items():
        if op in _CAT_STACK:
            out["Stack"] += n
        elif op in _CAT_ALU:
            out["ALU"] += n
        elif op in _CAT_LOAD:
            out["Load"] += n
        elif op in _CAT_STORE:
            out["Store"] += n
        elif op in _CAT_BRANCH:
            out["Branch"] += n
        elif op in _CAT_MUL:
            out["Mul"] += n
        else:
            out["Other"] += n
    return out


# RV64 ABI callee-saved registers — `sd`/`ld` to these at sp offsets are
# prologue/epilogue frame save/restore, not spills.
_CALLEE_SAVED = {
    "ra",
    "fp",
    "s0",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
    "s8",
    "s9",
    "s10",
    "s11",
    "fs0",
    "fs1",
    "fs2",
    "fs3",
    "fs4",
    "fs5",
    "fs6",
    "fs7",
    "fs8",
    "fs9",
    "fs10",
    "fs11",
}


_llvmlite_ready = False


def llvmlite_ir_to_riscv(
    ir_text: str,
    features: str = "",
    opt_level: int = 2,
    output_path: str = "",
) -> tuple[int, str, dict[str, int]]:
    """Compile LLVM IR → RISC-V assembly via llvmlite.

    Drop-in replacement for ``compare_codegen.llvm_ir_to_riscv``, using
    llvmlite's Python bindings instead of ctypes.  No ``lib`` parameter is
    needed — llvmlite manages the LLVM library internally.

    Attempts to load the system ``libLLVM-20.so`` so that codegen output is
    byte-identical to the existing ctypes path.  Falls back to llvmlite's
    bundled LLVM if the system library is not available.

    Args:
        ir_text: LLVM IR source text.
        features: Target features, e.g. ``""`` or ``"+f,+d"``.
        opt_level: 0=None, 1=Less, 2=Default, 3=Aggressive.
        output_path: If set, save assembly to this file.

    Returns:
        ``(instruction_count, assembly_text, opcode_breakdown)``.
    """
    global _llvmlite_ready
    if not _llvmlite_ready:
        from llvmlite import binding as llvm

        # Attempt to load the system libLLVM so that load_library_permanently
        # can resolve any additional LLVM plugins from the system install.
        # Note: llvmlite's codegen functions are pre-linked against its own
        # bundled LLVM, so minor output differences (+2 instrs) from LLVM
        # version evolution are expected.
        try:
            llvm.load_library_permanently("libLLVM-20.so")
        except Exception:
            pass
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
        _llvmlite_ready = True

    from llvmlite import binding as llvm
    from scratchv.standalone.compare_codegen import count_riscv_instrs

    mod = llvm.parse_assembly(ir_text)

    target = llvm.Target.from_triple("riscv64-unknown-elf")
    tm = target.create_target_machine(
        cpu="generic-rv64",
        features=features,
        opt=opt_level,
        reloc="default",
        codemodel="default",
    )

    asm = tm.emit_assembly(mod)

    if output_path:
        with open(output_path, "w") as f:
            f.write(asm)

    count, cats = count_riscv_instrs(asm)
    return count, asm, cats
