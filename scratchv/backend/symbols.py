"""The supported external function-symbol subset for RISC-V assembly."""

import re


# LLVM creates these section symbols even before an explicit section switch.
RESERVED_SECTION_SYMBOLS = frozenset({".text", ".data", ".bss", ".rodata"})


def validate_label_symbol(name: str, section_symbols=()) -> None:
    """Internal labels allow .L and $, but must be bare ASCII symbols."""
    if (not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z_.$][A-Za-z0-9_.$]*", name)
            or name == "."
            or name in RESERVED_SECTION_SYMBOLS
            or name in section_symbols):
        raise ValueError(f"unsupported RISC-V label symbol: {name!r}")


def label_reference(name: str) -> str:
    """Quote a leading dollar so LLVM parses a symbol expression, not a register."""
    validate_label_symbol(name)
    return f'"{name}"' if name.startswith("$") else name


def validate_call_symbol(name: str) -> None:
    """CALL requires a bare symbol in LLVM's RISC-V parser."""
    if (not isinstance(name, str)
            or not re.fullmatch(r"(?:[A-Za-z_]|\.[A-Za-z_])[A-Za-z0-9_.]*", name)
            or name in RESERVED_SECTION_SYMBOLS):
        raise ValueError(f"unsupported RISC-V call symbol: {name!r}")


def validate_function_symbol(name: str) -> None:
    # LLVM/GAS reserve .L* for assembler-local symbols; they cannot be exported
    # with .globl even though the spelling is valid for an internal label.
    if (not isinstance(name, str)
            or not re.fullmatch(r"(?:[A-Za-z_]|\.[A-Za-z_])[A-Za-z0-9_.]*", name)
            or name.startswith(".L")
            or name in RESERVED_SECTION_SYMBOLS):
        raise ValueError(
            f"unsupported RISC-V function symbol: {name!r}; "
            "use an ASCII identifier outside reserved sections and the .L local-symbol prefix"
        )
