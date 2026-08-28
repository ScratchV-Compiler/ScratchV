"""Functional unit tests for semantic RISC-V instruction comments."""

import pytest

from scratchv.backend.asm_beautifier import (
    _INST_COMMENTS,
    _gen_comment,
    beautify_asm,
)
from scratchv.backend.asm_parser_for_beautifier import INSTRUCTION_SPECS


_SAMPLE_OPERANDS = {
    "rd": "a0",
    "rs": "a1",
    "rs1": "a1",
    "rs2": "a2",
    "imm": "4",
    "memory": "8(sp)",
    "target": ".L1",
    "symbol": "data_object",
    "predecessor": "rw",
    "successor": "rw",
}


@pytest.mark.parametrize(
    ("opcode", "operands", "expected"),
    [
        ("add", ["a0", "a1", "a2"], "a0 = a1 + a2"),
        ("sub", ["a0", "a1", "a2"], "a0 = a1 - a2"),
        ("lw", ["t0", "8(sp)"], "t0 = MEM[sp + 8]"),
        ("sw", ["ra", "28(sp)"], "MEM[sp + 28] = ra"),
        ("beq", ["a0", "a1", ".L1"], "if a0 == a1 goto .L1"),
        ("jal", ["ra", "worker"], "ra = PC+4; goto worker"),
    ],
)
def test_registered_instruction_comment(
    opcode: str,
    operands: list[str],
    expected: str,
) -> None:
    assert _gen_comment(opcode, operands) == expected


def test_all_parser_instructions_have_working_templates() -> None:
    assert set(_INST_COMMENTS) == set(INSTRUCTION_SPECS)

    for opcode, spec in INSTRUCTION_SPECS.items():
        operands = [
            _SAMPLE_OPERANDS[role]
            for role in spec.operand_roles[:spec.min_operands]
        ]
        comment = _gen_comment(opcode, operands)

        assert comment, f"{opcode} did not generate a comment"


@pytest.mark.parametrize(
    ("opcode", "operands"),
    [
        ("add", ["a0"]),
        ("lw", ["t0"]),
        ("beq", ["a0", "a1"]),
        ("custom_op", ["x1", "x2"]),
        (".loc", ["1 2 0"]),
    ],
)
def test_incomplete_unknown_and_directive_have_no_template_comment(
    opcode: str,
    operands: list[str],
) -> None:
    assert _gen_comment(opcode, operands) == ""


@pytest.mark.parametrize("opcode", ["", None])
def test_empty_opcode_has_no_template_comment(opcode: str | None) -> None:
    assert _gen_comment(opcode, []) == ""


def test_original_and_automatic_comments_are_merged() -> None:
    result = beautify_asm("add a0,a1,a2 # user note")

    assert result.rstrip().endswith("# user note | a0 = a1 + a2")


def test_add_comments_false_keeps_only_original_comment() -> None:
    source = "sub a0,a1,a2 # user note"

    result = beautify_asm(source, add_comments=False)

    assert result.rstrip().endswith("# user note")
    assert "a0 = a1 - a2" not in result


def test_abi_aliases_only_change_generated_comment() -> None:
    source = "lw x5,8(x2)"

    result = beautify_asm(source, abi_register_names=True)

    assert "lw        x5, 8(x2)" in result
    assert "# t0 = MEM[sp + 8]" in result


def test_abi_aliases_fill_all_register_roles() -> None:
    source = "add x1,x2,x10"

    result = beautify_asm(source, abi_register_names=True)

    assert "add       x1, x2, x10" in result
    assert "# ra = sp + a0" in result


def test_abi_aliases_are_disabled_by_default() -> None:
    result = beautify_asm("add x1,x2,x10")

    assert "# x1 = x2 + x10" in result


def test_abi_aliases_do_not_change_non_register_roles() -> None:
    result = beautify_asm(
        "beq x1,x2,x10\nli x1,x2",
        abi_register_names=True,
    )

    branch, immediate = result.splitlines()
    assert "# if ra == sp goto x10" in branch
    assert "# ra = x2" in immediate


def test_jal_single_operand_uses_implicit_ra() -> None:
    assert _gen_comment("jal", ["worker"]) == "ra = PC+4; goto worker"


def test_nop_with_original_comment_does_not_add_no_operation() -> None:
    result = beautify_asm("nop # operator boundary")

    assert result.rstrip().endswith("# operator boundary")
    assert "no operation" not in result


def test_no_align_only_appends_comment_without_reformatting_code() -> None:
    result = beautify_asm("add a0,a1,a2", align=False)

    assert result == "add a0,a1,a2  # a0 = a1 + a2"
