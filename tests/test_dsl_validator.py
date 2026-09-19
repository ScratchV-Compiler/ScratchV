"""Integration tests for structured DSL validation."""

import pytest

from scratchv.frontend.dsl_errors import DSLSyntaxError
from scratchv.frontend.dsl_parser import DSLParser
from scratchv.frontend.dsl_extended import ExtendedDSLParser
from scratchv.frontend.dsl_validator import SourceBuffer


class TestSourceBuffer:
    def test_preserves_physical_lines_for_lf_and_crlf(self):
        lf = SourceBuffer("一\n\t二\n", filename="a.dsl")
        crlf = SourceBuffer("一\r\n\t二\r\n", filename="a.dsl")
        assert lf.lines == crlf.lines == ("一", "\t二", "")
        assert lf.line_text(2) == "\t二"
        assert lf.line_count == 3

    def test_empty_source_has_one_physical_line(self):
        source = SourceBuffer("")
        assert source.lines == ("",)
        assert source.line_text(1) == ""


@pytest.mark.parametrize(
    ("source", "code", "line", "col"),
    [
        ("retrun x", "E100", 1, 1),
        ("x = add(a, b", "E101", 1, 8),
        ("x = ad(a, b)", "E200", 1, 5),
        ("x = add(a)", "E201", 1, 9),
        ("x = relu(a, b)", "E201", 1, 10),
    ],
)
def test_base_parser_reports_structured_errors(source, code, line, col):
    parser = DSLParser()
    with pytest.raises(DSLSyntaxError) as caught:
        parser.parse(source, filename="bad.dsl")
    error = caught.value
    assert error.error_code == code
    assert (error.line, error.col) == (line, col)
    assert error.filename == "bad.dsl"
    assert error.source_line == source


def test_unsupported_op_has_conservative_spelling_hint():
    collector = DSLParser().validate("x = ad(a, b)")
    assert collector.errors[0].fix_hint == "did you mean 'add'?"


def test_validation_collects_independent_line_errors():
    collector = DSLParser().validate(
        "retrun x\ny = ad(a, b)\nz = relu(a, b)",
        filename="many.dsl",
    )
    assert [error.error_code for error in collector.errors] == [
        "E100", "E200", "E201",
    ]


def test_parser_state_is_clean_after_failed_parse():
    parser = DSLParser()
    with pytest.raises(DSLSyntaxError):
        parser.parse("x = add(a)")
    program = parser.parse("x = add(a, b)\nreturn x")
    assert len(program.functions) == 1
    assert len(program.functions[0].blocks[0].instructions) == 2


def test_negative_for_bound_is_structured_error_not_internal_exception():
    source = "for i = -1, 3\nendfor"
    collector = ExtendedDSLParser().validate(source)
    assert collector.errors[0].error_code == "E100"
    with pytest.raises(DSLSyntaxError) as caught:
        ExtendedDSLParser().parse(source)
    assert caught.value.error_code == "E100"


def test_unicode_identifier_remains_valid():
    program = DSLParser().parse("结果 = relu(x)\nreturn 结果")
    instructions = program.functions[0].blocks[0].instructions
    assert [instruction.opcode.name for instruction in instructions] == [
        "RELU", "RETURN",
    ]


def test_parser_and_signature_registry_have_same_operations():
    from scratchv.frontend.dsl_validator import OP_SIGNATURES

    assert set(OP_SIGNATURES) == DSLParser.supported_operations()


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("x = softmax(a, unknown:1)", "E202"),
        ("x = softmax(a, axis:nope)", "E203"),
    ],
)
def test_keyword_argument_diagnostics(source, code):
    collector = DSLParser().validate(source)
    assert [error.error_code for error in collector.errors] == [code]


@pytest.mark.parametrize(
    ("source", "expected_col"),
    [
        ("x = add a, b)", 13),
        ("x = add(a, b))", 14),
        ("x = add((a, b)", 8),
    ],
)
def test_unbalanced_call_parentheses_are_e101(source, expected_col):
    collector = DSLParser().validate(source)
    assert [error.error_code for error in collector.errors] == ["E101"]
    assert collector.errors[0].col == expected_col


def test_validation_error_limit_is_metadata_not_fake_error():
    source = "\n".join(f"bad statement {index}" for index in range(5))
    collector = DSLParser().validate(source, max_errors=3)
    assert collector.error_count == 3
    assert collector.limit_reached
    assert all(error.line > 0 for error in collector.errors)
    assert "--- 3 error(s) found ---" in collector.report()
    assert "further errors suppressed" in collector.report()


@pytest.mark.parametrize(
    ("source", "codes"),
    [
        ("endif", ["E110"]),
        ("if (a > b):\nx = add(a, b)", ["E111"]),
        ("else:\nx = add(a, b)", ["E112"]),
        (
            "if (a > b):\nelse:\nelse:\nendif",
            ["E112"],
        ),
        ("while (i < 3):\nendif", ["E110"]),
        (
            "if (a > b):\nwhile (i < 3):\nendif",
            ["E110"],
        ),
        (
            "if (a > b):\nwhile (i < 3):\nx = add(a, b)",
            ["E111", "E111"],
        ),
    ],
)
def test_extended_block_validation_and_recovery(source, codes):
    collector = ExtendedDSLParser().validate(source, filename="bad.dsl")
    assert [error.error_code for error in collector.errors] == codes


def test_extended_parser_reports_three_independent_errors():
    source = "endif\nx = ad(a, b)\nwhile (i < 3):"
    collector = ExtendedDSLParser().validate(source)
    assert [error.error_code for error in collector.errors] == [
        "E110", "E200", "E111",
    ]


def test_extended_parse_rejects_unterminated_block_before_ir_generation():
    parser = ExtendedDSLParser()
    with pytest.raises(DSLSyntaxError) as caught:
        parser.parse("if (a > b):\nx = add(a, b)", filename="bad.dsl")
    assert caught.value.error_code == "E111"
    assert caught.value.line == 1
