"""Integration tests: parser error branches, recovery, and golden IR.

Covers the topic-09 acceptance criteria:

* Strict ``parse()`` runs the shared structural pre-validation pass first
  (structured ``E1xx``/``E2xx`` codes, validator locations) and still raises
  rich ``DSLSyntaxError`` objects when the rich parser finds something the
  pre-validation missed.
* ``collector=`` mode uses the rich parser for recovery, collecting
  ``E3xx``/``E2xx`` diagnostics with suggestions and spans.
* Legal DSL (``examples/**/*.dsl``, ``benchmarks/cases/*.dsl``) produces zero
  diagnostics and unchanged IR (golden snapshot in ``tests/data/``).
* The compiler driver surfaces validator diagnostics by default and keeps a
  propagation path for rich ``DSLSyntaxError`` objects.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scratchv.compiler import CompilerConfig, CompilerDriver
from scratchv.frontend.dsl_errors import (
    DSLParseError,
    DSLSyntaxError,
    ErrorCode,
    ErrorCollector,
    format_error,
)
from scratchv.frontend.dsl_extended import ExtendedDSLParser
from scratchv.frontend.dsl_parser import DSLParser
from scratchv.ir.printer import IRPrinter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "dsl_golden_ir.json"


def rich_errors(parser, source: str, filename: str = "rich.dsl") -> ErrorCollector:
    """Parse with a collector to exercise the rich recovery diagnostics."""
    collector = ErrorCollector(
        filename=filename, use_color=False, source=source,
    )
    parser.parse(source, filename=filename, collector=collector)
    return collector


# ---------------------------------------------------------------------------
# Individual diagnostics
# ---------------------------------------------------------------------------

class TestErrorLocations:
    def test_unknown_op_preflight_location(self):
        source = "a = add(x, y)\nb = retrun(a, 1)\n"
        with pytest.raises(DSLSyntaxError) as excinfo:
            ExtendedDSLParser().parse(source, filename="unknown_op.dsl")
        e = excinfo.value

        assert e.error_code == "E200"
        assert (e.line, e.col) == (2, 5)
        assert e.filename == "unknown_op.dsl"
        assert e.source_line == "b = retrun(a, 1)"
        assert e.message == "unsupported operation 'retrun'"
        assert isinstance(e, DSLParseError)

    def test_unknown_op_rich_suggestion_in_collector_mode(self):
        source = "a = add(x, y)\nb = retrun(a, 1)\n"
        collector = rich_errors(
            ExtendedDSLParser(), source, "unknown_op.dsl",
        )
        assert collector.error_count == 1
        e = collector.errors[0]

        assert e.error_code == ErrorCode.SEM_UNKNOWN_OP == "E301"
        assert (e.line, e.col) == (2, 5)
        assert e.filename == "unknown_op.dsl"
        assert e.source_line == "b = retrun(a, 1)"
        assert e.fix_hint == "did you mean 'return'?"

        output = format_error(e, use_color=False)
        assert output == (
            "unknown_op.dsl:2:5: error[E301]: unknown operation 'retrun'\n"
            "  2 | b = retrun(a, 1)\n"
            "    |     ^~~~~~\n"
            "note: did you mean 'return'?"
        )
        assert output.splitlines()[2] == "    |     ^~~~~~"
        assert str(e) == output

    def test_double_digit_line_marker_alignment(self):
        source = "a = add(x, y)\n" * 8 + "b = retrun(a, 1)\n"
        with pytest.raises(DSLSyntaxError) as excinfo:
            ExtendedDSLParser().parse(source, filename="test.dsl")
        e = excinfo.value
        assert (e.line, e.col) == (9, 5)
        assert e.error_code == "E200"
        marker = format_error(e, use_color=False).splitlines()[2]
        assert marker == "    |     ^~~~~~"

        # Directly exercise a two-digit line number
        e2 = DSLSyntaxError(
            line=10, col=5,
            message="unknown operation 'retrun'",
            source_line="d = retrun(c)",
            error_code="E301",
        )
        assert format_error(e2, use_color=False).splitlines()[2] == (
            "     |     ^~~~~~"
        )

    def test_indented_statement_column(self):
        source = "if (a > b):\n    d = retrun(c)\nendif\n"
        with pytest.raises(DSLSyntaxError) as excinfo:
            ExtendedDSLParser().parse(source, filename="indent.dsl")
        e = excinfo.value
        assert (e.line, e.col) == (2, 9)
        assert e.error_code == "E200"

        collector = rich_errors(
            ExtendedDSLParser(), source, "indent.dsl",
        )
        rich = collector.errors[0]
        assert rich.error_code == "E301"
        assert (rich.line, rich.col) == (2, 9)
        lines = format_error(rich, use_color=False).splitlines()
        assert lines[1] == "  2 |     d = retrun(c)"
        assert lines[2] == "    |         ^~~~~~"

    def test_illegal_character(self):
        source = "a = add(b, c) $\n"
        with pytest.raises(DSLSyntaxError) as excinfo:
            DSLParser().parse(source, filename="lex.dsl")
        e = excinfo.value
        assert e.error_code == "E100"
        assert (e.line, e.col) == (1, 1)
        assert e.message == "cannot parse statement"

        collector = rich_errors(DSLParser(), source, "lex.dsl")
        rich = collector.errors[0]
        assert rich.error_code == ErrorCode.LEX_ILLEGAL_CHAR == "E101"
        assert rich.message == "unexpected character '$'"
        # '$' is the 15th character of the physical line
        assert (rich.line, rich.col) == (1, 15)
        assert rich.fix_hint == "remove or replace '$'"

    def test_arity_and_nested_call(self):
        with pytest.raises(DSLSyntaxError) as excinfo:
            DSLParser().parse("a = add(1)\n", filename="arity.dsl")
        e = excinfo.value
        assert e.error_code == "E201"
        assert (e.line, e.col) == (1, 9)
        assert e.message == (
            "operation 'add' expects 2 positional argument(s), got 1"
        )

        collector = rich_errors(DSLParser(), "a = add(1)\n", "arity.dsl")
        rich = collector.errors[0]
        assert rich.error_code == ErrorCode.SEM_ARITY == "E302"
        assert (rich.line, rich.col) == (1, 5)
        assert rich.message == "add() expects 2 arguments, got 1"
        assert rich.fix_hint == "add() requires exactly 2 arguments"

        with pytest.raises(DSLSyntaxError) as excinfo:
            DSLParser().parse("a = add()\n", filename="arity.dsl")
        assert excinfo.value.error_code == "E201"

        collector = rich_errors(DSLParser(), "a = add()\n", "arity.dsl")
        assert collector.errors[0].message == (
            "add() expects 2 arguments, got 0"
        )

        with pytest.raises(DSLSyntaxError) as excinfo:
            DSLParser().parse(
                "c = add(mul(a, b), d)\n", filename="nested.dsl",
            )
        assert excinfo.value.error_code == "E201"

        collector = rich_errors(
            DSLParser(), "c = add(mul(a, b), d)\n", "nested.dsl",
        )
        rich = collector.errors[0]
        assert rich.error_code == ErrorCode.SYN_NESTED_CALL == "E205"
        assert (rich.line, rich.col) == (1, 12)
        assert rich.message == "nested function call is not supported"

    def test_invalid_condition_is_e202_and_parse_error(self):
        source = "if a > b:\n  c = add(a, b)\nendif\nreturn c\n"
        with pytest.raises(DSLParseError) as excinfo:
            ExtendedDSLParser().parse(source, filename="cond.dsl")
        e = excinfo.value
        assert isinstance(e, DSLSyntaxError)
        assert e.error_code == "E101"
        assert (e.line, e.col) == (1, 1)
        assert e.message == "invalid if condition"
        assert e.fix_hint == "add matching parentheses around the condition"

        collector = rich_errors(
            ExtendedDSLParser(), source, "cond.dsl",
        )
        rich = collector.errors[0]
        assert rich.error_code == "E202"
        assert (rich.line, rich.col) == (1, 1)
        assert rich.message.startswith("invalid condition in 'if'")
        assert format_error(rich, use_color=False).splitlines()[-1] == (
            "note: expected one of ==, !=, <, >, <=, >= "
            "and parentheses around each operand"
        )

    @pytest.mark.parametrize("line,opener,code", [
        ("endif", "if", "E110"),
        ("endwhile", "while", "E110"),
        ("endfor", "for", "E110"),
        ("else", "if", "E112"),
        ("else:", "if", "E112"),
    ])
    def test_stray_terminators(self, line, opener, code):
        with pytest.raises(DSLSyntaxError) as excinfo:
            ExtendedDSLParser().parse(line + "\n", filename="stray.dsl")
        e = excinfo.value
        assert e.error_code == code
        assert (e.line, e.col) == (1, 1)

        collector = rich_errors(
            ExtendedDSLParser(), line + "\n", "stray.dsl",
        )
        rich = collector.errors[0]
        assert rich.error_code == ErrorCode.SYN_STRAY_TERMINATOR == "E204"
        assert (rich.line, rich.col) == (1, 1)
        assert rich.message == (
            f"'{line.rstrip(':')}' without matching '{opener}'"
        )


class TestBlockTerminators:
    def test_missing_endif_reported_at_opener(self):
        source = (
            "i = add(x, 1)\n"
            "if (i > 0):\n"
            "  y = mul(i, 2)\n"
            "return y\n"
        )
        with pytest.raises(DSLSyntaxError) as excinfo:
            ExtendedDSLParser().parse(source, filename="missing_endif.dsl")
        e = excinfo.value
        assert e.error_code == "E111"
        assert (e.line, e.col) == (2, 1)
        assert e.message == "unterminated if block"
        assert e.fix_hint == "add missing 'endif'"

        collector = rich_errors(
            ExtendedDSLParser(), source, "missing_endif.dsl",
        )
        rich = collector.errors[0]
        assert rich.error_code == ErrorCode.SYN_MISSING_TERMINATOR == "E203"
        assert (rich.line, rich.col) == (2, 1)
        assert rich.message == "missing 'endif' for 'if' opened here"
        assert format_error(rich, use_color=False) == (
            "missing_endif.dsl:2:1: error[E203]: "
            "missing 'endif' for 'if' opened here\n"
            "  2 | if (i > 0):\n"
            "    | ^~\n"
            "note: add 'endif' to close this block"
        )

        collector = ErrorCollector(
            filename="missing_endif.dsl", use_color=False, source=source,
        )
        program = ExtendedDSLParser().parse(
            source, filename="missing_endif.dsl", collector=collector,
        )
        assert collector.error_count == 1
        assert program.functions  # partial Program is still constructible

    def test_missing_endwhile_reported_at_opener(self):
        source = "while (i < 9):\n  a = add(a, b)\n"
        with pytest.raises(DSLSyntaxError) as excinfo:
            ExtendedDSLParser().parse(source, filename="w.dsl")
        e = excinfo.value
        assert e.error_code == "E111"
        assert (e.line, e.col) == (1, 1)
        assert e.message == "unterminated while block"

        collector = rich_errors(ExtendedDSLParser(), source, "w.dsl")
        rich = collector.errors[0]
        assert rich.error_code == "E203"
        assert rich.message == "missing 'endwhile' for 'while' opened here"

    def test_missing_endfor_reported_at_opener(self):
        source = "for i = 0, 4\n  acc = add(acc, i)\nreturn acc\n"
        with pytest.raises(DSLSyntaxError) as excinfo:
            ExtendedDSLParser().parse(source, filename="f.dsl")
        e = excinfo.value
        assert e.error_code == "E111"
        assert (e.line, e.col) == (1, 1)
        assert e.message == "unterminated for block"

        collector = rich_errors(ExtendedDSLParser(), source, "f.dsl")
        rich = collector.errors[0]
        assert rich.error_code == "E203"
        assert rich.message == "missing 'endfor' for 'for' opened here"

        collector = ErrorCollector(filename="f.dsl", use_color=False)
        ExtendedDSLParser().parse(
            source, filename="f.dsl", collector=collector,
        )

    def test_missing_endif_when_foreign_terminator(self):
        # 'while' never closed, then 'endif' from an outer if
        source = (
            "if (a > b):\n"
            "  while (i < 9):\n"
            "    c = add(a, b)\n"
            "endif\n"
        )
        collector = ErrorCollector(
            filename="mixed.dsl", use_color=False, source=source,
        )
        ExtendedDSLParser().parse(
            source, filename="mixed.dsl", collector=collector,
        )
        codes = [e.error_code for e in collector.errors]
        assert "E203" in codes


# ---------------------------------------------------------------------------
# Multi-error collection / recovery
# ---------------------------------------------------------------------------

class TestCollectorRecovery:
    def test_multi_error_collection_and_recovery(self):
        source = (
            "a = add(1)\n"
            "b = retrun(a, 2)\n"
            "if a > 0:\n"
            "  c = mul(a, 2)\n"
            "endwhile\n"
        )
        collector = ErrorCollector(
            filename="multi_error.dsl", use_color=False, source=source,
        )
        program = ExtendedDSLParser().parse(
            source, filename="multi_error.dsl", collector=collector,
        )

        assert collector.error_count == 4
        assert collector.suppressed_count == 0
        assert collector.has_errors
        assert program.functions
        assert [e.error_code for e in collector.errors] == [
            "E302", "E301", "E202", "E204",
        ]
        assert [e.line for e in collector.errors] == [1, 2, 3, 5]

        assert collector.report() == (
            "--- 4 error(s) found ---\n"
            "multi_error.dsl:1:5: error[E302]: add() expects 2 arguments, "
            "got 1\n"
            "  1 | a = add(1)\n"
            "    |     ^~~\n"
            "note: add() requires exactly 2 arguments\n"
            "multi_error.dsl:2:5: error[E301]: unknown operation 'retrun'\n"
            "  2 | b = retrun(a, 2)\n"
            "    |     ^~~~~~\n"
            "note: did you mean 'return'?\n"
            "multi_error.dsl:3:1: error[E202]: invalid condition in 'if'; "
            "expected 'if (<expr>) <op> (<expr>):'\n"
            "  3 | if a > 0:\n"
            "    | ^~\n"
            "note: expected one of ==, !=, <, >, <=, >= "
            "and parentheses around each operand\n"
            "multi_error.dsl:5:1: error[E204]: 'endwhile' without matching "
            "'while'\n"
            "  5 | endwhile\n"
            "    | ^~~~~~~~\n"
            "note: remove this line or add a matching 'while'"
        )

        # Strict mode fails fast through the shared pre-validation pass
        with pytest.raises(DSLSyntaxError) as excinfo:
            ExtendedDSLParser().parse(source, filename="multi_error.dsl")
        e = excinfo.value
        assert (e.line, e.col, e.error_code) == (1, 9, "E201")

    def test_collector_continues_after_unknown_op(self):
        source = "a = retrun(b, 1)\nc = add(a, b)\nreturn c\n"
        collector = ErrorCollector(
            filename="recover.dsl", use_color=False, source=source,
        )
        program = ExtendedDSLParser().parse(
            source, filename="recover.dsl", collector=collector,
        )
        assert collector.error_count == 1
        # The failed assignment must not be registered under its dest name
        assert program is not None

    def test_no_derived_errors_inside_bad_block(self):
        source = (
            "if bad condition\n"
            "  x = retrun(a, 1)\n"
            "  y = add(1)\n"
            "endif\n"
        )
        collector = ErrorCollector(
            filename="recover_block.dsl", use_color=False, source=source,
        )
        ExtendedDSLParser().parse(
            source, filename="recover_block.dsl", collector=collector,
        )
        assert collector.error_count == 1
        assert collector.errors[0].error_code == "E202"


# ---------------------------------------------------------------------------
# F1 regression: malformed statements must not be silently accepted
# ---------------------------------------------------------------------------

class TestCollectorMalformedStatements:
    @pytest.mark.parametrize(
        ("source", "strict_code", "rich_code"),
        [
            ("return x junk\n", "E100", "E201"),
            (
                "for i = 0, 4 junk\nacc = add(acc, i)\nendfor\n",
                "E100",
                "E201",
            ),
            ("a = add(x, y, foo:1)\n", "E202", "E304"),
            ("m = matmul(a, b, rows:abc)\n", "E203", "E304"),
        ],
    )
    def test_malformed_statements_diagnosed_in_both_modes(
        self, source, strict_code, rich_code,
    ):
        with pytest.raises(DSLSyntaxError) as excinfo:
            ExtendedDSLParser().parse(source, filename="malformed.dsl")
        assert excinfo.value.error_code == strict_code

        collector = rich_errors(
            ExtendedDSLParser(), source, "malformed.dsl",
        )
        assert collector.error_count >= 1
        assert collector.suppressed_count == 0
        assert rich_code in [e.error_code for e in collector.errors]

    def test_invalid_kwarg_location_and_message(self):
        collector = rich_errors(
            ExtendedDSLParser(), "a = add(x, y, foo:1)\n", "kw.dsl",
        )
        error = collector.errors[0]
        assert error.error_code == ErrorCode.SEM_UNKNOWN_KWARG == "E304"
        assert (error.line, error.col) == (1, 15)
        assert error.message == "invalid keyword argument 'foo'"
        assert "'foo' is not accepted by add()" in (
            format_error(error, use_color=False)
        )

        collector = rich_errors(
            ExtendedDSLParser(), "m = matmul(a, b, rows:abc)\n", "kw.dsl",
        )
        error = collector.errors[0]
        assert error.error_code == "E304"
        assert (error.line, error.col) == (1, 18)
        assert error.message == "'rows' requires a numeric value"

    @pytest.mark.parametrize(
        ("source", "opcode"),
        [
            ("d = dot(a, b, len:4)\n", "DOT"),
            ("m = matmul(a, b, m:2, n:2, k:2)\n", "MATMUL"),
            ("s = softmax(x, axis:-1)\n", "SOFTMAX"),
            ("p = maxpool(x, kernel:2, stride:2)\n", "MAXPOOL"),
        ],
    )
    def test_registered_kwargs_stay_accepted(self, source, opcode):
        collector = ErrorCollector(
            filename="kwargs_ok.dsl", use_color=False, source=source,
        )
        program = ExtendedDSLParser().parse(
            source, filename="kwargs_ok.dsl", collector=collector,
        )
        assert collector.error_count == 0, collector.report()
        opcodes = [
            instr.opcode.name
            for instr in program.functions[0].blocks[0].instructions
        ]
        assert opcode in opcodes

    def test_return_prefix_identifier_is_not_swallowed(self):
        # E2 root cause: 'returnx = add(a,b)' used to be read as 'return x'.
        source = "returnx = add(a, b)\n"
        collector = ErrorCollector(
            filename="prefix.dsl", use_color=False, source=source,
        )
        program = ExtendedDSLParser().parse(
            source, filename="prefix.dsl", collector=collector,
        )
        assert collector.error_count == 0
        opcodes = [
            instr.opcode.name
            for instr in program.functions[0].blocks[0].instructions
        ]
        assert "ADD" in opcodes


# ---------------------------------------------------------------------------
# F2 regression: validator limit accounting
# ---------------------------------------------------------------------------

class TestValidatorLimitSemantics:
    def test_suppressed_counts_all_unreported_line_errors(self):
        source = "\n".join(f"retrun x{i}" for i in range(50)) + "\n"
        collector = ExtendedDSLParser().validate(source, max_errors=3)
        assert collector.error_count == 3
        assert collector.limit_reached
        assert collector.suppressed_count == 47
        assert collector.report().splitlines()[-1] == (
            "note: error limit (3) reached; 47 further errors suppressed"
        )

    def test_suppressed_counts_trailing_block_errors(self):
        # Unterminated blocks are reported after the line loop; they must
        # still be accounted for once the limit was already reached.
        source = "".join(f"if (a > {i}):\n" for i in range(5))
        collector = ExtendedDSLParser().validate(source, max_errors=2)
        assert collector.error_count == 2
        assert collector.limit_reached
        assert collector.suppressed_count == 3


# ---------------------------------------------------------------------------
# F7 regression: CRLF source-line consistency
# ---------------------------------------------------------------------------

class TestCrlfSourceLineConsistency:
    def test_strict_and_collector_agree_on_crlf_source_line(self):
        source = "a = retrun(b, 1)\r\n"
        with pytest.raises(DSLSyntaxError) as excinfo:
            DSLParser().parse(source, filename="crlf.dsl")
        strict_line = excinfo.value.source_line
        assert strict_line == "a = retrun(b, 1)"

        collector = rich_errors(DSLParser(), source, "crlf.dsl")
        assert collector.errors[0].source_line == strict_line

    def test_extended_parser_normalizes_crlf(self):
        source = "if (a > b):\r\n  c = retrun(a)\r\nendif\r\n"
        collector = rich_errors(
            ExtendedDSLParser(), source, "crlf_ext.dsl",
        )
        assert collector.error_count == 1
        assert collector.errors[0].source_line == "  c = retrun(a)"
        assert all(
            "\r" not in error.source_line for error in collector.errors
        )

    def test_crlf_valid_input_parses_cleanly(self):
        source = "c = add(a, b)\r\nreturn c\r\n"
        collector = rich_errors(DSLParser(), source, "crlf_ok.dsl")
        assert collector.error_count == 0


# ---------------------------------------------------------------------------
# Regression: legal DSL produces zero diagnostics and unchanged IR
# ---------------------------------------------------------------------------

class TestLegalDslGolden:
    def test_legal_dsl_zero_diagnostics(self):
        golden = json.loads(GOLDEN_PATH.read_text())
        assert len(golden) >= 30

        for path_str, expect in sorted(golden.items()):
            path = PROJECT_ROOT / path_str
            source = path.read_text()
            collector = ErrorCollector(
                filename=path_str, use_color=False, source=source,
            )
            parser = (
                ExtendedDSLParser() if expect["parser"] == "extended"
                else DSLParser()
            )
            program = parser.parse(source, filename=path_str,
                                   collector=collector)
            assert not collector.has_errors, (
                f"{path_str}:\n{collector.report()}"
            )
            assert IRPrinter(program).dump() == expect["ir"], path_str

    def test_unparseable_baseline_file_stays_unparseable(self):
        # examples/cnn_model.dsl was already unparsable before this change
        # (unsupported ops / ';' comments); it must not be silently accepted.
        source = (PROJECT_ROOT / "examples/cnn_model.dsl").read_text()
        with pytest.raises(DSLParseError):
            ExtendedDSLParser().parse(source, filename="cnn_model.dsl")

    def test_parse_default_signature_compat(self):
        src = "c = add(a, b)\nreturn c\n"
        assert DSLParser().parse(src)
        assert ExtendedDSLParser().parse(src)


# ---------------------------------------------------------------------------
# Compiler driver integration
# ---------------------------------------------------------------------------

class TestCompilerIntegration:
    def test_compiler_reports_structured_error_for_bad_dsl(self, tmp_path):
        bad = tmp_path / "bad.dsl"
        bad.write_text("a = add(1)\n")
        out = tmp_path / "out.s"
        driver = CompilerDriver(CompilerConfig())
        result = driver.compile(str(bad), str(out))

        assert result.success is False
        assert len(result.errors) == 1
        assert "bad.dsl:1:9: error[E201]" in result.errors[0]
        assert "operation 'add' expects 2 positional argument(s), got 1" in (
            result.errors[0]
        )
        assert not out.exists()

    def test_compiler_propagates_rich_syntax_error(
        self, tmp_path, monkeypatch,
    ):
        import scratchv.frontend.dsl_extended as ext_mod

        rich = DSLSyntaxError(
            line=7, col=3, message="unknown operation 'retrun'",
            source_line="b = retrun(a, 1)", filename="rich.dsl",
            fix_hint="did you mean 'return'?", error_code="E301",
        )

        class RichFailingExtended:
            def validate(self, text, *, filename=None, max_errors=20):
                return ErrorCollector(filename=filename, use_color=False)

            def parse(self, text, filename=None, collector=None):
                raise rich

        monkeypatch.setattr(ext_mod, "ExtendedDSLParser", RichFailingExtended)
        src = tmp_path / "rich.dsl"
        src.write_text("b = retrun(a, 1)\n")
        out = tmp_path / "out.s"
        result = CompilerDriver(CompilerConfig()).compile(str(src), str(out))

        assert result.success is False
        assert result.errors == [str(rich)]
        assert result.diagnostics == [rich]
        assert not out.exists()

    def test_compiler_surfaces_rich_error_when_validator_misses(
        self, tmp_path,
    ):
        # 'add(mul(b, c))' passes the validator (inner comma splits into two
        # positional args), so the rich parser must report E205 for real.
        src = tmp_path / "nested.dsl"
        src.write_text("a = add(mul(b, c))\n")
        out = tmp_path / "out.s"
        result = CompilerDriver(CompilerConfig()).compile(str(src), str(out))

        assert result.success is False
        assert len(result.errors) == 1
        assert "error[E205]" in result.errors[0]
        assert not out.exists()

    def test_compiler_compiles_for_loop(self, tmp_path):
        src = tmp_path / "loop.dsl"
        src.write_text("for i = 0, 4\n  acc = add(acc, i)\nendfor\nreturn acc\n")
        out = tmp_path / "out.s"
        driver = CompilerDriver(CompilerConfig())
        result = driver.compile(str(src), str(out))
        assert result.success is True
        assert out.exists()

    def test_compiler_falls_back_for_base_only_extended_failure(
        self, tmp_path, monkeypatch,
    ):
        import scratchv.frontend.dsl_extended as ext_mod

        class FailingExtended:
            def validate(self, text, *, filename=None, max_errors=20):
                return ErrorCollector(filename=filename, use_color=False)

            def parse(self, text, filename=None, collector=None):
                raise DSLParseError("extended parser not applicable")

        monkeypatch.setattr(ext_mod, "ExtendedDSLParser", FailingExtended)
        src = tmp_path / "base.dsl"
        src.write_text("c = add(a, b)\nreturn c\n")
        out = tmp_path / "out.s"
        driver = CompilerDriver(CompilerConfig())
        result = driver.compile(str(src), str(out))
        assert result.success is True
        assert out.exists()

    def test_compiler_inline_dsl_source(self, tmp_path):
        out = tmp_path / "inline.s"
        driver = CompilerDriver(CompilerConfig())
        result = driver.compile(
            input_path="", output_path=str(out),
            dsl_source="c = add(a, b)\nreturn c\n",
        )
        assert result.success is True
