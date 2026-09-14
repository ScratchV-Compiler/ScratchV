"""Tests for the DSL error beautifier module."""

import io
import pytest
from scratchv.frontend.dsl_parser import DSLParseError
from scratchv.frontend.dsl_errors import (
    DSLSyntaxError,
    format_error,
    ErrorCollector,
    ErrorCode,
    make_error,
    Color,
    render_error,
    _compute_suggestion,
    _estimate_token_length,
)


class TestDSLSyntaxError:
    """Tests for the DSLSyntaxError exception class."""

    def test_create_basic_error(self):
        err = DSLSyntaxError(
            line=5,
            col=12,
            message="unexpected token",
            source_line="result = retrun(x)",
        )
        assert err.line == 5
        assert err.col == 12
        assert "unexpected token" in err.message
        assert "retrun" in err.source_line
        assert err.filename is None

    def test_create_with_filename(self):
        err = DSLSyntaxError(
            line=3,
            col=1,
            message="missing endif",
            filename="test.dsl",
        )
        assert err.filename == "test.dsl"

    def test_create_with_fix_hint(self):
        err = DSLSyntaxError(
            line=1,
            col=10,
            message="syntax error",
            fix_hint="did you mean 'return'?",
        )
        assert err.fix_hint == "did you mean 'return'?"

    def test_create_with_error_code(self):
        err = DSLSyntaxError(
            line=2,
            col=5,
            message="type error",
            error_code="E001",
        )
        assert err.error_code == "E001"

    def test_str_representation(self):
        err = DSLSyntaxError(
            line=3,
            col=1,
            message="test error",
            source_line="bad line",
        )
        s = str(err)
        assert "3:1" in s
        assert "test error" in s

    def test_error_is_exception(self):
        err = DSLSyntaxError(1, 1, "msg")
        with pytest.raises(DSLSyntaxError):
            raise err

    def test_error_is_compatible_with_dsl_parse_error(self):
        err = DSLSyntaxError(1, 1, "msg")
        assert isinstance(err, DSLParseError)

    def test_str_never_contains_ansi(self):
        err = DSLSyntaxError(1, 1, "msg", source_line="bad")
        assert "\033[" not in str(err)


class TestFormatError:
    """Tests for the format_error() function."""

    def test_basic_format_no_color(self):
        err = DSLSyntaxError(
            line=5,
            col=12,
            message="unexpected token 'retrun'",
            source_line="result = retrun(x)",
            filename="test.dsl",
        )
        output = format_error(err, use_color=False)
        assert "test.dsl:5:12:" in output
        assert "error:" in output
        assert "unexpected token" in output
        assert "result = retrun(x)" in output

    def test_format_with_fix_hint(self):
        err = DSLSyntaxError(
            line=10,
            col=1,
            message="unterminated block",
            source_line="while (i < 10):",
            fix_hint="missing 'endwhile'",
        )
        output = format_error(err, use_color=False)
        assert "note:" in output
        assert "endwhile" in output

    def test_format_auto_suggestion(self):
        err = DSLSyntaxError(
            line=5,
            col=12,
            message="unexpected keyword",
            source_line="result = retrun(x)",
        )
        output = format_error(err, use_color=False)
        assert "note:" in output or "did you mean" in output.lower()

    def test_format_no_source_line(self):
        err = DSLSyntaxError(
            line=1,
            col=1,
            message="file not found",
        )
        output = format_error(err, use_color=False)
        assert "file not found" in output
        # Source line marker should not appear
        assert "|" not in output

    def test_format_with_color(self):
        err = DSLSyntaxError(
            line=5,
            col=12,
            message="test error",
            source_line="some code here",
        )
        output = format_error(err, use_color=True)
        # ANSI codes should be present
        assert "\033[" in output

    def test_format_no_color(self):
        err = DSLSyntaxError(
            line=5,
            col=12,
            message="test error",
        )
        output = format_error(err, use_color=False)
        # No ANSI codes
        assert "\033[" not in output

    def test_format_with_error_code(self):
        err = DSLSyntaxError(
            line=1,
            col=1,
            message="test error",
            error_code="E001",
        )
        output = format_error(err, use_color=False)
        assert "error[E001]: test error" in output

    def test_format_unknown_filename(self):
        err = DSLSyntaxError(1, 1, "test error")
        assert format_error(err, use_color=False).startswith(
            "<dsl>:1:1: error:"
        )

    def test_format_uses_explicit_span(self):
        err = DSLSyntaxError(
            1, 5, "test error", source_line="x = retrun y", end_col=11,
        )
        marker = format_error(err, use_color=False).splitlines()[2]
        assert "^~~~~~" in marker

    def test_format_expands_tabs_for_marker(self):
        err = DSLSyntaxError(
            1, 2, "test error", source_line="\tbad", end_col=5,
        )
        output = format_error(err, use_color=False)
        assert "  1 |     bad" in output
        marker = output.splitlines()[2]
        assert marker.index("^") == output.splitlines()[1].index("b")

    def test_render_error_auto_color_uses_tty(self, monkeypatch):
        class TTY(io.StringIO):
            def isatty(self):
                return True

        monkeypatch.delenv("NO_COLOR", raising=False)
        output = render_error(
            DSLSyntaxError(1, 1, "bad"), stream=TTY(), use_color=None,
        )
        assert "\033[" in output

    def test_render_error_auto_color_honors_no_color(self, monkeypatch):
        class TTY(io.StringIO):
            def isatty(self):
                return True

        monkeypatch.setenv("NO_COLOR", "1")
        output = render_error(
            DSLSyntaxError(1, 1, "bad"), stream=TTY(), use_color=None,
        )
        assert "\033[" not in output

    def test_format_column_marker(self):
        err = DSLSyntaxError(
            line=3,
            col=5,
            message="expected something",
            source_line="abc def ghi",
        )
        output = format_error(err, use_color=False, show_column_marker=True)
        # Should have the caret marker
        assert "^" in output

    def test_format_without_column_marker(self):
        err = DSLSyntaxError(
            line=3,
            col=5,
            message="expected something",
            source_line="abc def ghi",
        )
        output = format_error(err, use_color=False, show_column_marker=False)
        assert "^" not in output


class TestErrorCollector:
    """Tests for the ErrorCollector class."""

    def test_empty_collector(self):
        collector = ErrorCollector()
        assert not collector.has_errors
        assert collector.error_count == 0
        assert len(collector.errors) == 0

    def test_add_single_error(self):
        collector = ErrorCollector(filename="test.dsl")
        err = DSLSyntaxError(1, 1, "test error")
        collector.add(err)
        assert collector.has_errors
        assert collector.error_count == 1

    def test_add_multiple_errors(self):
        collector = ErrorCollector()
        for i in range(5):
            collector.add(DSLSyntaxError(i + 1, 1, f"error {i}"))
        assert collector.error_count == 5

    def test_add_error_convenience(self):
        collector = ErrorCollector(filename="test.dsl")
        collector.add_error(
            line=3,
            col=5,
            message="missing parenthesis",
            source_line="a = add(b",
        )
        assert collector.error_count == 1
        err = collector.errors[0]
        assert err.line == 3
        assert err.col == 5
        assert err.filename == "test.dsl"

    def test_report_format(self):
        collector = ErrorCollector(filename="prog.dsl", use_color=False)
        collector.add_error(1, 1, "first error")
        collector.add_error(2, 3, "second error", source_line="bad")
        report = collector.report()
        assert "2 error(s) found" in report
        assert "first error" in report
        assert "second error" in report

    def test_report_no_errors(self):
        collector = ErrorCollector()
        assert collector.report() == ""

    def test_max_errors_limit(self):
        collector = ErrorCollector(max_errors=3)
        for i in range(10):
            collector.add(DSLSyntaxError(i + 1, 1, f"error {i}"))
        assert len(collector.errors) == 3
        assert collector.error_count == 3
        assert collector.limit_reached
        assert "further errors suppressed" in collector.report()

    @pytest.mark.parametrize("max_errors", [0, -1, -20])
    def test_non_positive_max_errors_rejected(self, max_errors):
        # A zero/negative limit silently suppressed every error; reject loud.
        with pytest.raises(ValueError, match="max_errors"):
            ErrorCollector(max_errors=max_errors)

    def test_deduplicates_and_sorts_errors(self):
        collector = ErrorCollector(use_color=False)
        second = DSLSyntaxError(2, 3, "second", error_code="E200")
        first = DSLSyntaxError(1, 2, "first", error_code="E100")
        collector.add(second)
        collector.add(first)
        collector.add(first)
        assert collector.errors == [first, second]

    def test_clear_resets_limit_state(self):
        collector = ErrorCollector(max_errors=1)
        collector.add(DSLSyntaxError(1, 1, "first"))
        collector.add(DSLSyntaxError(2, 1, "second"))
        assert collector.limit_reached
        collector.clear()
        assert not collector.limit_reached

    def test_duplicate_at_capacity_does_not_reach_limit(self):
        collector = ErrorCollector(max_errors=1)
        error = DSLSyntaxError(1, 1, "first")
        collector.add(error)
        collector.add(error)
        assert collector.error_count == 1
        assert not collector.limit_reached

    def test_clear_errors(self):
        collector = ErrorCollector()
        collector.add(DSLSyntaxError(1, 1, "test"))
        assert collector.has_errors
        collector.clear()
        assert not collector.has_errors

    def test_filename_auto_set(self):
        collector = ErrorCollector(filename="auto.dsl")
        err = DSLSyntaxError(1, 1, "test")  # no filename
        collector.add(err)
        # Error should get collector's filename
        assert collector.errors[0].filename == "auto.dsl"

    def test_errors_list_is_copy(self):
        collector = ErrorCollector()
        collector.add(DSLSyntaxError(1, 1, "test"))
        errors = collector.errors
        errors.append(DSLSyntaxError(2, 2, "extra"))
        # Original collector should not be affected
        assert collector.error_count == 1


class TestMakeError:
    """Tests for the make_error() factory function."""

    def test_make_error_basic(self):
        err = make_error(line=10, col=5, message="test")
        assert err.line == 10
        assert err.col == 5
        assert err.message == "test"

    def test_make_error_with_all_fields(self):
        err = make_error(
            line=1, col=2, message="msg",
            source_line="source", filename="f.dsl",
            fix_hint="hint", error_code="E001",
        )
        assert err.line == 1
        assert err.col == 2
        assert err.source_line == "source"
        assert err.filename == "f.dsl"
        assert err.fix_hint == "hint"
        assert err.error_code == "E001"


@pytest.mark.parametrize("line", [1, 9, 10, 99, 100, 1000])
@pytest.mark.parametrize("indent", ["", "    ", "\t", " \t"])
@pytest.mark.parametrize("use_color", [False, True])
def test_marker_aligns_with_token_across_line_number_widths(line, indent, use_color):
    import re

    source = indent + "x = ad(a, b)"
    col = source.index("ad") + 1
    error = DSLSyntaxError(
        line, col, "unsupported operation", source_line=source,
        error_code="E200", end_col=col + 2,
    )
    rendered = format_error(error, use_color=use_color)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", rendered)
    source_display, marker = plain.splitlines()[1:3]
    assert marker.index("^") == source_display.index("ad")


class TestColor:
    """Tests for ANSI color definitions."""

    def test_color_values_not_empty(self):
        for color in Color:
            assert color.value != ""

    def test_color_reset(self):
        assert Color.RESET.value == "\033[0m"


class TestErrorHierarchy:
    """Tests for the DSLParseError / DSLSyntaxError relationship."""

    def test_syntax_error_is_parse_error(self):
        assert issubclass(DSLSyntaxError, DSLParseError)
        with pytest.raises(DSLParseError):
            raise DSLSyntaxError(1, 1, "boom")

    def test_suggestion_alias(self):
        err = DSLSyntaxError(
            line=1, col=2, message="msg", suggestion="try x",
        )
        assert err.suggestion == "try x"
        assert err.fix_hint == "try x"
        err.suggestion = "try y"
        assert err.fix_hint == "try y"

    def test_fix_hint_wins_over_suggestion(self):
        err = DSLSyntaxError(
            line=1, col=2, message="msg",
            fix_hint="primary", suggestion="secondary",
        )
        assert err.fix_hint == "primary"
        assert err.suggestion == "primary"

    def test_positional_construction_compat(self):
        err = DSLSyntaxError(3, 4, "m", "src", "f.dsl", "hint", "E201")
        assert (err.line, err.col, err.message) == (3, 4, "m")
        assert (err.source_line, err.filename) == ("src", "f.dsl")
        assert (err.fix_hint, err.error_code) == ("hint", "E201")
        assert err.args == ("m",)

    def test_error_code_constants(self):
        assert ErrorCode.LEX_ILLEGAL_CHAR == "E101"
        assert ErrorCode.SYN_INVALID_STATEMENT == "E201"
        assert ErrorCode.SYN_INVALID_CONDITION == "E202"
        assert ErrorCode.SYN_MISSING_TERMINATOR == "E203"
        assert ErrorCode.SYN_STRAY_TERMINATOR == "E204"
        assert ErrorCode.SYN_NESTED_CALL == "E205"
        assert ErrorCode.SEM_UNKNOWN_OP == "E301"
        assert ErrorCode.SEM_ARITY == "E302"
        assert ErrorCode.SEM_UNKNOWN_KWARG == "E304"


class TestMarkerAlignment:
    """Tests for the gcc-style caret/gutter alignment."""

    def test_marker_alignment_single_digit(self):
        err = DSLSyntaxError(
            line=2, col=5,
            message="unknown operation 'retrun'",
            source_line="b = retrun(a, 1)",
            error_code="E301",
        )
        lines = format_error(err, use_color=False).splitlines()
        assert lines[2] == "    |     ^~~~~~"

    def test_marker_alignment_double_digit(self):
        err = DSLSyntaxError(
            line=10, col=5,
            message="unknown operation 'retrun'",
            source_line="b = retrun(a, 1)",
            error_code="E301",
        )
        lines = format_error(err, use_color=False).splitlines()
        assert lines[2] == "     |     ^~~~~~"

    def test_token_length_includes_underscore(self):
        assert _estimate_token_length("foo_bar(x)", 0) == 7
        assert _estimate_token_length("a_b", 0) == 3
        assert _estimate_token_length("", 0) == 1
        assert _estimate_token_length("abc", 99) == 1

    def test_no_filename_uses_placeholder(self):
        err = DSLSyntaxError(line=1, col=1, message="boom")
        output = format_error(err, use_color=False)
        assert output.startswith("<dsl>:1:1: error: boom")

    def test_col_clamped_to_line_end(self):
        err = DSLSyntaxError(
            line=1, col=99, message="boom", source_line="abc",
        )
        output = format_error(err, use_color=False)
        # Header keeps the original column
        assert "<dsl>:1:99: error: boom" in output
        # Marker is clamped to the end of the line and still emitted
        source_display = output.splitlines()[1]
        marker = output.splitlines()[2]
        assert source_display == "  1 | abc"
        assert marker.index("^") == len(source_display)

    def test_context_with_source(self):
        source = "a = add(1)\nb = retrun(a, 2)\n"
        err = DSLSyntaxError(
            line=2, col=5,
            message="unknown operation 'retrun'",
            source_line="b = retrun(a, 2)",
        )
        output = format_error(
            err, use_color=False, context_lines=1, source=source,
        )
        assert "  1 | a = add(1)" in output
        assert "\n\n" not in output

    def test_format_context_without_source_ignored(self):
        err = DSLSyntaxError(
            line=2, col=1, message="boom", source_line="abc",
        )
        output = format_error(err, use_color=False, context_lines=2)
        assert "\n\n" not in output
        assert output.splitlines()[1] == "  2 | abc"

    def test_arity_hint_reachable(self):
        hint = _compute_suggestion(
            "add() expects 2 arguments, got 1", "a = add(1)", "E302",
        )
        assert hint == "add() requires exactly 2 arguments"

    def test_spelling_hint_beats_difflib(self):
        hint = _compute_suggestion(
            "unknown operation 'retrun'", "b = retrun(a, 1)", "E301",
        )
        assert hint == "did you mean 'return'?"


class TestCollectorExtensions:
    """Tests for collector dedup, suppression and source context."""

    def test_collector_dedup(self):
        collector = ErrorCollector()
        collector.add(DSLSyntaxError(
            3, 5, "dup", source_line="x", error_code="E301",
        ))
        collector.add(DSLSyntaxError(
            3, 5, "dup", source_line="x", error_code="E301",
        ))
        assert collector.error_count == 1

    def test_collector_suppressed_count(self):
        collector = ErrorCollector(max_errors=3)
        for i in range(10):
            collector.add(DSLSyntaxError(i + 1, 1, f"error {i}"))
        assert collector.error_count == 3
        assert collector.suppressed_count == 7
        assert len(collector.errors) == 3
        assert "7 further errors suppressed" in collector.report()

    def test_suppressed_duplicates_counted_once(self):
        collector = ErrorCollector(max_errors=1)
        collector.add(DSLSyntaxError(1, 1, "stored"))
        duplicate = DSLSyntaxError(2, 2, "suppressed")
        collector.add(duplicate)
        collector.add(duplicate)
        collector.add(duplicate)
        # Repeated suppressed duplicates must not inflate the count.
        assert collector.error_count == 1
        assert collector.suppressed_count == 1
        collector.add(DSLSyntaxError(3, 3, "another"))
        assert collector.suppressed_count == 2
        assert "2 further errors suppressed" in collector.report()

    def test_collector_source_context(self):
        source = "a = add(1)\nb = retrun(a, 2)\n"
        err = DSLSyntaxError(
            line=2, col=5,
            message="unknown operation 'retrun'",
            source_line="b = retrun(a, 2)",
        )
        collector = ErrorCollector(
            use_color=False, source=source, context_lines=1,
        )
        collector.add(err)
        report = collector.report()
        assert "  1 | a = add(1)" in report
        assert "\n\n" not in report

    def test_clear_resets_suppressed(self):
        collector = ErrorCollector(max_errors=1)
        collector.add(DSLSyntaxError(1, 1, "one"))
        collector.add(DSLSyntaxError(2, 1, "two"))
        assert collector.suppressed_count == 1
        collector.clear()
        assert not collector.has_errors
        assert collector.suppressed_count == 0
