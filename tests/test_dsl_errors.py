"""Tests for the DSL error beautifier module."""

import pytest
from scratchv.frontend.dsl_errors import (
    DSLSyntaxError,
    format_error,
    ErrorCollector,
    make_error,
    Color,
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
        assert "E001" in output

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
        # Should only have max_errors items
        assert len(collector.errors) <= 4  # 3 real errors + 1 limit message

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


class TestColor:
    """Tests for ANSI color definitions."""

    def test_color_values_not_empty(self):
        for color in Color:
            assert color.value != ""

    def test_color_reset(self):
        assert Color.RESET.value == "\033[0m"
