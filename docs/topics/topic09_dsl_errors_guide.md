# DSL Error Beautifier - User Guide

## Overview

The `dsl_errors` module (`scratchv/frontend/dsl_errors.py`) provides gcc/clang-style error formatting for DSL syntax errors, with ANSI color support and fix suggestions.

## Components

### DSLSyntaxError

An enriched exception class with precise location information:

```python
from scratchv.frontend.dsl_errors import DSLSyntaxError

err = DSLSyntaxError(
    line=5,
    col=12,
    message="unexpected token 'retrun'",
    source_line="result = retrun(x)",
    filename="test.dsl",
)
```

Fields:
- `line` (int): 1-based line number
- `col` (int): 1-based column number
- `message` (str): Human-readable error description
- `source_line` (str): Content of the erroneous line
- `filename` (str, optional): Source filename
- `fix_hint` (str, optional): Suggested fix
- `error_code` (str, optional): Error code for categorization

### format_error()

Formats a DSLSyntaxError as a gcc/clang-style message:

```python
from scratchv.frontend.dsl_errors import format_error

print(format_error(err, use_color=True))
```

Output:
```
test.dsl:5:12: error: unexpected token 'retrun'
  5 | result = retrun(x)
    |          ^~~~~~~
note: did you mean 'return'?
```

Parameters:
- `use_color` (bool): Enable ANSI colors (default True)
- `context_lines` (int): Number of context lines before the error line (default 0)

### ErrorCollector

Collects multiple errors before reporting, allowing the parser to continue after the first error:

```python
from scratchv.frontend.dsl_errors import ErrorCollector

collector = ErrorCollector(filename="test.dsl", max_errors=20)

try:
    result = do_something()
except DSLSyntaxError as e:
    collector.add(e)

# Or add errors directly:
collector.add_error(
    line=10,
    col=5,
    message="missing closing ')'",
    source_line="c = add(a, b",
)

# Report all errors at once
print(collector.report())

# Or report and exit on error
collector.report_and_exit()
```

## Fix Suggestions

The error beautifier includes a built-in suggestion database for common mistakes:

| Error pattern  | Suggestion                     |
|----------------|--------------------------------|
| `retrun`       | did you mean 'return'?         |
| `endiff`       | did you mean 'endif'?          |
| `endwhie`      | did you mean 'endwhile'?       |
| `reul`         | did you mean 'relu'?           |
| missing paren  | missing closing ')'            |
| unterminated   | missing 'endif', 'endwhile', or 'endfor' |

## ANSI Color Scheme

| Element   | Color       |
|-----------|-------------|
| Location  | Bold white  |
| `error`   | Red         |
| Source    | Gray        |
| Marker ^  | Green       |
| `note`    | Cyan        |

## Integration with Parser

To integrate the error beautifier into the DSL parser:

```python
from scratchv.frontend.dsl_errors import DSLSyntaxError, make_error

# In parser code:
if error_condition:
    raise make_error(
        line=current_line,
        col=current_col,
        message="unexpected token",
        source_line=raw_line,
        filename=self.filename,
    )
```

## See Also

- `scratchv/frontend/dsl_parser.py` - Base DSL parser
- `scratchv/frontend/dsl_extended.py` - Extended parser with if/while
