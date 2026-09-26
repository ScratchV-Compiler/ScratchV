# Compiler Logger - User Guide

## Overview

The `scratchv.utils.logger` module provides structured, color-coded logging for the ScratchV compiler pipeline, replacing ad-hoc print() calls with proper log infrastructure.

## Quick Start

```python
from scratchv.utils.logger import init_logger, get_logger

# Initialize once at program start
init_logger(level="DEBUG", log_file="build.log")

# Get a logger for your module
log = get_logger("parser")
log.info("Parsing DSL source (%d lines)", len(lines))
log.debug("Line: %s", line)
log.warning("Unexpected token at line %d", lineno)
log.error("Parse failed: %s", error)
```

## Log Levels

| Level    | Purpose                                    | Console Color |
|----------|--------------------------------------------|---------------|
| DEBUG    | Detailed tracing for debugging             | Gray          |
| INFO     | Normal operational messages                | Green         |
| WARNING  | Non-critical issues                        | Yellow        |
| ERROR    | Errors that prevent completion             | Red           |
| CRITICAL | Fatal errors requiring immediate attention | Bold Red      |

## API Reference

### init_logger()

```python
def init_logger(
    level: str = "INFO",
    log_file: str | None = None,
    use_color: bool = True,
) -> None
```

Initializes the logging system. Must be called once at startup.

- `level`: One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
- `log_file`: Optional file path for log output (plain text, no color)
- `use_color`: Enable ANSI color output on console

### get_logger()

```python
def get_logger(name: str) -> logging.Logger
```

Get or create a named logger. If `init_logger()` hasn't been called, it auto-initializes with defaults.

- `name`: Logger name (e.g., `"parser"`, `"optimizer.peephole"`)
- The `"scratchv."` prefix is auto-prepended if missing.

### set_level()

```python
def set_level(level: str) -> None
```

Change the log level at runtime. Useful when toggling debug mode.

### shutdown()

```python
def shutdown() -> None
```

Flush and close all logging handlers. Call before exiting.

## Progress Indicators

### log_phase() (Context Manager)

```python
from scratchv.utils.logger import log_phase

with log_phase("parse", "Parsing DSL input"):
    program = parser.parse(source)
# Output: 12:34:56 INFO [scratchv.parse] Parsing DSL input... done (0.032s)
```

Automatically logs start, completion time, and failure if an exception occurs.

### log_progress()

```python
log_progress("optimize", current=5, total=10, description="Optimizing functions")
# Output: 12:34:57 INFO [scratchv.optimize] Optimizing functions [5/10] 50.0%
```

### log_step()

```python
log_step("codegen", "Selecting instructions")
# Output: 12:34:58 DEBUG [scratchv.codegen] -> Selecting instructions
```

## Module Convention

Each compiler module should have its own logger:

```python
import logging
from scratchv.utils.logger import get_logger

log = get_logger(__name__)  # uses module path as logger name
```

## Output Format

Console (with color):
```
HH:MM:SS LEVEL    [scratchv.module] Message text
```

File (plain text, always at DEBUG level):
```
YYYY-MM-DD HH:MM:SS LEVEL    [scratchv.module] Message text
```

## Replacing print() Calls

Before:
```python
print(f"Parsing {filename}...")
print(f"Error: {msg}", file=sys.stderr)
```

After:
```python
log.info("Parsing %s...", filename)
log.error("%s", msg)
```

## CLI Integration

Add to `main.py`:

```python
parser.add_argument("--log-level", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
                    default="INFO", help="Logging verbosity")
parser.add_argument("--log-file", default=None, help="Write logs to file")

# In main():
init_logger(level=args.log_level, log_file=args.log_file)
```

## See Also

- `scratchv/main.py` - CLI entry point for log configuration
- `scratchv/frontend/dsl_parser.py` - Example module using the base parser
