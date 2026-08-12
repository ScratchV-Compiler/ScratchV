# DSL Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate structured, location-aware diagnostics into ScratchV's base and extended DSL paths while preserving successful DSL IR output and leaving all non-DSL behavior unchanged.

**Architecture:** Add a DSL-only source/validation layer shared by both parsers. Validation runs before IR generation, reports stable diagnostic codes through `DSLSyntaxError` and `ErrorCollector`, and passes structured diagnostics through only the DSL branches of `CompilerDriver` and the CLI.

**Tech Stack:** Python 3.12-compatible type hints, dataclasses, regular expressions, argparse, pytest.

---

## File map and scope boundary

- Create `scratchv/frontend/dsl_validator.py`: source preservation, shared operator signatures, line validation, block-stack recovery, multi-error collection.
- Modify `scratchv/frontend/dsl_errors.py`: compatible diagnostic spans, plain/automatic rendering, deterministic collector semantics.
- Modify `scratchv/frontend/dsl_parser.py`: pre-IR validation and filename-aware errors; preserve successful IR generation.
- Modify `scratchv/frontend/dsl_extended.py`: use shared validation before extended IR generation; preserve successful control-flow IR.
- Modify `scratchv/frontend/__init__.py`: export new DSL-only public diagnostic interfaces.
- Modify `scratchv/compiler.py`: only DSL parsing and diagnostic result fields.
- Modify `scratchv/main.py`: only structured DSL diagnostic rendering and top-level internal-error exit handling.
- Create `tests/test_dsl_validator.py`: validator and parser integration tests.
- Create `tests/test_dsl_diagnostics_cli.py`: CompilerDriver/CLI DSL diagnostics tests.
- Modify `tests/test_dsl_errors.py`: error model, renderer and collector regression tests.
- Do not modify ONNX parsing, IR definitions, optimizer passes, backends, simulators, benchmarks, or their tests.

### Task 1: Diagnostic model and rendering

**Files:**
- Modify: `scratchv/frontend/dsl_errors.py`
- Test: `tests/test_dsl_errors.py`

- [ ] **Step 1: Write failing tests for compatibility, spans, tabs and automatic color**

Add tests asserting that `DSLSyntaxError` remains catchable as `DSLParseError`, accepts exclusive `end_col`, renders `error[E100]`, expands tabs at four-column stops, keeps `str(error)` ANSI-free, and `render_error(..., use_color=None)` honors `isatty()` and `NO_COLOR`.

- [ ] **Step 2: Verify the new renderer tests fail for missing behavior**

Run: `python -m pytest tests/test_dsl_errors.py -q`

Expected: failures for `end_col`, `render_error`, header placement, tab alignment, and collector limit metadata.

- [ ] **Step 3: Implement the minimal compatible diagnostic API**

Implement these interfaces without changing existing positional constructor compatibility:

```python
@dataclass
class DSLSyntaxError(DSLParseError):
    line: int
    col: int
    message: str
    source_line: str = ""
    filename: Optional[str] = None
    fix_hint: Optional[str] = None
    error_code: Optional[str] = None
    end_col: Optional[int] = None

def render_error(
    err: DSLSyntaxError,
    *,
    stream: TextIO,
    use_color: Optional[bool] = None,
) -> str: ...
```

Keep `format_error()` available, default unknown filenames to `<dsl>`, render codes as `error[E100]:`, and calculate display columns by expanding tabs to four-column stops.

- [ ] **Step 4: Make collector semantics deterministic**

Add `limit_reached`; stop at exactly `max_errors`; do not append a fake line-zero error; deduplicate by `(filename, line, col, error_code, message)`; sort reports by `(line, col, error_code or "")`; reset all state in `clear()`.

- [ ] **Step 5: Verify Task 1 is green**

Run: `python -m pytest tests/test_dsl_errors.py -q`

Expected: all tests in the file pass with no warnings.

### Task 2: Shared source buffer and base DSL validation

**Files:**
- Create: `scratchv/frontend/dsl_validator.py`
- Modify: `scratchv/frontend/dsl_parser.py`
- Test: `tests/test_dsl_validator.py`
- Test: `tests/test_parser.py`

- [ ] **Step 1: Write failing source-buffer and E100/E101/E200/E201 tests**

Cover LF/CRLF, blank input, trailing newline, Unicode and tabs. Add parser-facing cases for `retrun x`, missing call parenthesis, unsupported `ad`, and incorrect arity for unary/binary/keyword operators. Assert exact 1-based line/column, source line, filename and stable code.

- [ ] **Step 2: Verify base-validator RED behavior**

Run: `python -m pytest tests/test_dsl_validator.py tests/test_parser.py -q`

Expected: new tests fail because `SourceBuffer`, `DSLValidator.validate()` and filename-aware `parse()` do not exist.

- [ ] **Step 3: Add the shared syntax facts**

Define immutable `SourceBuffer`, `OpSignature`, base statement patterns, keyword sets, and `OP_SIGNATURES` for every operator dispatched by `DSLParser`. Validate assignment structure with `fullmatch`, split arguments without creating IR, and stop validation of a line after its first structural error.

- [ ] **Step 4: Add base block and signature validation**

Validate `for/endfor`, `return`, identifiers, parentheses, supported operations, positional counts, known/duplicate kwargs and numeric kwargs. Preserve the language rule that first-use variable names are valid inputs.

- [ ] **Step 5: Gate base parsing before IR generation**

Expose:

```python
def validate(
    self,
    text: str,
    *,
    filename: Optional[str] = None,
    max_errors: int = 20,
) -> ErrorCollector: ...

def parse(
    self,
    text: str,
    *,
    filename: Optional[str] = None,
) -> Program: ...
```

`parse()` raises the first `DSLSyntaxError` before resetting and using `IRBuilder`; successful programs retain their previous `Program.dump()` structure.

- [ ] **Step 6: Verify base validation and successful IR are green**

Run: `python -m pytest tests/test_dsl_validator.py tests/test_parser.py tests/test_dsl_errors.py -q`

Expected: all selected tests pass.

### Task 3: Extended block validation and recovery

**Files:**
- Modify: `scratchv/frontend/dsl_validator.py`
- Modify: `scratchv/frontend/dsl_extended.py`
- Test: `tests/test_dsl_validator.py`
- Test: `tests/test_dsl_extended.py`

- [ ] **Step 1: Write failing E110/E111/E112 recovery tests**

Cover stray closers, missing closers, `else` without `if`, duplicate `else`, `while ... endif`, `if ... while ... endif`, nested blocks at EOF, and three independent errors in one file. Assert mismatched closers do not also create recovered-block E111 errors.

- [ ] **Step 2: Verify extended-validator RED behavior**

Run: `python -m pytest tests/test_dsl_validator.py tests/test_dsl_extended.py -q`

Expected: new block-validation cases fail under the current silent-EOF behavior.

- [ ] **Step 3: Implement one shared block stack**

Use `BlockFrame(kind, line, col, saw_else)` in the validator. Match `endif/endwhile/endfor` against the stack; on mismatch, emit one E110 and recover by popping through an outer matching block or treating the closer as recovery for the top frame. Emit E111 only for frames left at EOF.

- [ ] **Step 4: Gate extended parsing with the shared validator**

Override validation only to enable extended rules, then validate before any labels, blocks or instructions are created. Keep the existing successful `if/else/while/for` IR path and reset parser state on every call.

- [ ] **Step 5: Verify extended validation and IR regression**

Run: `python -m pytest tests/test_dsl_validator.py tests/test_dsl_extended.py tests/test_parser.py -q`

Expected: all selected tests pass.

### Task 4: DSL-only CompilerDriver and CLI integration

**Files:**
- Modify: `scratchv/compiler.py`
- Modify: `scratchv/main.py`
- Modify: `scratchv/frontend/__init__.py`
- Create: `tests/test_dsl_diagnostics_cli.py`

- [ ] **Step 1: Write failing CompilerDriver and CLI tests**

Test a temporary bad `.dsl` through `CompilerDriver.compile()` and `main(argv)`. Assert `success is False`, `diagnostics` contains the original `DSLSyntaxError`, `errors` contains the complete plain rendering without `Parse error:` duplication, CLI exits 1, redirected stderr has no ANSI, and missing `endif` is not overwritten by a base-parser error.

- [ ] **Step 2: Verify integration RED behavior**

Run: `python -m pytest tests/test_dsl_diagnostics_cli.py -q`

Expected: failures for missing structured result fields, broad parser fallback and CLI rendering.

- [ ] **Step 3: Add compatible result metadata**

Append defaulted `diagnostics`, `diagnostic_limit_reached`, and `diagnostic_limit` fields to `CompileResult`. In the DSL branch only, use `ExtendedDSLParser.parse(text, filename=input_path)` and preserve `DSLSyntaxError` without broad fallback. Do not alter ONNX parsing or successful result semantics.

- [ ] **Step 4: Render diagnostics once at the CLI boundary**

When `result.diagnostics` is non-empty, render them to `sys.stderr` with automatic color and do not print `result.errors` again. Preserve exit 1 for input diagnostics. Convert unexpected top-level exceptions to `internal compiler error` and exit 2 without traceback in normal mode.

- [ ] **Step 5: Verify end-to-end DSL diagnostics**

Run: `python -m pytest tests/test_dsl_diagnostics_cli.py tests/test_dsl_errors.py tests/test_dsl_validator.py tests/test_parser.py tests/test_dsl_extended.py -q`

Expected: all selected tests pass with no ANSI in captured output.

### Task 5: Scope, regression and project verification

**Files:**
- Modify only files listed in the file map if verification exposes a DSL-specific defect.

- [ ] **Step 1: Run every DSL-related test**

Run: `python -m pytest tests/test_dsl_errors.py tests/test_dsl_validator.py tests/test_dsl_diagnostics_cli.py tests/test_parser.py tests/test_dsl_extended.py -q`

Expected: all pass.

- [ ] **Step 2: Run L1/L2 Harness if present**

Run: `python .Codex/harness/verify/run.py --level L1` and `python .Codex/harness/verify/run.py --level L2`.

If the Harness is absent, record that fact and run the repository-equivalent full command `python -m pytest tests -q` in the project environment containing declared dependencies.

- [ ] **Step 3: Prove the scope boundary**

Run: `git diff --name-only` and `git diff --check`.

Expected: only the DSL files, DSL sections of `compiler.py`/`main.py`, DSL tests, and this plan appear; no whitespace errors.

- [ ] **Step 4: Self-review diagnostic requirements**

Confirm E100, E101, E110, E111, E200 and E201; three-error collection; no partial IR; no parser fallback masking; plain redirected output; successful DSL IR regression; and no non-DSL behavior changes.

- [ ] **Step 5: Update shared memory only if a novel reusable lesson exists**

If `memory/memory.md` exists, merge one non-duplicate entry in `[2026-08-12] ...` format. If it does not exist, record the missing Harness facility and do not create unrelated infrastructure in this DSL patch.

- [ ] **Step 6: Stage, commit and push after all verification is green**

Run `git add` with the explicit reviewed file list, commit with an English message such as `feat: integrate structured DSL diagnostics`, then push the current branch without force.
