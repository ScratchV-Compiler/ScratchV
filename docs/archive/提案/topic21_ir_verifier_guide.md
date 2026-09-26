# IR Verifier - User Guide

## Overview

The `IRVerifier` in `scratchv/analysis/ir_verifier.py` validates ScratchV IR programs against a set of correctness rules. It produces a list of verification errors and warnings, designed to be run before and after optimization passes.

## Verification Rules

| Rule               | Level   | Description                                      |
|--------------------|---------|--------------------------------------------------|
| def-before-use     | ERROR   | All value operands must be defined before use    |
| label-existence    | ERROR   | Branch/jump targets must exist as block labels   |
| block-termination  | ERROR   | Every basic block must end with a terminator     |
| type-consistency   | WARNING | Operands of binary/NN ops must have compatible types |
| control-flow-integrity | ERROR | Unreachable instructions after branches/returns |
| ssa-validity       | ERROR   | Each value must be assigned exactly once (SSA)   |
| entry-existence    | ERROR   | Function must have at least one basic block      |

## Usage

### Basic Verification

```python
from scratchv.analysis.ir_verifier import IRVerifier
from scratchv.frontend.dsl_parser import DSLParser

parser = DSLParser()
program = parser.parse(dsl_source)

verifier = IRVerifier(program)
errors = verifier.verify()

if errors:
    for err in errors:
        print(err)
    raise SystemExit(1)
else:
    print("IR verification passed.")
```

### Programmatic Verification

```python
from scratchv.analysis.ir_verifier import verify_ir

passed, errors = verify_ir(program)
if not passed:
    print(f"Found {len(errors)} verification error(s)")
```

### Integration into Compiler Pipeline

```python
# Parse
program = parse_input(source)
verifier = IRVerifier(program)
pre_errors = verifier.verify()
assert not pre_errors, f"IR invalid before optimization: {pre_errors}"

# Optimize
run_optimizations(program)

# Verify after optimization
post_errors = IRVerifier(program).verify()
assert not post_errors, f"Optimization produced invalid IR: {post_errors}"

# Codegen
generate_code(program)
```

## Error Levels

- **ERROR**: Definite correctness issue that will cause incorrect compilation or runtime failures.
- **WARNING**: Potential issue that may or may not cause problems (e.g., type mismatches that might be intentional).

## VerificationError

```python
@dataclass
class VerificationError:
    level: ErrorLevel           # ERROR or WARNING
    message: str                # Human-readable description
    function_name: str | None   # Containing function
    block_name: str | None      # Containing basic block
    instruction_index: int | None  # Index of instruction
    value_name: str | None      # Problematic value name
    rule: str | None            # Rule identifier
```

Example formatted output:
```
[ERROR] (def-before-use) in 'main', block 'entry', instr #2, value 'c': value 'c' used before definition
[ERROR] (block-termination) in 'main', block 'loop_body': block does not end with a terminator
[WARNING] (type-consistency) in 'main', block 'entry', instr #0: operand type mismatch
```

## Rule Details

### Def-Before-Use

Checks that every value operand has been defined (assigned) in a previous instruction or passed as a function parameter. Constants are auto-defined on first use.

### Label Existence

Checks that all branch/jump targets (in `BR`, `BR_IF` instructions) reference existing basic block names in the same function.

### Block Termination

Every basic block must end with one of: `RETURN`, `BR`, `BR_IF`. Blocks without a terminator are flagged as errors; empty blocks are warnings.

### Type Consistency

For binary operations (`ADD`, `SUB`, `MUL`, `DIV`) and NN operations (`MATMUL`, `DOT`, `CONV`), both operands should have the same DataType. Mismatches produce warnings.

### Control Flow Integrity

- Unconditional jump (`BR`) must be the last instruction in its block.
- Conditional branch (`BR_IF`) must have exactly two targets (comma-separated).
- `RETURN` must be the last instruction in its block.

### SSA Validity

Each value (by name) must be assigned exactly once (SSA property). Multiple assignments to the same name are errors.

## CLI Integration

Add `--verify-ir` flag to the CLI:

```python
parser.add_argument("--verify-ir", action="store_true",
                    help="Validate IR after each pass")

if args.verify_ir:
    errors = IRVerifier(program).verify()
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        sys.exit(1)
```

## Extending the Verifier

To add a new verification rule:

1. Add a method to `IRVerifier` with the pattern `_check_<rule_name>()`
2. Call it from `_verify_function()`
3. Add error reporting using `_add_error()`

Example:
```python
def _check_my_rule(self, func: Function) -> None:
    for block in func.blocks:
        for i, instr in enumerate(block.instructions):
            if my_condition_violated(instr):
                self._add_error(
                    ErrorLevel.WARNING,
                    "custom rule violation message",
                    func_name=func.name,
                    block_name=block.name,
                    instruction_index=i,
                    rule="my-custom-rule",
                )
```

## See Also

- `scratchv/ir/types.py` - IR data structures
- `scratchv/analysis/cfg_builder.py` - CFG analysis
- `scratchv/optimizer/` - Optimization passes that should run verification
