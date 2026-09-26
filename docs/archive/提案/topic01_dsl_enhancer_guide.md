# DSL Frontend Enhancer - User Guide

## Overview

The `ExtendedDSLParser` in `scratchv/frontend/dsl_extended.py` extends the base `DSLParser` to support conditional branching (`if/else`) and loop constructs (`while`). It generates proper IR with labels, conditional branches, and loop structures.

## New Syntax

### if/else

```
if (condition):
    # then-body
else:
    # else-body
endif
```

- The condition is enclosed in parentheses and supports `==`, `!=`, `<`, `>`, `<=`, `>=`.
- Both operands can be variable names or numeric literals.
- The `else:` branch is optional.
- Blocks end with `endif`.

Example:
```
if (a > b):
    c = add(a, b)
else:
    c = mul(a, b)
endif
return c
```

### while

```
while (condition):
    # loop body
endwhile
```

- Condition syntax is the same as `if`.
- The loop evaluates the condition before each iteration, branching to the body or exit.
- Nested `while` and `if` are supported.

Example:
```
while (i < 10):
    acc = add(acc, x)
endwhile
return acc
```

## Usage

```python
from scratchv.frontend.dsl_extended import ExtendedDSLParser

parser = ExtendedDSLParser()
program = parser.parse(dsl_source_text)
```

## Generated IR Structure

### if/else IR

```
entry:
    cmp = cmp(a, b)            # comparison
    br_if cmp -> if_then, if_else
.if_then:
    c = add a b
    br -> if_end
.if_else:
    c = mul a b
    br -> if_end
.if_end:
    ret c
```

### while IR

```
entry:
    br -> while_hdr
.while_hdr:
    cmp = cmp(i, 10)
    br_if cmp -> while_body, while_exit
.while_body:
    acc = add acc x
    br -> while_hdr
.while_exit:
    ret acc
```

## Nested Constructs

Both `if` and `while` can be nested arbitrarily:

```
if (a > 0):
    while (i < 10):
        t = mul(a, i)
        acc = add(acc, t)
    endwhile
else:
    c = sub(a, b)
endif
return acc
```

The parser uses unique label counters to ensure no label collisions in nested constructs.

## Limitations

- Comparison operands must be existing variable names or numeric literals (no nested arithmetic in conditions).
- Boolean operators (`&&`, `||`) are not yet supported.
- The `for` loop from the base DSLParser is still available and can be combined with `if/else` and `while`.

## See Also

- `scratchv/frontend/dsl_parser.py` - Base DSL parser
- `scratchv/ir/builder.py` - IR builder used for code generation
- `docs/topics/topic09_dsl_errors_guide.md` - Error beautifier for DSL
