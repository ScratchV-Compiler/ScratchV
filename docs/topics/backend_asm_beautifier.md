# RISC-V Assembly Beautifier

## Overview

The RISC-V Assembly Beautifier (`scratchv.backend.asm_beautifier`) parses RISC-V assembly text and outputs a neatly formatted, aligned version with semantic comments and section headers. It improves readability of generated `.s` files significantly without changing the semantics.

## API

```python
from scratchv.backend.asm_beautifier import beautify_asm

pretty = beautify_asm(raw_asm_text, align=True, add_comments=True)
```

### `beautify_asm(asm_text, align=True, add_comments=True) -> str`

**Parameters:**
- `asm_text` (`str`): Raw RISC-V assembly source text.
- `align` (`bool`): If True, align labels, opcodes, and operands into fixed-width columns.
- `add_comments` (`bool`): If True, add semantic comments to each instruction line.

**Returns:** Formatted assembly string.

### `beautify_file(input_path, output_path=None, align=True, add_comments=True) -> str`

Read a `.s` file, beautify it, optionally write to output.

## Features

### Column Alignment
- Labels: left-aligned (up to 30 chars)
- Opcodes: fixed width 8-12 chars
- Operands: left-aligned (up to 40 chars)

### Semantic Comment Templates
Over 60 RISC-V instructions have human-readable comment templates:

| Instruction | Comment |
|-------------|---------|
| `add rd, rs1, rs2` | `rd = rs1 + rs2` |
| `lw rd, 0(rs1)` | `rd = MEM[rs1 + 0]` |
| `beq rs1, rs2, label` | `if rs1 == rs2 goto label` |
| `j label` | `goto label` |
| `li rd, imm` | `rd = imm` |

### Section Headers
Automatically detects `.text`, `.data`, `.bss`, `.rodata` directives and inserts:
```
# ============================================================
#  CODE SECTION
# ============================================================
```

### Function Headers
Detects function entry labels and inserts descriptive comments before each function.

## CLI Usage

```bash
python -m scratchv.backend.asm_beautifier input.s -o output.s
python -m scratchv.backend.asm_beautifier input.s --no-align --no-comments
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `input` | Input assembly file (required) |
| `-o, --output` | Output file (default: stdout) |
| `--no-align` | Disable column alignment |
| `--no-comments` | Disable semantic comments |

## Customizing Comment Templates

The comment template dictionary `_INST_COMMENTS` in `asm_beautifier.py` can be extended. Each entry maps a lowercase opcode mnemonic to a format string with the following placeholders:

- `{rd}`: Destination register
- `{rs1}`: First source register
- `{rs2}`: Second source register
- `{imm}`: Immediate value

Example:
```python
_INST_COMMENTS["fcvt.s.d"] = "{rd} = (float){rs1}  # f64 -> f32"
```
