# RISC-V Instruction Counter

## Overview

The RISC-V Instruction Counter (`scratchv.backend.inst_counter`) parses `.s` assembly files and produces categorized instruction statistics with text tables, matplotlib charts, and HTML reports.

## API

```python
from scratchv.backend.inst_counter import count_instructions

counts = count_instructions(asm_text)
# {'ALU': 42, 'MEM': 15, 'BRANCH': 8, 'JUMP': 3, 'PSEUDO': 10, 'MISC': 2}
```

### `count_instructions(asm_text) -> dict[str, int]`

Parse assembly text and return a dictionary mapping category to instruction count. All six standard categories are always present (even if 0). An additional `_detailed` key holds a `collections.Counter` of per-opcode counts.

### Instruction Categories

| Category | Examples |
|----------|----------|
| ALU | `add`, `sub`, `mul`, `div`, `addi`, `lui`, `xor`, `and`, etc. |
| MEM | `lw`, `sw`, `lh`, `sh`, `lb`, `sb`, `flw`, `fsw` |
| BRANCH | `beq`, `bne`, `blt`, `bge`, `bnez`, etc. |
| JUMP | `j`, `jal`, `jalr`, `ret` |
| PSEUDO | `li`, `mv`, `call`, `nop`, `la`, `not`, etc. |
| MISC | Everything else (directives, custom instructions) |

### `format_table(counts) -> str`

Produce a human-readable text table from count results.

### `generate_chart(counts, output_path, title=...) -> None`

Generate a bar chart + pie chart using matplotlib.

### `generate_html_report(counts, output_path, title=...) -> None`

Generate a standalone HTML report with tables and CSS styling.

### `compare_files(filepaths) -> ComparisonResult`

Compare instruction counts across multiple files. Returns a `ComparisonResult` object with `counts` and `diffs` dictionaries and a `to_dataframe()` method (requires pandas).

## CLI Usage

```bash
# Single file
python -m scratchv.backend.inst_counter program.s --chart stats.png --html report.html

# Multi-file comparison
python -m scratchv.backend.inst_counter program1.s program2.s --compare

# Verbose breakdown
python -m scratchv.backend.inst_counter program.s -v
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `files` | One or more assembly files |
| `--chart PATH` | Save bar+pie chart image |
| `--html PATH` | Save HTML report |
| `--compare` | Multi-file side-by-side comparison mode |
| `--verbose, -v` | Show per-instruction breakdown |

## Adding New Instruction Mappings

To add a new instruction to the categorizer, add an entry to the `_OPCODE_CATEGORIES` dict in `inst_counter.py`:

```python
_OPCODE_CATEGORIES["fcvt.s.d"] = "MISC"
```

Or create a new category by adding it to `_CATEGORY_ORDER` and mapping instructions to it.
