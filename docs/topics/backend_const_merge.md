# Constant Load Merge Optimization

## Overview

The Constant Load Merge Optimizer (`scratchv.backend.const_merge`) detects and optimizes RISC-V instruction sequences that load 32-bit constants. It performs two passes:

1. **lui+addi merging**: Combines `lui rd, imm_hi` + `addi rd, rd, imm_lo` into a single `li rd, full_value`.
2. **Redundant lui elimination**: Removes duplicate `lui` instructions that load the same upper immediate into the same register.

## API

```python
from scratchv.backend.const_merge import merge_constants

optimized_asm, changes = merge_constants(asm_text)
```

### `merge_constants(asm_text) -> tuple[str, int]`

**Parameters:**
- `asm_text` (`str`): Input RISC-V assembly text.

**Returns:** Tuple of `(optimized_assembly, number_of_changes)`.

## How It Works

### RISC-V Constant Loading

RISC-V loads 32-bit constants using two instructions:
```
lui rd, imm_hi       # rd = imm_hi << 12   (upper 20 bits)
addi rd, rd, imm_lo  # rd = rd + imm_lo   (lower 12 bits, sign-extended)
```

The final value is: `(imm_hi << 12) + sign_extend_12(imm_lo)`

### Pass 1: lui+addi Merging

Detects adjacent `lui` followed by `addi` where:
- The destination register matches (`lui rd` == `addi rd`)
- The addi reads from the same register (`addi rd, rd, imm_lo`)

Computes the full 32-bit constant and replaces with `li rd, final_value`.

### Pass 2: Redundant lui Elimination

Tracks the last `lui` value loaded into each register. If a subsequent `lui` loads the same upper immediate into the same register without the register being modified, it is removed.

## Example

### Input
```asm
  lui t0, 0x12345
  addi t0, t0, -256
  ...
  lui t0, 0x12345    ; redundant!
  addi t0, t0, 100
```

### Output
```asm
  li t0, 0x12344F00  # merged lui+addi -> 305418496
  ...
  # peephole: removed redundant lui t0, 0x12345
  addi t0, t0, 100
```

## Sign Extension

RISC-V `addi` sign-extends the 12-bit immediate. For example:
- `lui t0, 0x12345` loads `0x12345000`
- `addi t0, t0, 0x800` adds `-2048` (sign-extended from bit 11)
- Final value: `0x12345000 + (-2048) = 0x12344800`

The optimizer correctly handles sign extension when computing the merged constant.

## CLI Usage

```bash
python -m scratchv.backend.const_merge input.s -o output.s -v
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `input` | Input assembly file |
| `-o, --output` | Output file (default: stdout) |
| `-v, --verbose` | Print optimization statistics to stderr |
