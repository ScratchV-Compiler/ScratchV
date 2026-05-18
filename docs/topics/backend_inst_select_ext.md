# Extended Instruction Selector

## Overview

The Extended Instruction Selector (`scratchv.backend.inst_select_ext`) builds on the base `InstructionSelector` to add support for additional RISC-V operations and float64 (double-precision) data types.

## API

```python
from scratchv.backend.inst_select_ext import ExtendedInstructionSelector

selector = ExtendedInstructionSelector(program, enable_fp64=True)
machine_instrs = selector.run()
```

### `ExtendedInstructionSelector(program, enable_fp64=True, use_hardware_sqrt=False)`

**Parameters:**
- `program` (`Program`): The ScratchV IR Program to select instructions for.
- `enable_fp64` (`bool`): Enable float64 (D extension) support.
- `use_hardware_sqrt` (`bool`): Use `fsqrt.s`/`fsqrt.d` hardware instructions instead of library calls.

### `supported_ops` (property)

Returns a list of all supported opcodes.

## New Operations

### sqrt

Computes square root. Two modes:
- **Library call**: Emits `mv a0, src; call sqrtf; mv dst, a0` for float32, or `call sqrt` for float64.
- **Hardware**: Uses `fsqrt.s` or `fsqrt.d` if `use_hardware_sqrt=True`.

### min / max

- **Integer min**: Branchless sequence using `slt; sub; and; add`.
- **Integer max**: Uses the existing `MAX` pseudo-instruction from the base selector.
- **Float64**: Uses `fmin.d` / `fmax.d` hardware instructions.

Integer min branchless implementation:
```
slt tmp, a, b       # tmp = (a < b) ? 1 : 0
sub diff, b, a      # diff = b - a
and tmp, tmp, diff  # mask = tmp & diff
add dst, a, tmp     # dst = a + mask
```
This works because: if a < b, mask = b - a, so dst = a + (b - a) = b. If a >= b, mask = 0, so dst = a.

### abs

- **Integer abs**: `srai 31 + xor + sub` branchless sequence.
- **Float64**: Uses `fabs.d` hardware instruction.

### Integer Division (div, rem, mod)

Uses native RISC-V M-extension instructions:
- `div rd, rs1, rs2` for integer division
- `rem rd, rs1, rs2` for remainder
- `mod` is mapped to `rem` for non-negative cases

## Float64 (D Extension) Support

When `enable_fp64=True`, the extended selector automatically overrides arithmetic operations for float64 typed values:

| IR Op | RISC-V Instruction |
|-------|-------------------|
| `add` on f64 | `fadd.d` |
| `sub` on f64 | `fsub.d` |
| `mul` on f64 | `fmul.d` |
| `div` on f64 | `fdiv.d` |
| `neg` on f64 | `fneg.d` |
| `load` of f64 | `fld` |
| `store` of f64 | `fsd` |
| `load_const` of f64 | `li.d` (pseudo) |
| f64 comparison (lt) | `flt.d` |
| f64 comparison (eq) | `feq.d` |
| f64 -> f32 conversion | `fcvt.s.d` |
| f32 -> f64 conversion | `fcvt.d.s` |

## New MachineOp Codes

The extended selector adds these opcodes to the `MachineOp` enum at import time:

| OpCode | RISC-V Mnemonic |
|--------|----------------|
| `SQRT_S` | `fsqrt.s` |
| `SQRT_D` | `fsqrt.d` |
| `FMIN_D` | `fmin.d` |
| `FMAX_D` | `fmax.d` |
| `FABS_D` | `fabs.d` |
| `FNEG_D` | `fneg.d` |
| `FADD_D` | `fadd.d` |
| `FSUB_D` | `fsub.d` |
| `FMUL_D` | `fmul.d` |
| `FDIV_D` | `fdiv.d` |
| `FLT_D` | `flt.d` |
| `FEQ_D` | `feq.d` |
| `FCVT_S_D` | `fcvt.s.d` |
| `FCVT_D_S` | `fcvt.d.s` |
| `FLD` | `fld` |
| `FSD` | `fsd` |
| `SRAI` | `srai` |
| `XOR` | `xor` |
| `AND` | `and` |
| `SLT` | `slt` |
| `REM` | `rem` |

## Integration

The extended selector is a drop-in replacement for the base `InstructionSelector`:

```python
# Before:
from scratchv.backend import InstructionSelector
selector = InstructionSelector(program)

# After:
from scratchv.backend.inst_select_ext import ExtendedInstructionSelector
selector = ExtendedInstructionSelector(program)
```

All existing IR opcodes that the base selector handles continue to work.
