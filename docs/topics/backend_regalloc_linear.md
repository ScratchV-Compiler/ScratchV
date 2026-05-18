# Linear Scan Register Allocator

## Overview

The Linear Scan Register Allocator (`scratchv.backend.regalloc_linear`) implements a classic linear scan register allocation algorithm for individual RISC-V basic blocks. It computes live intervals for all virtual registers, allocates physical registers greedily, and generates spill code when registers run out.

## API

```python
from scratchv.backend.regalloc_linear import LinearScanAllocator, LsInstruction

allocator = LinearScanAllocator(phys_regs=["a0", "a1", "t0", "t1", "t2"])
intervals = allocator.compute_live_intervals(block_instructions)
allocation = allocator.allocate(intervals)
code = allocator.get_allocated_code(block_instructions)
```

### `LinearScanAllocator(phys_regs=None)`

**Parameters:**
- `phys_regs` (`list[str] | None`): Physical registers available for allocation. Defaults to all RISC-V integer registers (excluding `x0`, `sp`, `gp`, `tp`, `ra`).

### `compute_live_intervals(block) -> list[LiveInterval]`

Compute live intervals for all virtual registers in a basic block. Returns a list of `LiveInterval` objects sorted by start position.

Each `LiveInterval` contains:
- `vreg` (`str`): Virtual register name
- `start` (`int`): Instruction index of first definition
- `end` (`int`): Instruction index of last use (exclusive)
- `uses` (`set[int]`): Set of instruction indices where used

### `allocate(intervals) -> dict[str, str]`

Perform linear scan allocation. Returns a dict mapping virtual register names to physical register names.

### `spill(var) -> str | None`

Select a register to spill when no free registers are available. Uses the "farthest end" heuristic: spills the active interval that ends latest. This is called automatically during `allocate()`.

### `get_allocated_code(block) -> str`

Generate RISC-V assembly code with physical registers and inserted spill code.

## Algorithm

### Linear Scan Overview

1. **Compute live intervals**: Traverse the basic block, recording where each virtual register is first defined (start) and last used (end).

2. **Sort intervals**: Sort all intervals by increasing start position.

3. **Scan and allocate**:
   - For each interval in order:
     - Expire intervals from the active list whose `end <= current.start`
     - If a free register exists, assign it
     - If no free register, spill the active interval with the farthest end
   - Emit spill code: `sw` after definition, `lw` before use

### Spill Heuristic

When all physical registers are occupied, the allocator must evict one. It chooses the interval with the latest end position among active intervals, as this minimizes the total number of spills.

### Spill Code Generation

- **After definition**: Insert `sw reg, -N(sp)` to store the value to stack.
- **Before use**: Insert `lw reg, -N(sp)` to reload the value from stack.

## RISC-V Register Set

The default allocatable registers are 25 integer registers:

| Group | Registers | Count |
|-------|-----------|-------|
| Argument/temp | `a0-a7`, `t0-t6` | 15 |
| Saved | `s0-s11` | 12 |
| **Total** | | **27** (excluding x0, sp, gp, tp, ra) |

## Example

```python
from scratchv.backend.regalloc_linear import LinearScanAllocator, LsInstruction

# Build a block of instructions with virtual registers
block = [
    LsInstruction(0, "add", ["v1", "v2", "v3"], defines={"v1"}, uses={"v2", "v3"}),
    LsInstruction(1, "mul", ["v4", "v1", "v5"], defines={"v4"}, uses={"v1", "v5"}),
    LsInstruction(2, "add", ["v6", "v4", "v1"], defines={"v6"}, uses={"v4", "v1"}),
]

allocator = LinearScanAllocator()
intervals = allocator.compute_live_intervals(block)
mapping = allocator.allocate(intervals)
print(allocator.report())
```
