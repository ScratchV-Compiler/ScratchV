# Optimization Passes Guide

Six beginner-friendly optimization passes for ScratchV, ordered by difficulty.
Each pass can be implemented as a standalone task.

All IR passes define a stable `name` and implement
`OptimizationPass.optimize(self, program: Program) -> int`. The return value is
the non-negative number of transformations made by the current invocation.
Use the canonical factory when selecting an optimization level:

```python
from scratchv.compiler import create_optimization_pass_manager

manager = create_optimization_pass_manager("all")
report = manager.run(program)
print(report.total_changes)
```

`manager.run()` returns an immutable `OptimizationReport`. Its ordered
`executions` contain each pass name, change count, and elapsed time;
`total_changes` and `elapsed_seconds` summarize the complete pipeline.

The main CLI accepts `--opt-level none|basic|all`; `--optimize` remains a
compatibility alias. The standalone LLVM tool has a separate numeric
`--opt-level 0|1|2|3` option.

---

## 1. Constant Folding (⭐)

**Already implemented** in `scratchv/optimizer/constant_folding.py`.

Compile-time evaluation of constant expressions:
```
a = 3      →  (folded during IR construction)
b = 5
c = add(a, b)  →  c = 8 (replaced with load_const)
```

---

## 2. Dead Code Elimination (⭐⭐)

**Already implemented** in `scratchv/optimizer/dead_code.py`.

Removes instructions whose results are never referenced:
```
t1 = mul(a, b)     # no subsequent read of t1 → DELETE
t2 = add(t1, c)     # t2 is read by ret → KEEP
ret t2
```

---

## 3. Mul-Add Fusion / Instruction Combining (⭐)

Combines a `mul` followed by `add` into a combined operation
(reduces temporary register pressure):

```python
# Before                    # After
tmp = mul(a, b)              sum = mul_add(a, b, sum)
sum = add(tmp, sum)
```

**Implementation**: Pattern-match in IR:
```
IF: instruction[i] is MUL(dst, a, b)
AND instruction[i+1] is ADD(sum, dst, sum)
THEN: replace with single MUL_ADD(dst, a, b, sum) pseudo-op
```

---

## 4. Peephole Optimization (⭐)

Scans assembly for redundant patterns and removes them:

| Pattern | Replacement |
| :--- | :--- |
| `addi rd, rs, 0` | delete (no-op) |
| `li rd, 0` then `add rd, rd, rs` | `mv rd, rs` |
| `j L` immediately followed by `L:` | delete the jump |
| `mul rd, rs, 1` | `mv rd, rs` |
| `mul rd, rs, 0` | `li rd, 0` |

**Implementation** in `scratchv/optimizer/peephole.py`:
```python
class PeepholeOptimizer:
    def optimize(self, program: Program) -> int:
        changes = 0
        for func in program.functions:
            for block in func.blocks:
                changes += self._optimize_block(block)
        return changes

    def _optimize_block(self, block) -> int:
        changes = 0
        i = 0
        while i < len(block.instructions):
            if self._is_addi_zero(block.instructions[i]):
                block.instructions.pop(i)
                changes += 1
                continue
            elif self._is_jump_to_next(block, i):
                block.instructions.pop(i)
                changes += 1
                continue
            i += 1
        return changes
```

---

## 5. Loop Invariant Code Motion (LICM) (⭐⭐)

Moves computations that don't change inside a loop to before the loop.

**Example** (convolution inner loop):
```python
# Before (inside inner loop):
for out_y in range(H_out):
    for out_x in range(W_out):
        base = out_y * W_in     # invariant in inner loop!
        for ky in range(K):
            ...

# After:
for out_y in range(H_out):
    base = out_y * W_in         # hoisted out
    for out_x in range(W_out):
        for ky in range(K):
            ...
```

**Implementation** in `scratchv/optimizer/licm.py`:
```python
class LICM:
    def optimize(self, program: Program) -> int:
        changes = 0
        for func in program.functions:
            changes += self._find_loops_and_hoist(func)
        return changes

    def _find_loops_and_hoist(self, func) -> int:
        hoisted_count = 0
        # 1. Find FOR/ENDFOR pairs
        # 2. Identify instructions whose operands don't change in loop
        # 3. Move them before the FOR instruction
        # 4. Increment hoisted_count for each moved instruction
        return hoisted_count
```

---

## 6. Greedy Register Allocation (⭐⭐)

**Already implemented** in `scratchv/backend/register_alloc.py`.

Replaces naive fixed mapping with an LRU-based greedy allocator that
reuses registers efficiently and spills only when necessary.

---

## 📊 Measuring Optimization Impact

Use the TinyFive adapter to compare instruction counts:

```python
from scratchv.simulator.tinyfive import ProfiledMachine

def count_instrs(asm_code: str) -> int:
    m = ProfiledMachine()
    m.pc = 4 * 128
    for line in asm_code.split('\n'):
        # Feed assembly lines to TinyFive
        ...
    m.exe()
    return m.instr_count

before = count_instrs(asm_before_opt)
after  = count_instrs(asm_after_opt)
print(f"Reduction: {((before - after) / before) * 100:.1f}%")
```

---

## 🗺️ Suggested Timeline

| Week | Pass | Notes |
| :--- | :--- | :--- |
| W5 | Constant folding + DCE | IR building stage |
| W7 | Peephole + MulAdd fusion | Backend codegen stage |
| W9 | LICM | After loop support is solid |
| W10 | Register alloc improvement | Compare against naive alloc |
