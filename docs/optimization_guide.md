# Optimization Passes Guide

Six beginner-friendly optimization passes for ScratchV, ordered by difficulty.
Each pass can be implemented as a standalone task.

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
    def run(self, program: Program) -> int:
        for func in program.functions:
            for block in func.blocks:
                self._optimize_block(block)

    def _optimize_block(self, block):
        i = 0
        while i < len(block.instructions):
            if self._is_addi_zero(block.instructions[i]):
                block.instructions.pop(i)
                continue
            elif self._is_jump_to_next(block, i):
                block.instructions.pop(i)
                continue
            i += 1
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
    def run(self, program: Program) -> int:
        for func in program.functions:
            self._find_loops_and_hoist(func)

    def _find_loops_and_hoist(self, func):
        # 1. Find FOR/ENDFOR pairs
        # 2. Identify instructions whose operands don't change in loop
        # 3. Move them before the FOR instruction
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
