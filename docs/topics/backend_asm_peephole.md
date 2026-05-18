# Assembly-level Peephole Optimizer

## Overview

The Peephole Optimizer (`scratchv.backend.asm_peephole`) applies peephole optimization rules directly to RISC-V assembly text. It uses a sliding-window pattern matching approach with register wildcards to detect and replace suboptimal instruction sequences.

## API

```python
from scratchv.backend.asm_peephole import PeepholeOptimizer, PeepholeRule

optimizer = PeepholeOptimizer()
optimized_asm, num_changes = optimizer.optimize(raw_asm)
```

### `PeepholeOptimizer(rules=None)`

**Parameters:**
- `rules` (`list[PeepholeRule] | None`): Custom peephole rules. If None, five default rules are used.

### `optimize(asm_text) -> tuple[str, int]`

Apply peephole optimization to the assembly text. Returns the optimized assembly and the number of changes made.

### `report() -> str`

Return a human-readable report of rules applied and match counts.

## Default Rules (5 rules)

### 1. addi+addi Fusion
```
addi x1, x1, 3
addi x1, x1, 5
```
becomes:
```
addi x1, x1, 8    # peephole: addi+addi fusion
```

### 2. Redundant mv Pair Elimination
```
mv x1, x2
mv x2, x1
```
This is removed entirely when the two form a redundant swap.

### 3. li+addi Fusion
```
li x1, 10
addi x1, x1, 5
```
becomes:
```
li x1, 15    # peephole: li+addi fusion
```

### 4. beq x0,x0 -> j (Unconditional Jump)
```
beq x0, x0, label
```
becomes:
```
j label    # peephole: beq zero-zero to jump
```

### 5. Redundant mv Through Intermediate
```
mv x1, x2
mv x3, x1
```
becomes:
```
mv x3, x2    # peephole: redundant mv elimination
```

## Custom Rules

Rules are defined with `PeepholeRule`:

```python
from scratchv.backend.asm_peephole import PeepholeRule, PeepholeOptimizer

custom_rules = [
    PeepholeRule(
        name="nop elimination",
        pattern=["nop"],
        replacement=[],  # empty = delete
        register_constraints=[],
    ),
]

opt = PeepholeOptimizer(rules=custom_rules)
```

### `PeepholeRule` Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Human-readable rule name |
| `pattern` | `list[str]` | Opcode sequence to match. `"*"` matches any opcode. |
| `replacement` | `list[str]` | Replacement opcode sequence. Use `{var}` for template substitution. |
| `register_constraints` | `list[tuple]` | Index constraints: `(dst_instr_idx, src_instr_idx, src_op_idx)`. |

## CLI Usage

```bash
python -m scratchv.backend.asm_peephole input.s -o output.s --report
python -m scratchv.backend.asm_peephole input.s --list-rules
```

## Algorithm

The optimizer iterates to a fixed point (max 50 iterations) using a sliding window. For each position in the instruction list, it tries all rules. When a rule matches, the matched window is replaced and scanning continues from that position. The iteration repeats until no more rules fire.

Register wildcards bind on first use: the first time a variable like `rd0` is seen in a pattern, its value is captured. Subsequent uses must match the captured value.
