# Assembly Peephole Optimizer — Agent / Developer Guide

> **Audience**: AI coding agents, maintainers extending `asm_peephole.py`  
> **Source**: `scratchv/backend/asm_peephole.py`  
> **Topic index**: [../../../topic13/README.md](../../../topic13/README.md)  
> **Human design doc**: [../13-窥孔优化器-设计文档.md](../13-窥孔优化器-设计文档.md)  
> **Compare report**: [../../../benchmark_reports/peephole_compare.md](../../../benchmark_reports/peephole_compare.md)  
> **Last verified**: 2026-08-01 — **83/83** tests pass; **8** default rules  
> **Do NOT re-add**: `redundant mv pair elimination` (`mv x,y; mv y,x → delete`) — unsound

---

## Module Map (symbol → responsibility)

Line numbers drift; prefer symbols over exact lines.

| Symbol | Role |
|--------|------|
| `AsmLine` | Parsed assembly line dataclass |
| `PeepholeRule` | Rule definition (pattern + replacement + constraints) |
| `_LINE_RE` / `_parse_line` | Single-line parse |
| `_parse_asm` / `_lines_to_asm` | Full text ↔ `list[AsmLine]` |
| `_fits_simm12` / `_parse_imm` | Signed 12-bit check; base-aware int parse (`0x`, `0b`) |
| `_default_rules` | **8** built-in rules (no fake-swap delete) |
| `_operand_matches` | Wildcard operand binding |
| `_match_rule` | Match rule against window (+ label / imm / beq / mv-chain guards) |
| `AsmPeepholeOptimizer` | Fixed-point sliding-window optimizer |
| `_apply_replacement` | Rule-specific template expansion + label preserve |
| `main` | CLI (`python -m scratchv.backend.asm_peephole`) |

**Do NOT confuse with**: `scratchv/optimizer/peephole.py` (`IRPeepholeOptimizer`) — different layer, different API.

---

## Public API Contract

### Import

```python
from scratchv.backend.asm_peephole import (
    AsmPeepholeOptimizer,
    PeepholeRule,
    AsmLine,
)
# also re-exported: from scratchv.backend import AsmPeepholeOptimizer
```

### Primary usage

```python
opt = AsmPeepholeOptimizer()             # default 8 rules
opt = AsmPeepholeOptimizer(rules=[...])  # custom rules

optimized_text, num_changes = opt.optimize(asm_text)  # -> tuple[str, int]
report_str = opt.report()                # after optimize()
counts = opt.total_matches               # dict[rule_name, int]
```

### Invariants

1. `optimize()` is **pure** on input text (no file I/O; counters live on the instance).
2. `num_changes` = number of rule applications (not necessarily lines saved).
3. Empty `replacement=[]` means **delete** matched window; if the first line had a label, emit a bare `label:` line.
4. Fixed-point loop: max **50** iterations; stops when a full pass makes no match.
5. Application is **left-to-right greedy**; first matching rule in `self.rules` wins.
6. Mid-window labels (`window[i].label` for `i > 0`) **refuse** the match.
7. Immediate folding uses `_parse_imm` (not bare `int()`); never emit `(0x10+0x20)` style garbage.

---

## Compiler Integration

```python
# scratchv/compiler.py — _run_asm_passes()
if self.config.peephole_asm:
    from scratchv.backend.asm_peephole import AsmPeepholeOptimizer
    opt = AsmPeepholeOptimizer()
    asm_text, changes = opt.optimize(asm_text)
```

CLI flag: `scratchv ... --peephole-asm` (`scratchv/main.py`).

Pass order in `_run_asm_passes`: **peephole → const_merge → schedule → beautify**.

---

## PeepholeRule Schema

```python
PeepholeRule(
    name: str,                          # MUST be unique; used in _apply_replacement branches
    pattern: list[str],                 # opcodes, lowercase; len = window size
    replacement: list[str],             # templates; [] = delete
    register_constraints: list[tuple[int, int, int]],  # (dst_instr, src_instr, src_op_idx)
)
```

### register_constraints semantics

Each tuple `(dst_idx, src_instr_idx, src_op_idx)` requires:

```
window[dst_idx].operands[0] == window[src_instr_idx].operands[src_op_idx]
```

### Replacement templates

| Placeholder | Set by rule |
|-------------|-------------|
| `{rd}`, `{rs1}`, `{imm_sum}` | addi+addi fusion |
| `{rd}`, `{imm_sum}` | li+addi fusion |
| `{label}` | beq zero-zero to jump |
| `{rd1}`, `{rs2}` | redundant mv elimination |
| `{rd}`, `{rs}` | addi-zero to mv |

If `_parse_imm` fails after a match (should be rare), `_apply_replacement` returns the original window unchanged.

---

## Default Rules (quick reference)

| # | name | pattern | replacement | constraints / notes |
|---|------|---------|-------------|---------------------|
| 1 | `addi+addi fusion` | addi, addi | addi {rd} {rs1} {imm_sum} | (0,1,0),(0,1,1); sum ∈ [-2048,2047] |
| 2 | `li+addi fusion` | li, addi | li {rd} {imm_sum} | (0,1,0),(0,1,1); imms parseable |
| 3 | `beq zero-zero to jump` | beq | j {label} | ops[0,1] ∈ {x0, zero} |
| 4 | `redundant mv elimination` | mv, mv | mv {rd1} {rs2} | (0,1,1); **excludes** swap shape; mid may stay live |
| 5 | `addi-zero self elimination` | addi | [] | (0,0,1); imm==0 |
| 6 | `addi-zero to mv` | addi | mv {rd} {rs} | imm==0; rd≠rs |
| 7 | `nop elimination` | nop | [] | — |
| 8 | `mv-self elimination` | mv | [] | (0,0,1) |

**Removed (unsound)**: `redundant mv pair elimination` (`mv x,y; mv y,x → delete`).  
Both regs become original `y` — not a no-op. Tests assert the pair is **preserved**.

Helpers: `_fits_simm12`, `_parse_imm`. Mid-window labels refuse matching.

---

## How to Add a New Rule

### Step 1 — Define rule in `_default_rules()` or pass a custom list

```python
PeepholeRule(
    name="li-zero to mv",
    pattern=["li"],
    replacement=["mv {rd} x0"],
    register_constraints=[],
)
```

### Step 2 — Add special logic if needed

If replacement needs computed values, add a branch in `_apply_replacement()`:

```python
elif rule.name == "my new rule":
    derived["foo"] = ...
```

**Prefer**: keep logic generic; only add branches when template substitution is insufficient.

### Step 3 — Extra checks in `_match_rule()` when opcode-only match is insufficient

Example: beq rule checks `x0`/`zero` after generic matching; addi fusion checks simm12.

### Step 4 — Test

```python
# tests/test_asm_peephole.py
def test_my_rule():
    opt = AsmPeepholeOptimizer(rules=[my_rule])
    result, changes = opt.optimize("  ...\n")
    assert changes >= 1
```

Prefer also adding a `TestSemanticEquivalence` case when the rewrite changes values.

### Step 5 — Update docs

- Human: `docs/topics/13-窥孔优化器-设计文档.md` §6 rule table  
- This file: Default Rules table  
- Status index: `topic13/README.md` rule count / test count if they change  

---

## Verification Commands

```bash
cd /home/z/ScratchV-main   # or your repo root
source .venv/bin/activate

# All Topic-13 peephole tests (83 cases)
python -m pytest tests/test_asm_peephole*.py -v --tb=short

# By category
python -m pytest tests/ -m unit        -k peephole -v
python -m pytest tests/ -m integration -k peephole -v
python -m pytest tests/ -m stress      -k peephole -v
python -m pytest tests/ -m blackbox    -k peephole -v

# Benchmark / before-after (optional)
python benchmarks/bench_asm_peephole.py
python benchmarks/compare_peephole.py --markdown benchmark_reports/peephole_compare.md
```

**Expected**: **83 passed**; `--list-rules` must **not** print `redundant mv pair elimination`.

### Test file map

| Marker | File | Role |
|--------|------|------|
| `unit` | `tests/test_asm_peephole.py` | parse, match, 8 rules, semantic equivalence, labels/hex |
| `integration` | `tests/test_asm_peephole_integration.py` | CompilerDriver / passes / flag off |
| `stress` | `tests/test_asm_peephole_stress.py` | scale / hex batch / labels under load |
| `blackbox` | `tests/test_asm_peephole_blackbox.py` | CLI + fixtures; swap-delete absent |

Fixtures: `tests/fixtures/asm_peephole/*.s`  
(incl. `input_hex_fusion.s`, `input_addi_overflow.s`, `input_nop_mv_self.s`, `input_mv_chain.s`)

---

## Pitfalls for Agents

| Issue | Detail | Fix |
|-------|--------|-----|
| IR vs ASM peephole | Two modules, same concept | Edit `backend/asm_peephole.py` for Topic 13 |
| Re-adding fake swap delete | Looks clever, breaks semantics | Never restore `redundant mv pair elimination` |
| Duplicate parser | `_asm_parser.py` unused here | Do not unify unless task asks |
| Rule name typos | `_apply_replacement` branches on `rule.name` | Match strings exactly |
| addi imm overflow | RV addi imm is simm12 | Refuse fusion when sum out of range |
| hex immediates | Must use `_parse_imm` | Do not fold with bare `int()` |
| Labels | Mid-window label = refuse; lead label = preserve | Cover with tests |
| mv-chain liveness | Rule 4 best-effort without liveness | Document; see `test_mv_chain_unsound_when_mid_live` |
| x0 vs zero | Only beq special-cases aliases | Normalize if adding more zero checks |
| Infinite loop | Bad rules can oscillate | `max_iterations=50` + terminate tests |
| Greedy order | Rule A may block Rule B | Reorder or merge; document dependency |
| `_split_operands` | Defined but unused | Dead code; ignore or cleanup PR |
| CLI `--list-rules` | Still needs positional `input` | Known argparse limitation |

---

## Test Coverage Matrix (key cases)

| Test | Asserts |
|------|---------|
| `test_addi_addi_fusion` | imm merged to 8 |
| `test_li_addi_fusion` | li+addi → single li |
| `test_beq_zero_jump` / `test_beq_zero_alias` | beq x0/zero → j |
| `test_mv_swap_pair_not_deleted` | swap-shaped pair **preserved** (`changes == 0`) |
| `test_redundant_mv_elimination` | mv chain shortened |
| `test_addi_fusion_hex_immediates` | `0x10+0x20` → `48`, no `(` garbage |
| `test_label_preserved_on_fusion` / `_on_nop_deletion` | labels survive |
| `test_mid_label_blocks_fusion` | labeled 2nd insn blocks pair |
| `test_mv_chain_unsound_when_mid_live` | documents Rule 4 liveness gap |
| `TestSemanticEquivalence.*` | register-state checks for sound rewrites |
| `test_cli_list_rules` | removed rule name absent from stdout |

When adding rules: input asm → `optimize()` → assert tokens + `changes`, and prefer a semantic check.

---

## Dependencies

```
asm_peephole.py
  ├── stdlib: re, sys, dataclasses, typing
  └── (no scratchv internal imports)

Consumers:
  ├── scratchv/compiler.py (_run_asm_passes)
  ├── scratchv/backend/__init__.py (re-export AsmPeepholeOptimizer)
  ├── tests/test_asm_peephole*.py
  ├── benchmarks/bench_asm_peephole.py
  └── benchmarks/compare_peephole.py
```

---

## Modification Checklist (agents)

Before marking task complete:

- [ ] `python -m pytest tests/test_asm_peephole*.py -q` — all green (expect 83 unless count intentionally changed)
- [ ] New rule has ≥1 dedicated test (+ semantic case if values change)
- [ ] `rule.name` unique among `_default_rules()`
- [ ] If immediates folded: use `_parse_imm` + simm12 check where needed
- [ ] Labels: mid-window refuse / lead preserve / delete keeps bare label
- [ ] Did **not** re-introduce fake-swap delete
- [ ] `report()` / `total_matches` reflect the new rule
- [ ] Updated design doc §6 + this guide + `topic13/README.md` counts if rules/tests changed
- [ ] Did not break IR peephole (`optimizer/peephole.py`)

---

## Example: End-to-end agent task

**Task**: Add `li rd, 0` → `mv rd, x0` (optional micro-canonicalization).

1. Add to `_default_rules()` with a unique `name`.
2. In `_match_rule`, require `_parse_imm(ops[1]) == 0`.
3. In `_apply_replacement`, set `{rd}` from `ops[0]`.
4. Tests: positive rewrite + semantic equivalence + ensure `li rd, 1` untouched.
5. Run `pytest tests/test_asm_peephole*.py -q`; bump README/design counts if defaults changed.

---

## See Also

- [../../../topic13/README.md](../../../topic13/README.md) — Topic 13 completion index  
- [../13-窥孔优化器.md](../13-窥孔优化器.md) — beginner tutorial  
- [../13-窥孔优化器-设计文档.md](../13-窥孔优化器-设计文档.md) — human design spec  
- [../05-汇编代码美化器.md](../05-汇编代码美化器.md) — downstream asm pass  
- [../14-常量加载合并.md](../14-常量加载合并.md) — adjacent pass in pipeline  
- `scratchv/backend/_asm_parser.py` — shared parser (future unification target)  
