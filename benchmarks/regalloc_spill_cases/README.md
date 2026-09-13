# DSL Register-Spill Benchmark Suite

This suite compares spill traffic produced from the same straight-line DSL
programs by:

1. ScratchV's current `InstructionSelector` + `regalloc_linear.py` pipeline.
2. ScratchV's `LLVMCodegen` + LLVM's RISC-V backend (O2 by default).

Run it from the repository root:

```bash
python3 -m benchmarks.bench_regalloc_spill_compare
python3 -m benchmarks.bench_regalloc_spill_compare --json
python3 -m benchmarks.bench_regalloc_spill_compare \
  --json-output /tmp/spills.json --markdown /tmp/spills.md
```

Use `--phys-reg-count N` to draw a ScratchV pressure curve with fewer than the
pipeline's 19 default integer registers. Use `--llvm-opt-level 0..3` to compare
LLVM allocation modes.

## Cases

| Case | Pressure shape | Expected ScratchV boundary | Question answered |
|---|---|---:|---|
| `00_low_pressure_chain` | Short sequential live ranges | no spill | Does either backend spill below capacity? |
| `01_wide_fanout_32` | 32 values live before a balanced reduction | spill | What happens just above ScratchV's register limit? |
| `02_double_use_40` | 40 values consumed forward and in reverse | spill | How much traffic remains when LLVM must also spill? |
| `03_lifetime_holes_36` | Values used, idle for a region, then reused | spill | Can live-range splitting avoid whole-interval spills? |
| `04_hot_cold_48` | Cold values surround a frequently used hot chain | spill | Do use frequency and scheduling reduce spill traffic? |

## Measurement rules

- Cases use at most four external inputs, so LLVM stack-argument loads do not
  contaminate the spill count.
- Cases contain no explicit memory operations or control-flow allocas. Other
  stack-relative scalar loads/stores can therefore be treated as spill traffic.
- LLVM `sd`/`ld` and `fsd`/`fld` pairs for ABI callee-saved registers are
  reported as frame management and excluded from spill traffic.
- ScratchV spill slots use the public `spill_slot_count` allocator property;
  anchored `sw/lw/fsw/flw/sd/ld/fsd/fld` stack accesses are classified as
  spill stores/reloads.
- Exact counts may change when instruction selection, scheduling, register sets,
  LLVM versions, or allocation heuristics change. The suite asserts only the
  intended pressure boundary, not today's ratios.

The comparison is intentionally pipeline-level. ScratchV currently selects its
own arithmetic machine operations while LLVM uses the IR's floating-point type,
so the result includes register-class and instruction-scheduling effects in
addition to the allocator algorithm itself. This is the useful end-to-end gap,
but it should not be presented as an isolated algorithm-only comparison.
