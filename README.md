# ScratchV

**From ONNX to RISC-V binary — a minimal AI compiler built from scratch.**

ScratchV is an educational compiler project that implements a complete
toolchain: parse ONNX models (or a simple DSL), lower through a custom
intermediate representation (IR), apply optimizations, and emit RISC-V
assembly / LLVM IR / binary machine code.

---

## Project Structure

```
ScratchV/
├── scratchv/                # Main compiler package
│   ├── ir/                  #   Intermediate representation (three-address code)
│   │   ├── types.py         #     Value, Instruction, BasicBlock, Function, Program
│   │   ├── builder.py       #     IR construction helper (chainable API)
│   │   └── printer.py       #     IR text dump
│   ├── frontend/            #   Input parsing
│   │   ├── onnx_parser.py   #     ONNX model → IR (Conv, Gemm, Sigmoid, ...)
│   │   ├── dsl_parser.py    #     Simple DSL → IR (test without ONNX dep)
│   │   ├── dsl_extended.py  #     Extended DSL: if/else, while loops
│   │   └── dsl_errors.py    #     GCC-style error messages with fix suggestions
│   ├── optimizer/           #   IR → IR optimizations
│   │   ├── constant_folding.py  #     Compile-time constant evaluation
│   │   ├── dead_code.py         #     Unused instruction removal
│   │   ├── peephole.py          #     Redundant pattern elimination
│   │   ├── muladd_fusion.py     #     Mul+Add instruction combining
│   │   └── licm.py              #     Loop Invariant Code Motion
│   ├── analysis/             #   IR analysis & verification
│   │   ├── cfg_builder.py    #     CFG builder, dominator tree, loop detection
│   │   └── ir_verifier.py    #     7-rule IR validator (def-use, types, labels...)
│   ├── backend/             #   Code generation
│   │   ├── instruction_select.py #  IR → RISC-V pseudo-instructions
│   │   ├── inst_select_ext.py    #  Extended: sqrt/min/max/abs/float64 support
│   │   ├── register_alloc.py     #  Register allocation (naive + greedy)
│   │   ├── regalloc_linear.py    #  Linear-scan register allocator with spill
│   │   ├── inst_scheduler.py     #  List scheduler with DAG + latency model
│   │   ├── asm_emit.py           #  RISC-V assembly text emission
│   │   ├── asm_beautifier.py     #  Assembly beautifier (align + comment)
│   │   ├── asm_peephole.py       #  Assembly-level peephole optimizer
│   │   ├── const_merge.py        #  Constant-load merger (lui+addi → li)
│   │   ├── inst_counter.py       #  Instruction category counter + charts
│   │   ├── riscv_encoder.py      #  RISC-V RV32IM binary encoder
│   │   └── llvm_codegen.py       #  LLVM IR text generation
│   ├── verification/        #   Verification & comparison
│   │   └── verifier.py      #     ONNX Runtime + numpy reference comparison
│   ├── simulator/           #   Simulation
│   │   ├── tinyfive.py      #     TinyFive adapter with instruction counting
│   │   └── rv32_emulator.py #     RV32IM emulator with NN runtime hooks
│   ├── utils/               #   Utilities
│   │   └── logger.py        #     Colored logging with phase timing
│   └── main.py              #   CLI entry point
├── scratchv_dag/            # Standalone DAG / memory library
│   ├── sdnode.py            #   SDNode, MVT, SelectionDAG container
│   ├── selection_dag.py     #   DAGBuilder, DAGCombiner, DAGScheduler
│   ├── cache.py             #   4 MB L1 cache simulator (LRU, write-back)
│   ├── allocator.py         #   Buddy allocator with cache-line alignment
│   └── README.md            #   Standalone docs
├── benchmarks/              # Benchmark suite (23 DSL cases + per-topic bench)
│   ├── bench_runner.py      #   Automated test runner with HTML reports
│   ├── bench_*.py           #   Per-module performance benchmarks
│   └── cases/               #   23 DSL test cases (.dsl + .expected + .desc)
├── tests/                   # 348 unit tests
├── examples/                # DSL models, ONNX generator, pipeline demos
├── scripts/                 # Utility scripts (full pipeline, lint check)
├── docs/
│   ├── topics/              #   14 topic proposals + 14 implementation guides
│   ├── CODING_STANDARDS.md  #   Code style guide
│   ├── verification.md      #   Verification guide
│   ├── optimization_guide.md #  Optimization passes guide
│   ├── developer_guide.md   #   Internal architecture & extension guide
│   └── ScratchV.html        #   Project landing page (zh-CN)
├── models/                  # Generated / example ONNX models
├── CHANGELOG.md             # Release history
├── CONTRIBUTING.md          # Contribution guidelines
└── Makefile                 # Dev targets (test, clean, lint, …)
```

## Quick Start

### Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install onnx onnxruntime tinyfive pytest
```

### Compile an ONNX model

```bash
# Compile with RISC-V backend (→ RISC-V assembly + binary)
python3.12 -m scratchv models/graph/cnn.onnx -o cnn.s --dump-ir

# Compile with LLVM backend
python3.12 -m scratchv models/graph/cnn.onnx --backend llvm -o cnn.ll

# Full CNN pipeline: ONNX → IR → asm → binary → execution → MSE/MAE
python3.12 scripts/run_full_pipeline.py models/graph/cnn.onnx --dump-asm
```

### Compile a DSL model

```bash
# Simple arithmetic
python3.12 -m scratchv examples/simple_add.dsl -o output.s --dump-ir

# Full optimization pipeline
python3.12 -m scratchv examples/relu_test.dsl -o relu.s --optimize all --dump-ir

# Extended DSL (if/else, while loops)
python3.12 -c "
from scratchv.frontend.dsl_extended import ExtendedDSLParser
dsl = '''
  while (i < 10):
    acc = add(acc, x)
  endwhile
  return acc
'''
parser = ExtendedDSLParser()
prog = parser.parse(dsl)
print(prog.dump())
"
```

### Run tests

```bash
# All 348 tests
python3.12 -m pytest tests/ -v

# CNN pipeline tests only
python3.12 -m pytest tests/test_cnn_pipeline.py -v

# Backend topic tests
python3.12 -m pytest tests/test_asm_beautifier.py tests/test_inst_counter.py tests/test_asm_peephole.py -v
```

### Run benchmarks

```bash
# Full benchmark suite
python3.12 benchmarks/bench_runner.py

# Individual module benchmarks
python3.12 benchmarks/bench_regalloc_linear.py
python3.12 benchmarks/bench_inst_scheduler.py
```

## Command-line options

| Flag | Description |
| :--- | :--- |
| `-o FILE` | Output file (default: output.s for riscv, output.ll for llvm) |
| `--backend {riscv,llvm}` | Target backend (default: riscv) |
| `--dump-ir` | Print IR before and after optimization |
| `--optimize {none,basic,all}` | Optimization level (default: none) |
| `--reg-alloc {naive,greedy}` | Register allocation strategy (default: greedy) |
| `--verify` | Verify output against ONNX Runtime / numpy reference |
| `--rtol FLOAT` | Relative tolerance for verification (default: 1e-5) |
| `--atol FLOAT` | Absolute tolerance for verification (default: 1e-8) |

## Pipeline Overview

```
                        ┌──────────────────────────────────────────────┐
                        │           ScratchV Compiler                  │
                        │                                              │
ONNX Model ──▶ ONNX Parser ──▶ IR (3-addr) ──▶ Optimizer ──┐         │
                        │                              │     │         │
DSL Source ──▶ DSL Parser ────┘                       │     │         │
                                                      │     │         │
                          ┌───────────────────────────┘     │         │
                          ▼                                 │         │
               ┌─────────────────┐                          │         │
               │ Instruction Sel │──▶ Reg Alloc ──▶ Asm Emit│──▶ RISC-V .s
               └─────────────────┘       │                   │         │
                          │              ▼                   │         │
                          │    RISC-V Encoder ──▶ .bin      │         │
                          ▼                                  │         │
               ┌─────────────────┐                          │         │
               │ LLVM Codegen    │──▶ LLVM IR (.ll)         │         │
               └─────────────────┘                          │         │
                        │                                    │         │
                        ▼                                    │         │
               ┌──────────────────────────┐                  │         │
               │ Verification Framework   │                  │         │
               │ • ONNX Runtime reference │                  │         │
               │ • Numpy reference        │                  │         │
               │ • IR Trace Executor      │                  │         │
               │ • RV32 Emulator          │                  │         │
               │ • MSE / MAE comparison   │                  │         │
               └──────────────────────────┘                  │         │
└──────────────────────────────────────────────────────────────┘
```

## Topic Modules (14 课题)

Each topic includes: implementation module + docs/topics/ guide + tests + benchmark.

| # | Topic | Module | Difficulty |
| :--- | :--- | :--- | :--- |
| 1 | DSL Frontend Enhancer (if/else, while) | `scratchv/frontend/dsl_extended.py` | Medium |
| 5 | RISC-V Assembly Beautifier | `scratchv/backend/asm_beautifier.py` | Low |
| 6 | Compiler Benchmark Suite | `benchmarks/bench_runner.py` | Medium |
| 7 | Compiler Logging System | `scratchv/utils/logger.py` | Low |
| 9 | DSL Error Beautifier | `scratchv/frontend/dsl_errors.py` | Medium |
| 11 | CFG Builder + Loop Detection | `scratchv/analysis/cfg_builder.py` | High |
| 12 | Instruction Counter + Charts | `scratchv/backend/inst_counter.py` | High |
| 13 | Assembly Peephole Optimizer | `scratchv/backend/asm_peephole.py` | Low |
| 14 | Constant Load Merge | `scratchv/backend/const_merge.py` | Low |
| 17 | Linear Scan Register Allocator | `scratchv/backend/regalloc_linear.py` | High |
| 18 | Instruction Scheduler | `scratchv/backend/inst_scheduler.py` | High |
| 20 | Code Standards & Formatting | `.pre-commit-config.yaml` | Low |
| 21 | IR Verifier | `scratchv/analysis/ir_verifier.py` | Medium |
| 28 | Extended Instruction Selection | `scratchv/backend/inst_select_ext.py` | Medium |

See `docs/topics/` for detailed guides and the original topic proposals.

## License

MIT
