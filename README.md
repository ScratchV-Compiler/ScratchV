# ScratchV

**From ONNX to RISC-V assembly — a minimal compiler built from scratch.**

ScratchV is a 12-week educational project that implements a complete compiler
toolchain: parse an ONNX model (or a simple DSL), lower it through a custom
intermediate representation (IR), apply optimizations, and emit RISC-V assembly
code executable on QEMU, Spike, TinyFive, or real hardware.

---

## Project Structure

```
ScratchV/
├── scratchv/
│   ├── ir/                  # Intermediate representation (three-address code)
│   │   ├── types.py         #   Core types: Value, Instruction, BasicBlock, Function, Program
│   │   ├── builder.py       #   IR construction helper (chainable API)
│   │   └── printer.py       #   IR text dump
│   ├── frontend/            # Input parsing
│   │   ├── onnx_parser.py   #   ONNX model → IR
│   │   └── dsl_parser.py    #   Simple DSL → IR (test without ONNX dep)
│   ├── optimizer/           # IR → IR optimizations
│   │   ├── constant_folding.py  #   Compile-time constant evaluation
│   │   ├── dead_code.py         #   Unused instruction removal
│   │   ├── peephole.py          #   Redundant pattern elimination
│   │   ├── muladd_fusion.py     #   Mul+Add instruction combining
│   │   └── licm.py              #   Loop Invariant Code Motion
│   ├── backend/             # Code generation
│   │   ├── instruction_select.py #   IR → RISC-V pseudo-instructions
│   │   ├── register_alloc.py     #   Register allocation (naive + greedy)
│   │   ├── asm_emit.py           #   RISC-V assembly text emission
│   │   └── llvm_codegen.py       #   LLVM IR text generation
│   ├── verification/        # Verification & comparison
│   │   └── verifier.py      #   ONNX Runtime + numpy reference comparison
│   ├── simulator/           # Simulation
│   │   └── tinyfive.py      #   TinyFive adapter with instruction counting
│   └── main.py              # CLI entry point
├── tests/                   # 50+ unit tests (including LLVM codegen + verification)
├── examples/                # DSL models, ONNX generator, pipeline demos
├── docs/
│   ├── verification.md      # Guide: TinyFive, Spike, QEMU, LLVM IR, ONNX Runtime
│   └── optimization_guide.md # Optimization passes guide
└── models/                  # Generated ONNX models
```

## Quick Start

### Installation

**Recommended: virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install onnx numpy         # ONNX model support
pip install tinyfive           # assembly verification (optional)
```

**Alternative (pipx):**

```bash
pipx install .
pipx inject scratchv onnx numpy tinyfive
```

> Debian/Ubuntu users: if you get an "externally-managed-environment" error,
> use the venv method above, or append `--break-system-packages`:
> ```bash
> pip install --break-system-packages -e .
> ```

### Compile a DSL model

```bash
# Simple add (RISC-V backend)
scratchv examples/simple_add.dsl -o output.s --dump-ir

# LLVM IR backend
scratchv examples/simple_add.dsl --backend llvm -o output.ll --dump-ir

# Full optimization pipeline
scratchv examples/relu_test.dsl -o relu.s --optimize all --dump-ir

# Matrix multiply
scratchv examples/matmul_test.dsl -o matmul.s --optimize all
```

### Compile an ONNX model

```bash
# Generate test ONNX models
python examples/gen_onnx_model.py

# Compile with RISC-V backend
scratchv models/add.onnx -o add.s --optimize all

# Compile with LLVM backend
scratchv models/add.onnx --backend llvm -o add.ll --optimize all

# Verify against ONNX Runtime
scratchv models/add.onnx --verify
```

### Verify with TinyFive

```bash
python examples/verify_with_tinyfive.py examples/simple_add.dsl
```

### End-to-end pipeline demos

```bash
# Full pipeline (ONNX → LLVM IR → verification)
python examples/end_to_end_pipeline.py --backend llvm

# ONNX → LLVM IR → ONNX Runtime comparison
python examples/onnx_llvm_verification.py

# LLVM optimization impact analysis
python examples/llvm_optimization_pipeline.py
```

### Command-line options

| Flag | Description |
| :--- | :--- |
| `-o FILE` | Output file (default: output.s for riscv, output.ll for llvm) |
| `--backend {riscv,llvm}` | Target backend (default: riscv) |
| `--dump-ir` | Print IR before and after optimization |
| `--optimize {none,basic,all}` | Optimization level (default: none) |
| `--reg-alloc {naive,greedy}` | Register allocation strategy (default: greedy) |
| `--verify` | Verify output against ONNX Runtime / numpy reference |
| `--rtol FLOAT` | Relative tolerance for verification (default: 1e-5) |
| `--atol FLOAT` | Absolute tolerance for verification (default: 1e-8) | |

## Pipeline Overview

```
                        ┌──────────────────────────────────────────┐
                        │           ScratchV Compiler              │
                        │                                          │
ONNX Model ──▶ ONNX Parser ──▶ IR (3-addr) ──▶ Optimizer ──┐     │
                        │                              │     │
DSL Source ──▶ DSL Parser ────┘                       │     │
                                                      │     │
                          ┌───────────────────────────┘     │
                          ▼                                 │
               ┌─────────────────┐                          │
               │ Instruction Sel │──▶ Reg Alloc ──▶ Asm Emit │──▶ RISC-V Assembly
               └─────────────────┘                          │
                          │                                  │
                          ▼                                  │
                  ┌──────────────┐                           │
                  │ LLVM Codegen  │──▶ LLVM IR (.ll)         │
                  └──────────────┘       │                   │
                                         ▼                   │
                               ┌──────────────────┐          │
                               │ opt / llc / lli  │          │
                               │ (external tools) │          │
                               └──────────────────┘          │
                        ┌─────────────────────────┐          │
                        │ Verification Framework   │          │
                        │ • ONNX Runtime reference │          │
                        │ • Numpy reference        │          │
                        │ • DSL interpreter        │          │
                        │ • TinyFive simulator     │          │
                        └─────────────────────────┘          │
└──────────────────────────────────────────────────────────────┘
```

### Optimization Passes

| Pass | Level | Description |
| :--- | :--- | :--- |
| Constant Folding | basic | Evaluate constant expressions at compile time |
| Dead Code Elimination | basic | Remove unused instructions |
| Peephole | all | Eliminate redundant patterns (addi 0, mul 1, etc.) |
| Mul-Add Fusion | all | Combine consecutive mul+add instruction pairs |
| Loop Invariant Code Motion | all | Hoist invariant computations out of loops |

### Verification Tools

| Tool | Type | Best For |
| :--- | :--- | :--- |
| TinyFive | Python simulator | Quick validation, AI model testing, instruction counting |
| Spike | RISC-V golden model | ISA conformance testing |
| QEMU | System emulator | Full-system testing |
| Custom simulator | Python | Teaching, deep understanding |

See `docs/verification.md` for the complete verification guide.

## Roadmap (12 Weeks)

| Weeks | Phase | Goal |
| :--- | :--- | :--- |
| 1-2 | Setup | Toolchain, QEMU, run baseline benchmarks |
| 3-4 | IR | ONNX parser, core IR, basic ops (Add, Mul, MatMul) |
| 5-6 | Optimizer | CF + DCE + peephole, more ops (ReLU, MaxPool, GELU) |
| 7-8 | Backend I | Instruction selection, naive reg alloc, control flow |
| 9-10 | Backend II | Greedy reg alloc, LICM, muladd fusion, benchmark validation |
| 11-12 | Docs | Design doc, user manual, final presentation, perf analysis |

## DSL Syntax

```
# Arithmetic
result = add(a, b)
result = mul(a, b)

# Activation
y = relu(x)
y = gelu(x)
y = softmax(x, axis:-1)

# Linear algebra
c = matmul(A, B, m:2, n:2, k:2)
d = dot(a, b, len:4)

# Pooling
p = maxpool(x, kernel:2, stride:2)

# Return
return result
```

## License

MIT
