# Verification & Simulation Guide

After ScratchV generates RISC-V assembly, you need to verify it runs correctly.
This guide covers multiple approaches, from lightweight Python simulators to
industrial-grade tools.

---

## Overview

```
ScratchV .s output
        │
        ├── TinyFive (Python, lightweight, AI-friendly)    ← Recommended for dev
        ├── QEMU     (Industrial system emulator)
        ├── Spike    (RISC-V official ISA simulator, golden model)
        ├── Renode   (Embedded system simulator)
        └── Custom Python simulator (educational)
```

| Tool | Language | Lines | Precision | Best For |
| :--- | :--- | :--- | :--- | :--- |
| TinyFive | Python | <1000 | Functional | Quick validation, AI model testing |
| Spike | C++ | ~10K | Cycle-approx | ISA conformance, gold standard |
| QEMU | C | ~1M | Functional | Full system emulation |
| Custom sim | Python | ~500 | Functional | Teaching, deep understanding |

---

## 🐍 TinyFive (Recommended for ScratchV)

TinyFive is a pure-Python RISC-V simulator with <1000 lines of code.
It supports RV32IM and integrates seamlessly with Python AI frameworks.

### Installation

```bash
pip install tinyfive numpy
```

### Quick Start

```python
from tinyfive.machine import Machine

m = Machine(mem_size=4096)
m.x[11] = 6
m.x[12] = 7
m.MUL(10, 11, 12)
print(f"Result: {m.x[10]}")  # 42
```

### Three Usage Modes

**A. Direct API calls** — fastest for testing operator logic:

```python
m.LW(11, 0, 0)     # load from address 0 into x11
m.LW(12, 4, 0)     # load from address 4 into x12
m.MUL(10, 11, 12)  # x10 = x11 * x12
m.ADD(10, 10, 0)   # x10 = x10 + 0
```

**B. Assembly strings** — works with ScratchV's assembly output:

```python
m.pc = 4 * 128
m.asm('addi', 10, 0, 42)  # x10 = 0 + 42
m.exe()
print(f"Result: {m.x[10]}")
```

**C. Full assembly with labels**:

```python
m.pc = 4 * 128
m.lbl('start')
m.asm('addi', 10, 10, 1)
m.asm('bne', 10, 0, 'start')
m.exe(start='start')
```

### Performance Profiling

After execution, get instruction counts:

```python
m.print_perf()
# Output:
#   Ops counters: {'total': 50, 'load': 16, 'store': 8, 'mul': 0, ...}
#   x[] regfile : 5 out of 31 x-registers are used
```

### Custom Instructions

Extend TinyFive to prototype new instructions:

```python
class MyMachine(Machine):
    def FOO(self, rd, rs1, rs2):
        self.x[rd] = (self.x[rs1] + self.x[rs2]) * 2

m = MyMachine(mem_size=4000)
m.x[11] = 3; m.x[12] = 5
m.FOO(10, 11, 12)
print(m.x[10])  # 16
```

---

## 🏭 Spike (RISC-V Golden Model)

Spike is the official RISC-V ISA simulator used for conformance testing.

### Installation

```bash
# Build from source
git clone https://github.com/riscv-software-src/riscv-isa-sim.git
cd riscv-isa-sim
mkdir build && cd build
../configure --prefix=$RISCV
make && make install
```

### Usage

```bash
# Compile assembly with RISC-V GCC
riscv64-unknown-elf-gcc -march=rv32im -static -o prog.elf output.s

# Run with Spike + Proxy Kernel
spike pk prog.elf

# Get instruction count
spike -l --log-commits pk prog.elf 2>&1 | tail -5
```

### Adding Custom Instructions to Spike

1. Define the instruction encoding
2. Modify the decoder in Spike
3. Implement the execution logic
4. Test with `.insn` assembly directives

---

## 💻 QEMU (System Emulator)

QEMU provides full-system emulation with `-icount` mode for instruction counting.

```bash
# Install RISC-V toolchain
sudo apt-get install gcc-riscv64-linux-gnu qemu-user

# Compile and run
riscv64-linux-gnu-gcc -march=rv32im -static -o prog.elf output.s
qemu-riscv32-static prog.elf
```

---

## 📊 Performance Profiling

### Instruction Counting (Most Important for Compiler Optimization)

For ScratchV's optimization passes, **instruction count** is the primary metric.
Use a profiled TinyFive machine:

```python
class ProfiledMachine(Machine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instr_count = 0

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if name.isupper() and callable(attr) and name not in ('__init__', 'asm', 'exe'):
            def counted(*args, **kwargs):
                self.instr_count += 1
                return attr(*args, **kwargs)
            return counted
        return attr
```

### Cycle Counting

For more accurate performance estimation, use:

| Tool | Precision | Effort |
| :--- | :--- | :--- |
| Spike + `rdcycle` | Cycle-approx | Low |
| rvsim-core | Cycle-accurate | Medium |
| GVSoC | ~90% accuracy, 2500x faster | Medium |

---

## 🔗 Integration with ScratchV

Add verification to your workflow:

```bash
# 1. Compile with ScratchV (RISC-V backend)
scratchv model.onnx -o output.s --optimize

# 2. Compile with LLVM backend
scratchv model.onnx --backend llvm -o model.ll --optimize

# 3. Verify against ONNX Runtime
scratchv model.onnx --verify

# 4. LLVM IR toolchain
opt -O2 model.ll -o optimized.bc   # LLVM optimization
llc model.ll -o model.s            # LLVM → native assembly
lli model.ll                       # LLVM JIT execution

# 5. Compare instruction counts
#    (before vs after optimization)
```

---

## LLVM IR Verification

ScratchV can generate **LLVM IR** (`.ll`) as an alternative backend target:

- **Zero dependencies**: LLVM IR is generated as human-readable text
- **Optimization pipeline**: LLVM's `opt` tool applies additional optimization
- **JIT execution**: `lli` runs LLVM IR directly on your machine
- **Cross-compilation**: `llc` targets any architecture LLVM supports

### Pipeline

```
ONNX/DSL → ScratchV IR → Optimizer → LLVM IR → opt → lli/JIT → Result
                                           ↘
                                     ONNX Runtime → Reference → Compare
```

### Verification with numpy reference

```python
from scratchv.verification.verifier import numpy_reference, DSLInterpreter

# Numpy reference computation for any op
result = numpy_reference("Relu", np.array([-1.0, 0.0, 1.0]))

# Full DSL program interpretation
interpreter = DSLInterpreter()
result = interpreter.run(dsl_source, {"x": input_array})
```

### Verification with ONNX Runtime

```python
from scratchv.verification.verifier import verify_onnx_model

result = verify_onnx_model("model.onnx", verbose=True)
# Returns: {"success": bool, "max_error": float, ...}
```

### End-to-end verification

```bash
# Full pipeline demo
python examples/end_to_end_pipeline.py --backend llvm

# ONNX → LLVM → Reference comparison
python examples/onnx_llvm_verification.py

# Optimization impact analysis
python examples/llvm_optimization_pipeline.py
```

See `scratchv/verification/verifier.py` for the full verification framework.

See `scratchv/backend/llvm_codegen.py` for the LLVM IR codegen backend.
