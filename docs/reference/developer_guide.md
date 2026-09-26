# Developer Guide

This guide explains how ScratchV works internally and how to extend it.

---

## Architecture Overview

```
                     ┌─────────────────────────────────────────┐
                     │           ScratchV Compiler              │
                     │                                          │
  ONNX Model ──▶ ONNXParser ──▶ IR (3-addr) ──▶ Optimizer ──┐ │
                     │                              │       │ │
  DSL Source ──▶ DSLParser ────┘                     │       │ │
                                                      │       │ │
                            ┌─────────────────────────┘       │ │
                            ▼                                 │ │
                 ┌──────────────────────┐                      │ │
                 │ InstructionSelector  │──▶ RegAlloc ─▶ Asm   │─▶ .s
                 └──────────────────────┘                      │ │
                            │                                  │ │
                            ▼                                  │ │
                 ┌──────────────────────┐                      │ │
                 │ DAGBuilder / Sched   │──▶ (alt. pipeline)    │ │
                 │ (scratchv_dag)       │                       │ │
                 └──────────────────────┘                       │ │
                            │                                   │ │
                            ▼                                   │ │
                 ┌──────────────────────┐                      │ │
                 │ LLVMCodegen          │──▶ .ll ─▶ opt/lli     │ │
                 └──────────────────────┘                      │ │
                     ┌──────────────────────────┐               │ │
                     │ Verification Framework   │               │ │
                     │ ─ ONNX Runtime           │               │ │
                     │ ─ numpy reference        │               │ │
                     │ ─ DSL interpreter        │               │ │
                     │ ─ TinyFive sim           │               │ │
                     └──────────────────────────┘               │ │
                     ┌──────────────────────────┐               │ │
                     │ scratchv_dag              │               │ │
                     │ ─ SelectionDAG            │               │ │
                     │ ─ L1 cache simulator      │───────────────┘ │
                     │ ─ Memory allocator        │                 │
                     └──────────────────────────┘                  │
                     ┌──────────────────────────┐                  │
                     │ scratchv_dag              │                  │
                     │ ─ SDNode / MVT / DAG      │──────────────────┘
                     └──────────────────────────┘
```

## Package Map

| Package | Responsibility |
|---|---|
| `scratchv/ir/` | IR types, builder, printer |
| `scratchv/frontend/` | ONNX & DSL parsers |
| `scratchv/optimizer/` | IR → IR optimization passes |
| `scratchv/backend/` | Instruction selection, reg alloc, asm emit, LLVM codegen |
| `scratchv/verification/` | Verification against reference implementations |
| `scratchv/simulator/` | TinyFive adapter |
| `scratchv_dag/` | DAG-based instruction selection (standalone) |

## IR Reference

### Types (`scratchv/ir/types.py`)

```python
class OpCode(enum.Enum):
    ADD, SUB, MUL, DIV, NEG, EXP       # arithmetic
    LOAD, STORE, LOAD_CONST, ALLOCA     # memory
    FOR, ENDFOR, BR, BR_IF, RETURN      # control flow
    MATMUL, RELU, MAXPOOL, SOFTMAX, ... # neural-network ops

class Value:
    name: str
    dtype: DataType       # FLOAT32, INT32, FLOAT64, INT64
    is_constant: bool
    const_value: float | int | None
    shape: tuple[int, ...]

class Instruction:
    opcode: OpCode
    dest: Value | None
    operands: list[Value]
    attrs: dict           # e.g. {"value": 42} for load_const
    target: str | None    # branch target label

class BasicBlock:
    name: str
    instructions: list[Instruction]
    phi_nodes: list[Instruction]

class Function:
    name: str
    params: list[Value]
    returns: list[Value]
    blocks: list[BasicBlock]
    locals: list[Value]

class Program:
    functions: list[Function]
    global_values: list[Value]
```

### Builder (`scratchv/ir/builder.py`)

```python
builder = IRBuilder()
f = builder.new_function("add4")
bb = builder.new_block("entry")

a = builder.make_value("a")
b = builder.make_value("b")
s = builder.add(a, b)
builder.ret(s)
```

## Backend Pipeline

### Standard path (flat instruction selection)

```
IR → InstructionSelector → MachineInstr[] → RegisterAllocator → AsmEmitter → .s
```

- `InstructionSelector`: one handler per `OpCode`, emits `MachineInstr` with
  virtual registers.
- `RegisterAllocator`: two modes — `naive` (spill everything) and `greedy`
  (LRU-based, reuses callee-saved temps).
- `AsmEmitter`: `MachineInstr[]` → GAS-syntax RISC-V text.

### DAG path (experimental, via scratchv_dag)

```
IR → DAGBuilder → SelectionDAG → DAGCombiner → DAGScheduler → MachineInstr[]
```

The DAG path enables more advanced optimisations (pattern matching, better
constant folding) before scheduling back to linear instructions.

## Memory System

- `L1Cache`: set-associative cache simulator for performance estimation
  (default 4 MB, 8-way, 64 B lines, LRU replacement).
- `MemoryAllocator`: buddy allocator with cache-line alignment and
  scratchpad region (25 % of pool for explicit DMA transfers).

Both live in the standalone `scratchv_dag` package and are usable independently.

## Adding Support for a New ONNX Operator

1. **ONNX parser** (`scratchv/frontend/onnx_parser.py`):
   - Add a `_handle_<op>` method that reads inputs/outputs and emits IR.
   - Register it in the operator dispatch dict.

2. **Optional: IR opcode** (`scratchv/ir/types.py`):
   - Only if the operator cannot be decomposed into existing IR ops.

3. **Instruction selection** (`scratchv/backend/instruction_select.py`):
   - Add `_select_<op>` to lower the IR op to `MachineInstr`s.
   - For simple ops, one or two RISC-V instructions suffice.

4. **LLVM codegen** (`scratchv/backend/llvm_codegen.py`):
   - Add `_emit_<op>` to produce LLVM IR for the operator.

5. **Verification** (`scratchv/verification/verifier.py`):
   - Add a numpy reference function if existing helpers don't cover it.

6. **Tests**: add IR → assembly → verification test cases.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_ir.py -v

# Run a specific test
pytest tests/test_ir.py::TestIRBuilder::test_build_simple_add -v

# Run with coverage
pytest tests/ --cov=scratchv --cov=scratchv_dag --cov-report=html
```
