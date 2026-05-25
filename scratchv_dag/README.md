# scratchv_dag — LLVM-Style SelectionDAG & Cache-Aware Memory Allocator

**scratchv_dag** is a standalone Python package providing DAG-based instruction selection infrastructure inspired by LLVM's SelectionDAG, paired with a 4 MB L1 cache simulator and a buddy-system memory allocator designed for edge-NPU compiler toolchains.

It operates independently or as part of the [ScratchV](https://github.com/kinsomwang/ScratchV) ONNX→RISC-V compiler.

---

## Package Structure

```
scratchv_dag/
├── __init__.py          # Public API re-exports
├── sdnode.py            # Core DAG types: MVT, SDNodeOpcode, SDNode, SelectionDAG
├── selection_dag.py     # DAGBuilder, DAGCombiner, DAGScheduler
├── cache.py             # 4 MB L1 cache simulator (LRU, write-back)
├── allocator.py         # Buddy-system memory allocator with scratchpad
└── README.md
```

---

## Modules

### `sdnode` — SelectionDAG Core Types

LLVM-inspired DAG node representation:

| Type | Role |
|---|---|
| `MVT` | Machine Value Type (`i8`–`i64`, `f32`, `f64`, `Other`, `Void`) |
| `SDNodeOpcode` | 40+ node opcodes (arithmetic, memory, control, NN, RISC-V pseudo) |
| `SDNodeFlags` | Per-node flags (fast-math, volatile, alignment) |
| `SDValue` | Edge reference `(SDNode, result_index)` |
| `SDNode` | DAG node with opcode, result types, operand edges, and chain support |
| `SelectionDAG` | Node container with factory methods and deduplication |

### `selection_dag` — DAG Pipeline

Three stages transform IR → DAG → machine instructions:

```
┌──────────┐    ┌───────────┐    ┌─────────────┐    ┌──────────────┐
│  IR Insn │───▶│ DAGBuilder │───▶│ DAGCombiner │───▶│ DAGScheduler │
└──────────┘    └───────────┘    └─────────────┘    └──────────────┘
                                                          │
                                                          ▼
                                                   MachineInstr[]
```

- **DAGBuilder** — visits each IR instruction and builds the corresponding DAG sub-graph.
- **DAGCombiner** — peephole optimisations over the DAG (constant folding for integer and FP arithmetic).
- **DAGScheduler** — post-order topological sort that linearises the DAG into a `MachineInstr` list ready for register allocation.

### `cache` — 4 MB L1 Cache Simulator

Models a set-associative L1 data cache for edge-NPU performance estimation.

**Default configuration:**

| Parameter | Value |
|---|---|
| Capacity | 4 MB |
| Line size | 64 B |
| Associativity | 8-way |
| Write policy | Write-back + write-allocate |
| Hit latency | 2 cycles |
| Miss latency | 20 cycles |

```python
from scratchv_dag import L1Cache

cache = L1Cache()
cache.read(0x1000, 4)   # → latency in cycles
cache.write(0x2000, 8)  # → latency in cycles
print(cache.stats)       # CacheStats(hits=..., hit_rate=...)
```

All parameters are configurable via `CacheConfig`:

```python
from scratchv_dag import L1Cache, CacheConfig

cfg = CacheConfig(total_size=2*1024*1024, associativity=4)
cache = L1Cache(cfg)
```

### `allocator` — Cache-Aware Memory Allocator

Buddy-system allocator with L1-cache-line alignment and a scratchpad region for explicit DMA.

**Pool layout (default 4 MB):**

```
0x000000 ┌──────────────────────────────┐
         │  Scratchpad  (1 MB, 25 %)    │  ← uncached SRAM
0x100000 ├──────────────────────────────┤
         │  General     (3 MB, 75 %)    │  ← buddy-managed, cached
0x400000 └──────────────────────────────┘
```

```python
from scratchv_dag import MemoryAllocator, AllocationPolicy

alloc = MemoryAllocator(pool_size=4*1024*1024)

a = alloc.alloc(4096)                      # 64 B aligned
b = alloc.alloc(256, alignment=4096)       # 4K page aligned
s = alloc.scratchpad_alloc(1024)           # from scratchpad SRAM

alloc.free(a)
```

**Why cache-line alignment?** Edge NPUs often share cache lines across processing elements. Misaligned allocations cause false sharing and expensive L1 evictions. Defaulting to 64 B alignment avoids this at zero extra cost.

---

## Quick Start

```python
from scratchv_dag.sdnode import SelectionDAG, MVT

dag = SelectionDAG()
a = dag.get_constant(42, MVT.i32)
b = dag.get_constant(10, MVT.i32)
c = dag.get_add(a, b)
print(dag.dump())
```

```python
from scratchv_dag.cache import L1Cache
from scratchv_dag.allocator import MemoryAllocator

# Simulate a cache-friendly access pattern
cache = L1Cache()
for _ in range(10):
    for i in range(32):
        cache.read(i * 64, 4)
print(f"Hit rate: {cache.stats.hit_rate:.1%}")

# Allocate memory for two tensors
alloc = MemoryAllocator()
tensor_a = alloc.alloc(512 * 512 * 4)   # 512×512 f32
tensor_b = alloc.alloc(512 * 512 * 4)
```

---

## Python Compatibility

Requires **Python 3.8+**. The package is pure Python with no runtime dependencies beyond the standard library.

(When used with ScratchV, `onnx`, `numpy`, and `protobuf` are needed for ONNX parsing.)

---

## License

Same as ScratchV — see the [LICENSE](../LICENSE) file.
