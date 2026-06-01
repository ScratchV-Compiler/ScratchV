# Instruction Scheduler (List Scheduling)

## Overview

The Instruction Scheduler (`scratchv.backend.inst_scheduler`) reorders RISC-V instructions within a basic block to reduce pipeline stalls caused by data hazards. It uses list scheduling with critical path priority on a dependency DAG.

## API

```python
from scratchv.backend.inst_scheduler import InstructionScheduler, parse_instructions

scheduler = InstructionScheduler(latency_model={"add": 1, "lw": 2, "mul": 3})
instructions = parse_instructions(asm_text)
dag = scheduler.build_dag(instructions)
scheduled = scheduler.schedule(dag)
```

### `InstructionScheduler(latency_model=None)`

**Parameters:**
- `latency_model` (`dict[str, int] | None`): Opcode-to-latency mapping. Defaults to the standard RISC-V latency model.

### `build_dag(instructions) -> list[DAGNode]`

Build a dependency DAG from a list of `SchedInst` objects. Edges represent RAW (Read-After-Write), WAW (Write-After-Write) hazards.

**Edge types:**
- **RAW**: An instruction's operand use depends on a previous definition.
- **WAW**: Later definition of the same register must follow earlier definition.
- Edge weights equal the producer instruction's latency.

### `schedule(dag) -> list[SchedInst]`

Perform list scheduling. Returns instructions in scheduled order.

### `report(original, scheduled) -> str`

Generate a human-readable comparison report between original and scheduled order, including estimated cycle counts.

## Default RISC-V Latency Model

| Operation | Latency (cycles) |
|-----------|-----------------|
| Integer ALU (`add`, `sub`, `xor`, etc.) | 1 |
| Shift (`sll`, `srl`, `sra`) | 1 |
| Immediate ALU (`addi`, `ori`, etc.) | 1 |
| Multiplication (`mul`, `mulh`) | 3 |
| Division (`div`, `rem`) | 16 |
| Memory load (`lw`, `lh`, `lb`) | 2 |
| Memory store (`sw`, `sh`, `sb`) | 0* |
| Branch (`beq`, `bne`, etc.) | 1 |
| Jump (`j`, `jal`, `jalr`) | 0 |
| Pseudo (`li`, `mv`) | 1 |

*Store instructions are considered non-blocking for subsequent loads.

## Algorithm

### 1. DAG Construction

For each instruction, the scheduler identifies:
- **Reads** (`uses`): registers consumed as input
- **Writes** (`defines`): registers produced as output

Dependencies are added:
- RAW: use -> previous definition (forward edge with latency weight)
- WAW: definition -> previous definition of the same register

### 2. Priority Computation (Critical Path)

Each node's priority is the longest weighted path from that node to a leaf node (instruction with no successors). Higher priority = more urgent to schedule.

```
priority(node) = latency(node) + max(priority(successor) + edge_latency)
```

Computed via DFS topological sort in reverse order.

### 3. List Scheduling Loop

```
ready_queue = instructions with no unscheduled predecessors
while ready_queue is not empty:
    highest = instruction with max priority (then lowest original index)
    schedule(highest)
    mark highest as scheduled
    add newly-ready successors to ready_queue
```

## Example

**Input** (original order):
```asm
  lw t0, 0(a0)     # 2 cycles
  add t1, t0, t2   # RAW: depends on lw → stall 1 cycle
  mul t3, t1, t4   # RAW: depends on add → stall 3 cycles (mul latency)
```

**Scheduled order** (with independent instruction moved up):
```asm
  lw t0, 0(a0)     # 2 cycles
  lw t5, 4(a0)     # independent load, no stall
  add t1, t0, t2   # RAW satisfied
  mul t3, t1, t4   # RAW satisfied
```

**Result**: Pipeline stalls reduced, overall cycle count decreased.

## CLI Usage

```bash
python -m scratchv.backend.inst_scheduler input.s -o output.s --report
```
