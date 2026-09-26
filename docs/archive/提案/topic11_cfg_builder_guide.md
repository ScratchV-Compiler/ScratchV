# CFG Builder - User Guide

## Overview

The `CFGBuilder` in `scratchv/analysis/cfg_builder.py` constructs Control Flow Graphs from ScratchV IR programs, providing analysis capabilities including unreachable code elimination, dominator computation, and natural loop detection.

## Key Concepts

### Control Flow Graph (CFG)

A directed graph where:
- **Nodes** are basic blocks (straight-line code sequences)
- **Edges** represent control flow transitions between blocks

### Edge Types

| Type         | Description                          | DOT Style      |
|--------------|--------------------------------------|----------------|
| FALLTHROUGH  | Sequential transition to next block  | Solid line     |
| BRANCH       | Conditional branch (true/false)      | Dashed blue    |
| JUMP         | Unconditional jump                   | Solid red      |
| CALL         | Function call (reserved)             | Dotted purple  |

### Natural Loops

A natural loop is defined by:
1. A **header** node that dominates all nodes in the loop
2. At least one **back edge** pointing to the header
3. A **body** consisting of all nodes that can reach the back edge without going through the header

## Usage

### Building a CFG

```python
from scratchv.analysis.cfg_builder import CFGBuilder
from scratchv.frontend.dsl_parser import DSLParser

# Parse some IR
parser = DSLParser()
program = parser.parse(dsl_source)

# Build CFG
builder = CFGBuilder(program)
cfgs = builder.build()

# Get CFG for a specific function
cfg = cfgs["main"]
```

### Querying the CFG

```python
# Successors/predecessors
print(cfg.successors("entry"))       # list of target block names
print(cfg.predecessors("while_body")) # list of source block names

# Reachable nodes (DFS from entry)
reachable = cfg.reachable_nodes
```

### Eliminating Unreachable Code

```python
unreachable = builder.eliminate_unreachable(cfg)
print(f"Unreachable blocks: {unreachable}")
# ['dead_block_1', 'dead_block_2']
```

### Computing Dominators

```python
# Full dominator sets
dom_sets = builder.compute_dominators(cfg)
# What blocks does block_a dominate?
print(dom_sets["block_a"])

# Immediate dominator tree
idom = builder.compute_dominator_tree(cfg)
print(idom["block_b"])  # Immediate dominator of block_b
```

### Detecting Loops

```python
# Basic loop detection
loops = builder.detect_loops(cfg)
for loop in loops:
    print(f"Loop header: {loop.header}")
    print(f"  Body: {loop.body}")
    print(f"  Back edges: {loop.back_edges}")

# Nested loop detection
loops = builder.detect_nested_loops(cfg)
for loop in loops:
    print(f"Header: {loop.header}, Depth: {loop.nesting_depth}")
    if loop.parent:
        print(f"  Parent: {loop.parent}")
    if loop.children:
        print(f"  Children: {loop.children}")
```

### Generating Graphviz Output

```python
# DOT format for rendering with graphviz
from scratchv.analysis.cfg_builder import to_dot

dot_str = to_dot(cfg)
print(dot_str)

# Save to file and render
with open("cfg.dot", "w") as f:
    f.write(dot_str)
# $ dot -Tpng cfg.dot -o cfg.png
```

## DOT Visualization

Generated DOT output includes:
- Green nodes for entry block
- Red nodes for exit blocks (return)
- Blue nodes for loop headers (with `highlight_loops=True`)
- Instruction counts and terminator opcodes in node labels
- Color-coded edges by type

## Example Workflow

```python
from scratchv.frontend.dsl_extended import ExtendedDSLParser
from scratchv.analysis.cfg_builder import CFGBuilder, EdgeType

dsl = """
if (a > b):
    c = add(a, b)
else:
    c = mul(a, b)
endif
return c
"""

parser = ExtendedDSLParser()
program = parser.parse(dsl)

builder = CFGBuilder(program)
cfg = builder.build()["main"]

print(f"Nodes: {len(cfg.nodes)}")
print(f"Edges: {len(cfg.edges)}")

for edge in cfg.edges:
    print(f"  {edge.source} -> {edge.target} [{edge.edge_type.value}]")

loops = builder.detect_loops(cfg)
print(f"Loops detected: {len(loops)}")
```

## Algorithms

### Dominator Computation

Uses the iterative data-flow algorithm:
1. Initialize: entry dominates itself; all others dominated by all nodes
2. Iterate: transfer function is `OUT[B] = {B} U (intersection of OUT[P] for all predecessors P)`
3. Stop when no changes across full iteration

### Natural Loop Detection

1. Identify back edges: edge `A -> B` where B dominates A
2. For each back edge, find loop body: all nodes that can reach A without going through B
3. Add header B to the body set

### Unreachable Code Elimination

Mark-and-sweep approach:
1. DFS from entry block to mark reachable nodes
2. Unmarked nodes are unreachable and can be eliminated

## See Also

- `scratchv/ir/types.py` - IR data structures (BasicBlock, Function, Program)
- `scratchv/frontend/dsl_extended.py` - Extended parser generating CFGs with branches
- `docs/topics/topic21_ir_verifier_guide.md` - IR validation
