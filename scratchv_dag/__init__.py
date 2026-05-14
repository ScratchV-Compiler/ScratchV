"""
scratchv_dag — LLVM-style SelectionDAG infrastructure with cache-aware memory allocation.

This package provides a DAG-based instruction selection framework inspired by
LLVM's SelectionDAG, plus a 4 MB L1 cache simulator and a buddy-system memory
allocator designed for edge NPU scenarios. It operates as a standalone component
or as part of the ScratchV compiler toolchain.

Submodules:
    sdnode          Core SDNode / SelectionDAG types (opcodes, MVT, DAG container).
    selection_dag   DAG builder (IR → DAG), DAG combiner (constant folding),
                    and DAG scheduler (DAG → machine instructions).
    cache           4 MB set-associative L1 cache simulator with LRU replacement.
    allocator       Buddy-system memory allocator with cache-line alignment
                    and scratchpad region support.
"""

from __future__ import annotations

from scratchv_dag.sdnode import (
    MVT,
    SDNodeOpcode,
    SDNodeFlags,
    SDValue,
    SDNode,
    SelectionDAG,
)
from scratchv_dag.selection_dag import (
    DAGBuilder,
    DAGCombiner,
    DAGScheduler,
)
from scratchv_dag.cache import (
    L1Cache,
    CacheConfig,
    CacheStats,
)
from scratchv_dag.allocator import (
    MemoryAllocator,
    AllocationPolicy,
    MemoryRegion,
    AllocStats,
)

__all__ = [
    # sdnode
    "MVT", "SDNodeOpcode", "SDNodeFlags", "SDValue", "SDNode",
    "SelectionDAG",
    # selection_dag
    "DAGBuilder", "DAGCombiner", "DAGScheduler",
    # cache
    "L1Cache", "CacheConfig", "CacheStats",
    # allocator
    "MemoryAllocator", "AllocationPolicy", "MemoryRegion", "AllocStats",
]

__version__ = "0.1.0"
