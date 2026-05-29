"""
scratchv_dag — LLVM-style SelectionDAG with cache-aware allocation.

DAG-based instruction selection framework inspired by
LLVM's SelectionDAG, plus a 4 MB L1 cache simulator and buddy allocator
for edge NPU scenarios. Operates standalone or as part of ScratchV.


Submodules:
    sdnode          Core SDNode / SelectionDAG types.
    selection_dag   DAG builder (IR -> DAG), DAG combiner (const folding),
                    and DAG scheduler (DAG -> machine instructions).
    cache           4 MB L1 cache simulator with LRU replacement.
    allocator       Buddy-system memory allocator with cache-line
                    alignment and scratchpad region support.
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
