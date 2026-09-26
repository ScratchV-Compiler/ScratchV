"""Backward-compatible shim for the unified CFG core.

Historical Topic 11 code imported CFG structures from ``scratchv.ir.cfg``.
The real implementation now lives in ``scratchv.analysis.cfg``; this module
only re-exports it.
"""
from scratchv.analysis.cfg import (
    BlockId,
    InstructionId,
    ValueId,
    EdgeType,
    CFGEdge,
    CFGNode,
    NaturalLoop,
    ControlFlowGraph,
    CFG,
    CFGAdapter,
    build_cfg,
    build_cfg_from_instructions,
    partition_basic_blocks_with_names,
    CFGBuilder,
    compute_dominators,
    compute_dominator_tree,
    detect_loops,
    detect_nested_loops,
    to_dot,
    verify_cfg,
)
__all__ = [
    "BlockId",
    "InstructionId",
    "ValueId",
    "EdgeType",
    "CFGEdge",
    "CFGNode",
    "NaturalLoop",
    "ControlFlowGraph",
    "CFG",
    "CFGAdapter",
    "build_cfg",
    "build_cfg_from_instructions",
    "partition_basic_blocks_with_names",
    "CFGBuilder",
    "compute_dominators",
    "compute_dominator_tree",
    "detect_loops",
    "detect_nested_loops",
    "to_dot",
    "verify_cfg",
]