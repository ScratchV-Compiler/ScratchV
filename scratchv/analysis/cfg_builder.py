"""Backward-compatible shim for the unified CFG core.

New code should import from ``scratchv.analysis.cfg``.  This module is kept so
existing imports such as ``from scratchv.analysis.cfg_builder import CFGBuilder``
continue to work.
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