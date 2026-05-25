"""Code generation module — re-exports from scratchv_dag.

This package provides DAG-based instruction selection infrastructure.
The implementation lives in the standalone ``scratchv_dag`` package;
this module serves as a compatibility shim.
"""
# flake8: noqa
from scratchv_dag import (               # noqa: F401
    MVT,
    SDNodeOpcode,
    SDNodeFlags,
    SDValue,
    SDNode,
    SelectionDAG,
    DAGBuilder,
    DAGCombiner,
    DAGScheduler,
)

__all__ = [
    "MVT", "SDNodeOpcode", "SDNodeFlags",
    "SDValue", "SDNode", "SelectionDAG",
    "DAGBuilder", "DAGCombiner", "DAGScheduler",
]
