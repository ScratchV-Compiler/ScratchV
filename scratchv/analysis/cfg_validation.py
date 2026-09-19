"""Unified CFG validation and diagnostics.

The builder deliberately tolerates unreachable blocks so analyses can inspect
them.  ``verify_cfg`` is the explicit, non-destructive validation step that
reports structural problems with stable diagnostic codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from scratchv.analysis.cfg import ControlFlowGraph, EdgeType


@dataclass(frozen=True)
class CFGDiagnostic:
    """A structured validation finding."""

    severity: str
    code: str
    message: str
    block: Optional[str] = None
    edge: Optional[tuple[str, str]] = None


def _opcode_name(instr: Any) -> Optional[str]:
    opcode = getattr(instr, "opcode", None)
    if opcode is None:
        opcode = getattr(instr, "op", None)
    if opcode is None:
        return None
    return getattr(opcode, "name", str(opcode))


_IR_TERMINATORS = {"BR", "BR_IF", "RETURN"}
_MACHINE_TERMINATORS = {
    "J", "JAL", "JALR", "BEQ", "BNE", "BLT", "BGE", "BNEZ",
}
_TERMINATORS = _IR_TERMINATORS | _MACHINE_TERMINATORS


def _looks_like_terminator(instr: Any) -> bool:
    return _opcode_name(instr) in _TERMINATORS


def _check_entry(cfg: ControlFlowGraph) -> list[CFGDiagnostic]:
    diagnostics: list[CFGDiagnostic] = []
    if not cfg.nodes:
        diagnostics.append(
            CFGDiagnostic(
                severity="error",
                code="CFG_NO_ENTRY",
                message="CFG has no entry block",
            )
        )
        return diagnostics

    if cfg.entry not in cfg.nodes:
        diagnostics.append(
            CFGDiagnostic(
                severity="error",
                code="CFG_INVALID_ENTRY",
                message=f"entry block {cfg.entry!r} is not present",
                block=cfg.entry,
            )
        )
    return diagnostics


def _check_edges(cfg: ControlFlowGraph) -> list[CFGDiagnostic]:
    diagnostics: list[CFGDiagnostic] = []
    for edge in cfg.edges:
        if edge.source not in cfg.nodes:
            diagnostics.append(
                CFGDiagnostic(
                    severity="error",
                    code="CFG_DANGLING_SOURCE",
                    message=f"edge source {edge.source!r} does not exist",
                    edge=(edge.source, edge.target),
                )
            )
        if edge.target not in cfg.nodes:
            diagnostics.append(
                CFGDiagnostic(
                    severity="error",
                    code="CFG_DANGLING_TARGET",
                    message=f"edge target {edge.target!r} does not exist",
                    edge=(edge.source, edge.target),
                )
            )
    return diagnostics


def _check_terminators(cfg: ControlFlowGraph) -> list[CFGDiagnostic]:
    diagnostics: list[CFGDiagnostic] = []
    for name, node in cfg.nodes.items():
        instructions = (
            [] if isinstance(node.instructions, int)
            else list(node.instructions)
        )
        for index, instr in enumerate(instructions):
            if index == len(instructions) - 1:
                continue
            if _looks_like_terminator(instr):
                diagnostics.append(
                    CFGDiagnostic(
                        severity="error",
                        code="CFG_TERMINATOR_NOT_LAST",
                        message=(
                            "terminator instruction appears before end of "
                            "basic block"
                        ),
                        block=name,
                    )
                )
                break
    return diagnostics


def _check_control_edges(cfg: ControlFlowGraph) -> list[CFGDiagnostic]:
    diagnostics: list[CFGDiagnostic] = []
    by_source: dict[str, list[EdgeType]] = {}
    for edge in cfg.edges:
        by_source.setdefault(edge.source, []).append(edge.edge_type)

    for source, kinds in by_source.items():
        has_jump = EdgeType.JUMP in kinds
        has_fallthrough = EdgeType.FALLTHROUGH in kinds
        if has_jump and has_fallthrough:
            diagnostics.append(
                CFGDiagnostic(
                    severity="error",
                    code="CFG_JUMP_WITH_FALLTHROUGH",
                    message=(
                        "block has both an unconditional jump and a "
                        "fallthrough edge"
                    ),
                    block=source,
                )
            )
    return diagnostics


def _check_predecessor_consistency(
    cfg: ControlFlowGraph,
) -> list[CFGDiagnostic]:
    diagnostics: list[CFGDiagnostic] = []
    for name in cfg.nodes:
        for pred in cfg.predecessors(name):
            if pred not in cfg.nodes:
                diagnostics.append(
                    CFGDiagnostic(
                        severity="error",
                        code="CFG_BAD_PREDECESSOR",
                        message=(
                            f"predecessor {pred!r} of {name!r} is not a node"
                        ),
                        block=name,
                    )
                )
        for succ in cfg.successors(name):
            if succ not in cfg.nodes:
                diagnostics.append(
                    CFGDiagnostic(
                        severity="error",
                        code="CFG_BAD_SUCCESSOR",
                        message=(
                            f"successor {succ!r} of {name!r} is not a node"
                        ),
                        block=name,
                    )
                )
    return diagnostics


def verify_cfg(cfg: ControlFlowGraph) -> list[CFGDiagnostic]:
    """Validate structural invariants of a unified CFG.

    Returns a list of :class:`CFGDiagnostic` objects.  An empty list means the
    CFG satisfies the checked invariants.  The function never mutates ``cfg``.
    """

    diagnostics: list[CFGDiagnostic] = []
    diagnostics.extend(_check_entry(cfg))
    diagnostics.extend(_check_edges(cfg))
    diagnostics.extend(_check_terminators(cfg))
    diagnostics.extend(_check_control_edges(cfg))
    diagnostics.extend(_check_predecessor_consistency(cfg))

    # De-duplicate by diagnostic identity while preserving order.
    unique: list[CFGDiagnostic] = []
    seen: set[tuple[Any, ...]] = set()
    for item in diagnostics:
        key = (item.code, item.message, item.block, item.edge)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


__all__ = [
    "CFGDiagnostic",
    "verify_cfg",
]