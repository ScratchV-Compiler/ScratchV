"""Unified control-flow graph (CFG) core for ScratchV.

This module is the single source of truth for CFG data structures and graph
algorithms used by both IR-level optimisations and machine-level passes.

Design contract
---------------
* A CFG is built from a :class:`CFGAdapter`.  The adapter hides the concrete
  instruction representation (``Instruction`` for IR, ``MachineInstr`` for
  machine code) behind a small protocol.
* The builder partitions a function into basic blocks at labels and
  terminators.  Adapters are responsible for normalising loop sugar such as
  ``FOR``/``ENDFOR`` before the CFG is built.
* A function always has exactly one entry block.  An empty function produces
  an empty entry block rather than a CFG without nodes.
* ``FALLTHROUGH``, ``BRANCH`` and ``JUMP`` are control-flow edges.  ``CALL``
  is retained for compatibility and visualisation, but a call instruction is
  not a terminator: it stays inside its block and is followed by normal
  fallthrough.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

BlockId = str
InstructionId = str
ValueId = str


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------

class EdgeType(enum.Enum):
    """Kinds of control-flow edges in a CFG."""

    FALLTHROUGH = "fallthrough"
    BRANCH = "branch"
    JUMP = "jump"
    CALL = "call"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CFGEdge:
    """A directed edge between two basic blocks.

    ``condition`` is ``"true"``/``"false"`` for conditional branches and
    ``None`` for ordinary fallthrough or unconditional jumps.
    """

    source: str
    target: str
    edge_type: EdgeType = EdgeType.FALLTHROUGH
    condition: Optional[str] = None


@dataclass
class CFGNode:
    """A basic block in a CFG.

    ``instructions`` is the block's executable instruction sequence.
    Historically, standalone ``CFGNode`` instances also accepted an integer
    instruction *count*; that compatibility is preserved by keeping the field
    untyped.  Code that consumes a builder-produced CFG should treat it as a
    sequence.  ``instruction_ids`` is the canonical id list for analyses such
    as liveness.
    """

    name: str
    instructions: Any = 0
    is_entry: bool = False
    is_exit: bool = False
    terminator_opcode: Optional[str] = None
    block_id: Optional[BlockId] = None
    instruction_ids: list[InstructionId] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.block_id is None:
            self.block_id = self.name


@dataclass
class NaturalLoop:
    """A natural loop identified from dominator information."""

    header: str
    body: set[str] = field(default_factory=set)
    back_edges: list[tuple[str, str]] = field(default_factory=list)
    parent: Optional[str] = None
    children: list[str] = field(default_factory=list)
    nesting_depth: int = 0


# ---------------------------------------------------------------------------
# ControlFlowGraph
# ---------------------------------------------------------------------------

def _dot_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "")
    )


def _instruction_preview(node: CFGNode) -> tuple[int, list[str]]:
    """Return ``(count, preview_lines)`` for a node's instruction display."""
    if isinstance(node.instructions, int):
        count = node.instructions
        return count, []

    try:
        instructions = list(node.instructions)
    except TypeError:
        return 0, []
    preview = [str(instr)[:60] for instr in instructions[:3]]
    return len(instructions), preview


@dataclass
class ControlFlowGraph:
    """Control-flow graph for one function.

    Nodes are keyed by block name.  The builder guarantees ``entry`` is one of
    those keys.  Analyses are intentionally read-only with respect to the
    graph; callers that mutate ``nodes`` or ``edges`` are responsible for
    re-running any cached analysis.
    """

    function_name: str
    nodes: dict[str, CFGNode] = field(default_factory=dict)
    edges: list[CFGEdge] = field(default_factory=list)
    entry: str = "entry"

    # -- adjacency queries -------------------------------------------------

    def successors(self, block_name: str) -> list[str]:
        return [edge.target for edge in self.edges if edge.source == block_name]

    def predecessors(self, block_name: str) -> list[str]:
        return [edge.source for edge in self.edges if edge.target == block_name]

    def edges_from(self, block_name: str) -> list[CFGEdge]:
        return [edge for edge in self.edges if edge.source == block_name]

    # -- reachability ------------------------------------------------------

    @property
    def reachable_nodes(self) -> set[str]:
        visited: set[str] = set()
        stack = [self.entry]
        while stack:
            node = stack.pop()
            if node in visited or node not in self.nodes:
                continue
            visited.add(node)
            stack.extend(self.successors(node))
        return visited

    @property
    def unreachable_blocks(self) -> set[str]:
        """Return block names that cannot be reached from ``entry``.

        This method only reports unreachable blocks; it never mutates the CFG.
        """
        return set(self.nodes) - self.reachable_nodes

    # -- instruction ids ----------------------------------------------------

    def instruction_ids(self, block_name: str) -> list[InstructionId]:
        node = self.nodes.get(block_name)
        return list(node.instruction_ids) if node is not None else []

    # -- DOT output ----------------------------------------------------------

    def to_dot(
        self,
        highlight_loops: bool = False,
        loop_headers: Optional[set[str]] = None,
    ) -> str:
        """Render the CFG as a Graphviz directed graph."""

        lines = [f'digraph "CFG_{self.function_name}" {{']
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box, style=filled];")

        loop_set = set(loop_headers or ())
        if highlight_loops and not loop_headers:
            loop_set = {loop.header for loop in detect_loops(self)}

        for name, node in self.nodes.items():
            if node.is_entry:
                color = "#90EE90"
            elif name in loop_set:
                color = "#87CEEB"
            elif node.is_exit:
                color = "#FF6B6B"
            else:
                color = "lightyellow"

            count, preview = _instruction_preview(node)
            label_parts = [f"[{name}]"]
            label_parts.extend(preview)
            if count > len(preview):
                label_parts.append(f"... (+{count - len(preview)} more)")
            label = _dot_escape("\n".join(label_parts))

            lines.append(
                f'    "{_dot_escape(name)}" [fillcolor="{color}", '
                f'label="{label}"];'
            )

        for edge in self.edges:
            if edge.edge_type is EdgeType.FALLTHROUGH:
                style = "color=black"
            elif edge.edge_type is EdgeType.BRANCH:
                cond = f" [{edge.condition}]" if edge.condition else ""
                style = (
                    'color=blue, style=dashed, fontcolor=blue, '
                    f'label="{cond}"'
                )
            elif edge.edge_type is EdgeType.JUMP:
                style = "color=red"
            elif edge.edge_type is EdgeType.CALL:
                style = "color=purple, style=dotted"
            else:  # pragma: no cover - defensive
                style = "color=black"

            lines.append(
                f'    "{_dot_escape(edge.source)}" -> '
                f'"{_dot_escape(edge.target)}" [{style}];'
            )

        lines.append("}")
        return "\n".join(lines)


# Backwards-compatible name used by earlier ScratchV modules/tests.
CFG = ControlFlowGraph


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------

class CFGAdapter(Protocol):
    """Abstraction the CFG builder uses to inspect any instruction stream.

    Implementations must normalise control-flow sugar before returning
    ``blocks()`` so that the builder only has to understand labels,
    terminators, targets, and fallthrough.
    """

    @property
    def function_name(self) -> str: ...

    def blocks(self) -> Sequence[Any]: ...

    def block_name(self, block: Any) -> str: ...

    def instructions(self, block: Any) -> Sequence[Any]: ...

    def is_label(self, instr: Any) -> bool: ...

    def is_terminator(self, instr: Any) -> bool: ...

    def branch_targets(self, instr: Any) -> Sequence[str]: ...

    def has_fallthrough(self, instr: Any) -> bool: ...

    def opcode_name(self, instr: Any) -> str: ...


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _last_instruction(
    adapter: CFGAdapter,
    instructions: Sequence[Any],
) -> Optional[Any]:
    for instr in reversed(instructions):
        if not adapter.is_label(instr):
            return instr
    return None


def _add_edge(
    cfg: ControlFlowGraph,
    source: str,
    target: str,
    edge_type: EdgeType,
    condition: Optional[str] = None,
) -> None:
    cfg.edges.append(
        CFGEdge(source=source, target=target, edge_type=edge_type,
                condition=condition)
    )


def build_cfg(adapter: CFGAdapter) -> ControlFlowGraph:
    """Build a CFG for one function described by ``adapter``.

    Raises nothing for unreachable blocks; validation is handled separately by
    ``verify_cfg`` so that reporting is deterministic and non-destructive.
    """

    blocks = list(adapter.blocks())
    cfg = ControlFlowGraph(function_name=adapter.function_name)

    if not blocks:
        node = CFGNode(
            name="entry",
            instructions=[],
            is_entry=True,
            is_exit=False,
            block_id="entry",
            instruction_ids=[],
        )
        cfg.entry = "entry"
        cfg.nodes["entry"] = node
        return cfg

    names: list[str] = []
    seen_names: set[str] = set()
    for index, block in enumerate(blocks):
        name = adapter.block_name(block)
        if name in seen_names:
            raise ValueError(f"duplicate basic block name: {name}")
        seen_names.add(name)
        names.append(name)

        instructions = [
            instr for instr in adapter.instructions(block)
            if not adapter.is_label(instr)
        ]
        node = CFGNode(
            name=name,
            instructions=instructions,
            is_entry=(index == 0),
            block_id=name,
            instruction_ids=[f"{name}:{i}" for i in range(len(instructions))],
        )

        last = _last_instruction(adapter, instructions)
        if last is not None and adapter.is_terminator(last):
            node.terminator_opcode = adapter.opcode_name(last)
            targets = list(adapter.branch_targets(last))
            if not targets and not adapter.has_fallthrough(last):
                node.is_exit = True

        cfg.nodes[name] = node

    cfg.entry = names[0]

    for index, name in enumerate(names):
        node = cfg.nodes[name]
        if isinstance(node.instructions, int):
            instructions: list[Any] = []
        else:
            instructions = list(node.instructions)

        last = _last_instruction(adapter, instructions)
        next_name = names[index + 1] if index + 1 < len(names) else None

        if last is None:
            if next_name is not None:
                _add_edge(cfg, name, next_name, EdgeType.FALLTHROUGH)
            continue

        if not adapter.is_terminator(last):
            if next_name is not None:
                _add_edge(cfg, name, next_name, EdgeType.FALLTHROUGH)
            continue

        targets = list(adapter.branch_targets(last))
        has_fallthrough = adapter.has_fallthrough(last)

        if targets:
            # A multi-target or fallthrough terminator is conditional.
            edge_type = (
                EdgeType.BRANCH
                if len(targets) > 1 or has_fallthrough
                else EdgeType.JUMP
            )
            for target_index, target in enumerate(targets):
                condition: Optional[str] = None
                if edge_type is EdgeType.BRANCH:
                    if target_index == 0:
                        condition = "true"
                    elif target_index == 1:
                        condition = "false"
                    else:
                        condition = f"case_{target_index}"
                _add_edge(cfg, name, target, edge_type, condition)

            if has_fallthrough and next_name is not None:
                fall_condition = "false" if edge_type is EdgeType.BRANCH else None
                _add_edge(
                    cfg, name, next_name, EdgeType.FALLTHROUGH,
                    fall_condition,
                )
        elif has_fallthrough and next_name is not None:
            _add_edge(cfg, name, next_name, EdgeType.FALLTHROUGH)

    return cfg


# ---------------------------------------------------------------------------
# Generic instruction partitioning (IR-compatible)
# ---------------------------------------------------------------------------

def _instruction_opcode_name(instr: Any) -> Optional[str]:
    opcode = getattr(instr, "opcode", None)
    if opcode is None:
        opcode = getattr(instr, "op", None)
    if opcode is None:
        return None
    return getattr(opcode, "name", str(opcode))


def partition_basic_blocks_with_names(
    instructions: Sequence[Any],
    entry_name: str = "entry",
) -> list[tuple[str, list[Any]]]:
    """Partition a flat instruction list at labels and terminators.

    This helper is kept for compatibility with the Topic 11 API.  It does not
    normalise ``FOR``/``ENDFOR``; use :class:`IRCFGAdapter` for the unified
    builder path.
    """

    result: list[tuple[str, list[Any]]] = []
    current: list[Any] = []
    current_name: Optional[str] = None
    auto_id = 0

    for instr in instructions:
        op_name = _instruction_opcode_name(instr)

        if op_name == "LABEL":
            if current:
                result.append((current_name or f"b{auto_id}", current))
                auto_id += 1
            current_name = getattr(instr, "target", None) or f"L_{auto_id}"
            current = []
            continue

        if current_name is None:
            current_name = entry_name if not result else f"b{auto_id}"
            if result:
                auto_id += 1

        current.append(instr)

        if op_name in ("BR", "BR_IF", "RETURN"):
            result.append((current_name, current))
            current_name = None
            current = []

    if current:
        result.append((current_name or f"b{auto_id}", current))

    return result


@dataclass
class _PartitionedBlock:
    name: str
    instructions: list[Any]


class _PartitionedInstructionAdapter:
    """Adapter used by the legacy ``build_cfg_from_instructions`` helper."""

    def __init__(
        self,
        instructions: Sequence[Any],
        function_name: str,
        entry_name: str,
    ) -> None:
        partitioned = partition_basic_blocks_with_names(
            list(instructions), entry_name
        )
        self._function_name = function_name
        self._blocks = [
            _PartitionedBlock(name=name, instructions=instrs)
            for name, instrs in partitioned
        ]

    @property
    def function_name(self) -> str:
        return self._function_name

    def blocks(self) -> Sequence[Any]:
        return self._blocks

    def block_name(self, block: Any) -> str:
        return block.name

    def instructions(self, block: Any) -> Sequence[Any]:
        return block.instructions

    def is_label(self, instr: Any) -> bool:
        return _instruction_opcode_name(instr) == "LABEL"

    def is_terminator(self, instr: Any) -> bool:
        return _instruction_opcode_name(instr) in ("BR", "BR_IF", "RETURN")

    def branch_targets(self, instr: Any) -> Sequence[str]:
        op_name = _instruction_opcode_name(instr)
        if op_name == "BR":
            return [instr.target] if getattr(instr, "target", None) else []
        if op_name == "BR_IF":
            return [
                item.strip()
                for item in (getattr(instr, "target", "") or "").split(",")
                if item.strip()
            ]
        return []

    def has_fallthrough(self, instr: Any) -> bool:
        if _instruction_opcode_name(instr) != "BR_IF":
            return False
        return len(self.branch_targets(instr)) == 1

    def opcode_name(self, instr: Any) -> str:
        return _instruction_opcode_name(instr) or "unknown"


def build_cfg_from_instructions(
    instructions: Sequence[Any],
    function_name: str = "main",
    entry_name: str = "entry",
) -> ControlFlowGraph:
    """Build a CFG from a flat IR instruction list.

    Kept as a compatibility entry point.  For new code, prefer an adapter plus
    :func:`build_cfg`.
    """

    adapter = _PartitionedInstructionAdapter(
        instructions=instructions,
        function_name=function_name,
        entry_name=entry_name,
    )
    return build_cfg(adapter)


# ---------------------------------------------------------------------------
# Compatibility builder facade
# ---------------------------------------------------------------------------

class CFGBuilder:
    """Backwards-compatible builder facade.

    ``CFGBuilder(program).build()`` is equivalent to building one CFG per IR
    function through :class:`IRCFGAdapter`.  The graph-analysis methods below
    delegate to the module-level algorithms.
    """

    def __init__(self, program: Any = None):
        self.program = program

    def build(self, program: Any = None) -> dict[str, ControlFlowGraph]:
        target = program if program is not None else self.program
        if target is None:
            return {}

        from scratchv.analysis.adapters import IRCFGAdapter

        return {
            func.name: build_cfg(IRCFGAdapter(func))
            for func in target.functions
        }

    # -- reachability -------------------------------------------------------

    def eliminate_unreachable(self, cfg: ControlFlowGraph) -> set[str]:
        reachable = cfg.reachable_nodes
        removed = set(cfg.nodes) - reachable
        for name in removed:
            del cfg.nodes[name]
        cfg.edges = [
            edge
            for edge in cfg.edges
            if edge.source in reachable and edge.target in reachable
        ]
        return removed

    # -- dominators ---------------------------------------------------------

    def compute_dominators(
        self, cfg: ControlFlowGraph
    ) -> dict[str, set[str]]:
        return compute_dominators(cfg)

    def compute_dominator_tree(
        self, cfg: ControlFlowGraph
    ) -> dict[str, Optional[str]]:
        return compute_dominator_tree(cfg)

    # -- loops --------------------------------------------------------------

    def detect_loops(self, cfg: ControlFlowGraph) -> list[NaturalLoop]:
        return detect_loops(cfg)

    def detect_nested_loops(self, cfg: ControlFlowGraph) -> list[NaturalLoop]:
        return detect_nested_loops(cfg)


# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------

def compute_dominators(
    cfg: ControlFlowGraph,
) -> dict[str, set[str]]:
    """Compute the set of dominators for every block.

    ``Dom(entry) = {entry}`` and ``Dom(n) = all nodes`` initially for every
    other node.  Iteration stops when a fixed point is reached.
    """

    all_nodes = set(cfg.nodes)
    if not all_nodes:
        return {}

    dom: dict[str, set[str]] = {
        node: (all_nodes.copy() if node != cfg.entry else {cfg.entry})
        for node in all_nodes
    }

    changed = True
    while changed:
        changed = False
        for node in all_nodes:
            if node == cfg.entry:
                continue
            preds = cfg.predecessors(node)
            if preds:
                new_dom = set.intersection(
                    *(dom[pred] for pred in preds if pred in dom)
                )
            else:
                new_dom = set()
            new_dom.add(node)
            if new_dom != dom[node]:
                dom[node] = new_dom
                changed = True

    return dom


def compute_dominator_tree(
    cfg: ControlFlowGraph,
) -> dict[str, Optional[str]]:
    """Compute immediate dominators.

    For a node other than ``entry``, the immediate dominator is the strict
    dominator that dominates every other strict dominator of that node.  This
    is implemented in ``O(N^2)`` theoretical worst case, which is appropriate
    for the CFGs produced by ScratchV (typically a few dozen blocks).
    """

    dom_sets = compute_dominators(cfg)
    idom: dict[str, Optional[str]] = {}

    for node in dom_sets:
        if node == cfg.entry:
            idom[node] = None
            continue

        strict = dom_sets[node] - {node}
        if not strict:
            idom[node] = None
            continue

        # The immediate dominator is the strict dominator with the largest
        # dominator set: it is "closest" to ``node`` in the dominance partial
        # order.
        idom[node] = max(strict, key=lambda cand: len(dom_sets[cand]))

    return idom


def _loop_body(
    cfg: ControlFlowGraph,
    source: str,
    header: str,
) -> set[str]:
    body: set[str] = set()
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if node == header or node in body:
            continue
        body.add(node)
        for pred in cfg.predecessors(node):
            if pred not in body and pred != header:
                queue.append(pred)
    body.add(header)
    return body


def detect_loops(cfg: ControlFlowGraph) -> list[NaturalLoop]:
    """Detect natural loops from back edges in the dominator tree."""

    dom = compute_dominators(cfg)
    back_edges = [
        (edge.source, edge.target)
        for edge in cfg.edges
        if edge.target in dom.get(edge.source, set())
    ]

    loops: list[NaturalLoop] = []
    seen: dict[str, NaturalLoop] = {}

    for source, header in back_edges:
        if header in seen:
            loop = seen[header]
            loop.back_edges.append((source, header))
            loop.body |= _loop_body(cfg, source, header)
            continue

        loop = NaturalLoop(
            header=header,
            body=_loop_body(cfg, source, header),
            back_edges=[(source, header)],
        )
        loops.append(loop)
        seen[header] = loop

    return loops


def detect_nested_loops(cfg: ControlFlowGraph) -> list[NaturalLoop]:
    """Detect loops and populate ``parent``/``children``/``nesting_depth``."""

    loops = detect_loops(cfg)
    for outer in loops:
        outer.children = []
        outer.nesting_depth = 0

    for outer in loops:
        for inner in loops:
            if inner is outer:
                continue
            if (
                inner.header in outer.body
                and inner.body != outer.body
                and inner.body.issubset(outer.body)
            ):
                if inner.header not in outer.children:
                    outer.children.append(inner.header)
                inner.parent = outer.header
                inner.nesting_depth = max(
                    inner.nesting_depth, outer.nesting_depth + 1
                )

    return loops


def to_dot(
    cfg: ControlFlowGraph,
    highlight_loops: bool = True,
) -> str:
    """Generate a DOT string, optionally highlighting loop headers."""

    loop_headers: Optional[set[str]] = None
    if highlight_loops:
        loop_headers = {loop.header for loop in detect_loops(cfg)}
    return cfg.to_dot(
        highlight_loops=highlight_loops,
        loop_headers=loop_headers,
    )


def verify_cfg(cfg: ControlFlowGraph):
    """Validate a unified CFG without mutating it.

    The implementation lives in ``scratchv.analysis.cfg_validation``; this
    wrapper keeps the public entry point next to the CFG data structure.
    """
    from scratchv.analysis.cfg_validation import verify_cfg as _verify_cfg
    return _verify_cfg(cfg)


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