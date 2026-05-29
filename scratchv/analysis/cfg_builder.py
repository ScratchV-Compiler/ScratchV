"""Control Flow Graph (CFG) builder for ScratchV IR.

Constructs CFGs from IR programs, with support for:
- Basic block identification and edge construction
- Unreachable code elimination (DFS from entry)
- Natural loop detection via dominator tree
- Graphviz DOT output for visualization
- Dominator tree computation

Edge types:
    FALLTHROUGH - Sequential transition to next block
    BRANCH      - Conditional branch
    CALL        - Function call (reserved)
    JUMP        - Unconditional jump

Usage::

    from scratchv.analysis.cfg_builder import CFGBuilder

    builder = CFGBuilder(program)
    cfg = builder.build()
    print(cfg.to_dot())

    # Detect loops
    loops = builder.detect_loops(cfg)

    # Eliminate unreachable code
    builder.eliminate_unreachable(cfg)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

# moved import above
from scratchv.ir.types import (
    OpCode,
    Program,
    Function,
)


# ---------------------------------------------------------------------------
# CFG edge types
# ---------------------------------------------------------------------------

class EdgeType(enum.Enum):
    """Types of edges in the control flow graph."""
    FALLTHROUGH = "fallthrough"  # Sequential transition to next block
    BRANCH = "branch"             # Conditional branch (true/false)
    JUMP = "jump"                 # Unconditional jump
    CALL = "call"                 # Function call (reserved)


# ---------------------------------------------------------------------------
# CFG dataclass
# ---------------------------------------------------------------------------

@dataclass
class CFGEdge:
    """An edge connecting two basic blocks in a CFG.

    Attributes:
        source: Source basic block name.
        target: Target basic block name.
        edge_type: The type of control flow transition.
        condition: Optional condition label
            (e.g., "true", "false" for branches).
    """
    source: str
    target: str
    edge_type: EdgeType = EdgeType.FALLTHROUGH
    condition: Optional[str] = None


@dataclass
class CFGNode:
    """A node in the CFG, representing a basic block.

    Attributes:
        name: Block name (label).
        instructions: Number of instructions in the block.
        is_entry: Whether this block is the function entry.
        is_exit: Whether this block is an exit point.
        terminator_opcode: OpCode of the terminator instruction, if any.
    """
    name: str
    instructions: int = 0
    is_entry: bool = False
    is_exit: bool = False
    terminator_opcode: Optional[str] = None


@dataclass
class CFG:
    """A Control Flow Graph for a single function.

    Attributes:
        function_name: Name of the function this CFG belongs to.
        nodes: List of CFGNode objects keyed by block name.
        edges: List of CFGEdge objects.
        entry: Name of the entry block.
    """
    function_name: str
    nodes: dict[str, CFGNode] = field(default_factory=dict)
    edges: list[CFGEdge] = field(default_factory=list)
    entry: str = "entry"

    def successors(self, block_name: str) -> list[str]:
        """Return the successor block names for a given block."""
        return [
            e.target for e in self.edges
            if e.source == block_name
        ]

    def predecessors(self, block_name: str) -> list[str]:
        """Return the predecessor block names for a given block."""
        return [
            e.source for e in self.edges
            if e.target == block_name
        ]

    @property
    def reachable_nodes(self) -> set[str]:
        """Compute the set of reachable nodes via DFS from entry."""
        visited: set[str] = set()
        stack = [self.entry]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            if node not in self.nodes:
                continue
            visited.add(node)
            for succ in self.successors(node):
                if succ not in visited:
                    stack.append(succ)
        return visited

    def to_dot(
        self,
        highlight_loops: bool = False,
        loop_headers: Optional[set[str]] = None,
    ) -> str:
        """Generate Graphviz DOT format string for the CFG.

        Args:
            highlight_loops: If True, style loop header nodes differently.
            loop_headers: Set of block names that are loop headers.

        Returns:
            A string in Graphviz DOT format.
        """
        lines = [f'digraph "CFG_{self.function_name}" {{']
        lines.append('  rankdir=TB;')
        lines.append(
            '  node [shape=box, style=filled, fillcolor=lightyellow];'
        )

        loop_headers = loop_headers or set()

        for name, node in self.nodes.items():
            attrs = []
            if node.is_entry:
                attrs.append('fillcolor=lightgreen')
            if node.is_exit:
                attrs.append('fillcolor=lightcoral')
            if name in loop_headers:
                attrs.append('fillcolor=lightskyblue')
            attr_str = ", ".join(attrs) if attrs else ""
            label = f"{name}\\n({node.instructions} inst)"
            if node.terminator_opcode:
                label += f"\\n[{node.terminator_opcode}]"
            attr_prefix = ", " + attr_str if attr_str else ""
            lines.append(
                f'  {name} [label="{label}"{attr_prefix}];'
            )

        for edge in self.edges:
            style = {
                EdgeType.BRANCH: 'style=dashed, color=blue',
                EdgeType.JUMP: 'style=solid, color=red',
                EdgeType.FALLTHROUGH: 'style=solid',
                EdgeType.CALL: 'style=dotted, color=purple',
            }.get(edge.edge_type, "")

            label = ""
            if edge.condition:
                label = f', label="{edge.condition}"'

            lines.append(
                f'  {edge.source} -> {edge.target} [{style}{label}];'
            )

        lines.append("}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CFGBuilder
# ---------------------------------------------------------------------------

class CFGBuilder:
    """Builds control flow graphs from ScratchV IR Programs.

    Usage::

        builder = CFGBuilder(program)
        cfg = builder.build()
        builder.eliminate_unreachable(cfg)
        loops = builder.detect_loops(cfg)

    Attributes:
        program: The IR Program to analyze.
    """

    def __init__(self, program: Program):
        """Initialize the CFG builder.

        Args:
            program: A ScratchV IR Program.
        """
        self.program = program

    # -------------------------------------------------------------------
    # CFG construction
    # -------------------------------------------------------------------

    def build(self) -> dict[str, CFG]:
        """Build CFGs for all functions in the program.

        Returns:
            A dict mapping function name to CFG.
        """
        cfgs: dict[str, CFG] = {}
        for func in self.program.functions:
            cfgs[func.name] = self._build_function_cfg(func)
        return cfgs

    def _build_function_cfg(self, func: Function) -> CFG:
        """Build a CFG for a single function.

        Args:
            func: The function to analyze.

        Returns:
            A CFG object.
        """
        cfg = CFG(function_name=func.name)

        if not func.blocks:
            return cfg

        cfg.entry = func.blocks[0].name
        # _block_names and block_map are available for future use

        # Create nodes
        for i, block in enumerate(func.blocks):
            is_entry = (i == 0)
            node = CFGNode(
                name=block.name,
                instructions=len(block.instructions),
                is_entry=is_entry,
                is_exit=False,
                terminator_opcode=None,
            )
            # Check for terminator
            for instr in block.instructions:
                if instr.opcode in (
                    OpCode.RETURN, OpCode.BR, OpCode.BR_IF,
                    OpCode.FOR, OpCode.ENDFOR,
                ):
                    node.terminator_opcode = instr.opcode.name
                    if instr.opcode == OpCode.RETURN:
                        node.is_exit = True

            cfg.nodes[block.name] = node

        # Create edges
        for i, block in enumerate(func.blocks):
            insts = block.instructions
            if not insts:
                # Empty block falls through to next
                if i + 1 < len(func.blocks):
                    cfg.edges.append(CFGEdge(
                        source=block.name,
                        target=func.blocks[i + 1].name,
                        edge_type=EdgeType.FALLTHROUGH,
                    ))
                continue

            last_instr = insts[-1]

            if last_instr.opcode == OpCode.BR:
                target = last_instr.target
                if target:
                    cfg.edges.append(CFGEdge(
                        source=block.name,
                        target=target,
                        edge_type=EdgeType.JUMP,
                    ))

            elif last_instr.opcode == OpCode.BR_IF:
                target = last_instr.target
                if target and "," in target:
                    true_target, false_target = target.split(",", 1)
                    cfg.edges.append(CFGEdge(
                        source=block.name,
                        target=true_target.strip(),
                        edge_type=EdgeType.BRANCH,
                        condition="true",
                    ))
                    cfg.edges.append(CFGEdge(
                        source=block.name,
                        target=false_target.strip(),
                        edge_type=EdgeType.BRANCH,
                        condition="false",
                    ))

            elif last_instr.opcode == OpCode.RETURN:
                # No outgoing edges from return
                pass

            elif last_instr.opcode == OpCode.FOR:
                # FOR implicitly branches to the loop body and to the loop exit
                # Track next endfor for the exit target; fallthrough for now
                pass

            elif last_instr.opcode == OpCode.ENDFOR:
                pass

            else:
                # Fallthrough to next block
                if i + 1 < len(func.blocks):
                    cfg.edges.append(CFGEdge(
                        source=block.name,
                        target=func.blocks[i + 1].name,
                        edge_type=EdgeType.FALLTHROUGH,
                    ))

        return cfg

    # -------------------------------------------------------------------
    # Unreachable code elimination
    # -------------------------------------------------------------------

    def eliminate_unreachable(self, cfg: CFG) -> set[str]:
        """Compute and return the set of unreachable block names.

        Uses DFS from the entry block to mark reachable nodes, then
        identifies blocks that are not reachable.

        Args:
            cfg: The control flow graph to analyze.

        Returns:
            Set of unreachable block names.
        """
        reachable = cfg.reachable_nodes
        all_nodes = set(cfg.nodes.keys())
        unreachable = all_nodes - reachable
        return unreachable

    # -------------------------------------------------------------------
    # Dominator tree computation
    # -------------------------------------------------------------------

    def compute_dominators(self, cfg: CFG) -> dict[str, set[str]]:
        """Compute dominator sets for each block in the CFG.

        A block D dominates block B if every path from the entry to B
        must pass through D. Uses the iterative data-flow algorithm.

        Args:
            cfg: The control flow graph.

        Returns:
            Dict mapping block name to set of block names it dominates.
        """
        all_nodes = set(cfg.nodes.keys())
        if not all_nodes:
            return {}

        # Initialize: entry dominates itself; all others initially
        # dominated by everything
        dom: dict[str, set[str]] = {}
        for name in all_nodes:
            if name != cfg.entry:
                dom[name] = all_nodes.copy()
            else:
                dom[name] = {cfg.entry}

        changed = True
        while changed:
            changed = False
            for node in all_nodes:
                if node == cfg.entry:
                    continue
                preds = cfg.predecessors(node)
                if not preds:
                    continue
                # Intersection of all predecessors' dominator sets
                new_dom = dom[preds[0]].copy() if preds else set()
                for pred in preds[1:]:
                    new_dom &= dom[pred]
                new_dom.add(node)
                if new_dom != dom[node]:
                    dom[node] = new_dom
                    changed = True

        return dom

    def compute_dominator_tree(self, cfg: CFG) -> dict[str, Optional[str]]:
        """Compute the immediate dominator for each block.

        The immediate dominator of B is the unique node that strictly
        dominates B but does not strictly dominate any other strict
        dominator of B.

        Args:
            cfg: The control flow graph.

        Returns:
            Dict mapping block name to immediate dominator
            (or None for entry).
        """
        dom_sets = self.compute_dominators(cfg)
        idom: dict[str, Optional[str]] = {}

        for node, doms in dom_sets.items():
            if node == cfg.entry:
                idom[node] = None
                continue
            strict_doms = doms - {node}
            if not strict_doms:
                idom[node] = None
                continue
            # Find the strict dominator that doesn't dominate any other strict
            # dominator (i.e., the "closest" one)
            idom[node] = None
            for d in strict_doms:
                is_immediate = True
                for other in strict_doms:
                    if other != d and d in (dom_sets[other] - {other}):
                        is_immediate = False
                        break
                if is_immediate:
                    idom[node] = d
                    break

        return idom

    # -------------------------------------------------------------------
    # Natural loop detection
    # -------------------------------------------------------------------

    def detect_loops(self, cfg: CFG) -> list[NaturalLoop]:
        """Detect natural loops in the CFG.

        A natural loop has:
        - A header node that dominates all nodes in the loop
        - At least one back edge pointing to the header
        - A body consisting of all nodes that can reach the back edge
          without going through the header

        Args:
            cfg: The control flow graph.

        Returns:
            A list of NaturalLoop objects.
        """
        dom_sets = self.compute_dominators(cfg)
        back_edges: list[tuple[str, str]] = []

        # Find back edges: target dominates source
        for edge in cfg.edges:
            if edge.source in dom_sets.get(edge.target, set()):
                # source is dominated by target -> back edge
                back_edges.append((edge.source, edge.target))

        loops: list[NaturalLoop] = []
        for source, header in back_edges:
            # Find loop body: all nodes that can reach source without
            # going through header
            body: set[str] = set()
            stack = [source]
            while stack:
                node = stack.pop()
                if node == header:
                    continue
                if node in body:
                    continue
                body.add(node)
                for pred in cfg.predecessors(node):
                    if pred not in body:
                        stack.append(pred)

            # Create loop
            loop = NaturalLoop(
                header=header,
                body=body | {header},
                back_edges=[(source, header)],
            )
            loops.append(loop)

        return loops

    def detect_nested_loops(self, cfg: CFG) -> list[NaturalLoop]:
        """Detect loops including nesting relationships.

        After detecting loops, computes which loops are nested inside others.
        A loop L1 is nested inside L2 if L1's body is a subset of L2's body
        and L1 != L2.

        Args:
            cfg: The control flow graph.

        Returns:
            A list of NaturalLoop objects with nesting relationships populated.
        """
        loops = self.detect_loops(cfg)

        for i, outer in enumerate(loops):
            for j, inner in enumerate(loops):
                if i == j:
                    continue
                if (inner.header in outer.body
                        and inner.body.issubset(outer.body)):
                    if inner.body != outer.body:
                        inner.nesting_depth = outer.nesting_depth + 1
                        inner.parent = outer.header
                        outer.children.append(inner.header)

        return loops


# ---------------------------------------------------------------------------
# NaturalLoop dataclass
# ---------------------------------------------------------------------------

@dataclass
class NaturalLoop:
    """Represents a natural loop in a CFG.

    Attributes:
        header: The header block name (loop entry point).
        body: Set of block names in the loop body (including header).
        back_edges: List of (source, header) back edge pairs.
        parent: Header of the enclosing loop, if nested.
        children: Headers of loops nested inside this one.
        nesting_depth: Nesting depth (0 = outermost).
    """
    header: str
    body: set[str] = field(default_factory=set)
    back_edges: list[tuple[str, str]] = field(default_factory=list)
    parent: Optional[str] = None
    children: list[str] = field(default_factory=list)
    nesting_depth: int = 0


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def to_dot(cfg: CFG, highlight_loops: bool = True) -> str:
    """Generate Graphviz DOT format for a CFG, with optional loop highlighting.

    Args:
        cfg: The CFG to visualize.
        highlight_loops: Whether to detect and highlight loops.

    Returns:
        A DOT format string.
    """
    loop_headers: Optional[set[str]] = None
    if highlight_loops:
        builder = CFGBuilder(Program())  # dummy program for standalone use
        loops = builder.detect_loops(cfg)
        loop_headers = {loop.header for loop in loops}
    return cfg.to_dot(
        highlight_loops=highlight_loops, loop_headers=loop_headers
    )
