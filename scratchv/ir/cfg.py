"""控制流图 (CFG) 生成器 — ScratchV 课题 11。

从 IR Program 构建控制流图，支撑不可达代码消除、支配树、
自然循环检测等分析。输出 Graphviz DOT 用于可视化。

用法::

    from scratchv.ir.cfg import CFGBuilder

    builder = CFGBuilder()
    cfgs = builder.build(program)
    cfg = cfgs["main"]
    print(cfg.to_dot())
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field

from scratchv.ir.types import Instruction, OpCode


# ============================================================================
# 边类型 (W3)
# ============================================================================

class EdgeType(enum.Enum):
    """控制流图中边的类型。"""
    FALLTHROUGH = "fallthrough"
    BRANCH = "branch"
    JUMP = "jump"
    CALL = "call"


# ============================================================================
# 数据结构 (W3)
# ============================================================================

@dataclass
class CFGNode:
    name: str
    instructions: list[Instruction] = field(default_factory=list)
    is_entry: bool = False
    is_exit: bool = False
    terminator_opcode: str | None = None


@dataclass
class CFGEdge:
    source: str
    target: str
    edge_type: EdgeType = EdgeType.FALLTHROUGH
    condition: str | None = None


@dataclass
class NaturalLoop:
    """自然循环结构 (W9)。"""
    header: str
    body: set[str] = field(default_factory=set)
    back_edges: list[tuple[str, str]] = field(default_factory=list)
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    nesting_depth: int = 0


@dataclass
class CFG:
    """单个函数的控制流图。"""
    function_name: str
    nodes: dict[str, CFGNode] = field(default_factory=dict)
    edges: list[CFGEdge] = field(default_factory=list)
    entry: str = "entry"

    def successors(self, name: str) -> list[str]:
        return [e.target for e in self.edges if e.source == name]

    def predecessors(self, name: str) -> list[str]:
        return [e.source for e in self.edges if e.target == name]

    def to_dot(self, loop_headers: set[str] | None = None) -> str:
        """生成 Graphviz DOT 字符串 (W6)。"""
        lines = [f'digraph "CFG_{self.function_name}" {{']
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box, style=filled];")

        loop_set = loop_headers or set()

        for name, node in self.nodes.items():
            if node.is_entry:
                color = "#90EE90"
            elif name in loop_set:
                color = "#87CEEB"
            elif node.is_exit:
                color = "#FF6B6B"
            else:
                color = "lightyellow"

            label_parts = [f"[{name}]"]
            for instr in node.instructions[:3]:
                label_parts.append(str(instr)[:60])
            if len(node.instructions) > 3:
                label_parts.append(f"... (+{len(node.instructions) - 3} more)")
            label = "\\n".join(label_parts)

            lines.append(f'    "{name}" [fillcolor="{color}", label="{label}"];')

        for edge in self.edges:
            if edge.edge_type == EdgeType.FALLTHROUGH:
                style = "color=black"
            elif edge.edge_type == EdgeType.BRANCH:
                cond = f" [{edge.condition}]" if edge.condition else ""
                style = f'color=blue, style=dashed, fontcolor=blue, label="{cond}"'
            elif edge.edge_type == EdgeType.JUMP:
                style = "color=red"
            else:
                style = "color=purple, style=dotted"
            lines.append(f'    "{edge.source}" -> "{edge.target}" [{style}];')

        lines.append("}")
        return "\n".join(lines)


# ============================================================================
# 基本块划分 (W2)
# ============================================================================

def partition_basic_blocks_with_names(
    instructions: list[Instruction],
    entry_name: str = "entry",
) -> list[tuple[str, list[Instruction]]]:
    """将线性 IR 指令切分为带名称的基本块。"""
    result: list[tuple[str, list[Instruction]]] = []
    current: list[Instruction] = []
    current_name: str | None = None
    auto_id = 0

    for instr in instructions:
        if instr.opcode == OpCode.LABEL:
            if current:
                result.append((current_name or f"b{auto_id}", current))
                auto_id += 1
            current_name = instr.target or f"L_{auto_id}"
            current = []
            continue

        if current_name is None:
            current_name = entry_name if not result else f"b{auto_id}"
            if result:
                auto_id += 1

        current.append(instr)

        if instr.opcode in (OpCode.BR, OpCode.BR_IF, OpCode.RETURN):
            result.append((current_name, current))
            current_name = None
            current = []

    if current:
        result.append((current_name or f"b{auto_id}", current))

    return result


# ============================================================================
# CFG 构建 (W4)
# ============================================================================

def _add_fallthrough(cfg: CFG, name: str, names: list[str], idx: int) -> None:
    if idx + 1 < len(names):
        cfg.edges.append(CFGEdge(name, names[idx + 1], EdgeType.FALLTHROUGH))


def build_cfg_from_instructions(
    instructions: list[Instruction],
    function_name: str = "main",
    entry_name: str = "entry",
) -> CFG:
    """从线性指令列表构建完整 CFG。"""
    partitioned = partition_basic_blocks_with_names(instructions, entry_name)

    if not partitioned:
        cfg = CFG(function_name=function_name, entry=entry_name)
        cfg.nodes[entry_name] = CFGNode(name=entry_name)
        return cfg

    cfg = CFG(function_name=function_name, entry=partitioned[0][0])
    block_names = [name for name, _ in partitioned]

    for idx, (name, instrs) in enumerate(partitioned):
        node = CFGNode(name=name, instructions=instrs, is_entry=(idx == 0))
        if instrs:
            last = instrs[-1]
            if last.opcode == OpCode.RETURN:
                node.is_exit = True
                node.terminator_opcode = "return"
            elif last.opcode == OpCode.BR:
                node.terminator_opcode = "br"
            elif last.opcode == OpCode.BR_IF:
                node.terminator_opcode = "br_if"
        cfg.nodes[name] = node

    for idx, (name, instrs) in enumerate(partitioned):
        if not instrs:
            _add_fallthrough(cfg, name, block_names, idx)
            continue

        last = instrs[-1]

        if last.opcode == OpCode.RETURN:
            pass
        elif last.opcode == OpCode.BR:
            target = last.target or ""
            if target:
                cfg.edges.append(CFGEdge(name, target, EdgeType.JUMP))
        elif last.opcode == OpCode.BR_IF:
            targets = [t.strip() for t in (last.target or "").split(",") if t.strip()]
            if len(targets) >= 2:
                cfg.edges.append(CFGEdge(name, targets[0], EdgeType.BRANCH, "true"))
                cfg.edges.append(CFGEdge(name, targets[1], EdgeType.BRANCH, "false"))
            elif len(targets) == 1:
                cfg.edges.append(CFGEdge(name, targets[0], EdgeType.BRANCH, "true"))
                _add_fallthrough(cfg, name, block_names, idx)
            else:
                _add_fallthrough(cfg, name, block_names, idx)
        else:
            _add_fallthrough(cfg, name, block_names, idx)

    return cfg


# ============================================================================
# CFG 构建器 (整合 W7/W8/W9)
# ============================================================================

class CFGBuilder:
    """控制流图构建器 + 分析工具。"""

    # ---- 构建 ----

    def build(self, program) -> dict[str, CFG]:
        """直接从 IR Function 的 BasicBlock 构建 CFG（不拍平再切分）。

        parser 已经做好块划分时，直接信任其块结构；
        partition_basic_blocks_with_names 保留给纯指令列表的独立调用。
        """
        from scratchv.ir.types import Program
        cfgs: dict[str, CFG] = {}
        for func in program.functions:
            if not func.blocks:
                cfgs[func.name] = CFG(function_name=func.name)
                continue

            cfg = CFG(function_name=func.name,
                      entry=func.blocks[0].name)
            names = [b.name for b in func.blocks]

            # 创建节点
            for i, block in enumerate(func.blocks):
                node = CFGNode(
                    name=block.name,
                    instructions=block.instructions,
                    is_entry=(i == 0),
                )
                if block.instructions:
                    last = block.instructions[-1]
                    if last.opcode == OpCode.RETURN:
                        node.is_exit = True
                        node.terminator_opcode = "return"
                    elif last.opcode == OpCode.BR:
                        node.terminator_opcode = "br"
                    elif last.opcode == OpCode.BR_IF:
                        node.terminator_opcode = "br_if"
                cfg.nodes[block.name] = node

            # 添加边
            for i, block in enumerate(func.blocks):
                name = block.name
                instrs = block.instructions
                if not instrs:
                    _add_fallthrough(cfg, name, names, i)
                    continue

                last = instrs[-1]
                if last.opcode == OpCode.RETURN:
                    pass
                elif last.opcode == OpCode.BR:
                    t = last.target or ""
                    if t:
                        cfg.edges.append(CFGEdge(name, t, EdgeType.JUMP))
                elif last.opcode == OpCode.BR_IF:
                    targets = [x.strip() for x in (last.target or "").split(",") if x.strip()]
                    if len(targets) >= 2:
                        cfg.edges.append(CFGEdge(name, targets[0], EdgeType.BRANCH, "true"))
                        cfg.edges.append(CFGEdge(name, targets[1], EdgeType.BRANCH, "false"))
                    elif len(targets) == 1:
                        cfg.edges.append(CFGEdge(name, targets[0], EdgeType.BRANCH, "true"))
                        _add_fallthrough(cfg, name, names, i)
                    else:
                        _add_fallthrough(cfg, name, names, i)
                else:
                    _add_fallthrough(cfg, name, names, i)

            cfgs[func.name] = cfg
        return cfgs

    # ---- W7: 不可达消除 ----

    def eliminate_unreachable(self, cfg: CFG) -> list[str]:
        reachable = self._dfs(cfg, cfg.entry)
        removed = [n for n in list(cfg.nodes) if n not in reachable]
        for n in removed:
            del cfg.nodes[n]
        cfg.edges = [e for e in cfg.edges
                     if e.source in reachable and e.target in reachable]
        return removed

    @staticmethod
    def _dfs(cfg: CFG, start: str) -> set[str]:
        visited: set[str] = set()
        stack = [start]
        while stack:
            n = stack.pop()
            if n in visited or n not in cfg.nodes:
                continue
            visited.add(n)
            stack.extend(cfg.successors(n))
        return visited

    # ---- W8: 支配树 ----

    def compute_dominators(self, cfg: CFG) -> dict[str, set[str]]:
        all_nodes = set(cfg.nodes.keys())
        if not all_nodes:
            return {}
        dom = {n: all_nodes.copy() for n in all_nodes}
        dom[cfg.entry] = {cfg.entry}

        changed = True
        while changed:
            changed = False
            for node in all_nodes:
                if node == cfg.entry:
                    continue
                preds = cfg.predecessors(node)
                new_dom = {node}
                if preds:
                    new_dom |= set.intersection(*(dom[p] for p in preds if p in dom))
                if new_dom != dom[node]:
                    dom[node] = new_dom
                    changed = True
        return dom

    def compute_dominator_tree(self, cfg: CFG) -> dict[str, str | None]:
        dom_sets = self.compute_dominators(cfg)
        idom: dict[str, str | None] = {}
        for node in cfg.nodes:
            if node == cfg.entry:
                idom[node] = None
                continue
            strict = dom_sets.get(node, set()) - {node}
            for d in strict:
                if all(d not in (dom_sets.get(o, set()) - {o}) or o == d
                       for o in strict):
                    idom[node] = d
                    break
            else:
                idom[node] = cfg.entry
        return idom

    # ---- W9: 自然循环检测 ----

    def detect_loops(self, cfg: CFG) -> list[NaturalLoop]:
        dom = self.compute_dominators(cfg)
        back_edges = [(e.source, e.target) for e in cfg.edges
                      if e.target in dom.get(e.source, set())]

        loops: list[NaturalLoop] = []
        seen: dict[str, NaturalLoop] = {}

        for source, header in back_edges:
            if header in seen:
                seen[header].back_edges.append((source, header))
                seen[header].body |= self._loop_body(cfg, source, header)
                continue
            body = self._loop_body(cfg, source, header)
            loop = NaturalLoop(header=header, body=body,
                               back_edges=[(source, header)])
            loops.append(loop)
            seen[header] = loop
        return loops

    def _loop_body(self, cfg: CFG, source: str, header: str) -> set[str]:
        body: set[str] = set()
        queue = deque([source])
        while queue:
            n = queue.popleft()
            if n == header or n in body:
                continue
            body.add(n)
            for pred in cfg.predecessors(n):
                if pred not in body and pred != header:
                    queue.append(pred)
        return body | {header}

    def detect_nested_loops(self, cfg: CFG) -> list[NaturalLoop]:
        loops = self.detect_loops(cfg)
        for i, outer in enumerate(loops):
            for j, inner in enumerate(loops):
                if i == j:
                    continue
                if (inner.header in outer.body
                        and inner.body != outer.body
                        and inner.body.issubset(outer.body)):
                    inner.nesting_depth = max(
                        inner.nesting_depth, outer.nesting_depth + 1)
                    inner.parent = outer.header
                    outer.children.append(inner.header)
        return loops


