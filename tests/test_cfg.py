"""控制流图 (CFG) 模块测试 — 课题11。

覆盖: 基本块划分、CFG构建、边类型、DOT输出、
不可达消除、支配树、循环检测、边界情况。
"""

import pytest
from scratchv.ir.types import Program, Function, BasicBlock, Instruction, OpCode, Value, DataType
from scratchv.ir.cfg import (
    CFGNode, CFGEdge, CFG, EdgeType, NaturalLoop,
    CFGBuilder, partition_basic_blocks_with_names, build_cfg_from_instructions,
)
from scratchv.frontend.dsl_parser import DSLParser
from scratchv.frontend.dsl_extended import ExtendedDSLParser


# ============================================================================
# 辅助
# ============================================================================

def _parse(src: str) -> Program:
    return DSLParser().parse(src)

def _parse_ext(src: str) -> Program:
    return ExtendedDSLParser().parse(src)

def _inst(op: OpCode, dest=None, operands=None, target=None) -> Instruction:
    return Instruction(opcode=op,
        dest=Value(name=dest, dtype=DataType.FLOAT32) if dest else None,
        operands=operands or [], target=target)


# ============================================================================
# W2: 基本块划分
# ============================================================================

class TestPartition:
    def test_single_block(self):
        """无控制流的程序 -> 1个块。"""
        instrs = [_inst(OpCode.ADD, "c"), _inst(OpCode.RETURN)]
        blocks = partition_basic_blocks_with_names(instrs)
        assert len(blocks) == 1
        assert blocks[0][0] == "entry"

    def test_br_splits(self):
        """BR 之后切新块。"""
        instrs = [
            _inst(OpCode.ADD, "a"),
            _inst(OpCode.BR, target="L1"),
            _inst(OpCode.LABEL, target="L1"),
            _inst(OpCode.RETURN),
        ]
        blocks = partition_basic_blocks_with_names(instrs)
        assert len(blocks) == 2
        assert blocks[0][0] == "entry"
        assert blocks[1][0] == "L1"

    def test_label_not_in_instructions(self):
        """LABEL 不进任何块的指令列表。"""
        instrs = [
            _inst(OpCode.BR, target="L1"),
            _inst(OpCode.LABEL, target="L1"),
            _inst(OpCode.RETURN),
        ]
        blocks = partition_basic_blocks_with_names(instrs)
        for _, instr_list in blocks:
            for i in instr_list:
                assert i.opcode != OpCode.LABEL

    def test_auto_naming(self):
        """BR 后无 LABEL -> 自动命名 b0, b1。"""
        instrs = [
            _inst(OpCode.BR, target="L1"),
            _inst(OpCode.RETURN),
        ]
        blocks = partition_basic_blocks_with_names(instrs)
        names = [n for n, _ in blocks]
        assert "b0" in names


# ============================================================================
# W3: 数据结构
# ============================================================================

class TestDataStructures:
    def test_edge_type_values(self):
        assert EdgeType.FALLTHROUGH.value == "fallthrough"
        assert EdgeType.BRANCH.value == "branch"
        assert EdgeType.JUMP.value == "jump"
        assert EdgeType.CALL.value == "call"

    def test_cfg_successors_predecessors(self):
        cfg = CFG("test")
        cfg.nodes["A"] = CFGNode("A")
        cfg.nodes["B"] = CFGNode("B")
        cfg.edges = [CFGEdge("A", "B")]
        assert cfg.successors("A") == ["B"]
        assert cfg.predecessors("B") == ["A"]


# ============================================================================
# W4: CFG 构建
# ============================================================================

class TestCFGBuild:
    def test_if_else_blocks(self):
        prog = _parse_ext("if (a > b):\n    c = add(a, b)\nelse:\n"
                          "    c = mul(a, b)\nendif\nreturn c\n")
        cfg = CFGBuilder().build(prog)["main"]
        assert len(cfg.nodes) >= 4

    def test_if_else_edge_types(self):
        prog = _parse_ext("if (a > b):\n    c = add(a, b)\nelse:\n"
                          "    c = mul(a, b)\nendif\nreturn c\n")
        cfg = CFGBuilder().build(prog)["main"]
        jumps = sum(1 for e in cfg.edges if e.edge_type == EdgeType.JUMP)
        branches = sum(1 for e in cfg.edges if e.edge_type == EdgeType.BRANCH)
        assert jumps >= 2
        assert branches >= 2

    def test_empty_program(self):
        cfg = build_cfg_from_instructions([])
        assert cfg.entry in cfg.nodes
        assert cfg.nodes[cfg.entry].instructions == []

    def test_linear_no_jump(self):
        instrs = [_inst(OpCode.ADD, "c"), _inst(OpCode.RETURN)]
        cfg = build_cfg_from_instructions(instrs)
        jumps = sum(1 for e in cfg.edges if e.edge_type == EdgeType.JUMP)
        assert jumps == 0


# ============================================================================
# W6: DOT 输出
# ============================================================================

class TestDOT:
    def test_contains_digraph(self):
        cfg = build_cfg_from_instructions([_inst(OpCode.RETURN)])
        dot = cfg.to_dot()
        assert dot.startswith("digraph")

    def test_node_colors(self):
        """2块CFG：entry(绿) + exit(红)。"""
        instrs = [
            _inst(OpCode.BR, target="L_end"),
            _inst(OpCode.LABEL, target="L_end"),
            _inst(OpCode.RETURN),
        ]
        cfg = build_cfg_from_instructions(instrs)
        dot = cfg.to_dot()
        assert "#90EE90" in dot  # 入口绿色
        assert "#FF6B6B" in dot  # 出口红色


# ============================================================================
# W7: 不可达消除
# ============================================================================

class TestUnreachable:
    def test_removes_isolated(self):
        cfg = CFG("test")
        cfg.entry = "A"
        cfg.nodes["A"] = CFGNode("A")
        cfg.nodes["B"] = CFGNode("B")
        cfg.nodes["C"] = CFGNode("C")  # 孤立
        cfg.edges = [CFGEdge("A", "B")]
        builder = CFGBuilder()
        removed = builder.eliminate_unreachable(cfg)
        assert "C" in removed
        assert "C" not in cfg.nodes

    def test_all_reachable(self):
        prog = _parse("c = add(a, b)\nreturn c\n")
        builder = CFGBuilder()
        cfg = builder.build(prog)["main"]
        removed = builder.eliminate_unreachable(cfg)
        assert len(removed) == 0


# ============================================================================
# W8: 支配树
# ============================================================================

class TestDominators:
    def test_entry_self(self):
        cfg = build_cfg_from_instructions([_inst(OpCode.RETURN)])
        builder = CFGBuilder()
        dom = builder.compute_dominators(cfg)
        assert cfg.entry in dom[cfg.entry]

    def test_idom_entry_none(self):
        cfg = build_cfg_from_instructions([_inst(OpCode.RETURN)])
        builder = CFGBuilder()
        idom = builder.compute_dominator_tree(cfg)
        assert idom[cfg.entry] is None


# ============================================================================
# W9: 循环检测
# ============================================================================

class TestLoops:
    def test_no_loops_linear(self):
        cfg = build_cfg_from_instructions([_inst(OpCode.RETURN)])
        builder = CFGBuilder()
        loops = builder.detect_loops(cfg)
        assert len(loops) == 0

    def test_while_loop(self):
        prog = _parse_ext("while (i < 10):\n    acc = add(acc, x)\nendwhile\n"
                          "return acc\n")
        builder = CFGBuilder()
        cfg = builder.build(prog)["main"]
        loops = builder.detect_loops(cfg)
        assert len(loops) >= 1

    def test_nested(self):
        """嵌套 while：外层 depth=0。"""
        prog = _parse_ext(
            "while (i < 3):\n"
            "    while (j < 2):\n"
            "        c = add(a, b)\n"
            "        j = add(j, 1)\n"
            "    endwhile\n"
            "    i = add(i, 1)\n"
            "endwhile\n"
            "return c\n")
        builder = CFGBuilder()
        cfg = builder.build(prog)["main"]
        loops = builder.detect_nested_loops(cfg)
        depths = [l.nesting_depth for l in loops]
        assert 0 in depths


# ============================================================================
# 边界情况
# ============================================================================

class TestEdgeCases:
    def test_empty_function(self):
        func = Function(name="f")
        prog = Program()
        prog.add_function(func)
        cfgs = CFGBuilder().build(prog)
        assert "f" in cfgs

    def test_multi_function(self):
        f1, f2 = Function("f1"), Function("f2")
        f1.add_block(BasicBlock("entry"))
        f2.add_block(BasicBlock("entry"))
        prog = Program()
        prog.add_function(f1); prog.add_function(f2)
        cfgs = CFGBuilder().build(prog)
        assert "f1" in cfgs and "f2" in cfgs


