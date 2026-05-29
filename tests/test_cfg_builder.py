"""Tests for the CFG builder module."""

from scratchv.frontend.dsl_parser import DSLParser
from scratchv.frontend.dsl_extended import ExtendedDSLParser
from scratchv.analysis.cfg_builder import (
    CFGNode, CFGEdge, EdgeType, CFGBuilder,
    NaturalLoop, to_dot,
)
from scratchv.ir.types import Program  # noqa: F401


class TestCFGNode:
    """Tests for CFGNode dataclass."""

    def test_create_node(self):
        node = CFGNode(name="entry", instructions=3, is_entry=True)
        assert node.name == "entry"
        assert node.instructions == 3
        assert node.is_entry is True
        assert node.is_exit is False
        assert node.terminator_opcode is None

    def test_exit_node(self):
        node = CFGNode(name="exit", instructions=1, is_exit=True,
                       terminator_opcode="RETURN")
        assert node.is_exit is True
        assert node.terminator_opcode == "RETURN"


class TestCFGEdge:
    """Tests for CFGEdge dataclass."""

    def test_create_fallthrough_edge(self):
        edge = CFGEdge(source="A", target="B")
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.edge_type == EdgeType.FALLTHROUGH

    def test_create_branch_edge(self):
        edge = CFGEdge(source="A", target="B", edge_type=EdgeType.BRANCH,
                       condition="true")
        assert edge.edge_type == EdgeType.BRANCH
        assert edge.condition == "true"


class TestCFGBuilder:
    """Tests for the CFGBuilder class."""

    def test_build_simple_program(self):
        dsl = """
        c = add(a, b)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        builder = CFGBuilder(program)
        cfgs = builder.build()
        assert "main" in cfgs
        cfg = cfgs["main"]
        assert cfg.function_name == "main"
        assert "entry" in cfg.nodes

    def test_build_multiple_blocks(self):
        """For-loop creates single block with FOR/ENDFOR in base parser."""
        dsl = """
        for i = 0, 4
            c = add(a, b)
        endfor
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        builder = CFGBuilder(program)
        cfgs = builder.build()
        cfg = cfgs["main"]
        # FOR/ENDFOR are in the same block, so at least 1 node
        assert len(cfg.nodes) >= 1

    def test_empty_program(self):
        program = Program()
        builder = CFGBuilder(program)
        cfgs = builder.build()
        assert len(cfgs) == 0

    def test_successors(self):
        dsl = """
        c = add(a, b)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        builder = CFGBuilder(program)
        cfg = builder.build()["main"]
        succs = cfg.successors("entry")
        assert isinstance(succs, list)

    def test_predecessors(self):
        dsl = """
        c = add(a, b)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        builder = CFGBuilder(program)
        cfg = builder.build()["main"]
        preds = cfg.predecessors("entry")
        assert isinstance(preds, list)

    def test_reachable_nodes(self):
        dsl = """
        c = add(a, b)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        builder = CFGBuilder(program)
        cfg = builder.build()["main"]
        reachable = cfg.reachable_nodes
        assert "entry" in reachable

    def test_eliminate_unreachable(self):
        dsl = """
        c = add(a, b)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        builder = CFGBuilder(program)
        cfg = builder.build()["main"]
        unreachable = builder.eliminate_unreachable(cfg)
        # All blocks should be reachable in a simple program
        assert len(unreachable) == 0

    def test_dominators(self):
        dsl = """
        c = add(a, b)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        builder = CFGBuilder(program)
        cfg = builder.build()["main"]
        dom = builder.compute_dominators(cfg)
        assert "entry" in dom
        assert "entry" in dom["entry"]

    def test_immediate_dominator_tree(self):
        dsl = """
        c = add(a, b)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        builder = CFGBuilder(program)
        cfg = builder.build()["main"]
        idom = builder.compute_dominator_tree(cfg)
        assert "entry" in idom
        assert idom["entry"] is None  # entry has no immediate dominator

    def test_detect_loops_simple_no_loops(self):
        dsl = """
        c = add(a, b)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        builder = CFGBuilder(program)
        cfg = builder.build()["main"]
        loops = builder.detect_loops(cfg)
        assert len(loops) == 0

    def test_detect_loops_nested(self):
        """Nested loop detection should not crash."""
        dsl = """
        for i = 0, 3
            for j = 0, 2
                c = add(a, b)
            endfor
        endfor
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        builder = CFGBuilder(program)
        cfg = builder.build()["main"]
        loops = builder.detect_nested_loops(cfg)
        assert isinstance(loops, list)


class TestCFGToDot:
    """Tests for DOT generation."""

    def test_to_dot_basic(self):
        dsl = """
        c = add(a, b)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        builder = CFGBuilder(program)
        cfg = builder.build()["main"]
        dot = to_dot(cfg)
        assert "digraph" in dot
        assert 'CFG_main' in dot
        assert "entry" in dot

    def test_to_dot_edge_types(self):
        dsl = """
        c = add(a, b)
        return c
        """
        parser = DSLParser()
        program = parser.parse(dsl)
        builder = CFGBuilder(program)
        cfg = builder.build()["main"]
        dot = cfg.to_dot()
        assert "digraph" in dot
        assert "entry" in dot


class TestCFGWithIfElse:
    """Tests for CFG building with extended parser (if/else)."""

    def test_if_else_cfg(self):
        dsl = """
        if (a > b):
            c = add(a, b)
        else:
            c = mul(a, b)
        endif
        return c
        """
        parser = ExtendedDSLParser()
        program = parser.parse(dsl)
        builder = CFGBuilder(program)
        cfg = builder.build()["main"]
        # if/else should produce at least 4 blocks (entry + then + else +
        # merge)
        assert len(cfg.nodes) >= 4
        # Should have jump edges (BR instructions)
        jump_edges = [e for e in cfg.edges
                      if e.edge_type == EdgeType.JUMP]
        assert len(jump_edges) >= 2

    def test_while_cfg(self):
        dsl = """
        while (i < 10):
            acc = add(acc, x)
        endwhile
        return acc
        """
        parser = ExtendedDSLParser()
        program = parser.parse(dsl)
        builder = CFGBuilder(program)
        cfg = builder.build()["main"]
        assert len(cfg.nodes) >= 3
        # Should have jump edges
        jump_edges = [e for e in cfg.edges
                      if e.edge_type == EdgeType.JUMP]
        assert len(jump_edges) >= 1


class TestNaturalLoop:
    """Tests for the NaturalLoop dataclass."""

    def test_create_loop(self):
        loop = NaturalLoop(
            header="loop_hdr",
            body={"loop_hdr", "loop_body"},
            back_edges=[("loop_body", "loop_hdr")],
        )
        assert loop.header == "loop_hdr"
        assert "loop_body" in loop.body
        assert len(loop.back_edges) == 1
        assert loop.nesting_depth == 0

    def test_nested_loop_depth(self):
        inner = NaturalLoop(
            header="inner_hdr",
            nesting_depth=1,
            parent="outer_hdr",
        )
        assert inner.nesting_depth == 1
        assert inner.parent == "outer_hdr"
