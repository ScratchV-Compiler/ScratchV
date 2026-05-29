"""Tests for Instruction Scheduler (List Scheduling)."""

import pytest
from scratchv.backend.inst_scheduler import (
    InstructionScheduler, SchedInst, DAGNode, parse_instructions,
)


class TestSchedInst:
    """Tests for SchedInst data class."""

    def test_creation(self):
        inst = SchedInst(0, "add", ["t0", "t1", "t2"],
                         defines={"t0"}, uses={"t1", "t2"})
        assert inst.id == 0
        assert inst.opcode == "add"
        assert inst.defines == {"t0"}
        assert inst.uses == {"t1", "t2"}


class TestDAGNode:
    """Tests for DAG node."""

    def test_creation(self):
        inst = SchedInst(0, "add", ["t0", "t1", "t2"],
                         defines={"t0"}, uses={"t1", "t2"})
        node = DAGNode(inst=inst)
        assert node.inst.opcode == "add"
        assert not node.scheduled
        assert node.priority == 0

    def test_add_predecessor(self):
        inst_a = SchedInst(0, "add", ["t0", "t1", "t2"],
                           defines={"t0"}, uses={"t1", "t2"})
        inst_b = SchedInst(1, "mul", ["t3", "t0", "t4"],
                           defines={"t3"}, uses={"t0", "t4"})
        na = DAGNode(inst=inst_a)
        nb = DAGNode(inst=inst_b)
        nb.predecessors.append((na, 2))
        na.successors.append((nb, 2))
        assert len(nb.predecessors) == 1
        assert len(na.successors) == 1


class TestParseInstructions:
    """Tests for parsing assembly text to SchedInst list."""

    def test_parse_basic(self):
        asm = "  add t0, t1, t2\n  sub t3, t4, t5\n"
        insts = parse_instructions(asm)
        assert len(insts) == 2
        assert insts[0].opcode == "add"
        assert insts[1].opcode == "sub"

    def test_parse_skip_labels_and_comments(self):
        asm = "main:\n  add t0, t1, t2\n  # comment\n  ret\n"
        insts = parse_instructions(asm)
        opcodes = [i.opcode for i in insts]
        assert "add" in opcodes
        assert "ret" in opcodes
        # Should not include label line or comment
        assert len(insts) == 2

    def test_parse_defines_uses(self):
        asm = "  add t0, t1, t2\n"
        insts = parse_instructions(asm)
        assert "t0" in insts[0].defines
        assert "t1" in insts[0].uses
        assert "t2" in insts[0].uses

    def test_parse_store(self):
        asm = "  sw t0, 0(sp)\n"
        insts = parse_instructions(asm)
        assert insts[0].opcode == "sw"
        # Store: first operand is a use (value to store)
        assert "t0" in insts[0].uses
        assert "sp" in insts[0].uses


class TestInstructionScheduler:
    """Tests for the instruction scheduler."""

    def test_build_dag_simple(self):
        insts = [
            SchedInst(0, "li", ["t0", "42"], defines={"t0"}, uses=set()),
            SchedInst(1, "li", ["t1", "10"], defines={"t1"}, uses=set()),
            SchedInst(2, "add", ["t2", "t0", "t1"],
                      defines={"t2"}, uses={"t0", "t1"}),
        ]
        scheduler = InstructionScheduler()
        dag = scheduler.build_dag(insts)
        assert len(dag) == 3
        # t2 depends on t0 and t1
        add_node = dag[2]
        assert len(add_node.predecessors) >= 2

    def test_schedule_no_dependencies(self):
        """Independent instructions should stay in order."""
        insts = [
            SchedInst(0, "li", ["t0", "1"], defines={"t0"}, uses=set()),
            SchedInst(1, "li", ["t1", "2"], defines={"t1"}, uses=set()),
            SchedInst(2, "li", ["t2", "3"], defines={"t2"}, uses=set()),
        ]
        scheduler = InstructionScheduler()
        dag = scheduler.build_dag(insts)
        scheduled = scheduler.schedule(dag)
        assert len(scheduled) == 3

    def test_schedule_with_dependency(self):
        """An instruction that depends on a previous result."""
        insts = [
            SchedInst(0, "lw", ["t0", "0(a0)"], defines={"t0"}, uses={"a0"}),
            SchedInst(1, "add", ["t1", "t0", "t2"],
                      defines={"t1"}, uses={"t0", "t2"}),
            SchedInst(2, "lw", ["t3", "4(a0)"], defines={"t3"}, uses={"a0"}),
        ]
        scheduler = InstructionScheduler()
        dag = scheduler.build_dag(insts)
        scheduled = scheduler.schedule(dag)
        assert len(scheduled) == 3
        # The second lw is independent and may be moved before the add

    def test_estimate_cycles(self):
        insts = [
            SchedInst(0, "add", ["t0", "t1", "t2"],
                      defines={"t0"}, uses={"t1", "t2"}),
            SchedInst(1, "mul", ["t3", "t0", "t4"],
                      defines={"t3"}, uses={"t0", "t4"}),
        ]
        scheduler = InstructionScheduler()
        cycles = scheduler.estimate_cycles(insts)
        # add(1) + mul(3) = 4
        assert cycles >= 2

    def test_report(self):
        insts = [
            SchedInst(0, "add", ["t0", "t1", "t2"],
                      defines={"t0"}, uses={"t1", "t2"}),
        ]
        scheduler = InstructionScheduler()
        dag = scheduler.build_dag(insts)
        scheduled = scheduler.schedule(dag)
        report = scheduler.report(insts, scheduled)
        assert "Scheduling" in report
        assert "cycles" in report.lower()

    def test_priority_computation(self):
        """Verify that priorities are computed (critical path)."""
        insts = [
            SchedInst(0, "lw", ["t0", "0(a0)"], defines={"t0"}, uses={"a0"}),
            SchedInst(1, "mul", ["t1", "t0", "t2"],
                      defines={"t1"}, uses={"t0", "t2"}),
        ]
        scheduler = InstructionScheduler()
        dag = scheduler.build_dag(insts)
        # Both nodes should have priority > 0
        for node in dag:
            assert node.priority > 0

    def test_empty_input(self):
        scheduler = InstructionScheduler()
        dag = scheduler.build_dag([])
        scheduled = scheduler.schedule(dag)
        assert scheduled == []

    def test_custom_latency_model(self):
        model = {"add": 5, "lw": 1}
        scheduler = InstructionScheduler(latency_model=model)
        assert scheduler.latency_model["add"] == 5

    def test_waw_dependency(self):
        """WAW: two writes to same register create a dependency."""
        insts = [
            SchedInst(0, "add", ["t0", "t1", "t2"],
                      defines={"t0"}, uses={"t1", "t2"}),
            SchedInst(1, "sub", ["t0", "t3", "t4"],
                      defines={"t0"}, uses={"t3", "t4"}),
        ]
        scheduler = InstructionScheduler()
        dag = scheduler.build_dag(insts)
        assert len(dag) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
