"""Unified CFG infrastructure tests (Issue #58 scope).

These tests exercise the new ``scratchv.analysis`` APIs without relying on the
legacy ``scratchv.ir.cfg`` shim.  Assertions are exact where possible.
"""

from scratchv.analysis.cfg import (
    CFG,
    CFGEdge,
    CFGNode,
    CFGBuilder,
    EdgeType,
    build_cfg,
    build_cfg_from_instructions,
    detect_loops,
)
from scratchv.analysis.adapters import IRCFGAdapter, MachineCFGAdapter
from scratchv.analysis.cfg_validation import verify_cfg
from scratchv.analysis.liveness import analyze_liveness
from scratchv.analysis.usedef import IRUseDefProvider, MachineUseDefProvider
from scratchv.analysis.dataflow import (
    ConstantPropagation,
    Direction,
    run_dataflow,
)
import pytest

from scratchv.backend import regalloc_linear, regalloc_linear_v1_5
from scratchv.backend.machine_types import (
    MachineInstr,
    MachineOp,
    MachineOperand,
)
from scratchv.backend.riscv_encoder import RISCVAEncoder
from scratchv.backend.inst_scheduler import (
    SchedInst,
    machine_instrs_from_scheduled,
)
from scratchv.frontend.dsl_parser import DSLParser
from scratchv.ir.types import DataType, Function, Instruction, OpCode, Value


def _inst(op: OpCode, dest=None, operands=None, target=None, attrs=None):
    return Instruction(
        opcode=op,
        dest=Value(name=dest, dtype=DataType.FLOAT32) if dest else None,
        operands=operands or [],
        target=target,
        attrs=attrs or {},
    )


def _val(name: str, constant=False, value=None) -> Value:
    return Value(
        name=name,
        dtype=DataType.FLOAT32,
        is_constant=constant,
        const_value=value,
    )


def test_ir_adapter_empty_function_has_entry():
    cfg = build_cfg(IRCFGAdapter(Function(name="f")))
    assert cfg.function_name == "f"
    assert cfg.entry == "entry"
    assert cfg.entry in cfg.nodes
    assert cfg.nodes[cfg.entry].instructions == []
    assert verify_cfg(cfg) == []


def test_machine_adapter_conditional_branch_topology():
    instrs = [
        MachineInstr(MachineOp.LABEL, target="main"),
        MachineInstr(MachineOp.LABEL, target=".entry"),
        MachineInstr(
            MachineOp.BNEZ,
            MachineOperand.vreg("cond"),
            target=".L_true",
        ),
        MachineInstr(MachineOp.J, target=".L_false"),
        MachineInstr(MachineOp.LABEL, target=".L_true"),
        MachineInstr(MachineOp.JALR, MachineOperand.reg("zero"),
                     MachineOperand.reg("ra"), comment="ret"),
        MachineInstr(MachineOp.LABEL, target=".L_false"),
        MachineInstr(MachineOp.JALR, MachineOperand.reg("zero"),
                     MachineOperand.reg("ra"), comment="ret"),
    ]
    cfg = build_cfg(MachineCFGAdapter("main", instrs))

    assert cfg.entry == ".entry"
    assert ".entry" in cfg.nodes
    assert ".L_true" in cfg.nodes
    assert ".L_false" in cfg.nodes

    branch_edges = [
        edge for edge in cfg.edges
        if edge.source == ".entry" and edge.edge_type is EdgeType.BRANCH
    ]
    assert len(branch_edges) == 1
    assert branch_edges[0].target == ".L_true"

    fallthrough = [
        edge for edge in cfg.edges
        if edge.source == ".entry" and edge.edge_type is EdgeType.FALLTHROUGH
    ]
    assert len(fallthrough) == 1

    assert verify_cfg(cfg) == []


def test_verify_cfg_detects_dangling_target():
    cfg = CFG("f")
    cfg.entry = "entry"
    cfg.nodes["entry"] = CFGNode("entry")
    cfg.edges = [CFGEdge("entry", "missing", EdgeType.JUMP)]

    diagnostics = verify_cfg(cfg)
    codes = {diagnostic.code for diagnostic in diagnostics}
    assert "CFG_DANGLING_TARGET" in codes


def test_verify_cfg_detects_jump_with_fallthrough():
    cfg = CFG("f")
    cfg.entry = "entry"
    cfg.nodes.update(
        {"entry": CFGNode("entry"), "a": CFGNode("a"), "b": CFGNode("b")}
    )
    cfg.edges = [
        CFGEdge("entry", "a", EdgeType.JUMP),
        CFGEdge("entry", "b", EdgeType.FALLTHROUGH),
    ]

    diagnostics = verify_cfg(cfg)
    codes = {diagnostic.code for diagnostic in diagnostics}
    assert "CFG_JUMP_WITH_FALLTHROUGH" in codes


def test_liveness_linear_use_def_kill():
    instrs = [
        _inst(OpCode.ADD, "c", [_val("a"), _val("b")]),
        _inst(OpCode.RETURN, operands=[_val("c")]),
    ]
    cfg = build_cfg_from_instructions(instrs)
    result = analyze_liveness(cfg, IRUseDefProvider())

    block = result.blocks["entry"]
    assert block.uses == frozenset({"a", "b"})
    assert block.defs == frozenset({"c"})
    assert block.live_in == frozenset({"a", "b"})
    assert block.live_out == frozenset()

    assert result.live_before["entry:1"] == frozenset({"c"})
    assert result.live_after["entry:1"] == frozenset()
    assert result.live_before["entry:0"] == frozenset({"a", "b"})
    assert result.live_after["entry:0"] == frozenset({"c"})


def test_constant_propagation_folds_constants():
    instrs = [
        _inst(OpCode.LOAD_CONST, "a", attrs={"value": 1}),
        _inst(OpCode.LOAD_CONST, "b", attrs={"value": 2}),
        _inst(OpCode.ADD, "c", [_val("a"), _val("b")]),
        _inst(OpCode.RETURN, operands=[_val("c")]),
    ]
    cfg = build_cfg_from_instructions(instrs)
    result = ConstantPropagation(cfg).run()

    out = result.out_values["entry"]
    assert out["a"].value == 1
    assert out["b"].value == 2
    assert out["c"].value == 3


def test_machine_use_def_provider_call_clobbers():
    call = MachineInstr(MachineOp.CALL, target="helper")
    provider = MachineUseDefProvider()
    assert provider.uses(call) == frozenset()
    assert provider.defs(call) == frozenset()
    assert "ra" in provider.clobbers(call)
    assert "t0" in provider.clobbers(call)
    assert "a0" in provider.clobbers(call)


def test_scheduler_control_target_roundtrip():
    scheduled = [
        SchedInst(
            id=0,
            opcode="beq",
            operands=["t0", "t1", ".L_then"],
            uses={"t0", "t1"},
            target=".L_then",
        )
    ]
    machine = machine_instrs_from_scheduled(scheduled)
    assert machine[0].op is MachineOp.BEQ
    assert machine[0].target == ".L_then"
    assert machine[0].src1 is not None
    assert machine[0].src1.value == "t0"
    assert machine[0].src2 is not None
    assert machine[0].src2.value == "t1"


def test_for_loop_is_normalized_and_detected():
    program = DSLParser().parse(
        "for i = 0, 4\n"
        "    c = add(a, b)\n"
        "endfor\n"
        "return c\n"
    )
    cfg = CFGBuilder().build(program)["main"]
    loops = detect_loops(cfg)
    assert len(loops) == 1
    assert loops[0].header.startswith("for_hdr")
    assert "for_body" in loops[0].body or any(
        name.startswith("for_body") for name in loops[0].body
    )


def test_machine_use_def_provider_uses_central_semantics():
    v = MachineOperand.vreg
    provider = MachineUseDefProvider()

    bnez = MachineInstr(MachineOp.BNEZ, v("cond"), target=".taken")
    assert provider.uses(bnez) == frozenset({"cond"})
    assert provider.defs(bnez) == frozenset()

    sw = MachineInstr(MachineOp.SW, v("value"), v("addr"))
    assert provider.uses(sw) == frozenset({"value", "addr"})

    call = MachineInstr(MachineOp.CALL, target="helper")
    assert provider.implicit_defs(call) == frozenset({"ra"})
    assert {"ra", "a0", "t0"} <= provider.clobbers(call)


def test_forward_solver_transfers_when_input_equals_initial():
    cfg = CFG("f")
    cfg.entry = "entry"
    cfg.nodes = {
        "entry": CFGNode("entry"),
        "b1": CFGNode(
            "b1",
            instructions=[_inst(OpCode.LOAD_CONST, "x", attrs={"value": 7})],
        ),
    }
    cfg.edges = [CFGEdge("entry", "b1", EdgeType.FALLTHROUGH)]

    result = ConstantPropagation(cfg).run()
    assert result.out_values["b1"]["x"].value == 7


def test_backward_solver_processes_cycle_without_exit_block():
    cfg = CFG("loop")
    cfg.entry = "A"
    cfg.nodes = {
        "A": CFGNode("A"),
        "B": CFGNode("B"),
        "C": CFGNode("C"),
    }
    cfg.edges = [
        CFGEdge("A", "B", EdgeType.FALLTHROUGH),
        CFGEdge("B", "C", EdgeType.FALLTHROUGH),
        CFGEdge("C", "A", EdgeType.JUMP),
    ]

    class BackwardMarker:
        direction = Direction.BACKWARD

        def initial(self):
            return frozenset()

        def boundary(self, block):
            return frozenset()

        def meet(self, values):
            result = set()
            for value in values:
                result |= value
            return frozenset(result)

        def transfer(self, block, value):
            return frozenset(set(value) | {block})

    result = run_dataflow(cfg, BackwardMarker())

    assert result.in_values["A"] == frozenset({"A", "B", "C"})
    assert result.in_values["B"] == frozenset({"A", "B", "C"})
    assert result.in_values["C"] == frozenset({"A", "B", "C"})


def test_machine_cross_block_liveness_diamond_join():
    v = MachineOperand.vreg
    imm = MachineOperand.immediate
    machine = [
        MachineInstr(MachineOp.LABEL, target="main"),
        MachineInstr(MachineOp.LABEL, target=".entry"),
        MachineInstr(MachineOp.BNEZ, v("cond"), target=".then"),
        MachineInstr(MachineOp.LI, v("y"), imm(1)),
        MachineInstr(MachineOp.J, target=".join"),
        MachineInstr(MachineOp.LABEL, target=".then"),
        MachineInstr(MachineOp.LI, v("y"), imm(2)),
        MachineInstr(MachineOp.LABEL, target=".join"),
        MachineInstr(MachineOp.ADDI, v("z"), v("y"), imm(0)),
    ]
    cfg = build_cfg(MachineCFGAdapter("main", machine))
    result = analyze_liveness(cfg, MachineUseDefProvider())

    assert result.blocks[".join"].live_in == frozenset({"y"})
    assert result.blocks[".then"].live_out == frozenset({"y"})
    assert result.blocks[".entry"].live_in == frozenset({"cond"})


def test_machine_liveness_loop_with_multiple_backedges():
    v = MachineOperand.vreg
    imm = MachineOperand.immediate
    cfg = CFG("f")
    cfg.entry = "entry"
    cfg.nodes = {
        "entry": CFGNode("entry"),
        "header": CFGNode("header"),
        "body_a": CFGNode(
            "body_a",
            instructions=[MachineInstr(MachineOp.ADD, v("a"), v("x"), v("y"))],
        ),
        "body_b": CFGNode(
            "body_b",
            instructions=[MachineInstr(MachineOp.ADD, v("b"), v("x"), v("y"))],
        ),
        "exit": CFGNode(
            "exit",
            instructions=[MachineInstr(MachineOp.ADD, v("z"), v("a"), v("b"))],
        ),
    }
    cfg.edges = [
        CFGEdge("entry", "header", EdgeType.FALLTHROUGH),
        CFGEdge("header", "body_a", EdgeType.BRANCH, "true"),
        CFGEdge("header", "body_b", EdgeType.BRANCH, "false"),
        CFGEdge("header", "exit", EdgeType.FALLTHROUGH),
        CFGEdge("body_a", "header", EdgeType.JUMP),
        CFGEdge("body_b", "header", EdgeType.JUMP),
    ]

    result = analyze_liveness(cfg, MachineUseDefProvider())
    assert {"a", "b", "x", "y"} <= result.blocks["header"].live_in
    assert "b" in result.blocks["body_a"].live_in
    assert "a" in result.blocks["body_b"].live_in


def test_machine_liveness_phi_edge_uses():
    class PhiProvider(MachineUseDefProvider):
        def phi_defs(self, block):
            return frozenset({"p"}) if block == "join" else frozenset()

        def edge_uses(self, pred, succ):
            if (pred, succ) == ("then", "join"):
                return frozenset({"p_then"})
            if (pred, succ) == ("else", "join"):
                return frozenset({"p_else"})
            return frozenset()

    cfg = CFG("f")
    cfg.entry = "entry"
    cfg.nodes = {
        "entry": CFGNode("entry"),
        "then": CFGNode("then"),
        "else": CFGNode("else"),
        "join": CFGNode("join"),
    }
    cfg.edges = [
        CFGEdge("entry", "then", EdgeType.BRANCH, "true"),
        CFGEdge("entry", "else", EdgeType.BRANCH, "false"),
        CFGEdge("then", "join", EdgeType.FALLTHROUGH),
        CFGEdge("else", "join", EdgeType.FALLTHROUGH),
    ]

    result = analyze_liveness(cfg, PhiProvider())
    assert result.edge_live[("then", "join")] == frozenset({"p_then"})
    assert result.edge_live[("else", "join")] == frozenset({"p_else"})


def test_machine_call_live_after_instruction():
    v = MachineOperand.vreg
    machine = [
        MachineInstr(MachineOp.CALL, target="helper"),
        MachineInstr(MachineOp.ADDI, v("z"), v("x"), MachineOperand.immediate(1)),
    ]
    cfg = build_cfg(MachineCFGAdapter("f", machine))
    result = analyze_liveness(cfg, MachineUseDefProvider())

    assert result.live_after["entry:0"] == frozenset({"x"})
    assert result.live_before["entry:0"] == frozenset({"x"})
    assert result.live_after["entry:1"] == frozenset()


@pytest.mark.parametrize("module", [regalloc_linear, regalloc_linear_v1_5])
def test_linear_scan_consumes_unified_cfg_liveness(module):
    v = MachineOperand.vreg
    imm = MachineOperand.immediate
    machine = [
        MachineInstr(MachineOp.LABEL, comment="main"),
        MachineInstr(MachineOp.LI, v("condition"), imm(1)),
        MachineInstr(MachineOp.LI, v("carried"), imm(7)),
        MachineInstr(MachineOp.BNEZ, v("condition"), comment=".then"),
        MachineInstr(MachineOp.ADDI, v("else_value"), v("carried"), imm(1)),
        MachineInstr(MachineOp.J, comment=".join"),
        MachineInstr(MachineOp.LABEL, comment=".then"),
        MachineInstr(MachineOp.ADDI, v("then_value"), v("carried"), imm(2)),
        MachineInstr(MachineOp.LABEL, comment=".join"),
        MachineInstr(MachineOp.MV, v("result"), v("carried")),
    ]
    block = module.block_from_machine_instrs(machine)
    allocator = module.LinearScanAllocator(["t0", "t1", "s0"])
    allocator.compute_live_intervals(block)

    assert ".then" in allocator.cfg.by_name
    assert ".join" in allocator.cfg.by_name
    assert "carried" in allocator.cfg.by_name[".then"].live_in
    assert "carried" in allocator.cfg.by_name[".join"].live_in


@pytest.mark.parametrize("module", [regalloc_linear, regalloc_linear_v1_5])
def test_unified_cfg_spill_reload_executes_in_simulator(module):
    pytest.importorskip("tinyfive")
    from scratchv.simulator.tinyfive import ProfiledMachine

    v = MachineOperand.vreg
    imm = MachineOperand.immediate
    machine = [
        MachineInstr(MachineOp.LABEL, comment="main"),
        MachineInstr(MachineOp.LI, v("left"), imm(7)),
        MachineInstr(MachineOp.LI, v("right"), imm(9)),
        MachineInstr(MachineOp.LI, v("condition"), imm(1)),
        MachineInstr(MachineOp.BNEZ, v("condition"), comment=".then"),
        MachineInstr(MachineOp.ADD, v("result"), v("left"), v("right")),
        MachineInstr(MachineOp.J, comment=".join"),
        MachineInstr(MachineOp.LABEL, comment=".then"),
        MachineInstr(MachineOp.SUB, v("result"), v("left"), v("right")),
        MachineInstr(MachineOp.LABEL, comment=".join"),
        MachineInstr(MachineOp.MV, MachineOperand.reg("a0"), v("result")),
        MachineInstr(MachineOp.JALR, MachineOperand.reg("zero"),
                     MachineOperand.reg("ra"), comment="ret"),
    ]
    block = module.block_from_machine_instrs(machine)
    allocator = module.LinearScanAllocator(["t0", "t1"])
    asm = allocator.emit(block)
    binary = RISCVAEncoder().assemble(
        "li sp, 4096\n" + asm + "\n.done:\nj .done"
    )
    words = [
        int.from_bytes(binary[offset:offset + 4], "little")
        for offset in range(0, len(binary), 4)
    ]
    profile = ProfiledMachine(mem_size=8192)
    profile.load_binary(words, origin=0)
    profile.run(instructions=len(words) + 2, start=0, strict=True)

    assert profile.get_reg(10) & 0xFFFFFFFF == 0xFFFFFFFE
