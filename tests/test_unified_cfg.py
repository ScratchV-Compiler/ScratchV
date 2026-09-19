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
from scratchv.analysis.dataflow import ConstantPropagation
from scratchv.backend.machine_types import (
    MachineInstr,
    MachineOp,
    MachineOperand,
)
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
