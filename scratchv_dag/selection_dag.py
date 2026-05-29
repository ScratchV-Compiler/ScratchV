"""
SelectionDAG builder, combiner, and scheduler.

Translates a ScratchV IR Program into a SelectionDAG (DAGBuilder),
performs DAG-level peephole optimisations (DAGCombiner), then
linearises the DAG into a schedule of MachineInstrs (DAGScheduler).

The flow::

    Program  ──▶  DAGBuilder  ──▶  SelectionDAG  ──▶  DAGCombiner
                                                          │
                                                          ▼
    MachineInstr list  ◀───  DAGScheduler  ◀────────  clean DAG
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from scratchv_dag.sdnode import (
    MVT,
    SDNodeOpcode,
    SDValue,
    SelectionDAG,
)

# We re-use the existing backend's MachineInstr types for scheduling
# output so the DAG scheduler integrates directly into the ScratchV
# backend pipeline.
from scratchv.backend.register_alloc import (
    MachineInstr, MachineOp, MachineOperand,
)

# Re-export for convenience.
__all__ = [
    "DAGBuilder",
    "DAGCombiner",
    "DAGScheduler",
]


# ═══════════════════════════════════════════════════════════════════════════════
# IR → MVT mapping helper
# ═══════════════════════════════════════════════════════════════════════════════

def _ir_to_mvt(dtype: Any) -> MVT:
    """Map a ScratchV IR ``DataType`` to the corresponding ``MVT``."""
    from scratchv.ir.types import DataType
    return {
        DataType.FLOAT32: MVT.f32,
        DataType.FLOAT64: MVT.f64,
        DataType.INT32:   MVT.i32,
        DataType.INT64:   MVT.i64,
    }.get(dtype, MVT.i32)


# ═══════════════════════════════════════════════════════════════════════════════
# DAGBuilder — IR → SelectionDAG
# ═══════════════════════════════════════════════════════════════════════════════

class DAGBuilder:
    """Lower a ScratchV IR ``Program`` into a ``SelectionDAG``.

    Each IR instruction is visited by a dedicated handler that builds
    the corresponding DAG sub-graph.  Value names are tracked in a
    symbol table mapping them to their ``SDValue`` producer.

    Usage::

        builder = DAGBuilder(program)
        dag = builder.run()
    """

    def __init__(self, program: Any) -> None:
        # The IR program to lower.
        self.program = program
        # The DAG being built.
        self.dag = SelectionDAG()
        # IR value name → SDValue symbol table.
        self._value_map: Dict[str, SDValue] = {}
        # Current chain token (threaded through side-effecting ops).
        self._chain: SDValue = self.dag.entry_token
        # Loop context for ``for``/``endfor``.
        self._loop_ctx: Optional[Dict[str, Any]] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> SelectionDAG:
        """Build the DAG for all functions in the program."""
        for func in self.program.functions:
            self._build_function(func)
        return self.dag

    # ── Per-function lowering ──────────────────────────────────────────────

    def _build_function(self, func: Any) -> None:
        self._value_map.clear()
        self._chain = self.dag.entry_token

        # Map each function parameter to a CopyFromReg.
        for i, param in enumerate(func.params):
            reg_name = f"a{i}" if i < 8 else f"s{i - 8}"
            reg = self.dag.get_register(reg_name)
            val = self.dag.get_copy_from_reg(reg)
            self._chain = val.node.get_chain() or self._chain
            self._value_map[param.name] = val

        for block in func.blocks:
            for instr in block.instructions:
                self._build_instruction(instr)

    def _build_instruction(self, instr: Any) -> None:
        """Dispatch an IR instruction to its dedicated builder."""
        handler = getattr(self, f"_build_{instr.opcode.value}", None)
        if handler is None:
            raise ValueError(
                f"No DAG builder for opcode: {instr.opcode.value}"
            )
        handler(instr)

    # ── Operand resolution ─────────────────────────────────────────────────

    def _get_val(self, ir_val: Any) -> SDValue:
        """Resolve an IR operand to an SDValue.

        Constants are created on the fly; named values are looked up
        in the symbol table (falling back to Undef).
        """
        if ir_val.is_constant and ir_val.const_value is not None:
            vt = _ir_to_mvt(ir_val.dtype)
            if vt.is_float:
                return self.dag.get_constant_fp(
                    float(ir_val.const_value), vt
                )
            return self.dag.get_constant(
                int(ir_val.const_value), vt
            )
        name = ir_val.name
        if name not in self._value_map:
            # Safeguard: lazily create an Undef for forward references.
            self._value_map[name] = self.dag.get_undef(
                _ir_to_mvt(ir_val.dtype)
            )
        return self._value_map[name]

    def _set_val(self, ir_val: Any, sdval: SDValue) -> None:
        """Record an IR→SDValue binding."""
        self._value_map[ir_val.name] = sdval

    # ── Arithmetic ─────────────────────────────────────────────────────────

    def _build_add(self, instr: Any) -> None:
        lhs = self._get_val(instr.operands[0])
        rhs = self._get_val(instr.operands[1])
        if lhs.value_type.is_float:
            val = self.dag.get_fadd(lhs, rhs)
        else:
            val = self.dag.get_add(lhs, rhs)
        self._set_val(instr.dest, val)

    def _build_sub(self, instr: Any) -> None:
        lhs = self._get_val(instr.operands[0])
        rhs = self._get_val(instr.operands[1])
        if lhs.value_type.is_float:
            val = self.dag.get_fsub(lhs, rhs)
        else:
            val = self.dag.get_sub(lhs, rhs)
        self._set_val(instr.dest, val)

    def _build_mul(self, instr: Any) -> None:
        lhs = self._get_val(instr.operands[0])
        rhs = self._get_val(instr.operands[1])
        if lhs.value_type.is_float:
            val = self.dag.get_fmul(lhs, rhs)
        else:
            val = self.dag.get_mul(lhs, rhs)
        self._set_val(instr.dest, val)

    def _build_div(self, instr: Any) -> None:
        lhs = self._get_val(instr.operands[0])
        rhs = self._get_val(instr.operands[1])
        if lhs.value_type.is_float:
            val = self.dag.get_fdiv(lhs, rhs)
        else:
            val = self.dag.get_div(lhs, rhs)
        self._set_val(instr.dest, val)

    def _build_neg(self, instr: Any) -> None:
        src = self._get_val(instr.operands[0])
        if src.value_type.is_float:
            zero = self.dag.get_constant_fp(0.0, src.value_type)
            val = self.dag.get_fsub(zero, src)
        else:
            zero = self.dag.get_constant(0, src.value_type)
            val = self.dag.get_sub(zero, src)
        self._set_val(instr.dest, val)

    def _build_exp(self, instr: Any) -> None:
        src = self._get_val(instr.operands[0])
        callee = "expf" if src.value_type == MVT.f32 else "exp"
        val = self.dag.get_call(callee, [src], vt=src.value_type)
        self._chain = val.node.get_chain() or self._chain
        self._set_val(instr.dest, val)

    def _build_load_const(self, instr: Any) -> None:
        v = instr.attrs.get("value", 0)
        vt = _ir_to_mvt(instr.dest.dtype) if instr.dest else MVT.f32
        if vt.is_float:
            val = self.dag.get_constant_fp(float(v), vt)
        else:
            val = self.dag.get_constant(int(v), vt)
        self._set_val(instr.dest, val)

    # ── Memory ─────────────────────────────────────────────────────────────

    def _build_load(self, instr: Any) -> None:
        addr = self._get_val(instr.operands[0])
        vt = _ir_to_mvt(instr.dest.dtype) if instr.dest else MVT.i32
        val = self.dag.get_load(addr, vt, chain=self._chain)
        self._chain = val.node.get_chain() or self._chain
        self._set_val(instr.dest, val)

    def _build_store(self, instr: Any) -> None:
        addr = self._get_val(instr.operands[0])
        val = self._get_val(instr.operands[1])
        self._chain = self.dag.get_store(addr, val, chain=self._chain)

    def _build_alloca(self, instr: Any) -> None:
        size = instr.attrs.get("size", 4)
        vt = _ir_to_mvt(instr.dest.dtype) if instr.dest else MVT.i32
        val = self.dag.get_constant(size, vt)
        self._set_val(instr.dest, val)

    # ── Control flow ───────────────────────────────────────────────────────

    def _build_for(self, instr: Any) -> None:
        start = instr.attrs.get("start", 0)
        val = self.dag.get_constant(start, MVT.i32)
        self._value_map[instr.dest.name] = val
        self._loop_ctx = {
            "iv_name": instr.dest.name,
            "end": instr.attrs.get("end", 0),
        }

    def _build_endfor(self, instr: Any) -> None:
        if self._loop_ctx is None:
            return
        iv_name = self._loop_ctx["iv_name"]
        iv = self._value_map.get(iv_name)
        if iv is not None:
            inc = self.dag.get_add(iv, self.dag.get_constant(1, MVT.i32))
            self._value_map[iv_name] = inc
        self._loop_ctx = None

    def _build_br(self, instr: Any) -> None:
        self._chain = self.dag.get_br(instr.target or "", chain=self._chain)

    def _build_br_if(self, instr: Any) -> None:
        cond = self._get_val(instr.operands[0])
        targets = (instr.target or "").split(",")
        true_t = targets[0].strip() if targets else ""
        false_t = targets[1].strip() if len(targets) > 1 else ""
        self._chain = self.dag.get_br_cc(
            cond, true_t, false_t, chain=self._chain)

    def _build_return(self, instr: Any) -> None:
        vals = [self._get_val(instr.operands[0])] if instr.operands else None
        self._chain = self.dag.get_ret(vals, chain=self._chain)

    def _build_label(self, instr: Any) -> None:
        pass  # Labels are implicit in the DAG structure.

    # ── Neural-network ops ─────────────────────────────────────────────────

    def _build_relu(self, instr: Any) -> None:
        src = self._get_val(instr.operands[0])
        val = self.dag.get_call("relu", [src], vt=src.value_type)
        self._chain = val.node.get_chain() or self._chain
        self._set_val(instr.dest, val)

    def _build_gelu(self, instr: Any) -> None:
        src = self._get_val(instr.operands[0])
        val = self.dag.get_call("gelu", [src], vt=src.value_type)
        self._chain = val.node.get_chain() or self._chain
        self._set_val(instr.dest, val)

    def _build_softmax(self, instr: Any) -> None:
        src = self._get_val(instr.operands[0])
        val = self.dag.get_call("softmax", [src], vt=src.value_type)
        self._chain = val.node.get_chain() or self._chain
        self._set_val(instr.dest, val)

    def _build_matmul(self, instr: Any) -> None:
        a = self._get_val(instr.operands[0])
        b = self._get_val(instr.operands[1])
        m = instr.attrs.get("m", 1)
        n = instr.attrs.get("n", 1)
        k = instr.attrs.get("k", 1)
        vt = _ir_to_mvt(instr.dest.dtype) if instr.dest else MVT.f32
        val = self.dag.get_call(f"matmul_m{m}_n{n}_k{k}", [a, b], vt=vt)
        self._chain = val.node.get_chain() or self._chain
        self._set_val(instr.dest, val)

    def _build_dot(self, instr: Any) -> None:
        a = self._get_val(instr.operands[0])
        b = self._get_val(instr.operands[1])
        length = instr.attrs.get("length", 1)
        vt = _ir_to_mvt(instr.dest.dtype) if instr.dest else MVT.f32
        val = self.dag.get_call(f"dot_len{length}", [a, b], vt=vt)
        self._chain = val.node.get_chain() or self._chain
        self._set_val(instr.dest, val)


# ═══════════════════════════════════════════════════════════════════════════════
# DAGCombiner — DAG-level peephole optimisations
# ═══════════════════════════════════════════════════════════════════════════════

class DAGCombiner:
    """DAG-level peephole optimisations.

    Currently implements constant folding for integer and floating-point
    arithmetic.  Runs iteratively until no further folds are possible
    or a fixed iteration limit is reached.

    Usage::

        combiner = DAGCombiner(dag)
        n_folds  = combiner.run()
    """

    def __init__(self, dag: SelectionDAG) -> None:
        self.dag = dag
        self._changed = False

    def run(self) -> int:
        """Apply all DAG combines.  Returns the number of folds applied."""
        n_folds = 0
        for _ in range(32):  # safety limit
            self._changed = False
            # Iterate in reverse so we fold bottom-up.
            for node in reversed(self.dag._nodes):
                handler = getattr(self, f"_fold_{node.opcode.value}", None)
                if handler is not None:
                    handler(node)
                if self._changed:
                    n_folds += 1
            if not self._changed:
                break
        return n_folds

    # ── Folding helpers ────────────────────────────────────────────────────

    def _fold_ADD(self, node: Any) -> None:
        lhs, rhs = self._get_const_int_binop(node)
        if lhs is not None and rhs is not None:
            self._replace_with_constant(node, lhs + rhs)

    def _fold_SUB(self, node: Any) -> None:
        lhs, rhs = self._get_const_int_binop(node)
        if lhs is not None and rhs is not None:
            self._replace_with_constant(node, lhs - rhs)

    def _fold_MUL(self, node: Any) -> None:
        lhs, rhs = self._get_const_int_binop(node)
        if lhs is not None and rhs is not None:
            self._replace_with_constant(node, lhs * rhs)

    def _fold_DIV(self, node: Any) -> None:
        lhs, rhs = self._get_const_int_binop(node)
        if lhs is not None and rhs is not None and rhs != 0:
            self._replace_with_constant(node, lhs // rhs)

    def _fold_FADD(self, node: Any) -> None:
        self._fold_fp_binop(node, lambda a, b: a + b)

    def _fold_FSUB(self, node: Any) -> None:
        self._fold_fp_binop(node, lambda a, b: a - b)

    def _fold_FMUL(self, node: Any) -> None:
        self._fold_fp_binop(node, lambda a, b: a * b)

    def _fold_FDIV(self, node: Any) -> None:
        self._fold_fp_binop(node, lambda a, b: a / b)

    # ── Internal ───────────────────────────────────────────────────────────

    def _get_const_int_binop(
        self, node: Any
    ) -> Tuple[Optional[int], Optional[int]]:
        """Return (lhs, rhs) if both operands are integer Constants."""
        if len(node.operands) < 2:
            return None, None
        lhs = node.operands[0].node.get_constant_int()
        rhs = node.operands[1].node.get_constant_int()
        return lhs, rhs

    def _fold_fp_binop(self, node: Any, op: Any) -> None:
        """Constant-fold an FP binary op if both operands are ConstantFP."""
        lhs = node.operands[0].node.get_constant_fp()
        rhs = node.operands[1].node.get_constant_fp()
        if lhs is not None and rhs is not None:
            try:
                result = op(lhs, rhs)
                self._replace_with_fp_constant(node, result)
            except (ZeroDivisionError, OverflowError, ValueError):
                pass

    def _replace_with_constant(self, old_node: Any, val: int) -> None:
        """Replace *old_node* with a new Constant node."""
        new_val = self.dag.get_constant(val, old_node.value_type())
        old_node._attributes["replaced_by"] = new_val
        self._changed = True

    def _replace_with_fp_constant(self, old_node: Any, val: float) -> None:
        new_val = self.dag.get_constant_fp(val, old_node.value_type())
        old_node._attributes["replaced_by"] = new_val
        self._changed = True


# ═══════════════════════════════════════════════════════════════════════════════
# DAGScheduler — DAG → linear MachineInstr list
# ═══════════════════════════════════════════════════════════════════════════════

class DAGScheduler:
    """Schedule a ``SelectionDAG`` into a linear list of ``MachineInstr``\\s.

    Uses a post-order traversal (operands before consumers) to produce
    a valid topological schedule.  Each SDNode is mapped to one or more
    ``MachineInstr``\\s that the existing ScratchV backend can consume.

    Usage::

        scheduler = DAGScheduler(dag)
        instrs = scheduler.run()
    """

    def __init__(self, dag: SelectionDAG) -> None:
        self.dag = dag

    def run(self) -> List[MachineInstr]:
        """Produce a linearised instruction list from the DAG."""
        scheduled: Set[int] = set()
        result: List[MachineInstr] = []

        def _schedule(node: Any) -> None:
            if node.node_id in scheduled:
                return
            # Recurse into operands first (post-order).
            for op in node.operands:
                if op.node.node_id not in scheduled:
                    _schedule(op.node)
            scheduled.add(node.node_id)
            self._emit_node(node, result)

        for node in self.dag._nodes:
            _schedule(node)

        return result

    # ── Node emission ──────────────────────────────────────────────────────

    def _emit_node(self, node: Any, result: List[MachineInstr]) -> None:
        """Emit a single SDNode as 0+ MachineInstrs."""
        opcode = node.opcode
        machine_op = _SDNODE_TO_MACHINE_OP.get(opcode)
        if machine_op is None:
            return  # skip nodes without a direct lowering

        # Constants
        if opcode == SDNodeOpcode.Constant:
            val = node.get_constant_int() or 0
            dst = MachineOperand.vreg(f"t{node.node_id}")
            result.append(MachineInstr(
                MachineOp.LI, dst,
                MachineOperand.immediate(val),
                comment=f"const {val}",
            ))
            return

        if opcode == SDNodeOpcode.ConstantFP:
            val = node.get_constant_fp() or 0.0
            dst = MachineOperand.vreg(f"t{node.node_id}")
            result.append(MachineInstr(
                MachineOp.LI, dst,
                MachineOperand.immediate(int(val)),
                comment=f"constfp {val}",
            ))
            return

        if opcode == SDNodeOpcode.CopyFromReg:
            reg = node.get_attr("reg_name", "zero")
            dst = MachineOperand.vreg(f"t{node.node_id}")
            result.append(MachineInstr(
                MachineOp.MV, dst,
                MachineOperand.reg(reg),
                comment="copy_from_reg",
            ))
            return

        # Memory
        if opcode == SDNodeOpcode.LOAD:
            dst = MachineOperand.vreg(f"t{node.node_id}")
            addr = _op_to_operand(node.operands[1])
            result.append(MachineInstr(
                MachineOp.LW, dst, addr, comment="load"))
            return

        if opcode == SDNodeOpcode.STORE:
            addr = _op_to_operand(node.operands[1])
            val = _op_to_operand(node.operands[2])
            result.append(MachineInstr(
                MachineOp.SW, addr, val, comment="store"))
            return

        # Control
        if opcode == SDNodeOpcode.BR:
            target = node.get_attr("branch_target", "")
            result.append(MachineInstr(MachineOp.J, comment=target))
            return

        if opcode == SDNodeOpcode.BR_CC:
            cond = _op_to_operand(node.operands[1])
            true_t = node.get_attr("true_target", "")
            false_t = node.get_attr("false_target", "")
            result.append(MachineInstr(
                MachineOp.BNEZ, cond, comment=true_t))
            result.append(MachineInstr(
                MachineOp.J, comment=false_t))
            return

        if opcode == SDNodeOpcode.RET:
            result.append(MachineInstr(
                MachineOp.JALR, MachineOperand.vreg("zero"),
                MachineOperand.vreg("ra"),
                comment="ret",
            ))
            return

        if opcode == SDNodeOpcode.CALL:
            callee = node.get_attr("callee", "unknown")
            result.append(MachineInstr(MachineOp.CALL, comment=callee))
            if node.num_values > 0:
                dst = MachineOperand.vreg(f"t{node.node_id}")
                result.append(MachineInstr(
                    MachineOp.MV, dst, MachineOperand.vreg("a0"),
                ))
            return

        # Generic binary operation
        dst: MachineOperand | None = None
        src1: MachineOperand | None = None
        src2: MachineOperand | None = None
        if node.num_values > 0 and node._num_types > node.num_chain_results:
            dst = MachineOperand.vreg(f"t{node.node_id}")
        if len(node.operands) >= 2:
            src1 = _op_to_operand(node.operands[0])
            src2 = _op_to_operand(node.operands[1])
        result.append(MachineInstr(machine_op, dst, src1, src2))


# ── Helper ────────────────────────────────────────────────────────────

def _op_to_operand(sdval: SDValue) -> MachineOperand:
    """Convert an SDValue to a MachineOperand (vreg, imm, or phys reg)."""
    opc = sdval.node.opcode
    if opc == SDNodeOpcode.Constant:
        return MachineOperand.immediate(sdval.node.get_constant_int() or 0)
    if opc == SDNodeOpcode.ConstantFP:
        val = int(sdval.node.get_constant_fp() or 0.0)
        return MachineOperand.immediate(val)
    if opc == SDNodeOpcode.Register:
        return MachineOperand.reg(sdval.node.get_attr("reg_name", "zero"))
    return MachineOperand.vreg(f"t{sdval.node.node_id}")


# ── SDNode -> MachineOp lookup table ──────────────────────────────────

_SDNODE_TO_MACHINE_OP: Dict[SDNodeOpcode, MachineOp] = {
    SDNodeOpcode.ADD: MachineOp.ADD,
    SDNodeOpcode.SUB: MachineOp.SUB,
    SDNodeOpcode.MUL: MachineOp.MUL,
    SDNodeOpcode.DIV: MachineOp.DIV,
    SDNodeOpcode.FADD: MachineOp.ADD,
    SDNodeOpcode.FSUB: MachineOp.SUB,
    SDNodeOpcode.FMUL: MachineOp.MUL,
    SDNodeOpcode.FDIV: MachineOp.DIV,
    SDNodeOpcode.NEG: MachineOp.SUB,
    SDNodeOpcode.SETCC: MachineOp.SUB,
    SDNodeOpcode.LOAD: MachineOp.LW,
    SDNodeOpcode.STORE: MachineOp.SW,
    SDNodeOpcode.BR: MachineOp.J,
    SDNodeOpcode.BR_CC: MachineOp.BNEZ,
    SDNodeOpcode.RET: MachineOp.JALR,
    SDNodeOpcode.CALL: MachineOp.CALL,
    SDNodeOpcode.LI_Pseudo: MachineOp.LI,
    SDNodeOpcode.MV_Pseudo: MachineOp.MV,
    SDNodeOpcode.RELU:
        MachineOp.MAX,
}
