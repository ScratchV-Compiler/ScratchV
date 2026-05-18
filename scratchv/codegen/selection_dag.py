"""
SelectionDAG builder, combiner, and scheduler.

DAGBuilder  — Translates IR instructions into SelectionDAG nodes.
DAGCombiner — Peephole optimizations over the DAG (fold, simplify).
DAGScheduler — Schedules DAG into linearized MachineInstr list.
"""

from __future__ import annotations

from scratchv.ir.types import OpCode, Program, Function, BasicBlock, Instruction
from scratchv.codegen.sdnode import (
    MVT, SDNodeOpcode, SDNodeFlags, SDValue, SelectionDAG,
)
from scratchv.backend.register_alloc import MachineInstr, MachineOp, MachineOperand


# ═══════════════════════════════════════════════════════════
# IR type → MVT mapping
# ═══════════════════════════════════════════════════════════

def _ir_to_mvt(dtype) -> MVT:
    from scratchv.ir.types import DataType
    return {
        DataType.FLOAT32: MVT.f32,
        DataType.FLOAT64: MVT.f64,
        DataType.INT32: MVT.i32,
        DataType.INT64: MVT.i64,
    }.get(dtype, MVT.i32)


# ═══════════════════════════════════════════════════════════
# DAGBuilder — IR → SelectionDAG
# ═══════════════════════════════════════════════════════════

class DAGBuilder:
    """Build a SelectionDAG from a ScratchV IR Program."""

    def __init__(self, program: Program):
        self.program = program
        self.dag = SelectionDAG()
        # Maps IR value names → SDValue
        self._value_map: dict[str, SDValue] = {}
        self._chain = self.dag.entry_token

    def run(self) -> SelectionDAG:
        """Build the DAG for all functions."""
        for func in self.program.functions:
            self._build_function(func)
        return self.dag

    def _build_function(self, func: Function) -> None:
        self._value_map.clear()
        self._chain = self.dag.entry_token

        # Map function parameters to CopyFromReg nodes
        for i, param in enumerate(func.params):
            reg = self.dag.get_register(f"a{i}" if i < 8 else f"s{i-8}")
            val = self.dag.get_copy_from_reg(reg)
            self._chain = val.node.get_chain() or self._chain
            self._value_map[param.name] = val

        for block in func.blocks:
            self._build_block(block, func)

    def _build_block(self, block: BasicBlock, func: Function) -> None:
        for instr in block.instructions:
            self._build_instruction(instr)

    def _build_instruction(self, instr: Instruction) -> None:
        handler = getattr(self, f"_build_{instr.opcode.value}", None)
        if handler is None:
            raise ValueError(f"No DAG builder for opcode: {instr.opcode}")
        handler(instr)

    def _get_val(self, ir_val) -> SDValue:
        """Map an IR operand Value to an SDValue."""
        if ir_val.is_constant and ir_val.const_value is not None:
            vt = _ir_to_mvt(ir_val.dtype)
            if vt.is_float:
                return self.dag.get_constant_fp(float(ir_val.const_value), vt)
            return self.dag.get_constant(int(ir_val.const_value), vt)
        name = ir_val.name
        if name not in self._value_map:
            self._value_map[name] = self.dag.get_undef(_ir_to_mvt(ir_val.dtype))
        return self._value_map[name]

    def _set_val(self, ir_val, sdval: SDValue) -> None:
        self._value_map[ir_val.name] = sdval

    # ── Arithmetic ─────────────────────────────────────

    def _build_add(self, instr: Instruction) -> None:
        lhs = self._get_val(instr.operands[0])
        rhs = self._get_val(instr.operands[1])
        if lhs.value_type.is_float:
            val = self.dag.get_fadd(lhs, rhs)
        else:
            val = self.dag.get_add(lhs, rhs)
        self._set_val(instr.dest, val)

    def _build_sub(self, instr: Instruction) -> None:
        lhs = self._get_val(instr.operands[0])
        rhs = self._get_val(instr.operands[1])
        if lhs.value_type.is_float:
            val = self.dag.get_fsub(lhs, rhs)
        else:
            val = self.dag.get_sub(lhs, rhs)
        self._set_val(instr.dest, val)

    def _build_mul(self, instr: Instruction) -> None:
        lhs = self._get_val(instr.operands[0])
        rhs = self._get_val(instr.operands[1])
        if lhs.value_type.is_float:
            val = self.dag.get_fmul(lhs, rhs)
        else:
            val = self.dag.get_mul(lhs, rhs)
        self._set_val(instr.dest, val)

    def _build_div(self, instr: Instruction) -> None:
        lhs = self._get_val(instr.operands[0])
        rhs = self._get_val(instr.operands[1])
        if lhs.value_type.is_float:
            val = self.dag.get_fdiv(lhs, rhs)
        else:
            val = self.dag.get_div(lhs, rhs)
        self._set_val(instr.dest, val)

    def _build_neg(self, instr: Instruction) -> None:
        src = self._get_val(instr.operands[0])
        # neg = sub 0, x or fneg x
        if src.value_type.is_float:
            zero = self.dag.get_constant_fp(0.0, src.value_type)
            val = self.dag.get_fsub(zero, src)
        else:
            zero = self.dag.get_constant(0, src.value_type)
            val = self.dag.get_sub(zero, src)
        self._set_val(instr.dest, val)

    def _build_exp(self, instr: Instruction) -> None:
        src = self._get_val(instr.operands[0])
        val = self.dag.get_call("expf" if src.value_type == MVT.f32 else "exp",
                                [src], vt=src.value_type)
        self._chain = val.node.get_chain() or self._chain
        self._set_val(instr.dest, val)

    def _build_load_const(self, instr: Instruction) -> None:
        v = instr.attrs.get("value", 0)
        vt = _ir_to_mvt(instr.dest.dtype) if instr.dest else MVT.f32
        if vt.is_float:
            val = self.dag.get_constant_fp(float(v), vt)
        else:
            val = self.dag.get_constant(int(v), vt)
        self._set_val(instr.dest, val)

    def _build_load(self, instr: Instruction) -> None:
        addr = self._get_val(instr.operands[0])
        vt = _ir_to_mvt(instr.dest.dtype) if instr.dest else MVT.i32
        val = self.dag.get_load(addr, vt, chain=self._chain)
        self._chain = val.node.get_chain() or self._chain
        self._set_val(instr.dest, val)

    def _build_store(self, instr: Instruction) -> None:
        addr = self._get_val(instr.operands[0])
        val = self._get_val(instr.operands[1])
        chain = self.dag.get_store(addr, val, chain=self._chain)
        self._chain = chain

    def _build_alloca(self, instr: Instruction) -> None:
        size = instr.attrs.get("size", 4)
        vt = _ir_to_mvt(instr.dest.dtype) if instr.dest else MVT.i32
        # Represent as a constant pointer offset (from sp)
        val = self.dag.get_constant(size, vt)
        self._set_val(instr.dest, val)

    # ── Control flow ───────────────────────────────────

    def _build_for(self, instr: Instruction) -> None:
        iv = instr.dest
        start = instr.attrs.get("start", 0)
        end = instr.attrs.get("end", 0)
        val = self.dag.get_constant(start, MVT.i32)
        self._value_map[instr.dest.name] = val
        # Store loop context for endfor
        self._loop_ctx = {
            "iv_name": iv.name,
            "end": end,
        }

    def _build_endfor(self, instr: Instruction) -> None:
        ctx = getattr(self, "_loop_ctx", None)
        if ctx is None:
            return
        iv_name = ctx["iv_name"]
        iv = self._value_map.get(iv_name)
        if iv is not None:
            one = self.dag.get_constant(1, MVT.i32)
            inc = self.dag.get_add(iv, one)
            self._value_map[iv_name] = inc
        self._loop_ctx = None

    def _build_br(self, instr: Instruction) -> None:
        self._chain = self.dag.get_br(instr.target or "", chain=self._chain)

    def _build_br_if(self, instr: Instruction) -> None:
        cond = self._get_val(instr.operands[0])
        targets = (instr.target or "").split(",")
        true_t = targets[0].strip() if len(targets) > 0 else ""
        false_t = targets[1].strip() if len(targets) > 1 else ""
        self._chain = self.dag.get_br_cc(cond, true_t, false_t, chain=self._chain)

    def _build_return(self, instr: Instruction) -> None:
        vals = [self._get_val(instr.operands[0])] if instr.operands else None
        self._chain = self.dag.get_ret(vals, chain=self._chain)

    def _build_label(self, instr: Instruction) -> None:
        pass  # labels are implicit in DAG

    # ── NN ops ─────────────────────────────────────────

    def _build_relu(self, instr: Instruction) -> None:
        src = self._get_val(instr.operands[0])
        val = self.dag.get_call("relu", [src], vt=src.value_type)
        self._chain = val.node.get_chain() or self._chain
        self._set_val(instr.dest, val)

    def _build_gelu(self, instr: Instruction) -> None:
        src = self._get_val(instr.operands[0])
        val = self.dag.get_call("gelu", [src], vt=src.value_type)
        self._chain = val.node.get_chain() or self._chain
        self._set_val(instr.dest, val)

    def _build_softmax(self, instr: Instruction) -> None:
        src = self._get_val(instr.operands[0])
        val = self.dag.get_call("softmax", [src], vt=src.value_type)
        self._chain = val.node.get_chain() or self._chain
        self._set_val(instr.dest, val)

    def _build_matmul(self, instr: Instruction) -> None:
        a = self._get_val(instr.operands[0])
        b = self._get_val(instr.operands[1])
        m = instr.attrs.get("m", 1)
        n = instr.attrs.get("n", 1)
        k = instr.attrs.get("k", 1)
        val = self.dag.get_call(f"matmul_m{m}_n{n}_k{k}", [a, b],
                                vt=_ir_to_mvt(instr.dest.dtype) if instr.dest else MVT.f32)
        self._chain = val.node.get_chain() or self._chain
        self._set_val(instr.dest, val)

    def _build_dot(self, instr: Instruction) -> None:
        a = self._get_val(instr.operands[0])
        b = self._get_val(instr.operands[1])
        length = instr.attrs.get("length", 1)
        val = self.dag.get_call(f"dot_len{length}", [a, b],
                                vt=_ir_to_mvt(instr.dest.dtype) if instr.dest else MVT.f32)
        self._chain = val.node.get_chain() or self._chain
        self._set_val(instr.dest, val)


# ═══════════════════════════════════════════════════════════
# DAGCombiner — peephole optimizations on the DAG
# ═══════════════════════════════════════════════════════════

class DAGCombiner:
    """DAG-level peephole optimizations: constant folding, redundant removal."""

    def __init__(self, dag: SelectionDAG):
        self.dag = dag
        self._changed = False

    def run(self) -> int:
        """Run all DAG combines. Returns number of folds applied."""
        n_folds = 0
        # Iterate until stable
        for _ in range(32):  # limit iterations
            self._changed = False
            for node in reversed(self.dag._nodes):
                self._try_fold(node)
                if self._changed:
                    n_folds += 1
            if not self._changed:
                break
        return n_folds

    def _try_fold(self, node) -> None:
        """Try to fold a single node in-place."""
        handler = getattr(self, f"_fold_{node.opcode.value}", None)
        if handler is not None:
            handler(node)

    def _fold_ADD(self, node) -> None:
        """Constant fold: add(const, const) -> const"""
        lhs, rhs = self._get_const_binop(node)
        if lhs is not None and rhs is not None:
            val = self.dag.get_constant(lhs + rhs, node.value_type())
            self._replace_node(node, val)

    def _fold_SUB(self, node) -> None:
        lhs, rhs = self._get_const_binop(node)
        if lhs is not None and rhs is not None:
            val = self.dag.get_constant(lhs - rhs, node.value_type())
            self._replace_node(node, val)

    def _fold_MUL(self, node) -> None:
        lhs, rhs = self._get_const_binop(node)
        if lhs is not None and rhs is not None:
            val = self.dag.get_constant(lhs * rhs, node.value_type())
            self._replace_node(node, val)

    def _fold_DIV(self, node) -> None:
        lhs, rhs = self._get_const_binop(node)
        if lhs is not None and rhs is not None and rhs != 0:
            val = self.dag.get_constant(lhs // rhs, node.value_type())
            self._replace_node(node, val)

    def _fold_FADD(self, node) -> None:
        self._fold_fp_binop(node, lambda a, b: a + b)

    def _fold_FSUB(self, node) -> None:
        self._fold_fp_binop(node, lambda a, b: a - b)

    def _fold_FMUL(self, node) -> None:
        self._fold_fp_binop(node, lambda a, b: a * b)

    def _fold_FDIV(self, node) -> None:
        self._fold_fp_binop(node, lambda a, b: a / b)

    def _fold_fp_binop(self, node, op) -> None:
        lhs = self._get_fp_const(node, 0)
        rhs = self._get_fp_const(node, 1)
        if lhs is not None and rhs is not None:
            try:
                val = self.dag.get_constant_fp(op(lhs, rhs), node.value_type())
                self._replace_node(node, val)
            except (ZeroDivisionError, OverflowError, ValueError):
                pass

    def _get_const_binop(self, node):
        """Return (lhs_int, rhs_int) if both operands are Constant."""
        if len(node.operands) < 2:
            return None, None
        lhs = node.operands[0].node.get_constant_int()
        rhs = node.operands[1].node.get_constant_int()
        return lhs, rhs

    def _get_fp_const(self, node, idx: int):
        op = node.operands[idx]
        return op.node.get_constant_fp()

    def _replace_node(self, old_node, new_val: SDValue) -> None:
        """Replace all uses of old_node with new_val (simple)."""
        old_node._attributes["replaced_by"] = new_val
        self._changed = True


# ═══════════════════════════════════════════════════════════
# DAGScheduler — DAG → linear MachineInstr list
# ═══════════════════════════════════════════════════════════

class DAGScheduler:
    """Schedule a SelectionDAG into a linear sequence of MachineInstrs."""

    def __init__(self, dag: SelectionDAG):
        self.dag = dag

    def run(self) -> list[MachineInstr]:
        """Topological schedule: emit nodes in dependency order."""
        scheduled: set[int] = set()
        result: list[MachineInstr] = []
        label_counter = [0]

        def fresh_label(prefix="L"):
            label_counter[0] += 1
            return f"{prefix}_{label_counter[0]}"

        def schedule_node(node, chain_token=None):
            if node.node_id in scheduled:
                return
            # Schedule operands first (post-order DFS)
            for op in node.operands:
                if op.node.node_id not in scheduled:
                    schedule_node(op.node)
            scheduled.add(node.node_id)

            opcode = node.opcode
            try:
                machine_op = _SDNODE_TO_MACHINE_OP[opcode]
            except KeyError:
                # Skip nodes without a direct MachineOp mapping
                return

            dst = None
            src1 = None
            src2 = None
            comment = ""

            if opcode == SDNodeOpcode.Constant:
                dst = MachineOperand.vreg(f"t{node.node_id}")
                val = node.get_constant_int() or 0
                result.append(MachineInstr(
                    MachineOp.LI, dst, MachineOperand.immediate(val),
                    comment=f"const {val}"))
                return

            if opcode == SDNodeOpcode.ConstantFP:
                dst = MachineOperand.vreg(f"t{node.node_id}")
                val = node.get_constant_fp() or 0.0
                result.append(MachineInstr(
                    MachineOp.LI, dst, MachineOperand.immediate(int(val)),
                    comment=f"constfp {val}"))
                return

            if opcode == SDNodeOpcode.CopyFromReg:
                # Should be handled by register allocator
                reg = node._get_attr("reg_name", "zero")
                dst = MachineOperand.vreg(f"t{node.node_id}")
                result.append(MachineInstr(
                    MachineOp.MV, dst, MachineOperand.reg(reg),
                    comment="copy_from_reg"))
                return

            if opcode in (SDNodeOpcode.LOAD,):
                dst = MachineOperand.vreg(f"t{node.node_id}")
                src1 = _op_to_operand(node.operands[1], node, fresh_label)
                result.append(MachineInstr(
                    MachineOp.LW, dst, src1, comment="load"))
                return

            if opcode == SDNodeOpcode.STORE:
                src1 = _op_to_operand(node.operands[1], node, fresh_label)
                src2 = _op_to_operand(node.operands[2], node, fresh_label)
                result.append(MachineInstr(
                    MachineOp.SW, src1, src2, comment="store"))
                return

            if opcode == SDNodeOpcode.BR:
                target = node._get_attr("branch_target", "")
                result.append(MachineInstr(
                    MachineOp.J, comment=target))
                return

            if opcode == SDNodeOpcode.BR_CC:
                cond = _op_to_operand(node.operands[1], node, fresh_label)
                true_t = node._get_attr("true_target", "")
                false_t = node._get_attr("false_target", "")
                result.append(MachineInstr(
                    MachineOp.BNEZ, cond, comment=true_t))
                result.append(MachineInstr(
                    MachineOp.J, comment=false_t))
                return

            if opcode == SDNodeOpcode.RET:
                result.append(MachineInstr(
                    MachineOp.JALR, MachineOperand.vreg("zero"),
                    MachineOperand.vreg("ra"), comment="ret"))
                return

            if opcode == SDNodeOpcode.CALL:
                callee = node._get_attr("callee", "unknown")
                result.append(MachineInstr(
                    MachineOp.CALL, comment=callee))
                if node.num_values > 0:
                    dst = MachineOperand.vreg(f"t{node.node_id}")
                    result.append(MachineInstr(
                        MachineOp.MV, dst, MachineOperand.vreg("a0")))
                return

            # Generic binop emission
            if len(node.operands) >= 2:
                src1 = _op_to_operand(node.operands[0], node, fresh_label)
                src2 = _op_to_operand(node.operands[1], node, fresh_label)
            elif len(node.operands) >= 1:
                src1 = _op_to_operand(node.operands[0], node, fresh_label)

            if node.num_values > 0 and node._num_values > node.num_chain_results:
                dst = MachineOperand.vreg(f"t{node.node_id}")

            result.append(MachineInstr(machine_op, dst, src1, src2, comment))

        # Schedule all nodes
        for node in self.dag._nodes:
            schedule_node(node)

        return result


def _op_to_operand(sdval: SDValue, parent_node, fresh_label_fn) -> MachineOperand:
    """Convert SDValue to MachineOperand (vreg)."""
    if sdval.node.opcode == SDNodeOpcode.Constant:
        val = sdval.node.get_constant_int() or 0
        return MachineOperand.immediate(val)
    if sdval.node.opcode == SDNodeOpcode.ConstantFP:
        val = sdval.node.get_constant_fp() or 0.0
        return MachineOperand.immediate(int(val))
    if sdval.node.opcode == SDNodeOpcode.Register:
        name = sdval.node._get_attr("reg_name", "zero")
        return MachineOperand.reg(name)
    return MachineOperand.vreg(f"t{sdval.node.node_id}")


_SDNODE_TO_MACHINE_OP: dict[SDNodeOpcode, MachineOp] = {
    SDNodeOpcode.ADD:  MachineOp.ADD,
    SDNodeOpcode.SUB:  MachineOp.SUB,
    SDNodeOpcode.MUL:  MachineOp.MUL,
    SDNodeOpcode.DIV:  MachineOp.DIV,
    SDNodeOpcode.FADD: MachineOp.ADD,
    SDNodeOpcode.FSUB: MachineOp.SUB,
    SDNodeOpcode.FMUL: MachineOp.MUL,
    SDNodeOpcode.FDIV: MachineOp.DIV,
    SDNodeOpcode.NEG:  MachineOp.SUB,
    SDNodeOpcode.SETCC: MachineOp.SUB,  # placeholder
    SDNodeOpcode.LOAD: MachineOp.LW,
    SDNodeOpcode.STORE: MachineOp.SW,
    SDNodeOpcode.BR:   MachineOp.J,
    SDNodeOpcode.BR_CC: MachineOp.BNEZ,
    SDNodeOpcode.RET:  MachineOp.JALR,
    SDNodeOpcode.CALL: MachineOp.CALL,
    SDNodeOpcode.LI_Pseudo: MachineOp.LI,
    SDNodeOpcode.MV_Pseudo: MachineOp.MV,
    SDNodeOpcode.RELU: MachineOp.MAX,
}
