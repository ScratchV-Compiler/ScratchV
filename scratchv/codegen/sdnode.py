"""
SDNode: LLVM-style SelectionDAG node definitions for ScratchV.

Provides the core DAG node types, machine value types (MVT), opcodes,
and the SelectionDAG container used for DAG-based instruction selection.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════
# MVT — Machine Value Type
# ═══════════════════════════════════════════════════════════

class MVT(enum.Enum):
    """Machine Value Type — represents the type of a value in the DAG."""
    i8    = "i8"
    i16   = "i16"
    i32   = "i32"
    i64   = "i64"
    f32   = "f32"
    f64   = "f64"
    Other = "other"
    Void  = "void"

    @property
    def is_integer(self) -> bool:
        return self in (MVT.i8, MVT.i16, MVT.i32, MVT.i64)

    @property
    def is_float(self) -> bool:
        return self in (MVT.f32, MVT.f64)

    @property
    def size_bits(self) -> int:
        return {
            MVT.i8: 8, MVT.i16: 16, MVT.i32: 32, MVT.i64: 64,
            MVT.f32: 32, MVT.f64: 64,
        }.get(self, 0)

    @property
    def size_bytes(self) -> int:
        return self.size_bits // 8

    @staticmethod
    def from_size(bits: int, is_float: bool = False) -> MVT:
        if is_float:
            return {32: MVT.f32, 64: MVT.f64}.get(bits, MVT.f32)
        return {8: MVT.i8, 16: MVT.i16, 32: MVT.i32, 64: MVT.i64}.get(bits, MVT.i32)


# ═══════════════════════════════════════════════════════════
# SDNodeOpcode — DAG node operation codes
# ═══════════════════════════════════════════════════════════

class SDNodeOpcode(enum.Enum):
    """LLVM-inspired SelectionDAG node opcodes."""
    # ── Constants ──────────────────────────────────────
    Constant      = "Constant"        # integer constant
    ConstantFP    = "ConstantFP"      # floating-point constant
    Undef         = "Undef"           # undefined value
    TargetConstant = "TargetConstant" # target-specific constant (e.g. CSR)

    # ── Arithmetic ─────────────────────────────────────
    ADD     = "ADD"
    SUB     = "SUB"
    MUL     = "MUL"
    DIV     = "DIV"
    NEG     = "NEG"
    UDIV    = "UDIV"      # unsigned
    SRA     = "SRA"       # shift right arithmetic
    SRL     = "SRL"       # shift right logical
    SHL     = "SHL"       # shift left

    # Floating-point
    FADD    = "FADD"
    FSUB    = "FSUB"
    FMUL    = "FMUL"
    FDIV    = "FDIV"
    FNEG    = "FNEG"
    FABS    = "FABS"

    # ── Comparison ─────────────────────────────────────
    SETCC   = "SETCC"     # set condition (returns i1)
    BR_CC   = "BR_CC"     # branch on condition code

    # ── Type conversion ────────────────────────────────
    FP_EXTEND  = "FP_EXTEND"
    FP_TRUNC   = "FP_TRUNC"
    INT_TO_FP  = "INT_TO_FP"
    FP_TO_INT  = "FP_TO_INT"
    ANY_EXTEND = "ANY_EXTEND"
    TRUNCATE   = "TRUNCATE"
    BITCAST    = "BITCAST"

    # ── Memory ─────────────────────────────────────────
    LOAD      = "LOAD"
    STORE     = "STORE"
    TokenFactor = "TokenFactor"

    # ── Control ────────────────────────────────────────
    BR         = "BR"
    BRIND      = "BRIND"     # indirect branch
    RET        = "RET"
    CALL       = "CALL"

    # ── Pseudo ─────────────────────────────────────────
    CopyFromReg = "CopyFromReg"
    CopyToReg   = "CopyToReg"
    Register    = "Register"

    # ── Target-specific RISC-V ─────────────────────────
    LI_Pseudo     = "LI_Pseudo"
    MV_Pseudo     = "MV_Pseudo"
    CALL_Pseudo   = "CALL_Pseudo"
    RET_Pseudo    = "RET_Pseudo"
    LoadAddress   = "LoadAddress"

    # ── NN ops (low-level DAG nodes) ───────────────────
    RELU      = "RELU"
    MAXPOOL   = "MAXPOOL"
    GELU      = "GELU"
    MATMUL    = "MATMUL"

    # ── Properties ─────────────────────────────────────

    @property
    def has_chain(self) -> bool:
        """True if this op has side effects and needs a chain token."""
        return self in _OP_HAS_CHAIN

    @property
    def is_memop(self) -> bool:
        """True if this is a memory operation."""
        return self in _OP_IS_MEMOP

    @property
    def is_commutative(self) -> bool:
        return self in (SDNodeOpcode.ADD, SDNodeOpcode.MUL,
                        SDNodeOpcode.FADD, SDNodeOpcode.FMUL)


_OP_HAS_CHAIN = frozenset({
    SDNodeOpcode.LOAD, SDNodeOpcode.STORE,
    SDNodeOpcode.BR, SDNodeOpcode.BR_CC, SDNodeOpcode.BRIND,
    SDNodeOpcode.RET, SDNodeOpcode.CALL,
    SDNodeOpcode.TokenFactor,
    SDNodeOpcode.CopyToReg, SDNodeOpcode.CopyFromReg,
    SDNodeOpcode.CALL_Pseudo, SDNodeOpcode.RET_Pseudo,
})

_OP_IS_MEMOP = frozenset({
    SDNodeOpcode.LOAD, SDNodeOpcode.STORE,
})


# ═══════════════════════════════════════════════════════════
# SDNodeFlags
# ═══════════════════════════════════════════════════════════

@dataclass
class SDNodeFlags:
    """Flags attached to an SDNode."""
    no_nan: bool = False
    no_signed_zeros: bool = False
    no_infs: bool = False
    no_unsafe_fp: bool = False
    is_volatile: bool = False
    is_non_temporal: bool = False
    alignment: int = 0       # in bytes, 0 = default


# ═══════════════════════════════════════════════════════════
# SDValue — edge in the DAG (node + result index)
# ═══════════════════════════════════════════════════════════

@dataclass(slots=True)
class SDValue:
    """Reference to a value produced by an SDNode."""
    node: SDNode
    resno: int = 0

    @property
    def value_type(self) -> MVT:
        return self.node.value_type(self.resno)

    def __eq__(self, other) -> bool:
        if not isinstance(other, SDValue):
            return NotImplemented
        return self.node is other.node and self.resno == other.resno

    def __hash__(self) -> int:
        return id(self.node) ^ self.resno

    def __repr__(self) -> str:
        return f"t{self.node.node_id}.{self.resno}:{self.value_type.value}"

    def is_chain(self) -> bool:
        return self.resno == self.node.num_chain_results and self.value_type == MVT.Other

    def is_undef(self) -> bool:
        return self.node.opcode == SDNodeOpcode.Undef


# ═══════════════════════════════════════════════════════════
# SDNode — single DAG node
# ═══════════════════════════════════════════════════════════

class SDNode:
    """A node in the SelectionDAG. Each node produces one or more results.

    Layout:
        [chain result (MVT.Other)]? [data results ...]
    """

    _next_id: int = 0

    __slots__ = (
        "node_id", "opcode", "_value_types", "operands",
        "flags", "dbg_info", "_num_values", "num_chain_results",
        "_attributes",
    )

    def __init__(
        self,
        opcode: SDNodeOpcode,
        value_types: list[MVT],
        operands: list[SDValue],
        flags: SDNodeFlags | None = None,
        dbg_info: str = "",
    ):
        self.node_id = SDNode._next_id
        SDNode._next_id += 1
        self.opcode = opcode
        self._value_types = list(value_types)
        self.operands = list(operands)
        self.flags = flags or SDNodeFlags()
        self.dbg_info = dbg_info
        self._num_values = len(self._value_types)
        self.num_chain_results = 0
        self._attributes = {}

    # ── Value types ────────────────────────────────────

    def value_type(self, idx: int = 0) -> MVT:
        return self._value_types[idx] if idx < self._num_values else MVT.Void

    @property
    def num_values(self) -> int:
        """Number of non-chain value results."""
        return self._num_values - self.num_chain_results

    @property
    def has_chain(self) -> bool:
        return self.opcode.has_chain

    def get_chain(self) -> SDValue | None:
        """Get the chain operand, if any."""
        if self.has_chain:
            for op in self.operands:
                if op.is_chain():
                    return op
        return None

    # ── Constant accessors ─────────────────────────────

    def get_constant_int(self) -> int | None:
        """If this is a Constant node, return the integer value."""
        return self._get_attr("const_val")

    def get_constant_fp(self) -> float | None:
        if self.opcode == SDNodeOpcode.ConstantFP:
            return self._get_attr("const_fp")
        return None

    def _get_attr(self, key: str, default=None):
        return self._attributes.get(key, default)

    # ── Debug ──────────────────────────────────────────

    def __repr__(self) -> str:
        vt = ",".join(v.value for v in self._value_types)
        ops = ", ".join(str(op) for op in self.operands[:4])
        if len(self.operands) > 4:
            ops += f", ... (+{len(self.operands)-4})"
        return (f"t{self.node_id}: {self.opcode.value} [{vt}] "
                f"<- ({ops})")

    def dump(self, indent: str = "") -> str:
        lines = [f"{indent}Node t{self.node_id}:"]
        lines.append(f"{indent}  Opcode: {self.opcode.value}")
        lines.append(f"{indent}  Types:  {[v.value for v in self._value_types]}")
        lines.append(f"{indent}  Operands ({len(self.operands)}):")
        for op in self.operands:
            lines.append(f"{indent}    {op}")
        if self._attributes:
            lines.append(f"{indent}  Attrs: {self._attributes}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# SelectionDAG — container & node factory
# ═══════════════════════════════════════════════════════════

class SelectionDAG:
    """Manages all SDNodes and provides factory methods.

    The DAG uses a single chain token (EntryToken) that all side-effecting
    nodes implicitly depend upon as the root chain.
    """

    def __init__(self):
        self._nodes: list[SDNode] = []
        self._node_map: dict[tuple, SDNode] = {}  # dedup cache
        self._root: Optional[SDValue] = None
        self._debug_loc: dict[int, str] = {}
        # Reset node ID counter
        SDNode._next_id = 0
        # Create entry token (root chain)
        entry = self._new_node(
            SDNodeOpcode.TokenFactor,
            [MVT.Other],
            [],
            dbg_info="EntryToken",
        )
        entry.num_chain_results = 1
        self._entry_token = SDValue(entry, 0)

    # ── Properties ─────────────────────────────────────

    @property
    def entry_token(self) -> SDValue:
        return self._entry_token

    @property
    def root(self) -> SDValue | None:
        return self._root

    @root.setter
    def root(self, val: SDValue) -> None:
        self._root = val

    @property
    def nodes(self) -> list[SDNode]:
        return list(self._nodes)

    # ── Node creation ──────────────────────────────────

    def _new_node(
        self,
        opcode: SDNodeOpcode,
        value_types: list[MVT],
        operands: list[SDValue],
        flags: SDNodeFlags | None = None,
        dbg_info: str = "",
        **attrs,
    ) -> SDNode:
        node = SDNode(opcode, value_types, operands, flags, dbg_info)
        if opcode.has_chain:
            node.num_chain_results = 1
        node._attributes = attrs
        self._nodes.append(node)
        return node

    def get_constant(self, val: int, vt: MVT = MVT.i32) -> SDValue:
        """Get or create a Constant node."""
        key = ("const", vt, val)
        if key in self._node_map:
            return SDValue(self._node_map[key], 0)
        node = self._new_node(SDNodeOpcode.Constant, [vt], [], const_val=val)
        self._node_map[key] = node
        return SDValue(node, 0)

    def get_constant_fp(self, val: float, vt: MVT = MVT.f32) -> SDValue:
        key = ("constfp", vt, val)
        if key in self._node_map:
            return SDValue(self._node_map[key], 0)
        node = self._new_node(SDNodeOpcode.ConstantFP, [vt], [], const_fp=val)
        self._node_map[key] = node
        return SDValue(node, 0)

    def get_undef(self, vt: MVT = MVT.i32) -> SDValue:
        key = ("undef", vt)
        if key in self._node_map:
            return SDValue(self._node_map[key], 0)
        node = self._new_node(SDNodeOpcode.Undef, [vt], [])
        self._node_map[key] = node
        return SDValue(node, 0)

    def get_register(self, name: str, vt: MVT = MVT.i32) -> SDValue:
        node = self._new_node(SDNodeOpcode.Register, [vt], [],
                              reg_name=name)
        return SDValue(node, 0)

    def get_copy_from_reg(self, reg: SDValue, chain: SDValue | None = None) -> SDValue:
        chain = chain or self._entry_token
        node = self._new_node(
            SDNodeOpcode.CopyFromReg,
            [MVT.Other, reg.value_type],
            [chain, reg],
        )
        node.num_chain_results = 1
        return SDValue(node, 1)  # data result

    def get_copy_to_reg(self, reg: SDValue, val: SDValue,
                        chain: SDValue | None = None) -> SDValue:
        chain = chain or self._entry_token
        node = self._new_node(
            SDNodeOpcode.CopyToReg,
            [MVT.Other],
            [chain, reg, val],
        )
        node.num_chain_results = 1
        return SDValue(node, 0)  # chain result

    def get_add(self, lhs: SDValue, rhs: SDValue) -> SDValue:
        return self._get_binop(SDNodeOpcode.ADD, lhs, rhs)

    def get_sub(self, lhs: SDValue, rhs: SDValue) -> SDValue:
        return self._get_binop(SDNodeOpcode.SUB, lhs, rhs)

    def get_mul(self, lhs: SDValue, rhs: SDValue) -> SDValue:
        return self._get_binop(SDNodeOpcode.MUL, lhs, rhs)

    def get_div(self, lhs: SDValue, rhs: SDValue) -> SDValue:
        return self._get_binop(SDNodeOpcode.DIV, lhs, rhs)

    def get_fadd(self, lhs: SDValue, rhs: SDValue) -> SDValue:
        return self._get_binop(SDNodeOpcode.FADD, lhs, rhs)

    def get_fsub(self, lhs: SDValue, rhs: SDValue) -> SDValue:
        return self._get_binop(SDNodeOpcode.FSUB, lhs, rhs)

    def get_fmul(self, lhs: SDValue, rhs: SDValue) -> SDValue:
        return self._get_binop(SDNodeOpcode.FMUL, lhs, rhs)

    def get_fdiv(self, lhs: SDValue, rhs: SDValue) -> SDValue:
        return self._get_binop(SDNodeOpcode.FDIV, lhs, rhs)

    def _get_binop(self, opcode: SDNodeOpcode,
                   lhs: SDValue, rhs: SDValue) -> SDValue:
        vt = lhs.value_type
        node = self._new_node(opcode, [vt], [lhs, rhs])
        return SDValue(node, 0)

    def get_load(self, addr: SDValue, vt: MVT = MVT.i32,
                 chain: SDValue | None = None,
                 flags: SDNodeFlags | None = None) -> SDValue:
        """Create a LOAD node. Returns (chain, data)."""
        chain = chain or self._entry_token
        node = self._new_node(
            SDNodeOpcode.LOAD, [MVT.Other, vt],
            [chain, addr],
            flags=flags,
        )
        node.num_chain_results = 1
        return SDValue(node, 1)  # data result

    def get_store(self, addr: SDValue, val: SDValue,
                  chain: SDValue | None = None,
                  flags: SDNodeFlags | None = None) -> SDValue:
        """Create a STORE node. Returns the chain."""
        chain = chain or self._entry_token
        node = self._new_node(
            SDNodeOpcode.STORE, [MVT.Other],
            [chain, addr, val],
            flags=flags,
        )
        node.num_chain_results = 1
        return SDValue(node, 0)  # chain result

    def get_br(self, target: str, chain: SDValue | None = None) -> SDValue:
        chain = chain or self._entry_token
        node = self._new_node(SDNodeOpcode.BR, [MVT.Other],
                              [chain], branch_target=target)
        node.num_chain_results = 1
        return SDValue(node, 0)

    def get_br_cc(self, cond: SDValue, true_target: str, false_target: str,
                  chain: SDValue | None = None) -> SDValue:
        chain = chain or self._entry_token
        node = self._new_node(
            SDNodeOpcode.BR_CC, [MVT.Other],
            [chain, cond],
            true_target=true_target, false_target=false_target,
        )
        node.num_chain_results = 1
        return SDValue(node, 0)

    def get_ret(self, values: list[SDValue] | None = None,
                chain: SDValue | None = None) -> SDValue:
        chain = chain or self._entry_token
        ops = [chain] + (values or [])
        node = self._new_node(SDNodeOpcode.RET, [MVT.Other], ops)
        node.num_chain_results = 1
        return SDValue(node, 0)

    def get_call(self, callee: str, args: list[SDValue],
                 vt: MVT = MVT.i32,
                 chain: SDValue | None = None) -> SDValue:
        """Create a CALL node. Returns (chain, data)."""
        chain = chain or self._entry_token
        node = self._new_node(
            SDNodeOpcode.CALL, [MVT.Other, vt],
            [chain, self.get_target_constant(callee)] + args,
            callee=callee,
        )
        node.num_chain_results = 1
        return SDValue(node, 1)  # data result

    def get_target_constant(self, val: str | int, vt: MVT = MVT.i32) -> SDValue:
        node = self._new_node(SDNodeOpcode.TargetConstant, [vt],
                              [], target_val=val)
        return SDValue(node, 0)

    def get_token_factor(self, chains: list[SDValue]) -> SDValue:
        """Merge multiple chains into one."""
        if len(chains) == 1:
            return chains[0]
        node = self._new_node(SDNodeOpcode.TokenFactor, [MVT.Other], chains)
        node.num_chain_results = 1
        return SDValue(node, 0)

    # ── DAG lifetime ───────────────────────────────────

    def clear(self) -> None:
        self._nodes.clear()
        self._node_map.clear()
        self._root = None
        self._debug_loc.clear()
        SDNode._next_id = 0
        entry = self._new_node(
            SDNodeOpcode.TokenFactor, [MVT.Other], [],
            dbg_info="EntryToken",
        )
        entry.num_chain_results = 1
        self._entry_token = SDValue(entry, 0)

    def dump(self) -> str:
        lines = ["SelectionDAG:"]
        lines.append(f"  EntryToken: t{self._entry_token.node.node_id}")
        if self._root:
            lines.append(f"  Root:       {self._root}")
        lines.append("  Nodes:")
        for node in self._nodes:
            lines.append(f"    {node}")
        return "\n".join(lines)
