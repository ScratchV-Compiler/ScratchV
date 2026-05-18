"""
SDNode — LLVM-style SelectionDAG core types.

Provides machine value types (MVT), DAG node opcodes, node flags,
SDValue edges, SDNode definitions, and the SelectionDAG container.

Designed for DAG-based instruction selection in compilers targeting
RISC-V and similar architectures.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# MVT — Machine Value Type
# ═══════════════════════════════════════════════════════════════════════════════

class MVT(enum.Enum):
    """Machine Value Type for values flowing through the DAG.

    Attributes:
        i8 / i16 / i32 / i64:  Integer types of varying width.
        f32 / f64:             Floating-point types.
        Other:                 Token/chain type (side-effect ordering).
        Void:                  No value (e.g. void return).
    """

    i8 = "i8"
    i16 = "i16"
    i32 = "i32"
    i64 = "i64"
    f32 = "f32"
    f64 = "f64"
    Other = "other"
    Void = "void"

    @property
    def is_integer(self) -> bool:
        """True if this is an integer type (i8–i64)."""
        return self in (MVT.i8, MVT.i16, MVT.i32, MVT.i64)

    @property
    def is_float(self) -> bool:
        """True if this is a floating-point type (f32, f64)."""
        return self in (MVT.f32, MVT.f64)

    @property
    def size_bits(self) -> int:
        """Bit width of this type (0 for Other/Void)."""
        return {
            MVT.i8: 8, MVT.i16: 16, MVT.i32: 32, MVT.i64: 64,
            MVT.f32: 32, MVT.f64: 64,
        }.get(self, 0)

    @property
    def size_bytes(self) -> int:
        """Byte width of this type (0 for Other/Void)."""
        return self.size_bits // 8

    @staticmethod
    def from_size(bits: int, is_float: bool = False) -> MVT:
        """Resolve a bit width to the corresponding MVT.

        Args:
            bits: Bit width (8, 16, 32, or 64).
            is_float: If True, return a floating-point type.

        Returns:
            The corresponding MVT. Falls back to i32 for unknown widths.
        """
        if is_float:
            return {32: MVT.f32, 64: MVT.f64}.get(bits, MVT.f32)
        mapping = {8: MVT.i8, 16: MVT.i16, 32: MVT.i32, 64: MVT.i64}
        return mapping.get(bits, MVT.i32)


# ═══════════════════════════════════════════════════════════════════════════════
# SDNodeOpcode — DAG node operation codes
# ═══════════════════════════════════════════════════════════════════════════════

class SDNodeOpcode(enum.Enum):
    """LLVM-inspired SelectionDAG node opcodes.

    Each entry represents one kind of operation that can appear as a node
    in the DAG, including arithmetic, control flow, memory access, and
    target-specific pseudo-instructions.
    """

    # ── Constants ──────────────────────────────────────────────────────────
    Constant = "Constant"         # Integer constant
    ConstantFP = "ConstantFP"       # Floating-point constant
    Undef = "Undef"            # Undefined / poisoning value
    TargetConstant = "TargetConstant"   # Target-specific constant (CSR# etc.)

    # ── Integer arithmetic ─────────────────────────────────────────────────
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"     # Signed division
    UDIV = "UDIV"    # Unsigned division
    SRA = "SRA"     # Shift right arithmetic
    SRL = "SRL"     # Shift right logical
    SHL = "SHL"     # Shift left
    NEG = "NEG"     # 0 - x

    # ── Floating-point arithmetic ──────────────────────────────────────────
    FADD = "FADD"
    FSUB = "FSUB"
    FMUL = "FMUL"
    FDIV = "FDIV"
    FNEG = "FNEG"
    FABS = "FABS"

    # ── Comparison & branches ──────────────────────────────────────────────
    SETCC = "SETCC"   # Set on condition code → returns i1
    BR_CC = "BR_CC"   # Branch on condition code
    BR = "BR"      # Unconditional branch
    BRIND = "BRIND"   # Indirect branch (register target)
    RET = "RET"     # Return from function
    CALL = "CALL"    # Function call

    # ── Type conversion ────────────────────────────────────────────────────
    FP_EXTEND = "FP_EXTEND"
    FP_TRUNC = "FP_TRUNC"
    INT_TO_FP = "INT_TO_FP"
    FP_TO_INT = "FP_TO_INT"
    ANY_EXTEND = "ANY_EXTEND"
    TRUNCATE = "TRUNCATE"
    BITCAST = "BITCAST"

    # ── Memory ─────────────────────────────────────────────────────────────
    LOAD = "LOAD"
    STORE = "STORE"
    TokenFactor = "TokenFactor"

    # ── Pseudo / register ──────────────────────────────────────────────────
    CopyFromReg = "CopyFromReg"
    CopyToReg = "CopyToReg"
    Register = "Register"
    LI_Pseudo = "LI_Pseudo"
    MV_Pseudo = "MV_Pseudo"
    CALL_Pseudo = "CALL_Pseudo"
    RET_Pseudo = "RET_Pseudo"
    LoadAddress = "LoadAddress"

    # ── Neural-network ops ─────────────────────────────────────────────────
    RELU = "RELU"
    MAXPOOL = "MAXPOOL"
    GELU = "GELU"
    MATMUL = "MATMUL"

    # ── Property helpers ───────────────────────────────────────────────────

    @property
    def has_chain(self) -> bool:
        """True if this op carries side effects and needs a chain edge."""
        return self in _OP_HAS_CHAIN

    @property
    def is_memop(self) -> bool:
        """True if this is a memory load or store."""
        return self in _OP_IS_MEMOP

    @property
    def is_commutative(self) -> bool:
        """True if the operation is commutative (a+b == b+a)."""
        return self in (
            SDNodeOpcode.ADD, SDNodeOpcode.MUL,
            SDNodeOpcode.FADD, SDNodeOpcode.FMUL,
        )


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


# ═══════════════════════════════════════════════════════════════════════════════
# SDNodeFlags — per-node metadata
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SDNodeFlags:
    """Fine-grained flags attached to an SDNode.

    These mirror LLVM's SDNodeFlags and control later optimisations
    (e.g. fast-math flags enable more aggressive transforms).
    """

    no_nan: bool = False
    """Assume no NaN values (``fast`` flag for FP)."""

    no_signed_zeros: bool = False
    """Allow optimisations that ignore signed zero."""

    no_infs: bool = False
    """Assume no infinities."""

    no_unsafe_fp: bool = False
    """Allow all fast-math transforms."""

    is_volatile: bool = False
    """Memory access is volatile (must not be reordered)."""

    is_non_temporal: bool = False
    """Non-temporal memory access (bypass cache hint)."""

    alignment: int = 0
    """Known alignment in bytes (0 = default / unknown)."""


# ═══════════════════════════════════════════════════════════════════════════════
# SDValue — DAG edge reference
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SDValue:
    """A reference to a value produced by an SDNode.

    An SDValue pairs an SDNode with a result index, forming an edge
    in the DAG.  Result 0 is always the first non-chain value unless
    the node has no chain, in which case all results are data values.

    Attributes:
        node: The producer SDNode.
        resno: Which result of that node (0‑based).
    """

    node: "SDNode"
    resno: int = 0

    # ── Type query ────────────────────────────────────────────────────────

    @property
    def value_type(self) -> MVT:
        """The MVT of this value."""
        return self.node.value_type(self.resno)

    # ── Semantic predicates ───────────────────────────────────────────────

    def is_chain(self) -> bool:
        """True if this is a chain token (MVT.Other at the chain position)."""
        return (self.resno == self.node.num_chain_results
                and self.value_type == MVT.Other)

    def is_undef(self) -> bool:
        """True if this value originates from an Undef node."""
        return self.node.opcode == SDNodeOpcode.Undef

    # ── Equality — identity-based (by node pointer + result index) ─────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SDValue):
            return NotImplemented
        return self.node is other.node and self.resno == other.resno

    def __hash__(self) -> int:
        return id(self.node) ^ self.resno

    def __repr__(self) -> str:
        return f"t{self.node.node_id}.{self.resno}:{self.value_type.value}"


# ═══════════════════════════════════════════════════════════════════════════════
# SDNode — single DAG node
# ═══════════════════════════════════════════════════════════════════════════════

class SDNode:
    """A node in the SelectionDAG.

    Each node has an opcode, a list of result types, a list of operand SDValues
    (incoming edges), and optional metadata.  Nodes with side effects carry an
    implicit chain edge (``MVT.Other``) as their first result and operand.

    Layout convention per LLVM:
        [chain result (MVT.Other)]? [data result 0] [data result 1 …]

    Attributes:
        node_id:             Globally unique node identifier.
        opcode:              The operation this node performs.
        operands:            Incoming DAG edges (SDValues).
        flags:               Per-node flags (fast-math, volatility, …).
        dbg_info:            Optional debug / source location string.
        num_chain_results:   Number of chain-valued results (0 or 1).
    """

    __slots__ = (
        "node_id", "opcode", "_value_types", "operands",
        "flags", "dbg_info", "_num_types", "num_chain_results",
        "_attributes",
    )

    _next_id: int = 0

    def __init__(
        self,
        opcode: SDNodeOpcode,
        value_types: List[MVT],
        operands: List[SDValue],
        flags: Optional[SDNodeFlags] = None,
        dbg_info: str = "",
    ) -> None:
        self.node_id = SDNode._next_id
        SDNode._next_id += 1
        self.opcode = opcode
        self._value_types = list(value_types)
        self.operands = list(operands)
        self.flags = flags if flags is not None else SDNodeFlags()
        self.dbg_info = dbg_info
        self._num_types = len(self._value_types)
        self.num_chain_results = 0
        self._attributes: Dict[str, Any] = {}

    # ── Value type access ──────────────────────────────────────────────────

    def value_type(self, idx: int = 0) -> MVT:
        """Return the MVT of the *idx*-th result (0‑based)."""
        if 0 <= idx < self._num_types:
            return self._value_types[idx]
        return MVT.Void

    @property
    def num_values(self) -> int:
        """Number of non-chain data values produced by this node."""
        return self._num_types - self.num_chain_results

    # ── Chain helpers ──────────────────────────────────────────────────────

    @property
    def has_chain(self) -> bool:
        """True if the node has side effects and carries a chain."""
        return self.opcode.has_chain

    def get_chain(self) -> Optional[SDValue]:
        """Return the chain operand, or None if this node has no chain."""
        if self.has_chain:
            for op in self.operands:
                if op.is_chain():
                    return op
        return None

    # ── Constant accessors ─────────────────────────────────────────────────

    def get_constant_int(self) -> Optional[int]:
        """If this is a Constant node, return the stored integer value."""
        return self._attributes.get("const_val")

    def get_constant_fp(self) -> Optional[float]:
        """If this is a ConstantFP node, return the stored float value."""
        if self.opcode == SDNodeOpcode.ConstantFP:
            return self._attributes.get("const_fp")
        return None

    # ── Attribute bucket ───────────────────────────────────────────────────

    def get_attr(self, key: str, default: Any = None) -> Any:
        """Return an arbitrary attribute attached to this node."""
        return self._attributes.get(key, default)

    def set_attr(self, key: str, value: Any) -> None:
        """Attach an arbitrary attribute to this node."""
        self._attributes[key] = value

    # ── Debug ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        vt = ",".join(v.value for v in self._value_types)
        ops = ", ".join(str(op) for op in self.operands[:4])
        if len(self.operands) > 4:
            ops += f", … (+{len(self.operands) - 4})"
        return f"t{self.node_id}: {self.opcode.value} [{vt}] ← ({ops})"

    def dump(self, indent: str = "") -> str:
        """Return a multi-line debug dump of this node."""
        lines = [
            f"{indent}Node t{self.node_id}:",
            f"{indent}  Opcode: {self.opcode.value}",
            f"{indent}  Types:  {[v.value for v in self._value_types]}",
            f"{indent}  Operands ({len(self.operands)}):",
        ]
        for op in self.operands:
            lines.append(f"{indent}    {op}")
        if self._attributes:
            lines.append(f"{indent}  Attrs: {self._attributes}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SelectionDAG — DAG container & node factory
# ═══════════════════════════════════════════════════════════════════════════════

class SelectionDAG:
    """Owning container for SDNodes with factory methods.

    The DAG manages node lifetime, deduplication of constants, and
    provides a default *entry token* chain that all side-effecting
    nodes implicitly depend upon.  The *root* value is the DAG's
    terminal value (typically the return value or a token factor
    merging all side-effect chains).

    Typical usage::

        dag = SelectionDAG()
        a = dag.get_constant(42, MVT.i32)
        b = dag.get_constant(10, MVT.i32)
        c = dag.get_add(a, b)
        print(dag.dump())
    """

    def __init__(self) -> None:
        self._nodes: List[SDNode] = []
        # Deduplication cache:  (kind_key, ...) → SDNode
        self._node_map: Dict[Tuple, SDNode] = {}
        self._root: Optional[SDValue] = None
        self._debug_loc: Dict[int, str] = {}

        # Reset the global node counter so each DAG starts from t0.
        SDNode._next_id = 0

        # Create the entry chain token — all side-effecting nodes in
        # a function ultimately chain back to this.
        entry = self._new_node(
            SDNodeOpcode.TokenFactor, [MVT.Other], [],
            dbg_info="EntryToken",
        )
        entry.num_chain_results = 1
        self._entry_token = SDValue(entry, 0)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def entry_token(self) -> SDValue:
        """The DAG's root chain token (all side effects hang off this)."""
        return self._entry_token

    @property
    def root(self) -> Optional[SDValue]:
        """The terminal value of the DAG (return value / merged chain)."""
        return self._root

    @root.setter
    def root(self, val: SDValue) -> None:
        self._root = val

    @property
    def nodes(self) -> List[SDNode]:
        """A snapshot copy of all nodes currently in the DAG."""
        return list(self._nodes)

    # ── Low-level node creation ────────────────────────────────────────────

    def _new_node(
        self,
        opcode: SDNodeOpcode,
        value_types: List[MVT],
        operands: List[SDValue],
        flags: Optional[SDNodeFlags] = None,
        dbg_info: str = "",
        **attrs: Any,
    ) -> SDNode:
        """Allocate an SDNode, register it, and set its attribute bucket."""
        node = SDNode(opcode, value_types, operands, flags, dbg_info)
        if opcode.has_chain:
            node.num_chain_results = 1
        node._attributes = attrs
        self._nodes.append(node)
        return node

    # ── Factory methods — constants ────────────────────────────────────────

    def get_constant(self, val: int, vt: MVT = MVT.i32) -> SDValue:
        """Get or create a Constant node for integer *val*."""
        key: Tuple = ("const", vt, val)
        node = self._node_map.get(key)
        if node is None:
            node = self._new_node(SDNodeOpcode.Constant, [vt], [],
                                  const_val=val)
            self._node_map[key] = node
        return SDValue(node, 0)

    def get_constant_fp(self, val: float, vt: MVT = MVT.f32) -> SDValue:
        """Get or create a ConstantFP node for float *val*."""
        key: Tuple = ("constfp", vt, val)
        node = self._node_map.get(key)
        if node is None:
            node = self._new_node(SDNodeOpcode.ConstantFP, [vt], [],
                                  const_fp=val)
            self._node_map[key] = node
        return SDValue(node, 0)

    def get_undef(self, vt: MVT = MVT.i32) -> SDValue:
        """Get or create an Undef node of type *vt*."""
        key: Tuple = ("undef", vt)
        node = self._node_map.get(key)
        if node is None:
            node = self._new_node(SDNodeOpcode.Undef, [vt], [])
            self._node_map[key] = node
        return SDValue(node, 0)

    def get_target_constant(
        self, val: object, vt: MVT = MVT.i32
    ) -> SDValue:
        """Get or create a TargetConstant (target-specific literal)."""
        node = self._new_node(SDNodeOpcode.TargetConstant, [vt], [],
                              target_val=val)
        return SDValue(node, 0)

    # ── Factory methods — register transfer ────────────────────────────────

    def get_register(self, name: str, vt: MVT = MVT.i32) -> SDValue:
        """Create a Register node representing a named physical register."""
        node = self._new_node(SDNodeOpcode.Register, [vt], [],
                              reg_name=name)
        return SDValue(node, 0)

    def get_copy_from_reg(
        self, reg: SDValue,
        chain: Optional[SDValue] = None,
    ) -> SDValue:
        """Copy a value from a physical register.

        Returns the data value result (chain result is at index 0).
        """
        chain = chain or self._entry_token
        node = self._new_node(
            SDNodeOpcode.CopyFromReg,
            [MVT.Other, reg.value_type],
            [chain, reg],
        )
        node.num_chain_results = 1
        return SDValue(node, 1)  # data value

    def get_copy_to_reg(
        self, reg: SDValue, val: SDValue,
        chain: Optional[SDValue] = None,
    ) -> SDValue:
        """Copy a value to a physical register.  Returns the chain."""
        chain = chain or self._entry_token
        node = self._new_node(
            SDNodeOpcode.CopyToReg,
            [MVT.Other],
            [chain, reg, val],
        )
        node.num_chain_results = 1
        return SDValue(node, 0)

    # ── Factory methods — arithmetic ───────────────────────────────────────

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

    def _get_binop(
        self, opcode: SDNodeOpcode,
        lhs: SDValue, rhs: SDValue,
    ) -> SDValue:
        """Shared helper for binary operation node creation."""
        vt = lhs.value_type
        node = self._new_node(opcode, [vt], [lhs, rhs])
        return SDValue(node, 0)

    # ── Factory methods — memory ───────────────────────────────────────────

    def get_load(
        self,
        addr: SDValue,
        vt: MVT = MVT.i32,
        chain: Optional[SDValue] = None,
        flags: Optional[SDNodeFlags] = None,
    ) -> SDValue:
        """Create a LOAD node.  Returns the *data* result.

        The chain result is at index 0 if needed via ``node.get_chain()``.
        """
        chain = chain or self._entry_token
        node = self._new_node(
            SDNodeOpcode.LOAD, [MVT.Other, vt],
            [chain, addr], flags=flags,
        )
        node.num_chain_results = 1
        return SDValue(node, 1)

    def get_store(
        self,
        addr: SDValue,
        val: SDValue,
        chain: Optional[SDValue] = None,
        flags: Optional[SDNodeFlags] = None,
    ) -> SDValue:
        """Create a STORE node.  Returns the chain result."""
        chain = chain or self._entry_token
        node = self._new_node(
            SDNodeOpcode.STORE, [MVT.Other],
            [chain, addr, val], flags=flags,
        )
        node.num_chain_results = 1
        return SDValue(node, 0)

    # ── Factory methods — control flow ─────────────────────────────────────

    def get_br(
        self, target: str,
        chain: Optional[SDValue] = None,
    ) -> SDValue:
        """Create an unconditional branch to *target*."""
        chain = chain or self._entry_token
        node = self._new_node(
            SDNodeOpcode.BR, [MVT.Other],
            [chain], branch_target=target,
        )
        node.num_chain_results = 1
        return SDValue(node, 0)

    def get_br_cc(
        self, cond: SDValue,
        true_target: str, false_target: str,
        chain: Optional[SDValue] = None,
    ) -> SDValue:
        """Create a conditional branch."""
        chain = chain or self._entry_token
        node = self._new_node(
            SDNodeOpcode.BR_CC, [MVT.Other],
            [chain, cond],
            true_target=true_target, false_target=false_target,
        )
        node.num_chain_results = 1
        return SDValue(node, 0)

    def get_ret(
        self,
        values: Optional[List[SDValue]] = None,
        chain: Optional[SDValue] = None,
    ) -> SDValue:
        """Create a return node."""
        chain = chain or self._entry_token
        ops = [chain] + (values or [])
        node = self._new_node(SDNodeOpcode.RET, [MVT.Other], ops)
        node.num_chain_results = 1
        return SDValue(node, 0)

    def get_call(
        self,
        callee: str,
        args: List[SDValue],
        vt: MVT = MVT.i32,
        chain: Optional[SDValue] = None,
    ) -> SDValue:
        """Create a call node.  Returns the *data* result.

        The chain result is at index 0; the data result is at index 1.
        """
        chain = chain or self._entry_token
        tc = self.get_target_constant(callee)
        node = self._new_node(
            SDNodeOpcode.CALL, [MVT.Other, vt],
            [chain, tc] + args,
            callee=callee,
        )
        node.num_chain_results = 1
        return SDValue(node, 1)

    def get_token_factor(self, chains: List[SDValue]) -> SDValue:
        """Merge multiple chain tokens into one.

        If only one chain is given it is returned as-is.
        """
        if len(chains) == 1:
            return chains[0]
        node = self._new_node(
            SDNodeOpcode.TokenFactor, [MVT.Other], chains,
        )
        node.num_chain_results = 1
        return SDValue(node, 0)

    # ── DAG lifetime ───────────────────────────────────────────────────────

    def clear(self) -> None:
        """Reset the entire DAG, discarding all nodes."""
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
        """Return a human-readable dump of the entire DAG."""
        lines = ["SelectionDAG:"]
        lines.append(f"  EntryToken: t{self._entry_token.node.node_id}")
        if self._root is not None:
            lines.append(f"  Root:       {self._root}")
        lines.append(f"  Nodes ({len(self._nodes)}):")
        for node in self._nodes:
            lines.append(f"    {node}")
        return "\n".join(lines)
