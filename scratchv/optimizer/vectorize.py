"""FOR-loop strip-mining vectorizer (Topic 29, phase 1).

Rewrites straight-line ``FOR`` regions whose memory accesses follow the
element pattern ``base + i*elem_bytes`` into vector ops over strips of
``width`` lanes, plus a scalar remainder loop when the trip count is not
divisible by the width.

The transformed IR still contains vector ops; phase 1 lowers them to
scalar RV32IM code in the instruction selector via
``scratchv.backend.vector_scalar.VectorScalarExpander``.

Semantics: every accepted loop is transformed so that the vector body is
a per-strip regrouping of the original scalar body; lane k of a strip
corresponds exactly to iteration ``strip*width + k``.  Rejected loops are
left completely untouched and the reason is reported through
``PassResult.warnings`` / ``Vectorizer.last_report``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from scratchv.ir.types import (
    BasicBlock,
    DataType,
    Function,
    Instruction,
    OpCode,
    Program,
    Value,
)
from scratchv.pass_interface import CompilerPass, PassResult


# ── Stable rejection reasons (machine-readable) ──────────────────────────

REASON_NON_CONSTANT_BOUNDS = "non-constant-bounds"
REASON_UNSUPPORTED_STEP = "unsupported-loop-step"
REASON_UNSUPPORTED_START = "unsupported-loop-start"
REASON_TRIP_TOO_SMALL = "trip-count-too-small"
REASON_NESTED_CONTROL_FLOW = "nested-control-flow"
REASON_NO_ELEMENT_PATTERN = "no-memory-element-pattern"
REASON_NON_ELEMENTWISE_IV = "non-elementwise-iv-use"
REASON_ALIASING_STORE = "aliasing-store"
REASON_UNSUPPORTED_OP = "unsupported-op"
REASON_REGION_VALUE_ESCAPES = "region-value-escapes"


def validate_vector_width(width: object) -> int:
    """Validate a phase-1 strip width (``>= 2``) and return it.

    The CLI already restricts ``--vector-width`` to ``{2, 4}``, but the
    programmatic ``CompilerConfig`` API does not, and an unchecked width
    of ``0`` previously crashed with ``ZeroDivisionError`` inside the
    vectorizer (review F6).
    """
    if isinstance(width, bool) or not isinstance(width, int) or width < 2:
        raise ValueError(
            f"vector width must be an integer >= 2 (got {width!r})")
    return width


# ── Element op → vector op mapping ───────────────────────────────────────
#
# NOTE (deviation from design doc 2.2.1 C5): ``NEG`` is listed there as an
# element op, but the phase-1 vector instruction set has no ``VNEG``, so a
# region containing NEG is rejected as ``unsupported-op`` instead of being
# rewritten with a non-existent opcode.

_ELEMENT_VECTOR_OPS: dict[OpCode, OpCode] = {
    OpCode.ADD: OpCode.VADD,
    OpCode.SUB: OpCode.VSUB,
    OpCode.MUL: OpCode.VMUL,
    OpCode.DIV: OpCode.VDIV,
    OpCode.RELU: OpCode.VRELU,
}

_CONTROL_FLOW_OPS = frozenset({
    OpCode.FOR,
    OpCode.ENDFOR,
    OpCode.BR,
    OpCode.BR_IF,
    OpCode.LABEL,
    OpCode.RETURN,
})


# ── Report structures ────────────────────────────────────────────────────

@dataclass
class LoopVectorizationRecord:
    """Outcome of attempting to vectorize one ``FOR`` region."""

    function: str
    block: str
    index: int
    status: str
    reason: str = ""
    start: int = 0
    end: int = 0
    width: int = 0
    strips: int = 0
    remainder: int = 0
    vector_ops: int = 0


@dataclass
class _LoopRegion:
    for_index: int
    end_index: int
    for_instr: Instruction
    body: list[Instruction]


@dataclass
class _MemRef:
    """A LOAD/STORE inside the region with its decomposed address."""

    instr: Instruction
    index: int
    base: Optional[Value]
    offset: Optional[Value]
    canonical: bool

    @property
    def is_load(self) -> bool:
        return self.instr.opcode is OpCode.LOAD

    @property
    def lane_key(self) -> object:
        if self.canonical:
            return "canonical"
        if self.offset is not None:
            return self.offset.name
        return None


@dataclass
class _Plan:
    n: int
    width: int
    elem_bytes: int
    strips: int
    remainder: int
    iv_name: str
    outer_names: frozenset[str]
    offset_mul_indices: set[int] = field(default_factory=set)
    addr_add_bases: dict[int, Value] = field(default_factory=dict)
    mem_indices: set[int] = field(default_factory=set)
    elem_indices: set[int] = field(default_factory=set)
    const_indices: set[int] = field(default_factory=set)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class Vectorizer(CompilerPass):
    """Strip-mine vectorizer for straight-line ``FOR`` regions."""

    def __init__(self, program: Program, *, width: int = 4,
                 elem_bytes: int = 4) -> None:
        self.program = program
        self.width = validate_vector_width(width)
        self.elem_bytes = elem_bytes
        self._report: list[LoopVectorizationRecord] = []
        self._counter = 0
        self._used_names: set[str] = set()

    # ── CompilerPass API ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "vectorize"

    @property
    def last_report(self) -> list[LoopVectorizationRecord]:
        return list(self._report)

    def run(self, input_data: Any = None) -> PassResult:
        self._report = []
        warnings: list[str] = []
        changes = 0
        self._used_names = set()
        for func in self.program.functions:
            for block in func.blocks:
                for instr in block.instructions:
                    if instr.dest is not None:
                        self._used_names.add(instr.dest.name)
                    for op in instr.operands:
                        self._used_names.add(op.name)
            for param in func.params:
                self._used_names.add(param.name)

        for func in self.program.functions:
            for block in func.blocks:
                i = 0
                while i < len(block.instructions):
                    instr = block.instructions[i]
                    if instr.opcode is not OpCode.FOR:
                        i += 1
                        continue
                    region = self._collect_region(block, i)
                    if region is None:
                        i += 1
                        continue
                    rec = self._try_vectorize(func, block, region)
                    self._report.append(rec)
                    if rec.status == "vectorized":
                        changes += 1
                        i = self._skip_rewritten(block, region, rec)
                        continue
                    warnings.append(
                        f"loop {rec.function}:{rec.block}[{rec.index}] "
                        f"rejected: {rec.reason}"
                    )
                    i += 1

        return PassResult(
            data=self.program,
            changes=changes,
            message=(f"vectorized {changes}/{len(self._report)} loop(s), "
                     f"width={self.width}"),
            warnings=warnings,
        )

    # ── Region discovery ────────────────────────────────────────────────

    def _collect_region(self, block: BasicBlock,
                        for_index: int) -> Optional[_LoopRegion]:
        instrs = block.instructions
        depth = 0
        for j in range(for_index, len(instrs)):
            op = instrs[j].opcode
            if op is OpCode.FOR:
                depth += 1
            elif op is OpCode.ENDFOR:
                depth -= 1
                if depth == 0:
                    return _LoopRegion(
                        for_index=for_index,
                        end_index=j,
                        for_instr=instrs[for_index],
                        body=list(instrs[for_index + 1:j]),
                    )
        return None

    def _skip_rewritten(self, block: BasicBlock, region: _LoopRegion,
                        rec: LoopVectorizationRecord) -> int:
        """Index of the instruction after the rewritten region.

        The rewritten region contains one top-level ``FOR`` (the strip
        loop) plus, when there is a remainder, a second one inserted
        directly after the strip ``ENDFOR``.  The original ``end_index``
        is stale after the rewrite, so the position is found by scanning
        the block for the matching top-level ``ENDFOR`` (review F1).
        """
        target = 2 if rec.remainder else 1
        depth = 0
        closed = 0
        for j in range(region.for_index, len(block.instructions)):
            op = block.instructions[j].opcode
            if op is OpCode.FOR:
                depth += 1
            elif op is OpCode.ENDFOR:
                depth -= 1
                if depth == 0:
                    closed += 1
                    if closed == target:
                        return j + 1
        # Defensive: the rewrite always produces the expected loops.
        return region.end_index + 1

    # ── Decision tree (design doc 2.2.1 C1–C7) ──────────────────────────

    def _try_vectorize(self, func: Function, block: BasicBlock,
                       region: _LoopRegion) -> LoopVectorizationRecord:
        for_instr = region.for_instr
        start = for_instr.attrs.get("start")
        end = for_instr.attrs.get("end")
        step = for_instr.attrs.get("step")
        rec = LoopVectorizationRecord(
            function=func.name,
            block=block.name,
            index=region.for_index,
            status="rejected",
            start=start if _is_int(start) else 0,
            end=end if _is_int(end) else 0,
            width=self.width,
        )

        # C2/C3: constant bounds, start == 0, step == 1, n >= W.
        if not _is_int(start) or not _is_int(end) or not _is_int(step):
            return self._reject(rec, REASON_NON_CONSTANT_BOUNDS)
        if step != 1:
            return self._reject(rec, REASON_UNSUPPORTED_STEP)
        if start != 0:
            return self._reject(rec, REASON_UNSUPPORTED_START)
        n = end - start
        if n < self.width:
            return self._reject(rec, REASON_TRIP_TOO_SMALL)

        # C1: no nested control flow inside the region.
        if any(instr.opcode in _CONTROL_FLOW_OPS for instr in region.body):
            return self._reject(rec, REASON_NESTED_CONTROL_FLOW)

        defs: dict[str, Instruction] = {
            instr.dest.name: instr
            for instr in region.body if instr.dest is not None
        }
        outer_names = self._outer_names(func, defs)
        iv_name = for_instr.dest.name if for_instr.dest else ""

        # C4: at least one canonical element address chain.
        mem_refs = [
            self._mem_ref(instr, idx, defs, outer_names, iv_name)
            for idx, instr in enumerate(region.body)
            if instr.opcode in (OpCode.LOAD, OpCode.STORE)
        ]
        if not any(ref.canonical for ref in mem_refs):
            return self._reject(rec, REASON_NO_ELEMENT_PATTERN)

        # C7: aliasing stores (shallow, conservative base analysis).
        reason = self._alias_reason(mem_refs, n)
        if reason is not None:
            return self._reject(rec, reason)

        # C6: the induction variable may only feed canonical address MULs.
        reason = self._iv_use_reason(region.body, iv_name)
        if reason is not None:
            return self._reject(rec, reason)

        # C6 (live-out): values defined inside the region (including the
        # original induction variable) must not be used outside it — the
        # rewrite replaces or drops those definitions (review F2).
        reason = self._region_escape_reason(func, region, defs, iv_name)
        if reason is not None:
            return self._reject(rec, reason)

        plan = _Plan(
            n=n, width=self.width, elem_bytes=self.elem_bytes,
            strips=n // self.width, remainder=n % self.width,
            iv_name=iv_name, outer_names=frozenset(outer_names),
        )
        self._build_plan(plan, region, defs, mem_refs)

        # C5: classify every remaining instruction.
        reason = self._classify(plan, region, defs)
        if reason is not None:
            return self._reject(rec, reason)

        original_iv = region.for_instr.dest
        vector_ops, new_end_index = self._rewrite(block, region, plan)
        self._clone_remainder(block, region, plan, original_iv,
                              new_end_index)

        rec.status = "vectorized"
        rec.reason = ""
        rec.strips = plan.strips
        rec.remainder = plan.remainder
        rec.vector_ops = vector_ops
        return rec

    @staticmethod
    def _reject(rec: LoopVectorizationRecord,
                reason: str) -> LoopVectorizationRecord:
        rec.status = "rejected"
        rec.reason = reason
        return rec

    def _outer_names(self, func: Function,
                     defs: dict[str, Instruction]) -> set[str]:
        outer: set[str] = set()
        for block in func.blocks:
            for instr in block.instructions:
                if instr.dest is not None and instr.dest.name not in defs:
                    outer.add(instr.dest.name)
        for param in func.params:
            outer.add(param.name)
        return outer

    def _mem_ref(self, instr: Instruction, index: int,
                 defs: dict[str, Instruction], outer_names: set[str],
                 iv_name: str) -> _MemRef:
        addr = instr.operands[0]
        base: Optional[Value] = None
        offset: Optional[Value] = None

        if addr.name in defs:
            addr_def = defs[addr.name]
            if addr_def.opcode is OpCode.ADD and len(addr_def.operands) == 2:
                x, y = addr_def.operands
                x_local = x.name in defs
                y_local = y.name in defs
                if x_local and not y_local:
                    offset, base = x, y
                elif y_local and not x_local:
                    offset, base = y, x
        else:
            base = addr

        canonical = False
        if offset is not None and offset.name in defs:
            offset_def = defs[offset.name]
            if (offset_def.opcode is OpCode.MUL
                    and len(offset_def.operands) == 2):
                a, b = offset_def.operands
                other: Optional[Value] = None
                if a.name == iv_name:
                    other = b
                elif b.name == iv_name:
                    other = a
                if (other is not None and other.is_constant
                        and other.const_value is not None
                        and _is_int(other.const_value)
                        and other.const_value == self.elem_bytes):
                    canonical = True

        if canonical and base is not None:
            if not (base.is_constant or base.name in outer_names):
                canonical = False

        return _MemRef(instr=instr, index=index, base=base,
                       offset=offset, canonical=canonical)

    def _alias_reason(self, mem_refs: list[_MemRef],
                      n: int) -> Optional[str]:
        """C7: conservative aliasing check (review F3).

        Two references on the *same* base are only safe when the accesses
        are the canonical element chain with at most one store and one
        load and both use the same lane offset (in-place patterns).  Two
        references on *different* base names are only safe when both
        bases are constant absolute addresses with provably disjoint
        ``[base, base + n*elem_bytes)`` byte ranges; anything else is
        rejected because a shallow name-based analysis cannot rule out
        overlap (e.g. ``src = sub(out, 4)``).
        """
        by_base: dict[str, list[_MemRef]] = {}
        for ref in mem_refs:
            if ref.base is None:
                # Addresses that are not a canonical element chain are
                # rejected later by classification; they never receive a
                # vector form, so they cannot be reordered here.
                continue
            by_base.setdefault(ref.base.name, []).append(ref)

        for base_name, refs in by_base.items():
            store_refs = [ref for ref in refs if not ref.is_load]
            if len(store_refs) > 1:
                return REASON_ALIASING_STORE
            if not store_refs:
                continue
            load_refs = [ref for ref in refs if ref.is_load]
            if not load_refs:
                continue
            if len(load_refs) != 1:
                return REASON_ALIASING_STORE
            if load_refs[0].lane_key != store_refs[0].lane_key:
                return REASON_ALIASING_STORE

        store_bases = [
            name for name, refs in by_base.items()
            if any(not ref.is_load for ref in refs)
        ]
        for store_name in store_bases:
            store_base = by_base[store_name][0].base
            for other_name, other_refs in by_base.items():
                if other_name == store_name:
                    continue
                if not self._provably_disjoint(
                        store_base, other_refs[0].base, n):
                    return REASON_ALIASING_STORE
        return None

    def _provably_disjoint(self, lhs: Optional[Value],
                           rhs: Optional[Value], n: int) -> bool:
        """True when two constant bases cannot overlap over *n* elements."""
        if lhs is None or rhs is None:
            return False
        if not all(value.is_constant and value.const_value is not None
                   and _is_int(value.const_value) for value in (lhs, rhs)):
            return False
        span = n * self.elem_bytes
        assert lhs.const_value is not None and rhs.const_value is not None
        return abs(int(lhs.const_value) - int(rhs.const_value)) >= span

    def _region_escape_reason(self, func: Function, region: _LoopRegion,
                              defs: dict[str, Instruction],
                              iv_name: str) -> Optional[str]:
        """C6 live-out: no region-local definition may escape the region.

        ``_rewrite`` replaces or drops every definition inside
        ``region.body`` and redefines the ``FOR`` destination as the strip
        index, so a use of any of those values outside the region would
        become a dangling SSA reference (review F2).  Checking the whole
        function also covers the remainder-less case where the original
        induction variable is never redefined.
        """
        local_names = set(defs)
        if iv_name:
            local_names.add(iv_name)
        if not local_names:
            return None
        body_ids = {id(instr) for instr in region.body}
        for block in func.blocks:
            for instr in block.instructions:
                if id(instr) in body_ids:
                    continue
                for op in instr.operands:
                    if op.name in local_names:
                        return REASON_REGION_VALUE_ESCAPES
        return None

    def _iv_use_reason(self, body: list[Instruction],
                       iv_name: str) -> Optional[str]:
        """C6: the induction variable may only feed canonical address MULs."""
        canonical_mul_ids: set[int] = set()
        for instr in body:
            if instr.opcode is not OpCode.MUL or len(instr.operands) != 2:
                continue
            a, b = instr.operands
            if a.name != iv_name and b.name != iv_name:
                continue
            other = b if a.name == iv_name else a
            if (other.is_constant and other.const_value is not None
                    and _is_int(other.const_value)
                    and other.const_value == self.elem_bytes):
                canonical_mul_ids.add(id(instr))
        for instr in body:
            if id(instr) in canonical_mul_ids:
                continue
            for op in instr.operands:
                if op.name == iv_name:
                    return REASON_NON_ELEMENTWISE_IV
        return None

    def _build_plan(self, plan: _Plan, region: _LoopRegion,
                    defs: dict[str, Instruction],
                    mem_refs: list[_MemRef]) -> None:
        for ref in mem_refs:
            if not ref.canonical:
                continue
            plan.mem_indices.add(ref.index)
            offset_def = defs[ref.offset.name]
            plan.offset_mul_indices.add(_index_of(region.body, offset_def))
            addr_name = ref.instr.operands[0].name
            if addr_name in defs:
                addr_def = defs[addr_name]
                addr_index = _index_of(region.body, addr_def)
                plan.addr_add_bases[addr_index] = ref.base

    def _classify(self, plan: _Plan, region: _LoopRegion,
                  defs: dict[str, Instruction]) -> Optional[str]:
        # Values that will have a vector form after the rewrite: element
        # results and canonical LOAD results (which become VLOADs).
        vector_values: set[str] = set()
        for idx in plan.mem_indices:
            dest = region.body[idx].dest
            if dest is not None:
                vector_values.add(dest.name)
        for idx, instr in enumerate(region.body):
            if (idx in plan.offset_mul_indices
                    or idx in plan.mem_indices
                    or idx in plan.addr_add_bases):
                continue
            if instr.opcode is OpCode.LOAD_CONST:
                plan.const_indices.add(idx)
                continue
            if instr.opcode not in _ELEMENT_VECTOR_OPS:
                return REASON_UNSUPPORTED_OP
            if instr.dest is None:
                return REASON_UNSUPPORTED_OP
            for op in instr.operands:
                if op.is_constant:
                    continue
                if op.name in plan.outer_names:
                    continue
                if op.name in vector_values:
                    continue
                return REASON_UNSUPPORTED_OP
            plan.elem_indices.add(idx)
            vector_values.add(instr.dest.name)

        # Dead element/constant values must not sneak into the region.
        reachable: set[str] = set()
        stack: list[Value] = []
        for idx in plan.mem_indices:
            for op in region.body[idx].operands:
                stack.append(op)
        while stack:
            value = stack.pop()
            if value.name in reachable:
                continue
            reachable.add(value.name)
            defining = defs.get(value.name)
            if defining is None:
                continue
            for op in defining.operands:
                stack.append(op)
        for idx in plan.elem_indices | plan.const_indices:
            dest = region.body[idx].dest
            if dest is None or dest.name not in reachable:
                return REASON_UNSUPPORTED_OP
        return None

    # ── Rewrite (strip-mining) ──────────────────────────────────────────

    def _rewrite(self, block: BasicBlock, region: _LoopRegion,
                 plan: _Plan) -> tuple[int, int]:
        """Rewrite the region in place.

        Returns ``(vector_ops, new_end_index)`` where ``new_end_index`` is
        the position of the strip ``ENDFOR`` *after* the body replacement
        (``region.end_index`` is stale once ``new_body`` changes length;
        review F1).
        """
        old_iv = region.for_instr.dest
        iv_dtype = old_iv.dtype if old_iv is not None else DataType.INT32
        strip_iv = Value(name=self._fresh("v"), dtype=iv_dtype)
        c_scale_value = plan.width * plan.elem_bytes
        c_scale = Value(name=self._fresh("c"), dtype=DataType.INT32,
                        is_constant=True, const_value=c_scale_value)

        new_body: list[Instruction] = []
        val_map: dict[str, Value] = {}
        base_addrs: dict[str, Value] = {}
        bcast_cache: dict[str, Value] = {}
        boff: Optional[Value] = None
        vector_ops = 0

        for idx, instr in enumerate(region.body):
            if idx in plan.offset_mul_indices:
                if boff is None:
                    new_body.append(Instruction(
                        OpCode.LOAD_CONST, c_scale, [],
                        attrs={"value": c_scale_value}))
                    boff = Value(name=self._fresh("boff"),
                                 dtype=DataType.INT32)
                    new_body.append(Instruction(
                        OpCode.MUL, boff, [strip_iv, c_scale]))
                if instr.dest is not None:
                    val_map[instr.dest.name] = boff
                continue

            if idx in plan.addr_add_bases:
                base = plan.addr_add_bases[idx]
                assert base is not None and boff is not None
                key = base.name
                if key not in base_addrs:
                    pa = Value(name=self._fresh("pa"), dtype=base.dtype)
                    new_body.append(Instruction(OpCode.ADD, pa, [base, boff]))
                    base_addrs[key] = pa
                if instr.dest is not None:
                    val_map[instr.dest.name] = base_addrs[key]
                continue

            if instr.opcode is OpCode.LOAD:
                addr = val_map[instr.operands[0].name]
                dest = instr.dest
                assert dest is not None
                vec = Value(name=self._fresh("v"), dtype=dest.dtype,
                            shape=(plan.width,))
                new_body.append(Instruction(
                    OpCode.VLOAD, vec, [addr],
                    attrs={"width": plan.width,
                           "elem_bytes": plan.elem_bytes, "align": 4}))
                val_map[dest.name] = vec
                vector_ops += 1
                continue

            if instr.opcode is OpCode.STORE:
                addr = val_map[instr.operands[0].name]
                src = self._element_value(
                    instr.operands[1], plan, val_map, bcast_cache, new_body)
                new_body.append(Instruction(
                    OpCode.VSTORE, None, [addr, src],
                    attrs={"width": plan.width,
                           "elem_bytes": plan.elem_bytes, "align": 4}))
                vector_ops += 1
                continue

            if idx in plan.elem_indices:
                vec_op = _ELEMENT_VECTOR_OPS[instr.opcode]
                vec_dest = instr.dest
                assert vec_dest is not None
                operands = [
                    self._element_value(op, plan, val_map,
                                        bcast_cache, new_body)
                    for op in instr.operands
                ]
                out = Value(name=self._fresh("v"), dtype=vec_dest.dtype,
                            shape=(plan.width,))
                new_body.append(Instruction(
                    vec_op, out, operands, attrs={"width": plan.width}))
                val_map[vec_dest.name] = out
                vector_ops += 1
                continue

            if instr.opcode is OpCode.LOAD_CONST:
                # Constants remain referenceable as values; they are
                # materialized on demand by the backend (LI / VBCAST).
                continue

            # Defensive: classification should have rejected this.
            raise ValueError(
                f"vectorizer internal error: unexpected op "
                f"{instr.opcode.value} in classified region")

        region.for_instr.dest = strip_iv
        region.for_instr.attrs = {
            "start": 0,
            "end": plan.strips,
            "step": 1,
            "vector_width": plan.width,
            "elem_bytes": plan.elem_bytes,
            "orig_trip": plan.n,
        }
        block.instructions[region.for_index + 1:region.end_index] = new_body
        new_end_index = region.for_index + 1 + len(new_body)
        return vector_ops, new_end_index

    def _element_value(self, value: Value, plan: _Plan,
                       val_map: dict[str, Value],
                       bcast_cache: dict[str, Value],
                       out: list[Instruction]) -> Value:
        """Map a scalar/element operand to its vector form."""
        if value.name in val_map:
            return val_map[value.name]
        if value.is_constant or value.name in plan.outer_names:
            if value.name not in bcast_cache:
                vec = Value(name=self._fresh("v"), dtype=value.dtype,
                            shape=(plan.width,))
                out.append(Instruction(
                    OpCode.VBCAST, vec, [value],
                    attrs={"width": plan.width}))
                bcast_cache[value.name] = vec
            return bcast_cache[value.name]
        raise ValueError(
            f"vectorizer internal error: no vector form for '{value.name}'")

    def _clone_remainder(self, block: BasicBlock, region: _LoopRegion,
                         plan: _Plan, original_iv: Optional[Value],
                         new_end_index: int) -> int:
        """Insert the scalar remainder loop after the rewritten region.

        ``new_end_index`` is the strip ``ENDFOR`` position returned by
        ``_rewrite``; inserting at any other index either nests the
        remainder inside the strip loop or leaves it after ``return``
        (review F1).
        """
        if plan.remainder == 0:
            return 0
        new_for = Instruction(
            opcode=OpCode.FOR, dest=original_iv,
            attrs={"start": plan.strips * plan.width,
                   "end": plan.n, "step": 1})
        cloned = self._clone_instrs(region.body)
        endfor = Instruction(opcode=OpCode.ENDFOR)
        insert_at = new_end_index + 1
        block.instructions[insert_at:insert_at] = [new_for, *cloned, endfor]
        return 1

    def _clone_instrs(self, instrs: list[Instruction]) -> list[Instruction]:
        rename: dict[str, Value] = {}
        for instr in instrs:
            if instr.dest is None:
                continue
            old = instr.dest
            rename[old.name] = Value(
                name=self._fresh_clone_name(old.name),
                dtype=old.dtype,
                is_constant=old.is_constant,
                const_value=old.const_value,
                shape=old.shape,
            )
        cloned: list[Instruction] = []
        for instr in instrs:
            operands = [rename.get(op.name, op) for op in instr.operands]
            dest = rename.get(instr.dest.name) if instr.dest else None
            cloned.append(Instruction(
                opcode=instr.opcode,
                dest=dest,
                operands=operands,
                attrs=dict(instr.attrs),
                target=instr.target,
            ))
        return cloned

    # ── Fresh names ─────────────────────────────────────────────────────

    def _fresh(self, prefix: str = "v") -> str:
        while True:
            self._counter += 1
            name = f"{prefix}_{self._counter}"
            if name not in self._used_names:
                self._used_names.add(name)
                return name

    def _fresh_clone_name(self, base: str) -> str:
        name = f"{base}__rem"
        while name in self._used_names:
            name += "_"
        self._used_names.add(name)
        return name


def _index_of(instrs: list[Instruction], target: Instruction) -> int:
    for i, instr in enumerate(instrs):
        if instr is target:
            return i
    raise ValueError("instruction not found in region body")
