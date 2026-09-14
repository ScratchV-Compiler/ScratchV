"""IR-level loop unrolling for ScratchV (Topic 10).

Unrolls ``FOR`` / ``ENDFOR`` loops by duplicating the loop body:

- ``FULL``             - trip count ``N <= full_threshold``; the loop markers
                         are removed and the body is replicated ``N`` times.
- ``PARTIAL_EXACT``    - the largest divisor ``U`` of ``N`` (``2 <= U <=
                         max_factor``); the ``FOR`` loop is rewritten into a
                         group counter (``[start=0][end=q]``) with ``U``
                         replicated copies per group.
- ``PARTIAL_EPILOGUE`` - same as partial, but a remainder loop handles the
                         ``r = N % U`` leftover iterations.

The pass is conservative: every failed precondition records a reason in
``stats["skipped"]`` and leaves the IR untouched.  It never redefines an
existing value name -- the last copy of each body value reuses the original
name (rotating naming), which keeps the SSA-style single-assignment contract
checked by :class:`scratchv.analysis.ir_verifier.IRVerifier`.

The backend lowering always emits ``ADDI iv, iv, 1`` for ``ENDFOR`` and
ignores ``step``, so partial unrolling rewrites the induction variable into a
group counter and materialises the element index (``start + k + U * group``)
for each copy instead of touching ``ENDFOR``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

from scratchv.ir.types import (
    DataType,
    Function,
    Instruction,
    OpCode,
    Program,
    Value,
)

# Fixed skip-reason keys (missing keys are treated as 0 by consumers).
_SKIP_KEYS = (
    "bad_attrs", "step_not_one", "trip_lt_2", "already_unrolled",
    "body_has_branches", "nested_loop", "body_too_large", "multi_def",
    "carried_value", "no_factor", "growth_limit", "unprofitable",
    "unpaired", "internal_error",
)

_BARRIER_OPS = (OpCode.LABEL, OpCode.BR, OpCode.BR_IF)
_LOOP_OPS = (OpCode.FOR, OpCode.ENDFOR)


@dataclass
class UnrollPlan:
    """Chosen unrolling strategy for a single loop."""

    mode: str            # "full" | "partial_exact" | "partial_epilogue"
    U: int               # unroll factor (number of copies)
    q: int               # N // U
    r: int               # N % U
    estimated_added: int = 0
    dynamic_saving: int = 0

    @property
    def partial(self) -> bool:
        return self.mode != "full"


class LoopUnroll:
    """Unroll ``FOR`` / ``ENDFOR`` loops in an IR program (in place)."""

    _MAX_ITERATIONS = 32

    def __init__(
        self,
        program: Program,
        max_factor: int = 8,
        full_threshold: int = 8,
        body_limit: int = 64,
        max_growth: int = 512,
        epilogue: bool = False,
    ) -> None:
        self.program = program
        self._max_factor = max_factor
        self._full_threshold = full_threshold
        self._body_limit = body_limit
        self._max_growth = max_growth
        self._epilogue = epilogue
        self._stats = self._empty_stats()
        self._name_cache: dict[str, set[str]] = {}
        self._seen_loops: set[int] = set()
        # FOR instructions already transformed by this run (avoids counting
        # rescan artifacts such as a freshly created epilogue loop).
        self._handled_loops: set[int] = set()
        # id(FOR) -> (FOR object, skip reasons already counted this run).
        # Holding the object prevents id reuse from hiding later skips; the
        # reason set keeps repeated rescans from inflating the counters.
        self._skip_recorded: dict[int, tuple[Instruction, set[str]]] = {}
        self._last_scan_unpaired = False
        # Rolling "current copy" name map shared by ``_copy_region`` calls.
        self._copy_cur: dict[str, str] = {}
        self._epilogue_copy = False

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Statistics of the last :meth:`run`, in the fixed schema."""
        return self._stats

    def run(self) -> int:
        """Run unrolling on all functions; return the number of loops unrolled.

        The pass is repeatable: every loop written by the pass (main
        partially unrolled loop and remainder loop alike) carries an
        ``attrs["unrolled"]`` marker and is skipped on subsequent runs.
        """
        self._stats = self._empty_stats()
        self._name_cache = {}
        self._seen_loops = set()
        self._handled_loops = set()
        self._skip_recorded = {}
        self._stats["instructions_before"] = self._count_instructions()

        for func in self.program.functions:
            snapshot = self._snapshot(func)
            stats_snapshot = copy.deepcopy(self._stats)
            try:
                self._process_function(func)
            except Exception:
                self._restore(func, snapshot)
                self._stats = stats_snapshot
                self._stats["skipped"]["internal_error"] += 1

        before = self._stats["instructions_before"]
        after = self._count_instructions()
        self._stats["instructions_after"] = after
        self._stats["instructions_added"] = after - before
        return self._stats["loops_unrolled"]

    # ── Scanning ────────────────────────────────────────────────────────

    def _find_pairs(self, instrs: list) -> list[tuple[int, int]]:
        """Pair ``FOR`` / ``ENDFOR`` indices using a stack.

        Returns pairs ordered by closing ``ENDFOR``, i.e. innermost first.
        An unbalanced block is reported via ``skipped["unpaired"]`` and no
        pair is returned, so the whole block is left untouched.
        """
        stack: list[int] = []
        pairs: list[tuple[int, int]] = []
        imbalanced = False

        for i, ins in enumerate(instrs):
            if ins.opcode == OpCode.FOR:
                stack.append(i)
            elif ins.opcode == OpCode.ENDFOR:
                if not stack:
                    imbalanced = True
                    continue
                pairs.append((stack.pop(), i))

        if stack:
            imbalanced = True

        if imbalanced:
            self._stats["skipped"]["unpaired"] += 1
            self._last_scan_unpaired = True
            return []
        return pairs

    def _process_function(self, func: Function) -> None:
        for block in func.blocks:
            for _ in range(self._MAX_ITERATIONS):
                self._last_scan_unpaired = False
                pairs = self._find_pairs(block.instructions)
                if self._last_scan_unpaired:
                    break

                applied = False
                for for_idx, endfor_idx in pairs:
                    plan = self._select_plan(
                        block.instructions, for_idx, endfor_idx)
                    if plan is None:
                        continue
                    self._apply_unroll(
                        func, block, for_idx, endfor_idx, plan)
                    applied = True
                    break
                if not applied:
                    break

    # ── Plan selection ──────────────────────────────────────────────────

    def _select_plan(
        self, instrs: list, for_idx: int, endfor_idx: int,
    ) -> Optional[UnrollPlan]:
        """Choose a plan, or ``None`` with a recorded skip reason."""
        for_ins = instrs[for_idx]
        if id(for_ins) in self._handled_loops:
            return None
        key = id(for_ins)
        if key not in self._seen_loops:
            self._seen_loops.add(key)
            self._stats["loops_seen"] += 1

        attrs = for_ins.attrs
        start = attrs.get("start")
        end = attrs.get("end")
        step = attrs.get("step")
        if not (type(start) is int and type(end) is int and type(step) is int):
            self._skip_loop(for_ins, "bad_attrs")
            return None
        if step != 1:
            self._skip_loop(for_ins, "step_not_one")
            return None

        n_trip = end - start
        if n_trip < 2:
            self._skip_loop(for_ins, "trip_lt_2")
            return None
        if "unrolled" in attrs:
            self._skip_loop(for_ins, "already_unrolled")
            return None

        region = instrs[for_idx + 1:endfor_idx]
        if any(ins.opcode in _BARRIER_OPS for ins in region):
            self._skip_loop(for_ins, "body_has_branches")
            return None
        if any(ins.opcode in _LOOP_OPS for ins in region):
            self._skip_loop(for_ins, "nested_loop")
            return None
        if len(region) > self._body_limit:
            self._skip_loop(for_ins, "body_too_large")
            return None

        def_counts: dict[str, int] = {}
        for ins in region:
            if ins.dest is not None:
                def_counts[ins.dest.name] = (
                    def_counts.get(ins.dest.name, 0) + 1)
        iv = for_ins.dest
        iv_name = iv.name if iv is not None else ""
        if any(count > 1 for count in def_counts.values()):
            self._skip_loop(for_ins, "multi_def")
            return None
        if iv_name and any(
            ins.dest is not None and ins.dest.name == iv_name
            for ins in region
        ):
            self._skip_loop(for_ins, "multi_def")
            return None

        # Loop-carried (self/forward-referenced) body values cannot be
        # expressed by the single renamed epilogue copy: its uses stay bound
        # to the pre-loop names, so the second remainder iteration reads a
        # stale value.  Reject that shape instead of emitting wrong IR.
        has_forward_ref = self._has_forward_reference(
            region, set(def_counts), iv_name)

        body_size = len(region)
        iv_used = bool(iv_name) and any(
            op.name == iv_name for ins in region for op in ins.operands)
        extra_start = 1 if start != 0 else 0

        if n_trip <= self._full_threshold:
            candidates: list[tuple[str, int]] = [("full", n_trip)]
        else:
            limit = min(self._max_factor, n_trip - 1)
            divisors = [u for u in range(limit, 1, -1)
                        if n_trip % u == 0]
            if divisors:
                candidates = [("partial_exact", u) for u in divisors]
            elif self._epilogue and limit >= 2:
                candidates = [("partial_epilogue", limit)]
            else:
                self._skip_loop(for_ins, "no_factor")
                return None

        last_reason = "no_factor"
        for mode, factor in candidates:
            q, r = divmod(n_trip, factor)
            if mode == "partial_epilogue" and r > 1 and has_forward_ref:
                last_reason = "carried_value"
                continue
            if mode == "full":
                m_cost = factor if iv_used else 0
                s_cost = 0
                epilogue_cost = 0
            else:
                m_cost = (factor + extra_start) if iv_used else 0
                s_cost = (2 + extra_start) if iv_used else 0
                epilogue_cost = (body_size + 2) if (
                    mode == "partial_epilogue" and r > 0) else 0

            estimated_added = (
                (factor - 1) * body_size + m_cost + s_cost
                + epilogue_cost + 1)
            dynamic_saving = q * (3 * factor - 3 - m_cost) - s_cost
            if mode == "partial_epilogue" and r > 0:
                dynamic_saving -= 1

            if estimated_added > self._max_growth:
                last_reason = "growth_limit"
                continue
            if mode != "full" and dynamic_saving < 2:
                last_reason = "unprofitable"
                continue

            return UnrollPlan(
                mode=mode, U=factor, q=q, r=r,
                estimated_added=estimated_added,
                dynamic_saving=dynamic_saving,
            )

        self._skip_loop(for_ins, last_reason)
        return None

    # ── IR rewriting ────────────────────────────────────────────────────

    def _apply_unroll(
        self, func: Function, block, for_idx: int, endfor_idx: int,
        plan: UnrollPlan,
    ) -> None:
        instrs = block.instructions
        for_ins = instrs[for_idx]
        endfor_ins = instrs[endfor_idx]
        self._handled_loops.add(id(for_ins))
        iv = for_ins.dest
        iv_name = iv.name if iv is not None else ""
        iv_dtype = iv.dtype if iv is not None else DataType.INT32
        start = for_ins.attrs["start"]
        end = for_ins.attrs["end"]
        n_trip = end - start
        region = list(instrs[for_idx + 1:endfor_idx])
        body_defs = {
            ins.dest.name for ins in region if ins.dest is not None}
        orig_def_values = {
            ins.dest.name: ins.dest for ins in region
            if ins.dest is not None}

        iv_in_body = bool(iv_name) and any(
            op.name == iv_name for ins in region for op in ins.operands)
        iv_after = self._name_used_after(
            func, block, for_idx, endfor_idx, iv_name)

        new_body: list[Instruction] = []
        post_rename: dict[str, str] = {}

        if plan.mode == "full":
            self._copy_cur = {}
            self._epilogue_copy = False
            for k in range(plan.U):
                bind = None
                if iv_in_body:
                    if k == plan.U - 1:
                        bind, bind_ins = self._make_const(
                            func, start + k, iv_dtype, name=iv_name)
                    else:
                        bind, bind_ins = self._make_const(
                            func, start + k, iv_dtype,
                            base=f"{iv_name}__u{k}")
                    new_body.append(bind_ins)
                new_body.extend(self._copy_region(
                    func, region, body_defs, iv_name, bind,
                    k == plan.U - 1, k))
        else:
            for_ins.attrs.clear()
            for_ins.attrs.update({
                "start": 0, "end": plan.q, "step": 1, "unrolled": plan.U})

            setup: list[Instruction] = []
            u_tmp: Optional[Value] = None
            one_tmp: Optional[Value] = None
            start_tmp: Optional[Value] = None
            if iv_in_body:
                u_tmp, ins_u = self._make_const(
                    func, plan.U, iv_dtype, base="c_u")
                one_tmp, ins_one = self._make_const(
                    func, 1, iv_dtype, base="c_one")
                setup.extend([ins_u, ins_one])
                if start != 0:
                    start_tmp, ins_start = self._make_const(
                        func, start, iv_dtype, base="c_start")
                    setup.append(ins_start)

            self._copy_cur = {}
            self._epilogue_copy = False
            prev_bind: Optional[Value] = None
            for k in range(plan.U):
                bind = None
                if iv_in_body:
                    bind_name = self._fresh_name(func, f"{iv_name}__u{k}")
                    if k == 0 and start == 0:
                        bind, bind_ins = self._make_binary(
                            bind_name, OpCode.MUL, iv, u_tmp, iv_dtype)
                        new_body.append(bind_ins)
                    elif k == 0:
                        tmp_name = self._fresh_name(
                            func, f"{iv_name}__g0")
                        tmp, tmp_ins = self._make_binary(
                            tmp_name, OpCode.MUL, iv, u_tmp, iv_dtype)
                        bind, bind_ins = self._make_binary(
                            bind_name, OpCode.ADD, tmp, start_tmp, iv_dtype)
                        new_body.extend([tmp_ins, bind_ins])
                    else:
                        bind, bind_ins = self._make_binary(
                            bind_name, OpCode.ADD, prev_bind, one_tmp,
                            iv_dtype)
                        new_body.append(bind_ins)
                    prev_bind = bind
                new_body.extend(self._copy_region(
                    func, region, body_defs, iv_name, bind,
                    k == plan.U - 1, k))

            epilogue: list[Instruction] = []
            if plan.mode == "partial_epilogue" and plan.r > 0:
                ep_iv = Value(
                    name=self._fresh_name(
                        func, f"{iv_name}_ep" if iv_name else "iv_ep"),
                    dtype=iv_dtype, is_constant=False)
                ep_for = Instruction(
                    OpCode.FOR, ep_iv, [],
                    {"start": start + plan.q * plan.U, "end": end,
                     "step": 1, "unrolled": 1})
                self._handled_loops.add(id(ep_for))
                self._copy_cur = {}
                self._epilogue_copy = True
                ep_body = self._copy_region(
                    func, region, body_defs, iv_name, ep_iv, False, 0)
                self._epilogue_copy = False
                epilogue = [ep_for] + ep_body + [Instruction(OpCode.ENDFOR)]
                post_rename = {
                    name: self._copy_cur[name] for name in body_defs
                    if name in self._copy_cur}

            new_body = setup + [for_ins] + new_body + [endfor_ins] + epilogue

        instrs[for_idx:endfor_idx + 1] = new_body
        insert_at = for_idx + len(new_body)

        redirects: list[tuple[str, Value]] = []
        if iv_after:
            iv_final = Value(
                name=self._fresh_name(
                    func, f"{iv_name}_final" if iv_name else "iv_final"),
                dtype=iv_dtype, is_constant=False)
            instrs.insert(insert_at, Instruction(
                OpCode.LOAD_CONST, iv_final, [], {"value": start + n_trip}))
            insert_at += 1
            redirects.append((iv_name, iv_final))
        for old_name, new_name in post_rename.items():
            redirects.append((
                old_name,
                self._value_like(orig_def_values[old_name], new_name)))

        if redirects:
            self._redirect_uses(func, block, insert_at, redirects)

        self._stats["loops_unrolled"] += 1
        if plan.mode == "full":
            self._stats["full_unrolls"] += 1
        else:
            self._stats["partial_unrolls"] += 1
            if plan.mode == "partial_epilogue" and plan.r > 0:
                self._stats["partial_epilogues"] += 1

    def _copy_region(
        self, func: Function, region: list, body_defs: set[str],
        iv_name: str, bind_k: Optional[Value], last: bool, k: int,
    ) -> list[Instruction]:
        """Clone *region*, rotating body-definition names.

        References resolve through ``self._copy_cur``: a use maps to the most
        recent definition (earlier in the same copy, else the previous copy,
        else the original name for the first copy).  The last copy reuses the
        original names so that loop-carried values remain visible after the
        loop.
        """
        cur = self._copy_cur
        suffix = "__ep" if self._epilogue_copy else f"__u{k}"
        out: list[Instruction] = []

        for ins in region:
            operands = []
            for op in ins.operands:
                if iv_name and op.name == iv_name:
                    operands.append(bind_k if bind_k is not None else op)
                elif op.name in body_defs:
                    resolved = cur.get(op.name, op.name)
                    if resolved == op.name:
                        operands.append(op)
                    else:
                        operands.append(self._value_like(op, resolved))
                else:
                    operands.append(op)

            dest = ins.dest
            if dest is not None and dest.name in body_defs:
                if last:
                    new_name = dest.name
                else:
                    new_name = self._fresh_name(
                        func, f"{dest.name}{suffix}")
                cur[dest.name] = new_name
                dest = self._value_like(dest, new_name)

            out.append(Instruction(
                ins.opcode, dest, operands, dict(ins.attrs), ins.target))
        return out

    # ── Helpers ─────────────────────────────────────────────────────────

    def _fresh_name(self, func: Function, base: str) -> str:
        """Return a function-unique value name derived from *base*."""
        used = self._name_cache.get(func.name)
        if used is None:
            used = self._collect_names(func)
            self._name_cache[func.name] = used
        name = base
        counter = 0
        while name in used:
            counter += 1
            name = f"{base}_{counter}"
        used.add(name)
        return name

    def _collect_names(self, func: Function) -> set[str]:
        names: set[str] = set()
        for param in func.params:
            names.add(param.name)
        for local in func.locals:
            names.add(local.name)
        for block in func.blocks:
            for ins in block.instructions:
                if ins.dest is not None:
                    names.add(ins.dest.name)
                for op in ins.operands:
                    names.add(op.name)
        return names

    @staticmethod
    def _value_like(value: Value, name: str) -> Value:
        """Create a fresh, non-constant value carrying *value*'s type."""
        return Value(
            name=name, dtype=value.dtype, is_constant=False,
            const_value=None, shape=value.shape)

    def _make_const(
        self, func: Function, value: int, dtype: DataType,
        name: Optional[str] = None, base: Optional[str] = None,
    ) -> tuple[Value, Instruction]:
        """Create a non-constant ``LOAD_CONST`` value/instruction pair."""
        if name is None:
            name = self._fresh_name(func, base or "c")
        val = Value(name=name, dtype=dtype, is_constant=False)
        return val, Instruction(OpCode.LOAD_CONST, val, [], {"value": value})

    @staticmethod
    def _make_binary(
        name: str, opcode: OpCode, lhs: Value, rhs: Value, dtype: DataType,
    ) -> tuple[Value, Instruction]:
        """Create a non-constant binary-operation value/instruction pair."""
        dest = Value(name=name, dtype=dtype, is_constant=False)
        return dest, Instruction(opcode, dest, [lhs, rhs])

    @staticmethod
    def _has_forward_reference(
        region: list, body_def_names: set[str], iv_name: str,
    ) -> bool:
        """Whether the region reads a body-defined value before defining it.

        This covers both forward references (``t = add(u, one)`` before
        ``u = ...``) and self references (``acc = add(acc, t)``), i.e. the
        loop-carried values a single epilogue copy cannot express.
        """
        defined: set[str] = set()
        for ins in region:
            for op in ins.operands:
                if (op.name != iv_name and op.name in body_def_names
                        and op.name not in defined):
                    return True
            if ins.dest is not None:
                defined.add(ins.dest.name)
        return False

    def _name_used_after(
        self, func: Function, block, for_idx: int, endfor_idx: int,
        name: str,
    ) -> bool:
        """Whether *name* is used outside the loop.

        Scans later instructions in the same block plus every other block.
        """
        if not name:
            return False
        instrs = block.instructions
        for i in range(endfor_idx + 1, len(instrs)):
            if any(op.name == name for op in instrs[i].operands):
                return True
        for other in func.blocks:
            if other is block:
                continue
            for ins in other.instructions:
                if any(op.name == name for op in ins.operands):
                    return True
        return False

    def _redirect_uses(
        self, func: Function, block, start_idx: int,
        redirects: list[tuple[str, Value]],
    ) -> None:
        """Replace operand names after the loop according to *redirects*."""
        lookup = dict(redirects)
        for i in range(start_idx, len(block.instructions)):
            self._redirect_instruction(block.instructions[i], lookup)
        for other in func.blocks:
            if other is block:
                continue
            for ins in other.instructions:
                self._redirect_instruction(ins, lookup)

    @staticmethod
    def _redirect_instruction(
        instr: Instruction, lookup: dict[str, Value],
    ) -> None:
        for j, op in enumerate(instr.operands):
            replacement = lookup.get(op.name)
            if replacement is not None:
                instr.operands[j] = replacement

    def _snapshot(self, func: Function) -> list:
        """Save the instruction list, attrs and operands of every block.

        Operands are captured because ``_redirect_uses`` rewrites them in
        place; without them an exception mid-rewrite would leave a
        half-rewritten function behind.
        """
        return [
            (block, list(block.instructions),
             [(ins, dict(ins.attrs), list(ins.operands))
              for ins in block.instructions])
            for block in func.blocks
        ]

    @staticmethod
    def _restore(func: Function, snapshot: list) -> None:
        for block, instrs, saved in snapshot:
            block.instructions = instrs
            for ins, attrs, operands in saved:
                ins.attrs = attrs
                ins.operands = operands

    def _count_instructions(self) -> int:
        return sum(
            len(block.instructions)
            for func in self.program.functions
            for block in func.blocks
        )

    def _skip(self, reason: str) -> None:
        self._stats["skipped"][reason] += 1

    def _skip_loop(self, for_ins: Instruction, reason: str) -> None:
        """Record a skip reason for *for_ins* at most once per ``run()``.

        ``_process_function`` rescans the block after every applied loop, so
        the same untouched loop is examined repeatedly; counting it each
        time would inflate the user-visible statistics.
        """
        key = id(for_ins)
        entry = self._skip_recorded.get(key)
        if entry is None or entry[0] is not for_ins:
            entry = (for_ins, set())
            self._skip_recorded[key] = entry
        if reason in entry[1]:
            return
        entry[1].add(reason)
        self._skip(reason)

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "loops_seen": 0,
            "loops_unrolled": 0,
            "full_unrolls": 0,
            "partial_unrolls": 0,
            "partial_epilogues": 0,
            "instructions_before": 0,
            "instructions_after": 0,
            "instructions_added": 0,
            "skipped": {key: 0 for key in _SKIP_KEYS},
        }
