"""Read-only verification of shared IR structure, definitions and signatures.

CFG construction and loop pairing belong to ``cfg_builder``. Diagnostics use
original IR positions, including when a structured loop splits a basic block.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, replace
from typing import Optional

from scratchv.analysis.adapters import IRCFGAdapter
from scratchv.analysis.cfg import build_cfg, compute_dominators
from scratchv.analysis.cfg_validation import verify_cfg
from scratchv.analysis.ir_diagnostics import IRContextLine, capture_ir_context, ir_fix_hint
from scratchv.ir.types import DataType, OpCode, Program


class ErrorLevel(enum.Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class VerificationError:
    level: ErrorLevel
    message: str
    function_name: Optional[str] = None
    block_name: Optional[str] = None
    instruction_index: Optional[int] = None
    value_name: Optional[str] = None
    rule: Optional[str] = None
    stage: Optional[str] = None
    context: tuple[IRContextLine, ...] = ()
    fix_hint: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"[{self.level.value.upper()}][{self.rule}]"]
        for label, value in (("stage", self.stage), ("function", self.function_name),
                             ("block", self.block_name), ("instruction", self.instruction_index),
                             ("value", self.value_name)):
            if value is not None:
                parts.append(f"{label}={value}")
        return " ".join(parts) + ": " + self.message


@dataclass(frozen=True)
class OpcodeSpec:
    min_operands: int
    max_operands: Optional[int]
    has_dest: bool
    family: Optional[str] = None


# T: identical numeric types; F: identical floating point types.
OPCODE_SPECS = {}
for _ops, _count, _family in (
    ((OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV, OpCode.MATMUL, OpCode.DOT), 2, "T"),
    ((OpCode.NEG, OpCode.RELU, OpCode.MAXPOOL, OpCode.RESHAPE, OpCode.TRANSPOSE), 1, "T"),
    ((OpCode.EXP, OpCode.GELU, OpCode.SIGMOID, OpCode.SOFTMAX), 1, "F"),
    ((OpCode.CONV, OpCode.GEMM), 3, "T"),
):
    for _op in _ops:
        OPCODE_SPECS[_op] = OpcodeSpec(_count, _count, True, _family)
OPCODE_SPECS.update({
    OpCode.CONCAT: OpcodeSpec(1, None, True, "T"),
    OpCode.LOAD_CONST: OpcodeSpec(0, 0, True),
    OpCode.ALLOCA: OpcodeSpec(0, 0, True),
    OpCode.LOAD: OpcodeSpec(1, 1, True),
    OpCode.STORE: OpcodeSpec(2, 2, False),
    OpCode.BR: OpcodeSpec(0, 0, False),
    OpCode.BR_IF: OpcodeSpec(1, 2, False),
    OpCode.RETURN: OpcodeSpec(0, 1, False),
    OpCode.FOR: OpcodeSpec(0, 0, True),
    OpCode.ENDFOR: OpcodeSpec(0, 0, False),
})
_RULES = {name: i for i, name in enumerate((
    "def-before-use", "label-existence", "block-termination", "type-consistency",
    "control-flow-integrity", "ssa-validity", "entry-existence",
))}
_FLOATS = (DataType.FLOAT32, DataType.FLOAT64)
_INTS = (DataType.INT32, DataType.INT64)


def _valid_constant(value, dtype):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if dtype in _INTS:
        bits = 32 if dtype is DataType.INT32 else 64
        return isinstance(value, int) and -(1 << (bits - 1)) <= value < (1 << (bits - 1))
    return dtype in _FLOATS


def _same_constant(left, right):
    # Avoid conversion of arbitrary precision integer literals to floats.
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    if left == right:
        if left == 0:
            return math.copysign(1, left) == math.copysign(1, right)
        return True
    return False


@dataclass(frozen=True)
class _Definition:
    value: object
    location: tuple
    instruction: object = None


class IRVerifier:
    """Collect independent errors without repairing or executing the Program."""

    def __init__(self, program: Program):
        self.program = program

    def verify(self, *, stage: Optional[str] = None) -> list[VerificationError]:
        self._stage = stage
        self._issues = {}
        globals_ = {}
        ambiguous = set()
        for vi, value in enumerate(self.program.global_values):
            loc = (-1, -1, -1, "global", vi)
            self._check_value(value, loc)
            self._define(globals_, ambiguous, value, loc)
        for fi, func in enumerate(self.program.functions):
            self._verify_function(fi, func, globals_, ambiguous)
        return [issue for _, issue in sorted(self._issues.values(), key=lambda pair: pair[0])]

    def _emit(self, rule, reason, message, loc, value=None, level=ErrorLevel.ERROR):
        fi, bi, ii, kind, item = loc
        func = self.program.functions[fi] if fi >= 0 else None
        block = func.blocks[bi] if func is not None and bi >= 0 else None
        issue = VerificationError(level, message, func.name if func else None,
                                  block.name if block else None, ii if ii >= 0 else None,
                                  value, rule, self._stage,
                                  capture_ir_context(self.program, loc, value),
                                  ir_fix_hint(rule, reason))
        key = (loc, rule, value, reason)
        order = (fi, bi, ii, _RULES[rule], value or "", message, kind, item)
        self._issues.setdefault(key, (order, issue))

    def _check_value(self, value, loc):
        if not isinstance(value.dtype, DataType):
            self._emit("type-consistency", "invalid-dtype", f"invalid dtype: {value.dtype!r}", loc, value.name)
        elif value.is_constant and not _valid_constant(value.const_value, value.dtype):
            self._emit("type-consistency", "invalid-constant", "constant metadata is incompatible with dtype", loc, value.name)

    def _define(self, definitions, ambiguous, value, loc, instruction=None):
        if value.name in definitions:
            first_loc = definitions[value.name].location
            first = self._describe_location(first_loc)
            self._emit("ssa-validity", "duplicate-definition",
                       f"duplicate definition; first definition at {first}", loc, value.name)
            ambiguous.add(value.name)
        else:
            definitions[value.name] = _Definition(value, loc, instruction)

    def _describe_location(self, loc):
        fi, bi, ii, kind, item = loc
        if fi < 0:
            return f"global #{item}"
        func = self.program.functions[fi]
        if bi < 0:
            return f"function={func.name} {kind} #{item}"
        return (f"function={func.name} block={func.blocks[bi].name} "
                f"(block #{bi}) instruction={ii}")

    def _verify_function(self, fi, func, globals_, global_ambiguous):
        definitions = dict(globals_)
        ambiguous = set(global_ambiguous)
        function_loc = (fi, -1, -1, "function", 0)
        for pi, value in enumerate(func.params):
            loc = (fi, -1, -1, "param", pi)
            self._check_value(value, loc)
            self._define(definitions, ambiguous, value, loc)
        for ri, value in enumerate(func.returns):
            self._check_value(value, (fi, -1, -1, "return", ri))
        for li, value in enumerate(func.locals):
            self._check_value(value, (fi, -1, -1, "local", li))
        if len(func.returns) > 1:
            self._emit("type-consistency", "return-signature", "at most one return type is supported", function_loc)
        if not func.blocks:
            self._emit("entry-existence", "no-entry", "function has no basic blocks", function_loc)
            return
        for bi, block in enumerate(func.blocks):
            for ii, inst in enumerate(block.instructions):
                if inst.dest is not None:
                    loc = (fi, bi, ii, "instruction", 0)
                    self._check_value(inst.dest, loc)
                    self._define(definitions, ambiguous, inst.dest, loc, inst)

        cfg, instruction_nodes = self._analyze_cfg(fi, func)
        reachable = cfg.reachable_nodes if cfg is not None else set()
        # Analyze a reachable view; unreachable predecessors must not alter
        # dominance. The upstream graph and original IR remain untouched.
        live_cfg = replace(cfg, nodes={n: cfg.nodes[n] for n in reachable},
                           edges=[e for e in cfg.edges if e.source in reachable and e.target in reachable]) if cfg is not None else None
        dominators = compute_dominators(live_cfg) if live_cfg is not None else {}
        if cfg is not None:
            reachable_blocks = {bi for (bi, ii), node in instruction_nodes.items() if node in reachable}
            for bi, block in enumerate(func.blocks):
                if bi not in reachable_blocks:
                    self._emit("control-flow-integrity", "unreachable", "unreachable basic block",
                               (fi, bi, -1, "block", 0), level=ErrorLevel.WARNING)
        for bi, block in enumerate(func.blocks):
            for ii, inst in enumerate(block.instructions):
                loc = (fi, bi, ii, "instruction", 0)
                for operand in inst.operands:
                    self._check_value(operand, loc)
                    if operand.name in ambiguous:
                        continue
                    definition = definitions.get(operand.name)
                    if definition is None:
                        if not operand.is_constant:
                            self._emit("def-before-use", "undefined-value", "value has no definition", loc, operand.name)
                        continue
                    if (isinstance(operand.dtype, DataType) and isinstance(definition.value.dtype, DataType)
                            and operand.dtype != definition.value.dtype):
                        self._emit("type-consistency", "reference-type", "reference dtype differs from definition", loc, operand.name)
                    if definition.instruction is None:
                        continue
                    def_pos = definition.location[1:3]
                    use_node = instruction_nodes.get((bi, ii))
                    def_node = instruction_nodes.get(def_pos)
                    if (use_node is not None and use_node == def_node) or (cfg is None and def_pos[0] == bi):
                        if def_pos[1] >= ii:
                            self._emit("def-before-use", "use-before-definition", "value used before its definition", loc, operand.name)
                    elif cfg is not None and use_node in reachable and def_node not in dominators[use_node]:
                        self._emit("def-before-use", "not-dominating", "definition does not dominate this use", loc, operand.name)
                self._check_signature(inst, func, definitions, ambiguous, loc)

    def _analyze_cfg(self, fi, func):
        """Validate original IR, then consume the upstream graph without changing it.

        The adapter normalizes loops and splits at terminators, so original-IR
        invariants must be checked before that information is lost. No edges or
        synthetic control-flow nodes are constructed here.
        """
        invalid = False
        names = {}
        stack = []
        loop_ends = {}
        terminators = {OpCode.BR, OpCode.BR_IF, OpCode.RETURN}

        def report(rule, reason, bi, ii=-1):
            nonlocal invalid
            invalid = True
            self._emit(rule, reason, reason,
                       (fi, bi, ii, "instruction" if ii >= 0 else "block", 0))

        for bi, block in enumerate(func.blocks):
            if not isinstance(block.name, str) or not block.name.strip():
                report("label-existence", "block name must be nonempty", bi)
            if block.name in names:
                report("label-existence", "duplicate block name", bi)
            names[block.name] = bi
        for bi, block in enumerate(func.blocks):
            if block.phi_nodes:
                report("control-flow-integrity", "unsupported representation: phi_nodes", bi)
            if not block.instructions:
                report("block-termination", "empty block has no terminator", bi)
            terminated = False
            for ii, inst in enumerate(block.instructions):
                if terminated:
                    report("control-flow-integrity", "instruction after explicit terminator", bi, ii)
                    terminated = False
                if inst.opcode in terminators:
                    terminated = True
                if inst.opcode == OpCode.LABEL:
                    report("control-flow-integrity", "unsupported representation: LABEL", bi, ii)
                if inst.opcode == OpCode.FOR:
                    stack.append((bi, ii))
                elif inst.opcode == OpCode.ENDFOR:
                    if stack:
                        start_bi, start_ii = stack.pop()
                        loop = func.blocks[start_bi].instructions[start_ii]
                        loop_ends[id(loop.dest)] = (bi, ii)
                    else:
                        report("control-flow-integrity", "ENDFOR without FOR", bi, ii)
                    if bi == len(func.blocks) - 1 and ii == len(block.instructions) - 1:
                        report("control-flow-integrity", "loop has no exit continuation", bi, ii)
                if inst.opcode in (OpCode.BR, OpCode.BR_IF):
                    targets = inst.target.split(",") if isinstance(inst.target, str) else []
                    targets = [t.strip() for t in targets]
                    count = 1 if inst.opcode == OpCode.BR else 2
                    if len(targets) != count or not all(targets):
                        report("label-existence" if count == 1 else "control-flow-integrity",
                               "branch requires %d nonempty target(s)" % count, bi, ii)
                    else:
                        for target in dict.fromkeys(targets):
                            if target not in names:
                                report("label-existence", "branch target '%s' does not exist" % target, bi, ii)
                            elif target == func.blocks[0].name:
                                report("control-flow-integrity", "explicit branch to function entry", bi, ii)
            if block.instructions and block.instructions[-1].opcode not in terminators | {OpCode.FOR, OpCode.ENDFOR}:
                following = func.blocks[bi + 1].instructions if bi + 1 < len(func.blocks) else []
                if not following or following[0].opcode != OpCode.ENDFOR:
                    report("block-termination", "block does not end with a terminator", bi, len(block.instructions) - 1)
        for bi, ii in stack:
            report("control-flow-integrity", "FOR without ENDFOR", bi, ii)
        if invalid:
            return None, {}

        # Signature checks below report malformed FOR attributes precisely;
        # do not pass them to the adapter's int() conversion first.
        for block in func.blocks:
            for inst in block.instructions:
                if inst.opcode == OpCode.FOR and (inst.dest is None or any(
                        not _valid_constant(inst.attrs.get(key), DataType.INT32)
                        for key in ("start", "end", "step"))):
                    return None, {}
        try:
            cfg = build_cfg(IRCFGAdapter(func))
        except (ValueError, TypeError, OverflowError) as exc:
            self._emit("control-flow-integrity", "cfg-build", f"cannot build unified CFG: {exc}",
                       (fi, -1, -1, "function", 0))
            return None, {}
        # Ordinary instructions retain object identity through the adapter.
        # FOR's definition is the generated LOAD_CONST using the original dest.
        originals = {id(inst): (bi, ii) for bi, block in enumerate(func.blocks)
                     for ii, inst in enumerate(block.instructions)}
        loops = {id(inst.dest): (bi, ii) for bi, block in enumerate(func.blocks)
                 for ii, inst in enumerate(block.instructions) if inst.opcode == OpCode.FOR}
        instruction_nodes = {}
        for name, node in cfg.nodes.items():
            for inst in node.instructions:
                loc = originals.get(id(inst))
                if loc is None and inst.opcode == OpCode.LOAD_CONST:
                    loc = loops.get(id(inst.dest))
                elif loc is None and inst.opcode == OpCode.ADD:
                    loc = loop_ends.get(id(inst.dest))
                if loc is not None:
                    instruction_nodes[loc] = name
        for diagnostic in verify_cfg(cfg):
            loc = next((pos for pos, node in instruction_nodes.items() if node == diagnostic.block), None)
            position = (fi, *loc, "instruction", 0) if loc else (fi, -1, -1, "function", 0)
            self._emit("control-flow-integrity", diagnostic.code, diagnostic.message, position,
                       level=ErrorLevel.ERROR if diagnostic.severity == "error" else ErrorLevel.WARNING)
            invalid |= diagnostic.severity == "error"
        return (None if invalid else cfg), instruction_nodes

    def _check_signature(self, inst, func, definitions, ambiguous, loc):
        def error(reason, message):
            self._emit("type-consistency", reason, message, loc)

        if inst.opcode == OpCode.LABEL:
            return  # Unsupported control-flow representation is R5.
        spec = OPCODE_SPECS.get(inst.opcode)
        if spec is None:
            error("unknown-opcode", f"unknown opcode: {inst.opcode!r}")
            return
        count = len(inst.operands)
        valid_count = count >= spec.min_operands and (spec.max_operands is None or count <= spec.max_operands)
        if not valid_count:
            error("operand-count", f"expected {spec.min_operands}..{spec.max_operands} operands, got {count}")
        if (inst.dest is not None) != spec.has_dest:
            error("destination", "instruction requires a result" if spec.has_dest else "instruction must not have a result")
        values = inst.operands + ([inst.dest] if inst.dest is not None else [])
        if spec.family and values and all(isinstance(v.dtype, DataType) for v in values):
            types = [v.dtype for v in values]
            if len(set(types)) != 1 or (spec.family == "F" and types[0] not in _FLOATS):
                error("signature-types", "incompatible types: " + ", ".join(t.value for t in types))
        if inst.opcode == OpCode.LOAD_CONST and inst.dest is not None and isinstance(inst.dest.dtype, DataType):
            value = inst.attrs.get("value")
            if not _valid_constant(value, inst.dest.dtype):
                error("load-constant", "LOAD_CONST value is missing or incompatible with result dtype")
            elif (inst.dest.is_constant and _valid_constant(inst.dest.const_value, inst.dest.dtype)
                  and not _same_constant(value, inst.dest.const_value)):
                error("constant-disagreement", "LOAD_CONST value differs from result constant metadata")
        if inst.opcode == OpCode.FOR:
            if inst.dest is not None and isinstance(inst.dest.dtype, DataType) and inst.dest.dtype != DataType.INT32:
                error("loop-dtype", "FOR result must be i32")
            for key in ("start", "end", "step"):
                if not _valid_constant(inst.attrs.get(key), DataType.INT32):
                    error("loop-" + key, f"FOR {key} must be an i32 integer attribute")
            step = inst.attrs.get("step")
            if isinstance(step, int) and not isinstance(step, bool) and step <= 0:
                error("loop-step-direction", "FOR step must be positive")
        if not valid_count:
            return
        if inst.opcode == OpCode.BR_IF:
            if count == 1:
                if isinstance(inst.operands[0].dtype, DataType) and inst.operands[0].dtype not in _INTS:
                    error("condition-type", "BR_IF condition must be i32 or i64")
                if "cmp_op" in inst.attrs:
                    error("condition-comparison", "single-condition BR_IF must not carry cmp_op")
            else:
                if inst.attrs.get("cmp_op") not in ("==", "!=", "<", "<=", ">", ">="):
                    error("comparison", "BR_IF requires a supported cmp_op")
                if all(isinstance(v.dtype, DataType) for v in inst.operands) and inst.operands[0].dtype != inst.operands[1].dtype:
                    error("comparison-types", "BR_IF comparison types differ")
        if inst.opcode == OpCode.RETURN and len(func.returns) == 1:
            if count != 1:
                error("return-count", "declared return requires one value")
            elif (isinstance(inst.operands[0].dtype, DataType) and isinstance(func.returns[0].dtype, DataType)
                  and inst.operands[0].dtype != func.returns[0].dtype):
                error("return-type", "returned dtype differs from declared return")
        if inst.opcode in (OpCode.LOAD, OpCode.STORE):
            address = inst.operands[0]
            definition = definitions.get(address.name) if address.name not in ambiguous else None
            if definition is not None and definition.instruction is not None and definition.instruction.opcode == OpCode.ALLOCA:
                value = inst.dest if inst.opcode == OpCode.LOAD else inst.operands[1]
                if (value is not None and isinstance(value.dtype, DataType)
                        and isinstance(definition.value.dtype, DataType) and value.dtype != definition.value.dtype):
                    error("storage-type", "value dtype differs from ALLOCA element dtype")


def verify_ir(program: Program, *, stage: Optional[str] = None) -> tuple[bool, list[VerificationError]]:
    issues = IRVerifier(program).verify(stage=stage)
    return not any(issue.level is ErrorLevel.ERROR for issue in issues), issues
