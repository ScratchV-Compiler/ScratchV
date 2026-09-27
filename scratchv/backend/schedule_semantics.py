"""Typed, conservative instruction effects for the assembly scheduler.

Parsing happens here, once. Scheduling never interprets operand strings.
Unknown forms are boundaries, not instructions with guessed effects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ._asm_parser import canonical_reg, is_integer_reg

_FP_ALIASES = {f"ft{i}": f"f{i}" for i in range(8)}
_FP_ALIASES.update({"fs0": "f8", "fs1": "f9"})
_FP_ALIASES.update({f"fa{i}": f"f{i + 10}" for i in range(8)})
_FP_ALIASES.update({f"fs{i}": f"f{i + 16}" for i in range(2, 12)})
_FP_ALIASES.update({f"ft{i}": f"f{i + 20}" for i in range(8, 12)})

INTEGER_BINARY = frozenset(
    ["add", "sub", "sll", "srl", "sra", "xor", "or", "and", "slt", "sltu"]
)
MULTIPLY = frozenset(["mul", "mulh", "mulhu", "mulhsu"])
DIVIDE = frozenset(["div", "divu", "rem", "remu"])
INTEGER_IMMEDIATE = frozenset(
    ["addi", "xori", "ori", "andi", "slti", "sltiu", "slli", "srli", "srai"]
)
LOADS = frozenset(["lb", "lbu", "lh", "lhu", "lw", "flw", "fld"])
STORES = frozenset(["sb", "sh", "sw", "fsw", "fsd"])
BRANCHES = frozenset(
    ["beq", "bne", "blt", "bge", "bltu", "bgeu", "bgt", "ble", "bgtu", "bleu"]
)
ZERO_BRANCHES = frozenset(["beqz", "bnez", "bltz", "bgez", "blez", "bgtz"])
_ROUNDING = frozenset(["rne", "rtz", "rdn", "rup", "rmm", "dyn"])
_INTEGER = re.compile(r"[+-]?(?:0[xX][0-9a-fA-F]+|[0-9]+)\Z")
_LABEL = re.compile(r"(?:[A-Za-z_.$][\w.$]*|[0-9]+[fb])\Z")


def register_name(value: str) -> str | None:
    """Resolve integer and FP ABI aliases, rejecting non-register operands."""
    value = value.strip().lower()
    if is_integer_reg(value):
        return canonical_reg(value)
    value = _FP_ALIASES.get(value, value)
    if re.fullmatch(r"f(?:[0-9]|[12][0-9]|3[01])", value):
        return value
    return None


def integer(value: str) -> int | None:
    """Read a literal, without accepting symbols or assembler expressions."""
    if not _INTEGER.fullmatch(value):
        return None
    return int(value, 16 if "x" in value.lower() else 10)


@dataclass(frozen=True)
class MemoryAddress:
    base: str
    offset: int


def memory_address(value: str) -> MemoryAddress | None:
    """Accept GAS offset(base) and the backend's historical base(offset)."""
    match = re.fullmatch(r"\s*([^()]*)\(\s*([^()]*)\s*\)\s*", value)
    if not match:
        return None
    left, right = (part.strip() for part in match.groups())
    base = register_name(right)
    offset = integer(left or "0")
    if base is None or not base.startswith("x") or offset is None:
        base = register_name(left)
        offset = integer(right or "0")
    if base is None or not base.startswith("x") or offset is None:
        return None
    if not -2048 <= offset <= 2047:
        return None
    return MemoryAddress(base, offset)


@dataclass(frozen=True)
class InstructionEffects:
    defines: frozenset[str] = frozenset()
    uses: frozenset[str] = frozenset()
    memory: str = "none"
    address: MemoryAddress | None = None
    control: str = "none"
    target: str | None = None
    fp_flags: bool = False
    barrier_reason: str = ""


def describe(opcode: str, operands: tuple[str, ...]) -> InstructionEffects:
    """Describe an exact supported form, or explain why it is pinned."""
    op = opcode.lower()
    args = operands

    def boundary(reason: str, control: str = "none", target: str | None = None) -> InstructionEffects:
        return InstructionEffects(control=control, target=target, barrier_reason=reason)

    def regs(
        pattern: str, *, flags: bool = False, rounding: bool = False
    ) -> InstructionEffects:
        actual = args
        # Floating rounding operands are optional; never guess extra operands.
        if rounding and len(actual) == len(pattern) + 1 and actual[-1] in _ROUNDING:
            actual = actual[:-1]
        if len(actual) != len(pattern):
            return boundary("unsupported operand form")
        names = [register_name(arg) for arg in actual]
        if any(
            name is None or not name.startswith(kind.lower())
            for name, kind in zip(names, pattern)
        ):
            return boundary("invalid register operand")
        return InstructionEffects(
            defines=frozenset(names[:1]) - {"x0"},
            uses=frozenset(names[1:]) - {"x0"},
            fp_flags=flags,
        )

    if op in INTEGER_BINARY | MULTIPLY | DIVIDE:
        return regs("xxx")
    if op in INTEGER_IMMEDIATE:
        if len(args) != 3:
            return boundary("unsupported operand form")
        imm = integer(args[2])
        limit = (0, 31) if op in {"slli", "srli", "srai"} else (-2048, 2047)
        if imm is None or not limit[0] <= imm <= limit[1]:
            return boundary("unsupported immediate")
        dst, src = register_name(args[0]), register_name(args[1])
        if not dst or not src or not dst.startswith("x") or not src.startswith("x"):
            return boundary("invalid register operand")
        return InstructionEffects(frozenset({dst}) - {"x0"}, frozenset({src}) - {"x0"})
    if op in {"mv", "neg", "not", "seqz", "snez", "sltz", "sgtz"}:
        return regs("xx")
    if op == "nop" and not args:
        return InstructionEffects()
    if op in {"li", "lui"}:
        if len(args) != 2:
            return boundary("unsupported operand form")
        dst, imm = register_name(args[0]), integer(args[1])
        low, high = (-2048, 2047) if op == "li" else (0, (1 << 20) - 1)
        if not dst or not dst.startswith("x") or imm is None or not low <= imm <= high:
            return boundary("unknown or multi-instruction immediate expansion")
        return InstructionEffects(defines=frozenset({dst}) - {"x0"})
    if op in LOADS | STORES:
        if len(args) != 2:
            return boundary("unsupported memory form")
        value, addr = args
        if op in STORES and "(" in value:
            addr, value = value, addr
        address = memory_address(addr)
        reg = register_name(value)
        kind = "f" if op.startswith("f") else "x"
        if address is None or not reg or not reg.startswith(kind):
            return boundary("unsupported memory form")
        load = op in LOADS
        return InstructionEffects(
            defines=(frozenset({reg}) - {"x0"}) if load else frozenset(),
            uses=(frozenset({address.base}) | (frozenset() if load else {reg}))
            - {"x0"},
            memory="read" if load else "write",
            address=address,
        )
    if op in BRANCHES | ZERO_BRANCHES:
        count = 3 if op in BRANCHES else 2
        if len(args) != count or not _LABEL.fullmatch(args[-1]):
            return boundary("unsupported branch target", "branch")
        names = [register_name(arg) for arg in args[:-1]]
        if any(name is None or not name.startswith("x") for name in names):
            return boundary("invalid branch register", "branch")
        return InstructionEffects(
            uses=frozenset(names) - {"x0"}, control="branch", target=args[-1]
        )
    if op == "j" and len(args) == 1 and _LABEL.fullmatch(args[0]):
        return InstructionEffects(control="jump", target=args[0])
    if op == "ret" and not args:
        return InstructionEffects(uses=frozenset({"x1"}), control="return")
    if op == "jr" and len(args) == 1:
        src = register_name(args[0])
        if src and src.startswith("x"):
            return InstructionEffects(uses=frozenset({src}) - {"x0"}, control="jump")
    if op == "jal":
        if (
            len(args) == 2
            and register_name(args[0]) == "x0"
            and _LABEL.fullmatch(args[1])
        ):
            return InstructionEffects(control="jump", target=args[1])
        if (
            (len(args) == 1 or (len(args) == 2 and register_name(args[0]) is not None
                               and register_name(args[0]).startswith("x")))
            and _LABEL.fullmatch(args[-1])
        ):
            return boundary("call or unsupported jal", "call", args[-1])
        return boundary("call or unsupported jal", "call")
    if op == "jalr":
        # Only the no-link form can be a modeled terminator. Calls are barriers.
        if len(args) == 3 and register_name(args[0]) == "x0":
            src, offset = register_name(args[1]), integer(args[2])
            if (
                src
                and src.startswith("x")
                and offset is not None
                and -2048 <= offset <= 2047
            ):
                return InstructionEffects(
                    uses=frozenset({src}) - {"x0"}, control="jump"
                )
        if len(args) == 2 and register_name(args[0]) == "x0":
            src = register_name(args[1])
            if src and src.startswith("x"):
                return InstructionEffects(
                    uses=frozenset({src}) - {"x0"}, control="jump"
                )
            addr = memory_address(args[1])
            if addr:
                return InstructionEffects(
                    uses=frozenset({addr.base}) - {"x0"}, control="jump"
                )
        return boundary("call or unsupported jalr", "call")
    if op in {"call", "tail"}:
        target = args[0] if len(args) == 1 and _LABEL.fullmatch(args[0]) else None
        return boundary("call boundary", "call", target)
    if op.startswith("f"):
        parts = op.split(".")
        stem = parts[0]
        if len(parts) == 2 and parts[1] in {"s", "d"}:
            if stem in {"fadd", "fsub", "fmul", "fdiv", "fmin", "fmax"}:
                return regs("fff", flags=True, rounding=stem not in {"fmin", "fmax"})
            if stem == "fsqrt":
                return regs("ff", flags=True, rounding=True)
            if stem in {"fmadd", "fmsub", "fnmadd", "fnmsub"}:
                return regs("ffff", flags=True, rounding=True)
            if stem in {"feq", "flt", "fle"}:
                return regs("xff", flags=True)
            if stem == "fclass":
                return regs("xf")
            if stem in {"fsgnj", "fsgnjn", "fsgnjx"}:
                return regs("fff")
            if stem in {"fmv", "fabs", "fneg"}:
                return regs("ff")
        if op in {"fmv.x.w", "fmv.x.s"}:
            return regs("xf")
        if op in {"fmv.w.x", "fmv.s.x"}:
            return regs("fx")
        if len(parts) == 3 and stem == "fcvt":
            dst, src = parts[1:]
            if (
                dst in {"s", "d", "w", "wu"}
                and src in {"s", "d", "w", "wu"}
                and (dst in {"s", "d"} or src in {"s", "d"})
                and dst != src
            ):
                return regs(
                    ("f" if dst in {"s", "d"} else "x")
                    + ("f" if src in {"s", "d"} else "x"),
                    flags=True,
                    rounding=True,
                )
    return boundary("unknown opcode or unsupported form")


@dataclass(frozen=True)
class SchedInst:
    """One immutable instruction, including its exact original source line.

    Legacy def/use constructor arguments are accepted, but recomputed from the
    opcode description: caller hints must not override instruction semantics.
    Registers in the resulting object use canonical xN/fN names.
    """

    id: int
    opcode: str
    operands: tuple[str, ...] = ()
    defines: frozenset[str] = frozenset()
    uses: frozenset[str] = frozenset()
    raw_line: str = ""
    region: int = 0
    target: str | None = field(default=None, kw_only=True)
    effects: InstructionEffects = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "opcode", self.opcode.lower())
        object.__setattr__(self, "operands", tuple(self.operands))
        effects = describe(self.opcode, self.operands)
        if self.target is not None and effects.target not in {None, self.target}:
            raise ValueError("Conflicting instruction targets")
        if self.target is None:
            object.__setattr__(self, "target", effects.target)
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "defines", effects.defines)
        object.__setattr__(self, "uses", effects.uses)

    @property
    def terminator(self) -> bool:
        return self.effects.control in {"branch", "jump", "return"}

    @property
    def movable(self) -> bool:
        return not self.effects.barrier_reason and not self.terminator


class ScheduleError(ValueError):
    """An invalid schedule or an unsupported use of the low-level API."""
