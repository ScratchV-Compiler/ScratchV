"""Assembly-level Peephole Optimizer for RISC-V.

Applies peephole optimization rules to RISC-V assembly text using
sliding-window pattern matching with register wildcards.

Parsing is delegated to the shared backend parser
(``scratchv.backend._asm_parser``) so beautifier / peephole / const-merge
share one AsmLine representation.

Usage::

    from scratchv.backend.asm_peephole import AsmPeepholeOptimizer
    opt = AsmPeepholeOptimizer()
    optimized_text, changes = opt.optimize(asm_text)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

from scratchv.backend._asm_parser import (
    ParsedAsmLine,
    lines_to_asm as _shared_lines_to_asm,
    parse_asm as _shared_parse_asm,
    parse_line as _shared_parse_line,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

# Backward-compatible alias: peephole historically used ``AsmLine``.
AsmLine = ParsedAsmLine


@dataclass
class PeepholeRule:
    """A peephole optimization rule.

    Parameters
    ----------
    name:
        Human-readable rule name.
    pattern:
        List of opcode strings (lowercase). Use ``*`` as a wildcard that
        matches any opcode. Operands are matched positionally with support
        for register wildcards (see ``register_constraints``).
    replacement:
        List of opcode strings for replacement. Use ``{0}``, ``{1}`` etc.
        to reference registers captured from the pattern.
    register_constraints:
        Optional list of tuples ``(dst_instr, src_instr, src_op)`` requiring
        ``window[dst_instr].operands[0] == window[src_instr].operands[src_op]``.
        Indices are **0-based positions inside the match window** (not global
        line numbers). Example: ``(0, 1, 1)`` means "rd of window[0] must
        equal operand 1 of window[1]".
    """
    name: str
    pattern: list[str]
    replacement: list[str]
    register_constraints: list[tuple[int, int, int]] = field(
        default_factory=list
    )

    def __repr__(self) -> str:
        return f"PeepholeRule({self.name!r})"


# ---------------------------------------------------------------------------
# Parsing — thin wrappers over shared ``_asm_parser``
# ---------------------------------------------------------------------------

def _parse_line(line: str, lineno: int = 0) -> AsmLine:
    """Parse a single assembly line (shared parser)."""
    return _shared_parse_line(line, lineno=lineno)


def _parse_asm(asm_text: str) -> list[AsmLine]:
    """Parse full assembly text (shared parser).

    Leading/trailing blank lines from ``strip()`` are dropped so behaviour
    matches the historical peephole parser used by existing tests.
    """
    return _shared_parse_asm(asm_text.strip())


def _lines_to_asm(lines: list[AsmLine]) -> str:
    """Convert parsed lines back to assembly text (shared formatter)."""
    return _shared_lines_to_asm(lines)


def _count_opcodes(lines: list[AsmLine]) -> int:
    """Count real instruction opcodes (labels and directives excluded)."""
    return sum(
        1 for al in lines
        if al.opcode is not None and not al.is_directive
    )


# ---------------------------------------------------------------------------
# Register alias helpers (x0 == zero)
# ---------------------------------------------------------------------------

_ZERO_ALIASES = frozenset({"x0", "zero"})


def _canon_reg(name: str) -> str:
    """Normalize register aliases so ``x0`` and ``zero`` compare equal."""
    if name in _ZERO_ALIASES:
        return "x0"
    return name


def _regs_equal(a: str, b: str) -> bool:
    """Equality with zero-register alias awareness."""
    return _canon_reg(a) == _canon_reg(b)


def _is_zero_reg(name: str) -> bool:
    """Return True if *name* is the hardwired-zero register."""
    return name in _ZERO_ALIASES


# ---------------------------------------------------------------------------
# Immediate helpers (RISC-V I-type signed 12-bit)
# ---------------------------------------------------------------------------

_SIMM12_MIN = -2048
_SIMM12_MAX = 2047


def _fits_simm12(value: int) -> bool:
    """Return True if *value* fits in a signed 12-bit immediate."""
    return _SIMM12_MIN <= value <= _SIMM12_MAX


def _parse_imm(text: str) -> Optional[int]:
    """Parse an immediate operand; return None if not a plain integer."""
    try:
        return int(text, 0)  # accepts 10, 0x10, 0b10
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Default peephole rules
# ---------------------------------------------------------------------------

def _default_rules() -> list[PeepholeRule]:
    """Return the default peephole optimization rules."""
    return [
        # Rule 1: addi x, x, a; addi x, x, b -> addi x, x, a+b
        # (only when a+b fits signed 12-bit immediate)
        PeepholeRule(
            name="addi+addi fusion",
            pattern=["addi", "addi"],
            replacement=["addi {rd} {rs1} {imm_sum}"],
            register_constraints=[(0, 1, 0), (0, 1, 1)],
        ),

        # NOTE: former "mv x,y; mv y,x -> delete" was unsound (not a true
        # swap / no-op under RISC-V). Removed; Rule 5 still covers mv chains
        # when the intermediate is unused by later code (best-effort).

        # Rule 2: li x, a; addi x, x, b -> li x, a+b
        PeepholeRule(
            name="li+addi fusion",
            pattern=["li", "addi"],
            replacement=["li {rd} {imm_sum}"],
            register_constraints=[(0, 1, 0), (0, 1, 1)],
        ),

        # Rule 3: beq x0, x0, label -> j label
        PeepholeRule(
            name="beq zero-zero to jump",
            pattern=["beq"],
            replacement=["j {label}"],
            register_constraints=[],
        ),

        # Rule 4: mv a, b; mv c, a -> mv c, b (skip intermediate register a)
        # Unsound if `a` is live after the pair; callers/tests must treat as
        # best-effort without liveness analysis.
        PeepholeRule(
            name="redundant mv elimination",
            pattern=["mv", "mv"],
            replacement=["mv {rd1} {rs2}"],
            register_constraints=[(0, 1, 1)],
        ),

        # Rule 5: addi rd, rd, 0 -> deleted (no-op)
        PeepholeRule(
            name="addi-zero self elimination",
            pattern=["addi"],
            replacement=[],
            register_constraints=[(0, 0, 1)],  # rd == rs1
        ),

        # Rule 6: addi rd, rs, 0 (rd != rs) -> mv rd, rs
        PeepholeRule(
            name="addi-zero to mv",
            pattern=["addi"],
            replacement=["mv {rd} {rs}"],
            register_constraints=[],
        ),

        # Rule 7: nop -> deleted
        PeepholeRule(
            name="nop elimination",
            pattern=["nop"],
            replacement=[],
            register_constraints=[],
        ),

        # Rule 8: mv x, x -> deleted
        PeepholeRule(
            name="mv-self elimination",
            pattern=["mv"],
            replacement=[],
            register_constraints=[(0, 0, 1)],  # rd == rs
        ),
    ]


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def _operand_matches(pattern_op: str, actual_op: str,
                     bindings: dict[str, str]) -> bool:
    """Check if an operand matches, updating bindings for wildcards.

    Pattern operands can be:
    - A literal string like 'x0', 'zero': must match exactly.
    - A wildcard variable like 'rd', 'rs1', 'rs2', 'imm', 'label':
      binds to the actual value on first encounter, then checks equality.
    - A string with wildcards like 'rd{1}' for second instruction's rd.
    """
    if pattern_op == "*":
        return True
    if pattern_op in bindings:
        return bindings[pattern_op] == actual_op
    # First encounter: bind it
    bindings[pattern_op] = actual_op
    return True


def _match_rule(
        rule: PeepholeRule, window: list[AsmLine],
) -> Optional[dict[str, str]]:
    """Try to match a rule against a window of AsmLine objects.

    Returns a dict of bindings (wildcard -> value) on success, None on failure.
    """
    if len(window) != len(rule.pattern):
        return None

    bindings: dict[str, str] = {}

    for i, (pat_op, line) in enumerate(zip(rule.pattern, window)):
        # Match opcode
        if pat_op != "*" and pat_op != line.opcode:
            return None

        if line.opcode is None:
            return None

        # Match operands
        operands = line.operands
        # Define expected pattern operands based on the instruction index
        expected_ops = [f"rd{i}", f"rs{i}_1", f"rs{i}_2", f"imm{i}"]
        for j, actual in enumerate(operands):
            if j < len(expected_ops):
                if not _operand_matches(expected_ops[j], actual, bindings):
                    return None

    # Never match a window that would drop a mid-window label (jump target).
    for i, line in enumerate(window):
        if i > 0 and line.label:
            return None

    # Apply register constraints (window-local indices; x0/zero aliases).
    for constraint in rule.register_constraints:
        dst_instr, src_instr, src_op = constraint
        if src_instr >= len(window) or dst_instr >= len(window):
            return None
        if not window[dst_instr].operands or not window[src_instr].operands:
            return None
        if src_op >= len(window[src_instr].operands):
            return None
        dst_rd = window[dst_instr].operands[0]
        src_val = window[src_instr].operands[src_op]
        if not _regs_equal(dst_rd, src_val):
            return None

    # beq x0/zero, x0/zero, label -> j label
    if rule.name == "beq zero-zero to jump":
        ops = window[0].operands
        if len(ops) < 2:
            return None
        if not (_is_zero_reg(ops[0]) and _is_zero_reg(ops[1])):
            return None

    # mv-chain rule must not match swap-shaped pairs: mv x,y; mv y,x
    # (that pattern is not a no-op and must be left untouched).
    # Guard operand lengths to avoid IndexError on malformed / short lines.
    if rule.name == "redundant mv elimination":
        if (
            len(window) >= 2
            and len(window[0].operands) >= 2
            and len(window[1].operands) >= 2
            and _regs_equal(
                window[1].operands[0], window[0].operands[1]
            )
        ):
            return None

    # addi+addi: both immediates must parse and sum must fit simm12
    if rule.name == "addi+addi fusion":
        ops0, ops1 = window[0].operands, window[1].operands
        if len(ops0) < 3 or len(ops1) < 3:
            return None
        imm1, imm2 = _parse_imm(ops0[2]), _parse_imm(ops1[2])
        if imm1 is None or imm2 is None:
            return None
        if not _fits_simm12(imm1 + imm2):
            return None

    # li+addi: immediates must be integers (li can hold any 32-bit result)
    if rule.name == "li+addi fusion":
        ops0, ops1 = window[0].operands, window[1].operands
        if len(ops0) < 2 or len(ops1) < 3:
            return None
        if _parse_imm(ops0[1]) is None or _parse_imm(ops1[2]) is None:
            return None

    # addi rd, rd, 0 -> delete
    if rule.name == "addi-zero self elimination":
        ops = window[0].operands
        if len(ops) < 3 or _parse_imm(ops[2]) != 0:
            return None

    # addi rd, rs, 0 (rd != rs) -> mv rd, rs
    # Non-integer immediates (labels/symbols) are skipped via _parse_imm.
    if rule.name == "addi-zero to mv":
        ops = window[0].operands
        if len(ops) < 3 or _parse_imm(ops[2]) != 0:
            return None
        if _regs_equal(ops[0], ops[1]):
            return None  # handled by addi-zero self elimination

    return bindings


# ---------------------------------------------------------------------------
# Peephole optimizer
# ---------------------------------------------------------------------------

class AsmPeepholeOptimizer:
    """Sliding-window peephole optimizer for RISC-V assembly.

    Parameters
    ----------
    rules:
        List of peephole rules. If None, uses the built-in default rules.

    Usage::

        opt = AsmPeepholeOptimizer()
        optimized_asm, num_changes = opt.optimize(asm_text)
    """

    def __init__(self, rules: Optional[list[PeepholeRule]] = None):
        self.rules: list[PeepholeRule] = (
            rules if rules is not None else _default_rules()
        )
        self._total_matches: dict[str, int] = {}
        self._instr_before: int = 0
        self._instr_after: int = 0
        self._iterations: int = 0

    @property
    def total_matches(self) -> dict[str, int]:
        """Return per-rule match counts from the last ``optimize()`` call."""
        return dict(self._total_matches)

    @property
    def instructions_before(self) -> int:
        """Opcode count in the input of the last ``optimize()`` call."""
        return self._instr_before

    @property
    def instructions_after(self) -> int:
        """Opcode count in the output of the last ``optimize()`` call."""
        return self._instr_after

    @property
    def instructions_saved(self) -> int:
        """Static instructions removed by the last ``optimize()`` call."""
        return max(0, self._instr_before - self._instr_after)

    @property
    def iterations(self) -> int:
        """Number of fixed-point passes performed by the last ``optimize()``."""
        return self._iterations

    def optimize(self, asm_text: str) -> tuple[str, int]:
        """Apply peephole optimization to assembly text.

        Parameters
        ----------
        asm_text:
            Input RISC-V assembly text.

        Returns
        -------
        Tuple of (optimized_asm_string, total_number_of_changes).
        """
        lines = _parse_asm(asm_text)
        self._total_matches = {r.name: 0 for r in self.rules}
        self._instr_before = _count_opcodes(lines)
        total_changes = 0

        # Iterate until a fixed point is reached
        changed = True
        iteration = 0
        max_iterations = 50  # safety limit

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            new_lines: list[AsmLine] = []
            i = 0

            while i < len(lines):
                matched = False
                for rule in self.rules:
                    window_size = len(rule.pattern)
                    if i + window_size > len(lines):
                        continue

                    window = lines[i:i + window_size]
                    bindings = _match_rule(rule, window)

                    if bindings is not None:
                        # Apply replacement
                        replacement_lines = self._apply_replacement(
                            rule, window, bindings)
                        new_lines.extend(replacement_lines)
                        self._total_matches[rule.name] += 1
                        total_changes += 1
                        i += window_size
                        matched = True
                        changed = True
                        break

                if not matched:
                    new_lines.append(lines[i])
                    i += 1

            lines = new_lines

        self._instr_after = _count_opcodes(lines)
        self._iterations = iteration
        return _lines_to_asm(lines), total_changes

    def _apply_replacement(self, rule: PeepholeRule,
                           window: list[AsmLine],
                           bindings: dict[str, str]) -> list[AsmLine]:
        """Generate replacement lines from a matched rule.

        Supports template substitution using bindings and simple constant
        folding (e.g., {imm_sum} for addi+addi fusion).
        """
        result: list[AsmLine] = []
        lead_label = window[0].label if window else None

        # Deletion: keep a bare label so jump targets are not lost.
        if not rule.replacement:
            if lead_label:
                return [AsmLine(
                    raw=f"{lead_label}:",
                    label=lead_label,
                    lineno=window[0].lineno,
                )]
            return []

        # Compute derived values
        derived: dict[str, str] = {}
        if rule.name == "addi+addi fusion":
            imm1 = _parse_imm(
                window[0].operands[2] if len(window[0].operands) > 2 else "0"
            )
            imm2 = _parse_imm(
                window[1].operands[2] if len(window[1].operands) > 2 else "0"
            )
            if imm1 is None or imm2 is None:
                return list(window)
            derived["imm_sum"] = str(imm1 + imm2)
            derived["rd"] = (
                window[0].operands[0] if window[0].operands else "x0"
            )
            derived["rs1"] = (
                window[0].operands[1]
                if len(window[0].operands) > 1 else "x0"
            )

        elif rule.name == "li+addi fusion":
            imm1 = _parse_imm(
                window[0].operands[1] if len(window[0].operands) > 1 else "0"
            )
            imm2 = _parse_imm(
                window[1].operands[2] if len(window[1].operands) > 2 else "0"
            )
            if imm1 is None or imm2 is None:
                return list(window)
            derived["imm_sum"] = str(imm1 + imm2)
            derived["rd"] = (
                window[0].operands[0] if window[0].operands else "x0"
            )

        elif rule.name == "beq zero-zero to jump":
            derived["label"] = (
                window[0].operands[2]
                if len(window[0].operands) > 2 else "L0"
            )

        elif rule.name == "redundant mv elimination":
            derived["rd1"] = (
                window[1].operands[0] if window[1].operands else "x0"
            )
            derived["rs2"] = (
                window[0].operands[1]
                if len(window[0].operands) > 1 else "x0"
            )

        elif rule.name == "addi-zero to mv":
            derived["rd"] = (
                window[0].operands[0] if window[0].operands else "x0"
            )
            derived["rs"] = (
                window[0].operands[1]
                if len(window[0].operands) > 1 else "x0"
            )

        # Generate replacement lines from template
        for idx, repl_op_str in enumerate(rule.replacement):
            # Substitute template variables
            repl = repl_op_str
            for key, val in derived.items():
                repl = repl.replace(f"{{{key}}}", val)
            for key, val in bindings.items():
                repl = repl.replace(f"{{{key}}}", val)

            parts = repl.split()
            if not parts:
                continue
            opcode = parts[0]
            operands = parts[1:] if len(parts) > 1 else []
            comment = f"peephole: {rule.name}"

            result.append(AsmLine(
                raw=repl,
                label=lead_label if idx == 0 else None,
                opcode=opcode,
                operands=operands,
                comment=comment,
            ))

        return result

    def report(self) -> str:
        """Return a human-readable report of the last ``optimize()`` call.

        Includes rule match counts and static instruction savings
        (opcode lines before/after).
        """
        total = sum(self._total_matches.values())
        before = self._instr_before
        after = self._instr_after
        saved = self.instructions_saved
        if before > 0:
            pct = 100.0 * saved / before
            saved_str = f"{saved} ({pct:.1f}%)"
        else:
            saved_str = str(saved)

        lines = [
            "Peephole Optimizer Report",
            f"  Instructions before: {before}",
            f"  Instructions after:  {after}",
            f"  Instructions saved:  {saved_str}",
            f"  Rule applications:   {total}",
            f"  Fixed-point passes:  {self._iterations}",
        ]
        if total > 0:
            lines.append("  Rules applied:")
            for name, count in self._total_matches.items():
                if count > 0:
                    lines.append(f"    {name}: {count} time(s)")
        else:
            lines.append("  No optimization opportunities found.")
        # Keep legacy key for scripts/tests that grep "Total changes"
        lines.append(f"  Total changes: {total}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the peephole optimizer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RISC-V Assembly Peephole Optimizer",
    )
    parser.add_argument(
        "input", type=str,
        help="Input assembly file (.s)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file (default: print to stdout)",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print optimization report",
    )
    parser.add_argument(
        "--list-rules", action="store_true",
        help="List all available rules and exit",
    )

    args = parser.parse_args()

    if args.list_rules:
        for rule in _default_rules():
            print(f"  {rule.name}")
        return

    with open(args.input, "r") as f:
        asm_text = f.read()

    opt = AsmPeepholeOptimizer()
    result, changes = opt.optimize(asm_text)

    if args.report:
        print(opt.report(), file=sys.stderr)
        print(
            f"Summary: {changes} rule application(s), "
            f"{opt.instructions_saved} instruction(s) saved "
            f"({opt.instructions_before} -> {opt.instructions_after})",
            file=sys.stderr,
        )

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        print(
            f"Optimized assembly written to {args.output}",
            file=sys.stderr,
        )
    else:
        print(result)


if __name__ == "__main__":
    main()
