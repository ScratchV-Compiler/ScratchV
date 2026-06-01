"""Assembly-level Peephole Optimizer for RISC-V.

Applies peephole optimization rules to RISC-V assembly text using
sliding-window pattern matching with register wildcards.

Usage::

    from scratchv.backend.asm_peephole import PeepholeOptimizer
    opt = PeepholeOptimizer()
    optimized_text, changes = opt.optimize(asm_text)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class AsmLine:
    """Represents one parsed line of assembly."""
    raw: str
    label: Optional[str] = None
    opcode: Optional[str] = None
    operands: list[str] = field(default_factory=list)
    comment: Optional[str] = None
    lineno: int = 0

    def __str__(self) -> str:
        if self.label is not None and self.opcode is None:
            return self.raw
        parts = []
        if self.label:
            parts.append(f"{self.label}:")
        if self.opcode:
            parts.append(f"  {self.opcode}")
            if self.operands:
                parts.append(" " + ", ".join(self.operands))
        if self.comment:
            parts.append(f"  # {self.comment}")
        return "".join(parts)


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
        Optional list of index-pair tuples ``(i, j)`` specifying that the
        destination register of instruction i must equal some operand of
        instruction j for the rule to fire.
        Format: ``(dst_index, src_instruction_index, src_operand_index)``.
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
# Parsing
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(
    r'^\s*'
    r'(?P<label>[A-Za-z_.][A-Za-z0-9_.]*:)?\s*'
    r'(?P<opcode>\.?\w[\w.]*)?\s*'
    r'(?P<operands>[^#]*?)'
    r'(?:\s*#\s*(?P<comment>.*))?'
    r'$'
)


def _parse_line(line: str, lineno: int = 0) -> AsmLine:
    """Parse a single assembly line into an AsmLine object."""
    m = _LINE_RE.match(line)
    if m is None:
        return AsmLine(raw=line, lineno=lineno)

    label = m.group("label")
    if label is not None:
        label = label.rstrip(":")

    opcode = m.group("opcode")
    if opcode is not None:
        opcode = opcode.strip().lower()

    operands_raw = (m.group("operands") or "").strip()
    operands_list = [
        o.strip() for o in operands_raw.split(",") if o.strip()
    ]

    comment = m.group("comment")
    if comment is not None:
        comment = comment.strip()

    return AsmLine(
        raw=line,
        label=label,
        opcode=opcode,
        operands=operands_list,
        comment=comment,
        lineno=lineno,
    )


def _split_operands(s: str) -> list[str]:
    """Split operand string by comma, respecting parentheses."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
    if current or not parts:
        parts.append("".join(current).strip())
    return [p for p in parts if p]


def _parse_asm(asm_text: str) -> list[AsmLine]:
    """Parse full assembly text into a list of AsmLine objects."""
    lines = asm_text.strip().split("\n")
    result = []
    line_count = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Preserve empty lines but as minimal records
        if not stripped:
            result.append(AsmLine(raw=line, lineno=i))
            continue
        al = _parse_line(line, lineno=i)
        result.append(al)
        if al.opcode is not None:
            line_count += 1
    return result


def _lines_to_asm(lines: list[AsmLine]) -> str:
    """Convert a list of AsmLine objects back to assembly text."""
    output = []
    for al in lines:
        if al.opcode is None and al.label is None and not al.raw.strip():
            output.append("")
        elif al.label is not None and al.opcode is None and not al.operands:
            output.append(f"{al.label}:")
        else:
            parts = []
            if al.label:
                parts.append(f"{al.label}:")
            if al.opcode:
                parts.append(f"  {al.opcode}")
                if al.operands:
                    parts.append(" " + ", ".join(al.operands))
            if al.comment:
                parts.append(f"  # {al.comment}")
            output.append("".join(parts))
    return "\n".join(output)


# ---------------------------------------------------------------------------
# Default peephole rules
# ---------------------------------------------------------------------------

def _default_rules() -> list[PeepholeRule]:
    """Return the set of five default peephole optimization rules."""
    return [
        # Rule 1: addi x, x, a; addi x, x, b -> addi x, x, a+b
        PeepholeRule(
            name="addi+addi fusion",
            pattern=["addi", "addi"],
            replacement=["addi {rd} {rs1} {imm_sum}"],
            register_constraints=[(0, 1, 0), (0, 1, 1)],
        ),

        # Rule 2: mv x, y; mv y, x -> deleted (redundant swap)
        PeepholeRule(
            name="redundant mv pair elimination",
            pattern=["mv", "mv"],
            replacement=[],  # deleted entirely
            register_constraints=[(0, 1, 0), (1, 0, 0)],
        ),

        # Rule 3: li x, a; addi x, x, b -> li x, a+b
        PeepholeRule(
            name="li+addi fusion",
            pattern=["li", "addi"],
            replacement=["li {rd} {imm_sum}"],
            register_constraints=[(0, 1, 0), (0, 1, 1)],
        ),

        # Rule 4: beq x0, x0, label -> j label
        PeepholeRule(
            name="beq zero-zero to jump",
            pattern=["beq"],
            replacement=["j {label}"],
            register_constraints=[],
        ),

        # Rule 5: mv a, b; ... (a not used) mv c, a -> mv c, b
        # (redundant move through intermediate)
        PeepholeRule(
            name="redundant mv elimination",
            pattern=["mv", "mv"],
            replacement=["mv {rd1} {rs2}"],
            register_constraints=[(1, 0, 0)],
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

    # Check register constraints
    for dst_idx, src_instr_idx, src_op_idx in rule.register_constraints:
        if dst_idx >= len(window) or src_instr_idx >= len(window):
            return None
        dst_line = window[dst_idx]
        src_line = window[src_instr_idx]
        if not dst_line.operands or not src_line.operands:
            return None
        if src_op_idx >= len(src_line.operands):
            return None

    # Apply actual constraints
    for constraint in rule.register_constraints:
        dst_instr, src_instr, src_op = constraint
        if src_instr >= len(window) or dst_instr >= len(window):
            return None
        if not window[dst_instr].operands or not window[src_instr].operands:
            return None
        dst_rd = (
            window[dst_instr].operands[0]
            if window[dst_instr].operands else None
        )
        if src_op < len(window[src_instr].operands):
            src_val = window[src_instr].operands[src_op]
        else:
            return None
        if dst_rd != src_val:
            return None

    # For Rule 4 (beq x0,x0 -> j): both operands must be zero
    if rule.name == "beq zero-zero to jump":
        ops = window[0].operands
        if len(ops) < 2:
            return None
        if ops[0] not in ("x0", "zero") or ops[1] not in ("x0", "zero"):
            return None

    return bindings


# ---------------------------------------------------------------------------
# Peephole optimizer
# ---------------------------------------------------------------------------

class PeepholeOptimizer:
    """Sliding-window peephole optimizer for RISC-V assembly.

    Parameters
    ----------
    rules:
        List of peephole rules. If None, uses the five default rules.

    Usage::

        opt = PeepholeOptimizer()
        optimized_asm, num_changes = opt.optimize(asm_text)
    """

    def __init__(self, rules: Optional[list[PeepholeRule]] = None):
        self.rules: list[PeepholeRule] = (
            rules if rules is not None else _default_rules()
        )
        self._total_matches: dict[str, int] = (
            {}  # rule_name -> match count
        )

    @property
    def total_matches(self) -> dict[str, int]:
        """Return per-rule match counts from the last ``optimize()`` call."""
        return dict(self._total_matches)

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

        return _lines_to_asm(lines), total_changes

    def _apply_replacement(self, rule: PeepholeRule,
                           window: list[AsmLine],
                           bindings: dict[str, str]) -> list[AsmLine]:
        """Generate replacement lines from a matched rule.

        Supports template substitution using bindings and simple constant
        folding (e.g., {imm_sum} for addi+addi fusion).
        """
        result: list[AsmLine] = []

        # Compute derived values
        derived: dict[str, str] = {}
        if rule.name == "addi+addi fusion":
            # Try to compute imm1 + imm2
            imm1_str = (
                window[0].operands[2]
                if len(window[0].operands) > 2 else "0"
            )
            imm2_str = (
                window[1].operands[2]
                if len(window[1].operands) > 2 else "0"
            )
            try:
                imm_sum = int(imm1_str) + int(imm2_str)
                derived["imm_sum"] = str(imm_sum)
            except ValueError:
                derived["imm_sum"] = f"({imm1_str}+{imm2_str})"
            derived["rd"] = (
                window[0].operands[0] if window[0].operands else "x0"
            )
            derived["rs1"] = (
                window[0].operands[1]
                if len(window[0].operands) > 1 else "x0"
            )

        elif rule.name == "li+addi fusion":
            imm1_str = (
                window[0].operands[1]
                if len(window[0].operands) > 1 else "0"
            )
            imm2_str = (
                window[1].operands[2]
                if len(window[1].operands) > 2 else "0"
            )
            try:
                imm_sum = int(imm1_str) + int(imm2_str)
                derived["imm_sum"] = str(imm_sum)
            except ValueError:
                derived["imm_sum"] = f"({imm1_str}+{imm2_str})"
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

        # Generate replacement lines from template
        for repl_op_str in rule.replacement:
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
                opcode=opcode,
                operands=operands,
                comment=comment,
            ))

        return result

    def report(self) -> str:
        """Return a human-readable report of optimizations applied."""
        total = sum(self._total_matches.values())
        lines = []
        lines.append("Peephole Optimizer Report")
        lines.append(f"  Total changes: {total}")
        if total > 0:
            lines.append("  Rules applied:")
            for name, count in self._total_matches.items():
                if count > 0:
                    lines.append(f"    {name}: {count} time(s)")
        else:
            lines.append("  No optimization opportunities found.")
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

    opt = PeepholeOptimizer()
    result, changes = opt.optimize(asm_text)

    if args.report:
        print(opt.report(), file=sys.stderr)
        print(f"Total changes: {changes}", file=sys.stderr)

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
