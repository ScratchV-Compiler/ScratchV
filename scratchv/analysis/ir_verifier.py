"""IR verification pass for ScratchV.

Validates IR programs against a set of correctness rules, producing
a list of verification errors or warnings. Designed to be run before
and after optimization passes to catch bugs early.

Verification rules:
    1. Def-before-use: All value operands must be defined before use.
    2. Label existence: Branch/jump targets must exist as block labels.
    3. Block termination: Every basic block must end with a terminator
       (return, branch, or jump).
    4. Type consistency: Operands of arithmetic/nn ops must have
       compatible types.
    5. Control flow integrity: Blocks after unconditional jumps must
       be unreachable. Conditional branches must have exactly two
       targets specified.
    6. SSA validity: Each value must be assigned exactly once (SSA).
    7. Entry existence: Every function must contain a basic block.

Usage::

    from scratchv.analysis.ir_verifier import IRVerifier

    verifier = IRVerifier(program)
    errors = verifier.verify()
    if errors:
        for err in errors:
            print(err)
    else:
        print("IR verification passed.")
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

# moved import above
from scratchv.ir.types import (
    OpCode,
    Function,
    Instruction,
    Program,
)


# ---------------------------------------------------------------------------
# Error level
# ---------------------------------------------------------------------------

class ErrorLevel(enum.Enum):
    """Severity level for verification issues."""
    ERROR = "error"
    WARNING = "warning"


# ---------------------------------------------------------------------------
# VerificationError
# ---------------------------------------------------------------------------

@dataclass
class VerificationError:
    """A single verification issue found in IR.

    Attributes:
        level: Severity (ERROR or WARNING).
        message: Human-readable description of the issue.
        function_name: Name of the function containing the issue.
        block_name: Name of the basic block (if applicable).
        instruction_index: Index of the instruction (if applicable).
        value_name: Name of the problematic value (if applicable).
        rule: Identifier for the verification rule violated.
    """
    level: ErrorLevel
    message: str
    function_name: Optional[str] = None
    block_name: Optional[str] = None
    instruction_index: Optional[int] = None
    value_name: Optional[str] = None
    rule: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"[{self.level.value.upper()}]"]
        if self.rule:
            parts.append(f"({self.rule})")
        if self.function_name:
            parts.append(f"in '{self.function_name}'")
        if self.block_name:
            parts.append(f", block '{self.block_name}'")
        if self.instruction_index is not None:
            parts.append(f", instr #{self.instruction_index}")
        if self.value_name:
            parts.append(f", value '{self.value_name}'")
        parts.append(f": {self.message}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# IRVerifier
# ---------------------------------------------------------------------------

class IRVerifier:
    """Verify the correctness of a ScratchV IR Program.

    Usage::

        from scratchv.analysis.ir_verifier import IRVerifier
        verifier = IRVerifier(program)
        errors = verifier.verify()
        if errors:
            for e in errors:
                print(e)
            raise SystemExit(1)

    The verifier can be run repeatedly on the same program as it does
    not mutate any state.
    """

    def __init__(self, program: Program):
        """Initialize the verifier.

        Args:
            program: The IR Program to verify.
        """
        self.program = program
        self._errors: list[VerificationError] = []

    # -------------------------------------------------------------------
    # Main verification entry point
    # -------------------------------------------------------------------

    def verify(self) -> list[VerificationError]:
        """Run all verification checks on the program.

        Returns:
            A list of VerificationError objects. An empty list means
            the program passed all checks.
        """
        self._errors = []
        self._check_global_ssa_validity()

        for func in self.program.functions:
            self._verify_function(func)

        return self._errors

    # -------------------------------------------------------------------
    # Per-function verification
    # -------------------------------------------------------------------

    def _verify_function(self, func: Function) -> None:
        """Run all checks on a single function.

        Args:
            func: The function to verify.
        """
        # Collect all block names for label checks
        block_names: set[str] = set()
        for block in func.blocks:
            if block.name in block_names:
                self._add_error(
                    ErrorLevel.ERROR,
                    f"duplicate block label '{block.name}'",
                    func_name=func.name,
                    block_name=block.name,
                    rule="label-existence",
                )
            block_names.add(block.name)

        # Check 1: Def-before-use per function
        self._check_def_before_use(func)

        # Check 2: Block termination
        self._check_block_termination(func)

        # Check 3: Label existence in branches/jumps
        self._check_label_existence(func, block_names)

        # Check 4: Type consistency
        self._check_type_consistency(func)

        # Check 5: Control flow integrity
        self._check_control_flow_integrity(func, block_names)

        # Check 6: SSA validity
        self._check_ssa_validity(func)

        # Check 7: Entry block existence
        if len(func.blocks) == 0:
            self._add_error(
                ErrorLevel.ERROR,
                "function has no basic blocks",
                func_name=func.name,
                rule="entry-existence",
            )

    # -------------------------------------------------------------------
    # Rule 1: Def-before-use
    # -------------------------------------------------------------------

    def _check_def_before_use(self, func: Function) -> None:
        """Ensure all value operands are defined before use.

        Function parameters, program globals, and constants are available at
        function entry.  Instruction results are available only after their
        defining instruction and only in blocks dominated by that definition.

        Args:
            func: The function to check.
        """
        entry_values = {param.name for param in func.params}
        entry_values.update(value.name for value in self.program.global_values)

        definitions: dict[str, list[tuple[str, int]]] = {}
        for block in func.blocks:
            for i, instr in enumerate(block.instructions):
                if instr.dest is not None:
                    definitions.setdefault(instr.dest.name, []).append(
                        (block.name, i),
                    )

        dominators = self._compute_dominators(func)

        for block in func.blocks:
            for i, instr in enumerate(block.instructions):
                for op in instr.operands:
                    if op.is_constant or op.name in entry_values:
                        continue

                    sites = definitions.get(op.name, [])
                    is_defined = any(
                        (
                            def_block == block.name
                            and def_index < i
                        )
                        or (
                            def_block != block.name
                            and def_block in dominators.get(block.name, set())
                        )
                        for def_block, def_index in sites
                    )
                    if is_defined:
                        continue

                    self._add_error(
                        ErrorLevel.ERROR,
                        f"value '{op.name}' is used before a dominating "
                        "definition",
                        func_name=func.name,
                        block_name=block.name,
                        instruction_index=i,
                        value_name=op.name,
                        rule="def-before-use",
                    )

    def _compute_dominators(self, func: Function) -> dict[str, set[str]]:
        """Compute block dominators for the reachable control-flow graph."""
        if not func.blocks:
            return {}

        block_names = {block.name for block in func.blocks}
        successors: dict[str, set[str]] = {
            block.name: set() for block in func.blocks
        }

        for index, block in enumerate(func.blocks):
            if not block.instructions:
                if index + 1 < len(func.blocks):
                    successors[block.name].add(func.blocks[index + 1].name)
                continue
            terminator = block.instructions[-1]
            for target in self._branch_targets(terminator):
                if target in block_names:
                    successors[block.name].add(target)
            if (
                terminator.opcode not in {
                    OpCode.BR, OpCode.BR_IF, OpCode.RETURN,
                }
                and index + 1 < len(func.blocks)
            ):
                successors[block.name].add(func.blocks[index + 1].name)

        entry = func.blocks[0].name
        reachable = {entry}
        worklist = [entry]
        while worklist:
            current = worklist.pop()
            for successor in successors[current]:
                if successor not in reachable:
                    reachable.add(successor)
                    worklist.append(successor)

        predecessors: dict[str, set[str]] = {
            name: set() for name in block_names
        }
        for source, targets in successors.items():
            for target in targets:
                predecessors[target].add(source)

        dominators = {
            name: ({entry} if name == entry else set(reachable))
            for name in reachable
        }
        changed = True
        while changed:
            changed = False
            for name in reachable:
                if name == entry:
                    continue
                reachable_predecessors = predecessors[name] & reachable
                if reachable_predecessors:
                    common = set.intersection(
                        *(dominators[pred] for pred in reachable_predecessors)
                    )
                    updated = {name} | common
                else:
                    updated = {name}
                if updated != dominators[name]:
                    dominators[name] = updated
                    changed = True

        for name in block_names - reachable:
            dominators[name] = {name}
        return dominators

    @staticmethod
    def _branch_targets(instr: Instruction) -> list[str]:
        """Return normalized targets for a branch instruction."""
        if instr.opcode == OpCode.BR:
            return [(instr.target or "").strip()]
        if instr.opcode == OpCode.BR_IF:
            return [
                target.strip()
                for target in (instr.target or "").split(",")
            ]
        return []

    @staticmethod
    def _has_two_branch_targets(targets: list[str]) -> bool:
        """Return whether a conditional branch has two usable targets."""
        return len(targets) == 2 and all(targets)

    # -------------------------------------------------------------------
    # Rule 2: Block termination
    # -------------------------------------------------------------------

    def _check_block_termination(self, func: Function) -> None:
        """Ensure every basic block ends with a terminator instruction.

        Valid terminators: RETURN, BR, BR_IF. Empty blocks are flagged.

        Args:
            func: The function to check.
        """
        terminators = {OpCode.RETURN, OpCode.BR, OpCode.BR_IF}

        for block in func.blocks:
            if not block.instructions:
                self._add_error(
                    ErrorLevel.WARNING,
                    "block has no instructions (no terminator)",
                    func_name=func.name,
                    block_name=block.name,
                    rule="block-termination",
                )
                continue

            last_instr = block.instructions[-1]
            if last_instr.opcode not in terminators:
                self._add_error(
                    ErrorLevel.ERROR,
                    f"block does not end with a terminator "
                    f"(last instruction is '{last_instr.opcode.value}')",
                    func_name=func.name,
                    block_name=block.name,
                    instruction_index=len(block.instructions) - 1,
                    rule="block-termination",
                )

    # -------------------------------------------------------------------
    # Rule 3: Label existence
    # -------------------------------------------------------------------

    def _check_label_existence(
        self, func: Function, block_names: set[str],
    ) -> None:
        """Ensure all branch/jump targets refer to existing blocks.

        Args:
            func: The function to check.
            block_names: Set of valid block names in this function.
        """
        for block in func.blocks:
            for i, instr in enumerate(block.instructions):
                if instr.opcode not in {OpCode.BR, OpCode.BR_IF}:
                    continue

                targets = self._branch_targets(instr)
                if instr.opcode == OpCode.BR_IF:
                    if not self._has_two_branch_targets(targets):
                        self._add_error(
                            ErrorLevel.ERROR,
                            "conditional branch must specify two non-empty "
                            "targets",
                            func_name=func.name,
                            block_name=block.name,
                            instruction_index=i,
                            rule="label-existence",
                        )
                for target in targets:
                    if not target:
                        if instr.opcode == OpCode.BR:
                            self._add_error(
                                ErrorLevel.ERROR,
                                "branch target is missing",
                                func_name=func.name,
                                block_name=block.name,
                                instruction_index=i,
                                rule="label-existence",
                            )
                        continue
                    if target not in block_names:
                        self._add_error(
                            ErrorLevel.ERROR,
                            f"branch target '{target}' does not exist",
                            func_name=func.name,
                            block_name=block.name,
                            instruction_index=i,
                            rule="label-existence",
                        )

    # -------------------------------------------------------------------
    # Rule 4: Type consistency
    # -------------------------------------------------------------------

    def _check_type_consistency(self, func: Function) -> None:
        """Ensure operands of binary/arithmetic ops have consistent types.

        Args:
            func: The function to check.
        """
        binary_ops = {
            OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV,
        }
        nn_ops = {
            OpCode.MATMUL, OpCode.DOT, OpCode.CONV,
        }

        for block in func.blocks:
            for i, instr in enumerate(block.instructions):
                if (
                    instr.opcode in binary_ops | nn_ops
                    and len(instr.operands) >= 2
                ):
                    expected = instr.operands[0]
                    for operand in instr.operands[1:]:
                        if expected.dtype == operand.dtype:
                            continue
                        self._add_error(
                            ErrorLevel.WARNING,
                            f"operand type mismatch: '{expected.name}' is "
                            f"{expected.dtype.value}, '{operand.name}' is "
                            f"{operand.dtype.value}",
                            func_name=func.name,
                            block_name=block.name,
                            instruction_index=i,
                            rule="type-consistency",
                        )

    # -------------------------------------------------------------------
    # Rule 5: Control flow integrity
    # -------------------------------------------------------------------

    def _check_control_flow_integrity(
        self, func: Function, block_names: set[str],
    ) -> None:
        """Check control flow integrity.

        - Unconditional jump (BR) must not be followed by instructions
          in the same block.
        - Conditional branch (BR_IF) must have exactly two targets.
        - RETURN must be the last instruction in a block.

        Args:
            func: The function to check.
            block_names: Valid block names.
        """
        terminators = {OpCode.BR, OpCode.BR_IF, OpCode.RETURN}

        for block in func.blocks:
            for i, instr in enumerate(block.instructions):
                if (
                    instr.opcode in terminators
                    and i < len(block.instructions) - 1
                ):
                    self._add_error(
                        ErrorLevel.ERROR,
                        f"unreachable instructions after "
                        f"{instr.opcode.value}",
                        func_name=func.name,
                        block_name=block.name,
                        instruction_index=i,
                        rule="control-flow-integrity",
                    )

                if instr.opcode == OpCode.BR_IF:
                    # Must have exactly two targets
                    targets = self._branch_targets(instr)
                    if not self._has_two_branch_targets(targets):
                        self._add_error(
                            ErrorLevel.ERROR,
                            "conditional branch must have exactly two "
                            "non-empty targets",
                            func_name=func.name,
                            block_name=block.name,
                            instruction_index=i,
                            rule="control-flow-integrity",
                        )
                    if not instr.operands:
                        self._add_error(
                            ErrorLevel.ERROR,
                            "conditional branch has no condition operand",
                            func_name=func.name,
                            block_name=block.name,
                            instruction_index=i,
                            rule="control-flow-integrity",
                        )

    # -------------------------------------------------------------------
    # Rule 6: SSA validity
    # -------------------------------------------------------------------

    def _check_global_ssa_validity(self) -> None:
        """Ensure program-global values have unique SSA names."""
        assigned: set[str] = set()
        for value in self.program.global_values:
            if value.name in assigned:
                self._add_error(
                    ErrorLevel.ERROR,
                    f"global value '{value.name}' assigned multiple times "
                    "(SSA violation)",
                    value_name=value.name,
                    rule="ssa-validity",
                )
            else:
                assigned.add(value.name)

    def _check_ssa_validity(self, func: Function) -> None:
        """Check SSA validity: each value must be assigned exactly once.

        Args:
            func: The function to check.
        """
        assigned = {value.name for value in self.program.global_values}

        for param in func.params:
            if param.name in assigned:
                self._add_error(
                    ErrorLevel.ERROR,
                    f"value '{param.name}' assigned multiple times "
                    "(SSA violation)",
                    func_name=func.name,
                    value_name=param.name,
                    rule="ssa-validity",
                )
            else:
                assigned.add(param.name)

        for block in func.blocks:
            for i, instr in enumerate(block.instructions):
                if instr.dest is not None:
                    if instr.dest.name in assigned:
                        self._add_error(
                            ErrorLevel.ERROR,
                            f"value '{instr.dest.name}' assigned multiple "
                            f"times (SSA violation)",
                            func_name=func.name,
                            block_name=block.name,
                            instruction_index=i,
                            value_name=instr.dest.name,
                            rule="ssa-validity",
                        )
                    else:
                        assigned.add(instr.dest.name)

    # -------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------

    def _add_error(
        self,
        level: ErrorLevel,
        message: str,
        func_name: Optional[str] = None,
        block_name: Optional[str] = None,
        instruction_index: Optional[int] = None,
        value_name: Optional[str] = None,
        rule: Optional[str] = None,
    ) -> None:
        """Add a verification error to the internal list.

        Args:
            level: Error severity.
            message: Error description.
            func_name: Function name context.
            block_name: Block name context.
            instruction_index: Instruction index context.
            value_name: Value name context.
            rule: Rule identifier.
        """
        self._errors.append(VerificationError(
            level=level,
            message=message,
            function_name=func_name,
            block_name=block_name,
            instruction_index=instruction_index,
            value_name=value_name,
            rule=rule,
        ))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def verify_ir(program: Program) -> tuple[bool, list[VerificationError]]:
    """Quick verification function for programmatic use.

    Args:
        program: The IR Program to verify.

    Returns:
        A tuple (passed, errors) where passed is True if no errors
        (only warnings at most), and errors is the list of all issues.
    """
    verifier = IRVerifier(program)
    errors = verifier.verify()
    real_errors = [e for e in errors if e.level == ErrorLevel.ERROR]
    return len(real_errors) == 0, errors
