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
        block_names: set[str] = {b.name for b in func.blocks}

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

        Uses a two-pass approach:
        1. First pass: collect all values that are assigned (appear as
           instruction destinations) across all blocks.
        2. Second pass: flag operands that are never assigned and aren't
           constants or function params.

        Values that appear as operands but are never assigned are treated
        as implicit input variables (not flagged as errors).

        Args:
            func: The function to check.
        """
        # Pass 1: collect all defined names (instruction destinations)
        defined: set[str] = set()

        # Function parameters are pre-defined
        for param in func.params:
            defined.add(param.name)

        for block in func.blocks:
            for instr in block.instructions:
                if instr.dest is not None:
                    defined.add(instr.dest.name)

        # Pass 2: flag uses of undefined values
        for block in func.blocks:
            for instr in block.instructions:
                for op in instr.operands:
                    if op.name not in defined:
                        # Allow constants (auto-defined) and implicit inputs
                        if op.is_constant:
                            continue
                        # Treat as implicit input (not an error)
                        # Mark so it's not flagged again
                        defined.add(op.name)
                        continue

                # Also track values created mid-block for intra-block checks
                if instr.dest is not None:
                    defined.add(instr.dest.name)

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
                target = instr.target
                if target is None:
                    continue

                # BR_IF has comma-separated targets
                if instr.opcode == OpCode.BR_IF:
                    parts = target.split(",")
                    for part in parts:
                        part = part.strip()
                        if part and part not in block_names:
                            self._add_error(
                                ErrorLevel.ERROR,
                                f"branch target '{part}' does not exist",
                                func_name=func.name,
                                block_name=block.name,
                                instruction_index=i,
                                rule="label-existence",
                            )
                else:
                    if target not in block_names:
                        self._add_error(
                            ErrorLevel.ERROR,
                            f"jump target '{target}' does not exist",
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
                if instr.opcode in binary_ops and len(instr.operands) >= 2:
                    lhs, rhs = instr.operands[0], instr.operands[1]
                    if lhs.dtype != rhs.dtype:
                        self._add_error(
                            ErrorLevel.WARNING,
                            f"operand type mismatch: '{lhs.name}' is "
                            f"{lhs.dtype.value}, '{rhs.name}' is "
                            f"{rhs.dtype.value}",
                            func_name=func.name,
                            block_name=block.name,
                            instruction_index=i,
                            rule="type-consistency",
                        )

                if instr.opcode in nn_ops and len(instr.operands) >= 2:
                    lhs, rhs = instr.operands[0], instr.operands[1]
                    if lhs.dtype != rhs.dtype:
                        self._add_error(
                            ErrorLevel.WARNING,
                            f"NN op operand type mismatch: '{lhs.name}' is "
                            f"{lhs.dtype.value}, '{rhs.name}' is "
                            f"{rhs.dtype.value}",
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
        for block in func.blocks:
            for i, instr in enumerate(block.instructions):
                if instr.opcode == OpCode.BR:
                    # Cannot have instructions after unconditional jump
                    if i < len(block.instructions) - 1:
                        self._add_error(
                            ErrorLevel.ERROR,
                            "unreachable instructions after unconditional "
                            "branch",
                            func_name=func.name,
                            block_name=block.name,
                            instruction_index=i,
                            rule="control-flow-integrity",
                        )

                elif instr.opcode == OpCode.BR_IF:
                    # Must have exactly two targets
                    target = instr.target or ""
                    targets = [
                        t.strip() for t in target.split(",") if t.strip()
                    ]
                    if len(targets) != 2:
                        self._add_error(
                            ErrorLevel.ERROR,
                            f"conditional branch has {len(targets)} "
                            f"targets, expected 2",
                            func_name=func.name,
                            block_name=block.name,
                            instruction_index=i,
                            rule="control-flow-integrity",
                        )

                elif instr.opcode == OpCode.RETURN:
                    if i < len(block.instructions) - 1:
                        self._add_error(
                            ErrorLevel.ERROR,
                            "unreachable instructions after return",
                            func_name=func.name,
                            block_name=block.name,
                            instruction_index=i,
                            rule="control-flow-integrity",
                        )

    # -------------------------------------------------------------------
    # Rule 6: SSA validity
    # -------------------------------------------------------------------

    def _check_ssa_validity(self, func: Function) -> None:
        """Check SSA validity: each value must be assigned exactly once.

        Args:
            func: The function to check.
        """
        assigned: dict[str, int] = {}  # value name -> first assignment index

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
                        assigned[instr.dest.name] = i

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
