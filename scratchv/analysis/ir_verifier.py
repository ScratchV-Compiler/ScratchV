"""IR verification and optimization-pipeline checks for ScratchV."""

from __future__ import annotations

import argparse
import enum
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from scratchv.ir.types import (
    BasicBlock, DataType, Function, Instruction, OpCode, Program, Value,
)
from scratchv.pass_interface import PassResult


class ErrorLevel(enum.Enum):
    """Severity level for a verification issue."""

    ERROR = "error"
    WARNING = "warning"


@dataclass
class VerificationError:
    """One verifier diagnostic with its IR location and violated rule."""

    level: ErrorLevel
    message: str
    function_name: Optional[str] = None
    block_name: Optional[str] = None
    instruction_index: Optional[int] = None
    value_name: Optional[str] = None
    rule: Optional[str] = None

    def __str__(self) -> str:
        location: list[str] = []
        if self.function_name is not None:
            location.append(f"function '{self.function_name}'")
        if self.block_name is not None:
            location.append(f"block '{self.block_name}'")
        if self.instruction_index is not None:
            location.append(f"instruction #{self.instruction_index}")
        if self.value_name is not None:
            location.append(f"value '{self.value_name}'")
        where = f" in {', '.join(location)}" if location else ""
        rule = f" [{self.rule}]" if self.rule else ""
        return f"{self.level.value.upper()}{rule}{where}: {self.message}"


class IRVerificationFailure(RuntimeError):
    """Raised when pipeline verification finds invalid IR."""

    def __init__(self, phase: str, issues: Sequence[VerificationError]) -> None:
        self.phase = phase
        self.issues = list(issues)
        details = "\n".join(f"  {issue}" for issue in self.issues)
        super().__init__(
            f"IR verification failed during {phase} "
            f"({len(self.issues)} issue(s)):\n{details}"
        )


def _opcode_name(opcode: object) -> str:
    raw = getattr(opcode, "name", getattr(opcode, "value", opcode))
    return str(raw).upper()


def _type_name(dtype: object) -> str:
    return str(getattr(dtype, "value", dtype))


def _targets(instruction: Instruction) -> list[str]:
    target = getattr(instruction, "target", None)
    if target is None:
        return []
    if isinstance(target, str):
        return [part.strip() for part in target.split(",") if part.strip()]
    if isinstance(target, (list, tuple)):
        return [str(part).strip() for part in target if str(part).strip()]
    text = str(target).strip()
    return [text] if text else []


BINARY_OPS = {"ADD", "SUB", "MUL", "DIV", "MATMUL", "DOT"}
TERNARY_OPS = {"CONV"}
CONDITIONAL_BRANCHES = {"BR_IF"}
UNCONDITIONAL_BRANCHES = {"BR"}
RETURNS = {"RETURN"}
TERMINATORS = CONDITIONAL_BRANCHES | UNCONDITIONAL_BRANCHES | RETURNS


class IRVerifier:
    """Validate a ScratchV IR program without mutating it."""

    def __init__(self, program: Program):
        self.program = program
        self._errors: list[VerificationError] = []

    def verify(self) -> list[VerificationError]:
        """Backward-compatible alias for :meth:`verify_program`."""
        return self.verify_program()

    def verify_program(self) -> list[VerificationError]:
        """Verify every function in the configured program."""
        self._errors = []
        functions = getattr(self.program, "functions", None)
        if not self._is_sequence(functions):
            self._add_error(
                "program must contain a sequence of functions",
                rule="program-structure",
            )
            return list(self._errors)
        for function in functions:
            if not isinstance(getattr(function, "name", None), str):
                self._add_error(
                    "program entry must be a function with a name",
                    rule="program-structure",
                )
                continue
            self._verify_function(function)
        return list(self._errors)

    def verify_function(self, function: Function) -> list[VerificationError]:
        """Verify one function, independently of the configured program."""
        self._errors = []
        self._verify_function(function)
        return list(self._errors)

    def verify_basic_block(
        self,
        block: BasicBlock,
        function: Function,
        *,
        defined: Optional[set[str]] = None,
        block_names: Optional[set[str]] = None,
    ) -> list[VerificationError]:
        """Verify one block with optional surrounding definition context."""
        self._errors = []
        known = set(defined or (param.name for param in function.params))
        labels = set(block_names or (item.name for item in function.blocks))
        self._verify_basic_block(block, function, known, labels, {})
        return list(self._errors)

    @staticmethod
    def _is_sequence(value: object) -> bool:
        return isinstance(value, Sequence) and not isinstance(
            value, (str, bytes)
        )

    def _verify_function(self, function: Function) -> None:
        blocks_value = getattr(function, "blocks", None)
        params_value = getattr(function, "params", None)
        if not self._is_sequence(blocks_value):
            self._add_error(
                "function must contain a sequence of basic blocks",
                function=function,
                rule="program-structure",
            )
            return
        if not self._is_sequence(params_value):
            self._add_error(
                "function parameters must be a sequence",
                function=function,
                rule="program-structure",
            )
            return
        blocks = list(blocks_value)
        params = list(params_value)
        if not blocks:
            self._add_error(
                "function has no basic blocks",
                function=function,
                rule="entry-existence",
            )
            return
        if not self._validate_structure(function, blocks, params):
            return

        names = [block.name for block in blocks]
        seen: set[str] = set()
        for block in blocks:
            if block.name in seen:
                self._add_error(
                    f"duplicate basic-block name '{block.name}'",
                    function=function,
                    block=block,
                    rule="label-existence",
                )
            seen.add(block.name)

        definition_sites = self._collect_definition_sites(blocks)
        all_defined_names = {
            instruction.dest.name
            for block in blocks
            for instruction in block.instructions
            if instruction.dest is not None
        }
        dominators = self._compute_dominators(blocks)
        parameter_names = {param.name for param in params}
        assigned: dict[str, tuple[str, int]] = {}
        for block in blocks:
            defined = set(parameter_names)
            for name, (definition_block, _) in definition_sites.items():
                if (
                    definition_block != block.name
                    and definition_block in dominators.get(block.name, set())
                ):
                    defined.add(name)
            self._verify_basic_block(
                block, function, defined, set(names), assigned,
                all_defined_names,
            )

    def _validate_structure(
        self,
        function: Function,
        blocks: Sequence[object],
        params: Sequence[object],
    ) -> bool:
        valid = True

        def valid_value(candidate: object) -> bool:
            return (
                isinstance(getattr(candidate, "name", None), str)
                and bool(getattr(candidate, "name", ""))
                and isinstance(getattr(candidate, "dtype", None), DataType)
                and isinstance(
                    getattr(candidate, "is_constant", None), bool
                )
            )

        for parameter in params:
            if not valid_value(parameter):
                self._add_error(
                    "function parameter must have a non-empty name and type",
                    function=function,
                    rule="program-structure",
                )
                valid = False
        for block in blocks:
            if not isinstance(getattr(block, "name", None), str) or not getattr(
                block, "name", ""
            ):
                self._add_error(
                    "basic block must have a non-empty name",
                    function=function,
                    rule="program-structure",
                )
                valid = False
                continue
            instructions = getattr(block, "instructions", None)
            if not self._is_sequence(instructions):
                self._add_error(
                    "basic-block instructions must be a sequence",
                    function=function,
                    block=block,
                    rule="program-structure",
                )
                valid = False
                continue
            for index, instruction in enumerate(instructions):
                if not isinstance(getattr(instruction, "opcode", None), OpCode):
                    self._add_error(
                        "instruction must define a recognized OpCode",
                        function=function,
                        block=block,
                        index=index,
                        rule="program-structure",
                    )
                    valid = False
                    continue
                if not hasattr(instruction, "dest"):
                    self._add_error(
                        "instruction must define a destination field",
                        function=function,
                        block=block,
                        index=index,
                        rule="program-structure",
                    )
                    valid = False
                    continue
                operands = getattr(instruction, "operands", None)
                if not self._is_sequence(operands):
                    self._add_error(
                        "instruction operands must be a sequence",
                        function=function,
                        block=block,
                        index=index,
                        rule="program-structure",
                    )
                    valid = False
                    continue
                destination = getattr(instruction, "dest", None)
                if destination is not None and not valid_value(destination):
                    self._add_error(
                        "instruction destination must have a name and type",
                        function=function,
                        block=block,
                        index=index,
                        rule="program-structure",
                    )
                    valid = False
                for operand in operands:
                    if not valid_value(operand):
                        self._add_error(
                            "instruction operand must have a name and type",
                            function=function,
                            block=block,
                            index=index,
                            rule="program-structure",
                        )
                        valid = False
        return valid

    @staticmethod
    def _collect_definition_sites(
        blocks: Sequence[BasicBlock],
    ) -> dict[str, tuple[str, int]]:
        definitions: dict[str, tuple[str, int]] = {}
        for block in blocks:
            for index, instruction in enumerate(block.instructions):
                if instruction.dest is not None:
                    definitions.setdefault(
                        instruction.dest.name, (block.name, index)
                    )
                if _opcode_name(instruction.opcode) in TERMINATORS:
                    break
        return definitions

    @staticmethod
    def _compute_dominators(
        blocks: Sequence[BasicBlock],
    ) -> dict[str, set[str]]:
        names = [block.name for block in blocks]
        known = set(names)
        successors: dict[str, set[str]] = {name: set() for name in names}
        for block in blocks:
            for instruction in block.instructions:
                opcode = _opcode_name(instruction.opcode)
                if opcode in CONDITIONAL_BRANCHES | UNCONDITIONAL_BRANCHES:
                    successors[block.name].update(
                        target for target in _targets(instruction)
                        if target in known
                    )
                if opcode in TERMINATORS:
                    break

        entry = names[0]
        reachable: set[str] = set()
        pending = [entry]
        while pending:
            name = pending.pop()
            if name in reachable:
                continue
            reachable.add(name)
            pending.extend(successors.get(name, set()) - reachable)
        predecessors: dict[str, set[str]] = {name: set() for name in names}
        for source, targets in successors.items():
            for target in targets:
                predecessors[target].add(source)
        dominators = {
            name: ({name} if name == entry or name not in reachable
                   else set(reachable))
            for name in names
        }
        changed = True
        while changed:
            changed = False
            for name in names:
                if name == entry or name not in reachable:
                    continue
                incoming = [
                    dominators[pred]
                    for pred in predecessors[name]
                    if pred in reachable
                ]
                common = set.intersection(*incoming) if incoming else set()
                updated = {name} | common
                if updated != dominators[name]:
                    dominators[name] = updated
                    changed = True
        return dominators

    def _verify_basic_block(
        self,
        block: BasicBlock,
        function: Function,
        defined: set[str],
        block_names: set[str],
        assigned: dict[str, tuple[str, int]],
        all_defined_names: Optional[set[str]] = None,
    ) -> None:
        instructions = list(block.instructions)
        definitions = all_defined_names or {
            instruction.dest.name
            for instruction in instructions
            if instruction.dest is not None
        }
        for index, instruction in enumerate(instructions):
            opcode = _opcode_name(instruction.opcode)
            operands = list(instruction.operands)
            for operand in operands:
                if operand.is_constant or operand.name in defined:
                    continue
                # Never-assigned names are implicit DSL inputs. Names assigned
                # in the function must dominate each use.
                if operand.name in definitions:
                    self._add_error(
                        f"value '{operand.name}' is used before definition",
                        function=function,
                        block=block,
                        index=index,
                        value=operand.name,
                        rule="def-before-use",
                    )
            if opcode in BINARY_OPS:
                if len(operands) != 2:
                    self._add_error(
                        f"binary operation {opcode} has {len(operands)} "
                        "operand(s), expected 2",
                        function=function,
                        block=block,
                        index=index,
                        rule="type-consistency",
                    )
                elif operands[0].dtype != operands[1].dtype:
                    self._add_error(
                        f"operand types differ: "
                        f"{operands[0].name}:{_type_name(operands[0].dtype)} "
                        f"vs {operands[1].name}:{_type_name(operands[1].dtype)}",
                        function=function,
                        block=block,
                        index=index,
                        rule="type-consistency",
                        level=ErrorLevel.WARNING,
                    )
            elif opcode in TERNARY_OPS:
                if len(operands) != 3:
                    self._add_error(
                        f"operation {opcode} has {len(operands)} operand(s), "
                        "expected 3",
                        function=function,
                        block=block,
                        index=index,
                        rule="type-consistency",
                    )
                elif any(
                    operand.dtype != operands[0].dtype
                    for operand in operands[1:]
                ):
                    self._add_error(
                        f"{opcode} operand types are inconsistent",
                        function=function,
                        block=block,
                        index=index,
                        rule="type-consistency",
                        level=ErrorLevel.WARNING,
                    )
            targets = _targets(instruction)
            if opcode in CONDITIONAL_BRANCHES:
                if not 1 <= len(operands) <= 2:
                    self._add_error(
                        f"conditional branch has {len(operands)} condition "
                        "operand(s), expected 1 or 2",
                        function=function,
                        block=block,
                        index=index,
                        rule="control-flow-integrity",
                    )
                if len(targets) != 2:
                    self._add_error(
                        f"conditional branch has {len(targets)} target(s), "
                        "expected exactly 2",
                        function=function,
                        block=block,
                        index=index,
                        rule="control-flow-integrity",
                    )
                self._check_targets(
                    targets, block_names, function, block, index
                )
            elif opcode in UNCONDITIONAL_BRANCHES:
                if len(targets) != 1:
                    self._add_error(
                        f"unconditional branch has {len(targets)} target(s), "
                        "expected exactly 1",
                        function=function,
                        block=block,
                        index=index,
                        rule="control-flow-integrity",
                    )
                self._check_targets(
                    targets, block_names, function, block, index
                )
            if opcode in TERMINATORS and index != len(instructions) - 1:
                self._add_error(
                    "instruction after terminator is unreachable",
                    function=function,
                    block=block,
                    index=index + 1,
                    rule="control-flow-integrity",
                )
            if instruction.dest is not None:
                destination = instruction.dest
                if destination.name in assigned:
                    first_block, first_index = assigned[destination.name]
                    self._add_error(
                        "value is assigned more than once; first assignment "
                        f"was in block '{first_block}', instruction "
                        f"#{first_index}",
                        function=function,
                        block=block,
                        index=index,
                        value=destination.name,
                        rule="ssa-validity",
                    )
                else:
                    assigned[destination.name] = (block.name, index)
                defined.add(destination.name)
        if not instructions:
            self._add_error(
                "basic block is empty and has no terminator",
                function=function,
                block=block,
                rule="block-termination",
            )
        elif _opcode_name(instructions[-1].opcode) not in TERMINATORS:
            self._add_error(
                "basic block must end with BR, BR_IF, or RETURN; found "
                f"{_opcode_name(instructions[-1].opcode)}",
                function=function,
                block=block,
                index=len(instructions) - 1,
                rule="block-termination",
            )

    def _check_targets(
        self,
        targets: Iterable[str],
        block_names: set[str],
        function: Function,
        block: BasicBlock,
        index: int,
    ) -> None:
        for target in targets:
            if target not in block_names:
                self._add_error(
                    f"branch target '{target}' does not exist",
                    function=function,
                    block=block,
                    index=index,
                    value=target,
                    rule="label-existence",
                )

    def _add_error(
        self,
        message: str,
        *,
        function: Optional[Function] = None,
        block: Optional[BasicBlock] = None,
        index: Optional[int] = None,
        value: Optional[str] = None,
        rule: Optional[str] = None,
        level: ErrorLevel = ErrorLevel.ERROR,
    ) -> None:
        self._errors.append(VerificationError(
            level=level,
            message=message,
            function_name=getattr(function, "name", None),
            block_name=getattr(block, "name", None),
            instruction_index=index,
            value_name=value,
            rule=rule,
        ))


def verify_ir(program: Program) -> tuple[bool, list[VerificationError]]:
    """Return ``(passed, issues)``; warnings do not fail verification."""
    issues = IRVerifier(program).verify_program()
    return not any(issue.level == ErrorLevel.ERROR for issue in issues), issues


def verify_or_raise(program: Program, phase: str) -> None:
    """Raise a detailed failure when ``program`` contains verifier errors."""
    passed, issues = verify_ir(program)
    if not passed:
        raise IRVerificationFailure(phase, issues)


OptimizationPass = Callable[[Program], Optional[Program]]


def run_optimization_pipeline(
    program: Program,
    passes: Iterable[OptimizationPass | object],
    *,
    verify_ir_enabled: bool = False,
) -> Program:
    """Run passes and optionally verify immediately before and after each."""
    current = program
    for number, optimization_pass in enumerate(passes, start=1):
        pass_name = getattr(
            optimization_pass,
            "name",
            getattr(optimization_pass, "__name__",
                    type(optimization_pass).__name__),
        )
        if verify_ir_enabled:
            verify_or_raise(current, f"before pass #{number} ({pass_name})")
        runner = getattr(optimization_pass, "run", optimization_pass)
        if not callable(runner):
            raise TypeError(f"optimization pass {pass_name!r} is not callable")
        accepts_input = bool(inspect.signature(runner).parameters)
        result = runner(current) if accepts_input else runner()
        if isinstance(result, PassResult):
            if result.data is None:
                detail = result.message or "pass returned no output data"
                raise RuntimeError(f"optimization pass {pass_name!r} failed: {detail}")
            current = result.data
        elif isinstance(result, Program):
            current = result
        elif result is not None and accepts_input:
            current = result
        if verify_ir_enabled:
            verify_or_raise(current, f"after pass #{number} ({pass_name})")
    return current


def _expect_object(data: object, location: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{location} must be a JSON object")
    return data


def _list_field(data: dict[str, Any], key: str, location: str) -> list[Any]:
    value = data.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{location}.{key} must be a JSON array")
    return value


def _required_text(data: dict[str, Any], key: str, location: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location}.{key} must be a non-empty string")
    return value


def _parse_type(raw: object) -> DataType:
    text = str(raw)
    for data_type in DataType:
        if text.lower() in {data_type.name.lower(), data_type.value.lower()}:
            return data_type
    raise ValueError(f"unknown IR data type {text!r}")


def _parse_opcode(raw: object) -> OpCode:
    text = str(raw)
    for opcode in OpCode:
        if text.lower() in {opcode.name.lower(), opcode.value.lower()}:
            return opcode
    raise ValueError(f"unknown IR opcode {text!r}")


def _load_value(data: object, location: str) -> Value:
    value_data = _expect_object(data, location)
    constant = value_data.get("is_constant", False)
    if not isinstance(constant, bool):
        raise ValueError(f"{location}.is_constant must be a boolean")
    dtype = value_data.get("dtype", DataType.FLOAT32.value)
    if not isinstance(dtype, str):
        raise ValueError(f"{location}.dtype must be a string")
    return Value(
        name=_required_text(value_data, "name", location),
        dtype=_parse_type(dtype),
        is_constant=constant,
        const_value=value_data.get("value"),
    )


def load_program_json(path: str | Path) -> Program:
    """Load the verifier's compact JSON representation into ScratchV IR."""
    with Path(path).open("r", encoding="utf-8") as source:
        data = _expect_object(json.load(source), "root")
    program = Program()
    for function_index, raw_function in enumerate(
        _list_field(data, "functions", "root")
    ):
        function_location = f"root.functions[{function_index}]"
        function_data = _expect_object(raw_function, function_location)
        function = Function(
            name=_required_text(function_data, "name", function_location),
            params=[
                _load_value(item, f"{function_location}.params[{index}]")
                for index, item in enumerate(
                    _list_field(function_data, "params", function_location)
                )
            ],
        )
        for block_index, raw_block in enumerate(
            _list_field(function_data, "blocks", function_location)
        ):
            block_location = f"{function_location}.blocks[{block_index}]"
            block_data = _expect_object(raw_block, block_location)
            block = BasicBlock(
                _required_text(block_data, "name", block_location)
            )
            for instruction_index, raw_instruction in enumerate(
                _list_field(block_data, "instructions", block_location)
            ):
                instruction_location = (
                    f"{block_location}.instructions[{instruction_index}]"
                )
                instruction_data = _expect_object(
                    raw_instruction, instruction_location
                )
                destination = instruction_data.get("dest")
                operands = _list_field(
                    instruction_data, "operands", instruction_location
                )
                target = instruction_data.get("target")
                if isinstance(target, list):
                    if not all(isinstance(item, str) for item in target):
                        raise ValueError(
                            f"{instruction_location}.target items must be strings"
                        )
                    target = ",".join(target)
                if target is not None and not isinstance(target, str):
                    raise ValueError(
                        f"{instruction_location}.target must be a string or array"
                    )
                block.add(Instruction(
                    opcode=_parse_opcode(_required_text(
                        instruction_data, "opcode", instruction_location
                    )),
                    dest=(
                        _load_value(destination, f"{instruction_location}.dest")
                        if destination is not None else None
                    ),
                    operands=[
                        _load_value(
                            item,
                            f"{instruction_location}.operands[{index}]",
                        )
                        for index, item in enumerate(operands)
                    ],
                    target=target,
                ))
            function.add_block(block)
        program.add_function(function)
    return program


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate ScratchV IR")
    parser.add_argument("ir_file", type=Path, help="IR program in JSON format")
    parser.add_argument(
        "--verify-ir", action="store_true",
        help="verify IR and return status 1 when invalid",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        program = load_program_json(args.ir_file)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot load IR file '{args.ir_file}': {exc}", file=sys.stderr)
        return 2
    if not args.verify_ir:
        print("IR loaded; verification disabled (use --verify-ir to enable it)")
        return 0
    passed, issues = verify_ir(program)
    if not passed:
        print(
            f"IR verification failed ({len(issues)} issue(s)):",
            file=sys.stderr,
        )
        for issue in issues:
            print(f"  {issue}", file=sys.stderr)
        return 1
    print("IR verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
