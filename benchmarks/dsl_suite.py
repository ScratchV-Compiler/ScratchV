"""Core implementation of the ScratchV DSL benchmark suite (topic 06).

The suite discovers ``*.dsl`` cases under configurable roots, validates the
``*.meta.json`` contract, and evaluates each case through four stages:
compile, assemble, instruction budget and semantic golden check.  It is
consumed by ``tests/test_dsl_suite.py`` (pytest gate) and by
``benchmarks/run_suite.py`` (JSON/Markdown report + exit code CLI).

Design constraints:
    - ``scratchv/**`` is read-only for this suite; compiler defects are
      recorded as ``xfail`` declarations in case metadata, never fixed here.
    - No silent pass: every stage exception is captured into a case outcome.
"""

from __future__ import annotations

import datetime
import json
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scratchv.backend._asm_parser import parse_asm


DEFAULT_ROOTS: tuple[str, ...] = ("benchmarks/cases", "tests/stress")
SCHEMA_VERSION: int = 1
SUITE_NAME: str = "dsl-benchmark"

STAGE_COMPILE: str = "compile"
STAGE_ASSEMBLE: str = "assemble"
STAGE_BUDGET: str = "budget"
STAGE_SEMANTIC: str = "semantic"
STAGE_META: str = "meta"

ALL_STAGES: tuple[str, ...] = (
    STAGE_COMPILE,
    STAGE_ASSEMBLE,
    STAGE_BUDGET,
    STAGE_SEMANTIC,
)

_STAGE_RESULT_KEYS: dict[str, str] = {
    STAGE_COMPILE: "compile_ok",
    STAGE_ASSEMBLE: "asm_encodable",
    STAGE_BUDGET: "inst_budget_ok",
    STAGE_SEMANTIC: "semantic_ok",
}

ASSERT_COMPILE_OK: str = "compile_ok"
ASSERT_ASM_ENCODABLE: str = "asm_encodable"
ASSERT_INST_BUDGET: str = "inst_budget"
ASSERT_SEMANTIC_GOLDEN: str = "semantic_golden"

DEFAULT_ASSERTIONS: tuple[str, ...] = (
    ASSERT_COMPILE_OK,
    ASSERT_ASM_ENCODABLE,
    ASSERT_INST_BUDGET,
    ASSERT_SEMANTIC_GOLDEN,
)

ORACLE_INTERPRETER: str = "interpreter"
ORACLE_EXECUTION: str = "execution"
ORACLE_NONE: str = "none"

FLOW_LINEAR: str = "linear"
FLOW_CONTROL: str = "control"

STATUS_PASS: str = "pass"
STATUS_FAIL: str = "fail"
STATUS_XFAIL: str = "xfail"
STATUS_XPASS: str = "xpass"
STATUS_SKIP: str = "skip"

ALLOWED_CATEGORIES: frozenset[str] = frozenset(
    {"arith", "nn", "control", "complex", "const", "stress"}
)

_CONTROL_KEYWORDS: tuple[str, ...] = (
    "if (", "else:", "endif", "while (", "endwhile", "for ", "endfor",
)

_REGISTER_RE = re.compile(r"^(a[0-7]|s[0-9]|s1[01])$")

_KNOWN_META_FIELDS: frozenset[str] = frozenset(
    {
        "schema_version", "description", "category", "flow", "oracle",
        "inputs", "input_registers", "expected_output_type",
        "expected_return", "rtol", "atol", "assertions",
        "max_instructions", "timeout_s", "xfail",
    }
)

_OP_PATTERN = (
    r"\b(add|sub|mul|div|relu|gelu|exp|neg|"
    r"matmul|dot|maxpool|softmax)\(([^)]+)"
)

_INPUT_KEYWORDS: frozenset[str] = frozenset(
    {
        "add", "sub", "mul", "div", "relu", "gelu", "exp", "neg",
        "matmul", "dot", "maxpool", "softmax", "return", "for",
        "endfor", "if", "else", "endif", "while", "endwhile",
        "m", "n", "k", "rows", "cols", "inner", "len",
        "axis", "kernel", "stride", "padding",
        "out_channels", "kernel_size",
        "transA", "transB", "alpha", "beta",
        "i", "j", "t1", "t2", "t3", "t4",
        "acc", "sum", "tmp",
    }
)


class DSLSuiteError(Exception):
    """Base error for suite configuration problems."""


class CaseSpecError(DSLSuiteError):
    """Raised when a case specification cannot be loaded at all."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XFailSpec:
    """Declared, located and owned expected-failure description."""

    stages: tuple[str, ...]
    reason: str
    owner: str
    strict: bool = False


@dataclass(frozen=True)
class CaseSpec:
    """Immutable case specification produced at discovery time."""

    case_id: str
    pytest_id: str
    name: str
    root: Path
    dsl_path: Path
    meta_path: Path | None
    expected_path: Path | None
    description: str
    category: str
    flow: str
    oracle: str
    inputs: dict[str, Any]
    input_registers: dict[str, str]
    expected_return: Any
    expected_text: str
    rtol: float
    atol: float
    assertions: tuple[str, ...]
    max_instructions: int | None
    timeout_s: float
    xfail: XFailSpec | None
    skip_reason: str | None
    meta_errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass
class CompileOutcome:
    ok: bool
    asm_text: str
    output_path: Path | None
    ir_instruction_count: int
    duration_s: float
    error: str | None


@dataclass
class AssembleOutcome:
    ok: bool
    binary_len: int
    instruction_count: int | None
    duration_s: float
    error: str | None


@dataclass
class BudgetOutcome:
    ok: bool | None
    instruction_count: int | None
    limit: int | None
    skipped: bool
    error: str | None


@dataclass
class SemanticOutcome:
    ok: bool | None
    oracle: str
    expected: Any
    actual: Any
    duration_s: float
    blocked_reason: str | None
    error: str | None


@dataclass
class CaseOutcome:
    case_id: str
    pytest_id: str
    status: str
    compile: CompileOutcome
    assemble: AssembleOutcome | None
    budget: BudgetOutcome | None
    semantic: SemanticOutcome | None
    error_stage: str | None
    error: str | None
    xfail: XFailSpec | None


@dataclass
class SuiteReport:
    """Aggregated suite result with JSON/Markdown rendering."""

    results: list[CaseOutcome]
    roots: tuple[str, ...]
    compiler: dict[str, str]
    generated_at: str
    specs: dict[str, CaseSpec] = field(default_factory=dict)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.status == STATUS_PASS)

    @property
    def xfail_count(self) -> int:
        return sum(1 for r in self.results if r.status == STATUS_XFAIL)

    @property
    def xpass_count(self) -> int:
        return sum(1 for r in self.results if r.status == STATUS_XPASS)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if r.status == STATUS_FAIL)

    @property
    def skip_count(self) -> int:
        return sum(1 for r in self.results if r.status == STATUS_SKIP)

    def summary(self) -> dict[str, int]:
        return {
            "total": len(self.results),
            "passed": self.pass_count,
            "xfailed": self.xfail_count,
            "xpassed": self.xpass_count,
            "failed": self.fail_count,
            "skipped": self.skip_count,
        }

    def stage_summary(self) -> dict[str, dict[str, int]]:
        """Per-stage verdict aggregation with xfail attribution.

        ``passed`` counts true verdicts not excused by an unexpected pass,
        ``xfail`` counts failures excused for that stage by the case's
        declaration, ``failed`` counts unexcused failures, ``xpass`` counts
        declared stages of a case that unexpectedly passed, and ``blocked``
        counts stages that never executed (skipped case, earlier failure, or
        the assertion was disabled).
        """
        counts: dict[str, dict[str, int]] = {
            stage: {
                "passed": 0, "xfail": 0, "failed": 0,
                "xpass": 0, "blocked": 0,
            }
            for stage in ALL_STAGES
        }
        for r in self.results:
            declared = set(r.xfail.stages) if r.xfail else set()
            stages = _stage_map(r)
            for stage, key in _STAGE_RESULT_KEYS.items():
                bucket = counts[stage]
                verdict = stages[key]
                declared_here = stage in declared
                if verdict is None:
                    bucket["blocked"] += 1
                elif verdict is False:
                    if declared_here and r.status == STATUS_XFAIL:
                        bucket["xfail"] += 1
                    else:
                        bucket["failed"] += 1
                elif declared_here and r.status == STATUS_XPASS:
                    bucket["xpass"] += 1
                else:
                    bucket["passed"] += 1
        return counts

    def owner_summary(self) -> dict[str, dict[str, int]]:
        """Per-owner xfail ledger, sorted by owner for stable output."""
        owners: dict[str, dict[str, int]] = {}
        for r in self.results:
            if r.xfail is None:
                continue
            bucket = owners.setdefault(
                r.xfail.owner,
                {"declared": 0, "xfailed": 0, "xpassed": 0, "failed": 0},
            )
            bucket["declared"] += 1
            if r.status == STATUS_XFAIL:
                bucket["xfailed"] += 1
            elif r.status == STATUS_XPASS:
                bucket["xpassed"] += 1
            elif r.status == STATUS_FAIL:
                bucket["failed"] += 1
        return dict(sorted(owners.items()))

    def xpassed_cases(self) -> list[str]:
        """Case ids whose declared xfail stages unexpectedly passed."""
        return sorted(
            r.case_id for r in self.results if r.status == STATUS_XPASS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "suite": SUITE_NAME,
            "generated_at": self.generated_at,
            "roots": list(self.roots),
            "compiler": dict(self.compiler),
            "summary": self.summary(),
            "stage_summary": self.stage_summary(),
            "owner_summary": self.owner_summary(),
            "xpassed_cases": self.xpassed_cases(),
            "results": [self._result_to_dict(r) for r in self.results],
        }

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def to_markdown(self) -> str:
        summary = self.summary()
        lines = [
            "# ScratchV DSL Benchmark Suite",
            "",
            f"- generated_at: {self.generated_at}",
            f"- roots: {', '.join(self.roots)}",
            "- compiler: "
            + ", ".join(f"{k}={v}" for k, v in self.compiler.items()),
            "",
            "## Summary",
            "",
            "| total | passed | xfailed | xpassed | failed | skipped |",
            "|-------|--------|---------|---------|--------|---------|",
            "| {total} | {passed} | {xfailed} | {xpassed} | {failed} | "
            "{skipped} |".format(**summary),
        ]
        if summary["xpassed"]:
            lines += [
                "",
                f"**WARNING: {summary['xpassed']} unexpected pass(es) "
                "(xpass)** — declared xfail stages now pass; remove the "
                "obsolete declaration or run with `--strict-xfail`.",
            ]
        lines += [
            "",
            "## Stage summary",
            "",
            "| stage | passed | xfail | failed | xpass | blocked |",
            "|-------|--------|-------|--------|-------|---------|",
        ]
        for stage, counts in self.stage_summary().items():
            lines.append(
                f"| {stage} | {counts['passed']} | {counts['xfail']} | "
                f"{counts['failed']} | {counts['xpass']} | "
                f"{counts['blocked']} |"
            )
        owners = self.owner_summary()
        lines += [
            "",
            "## Owner summary",
            "",
            "| owner | declared | xfailed | xpassed | failed |",
            "|-------|----------|---------|---------|--------|",
        ]
        if owners:
            for owner, counts in owners.items():
                lines.append(
                    f"| {owner} | {counts['declared']} | "
                    f"{counts['xfailed']} | {counts['xpassed']} | "
                    f"{counts['failed']} |"
                )
        else:
            lines.append("| - | - | - | - | - |")
        lines += [
            "",
            "## Cases",
            "",
            "| case_id | status | compile | asm | inst | semantic | oracle | "
            "xfail reason |",
            "|---------|--------|---------|-----|------|----------|--------|"
            "-------------|",
        ]
        for r in self.results:
            spec = self.specs.get(r.case_id)
            oracle = spec.oracle if spec else "-"
            reason = r.xfail.reason if r.xfail else ""
            reason = reason.replace("|", "\\|")
            lines.append(
                f"| {r.case_id} | {r.status} | {_mark(r.compile.ok)} | "
                f"{_mark(r.assemble.ok) if r.assemble else '-'} | "
                f"{_mark(r.budget.ok) if r.budget else '-'} | "
                f"{_mark(r.semantic.ok) if r.semantic else '-'} | "
                f"{oracle} | {reason} |"
            )

        problems = [r for r in self.results if r.status == STATUS_FAIL]
        lines += ["", "## Failures", ""]
        if problems:
            groups: dict[str, list[CaseOutcome]] = {}
            for result in problems:
                groups.setdefault(result.error_stage or "unknown", []).append(
                    result
                )
            for stage, staged in groups.items():
                lines.append(f"### {stage}")
                lines.append("")
                for result in staged:
                    lines.append(f"- **{result.case_id}**: {result.error}")
                lines.append("")
        else:
            lines.append("No hard failures.")

        xpassed = [r for r in self.results if r.status == STATUS_XPASS]
        expected_failures = [
            r for r in self.results if r.status == STATUS_XFAIL
        ]
        lines += ["", "## Unexpected passes (xpass — action required)", ""]
        if xpassed:
            for result in xpassed:
                stages = (
                    ", ".join(result.xfail.stages) if result.xfail else "-"
                )
                owner = result.xfail.owner if result.xfail else "-"
                lines.append(
                    f"- **{result.case_id}**: declared stages now pass "
                    f"(owner={owner}, stages={stages})"
                )
        else:
            lines.append("None.")
        lines += ["", "## Expected failures (xfail)", ""]
        if expected_failures:
            for result in expected_failures:
                which = result.error_stage or "unknown"
                lines.append(f"- {result.case_id} ({which})")
        else:
            lines.append("None.")

        lines += [
            "",
            "## 数据缺口与缺陷登记",
            "",
            "- C1 backend/regalloc: the linear-scan emitter keeps branch "
            "targets in comments instead of operands (greedy path emits them "
            "correctly); affected assemble/semantic stages are xfailed.",
            "- C2 backend/asm-emitter: constant materialisation emits invalid "
            "`mv rd, imm`.",
            "- C3 backend/regalloc: spill/reload stack slots read uninitialised "
            "frames (reg_pressure_32 returns 178 instead of 64).",
            "- C4 backend/op-lowering: matmul/dot/maxpool/softmax are not "
            "lowered; execution oracle would expose it.",
            "- C5 backend/numeric semantics: float division/activations are "
            "executed as integer instructions.",
            "- C6 backend/regalloc: input variable to physical register mapping "
            "has no published contract; declared in `input_registers`.",
            "- B1 verification: `DSLInterpreter` does not implement control "
            "flow; control cases use the execution oracle.",
            "- D1/D2: five control cases had no golden and two while loops had "
            "no induction step; fixed in this suite's data files.",
            "",
        ]
        return "\n".join(lines)

    def save_markdown(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown())

    def _result_to_dict(self, r: CaseOutcome) -> dict[str, Any]:
        spec = self.specs.get(r.case_id)
        skipped = r.status == STATUS_SKIP
        asm_count: int | None = None
        if r.assemble is not None:
            asm_count = r.assemble.instruction_count
        elif r.budget is not None:
            asm_count = r.budget.instruction_count

        expected: Any = None
        actual: Any = None
        if r.semantic is not None:
            expected = r.semantic.expected
            actual = r.semantic.actual
        if expected is None and spec is not None:
            expected = (
                spec.expected_return
                if spec.expected_return is not None
                else (spec.expected_text or None)
            )

        return {
            "case_id": r.case_id,
            "pytest_id": r.pytest_id,
            "category": spec.category if spec else "unknown",
            "flow": spec.flow if spec else "unknown",
            "oracle": spec.oracle if spec else "unknown",
            "status": r.status,
            "stages": _stage_map(r),
            "metrics": {
                "ir_instructions": None if skipped
                else r.compile.ir_instruction_count,
                "asm_instructions": None if skipped else asm_count,
                "compile_time_s": 0.0 if skipped else r.compile.duration_s,
                "assemble_time_s": 0.0 if skipped or r.assemble is None
                else r.assemble.duration_s,
                "semantic_time_s": 0.0 if skipped or r.semantic is None
                else r.semantic.duration_s,
            },
            "expected": _jsonable(expected),
            "actual": None if skipped else _jsonable(actual),
            "xfail": _xfail_to_dict(r.xfail),
            "error_stage": r.error_stage,
            "error": r.error,
            "warnings": list(spec.warnings) if spec else [],
        }


def _mark(value: bool | None) -> str:
    if value is None:
        return "-"
    return "ok" if value else "FAIL"


def _stage_map(r: CaseOutcome) -> dict[str, bool | None]:
    """Per-stage verdicts with the report's ``null`` semantics.

    Skipped cases expose ``None`` for every stage so that the JSON schema and
    the aggregate summaries agree on what "did not execute" means.
    """
    skipped = r.status == STATUS_SKIP
    return {
        "compile_ok": None if skipped else r.compile.ok,
        "asm_encodable": None if skipped or r.assemble is None
        else r.assemble.ok,
        "inst_budget_ok": None if skipped or r.budget is None
        else r.budget.ok,
        "semantic_ok": None if skipped or r.semantic is None
        else r.semantic.ok,
    }


def _xfail_to_dict(spec: XFailSpec | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    return {
        "stages": list(spec.stages),
        "reason": spec.reason,
        "owner": spec.owner,
        "strict": spec.strict,
    }


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def infer_flow(source: str) -> str:
    """Return ``control`` when the source uses extended control flow."""
    return (
        FLOW_CONTROL
        if any(keyword in source for keyword in _CONTROL_KEYWORDS)
        else FLOW_LINEAR
    )


def default_inputs(source: str, *, fill: float = 1.0) -> dict[str, np.ndarray]:
    """Default vector inputs for the interpreter oracle (``[1, 2, 3, 4]``)."""
    base = np.array([fill, 2.0 * fill, 3.0 * fill, 4.0 * fill], dtype=np.float32)
    names: set[str] = set()
    for match in re.finditer(_OP_PATTERN, source):
        for arg in match.group(2).split(","):
            arg = arg.strip().split(":")[0].strip()
            if arg and not arg[0].isdigit():
                names.add(arg)
    names = {
        name for name in names
        if name.lower() not in _INPUT_KEYWORDS and not name.startswith("_")
    }
    return {name: base.copy() for name in names}


def _as_vector(value: Any) -> np.ndarray | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, np.ndarray):
        try:
            return np.asarray(value, dtype=float).ravel()
        except (TypeError, ValueError):
            return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.array([float(value)], dtype=float)
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value, dtype=float).ravel()
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parts = [p for p in re.split(r"[,;\s]+", text.strip("[]")) if p]
        try:
            return np.array([float(p) for p in parts], dtype=float)
        except ValueError:
            return None
    return None


def compare_values(
    actual: Any,
    expected: Any,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> bool:
    """Numerically compare scalars, vectors or stringified vectors."""
    if isinstance(actual, str) and isinstance(expected, str):
        if actual.strip() == expected.strip():
            return True
    vector_a = _as_vector(actual)
    vector_e = _as_vector(expected)
    if vector_a is None or vector_e is None:
        return False
    if vector_a.shape != vector_e.shape:
        return False
    if vector_a.size == 0:
        return True
    return bool(np.allclose(
        vector_a, vector_e, rtol=rtol, atol=atol, equal_nan=True,
    ))


def count_asm_instructions(asm_text: str) -> int:
    """Count literal instructions with the shared ``parse_asm`` semantics."""
    return sum(
        1 for line in parse_asm(asm_text)
        if line.opcode and not line.is_directive
    )


def _stringify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=6, suppress_small=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return _stringify(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not (
        isinstance(value, bool)
    )


# ---------------------------------------------------------------------------
# Metadata loading and validation
# ---------------------------------------------------------------------------


def infer_category(name: str, source: str, flow: str, root: Path) -> str:
    if root.name == "stress" or name.startswith("stress"):
        return "stress"
    if flow == FLOW_CONTROL:
        return "control"
    if "constant" in name:
        return "const"
    lowered = source.lower()
    if any(
        op in lowered
        for op in ("matmul(", "dot(", "relu(", "gelu(", "softmax(", "maxpool(")
    ):
        return "nn"
    if len(re.findall(r"\w+\s*=\s*\w+\(", source)) >= 3:
        return "complex"
    if any(op in lowered for op in ("add(", "sub(", "mul(", "div(")):
        return "arith"
    return "unknown"


def _parse_xfail(value: Any, errors: list[str]) -> XFailSpec | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        errors.append("xfail must be a JSON object")
        return None
    stages_raw = value.get("stages")
    if isinstance(stages_raw, str):
        stages: tuple[str, ...] = (stages_raw,)
    elif isinstance(stages_raw, list) and all(
        isinstance(s, str) for s in stages_raw
    ):
        stages = tuple(stages_raw)
    else:
        errors.append(
            "xfail.stages must be a string or a list of strings"
        )
        stages = ()
    if not stages:
        errors.append("xfail.stages must not be empty")
    for stage in stages:
        if stage not in ALL_STAGES:
            errors.append(
                f"xfail.stages contains unknown stage '{stage}' "
                f"(expected one of {list(ALL_STAGES)})"
            )
    reason = value.get("reason", "")
    owner = value.get("owner", "")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("xfail.reason must be a non-empty string")
    if not isinstance(owner, str) or not owner.strip():
        errors.append("xfail.owner must be a non-empty string")
    strict = value.get("strict", False)
    if not isinstance(strict, bool):
        errors.append("xfail.strict must be a boolean")
        strict = False
    return XFailSpec(
        stages=stages,
        reason=reason if isinstance(reason, str) else "",
        owner=owner if isinstance(owner, str) else "",
        strict=strict,
    )


def _validate_inputs(value: Any, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append("inputs must be a JSON object")
        return {}
    for key, item in value.items():
        if _is_number(item):
            continue
        if (
            isinstance(item, list)
            and all(_is_number(element) for element in item)
        ):
            continue
        errors.append(
            f"inputs['{key}'] must be a number or a one-dimensional list "
            "of numbers"
        )
    return dict(value)


def _validate_input_registers(
    value: Any,
    inputs: dict[str, Any],
    oracle: str,
    errors: list[str],
) -> dict[str, str]:
    if not isinstance(value, dict):
        errors.append("input_registers must be a JSON object")
        return {}
    registers: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(item, str):
            errors.append(
                f"input_registers['{key}'] must be a register name string"
            )
            continue
        if not _REGISTER_RE.match(item):
            errors.append(
                f"input_registers['{key}'] = '{item}' does not match "
                r"^(a[0-7]|s[0-9]|s1[01])$"
            )
            continue
        registers[str(key)] = item
    if oracle == ORACLE_EXECUTION and inputs and not registers:
        errors.append(
            "input_registers is required when oracle=execution and inputs "
            "is non-empty"
        )
    if registers and inputs and set(registers) != set(inputs):
        errors.append(
            "input_registers keys must match inputs keys "
            f"(inputs={sorted(inputs)}, registers={sorted(registers)})"
        )
    return registers


def _validate_assertions(value: Any, errors: list[str]) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_ASSERTIONS
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        errors.append("assertions must be a list of strings")
        return DEFAULT_ASSERTIONS
    unknown = [item for item in value if item not in DEFAULT_ASSERTIONS]
    if unknown:
        errors.append(f"assertions contains unknown entries: {unknown}")
    valid = tuple(item for item in value if item in DEFAULT_ASSERTIONS)
    if not valid:
        errors.append("assertions must contain at least one known assertion")
        return DEFAULT_ASSERTIONS
    return valid


def _validate_expected_return(
    value: Any, errors: list[str]
) -> Any:
    if value is None:
        return None
    if _is_number(value):
        return value
    if (
        isinstance(value, list)
        and all(_is_number(element) for element in value)
    ):
        return value
    errors.append("expected_return must be a number or a list of numbers")
    return None


def load_case_spec(dsl_path: Path, root: Path) -> CaseSpec:
    """Load and validate one case; contract violations land in ``meta_errors``."""
    dsl_path = Path(dsl_path)
    root = Path(root)
    relative = dsl_path.relative_to(root)
    rel_stem = relative.with_suffix("").as_posix()
    case_id = f"{root.name}/{rel_stem}"
    pytest_id = case_id.replace("/", "-")

    source = dsl_path.read_text()
    inferred_flow = infer_flow(source)
    errors: list[str] = []
    warnings: list[str] = []

    meta_path = dsl_path.with_suffix(".meta.json")
    expected_path = dsl_path.with_suffix(".expected")
    expected_text = ""
    if expected_path.exists():
        expected_text = expected_path.read_text().strip()

    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"meta.json parse error: {exc}")
            loaded = {}
        if not isinstance(loaded, dict):
            errors.append("meta.json must decode to a JSON object")
            loaded = {}
        meta = loaded
    else:
        meta_path = None
        errors.append("meta.json not found")

    for key in meta:
        if key not in _KNOWN_META_FIELDS:
            warnings.append(f"unknown meta field: {key}")

    schema_version = meta.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        errors.append(
            f"schema_version {schema_version!r} is not supported "
            f"(expected {SCHEMA_VERSION})"
        )

    description = meta.get("description")
    if description is None:
        desc_path = dsl_path.with_suffix(".desc")
        if desc_path.exists():
            description = desc_path.read_text().strip()
    if not isinstance(description, str) or not description.strip():
        errors.append("description must be a non-empty string")
        description = description if isinstance(description, str) else ""

    meta_flow = meta.get("flow")
    if meta_flow is None:
        flow = inferred_flow
    elif meta_flow not in (FLOW_LINEAR, FLOW_CONTROL):
        errors.append(f"flow must be '{FLOW_LINEAR}' or '{FLOW_CONTROL}'")
        flow = inferred_flow
    else:
        flow = meta_flow
        if flow != inferred_flow:
            errors.append(
                f"flow conflict: meta declares '{flow}' but the source "
                f"implies '{inferred_flow}'"
            )

    inferred_category = infer_category(
        dsl_path.stem, source, flow, root,
    )
    category = meta.get("category", inferred_category)
    if not isinstance(category, str) or category not in ALLOWED_CATEGORIES:
        errors.append(
            f"category {category!r} is not one of "
            f"{sorted(ALLOWED_CATEGORIES)}"
        )
        category = inferred_category

    oracle = meta.get("oracle")
    if oracle is None:
        oracle = (
            ORACLE_EXECUTION if flow == FLOW_CONTROL else ORACLE_INTERPRETER
        )
        warnings.append(f"oracle not declared; inferred '{oracle}'")
    elif oracle not in (ORACLE_INTERPRETER, ORACLE_EXECUTION, ORACLE_NONE):
        errors.append(
            f"oracle {oracle!r} is not one of "
            f"'{ORACLE_INTERPRETER}', '{ORACLE_EXECUTION}', '{ORACLE_NONE}'"
        )
        oracle = ORACLE_NONE
    if oracle == ORACLE_INTERPRETER and flow != FLOW_LINEAR:
        errors.append(
            "oracle/flow conflict: oracle=interpreter requires flow=linear"
        )
    if oracle == ORACLE_INTERPRETER:
        warnings.append(
            "interpreter oracle validates DSL semantics only; generated "
            "code is not executed (blind spots: C2 constant materialisation, "
            "C4 op lowering, C5 float semantics)"
        )

    inputs = _validate_inputs(meta.get("inputs", {}), errors)
    input_registers = _validate_input_registers(
        meta.get("input_registers", {}), inputs, oracle, errors,
    )

    expected_output_type = meta.get("expected_output_type", "return_value")
    if expected_output_type != "return_value":
        errors.append(
            "expected_output_type must be 'return_value' for this schema "
            f"version (got {expected_output_type!r})"
        )

    expected_return = _validate_expected_return(
        meta.get("expected_return"), errors,
    )
    if (
        oracle != ORACLE_NONE
        and expected_return is None
        and not (oracle == ORACLE_INTERPRETER and expected_text)
    ):
        errors.append(
            "expected_return is required when oracle != none "
            "(interpreter cases may fall back to a non-empty .expected)"
        )

    rtol = meta.get("rtol", 1e-3)
    atol = meta.get("atol", 1e-6)
    for name, value in (("rtol", rtol), ("atol", atol)):
        if not _is_number(value) or float(value) < 0:
            errors.append(f"{name} must be a non-negative number")
            if name == "rtol":
                rtol = 1e-3
            else:
                atol = 1e-6

    assertions = _validate_assertions(meta.get("assertions"), errors)

    max_instructions = meta.get("max_instructions")
    if max_instructions is not None:
        if (
            not isinstance(max_instructions, int)
            or isinstance(max_instructions, bool)
            or max_instructions <= 0
        ):
            errors.append("max_instructions must be a positive integer")
            max_instructions = None

    timeout_s = meta.get("timeout_s", 30.0)
    if not _is_number(timeout_s) or float(timeout_s) <= 0:
        errors.append("timeout_s must be a positive number")
        timeout_s = 30.0

    xfail = _parse_xfail(meta.get("xfail"), errors)
    if oracle == ORACLE_NONE:
        if xfail is None:
            errors.append("oracle=none requires an explicit xfail declaration")
        elif STAGE_SEMANTIC not in xfail.stages:
            errors.append(
                "oracle=none requires xfail.stages to include 'semantic'"
            )

    return CaseSpec(
        case_id=case_id,
        pytest_id=pytest_id,
        name=dsl_path.stem,
        root=root,
        dsl_path=dsl_path,
        meta_path=meta_path,
        expected_path=expected_path if expected_path.exists() else None,
        description=description,
        category=category,
        flow=flow,
        oracle=oracle,
        inputs=inputs,
        input_registers=input_registers,
        expected_return=expected_return,
        expected_text=expected_text,
        rtol=float(rtol),
        atol=float(atol),
        assertions=assertions,
        max_instructions=max_instructions,
        timeout_s=float(timeout_s),
        xfail=xfail,
        skip_reason=None,
        meta_errors=tuple(errors),
        warnings=tuple(warnings),
    )


def discover_cases(
    roots: Sequence[str | Path] = DEFAULT_ROOTS,
    *,
    verbose: bool = False,
) -> list[CaseSpec]:
    """Discover cases under *roots* following the suite's skip rules."""
    cases: list[CaseSpec] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            if verbose:
                print(f"warning: root not found, skipping: {root_path}")
            continue
        for dsl_path in root_path.rglob("*.dsl"):
            relative = dsl_path.relative_to(root_path)
            if dsl_path.name.startswith("_"):
                continue
            if "fixtures" in relative.parts[:-1]:
                continue
            try:
                spec = load_case_spec(dsl_path, root_path)
            except Exception as exc:  # pragma: no cover - defensive
                spec = _broken_spec(dsl_path, root_path, exc)
            skip_path = dsl_path.with_suffix(".skip")
            if skip_path.exists():
                reason = skip_path.read_text().strip()
                spec = replace(
                    spec, skip_reason=reason or "skipped by marker file",
                )
            cases.append(spec)
    cases.sort(key=lambda case: case.case_id)
    return cases


def _broken_spec(dsl_path: Path, root: Path, exc: Exception) -> CaseSpec:
    relative = dsl_path.relative_to(root)
    rel_stem = relative.with_suffix("").as_posix()
    case_id = f"{root.name}/{rel_stem}"
    return CaseSpec(
        case_id=case_id,
        pytest_id=case_id.replace("/", "-"),
        name=dsl_path.stem,
        root=root,
        dsl_path=dsl_path,
        meta_path=None,
        expected_path=None,
        description="",
        category="unknown",
        flow=FLOW_LINEAR,
        oracle=ORACLE_NONE,
        inputs={},
        input_registers={},
        expected_return=None,
        expected_text="",
        rtol=1e-3,
        atol=1e-6,
        assertions=DEFAULT_ASSERTIONS,
        max_instructions=None,
        timeout_s=30.0,
        xfail=None,
        skip_reason=None,
        meta_errors=(f"case loading failed: {type(exc).__name__}: {exc}",),
    )


# ---------------------------------------------------------------------------
# Execution oracle
# ---------------------------------------------------------------------------


_REG_NUMS: dict[str, int] = {
    "a0": 10, "a1": 11, "a2": 12, "a3": 13,
    "a4": 14, "a5": 15, "a6": 16, "a7": 17,
    "s0": 8, "s1": 9,
    "s2": 18, "s3": 19, "s4": 20, "s5": 21,
    "s6": 22, "s7": 23, "s8": 24, "s9": 25,
    "s10": 26, "s11": 27,
}


EXECUTION_MAX_STEPS: int = 100_000_000


def _machine_pc(machine: Any) -> int:
    """Read the PC, tolerating scalar and one-element-array TinyFive PCs.

    ``ProfiledMachine.pc`` assumes the PC is indexable, which breaks after a
    ``jalr`` stores a NumPy scalar (pre-existing simulator adapter issue).
    The suite reads the raw attribute instead of touching ``scratchv/**``.
    """
    raw = machine._machine.pc
    if hasattr(raw, "__len__"):
        return int(raw[0])
    return int(raw)


class ExecutionOracle:
    """Execute assembled RISC-V and read the ``a0`` return register.

    Execution is stepped instruction by instruction so that loops run to
    completion (the previous ``instructions=len(words)`` bound truncated any
    dynamic execution longer than the static code size) while a real wall
    clock ``timeout_s`` still bounds runaway programs.  The return address
    register ``ra`` is preloaded with a sentinel just past the code, so the
    compiler's ``jalr zero, ra`` epilogue halts the machine instead of
    jumping back to address 0.
    """

    def __init__(
        self,
        *,
        mem_size: int = 128 * 1024 * 1024,
        max_steps: int = EXECUTION_MAX_STEPS,
    ) -> None:
        self.mem_size = mem_size
        self.max_steps = max_steps

    def available(self) -> bool:
        try:
            from scratchv.simulator.tinyfive import ProfiledMachine
        except ImportError:
            return False
        return bool(ProfiledMachine(mem_size=4096).available)

    def execute(
        self,
        asm_text: str,
        input_registers: dict[str, str],
        inputs: dict[str, object],
        *,
        timeout_s: float = 5.0,
    ) -> SemanticOutcome:
        started = time.perf_counter()

        def elapsed() -> float:
            return time.perf_counter() - started

        if not self.available():
            return SemanticOutcome(
                ok=None, oracle=ORACLE_EXECUTION, expected=None, actual=None,
                duration_s=0.0, blocked_reason="tinyfive not installed",
                error=None,
            )
        try:
            from scratchv.backend.riscv_encoder import assemble_to_binary
            from scratchv.simulator.tinyfive import ProfiledMachine

            binary = assemble_to_binary(asm_text)
            if not binary:
                return SemanticOutcome(
                    ok=False, oracle=ORACLE_EXECUTION, expected=None,
                    actual=None, duration_s=elapsed(),
                    blocked_reason=None,
                    error="assembler produced empty binary",
                )
            words = [
                int.from_bytes(binary[i:i + 4], "little")
                for i in range(0, len(binary), 4)
            ]
            machine = ProfiledMachine(mem_size=self.mem_size)
            machine.load_binary(words, origin=0)
            for name, register in input_registers.items():
                value = float(inputs.get(name, 0))
                if value != int(value):
                    return SemanticOutcome(
                        ok=None, oracle=ORACLE_EXECUTION, expected=None,
                        actual=None, duration_s=elapsed(),
                        blocked_reason=(
                            "non-integer input not supported by integer "
                            "register semantics"
                        ),
                        error=None,
                    )
                machine.set_reg(_REG_NUMS[register], int(value))
            code_end = len(words) * 4
            machine.set_reg(1, code_end)
            steps = 0
            while 0 <= _machine_pc(machine) < code_end:
                if steps >= self.max_steps:
                    return SemanticOutcome(
                        ok=False, oracle=ORACLE_EXECUTION, expected=None,
                        actual=machine.get_reg(10), duration_s=elapsed(),
                        blocked_reason=None,
                        error=(
                            "execution step limit exceeded "
                            f"({self.max_steps} instructions)"
                        ),
                    )
                if elapsed() >= timeout_s:
                    return SemanticOutcome(
                        ok=False, oracle=ORACLE_EXECUTION, expected=None,
                        actual=machine.get_reg(10), duration_s=elapsed(),
                        blocked_reason=None,
                        error=(
                            f"execution timeout after {timeout_s:g}s "
                            f"({steps} instructions executed)"
                        ),
                    )
                pc_before = _machine_pc(machine)
                machine.run(instructions=1, start=pc_before, strict=True)
                steps += 1
                if _machine_pc(machine) == pc_before:
                    return SemanticOutcome(
                        ok=False, oracle=ORACLE_EXECUTION, expected=None,
                        actual=machine.get_reg(10), duration_s=elapsed(),
                        blocked_reason=None,
                        error=(
                            "unsupported instruction at "
                            f"pc={pc_before:#x} (decoder made no progress)"
                        ),
                    )
            final_pc = _machine_pc(machine)
            if final_pc != code_end:
                return SemanticOutcome(
                    ok=False, oracle=ORACLE_EXECUTION, expected=None,
                    actual=machine.get_reg(10), duration_s=elapsed(),
                    blocked_reason=None,
                    error=(
                        f"pc left the code region: {final_pc:#x} "
                        f"(code_end={code_end:#x})"
                    ),
                )
            actual = machine.get_reg(10)
            return SemanticOutcome(
                ok=None, oracle=ORACLE_EXECUTION, expected=None,
                actual=actual, duration_s=elapsed(),
                blocked_reason=None, error=None,
            )
        except Exception as exc:
            return SemanticOutcome(
                ok=False, oracle=ORACLE_EXECUTION, expected=None, actual=None,
                duration_s=elapsed(), blocked_reason=None,
                error=f"{type(exc).__name__}: {exc}",
            )


def _first_bad_asm_line(
    asm_text: str,
) -> tuple[int, str, str] | None:
    from scratchv.backend.riscv_encoder import RISCVAEncoder

    for index, line in enumerate(asm_text.splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith((".", "#")) or ":" in stripped:
            continue
        try:
            RISCVAEncoder().assemble(stripped + "\n")
        except Exception as exc:
            return index + 1, stripped, f"{type(exc).__name__}: {exc}"
    return None


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------


def _empty_compile_outcome() -> CompileOutcome:
    return CompileOutcome(
        ok=False, asm_text="", output_path=None,
        ir_instruction_count=0, duration_s=0.0, error=None,
    )


class DSLSuiteRunner:
    """Discover, execute and grade DSL benchmark cases."""

    def __init__(
        self,
        roots: tuple[str | Path, ...] = DEFAULT_ROOTS,
        *,
        workdir: str | Path | None = None,
        backend: str = "riscv",
        optimize_level: str = "all",
        reg_alloc: str = "linear",
        timeout_s: float = 30.0,
        strict_xfail: bool = False,
        verbose: bool = False,
    ) -> None:
        self._roots = tuple(Path(root) for root in roots)
        self.root_strings: tuple[str, ...] = tuple(str(root) for root in roots)
        if workdir is None:
            self.workdir = Path(tempfile.mkdtemp(prefix="dsl_suite_"))
            self._owns_workdir = True
        else:
            self.workdir = Path(workdir)
            self.workdir.mkdir(parents=True, exist_ok=True)
            self._owns_workdir = False
        self.backend = backend
        self.optimize_level = optimize_level
        self.reg_alloc = reg_alloc
        self.timeout_s = timeout_s
        self.strict_xfail = strict_xfail
        self.verbose = verbose
        self._cases: list[CaseSpec] | None = None

    @property
    def compiler_info(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "optimize_level": self.optimize_level,
            "reg_alloc": self.reg_alloc,
        }

    def discover(self) -> list[CaseSpec]:
        if self._cases is None:
            self._cases = discover_cases(self._roots, verbose=self.verbose)
        return list(self._cases)

    def cleanup(self) -> None:
        if self._owns_workdir:
            shutil.rmtree(self.workdir, ignore_errors=True)

    def compile_case(self, spec: CaseSpec) -> CompileOutcome:
        out_path = self.workdir / f"{spec.pytest_id}.s"
        started = time.perf_counter()
        asm_text = ""
        ok = False
        error: str | None = None
        output_path: Path | None = None
        try:
            from scratchv.compiler import CompilerConfig, CompilerDriver

            driver = CompilerDriver(
                CompilerConfig(
                    backend=self.backend,
                    optimize_level=self.optimize_level,
                    reg_alloc=self.reg_alloc,
                )
            )
            result = driver.compile(str(spec.dsl_path), str(out_path))
            asm_text = result.output_text or ""
            ok = bool(result.success and asm_text)
            if ok:
                output_path = out_path
            else:
                messages = list(result.errors) or ["compilation failed"]
                error = "; ".join(messages)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        duration = time.perf_counter() - started
        return CompileOutcome(
            ok=ok,
            asm_text=asm_text,
            output_path=output_path,
            ir_instruction_count=self._count_ir_instructions(spec),
            duration_s=duration,
            error=error,
        )

    def _count_ir_instructions(self, spec: CaseSpec) -> int:
        try:
            source = spec.dsl_path.read_text()
        except OSError:
            return 0
        program = None
        try:
            from scratchv.frontend.dsl_extended import ExtendedDSLParser

            program = ExtendedDSLParser().parse(source)
        except Exception:
            try:
                from scratchv.frontend.dsl_parser import DSLParser

                program = DSLParser().parse(source)
            except Exception:
                return 0
        return sum(
            1
            for function in program.functions
            for block in function.blocks
            for _ in block.instructions
        )

    def assemble_asm(self, asm_text: str) -> AssembleOutcome:
        started = time.perf_counter()
        binary: bytearray | None = None
        error: str | None = None
        try:
            from scratchv.backend.riscv_encoder import assemble_to_binary

            binary = assemble_to_binary(asm_text)
        except Exception as exc:
            detail = _first_bad_asm_line(asm_text)
            if detail is not None:
                line_no, bad_line, _ = detail
                error = (
                    f"{type(exc).__name__}: {exc} at line {line_no}: "
                    f"{bad_line!r}"
                )
            else:
                error = f"{type(exc).__name__}: {exc}"
        duration = time.perf_counter() - started
        instruction_count: int | None
        try:
            instruction_count = count_asm_instructions(asm_text)
        except Exception:
            instruction_count = None
        return AssembleOutcome(
            ok=bool(binary),
            binary_len=len(binary) if binary else 0,
            instruction_count=instruction_count,
            duration_s=duration,
            error=error,
        )

    def check_instruction_budget(
        self, spec: CaseSpec, asm_text: str,
    ) -> BudgetOutcome:
        count = count_asm_instructions(asm_text)
        if spec.max_instructions is None:
            return BudgetOutcome(
                ok=None, instruction_count=count, limit=None,
                skipped=True, error=None,
            )
        ok = count <= spec.max_instructions
        return BudgetOutcome(
            ok=ok,
            instruction_count=count,
            limit=spec.max_instructions,
            skipped=False,
            error=None if ok else f"{count} > {spec.max_instructions}",
        )

    def evaluate_semantics(
        self, spec: CaseSpec, compile_outcome: CompileOutcome,
    ) -> SemanticOutcome:
        if spec.oracle == ORACLE_NONE:
            return SemanticOutcome(
                ok=None, oracle=ORACLE_NONE, expected=spec.expected_return,
                actual=None, duration_s=0.0,
                blocked_reason="oracle=none", error=None,
            )
        if spec.oracle == ORACLE_INTERPRETER:
            return self._interpreter_semantics(spec)
        return self._execution_semantics(spec, compile_outcome)

    def _interpreter_semantics(self, spec: CaseSpec) -> SemanticOutcome:
        started = time.perf_counter()
        source = spec.dsl_path.read_text()
        if spec.inputs:
            inputs = {
                name: np.asarray(value, dtype=np.float32)
                for name, value in spec.inputs.items()
            }
        else:
            inputs = default_inputs(source)
        try:
            from scratchv.verification.verifier import DSLInterpreter

            result = DSLInterpreter().run(source, inputs)
            actual = _stringify(result)
            expected = (
                spec.expected_return
                if spec.expected_return is not None
                else spec.expected_text
            )
            ok = compare_values(
                result, expected, rtol=spec.rtol, atol=spec.atol,
            )
            error = None
            if not ok:
                error = (
                    f"semantic mismatch: expected {expected!r}, "
                    f"actual {actual!r}"
                )
            return SemanticOutcome(
                ok=ok, oracle=ORACLE_INTERPRETER, expected=expected,
                actual=actual, duration_s=time.perf_counter() - started,
                blocked_reason=None, error=error,
            )
        except Exception as exc:
            return SemanticOutcome(
                ok=False, oracle=ORACLE_INTERPRETER,
                expected=spec.expected_return, actual=None,
                duration_s=time.perf_counter() - started,
                blocked_reason=None, error=f"{type(exc).__name__}: {exc}",
            )

    def _execution_semantics(
        self, spec: CaseSpec, compile_outcome: CompileOutcome,
    ) -> SemanticOutcome:
        oracle = ExecutionOracle()
        outcome = oracle.execute(
            compile_outcome.asm_text,
            spec.input_registers,
            spec.inputs,
            timeout_s=min(spec.timeout_s, self.timeout_s),
        )
        if outcome.ok is None and (
            outcome.blocked_reason or outcome.error
        ):
            return replace(outcome, expected=spec.expected_return)
        if outcome.ok is False and outcome.error:
            return replace(outcome, expected=spec.expected_return)
        ok = compare_values(
            outcome.actual, spec.expected_return,
            rtol=spec.rtol, atol=spec.atol,
        )
        error = None
        if not ok:
            error = (
                f"semantic mismatch: expected {spec.expected_return!r}, "
                f"actual {outcome.actual!r}"
            )
        return replace(
            outcome, ok=ok, expected=spec.expected_return, error=error,
        )

    def run_case(self, spec: CaseSpec) -> CaseOutcome:
        if spec.meta_errors:
            return CaseOutcome(
                case_id=spec.case_id,
                pytest_id=spec.pytest_id,
                status=STATUS_FAIL,
                compile=_empty_compile_outcome(),
                assemble=None,
                budget=None,
                semantic=None,
                error_stage=STAGE_META,
                error="; ".join(spec.meta_errors),
                xfail=spec.xfail,
            )
        if spec.skip_reason:
            return CaseOutcome(
                case_id=spec.case_id,
                pytest_id=spec.pytest_id,
                status=STATUS_SKIP,
                compile=_empty_compile_outcome(),
                assemble=None,
                budget=None,
                semantic=None,
                error_stage=None,
                error=None,
                xfail=spec.xfail,
            )

        stage_order = {
            STAGE_COMPILE: 0,
            STAGE_ASSEMBLE: 1,
            STAGE_BUDGET: 2,
            STAGE_SEMANTIC: 3,
        }
        xfail_stages = set(spec.xfail.stages) if spec.xfail else set()
        failures: list[tuple[str, str]] = []
        blocked: list[tuple[str, str]] = []

        compile_outcome = self.compile_case(spec)
        assemble_outcome: AssembleOutcome | None = None
        budget_outcome: BudgetOutcome | None = None
        semantic_outcome: SemanticOutcome | None = None

        if ASSERT_COMPILE_OK in spec.assertions and not compile_outcome.ok:
            failures.append(
                (STAGE_COMPILE, compile_outcome.error or "compile failed"),
            )
        else:
            if compile_outcome.asm_text:
                assemble_outcome = self.assemble_asm(compile_outcome.asm_text)
            else:
                assemble_outcome = AssembleOutcome(
                    ok=False, binary_len=0, instruction_count=None,
                    duration_s=0.0, error="no assembly text produced",
                )
            if (
                ASSERT_ASM_ENCODABLE in spec.assertions
                and not assemble_outcome.ok
            ):
                failures.append(
                    (
                        STAGE_ASSEMBLE,
                        assemble_outcome.error or "assembly not encodable",
                    ),
                )

            if ASSERT_INST_BUDGET in spec.assertions:
                if spec.max_instructions is None:
                    budget_outcome = self.check_instruction_budget(
                        spec, compile_outcome.asm_text,
                    )
                elif assemble_outcome.ok:
                    budget_outcome = self.check_instruction_budget(
                        spec, compile_outcome.asm_text,
                    )
                    if budget_outcome.ok is False:
                        failures.append(
                            (
                                STAGE_BUDGET,
                                budget_outcome.error
                                or "instruction budget exceeded",
                            ),
                        )
                else:
                    budget_outcome = BudgetOutcome(
                        ok=None,
                        instruction_count=assemble_outcome.instruction_count,
                        limit=spec.max_instructions,
                        skipped=False,
                        error="blocked by assemble stage",
                    )
                    blocked.append(
                        (STAGE_BUDGET, "blocked by assemble stage"),
                    )
            elif assemble_outcome.ok:
                budget_outcome = self.check_instruction_budget(
                    spec, compile_outcome.asm_text,
                )

            if ASSERT_SEMANTIC_GOLDEN in spec.assertions:
                if (
                    spec.oracle == ORACLE_EXECUTION
                    and not assemble_outcome.ok
                ):
                    semantic_outcome = SemanticOutcome(
                        ok=None, oracle=spec.oracle,
                        expected=spec.expected_return, actual=None,
                        duration_s=0.0,
                        blocked_reason="blocked by assemble stage",
                        error=None,
                    )
                    blocked.append(
                        (STAGE_SEMANTIC, "blocked by assemble stage"),
                    )
                else:
                    semantic_outcome = self.evaluate_semantics(
                        spec, compile_outcome,
                    )
                    if semantic_outcome.ok is False:
                        failures.append(
                            (
                                STAGE_SEMANTIC,
                                semantic_outcome.error
                                or "semantic golden check failed",
                            ),
                        )
                    elif semantic_outcome.ok is None:
                        blocked.append(
                            (
                                STAGE_SEMANTIC,
                                semantic_outcome.blocked_reason
                                or "semantic check blocked",
                            ),
                        )

        hard = [
            (stage, message)
            for stage, message in failures if stage not in xfail_stages
        ]
        covered = [
            (stage, message)
            for stage, message in failures if stage in xfail_stages
        ]
        blocked_covered = [
            (stage, message)
            for stage, message in blocked if stage in xfail_stages
        ]
        blocked_uncovered = [
            (stage, message)
            for stage, message in blocked if stage not in xfail_stages
        ]

        status = STATUS_PASS
        error_stage: str | None = None
        error: str | None = None
        if hard or blocked_uncovered:
            status = STATUS_FAIL
            error_stage, error = min(
                hard + blocked_uncovered,
                key=lambda item: stage_order[item[0]],
            )
        elif covered or blocked_covered:
            status = STATUS_XFAIL
            error_stage, error = min(
                covered + blocked_covered,
                key=lambda item: stage_order[item[0]],
            )
        elif spec.xfail is not None:
            status = STATUS_XPASS
            if self.strict_xfail:
                status = STATUS_FAIL
                error = (
                    "strict xfail: declared failing stage(s) now pass "
                    f"({', '.join(spec.xfail.stages)})"
                )

        return CaseOutcome(
            case_id=spec.case_id,
            pytest_id=spec.pytest_id,
            status=status,
            compile=compile_outcome,
            assemble=assemble_outcome,
            budget=budget_outcome,
            semantic=semantic_outcome,
            error_stage=error_stage,
            error=error,
            xfail=spec.xfail,
        )

    def run_cases(self, cases: Sequence[CaseSpec]) -> SuiteReport:
        results = [self.run_case(case) for case in cases]
        return SuiteReport(
            results=results,
            roots=self.root_strings,
            compiler=self.compiler_info,
            generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
            specs={case.case_id: case for case in cases},
        )

    def run_all(self) -> SuiteReport:
        return self.run_cases(self.discover())
