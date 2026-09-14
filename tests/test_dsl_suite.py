"""Pytest gate for the ScratchV DSL benchmark suite (topic 06).

Each stage of each discovered case becomes an independently addressable
test node (``case_id + stage``), so failures pinpoint the exact stage.
Failures already located, owned and declared in ``*.meta.json`` are marked
``xfail`` (non-strict by default; xpasses are visible but do not fail CI).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from benchmarks.dsl_suite import (
    ASSERT_INST_BUDGET,
    ORACLE_EXECUTION,
    ORACLE_NONE,
    STAGE_ASSEMBLE,
    STAGE_BUDGET,
    STAGE_COMPILE,
    STAGE_SEMANTIC,
    STATUS_FAIL,
    STATUS_XFAIL,
    STATUS_XPASS,
    CaseOutcome,
    CaseSpec,
    CompileOutcome,
    DSLSuiteRunner,
    ExecutionOracle,
    SemanticOutcome,
    SuiteReport,
    compare_values,
    discover_cases,
    load_case_spec,
)
from benchmarks.run_suite import _print_summary

ALL_CASES: tuple[CaseSpec, ...] = tuple(discover_cases())
if not ALL_CASES:
    pytest.fail("dsl suite discovery found 0 cases", pytrace=False)


def marks_for(case: CaseSpec, stage: str) -> list[pytest.MarkDecorator]:
    """Return the xfail marker for *stage* when the case declares it."""
    if case.xfail is not None and stage in case.xfail.stages:
        return [
            pytest.mark.xfail(
                reason=case.xfail.reason,
                strict=case.xfail.strict,
            )
        ]
    return []


def params_for(stage: str | None) -> list[pytest.ParameterSet]:
    """Build per-case parameters; marks must be injected per parameter."""
    return [
        pytest.param(
            case,
            id=case.pytest_id,
            marks=marks_for(case, stage) if stage else [],
        )
        for case in ALL_CASES
    ]


@pytest.fixture(scope="session")
def suite_runner() -> DSLSuiteRunner:
    runner = DSLSuiteRunner(verbose=False)
    yield runner
    runner.cleanup()


@pytest.fixture(scope="session")
def compile_results(suite_runner: DSLSuiteRunner) -> dict:
    return {
        case.case_id: suite_runner.compile_case(case) for case in ALL_CASES
    }


@pytest.fixture(scope="session")
def asm_results(
    suite_runner: DSLSuiteRunner, compile_results: dict,
) -> dict:
    return {
        case_id: suite_runner.assemble_asm(outcome.asm_text)
        for case_id, outcome in compile_results.items()
    }


@pytest.fixture(scope="session")
def semantic_results(
    suite_runner: DSLSuiteRunner, compile_results: dict, asm_results: dict,
) -> dict:
    results: dict = {}
    for case in ALL_CASES:
        if case.skip_reason or case.oracle == ORACLE_NONE:
            results[case.case_id] = None
            continue
        if case.oracle == ORACLE_EXECUTION and not asm_results[case.case_id].ok:
            results[case.case_id] = SemanticOutcome(
                ok=None,
                oracle=case.oracle,
                expected=case.expected_return,
                actual=None,
                duration_s=0.0,
                blocked_reason="blocked by assemble stage",
                error=None,
            )
            continue
        results[case.case_id] = suite_runner.evaluate_semantics(
            case, compile_results[case.case_id],
        )
    return results


@pytest.mark.parametrize("case", params_for(None))
def test_meta_contract(case: CaseSpec) -> None:
    assert case.meta_errors == (), (
        f"meta contract violations: {case.meta_errors}"
    )
    if case.xfail is not None:
        assert case.xfail.owner.strip(), "xfail.owner must be non-empty"
        assert case.xfail.reason.strip(), "xfail.reason must be non-empty"


@pytest.mark.parametrize("case", params_for(STAGE_COMPILE))
def test_compile_ok(case: CaseSpec, compile_results: dict) -> None:
    if case.skip_reason:
        pytest.skip(case.skip_reason)
    outcome = compile_results[case.case_id]
    assert outcome.ok, f"compile failed: {outcome.error}"


@pytest.mark.parametrize("case", params_for(STAGE_ASSEMBLE))
def test_asm_encodable(case: CaseSpec, asm_results: dict) -> None:
    if case.skip_reason:
        pytest.skip(case.skip_reason)
    outcome = asm_results[case.case_id]
    assert outcome.ok, f"assemble failed: {outcome.error}"


@pytest.mark.parametrize("case", params_for(STAGE_BUDGET))
def test_instruction_budget(case: CaseSpec, asm_results: dict) -> None:
    if case.skip_reason:
        pytest.skip(case.skip_reason)
    if ASSERT_INST_BUDGET not in case.assertions:
        pytest.skip("inst_budget assertion disabled")
    if case.max_instructions is None:
        pytest.skip("no max_instructions declared")
    outcome = asm_results[case.case_id]
    assert outcome.ok, "blocked by assemble stage"
    assert outcome.instruction_count is not None
    assert outcome.instruction_count <= case.max_instructions, (
        f"{outcome.instruction_count} > {case.max_instructions}"
    )


@pytest.mark.parametrize("case", params_for(STAGE_SEMANTIC))
def test_semantic_golden(
    case: CaseSpec, semantic_results: dict,
) -> None:
    if case.skip_reason:
        pytest.skip(case.skip_reason)
    if case.oracle == ORACLE_NONE:
        pytest.skip("oracle=none")
    outcome = semantic_results[case.case_id]
    assert outcome is not None
    assert outcome.ok, outcome.blocked_reason or outcome.error


def test_compare_values_handles_scalars_vectors_and_text() -> None:
    assert compare_values([2, 4, 6, 8], "[2. 4. 6. 8.]")
    assert compare_values(30.0, "30.")
    assert compare_values(
        [0.032059, 0.087144, 0.236883, 0.643914],
        [0.0320586, 0.08714432, 0.23688282, 0.64391428],
    )
    assert not compare_values([1.0, 2.0], [1.0, 2.0, 3.0])
    assert not compare_values(1.0, 2.0)
    assert not compare_values(None, 1.0)


def _write_pseudo_case(
    directory: Path,
    name: str,
    source: str,
    meta: dict,
) -> Path:
    dsl_path = directory / f"{name}.dsl"
    dsl_path.write_text(source)
    (directory / f"{name}.meta.json").write_text(json.dumps(meta))
    return dsl_path


@pytest.mark.parametrize(
    ("name", "source", "meta", "fragment"),
    [
        (
            "interpreter_control",
            "while (i < 3):\n  i = add(i, 1)\nendwhile\nreturn i\n",
            {
                "description": "pseudo case",
                "oracle": "interpreter",
                "expected_return": 3,
            },
            "oracle/flow conflict",
        ),
        (
            "execution_missing_registers",
            "c = add(a, b)\nreturn c\n",
            {
                "description": "pseudo case",
                "oracle": "execution",
                "inputs": {"a": 1, "b": 2},
                "expected_return": 3,
            },
            "input_registers is required",
        ),
        (
            "none_without_xfail",
            "c = add(a, b)\nreturn c\n",
            {
                "description": "pseudo case",
                "oracle": "none",
            },
            "oracle=none requires",
        ),
    ],
)
def test_meta_contract_rejects_invalid_metadata(
    tmp_path: Path,
    name: str,
    source: str,
    meta: dict,
    fragment: str,
) -> None:
    dsl_path = _write_pseudo_case(tmp_path, name, source, meta)
    spec = load_case_spec(dsl_path, tmp_path)
    assert any(fragment in error for error in spec.meta_errors), (
        f"expected {fragment!r} in {spec.meta_errors}"
    )


MANUAL_LOOP_ASM = (
    "addi a0, zero, 0\n"
    "addi t0, zero, 0\n"
    "addi t1, zero, 4\n"
    "loop:\n"
    "addi a0, a0, 1\n"
    "addi t0, t0, 1\n"
    "blt t0, t1, loop\n"
    "jalr zero, ra\n"
)

INFINITE_LOOP_ASM = (
    "addi t0, zero, 0\n"
    "loop:\n"
    "addi t0, t0, 1\n"
    "j loop\n"
)


def _require_tinyfive() -> None:
    if not ExecutionOracle().available():
        pytest.skip("tinyfive not installed")


def test_execution_oracle_completes_manual_loop() -> None:
    """F1 regression: static word count must not truncate dynamic loops."""
    _require_tinyfive()
    outcome = ExecutionOracle().execute(MANUAL_LOOP_ASM, {}, {}, timeout_s=5.0)
    assert outcome.error is None, outcome.error
    assert outcome.blocked_reason is None
    assert outcome.actual == 4


def test_execution_oracle_validates_compiled_code(
    tmp_path: Path, suite_runner: DSLSuiteRunner,
) -> None:
    """F1/F3 regression: green codegen + simulation path for the oracle."""
    dsl_path = _write_pseudo_case(
        tmp_path,
        "execution_add",
        "c = add(a, b)\nreturn c\n",
        {
            "description": "codegen+execution probe",
            "oracle": "execution",
            "inputs": {"a": 2, "b": 3},
            "input_registers": {"a": "a0", "b": "a1"},
            "expected_return": 5,
        },
    )
    spec = load_case_spec(dsl_path, tmp_path)
    assert spec.meta_errors == (), spec.meta_errors
    _require_tinyfive()
    compile_outcome = suite_runner.compile_case(spec)
    assert compile_outcome.ok, compile_outcome.error
    outcome = suite_runner.evaluate_semantics(spec, compile_outcome)
    assert outcome.ok is True, outcome.blocked_reason or outcome.error


def test_execution_oracle_times_out() -> None:
    """F6 regression: timeout_s must bound a non-terminating execution."""
    _require_tinyfive()
    started = time.perf_counter()
    outcome = ExecutionOracle().execute(
        INFINITE_LOOP_ASM, {}, {}, timeout_s=0.1,
    )
    elapsed = time.perf_counter() - started
    assert outcome.ok is False
    assert outcome.error is not None and "timeout" in outcome.error
    assert elapsed < 5.0


def test_runner_timeout_budget_caps_execution(tmp_path: Path) -> None:
    """F6 regression: the runner/CLI budget reaches the execution oracle."""
    dsl_path = _write_pseudo_case(
        tmp_path,
        "timeout_case",
        "c = add(a, b)\nreturn c\n",
        {
            "description": "timeout budget probe",
            "oracle": "execution",
            "inputs": {"a": 1, "b": 2},
            "input_registers": {"a": "a0", "b": "a1"},
            "expected_return": 3,
            "timeout_s": 30.0,
        },
    )
    spec = load_case_spec(dsl_path, tmp_path)
    assert spec.meta_errors == (), spec.meta_errors
    _require_tinyfive()
    runner = DSLSuiteRunner(roots=(tmp_path,), timeout_s=0.1)
    try:
        synthetic = CompileOutcome(
            ok=True, asm_text=INFINITE_LOOP_ASM, output_path=None,
            ir_instruction_count=0, duration_s=0.0, error=None,
        )
        outcome = runner.evaluate_semantics(spec, synthetic)
        assert outcome.ok is False
        assert outcome.error is not None and "timeout" in outcome.error
    finally:
        runner.cleanup()


def test_blocked_semantic_without_xfail_is_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F4 regression: uncovered blocked stages must be hard failures."""
    dsl_path = _write_pseudo_case(
        tmp_path,
        "blocked_execution",
        "c = add(a, b)\nreturn c\n",
        {
            "description": "blocked semantic probe",
            "oracle": "execution",
            "inputs": {"a": 1, "b": 2},
            "input_registers": {"a": "a0", "b": "a1"},
            "expected_return": 3,
        },
    )
    spec = load_case_spec(dsl_path, tmp_path)
    assert spec.meta_errors == (), spec.meta_errors
    assert spec.xfail is None
    monkeypatch.setattr(ExecutionOracle, "available", lambda self: False)
    runner = DSLSuiteRunner(roots=(tmp_path,))
    try:
        outcome = runner.run_case(spec)
    finally:
        runner.cleanup()
    assert outcome.status == STATUS_FAIL
    assert outcome.error_stage == STAGE_SEMANTIC


def test_compile_stage_xfail_is_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F5 regression: xfail(compile) must reach the compile test node."""
    dsl_path = _write_pseudo_case(
        tmp_path,
        "compile_xfail",
        "this is not a valid dsl program !!!\n",
        {
            "description": "compile-stage xfail probe",
            "category": "arith",
            "oracle": "interpreter",
            "expected_return": 0,
            "xfail": {
                "stages": ["compile"],
                "reason": "probe: compile stage fails",
                "owner": "tests",
            },
        },
    )
    spec = load_case_spec(dsl_path, tmp_path)
    assert spec.meta_errors == (), spec.meta_errors
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "ALL_CASES", (spec,))
    params = params_for(STAGE_COMPILE)
    assert params and params[0].marks, "compile-stage xfail mark missing"
    assert any(mark.name == "xfail" for mark in params[0].marks)
    assert not params_for(None)[0].marks
    runner = DSLSuiteRunner(roots=(tmp_path,))
    try:
        outcome = runner.run_case(spec)
    finally:
        runner.cleanup()
    assert outcome.status == STATUS_XFAIL
    assert outcome.error_stage == STAGE_COMPILE


BRANCH_TARGET_XFAIL_CASES = frozenset({
    "cases/009_maxpool",
    "cases/013_for_sum",
    "cases/014_for_dot",
    "cases/015_for_relu",
    "cases/016_if_simple",
    "cases/017_while_sum",
    "cases/018_nested_if",
    "cases/019_nested_loop",
    "cases/021_dsl_if_else",
    "cases/022_dsl_while_sum",
    "stress/reg_pressure_loop",
})


def test_branch_target_xfails_blame_linear_scan() -> None:
    """F2 regression: assemble xfails must blame the linear-scan emitter."""
    discovered = {case.case_id for case in ALL_CASES}
    missing = BRANCH_TARGET_XFAIL_CASES - discovered
    assert not missing, f"cases disappeared: {sorted(missing)}"
    for case in ALL_CASES:
        if case.case_id not in BRANCH_TARGET_XFAIL_CASES:
            continue
        assert case.xfail is not None
        assert STAGE_ASSEMBLE in case.xfail.stages
        assert case.xfail.owner == "backend/regalloc", case.case_id
        assert "linear" in case.xfail.reason, case.case_id
        assert "first bad line" in case.xfail.reason, case.case_id


@pytest.mark.parametrize(
    "case_id", ["cases/013_for_sum", "cases/016_if_simple"],
)
def test_greedy_allocator_encodes_branch_targets(case_id: str) -> None:
    """F2 regression: same encoder encodes branch targets on greedy path."""
    spec = next(case for case in ALL_CASES if case.case_id == case_id)
    runner = DSLSuiteRunner(roots=(spec.root,), reg_alloc="greedy")
    try:
        compile_outcome = runner.compile_case(spec)
        assert compile_outcome.ok, compile_outcome.error
        assemble_outcome = runner.assemble_asm(compile_outcome.asm_text)
        assert assemble_outcome.ok, assemble_outcome.error
    finally:
        runner.cleanup()


def test_interpreter_blind_spot_reported_as_warning(
    suite_runner: DSLSuiteRunner,
) -> None:
    """F3 regression: interpreter blind spots must surface in the report."""
    spec = next(
        case for case in ALL_CASES
        if case.case_id == "cases/020_constant_propagation"
    )
    assert any("interpreter" in warning for warning in spec.warnings)
    outcome = suite_runner.run_case(spec)
    report = SuiteReport(
        results=[outcome],
        roots=("benchmarks/cases",),
        compiler=suite_runner.compiler_info,
        generated_at="test",
        specs={spec.case_id: spec},
    )
    warnings = report.to_dict()["results"][0]["warnings"]
    assert warnings, "interpreter blind spot warning missing from report"
    assert any("interpreter" in warning for warning in warnings)


def test_xpass_warning_is_printed(capsys: pytest.CaptureFixture) -> None:
    """F8 regression: unexpected passes must produce a visible warning."""
    result = CaseOutcome(
        case_id="pseudo/xpass",
        pytest_id="pseudo-xpass",
        status=STATUS_XPASS,
        compile=CompileOutcome(
            ok=True, asm_text="", output_path=None,
            ir_instruction_count=0, duration_s=0.0, error=None,
        ),
        assemble=None,
        budget=None,
        semantic=None,
        error_stage=None,
        error=None,
        xfail=None,
    )
    report = SuiteReport(
        results=[result], roots=("pseudo",), compiler={},
        generated_at="test",
    )
    _print_summary(report, quiet=True)
    captured = capsys.readouterr()
    assert "xpass" in captured.err.lower()
