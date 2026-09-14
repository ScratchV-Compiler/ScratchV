"""Hermetic contract tests for the DSL benchmark suite report (topic 06).

``SuiteReport`` produces the CI artifacts consumed by the job summary
(``benchmark_reports/dsl_suite.md``) and by dashboards/history
(``benchmark_reports/dsl_suite.json``).  These tests pin that contract
without running the 25 real cases: outcomes and case metadata are
constructed (or loaded from a temporary one-case root), so the tests stay
fast and independent of compiler progress.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.dsl_suite import (
    ALL_STAGES,
    SCHEMA_VERSION,
    STAGE_ASSEMBLE,
    STAGE_SEMANTIC,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIP,
    STATUS_XFAIL,
    STATUS_XPASS,
    SUITE_NAME,
    AssembleOutcome,
    BudgetOutcome,
    CaseOutcome,
    CaseSpec,
    CompileOutcome,
    SemanticOutcome,
    SuiteReport,
    load_case_spec,
)

GENERATED_AT = "2026-09-15T00:00:00"

TOP_LEVEL_KEYS = frozenset({
    "schema_version", "suite", "generated_at", "roots", "compiler",
    "summary", "stage_summary", "owner_summary", "xpassed_cases", "results",
})
RESULT_KEYS = frozenset({
    "case_id", "pytest_id", "category", "flow", "oracle", "status",
    "stages", "metrics", "expected", "actual", "xfail", "error_stage",
    "error", "warnings",
})
STAGE_KEYS = frozenset({
    "compile_ok", "asm_encodable", "inst_budget_ok", "semantic_ok",
})
METRIC_KEYS = frozenset({
    "ir_instructions", "asm_instructions", "compile_time_s",
    "assemble_time_s", "semantic_time_s",
})
STAGE_SUMMARY_KEYS = frozenset({
    "passed", "xfail", "failed", "xpass", "blocked",
})
OWNER_SUMMARY_KEYS = frozenset({
    "declared", "xfailed", "xpassed", "failed",
})


def _make_spec(
    root: Path, name: str, *, xfail: dict | None = None,
) -> CaseSpec:
    """Write and load one minimal valid interpreter-oracle case."""
    root.mkdir(parents=True, exist_ok=True)
    dsl_path = root / f"{name}.dsl"
    dsl_path.write_text("c = add(a, b)\nreturn c\n")
    meta: dict = {
        "description": f"report contract probe {name}",
        "category": "arith",
        "oracle": "interpreter",
        "expected_return": 3,
    }
    if xfail is not None:
        meta["xfail"] = xfail
    dsl_path.with_suffix(".meta.json").write_text(json.dumps(meta))
    spec = load_case_spec(dsl_path, root)
    assert spec.meta_errors == (), spec.meta_errors
    return spec


def _outcome(
    spec: CaseSpec,
    status: str,
    *,
    compile_ok: bool = True,
    assemble_ok: bool | None = True,
    budget_ok: bool | None = True,
    semantic_ok: bool | None = True,
    error_stage: str | None = None,
    error: str | None = None,
    actual: object | None = None,
) -> CaseOutcome:
    """Construct one case outcome with the requested per-stage verdicts."""
    compile_outcome = CompileOutcome(
        ok=compile_ok,
        asm_text="addi a0, zero, 3\n" if compile_ok else "",
        output_path=None,
        ir_instruction_count=1 if compile_ok else 0,
        duration_s=0.0,
        error=None if compile_ok else "compile failed",
    )
    assemble = (
        None if assemble_ok is None
        else AssembleOutcome(
            ok=assemble_ok,
            binary_len=4 if assemble_ok else 0,
            instruction_count=1 if assemble_ok else None,
            duration_s=0.0,
            error=None if assemble_ok else "assemble failed",
        )
    )
    budget = (
        None if budget_ok is None
        else BudgetOutcome(
            ok=budget_ok,
            instruction_count=1,
            limit=2,
            skipped=False,
            error=None if budget_ok else "1 > 2",
        )
    )
    semantic = (
        None if semantic_ok is None
        else SemanticOutcome(
            ok=semantic_ok,
            oracle=spec.oracle,
            expected=3,
            actual=actual if actual is not None else (3 if semantic_ok else 4),
            duration_s=0.0,
            blocked_reason=None,
            error=None if semantic_ok else "semantic mismatch",
        )
    )
    return CaseOutcome(
        case_id=spec.case_id,
        pytest_id=spec.pytest_id,
        status=status,
        compile=compile_outcome,
        assemble=assemble,
        budget=budget,
        semantic=semantic,
        error_stage=error_stage,
        error=error,
        xfail=spec.xfail,
    )


def _report_from(
    results: list[CaseOutcome], specs: list[CaseSpec],
) -> SuiteReport:
    return SuiteReport(
        results=results,
        roots=("probe",),
        compiler={
            "backend": "riscv",
            "optimize_level": "all",
            "reg_alloc": "linear",
        },
        generated_at=GENERATED_AT,
        specs={spec.case_id: spec for spec in specs},
    )


def _build_mixed_report(tmp_path: Path) -> SuiteReport:
    """One pass, one xfail, one xpass, one hard fail and one skip."""
    root = tmp_path / "probe"
    pass_spec = _make_spec(root, "pass_case")
    xfail_spec = _make_spec(root, "xfail_case", xfail={
        "stages": ["assemble", "budget"],
        "reason": "C1 probe: linear emitter drops branch targets",
        "owner": "backend/regalloc",
    })
    xpass_spec = _make_spec(root, "xpass_case", xfail={
        "stages": ["semantic"],
        "reason": "C4 probe: op lowering now works",
        "owner": "backend/op-lowering",
    })
    fail_spec = _make_spec(root, "fail_case")
    skip_spec = _make_spec(root, "skip_case")
    results = [
        _outcome(
            pass_spec, STATUS_PASS, actual=np.array([1.0, 2.0, 3.0]),
        ),
        _outcome(
            xfail_spec, STATUS_XFAIL,
            assemble_ok=False, budget_ok=None, semantic_ok=None,
            error_stage=STAGE_ASSEMBLE,
            error="C1 probe: linear emitter drops branch targets",
        ),
        _outcome(xpass_spec, STATUS_XPASS),
        _outcome(
            fail_spec, STATUS_FAIL,
            assemble_ok=False, budget_ok=None, semantic_ok=None,
            error_stage=STAGE_ASSEMBLE, error="assemble failed",
        ),
        _outcome(
            skip_spec, STATUS_SKIP,
            compile_ok=False, assemble_ok=None,
            budget_ok=None, semantic_ok=None,
        ),
    ]
    return _report_from(results, [
        pass_spec, xfail_spec, xpass_spec, fail_spec, skip_spec,
    ])


def _recomputed_summary(report: SuiteReport) -> dict[str, int]:
    statuses = [r.status for r in report.results]
    return {
        "total": len(statuses),
        "passed": statuses.count(STATUS_PASS),
        "xfailed": statuses.count(STATUS_XFAIL),
        "xpassed": statuses.count(STATUS_XPASS),
        "failed": statuses.count(STATUS_FAIL),
        "skipped": statuses.count(STATUS_SKIP),
    }


def assert_report_consistent(report: SuiteReport) -> None:
    """Gate: summary/aggregates must be recomputable from raw outcomes."""
    summary = report.summary()
    recomputed = _recomputed_summary(report)
    assert summary == recomputed, (
        f"summary() drifted from raw case statuses: {summary} != {recomputed}"
    )
    assert summary["total"] == sum(
        summary[key]
        for key in ("passed", "xfailed", "xpassed", "failed", "skipped")
    ), summary
    for stage, counts in report.stage_summary().items():
        assert set(counts) == STAGE_SUMMARY_KEYS, (stage, counts)
        assert sum(counts.values()) == summary["total"], (stage, counts)
    for owner, counts in report.owner_summary().items():
        assert set(counts) == OWNER_SUMMARY_KEYS, (owner, counts)
    data = report.to_dict()
    assert data["summary"] == summary
    assert data["stage_summary"] == report.stage_summary()
    assert data["owner_summary"] == report.owner_summary()
    assert data["xpassed_cases"] == report.xpassed_cases()


# ---------------------------------------------------------------------------
# summary() counts and boundaries
# ---------------------------------------------------------------------------


def test_summary_counts_match_constructed_outcomes(tmp_path: Path) -> None:
    report = _build_mixed_report(tmp_path)
    assert report.summary() == {
        "total": 5, "passed": 1, "xfailed": 1,
        "xpassed": 1, "failed": 1, "skipped": 1,
    }
    assert report.pass_count == 1
    assert report.xfail_count == 1
    assert report.xpass_count == 1
    assert report.fail_count == 1
    assert report.skip_count == 1


def test_summary_all_pass_and_empty_suite(tmp_path: Path) -> None:
    root = tmp_path / "probe_all_pass"
    specs = [_make_spec(root, f"pass_{i}") for i in range(3)]
    report = _report_from(
        [_outcome(spec, STATUS_PASS) for spec in specs], specs,
    )
    assert report.summary() == {
        "total": 3, "passed": 3, "xfailed": 0,
        "xpassed": 0, "failed": 0, "skipped": 0,
    }

    empty = SuiteReport(
        results=[], roots=(), compiler={}, generated_at=GENERATED_AT,
    )
    assert empty.summary() == {
        "total": 0, "passed": 0, "xfailed": 0,
        "xpassed": 0, "failed": 0, "skipped": 0,
    }
    assert set(empty.stage_summary()) == set(ALL_STAGES)
    assert all(
        sum(counts.values()) == 0
        for counts in empty.stage_summary().values()
    )
    assert empty.owner_summary() == {}
    assert empty.xpassed_cases() == []
    assert_report_consistent(empty)


def test_summary_single_status_xfail_and_xpass(tmp_path: Path) -> None:
    root = tmp_path / "probe_single"
    xfail_spec = _make_spec(root, "xfail_only", xfail={
        "stages": ["assemble"],
        "reason": "probe: assemble stage fails",
        "owner": "backend/regalloc",
    })
    xpass_spec = _make_spec(root, "xpass_only", xfail={
        "stages": ["semantic"],
        "reason": "probe: semantic stage now passes",
        "owner": "backend/op-lowering",
    })

    xfail_report = _report_from(
        [_outcome(
            xfail_spec, STATUS_XFAIL,
            assemble_ok=False, budget_ok=None, semantic_ok=None,
            error_stage=STAGE_ASSEMBLE, error="assemble failed",
        )],
        [xfail_spec],
    )
    assert xfail_report.summary() == {
        "total": 1, "passed": 0, "xfailed": 1,
        "xpassed": 0, "failed": 0, "skipped": 0,
    }
    assert_report_consistent(xfail_report)

    xpass_report = _report_from(
        [_outcome(xpass_spec, STATUS_XPASS)], [xpass_spec],
    )
    assert xpass_report.summary() == {
        "total": 1, "passed": 0, "xfailed": 0,
        "xpassed": 1, "failed": 0, "skipped": 0,
    }
    assert xpass_report.stage_summary()[STAGE_SEMANTIC]["xpass"] == 1
    assert_report_consistent(xpass_report)


# ---------------------------------------------------------------------------
# stage / owner aggregation
# ---------------------------------------------------------------------------


def test_stage_and_owner_aggregation(tmp_path: Path) -> None:
    report = _build_mixed_report(tmp_path)

    assert report.stage_summary() == {
        "compile": {
            "passed": 4, "xfail": 0, "failed": 0, "xpass": 0, "blocked": 1,
        },
        "assemble": {
            "passed": 2, "xfail": 1, "failed": 1, "xpass": 0, "blocked": 1,
        },
        "budget": {
            "passed": 2, "xfail": 0, "failed": 0, "xpass": 0, "blocked": 3,
        },
        "semantic": {
            "passed": 1, "xfail": 0, "failed": 0, "xpass": 1, "blocked": 3,
        },
    }
    assert report.owner_summary() == {
        "backend/op-lowering": {
            "declared": 1, "xfailed": 0, "xpassed": 1, "failed": 0,
        },
        "backend/regalloc": {
            "declared": 1, "xfailed": 1, "xpassed": 0, "failed": 0,
        },
    }
    assert report.xpassed_cases() == ["probe/xpass_case"]
    assert_report_consistent(report)


# ---------------------------------------------------------------------------
# JSON schema
# ---------------------------------------------------------------------------


def test_to_dict_schema_is_stable_and_json_serializable(
    tmp_path: Path,
) -> None:
    report = _build_mixed_report(tmp_path)
    data = report.to_dict()

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["suite"] == SUITE_NAME
    assert data["generated_at"] == GENERATED_AT
    assert TOP_LEVEL_KEYS <= set(data)
    assert len(data["results"]) == 5

    payload = json.dumps(data, sort_keys=True)
    assert json.loads(payload) == data

    for entry in data["results"]:
        assert RESULT_KEYS <= set(entry), entry["case_id"]
        assert set(entry["stages"]) == STAGE_KEYS, entry["case_id"]
        assert set(entry["metrics"]) == METRIC_KEYS, entry["case_id"]
        assert entry["case_id"] in report.specs
    skipped = data["results"][4]
    assert skipped["status"] == STATUS_SKIP
    assert all(value is None for value in skipped["stages"].values())

    passed = data["results"][0]
    assert isinstance(passed["actual"], str), (
        "numpy actual values must be stringified for JSON transport"
    )


def test_save_helpers_write_renderer_output(tmp_path: Path) -> None:
    report = _build_mixed_report(tmp_path)
    json_path = tmp_path / "out" / "suite.json"
    md_path = tmp_path / "out" / "suite.md"
    report.save_json(json_path)
    report.save_markdown(md_path)

    assert json.loads(json_path.read_text()) == report.to_dict()
    assert md_path.read_text() == report.to_markdown()


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_to_markdown_has_summary_row_per_case_and_prominent_xpass(
    tmp_path: Path,
) -> None:
    report = _build_mixed_report(tmp_path)
    markdown = report.to_markdown()

    assert (
        "| total | passed | xfailed | xpassed | failed | skipped |"
    ) in markdown
    assert "| 5 | 1 | 1 | 1 | 1 | 1 |" in markdown
    for r in report.results:
        assert markdown.count(f"| {r.case_id} |") == 1, r.case_id

    assert "**WARNING: 1 unexpected pass(es) (xpass)**" in markdown
    assert "## Unexpected passes (xpass — action required)" in markdown
    assert "- **probe/xpass_case**: declared stages now pass" in markdown
    assert "owner=backend/op-lowering" in markdown
    assert "## Expected failures (xfail)" in markdown
    assert "- probe/xfail_case (assemble)" in markdown

    assert "## Stage summary" in markdown
    assert "| compile | 4 | 0 | 0 | 0 | 1 |" in markdown
    assert "| assemble | 2 | 1 | 1 | 0 | 1 |" in markdown
    assert "| budget | 2 | 0 | 0 | 0 | 3 |" in markdown
    assert "| semantic | 1 | 0 | 0 | 1 | 3 |" in markdown

    assert "## Owner summary" in markdown
    assert "| backend/op-lowering | 1 | 0 | 1 | 0 |" in markdown
    assert "| backend/regalloc | 1 | 1 | 0 | 0 |" in markdown


def test_to_markdown_omits_xpass_warning_when_clean(tmp_path: Path) -> None:
    root = tmp_path / "probe_clean"
    spec = _make_spec(root, "pass_case")
    report = _report_from([_outcome(spec, STATUS_PASS)], [spec])
    markdown = report.to_markdown()
    assert "**WARNING:" not in markdown
    assert "## Unexpected passes (xpass — action required)" in markdown
    assert "None." in markdown


# ---------------------------------------------------------------------------
# Determinism and the anti-vacuity gate
# ---------------------------------------------------------------------------


def test_render_is_byte_stable_for_equal_inputs(tmp_path: Path) -> None:
    first = _build_mixed_report(tmp_path)
    second = _build_mixed_report(tmp_path)

    assert first.to_dict() == second.to_dict()
    assert json.dumps(first.to_dict(), sort_keys=True) == (
        json.dumps(second.to_dict(), sort_keys=True)
    )
    assert first.to_markdown() == second.to_markdown()
    assert first.to_markdown().encode() == first.to_markdown().encode()


def test_report_gate_is_not_vacuous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _build_mixed_report(tmp_path)
    assert_report_consistent(report)

    monkeypatch.setattr(
        SuiteReport, "pass_count", property(lambda self: 99),
    )
    assert report.summary()["passed"] == 99
    with pytest.raises(AssertionError):
        assert_report_consistent(report)
