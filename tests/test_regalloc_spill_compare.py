"""Tests for the DSL register-spill comparison benchmark."""

from pathlib import Path

import pytest

from benchmarks.bench_regalloc_spill_compare import (
    CASE_DIR,
    EXPECTED_SCRATCHV_SPILL,
    StackAccessStats,
    classify_llvm_stack_accesses,
    classify_scratchv_stack_accesses,
    compile_scratchv,
    discover_cases,
)


def test_discover_cases_returns_the_complete_pressure_suite() -> None:
    cases = discover_cases(CASE_DIR)

    assert [case.stem for case in cases] == [
        "00_low_pressure_chain",
        "01_wide_fanout_32",
        "02_double_use_40",
        "03_lifetime_holes_36",
        "04_hot_cold_48",
    ]


def test_llvm_stack_classifier_excludes_abi_frame_saves() -> None:
    asm = """
        fsd fs0, 24(sp)
        sd ra, 16(sp)
        fsw ft0, 12(sp)
        flw ft1, 12(sp)
        sw a0, 8(sp)
        lw a1, 8(sp)
        fld fs0, 24(sp)
        ld ra, 16(sp)
    """

    assert classify_llvm_stack_accesses(asm) == StackAccessStats(
        spill_slots=2,
        spill_stores=2,
        reloads=2,
        frame_saves=2,
        frame_restores=2,
    )


def test_scratchv_stack_classifier_covers_all_supported_widths() -> None:
    asm = """
        sw t0, -4(sp)
        lw t0, -4(sp)
        fsw ft0, -8(sp)
        flw ft0, -8(sp)
        sd t1, -16(sp)
        ld t1, -16(sp)
        fsd ft1, -24(sp)
        fld ft1, -24(sp)
        # sw t2, -28(sp)
        addi t0, t0, 1  # lw t3, -32(sp)
    """

    assert classify_scratchv_stack_accesses(asm) == StackAccessStats(
        spill_slots=4,
        spill_stores=4,
        reloads=4,
    )


@pytest.mark.parametrize(
    ("case_name", "expects_spill"),
    [
        ("00_low_pressure_chain.dsl", False),
        ("01_wide_fanout_32.dsl", True),
        ("02_double_use_40.dsl", True),
        ("03_lifetime_holes_36.dsl", True),
        ("04_hot_cold_48.dsl", True),
    ],
)
def test_cases_straddle_the_scratchv_spill_boundary(
    case_name: str,
    expects_spill: bool,
) -> None:
    source = (CASE_DIR / case_name).read_text()

    result = compile_scratchv(source)

    assert (result.stack.spill_slots > 0) is expects_spill
    assert result.peak_live is not None and result.peak_live > 0
    assert result.virtual_registers is not None and result.virtual_registers > 0
    assert result.physical_registers == 19


def test_expected_spill_metadata_covers_every_case() -> None:
    assert set(EXPECTED_SCRATCHV_SPILL) == {
        case.stem for case in discover_cases(CASE_DIR)
    }


def test_discover_cases_rejects_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No DSL benchmark cases"):
        discover_cases(tmp_path)
