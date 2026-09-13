"""Compare ScratchV and LLVM spill traffic on register-pressure DSL cases.

The suite deliberately uses straight-line scalar programs with at most four
external inputs.  That keeps LLVM stack-argument traffic and explicit DSL
allocas out of the measurement, so stack-relative loads/stores are a useful
proxy for register spills.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CASE_DIR = Path(__file__).with_name("regalloc_spill_cases")

EXPECTED_SCRATCHV_SPILL: dict[str, bool] = {
    "00_low_pressure_chain": False,
    "01_wide_fanout_32": True,
    "02_double_use_40": True,
    "03_lifetime_holes_36": True,
    "04_hot_cold_48": True,
}

_STACK_ACCESS_RE = re.compile(
    r"^\s*(?P<op>sd|ld|sw|lw|fsd|fld|fsw|flw)\s+"
    r"(?P<reg>[^,]+),\s*(?P<offset>-?\d+)\(sp\)"
)
_SAVED_REGS = frozenset(
    {"ra", "fp", *(f"s{i}" for i in range(12)), *(f"fs{i}" for i in range(12))}
)
_STORE_OPS = frozenset({"sd", "sw", "fsd", "fsw"})
_LOAD_OPS = frozenset({"ld", "lw", "fld", "flw"})

@dataclass(frozen=True)
class StackAccessStats:
    """Stack accesses split into allocator traffic and ABI frame traffic."""

    spill_slots: int = 0
    spill_stores: int = 0
    reloads: int = 0
    frame_saves: int = 0
    frame_restores: int = 0

    @property
    def spill_traffic(self) -> int:
        """Return the total number of spill stores and reloads."""
        return self.spill_stores + self.reloads


@dataclass(frozen=True)
class BackendStats:
    """Register-allocation measurements for one backend."""

    instructions: int
    virtual_registers: int | None
    peak_live: int | None
    physical_registers: int | None
    stack: StackAccessStats
    assembly: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe summary without embedding full assembly."""
        result = asdict(self)
        result.pop("assembly")
        result["stack"]["spill_traffic"] = self.stack.spill_traffic
        return result


@dataclass(frozen=True)
class ComparisonResult:
    """ScratchV/LLVM measurements for one DSL case."""

    name: str
    description: str
    scratchv: BackendStats
    llvm: BackendStats | None = None
    llvm_error: str = ""
    expect_scratchv_spill: bool | None = None

    @property
    def scratchv_expectation_met(self) -> bool | None:
        """Return whether ScratchV crossed the expected spill boundary."""

        if self.expect_scratchv_spill is None:
            return None
        spilled = self.scratchv.stack.spill_stores > 0
        return spilled is self.expect_scratchv_spill

    @property
    def spill_traffic_ratio(self) -> float | None:
        """Return ScratchV spill traffic divided by LLVM spill traffic."""
        if self.llvm is None:
            return None
        llvm_traffic = self.llvm.stack.spill_traffic
        scratchv_traffic = self.scratchv.stack.spill_traffic
        if llvm_traffic == 0:
            return float("inf") if scratchv_traffic else 1.0
        return scratchv_traffic / llvm_traffic

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe comparison summary."""
        ratio = self.spill_traffic_ratio
        return {
            "name": self.name,
            "description": self.description,
            "scratchv": self.scratchv.to_dict(),
            "llvm": self.llvm.to_dict() if self.llvm is not None else None,
            "llvm_error": self.llvm_error,
            "expect_scratchv_spill": self.expect_scratchv_spill,
            "scratchv_expectation_met": self.scratchv_expectation_met,
            "spill_traffic_ratio": "inf" if ratio == float("inf") else ratio,
        }


def discover_cases(case_dir: Path = CASE_DIR) -> list[Path]:
    """Return the sorted DSL pressure cases in *case_dir*."""
    cases = sorted(case_dir.glob("*.dsl"))
    if not cases:
        raise ValueError(f"No DSL benchmark cases found in {case_dir}")
    return cases


def _classify_stack_accesses(
    assembly: str,
    *,
    exclude_abi_frame: bool,
) -> StackAccessStats:
    """Classify anchored scalar stack accesses for either backend.

    The benchmark cases intentionally avoid stack arguments and explicit
    memory operations, so remaining stack-relative scalar accesses are spill
    traffic.  LLVM frame saves/restores can be excluded explicitly.
    """
    spill_offsets: set[int] = set()
    spill_stores = reloads = frame_saves = frame_restores = 0

    for line in assembly.splitlines():
        match = _STACK_ACCESS_RE.match(line)
        if match is None:
            continue
        op = match.group("op")
        reg = match.group("reg").strip()
        offset = int(match.group("offset"))

        is_frame_access = (
            exclude_abi_frame
            and op in {"sd", "ld", "fsd", "fld"}
            and reg in _SAVED_REGS
        )
        if is_frame_access:
            if op in _STORE_OPS:
                frame_saves += 1
            else:
                frame_restores += 1
            continue

        spill_offsets.add(offset)
        if op in _STORE_OPS:
            spill_stores += 1
        elif op in _LOAD_OPS:
            reloads += 1

    return StackAccessStats(
        spill_slots=len(spill_offsets),
        spill_stores=spill_stores,
        reloads=reloads,
        frame_saves=frame_saves,
        frame_restores=frame_restores,
    )


def classify_scratchv_stack_accesses(assembly: str) -> StackAccessStats:
    """Count ScratchV integer, floating-point, and wide spill traffic."""

    return _classify_stack_accesses(assembly, exclude_abi_frame=False)


def classify_llvm_stack_accesses(assembly: str) -> StackAccessStats:
    """Count LLVM spill traffic while excluding ABI frame management."""

    return _classify_stack_accesses(assembly, exclude_abi_frame=True)


def _peak_live(intervals: Sequence[Any]) -> int:
    """Calculate maximum simultaneous live intervals."""
    if not intervals:
        return 0
    first = min(interval.start for interval in intervals)
    last = max(interval.end for interval in intervals)
    return max(
        sum(interval.start <= position < interval.end for interval in intervals)
        for position in range(first, last)
    )


def _instruction_count(assembly: str) -> int:
    """Count RISC-V instructions using the project's canonical counter."""
    from scratchv.standalone.compare_codegen import count_riscv_instrs

    count, _ = count_riscv_instrs(assembly)
    return count


def compile_scratchv(
    source: str,
    phys_regs: list[str] | None = None,
) -> BackendStats:
    """Compile a DSL source through ScratchV's current linear allocator."""
    from scratchv.backend.instruction_select import InstructionSelector
    from scratchv.backend.regalloc_linear import (
        LinearScanAllocator,
        block_from_machine_instrs,
    )
    from scratchv.frontend.dsl_parser import DSLParser

    program = DSLParser().parse(source)
    machine = InstructionSelector(program).run()
    block = block_from_machine_instrs(machine)
    allocator = LinearScanAllocator(phys_regs=phys_regs)
    intervals = allocator.compute_live_intervals(block)
    allocator.allocate(intervals)
    assembly = allocator.get_allocated_code(block)

    stack = classify_scratchv_stack_accesses(assembly)
    if stack.spill_slots != allocator.spill_slot_count:
        raise RuntimeError(
            "ScratchV spill metric mismatch: "
            f"allocator={allocator.spill_slot_count}, assembly={stack.spill_slots}"
        )
    return BackendStats(
        instructions=_instruction_count(assembly),
        virtual_registers=len(intervals),
        peak_live=_peak_live(intervals),
        physical_registers=len(allocator.phys_regs),
        stack=stack,
        assembly=assembly,
    )


def compile_llvm(source: str, opt_level: int = 2) -> BackendStats:
    """Compile the same DSL source through LLVM's RISC-V backend."""
    from benchmarks.test_regalloc.bench_utils import llvmlite_ir_to_riscv
    from scratchv.backend.llvm_codegen import LLVMCodegen
    from scratchv.frontend.dsl_parser import DSLParser

    program = DSLParser().parse(source)
    ir_text = LLVMCodegen(program).emit()
    count, assembly, _ = llvmlite_ir_to_riscv(
        ir_text,
        features="+m,+f,+d",
        opt_level=opt_level,
    )
    return BackendStats(
        instructions=count,
        virtual_registers=None,
        peak_live=None,
        physical_registers=None,
        stack=classify_llvm_stack_accesses(assembly),
        assembly=assembly,
    )


def _description(source: str) -> str:
    """Extract the first comment line as the case description."""
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("# ")
        if stripped:
            break
    return ""


def run_case(
    path: Path,
    *,
    llvm_opt_level: int = 2,
    phys_regs: list[str] | None = None,
) -> ComparisonResult:
    """Compile one DSL case with both backends."""
    source = path.read_text()
    scratchv = compile_scratchv(source, phys_regs=phys_regs)
    expected_spill = EXPECTED_SCRATCHV_SPILL.get(path.stem)
    try:
        llvm = compile_llvm(source, opt_level=llvm_opt_level)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return ComparisonResult(
            name=path.stem,
            description=_description(source),
            scratchv=scratchv,
            llvm_error=f"{type(exc).__name__}: {exc}",
            expect_scratchv_spill=expected_spill,
        )
    return ComparisonResult(
        name=path.stem,
        description=_description(source),
        scratchv=scratchv,
        llvm=llvm,
        expect_scratchv_spill=expected_spill,
    )


def run_suite(
    case_dir: Path = CASE_DIR,
    *,
    llvm_opt_level: int = 2,
    phys_regs: list[str] | None = None,
) -> list[ComparisonResult]:
    """Run all discovered DSL pressure cases."""
    return [
        run_case(path, llvm_opt_level=llvm_opt_level, phys_regs=phys_regs)
        for path in discover_cases(case_dir)
    ]


def _format_ratio(ratio: float | None) -> str:
    if ratio is None:
        return "n/a"
    if ratio == float("inf"):
        return "inf"
    return f"{ratio:.2f}x"


def format_table(results: Sequence[ComparisonResult]) -> str:
    """Render a compact console comparison table."""
    rows = [
        "Case                     Peak | ScratchV slots S/R/T | "
        "LLVM slots S/R/T | Ratio | Expect",
        "-" * 95,
    ]
    for result in results:
        sv = result.scratchv.stack
        if result.llvm is None:
            llvm_text = " unavailable "
        else:
            ll = result.llvm.stack
            llvm_text = (
                f"{ll.spill_slots:>3} {ll.spill_stores:>3}/"
                f"{ll.reloads:<3}/{ll.spill_traffic:<3}"
            )
        expectation = result.scratchv_expectation_met
        expectation_text = (
            "n/a" if expectation is None else ("PASS" if expectation else "FAIL")
        )
        rows.append(
            f"{result.name:<24} {result.scratchv.peak_live or 0:>4} | "
            f"{sv.spill_slots:>3} {sv.spill_stores:>3}/"
            f"{sv.reloads:<3}/{sv.spill_traffic:<3} | "
            f"{llvm_text:<19} | {_format_ratio(result.spill_traffic_ratio):>6} | "
            f"{expectation_text}"
        )
    return "\n".join(rows)


def format_markdown(
    results: Sequence[ComparisonResult],
    llvm_opt_level: int,
) -> str:
    """Render a Markdown spill-comparison report."""
    lines = [
        "# DSL Register Spill Comparison",
        "",
        f"LLVM target: `riscv64-unknown-elf`, optimization level: `O{llvm_opt_level}`.",
        "ScratchV uses the current default register set from `regalloc_linear.py`.",
        "",
        (
            "| Case | Peak live | ScratchV slots | ScratchV store/reload | "
            "LLVM slots | LLVM store/reload | Traffic ratio | Expected boundary |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        sv = result.scratchv.stack
        if result.llvm is None:
            llvm_slots = llvm_traffic = "n/a"
        else:
            ll = result.llvm.stack
            llvm_slots = str(ll.spill_slots)
            llvm_traffic = f"{ll.spill_stores}/{ll.reloads}"
        expectation = result.scratchv_expectation_met
        expectation_text = (
            "n/a" if expectation is None else ("PASS" if expectation else "FAIL")
        )
        lines.append(
            f"| {result.name} | {result.scratchv.peak_live} | "
            f"{sv.spill_slots} | {sv.spill_stores}/{sv.reloads} | "
            f"{llvm_slots} | {llvm_traffic} | "
            f"{_format_ratio(result.spill_traffic_ratio)} | {expectation_text} |"
        )
    lines.extend(
        [
            "",
            (
                "`store/reload` counts only allocator-related stack traffic. "
                "ABI callee-saved frame saves/restores are excluded."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare ScratchV and LLVM spill traffic on DSL benchmarks",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=CASE_DIR,
        help="directory containing .dsl cases",
    )
    parser.add_argument(
        "--llvm-opt-level",
        type=int,
        choices=range(4),
        default=2,
        metavar="N",
        help="LLVM optimization level (default: 2)",
    )
    parser.add_argument(
        "--phys-reg-count",
        type=int,
        default=0,
        help="limit ScratchV physical registers; 0 uses the pipeline default",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="write JSON to stdout instead of the text table",
    )
    parser.add_argument("--json-output", type=Path, help="write JSON report to a file")
    parser.add_argument("--markdown", type=Path, help="write Markdown report to a file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parser().parse_args(argv)

    phys_regs = None
    if args.phys_reg_count:
        from scratchv.backend.regalloc_linear import _INT_REGS

        if not 1 <= args.phys_reg_count <= len(_INT_REGS):
            raise SystemExit(f"--phys-reg-count must be between 1 and {len(_INT_REGS)}")
        phys_regs = list(_INT_REGS[: args.phys_reg_count])

    results = run_suite(
        args.cases,
        llvm_opt_level=args.llvm_opt_level,
        phys_regs=phys_regs,
    )
    if not results:
        raise SystemExit("No benchmark cases matched")
    payload = {
        "llvm_opt_level": args.llvm_opt_level,
        "scratchv_phys_reg_count": results[0].scratchv.physical_registers,
        "cases": [result.to_dict() for result in results],
    }
    json_text = json.dumps(payload, indent=2, sort_keys=True)

    if args.json:
        print(json_text)
    else:
        print(format_table(results))
        for result in results:
            if result.llvm_error:
                print(f"LLVM unavailable for {result.name}: {result.llvm_error}")

    if args.json_output is not None:
        args.json_output.write_text(json_text + "\n")
    if args.markdown is not None:
        args.markdown.write_text(format_markdown(results, args.llvm_opt_level))

    llvm_failed = any(result.llvm is None for result in results)
    expectation_failed = any(
        result.scratchv_expectation_met is False for result in results
    )
    return int(llvm_failed or expectation_failed)


if __name__ == "__main__":
    raise SystemExit(main())
