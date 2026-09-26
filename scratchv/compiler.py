"""Compiler driver and pass manager for ScratchV.

Provides a ``PassManager`` that runs IR optimization passes and a
``CompilerDriver`` that orchestrates the full compilation pipeline:
parse → optimise → codegen → verify → emit.

Usage::

    from scratchv.compiler import CompilerDriver, CompilerConfig

    driver = CompilerDriver(CompilerConfig(
        backend="riscv",
        optimize_level="all",
        dump_ir=True,
    ))
    result = driver.compile("model.onnx", "output.s")
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scratchv.ir.types import Program
from scratchv.pass_interface import (
    OptimizationPassError,
    OptimizationReport,
)
from scratchv.pass_manager import PassManager, create_optimization_pass_manager

__all__ = [
    "CompileResult",
    "CompilerConfig",
    "CompilerDriver",
    "PassManager",
    "create_optimization_pass_manager",
]

# ═══════════════════════════════════════════════════════════════════════════════
# CompilerConfig
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CompilerConfig:
    """All compiler options in one place.

    Attributes:
        backend:        ``"riscv"`` or ``"llvm"``.
        optimize_level: ``"none"``, ``"basic"``, or ``"all"``.
        passes:         Explicit ordered IR pass names; None uses the preset.
        disabled_passes: IR pass names excluded from the selected pipeline.
        reg_alloc:      ``"naive"`` or ``"greedy"`` (also ``"linear"``).
        dump_ir:        Print IR dumps during compilation.
        verify:         Run ONNX Runtime / numpy verification.
        rtol:           Relative tolerance for verification.
        atol:           Absolute tolerance for verification.
        use_logger:     Use structured logger instead of print().
        log_level:      Log level (DEBUG, INFO, WARNING, ERROR).
        use_dag_isel:   Use DAG-based instruction selection.
        beautify_asm:   Run assembly beautifier on output.
        peephole_asm:   Run assembly-level peephole optimiser.
        const_merge:    Run constant-load merge pass.
        schedule:       Run instruction scheduler.
        count_instr:    Print instruction count statistics.
        cycle_stats:    Run 5-stage pipeline cycle estimation (detailed).
        enable_forwarding:  Enable forwarding in cycle estimator.
        branch_predictor:   Branch predictor mode for cycle estimator.
    """

    backend: str = "riscv"
    optimize_level: str = "none"
    reg_alloc: str = "greedy"
    dump_ir: bool = False
    verify: bool = False
    rtol: float = 1e-5
    atol: float = 1e-8
    use_logger: bool = False
    log_level: str = "INFO"
    use_dag_isel: bool = False
    beautify_asm: bool = False
    peephole_asm: bool = False
    const_merge: bool = False
    schedule: bool = False
    count_instr: bool = False
    cycle_stats: bool = False
    enable_forwarding: bool = True
    branch_predictor: str = "always_not_taken"
    passes: tuple[str, ...] | None = None
    disabled_passes: tuple[str, ...] = ()


# ═══════════════════════════════════════════════════════════════════════════════
# CompileResult
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CompileResult:
    """Result of a full compilation.

    Attributes:
        success:      Whether compilation succeeded.
        output_text:  Generated assembly / LLVM IR text.
        output_path:  Path the output was written to.
        ir_dump:      Optional IR dump text (if --dump-ir was set).
        stats:        Aggregated statistics from all passes.
        errors:       List of fatal error messages.
        warnings:     List of non-fatal warning messages.
    """

    success: bool
    output_text: str = ""
    output_path: str = ""
    ir_dump: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: list[Any] = field(default_factory=list)
    diagnostic_limit_reached: bool = False
    diagnostic_limit: int = 20

    def summary(self) -> str:
        """Return a one-line summary."""
        if self.success:
            return f"OK → {self.output_path} ({len(self.output_text)} bytes)"
        return f"FAILED: {'; '.join(self.errors)}"


# ═══════════════════════════════════════════════════════════════════════════════
# CompilerDriver
# ═══════════════════════════════════════════════════════════════════════════════


class CompilerDriver:
    """Orchestrates the full compilation pipeline.

    Encapsulates all knowledge about how to run the compiler.  The CLI
    (``main.py``) only translates command-line arguments into a
    ``CompilerConfig`` and delegates to the driver.

    Usage::

        driver = CompilerDriver(CompilerConfig(backend="riscv",
                                                optimize_level="all"))
        result = driver.compile("model.onnx", "output.s")
    """

    def __init__(self, config: CompilerConfig | None = None):
        self.config = config or CompilerConfig()
        self._last_register_map: dict[str, str] = {}
        self._assembly_report = OptimizationReport("assembly", (), 0, 0.0)

    # ── Public API ──────────────────────────────────────────────────────────

    def compile(
        self,
        input_path: str,
        output_path: str | None = None,
        dsl_source: str | None = None,
    ) -> CompileResult:
        """Compile an input file and write output.

        Args:
            input_path:  Path to .onnx or .dsl file.
            output_path: Output file path (auto-derived if None).
            dsl_source:  Inline DSL source (used with ``--dsl`` flag).

        Returns:
            A ``CompileResult`` with output text and statistics.
        """
        warnings: list[str] = []
        self._last_register_map = {}
        self._assembly_report = OptimizationReport("assembly", (), 0, 0.0)

        # Resolve output path
        if output_path is None:
            output_path = "output.ll" if self.config.backend == "llvm" else "output.s"

        use_dsl = dsl_source is not None or (input_path and input_path.endswith(".dsl"))

        if use_dsl:
            source = dsl_source
            if source is None and input_path:
                with open(input_path) as source_file:
                    source = source_file.read()
            from scratchv.frontend.dsl_extended import ExtendedDSLParser

            parser = ExtendedDSLParser()
            collector = parser.validate(
                source or "",
                filename=input_path or "<dsl>",
            )
            if collector.has_errors:
                diagnostics = collector.errors
                return CompileResult(
                    success=False,
                    errors=[str(error) for error in diagnostics],
                    diagnostics=diagnostics,
                    diagnostic_limit_reached=collector.limit_reached,
                    diagnostic_limit=collector.max_errors,
                )

        # --- 1. Parse ---
        try:
            program = self._parse(input_path, dsl_source)
        except Exception as e:
            if use_dsl:
                from scratchv.frontend.dsl_errors import DSLSyntaxError

                if isinstance(e, DSLSyntaxError):
                    return CompileResult(
                        success=False,
                        errors=[str(e)],
                        diagnostics=[e],
                    )
                raise
            return CompileResult(
                success=False,
                errors=[f"Parse error: {e}"],
            )

        ir_dump_before = ""
        if self.config.dump_ir:
            from scratchv.ir.printer import IRPrinter

            ir_dump_before = IRPrinter(program).dump()

        # --- 2. Verify IR (if configured) ---
        if self.config.use_logger:
            self._verify_ir(program, warnings)

        # --- 3. Optimize ---
        opt_message = ""
        try:
            optimization_report = self._run_optimizations(program)
        except (OptimizationPassError, TypeError, ValueError) as exc:
            if isinstance(exc, OptimizationPassError):
                completed_report = exc.completed_report
            else:
                completed_report = OptimizationReport("optimizer", (), 0, 0.0)
            optimization_stats = self._optimization_stats(completed_report)
            return CompileResult(
                success=False,
                errors=[f"Optimization error: {exc}"],
                stats={
                    "optimization": optimization_stats,
                    "opt_message": opt_message,
                    "cycle_report": "",
                },
            )
        opt_message = self._optimization_message(optimization_report)

        ir_dump_after = ""
        if self.config.dump_ir:
            from scratchv.ir.printer import IRPrinter

            ir_dump_after = IRPrinter(program).dump()

        ir_dump = ""
        if self.config.dump_ir:
            ir_dump = (
                "; --- IR Dump (before) ---\n"
                + ir_dump_before
                + "\n; --- IR Dump (after"
                + (f" {opt_message}" if opt_message else "")
                + ") ---\n"
                + ir_dump_after
            )

        # --- 4. Code generation ---
        try:
            asm_text = self._generate_code(program)
        except Exception as e:  # noqa: BLE001
            return CompileResult(
                success=False,
                errors=[f"Codegen error: {e}"],
                ir_dump=ir_dump,
            )

        # --- 5. Post-codegen passes ---
        try:
            asm_text = self._run_asm_passes(asm_text, warnings)
        except OptimizationPassError as exc:
            return CompileResult(
                success=False,
                errors=[f"Assembly pass error: {exc}"],
                stats={"assembly": self._pass_stats(exc.completed_report)},
                warnings=warnings,
            )

        # --- 6. Cycle estimation ---
        cycle_report = ""
        if self.config.cycle_stats:
            from scratchv.backend.cycle_estimator import (
                PipelineConfig,
                PipelineCycleEstimator,
            )

            pconfig = PipelineConfig(
                enable_forwarding=self.config.enable_forwarding,
                branch_predictor=self.config.branch_predictor,
            )
            estimator = PipelineCycleEstimator(pconfig)
            try:
                cstats = estimator.estimate(asm_text)
                cycle_report = estimator.report(cstats)
                warnings.append(estimator.report_short(cstats))
            except Exception as e:  # noqa: BLE001
                warnings.append(f"Cycle estimation failed: {e}")

        # --- 7. Write output ---
        with open(output_path, "w") as f:
            f.write(asm_text)

        return CompileResult(
            success=True,
            output_text=asm_text,
            output_path=output_path,
            ir_dump=ir_dump,
            stats={
                "optimization": self._optimization_stats(optimization_report),
                "opt_message": opt_message,
                "cycle_report": cycle_report,
                "register_map": dict(self._last_register_map),
                "assembly": self._pass_stats(self._assembly_report),
            },
            warnings=warnings,
        )

    # ── Internal: parse ─────────────────────────────────────────────────────

    def _parse(self, input_path: str, dsl_source: str | None = None):
        """Parse input into an IR Program."""
        use_dsl = dsl_source is not None or (input_path and input_path.endswith(".dsl"))

        if use_dsl:
            source = dsl_source
            if source is None and input_path:
                with open(input_path) as f:
                    source = f.read()
            from scratchv.frontend.dsl_extended import ExtendedDSLParser

            return ExtendedDSLParser().parse(
                source or "",
                filename=input_path or "<dsl>",
            )
        else:
            from scratchv.frontend.onnx_parser import ONNXParser

            return ONNXParser().parse(input_path)

    # ── Internal: verify IR ─────────────────────────────────────────────────

    def _verify_ir(self, program, warnings: list[str]) -> None:
        """Run IR verifier and collect warnings."""
        from scratchv.analysis.ir_verifier import IRVerifier

        verifier = IRVerifier(program)
        issues = verifier.verify()
        for issue in issues:
            msg = str(issue)
            if issue.level.value == "error":
                warnings.append(f"IR: {msg}")
            else:
                warnings.append(f"IR(warning): {msg}")

    # ── Internal: optimizations ─────────────────────────────────────────────

    def _run_optimizations(self, program: Program) -> OptimizationReport:
        """Run all configured optimization passes."""
        manager = create_optimization_pass_manager(
            self.config.optimize_level,
            passes=self.config.passes,
            disabled_passes=self.config.disabled_passes,
        )
        return manager.run(program)

    def _optimization_stats(self, report: OptimizationReport) -> dict[str, Any]:
        """Convert an immutable report to the CompileResult stats schema."""
        return {"level": self.config.optimize_level, **self._pass_stats(report)}

    @staticmethod
    def _pass_stats(report: OptimizationReport) -> dict[str, Any]:
        """Serialize ordered execution statistics for either pipeline stage."""
        return {
            "total_changes": report.total_changes,
            "elapsed_seconds": report.elapsed_seconds,
            "passes": [
                {
                    "index": item.index,
                    "name": item.name,
                    "changes": item.changes,
                    "elapsed_seconds": item.elapsed_seconds,
                }
                for item in report.executions
            ],
        }

    @staticmethod
    def _optimization_message(report: OptimizationReport) -> str:
        """Build the legacy human-readable optimization summary."""
        return "; ".join(
            f"[{item.name}] {item.changes} change(s)" for item in report.executions
        )

    # ── Internal: code generation ───────────────────────────────────────────

    def _generate_code(self, program) -> str:
        """Run code generation (instruction selection + regalloc + emit)."""
        if self.config.backend == "llvm":
            from scratchv.backend.llvm_codegen import LLVMCodegen

            return LLVMCodegen(program).emit()

        # RISC-V backend
        if self.config.use_dag_isel:
            return self._generate_riscv_dag(program)
        return self._generate_riscv_linear(program)

    def _generate_riscv_linear(self, program) -> str:
        """Standard RISC-V pipeline."""
        from scratchv.backend.asm_emit import AsmEmitter
        from scratchv.backend.instruction_select import InstructionSelector
        from scratchv.backend.register_alloc import RegisterAllocator

        selector = InstructionSelector(program)
        machine_instrs = selector.run()

        # Linear-scan path: allocate on *unallocated* MachineInstrs.
        # (Previously RegisterAllocator ran first with mode="linear", which
        # fell through to greedy — so LinearScan never saw virtual regs.)
        if self.config.reg_alloc == "linear":
            from scratchv.backend.regalloc_linear import (
                LinearScanAllocator,
                block_from_machine_instrs,
            )

            ls_insts = block_from_machine_instrs(machine_instrs)
            lsa = LinearScanAllocator()
            assembly = lsa.emit(ls_insts)
            self._last_register_map = dict(lsa.alloc_map)
            from scratchv.backend.abi_frame import apply_abi_frames

            return apply_abi_frames(assembly, lsa.spill_slot_count)

        mode = (
            self.config.reg_alloc
            if self.config.reg_alloc
            in (
                "naive",
                "greedy",
            )
            else "greedy"
        )
        alloc = RegisterAllocator(machine_instrs, mode=mode)
        allocated = alloc.run()
        self._last_register_map = alloc.register_map
        emitter = AsmEmitter(allocated)
        assembly = emitter.emit()
        from scratchv.backend.abi_frame import apply_abi_frames

        return apply_abi_frames(assembly, alloc.spill_slot_count)

    def _generate_riscv_dag(self, program) -> str:
        """DAG-based instruction selection pipeline."""
        from scratchv.backend.asm_emit import AsmEmitter
        from scratchv.backend.register_alloc import RegisterAllocator
        from scratchv_dag.selection_dag import DAGBuilder, DAGCombiner, DAGScheduler

        builder = DAGBuilder(program)
        dag = builder.run()

        combiner = DAGCombiner(dag)
        combiner.run()

        scheduler = DAGScheduler(dag)
        machine_instrs = scheduler.run()

        if self.config.reg_alloc == "linear":
            from scratchv.backend.regalloc_linear import (
                LinearScanAllocator,
                block_from_machine_instrs,
            )

            ls_insts = block_from_machine_instrs(machine_instrs)
            lsa = LinearScanAllocator()
            assembly = lsa.emit(ls_insts)
            self._last_register_map = dict(lsa.alloc_map)
            from scratchv.backend.abi_frame import apply_abi_frames

            return apply_abi_frames(assembly, lsa.spill_slot_count)

        mode = (
            self.config.reg_alloc
            if self.config.reg_alloc
            in (
                "naive",
                "greedy",
            )
            else "greedy"
        )
        alloc = RegisterAllocator(machine_instrs, mode=mode)
        allocated = alloc.run()
        self._last_register_map = alloc.register_map

        emitter = AsmEmitter(allocated)
        assembly = emitter.emit()
        from scratchv.backend.abi_frame import apply_abi_frames

        return apply_abi_frames(assembly, alloc.spill_slot_count)

    # ── Internal: post-codegen passes ───────────────────────────────────────

    def _run_asm_passes(self, asm_text: str, warnings: list[str]) -> str:
        """Run assembly-level passes (peephole, const-merge, beautify, etc.)."""
        from scratchv.assembly_passes import create_assembly_registry

        selected = []
        for name, enabled in (
            ("asm-peephole", self.config.peephole_asm),
            ("const-merge", self.config.const_merge),
            ("schedule", self.config.schedule),
            ("beautify", self.config.beautify_asm),
            ("count-instr", self.config.count_instr),
        ):
            if enabled:
                selected.append(name)
        manager = create_assembly_registry().build(selected, pipeline_name="assembly")
        result = manager.run_pipeline(asm_text)
        self._assembly_report = result.report
        warnings.extend(result.warnings)
        return result.data
