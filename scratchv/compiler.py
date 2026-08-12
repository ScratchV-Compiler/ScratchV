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

import time
from dataclasses import dataclass, field
from typing import Any

from scratchv.ir.types import Program
from scratchv.pass_interface import (
    OptimizationPass,
    OptimizationPassError,
    OptimizationReport,
    PassExecutionStats,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CompilerConfig
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CompilerConfig:
    """All compiler options in one place.

    Attributes:
        backend:        ``"riscv"`` or ``"llvm"``.
        optimize_level: ``"none"``, ``"basic"``, or ``"all"``.
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
    reg_alloc: str = "linear"
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


# ═══════════════════════════════════════════════════════════════════════════════
# PassManager
# ═══════════════════════════════════════════════════════════════════════════════

class PassManager:
    """Register and run IR optimization passes in a deterministic order.

    Usage::

        pm = PassManager()
        pm.register(ConstantFolder())
        pm.register(DeadCodeEliminator())
        report = pm.run(program)
    """

    def __init__(self, name: str = "pipeline"):
        if not isinstance(name, str):
            raise TypeError("pipeline name must be a string")
        if not name.strip():
            raise ValueError("pipeline name must not be empty")
        self._name = name
        self._passes: list[OptimizationPass] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def passes(self) -> list[OptimizationPass]:
        return list(self._passes)

    def register(self, pass_: OptimizationPass) -> "PassManager":
        """Append an optimization pass and return ``self`` for chaining."""
        if not isinstance(pass_, OptimizationPass):
            raise TypeError("registered pass must implement OptimizationPass")
        if not isinstance(pass_.name, str):
            raise TypeError("optimization pass name must be a string")
        if not pass_.name.strip():
            raise ValueError("optimization pass name must not be empty")
        self._passes.append(pass_)
        return self

    def add(self, pass_: OptimizationPass) -> "PassManager":
        """Compatibility alias for :meth:`register`."""
        return self.register(pass_)

    def run(self, program: Program) -> OptimizationReport:
        """Optimize ``program`` in place and return ordered execution stats."""
        if not isinstance(program, Program):
            raise TypeError("PassManager.run() requires a Program")

        executions: list[PassExecutionStats] = []
        for index, pass_ in enumerate(self._passes):
            t0 = time.perf_counter()
            try:
                changes = pass_.optimize(program)
                self._validate_change_count(changes)
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                completed_report = self._make_report(executions)
                raise OptimizationPassError(
                    pass_index=index,
                    pass_name=pass_.name,
                    elapsed_seconds=elapsed,
                    completed_report=completed_report,
                    cause=exc,
                ) from exc
            elapsed = time.perf_counter() - t0
            executions.append(
                PassExecutionStats(
                    index=index,
                    name=pass_.name,
                    changes=changes,
                    elapsed_seconds=elapsed,
                )
            )

        return self._make_report(executions)

    @staticmethod
    def _validate_change_count(changes: int) -> None:
        if isinstance(changes, bool) or not isinstance(changes, int):
            raise TypeError("optimization pass must return an integer change count")
        if changes < 0:
            raise ValueError("optimization pass change count must be non-negative")

    def _make_report(self, executions: list[PassExecutionStats]) -> OptimizationReport:
        records = tuple(executions)
        return OptimizationReport(
            pipeline_name=self._name,
            executions=records,
            total_changes=sum(item.changes for item in records),
            elapsed_seconds=sum(item.elapsed_seconds for item in records),
        )

    def report(self) -> str:
        """Return a summary of all registered passes."""
        lines = [f"PassManager '{self._name}' ({len(self._passes)} passes):"]
        for p in self._passes:
            lines.append(f"  {p.name}")
        return "\n".join(lines)


def create_optimization_pass_manager(level: str) -> PassManager:
    """Build the canonical ``none``/``basic``/``all`` IR pipeline."""
    if not isinstance(level, str):
        raise TypeError("optimization level must be a string")
    if level not in {"none", "basic", "all"}:
        raise ValueError("optimization level must be one of: none, basic, all")

    manager = PassManager("optimizer")
    if level == "none":
        return manager

    from scratchv.optimizer.constant_folding import ConstantFolder
    from scratchv.optimizer.dead_code import DeadCodeEliminator

    manager.register(ConstantFolder())
    manager.register(DeadCodeEliminator())

    if level == "all":
        from scratchv.optimizer.licm import LICM
        from scratchv.optimizer.muladd_fusion import MulAddFusion
        from scratchv.optimizer.peephole import IRPeepholeOptimizer

        manager.register(IRPeepholeOptimizer())
        manager.register(MulAddFusion())
        manager.register(LICM())

    return manager


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

    # ── Public API ──────────────────────────────────────────────────────────

    def compile(self, input_path: str, output_path: str | None = None,
                dsl_source: str | None = None) -> CompileResult:
        """Compile an input file and write output.

        Args:
            input_path:  Path to .onnx or .dsl file.
            output_path: Output file path (auto-derived if None).
            dsl_source:  Inline DSL source (used with ``--dsl`` flag).

        Returns:
            A ``CompileResult`` with output text and statistics.
        """
        warnings: list[str] = []

        # Resolve output path
        if output_path is None:
            output_path = "output.ll" if self.config.backend == "llvm" else "output.s"

        # --- 1. Parse ---
        try:
            program = self._parse(input_path, dsl_source)
        except Exception as e:
            return CompileResult(
                success=False, errors=[f"Parse error: {e}"],
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
                "; --- IR Dump (before) ---\n" + ir_dump_before +
                "\n; --- IR Dump (after" +
                (f" {opt_message}" if opt_message else "") +
                ") ---\n" + ir_dump_after
            )

        # --- 4. Code generation ---
        try:
            asm_text = self._generate_code(program)
        except Exception as e:
            return CompileResult(
                success=False, errors=[f"Codegen error: {e}"],
                ir_dump=ir_dump,
            )

        # --- 5. Post-codegen passes ---
        asm_text = self._run_asm_passes(asm_text, warnings)

        # --- 6. Cycle estimation ---
        cycle_report = ""
        if self.config.cycle_stats:
            from scratchv.backend.cycle_estimator import (
                PipelineCycleEstimator, PipelineConfig,
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
            except Exception as e:
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
            },
            warnings=warnings,
        )

    # ── Internal: parse ─────────────────────────────────────────────────────

    def _parse(self, input_path: str, dsl_source: str | None = None):
        """Parse input into an IR Program."""
        use_dsl = (
            dsl_source is not None
            or (input_path and input_path.endswith(".dsl"))
        )

        if use_dsl:
            source = dsl_source
            if source is None and input_path:
                with open(input_path) as f:
                    source = f.read()
            # Try extended DSL first
            try:
                from scratchv.frontend.dsl_extended import ExtendedDSLParser
                return ExtendedDSLParser().parse(source)
            except Exception:
                from scratchv.frontend.dsl_parser import DSLParser
                return DSLParser().parse(source)
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
        manager = create_optimization_pass_manager(self.config.optimize_level)
        return manager.run(program)

    def _optimization_stats(self, report: OptimizationReport) -> dict[str, Any]:
        """Convert an immutable report to the CompileResult stats schema."""
        return {
            "level": self.config.optimize_level,
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
            f"[{item.name}] {item.changes} change(s)"
            for item in report.executions
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
        from scratchv.backend.instruction_select import InstructionSelector
        from scratchv.backend.register_alloc import RegisterAllocator
        from scratchv.backend.asm_emit import AsmEmitter

        selector = InstructionSelector(program)
        machine_instrs = selector.run()

        # Linear-scan: skip greedy allocator, use liveness-driven allocator
        if self.config.reg_alloc == "linear":
            from scratchv.backend.regalloc_linear import (
                LinearScanAllocator, block_from_machine_instrs,
            )
            ls_insts = block_from_machine_instrs(machine_instrs)
            lsa = LinearScanAllocator()
            return lsa.emit(ls_insts)

        alloc = RegisterAllocator(machine_instrs, mode=self.config.reg_alloc)
        allocated = alloc.run()
        emitter = AsmEmitter(allocated)
        return emitter.emit()

    def _generate_riscv_dag(self, program) -> str:
        """DAG-based instruction selection pipeline."""
        from scratchv_dag.selection_dag import DAGBuilder, DAGCombiner, DAGScheduler
        from scratchv.backend.register_alloc import RegisterAllocator
        from scratchv.backend.asm_emit import AsmEmitter

        builder = DAGBuilder(program)
        dag = builder.run()

        combiner = DAGCombiner(dag)
        combiner.run()

        scheduler = DAGScheduler(dag)
        machine_instrs = scheduler.run()

        alloc = RegisterAllocator(machine_instrs, mode=self.config.reg_alloc)
        allocated = alloc.run()

        emitter = AsmEmitter(allocated)
        return emitter.emit()

    # ── Internal: post-codegen passes ───────────────────────────────────────

    def _run_asm_passes(self, asm_text: str, warnings: list[str]) -> str:
        """Run assembly-level passes (peephole, const-merge, beautify, etc.)."""
        if self.config.peephole_asm:
            from scratchv.backend.asm_peephole import AsmPeepholeOptimizer
            opt = AsmPeepholeOptimizer()
            asm_text, changes = opt.optimize(asm_text)
            if changes:
                warnings.append(f"Asm peephole: {changes} changes")

        if self.config.const_merge:
            from scratchv.backend.const_merge import merge_constants
            asm_text, changes = merge_constants(asm_text)
            if changes:
                warnings.append(f"Const merge: {changes} changes")

        if self.config.schedule:
            from scratchv.backend.inst_scheduler import (
                InstructionScheduler, parse_instructions,
            )
            sched = InstructionScheduler()
            insts = parse_instructions(asm_text)
            dag = sched.build_dag(insts)
            scheduled = sched.schedule(dag)
            asm_text = "\n".join(
                f"  {inst.opcode} " + ", ".join(inst.operands)
                for inst in scheduled
            )

        if self.config.beautify_asm:
            from scratchv.backend.asm_beautifier import beautify_asm
            asm_text = beautify_asm(asm_text)

        if self.config.count_instr:
            from scratchv.backend.inst_counter import count_instructions
            counts = count_instructions(asm_text)
            total = sum(v for k, v in counts.items()
                        if not k.startswith("_") and isinstance(v, int))
            warnings.append(f"Instruction count: {total}")

        return asm_text
