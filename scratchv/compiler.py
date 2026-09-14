"""Compiler driver and pass manager for ScratchV.

Provides a ``PassManager`` that chains compiler passes together and a
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

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Optional

from scratchv.pass_interface import CompilerPass, PassResult
from scratchv.utils.logger import (
    get_logger,
    init_logger,
    log_phase,
    log_progress,
    log_step,
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
        log_level:      Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file:       Optional plain-text log file (always DEBUG level).
        log_color:      Enable ANSI colors on the console handler.
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
    log_file: str | None = None
    log_color: bool = True
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
    """Manages a sequence of compiler passes and runs them in order.

    Each pass receives the output of the previous pass as its input. The
    first pass receives the initial ``input_data`` provided to ``run()``.

    Usage::

        pm = PassManager()
        pm.add(ConstantFolder(program))
        pm.add(DeadCodeEliminator(program))
        result = pm.run(program)
    """

    def __init__(self, name: str = "pipeline",
                 log: logging.Logger | None = None):
        self._name = name
        self._passes: list[CompilerPass] = []
        self._log = log

    @property
    def name(self) -> str:
        return self._name

    @property
    def passes(self) -> list[CompilerPass]:
        return list(self._passes)

    def add(self, pass_: CompilerPass) -> "PassManager":
        """Add a pass to the end of the pipeline.  Returns self for chaining."""
        self._passes.append(pass_)
        return self

    def run(self, input_data: Any) -> PassResult:
        """Run all passes sequentially.

        Returns the final ``PassResult``.  If any pass returns ``None``
        data the pipeline stops early and returns the last result.
        """
        data = input_data
        total_changes = 0
        messages: list[str] = []
        all_warnings: list[str] = []
        timings: dict[str, float] = {}
        total = len(self._passes)

        for i, p in enumerate(self._passes, 1):
            if self._log is not None:
                log_progress(self._log.name, i, total, f"pass {p.name}")
            t0 = time.perf_counter()
            try:
                result = p.run(data)
            except Exception as exc:
                if self._log is not None:
                    self._log.error("pass '%s' failed: %s", p.name, exc)
                return PassResult(
                    data=None,
                    changes=total_changes,
                    message=f"Pass '{p.name}' failed: {exc}",
                    warnings=all_warnings,
                )
            elapsed = time.perf_counter() - t0
            timings[p.name] = elapsed
            if self._log is not None:
                self._log.debug("pass %s: %d change(s) in %.3fs",
                                p.name, result.changes, elapsed)

            if result.data is None:
                if self._log is not None:
                    self._log.warning("pipeline stopped after '%s': %s",
                                      p.name, result.message)
                return PassResult(
                    data=None,
                    changes=total_changes,
                    message=f"Pipeline stopped after '{p.name}': {result.message}",
                    warnings=all_warnings + result.warnings,
                )

            data = result.data
            total_changes += result.changes
            if result.message:
                messages.append(f"[{p.name}] {result.message}")
            all_warnings.extend(result.warnings)

        if self._log is not None:
            elapsed_total = sum(timings.values())
            self._log.info("%s: %d pass(es), %d change(s) in %.3fs",
                           self._name, total, total_changes, elapsed_total)

        return PassResult(
            data=data,
            changes=total_changes,
            message="; ".join(messages) if messages else "pipeline complete",
            warnings=all_warnings,
        )

    def report(self) -> str:
        """Return a summary of all registered passes."""
        lines = [f"PassManager '{self._name}' ({len(self._passes)} passes):"]
        for p in self._passes:
            lines.append(f"  {p.name}")
        return "\n".join(lines)


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
        self._log: logging.Logger | None = None

    # ── Internal: logging helpers ───────────────────────────────────────────

    def _logger(self) -> logging.Logger | None:
        """Return the driver logger, or ``None`` when logging is disabled."""
        if not self.config.use_logger:
            return None
        if self._log is None:
            self._log = get_logger("compiler")
        return self._log

    def _pass_logger(self) -> logging.Logger | None:
        """Return the per-pass detail logger (``scratchv.compiler.passes``)."""
        if self._logger() is None:
            return None
        return get_logger("compiler.passes")

    def _phase(self, name: str, description: str):
        """Return a ``log_phase`` context or a no-op when logging is off."""
        if not self.config.use_logger:
            return nullcontext()
        return log_phase(name, description)

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
        errors: list[str] = []
        warnings: list[str] = []

        # Initialise structured logging once per driver instance.
        if self.config.use_logger and self._log is None:
            init_logger(level=self.config.log_level,
                        log_file=self.config.log_file,
                        use_color=self.config.log_color)
            self._log = get_logger("compiler")
            self._log.debug("config: backend=%s optimize=%s reg_alloc=%s",
                            self.config.backend, self.config.optimize_level,
                            self.config.reg_alloc)

        # Resolve output path
        if output_path is None:
            output_path = "output.ll" if self.config.backend == "llvm" else "output.s"

        use_dsl = (
            dsl_source is not None
            or (input_path and input_path.endswith(".dsl"))
        )

        if use_dsl:
            source = dsl_source
            if source is None and input_path:
                with open(input_path) as source_file:
                    source = source_file.read()
            from scratchv.frontend.dsl_extended import ExtendedDSLParser
            parser = ExtendedDSLParser()
            collector = parser.validate(
                source or "", filename=input_path or "<dsl>",
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
            with self._phase(
                "compiler.parse",
                f"Parsing {input_path or '<inline DSL>'}",
            ):
                program = self._parse(input_path, dsl_source)
        except Exception as e:
            if self._log is not None:
                self._log.debug("parse exception", exc_info=True)
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
        if self.config.optimize_level != "none":
            with self._phase("compiler.optimize", "Running optimization passes"):
                opt_result = self._run_optimizations(program)
            opt_message = opt_result.message

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
            with self._phase(
                "compiler.codegen",
                f"Generating {self.config.backend} code",
            ):
                asm_text = self._generate_code(program)
        except Exception as e:
            if self._log is not None:
                self._log.debug("codegen exception", exc_info=True)
            return CompileResult(
                success=False, errors=[f"Codegen error: {e}"],
                ir_dump=ir_dump,
            )

        # --- 5. Post-codegen passes ---
        with self._phase("compiler.asm", "Running assembly passes"):
            asm_text = self._run_asm_passes(asm_text, warnings)

        # --- 6. Cycle estimation ---
        cycle_report = ""
        if self.config.cycle_stats:
            with self._phase("compiler.cycle", "Estimating pipeline cycles"):
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
        with self._phase("compiler.emit", f"Writing {output_path}"):
            with open(output_path, "w") as f:
                f.write(asm_text)

        if self._log is not None:
            for w in warnings:
                self._log.warning("%s", w)
            self._log.info("compilation succeeded: %s (%d bytes)",
                           output_path, len(asm_text))

        return CompileResult(
            success=True,
            output_text=asm_text,
            output_path=output_path,
            ir_dump=ir_dump,
            stats={"opt_message": opt_message, "cycle_report": cycle_report},
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
            from scratchv.frontend.dsl_extended import ExtendedDSLParser
            program = ExtendedDSLParser().parse(
                source or "", filename=input_path or "<dsl>",
            )
            if self._log is not None:
                self._log.debug("parser: extended-dsl, input=%s",
                                input_path or "<inline DSL>")
            return program
        else:
            from scratchv.frontend.onnx_parser import ONNXParser
            program = ONNXParser().parse(input_path)
            if self._log is not None:
                self._log.debug("parser: onnx, input=%s", input_path)
            return program

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
        if self._log is not None:
            self._log.debug("IR verifier: %d issue(s)", len(issues))

    # ── Internal: optimizations ─────────────────────────────────────────────

    def _run_optimizations(self, program) -> PassResult:
        """Run all configured optimization passes."""
        from scratchv.optimizer.constant_folding import ConstantFolder
        from scratchv.optimizer.dead_code import DeadCodeEliminator

        pm = PassManager("optimizer", log=self._pass_logger())
        pm.add(_PassAdapter("constant-folding", ConstantFolder(program)))
        pm.add(_PassAdapter("dead-code-elim", DeadCodeEliminator(program)))

        if self.config.optimize_level == "all":
            from scratchv.optimizer.peephole import IRPeepholeOptimizer
            from scratchv.optimizer.muladd_fusion import MulAddFusion
            from scratchv.optimizer.licm import LICM

            pm.add(_PassAdapter("ir-peephole", IRPeepholeOptimizer(program)))
            pm.add(_PassAdapter("muladd-fusion", MulAddFusion(program)))
            pm.add(_PassAdapter("licm", LICM(program)))

        return pm.run(program)

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

        if self._log is not None:
            log_step("compiler.codegen", "instruction selection")
        selector = InstructionSelector(program)
        machine_instrs = selector.run()

        # Linear-scan: skip greedy allocator, use liveness-driven allocator
        if self.config.reg_alloc == "linear":
            if self._log is not None:
                log_step("compiler.codegen",
                         "register allocation (linear-scan)")
            from scratchv.backend.regalloc_linear import (
                LinearScanAllocator, block_from_machine_instrs,
            )
            ls_insts = block_from_machine_instrs(machine_instrs)
            lsa = LinearScanAllocator()
            if self._log is not None:
                log_step("compiler.codegen", "assembly emission")
            return lsa.emit(ls_insts)

        if self._log is not None:
            log_step("compiler.codegen",
                     f"register allocation ({self.config.reg_alloc})")
        alloc = RegisterAllocator(machine_instrs, mode=self.config.reg_alloc)
        allocated = alloc.run()
        if self._log is not None:
            log_step("compiler.codegen", "assembly emission")
        emitter = AsmEmitter(allocated)
        return emitter.emit()

    def _generate_riscv_dag(self, program) -> str:
        """DAG-based instruction selection pipeline."""
        from scratchv_dag.selection_dag import DAGBuilder, DAGCombiner, DAGScheduler
        from scratchv.backend.register_alloc import RegisterAllocator
        from scratchv.backend.asm_emit import AsmEmitter

        if self._log is not None:
            log_step("compiler.codegen", "DAG build")
        builder = DAGBuilder(program)
        dag = builder.run()

        if self._log is not None:
            log_step("compiler.codegen", "DAG combine")
        combiner = DAGCombiner(dag)
        combiner.run()

        if self._log is not None:
            log_step("compiler.codegen", "DAG scheduling")
        scheduler = DAGScheduler(dag)
        machine_instrs = scheduler.run()

        if self._log is not None:
            log_step("compiler.codegen",
                     f"register allocation ({self.config.reg_alloc})")
        alloc = RegisterAllocator(machine_instrs, mode=self.config.reg_alloc)
        allocated = alloc.run()

        if self._log is not None:
            log_step("compiler.codegen", "assembly emission")
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
            if self._log is not None:
                if changes:
                    self._log.info("asm peephole: %d change(s)", changes)
                else:
                    self._log.debug("asm peephole: no changes")

        if self.config.const_merge:
            from scratchv.backend.const_merge import merge_constants_detailed
            asm_text, stats = merge_constants_detailed(asm_text)
            if stats.total_changes:
                warnings.append(
                    f"Const merge: {stats.total_changes} changes "
                    f"({stats.merged_pairs} pairs, "
                    f"{stats.redundant_lui_removed} redundant lui)"
                )
            if self._log is not None:
                self._log.info(
                    "const merge: %d change(s) (%d pairs, %d redundant lui)",
                    stats.total_changes, stats.merged_pairs,
                    stats.redundant_lui_removed,
                )

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
            if self._log is not None:
                self._log.info(
                    "instruction scheduling applied (%d instructions)",
                    len(scheduled),
                )

        if self.config.beautify_asm:
            from scratchv.backend.asm_beautifier import beautify_asm
            asm_text = beautify_asm(asm_text)
            if self._log is not None:
                self._log.debug("assembly beautified (%d bytes)",
                                len(asm_text))

        if self.config.count_instr:
            from scratchv.backend.inst_counter import count_instructions
            counts = count_instructions(asm_text)
            total = sum(v for k, v in counts.items()
                        if not k.startswith("_") and isinstance(v, int))
            warnings.append(f"Instruction count: {total}")
            if self._log is not None:
                self._log.info("instruction count: %d", total)

        return asm_text


# ═══════════════════════════════════════════════════════════════════════════════
# _PassAdapter — wraps legacy passes that don't implement CompilerPass
# ═══════════════════════════════════════════════════════════════════════════════

class _PassAdapter(CompilerPass):
    """Adapter that wraps a legacy pass object into the ``CompilerPass`` API.

    Legacy passes are expected to have a ``run()`` method that returns an
    integer (number of changes) and mutate the data in place.
    """

    def __init__(self, name: str, legacy_pass: Any):
        self._name = name
        self._legacy = legacy_pass

    @property
    def name(self) -> str:
        return self._name

    def run(self, input_data: Any) -> PassResult:
        changes = self._legacy.run()
        return PassResult(
            data=input_data,
            changes=changes,
            message=f"{changes} change(s)",
        )
