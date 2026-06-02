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

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from scratchv.pass_interface import CompilerPass, PassResult


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

    def __init__(self, name: str = "pipeline"):
        self._name = name
        self._passes: list[CompilerPass] = []

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

        for p in self._passes:
            t0 = time.perf_counter()
            try:
                result = p.run(data)
            except Exception as exc:
                return PassResult(
                    data=None,
                    changes=total_changes,
                    message=f"Pass '{p.name}' failed: {exc}",
                    warnings=all_warnings,
                )
            elapsed = time.perf_counter() - t0
            timings[p.name] = elapsed

            if result.data is None:
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
        errors: list[str] = []
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
        if self.config.optimize_level != "none":
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
            asm_text = self._generate_code(program)
        except Exception as e:
            return CompileResult(
                success=False, errors=[f"Codegen error: {e}"],
                ir_dump=ir_dump,
            )

        # --- 5. Post-codegen passes ---
        asm_text = self._run_asm_passes(asm_text, warnings)

        # --- 6. Write output ---
        with open(output_path, "w") as f:
            f.write(asm_text)

        return CompileResult(
            success=True,
            output_text=asm_text,
            output_path=output_path,
            ir_dump=ir_dump,
            stats={"opt_message": opt_message},
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

    def _run_optimizations(self, program) -> PassResult:
        """Run all configured optimization passes."""
        from scratchv.optimizer.constant_folding import ConstantFolder
        from scratchv.optimizer.dead_code import DeadCodeEliminator

        pm = PassManager("optimizer")
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

        selector = InstructionSelector(program)
        machine_instrs = selector.run()

        alloc = RegisterAllocator(machine_instrs, mode=self.config.reg_alloc)
        allocated = alloc.run()

        # Optional: use linear-scan instead
        if self.config.reg_alloc == "linear":
            from scratchv.backend.regalloc_linear import (
                LinearScanAllocator, block_from_machine_instrs,
            )
            ls_insts = block_from_machine_instrs(allocated)
            lsa = LinearScanAllocator()
            intervals = lsa.compute_live_intervals(ls_insts)
            lsa.allocate(intervals)
            # Use linear-scan allocated code as assembly directly
            return lsa.get_allocated_code(ls_insts)

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
