#!/usr/bin/env python3
"""ScratchV CLI: ONNX model → RISC-V assembly / LLVM IR compiler.

Usage:
    scratchv model.onnx -o output.s              # RISC-V assembly
    scratchv model.onnx --backend llvm -o out.ll # LLVM IR
    scratchv model.onnx --verify                 # verify against ONNX Runtime
    scratchv --dsl source.dsl -o output.s
    scratchv model.onnx --optimize all --beautify --peephole-asm --count-instr
"""

from __future__ import annotations

import argparse
import sys

from scratchv.compiler import CompilerConfig, CompilerDriver, CompileResult


# ═══════════════════════════════════════════════════════════════════════════════
# CLI argument parser
# ═══════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ScratchV: ONNX model -> RISC-V assembly / LLVM IR",
    )
    # ── Input / output ──────────────────────────────────────────────────
    parser.add_argument("input", nargs="?", help="Input file (.onnx or .dsl)")
    parser.add_argument("-o", "--output", default=None, help="Output file")
    parser.add_argument("--dsl", help="Use DSL parser instead of ONNX")

    # ── Backend ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--backend", choices=["riscv", "llvm"], default="riscv",
        help="Target backend (default: riscv)",
    )

    # ── Optimizations ───────────────────────────────────────────────────
    parser.add_argument(
        "--optimize", choices=["none", "basic", "all"],
        default="none",
        help="Optimization level (none, basic, all)",
    )

    # ── Register allocation ─────────────────────────────────────────────
    parser.add_argument(
        "--reg-alloc", choices=["naive", "greedy", "linear"],
        default="greedy",
        help="Register allocation strategy (default: greedy)",
    )

    # ── Debug ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--dump-ir", action="store_true",
        help="Dump IR before and after optimization",
    )

    # ── Verification ────────────────────────────────────────────────────
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify output against ONNX Runtime / numpy reference",
    )
    parser.add_argument("--rtol", type=float, default=1e-5,
                        help="Relative tolerance for verification")
    parser.add_argument("--atol", type=float, default=1e-8,
                        help="Absolute tolerance for verification")

    # ── Topic module flags ──────────────────────────────────────────────
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Enable structured logging at given level",
    )
    parser.add_argument(
        "--verify-ir", action="store_true",
        help="Run IR verifier before and after optimization (Topic 21)",
    )
    parser.add_argument(
        "--beautify", action="store_true",
        help="Run assembly beautifier on output (Topic 5)",
    )
    parser.add_argument(
        "--peephole-asm", action="store_true",
        help="Run assembly-level peephole optimizer (Topic 13)",
    )
    parser.add_argument(
        "--const-merge", action="store_true",
        help="Run constant-load merge pass (Topic 14)",
    )
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run instruction scheduler (Topic 18)",
    )
    parser.add_argument(
        "--count-instr", action="store_true",
        help="Print instruction count statistics (Topic 12)",
    )
    parser.add_argument(
        "--dag-isel", action="store_true",
        help="Use DAG-based instruction selection (scratchv_dag)",
    )
    parser.add_argument(
        "--extended-isel", action="store_true",
        help="Use extended instruction selector with fp64/sqrt/min/max/abs support (Topic 28)",
    )

    # ── Meta ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--version", action="version",
        version="ScratchV 0.3.0",
    )
    return parser


# ═══════════════════════════════════════════════════════════════════════════════
# Config builder
# ═══════════════════════════════════════════════════════════════════════════════

def args_to_config(args: argparse.Namespace) -> CompilerConfig:
    """Translate parsed CLI arguments to a CompilerConfig."""
    return CompilerConfig(
        backend=args.backend,
        optimize_level=args.optimize,
        reg_alloc=args.reg_alloc,
        dump_ir=args.dump_ir,
        verify=args.verify,
        rtol=args.rtol,
        atol=args.atol,
        use_logger=args.log_level is not None,
        log_level=args.log_level or "INFO",
        use_dag_isel=args.dag_isel,
        beautify_asm=args.beautify,
        peephole_asm=args.peephole_asm,
        const_merge=args.const_merge,
        schedule=args.schedule,
        count_instr=args.count_instr,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════════

def run_verification(args: argparse.Namespace, program) -> None:
    """Run verification if requested."""
    from scratchv.verification.verifier import verify_dsl

    input_path = args.input
    use_dsl = args.dsl is not None or (
        input_path and input_path.endswith(".dsl"))

    if use_dsl:
        with open(input_path or args.dsl) as f:
            source = f.read()

        import numpy as np
        import re
        input_vars: set[str] = set()
        op_pat = (
            r'\b(add|sub|mul|div|relu|gelu|exp|neg|'
            r'matmul|dot|maxpool|softmax)\(([^)]+)'
        )
        for m in re.finditer(op_pat, source):
            args_text = m.group(2)
            for arg in args_text.split(","):
                arg = arg.strip().split(":")[0].strip()
                if arg and not arg[0].isdigit():
                    input_vars.add(arg)
        skip = (
            "add", "sub", "mul", "div", "relu", "gelu", "exp", "neg",
            "matmul", "dot", "maxpool", "softmax",
            "return", "for", "endfor",
        )
        input_vars = {v for v in input_vars if v.lower() not in skip}

        feed_dict = {
            v: np.random.randn(4).astype(np.float32)
            for v in input_vars
        }
        result = verify_dsl(
            source, feed_dict,
            rtol=args.rtol, atol=args.atol)
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        err = result['max_error']
        msg = f"  Verification: {status}  (max error: {err:.6e})"
        print(msg, file=sys.stderr)
    else:
        from scratchv.verification.verifier import verify_onnx_model

        def compiler_fn(inputs):
            from scratchv.frontend.onnx_parser import ONNXParser
            parser = ONNXParser()
            return parser.parse(args.input)

        verify_onnx_model(
            args.input,
            compiler_output_fn=compiler_fn,
            rtol=args.rtol,
            atol=args.atol,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.input is None and args.dsl is None:
        parser.print_help()
        return 1

    # Build config and driver
    config = args_to_config(args)
    driver = CompilerDriver(config)

    # Compile
    result: CompileResult = driver.compile(
        input_path=args.input or "",
        output_path=args.output,
        dsl_source=args.dsl if hasattr(args, 'dsl') else None,
    )

    # Report
    if result.ir_dump:
        print(result.ir_dump, file=sys.stderr)

    if result.success:
        print(f"OK {args.backend.upper()} output written to {result.output_path}",
              file=sys.stderr)
        for w in result.warnings:
            print(f"  note: {w}", file=sys.stderr)
        if result.stats.get("opt_message"):
            print(f"  optimizer: {result.stats['opt_message']}", file=sys.stderr)

        # Verification
        if args.verify:
            run_verification(args, None)

        return 0
    else:
        for err in result.errors:
            print(f"Error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
