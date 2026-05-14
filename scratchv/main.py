#!/usr/bin/env python3
"""ScratchV CLI: ONNX model → RISC-V assembly / LLVM IR compiler.

Usage:
    scratchv model.onnx -o output.s              # RISC-V assembly
    scratchv model.onnx --backend llvm -o out.ll # LLVM IR
    scratchv model.onnx --verify                 # verify against ONNX Runtime
    scratchv --dsl source.dsl -o output.s
"""

from __future__ import annotations

import argparse
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ScratchV: ONNX model to RISC-V assembly / LLVM IR compiler",
    )
    parser.add_argument("input", nargs="?", help="Input file (.onnx or .dsl)")
    parser.add_argument("-o", "--output", default=None, help="Output file")
    parser.add_argument("--dsl", help="Use DSL parser instead of ONNX (or pass .dsl file as input)")
    parser.add_argument("--backend", choices=["riscv", "llvm"], default="riscv",
                        help="Target backend (default: riscv)")
    parser.add_argument("--dump-ir", action="store_true", help="Dump IR before codegen")
    parser.add_argument("--optimize", choices=["none", "basic", "all"], default="none",
                        help="Optimization level: basic (fold+dce), all (+peephole+fuse+licm)")
    parser.add_argument("--reg-alloc", choices=["naive", "greedy"], default="greedy",
                        help="Register allocation strategy (default: greedy)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify output against ONNX Runtime reference")
    parser.add_argument("--rtol", type=float, default=1e-5,
                        help="Relative tolerance for verification")
    parser.add_argument("--atol", type=float, default=1e-8,
                        help="Absolute tolerance for verification")
    parser.add_argument("--version", action="version", version="ScratchV 0.1.0")
    return parser


def parse_input(args) -> object:
    """Parse input file (ONNX or DSL) into an IR Program."""
    input_path = args.input
    use_dsl = args.dsl is not None or (input_path and input_path.endswith(".dsl"))

    if use_dsl:
        from scratchv.frontend.dsl_parser import DSLParser
        with open(input_path or args.dsl) as f:
            source = f.read()
        dsl_parser = DSLParser()
        return dsl_parser.parse(source)
    else:
        from scratchv.frontend.onnx_parser import ONNXParser
        onnx_parser = ONNXParser()
        return onnx_parser.parse(input_path)


def run_optimizer(program, level: str, dump_ir: bool):
    """Run optimizations on the IR program. Returns stats string."""
    from scratchv.optimizer.constant_folding import ConstantFolder
    from scratchv.optimizer.dead_code import DeadCodeEliminator

    folder = ConstantFolder(program)
    folded = folder.run()
    elim = DeadCodeEliminator(program)
    eliminated = elim.run()

    stats_str = f"{folded} folded, {eliminated} eliminated"

    if level == "all":
        from scratchv.optimizer.peephole import PeepholeOptimizer
        from scratchv.optimizer.muladd_fusion import MulAddFusion
        from scratchv.optimizer.licm import LICM

        peep = PeepholeOptimizer(program)
        peeped = peep.run()
        fuse = MulAddFusion(program)
        fused = fuse.run()
        licm = LICM(program)
        hoisted = licm.run()
        stats_str += f", {peeped} peep-hole, {fused} fused, {hoisted} hoisted"

    if dump_ir:
        from scratchv.ir.printer import IRPrinter
        print(f"; --- After optimization: {stats_str} ---", file=sys.stderr)
        printer = IRPrinter(program)
        print(printer.dump(), file=sys.stderr)

    return stats_str


def generate_riscv_backend(program, reg_alloc: str) -> str:
    """Generate RISC-V assembly from IR program."""
    from scratchv.backend.instruction_select import InstructionSelector
    from scratchv.backend.register_alloc import RegisterAllocator
    from scratchv.backend.asm_emit import AsmEmitter

    selector = InstructionSelector(program)
    machine_instrs = selector.run()

    alloc = RegisterAllocator(machine_instrs, mode=reg_alloc)
    allocated = alloc.run()

    emitter = AsmEmitter(allocated)
    return emitter.emit()


def generate_llvm_backend(program) -> str:
    """Generate LLVM IR from ScratchV IR program."""
    from scratchv.backend.llvm_codegen import LLVMCodegen

    codegen = LLVMCodegen(program)
    return codegen.emit()


def run_verification(args, program) -> None:
    """Run verification if requested."""
    from scratchv.verification.verifier import verify_dsl

    input_path = args.input
    use_dsl = args.dsl is not None or (input_path and input_path.endswith(".dsl"))

    if use_dsl:
        with open(input_path or args.dsl) as f:
            source = f.read()

        # Generate some random test inputs
        import numpy as np
        # Extract variable names from DSL (simple heuristic)
        import re
        input_vars = set()
        for m in re.finditer(r'\b(add|sub|mul|div|relu|gelu|exp|neg|matmul|dot|maxpool|softmax)\(([^)]+)', source):
            args_text = m.group(2)
            for arg in args_text.split(","):
                arg = arg.strip().split(":")[0].strip()
                if arg and not arg[0].isdigit():
                    input_vars.add(arg)
        # Remove return/loop variable names
        input_vars = {v for v in input_vars if v.lower() not in (
            "add", "sub", "mul", "div", "relu", "gelu", "exp", "neg",
            "matmul", "dot", "maxpool", "softmax", "return", "for", "endfor"
        )}

        feed_dict = {v: np.random.randn(4).astype(np.float32) for v in input_vars}
        result = verify_dsl(source, feed_dict, rtol=args.rtol, atol=args.atol)
        status = "✓ PASS" if result["success"] else "✗ FAIL"
        print(f"  Verification: {status}  (max error: {result['max_error']:.6e})", file=sys.stderr)
    else:
        # ONNX model verification
        from scratchv.verification.verifier import verify_onnx_model

        def compiler_fn(inputs):
            """Run the full compiler pipeline on given inputs."""
            # Re-parse with concrete inputs
            from scratchv.frontend.onnx_parser import ONNXParser
            parser = ONNXParser()
            prog = parser.parse(args.input)

            if args.optimize != "none":
                run_optimizer(prog, args.optimize, False)

            # Compile and return a placeholder
            # Full JIT execution needs runtime linking — see docs/verification.md
            return {}

        result = verify_onnx_model(
            args.input,
            compiler_output_fn=compiler_fn,
            rtol=args.rtol,
            atol=args.atol,
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.input is None and args.dsl is None:
        parser.print_help()
        return 1

    # --- Resolve output path ---
    if args.output is None:
        if args.backend == "llvm":
            args.output = "output.ll"
        else:
            args.output = "output.s"

    # --- Parse input ---
    try:
        program = parse_input(args)
    except Exception as e:
        print(f"Error parsing input: {e}", file=sys.stderr)
        return 1

    # --- Dump IR if requested (before optimization) ---
    if args.dump_ir:
        from scratchv.ir.printer import IRPrinter
        printer = IRPrinter(program)
        print("; --- IR Dump (before optimization) ---", file=sys.stderr)
        print(printer.dump(), file=sys.stderr)

    # --- Optimize ---
    if args.optimize != "none":
        run_optimizer(program, args.optimize, args.dump_ir)

    # --- Code generation ---
    try:
        if args.backend == "llvm":
            asm_text = generate_llvm_backend(program)
        else:
            asm_text = generate_riscv_backend(program, args.reg_alloc)
    except Exception as e:
        print(f"Error during code generation: {e}", file=sys.stderr)
        return 1

    with open(args.output, "w") as f:
        f.write(asm_text)

    print(f"✓ {args.backend.upper()} output written to {args.output}", file=sys.stderr)

    # --- Verify ---
    if args.verify:
        run_verification(args, program)

    return 0


if __name__ == "__main__":
    sys.exit(main())
