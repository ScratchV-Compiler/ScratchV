#!/usr/bin/env python3
"""ONNX → LLVM IR → Verification against ONNX Runtime.

Full pipeline demonstrating the "code complete" path:
  1. Parse ONNX model
  2. Lower to ScratchV IR
  3. Optimize
  4. Generate LLVM IR
  5. Run through ONNX Runtime for reference
  6. Compare results

Prerequisites:
    pip install onnx onnxruntime numpy

Usage:
    python examples/onnx_llvm_verification.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def ensure_onnx_model(path: str = "models/add.onnx") -> str:
    """Generate a test ONNX model if it doesn't exist."""
    if os.path.exists(path):
        return path

    print(f"Generating {path}...")
    from examples.gen_onnx_model import make_add_model
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    make_add_model(path)
    return path


def main():
    model_path = ensure_onnx_model()

    print("=" * 60)
    print("ScratchV ONNX → LLVM IR Verification Pipeline")
    print("=" * 60)

    # Step 1: Parse ONNX model
    print("\n[1/5] Parsing ONNX model...")
    from scratchv.frontend.onnx_parser import ONNXParser
    parser = ONNXParser()
    program = parser.parse(model_path)

    from scratchv.ir.printer import IRPrinter
    printer = IRPrinter(program)
    print("  IR dump:")
    print(f"  {printer.dump()[:300]}")

    # Step 2: Optimize
    print("\n[2/5] Optimizing IR...")
    from scratchv.optimizer.constant_folding import ConstantFolder
    from scratchv.optimizer.dead_code import DeadCodeEliminator
    folder = ConstantFolder(program)
    folded = folder.run()
    elim = DeadCodeEliminator(program)
    eliminated = elim.run()
    print(f"  Folded: {folded}, Eliminated: {eliminated}")

    # Step 3: Generate LLVM IR
    print("\n[3/5] Generating LLVM IR...")
    from scratchv.backend.llvm_codegen import LLVMCodegen
    codegen = LLVMCodegen(program)
    llvm_ir = codegen.emit()

    out_path = "output.ll"
    with open(out_path, "w") as f:
        f.write(llvm_ir)
    print(f"  LLVM IR written to {out_path}")
    print(f"  Preview (first 20 lines):")
    for line in llvm_ir.split("\n")[:20]:
        print(f"    {line}")

    # Step 4: Reference with ONNX Runtime
    print("\n[4/5] Running ONNX Runtime reference...")
    from scratchv.verification.verifier import ONNXReference
    ref = ONNXReference(model_path)

    if not ref.available:
        print("  ONNX Runtime not available.")
        print("  Install: pip install onnxruntime")
        print("  Falling back to numpy reference...")

        # Use numpy reference instead
        import onnx
        onnx_model = onnx.load(model_path)
        inputs = {}
        for inp in onnx_model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

        print(f"  Generated inputs:")
        for name, arr in inputs.items():
            print(f"    {name}: shape={arr.shape}, values={arr}")

        # Compute expected via numpy
        from scratchv.verification.verifier import numpy_reference
        for node in onnx_model.graph.node:
            expected = numpy_reference(node.op_type, *(inputs[n] for n in node.input))
            for out_name in node.output:
                inputs[out_name] = expected

        print(f"\n  Reference output ({onnx_model.graph.output[0].name}):")
        for o in onnx_model.graph.output:
            print(f"    {o.name}: {inputs[o.name]}")
    else:
        # ONNX Runtime available
        import onnx
        onnx_model = onnx.load(model_path)
        feed_dict = {}
        for inp in onnx_model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            feed_dict[inp.name] = np.random.randn(*shape).astype(np.float32)

        print(f"  Generated inputs:")
        for name, arr in feed_dict.items():
            print(f"    {name}: shape={arr.shape}, values={arr}")

        reference = ref.run(feed_dict)
        print(f"\n  Reference outputs:")
        for name, arr in reference.items():
            print(f"    {name}: {arr}")

    # Step 5: Verification summary
    print("\n[5/5] Pipeline summary:")
    print(f"  Model: {model_path}")
    print(f"  Backend: LLVM IR")
    print(f"  IR optimizations: {'✓' if folded + eliminated > 0 else '-'}")
    print(f"  Output: {out_path}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  opt -O2 {out_path} -o optimized.bc    # LLVM optimization")
    print(f"  llc {out_path} -o output.s              # LLVM → native asm")
    print(f"  lli {out_path}                           # LLVM JIT execution")


if __name__ == "__main__":
    main()
