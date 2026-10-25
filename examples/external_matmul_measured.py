"""Inspect historical measured MatMul settings, or generate a tiny fallback graph.

Run from the repository root: python -m examples.external_matmul_measured.
This example never builds or executes a model. Historical timings are explicit
caller evidence, not a portable performance guarantee or an automatic tuner.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

# Direct execution from another cwd must import this checkout, not an installed copy.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scratchv.backend.external_matmul import MatMulMeasurement, MatMulPerformance
from scratchv.backend.nmsis import nmsis_matmul
from scratchv.backend.openblas import openblas_matmul
from scratchv.backend.tensor_c_codegen import TensorCCodegen
from scratchv.ir.builder import IRBuilder
from scratchv.ir.types import Value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("host", "riscv64"), default="host")
    parser.add_argument("--generate", type=Path, metavar="MODEL_C",
                        help="generate only a tiny, unmeasured graph; no compilation or execution")
    parser.add_argument("--use-historical-profile", action="store_true",
                        help="explicitly use historical timings during generation; not fresh measurements")
    parser.add_argument("--openblas-include", default=os.environ.get("SCRATCHV_OPENBLAS_INCLUDE_DIR"),
                        help="OpenBLAS headers; default SCRATCHV_OPENBLAS_INCLUDE_DIR or system search")
    parser.add_argument("--openblas-lib", default=os.environ.get("SCRATCHV_OPENBLAS_LIBRARY_DIR"),
                        help="OpenBLAS library directory; default SCRATCHV_OPENBLAS_LIBRARY_DIR or system search")
    parser.add_argument("--nmsis-root", default=os.environ.get("SCRATCHV_NMSIS_ROOT"),
                        help="NMSIS checkout; default SCRATCHV_NMSIS_ROOT, required for RISC-V generation")
    args = parser.parse_args(argv)
    if args.use_historical_profile and args.generate is None:
        parser.error("--use-historical-profile requires --generate")
    if args.generate is not None and args.generate.exists():
        parser.error(f"Refusing to overwrite existing output: {args.generate}")

    path = Path(__file__).with_name("external_matmul_profiles") / f"{args.target}-historical.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    settings = saved["profile"]
    # Default display requires no native libraries, NMSIS checkout or compiler.
    report = {
        "action": "display_only",
        "target": args.target,
        "historical_measurements": len(settings["measurements"]),
        "historical_environment": settings["environment"],
        "historical_selection_counts": saved["historical_full_model"]["selection_counts"],
        "generation_options": {key: settings[key] for key in (
            "matmul_policy", "matmul_tile", "matmul_rhs_layout", "kernel_calls",
            "qemu_workspace_layout")},
        "historical_harts": settings["harts"],
        "current_environment_validated": False,
        "notice": saved["use"],
    }
    if args.generate is not None:
        try:
            if args.target == "host":
                provider = openblas_matmul(
                    include_dirs=(args.openblas_include,) if args.openblas_include else (),
                    library_dirs=(args.openblas_lib,) if args.openblas_lib else ())
            else:
                if not args.nmsis_root:
                    parser.error("RISC-V generation needs --nmsis-root or SCRATCHV_NMSIS_ROOT")
                provider = nmsis_matmul(args.nmsis_root, parallel_rows=True)
            # Dependency paths belong to this checkout/machine, never the JSON.
            for key, expected in saved["provider_contract"].items():
                if key == "targets":
                    expected = tuple(expected)
                if getattr(provider, key) != expected:
                    raise ValueError(f"Provider contract differs from the historical example: {key}")
            profile = None
            if args.use_historical_profile:
                # For real work, replace these records with same-environment
                # measurements. The existing dataclasses validate their values.
                values = dict(settings)
                values["measurements"] = tuple(
                    MatMulMeasurement(**row) for row in values["measurements"])
                profile = MatMulPerformance(provider=provider, **values)

            x, first, last = (Value("x", shape=(3, 5)), Value("first", shape=(5, 7)),
                              Value("last", shape=(7, 11)))
            graph = IRBuilder()
            graph.new_function("main", [x])
            graph.new_block("entry")
            graph.program.global_values.extend((first, last))
            graph.ret(graph.matmul(graph.relu(graph.matmul(x, first)), last))
            weights = {value.name: np.arange(np.prod(value.shape), dtype=np.float32).reshape(value.shape) / 32
                       for value in (first, last)}
            artifact = TensorCCodegen(
                graph.program, weights, constant_storage="external",
                matmul_implementation="measured", external_matmul=provider,
                matmul_target=args.target, external_matmul_numerics="allow_provider_fp32",
                matmul_performance=profile, **report["generation_options"]).generate()
            # Neither tiny shape exists in the historical measurements. The
            # code generator must preserve the original optimized implementation.
            assert all(row.implementation == "builtin" for row in artifact.matmul_selections)
            assert artifact.external_matmul is None
            args.generate.parent.mkdir(parents=True, exist_ok=True)
            with args.generate.open("x", encoding="utf-8") as handle:
                handle.write(artifact.source)
            report.update(
                action="generated_only", source=str(args.generate),
                historical_profile_used=profile is not None,
                selections=[asdict(row) for row in artifact.matmul_selections],
                panel_count=len(artifact.matmul_rhs_panels),
                external_library_selected=False,
                artifact_function=artifact.function_name,
                note="C generation and selection check only; weights remain in the in-memory artifact. No compilation, execution or performance validation.")
        except (ValueError, OSError) as error:
            parser.error(str(error))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
