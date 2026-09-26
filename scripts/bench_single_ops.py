#!/usr/bin/env python3
"""Benchmark single-operator ONNX models — collect static/dynamic instruction data.

For each single-op model in models/single_op/:
  1. Compile with ScratchV → static instruction count
  2. Map to per-layer dynamic instruction estimates (ScratchV + LLVM)
  3. Aggregate by operator type

Output: benchmark_reports/single_op_bench.json
"""

from __future__ import annotations
import argparse
import json, subprocess, sys, tempfile, time
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

from benchmarks.test_regalloc.bench_cnn import _compile_onnx
from benchmarks.test_regalloc.semantics_compare import (
    DEFAULT_PRESSURE_REGS,
    compare_semantics,
    machine_instruction_summary,
    operand_semantics_differences,
)
from scratchv.backend.machine_types import ALL_REGS

PROJ = Path(__file__).resolve().parent.parent
SINGLE_OP_DIR = PROJ / "models" / "single_op"
OUTPUT_FILE = PROJ / "benchmark_reports" / "single_op_bench.json"
OUTPUT_MD = PROJ / "benchmark_reports" / "single_op_bench.md"


# ── Mapping: single-op model name → layer name in estimation functions ──
# These correspond to the per-layer keys in estimate_cnn_model() output
# and the compute_layer_dims() from llvm_cache_compare.py
# (op_type, sv_layer_name, llvm_layer_name)
MODEL_TO_LAYER = {
    "cnn_layer1.0_Conv":      ("conv",     "Conv1",    "Conv1 (3→32, 3×3)"),
    "cnn_layer1.1_Relu":      ("relu",     "ReLU1",    "ReLU1"),
    "cnn_layer1.2_MaxPool":   ("maxpool",  "MaxPool1", "MaxPool1 (2×2)"),
    "cnn_layer2.0_Conv":      ("conv",     "Conv2",    "Conv2 (32→32, 3×3)"),
    "cnn_layer2.1_Relu":      ("relu",     "ReLU2",    "ReLU2"),
    "cnn_layer2.2_MaxPool":   ("maxpool",  "MaxPool2", "MaxPool2 (2×2)"),
    "cnn_layer3.0_Conv":      ("conv",     "Conv3",    "Conv3 (32→64, 3×3)"),
    "cnn_layer3.1_Relu":      ("relu",     "ReLU3",    "ReLU3"),
    "cnn_layer3.2_MaxPool":   ("maxpool",  "MaxPool3", "MaxPool3 (2×2)"),
    "cnn_fc1_Gemm":           ("gemm",     "FC1",      "FC1 (53824→128)"),
    "cnn_relu1_Relu":         ("relu",     "ReLU (after FC1)", "ReLU4"),
    "cnn_fc2_Gemm":           ("gemm",     "FC2",      "FC2 (128→1)"),
    "cnn_sigmoid1_Sigmoid":   ("sigmoid",  "Sigmoid",  "Sigmoid"),
}

# ── Per-operator per-MAC/EL instruction category breakdowns ──

CATEGORIES = ["alu_r", "alu_i", "fp", "shift", "load", "store", "branch", "jump", "upper"]

# ScratchV Q16.16 RV32IM per-MAC instruction distribution
# Based on analysis of generated assembly inner loops (v0.3.0, K=3 unrolled)
SV_CONV_PER_MAC = {
    "alu_r": 2.0,    # mul + add (Q16.16 MAC)
    "alu_i": 2.5,    # addi (pointer increments, loop counters)
    "fp": 0.0,
    "shift": 1.0,    # srai (Q16.16 fixup)
    "load": 2.0,     # lw (weight + input)
    "store": 0.0,    # accumulator in register
    "branch": 0.3,   # bne (loop control, amortized)
    "jump": 0.1,
    "upper": 0.1,
}

SV_FC_PER_MAC = {
    "alu_r": 2.0,
    "alu_i": 3.0,
    "fp": 0.0,
    "shift": 1.0,
    "load": 2.0,
    "store": 0.0,
    "branch": 0.5,
    "jump": 0.2,
    "upper": 0.3,
}

SV_MAXPOOL_PER_EL = {
    "alu_r": 1.0,    # max comparison
    "alu_i": 4.0,    # address calc + loop
    "load": 3.0,     # lw for pool window
    "store": 1.0,
    "branch": 1.5,
    "jump": 0.5,
    "upper": 0.5,
    "fp": 0.0,
    "shift": 0.0,
}

SV_RELU_PER_EL = {
    "alu_i": 2.0,     # comparison + branch address calc
    "load": 2.0,      # lw input
    "store": 1.0,     # sw output
    "branch": 1.0,
    "jump": 0.5,
    "upper": 0.5,
    "alu_r": 0.5,
    "fp": 0.0,
    "shift": 0.0,
}

SV_SIGMOID_PER_EL = {
    "alu_r": 3.0,     # integer sigmoid approximation
    "alu_i": 5.0,
    "load": 2.0,
    "store": 1.0,
    "branch": 3.0,
    "jump": 1.0,
    "upper": 1.0,
    "fp": 0.0,
    "shift": 2.0,
}

# LLVM RV64FD float32 per-MAC instruction distribution (from llvm_cache_compare.py)
LLVM_CONV_PER_MAC = {
    "alu_i": 2.0, "alu_r": 0.5, "load": 2.0, "store": 0.0,
    "fp": 2.0, "branch": 0.3, "jump": 0.1, "upper": 0.1, "shift": 0.0,
}
LLVM_FC_PER_MAC = {
    "alu_i": 1.5, "alu_r": 0.5, "load": 2.0, "store": 0.0,
    "fp": 2.0, "branch": 0.3, "upper": 0.2, "jump": 0.0, "shift": 0.0,
}
LLVM_MAXPOOL_PER_EL = {
    "alu_i": 4.0, "alu_r": 1.0, "load": 3.0, "store": 0.5,
    "fp": 1.0, "branch": 1.5, "jump": 0.5, "upper": 0.5, "shift": 0.0,
}
LLVM_RELU_PER_EL = {
    "alu_i": 1.0, "load": 1.0, "store": 1.0, "fp": 0.5,
    "branch": 0.5, "alu_r": 0.0, "jump": 0.0, "upper": 0.0, "shift": 0.0,
}
LLVM_SIGMOID_PER_EL = {
    "alu_i": 5.0, "alu_r": 3.0, "load": 3.0, "store": 2.0,
    "fp": 8.0, "branch": 2.0, "jump": 1.0, "upper": 1.0, "shift": 0.0,
}


def _compute_per_category(layer_counts: dict, per_unit: dict, multiplier: int) -> dict:
    """Compute per-category instruction counts.

    Args:
        layer_counts: {'macs': N, 'elements': M} from compute_layer_dims
        per_unit: per-MAC or per-element category distribution dict
        multiplier: macs for conv/gemm, elements for relu/maxpool/sigmoid
    Returns:
        dict of category -> int
    """
    result = {}
    for cat in CATEGORIES:
        result[cat] = int(per_unit.get(cat, 0) * multiplier)
    return result


def _compile_scratchv(model_path: str) -> dict:
    """Compile a single-op model with ScratchV, return static instruction count."""
    t0 = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="scratchv-single-op-") as temp_dir:
            bin_out = str(Path(temp_dir) / f"{Path(model_path).stem}.bin")
            cmd = [
                sys.executable,
                str(PROJ / "scratchv" / "standalone" / "onnx_to_riscv_standalone.py"),
                model_path, "-o", bin_out,
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                cwd=str(PROJ),
            )
            elapsed = time.perf_counter() - t0
            output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "static_insns": 0, "code_size_bytes": 0, "elapsed_s": 30}
    except Exception as e:
        return {
            "status": "error",
            "static_insns": 0,
            "code_size_bytes": 0,
            "elapsed_s": 0,
            "error": str(e),
        }

    # Parse static instruction count from output
    static_insns = 0
    code_size = 0
    for line in output.splitlines():
        if "instructions" in line and "Code size:" in line:
            try:
                parts = line.split("(")[1].split()[0]
                static_insns = int(parts.replace(",", ""))
            except (ValueError, IndexError):
                pass
        if "Code size:" in line and "bytes" in line:
            try:
                parts = line.split("Code size:")[1].split()
                code_size = int(parts[0].replace(",", ""))
            except (ValueError, IndexError):
                pass

    if proc.returncode != 0:
        return {"status": "failed", "static_insns": static_insns,
                "code_size_bytes": code_size, "elapsed_s": elapsed,
                "stderr": proc.stderr[:200]}

    return {"status": "ok", "static_insns": static_insns,
            "code_size_bytes": code_size, "elapsed_s": elapsed}


def _get_layer_data() -> tuple[dict, dict, dict, dict]:
    """Get per-layer dynamic instruction estimates for ScratchV and LLVM.

    Returns:
        (sv_per_layer, llvm_per_layer, sv_per_layer_cats, llvm_per_layer_cats)
        where _cats variants are dicts of layer_name -> {category: count}
    """
    from scratchv.standalone.llvm_cache_compare import compute_layer_dims

    layers = compute_layer_dims()

    sv_per_layer = {}
    llvm_per_layer = {}
    sv_per_layer_cats = {}
    llvm_per_layer_cats = {}

    for layer in layers:
        name = layer.name
        macs = layer.macs
        elements = layer.elements

        if name.startswith("Conv"):
            sv_mult = macs
            ll_mult = macs
            sv_unit = SV_CONV_PER_MAC
            ll_unit = LLVM_CONV_PER_MAC
        elif name.startswith("FC"):
            sv_mult = macs
            ll_mult = macs
            sv_unit = SV_FC_PER_MAC
            ll_unit = LLVM_FC_PER_MAC
        elif name.startswith("MaxPool"):
            sv_mult = elements
            ll_mult = elements
            sv_unit = SV_MAXPOOL_PER_EL
            ll_unit = LLVM_MAXPOOL_PER_EL
        elif name.startswith("ReLU"):
            sv_mult = elements
            ll_mult = elements
            sv_unit = SV_RELU_PER_EL
            ll_unit = LLVM_RELU_PER_EL
        elif name.startswith("Sigmoid"):
            sv_mult = elements
            ll_mult = elements
            sv_unit = SV_SIGMOID_PER_EL
            ll_unit = LLVM_SIGMOID_PER_EL
        elif name.startswith("Reshape"):
            sv_mult = elements
            ll_mult = elements
            sv_unit = SV_RELU_PER_EL   # ReLU-like overhead
            ll_unit = LLVM_RELU_PER_EL
        else:
            continue

        # Per-category breakdown
        sv_cats = {}
        ll_cats = {}
        sv_total = 0
        ll_total = 0
        for cat in CATEGORIES:
            sv_c = int(sv_unit.get(cat, 0) * sv_mult)
            ll_c = int(ll_unit.get(cat, 0) * ll_mult)
            sv_cats[cat] = sv_c
            ll_cats[cat] = ll_c
            sv_total += sv_c
            ll_total += ll_c

        sv_per_layer[name] = sv_total
        llvm_per_layer[name] = ll_total
        sv_per_layer_cats[name] = sv_cats
        llvm_per_layer_cats[name] = ll_cats

    return sv_per_layer, llvm_per_layer, sv_per_layer_cats, llvm_per_layer_cats


def _parse_pressure_regs(value: str) -> tuple[int, ...]:
    """Parse and validate a comma-separated physical-register sweep."""

    try:
        counts = tuple(dict.fromkeys(int(item) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pressure registers must be integers") from exc
    if not counts or any(count < 2 or count > len(ALL_REGS) for count in counts):
        raise argparse.ArgumentTypeError(
            f"pressure registers must be between 2 and {len(ALL_REGS)}"
        )
    return counts


def _collect_regalloc_detail(
    model_path: str, pressure_regs: tuple[int, ...]
) -> dict[str, object]:
    """Collect exact Machine IR and spill metrics for one isolated operator."""

    try:
        machine, ir_instructions, virtual_registers = _compile_onnx(model_path)
        comparisons = {
            str(count): compare_semantics(machine, list(ALL_REGS[:count]))
            for count in pressure_regs
        }
        return {
            "status": "ok",
            "ir_instructions": ir_instructions,
            "virtual_registers": virtual_registers,
            **machine_instruction_summary(machine),
            "semantics_differences": operand_semantics_differences(machine),
            "pressure_sweep": comparisons,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "ir_instructions": 0,
            "virtual_registers": 0,
            "machine_instructions": 0,
            "opcode_counts": {},
            "pseudo_counts": {},
            "semantics_differences": [],
            "pressure_sweep": {},
        }


def _operator_regalloc_aggregates(
    results: dict[str, dict], pressure_regs: tuple[int, ...]
) -> dict[str, dict[str, object]]:
    """Aggregate exact allocation metrics by ONNX operator type."""

    metrics = (
        "pressure_peak",
        "spill_slots",
        "spill_stores",
        "reloads",
        "static_instructions",
        "encoded_instructions",
    )
    grouped: dict[str, dict[str, object]] = {}
    for model in results.values():
        op_type = model["op_type"]
        target = grouped.setdefault(
            op_type,
            {
                "model_count": 0,
                "machine_instructions": 0,
                "semantics_differences": 0,
                "pseudo_counts": defaultdict(int),
                "pressure_sweep": {},
            },
        )
        detail = model.get("regalloc_detail", {})
        if detail.get("status") != "ok":
            continue
        target["model_count"] += 1
        target["machine_instructions"] += detail["machine_instructions"]
        target["semantics_differences"] += len(
            detail["semantics_differences"]
        )
        for opcode, count in detail["pseudo_counts"].items():
            target["pseudo_counts"][opcode] += count
        for count in pressure_regs:
            comparison = detail["pressure_sweep"][str(count)]
            row = target["pressure_sweep"].setdefault(
                str(count),
                {
                    "physical_register_count": count,
                    "before": {metric: 0 for metric in metrics},
                    "after": {metric: 0 for metric in metrics},
                    "before_valid": True,
                    "after_valid": True,
                },
            )
            for side in ("before", "after"):
                for metric in metrics:
                    value = comparison[side][metric]
                    if metric == "pressure_peak":
                        row[side][metric] = max(row[side][metric], value)
                    else:
                        row[side][metric] += value
                row[f"{side}_valid"] = (
                    row[f"{side}_valid"] and comparison[side]["valid"]
                )

    for aggregate in grouped.values():
        aggregate["pseudo_counts"] = dict(
            sorted(aggregate["pseudo_counts"].items())
        )
    return grouped


def _regalloc_markdown(
    aggregates: dict[str, dict[str, object]], pressure_regs: tuple[int, ...]
) -> str:
    """Build the operator-granularity semantics comparison report."""

    lines = [
        "# Topic 17 single-operator register-allocation comparison",
        "",
        (
            "Before uses the historical positional `dst/src` inference. "
            "After uses `machine_semantics.py`; both sides use the same "
            "current allocator and isolated ONNX operator models."
        ),
        "",
        "| Operator | Registers | Models | Machine IR | Changed semantics | Pseudos | "
        "Pressure before/after | Slots before/after | Stores before/after | "
        "Reloads before/after | Static instructions before/after | "
        "Encoded RV32 before/after | Valid |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for op_type in sorted(aggregates):
        aggregate = aggregates[op_type]
        pseudos = ", ".join(
            f"{opcode}:{count}"
            for opcode, count in aggregate["pseudo_counts"].items()
        ) or "-"
        for count in pressure_regs:
            row = aggregate["pressure_sweep"].get(str(count))
            if row is None:
                continue
            before = row["before"]
            after = row["after"]
            valid = (
                "PASS"
                if row["before_valid"] and row["after_valid"]
                else "FAIL"
            )
            lines.append(
                f"| {op_type} | {count} | {aggregate['model_count']} | "
                f"{aggregate['machine_instructions']} | "
                f"{aggregate['semantics_differences']} | {pseudos} | "
                f"{before['pressure_peak']} / {after['pressure_peak']} | "
                f"{before['spill_slots']} / {after['spill_slots']} | "
                f"{before['spill_stores']} / {after['spill_stores']} | "
                f"{before['reloads']} / {after['reloads']} | "
                f"{before['static_instructions']} / "
                f"{after['static_instructions']} | "
                f"{before['encoded_instructions']} / "
                f"{after['encoded_instructions']} | {valid} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark isolated ONNX operators and register allocation"
    )
    parser.add_argument("--output-json", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--output-md", type=Path, default=OUTPUT_MD)
    parser.add_argument(
        "--pressure-regs",
        type=_parse_pressure_regs,
        default=DEFAULT_PRESSURE_REGS,
        help="comma-separated register-bank sizes (default: 2,3,5,8,12,19)",
    )
    args = parser.parse_args()
    pressure_regs = tuple(args.pressure_regs)

    print("Collecting per-layer analytical data...")
    sv_per_layer, llvm_per_layer, sv_cats, llvm_cats = _get_layer_data()

    print(f"  Layers: {list(sv_per_layer.keys())}")
    print(f"  Categories: {CATEGORIES}")

    # Discover all single-op models
    models = {}
    for op_dir in SINGLE_OP_DIR.iterdir():
        if not op_dir.is_dir():
            continue
        op_type = op_dir.name
        for onnx_file in op_dir.glob("*.onnx"):
            model_name = onnx_file.stem  # e.g., "cnn_layer1.0_Conv"
            models[model_name] = {
                "path": str(onnx_file),
                "op_type": op_type,
            }

    print(f"\nFound {len(models)} single-op models in {SINGLE_OP_DIR}")
    if not models:
        print("No single-operator models were found", file=sys.stderr)
        return 1

    # Compile each model with ScratchV and collect data
    results = {}
    for model_name in sorted(models.keys()):
        info = models[model_name]
        path = info["path"]
        op_type = info["op_type"]

        print(f"  Compiling: {model_name} ({op_type})...", end=" ", flush=True)
        comp = _compile_scratchv(path)
        regalloc_detail = _collect_regalloc_detail(path, pressure_regs)
        print(
            f"static={comp['static_insns']} insns, {comp['status']}, "
            f"regalloc={regalloc_detail['status']}"
        )

        # Map to per-layer dynamic data (both dicts use LLVM layer names)
        layer_info = MODEL_TO_LAYER.get(model_name, (op_type, None, None))
        llvm_layer = layer_info[2] if len(layer_info) > 2 else None
        sv_layer = llvm_layer  # SV and LLVM now share the same key (LLVM layer names)

        sv_dyn = sv_per_layer.get(sv_layer, 0) if sv_layer else 0
        llvm_dyn = llvm_per_layer.get(llvm_layer, 0) if llvm_layer else 0
        sv_cat = sv_cats.get(sv_layer, {}) if sv_layer else {}
        llvm_cat = llvm_cats.get(llvm_layer, {}) if llvm_layer else {}

        # Compute percentages for each category
        sv_cat_pct = {}
        llvm_cat_pct = {}
        for cat in CATEGORIES:
            sv_cat_pct[cat] = round(sv_cat.get(cat, 0) / max(sv_dyn, 1) * 100, 1)
            llvm_cat_pct[cat] = round(llvm_cat.get(cat, 0) / max(llvm_dyn, 1) * 100, 1)

        results[model_name] = {
            "op_type": op_type,
            "network": "cnn",
            "path": path,
            "scratchv_static_insns": comp["static_insns"],
            "scratchv_code_size_bytes": comp["code_size_bytes"],
            "scratchv_dynamic_insns": int(sv_dyn),
            "llvm_dynamic_insns": int(llvm_dyn),
            "dynamic_ratio": round(sv_dyn / max(llvm_dyn, 1), 2) if llvm_dyn else 0,
            "sv_category_breakdown": sv_cat,
            "llvm_category_breakdown": llvm_cat,
            "sv_category_pct": sv_cat_pct,
            "llvm_category_pct": llvm_cat_pct,
            "compile_status": comp["status"],
            "regalloc_detail": regalloc_detail,
        }

    # ── Aggregate by operator type ──
    by_op_type = defaultdict(lambda: {
        "models": [],
        "total_sv_static": 0,
        "total_sv_dynamic": 0,
        "total_llvm_dynamic": 0,
        "sv_categories": defaultdict(int),
        "llvm_categories": defaultdict(int),
    })

    for model_name, data in results.items():
        ot = data["op_type"]
        by_op_type[ot]["models"].append(model_name)
        by_op_type[ot]["total_sv_static"] += data["scratchv_static_insns"]
        by_op_type[ot]["total_sv_dynamic"] += data["scratchv_dynamic_insns"]
        by_op_type[ot]["total_llvm_dynamic"] += data["llvm_dynamic_insns"]
        for cat in CATEGORIES:
            by_op_type[ot]["sv_categories"][cat] += data["sv_category_breakdown"].get(cat, 0)
            by_op_type[ot]["llvm_categories"][cat] += data["llvm_category_breakdown"].get(cat, 0)

    # Compute aggregate ratios and category percentages
    aggregates = {}
    for ot, agg in by_op_type.items():
        total_sv = agg["total_sv_dynamic"]
        total_llvm = agg["total_llvm_dynamic"]
        sv_cat_pct = {}
        llvm_cat_pct = {}
        for cat in CATEGORIES:
            sv_cat_pct[cat] = round(agg["sv_categories"][cat] / max(total_sv, 1) * 100, 1)
            llvm_cat_pct[cat] = round(agg["llvm_categories"][cat] / max(total_llvm, 1) * 100, 1)

        aggregates[ot] = {
            "model_count": len(agg["models"]),
            "models": agg["models"],
            "total_scratchv_static_insns": agg["total_sv_static"],
            "total_scratchv_dynamic_insns": total_sv,
            "total_llvm_dynamic_insns": total_llvm,
            "dynamic_ratio": round(total_sv / max(total_llvm, 1), 2),
            "sv_categories": dict(agg["sv_categories"]),
            "llvm_categories": dict(agg["llvm_categories"]),
            "sv_category_pct": sv_cat_pct,
            "llvm_category_pct": llvm_cat_pct,
        }

    operator_regalloc = _operator_regalloc_aggregates(results, pressure_regs)

    # ── Build output ──
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": "ScratchV",
        "source_model": "cnn.onnx (3×Conv + 3×MaxPool + 2×FC)",
        "models": results,
        "aggregates": aggregates,
        "regalloc_comparison": {
            "baseline": "legacy positional dst/src inference",
            "current": "machine_semantics.py",
            "method": "same allocator, isolated ONNX operator models",
            "physical_register_counts": list(pressure_regs),
            "by_operator": operator_regalloc,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        _regalloc_markdown(operator_regalloc, pressure_regs),
        encoding="utf-8",
    )
    print(
        f"\nJSON: {args.output_json} ({len(json.dumps(output)):,} bytes)"
    )
    print(f"Markdown: {args.output_md}")

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("Operator-type aggregates:")
    print(f"{'Op Type':<12} {'Count':>5} {'SV Dynamic':>15} {'LLVM Dynamic':>15} {'Ratio':>8}")
    print("-" * 60)
    for ot in ["conv", "gemm", "maxpool", "relu", "sigmoid"]:
        if ot in aggregates:
            a = aggregates[ot]
            print(f"{ot:<12} {a['model_count']:>5} {a['total_scratchv_dynamic_insns']:>15,} "
                  f"{a['total_llvm_dynamic_insns']:>15,} {a['dynamic_ratio']:>7.2f}x")

    all_ok = all(
        model["compile_status"] == "ok"
        and model["regalloc_detail"]["status"] == "ok"
        and all(
            comparison["after"]["valid"]
            for comparison in model["regalloc_detail"]["pressure_sweep"].values()
        )
        for model in results.values()
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
