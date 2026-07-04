#!/usr/bin/env python3
"""Benchmark single-operator ONNX models — collect static/dynamic instruction data.

For each single-op model in models/single_op/:
  1. Compile with ScratchV → static instruction count
  2. Map to per-layer dynamic instruction estimates (ScratchV + LLVM)
  3. Aggregate by operator type

Output: benchmark_reports/single_op_bench.json
"""

from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

PROJ = Path(__file__).resolve().parent.parent
SINGLE_OP_DIR = PROJ / "models" / "single_op"
OUTPUT_FILE = PROJ / "benchmark_reports" / "single_op_bench.json"


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
    "cnn_PPQ_Operation_6":   ("reshape",  "Reshape (flatten)", "Reshape (flatten)"),
    "cnn_PPQ_Operation_12":  ("reshape",  "Reshape (flatten)", "Reshape (flatten)"),
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

SV_RESHAPE_PER_EL = {
    "alu_r": 0.0,     # no computation — pure data copy
    "alu_i": 5.0,     # addi (loop counter + address calc: slli+add for src + slli+add for dst)
    "load": 1.0,      # lw (load element)
    "store": 1.0,     # sw (store element)
    "branch": 1.0,    # bne (loop control)
    "jump": 0.0,
    "upper": 0.5,     # li32 for num_el (amortized)
    "fp": 0.0,
    "shift": 2.0,     # slli (address calc × 2 per element)
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

LLVM_RESHAPE_PER_EL = {
    "alu_i": 3.0,      # LLVM memcpy — mostly GEP + pointer arithmetic
    "load": 1.0,       # load element
    "store": 1.0,      # store element
    "branch": 1.0,     # loop control
    "alu_r": 0.0,      # no compute
    "jump": 0.0, "upper": 0.0, "fp": 0.0, "shift": 0.0,
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
    bin_out = f"/tmp/_singleop_{os.path.basename(model_path)}.bin"
    cmd = [
        sys.executable,
        str(PROJ / "scratchv" / "standalone" / "onnx_to_riscv_standalone.py"),
        model_path, "-o", bin_out,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            cwd=str(PROJ),
        )
        elapsed = time.perf_counter() - t0
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "static_insns": 0, "code_size_bytes": 0, "elapsed_s": 30}
    except Exception as e:
        return {"status": "error", "static_insns": 0, "code_size_bytes": 0, "elapsed_s": 0, "error": str(e)}

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
            sv_unit = SV_RESHAPE_PER_EL
            ll_unit = LLVM_RESHAPE_PER_EL
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


def main():
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

    # Compile each model with ScratchV and collect data
    results = {}
    for model_name in sorted(models.keys()):
        info = models[model_name]
        path = info["path"]
        op_type = info["op_type"]

        print(f"  Compiling: {model_name} ({op_type})...", end=" ", flush=True)
        comp = _compile_scratchv(path)
        print(f"static={comp['static_insns']} insns, {comp['status']}")

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

    # ── Build output ──
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": "ScratchV",
        "source_model": "cnn.onnx (3×Conv + 3×MaxPool + 2×Gemm + 4×ReLU + 1×Sigmoid + 2×Reshape)",
        "models": results,
        "aggregates": aggregates,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n→ {OUTPUT_FILE} ({len(json.dumps(output)):,} bytes)")

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("Operator-type aggregates:")
    print(f"{'Op Type':<12} {'Count':>5} {'SV Dynamic':>15} {'LLVM Dynamic':>15} {'Ratio':>8}")
    print("-" * 60)
    for ot in ["conv", "gemm", "maxpool", "relu", "sigmoid", "reshape"]:
        if ot in aggregates:
            a = aggregates[ot]
            print(f"{ot:<12} {a['model_count']:>5} {a['total_scratchv_dynamic_insns']:>15,} "
                  f"{a['total_llvm_dynamic_insns']:>15,} {a['dynamic_ratio']:>7.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
