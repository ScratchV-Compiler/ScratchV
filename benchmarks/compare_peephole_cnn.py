#!/usr/bin/env python3
"""Run a real standalone CNN A/B peephole benchmark."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.compare_peephole_html import generate_html_report  # noqa: E402
from scratchv.standalone.onnx_to_riscv_standalone import CNNRISCVGenerator  # noqa: E402
from scratchv.standalone.onnx_to_riscv_standalone import MemoryPlan  # noqa: E402
from scratchv.standalone.onnx_to_riscv_standalone import ONNXModel  # noqa: E402
from scratchv.standalone.onnx_to_riscv_standalone import (  # noqa: E402
    patch_gp_data_base,
)
from scratchv.standalone.peephole_relocator import PeepholeFinalizer  # noqa: E402


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _memory_plan_payload(memory: MemoryPlan) -> dict:
    return {
        "weight_offsets": dict(memory.weight_offsets),
        "workspace_offsets": dict(memory.workspace_offsets),
        "data_size": memory.data_size,
        "workspace_size": memory.workspace_size,
    }


def _prepare_model(model_path: Path) -> tuple[ONNXModel, MemoryPlan, bytes]:
    model = ONNXModel.from_file(str(model_path))
    memory = MemoryPlan()
    weight_data = memory.layout_weights(model.initializers)
    if model.inputs:
        input_tensor = model.inputs[0]
        memory.alloc_workspace(input_tensor.name, input_tensor.num_elements)
    return model, memory, weight_data


def _compile_variant(
    model: ONNXModel,
    initial_memory: MemoryPlan,
    weight_data: bytes,
    *,
    peephole_enabled: bool,
) -> dict:
    memory = copy.deepcopy(initial_memory)
    finalizer = PeepholeFinalizer(enabled=peephole_enabled)
    generator = CNNRISCVGenerator(
        model,
        memory,
        finalize_strategy=finalizer,
    )
    code_bytes = generator.generate()
    code_bytes, gp_patch = patch_gp_data_base(
        generator,
        code_bytes,
        sync_listing=True,
        strict=True,
    )
    binary = code_bytes + weight_data
    optimizer = finalizer.optimizer
    rule_matches = (
        optimizer.total_matches
        if peephole_enabled
        else {rule.name: 0 for rule in optimizer.rules}
    )
    return {
        "peephole_enabled": peephole_enabled,
        "input_assembly": finalizer.input_assembly,
        "input_assembly_sha256": _sha256(finalizer.input_assembly.encode()),
        "assembly_listing": generator.emit.disassemble(),
        "binary": binary,
        "binary_sha256": _sha256(binary),
        "code_size": len(code_bytes),
        "static_instructions": len(code_bytes) // 4,
        "memory_plan": _memory_plan_payload(memory),
        "machine_code_success": bool(binary) and len(code_bytes) % 4 == 0,
        "relocation_validation": finalizer.relocation_validation
        and not generator.emit.pending_fixups,
        "gp_patch": gp_patch,
        "rule_matches": rule_matches,
        "rule_applications": sum(rule_matches.values()),
        "fixed_point_iterations": optimizer.iterations if peephole_enabled else 0,
    }


def compare_cnn_model(model_path: str | Path) -> dict:
    """Compile one model twice; peephole is the only A/B variable."""
    path = Path(model_path)
    model, initial_memory, weight_data = _prepare_model(path)
    baseline = _compile_variant(
        model,
        initial_memory,
        weight_data,
        peephole_enabled=False,
    )
    optimized = _compile_variant(
        model,
        initial_memory,
        weight_data,
        peephole_enabled=True,
    )

    parameters = {
        "baseline": {
            "backend": "standalone-rv32im",
            "optimize_level": "none",
            "const_merge": False,
            "compact_constants": False,
            "peephole_enabled": False,
        },
        "optimized": {
            "backend": "standalone-rv32im",
            "optimize_level": "none",
            "const_merge": False,
            "compact_constants": False,
            "peephole_enabled": True,
        },
    }
    comparable = {
        key: value
        for key, value in parameters["baseline"].items()
        if key != "peephole_enabled"
    } == {
        key: value
        for key, value in parameters["optimized"].items()
        if key != "peephole_enabled"
    }
    memory_equal = baseline["memory_plan"] == optimized["memory_plan"]
    input_hash_equal = (
        baseline["input_assembly_sha256"] == optimized["input_assembly_sha256"]
    )
    before = baseline["static_instructions"]
    after = optimized["static_instructions"]
    saved = before - after
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": path.name,
            "path": str(path),
            "nodes": len(model.nodes),
            "initializers": len(model.initializers),
        },
        "compiler_parameters": parameters,
        "compiler_parameters_equal_except_peephole": comparable,
        "memory_plan_equal": memory_equal,
        "input_assembly_sha256": {
            "baseline": baseline["input_assembly_sha256"],
            "optimized": optimized["input_assembly_sha256"],
            "equal": input_hash_equal,
        },
        "binary_sha256": {
            "baseline": baseline["binary_sha256"],
            "optimized": optimized["binary_sha256"],
        },
        "instructions": {
            "baseline": before,
            "optimized": after,
            "saved": saved,
            "saved_percent": round(100.0 * saved / before, 3) if before else 0.0,
        },
        "code_size": {
            "baseline": baseline["code_size"],
            "optimized": optimized["code_size"],
            "saved": baseline["code_size"] - optimized["code_size"],
        },
        "machine_code": {
            "baseline": {"success": baseline["machine_code_success"]},
            "optimized": {"success": optimized["machine_code_success"]},
        },
        "relocation_validation": {
            "baseline": baseline["relocation_validation"],
            "optimized": optimized["relocation_validation"],
            "labels": True,
        },
        "gp_patch_validation": {
            "baseline": bool(baseline["gp_patch"]["validation"]),
            "optimized": bool(optimized["gp_patch"]["validation"]),
            "code_size_used": {
                "baseline": baseline["gp_patch"]["code_size"],
                "optimized": optimized["gp_patch"]["code_size"],
            },
        },
        "peephole": {
            "enabled": True,
            "total_matches": optimized["rule_matches"],
            "rule_applications": optimized["rule_applications"],
            "fixed_point_iterations": optimized["fixed_point_iterations"],
        },
        "baseline": {
            "assembly_listing": baseline["assembly_listing"],
            "memory_plan": baseline["memory_plan"],
        },
        "optimized": {
            "assembly_listing": optimized["assembly_listing"],
            "memory_plan": optimized["memory_plan"],
        },
    }


def _html_payload(report: dict) -> tuple[dict, list[tuple[str, str]]]:
    rules = report["peephole"]["total_matches"]
    before = report["instructions"]["baseline"]
    after = report["instructions"]["optimized"]
    saved = report["instructions"]["saved"]
    payload = {
        "generated_at": report["generated_at"],
        "metadata": {
            "python": platform.python_version(),
            "comparison": "standalone CNN, identical compiler parameters",
        },
        "summary": {
            "case_count": 1,
            "unchanged_cases": int(saved == 0),
            "before_instructions": before,
            "after_instructions": after,
            "reduced_instructions": saved,
            "reduction_percent": report["instructions"]["saved_percent"],
            "changes": report["peephole"]["rule_applications"],
            "rule_matches": rules,
        },
        "cases": [
            {
                "case_id": report["model"]["name"],
                "description": "真实 standalone RV32IM 编译结果",
                "peephole_off": {"instructions": before},
                "peephole_on": {"instructions": after},
                "reduced_instructions": saved,
                "reduction_percent": report["instructions"]["saved_percent"],
                "changes": report["peephole"]["rule_applications"],
                "rule_matches": rules,
            }
        ],
    }
    parameters_equal = (
        "是" if report["compiler_parameters_equal_except_peephole"] else "否"
    )
    rows = [
        ("模型", report["model"]["path"]),
        ("优化前代码大小", f'{report["code_size"]["baseline"]:,} B'),
        ("优化后代码大小", f'{report["code_size"]["optimized"]:,} B'),
        ("规则应用总次数", str(report["peephole"]["rule_applications"])),
        ("fixed-point 迭代次数", str(report["peephole"]["fixed_point_iterations"])),
        ("A/B 编译参数（除窥孔开关外）一致", parameters_equal),
        (
            "A/B 输入汇编哈希一致",
            "是" if report["input_assembly_sha256"]["equal"] else "否",
        ),
        (
            "baseline 机器码生成",
            "是" if report["machine_code"]["baseline"]["success"] else "否",
        ),
        (
            "optimized 机器码生成",
            "是" if report["machine_code"]["optimized"]["success"] else "否",
        ),
        ("重定位校验", "是" if all(report["relocation_validation"].values()) else "否"),
        (
            "GP 修补校验",
            (
                "是"
                if all(
                    report["gp_patch_validation"][key]
                    for key in ("baseline", "optimized")
                )
                else "否"
            ),
        ),
    ]
    return payload, rows


def generate_cnn_html(report: dict) -> str:
    """Render CNN data through the shared peephole HTML renderer."""
    payload, rows = _html_payload(report)
    return generate_html_report(
        payload,
        title="ScratchV 窥孔优化器 CNN Benchmark",
        comparison_note=(
            "baseline 与 optimized 使用同一模型、MemoryPlan、workspace、权重布局和编译参数；"
            "唯一变量是是否执行窥孔优化。"
        ),
        show_case_polarity=False,
        extra_environment_rows=rows,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Standalone CNN peephole A/B benchmark"
    )
    parser.add_argument("--model", type=Path, default=ROOT / "models/graph/cnn.onnx")
    parser.add_argument(
        "--json", type=Path, default=Path("benchmark_reports/cnn_peephole_compare.json")
    )
    args = parser.parse_args(argv)

    report = compare_cnn_model(args.model)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "CNN standalone peephole: "
        f'{report["instructions"]["baseline"]} -> '
        f'{report["instructions"]["optimized"]} instructions, '
        f'{report["instructions"]["saved"]} saved'
    )
    print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
