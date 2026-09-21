"""Same-input CNN scheduling cost and post-register-allocation safety audit.

Run with ``python -m benchmarks.bench_cnn_schedule``. The tracked ONNX model
is required; a generated replacement is never silently substituted. Costs
are sums over modeled regions, not whole-network execution times.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import tempfile

from benchmarks.run_inst_scheduler_case import run_case
from scratchv.backend.inst_scheduler import parse_instructions
from scratchv.backend.schedule_semantics import (
    LOADS, STORES, memory_address, register_name,
)
from scratchv.backend.schedule_verify import verify_schedule
from scratchv.compiler import CompilerConfig, CompilerDriver

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/graph/cnn.onnx"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assembly_resources(source: str) -> dict:
    """Count physical names and stack accesses, including unmodeled opcodes.

    Stack accesses include saves/restores as well as spills. An sp-relative
    offset count is not an allocator spill count or an allocated frame size.
    """
    instructions = parse_instructions(source)
    registers, offsets = set(), set()
    stack_loads = stack_stores = loads = stores = 0
    stack_sequence, stack_pointer_writes = [], []
    for inst in instructions:
        registers.update((inst.uses | inst.defines) - {"x0"})
        for operand in inst.operands:
            for token in re.split(r"[\s(),]+", operand):
                reg = register_name(token)
                if reg and reg != "x0":
                    registers.add(reg)
        load = inst.opcode in LOADS | {"ld"}
        store = inst.opcode in STORES | {"sd"}
        loads += load
        stores += store
        for operand in (inst.operands if load or store else ()):
            address = memory_address(operand)
            if address and address.base == "x2":
                stack_loads += load
                stack_stores += store
                offsets.add(address.offset)
                stack_sequence.append((inst.opcode, tuple(inst.operands)))
            elif "(" in operand and any(
                register_name(token) == "x2" for token in re.split(r"[\s(),]+", operand)
            ):
                raise ValueError(f"Cannot classify stack address: {inst.raw_line}")
        # Include opaque instructions which name sp as their first operand.
        if inst.operands and register_name(inst.operands[0]) == "x2" and not store:
            stack_pointer_writes.append((inst.opcode, tuple(inst.operands)))
    return {
        "source_instructions": len(instructions),
        "physical_registers": sorted(registers),
        "physical_register_count": len(registers),
        "loads": loads, "stores": stores,
        "stack_loads": stack_loads, "stack_stores": stack_stores,
        "stack_offsets": sorted(offsets), "stack_offset_count": len(offsets),
        "stack_access_sequence": stack_sequence,
        "stack_pointer_operations": stack_pointer_writes,
    }


def _local_live_peak(instructions, live_out: set[str]) -> int:
    """Backward liveness for a region with an explicit, shared exit contract."""
    live = set(live_out)
    peak = len(live)
    for inst in reversed(instructions):
        live.difference_update(inst.defines)
        live.update(inst.uses)
        peak = max(peak, len(live))
    return peak


def audit_side_effects(before: str, after: str, regions: list[dict]) -> dict:
    """Independently check each region's read origins, final writes and layout.

    Rebuild candidate identities from the original lines, never from the
    scheduler's dependency graph. Unknown instructions and unmodeled layouts
    must remain exactly in place. Local liveness includes all region final
    definitions as live-out; it does not claim whole-CFG register pressure.
    """
    a, b = assembly_resources(before), assembly_resources(after)
    before_lines, after_lines = before.splitlines(), after.splitlines()
    original = parse_instructions(before)
    errors, pressure = [], []
    checked_lines: set[int] = set()
    for row in regions:
        if row["original_cycles"] is None:
            continue
        start, end = row["start_line"], row["end_line"]
        body = [inst for inst in original if start <= inst.id + 1 <= end]
        available = defaultdict(deque)
        for inst in body:
            available[inst.raw_line].append(inst)
        candidate = []
        try:
            for inst in body:
                candidate.append(available[after_lines[inst.id]].popleft())
            verify_schedule(body, candidate)
        except (IndexError, ValueError) as exc:
            errors.append(f"Lines {start}-{end}: {exc}")
            continue
        checked_lines.update(inst.id for inst in body)
        live_out = set().union(*(inst.defines for inst in body))
        pressure.append({
            "start_line": start, "end_line": end,
            "live_out": sorted(live_out),
            "before": _local_live_peak(body, live_out),
            "after": _local_live_peak(candidate, live_out),
        })
    fixed_lines_equal = len(before_lines) == len(after_lines) and all(
        before_lines[i] == after_lines[i]
        for i in range(len(before_lines)) if i not in checked_lines
    )
    checks = {
        "instruction_multiset_equal": Counter(
            (i.opcode, tuple(i.operands)) for i in original
        ) == Counter((i.opcode, tuple(i.operands)) for i in parse_instructions(after)),
        "physical_registers_equal": a["physical_registers"] == b["physical_registers"],
        "stack_accesses_equal": a["stack_access_sequence"] == b["stack_access_sequence"],
        "stack_offsets_equal": a["stack_offsets"] == b["stack_offsets"],
        "stack_pointer_operations_equal": a["stack_pointer_operations"] == b["stack_pointer_operations"],
        "fixed_lines_equal": fixed_lines_equal,
        "region_dataflow_equal": not errors,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "before": a, "after": b, "checks": checks, "errors": errors,
        "verified_regions": len(pressure),
        "local_liveness": {
            "scope": "Per modeled region; all final definitions live at exit; not whole-CFG pressure",
            "peak_before": max((r["before"] for r in pressure), default=None),
            "peak_after": max((r["after"] for r in pressure), default=None),
            "increased_regions": sum(r["after"] > r["before"] for r in pressure),
            "regions": pressure,
        },
        "execution_verified": False,
    }


def analyze_assembly(path: Path, label: str, llvm_mca: str | None = None) -> dict:
    report = run_case(path, static_only=True, repeats=1, llvm_mca=llvm_mca)
    report["case"] = label
    report["output_sha256"] = sha256(report["assembly"]["after"].encode("utf-8"))
    report["side_effects"] = audit_side_effects(
        report["assembly"]["before"], report["assembly"]["after"],
        report["scheduling"]["regions"],
    )
    stats = report["scheduling"]
    modeled = stats["modeled_instructions"] > 0
    report["metric"] = {
        "name": "sum_of_modeled_region_cycles",
        "before": stats["original_cycles"] if modeled else None,
        "after": stats["final_cycles"] if modeled else None,
        "saved": stats["saved_cycles"] if modeled else None,
        "reduction_percent": 100 * stats["saved_cycles"] / stats["original_cycles"] if modeled else None,
    }
    if report["side_effects"]["status"] != "passed":
        report["status"] = "failed"
        report["errors"].append("Post-allocation side-effect checks failed")
    # Runtime of the report generator is not part of this reproducible metric.
    report.pop("timing")
    stats.pop("elapsed_seconds", None)
    return report


def run_benchmark(model_path: Path = DEFAULT_MODEL, *, standalone_asm: Path | None = None,
                  llvm_mca: str | None = None, execute: bool = False,
                  artifacts: Path | None = None) -> dict:
    import onnx
    from scratchv.backend.llvm_mca import model_metadata
    from benchmarks.schedule_analysis import cfg_liveness
    from benchmarks.cnn_schedule_execution import assemble_listing, execute_pair
    from scratchv.standalone.onnx_to_riscv_standalone import convert_onnx_to_riscv

    model_path = model_path.resolve()
    model_bytes = model_path.read_bytes()
    model = onnx.load_model_from_string(model_bytes)
    cpu_model = model_metadata(llvm_mca)
    cases = []
    with tempfile.TemporaryDirectory(prefix="cnn-schedule-") as directory:
        work = Path(directory)
        out = artifacts.resolve() if artifacts else work
        out.mkdir(parents=True, exist_ok=True)
        metadata = {}
        generated, scheduled = out / "standalone-before.s", out / "standalone-after.s"
        baseline_bin, scheduled_bin = out / "standalone-before.bin", out / "standalone-after.bin"
        with redirect_stdout(io.StringIO()):
            for enabled, asm, binary in ((False, generated, baseline_bin), (True, scheduled, scheduled_bin)):
                current_metadata = {}
                rc = convert_onnx_to_riscv(str(model_path), str(binary), str(asm), const_merge=True,
                                          schedule=enabled, metadata=current_metadata, llvm_mca=llvm_mca)
                if rc:
                    raise RuntimeError(f"Standalone CNN compilation failed: {rc}")
                if metadata and metadata != current_metadata:
                    raise ValueError("Scheduling changed the CNN memory layout or code size")
                metadata = current_metadata
                encoded = assemble_listing(asm, work)
                if bytes(encoded) != binary.read_bytes()[:metadata["code_bytes"]]:
                    raise ValueError("Standalone listing does not encode the generated machine words")
        if standalone_asm is not None and standalone_asm.read_bytes() != generated.read_bytes():
            raise ValueError("Standalone assembly does not match this ONNX model and --const-merge")
        primary = analyze_assembly(generated, "standalone/const-merge", llvm_mca)
        if primary["assembly"]["after"] != scheduled.read_text():
            raise ValueError("Standalone scheduled binary differs from the public assembly pass")
        if baseline_bin.read_bytes()[metadata["code_bytes"]:] != scheduled_bin.read_bytes()[metadata["code_bytes"]:]:
            raise ValueError("Scheduling changed weight bytes")
        primary["role"] = "primary"
        primary["binary"] = {"code_bytes": metadata["code_bytes"], "listing_roundtrip_equal": True,
                             "weights_equal": True, "before_sha256": sha256(baseline_bin.read_bytes()),
                             "after_sha256": sha256(scheduled_bin.read_bytes()),
                             "code_generator_sha256": sha256((ROOT / "scratchv/standalone/onnx_to_riscv_standalone.py").read_bytes())}
        primary["execution"] = execute_pair(baseline_bin, scheduled_bin, metadata, work) if execute else {"status": "not_run"}
        if execute and primary["execution"]["status"] != "passed":
            primary["status"] = "failed"
        cases.append(primary)
        for allocator in ("greedy", "linear"):
            path = out / f"compiler-{allocator}-before.s"
            config = CompilerConfig(reg_alloc=allocator, optimize_level="none")
            compiled = CompilerDriver(config).compile(str(model_path), str(path))
            if not compiled.success:
                raise RuntimeError("; ".join(compiled.errors))
            case = analyze_assembly(path, f"CompilerDriver/{allocator}", llvm_mca)
            (out / f"compiler-{allocator}-after.s").write_text(case["assembly"]["after"])
            case["role"] = "supplemental"
            case["execution"] = {"status": "not_run", "reason": "simplified ONNX lowering; not the full CNN path"}
            cases.append(case)
        for case in cases:
            case["input_sha256"] = sha256(case["assembly"]["before"].encode())
            case["side_effects"]["cfg_liveness"] = {
                phase: cfg_liveness(case["assembly"][phase]) for phase in ("before", "after")}
            case["side_effects"]["execution_verified"] = case["execution"]["status"] == "passed"
        primary["acceptance"] = {
            "static_safety": primary["side_effects"]["status"] == "passed",
            "coverage": primary["scheduling"]["unmodeled_by_opcode"] == {"auipc": 1},
            "llvm_cycles_improve": primary["scheduling"]["saved_cycles"] > 0,
            "regions_nonregressing": all(r["final_cycles"] <= r["original_cycles"]
                                          for r in primary["scheduling"]["regions"]
                                          if r["original_cycles"] is not None),
            "execution_equal": primary["execution"]["status"] == "passed" if execute else None,
        }
        if any(value is False for value in primary["acceptance"].values()):
            primary["status"] = "failed"
    return {
        "schema_version": 4, "benchmark_type": "cnn-scheduling-ab",
        "status": "passed" if all(c["status"] == "passed" for c in cases) else "failed",
        "validation_scope": "static_and_llvm" + ("_and_execution" if execute else ""),
        "whole_cnn_execution": primary["execution"]["status"],
        "input": {"path": model_path.relative_to(ROOT).as_posix() if model_path.is_relative_to(ROOT) else str(model_path),
                  "sha256": sha256(model_bytes), "bytes": len(model_bytes), "nodes": len(model.graph.node),
                  "operators": dict(sorted(Counter(n.op_type for n in model.graph.node).items()))},
        "environment": {"python": platform.python_version(), "onnx": onnx.__version__,
                        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "random"), "llvm_mca": cpu_model["version_output"]},
        "model": cpu_model,
        "metrics": {
            "peak_parallelism": "max instructions issued in one cycle; region maxima aggregated by max",
            "critical_path": "unavailable: this adapter does not collect critical-path analysis for this in-order target",
            "bubbles": "zero-issue cycles B = issue_span - count(distinct issue cycles); excludes drain; summed across regions",
            "issue_span": "last issue cycle + 1, with first issue normalized to zero; summed across regions",
            "bubble_ratio": "sum(bubbles) / sum(issue_span)",
            "scope": "each modeled region once, ready inputs; no loop weighting, cache misses or branch prediction",
            "llvm_cycles": "max(execution completion, last issue + 1) - first issue; simulator bookkeeping excluded",
        },
        "cases": cases,
    }


def markdown(report: dict) -> str:
    lines = ["# Topic 18：CNN 指令调度验证", "", f"验证状态：**{report['status']}**。"]
    if "error" in report:
        return "\n".join(lines + ["", report["error"], ""])
    lines += [
        f"输入：`{report['input']['path']}`，{report['input']['nodes']} 个节点；SHA-256：`{report['input']['sha256']}`。",
        "主路径为 standalone CNN 编译器，固定启用常量合并；CompilerDriver 两种分配器仅作补充回归。",
        "A/B 两侧使用相同版本的代码生成器，仅切换调度；生成器哈希保存在 JSON 中。",
        f"CPU 模型：LLVM {report['model']['llvm_version']} / `{report['model']['cpu']}`，RV32IMFD，禁用压缩指令。",
        "调度候选与原序均由 llvm-mca 分析；仅在完成周期严格减少时采用候选。指令延迟、旁路和资源占用由 LLVM 提供。",
        "配置：`-iterations=1 -noalias=false -timeline -timeline-max-cycles=0 -timeline-max-iterations=1`。模型定义链接、工具版本、输入和原始 LLVM JSON 均保存在报告 JSON 中。",
        "", "## 指标与结果", "",
        "| 指标 | 定义 |", "|---|---|",
        "| 最大指令并行数 | LLVM Timeline 中单周期发射指令数的峰值；跨区域取最大值 |",
        "| 关键路径长度 | N/A：当前接口未采集该顺序执行 CPU 的关键路径分析；不另行估算 |",
        "| 流水线气泡 | 发射区间内没有指令发射的周期数 B=S−U；S 为首末发射周期覆盖的长度，U 为有发射的周期数。排除尾部排空，不代表逐流水级气泡数 |",
        "| 完成周期 | 最后执行完成与最后发射加一的较大值，减去首发射周期；排除 LLVM 最后一个管理周期，包含排空 |",
        "| 汇总 | 周期和气泡跨区域求和；气泡率为 ΣB/ΣS，IPC 为建模指令数/完成周期合计 |",
        "", "表中数值均为调度前→后。", "",
        "| 路径 | 建模/输入指令 | 换序区域 | 峰值并行数 | 关键路径 | 周期合计 | 气泡周期 | 气泡率 | IPC |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        s = case["scheduling"]
        a, b = s["metrics_before"], s["metrics_after"]
        lines.append(f"| {case['case']} | {s['modeled_instructions']}/{s['input_instructions']} | {s['applied_regions']} | "
                     f"{a['peak_parallelism']}→{b['peak_parallelism']} | N/A | {a['cycles']}→{b['cycles']} | "
                     f"{a['bubbles']}→{b['bubbles']} | {a['bubble_ratio']:.1%}→{b['bubble_ratio']:.1%} | {a['ipc']:.3f}→{b['ipc']:.3f} |")
    primary = report["cases"][0]
    a, b = primary["scheduling"]["metrics_before"], primary["scheduling"]["metrics_after"]
    if a["cycles"]:
        lines += ["", f"主路径局部周期合计减少 {a['cycles'] - b['cycles']}（{(a['cycles'] - b['cycles']) / a['cycles']:.2%}），"
                  f"气泡减少 {a['bubbles'] - b['bubbles']} 个周期。"]
    lines += ["", "## 胜/平/负区域", "",
              "对照对象：同一编译路径、同一静态区域，A 为关闭调度的原始顺序，B 为最终输出顺序；两侧使用相同 LLVM 配置。不跨分配器比较。研究目标是判断换序能否减少该 CPU 模型下的局部等待。",
              "区域由标签、控制转移、未知指令和其他汇编文本边界划分，可能只是基本块的一部分；区域内终止指令固定。",
              "胜：T_B < T_A；平：T_B = T_A；负：T_B > T_A。T 为完成周期，与峰值并行数或气泡是否变化无关。每个成功分析的静态区域计一次，包含未换序区域；不按循环次数或收益大小加权。",
              "无收益和退化候选恢复原序，因此最终结果的负区域应为零；JSON 保留候选周期，可审查被拒绝的候选。工具失败会使 benchmark 失败，不视为持平。",
              "", "| 路径 | 对照区域数 | 胜/平/负区域 |", "|---|---:|---:|"]
    for case in report["cases"]:
        rows = [r for r in case["scheduling"]["regions"] if r["original_cycles"] is not None]
        changes = [r["original_cycles"] - r["final_cycles"] for r in rows]
        lines.append(f"| {case['case']} | {len(rows)} | {sum(d > 0 for d in changes)}/{sum(d == 0 for d in changes)}/{sum(d < 0 for d in changes)} |")
    lines += [
              "", "## 副作用与执行", "",
              "| 路径 | 物理寄存器数 | 栈加载/存储 | CFG 活跃峰值 | 结构检查 | 完整执行 |",
              "|---|---:|---:|---:|---|---|"]
    for case in report["cases"]:
        side = case["side_effects"]
        a, b = side["before"], side["after"]
        ca, cb = side["cfg_liveness"]["before"], side["cfg_liveness"]["after"]
        peak = f"{ca['peak']}→{cb['peak']}" if ca["status"] == cb["status"] == "completed" else "N/A"
        lines.append(f"| {case['case']} | {a['physical_register_count']}→{b['physical_register_count']} | "
                     f"{a['stack_loads']}/{a['stack_stores']}→{b['stack_loads']}/{b['stack_stores']} | {peak} | "
                     f"{side['status']} | {case['execution']['status']} |")
    primary = report["cases"][0]
    execution = primary["execution"]
    lines += ["", "结构检查包括指令/寄存器集合、读值来源、最终写入、访存顺序、控制流边界及 sp 操作。",
              "CFG 活跃性用固定点求解循环，出口契约为内存输出、sp 保留、ret 读取 ra；含未知语义的补充路径标为 N/A。",
              "调度发生在寄存器分配之后，没有再次分配；栈访存包含保存/恢复，不能直接当作精确 spill 数。"]
    if execution["status"] == "passed":
        lines.append(f"完整 standalone CNN 使用 QEMU 执行 {len(execution['samples'])} 组固定 Q16.16 输入，比较全部整数寄存器、工作区及输出：{execution['status']}。")
        lines.append("同时验证汇编重新编码与二进制一致、权重字节不变及缓冲区边界。执行结果在这些输入下前后一致；该检查不验证定点 CNN 与 ONNX 浮点参考的数值精度。")
        lines += ["", "| 输入种子 | 调度前输出（Q16.16 原始整数） | 调度后输出 | 全部状态一致 |",
                  "|---|---|---|---|"]
        for sample in execution["samples"]:
            lines.append(f"| {sample['seed']} | {sample['before']['output_q16']} | "
                         f"{sample['after']['output_q16']} | {sample['equal']} |")
    elif execution["status"] == "baseline_failed":
        lines.append("**完整 CNN 基线执行失败**，benchmark 未通过。失败状态及输入哈希保存在 JSON，不能据此判断调度前后执行等价。")
    lines += ["", "## 覆盖与限制", "",
              "覆盖率 = 纳入模型的源汇编指令条数 / 输入源汇编指令条数。标签和注释不计数，未展开的伪指令按一条计数；覆盖率不表示动态执行占比或发生换序的比例。",
              "未建模指令仍保留在输出中，但不计入周期、峰值并行数和气泡指标；未估算的开销不视为零周期，指令形式是否可合法编码需另行验证。",
              "", "| 路径 | 建模/输入指令 | 覆盖率 | 未建模指令及条数 |", "|---|---:|---:|---|"]
    missing_opcodes = set()
    invalid_slt = []
    for case in report["cases"]:
        stats = case["scheduling"]
        missing = stats["unmodeled_by_opcode"]
        missing_opcodes.update(missing)
        detail = "、".join(f"`{op}` × {count}" for op, count in missing.items()) or "无"
        coverage = f"{stats['coverage_ratio']:.1%}" if stats["input_instructions"] else "N/A"
        lines.append(f"| {case['case']} | {stats['modeled_instructions']}/{stats['input_instructions']} | {coverage} | {detail} |")
        for inst in parse_instructions(case["assembly"]["before"]):
            if inst.opcode == "slt" and inst.effects.barrier_reason == "invalid register operand":
                invalid_slt.append(f"{case['case']} 第 {inst.id + 1} 行：`slt {', '.join(inst.operands)}`")
    lines += ["", "未建模原因与处理：", ""]
    if "auipc" in missing_opcodes:
        lines.append("- **AUIPC：地址相关，固定原位。** 它计算“自身指令地址 + 立即数偏移”，主路径用它建立数据基地址。移动指令而不重算偏移会改变结果；当前不实现该地址重算，也不对 AUIPC 估算周期。活跃性分析仍可识别其寄存器写入。")
    if "max" in missing_opcodes:
        lines.append("- **max：尚未展开的项目伪指令。** 当前 RV32IM 路径中的 `max rd, rs, 0` 会展开成比较、分支和复制等多条指令。调度器在展开前处理文本，尚未对该展开序列建模，不能直接赋予一条普通指令的周期。")
    if invalid_slt:
        lines.append("- **slt：寄存器操作数形式无效。** " + "；".join(invalid_slt) + "。`slt` 要求寄存器操作数，立即数比较应生成 `slti`。正常寄存器形式的 `slt` 已支持；这里需要修正 CompilerDriver 指令选择，不能仅补充周期参数。")
    lines += ["",
              "未建模指令作为调度边界：自身固定，前后指令分别在各自区域内分析，禁止跨越该边界移动。其他未建模原因及所在行见 JSON 的 `scheduling.diagnostics`；CFG 活跃性遇到未知寄存器语义时标为 N/A，不能解释为零活跃寄存器。",
              "",
              "测试对象与结论范围：", "",
              "- **CompilerDriver 是补充编译回归。** 它将 Conv/Gemm 等算子简化为少量乘加，没有展开完整张量计算；与 standalone 的静态指令数差异不能归因于调度。完整执行列的 `not_run` 表示未执行机器码做结果验证；静态结构检查通过不等于完整 CNN 推理正确。",
              "- **静态区域只计一次。** 完整 standalone 代码也包含循环；例如 4 条循环体指令执行 1000 次，静态仍是 4 条，动态执行量为 4000 次。报告不按循环次数或分支频率加权，各区域从周期 0、入口操作数就绪开始独立估算，没有贯通跨区域的动态等待。",
              "- **性能结论限于模型。** llvm-mca 使用 SiFive7 的资源约束；报告未测量真实硬件周期、缓存未命中或分支预测开销，因此局部周期合计下降不能直接换算为整网运行时间下降。",
              "- **执行验证覆盖已测输入。** QEMU 用于比较调度前后执行状态，不用于测量硬件加速；固定输入上的一致性不等于所有输入的证明，也不验证定点结果与 ONNX 浮点参考的精度。", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--standalone-asm", type=Path)
    parser.add_argument("--llvm-mca", default=os.environ.get("LLVM_MCA", "llvm-mca"))
    parser.add_argument("--artifacts", type=Path, default=Path("benchmark_reports/cnn_schedule"))
    parser.add_argument("--json", type=Path, default=Path("benchmark_reports/inst_scheduler_cnn.json"))
    parser.add_argument("--markdown", type=Path, default=Path("benchmark_reports/inst_scheduler_cnn.md"))
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args.model.resolve(), standalone_asm=args.standalone_asm,
                               llvm_mca=args.llvm_mca, execute=True, artifacts=args.artifacts)
    except Exception as exc:
        report = {"benchmark_type": "cnn-scheduling-ab", "status": "failed", "error": str(exc)}
    for path in (args.json, args.markdown):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rendered = markdown(report)
    args.markdown.write_text(rendered, encoding="utf-8")
    print(rendered)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
