"""Audit real compiler coverage and independent llvm-mca scheduling A/B results.

Run from the repository root. llvm-mca is required: missing tools, compilation
failures and unsupported MCA inputs are errors, never silently skipped cases.
MCA cycles are target-model estimates, not hardware measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from benchmarks.bench_inst_scheduler import _gen_instructions
from scratchv.backend.inst_scheduler import ScheduleConfig, parse_instructions, schedule_assembly
from scratchv.backend.schedule_semantics import memory_address
from scratchv.compiler import CompilerConfig, CompilerDriver

CPUS = ("rocket-rv32", "sifive-e76")
SIZES = (10, 50, 100, 200, 500, 1000)
SEEDS = (42, 1, 2)
CHAINS = (1, 2, 3, 8)


def mca_cycles(executable: str, cpu: str, source: str) -> int:
    result = subprocess.run(
        [executable, "-mtriple=riscv32", f"-mcpu={cpu}", "-mattr=+m,+f,+d", "-iterations=1"],
        input=source, text=True, capture_output=True, timeout=30, check=True,
    )
    # Never accept partial analysis (e.g. an ignored unsupported instruction).
    if result.stderr.strip():
        raise RuntimeError(f"llvm-mca diagnostic for {cpu}: {result.stderr.strip()}")
    match = re.search(r"^Total Cycles:\s*(\d+)$", result.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError(f"llvm-mca did not report Total Cycles for {cpu}")
    return int(match.group(1))


def compare(executable: str, before: str, after: str) -> dict:
    results = {}
    for cpu in CPUS:
        original = mca_cycles(executable, cpu, before)
        final = original if before == after else mca_cycles(executable, cpu, after)
        results[cpu] = {"before": original, "after": final, "saved": original - final}
    return results


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for cpu in CPUS:
        samples = [row["mca"][cpu] for row in rows]
        summary[cpu] = {
            "samples": len(samples),
            "before": sum(row["before"] for row in samples),
            "after": sum(row["after"] for row in samples),
            "saved": sum(row["saved"] for row in samples),
            "win": sum(row["saved"] > 0 for row in samples),
            "tie": sum(row["saved"] == 0 for row in samples),
            "loss": sum(row["saved"] < 0 for row in samples),
        }
    return summary


def canonical_region(source: str, start: int, end: int) -> str:
    """Render modeled body instructions in GAS syntax for independent MCA.

    Branches stay fixed and are excluded from these local body estimates;
    source line boundaries and original source hashes remain in the report.
    """
    lines = []
    for inst in parse_instructions(source):
        if not start <= inst.id + 1 <= end or inst.terminator:
            continue
        if inst.effects.barrier_reason:
            raise ValueError(f"unmodeled instruction in MCA region: {inst.raw_line}")
        operands = list(inst.operands)
        address = inst.effects.address
        if address is not None:
            reg = operands[1] if memory_address(operands[0]) else operands[0]
            operands = [reg, f"{address.offset}({address.base})"]
        lines.append(inst.opcode + " " + ", ".join(operands))
    return "\n".join(lines) + "\n"


def run_audit(executable: str, root: Path) -> dict:
    version = subprocess.run([executable, "--version"], text=True, capture_output=True,
                             timeout=10, check=True).stdout.splitlines()[:3]
    synthetic = []
    for size in SIZES:
        for seed in SEEDS:
            for chains in CHAINS:
                source = "\n".join(i.raw_line for i in _gen_instructions(size, seed, chains)) + "\n"
                result = schedule_assembly(source, ScheduleConfig(strict=True))
                synthetic.append({
                    "size": size, "seed": seed, "chains": chains,
                    "input_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "changed": result.changed,
                    "model_before": result.stats["original_cycles"],
                    "model_after": result.stats["final_cycles"],
                    "sensitivity_rejected": result.stats.get("sensitivity_rejected_regions", 0),
                    "mca": compare(executable, source, result.asm_text),
                })

    paths = sorted((root / "benchmarks/cases").glob("[0-9]*.dsl"))
    cnn = root / "models/graph/cnn.onnx"
    if not cnn.exists():
        raise FileNotFoundError(f"real-input audit requires {cnn}; run scripts/gen_minimal_cnn.py")
    paths.append(cnn)
    real = []
    real_regions = []
    for allocator in ("greedy", "linear"):
        driver = CompilerDriver(CompilerConfig(reg_alloc=allocator, schedule=True, schedule_strict=True))
        for path in paths:
            source = driver._generate_code(driver._parse(str(path)))
            stats, warnings = {}, []
            final = driver._run_asm_passes(source, warnings, stats)
            schedule = stats["schedule"]
            name = path.relative_to(root).as_posix()
            real.append({
                "input": name, "allocator": allocator,
                "input_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "input_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "output_sha256": hashlib.sha256(final.encode()).hexdigest(),
                "changed": source != final, "scheduling": schedule,
                "source_instructions_before": len(parse_instructions(source)),
                "source_instructions_after": len(parse_instructions(final)),
                "execution_verified": False,
            })
            for region in schedule["regions"]:
                if region["status"] != "applied":
                    continue
                start, end = region["start_line"], region["end_line"]
                before = canonical_region(source, start, end)
                after = canonical_region(final, start, end)
                real_regions.append({
                    "input": name, "allocator": allocator, "start_line": start, "end_line": end,
                    "scope": "applied region body; fixed terminators excluded",
                    "before": before, "after": after, "mca": compare(executable, before, after),
                })

    coverage = {}
    for allocator in ("greedy", "linear"):
        rows = [r for r in real if r["allocator"] == allocator]
        total = sum(r["scheduling"]["input_instructions"] for r in rows)
        modeled = sum(r["scheduling"]["modeled_instructions"] for r in rows)
        sizes = [region["instructions"] for r in rows for region in r["scheduling"]["regions"]]
        coverage[allocator] = {
            "files": len(rows), "changed_files": sum(r["changed"] for r in rows),
            "input_instructions": total, "modeled_instructions": modeled,
            "unmodeled_instructions": total - modeled,
            "coverage_ratio": modeled / total if total else 0.0,
            "mean_region_size": sum(sizes) / len(sizes) if sizes else 0.0,
            "max_region_size": max(sizes, default=0),
            "instruction_count_preserved": all(
                r["source_instructions_before"] == r["source_instructions_after"]
                for r in rows
            ),
        }
    synthetic_summary = summarize(synthetic)
    real_summary = summarize(real_regions)
    # The review asks for positive aggregate savings and at least as many wins
    # as losses. Every sample, including losses and ties, remains in the JSON.
    passed = all(s["saved"] > 0 and s["win"] >= s["loss"] for s in synthetic_summary.values())
    passed &= all(s["saved"] >= 0 and s["win"] >= s["loss"] for s in real_summary.values())
    passed &= all(s["instruction_count_preserved"] for s in coverage.values())
    return {
        "benchmark_type": "scheduler-review-audit", "status": "passed" if passed else "failed",
        "llvm_mca_version": version, "iterations": 1, "cpus": list(CPUS),
        "scope": "LLVM static CPU models; no hardware timing or whole-program execution",
        "model": ScheduleConfig().model.name,
        "synthetic_summary": synthetic_summary, "real_summary": real_summary,
        "coverage": coverage, "synthetic": synthetic, "real": real, "real_regions": real_regions,
    }


def markdown(report: dict) -> str:
    lines = [
        "# Topic 18：Review 迭代独立验证", "",
        f"验收：**{report['status']}**；调度模型：`{report['model']}`。",
        "llvm-mca：" + "; ".join(report["llvm_mca_version"]) + "。",
        "每个样例运行 1 次迭代；这是 LLVM CPU 模型估算，不是硬件计时。", "",
        "| 语料 | CPU | 原序周期 | 最终周期 | 节省 | 胜 / 平 / 负 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for key, label in (("synthetic_summary", "72 个合成样例"), ("real_summary", "真实产物已应用区域")):
        for cpu, row in report[key].items():
            lines.append(f"| {label} | {cpu} | {row['before']} | {row['after']} | {row['saved']} | "
                         f"{row['win']} / {row['tie']} / {row['loss']} |")
    lines += ["", "真实产物按已应用区域的指令体估算，排除固定终止指令；未变化文件也保留在 JSON 中。", "",
              "| 分配器 | 变化文件 / 总文件 | 建模 / 输入指令 | 覆盖率 | 平均 / 最大区域长度 |",
              "|---|---:|---:|---:|---:|"]
    for allocator, row in report["coverage"].items():
        lines.append(f"| {allocator} | {row['changed_files']} / {row['files']} | "
                     f"{row['modeled_instructions']} / {row['input_instructions']} | "
                     f"{row['coverage_ratio']:.1%} | {row['mean_region_size']:.2f} / {row['max_region_size']} |")
    lines += ["", "合成验收要求每个 CPU 总节省 > 0 且胜例数 ≥ 负例数；真实已应用区域要求总节省 ≥ 0 且胜例数 ≥ 负例数。",
              "通过总体门槛不代表每个样例都加速。完整 JSON 保留负例、零收益、未建模原因和输入哈希。", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm-mca", default="llvm-mca")
    parser.add_argument("--json", type=Path, default=Path("benchmark_reports/inst_scheduler_audit.json"))
    parser.add_argument("--markdown", type=Path, default=Path("benchmark_reports/inst_scheduler_audit.md"))
    args = parser.parse_args(argv)
    executable = shutil.which(args.llvm_mca)
    if executable is None:
        parser.error(f"llvm-mca is required: {args.llvm_mca}")
    report = run_audit(executable, Path.cwd())
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    args.markdown.write_text(markdown(report))
    print(markdown(report))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
