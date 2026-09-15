"""Benchmark legal physical-register regions through the production scheduler.

Run: python -m benchmarks.bench_inst_scheduler --repeats 20 --json report.json --markdown report.md
Reported cycles are sums of local static model estimates, not CPU measurements.
These generated inputs are synthetic; compiler-output A/B and real execution
are reported separately by benchmarks.run_inst_scheduler_case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

from scratchv.backend.inst_scheduler import (
    InstructionScheduler,
    SchedInst,
    ScheduleConfig,
    schedule_assembly,
)


def _gen_instructions(
    num_insts: int, seed: int = 42, dep_chains: int = 3
) -> list[SchedInst]:
    """Generate real opcodes/operands; keep a0 as an unchanged memory base."""
    rng = random.Random(seed)
    registers = [f"x{i}" for i in list(range(5, 10)) + list(range(11, 32))]
    chain_count = max(1, min(dep_chains, len(registers)))
    groups = [registers[i::chain_count] for i in range(chain_count)]
    previous = [group[0] for group in groups]
    instructions = []
    for index in range(num_insts):
        chain = index % chain_count
        dst, src = rng.choice(groups[chain]), previous[chain]
        op = rng.choice(["add", "sub", "mul", "div", "lw", "sw", "addi", "li", "mv"])
        if op in {"lw", "sw"}:
            operands = [dst, f"{rng.randrange(8) * 4}(a0)"]
        elif op == "li":
            operands = [dst, str(rng.randrange(-100, 100))]
        elif op == "mv":
            operands = [dst, src]
        elif op == "addi":
            operands = [dst, src, str(rng.randrange(-16, 16))]
        else:
            operands = [dst, src, rng.choice(groups[chain])]
        instructions.append(
            SchedInst(index, op, operands, raw_line=f"  {op} " + ", ".join(operands))
        )
        if op != "sw":
            previous[chain] = dst
    return instructions


def bench_build_dag(insts: list[SchedInst], repeats: int = 20) -> dict:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        dag = InstructionScheduler().build_dag(insts)
        times.append(time.perf_counter() - start)
    return {"num_nodes": len(dag), "mean_s": statistics.mean(times)}


def bench_schedule(
    insts: list[SchedInst], repeats: int = 20, max_region_size: int = 1024
) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    source = "\n".join(inst.raw_line for inst in insts) + "\n"
    times = []
    config = ScheduleConfig(strict=True, max_region_size=max_region_size)
    for _ in range(repeats):
        start = time.perf_counter()
        result = schedule_assembly(source, config)
        times.append(time.perf_counter() - start)
    stats = result.stats
    return {
        "benchmark_type": "synthetic",
        "input_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "repeats": repeats,
        "max_region_size": max_region_size,
        "num_insts": len(insts),
        "orig_cycles": stats["original_cycles"],
        "sched_cycles": stats["final_cycles"],
        "improvement": stats["saved_cycles"],
        "modeled": stats["modeled_instructions"],
        "skipped": stats["skipped_regions"],
        "moved": stats["moved_instructions"],
        "applied_regions": stats["applied_regions"],
        "unchanged_regions": sum(
            row["status"] == "no_improvement" for row in stats["regions"]
        ),
        "sensitivity_rejected_regions": stats["sensitivity_rejected_regions"],
        "execution_verified": False,
        "model": stats["model"],
        "mean_s": statistics.mean(times),
        "stdev_s": statistics.stdev(times) if repeats > 1 else 0,
    }


def _markdown(records: list[dict]) -> str:
    lines = [
        "# 课题 18：合成指令调度 Benchmark",
        "",
        "固定随机种子生成合法物理寄存器指令，单独统计优化器耗时和局部模型收益。",
        "这些数据不是实际工作负载；未执行汇编，也未测量硬件周期。",
        "",
        "| 指令数 | 已建模 | 原序周期（估算） | 最终周期（估算） | 减少 | 移动指令 | 应用区域 | 无收益区域 | 敏感性回退区域 | 跳过区域 | 耗时均值 ± 标准差（ms） |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        modeled = row["modeled"] > 0
        before = row["orig_cycles"] if modeled else "N/A"
        after = row["sched_cycles"] if modeled else "N/A"
        saved = row["improvement"] if modeled else "N/A"
        lines.append(
            f"| {row['num_insts']} | {row['modeled']} | {before} | {after} | {saved} | "
            f"{row['moved']} | {row['applied_regions']} | {row['unchanged_regions']} | "
            f"{row['sensitivity_rejected_regions']} | {row['skipped']} | "
            f"{row['mean_s'] * 1000:.3f} ± {row['stdev_s'] * 1000:.3f} |"
        )
    if records:
        row = records[0]
        lines.extend(
            [
                "",
                f"模型：`{row['model']}`；重复次数：{row['repeats']}；区域上限：{row['max_region_size']}。",
                "未建模区域显示 N/A，不表示零周期；零收益与跳过样例均保留。",
                "JSON 保留数组格式，逐条记录样例类型、输入 SHA-256、种子和统计口径。",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-region-size", type=int, default=1024)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_region_size < 1:
        parser.error("repeats and max-region-size must be positive")
    records = []
    print(
        "Synthetic inputs; local static model estimates, not measured hardware cycles"
    )
    print(
        f"{'Size':>6} {'Time(ms)':>10} {'Before':>8} {'After':>8} {'Saved':>8} {'Modeled':>8} {'Skipped':>8}"
    )
    for size in [10, 50, 100, 200, 500, 1000, 5000]:
        row = bench_schedule(
            _gen_instructions(size), args.repeats, args.max_region_size
        )
        row["seed"] = 42
        records.append(row)
        before = str(row["orig_cycles"]) if row["modeled"] else "N/A"
        after = str(row["sched_cycles"]) if row["modeled"] else "N/A"
        saved = str(row["improvement"]) if row["modeled"] else "N/A"
        print(
            f"{size:6} {row['mean_s'] * 1000:10.3f} {before:>8} "
            f"{after:>8} {saved:>8} {row['modeled']:8} {row['skipped']:8}"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(records), encoding="utf-8")


if __name__ == "__main__":
    main()
