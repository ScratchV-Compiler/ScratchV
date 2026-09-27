"""Compare scheduling-disabled compilation against a separate base checkout."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect(root: Path) -> dict:
    # Run only in an isolated process. Never reuse modules from another checkout.
    sys.path.insert(0, str(root))
    from scratchv import compiler
    from scratchv.backend.riscv_encoder import assemble_to_binary
    from scratchv.standalone.onnx_to_riscv_standalone import convert_onnx_to_riscv
    from scripts.run_topic06_benchmarks import run_simulation

    if not Path(compiler.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Compiler imported outside the selected checkout")
    cases = sorted((root / "benchmarks/cases").glob("*.dsl"))
    cases += sorted((root / "tests/topic06/cases").rglob("*.dsl"))
    cases.append(root / "models/graph/cnn.onnx")
    if len(cases) < 2 or any(not path.is_file() for path in cases):
        raise ValueError("Missing comparison corpus")
    corpus = {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in cases}
    rows = {}
    with tempfile.TemporaryDirectory(prefix="schedule-disabled-") as directory:
        work = Path(directory)
        for path in cases:
            name = path.relative_to(root).as_posix()
            for allocator in ("greedy", "linear", "naive"):
                for dag in (False, True):
                    for optimization in ("none", "all"):
                        key = f"{name}|{allocator}|dag={dag}|opt={optimization}"
                        driver = compiler.CompilerDriver(compiler.CompilerConfig(
                            reg_alloc=allocator, use_dag_isel=dag,
                            optimize_level=optimization, schedule=False))
                        try:
                            result = driver.compile(str(path), str(work / "out.s"))
                        except Exception as exc:
                            rows[key] = {"compile": "exception", "error": str(exc).replace(str(root), "<root>")}
                            continue
                        row = {"compile": "passed" if result.success else "failed"}
                        if not result.success:
                            row["errors"] = [e.replace(str(root), "<root>") for e in result.errors]
                            rows[key] = row
                            continue
                        assembly = result.output_text
                        row.update(assembly_sha256=digest(assembly.encode()),
                                   register_map=result.stats["register_map"],
                                   spills=assembly.count("[regalloc:spill]"),
                                   reloads=assembly.count("[regalloc:reload]"))
                        try:
                            code = bytes(assemble_to_binary(assembly))
                            row.update(encoding="passed", binary_sha256=digest(code), code_bytes=len(code))
                        except Exception as exc:
                            row.update(encoding="failed", encoding_error=str(exc))
                        # The existing Topic 06 executable corpus has scalar inputs.
                        if (name.startswith("tests/topic06/cases/")
                                and path.parent.name in {"activation", "elementwise", "loop"}
                                and allocator == "greedy" and not dag):
                            meta_path = path.with_suffix(".meta.json")
                            corpus[meta_path.relative_to(root).as_posix()] = digest(meta_path.read_bytes())
                            meta = json.loads(meta_path.read_text())
                            initial = {result.stats["register_map"][n]: v for n, v in meta["inputs"].items()
                                       if n in result.stats["register_map"]}
                            row["execution"] = run_simulation(work / "out.s", initial)
                            row["expected"] = meta["expected_return"]
                        rows[key] = row
        for compact in (False, True):
            binary, assembly = work / "cnn.bin", work / "cnn.s"
            with redirect_stdout(io.StringIO()):
                rc = convert_onnx_to_riscv(str(cases[-1]), str(binary), str(assembly), const_merge=compact)
            if rc:
                raise RuntimeError(f"Standalone compilation failed: {rc}")
            rows[f"standalone|const_merge={compact}"] = {
                "compile": "passed", "binary_sha256": digest(binary.read_bytes()),
                "assembly_sha256": digest(assembly.read_bytes()), "bytes": binary.stat().st_size}
    imported = {name: str(Path(module.__file__).resolve().relative_to(root))
                for name, module in list(sys.modules.items())
                if (name == "scratchv" or name.startswith(("scratchv.", "scratchv_dag")))
                and getattr(module, "__file__", None)}
    source_hashes = {name: digest((root / path).read_bytes()) for name, path in imported.items()}
    return {"root": str(root), "corpus": corpus, "rows": rows,
            "source_sha256": source_hashes, "python": sys.version.split()[0]}


def worker(root: Path) -> dict:
    env = {**os.environ, "PYTHONPATH": str(root), "PYTHONHASHSEED": "0"}
    with tempfile.TemporaryDirectory(prefix="schedule-compare-") as directory:
        output = Path(directory) / "worker.json"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker-root", str(root),
                        "--json", str(output)], cwd=root, env=env, check=True,
                       capture_output=True, text=True, timeout=240)
        return json.loads(output.read_text())


def compare(base: dict, current: dict) -> dict:
    if base["corpus"] != current["corpus"]:
        raise ValueError("Baseline and current comparison inputs differ")
    before, after = base["rows"], current["rows"]
    mismatches = [key for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)]
    executions = [r for r in after.values() if "execution" in r]
    execution_matches = sum(r["execution"].get("success") is True
                            and r["execution"].get("return_value") == r["expected"] for r in executions)
    preexisting = [key for key, row in before.items()
                   if row.get("compile") != "passed" or row.get("encoding") == "failed"]
    return {"status": "failed" if mismatches or execution_matches != len(executions) else "passed", "cases": len(after),
            "mismatches": mismatches, "preexisting_compile_or_encoding_failures": preexisting,
            "execution_cases": len(executions),
            "execution_matches_reference": execution_matches,
            "scope": "same-input base/current with scheduling disabled; equality is not proof of baseline correctness",
            "baseline": base, "current": current}


def markdown(report: dict) -> str:
    return "\n".join([
        "# Topic 18：关闭调度的主分支对照", "", f"状态：**{report['status']}**。", "",
        f"比较 {report['cases']} 组输出，差异 {len(report['mismatches'])} 组。",
        "CompilerDriver 覆盖仓库 DSL 与 CNN、greedy/linear/naive、DAG 开关及 none/all 优化；",
        "检查汇编、可编码输出的机器码、寄存器映射和 spill/reload。standalone 比较常量合并开关下的默认汇编及完整二进制。",
        f"TinyFive 执行 {report['execution_cases']} 组标量用例，其中 {report['execution_matches_reference']} 组符合独立预期；执行状态和计数也参与版本对照。",
        f"基线已有 {len(report['preexisting_compile_or_encoding_failures'])} 组编译或编码失败，逐项保留在 JSON；一致性通过不表示这些路径正确。",
        "所有输入和实际导入源码的 SHA-256 均保存在 JSON。", "",
        *[f"- 差异：`{key}`" for key in report["mismatches"]], ""])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--json", type=Path, default=Path("benchmark_reports/schedule_disabled.json"))
    parser.add_argument("--markdown", type=Path, default=Path("benchmark_reports/schedule_disabled.md"))
    args = parser.parse_args()
    if args.worker_root:
        report = collect(args.worker_root.resolve())
    else:
        if args.baseline_root is None or args.baseline_root.resolve() == ROOT:
            parser.error("--baseline-root must select a separate base checkout")
        report = compare(worker(args.baseline_root.resolve()), worker(ROOT))
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(report), encoding="utf-8")
        print(markdown(report))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return int(report.get("status") == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
