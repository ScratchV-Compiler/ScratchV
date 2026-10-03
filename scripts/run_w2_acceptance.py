#!/usr/bin/env python3
"""Run fresh W2 acceptance gates, without installing dependencies or publishing.

Use an existing pinned Python environment and explicit local asset/tool paths.
All seven gates are required for PASS. A successful --gates subset is PARTIAL
(exit 2); failures exit 1. Reports never certify independent-person signoff,
full-model numerical execution, or a complete text-generation loop.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
GATES = ("frontend", "backend", "runtime", "runtime-model", "small-ir", "small-qemu", "full-parse")
SCRIPTS = {
    "backend": "probes/w2_backend_ops/run.py", "runtime": "probes/w2_runtime/run.py",
    "runtime-model": "probes/w2_runtime_model/run.py", "small-ir": "probes/w2_qwen3_small/run.py",
    "small-qemu": "probes/w2_qwen3_small/riscv.py", "full-parse": "probes/w2_qwen3_parse/run.py",
}
SMALL_CASES = ("full_seed_0", "full_seed_42", "one_token", "short_17", "short_255",
               "changed_future", "changed_padding")
FAMILIES = {"matmul", "elementwise", "softmax", "rmsnorm", "rope", "swiglu", "gqa"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprints(root=ROOT):
    files = {root / "pyproject.toml", root / "scripts/run_w2_acceptance.py",
             root / "scripts/check_w2_ci_results.py"}
    for directory in ("scratchv", "probes", "requirements"):
        files.update(p for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".json", ".txt"}
                     and not any(part in {"out", "output", "__pycache__"} for part in p.relative_to(root).parts))
    files.update((root / "tests").glob("test_qwen3*.py"))
    files.update((root / "tests").glob("test_w2*.py"))
    files.add(root / "tests/test_llm_inputs.py")
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files) if p.is_file()}


def checkout_evidence(root=ROOT):
    """A source archive is valid evidence too; Git absence is not a failure."""
    if not (root / ".git").exists():
        # A copied source tree may sit under another checkout's ignored output/.
        # Git's parent-directory discovery must not attribute that archive to it.
        return {"head": None, "status": "unknown", "reason": "Source directory has no .git metadata"}
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=root, capture_output=True,
                                  check=True, timeout=15).stdout
        return {"head": git("rev-parse", "HEAD").decode().strip(),
                "status": git("status", "--porcelain=v1", "--untracked-files=all").decode("utf-8", "replace"),
                "tracked_diff_sha256": hashlib.sha256(git("diff", "--binary", "HEAD", "--")).hexdigest(),
                "diff_scope": "Tracked staged and unstaged changes; source_sha256 also covers new source files"}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"head": None, "status": "unknown", "error": f"{type(exc).__name__}: {exc}"}


@dataclass(frozen=True)
class Gate:
    name: str
    command: tuple[str, ...]
    output: Path
    dependencies: tuple[str, ...] = ()


def build_plan(args, root=ROOT):
    selected = set(args.gates)
    if "small-qemu" in selected:
        selected.add("small-ir")
    if "runtime-model" in selected:
        selected.add("runtime")
    plans = []
    for name in GATES:
        if name not in selected:
            continue
        out = args.output_dir / name
        command = [args.python, "-B", "-X", "utf8"]
        if name == "frontend":
            command += ["-m", "pytest", "tests/test_qwen3_frontend_patterns.py", "-q", "-p", "no:cacheprovider",
                        "--junit-xml", str(out / "tests.xml"), "--basetemp", str(args.output_dir / "frontend-pytest-temp")]
        else:
            command += [str(root / SCRIPTS[name]), "--output-dir", str(out)]
        if name in {"backend", "small-qemu"}:
            command += ["--timeout", str(args.qemu_timeout)]
            for option in ("cc", "qemu"):
                if getattr(args, option):
                    command += ["--" + option, getattr(args, option)]
        if name in {"runtime", "runtime-model"}:
            command += ["--tokenizer-dir", str(args.tokenizer_dir)]
        if name == "runtime":
            command += ["--mode", args.tokenizer_mode]
        if name == "small-qemu":
            command += ["--model-dir", str(args.output_dir / "small-ir")]
        if name == "full-parse":
            command += ["--model-dir", str(args.model_dir), "--mode", args.model_mode,
                        "--timeout", str(args.parse_timeout)]
        dependencies = (("small-ir",) if name == "small-qemu" else
                        ("runtime",) if name == "runtime-model" else ())
        plans.append(Gate(name, tuple(command), out, dependencies))
    return plans


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_numeric_tree(value):
    """Reject contradictory nested failures and NaN/Infinity numeric evidence."""
    if isinstance(value, dict):
        if "passed" in value:
            require(value["passed"] is True, "Nested validation did not pass")
        if "max_abs" in value:
            number = value["max_abs"]
            require(type(number) in (int, float) and math.isfinite(number) and 0 <= number < 1e-5,
                    "Numeric error is missing, nonfinite or outside strict 1e-5 threshold")
        for child in value.values():
            check_numeric_tree(child)
    elif isinstance(value, list):
        for child in value:
            check_numeric_tree(child)
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite report value cannot be accepted as evidence")


def checked_rows(rows, count, names=None):
    require(isinstance(rows, list) and len(rows) == count, f"Expected {count} complete cases")
    require(all(isinstance(row, dict) and row.get("passed") is True for row in rows), "Incomplete case results")
    if names is not None:
        require([row.get("name") for row in rows] == list(names), "Missing, duplicated or reordered cases")
    return rows


def validate_gate_report(gate, sources, root=ROOT):
    if gate.name == "frontend":
        path = gate.output / "tests.xml"
        cases = ET.parse(path).getroot().findall(".//testcase")
        require(len(cases) == 22, "Frontend must execute all 22 pattern tests")
        require(len({(case.get("classname"), case.get("name")) for case in cases}) == 22,
                "Duplicated frontend test results")
        require(not any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")),
                "Frontend failures, errors and skips are not passing acceptance")
        return {"tests": len(cases), "report_sha256": sha256(path)}
    path = gate.output / "report.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    require(data.get("passed") is True and data.get("stage") == "complete",
            str(data.get("error", "Child gate did not complete successfully")))
    for name in ("report.md", "report.html"):
        require((gate.output / name).is_file(), f"Missing child report {name}")
    hashes = data.get("source_sha256", data.get("provenance", {}).get("source_sha256"))
    require(isinstance(hashes, dict) and bool(hashes), "Missing child source fingerprints")
    require(hashes.get(SCRIPTS[gate.name]) == sources.get(SCRIPTS[gate.name]), "Gate script fingerprint mismatch")
    require(all(name in sources and sources[name] == value for name, value in hashes.items()),
            "Child source fingerprints do not match this acceptance run")
    check_numeric_tree(data)
    summary = {"report_sha256": sha256(path), "child_seconds": data.get("seconds")}
    if gate.name == "backend":
        require(data.get("backend") == "ScratchV tensor-c -> RV64GC -> QEMU", "Unexpected backend gate identity")
        rows = checked_rows(data.get("cases"), 16)
        require(len({row.get("name") for row in rows}) == 16 and {row.get("family") for row in rows} == FAMILIES,
                "Missing backend families or duplicated cases")
        for row in rows:
            runs = checked_rows(row.get("executions"), 2)
            require([run.get("optimization") for run in runs] == ["none", "all"], "Incomplete optimization levels")
            for run in runs:
                require(all(run.get(key, {}).get("passed") is True for key in
                            ("ir_vs_ort", "ir_vs_numpy", "qemu_vs_ort", "qemu_vs_numpy")), "Missing backend comparisons")
                require(bool(run.get("command")) and bool(run.get("elf_sha256")), "Missing real QEMU evidence")
                require((gate.output / row["name"] / run["optimization"] / "qemu.npy").is_file(), "Missing QEMU output")
        require(data.get("completed_executions") == data.get("passed_executions") == 32, "Incomplete backend executions")
        summary.update(executions=32, qemu_seconds=data.get("qemu_process_wall_seconds", {}).get("total"))
    elif gate.name == "runtime":
        require(data.get("gate") == "unit:runtime", "Unexpected runtime gate identity")
        corpus = json.loads((root / "probes/w2_runtime/corpus.json").read_text(encoding="utf-8"))
        rows = checked_rows(data.get("tokenizer_cases"), 20, [row["name"] for row in corpus["cases"]])
        require(all(row.get("checks") == 24 for row in rows), "Incomplete tokenizer comparisons")
        checked_rows(data.get("runtime_cases"), 5)
        summary.update(corpus_cases=20, tokenizer_comparisons=480, runtime_cases=5)
    elif gate.name == "runtime-model":
        require(data.get("gate") == "integration:runtime-model", "Unexpected runtime-model gate identity")
        require(data.get("config", {}).get("vocab_size") == 151936
                and data.get("config", {}).get("num_hidden_layers") == 2, "Missing real tokenizer/model vocabulary contract")
        require(data.get("completed_model_invocations") == 8, "Incomplete runtime-model forward invocations")
        rows = checked_rows(data.get("cases"), 2, ("chinese", "special_pad_in_prompt"))
        require(rows[1].get("pad_id_inside_prompt") is True, "Missing in-prompt PAD test")
        for row in rows:
            require(row.get("torch", {}).get("contract", {}).get("passed") is True
                    and row.get("torch", {}).get("sampling", {}).get("passed") is True,
                    "Missing independent model and sampling reference")
            executions = checked_rows(row.get("executions"), 3)
            require([run.get("engine") for run in executions] == ["ort", "ir-none", "ir-all"],
                    "Missing runtime-model execution engines")
            for run in executions:
                require(run.get("vs_torch", {}).get("passed") is True
                        and run.get("sampling", {}).get("passed") is True, "Missing runtime-model comparisons")
                if run["engine"].startswith("ir-"):
                    require(run.get("vs_ort", {}).get("passed") is True, "Missing IR/ORT comparison")
        summary.update(cases=2, executions=6)
    elif gate.name == "small-ir":
        require(data.get("config", {}).get("num_hidden_layers") == 2, "Expected real two-layer configuration")
        checked_rows(data.get("cases"), 7, SMALL_CASES)
        checked_rows(data.get("invariants"), 6)
        for filename, field in (("model.onnx", "model_sha256"), ("diagnostics.onnx", "diagnostics_sha256")):
            require(sha256(gate.output / filename) == data.get("onnx", {}).get(field), "Export artifact hash mismatch")
        require((gate.output / "checkpoints.json").is_file(), "Missing diagnostic checkpoint schema")
        summary["cases"] = 7
    elif gate.name == "small-qemu":
        rows = checked_rows(data.get("cases"), 7, SMALL_CASES)
        groups = [(graph, level) for graph in ("normal", "diagnostic") for level in ("none", "all")]
        for row in rows:
            runs = checked_rows(row.get("qemu"), 4)
            require([(run.get("graph"), run.get("optimization")) for run in runs] == groups, "Missing QEMU graph/level")
            require(all(run.get("status") == "success" and run.get("command") for run in runs), "Missing QEMU execution")
            for run in runs:
                require((gate.output / "runs" / row["name"] / f"{run['graph']}-{run['optimization']}" / "qemu.npy").is_file(),
                        "Missing two-layer QEMU output")
        timing = data.get("timing", {})
        require(timing.get("completed_count") == timing.get("status_counts", {}).get("success") == 28,
                "Incomplete two-layer QEMU executions")
        require(data.get("export_evidence", {}).get("report_sha256") == sha256(gate.output.parent / "small-ir/report.json"),
                "QEMU did not use the fresh export from this run")
        summary.update(executions=28, qemu_seconds=timing.get("qemu_process_wall_seconds_total"))
    elif gate.name == "full-parse":
        require(data.get("gate") == "frontend:parse-qwen3" and data.get("schema_version") == 1,
                "Unexpected full parse gate identity")
        require(data.get("graph", {}).get("config", {}).get("num_hidden_layers") == 28, "Missing complete model configuration")
        layers = checked_rows(data.get("graph", {}).get("layers"), 28)
        require([row.get("layer") for row in layers] == list(range(28)), "Incomplete 28-layer chain")
        require(data.get("audit", {}).get("passed") is True and data.get("ir_verifier", {}).get("passed") is True,
                "Missing binding audit or IR verification")
        for name in ("nodes.json", "bindings.json", "ir.txt"):
            require(sha256(gate.output / name) == data.get("evidence", {}).get(name, {}).get("sha256"),
                    f"Missing or altered parse evidence: {name}")
        summary["layers"] = 28
    return summary


def terminate_process_tree(process):
    """The gates spawn compilers/QEMU/workers; stop owned descendants on timeout."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=20, check=True)
        else:
            try:
                # Let the probe unwind _run_process and stop its separate
                # compiler/QEMU/worker session before the final group kill.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=20)


def child_failure(row, gate, log):
    try:
        child = json.loads((gate.output / "report.json").read_text(encoding="utf-8"))
        row.update(child_stage=child.get("stage"), child_error=child.get("error"))
    except (OSError, ValueError):
        pass
    return row.get("child_error") or f"Child exited {row['returncode']}; inspect {log.name}"


def run_gate(gate, sources, *, timeout, root=ROOT):
    started = time.perf_counter()
    row = {"name": gate.name, "status": "FAIL", "passed": False, "command": list(gate.command),
           "output": str(gate.output), "timeout_seconds": timeout}
    log = gate.output.parent / "logs" / (gate.name + ".log")
    row["log"] = str(log)
    try:
        require(not gate.output.exists(), "Gate output already exists; cannot reuse previous evidence")
        log.parent.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.update(PYTHONHASHSEED="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        with log.open("xb") as stream:
            process = subprocess.Popen(gate.command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                       env=env, **options)
            try:
                row["returncode"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                row["error"] = f"Gate exceeded {timeout} seconds"
                try:
                    terminate_process_tree(process)
                except (OSError, subprocess.SubprocessError) as exc:
                    row["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                raise
            if row["returncode"] != 0:
                # Save the primary failure before closing the log; close itself
                # can fail and must not replace a child model/tool error.
                row["error"] = child_failure(row, gate, log)
                raise ValueError(row["error"])
        row["validation"] = validate_gate_report(gate, sources, root)
        row.update(status="PASS", passed=True)
    except Exception as exc:
        row.setdefault("error", f"{type(exc).__name__}: {exc}")
    row["seconds"] = time.perf_counter() - started
    return row


def finalize_status(report):
    if report.get("error") or any(row["status"] in {"FAIL", "BLOCKED"} for row in report["gates"]):
        report.update(status="FAIL", passed=False)
    elif all(row["status"] == "PASS" for row in report["gates"]) and len(report["gates"]) == len(GATES):
        report.update(status="PASS", passed=True)
    else:
        report.update(status="PARTIAL", passed=False)


def write_reports(out, report):
    errors = []

    def fail(name, exc):
        errors.append(f"{name}: {type(exc).__name__}: {exc}")
        report.setdefault("error", "Could not preserve complete W2 acceptance evidence")
        report.update(passed=False, status="FAIL", report_write_errors=list(errors))

    def write(name, text):
        temporary = out / ("." + name + ".tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(out / name)
            return True
        except OSError as exc:
            fail(name, exc)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def views():
        lines = ["# W2 local acceptance", "", f"Result: **{report['status']}**", "",
                 "Fresh executions only. Subsets and blocked gates cannot certify all W2 technical gates.", "",
                 "| Gate | Result | Wall seconds | Reason |", "|---|---|---:|---|"]
        for row in report["gates"]:
            reason = str(row.get("error", "")).replace("|", "/").replace("\n", " ")
            lines.append(f"| {row['name']} | {row['status']} | {row.get('seconds', 0):.3f} | {reason} |")
        if report.get("error"):
            lines += ["", "Error: " + report["error"]]
        lines += ["", "Source hashes and checkout metadata: report.json. Per-gate logs: logs/.",
                  "This does not establish team signoff, full-model IR/QEMU correctness or text generation."]
        markdown = "\n".join(lines) + "\n"
        return {"report.md": markdown, "report.html": '<!doctype html><meta charset="utf-8"><title>W2 acceptance</title>'
                '<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:32px auto}pre{white-space:pre-wrap}</style><pre>'
                + escape(markdown) + "</pre>"}

    for name, value in views().items():
        write(name, value)
    if errors:
        for name, value in views().items():
            write(name, value)
    if not write("report.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"):
        for name, value in views().items():
            write(name, value)
        write("report.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if errors:
        print("; ".join(errors), file=sys.stderr)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--python", default=sys.executable, help="Existing pinned Python interpreter; no installation")
    cli.add_argument("--output-dir", type=Path, required=True, help="Must not exist, even if empty")
    cli.add_argument("--tokenizer-dir", type=Path, help="Verified local official tokenizer assets")
    cli.add_argument("--tokenizer-mode", choices=("verify", "download"), default="verify")
    cli.add_argument("--model-dir", type=Path, help="Pinned full Qwen3 ONNX release directory")
    cli.add_argument("--model-mode", choices=("verify", "download"), default="verify")
    cli.add_argument("--cc", help="RISC-V compiler / Zig executable")
    cli.add_argument("--qemu", help="qemu-system-riscv64 executable")
    cli.add_argument("--gate-timeout", type=float, default=1800, help="Wall seconds per gate including child workers")
    cli.add_argument("--qemu-timeout", type=float, default=180, help="Seconds per QEMU execution")
    cli.add_argument("--parse-timeout", type=float, default=300, help="Full frontend worker timeout")
    cli.add_argument("--gates", nargs="+", choices=GATES, default=list(GATES),
                     help="Optional subset: dependencies auto-added; successful subset exits 2 (PARTIAL)")
    args = cli.parse_args(argv)
    for field in ("gate_timeout", "qemu_timeout", "parse_timeout"):
        if not math.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
            cli.error("Timeouts must be finite and positive")
    if len(set(args.gates)) != len(args.gates):
        cli.error("Duplicate gates are not allowed")
    args.output_dir = args.output_dir.resolve()
    for field in ("tokenizer_dir", "model_dir"):
        if getattr(args, field) is not None:
            setattr(args, field, getattr(args, field).resolve())
    if args.output_dir.exists():
        cli.error("--output-dir must not exist; preserve previous evidence")
    args.output_dir.mkdir(parents=True)
    report = {"schema_version": 1, "gate": "acceptance:w2", "status": "FAIL", "passed": False,
              "created_at": datetime.now(timezone.utc).isoformat(), "gates": [], "requested_gates": args.gates,
              "python": args.python, "checkout": checkout_evidence()}
    started = time.perf_counter()
    try:
        report["source_sha256"] = source_fingerprints()
        plan = {gate.name: gate for gate in build_plan(args)}
        states = {}
        for name in GATES:
            gate = plan.get(name)
            missing = ("--tokenizer-dir" if name in {"runtime", "runtime-model"} and args.tokenizer_dir is None else
                       "--model-dir" if name == "full-parse" and args.model_dir is None else None)
            if gate is None:
                row = {"name": name, "status": "NOT_RUN", "passed": False, "error": "Not selected"}
            elif missing or any(states.get(dependency) != "PASS" for dependency in gate.dependencies):
                row = {"name": name, "status": "BLOCKED", "passed": False,
                       "error": "Missing " + missing if missing else "Required preceding gate did not pass"}
            else:
                print(f"[W2] Running {name}", flush=True)
                row = run_gate(gate, report["source_sha256"], timeout=args.gate_timeout)
            report["gates"].append(row)
            states[name] = row["status"]
            print(f"[W2] {name}: {row['status']}", flush=True)
        final_sources = source_fingerprints()
        if final_sources != report["source_sha256"]:
            report["error"] = "Source files changed while acceptance was running; rerun after edits finish"
            report["final_source_sha256"] = final_sources
    except Exception as exc:
        report.setdefault("error", f"{type(exc).__name__}: {exc}")
    report["seconds"] = time.perf_counter() - started
    finalize_status(report)
    write_reports(args.output_dir, report)
    print(f"[W2] {report['status']}: {args.output_dir / 'report.json'}", flush=True)
    return 0 if report["passed"] else 2 if report["status"] == "PARTIAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
