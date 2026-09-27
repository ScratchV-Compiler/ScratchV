"""LLVM MCA is the sole timing authority for the RV32 scheduling pass.

No latency table, bypass approximation or resource simulator is maintained here.
The adapter validates LLVM output and counts events in its issue timeline.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache
import json
import os
import re
import shutil
import subprocess

CPU = "sifive-e76"
FLAGS = ("-mtriple=riscv32", f"-mcpu={CPU}", "-mattr=+m,+f,+d,-c")
OPTIONS = ("-iterations=1", "-noalias=false", "-timeline",
           "-timeline-max-cycles=0", "-timeline-max-iterations=1", "-json")
CRITICAL_PATH_REASON = "Critical-path analysis is not collected by this adapter for in-order sifive-e76"
_VERSION_RE = re.compile(r"\bLLVM version (\d+\.\d+\.\d+)\b")
_RETURN_WARNING = (
    "warning: found a return instruction in the input assembly sequence.\n"
    "note: program counter updates are ignored."
)


class LLVMError(RuntimeError):
    """A tool failure must not silently select a substitute timing model."""


def executable_path(executable: str | None = None) -> str:
    requested = executable or os.environ.get("LLVM_MCA", "llvm-mca")
    found = shutil.which(requested)
    if not found:
        raise LLVMError(f"llvm-mca not found: {requested}; install LLVM or set LLVM_MCA")
    return os.path.realpath(found)


def _identity(executable: str) -> tuple:
    stat = os.stat(executable)
    return executable, stat.st_size, stat.st_mtime_ns


def _run(command: list[str], source: str | None = None) -> str:
    try:
        result = subprocess.run(command, input=source, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LLVMError(f"llvm-mca execution failed: {exc}") from exc
    # PC updates are intentionally ignored for static regions with fixed
    # terminators. Reject every other diagnostic, including partial parsing.
    diagnostic = result.stderr.strip()
    expected_warning = source is not None and diagnostic == _RETURN_WARNING
    if result.returncode or (diagnostic and not expected_warning):
        raise LLVMError(f"llvm-mca failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


@lru_cache(maxsize=8)
def _version(identity: tuple) -> str:
    lines = _run([identity[0], "--version"]).splitlines()
    version = next((line.strip() for line in lines if _VERSION_RE.search(line)),
                   lines[0] if lines else "(empty version output)")
    if not _VERSION_RE.search(version):
        raise LLVMError(f"Unrecognized LLVM version output: {version}")
    return version


def check_version(executable: str | None = None) -> str:
    return _version(_identity(executable_path(executable)))


def model_metadata(executable: str | None = None) -> dict:
    path = executable_path(executable)
    output = check_version(path)
    version = _VERSION_RE.search(output).group(1)
    source_url = (f"https://github.com/llvm/llvm-project/blob/llvmorg-{version}/"
                  "llvm/lib/Target/RISCV/RISCVSchedSiFive7.td")
    return {"name": f"llvm-mca/{CPU}", "cpu": CPU, "llvm_version": version,
            "version_output": output, "executable": path,
            "flags": list(FLAGS), "options": list(OPTIONS), "source_url": source_url,
            "critical_path_status": "unavailable", "critical_path_reason": CRITICAL_PATH_REASON}


def canonical_source(instructions) -> str:
    lines = []
    for inst in instructions:
        if inst.effects.barrier_reason:
            raise LLVMError(f"Unsupported instruction: {inst.raw_line or inst.opcode}")
        args = list(inst.operands)
        if inst.effects.memory != "none":
            value = args[1] if "(" in args[0] else args[0]
            address = inst.effects.address
            args = [value, f"{address.offset}({address.base})"]
        if inst.effects.target:
            args[-1] = ".Lmca_target"
        lines.append(inst.opcode + " " + ", ".join(args))
    return "\n".join(lines) + "\n.Lmca_target:\n"


@lru_cache(maxsize=512)
def _analyze(identity: tuple, source: str) -> dict:
    _version(identity)
    try:
        return json.loads(_run([identity[0], *FLAGS, *OPTIONS], source))
    except (ValueError, IndexError) as exc:
        raise LLVMError("Invalid llvm-mca JSON output") from exc


def metrics(instructions, executable: str | None = None) -> dict:
    """Measure one region once; retain the complete LLVM JSON for reproduction."""
    if not instructions:
        raise LLVMError("Cannot analyze an empty region")
    source = canonical_source(instructions)
    raw = _analyze(_identity(executable_path(executable)), source)
    try:
        if len(raw["CodeRegions"]) != 1 or raw["TargetInfo"]["CPUName"] != CPU:
            raise ValueError("Unexpected target or region count")
        region = raw["CodeRegions"][0]
        summary = region["SummaryView"]
        timeline = region["TimelineView"]["TimelineInfo"]
        n = len(instructions)
        if len(timeline) != n or summary["TotaluOps"] != n or summary["Instructions"] != n:
            raise ValueError("LLVM expanded or omitted instructions in a modeled region")
        issues = [row["CycleIssued"] for row in timeline]
        if any(row["CycleIssued"] < 0 or row["CycleExecuted"] < row["CycleIssued"] for row in timeline):
            raise ValueError("Incomplete LLVM execution timeline")
        start = min(issues)
        cycles = max(max(row["CycleExecuted"], row["CycleIssued"] + 1) for row in timeline) - start
        counts = Counter(issues)
        span = max(issues) - start + 1
        bubbles = span - len(counts)
        return {
            "cycles": cycles, "peak_parallelism": max(counts.values()),
            "critical_path": None, "critical_path_reason": CRITICAL_PATH_REASON,
            "bubbles": bubbles, "bubble_ratio": bubbles / span,
            "issue_span": span, "drain_cycles": cycles - span,
            "dispatch_width": summary["DispatchWidth"], "ipc": n / cycles,
            "issue_cycles": [t - start for t in issues],
            "llvm_total_cycles": summary["TotalCycles"], "source": source,
            "llvm_output": deepcopy(raw),
        }
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLVMError(f"Invalid llvm-mca analysis: {exc}") from exc
