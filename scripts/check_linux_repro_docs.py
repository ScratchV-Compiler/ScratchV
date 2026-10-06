#!/usr/bin/env python3
"""Reject host-specific reproduction instructions; optionally parse Bash blocks.

This checks documentation, not execution or numerical acceptance. Historical
platform descriptions and portable implementation details remain valid.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*([\w+-]*)\s*$")
HOST_PATTERNS = (
    (re.compile(r"(?<![\w])(?:[A-Za-z]:[\\/])(?!/)"), "personal drive path"),
    (re.compile(r"\$env:|\$LASTEXITCODE\b", re.I), "PowerShell variable"),
    (re.compile(r"\bScripts[\\/]python(?:\.exe)?\b", re.I), "non-Linux virtualenv path"),
)


def document_paths(root):
    paths = set()
    for folder in (root / "docs", root / "probes", root / "benchmarks"):
        if folder.is_dir():
            paths.update(p for p in folder.rglob("*") if p.is_file() and p.suffix in (".md", ".txt"))
    paths.update(p for p in (root / "README.md", root / "CONTRIBUTING.md") if p.is_file())
    return sorted(paths)


def inspect_document(path, *, bash=None):
    problems, blocks = [], []
    active, start, language, body = None, 0, "", []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for pattern, label in HOST_PATTERNS:
            if pattern.search(line):
                problems.append(f"{path}:{number}: {label}")
        match = FENCE.match(line)
        if match and active is None:
            active, language = match.group(1), match.group(2).lower()
            start, body = number, []
            if language in ("powershell", "pwsh", "bat", "cmd"):
                problems.append(f"{path}:{number}: use a Linux Bash example")
        elif match and active and match.group(1)[0] == active[0] and len(match.group(1)) >= len(active) and not match.group(2):
            if language in ("bash", "sh", "shell"):
                blocks.append((start, "\n".join(body) + "\n"))
            active = None
        elif active:
            body.append(line)
            if language in ("bash", "sh", "shell") and re.search(r"\b[\w.-]+\.exe\b", line, re.I):
                problems.append(f"{path}:{number}: non-Linux executable in shell example")
    if active:
        problems.append(f"{path}:{start}: unterminated fenced block")
    if bash:
        for start, content in blocks:
            try:
                result = subprocess.run([bash, "--noprofile", "--norc", "-n"], input=content,
                                        capture_output=True, text=True, encoding="utf-8", timeout=10)
            except (OSError, subprocess.TimeoutExpired) as exc:
                problems.append(f"{path}:{start}: Bash validation could not run: {exc}")
                continue
            if result.returncode:
                problems.append(f"{path}:{start}: invalid Bash: {result.stderr.strip()}")
    return problems, len(blocks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-bash", action="store_true", help="Require Bash and syntax-check every shell block")
    parser.add_argument("--bash", help="Explicit Bash executable; defaults to PATH")
    args = parser.parse_args(argv)
    bash = (args.bash or shutil.which("bash")) if args.check_bash else None
    if args.check_bash and not bash:
        parser.exit(1, "FAIL: Bash is required for --check-bash\n")
    paths = document_paths(args.root.resolve())
    if not paths:
        parser.exit(1, "FAIL: no reproduction documents found\n")
    problems, blocks = [], 0
    for path in paths:
        findings, count = inspect_document(path, bash=bash)
        problems.extend(findings)
        blocks += count
    for problem in problems:
        print(problem)
    print(f"{'FAIL' if problems else 'PASS'}: {len(paths)} documents, {blocks} shell blocks; "
          f"Bash syntax {'checked' if bash else 'not requested'}; no model execution")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
