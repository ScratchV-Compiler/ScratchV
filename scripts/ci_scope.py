#!/usr/bin/env python3
"""Select optional PR checks from the complete, local merge-base diff."""
from __future__ import annotations

from fnmatch import fnmatchcase
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import subprocess
import sys


SCOPES = ("llm", "docs", "w4")
SHARED_PATHS = (".github/workflows/**", "scripts/ci_*", "scripts/summarize_ci.py", "tests/test_ci_*")
SHA = re.compile(r"[0-9a-fA-F]{40}\Z")
ROOT = Path(__file__).resolve().parents[1]


def read_config(path: Path) -> dict[str, list[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != set(SCOPES):
        raise ValueError("scope configuration must define llm, docs and w4")
    for patterns in value.values():
        if not isinstance(patterns, list) or any(
            not isinstance(pattern, str) or not pattern for pattern in patterns
        ):
            raise ValueError("scope configuration values must be lists of nonempty patterns")
    return value


def pr_commits(event_path: Path) -> tuple[str, str]:
    event = json.loads(event_path.read_text(encoding="utf-8"))
    try:
        base = event["pull_request"]["base"]["sha"]
        head = event["pull_request"]["head"]["sha"]
    except (KeyError, TypeError):
        raise ValueError("pull request event must contain base and head SHA") from None
    if any(not isinstance(value, str) or not SHA.fullmatch(value) for value in (base, head)):
        raise ValueError("pull request SHAs must each contain exactly 40 hexadecimal characters")
    return base, head


def changed_paths(root: Path, base: str, head: str) -> list[str]:
    # A rename out of a scope still changes its old path. Disable rename folding
    # so both the removed and added names participate in routing.
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--no-textconv", "--no-renames",
         "--name-only", "-z", f"{base}...{head}", "--"],
        cwd=root, capture_output=True, check=True, timeout=60,
    )
    if result.stdout and not result.stdout.endswith(b"\0"):
        raise ValueError("git did not produce a complete NUL-delimited path list")
    return [value.decode("utf-8", errors="surrogateescape")
            for value in result.stdout.split(b"\0") if value]


def matches_path(path: str, pattern: str) -> bool:
    """Match path segments: * stays in one segment, ** spans zero or more."""
    segments, patterns = path.split("/"), pattern.split("/")

    @lru_cache(maxsize=None)
    def match(position: int, pattern_position: int) -> bool:
        if pattern_position == len(patterns):
            return position == len(segments)
        current = patterns[pattern_position]
        if current == "**":
            return match(position, pattern_position + 1) or (
                position < len(segments) and match(position + 1, pattern_position)
            )
        return (position < len(segments) and fnmatchcase(segments[position], current)
                and match(position + 1, pattern_position + 1))

    return match(0, 0)


def select_scopes(config: dict[str, list[str]], paths: list[str]) -> dict[str, bool]:
    shared = any(matches_path(path, pattern) for path in paths for pattern in SHARED_PATHS)
    return {
        name: bool(patterns) and (shared or any(
            matches_path(path, pattern) for path in paths for pattern in patterns
        ))
        for name, patterns in config.items()
    }


def detect_scopes(root: Path, event_name: str, event_path: Path | None,
                  config_path: Path) -> tuple[dict[str, bool], int]:
    config = read_config(config_path)
    if event_name not in {"pull_request", "push", "schedule", "workflow_dispatch"}:
        raise ValueError("unsupported or missing GitHub event name")
    if event_name != "pull_request":
        return dict.fromkeys(SCOPES, False), 0
    if event_path is None:
        raise ValueError("GITHUB_EVENT_PATH is required for pull requests")
    base, head = pr_commits(event_path)
    paths = changed_paths(root, base, head)
    return select_scopes(config, paths), len(paths)


def main() -> int:
    try:
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        scopes, count = detect_scopes(
            ROOT, os.environ.get("GITHUB_EVENT_NAME", ""),
            Path(event_path) if event_path else None, ROOT / "scripts/ci_scopes.json",
        )
        destination = os.environ.get("GITHUB_OUTPUT")
        if not destination:
            raise ValueError("GITHUB_OUTPUT is required")
        lines = "".join(f"{name}={str(scopes[name]).lower()}\n" for name in SCOPES)
        with open(destination, "a", encoding="utf-8") as stream:
            stream.write(lines)
        print(f"Changed path count: {count}")
        print(lines, end="")
        return 0
    except (OSError, ValueError, subprocess.SubprocessError):
        # Event contents, Git stderr and filenames may be attacker-controlled.
        # Keep the public diagnostic bounded and do not echo those values.
        print("CI scope detection failed: check configuration, event SHAs and fetched Git history.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
