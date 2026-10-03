#!/usr/bin/env python3
"""Fail W2 CI acceptance if a required job failed, was skipped or is absent."""
from __future__ import annotations

import json
import os
import sys

REQUIRED_JOBS = ("llm-deploy", "full-qwen3-frontend")


def validate_results(results):
    if not isinstance(results, dict):
        raise ValueError("W2_JOB_RESULTS must contain a JSON object")
    failures = []
    for name in REQUIRED_JOBS:
        job = results.get(name)
        outcome = job.get("result") if isinstance(job, dict) else None
        if outcome != "success":
            failures.append(f"{name}: {outcome or 'missing result'}")
    if failures:
        raise ValueError("Required W2 jobs did not succeed: " + "; ".join(failures))


def main():
    print("## W2 required acceptance\n")
    try:
        validate_results(json.loads(os.environ.get("W2_JOB_RESULTS", "{}")))
    except (ValueError, TypeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print("PASS: small numerical/runtime gates and complete 28-layer frontend audit succeeded.")
    print("\nThis does not certify second-person reproduction or full-model IR/QEMU inference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
