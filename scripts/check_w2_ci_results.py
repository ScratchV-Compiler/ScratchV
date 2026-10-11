#!/usr/bin/env python3
"""Fail W2 CI acceptance if a required job failed, was skipped or is absent."""
from __future__ import annotations

import json
import os
import sys

REQUIRED_JOBS = ("llm-deploy", "full-qwen3-frontend")


def validate_results(results):
    if not isinstance(results, dict):
        raise ValueError("W2_JOB_RESULTS 必须是 JSON 对象")
    failures = []
    for name in REQUIRED_JOBS:
        job = results.get(name)
        outcome = job.get("result") if isinstance(job, dict) else None
        if outcome != "success":
            failures.append(f"{name}: {outcome or 'missing result'}")
    if failures:
        raise ValueError("以下 W2 验收 job 未成功: " + "; ".join(failures))


def main():
    print("## W2 验收\n")
    try:
        validate_results(json.loads(os.environ.get("W2_JOB_RESULTS", "{}")))
    except (ValueError, TypeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print("PASS: 小模型数值/运行时门槛与完整 28 层前端审计均成功。")
    print("\n这不证明他人复现，也不证明完整模型 IR/QEMU 推理。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
