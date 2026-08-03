#!/usr/bin/env python3
"""Run all 3 register allocation benchmarks and produce a report."""

import argparse
import datetime
import json
import os
import sys
import time


from benchmarks.test_regalloc import bench_simple, bench_dense, bench_cnn


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _make_html(results: dict, total_time: float) -> str:
    """Generate an HTML report."""
    rows = ""
    for name, r in results.items():
        if not isinstance(r, dict):
            continue
        v = "✓" if r.get("valid", True) else "✗"
        c = "#22863a" if r.get("valid", True) else "#cb2431"
        ms = f"{r.get('mean_s', 0) * 1000:.3f}"
        sd = f"{r.get('stdev_s', 0) * 1000:.3f}"
        rows += (
            f"<tr><td>{name}</td><td>{ms}</td><td>{sd}</td>"
            f"<td>{r.get('vreg_count', '-')}</td>"
            f"<td>{r.get('reg_spill_count', r.get('spills', '-'))}</td>"
            f"<td>{r.get('peak_active', '-')}</td>"
            f"<td>{r.get('reloads', '-')}</td>"
            f"<td>{r.get('asm_lines', '-')}</td>"
            f"<td style='color:{c}'>{v}</td></tr>\n"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Register Allocation Benchmark Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         max-width: 1000px; margin: 40px auto; padding: 0 20px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
  th {{ background: #f6f8fa; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f6f8fa; }}
</style>
</head>
<body>
<h1>Register Allocation Benchmark Report</h1>
<p>Generated: {datetime.datetime.now().isoformat()} | Total: {total_time * 1000:.1f}ms</p>
<table>
<tr><th>Benchmark</th><th>Mean(ms)</th><th>Std(ms)</th><th>Vregs</th>
<th>Spills</th><th>Peak</th><th>Reloads</th><th>Asm</th><th>Valid</th></tr>
{rows}
</table>
</body>
</html>"""


def _make_markdown(results: dict) -> str:
    """Generate a Markdown report."""
    lines = [
        "# Register Allocation Benchmark Report",
        "",
        f"**Generated**: {datetime.datetime.now().isoformat()}",
        "",
        "| Benchmark | Mean(ms) | Std(ms) | Vregs | Spills | Peak | "
        "Reloads | Asm | Valid |",
        "|-----------|----------|---------|-------|--------|------|"
        "---------|-----|-------|",
    ]
    for name, r in results.items():
        if not isinstance(r, dict):
            continue
        ms = f"{r.get('mean_s', 0) * 1000:.3f}"
        sd = f"{r.get('stdev_s', 0) * 1000:.3f}"
        v = "✓" if r.get("valid", True) else "✗"
        lines.append(
            f"| {name} | {ms} | {sd} | {r.get('vreg_count', '-')} | "
            f"{r.get('reg_spill_count', r.get('spills', '-'))} | "
            f"{r.get('peak_active', '-')} | "
            f"{r.get('reloads', '-')} | {r.get('asm_lines', '-')} | "
            f"{v} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Register Allocation Benchmark Suite")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-html", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    print("=" * 60)
    print("  ScratchV — Register Allocation Benchmark Suite")
    print("=" * 60)

    t0 = time.perf_counter()
    results: dict = {}

    # Benchmark 1 — Simple (no-spill)
    r1 = bench_simple.run_bench(repeats=args.repeats)
    results["1. Simple Arithmetic"] = r1
    print(
        f"  1. Simple:  reg_spill_count={r1['reg_spill_count']}, "
        f"mean={r1['mean_s'] * 1000:.3f}ms  "
        f"{'✓' if r1.get('valid') else '✗'}"
    )

    # Benchmark 2 — Dense (spill)
    r2 = bench_dense.run_bench(repeats=args.repeats)
    results["2. Dense Computation"] = r2
    print(
        f"  2. Dense:   reg_spill_count={r2['reg_spill_count']}, "
        f"mean={r2['mean_s'] * 1000:.3f}ms  "
        f"{'✓' if r2.get('valid') else '✗'}"
    )

    # Benchmark 3 — CNN Integration
    cnn_default = "models/graph/cnn.onnx"
    r3 = bench_cnn.run_bench(cnn_path=cnn_default, repeats=args.repeats)
    results["3. CNN Integration"] = r3
    print(
        f"  3. CNN:     reg_spill_count={r3['reg_spill_count']}, "
        f"mean={r3['mean_s'] * 1000:.3f}ms  "
        f"{'✓' if r3.get('valid') else '✗'}"
    )

    total_time = time.perf_counter() - t0

    # Summary
    all_ok = all(r.get("valid", True) for r in results.values() if isinstance(r, dict))
    print(f"\n  Total: {total_time * 1000:.1f}ms  {'PASS' if all_ok else 'HAD ERRORS'}")

    # Reports
    if args.output_json:
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_time_s": total_time,
            "repeats": args.repeats,
            "results": {
                name: {
                    k: v
                    for k, v in r.items()
                    if not k.startswith("_") and k != "asm_errors"
                }
                for name, r in results.items()
                if isinstance(r, dict)
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  JSON report: {args.output_json}")

    if args.output_html:
        with open(args.output_html, "w") as f:
            f.write(_make_html(results, total_time))
        print(f"  HTML report: {args.output_html}")

    if args.output_md:
        with open(args.output_md, "w") as f:
            f.write(_make_markdown(results))
        print(f"  Markdown:    {args.output_md}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)

