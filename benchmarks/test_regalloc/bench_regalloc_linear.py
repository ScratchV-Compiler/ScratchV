#!/usr/bin/env python3
"""Run all register allocation benchmarks and produce a report."""

import argparse
import datetime
import json
import os
import sys
import time


from benchmarks.test_regalloc import bench_cnn, bench_dense, bench_pseudo, bench_simple


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _optimization_comparison(results: dict) -> dict | None:
    cnn = results.get("3. CNN Integration", {})
    if not isinstance(cnn, dict):
        return None
    comparison = cnn.get("optimization_comparison")
    return comparison if isinstance(comparison, dict) else None


def _format_comparison_value(metric: dict) -> tuple[str, str]:
    unit = metric.get("unit", "")

    def format_one(value: float) -> str:
        if unit == "ms":
            return f"{value:.3f} ms"
        return f"{value:g} {unit}".strip()

    return format_one(metric["before"]), format_one(metric["after"])


def _format_improvement(value: float | None) -> str:
    if value is None:
        return "n/a (zero baseline)"
    if value > 0:
        return f"{value:.2f}% better"
    if value < 0:
        return f"{abs(value):.2f}% worse"
    return "unchanged"


def _comparison_markdown(results: dict) -> list[str]:
    comparison = _optimization_comparison(results)
    if comparison is None:
        return []

    baseline = comparison["baseline"]
    optimized = comparison["optimized"]
    lines = [
        "",
        "## CNN allocator comparison: Greedy vs LinearScan",
        "",
        (
            "Both allocators receive the same selected Machine IR in the same "
            "process. Lower is better for every metric below."
        ),
        "",
        f"| Metric | Before ({baseline}) | After ({optimized}) | Optimization |",
        "|---|---:|---:|---:|",
    ]
    for metric in comparison["metrics"].values():
        before, after = _format_comparison_value(metric)
        lines.append(
            f"| {metric['label']} | {before} | {after} | "
            f"{_format_improvement(metric['improvement_pct'])} |"
        )

    correctness = comparison["correctness"]
    before_status = "PASS" if correctness["before"] else "FAIL"
    after_status = "PASS" if correctness["after"] else "FAIL"
    lines.extend(
        [
            "",
            f"Correctness (assembly + emulator): **{before_status} -> "
            f"{after_status}**.",
        ]
    )
    return lines


def _pseudo_detail_markdown(results: dict) -> list[str]:
    """Render instruction-level pseudo semantics and pressure metrics."""

    pseudo = results.get("4. Pseudo Instructions", {})
    cases = pseudo.get("cases", []) if isinstance(pseudo, dict) else []
    if not cases:
        return []

    lines = [
        "",
        "## Pseudo-instruction detail",
        "",
        (
            "`Before` reproduces the legacy positional `dst/src` inference; "
            "`After` uses `machine_semantics.py`. The pressure columns use "
            "a two-register bank."
        ),
        "",
        "| Pseudo | Before def/use | After def/use | RV32 instructions | "
        "Pressure before/after | Spill stores before/after | "
        "Pressure case before/after | Execute |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for case in cases:
        sweep = case.get("pressure_sweep", [])
        pressure = next(
            (
                point for point in sweep
                if point.get("physical_register_count") == 2
            ),
            None,
        )
        before = pressure.get("before", {}) if pressure else {}
        after = pressure.get("after", {}) if pressure else {}

        def fmt_semantics(prefix: str) -> str:
            defs = ",".join(case.get(f"{prefix}_defs", [])) or "-"
            uses = ",".join(case.get(f"{prefix}_uses", [])) or "-"
            return f"{defs} / {uses}"

        execute = "PASS" if case.get("valid") else "FAIL"
        pressure_valid = (
            f"{'PASS' if before.get('valid') else 'FAIL'} / "
            f"{'PASS' if after.get('valid') else 'FAIL'}"
            if pressure else "-"
        )
        lines.append(
            f"| `{case['name']}` | {fmt_semantics('legacy')} | "
            f"{fmt_semantics('semantic')} | "
            f"{case.get('expanded_rv32_instructions', '-')} | "
            f"{before.get('pressure_peak', '-')} / "
            f"{after.get('pressure_peak', '-')} | "
            f"{before.get('spill_stores', '-')} / "
            f"{after.get('spill_stores', '-')} | {pressure_valid} | "
            f"{execute} |"
        )
    return lines


def _comparison_html(results: dict) -> str:
    comparison = _optimization_comparison(results)
    if comparison is None:
        return ""

    rows = []
    for metric in comparison["metrics"].values():
        before, after = _format_comparison_value(metric)
        rows.append(
            f"<tr><td>{metric['label']}</td><td>{before}</td>"
            f"<td>{after}</td><td>"
            f"{_format_improvement(metric['improvement_pct'])}</td></tr>"
        )
    correctness = comparison["correctness"]
    before_status = "PASS" if correctness["before"] else "FAIL"
    after_status = "PASS" if correctness["after"] else "FAIL"
    return f"""
<h2>CNN allocator comparison: Greedy vs LinearScan</h2>
<p>Both allocators receive the same selected Machine IR in the same process.
Lower is better for every metric below.</p>
<table>
<tr><th>Metric</th><th>Before ({comparison['baseline']})</th>
<th>After ({comparison['optimized']})</th><th>Optimization</th></tr>
{''.join(rows)}
</table>
<p>Correctness (assembly + emulator): <strong>{before_status} &rarr;
{after_status}</strong>.</p>"""


def _pseudo_detail_html(results: dict) -> str:
    """Render the instruction-level pseudo table for the HTML artifact."""

    pseudo = results.get("4. Pseudo Instructions", {})
    cases = pseudo.get("cases", []) if isinstance(pseudo, dict) else []
    if not cases:
        return ""

    rows = []
    for case in cases:
        sweep = case.get("pressure_sweep", [])
        point = next(
            (
                item for item in sweep
                if item.get("physical_register_count") == 2
            ),
            {},
        )
        before = point.get("before", {})
        after = point.get("after", {})
        legacy_defs = ",".join(case.get("legacy_defs", [])) or "-"
        legacy_uses = ",".join(case.get("legacy_uses", [])) or "-"
        semantic_defs = ",".join(case.get("semantic_defs", [])) or "-"
        semantic_uses = ",".join(case.get("semantic_uses", [])) or "-"
        execute = "PASS" if case.get("valid") else "FAIL"
        pressure_valid = (
            f"{'PASS' if before.get('valid') else 'FAIL'} / "
            f"{'PASS' if after.get('valid') else 'FAIL'}"
            if point else "-"
        )
        rows.append(
            f"<tr><td><code>{case['name']}</code></td>"
            f"<td>{legacy_defs} / {legacy_uses}</td>"
            f"<td>{semantic_defs} / {semantic_uses}</td>"
            f"<td>{case.get('expanded_rv32_instructions', '-')}</td>"
            f"<td>{before.get('pressure_peak', '-')} / "
            f"{after.get('pressure_peak', '-')}</td>"
            f"<td>{before.get('spill_stores', '-')} / "
            f"{after.get('spill_stores', '-')}</td>"
            f"<td>{pressure_valid}</td><td>{execute}</td></tr>"
        )
    return f"""
<h2>Pseudo-instruction detail</h2>
<p><code>Before</code> uses legacy positional inference; <code>After</code>
uses <code>machine_semantics.py</code>. Pressure metrics use two registers.</p>
<table>
<tr><th>Pseudo</th><th>Before def/use</th><th>After def/use</th>
<th>RV32 instructions</th><th>Pressure before/after</th>
<th>Spill stores before/after</th><th>Pressure case before/after</th>
<th>Execute</th></tr>
{''.join(rows)}
</table>"""


def _make_html(results: dict, total_time: float) -> str:
    """Generate an HTML report."""
    rows = ""
    for name, r in results.items():
        if not isinstance(r, dict):
            continue
        v = "PASS" if r.get("valid", True) else "FAIL"
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
    comparison_html = _comparison_html(results)
    pseudo_detail_html = _pseudo_detail_html(results)

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
{comparison_html}
{pseudo_detail_html}
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
        v = "PASS" if r.get("valid", True) else "FAIL"
        lines.append(
            f"| {name} | {ms} | {sd} | {r.get('vreg_count', '-')} | "
            f"{r.get('reg_spill_count', r.get('spills', '-'))} | "
            f"{r.get('peak_active', '-')} | "
            f"{r.get('reloads', '-')} | {r.get('asm_lines', '-')} | "
            f"{v} |"
        )
    lines.extend(_comparison_markdown(results))
    lines.extend(_pseudo_detail_markdown(results))
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
    print("  ScratchV - Register Allocation Benchmark Suite")
    print("=" * 60)

    t0 = time.perf_counter()
    results: dict = {}

    # Benchmark 1 — Simple (no-spill)
    r1 = bench_simple.run_bench(repeats=args.repeats)
    results["1. Simple Arithmetic"] = r1
    print(
        f"  1. Simple:  reg_spill_count={r1['reg_spill_count']}, "
        f"mean={r1['mean_s'] * 1000:.3f}ms  "
        f"{'PASS' if r1.get('valid') else 'FAIL'}"
    )

    # Benchmark 2 — Dense (spill)
    r2 = bench_dense.run_bench(repeats=args.repeats)
    results["2. Dense Computation"] = r2
    print(
        f"  2. Dense:   reg_spill_count={r2['reg_spill_count']}, "
        f"mean={r2['mean_s'] * 1000:.3f}ms  "
        f"{'PASS' if r2.get('valid') else 'FAIL'}"
    )

    # Benchmark 3 — CNN Integration And Comparation With LLVM
    cnn_default = "models/graph/cnn.onnx"
    r3 = bench_cnn.run_bench(cnn_path=cnn_default, repeats=args.repeats)
    results["3. CNN Integration"] = r3
    print(
        f"  3. CNN:     reg_spill_count={r3['reg_spill_count']}, "
        f"mean={r3['mean_s'] * 1000:.3f}ms  "
        f"{'PASS' if r3.get('valid') else 'FAIL'}"
    )

    r4 = bench_pseudo.run_bench(repeats=args.repeats)
    results["4. Pseudo Instructions"] = r4
    print(
        f"  4. Pseudo:  cases={r4['case_count']}, "
        f"mean={r4['mean_s'] * 1000:.3f}ms  "
        f"{'PASS' if r4.get('valid') else 'FAIL'}"
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
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  JSON report: {args.output_json}")

    if args.output_html:
        with open(args.output_html, "w", encoding="utf-8") as f:
            f.write(_make_html(results, total_time))
        print(f"  HTML report: {args.output_html}")

    if args.output_md:
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(_make_markdown(results))
        print(f"  Markdown:    {args.output_md}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
