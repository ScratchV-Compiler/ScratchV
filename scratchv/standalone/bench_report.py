#!/usr/bin/env python3
"""Benchmark report generator — HTML, JSON, Markdown outputs for CI visualization.

Generates rich, self-contained HTML reports with CSS bar charts, JSON for
machine parsing, and GitHub Actions job summaries. Zero external dependencies.

Usage:
    # Generate all report formats
    python bench_report.py --json /tmp/bench.json --html /tmp/bench.html \\
        --md /tmp/bench.md --code-size 3140

    # From emulation data
    python bench_report.py --perf-json /tmp/perf.json --output-dir reports/
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report Generator (pure HTML+CSS, no JS libraries)
# ═══════════════════════════════════════════════════════════════════════════

HTML_CSS = """
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f7fa; color: #2d3748; padding: 24px; max-width: 1100px; margin: 0 auto; }
  h1 { font-size: 24px; margin-bottom: 8px; color: #1a202c; }
  .subtitle { color: #718096; font-size: 14px; margin-bottom: 24px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
           gap: 16px; margin-bottom: 24px; }
  .card { background: #fff; border-radius: 10px; padding: 18px 20px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 4px solid #4299e1; }
  .card.green { border-left-color: #48bb78; }
  .card.orange { border-left-color: #ed8936; }
  .card.purple { border-left-color: #9f7aea; }
  .card.red { border-left-color: #fc8181; }
  .card .label { font-size: 12px; text-transform: uppercase; color: #a0aec0;
                 letter-spacing: 0.5px; margin-bottom: 4px; }
  .card .value { font-size: 28px; font-weight: 700; color: #1a202c; }
  .card .unit { font-size: 13px; color: #718096; }
  section { background: #fff; border-radius: 10px; padding: 24px;
            margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  section h2 { font-size: 18px; margin-bottom: 16px; padding-bottom: 8px;
               border-bottom: 2px solid #e2e8f0; color: #2d3748; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; padding: 10px 12px; font-size: 12px; text-transform: uppercase;
       letter-spacing: 0.5px; color: #a0aec0; border-bottom: 2px solid #e2e8f0; }
  td { padding: 10px 12px; font-size: 14px; border-bottom: 1px solid #edf2f7; }
  tr:hover td { background: #f7fafc; }
  .bar-cell { width: 100%; min-width: 120px; }
  .bar-bg { height: 18px; background: #edf2f7; border-radius: 9px; overflow: hidden;
            position: relative; }
  .bar-fill { height: 100%; border-radius: 9px; transition: width 0.3s ease; }
  .bar-fill.compute { background: linear-gradient(90deg, #4299e1, #3182ce); }
  .bar-fill.memory { background: linear-gradient(90deg, #ed8936, #dd6b20); }
  .bar-fill.branch { background: linear-gradient(90deg, #48bb78, #38a169); }
  .bar-fill.upper { background: linear-gradient(90deg, #9f7aea, #805ad5); }
  .bar-fill.shift { background: linear-gradient(90deg, #38b2ac, #319795); }
  .bar-fill.neutral { background: linear-gradient(90deg, #a0aec0, #718096); }
  .bar-label { font-size: 12px; color: #4a5568; margin-left: 8px; white-space: nowrap; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px;
         font-weight: 600; }
  .tag.ok { background: #c6f6d5; color: #22543d; }
  .tag.warn { background: #fefcbf; color: #744210; }
  .tag.info { background: #bee3f8; color: #2a4365; }
  .progress-ring { display: flex; align-items: center; gap: 16px; }
  .cm-gauge { width: 120px; height: 120px; position: relative; }
  .cm-gauge svg { transform: rotate(-90deg); }
  .cm-gauge .bg { fill: none; stroke: #edf2f7; stroke-width: 10; }
  .cm-gauge .fg { fill: none; stroke: #4299e1; stroke-width: 10; stroke-linecap: round;
                  transition: stroke-dashoffset 0.5s ease; }
  .cm-gauge .pct { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                   font-size: 22px; font-weight: 700; }
  .footer { text-align: center; color: #a0aec0; font-size: 12px; margin-top: 32px; }
  @media (max-width: 600px) {
    body { padding: 12px; }
    .cards { grid-template-columns: 1fr 1fr; }
  }
</style>
"""


def _bar_chart_html(rows: list[tuple[str, float, str]],
                    max_pct: float = 100.0) -> str:
    """Generate HTML table rows with CSS bar charts."""
    lines = []
    for label, pct, color_class in rows:
        w = max(pct, 0.5)  # minimum visible width for very small values
        lines.append(
            f'<tr><td style="width:180px">{label}</td>'
            f'<td class="bar-cell"><div class="bar-bg">'
            f'<div class="bar-fill {color_class}" style="width:{w/max_pct*100:.1f}%"></div>'
            f'</div></td>'
            f'<td style="width:80px;text-align:right;font-weight:600">{pct:.1f}%</td></tr>'
        )
    return "\n".join(lines)


def generate_html_report(
    code_size: int,
    static_insns: int,
    est_data: dict,
    emu_data: dict | None = None,
    binary_path: str = "",
    model_name: str = "cnn.onnx",
) -> str:
    """Generate a self-contained HTML benchmark report.

    Args:
        code_size: Code section size in bytes.
        static_insns: Static instruction count.
        est_data: Analytical estimation dict from estimate_cnn_model().
        emu_data: Optional emulation PerfCounters data dict (from --perf-json).
        binary_path: Path to the compiled binary.
        model_name: Name of the ONNX model.
    """
    total_est = est_data.get("total_estimated", 0)
    cm_ratio = est_data.get("cm_ratio", 0)
    compute_pct = est_data.get("compute_ratio", 0)
    memory_pct = est_data.get("memory_ratio", 0)
    branch_pct = est_data.get("branch_ratio", 0)
    est_hw_50 = est_data.get("est_hw_time_50mhz", 0)
    est_hw_100 = est_data.get("est_hw_time_100mhz", 0)
    per_layer = est_data.get("per_layer", {})

    # ── Build HTML ────────────────────────────────────────────────────
    parts = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head><meta charset="UTF-8">')
    parts.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    parts.append(f"<title>ScratchV Benchmark — {model_name}</title>")
    parts.append(HTML_CSS)
    parts.append("</head><body>")

    # Header
    parts.append(f"<h1>ScratchV CNN RISC-V Benchmark</h1>")
    parts.append(
        f'<div class="subtitle">Model: <code>{model_name}</code> | '
        f'Code: {code_size:,} B ({static_insns} static insns) | '
        f'Generated: {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}</div>'
    )

    # Summary cards
    cm_tag = "ok" if cm_ratio > 2 else ("warn" if cm_ratio > 1 else "info")
    parts.append('<div class="cards">')
    parts.append(
        f'<div class="card"><div class="label">Total Instructions</div>'
        f'<div class="value">{total_est / 1e9:.2f}<span class="unit">B</span></div></div>'
    )
    parts.append(
        f'<div class="card green"><div class="label">C/M Ratio</div>'
        f'<div class="value">{cm_ratio:.1f}<span class="unit">compute-heavy</span></div></div>'
    )
    parts.append(
        f'<div class="card orange"><div class="label">Est. HW Time @50MHz</div>'
        f'<div class="value">{est_hw_50:.2f}<span class="unit">s</span></div></div>'
    )
    parts.append(
        f'<div class="card purple"><div class="label">Est. HW Time @100MHz</div>'
        f'<div class="value">{est_hw_100:.2f}<span class="unit">s</span></div></div>'
    )
    parts.append(
        f'<div class="card"><div class="label">Binary Size</div>'
        f'<div class="value">{code_size / 1024:.1f}<span class="unit">KB code</span></div></div>'
    )
    parts.append('</div>')

    # Instruction Mix section
    parts.append("<section><h2>Dynamic Instruction Mix (estimated)</h2>")
    parts.append("<table>")
    parts.append("<tr><th>Category</th><th>Distribution</th><th>Share</th></tr>")
    # Approximate breakdown from estimation
    other_pct = 100 - compute_pct - memory_pct - branch_pct
    mix_rows = [
        ("Compute (ALU + Shift)", compute_pct, "compute"),
        ("Memory (Load + Store)", memory_pct, "memory"),
        ("Branch (BEQ/BNE/BLT/BGE)", branch_pct, "branch"),
        ("Upper (LUI/AUIPC)", 5.0, "upper"),
        ("Other (Jump/NOP)", max(other_pct - 5.0, 0.5), "neutral"),
    ]
    parts.append(_bar_chart_html(mix_rows))
    parts.append("</table></section>")

    # Per-Layer Breakdown
    if per_layer:
        parts.append("<section><h2>Per-Operator Instruction Breakdown</h2>")
        parts.append("<table>")
        parts.append("<tr><th>Layer</th><th>Instructions</th><th>Share</th></tr>")
        for name, insns in per_layer.items():
            pct = insns / max(total_est, 1) * 100
            color = "compute" if "Conv" in name or "FC" in name else "memory"
            insn_str = f"{insns / 1e9:.2f}B" if insns > 1e9 else (
                f"{insns / 1e6:.1f}M" if insns > 1e6 else f"{insns:,}"
            )
            parts.append(
                f'<tr><td>{name}</td>'
                f'<td class="bar-cell"><div class="bar-bg">'
                f'<div class="bar-fill {color}" style="width:{max(pct, 0.3):.1f}%"></div>'
                f'</div></td>'
                f'<td style="width:100px;text-align:right">'
                f'<span style="font-weight:600">{insn_str}</span>'
                f'<span style="color:#a0aec0;margin-left:6px">({pct:.1f}%)</span></td></tr>'
            )
        parts.append("</table></section>")

    # Emulation data (if available)
    if emu_data:
        parts.append("<section><h2>Emulation Profile (sampled)</h2>")
        parts.append("<table>")
        parts.append("<tr><th>Metric</th><th>Value</th></tr>")
        for k, v in emu_data.items():
            if isinstance(v, float):
                parts.append(f"<tr><td>{k}</td><td>{v:.2f}</td></tr>")
            elif isinstance(v, dict):
                continue
            else:
                parts.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
        parts.append("</table></section>")

    # C/M Gauge
    parts.append("<section><h2>Compute-to-Memory Ratio</h2>")
    parts.append('<div class="progress-ring">')
    # Simple SVG gauge
    circumference = 2 * 3.14159 * 42
    offset = circumference * (1 - min(cm_ratio / 10, 1.0))
    parts.append(
        f'<svg width="100" height="100" viewBox="0 0 100 100">'
        f'<circle class="bg" cx="50" cy="50" r="42"/>'
        f'<circle class="fg" cx="50" cy="50" r="42" '
        f'stroke-dasharray="{circumference:.1f}" '
        f'stroke-dashoffset="{offset:.1f}"/>'
        f'</svg>'
    )
    classification = "Compute-Heavy" if cm_ratio > 2 else (
        "Balanced" if cm_ratio > 1 else "Memory-Heavy")
    parts.append(
        f'<div><div style="font-size:32px;font-weight:700">{cm_ratio:.1f}</div>'
        f'<div style="color:#718096">{classification}</div></div>'
    )
    parts.append("</div></section>")

    # Footer
    parts.append(
        f'<div class="footer">ScratchV Standalone RISC-V Compiler | '
        f'Report generated {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}</div>'
    )
    parts.append("</body></html>")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# JSON / Markdown / GitHub Actions summary generators
# ═══════════════════════════════════════════════════════════════════════════


def generate_json_report(
    code_size: int, static_insns: int, est_data: dict,
    emu_data: dict | None = None, model_name: str = "cnn.onnx",
) -> str:
    """Generate machine-parseable JSON benchmark report."""
    report = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": {
            "size_bytes": code_size,
            "static_instructions": static_insns,
        },
        "estimation": {
            "total_instructions": est_data.get("total_estimated", 0),
            "compute_ratio_pct": round(est_data.get("compute_ratio", 0), 1),
            "memory_ratio_pct": round(est_data.get("memory_ratio", 0), 1),
            "branch_ratio_pct": round(est_data.get("branch_ratio", 0), 1),
            "cm_ratio": round(est_data.get("cm_ratio", 0), 2),
            "est_hw_time_50mhz_s": round(est_data.get("est_hw_time_50mhz", 0), 2),
            "est_hw_time_100mhz_s": round(est_data.get("est_hw_time_100mhz", 0), 2),
        },
        "per_layer": {},
    }
    # Per-layer breakdown
    total_est = max(est_data.get("total_estimated", 1), 1)
    for name, insns in est_data.get("per_layer", {}).items():
        report["per_layer"][name] = {
            "instructions": insns,
            "pct": round(insns / total_est * 100, 1),
        }

    # Emulation data summary
    if emu_data:
        report["emulation"] = {
            k: v for k, v in emu_data.items()
            if not isinstance(v, dict)
        }

    return json.dumps(report, indent=2)


def generate_github_summary(
    code_size: int, static_insns: int, est_data: dict,
) -> str:
    """Generate GitHub Actions job summary markdown."""
    total_est = est_data.get("total_estimated", 0)
    cm_ratio = est_data.get("cm_ratio", 0)
    per_layer = est_data.get("per_layer", {})

    lines = []
    lines.append("# ScratchV CNN RISC-V Benchmark")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total dynamic instructions | **{total_est / 1e9:.2f} B** |")
    lines.append(f"| C/M ratio | **{cm_ratio:.1f}** (compute-heavy) |")
    lines.append(f"| Est. HW time @ 50 MHz | **{est_data.get('est_hw_time_50mhz', 0):.1f} s** |")
    lines.append(f"| Est. HW time @ 100 MHz | **{est_data.get('est_hw_time_100mhz', 0):.1f} s** |")
    lines.append(f"| Code size | {code_size:,} B ({static_insns} insns) |")
    lines.append(f"| Compute % | {est_data.get('compute_ratio', 0):.1f}% |")
    lines.append(f"| Memory % | {est_data.get('memory_ratio', 0):.1f}% |")
    lines.append("")

    lines.append("## Per-Layer Breakdown")
    lines.append("")
    lines.append("| Layer | Instructions | % |")
    lines.append("|-------|-------------|---|")
    for name, insns in per_layer.items():
        pct = insns / max(total_est, 1) * 100
        if insns > 1e9:
            s = f"{insns / 1e9:.2f}B"
        elif insns > 1e6:
            s = f"{insns / 1e6:.1f}M"
        else:
            s = f"{insns:,}"
        lines.append(f"| {name} | {s} | {pct:.1f}% |")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="ScratchV Benchmark Report Generator"
    )
    parser.add_argument("--code-size", type=int, default=3140,
                        help="Code section size in bytes")
    parser.add_argument("--static-insns", type=int, default=785,
                        help="Static instruction count")
    parser.add_argument("--model", default="cnn.onnx",
                        help="Model name for report titles")
    parser.add_argument("--html", default="",
                        help="Output HTML report path")
    parser.add_argument("--json-out", default="",
                        help="Output JSON report path")
    parser.add_argument("--md", default="",
                        help="Output markdown (GitHub summary) path")
    parser.add_argument("--perf-json", default="",
                        help="Input: emulation perf JSON (from --benchmark --json)")
    parser.add_argument("--output-dir", default="benchmark_reports",
                        help="Output directory for all report formats")
    args = parser.parse_args()

    # Get estimation data
    from scratchv.standalone.benchmark import estimate_cnn_model
    est_data = estimate_cnn_model()

    emu_data = None
    if args.perf_json and os.path.exists(args.perf_json):
        with open(args.perf_json) as f:
            emu_data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate reports
    # 1. HTML
    html_path = args.html or os.path.join(args.output_dir, "benchmark.html")
    html = generate_html_report(
        code_size=args.code_size,
        static_insns=args.static_insns,
        est_data=est_data,
        emu_data=emu_data,
        model_name=args.model,
    )
    with open(html_path, "w") as f:
        f.write(html)
    print(f"HTML report: {html_path}")

    # 2. JSON
    json_path = args.json_out or os.path.join(args.output_dir, "benchmark.json")
    json_str = generate_json_report(
        code_size=args.code_size,
        static_insns=args.static_insns,
        est_data=est_data,
        emu_data=emu_data,
        model_name=args.model,
    )
    with open(json_path, "w") as f:
        f.write(json_str)
    print(f"JSON report: {json_path}")

    # 3. GitHub Actions summary
    md_path = args.md or os.path.join(args.output_dir, "github_summary.md")
    md = generate_github_summary(
        code_size=args.code_size,
        static_insns=args.static_insns,
        est_data=est_data,
    )
    with open(md_path, "w") as f:
        f.write(md)
    print(f"GitHub summary: {md_path}")

    # Also print the summary to stdout for CI log
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
