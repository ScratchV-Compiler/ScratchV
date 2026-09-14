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
import re
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
    optimization: dict | None = None,
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
    if optimization:
        code_label = (
            f"{optimization['code_size_before']:,} → "
            f"{optimization['code_size_after']:,} B "
            f"({optimization['machine_instructions_before']} → "
            f"{optimization['machine_instructions_after']} static insns)"
        )
    else:
        code_label = f"{code_size:,} B ({static_insns} static insns)"
    parts.append(
        f'<div class="subtitle">Model: <code>{model_name}</code> | '
        f'Code: {code_label} | '
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

    if optimization:
        size_before = optimization["code_size_before"]
        size_reduction = optimization["code_size_reduction"]
        reduction_pct = size_reduction / max(size_before, 1) * 100
        parts.append("<section><h2>Constant-Merge A/B Result</h2>")
        parts.append("<table>")
        parts.append("<tr><th>Metric</th><th>Before</th><th>After</th><th>Reduction</th></tr>")
        parts.append(
            f"<tr><td>Machine instructions</td>"
            f"<td>{optimization['machine_instructions_before']:,}</td>"
            f"<td>{optimization['machine_instructions_after']:,}</td>"
            f"<td>{optimization['machine_instruction_reduction']:,}</td></tr>"
        )
        parts.append(
            f"<tr><td>Code size</td><td>{size_before:,} B</td>"
            f"<td>{optimization['code_size_after']:,} B</td>"
            f"<td>{size_reduction:,} B ({reduction_pct:.1f}%)</td></tr>"
        )
        parts.append(
            f"<tr><td>Source instructions</td>"
            f"<td>{optimization['source_instructions_before']:,}</td>"
            f"<td>{optimization['source_instructions_after']:,}</td>"
            f"<td>{optimization['source_instruction_reduction']:,}</td></tr>"
        )
        parts.append("</table></section>")

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
    optimization: dict | None = None,
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

    if optimization:
        report["constant_merge"] = optimization

    return json.dumps(report, indent=2)


def generate_github_summary(
    code_size: int, static_insns: int, est_data: dict,
    optimization: dict | None = None,
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
    if optimization:
        size_before = optimization["code_size_before"]
        size_after = optimization["code_size_after"]
        size_reduction = optimization["code_size_reduction"]
        reduction_pct = size_reduction / max(size_before, 1) * 100
        lines.append(
            f"| Code size | {size_before:,} B → **{size_after:,} B** "
            f"(-{size_reduction:,} B, {reduction_pct:.1f}%) |"
        )
        lines.append(
            "| Static machine instructions | "
            f"{optimization['machine_instructions_before']:,} → "
            f"**{optimization['machine_instructions_after']:,}** "
            f"(-{optimization['machine_instruction_reduction']:,}) |"
        )
    else:
        lines.append(f"| Code size | {code_size:,} B ({static_insns} insns) |")
    lines.append(f"| Compute % | {est_data.get('compute_ratio', 0):.1f}% |")
    lines.append(f"| Memory % | {est_data.get('memory_ratio', 0):.1f}% |")
    lines.append("")

    if optimization:
        lines.append("## Constant-Merge Details")
        lines.append("")
        lines.append("| Metric | Before | After | Reduction |")
        lines.append("|--------|-------:|------:|----------:|")
        lines.append(
            "| Source assembly instructions | "
            f"{optimization['source_instructions_before']:,} | "
            f"{optimization['source_instructions_after']:,} | "
            f"{optimization['source_instruction_reduction']:,} |"
        )
        lines.append(
            "| Encoded machine instructions | "
            f"{optimization['machine_instructions_before']:,} | "
            f"{optimization['machine_instructions_after']:,} | "
            f"{optimization['machine_instruction_reduction']:,} |"
        )
        lines.append(
            f"| Code size (bytes) | {size_before:,} | {size_after:,} | "
            f"{size_reduction:,} ({reduction_pct:.1f}%) |"
        )
        lines.append("")
        lines.append("| Pass metric | Value |")
        lines.append("|-------------|------:|")
        lines.append(
            f"| Structural candidates | {optimization['candidate_pairs']:,} |"
        )
        lines.append(
            f"| Merged `lui`/`addi` pairs | {optimization['merged_pairs']:,} |"
        )
        lines.append(
            "| Redundant `lui` removed | "
            f"{optimization['redundant_lui_removed']:,} |"
        )
        lines.append("")
        lines.append(
            "> Source-level merges can exceed machine-instruction reduction: "
            "large `li` pseudo-instructions still encode as `lui` + `addi`."
        )
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
# Schema v2 renderers (rv32_bench.py reports)
# ═══════════════════════════════════════════════════════════════════════════

_MISSING = object()


def _dig(data: Any, path: str) -> Any:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _shape(shape: Any) -> str:
    if not shape:
        return "?"
    return "×".join(str(d) for d in shape)


def render_markdown(report: dict) -> str:
    """Render a schema v2 report as Markdown with provenance tags."""
    model = report.get("model") or {}
    env = report.get("environment") or {}
    targets = report.get("targets") or {}
    sv = report.get("scratchv") or {}
    llvm = report.get("llvm") or {}
    cmp_ = report.get("comparison") or {}

    sv_compile = sv.get("compile") or {}
    sv_dyn = sv.get("dynamic") or {}
    sv_mix = sv.get("static_instruction_mix") or {}
    sv_out = sv.get("output") or {}
    ll_compile = llvm.get("compile") or {}
    ll_dyn = llvm.get("dynamic") or {}
    tgt_sv = targets.get("scratchv") or {}
    tgt_ll = targets.get("llvm") or {}

    sha = str(model.get("sha256") or "")
    completion = sv_dyn.get("completion")
    dyn_rows = [
        ("source", _fmt(sv_dyn.get("source")), _fmt(ll_dyn.get("source"))),
        ("completion", _fmt(completion), _fmt(ll_dyn.get("completion"))),
        ("executed", _fmt(sv_dyn.get("executed")), _fmt(ll_dyn.get("executed"))),
    ]
    sv_ops = sv_dyn.get("ops") or {}
    for key in ("total", "load", "store", "mul", "add", "madd", "branch"):
        dyn_rows.append(
            (f"ops.{key}", _fmt(sv_ops.get(key)), _fmt(None)),
        )

    lines = []
    lines.append(f"# RV32 Benchmark Report — {_fmt(model.get('path'))}")
    lines.append("")
    lines.append(f"- Generated: {_fmt(report.get('generated_at'))} | "
                 f"schema: {_fmt(report.get('schema_version'))}")
    lines.append(
        f"- Model: {_fmt(model.get('input_name'))}{_shape(model.get('input_shape'))}"
        f" → {_fmt(model.get('output_name'))}{_shape(model.get('output_shape'))}"
        f" | sha256={sha[:12]} | bytes={_fmt(model.get('bytes'))}"
    )
    lines.append(
        f"- Targets: ScratchV {_fmt(tgt_sv.get('isa'))}/{_fmt(tgt_sv.get('numeric_format'))}"
        f" | LLVM {_fmt(tgt_ll.get('isa'))}/{_fmt(tgt_ll.get('numeric_format'))}"
        f" (opt={_fmt(tgt_ll.get('opt_level'))})"
    )
    lines.append(
        f"- Environment: python={_fmt(env.get('python'))} "
        f"numpy={_fmt(env.get('numpy'))} tinyfive={_fmt(env.get('tinyfive'))} "
        f"llvmlite={_fmt(env.get('llvmlite'))}"
    )
    lines.append("")

    lines.append("## 1. Compilation [static]")
    lines.append("")
    lines.append("| Metric | ScratchV | LLVM |")
    lines.append("|--------|----------|------|")
    lines.append(f"| status | {_fmt(sv_compile.get('status'))} | {_fmt(ll_compile.get('status'))} |")
    lines.append(f"| code bytes | {_fmt(sv_compile.get('code_bytes'))} | — |")
    lines.append(f"| data offset | {_fmt(sv_compile.get('data_offset'))} | — |")
    lines.append(f"| data bytes | {_fmt(sv_compile.get('data_bytes'))} | — |")
    ll_static = (
        _fmt(ll_compile.get("static_insns"))
        if ll_compile.get("status") == "success" else "—"
    )
    lines.append(
        f"| static insns [asm_scan] | {_fmt(sv_compile.get('static_insns'))} "
        f"| {ll_static} |"
    )
    lines.append("")
    lines.append("### Static instruction mix [static]")
    lines.append("")
    lines.append("| class | count |")
    lines.append("|-------|-------|")
    for key in ("load", "store", "mul", "add", "madd", "branch", "other"):
        lines.append(f"| {key} | {_fmt(sv_mix.get(key))} |")
    lines.append("")

    lines.append("## 2. Dynamic Execution [measured]")
    lines.append("")
    lines.append("| Metric | ScratchV | LLVM |")
    lines.append("|--------|----------|------|")
    for label, sv_cell, ll_cell in dyn_rows:
        lines.append(f"| {label} | {sv_cell} | {ll_cell} |")
    lines.append("")
    if completion == "budget_exhausted":
        lines.append(
            f"> [measured/budget] budget exhausted at {sv_dyn.get('limit')} "
            "instructions; dynamic counts are partial and excluded from "
            "comparison."
        )
        lines.append("")
    elif completion == "timeout":
        lines.append(
            f"> [measured/timeout] wall-clock timeout after "
            f"{_fmt(sv_dyn.get('elapsed_s'))}s; dynamic counts are partial."
        )
        lines.append("")
    elif sv_dyn.get("source") == "unavailable":
        lines.append(
            f"> [unavailable] ScratchV dynamic section omitted: "
            f"{_fmt(sv_dyn.get('reason'))}"
        )
        lines.append("")
    if ll_dyn.get("source") == "unavailable":
        lines.append(
            f"> [unavailable] LLVM dynamic section omitted: "
            f"{_fmt(ll_dyn.get('reason'))}"
        )
        lines.append("")
    if sv_dyn.get("source") == "simulated":
        lines.append(
            f"> Simulated by {_fmt(sv_dyn.get('simulator'))} "
            f"{_fmt(sv_dyn.get('simulator_version'))} | "
            f"completion={_fmt(completion)} | executed={_fmt(sv_dyn.get('executed'))} "
            f"| limit={_fmt(sv_dyn.get('limit'))} | "
            f"memory={_fmt(sv_dyn.get('memory_size_bytes'))} "
            f"| seed={_fmt(sv_dyn.get('input_seed'))} "
            f"| halt=0x{int(sv_dyn.get('halt_addr') or 0):x}"
        )
        lines.append("")

    lines.append("## 3. Comparison [measured]")
    lines.append("")
    ratio = cmp_.get("dynamic_instruction_ratio")
    if ratio is None:
        lines.append(
            f"- comparison ratio: **null** — "
            f"{_fmt(cmp_.get('incomparable_reason'))}"
        )
    else:
        lines.append(
            f"- dynamic_instruction_ratio: **{ratio:g}** "
            "(ScratchV / LLVM, both sides halted)"
        )
    lines.append("")

    lines.append("## 4. Analytical Warnings [estimated]")
    lines.append("")
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(f"- {w}" for w in warnings)
    else:
        lines.append("- none")
    lines.append("")
    errors = report.get("errors") or []
    if errors:
        lines.append("## 5. Errors")
        lines.append("")
        lines.extend(f"- {e}" for e in errors)
        lines.append("")

    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- model sha256={sha}")
    lines.append(
        f"- binary sha256={_fmt(sv_compile.get('binary_sha256'))} "
        f"| data_offset={_fmt(sv_compile.get('data_offset'))} "
        f"({_fmt(sv_compile.get('data_offset_source'))})"
    )
    lines.append(
        f"- static_source={_fmt(sv_compile.get('static_source'))} "
        f"| llvm static_source={_fmt(ll_compile.get('static_source'))}"
    )
    if sv_dyn.get("source") != "simulated":
        out_tag = "[unavailable]"
    elif sv_out.get("partial"):
        out_tag = "[measured/partial]"
    else:
        out_tag = "[measured]"
    lines.append(
        f"- output {out_tag}: {_fmt(sv_out.get('raw_hex'))} "
        f"(addr=0x{int(sv_out.get('addr') or 0):x}, "
        f"elements={_fmt(sv_out.get('elements'))}, "
        f"completion={_fmt(sv_out.get('completion'))})"
    )
    return "\n".join(lines)


def _md_to_html(md: str) -> str:
    from html import escape
    lines = md.splitlines()
    out: list[str] = []
    in_table = False
    in_list = False

    def close_blocks():
        nonlocal in_table, in_list
        if in_table:
            out.append("</table>")
            in_table = False
        if in_list:
            out.append("</ul>")
            in_list = False

    def inline(text: str) -> str:
        text = escape(text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        return text

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= {"-", " "} and c for c in cells):
                continue
            if not in_table:
                close_blocks()
                out.append("<table>")
                in_table = True
            tag = "th" if not any(mark.startswith("<tr>") for mark in out[-2:]) else "td"
            out.append("<tr>" + "".join(
                f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if stripped.startswith("- "):
            if not in_list:
                close_blocks()
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(stripped[2:])}</li>")
            continue
        close_blocks()
        if stripped.startswith("### "):
            out.append(f"<h3>{inline(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            out.append(f"<h2>{inline(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            out.append(f"<h1>{inline(stripped[2:])}</h1>")
        elif stripped.startswith("> "):
            out.append(f"<blockquote>{inline(stripped[2:])}</blockquote>")
        elif stripped:
            out.append(f"<p>{inline(stripped)}</p>")
    close_blocks()
    body = "\n".join(out)
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>RV32 Benchmark Report</title>"
        f"{HTML_CSS}</head><body>\n{body}\n</body></html>"
    )


def render_html(report: dict) -> str:
    """Wrap the schema v2 Markdown rendering in a self-contained HTML shell."""
    return _md_to_html(render_markdown(report))


def render_bench_json(report: dict) -> str:
    """Serialize the schema v2 report as JSON."""
    return json.dumps(report, sort_keys=False, indent=2, default=str)


def render_github_summary(report: dict) -> str:
    """Render a compact GitHub Actions summary using measured/static data only."""
    model = report.get("model") or {}
    env = report.get("environment") or {}
    sv = report.get("scratchv") or {}
    sv_compile = sv.get("compile") or {}
    sv_dyn = sv.get("dynamic") or {}
    sv_ops = sv_dyn.get("ops") or {}
    cmp_ = report.get("comparison") or {}
    sha = str(model.get("sha256") or "")

    lines = []
    lines.append("# RV32 Benchmark Summary")
    lines.append("")
    lines.append(f"- Model: `{_fmt(model.get('path'))}` (sha256={sha[:12]})")
    lines.append(f"- schema: {_fmt(report.get('schema_version'))}")
    lines.append("")
    lines.append("| Metric | Value | Source |")
    lines.append("|--------|-------|--------|")
    lines.append(
        f"| completion | {_fmt(sv_dyn.get('completion'))} | "
        f"[{_fmt(sv_dyn.get('source'))}] |"
    )
    lines.append(f"| executed | {_fmt(sv_dyn.get('executed'))} | [measured] |")
    lines.append(f"| ops.total | {_fmt(sv_ops.get('total'))} | [measured] |")
    lines.append(
        f"| static insns | {_fmt(sv_compile.get('static_insns'))} | [static] |"
    )
    lines.append(
        f"| model bytes | {_fmt(model.get('bytes'))} | [static] |"
    )
    lines.append("")
    ratio = cmp_.get("dynamic_instruction_ratio")
    if ratio is None:
        lines.append(
            f"> ratio: null — {_fmt(cmp_.get('incomparable_reason'))}"
        )
    else:
        lines.append(f"> dynamic_instruction_ratio: {ratio:g}")
    lines.append("")
    lines.append(
        f"> tinyfive={_fmt(env.get('tinyfive'))} "
        f"| memory={_fmt(sv_dyn.get('memory_size_bytes'))} "
        f"| seed={_fmt(sv_dyn.get('input_seed'))}"
    )
    warnings = report.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("## Warnings [estimated]")
        lines.extend(f"- {w}" for w in warnings)
    return "\n".join(lines)


def validate_report_schema(report: dict) -> list[str]:
    """Structural validation of a schema v2 report; empty list means valid."""
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["report is not a dict"]

    def require(path: str, predicate=None, description: str = ""):
        value = _dig(report, path)
        if value is _MISSING:
            errors.append(f"missing required field: {path}")
        elif predicate is not None and not predicate(value):
            errors.append(
                f"invalid {description or path}: {value!r}"
            )
        return value

    def is_int(value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    require("schema_version", lambda v: v == "rv32-bench/2")
    require("generated_at", lambda v: isinstance(v, str) and bool(v))
    require("generator.script", lambda v: isinstance(v, str) and bool(v))
    require("model.path", lambda v: isinstance(v, str) and bool(v))
    require(
        "model.sha256",
        lambda v: isinstance(v, str) and bool(re.fullmatch(r"[0-9a-f]{64}", v)),
    )
    require("model.bytes", is_int)
    require("environment", lambda v: isinstance(v, dict))
    require("targets.scratchv.isa", lambda v: isinstance(v, str) and bool(v))
    require("targets.llvm.isa", lambda v: isinstance(v, str) and bool(v))

    for side in ("scratchv", "llvm"):
        require(f"{side}.compile.status", lambda v: isinstance(v, str) and bool(v))
        require(f"{side}.compile.static_source", lambda v: v == "asm_scan")
        dyn = _dig(report, f"{side}.dynamic")
        if dyn is _MISSING or not isinstance(dyn, dict):
            errors.append(f"missing required field: {side}.dynamic")
            continue
        source = dyn.get("source")
        if source not in ("simulated", "unavailable"):
            errors.append(f"invalid {side}.dynamic.source: {source!r}")
            continue
        if source == "simulated":
            for key in ("simulator", "simulator_version", "completion",
                        "executed", "memory_size_bytes", "input_seed"):
                if dyn.get(key) is None:
                    errors.append(
                        f"missing required field: {side}.dynamic.{key}"
                    )
            if dyn.get("completion") not in (
                "halted", "budget_exhausted", "timeout"
            ):
                errors.append(
                    f"invalid {side}.dynamic.completion: "
                    f"{dyn.get('completion')!r}"
                )
            if not is_int(dyn.get("executed")):
                errors.append(
                    f"invalid {side}.dynamic.executed: {dyn.get('executed')!r}"
                )
            ops = dyn.get("ops")
            if not isinstance(ops, dict):
                errors.append(f"missing required field: {side}.dynamic.ops")
            else:
                for key in ("total", "load", "store", "mul", "add",
                            "madd", "branch"):
                    if not is_int(ops.get(key)):
                        errors.append(
                            f"invalid {side}.dynamic.ops.{key}: "
                            f"{ops.get(key)!r}"
                        )
        else:
            require(f"{side}.dynamic.reason",
                    lambda v: isinstance(v, str) and bool(v))
            if dyn.get("ops") is not None:
                errors.append(
                    f"{side}.dynamic.ops must be null when unavailable"
                )

    out = _dig(report, "scratchv.output")
    if out is _MISSING or not isinstance(out, dict):
        errors.append("missing required field: scratchv.output")
    else:
        if not isinstance(out.get("partial"), bool):
            errors.append(
                f"invalid scratchv.output.partial: {out.get('partial')!r}"
            )
        if not isinstance(out.get("completion"), str) or \
                not out.get("completion"):
            errors.append(
                "invalid scratchv.output.completion: "
                f"{out.get('completion')!r}"
            )
        if not is_int(out.get("elements")):
            errors.append(
                f"invalid scratchv.output.elements: "
                f"{out.get('elements')!r}"
            )
        q16 = out.get("q16_16")
        if out.get("partial") is False:
            if out.get("raw_hex") is None:
                errors.append(
                    "missing required field: scratchv.output.raw_hex"
                )
            if not isinstance(q16, list):
                errors.append(
                    f"invalid scratchv.output.q16_16: {q16!r}"
                )
        elif q16 is not None and not isinstance(q16, list):
            errors.append(
                f"invalid scratchv.output.q16_16: {q16!r}"
            )
        if isinstance(q16, list) and is_int(out.get("elements")) and \
                len(q16) != out["elements"]:
            errors.append(
                f"invalid scratchv.output.q16_16 length: {len(q16)} "
                f"!= elements {out['elements']}"
            )

    comparison = _dig(report, "comparison")
    if comparison is _MISSING or not isinstance(comparison, dict):
        errors.append("missing required field: comparison")
    else:
        if "dynamic_instruction_ratio" not in comparison:
            errors.append(
                "missing required field: comparison.dynamic_instruction_ratio"
            )
        require("comparison.incomparable_reason",
                lambda v: v is None or (isinstance(v, str) and bool(v)))
    require("warnings", lambda v: isinstance(v, list))
    require("errors", lambda v: isinstance(v, list))
    return errors


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
