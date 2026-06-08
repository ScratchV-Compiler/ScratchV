"""Generate ScratchV optimization history page for GitHub Pages."""

from __future__ import annotations
import json, os, sys
from pathlib import Path
from datetime import datetime

PROJ = Path(__file__).resolve().parent.parent.parent
HISTORY_FILE = PROJ / "benchmark_reports" / "optimization_history.json"
OUTPUT_FILE = PROJ / "benchmark_reports" / "history.html"

CSS = """*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;line-height:1.6}
.wrap{max-width:900px;margin:0 auto;padding:24px 20px}
h1{font-size:1.4rem;font-weight:800;color:#f8fafc}
h2{font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:12px}
.sub{font-size:.75rem;color:#64748b;margin-top:4px}
.header{background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #334155;border-radius:12px;padding:24px 28px;margin-bottom:24px}
.header h1{margin-bottom:4px}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px}
@media(max-width:700px){.kpi-grid{grid-template-columns:repeat(2,1fr)}}
.kpi{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:14px 16px;text-align:center}
.kpi .v{font-size:1.6rem;font-weight:800}
.kpi .v.g{color:#22c55e}.kpi .v.r{color:#ef4444}.kpi .v.y{color:#f59e0b}
.kpi .l{font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.4px;margin-top:2px}
/* ── Chart ── */
.chart-section{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:24px 20px 16px;margin-bottom:24px}
.chart-section h2{margin-bottom:4px}
.chart-section .chart-sub{font-size:.7rem;color:#64748b;margin-bottom:16px}
.chart-container{width:100%;overflow-x:auto}
.chart-legend{display:flex;gap:20px;flex-wrap:wrap;justify-content:center;margin-top:12px;font-size:.72rem;color:#94a3b8}
.chart-legend .leg-item{display:flex;align-items:center;gap:6px}
.chart-legend .leg-line{width:24px;height:2px;border-radius:1px}
.chart-legend .leg-dot{width:8px;height:8px;border-radius:50%}
/* ── Timeline ── */
.timeline{position:relative;padding-left:32px}
.timeline::before{content:'';position:absolute;left:11px;top:8px;bottom:8px;width:2px;background:#334155}
.milestone{position:relative;margin-bottom:28px}
.milestone::before{content:'';position:absolute;left:-24px;top:6px;width:12px;height:12px;border-radius:50%;border:2px solid #64748b;background:#1e293b}
.milestone.optimized::before{background:#22c55e;border-color:#22c55e}
.milestone.baseline::before{background:#64748b;border-color:#64748b}
.milestone .card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:18px 20px}
.milestone .card:hover{border-color:#475569}
.milestone .top{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:6px}
.milestone .ver{font-size:.75rem;font-weight:800;color:#f8fafc}
.milestone .date{font-size:.7rem;color:#475569}
.milestone .title{font-size:.95rem;font-weight:700;color:#f1f5f9;margin-bottom:6px}
.milestone .desc{font-size:.78rem;color:#94a3b8;line-height:1.6;margin-bottom:12px}
/* ── Changes list ── */
.milestone .changes-title{font-size:.7rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.4px;margin-bottom:6px}
.milestone .changes-list{list-style:none;padding:0;margin:0 0 14px}
.milestone .changes-list li{font-size:.74rem;color:#cbd5e1;line-height:1.6;padding:3px 0 3px 16px;position:relative}
.milestone .changes-list li::before{content:'▸';position:absolute;left:0;color:#f59e0b;font-size:.65rem}
/* ── Delta bar ── */
.delta-bar{display:flex;align-items:center;gap:12px;margin-bottom:12px;font-size:.72rem;flex-wrap:wrap}
.delta-bar .delta-item{display:flex;align-items:center;gap:4px;background:#0f172a;border-radius:4px;padding:4px 10px}
.delta-bar .delta-label{color:#64748b}
.delta-bar .delta-val{font-weight:700;font-variant-numeric:tabular-nums}
.delta-bar .delta-val.up{color:#22c55e}
.delta-bar .delta-val.down{color:#ef4444}
/* ── Metrics grid ── */
.milestone .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;font-size:.72rem}
@media(max-width:600px){.milestone .metrics{grid-template-columns:repeat(2,1fr)}}
.milestone .metrics .m{background:#0f172a;border-radius:4px;padding:6px 10px}
.milestone .metrics .m .mv{font-weight:700;font-variant-numeric:tabular-nums}
.milestone .metrics .m .ml{color:#64748b;font-size:.62rem}
/* ── Footer ── */
.ft{text-align:center;color:#475569;font-size:.65rem;padding:16px;margin-top:8px}
.ft a{color:#64748b}
"""


def _f(n):
    if n is None: return "—"
    if isinstance(n, float) and n >= 1e9: return f"{n/1e9:.2f}B"
    if isinstance(n, float) and n >= 1e6: return f"{n/1e6:.1f}M"
    if isinstance(n, int) and n >= 1e9: return f"{n/1e9:.2f}B"
    if isinstance(n, int) and n >= 1e6: return f"{n/1e6:.1f}M"
    return f"{n:,.0f}" if isinstance(n, (int, float)) else str(n)


def _svg_chart(baseline: dict, milestones: list) -> str:
    """Generate an SVG line chart: LLVM baseline + ScratchV progress curve."""
    W, H = 760, 320
    ML, MR, MT, MB = 90, 30, 24, 56  # margins
    PW = W - ML - MR  # plot width
    PH = H - MT - MB  # plot height

    # Find Y range
    vals = [baseline["dynamic_insns"]] + [m["dynamic_insns"] for m in milestones]
    y_min = baseline["dynamic_insns"] * 0.85  # leave some space below baseline
    y_max = max(vals) * 1.08

    def _fy(v):
        return MT + PH - (v - y_min) / (y_max - y_min) * PH

    def _fx(i):
        """X position for milestone index i (0..n-1)."""
        if len(milestones) <= 1:
            return ML + PW / 2
        return ML + i / (len(milestones) - 1) * PW

    # Y-axis ticks
    y_ticks = []
    step = 1e9  # 1B steps
    for v in range(int(y_min / step) * int(step), int(y_max) + int(step), int(step)):
        if v > y_min and v < y_max:
            y_ticks.append(v)

    svg = f"""<svg viewBox="0 0 {W} {H}" width="100%" height="auto" style="max-width:{W}px;font-family:system-ui,sans-serif">
  <!-- grid -->
  <line x1="{ML}" y1="{_fy(baseline['dynamic_insns'])}" x2="{ML+PW}" y2="{_fy(baseline['dynamic_insns'])}" stroke="#22c55e" stroke-dasharray="6,4" stroke-width="1.5" opacity="0.6"/>
  <text x="{ML-10}" y="{_fy(baseline['dynamic_insns'])+4}" text-anchor="end" fill="#22c55e" font-size="11" font-weight="700">LLVM {_f(baseline['dynamic_insns'])}</text>"""

    for yv in y_ticks:
        yp = _fy(yv)
        svg += f'\n  <line x1="{ML}" y1="{yp}" x2="{ML+PW}" y2="{yp}" stroke="#334155" stroke-width="1"/>'
        svg += f'\n  <text x="{ML-8}" y="{yp+4}" text-anchor="end" fill="#64748b" font-size="10">{_f(yv)}</text>'

    # Y-axis line
    svg += f'\n  <line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" stroke="#475569" stroke-width="1"/>'

    # X-axis line
    svg += f'\n  <line x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}" stroke="#475569" stroke-width="1"/>'

    # ScratchV polyline + dots
    points = []
    for i, m in enumerate(milestones):
        x = _fx(i)
        y = _fy(m["dynamic_insns"])
        points.append(f"{x:.0f},{y:.1f}")

    if len(points) >= 2:
        svg += f'\n  <polyline points="{" ".join(points)}" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linejoin="round"/>'

    for i, m in enumerate(milestones):
        x = _fx(i)
        y = _fy(m["dynamic_insns"])
        ratio_str = f"{m['vs_llvm_ratio']:.1f}x"
        svg += f'\n  <circle cx="{x:.0f}" cy="{y:.1f}" r="5" fill="#f59e0b" stroke="#0f172a" stroke-width="2"/>'
        # Label above dot
        svg += f'\n  <text x="{x:.0f}" y="{y-10:.1f}" text-anchor="middle" fill="#f59e0b" font-size="11" font-weight="700">{_f(m["dynamic_insns"])}</text>'
        # Version below x-axis
        svg += f'\n  <text x="{x:.0f}" y="{MT+PH+18}" text-anchor="middle" fill="#94a3b8" font-size="11">{m["version"]}</text>'
        # vs LLVM ratio below version
        svg += f'\n  <text x="{x:.0f}" y="{MT+PH+34}" text-anchor="middle" fill="#64748b" font-size="10">{ratio_str} vs LLVM</text>'

    svg += '\n</svg>'
    return svg


def generate(history: dict | None = None) -> str:
    if history is None:
        history = json.loads(HISTORY_FILE.read_text())

    baseline = history["baseline"]
    milestones = history["milestones"]
    first = milestones[0]
    last = milestones[-1]

    # ── KPI cards ──
    dyn_reduction = (1 - last["dynamic_insns"] / first["dynamic_insns"]) * 100
    ratio_change = first["vs_llvm_ratio"] - last["vs_llvm_ratio"]
    time_reduction = (1 - last["time_100mhz_s"] / first["time_100mhz_s"]) * 100
    first_ratio_cls = "y" if first["vs_llvm_ratio"] > 4 else "r"
    last_ratio_cls = "y" if last["vs_llvm_ratio"] > 1.5 else "g"

    h = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScratchV · 优化历史</title><style>{CSS}</style></head><body><div class="wrap">
<div class="header">
<h1>ScratchV 编译器优化历史</h1>
<div class="sub">模型: {history["model"]} &nbsp;|&nbsp; Baseline: {baseline["name"]} ({_f(baseline["dynamic_insns"])} 动态指令) &nbsp;|&nbsp; 更新于 {datetime.now().strftime("%Y-%m-%d")}</div>
</div>

<div class="kpi-grid">
<div class="kpi"><div class="v g">−{dyn_reduction:.0f}%</div><div class="l">动态指令减少</div></div>
<div class="kpi"><div class="v{' g' if ratio_change>0 else ' r'}">{ratio_change:.1f}x</div><div class="l">vs LLVM 差距缩小</div></div>
<div class="kpi"><div class="v g">−{time_reduction:.0f}%</div><div class="l">推理时间减少 @100MHz</div></div>
<div class="kpi"><div class="v y">{last['vs_llvm_ratio']:.2f}x</div><div class="l">当前 vs LLVM</div></div>
</div>

<div class="chart-section">
<h2>动态指令趋势</h2>
<div class="chart-sub">绿色虚线 = LLVM float32 baseline（{_f(baseline["dynamic_insns"])}） · 橙色实线 = ScratchV 各版本 · 越低越好</div>
<div class="chart-container">
{_svg_chart(baseline, milestones)}
</div>
<div class="chart-legend">
<div class="leg-item"><span class="leg-line" style="background:#22c55e;border:1px dashed #22c55e"></span> LLVM baseline ({_f(baseline["dynamic_insns"])})</div>
<div class="leg-item"><span class="leg-dot" style="background:#f59e0b"></span> ScratchV</div>
</div>
</div>

<h2 style="margin-bottom:16px">优化时间线</h2>
<div class="timeline">"""

    for m in reversed(milestones):
        tag_cls = m.get("tag", "")
        # Find previous milestone for comparison
        prev = None
        for pm in milestones:
            if pm is m:
                break
            prev = pm

        h += f"""<div class="milestone {tag_cls}">
<div class="card">
<div class="top"><span class="ver">{m["version"]}</span><span class="date">{m["date"]}</span></div>
<div class="title">{m["title"]}</div>
<div class="desc">{m["description"]}</div>"""

        # ── Optimization changes list ──
        changes = m.get("changes", [])
        if changes:
            h += '<div class="changes-title">优化要点</div><ul class="changes-list">'
            for c in changes:
                h += f"<li>{c}</li>"
            h += "</ul>"

        # ── Delta from previous version ──
        if prev:
            delta_dyn = (1 - m["dynamic_insns"] / prev["dynamic_insns"]) * 100
            delta_ratio = prev["vs_llvm_ratio"] - m["vs_llvm_ratio"]
            delta_time = (1 - m["time_100mhz_s"] / prev["time_100mhz_s"]) * 100
            h += '<div class="delta-bar">'
            h += '<span style="color:#64748b;font-weight:600">较上一版:</span>'
            if delta_dyn > 0:
                h += f'<span class="delta-item"><span class="delta-label">动态指令</span><span class="delta-val up">−{delta_dyn:.1f}%</span></span>'
            if delta_ratio > 0:
                h += f'<span class="delta-item"><span class="delta-label">vs LLVM</span><span class="delta-val up">−{delta_ratio:.2f}x</span></span>'
            if delta_time > 0:
                h += f'<span class="delta-item"><span class="delta-label">推理时间</span><span class="delta-val up">−{delta_time:.1f}%</span></span>'
            h += '</div>'

        # ── Metrics grid ──
        h += '<div class="metrics">'
        h += f'<div class="m"><div class="mv">{_f(m["dynamic_insns"])}</div><div class="ml">动态指令</div></div>'
        h += f'<div class="m"><div class="mv">{m["vs_llvm_ratio"]:.2f}x</div><div class="ml">vs LLVM 比值</div></div>'
        h += f'<div class="m"><div class="mv">{m["per_mac_insns"]} instr/MAC</div><div class="ml">内层循环效率</div></div>'
        h += f'<div class="m"><div class="mv">{m["time_100mhz_s"]:.1f}s</div><div class="ml">@100MHz 推理时间</div></div>'
        h += "</div>"

        # ── Comparison to LLVM (absolute) ──
        ratio = m["vs_llvm_ratio"]
        ratio_color = "#22c55e" if ratio <= 1.5 else ("#f59e0b" if ratio <= 4 else "#ef4444")
        gap = m["dynamic_insns"] - baseline["dynamic_insns"]
        h += f'<div style="margin-top:10px;font-size:.7rem;color:#64748b">'
        h += f'与 LLVM 差距: <span style="color:{ratio_color};font-weight:700">{_f(gap)}</span> 条指令 '
        h += f'(<span style="color:{ratio_color};font-weight:700">{ratio:.2f}x</span>)'
        h += '</div>'

        h += "</div></div>"

    h += f"""</div>

<div class="ft">
ScratchV Compiler · <a href="https://github.com/ScratchV-Compiler/ScratchV">GitHub</a>
&nbsp;·&nbsp; <a href="optimization_history.json">JSON data</a>
&nbsp;·&nbsp; Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}
</div></div></body></html>"""
    return h


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate optimization history page")
    p.add_argument("-o", "--output", default=str(OUTPUT_FILE),
                   help=f"Output HTML path (default: {OUTPUT_FILE})")
    p.add_argument("--json", default=str(HISTORY_FILE),
                   help="History JSON input")
    args = p.parse_args()

    if not os.path.exists(args.json):
        print(f"ERROR: {args.json} not found", file=sys.stderr)
        return 1

    with open(args.json) as f:
        history = json.load(f)

    html = generate(history)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"→ {args.output} ({len(html):,}B)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
