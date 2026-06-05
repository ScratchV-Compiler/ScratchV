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
.wrap{max-width:860px;margin:0 auto;padding:24px 20px}
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
.timeline{position:relative;padding-left:32px}
.timeline::before{content:'';position:absolute;left:11px;top:8px;bottom:8px;width:2px;background:#334155}
.milestone{position:relative;margin-bottom:28px}
.milestone::before{content:'';position:absolute;left:-24px;top:6px;width:12px;height:12px;border-radius:50%;border:2px solid #64748b;background:#1e293b}
.milestone.optimized::before{background:#22c55e;border-color:#22c55e}
.milestone.baseline::before{background:#64748b;border-color:#64748b}
.milestone .card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:16px 20px}
.milestone .card:hover{border-color:#475569}
.milestone .top{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:6px}
.milestone .ver{font-size:.7rem;color:#64748b;font-family:monospace}
.milestone .date{font-size:.7rem;color:#475569}
.milestone .title{font-size:.95rem;font-weight:700;color:#f1f5f9;margin-bottom:4px}
.milestone .desc{font-size:.75rem;color:#94a3b8;line-height:1.5;margin-bottom:10px}
.milestone .metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;font-size:.72rem}
@media(max-width:600px){.milestone .metrics{grid-template-columns:repeat(2,1fr)}}
.milestone .metrics .m{background:#0f172a;border-radius:4px;padding:6px 10px}
.milestone .metrics .m .mv{font-weight:700;font-variant-numeric:tabular-nums}
.milestone .metrics .m .ml{color:#64748b;font-size:.62rem}
.arrow{display:inline-block;color:#64748b;margin:0 4px}
.improvement{display:inline-block;padding:1px 6px;border-radius:4px;font-size:.68rem;font-weight:700}
.improvement.g{background:#052e16;color:#22c55e}
.badge{display:inline-block;padding:1px 7px;border-radius:5px;font-size:.66rem;font-weight:700}
.badge.r{background:#450a0a;color:#ef4444}.badge.y{background:#422006;color:#f59e0b}.badge.g{background:#052e16;color:#22c55e}
.chart-bar{display:flex;align-items:center;gap:10px;margin:4px 0;font-size:.72rem}
.chart-bar .label{width:100px;text-align:right;color:#94a3b8;flex-shrink:0}
.chart-bar .bar-wrap{flex:1;background:#1e293b;border-radius:4px;height:22px;position:relative;overflow:hidden}
.chart-bar .bar-fill{height:100%;border-radius:4px;display:flex;align-items:center;padding-left:8px;font-size:.65rem;font-weight:700;white-space:nowrap}
.bar-llvm{background:rgba(34,197,94,.2)}.bar-llvm .bar-fill{background:#22c55e;width:14%}
.bar-sv0 .bar-fill{background:#ef4444;width:100%}
.bar-sv1 .bar-fill{background:#f59e0b;width:41%}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;font-size:.68rem;color:#64748b}
.legend span{display:flex;align-items:center;gap:4px}
.legend .dot{width:8px;height:8px;border-radius:2px}
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


def _badge(v):
    if isinstance(v, str): return v
    if v <= 1.5: return f'<span class="badge g">{v:.2f}x</span>'
    if v <= 4: return f'<span class="badge y">{v:.2f}x</span>'
    return f'<span class="badge r">{v:.2f}x</span>'


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

    h = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScratchV · 优化历史</title><style>{CSS}</style></head><body><div class="wrap">
<div class="header">
<h1>ScratchV 编译器优化历史</h1>
<div class="sub">模型: {history["model"]} &nbsp;|&nbsp; Baseline: {baseline["name"]} ({_f(baseline["dynamic_insns"])} 动态操作) &nbsp;|&nbsp; 更新于 {datetime.now().strftime("%Y-%m-%d")}</div>
</div>

<div class="kpi-grid">
<div class="kpi"><div class="v g">{dyn_reduction:.0f}%</div><div class="l">动态指令减少</div></div>
<div class="kpi"><div class="v{' g' if ratio_change>0 else ' r'}">{ratio_change:.1f}x</div><div class="l">vs LLVM 差距缩小</div></div>
<div class="kpi"><div class="v g">{time_reduction:.0f}%</div><div class="l">推理时间减少 @100MHz</div></div>
<div class="kpi"><div class="v y">{last['vs_llvm_ratio']:.2f}x</div><div class="l">当前 vs LLVM</div></div>
</div>

<div class="kpi" style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:18px 22px;margin-bottom:24px">
<h2 style="margin-bottom:12px">动态指令趋势（越低越好）</h2>
<div class="chart-bar"><div class="label">LLVM</div><div class="bar-wrap bar-llvm"><div class="bar-fill">{_f(baseline["dynamic_insns"])}</div></div></div>"""
    for m in milestones:
        width_pct = int(m["dynamic_insns"] / first["dynamic_insns"] * 100)
        cls = "bar-sv1" if m["tag"] == "optimized" else "bar-sv0"
        h += f"""<div class="chart-bar"><div class="label">{m["version"]}</div><div class="bar-wrap {cls}"><div class="bar-fill" style="width:{width_pct}%">{_f(m["dynamic_insns"])} · {m["vs_llvm_ratio"]:.1f}x LLVM</div></div></div>"""
    h += f"""<div class="legend">
<span><span class="dot" style="background:#22c55e"></span>LLVM float32 (baseline)</span>
<span><span class="dot" style="background:#ef4444"></span>v0.1.0 初始</span>
<span><span class="dot" style="background:#f59e0b"></span>v0.2.0 优化后</span>
</div></div>

<h2 style="margin-bottom:16px">优化时间线</h2>
<div class="timeline">"""

    for m in reversed(milestones):
        tag_cls = m.get("tag", "")
        prev = None
        for pm in milestones:
            if pm is m: break
            prev = pm
        h += f"""<div class="milestone {tag_cls}">
<div class="card">
<div class="top"><span class="ver">{m["version"]}</span><span class="date">{m["date"]}</span></div>
<div class="title">{m["title"]}</div>
<div class="desc">{m["description"]}</div>
<div class="metrics">"""
        h += f"""<div class="m"><div class="mv">{_f(m["dynamic_insns"])}</div><div class="ml">动态指令</div></div>"""
        h += f"""<div class="m"><div class="mv">{m["vs_llvm_ratio"]:.2f}x</div><div class="ml">vs LLVM 比值</div></div>"""
        h += f"""<div class="m"><div class="mv">{m["per_mac_insns"]} instr/MAC</div><div class="ml">内层循环效率</div></div>"""
        if prev:
            delta_dyn = (1 - m["dynamic_insns"] / prev["dynamic_insns"]) * 100
            delta_ratio = prev["vs_llvm_ratio"] - m["vs_llvm_ratio"]
            if delta_dyn > 0:
                h += f"""<div class="m"><div class="mv" style="color:#22c55e">−{delta_dyn:.1f}%</div><div class="ml">较上一版</div></div>"""
            if delta_ratio > 0:
                h += f"""<div class="m"><div class="mv" style="color:#22c55e">−{delta_ratio:.2f}x</div><div class="ml">vs LLVM 改善</div></div>"""
        h += f"""<div class="m"><div class="mv">{m["time_100mhz_s"]:.1f}s</div><div class="ml">@100MHz 推理时间</div></div>"""
        h += "</div></div></div>"

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
