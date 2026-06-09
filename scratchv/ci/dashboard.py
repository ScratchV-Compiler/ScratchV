"""Performance dashboard — LLVM baseline vs ScratchV (instruction-set focus).

Two granularities:
  1. Instruction-level (by category): ALU R/I, FP, Shift, Load, Store, Branch, Jump, Upper
  2. Operator-level (by type): Conv, Gemm, MaxPool, ReLU, Sigmoid

Removed: cycle estimates, CPI profiles, memory ops, TinyFive kernel (not needed for
instruction-level analysis).
"""

from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

PROJ = Path(__file__).resolve().parent.parent.parent
SINGLE_OP_BENCH = PROJ / "benchmark_reports" / "single_op_bench.json"


def _run(script, args):
    """Run tool, return parsed JSON."""
    tmp = None
    for i, a in enumerate(args):
        if a == "--json-output" and i + 1 < len(args):
            tmp = args[i + 1]; break
        if a == "--json" and i + 1 < len(args) and not args[i + 1].startswith("--"):
            tmp = args[i + 1]; break
    if not tmp:
        tmp = "/tmp/_bench_tmp.json"
        args = args + ["--json-output", tmp]
    subprocess.run([sys.executable, str(PROJ / script)] + args,
                   capture_output=True, cwd=str(PROJ), timeout=60)
    if os.path.exists(tmp):
        with open(tmp) as f:
            return json.load(f)
    return {}


def collect():
    """Collect LLVM vs ScratchV comparison data."""
    llvm_path = "/tmp/_llvm_bench.json"
    return _run("scratchv/standalone/llvm_cache_compare.py", ["--json-output", llvm_path])


def _f(n):
    if not n:
        return "—"
    if isinstance(n, float):
        n = int(n)
    return f"{n:,}"


def _ratio(sv, ll):
    """ScratchV / LLVM. Always appends × suffix."""
    if not ll:
        return "—"
    v = sv / ll
    return f"{v:.1f}×" if v >= 10 else f"{v:.2f}×"


def _ratio_float(sv, ll):
    """Return raw ratio float for comparisons."""
    if not ll:
        return 0
    return sv / ll


def _badge(v):
    try:
        fv = float(v.replace("×", ""))
    except (ValueError, TypeError):
        return v
    if fv <= 1.5:
        return f'<span class="badge g">{v}</span>'
    if fv <= 4:
        return f'<span class="badge y">{v}</span>'
    return f'<span class="badge r">{v}</span>'


CSS = """*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;line-height:1.5}
.wrap{max-width:960px;margin:0 auto;padding:20px}
.hdr{background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #334155;border-radius:12px;padding:20px 28px;margin-bottom:18px}
.hdr h1{font-size:1.15rem;color:#f1f5f9}
.hdr .sub{font-size:.75rem;color:#94a3b8;margin-top:3px}
.hdr .sub b{color:#e2e8f0}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:18px}
@media(max-width:700px){.kpi-grid{grid-template-columns:repeat(2,1fr)}}
.grid2{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:18px}
@media(max-width:600px){.grid2{grid-template-columns:1fr}}
.kpi{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:16px;text-align:center}
.kpi .v{font-size:1.8rem;font-weight:800;color:#f59e0b}
.kpi .v.g{color:#22c55e}
.kpi .v.y{color:#f59e0b}
.kpi .v.r{color:#ef4444}
.kpi .l{font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.4px;margin-top:3px}
.kpi .d{font-size:.65rem;color:#94a3b8;margin-top:2px}
.sec{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:20px;margin-bottom:16px}
.sec h2{font-size:.9rem;font-weight:700;margin-bottom:12px;color:#f1f5f9}
.sec .sec-sub{font-size:.7rem;color:#64748b;margin-bottom:12px}
.chart-section{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:24px 20px 16px;margin-bottom:18px}
.chart-section h2{margin-bottom:4px}
.chart-section .chart-sub{font-size:.7rem;color:#64748b;margin-bottom:16px}
.chart-container{width:100%;overflow-x:auto}
.chart-legend{display:flex;gap:20px;flex-wrap:wrap;justify-content:center;margin-top:12px;font-size:.72rem;color:#94a3b8}
.chart-legend .leg-item{display:flex;align-items:center;gap:6px}
.chart-legend .leg-line{width:24px;height:2px;border-radius:1px}
.chart-legend .leg-dot{width:8px;height:8px;border-radius:50%}
.version-selector{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin-bottom:16px}
.version-selector label.vs-label{display:flex;align-items:center;gap:5px;font-size:.72rem;color:#cbd5e1;cursor:pointer;padding:4px 10px;border-radius:4px;background:#0f172a;border:1px solid #334155;user-select:none}
.version-selector label.vs-label:hover{border-color:#475569}
.version-selector label.vs-label.checked{border-color:#f59e0b;background:#1e293b}
.version-selector label:not(.vs-label){font-size:.65rem;color:#64748b;cursor:pointer;padding:4px 6px}
.version-selector label:not(.vs-label):hover{color:#94a3b8}
.version-selector input[type=checkbox]{display:none}
.timeline{position:relative;padding-left:32px}
.timeline::before{content:'';position:absolute;left:11px;top:8px;bottom:8px;width:2px;background:#334155}
.milestone{position:relative;margin-bottom:28px}
.milestone::before{content:'';position:absolute;left:-24px;top:6px;width:12px;height:12px;border-radius:50%;border:2px solid #64748b;background:#1e293b}
.milestone.optimized::before{background:#22c55e;border-color:#22c55e}
.milestone.baseline::before{background:#64748b;border-color:#64748b}
.milestone .card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:18px 20px}
.milestone .card:hover{border-color:#475569}
.card summary{list-style:none;cursor:pointer;outline:none}
.card summary::-webkit-details-marker{display:none}
.card summary::marker{display:none;content:''}
.card .top{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:6px}
.card .ver{font-size:.75rem;font-weight:800;color:#f8fafc}
.card .date{font-size:.7rem;color:#475569}
.card .title{font-size:.95rem;font-weight:700;color:#f1f5f9;margin-bottom:6px}
.card .desc{font-size:.78rem;color:#94a3b8;line-height:1.6;margin-bottom:12px}
.card .expand-hint{font-size:.65rem;color:#475569;margin-top:4px}
.changes-title{font-size:.7rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.4px;margin-bottom:6px}
.changes-list{list-style:none;padding:0;margin:0 0 14px}
.changes-list li{font-size:.74rem;color:#cbd5e1;line-height:1.6;padding:3px 0 3px 16px;position:relative}
.changes-list li::before{content:'▸';position:absolute;left:0;color:#f59e0b;font-size:.65rem}
.delta-bar{display:flex;align-items:center;gap:12px;margin-bottom:12px;font-size:.72rem;flex-wrap:wrap}
.delta-bar .delta-item{display:flex;align-items:center;gap:4px;background:#0f172a;border-radius:4px;padding:4px 10px}
.delta-bar .delta-label{color:#64748b}
.delta-bar .delta-val{font-weight:700;font-variant-numeric:tabular-nums}
.delta-bar .delta-val.up{color:#22c55e}
.delta-bar .delta-val.down{color:#ef4444}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;font-size:.72rem}
@media(max-width:600px){.metrics{grid-template-columns:repeat(2,1fr)}}
.metrics .m{background:#0f172a;border-radius:4px;padding:6px 10px}
.metrics .m .mv{font-weight:700;font-variant-numeric:tabular-nums}
.metrics .m .ml{color:#64748b;font-size:.62rem}
.detail-section{margin-top:12px;padding-top:12px;border-top:1px solid #334155}
.detail-section h4{font-size:.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.3px;margin-bottom:8px}
.detail-table{width:100%;border-collapse:collapse;font-size:.7rem;margin-bottom:12px}
.detail-table th{background:#0f172a;padding:4px 8px;text-align:left;font-weight:600;color:#64748b;font-size:.6rem;text-transform:uppercase}
.detail-table td{padding:3px 8px;border-bottom:1px solid #1e293b}
.detail-table .n{text-align:right;font-variant-numeric:tabular-nums}
.detail-table tr:hover td{background:#0f172a}
.detail-badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:.6rem;font-weight:700}
.detail-badge.g{background:#064e3b;color:#22c55e}
.detail-badge.y{background:#78350f;color:#f59e0b}
.detail-badge.r{background:#7f1d1d;color:#ef4444}
table{width:100%;border-collapse:collapse;font-size:.78rem}
th{background:#334155;padding:6px 10px;text-align:left;font-weight:600;color:#94a3b8;font-size:.65rem;text-transform:uppercase;letter-spacing:.3px}
td{padding:5px 10px;border-bottom:1px solid #334155}
tr:hover td{background:#1e293b}
.n{text-align:right;font-variant-numeric:tabular-nums}
.hl td{font-weight:700;background:#450a0a}
.hl td:first-child{color:#ef4444}
.ll{color:#22c55e;font-weight:600}
.badge{display:inline-block;padding:1px 7px;border-radius:6px;font-size:.65rem;font-weight:700}
.badge.r{background:#fee2e2;color:#dc2626}
.badge.y{background:#ffedd5;color:#ea580c}
.badge.g{background:#dcfce7;color:#16a34a}
.note{font-size:.68rem;color:#94a3b8;margin-top:10px;padding:10px 14px;background:#0f172a;border-radius:6px;line-height:1.6}
.hist{font-size:.68rem;color:#94a3b8;margin-top:8px;text-align:center}
.ft{text-align:center;color:#475569;font-size:.65rem;padding:16px}
.ft a{color:#64748b}
"""


# ── History ────────────────────────────────────────────────────────────
HISTORY_FILE = PROJ / "benchmark_reports" / "history.json"


def _update_history(costs: dict):
    try:
        hist = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else {"runs": []}
    except Exception:
        hist = {"runs": []}
    hist["runs"].append({"ts": datetime.now(timezone.utc).isoformat(), "costs": costs})
    hist["runs"] = hist["runs"][-50:]
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(hist, indent=2))


def _history_html() -> str:
    try:
        if not HISTORY_FILE.exists():
            return ""
        hist = json.loads(HISTORY_FILE.read_text())
        runs = hist.get("runs", [])
        if len(runs) < 2:
            return (f'<div class="hist">{len(runs)} run recorded · '
                    f'<a href="history.json" style="color:#94a3b8">history.json</a></div>')
        cur = runs[-1]["costs"]
        prev = runs[-2]["costs"]

        def delta(k):
            try:
                cv = float(cur[k].replace("×", ""))
                pv = float(prev[k].replace("×", ""))
                d = cv - pv
                if abs(d) < 0.01:
                    return "—"
                return f"{'+' if d > 0 else ''}{d:.2f}"
            except Exception:
                return "?"
        return (f'<div class="hist">Δ prev → curr: insn {delta("insn")} | '
                f'static {delta("static")} · {len(runs)} runs · '
                f'<a href="history.json" style="color:#94a3b8">history.json</a></div>')
    except Exception:
        return ""


# ── SVG Version vs LLVM Progress Chart ──────────────────────────────────

OPTIMIZATION_HISTORY = PROJ / "benchmark_reports" / "optimization_history.json"


def _version_progress_chart(milestones: list, baseline_dyn: int) -> str:
    """Generate SVG horizontal bar chart showing each version's ratio vs LLVM.

    The goal is to show progress toward the LLVM baseline (1.0x).
    Each version gets a bar proportional to its vs_llvm_ratio.
    """
    if not milestones:
        return '<div class="note">No version data available in optimization_history.json.</div>'

    W, H = 720, 50 + len(milestones) * 50
    ML, MR, MT, MB = 110, 130, 10, 20
    PW = W - ML - MR
    bar_h = 26
    gap = 24

    ratios = [m.get("vs_llvm_ratio", 1) for m in milestones]
    max_ratio = max(ratios)
    chart_max = max(max_ratio * 1.15, 2.0)

    svg = f'<svg viewBox="0 0 {W} {H}" width="100%" height="auto" style="max-width:{W}px;font-family:system-ui,sans-serif">'

    # LLVM baseline at 1.0x
    baseline_x = ML + (1.0 / chart_max) * PW
    svg += f'\n  <line x1="{baseline_x:.0f}" y1="{MT}" x2="{baseline_x:.0f}" y2="{MT + len(milestones) * (bar_h + gap)}" stroke="#22c55e" stroke-dasharray="8,4" stroke-width="2" opacity="0.7"/>'
    svg += f'\n  <text x="{baseline_x:.0f}" y="{MT - 6}" text-anchor="middle" fill="#22c55e" font-size="10" font-weight="700">LLVM 1.0x</text>'

    # Target arrow at bottom right
    svg += f'\n  <text x="{baseline_x + 4:.0f}" y="{MT + len(milestones) * (bar_h + gap) + 14}" fill="#22c55e" font-size="10">🎯 LLVM baseline (目标)</text>'

    y = MT
    first_ratio = milestones[0].get("vs_llvm_ratio", 1)
    last_ratio = milestones[-1].get("vs_llvm_ratio", 1)

    for i, m in enumerate(milestones):
        ratio = m.get("vs_llvm_ratio", 1)
        bar_w = max(ratio / chart_max * PW, 3)

        # Color
        if ratio <= 1.5:
            color = "#22c55e"
        elif ratio <= 4:
            color = "#f59e0b"
        else:
            color = "#ef4444"

        # Version label
        ver = m.get("version", f"v{i}")
        svg += f'\n  <text x="{ML - 8}" y="{y + bar_h - 9}" text-anchor="end" fill="#e2e8f0" font-size="13" font-weight="600">{ver}</text>'

        # Bar with gradient look (lighter fill)
        svg += f'\n  <rect x="{ML}" y="{y}" width="{bar_w:.0f}" height="{bar_h}" rx="4" fill="{color}" opacity="0.8"/>'

        # Ratio text
        svg += f'\n  <text x="{ML + bar_w + 8:.0f}" y="{y + bar_h - 9}" fill="{color}" font-size="13" font-weight="700">{ratio:.2f}x</text>'

        # Dynamic insns on right
        dyn = m.get("dynamic_insns", 0)
        svg += f'\n  <text x="{ML + PW + 4}" y="{y + bar_h - 9}" fill="#64748b" font-size="11">{_f(dyn)}</text>'

        # Delta from previous version
        if i > 0:
            prev_ratio = milestones[i - 1].get("vs_llvm_ratio", 1)
            delta = prev_ratio - ratio
            if delta > 0:
                delta_str = f"-{delta:.2f}x"
                delta_color = "#22c55e"
            else:
                delta_str = f"+{-delta:.2f}x"
                delta_color = "#ef4444"
            svg += f'\n  <text x="{ML + bar_w + 80:.0f}" y="{y + bar_h - 9}" fill="{delta_color}" font-size="11">{delta_str}</text>'

        y += bar_h + gap

    # Progress summary
    total_reduction = first_ratio - last_ratio
    pct_done = (first_ratio - last_ratio) / max(first_ratio - 1.0, 0.01) * 100
    gap_remaining = last_ratio - 1.0

    ly = y + 4
    svg += f'\n  <text x="{ML}" y="{ly}" fill="#94a3b8" font-size="11">Progress: {first_ratio:.1f}x → {last_ratio:.1f}x ({total_reduction:+.1f}x, {pct_done:.0f}% toward LLVM) · Gap remaining: {gap_remaining:.1f}x</text>'

    svg += '\n</svg>'
    return svg


CATEGORIES = ["alu_r", "alu_i", "fp", "shift", "load", "store", "branch", "jump", "upper"]

CAT_LABELS = {
    "alu_r": "ALU R", "alu_i": "ALU I", "fp": "FP", "shift": "Shift",
    "load": "Load", "store": "Store", "branch": "Branch", "jump": "Jump",
    "upper": "Upper",
}


def _op_category_table(aggregates: dict) -> str:
    """Generate per-operator instruction category breakdown with SVG % bars.

    Shows the top categories for each operator type, with percentages for
    both compilers side-by-side.
    """
    if not aggregates:
        return ""

    h = '<div style="margin-top:20px"><h3 style="font-size:.82rem;font-weight:700;color:#f1f5f9;margin-bottom:12px">Per-Operator Instruction Type Breakdown</h3>'
    h += '<div class="sec-sub">Category distribution (% of total dynamic instructions) per operator type — compare compiler instruction mix patterns</div>'

    for op_name in ["conv", "gemm", "maxpool", "relu", "sigmoid"]:
        agg = aggregates.get(op_name)
        if not agg:
            continue

        sv_pct = agg.get("sv_category_pct", {})
        llvm_pct = agg.get("llvm_category_pct", {})

        # Top categories by SV %
        top_cats = sorted(sv_pct.keys(), key=lambda c: max(sv_pct.get(c, 0), llvm_pct.get(c, 0)), reverse=True)[:5]

        h += f'<details style="margin-bottom:10px"><summary style="cursor:pointer;font-size:.75rem;font-weight:600;color:#e2e8f0;padding:6px 10px;background:#0f172a;border-radius:6px">{op_name} ({agg.get("model_count", 0)} ops, ratio: {agg.get("dynamic_ratio", 0):.2f}x)</summary>'
        h += '<div style="padding:8px 0"><table class="detail-table"><tr><th>Category</th><th class="n">SV %</th><th>SV bar</th><th class="n">LLVM %</th><th>LLVM bar</th></tr>'

        for cat in sorted(CATEGORIES, key=lambda c: max(sv_pct.get(c, 0), llvm_pct.get(c, 0)), reverse=True):
            sv_v = sv_pct.get(cat, 0)
            ll_v = llvm_pct.get(cat, 0)
            if sv_v == 0 and ll_v == 0:
                continue
            label = CAT_LABELS.get(cat, cat)
            max_w = 120
            sv_w = max(1, int(sv_v / 50 * max_w)) if sv_v > 0 else 0
            ll_w = max(1, int(ll_v / 50 * max_w)) if ll_v > 0 else 0
            sv_color = "#f59e0b" if sv_v > ll_v else "#22c55e"
            ll_color = "#3b82f6"
            h += f'<tr><td>{label}</td>'
            h += f'<td class="n">{sv_v:.1f}%</td>'
            h += f'<td><svg width="{max_w}" height="14"><rect x="0" y="2" width="{sv_w}" height="10" rx="2" fill="{sv_color}" opacity="0.7"/></svg></td>'
            h += f'<td class="n">{ll_v:.1f}%</td>'
            h += f'<td><svg width="{max_w}" height="14"><rect x="0" y="2" width="{ll_w}" height="10" rx="2" fill="{ll_color}" opacity="0.7"/></svg></td>'
            h += '</tr>'

        h += '</table></div></details>'

    h += '<div style="font-size:.65rem;color:#64748b;margin-top:6px">Orange = ScratchV dominant · Green = SV lower · Blue = LLVM. Longer bar = higher % of total instructions.</div>'
    h += '</div>'
    return h


# ── SVG Operator Bar Chart ──────────────────────────────────────────────


def _op_bar_chart(aggregates: dict) -> str:
    """Generate SVG horizontal bar chart for operator-type comparison.

    Shows ScratchV/LLVM dynamic instruction ratio per operator type.
    Lower = better (target: <1.5x).
    """
    if not aggregates:
        return '<div class="note">No single-operator data available. Run `make bench-single-ops` first.</div>'

    # Sort by ratio descending
    items = sorted(aggregates.items(), key=lambda x: x[1].get("dynamic_ratio", 0), reverse=True)

    W, H = 700, 40 + len(items) * 44
    ML, MR, MT, MB = 110, 120, 10, 20
    PW = W - ML - MR
    bar_h = 24
    gap = 20

    # Find max ratio for scaling (cap at 5x)
    max_ratio = max((v.get("dynamic_ratio", 0) for _, v in items), default=1)
    chart_max = max(max_ratio * 1.2, 1.5)

    svg = f'<svg viewBox="0 0 {W} {H}" width="100%" height="auto" style="max-width:{W}px;font-family:system-ui,sans-serif">'

    y = MT
    for name, data in items:
        ratio = data.get("dynamic_ratio", 0)
        bar_w = max(ratio / chart_max * PW, 2) if ratio > 0 else 0

        # Color coding
        if ratio <= 1.5:
            color = "#22c55e"
        elif ratio <= 4:
            color = "#f59e0b"
        else:
            color = "#ef4444"

        # Label (left)
        svg += f'\n  <text x="{ML - 8}" y="{y + bar_h - 8}" text-anchor="end" fill="#e2e8f0" font-size="13" font-weight="600">{name}</text>'

        # Bar
        svg += f'\n  <rect x="{ML}" y="{y}" width="{bar_w:.0f}" height="{bar_h}" rx="4" fill="{color}" opacity="0.85"/>'

        # Ratio text on bar
        ratio_text = f"{ratio:.2f}x"
        svg += f'\n  <text x="{ML + bar_w + 6:.0f}" y="{y + bar_h - 8}" fill="{color}" font-size="12" font-weight="700">{ratio_text}</text>'

        # LLVM baseline line at 1.0x
        baseline_x = ML + (1.0 / chart_max) * PW
        if ratio > 1.0:
            svg += f'\n  <line x1="{baseline_x:.0f}" y1="{y}" x2="{baseline_x:.0f}" y2="{y + bar_h}" stroke="#22c55e" stroke-dasharray="3,3" stroke-width="1" opacity="0.5"/>'

        # Dynamic insn count (right side, small)
        sv_dyn = data.get("total_scratchv_dynamic_insns", 0)
        svg += f'\n  <text x="{ML + PW + 6}" y="{y + bar_h - 8}" fill="#64748b" font-size="10">{_f(sv_dyn)}</text>'

        y += bar_h + gap

    # Legend
    ly = y + 8
    svg += f'\n  <text x="{ML}" y="{ly}" fill="#64748b" font-size="10">Bar width = ScratchV / LLVM dynamic instruction ratio · Green &lt;1.5x · Yellow 1.5–4x · Red &gt;4x</text>'
    svg += f'\n  <text x="{ML}" y="{ly + 16}" fill="#475569" font-size="9">Green dashed line = LLVM baseline (1.0x) · Numbers on right = ScratchV dynamic instruction count</text>'

    svg += '\n</svg>'
    return svg


# ── KPI Cards ──────────────────────────────────────────────────────────


def _kpi_cards(milestones: list) -> str:
    """Generate 4-card KPI grid from milestone data."""
    if not milestones or len(milestones) < 1:
        return ""

    first = milestones[0]
    last = milestones[-1]

    dyn_reduction = (first["dynamic_insns"] - last["dynamic_insns"]) / max(first["dynamic_insns"], 1) * 100
    gap_shrink = first["vs_llvm_ratio"] - last["vs_llvm_ratio"]
    time_reduction = (first.get("time_100mhz_s", 0) - last.get("time_100mhz_s", 0)) / max(first.get("time_100mhz_s", 1), 1) * 100
    current_ratio = last["vs_llvm_ratio"]

    def _color_green(v):
        return "g" if v >= 0 else "r"

    def _color_ratio(r):
        if r <= 1.5:
            return "g"
        elif r <= 4:
            return "y"
        return "r"

    h = '<div class="kpi-grid">'
    h += f'<div class="kpi"><div class="v g">−{dyn_reduction:.0f}%</div><div class="l">动态指令减少</div></div>'
    h += f'<div class="kpi"><div class="v {_color_green(gap_shrink)}">{gap_shrink:.1f}x</div><div class="l">vs LLVM 差距缩小</div></div>'
    h += f'<div class="kpi"><div class="v {_color_green(time_reduction)}">−{time_reduction:.0f}%</div><div class="l">推理时间减少 @100MHz</div></div>'
    h += f'<div class="kpi"><div class="v {_color_ratio(current_ratio)}">{current_ratio:.2f}x</div><div class="l">当前 vs LLVM</div></div>'
    h += '</div>'
    return h


# ── Trend SVG Chart ────────────────────────────────────────────────────


def _trend_chart_svg(milestones: list, baseline_dyn: int) -> str:
    """Generate SVG line chart with JS version toggle — dynamic instructions over versions."""
    if not milestones:
        return ""

    W, H = 800, 360
    ML, MR, MT, MB = 100, 40, 24, 56

    # Y-axis range: 0 to max_dyn * 1.1, at least 8B
    max_dyn = max(m.get("dynamic_insns", 0) for m in milestones)
    y_max = max(max_dyn * 1.08, 8_000_000_000)
    chart_h = H - MT - MB

    def y_pos(dyn):
        return MT + chart_h - (dyn / y_max) * chart_h

    # Baseline line
    baseline_y = y_pos(baseline_dyn) if baseline_dyn else y_pos(1_059_548_774)

    svg = f'<svg id="chart-svg" viewBox="0 0 {W} {H}" width="100%" height="auto" style="max-width:{W}px;font-family:system-ui,sans-serif">'

    # LLVM baseline dashed line
    svg += f'\n  <line x1="{ML}" y1="{baseline_y}" x2="{W - MR}" y2="{baseline_y}" stroke="#22c55e" stroke-dasharray="6,4" stroke-width="1.5" opacity="0.6"/>'
    svg += f'\n  <text x="{ML - 10}" y="{baseline_y + 4}" text-anchor="end" fill="#22c55e" font-size="11" font-weight="700">LLVM {baseline_dyn / 1e9:.2f}B</text>'

    # Y-axis grid lines and labels (0 to y_max in 1B steps)
    for b in range(1, int(y_max / 1e9) + 1):
        gy = y_pos(b * 1_000_000_000)
        svg += f'\n  <line x1="{ML}" y1="{gy}" x2="{W - MR}" y2="{gy}" stroke="#334155" stroke-width="1"/>'
        svg += f'\n  <text x="{ML - 8}" y="{gy + 4}" text-anchor="end" fill="#64748b" font-size="10">{b}.00B</text>'

    # X and Y axis lines
    svg += f'\n  <line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + chart_h}" stroke="#475569" stroke-width="1"/>'
    svg += f'\n  <line x1="{ML}" y1="{MT + chart_h}" x2="{W - MR}" y2="{MT + chart_h}" stroke="#475569" stroke-width="1"/>'

    # Scatter plot
    num_versions = len(milestones)
    points = []
    for i, m in enumerate(milestones):
        dyn = m.get("dynamic_insns", 0)
        ratio = m.get("vs_llvm_ratio", 0)
        if num_versions > 1:
            x = ML + i * (W - ML - MR) / (num_versions - 1)
        else:
            x = ML + (W - ML - MR) / 2
        cy = y_pos(dyn)
        points.append((x, cy, dyn, ratio, m.get("version", f"v{i}")))

    # Polyline connecting all points
    pts_str = " ".join(f"{x:.1f},{cy:.1f}" for x, cy, _, _, _ in points)
    svg += f'\n  <polyline class="sv-polyline" points="{pts_str}" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linejoin="round"/>'

    # Data points, labels, version labels, ratio labels
    label_offsets = []
    for i, (x, cy, dyn, ratio, ver) in enumerate(points):
        # Dot
        svg += f'\n  <circle class="data-point" data-index="{i}" cx="{x:.1f}" cy="{cy:.1f}" r="5" fill="#f59e0b" stroke="#0f172a" stroke-width="2"/>'
        # Dynamic insn label above dot
        dy_label = -10
        # Avoid overlap with previous label
        for px, py in label_offsets:
            if abs(x - px) < 80 and abs(cy + dy_label - py) < 16:
                dy_label -= 18
        label_offsets.append((x, cy + dy_label))
        svg += f'\n  <text class="data-label" data-index="{i}" x="{x:.1f}" y="{cy + dy_label:.1f}" text-anchor="middle" fill="#f59e0b" font-size="11" font-weight="700">{dyn / 1e9:.2f}B</text>'
        # Version label below
        svg += f'\n  <text class="ver-label" data-index="{i}" x="{x:.1f}" y="{MT + chart_h + 20}" text-anchor="middle" fill="#94a3b8" font-size="11">{ver}</text>'
        # Ratio label
        svg += f'\n  <text class="ratio-label" data-index="{i}" x="{x:.1f}" y="{MT + chart_h + 38}" text-anchor="middle" fill="#64748b" font-size="10">{ratio:.1f}x vs LLVM</text>'

    svg += '\n</svg>'

    # JSON data for JS
    json_data = [{"index": i, "version": m.get("version", f"v{i}"),
                   "dynamic_insns": m.get("dynamic_insns", 0),
                   "vs_llvm_ratio": m.get("vs_llvm_ratio", 0)} for i, m in enumerate(milestones)]

    # Version selector + JS
    h = '<div class="chart-section"><h2>动态指令趋势</h2>'
    h += f'<div class="chart-sub">绿色虚线 = LLVM float32 baseline ({baseline_dyn / 1e9:.2f}B) | 橙色折线 = ScratchV 各版本 | 勾选版本框进行对比</div>'

    # Version checkboxes
    h += '<div class="version-selector">'
    for i, m in enumerate(milestones):
        ver = m.get("version", f"v{i}")
        tag = m.get("tag", "")
        dot_color = "#22c55e" if tag == "optimized" else "#64748b"
        checked = " checked" if i >= len(milestones) - 2 else ""  # default: show last 2 + baseline
        h += f'<label class="vs-label checked"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{dot_color};margin-right:2px"></span><input type="checkbox" class="ver-cb" data-index="{i}"{checked}>{ver}</label>'
    h += '<label id="btn-select-all" style="background:transparent;border:none;font-size:.65rem;color:#64748b;cursor:pointer">全选</label>'
    h += '<label id="btn-select-optimized" style="background:transparent;border:none;font-size:.65rem;color:#64748b;cursor:pointer">仅优化版</label>'
    h += '</div>'

    h += f'<div class="chart-container">{svg}</div>'
    h += f'<script id="chart-data" type="application/json" data-baseline="{baseline_dyn}">{json.dumps(json_data)}</script>'

    # Legend
    h += '<div class="chart-legend">'
    h += '<div class="leg-item"><span class="leg-line" style="background:#22c55e;border:1px dashed #22c55e"></span> LLVM baseline</div>'
    h += '<div class="leg-item"><span class="leg-dot" style="background:#f59e0b"></span> ScratchV</div>'
    h += '</div></div>'

    # JS for interactivity
    h += """<script>
(function() {
  const chartSvg = document.getElementById('chart-svg');
  if (!chartSvg) return;
  const chartDataEl = document.getElementById('chart-data');
  if (!chartDataEl) return;
  let milestones;
  try { milestones = JSON.parse(chartDataEl.textContent); } catch(e) { return; }

  function getAllCBs() { return document.querySelectorAll('.ver-cb'); }
  function setAllIndexed(index, checked) {
    getAllCBs().forEach(cb => { if (parseInt(cb.dataset.index) === index) cb.checked = checked; });
  }

  function redrawChart() {
    const selected = new Set();
    getAllCBs().forEach(cb => { if (cb.checked) selected.add(parseInt(cb.dataset.index)); });
    const circles = chartSvg.querySelectorAll('.data-point');
    const labels = chartSvg.querySelectorAll('.data-label');
    const verLabels = chartSvg.querySelectorAll('.ver-label');
    const ratioLabels = chartSvg.querySelectorAll('.ratio-label');
    const polyline = chartSvg.querySelector('.sv-polyline');
    const visiblePoints = [];

    circles.forEach(c => {
      const idx = parseInt(c.dataset.index);
      if (selected.has(idx)) { c.style.display = ''; visiblePoints.push({x: c.getAttribute('cx'), y: c.getAttribute('cy')}); }
      else { c.style.display = 'none'; }
    });
    labels.forEach(l => { l.style.display = selected.has(parseInt(l.dataset.index)) ? '' : 'none'; });
    verLabels.forEach(l => { l.style.display = selected.has(parseInt(l.dataset.index)) ? '' : 'none'; });
    ratioLabels.forEach(l => { l.style.display = selected.has(parseInt(l.dataset.index)) ? '' : 'none'; });

    if (polyline && visiblePoints.length >= 2) {
      polyline.setAttribute('points', visiblePoints.map(p => p.x + ',' + p.y).join(' '));
      polyline.style.display = '';
    } else if (polyline) { polyline.style.display = 'none'; }

    document.querySelectorAll('.vs-label').forEach(lbl => {
      const cb = lbl.querySelector('.ver-cb');
      if (cb) { if (cb.checked) lbl.classList.add('checked'); else lbl.classList.remove('checked'); }
    });
  }

  getAllCBs().forEach(cb => {
    cb.addEventListener('change', function() {
      setAllIndexed(parseInt(this.dataset.index), this.checked);
      redrawChart();
    });
  });

  const btnAll = document.getElementById('btn-select-all');
  const btnOpt = document.getElementById('btn-select-optimized');
  if (btnAll) btnAll.addEventListener('click', function(e) { e.preventDefault(); getAllCBs().forEach(cb => { cb.checked = true; }); redrawChart(); });
  if (btnOpt) btnOpt.addEventListener('click', function(e) { e.preventDefault(); getAllCBs().forEach(cb => { const idx = parseInt(cb.dataset.index); cb.checked = milestones[idx] && milestones[idx].vs_llvm_ratio <= 3.0; }); redrawChart(); });
  redrawChart();
})();
</script>"""
    return h


# ── Timeline ───────────────────────────────────────────────────────────


def _timeline_html(milestones: list, baseline_dyn: int = 0) -> str:
    """Generate expandable timeline cards with per-version instruction breakdown.

    Args:
        milestones: List of milestone dicts from optimization_history.json
        baseline_dyn: LLVM baseline dynamic instruction count for gap calculation
    """
    if not milestones:
        return ""

    if baseline_dyn <= 0:
        baseline_dyn = 1_059_548_774  # fallback default

    h = '<h2 style="margin-bottom:16px">优化时间线 — 点击展开查看详情</h2>'
    h += '<div class="timeline">'

    CAT_LABELS_REV = {
        "alu_r": "ALU R-type", "alu_i": "ALU I-type", "fp": "FP", "shift": "Shift",
        "load": "Load", "store": "Store", "branch": "Branch", "jump": "Jump",
        "upper": "Upper imm.",
    }

    for i, m in enumerate(milestones):
        ver = m.get("version", f"v{i}")
        tag = m.get("tag", "baseline")
        date = m.get("date", "")
        title = m.get("title", "")
        desc = m.get("description", "")
        changes = m.get("changes", [])
        dyn = m.get("dynamic_insns", 0)
        ratio = m.get("vs_llvm_ratio", 0)
        time_s = m.get("time_100mhz_s", 0)
        per_mac = m.get("per_mac_insns", 0)
        breakdown = m.get("instruction_breakdown", {})
        cats = breakdown.get("categories", {})
        per_op = breakdown.get("per_operator", {})
        total_dyn = breakdown.get("dynamic_insns", dyn)

        # Delta from previous version
        delta_html = ""
        if i > 0:
            prev = milestones[i - 1]
            prev_dyn = prev.get("dynamic_insns", 1)
            prev_ratio = prev.get("vs_llvm_ratio", 1)
            dyn_delta = (dyn - prev_dyn) / max(prev_dyn, 1) * 100
            ratio_delta = prev_ratio - ratio
            time_delta = (time_s - prev.get("time_100mhz_s", 1)) / max(prev.get("time_100mhz_s", 1), 1) * 100
            dyn_sign = "up" if dyn_delta < 0 else "down"
            ratio_sign = "up" if ratio_delta > 0 else "down"
            time_sign = "up" if time_delta < 0 else "down"
            delta_html = f'<div class="delta-bar"><span style="color:#64748b;font-weight:600">较上一版:</span>'
            delta_html += f'<span class="delta-item"><span class="delta-label">动态指令</span><span class="delta-val {dyn_sign}">{dyn_delta:+.1f}%</span></span>'
            delta_html += f'<span class="delta-item"><span class="delta-label">vs LLVM</span><span class="delta-val {ratio_sign}">{ratio_delta:+.2f}x</span></span>'
            delta_html += f'<span class="delta-item"><span class="delta-label">推理时间</span><span class="delta-val {time_sign}">{time_delta:+.1f}%</span></span>'
            delta_html += '</div>'

        # Gap vs LLVM
        gap_dyn = dyn - baseline_dyn
        gap_color = "#ef4444" if ratio > 4 else ("#f59e0b" if ratio > 1.5 else "#22c55e")

        # Milestone class
        ms_class = "milestone optimized" if tag == "optimized" else "milestone baseline"

        # Open first (latest) card by default
        open_attr = " open" if i == len(milestones) - 1 else ""

        # Expand hint
        expand_text = "▾ 点击收起详情" if i == len(milestones) - 1 else "▸ 点击展开详情（指令分类 + 算子对比）"

        h += f'<div class="{ms_class}"><div class="card"><details{open_attr}><summary>'
        h += f'<div class="top"><span class="ver">{ver}</span><span class="date">{date}</span></div>'
        h += f'<div class="title">{title}</div>'
        h += f'<div class="desc">{desc}</div>'
        h += f'<div class="expand-hint">{expand_text}</div>'
        h += '</summary>'

        # Changes list
        if changes:
            h += '<div class="changes-title">优化要点</div><ul class="changes-list">'
            for c in changes:
                h += f'<li>{c}</li>'
            h += '</ul>'

        h += delta_html

        # Metrics grid
        h += '<div class="metrics">'
        h += f'<div class="m"><div class="mv">{dyn / 1e9:.2f}B</div><div class="ml">动态指令</div></div>'
        h += f'<div class="m"><div class="mv">{ratio:.2f}x</div><div class="ml">vs LLVM 比值</div></div>'
        h += f'<div class="m"><div class="mv">{per_mac} instr/MAC</div><div class="ml">内层循环效率</div></div>'
        h += f'<div class="m"><div class="mv">{time_s:.1f}s</div><div class="ml">@100MHz 推理时间</div></div>'
        h += '</div>'

        # Gap summary
        h += f'<div style="margin-top:10px;font-size:.7rem;color:#64748b">与 LLVM 差距: <span style="color:{gap_color};font-weight:700">{gap_dyn / 1e9:.2f}B</span> 条指令 (<span style="color:{gap_color};font-weight:700">{ratio:.2f}x</span>)</div>'

        # Detail section: instruction breakdown
        h += '<div class="detail-section"><h4>Dynamic Instruction Breakdown</h4>'
        h += '<table class="detail-table"><tr><th>Category</th><th class="n">Count</th><th class="n">%</th></tr>'
        for cat_key in ["alu_r", "alu_i", "fp", "shift", "load", "store", "branch", "jump", "upper"]:
            count = cats.get(cat_key, 0)
            if count == 0:
                h += f'<tr><td>{CAT_LABELS_REV.get(cat_key, cat_key)}</td><td class="n">0</td><td class="n">0.0%</td></tr>'
            else:
                pct = count / max(total_dyn, 1) * 100
                count_str = f"{count / 1e6:.1f}M" if count >= 1e6 else f"{count:,}"
                h += f'<tr><td>{CAT_LABELS_REV.get(cat_key, cat_key)}</td><td class="n">{count_str}</td><td class="n">{pct:.1f}%</td></tr>'
        total_count_str = f"{total_dyn / 1e9:.2f}B" if total_dyn >= 1e9 else f"{total_dyn:,}"
        h += f'<tr style="font-weight:700"><td>Total</td><td class="n">{total_count_str}</td><td class="n">100%</td></tr>'
        h += '</table>'

        # Per-operator breakdown
        if per_op:
            h += '<h4 style="margin-top:12px">Per-Operator Type</h4>'
            h += '<table class="detail-table"><tr><th>Op Type</th><th class="n">Dynamic Insns</th><th class="n">vs LLVM</th></tr>'
            for op_name in ["conv", "gemm", "maxpool", "relu", "sigmoid"]:
                op_data = per_op.get(op_name)
                if not op_data:
                    continue
                op_dyn = op_data.get("dynamic_insns", 0)
                op_ratio = op_data.get("ratio", 0)
                badge_class = "g" if op_ratio <= 1.5 else ("y" if op_ratio <= 4 else "r")
                op_dyn_str = f"{op_dyn / 1e9:.2f}B" if op_dyn >= 1e9 else (f"{op_dyn / 1e6:.1f}M" if op_dyn >= 1e6 else f"{op_dyn:,}")
                h += f'<tr><td>{op_name}</td><td class="n">{op_dyn_str}</td><td class="n"><span class="detail-badge {badge_class}">{op_ratio:.2f}x</span></td></tr>'
            h += '</table>'

        h += '</div></details></div></div>'

    h += '</div>'
    return h


# ═══════════════════════════════════════════════════════════════════════


def generate(ld=None, single_op_data=None, history_data=None):
    if ld is None:
        ld = collect()
    ld = ld or {}

    L = ld.get("llvm", {})
    S = ld.get("scratchv", {})

    Ld = L.get("dynamic_instructions", {})
    Sd = S.get("dynamic_instructions", {})

    Lt = Ld.get("total", 0)
    St = Sd.get("total", 0)

    # Static instruction counts
    Ls = L.get("static_insns", 0)
    Ss = S.get("static_insns", 0)

    R_insn = _ratio(St, Lt)
    R_static = "—"

    # History
    _update_history({"insn": R_insn, "static": R_static})

    # Load history data for version progress
    if history_data is None and OPTIMIZATION_HISTORY.exists():
        try:
            history_data = json.loads(OPTIMIZATION_HISTORY.read_text())
        except Exception:
            history_data = None

    milestones = history_data.get("milestones", []) if history_data else []
    baseline_dyn = history_data.get("baseline", {}).get("dynamic_insns", 0) if history_data else 0

    # ── Build HTML ──
    h = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScratchV · Dashboard</title><style>{CSS}</style></head><body><div class="wrap">
<div class="hdr"><h1>ScratchV vs LLVM — 追赶 LLVM 进度</h1>
<div class="sub"><b>LLVM RV64FD (float32)</b> — baseline (target ≤1.0x) &nbsp;|&nbsp; <b>ScratchV RV32IM (Q16.16)</b> — {R_insn} current</div></div>

{_kpi_cards(milestones)}
{_history_html()}"""

    # ── Section 1: Version vs LLVM Progress ──
    if milestones:
        h += """
<div class="sec"><h2>1. Version vs LLVM Progress · 版本追赶进度</h2>
<div class="sec-sub">Each version's dynamic instruction ratio vs LLVM baseline (1.0x = LLVM). Goal: ≤1.0x (beat LLVM).</div>"""
        h += _version_progress_chart(milestones, baseline_dyn)
        h += "</div>"

    # ── Trend Chart (line chart with JS interactivity) ──
    if milestones:
        h += _trend_chart_svg(milestones, baseline_dyn)

    # ── Section 2: Dynamic Instruction Distribution ──
    h += """
<div class="sec"><h2>2. Dynamic Instruction Distribution · 指令粒度</h2>
<div class="sec-sub">Per-category breakdown with percentages — identifies compilation instruction bottlenecks</div><table>
<tr><th>Category</th><th class="n">LLVM</th><th class="n">LLVM %</th><th class="n">ScratchV</th><th class="n">SV %</th><th class="n">SV/LLVM</th></tr>"""

    rows = [
        ("ALU R-type", "alu_r", "alu_r"),
        ("ALU I-type", "alu_i", "alu_i"),
        ("FP", "fp", "fp"),
        ("Shift", "shift", "shift"),
        ("Load", "load", "load"),
        ("Store", "store", "store"),
        ("Branch", "branch", "branch"),
        ("Jump", "jump", "jump"),
        ("Upper immediate", "upper", "upper"),
    ]
    max_ratio_cat = ("", 0)
    for name, lk, sk in rows:
        lv = Ld.get(lk, 0)
        sv = Sd.get(sk, 0)
        if lv or sv:
            r = _ratio(sv, lv) if lv else "—"
            r_float = _ratio_float(sv, lv) if lv else 0
            if r_float > max_ratio_cat[1]:
                max_ratio_cat = (name, r_float)
            lv_pct = lv / max(Lt, 1) * 100
            sv_pct = sv / max(St, 1) * 100
            h += f"<tr><td>{name}</td><td class='n'><span class='ll'>{_f(lv)}</span></td><td class='n'>{lv_pct:.1f}%</td><td class='n'>{_f(sv)}</td><td class='n'>{sv_pct:.1f}%</td><td class='n'>{_badge(r)}</td></tr>"

    h += f"""<tr class="hl"><td><b>Total</b></td><td class='n'><b class='ll'>{_f(Lt)}</b></td><td class='n'>100%</td><td class='n'><b>{_f(St)}</b></td><td class='n'>100%</td><td class='n'><b>{_badge(R_insn)}</b></td></tr></table>
<div class="note"><b>Biggest bottleneck:</b> {max_ratio_cat[0]} ({max_ratio_cat[1]:.2f}x vs LLVM). "
Store ratio {_ratio(Sd.get('store', 0), Ld.get('store', 0))}: LLVM keeps accumulators in FP registers (few stores). ScratchV spills to stack every MAC due to limited registers.</div></div>"""

    # ── Section 3: Operator Granularity ──
    h += """
<div class="sec"><h2>3. Operator Comparison · 算子粒度</h2>
<div class="sec-sub">Per-operator-type dynamic instruction ratio (ScratchV / LLVM) — identifies which operator types have the most optimization headroom</div>"""

    # Load single-op benchmark data
    if single_op_data is None and SINGLE_OP_BENCH.exists():
        try:
            single_op_data = json.loads(SINGLE_OP_BENCH.read_text())
        except Exception:
            single_op_data = None

    if single_op_data:
        aggregates = single_op_data.get("aggregates", {})
        h += _op_bar_chart(aggregates)

        # Per-operator instruction type breakdown
        h += _op_category_table(aggregates)

        # Add per-model breakdown table
        models = single_op_data.get("models", {})
        if models:
            h += '<table style="margin-top:16px"><tr><th>Model</th><th>Op Type</th><th class="n">SV Static</th><th class="n">SV Dynamic</th><th class="n">LLVM Dynamic</th><th class="n">Ratio</th></tr>'
            for model_name in sorted(models.keys()):
                m = models[model_name]
                r = m.get("dynamic_ratio", 0)
                ratio_str = f"{r:.2f}x" if r else "—"
                h += f"<tr><td>{model_name}</td><td>{m['op_type']}</td><td class='n'>{m.get('scratchv_static_insns', 0)}</td><td class='n'>{_f(m.get('scratchv_dynamic_insns', 0))}</td><td class='n'>{_f(m.get('llvm_dynamic_insns', 0))}</td><td class='n'>{_badge(ratio_str)}</td></tr>"
            h += "</table>"
    else:
        h += '<div class="note">No single-operator data available. Run <code>make bench-single-ops</code> to generate per-operator benchmarks.</div>'

    h += "</div>"

    # ── Section 4: Optimization Timeline ──
    if milestones:
        h += '<div class="sec"><h2>4. Optimization Timeline · 优化时间线</h2>'
        h += '<div class="sec-sub">Click to expand each version — instruction category breakdown with percentages and per-operator comparison</div>'
        h += _timeline_html(milestones, baseline_dyn)
        h += '</div>'

    # ── Footer ──
    h += f"""
<div class="ft">ScratchV CI · LLVM RV64FD baseline · <a href="https://github.com/ScratchV-Compiler/ScratchV">GitHub</a> · <a href="history.html">Optimization History</a> · Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</div></body></html>"""
    return h


def generate_dashboard_html(json_path="", json_data=None, embed_json=False, title="ScratchV"):
    """Backward-compatible entry point called by ci_benchmark.py."""
    ld = None
    single_op_data = None
    history_data = None

    if json_data and isinstance(json_data, dict):
        ld = json_data
    elif json_path and os.path.exists(json_path):
        with open(json_path) as f:
            ld = json.load(f)

    # Load single-op data if available
    if SINGLE_OP_BENCH.exists():
        try:
            single_op_data = json.loads(SINGLE_OP_BENCH.read_text())
        except Exception:
            pass

    # Load optimization history
    if OPTIMIZATION_HISTORY.exists():
        try:
            history_data = json.loads(OPTIMIZATION_HISTORY.read_text())
        except Exception:
            pass

    return generate(ld, single_op_data, history_data)


def main():
    import argparse
    p = argparse.ArgumentParser(description="ScratchV vs LLVM instruction dashboard")
    p.add_argument("--llvm-json", help="Path to llvm_cache_compare JSON output")
    p.add_argument("--single-op-json", default=str(SINGLE_OP_BENCH),
                   help="Path to single_op_bench.json")
    p.add_argument("--history-json", default=str(OPTIMIZATION_HISTORY),
                   help="Path to optimization_history.json")
    p.add_argument("-o", "--output", default="benchmark_reports/dashboard.html")
    p.add_argument("--run", action="store_true",
                   help="Auto-collect data via subprocess calls")
    a = p.parse_args()

    ld = None
    single_op_data = None
    history_data = None

    if a.llvm_json and os.path.exists(a.llvm_json):
        with open(a.llvm_json) as f:
            ld = json.load(f)

    if os.path.exists(a.single_op_json):
        try:
            single_op_data = json.loads(open(a.single_op_json).read())
        except Exception:
            pass

    if os.path.exists(a.history_json):
        try:
            history_data = json.loads(open(a.history_json).read())
        except Exception:
            pass

    if a.run or ld is None:
        print("collecting data...", file=sys.stderr)
        ld = collect()

    html = generate(ld, single_op_data, history_data)
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w") as f:
        f.write(html)
    print(f"→ {a.output} ({len(html):,}B)", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
