"""Performance comparison — LLVM baseline vs ScratchV."""

from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

PROJ = Path(__file__).resolve().parent.parent.parent

def _run(script, args):
    """Run tool, return parsed JSON. Reads from the --json-output path in args."""
    tmp = None
    for i, a in enumerate(args):
        if a == "--json-output" and i+1 < len(args):
            tmp = args[i+1]
            break
        if a == "--json" and i+1 < len(args) and not args[i+1].startswith("--"):
            tmp = args[i+1]
            break
    if not tmp:
        tmp = "/tmp/_bench_tmp.json"
        args = args + ["--json-output", tmp]
    subprocess.run([sys.executable, str(PROJ/script)] + args,
                   capture_output=True, cwd=str(PROJ), timeout=60)
    if os.path.exists(tmp):
        with open(tmp) as f: return json.load(f)
    return {}


def collect():
    llvm_path = "/tmp/_llvm_bench.json"
    tf_path   = "/tmp/_tf_bench.json"
    return (
        _run("scratchv/standalone/llvm_cache_compare.py", ["--json-output", llvm_path]),
        _run("scratchv/standalone/tinyfive_compare.py", ["--json", tf_path]),
    )

def _f(n):
    if not n: return "—"
    if isinstance(n, float): n = int(n)
    return f"{n:,}"

def _ratio(sv, ll):
    """ScratchV / LLVM. Always appends × suffix."""
    if not ll: return "—"
    v = sv / ll
    return f"{v:.1f}×" if v >= 10 else f"{v:.2f}×"

CSS = """*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#f8fafc;color:#1e293b;line-height:1.5}
.wrap{max-width:920px;margin:0 auto;padding:20px}
.hdr{background:linear-gradient(135deg,#0f172a,#1e293b);color:#f1f5f9;padding:20px 28px;border-radius:10px;margin-bottom:18px}
.hdr h1{font-size:1.15rem}
.hdr .sub{font-size:.75rem;color:#94a3b8;margin-top:3px}
.hdr .sub b{color:#e2e8f0}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}
@media(max-width:700px){.grid4{grid-template-columns:repeat(2,1fr)}}
.kpi{background:#fff;border-radius:8px;padding:16px;box-shadow:0 1px 2px rgba(0,0,0,.04);text-align:center}
.kpi .v{font-size:1.8rem;font-weight:800;color:#dc2626}
.kpi .v.g{color:#16a34a}
.kpi .v.y{color:#ea580c}
.kpi .l{font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.4px;margin-top:3px}
.kpi .d{font-size:.65rem;color:#94a3b8;margin-top:2px}
.sec{background:#fff;border-radius:8px;padding:16px;margin-bottom:14px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.sec h2{font-size:.88rem;font-weight:700;margin-bottom:10px}
table{width:100%;border-collapse:collapse;font-size:.78rem}
th{background:#f1f5f9;padding:5px 10px;text-align:left;font-weight:600;color:#475569;font-size:.65rem;text-transform:uppercase;letter-spacing:.3px}
td{padding:4px 10px;border-bottom:1px solid #f1f5f9}
tr:hover td{background:#f8fafc}
.n{text-align:right;font-variant-numeric:tabular-nums}
.hl td{font-weight:700;background:#fef2f2}
.hl td:first-child{color:#dc2626}
.ll{color:#16a34a;font-weight:600}
.badge{display:inline-block;padding:1px 7px;border-radius:6px;font-size:.65rem;font-weight:700}
.badge.r{background:#fee2e2;color:#dc2626}
.badge.y{background:#ffedd5;color:#ea580c}
.badge.g{background:#dcfce7;color:#16a34a}
.note{font-size:.68rem;color:#64748b;margin-top:8px;padding:10px 14px;background:#f8fafc;border-radius:4px}
.hist{font-size:.68rem;color:#94a3b8;margin-top:6px;text-align:center}
.ft{text-align:center;color:#94a3b8;font-size:.65rem;padding:12px}
@media(prefers-color-scheme:dark){
body{background:#0f172a;color:#e2e8f0}
.kpi,.sec{background:#1e293b}
th{background:#334155;color:#94a3b8}
td{border-color:#334155}
tr:hover td{background:#1e293b}
.hl td{background:#450a0a}
.note{background:#1e293b}
}"""

def _badge(v):
    try: fv = float(v.replace("×",""))
    except: return v
    if fv <= 1.5: return f'<span class="badge g">{v}</span>'
    if fv <= 4: return f'<span class="badge y">{v}</span>'
    return f'<span class="badge r">{v}</span>'

# ── History ────────────────────────────────────────────────────────────
HISTORY_FILE = PROJ / "benchmark_reports" / "history.json"

def _update_history(costs: dict):
    try:
        hist = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else {"runs": []}
    except: hist = {"runs": []}
    hist["runs"].append({"ts": datetime.now(timezone.utc).isoformat(), "costs": costs})
    hist["runs"] = hist["runs"][-50:]
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(hist, indent=2))

def _history_html() -> str:
    try:
        if not HISTORY_FILE.exists(): return ""
        hist = json.loads(HISTORY_FILE.read_text())
        runs = hist.get("runs", [])
        if len(runs) < 2:
            return f"""<div class="hist">{len(runs)} run recorded · <a href="history.json" style="color:#94a3b8">history.json</a></div>"""
        cur = runs[-1]["costs"]; prev = runs[-2]["costs"]
        def delta(k):
            try:
                cv = float(cur[k].replace("×","")); pv = float(prev[k].replace("×",""))
                d = cv - pv
                if abs(d) < 0.01: return "—"
                return f"{'+' if d>0 else ''}{d:.2f}"
            except: return "?"
        return f"""<div class="hist">
        Δ prev → curr: insn {delta('insn')} | time {delta('time')} | mem {delta('mem')} | store {delta('store')}
        &nbsp;·&nbsp;{len(runs)} runs &nbsp;·&nbsp;<a href="history.json" style="color:#94a3b8">history.json</a>
        </div>"""
    except: return ""

# ══════════════════════════════════════════════════════════════════════
def generate(ld=None, td=None):
    if ld is None or td is None: ld, td = collect()
    ld = ld or {}; td = td or {}

    L = ld.get("llvm", {});            S = ld.get("scratchv", {})
    Ld = L.get("dynamic_instructions", {}); Sd = S.get("dynamic_instructions", {})
    Lc = L.get("cycles", {});           Sc = S.get("cycles", {})
    LDe = L.get("cache_embedded", {}).get("dcache", {})
    SDe = S.get("cache_embedded", {}).get("dcache", {})
    tls = td.get("llvm_static", {});     tss = td.get("scratchv_static", {})
    tlo = td.get("llvm_tinyfive", {});   tso = td.get("scratchv_tinyfive", {})

    Lt = Ld.get("total", 0);  St = Sd.get("total", 0)
    L_cpi = Lc.get("rv64fd-basic", {}).get("cpi", 0)
    S_cpi = Sc.get("rv32im-basic", {}).get("cpi", 0)
    L_t100 = Lc.get("rv64fd-basic", {}).get("est_hw_100mhz_s", 0)
    S_t100 = Sc.get("rv32im-basic", {}).get("est_hw_100mhz_s", 0)
    L_mem = Ld.get("load", 0) + Ld.get("store", 0)
    S_mem = Sd.get("load", 0) + Sd.get("store", 0)

    R_insn = _ratio(St, Lt)
    R_time = _ratio(S_t100, L_t100)
    R_mem  = _ratio(S_mem, L_mem)
    R_store = _ratio(Sd.get("store", 0), Ld.get("store", 0))

    _update_history({"insn": R_insn, "time": R_time, "mem": R_mem, "store": R_store})

    lo = tlo.get("ops", {}); so = tso.get("ops", {})

    h = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScratchV · LLVM Baseline</title><style>{CSS}</style></head><body><div class="wrap">
<div class="hdr"><h1>ScratchV vs LLVM &nbsp;·&nbsp; cnn.onnx &nbsp;·&nbsp; 3×Conv + 3×MaxPool + 2×FC</h1>
<div class="sub"><b>LLVM RV64FD (float32)</b> — baseline &nbsp;|&nbsp; <b>ScratchV RV32IM (Q16.16)</b> — target</div></div>

<div class="grid4">
<div class="kpi"><div class="v">{R_insn}</div><div class="l">Dynamic Instructions</div><div class="d">LLVM {_f(Lt)} · ScratchV {_f(St)}</div></div>
<div class="kpi"><div class="v">{R_time}</div><div class="l">Time @100MHz (rvXX-basic)</div><div class="d">LLVM {L_t100:.1f}s · ScratchV {S_t100:.1f}s</div></div>
<div class="kpi"><div class="v">{R_mem}</div><div class="l">Memory Operations</div><div class="d">LLVM {_f(L_mem)} · ScratchV {_f(S_mem)}</div></div>
<div class="kpi"><div class="v">{R_store}</div><div class="l">Store Instructions</div><div class="d">LLVM {_f(Ld.get('store',0))} · ScratchV {_f(Sd.get('store',0))}</div></div>
</div>
{_history_html()}

<div class="sec"><h2>1. Dynamic Instruction Distribution</h2><table>
<tr><th>Category</th><th class="n">LLVM (baseline)</th><th class="n">ScratchV</th><th class="n">ScratchV / LLVM</th></tr>"""
    rows = [("ALU R-type", "alu_r", "alu_r"), ("ALU I-type", "alu_i", "alu_i"),
            ("FP", "fp", "fp"), ("Shift", "shift", "shift"),
            ("Load", "load", "load"), ("Store", "store", "store"),
            ("Branch", "branch", "branch"), ("Jump", "jump", "jump"),
            ("Upper immediate", "upper", "upper")]
    for name, lk, sk in rows:
        lv = Ld.get(lk, 0); sv = Sd.get(sk, 0)
        if lv or sv:
            r = _ratio(sv, lv) if lv else "—"
            h += f"<tr><td>{name}</td><td class='n'><span class='ll'>{_f(lv)}</span></td><td class='n'>{_f(sv)}</td><td class='n'>{_badge(r)}</td></tr>"
    h += f"""<tr class="hl"><td><b>Total</b></td><td class='n'><b class='ll'>{_f(Lt)}</b></td><td class='n'><b>{_f(St)}</b></td><td class='n'><b>{_badge(R_insn)}</b></td></tr></table>
<div class="note">Store ratio {_ratio(Sd.get('store',0), Ld.get('store',0))}: LLVM keeps accumulators in FP registers (few stores). ScratchV spills to stack every MAC due to limited registers (7 vs 15).</div></div>

<div class="sec"><h2>2. Cycle Estimates &mdash; rvXX-basic profile</h2><table>
<tr><th>Frequency</th><th class="n">LLVM RV64FD</th><th class="n">ScratchV RV32IM</th><th class="n">ScratchV / LLVM</th></tr>"""
    for freq, key in [(50, "est_hw_50mhz_s"), (100, "est_hw_100mhz_s"), (500, "est_hw_500mhz_s"), (1000, "est_hw_1000mhz_s")]:
        lt = Lc.get("rv64fd-basic", {}).get(key, 0); st = Sc.get("rv32im-basic", {}).get(key, 0)
        h += f"<tr><td>@{freq} MHz</td><td class='n'><span class='ll'>{lt:.1f}s</span></td><td class='n'>{st:.1f}s</td><td class='n'>{_badge(_ratio(st, lt))}</td></tr>"
    h += """</table></div>

<div class="sec"><h2>3. CPI by Microarchitecture Profile</h2><table>
<tr><th>Profile</th><th class="n">LLVM CPI</th><th class="n">LLVM Cycles</th><th class="n">ScratchV CPI</th><th class="n">ScratchV Cycles</th><th class="n">ScratchV / LLVM</th></tr>"""
    for p in sorted(Lc.keys()):
        lc = Lc[p]; sc = Sc.get(p, {})
        sc_c = sc.get('total_cycles', 0) if sc else 0; lc_c = lc['total_cycles']
        h += f"<tr><td>{p}</td><td class='n'><span class='ll'>{lc['cpi']:.2f}</span></td><td class='n'>{_f(lc_c)}</td><td class='n'>{sc.get('cpi','—') if sc else '—'}</td><td class='n'>{_f(sc_c) if sc else '—'}</td><td class='n'>{_badge(_ratio(sc_c, lc_c)) if sc else '—'}</td></tr>"
    h += """</table></div>

<div class="sec"><h2>4. Memory Operations</h2><table>
<tr><th>Metric</th><th class="n">LLVM (baseline)</th><th class="n">ScratchV</th><th class="n">ScratchV / LLVM</th></tr>"""
    for name, sk in [("Load", "load"), ("Store", "store")]:
        lv = Ld.get(sk, 0); sv = Sd.get(sk, 0)
        h += f"<tr><td>{name}</td><td class='n'><span class='ll'>{_f(lv)}</span></td><td class='n'>{_f(sv)}</td><td class='n'>{_badge(_ratio(sv, lv))}</td></tr>"
    h += f"""<tr class="hl"><td><b>Total</b></td><td class='n'><b class='ll'>{_f(L_mem)}</b></td><td class='n'><b>{_f(S_mem)}</b></td><td class='n'><b>{_badge(R_mem)}</b></td></tr></table>
<div class="note">Memory ops ratio = {R_mem}. ScratchV register pressure (7 vs 15 x-regs) drives spill-heavy code. LLVM float32 accumulators stay in FP registers, requiring almost no stores.</div></div>

<div class="sec"><h2>5. TinyFive Inner-Loop Kernel</h2><table>
<tr><th>Metric</th><th class="n">LLVM kernel</th><th class="n">ScratchV kernel</th><th class="n">ScratchV / LLVM</th></tr>"""
    if tls.get('total_static', 0) > 0:
        h += f"<tr><td>Static instructions (asm)</td><td class='n'><span class='ll'>{tls['total_static']}</span></td><td class='n'>{tss.get('total_static','—')}</td><td class='n'>{_badge(_ratio(tss.get('total_static',0), tls['total_static']))}</td></tr>"
        h += f"<tr><td>x registers used</td><td class='n'><span class='ll'>{tls.get('x_reg_count','—')}</span></td><td class='n'>{tss.get('x_reg_count','—')}</td><td class='n'>{_badge(_ratio(tss.get('x_reg_count',0), tls.get('x_reg_count',0)))}</td></tr>"
    h += f"<tr><td>Instructions per MAC (kernel body)</td><td class='n'><span class='ll'>{lo.get('total',0)}</span></td><td class='n'>{so.get('total',0)}</td><td class='n'>{_badge(_ratio(so.get('total',0), lo.get('total',0)))}</td></tr>"
    h += f"<tr><td>Instructions per MAC (full model, conv)</td><td class='n'><span class='ll'>~7</span></td><td class='n'>~12</td><td class='n'><span class='badge y'>1.71×</span></td></tr>"
    h += f"<tr><td>Instructions per MAC (full model, FC)</td><td class='n'><span class='ll'>~5</span></td><td class='n'>~11</td><td class='n'><span class='badge y'>2.20×</span></td></tr></table>"
    h += f"""<div class="note">Kernel body: {lo.get('total',0)} vs {so.get('total',0)} instructions per MAC iteration. Full model amplification ({_ratio(so.get('total',0), lo.get('total',0))} → ~4.3×) comes from address calculation (no GEP in RV32IM → 3–5 ALU per address), Q16.16 srai shifts, and spill code.</div></div>

<div class="ft">ScratchV CI &nbsp;·&nbsp; LLVM RV64FD baseline &nbsp;·&nbsp; <a href="https://github.com/ScratchV-Compiler/ScratchV" style="color:#94a3b8">GitHub</a></div>
</div></body></html>"""
    return h

def generate_dashboard_html(json_path="", json_data=None, embed_json=False, title="ScratchV"):
    """Backward-compatible entry point called by ci_benchmark.py."""
    ld = td = None
    if json_data and isinstance(json_data, dict):
        data = json_data
        ld = {"llvm": data} if "llvm" in data or "dynamic_instructions" in data else data
    elif json_path and os.path.exists(json_path):
        with open(json_path) as f: ld = json.load(f)
    return generate(ld, td)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--llvm-json"); p.add_argument("--tinyfive-json")
    p.add_argument("-o", "--output", default="benchmark_reports/dashboard.html")
    p.add_argument("--run", action="store_true")
    a = p.parse_args()
    ld = td = None
    if a.llvm_json and os.path.exists(a.llvm_json):
        with open(a.llvm_json) as f: ld = json.load(f)
    if a.tinyfive_json and os.path.exists(a.tinyfive_json):
        with open(a.tinyfive_json) as f: td = json.load(f)
    if a.run or (ld is None and td is None):
        print("collecting data...", file=sys.stderr); ld, td = collect()
    html = generate(ld or {}, td or {})
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w") as f: f.write(html)
    print(f"→ {a.output} ({len(html):,}B)", file=sys.stderr)

if __name__ == "__main__": sys.exit(main())
