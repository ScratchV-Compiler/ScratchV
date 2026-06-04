"""ScratchV vs LLVM cost overhead dashboard. LLVM = baseline. ScratchV = experiment."""

from __future__ import annotations
import json, os, subprocess, sys, tempfile, time
from pathlib import Path
from datetime import datetime, timezone

PROJ = Path(__file__).resolve().parent.parent.parent

def _run(script, args):
    tmp = tempfile.mktemp(suffix=".json")
    subprocess.run([sys.executable,str(PROJ/script)]+args, capture_output=True, cwd=str(PROJ), timeout=60)
    if os.path.exists(tmp):
        with open(tmp) as f: return json.load(f)
    return {}

def collect(): return (
    _run("scratchv/standalone/llvm_cache_compare.py",["--json-output","/tmp/_llvm.json"]),
    _run("scratchv/standalone/tinyfive_compare.py",["--json","/tmp/_tf.json"]),
)

def _f(n):
    if not n: return "—"
    if isinstance(n,float): n=int(n)
    return f"{n:,}"
def _cost(sv, ll):
    """How many times ScratchV costs vs LLVM. >1 = ScratchV is more expensive."""
    if not ll: return "—"
    return f"{sv/ll:.2f}" if sv/ll < 10 else f"{sv/ll:.1f}"

CSS="""*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#f8fafc;color:#1e293b;line-height:1.5}
.wrap{max-width:900px;margin:0 auto;padding:20px}
.hdr{background:linear-gradient(135deg,#0f172a,#1e293b);color:#f1f5f9;padding:22px 28px;border-radius:10px;margin-bottom:20px}
.hdr h1{font-size:1.2rem}
.hdr .sub{font-size:.78rem;color:#94a3b8;margin-top:4px}
.cost-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
@media(max-width:700px){.cost-grid{grid-template-columns:repeat(2,1fr)}}
.cost-card{background:#fff;border-radius:8px;padding:18px;box-shadow:0 1px 2px rgba(0,0,0,.05);text-align:center;position:relative}
.cost-card .cost{font-size:2rem;font-weight:800;color:#dc2626}
.cost-card .cost.good{color:#16a34a}
.cost-card .cost.warn{color:#ea580c}
.cost-card .label{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-top:4px}
.cost-card .detail{font-size:.68rem;color:#94a3b8;margin-top:2px}
.cost-card .detail span{font-weight:600}
.sec{background:#fff;border-radius:8px;padding:18px;margin-bottom:14px;box-shadow:0 1px 2px rgba(0,0,0,.05)}
.sec h2{font-size:.9rem;font-weight:700;margin-bottom:10px}
table{width:100%;border-collapse:collapse;font-size:.8rem}
th{background:#f1f5f9;padding:6px 10px;text-align:left;font-weight:600;color:#475569;font-size:.68rem;text-transform:uppercase;letter-spacing:.3px}
td{padding:5px 10px;border-bottom:1px solid #f1f5f9}
tr:hover td{background:#f8fafc}
.n{text-align:right;font-variant-numeric:tabular-nums}
.hl td{font-weight:700;background:#fef2f2}
.hl td:first-child{color:#dc2626}
.ll{color:#16a34a;font-weight:600}
.cost-badge{display:inline-block;padding:1px 8px;border-radius:8px;font-size:.7rem;font-weight:700}
.cost-badge.high{background:#fee2e2;color:#dc2626}
.cost-badge.mid{background:#ffedd5;color:#ea580c}
.cost-badge.low{background:#dcfce7;color:#16a34a}
.history{font-size:.72rem;color:#94a3b8;margin-top:8px;text-align:center}
.footer{text-align:center;color:#94a3b8;font-size:.68rem;padding:14px}
@media(prefers-color-scheme:dark){
body{background:#0f172a;color:#e2e8f0}
.cost-card,.sec{background:#1e293b}
th{background:#334155;color:#94a3b8}
td{border-color:#334155}
tr:hover td{background:#1e293b}
.hl td{background:#450a0a}
}"""

def _badge(v):
    """Color-coded cost badge."""
    try: fv=float(v.replace('×',''))
    except: return v
    if fv<=1.5: return f'<span class="cost-badge low">{v}</span>'
    if fv<=4: return f'<span class="cost-badge mid">{v}</span>'
    return f'<span class="cost-badge high">{v}</span>'

# ── Historical tracking ─────────────────────────────────────────────────────
HISTORY_FILE = PROJ / "benchmark_reports" / "cost_history.json"

def _update_history(costs: dict):
    """Append current run to cost history."""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE) as f: hist = json.load(f)
        else:
            hist = {"runs": []}
    except: hist = {"runs": []}

    hist["runs"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "costs": costs,
    })
    # Keep last 50 runs
    hist["runs"] = hist["runs"][-50:]

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(hist, f, indent=2)

def _history_html() -> str:
    """Generate mini trend table from history."""
    try:
        if not HISTORY_FILE.exists(): return ""
        with open(HISTORY_FILE) as f: hist = json.load(f)
        runs = hist.get("runs", [])
        if len(runs) < 2: return ""

        latest = runs[-1]["costs"]
        prev = runs[-2]["costs"]

        def _delta(key):
            try:
                cur = float(latest.get(key, "0").replace("×",""))
                old = float(prev.get(key, "0").replace("×",""))
                d = cur - old
                if abs(d) < 0.01: return "→ 持平"
                return f"{'↑ +' if d>0 else '↓ '}{abs(d):.2f}"
            except: return "—"

        return f"""<div class="history">
        上次 → 本次: 指令 {_delta('insn_cost')} | 耗时 {_delta('time_cost')} | 访存 {_delta('mem_cost')} | Store {_delta('store_cost')}
        <br>共 {len(runs)} 次记录 · <a href="cost_history.json" style="color:#94a3b8">历史数据</a>
        </div>"""
    except: return ""

def generate(ld=None, td=None):
    if ld is None or td is None: ld, td = collect()
    ld=ld or {}; td=td or {}
    L=ld.get("llvm",{}); S=ld.get("scratchv",{})
    Ld=L.get("dynamic_instructions",{}); Sd=S.get("dynamic_instructions",{})
    Lc=L.get("cycles",{}); Sc=S.get("cycles",{})
    LDe=L.get("cache_embedded",{}).get("dcache",{})
    SDe=S.get("cache_embedded",{}).get("dcache",{})
    tls=td.get("llvm_static",{}); tss=td.get("scratchv_static",{})
    tlo=td.get("llvm_tinyfive",{}); tso=td.get("scratchv_tinyfive",{})

    Lt=Ld.get("total",0); St=Sd.get("total",0)
    Lcp=Lc.get("rv64fd-basic",{}).get("cpi",0)
    Scp=Sc.get("rv32im-basic",{}).get("cpi",0)
    Lt1=Lc.get("rv64fd-basic",{}).get("est_hw_100mhz_s",0)
    St1=Sc.get("rv32im-basic",{}).get("est_hw_100mhz_s",0)
    Lmem=Ld.get("load",0)+Ld.get("store",0)
    Smem=Sd.get("load",0)+Sd.get("store",0)
    sv_mb=SDe.get('total_miss_bytes',0); ll_mb=LDe.get('total_miss_bytes',0)
    sv_m=SDe.get('misses',0); ll_m=LDe.get('misses',0)

    # Cost multiples
    insn_cost = _cost(St, Lt)
    time_cost = _cost(St1, Lt1)
    mem_cost  = _cost(Smem, Lmem)
    store_cost = _cost(Sd.get('store',0), Ld.get('store',0))

    costs = {"insn_cost": insn_cost, "time_cost": time_cost,
             "mem_cost": mem_cost, "store_cost": store_cost}
    _update_history(costs)

    lo=tlo.get("ops",{}); so=tso.get("ops",{})
    h=f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScratchV · Cost vs LLVM</title><style>{CSS}</style></head><body><div class="wrap">
<div class="hdr"><h1>📊 ScratchV 成本对比 — LLVM 基准线</h1>
<div class="sub">cnn.onnx · LLVM RV64FD (float32) = 基准 · ScratchV RV32IM (Q16.16) = 实验组 · 数值 = ScratchV 多花多少倍</div></div>

<div class="cost-grid">
<div class="cost-card"><div class="cost {'good' if float(insn_cost.replace('×',''))<=2 else 'warn' if float(insn_cost.replace('×',''))<=4 else ''}">{insn_cost}</div><div class="label">指令开销</div><div class="detail">LLVM <span>{_f(Lt)}</span> → ScratchV <span>{_f(St)}</span></div></div>
<div class="cost-card"><div class="cost">{time_cost}</div><div class="label">时间开销 @100MHz</div><div class="detail">LLVM <span>{Lt1:.1f}s</span> → ScratchV <span>{St1:.1f}s</span></div></div>
<div class="cost-card"><div class="cost">{mem_cost}</div><div class="label">访存开销</div><div class="detail">LLVM <span>{_f(Lmem)}</span> → ScratchV <span>{_f(Smem)}</span></div></div>
<div class="cost-card"><div class="cost">{store_cost}</div><div class="label">Store 开销</div><div class="detail">LLVM <span>{_f(Ld.get('store',0))}</span> → ScratchV <span>{_f(Sd.get('store',0))}</span></div></div>
</div>
{_history_html()}

<div class="sec"><h2>指令分布开销</h2><table><tr><th>类别</th><th class="n">LLVM</th><th class="n">ScratchV</th><th class="n">ScratchV 多花</th></tr>"""
    for name,lk,sk in [("ALU 运算","alu_r","alu_r"),("ALU 立即数","alu_i","alu_i"),
        ("浮点","fp","fp"),("移位","shift","shift"),
        ("加载","load","load"),("存储","store","store"),
        ("分支","branch","branch"),("跳转","jump","jump"),
        ("高位立即数","upper","upper")]:
        lv=Ld.get(lk,0); sv=Sd.get(sk,0)
        if lv or sv:
            c=_cost(sv,lv) if lv else "—"
            h+=f"<tr><td>{name}</td><td class='n'><span class='ll'>{_f(lv)}</span></td><td class='n'>{_f(sv)}</td><td class='n'>{_badge(c)}</td></tr>"
    h+=f"""<tr class="hl"><td><b>总计</b></td><td class='n'><b class='ll'>{_f(Lt)}</b></td><td class='n'><b>{_f(St)}</b></td><td class='n'><b>{_badge(insn_cost)}</b></td></tr></table></div>

<div class="sec"><h2>耗时开销 (rvXX-basic)</h2><table><tr><th>频率</th><th class="n">LLVM</th><th class="n">ScratchV</th><th class="n">ScratchV 多花</th></tr>"""
    for freq,key in [(50,"est_hw_50mhz_s"),(100,"est_hw_100mhz_s"),(500,"est_hw_500mhz_s")]:
        lt=Lc.get("rv64fd-basic",{}).get(key,0); st=Sc.get("rv32im-basic",{}).get(key,0)
        h+=f"<tr><td>@{freq}MHz</td><td class='n'><span class='ll'>{lt:.1f}s</span></td><td class='n'>{st:.1f}s</td><td class='n'>{_badge(_cost(st,lt))}</td></tr>"
    h+=f"""</table></div>

<div class="sec"><h2>访存开销</h2><table><tr><th>指标</th><th class="n">LLVM</th><th class="n">ScratchV</th><th class="n">ScratchV 多花</th></tr>
<tr><td>加载 (Load)</td><td class='n'><span class='ll'>{_f(Ld.get('load',0))}</span></td><td class='n'>{_f(Sd.get('load',0))}</td><td class='n'>{_badge(_cost(Sd.get('load',0),Ld.get('load',0)))}</td></tr>
<tr><td>存储 (Store)</td><td class='n'><span class='ll'>{_f(Ld.get('store',0))}</span></td><td class='n'>{_f(Sd.get('store',0))}</td><td class='n'>{_badge(_cost(Sd.get('store',0),Ld.get('store',0)))}</td></tr>
<tr class="hl"><td><b>访存总计</b></td><td class='n'><b class='ll'>{_f(Lmem)}</b></td><td class='n'><b>{_f(Smem)}</b></td><td class='n'><b>{_badge(mem_cost)}</b></td></tr>
</table>
<div class="insight" style="font-size:.72rem;color:#64748b;margin-top:8px">Store 开销巨大（{_cost(Sd.get('store',0),Ld.get('store',0))}）：LLVM 累加器留在 FP 寄存器中几乎不需要 store，ScratchV 每次 MAC 都需要 sw 到栈上 spill。这是寄存器不足（7 vs 15）的直接后果。</div></div>

<div class="sec"><h2>TinyFive 内核开销</h2><table><tr><th>指标</th><th class="n">LLVM</th><th class="n">ScratchV</th><th class="n">ScratchV 多花</th></tr>
{f"<tr><td>静态指令</td><td class='n'><span class='ll'>{tls.get('total_static','—')}</span></td><td class='n'>{tss.get('total_static','—')}</td><td class='n'>{_badge(_cost(tss.get('total_static',0),tls.get('total_static',0)))}</td></tr>" if tls.get('total_static',0) > 0 else ""}
{f"<tr><td>x 寄存器</td><td class='n'><span class='ll'>{tls.get('x_reg_count','—')}</span></td><td class='n'>{tss.get('x_reg_count','—')}</td><td class='n'>{_badge(_cost(tss.get('x_reg_count',0),tls.get('x_reg_count',0)))}</td></tr>" if tls.get('x_reg_count',0) > 0 else ""}
<tr><td>每 MAC 指令 (内核)</td><td class='n'><span class='ll'>{lo.get('total',0)}</span></td><td class='n'>{so.get('total',0)}</td><td class='n'>{_badge(_cost(so.get('total',0),lo.get('total',0)))}</td></tr>
<tr><td>每 MAC 指令 (全模型)</td><td class='n'><span class='ll'>~7</span></td><td class='n'>~30</td><td class='n'><span class="cost-badge high">4.3×</span></td></tr>
</table>
<div class="insight" style="font-size:.72rem;color:#64748b;margin-top:8px">内核仅 {_cost(so.get('total',0),lo.get('total',0))}，扩展到全模型放大到 4.3×。差异来源：Q16.16 srai 移位、地址计算（无 GEP → 3-5 条 ALU/地址）、spill 存储。</div></div>

<div class="sec"><h2>CPI 对比</h2><table><tr><th>Profile</th><th class="n">LLVM CPI</th><th class="n">LLVM Cycles</th><th class="n">ScratchV CPI</th><th class="n">ScratchV Cycles</th><th class="n">多花</th></tr>"""
    for p in sorted(Lc.keys()):
        lc=Lc[p]; sc=Sc.get(p,{})
        sc_total=sc.get('total_cycles',0) if sc else 0; lc_total=lc['total_cycles']
        h+=f"<tr><td>{p}</td><td class='n'><span class='ll'>{lc['cpi']:.2f}</span></td><td class='n'>{_f(lc_total)}</td><td class='n'>{sc.get('cpi','—') if sc else '—'}</td><td class='n'>{_f(sc_total) if sc else '—'}</td><td class='n'>{_badge(_cost(sc_total,lc_total)) if sc else '—'}</td></tr>"
    h+="</table></div>"

    h+=f"""<div class="footer">ScratchV CI · 每次优化 commit 都应该让上面的数字变小 · <a href="https://github.com/ScratchV-Compiler/ScratchV" style="color:#94a3b8">GitHub</a></div>
</div></body></html>"""
    return h

def generate_dashboard_html(*a,**kw): return generate()

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--llvm-json"); p.add_argument("--tinyfive-json")
    p.add_argument("-o","--output",default="benchmark_reports/dashboard.html")
    p.add_argument("--run",action="store_true")
    a=p.parse_args()
    ld=td=None
    if a.llvm_json and os.path.exists(a.llvm_json):
        with open(a.llvm_json) as f: ld=json.load(f)
    if a.tinyfive_json and os.path.exists(a.tinyfive_json):
        with open(a.tinyfive_json) as f: td=json.load(f)
    if a.run or (ld is None and td is None):
        print("collecting...",file=sys.stderr); ld,td=collect()
    html=generate(ld or {}, td or {})
    os.makedirs(os.path.dirname(a.output) or ".",exist_ok=True)
    with open(a.output,"w") as f: f.write(html)
    print(f"→ {a.output} ({len(html):,}B)",file=sys.stderr)

if __name__=="__main__": sys.exit(main())
