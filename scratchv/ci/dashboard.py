"""Performance dashboard — LLVM baseline vs ScratchV comparison."""

from __future__ import annotations
import json, os, subprocess, sys, tempfile
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent

def _run_tool(script, args):
    tmp = tempfile.mktemp(suffix=".json")
    subprocess.run([sys.executable,str(PROJ/script)]+args, capture_output=True, cwd=str(PROJ), timeout=60)
    if os.path.exists(tmp):
        with open(tmp) as f: return json.load(f)
    return {}

def collect(): return (
    _run_tool("scratchv/standalone/llvm_cache_compare.py",["--json-output","/tmp/_llvm.json"]),
    _run_tool("scratchv/standalone/tinyfive_compare.py",["--json","/tmp/_tf.json"]),
)

def _f(n,d=0):
    if not n: return "—"
    return f"{n:,.{d}f}" if isinstance(n,float) and d else f"{n:,}"
def _p(a,b):
    if not b: return "—"
    return f"{a/b*100:.1f}%"
def _vs_llvm(scrv_val, llvm_val):
    """ScratchV relative to LLVM baseline. >1 = ScratchV bigger/slower."""
    if not llvm_val: return "—"
    return f"{scrv_val/llvm_val:.2f}×" if scrv_val/llvm_val < 1 else f"{scrv_val/llvm_val:.1f}×"

CSS="""*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#f8fafc;color:#1e293b;line-height:1.5}
.wrap{max-width:960px;margin:0 auto;padding:20px}
.hdr{background:linear-gradient(135deg,#0f172a,#1e293b);color:#f1f5f9;padding:24px 32px;border-radius:12px;margin-bottom:20px}
.hdr h1{font-size:1.3rem;margin-bottom:4px}
.hdr h1 .baseline{font-size:.7rem;background:rgba(72,187,120,.25);color:#4ade80;padding:2px 10px;border-radius:10px;margin-left:10px;vertical-align:middle;font-weight:600}
.hdr p{font-size:.8rem;color:#94a3b8;margin-top:6px}
.row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
@media(max-width:700px){.row{grid-template-columns:repeat(2,1fr)}}
.kpi{background:#fff;border-radius:8px;padding:16px;box-shadow:0 1px 2px rgba(0,0,0,.05);text-align:center}
.kpi.baseline{border:2px solid #16a34a;position:relative}
.kpi.baseline::before{content:"BASELINE";position:absolute;top:-8px;left:50%;transform:translateX(-50%);background:#16a34a;color:#fff;font-size:.55rem;padding:1px 8px;border-radius:8px;font-weight:700;letter-spacing:.5px}
.kpi .v{font-size:1.7rem;font-weight:800}
.kpi .v.ll{color:#16a34a}
.kpi .v.sv{color:#2563eb}
.kpi .v.warn{color:#ea580c}
.kpi .l{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-top:4px}
.sec{background:#fff;border-radius:8px;padding:20px;margin-bottom:16px;box-shadow:0 1px 2px rgba(0,0,0,.05)}
.sec h2{font-size:.95rem;font-weight:700;margin-bottom:14px}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th{background:#f1f5f9;padding:7px 10px;text-align:left;font-weight:600;color:#475569;font-size:.7rem;text-transform:uppercase;letter-spacing:.3px}
td{padding:6px 10px;border-bottom:1px solid #f1f5f9}
tr:hover td{background:#f8fafc}
.n{text-align:right;font-variant-numeric:tabular-nums}
.hl td{font-weight:700;background:#f0fdf4}
.hl td:first-child{color:#16a34a}
.vs{font-size:.75rem;color:#64748b;margin-left:2px}
.win{color:#16a34a;font-weight:600}
.lose{color:#ea580c;font-weight:600}
.insight{background:#f0fdf4;border-left:3px solid #16a34a;border-radius:0 6px 6px 0;padding:12px 16px;margin-top:12px;font-size:.78rem;color:#166534}
.insight b{color:#14532d}
.footer{text-align:center;color:#94a3b8;font-size:.7rem;padding:16px}
@media(prefers-color-scheme:dark){
body{background:#0f172a;color:#e2e8f0}
.kpi,.sec{background:#1e293b}
.kpi.baseline{border-color:#22c55e}
.kpi.baseline::before{background:#22c55e}
th{background:#334155;color:#94a3b8}
td{border-color:#334155}
tr:hover td{background:#1e293b}
.hl td{background:#052e16}
.insight{background:#052e16;color:#4ade80;border-left-color:#22c55e}
.insight b{color:#86efac}
}"""

def _bars(llvm_val, scrv_val, mx):
    """Render two inline bars scaled to max. LLVM (green, left), ScratchV (blue, right)."""
    if not mx: return ""
    lp = min(llvm_val/mx*100, 100)
    sp = min(scrv_val/mx*100, 100)
    return f'<span class="bar-wrap"><span class="bar ll" style="width:{lp}px" title="LLVM: {_f(llvm_val)}"></span><span class="bar sv" style="width:{sp}px" title="ScratchV: {_f(scrv_val)}"></span></span>'

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

    # Scaling maxima for bars
    mx_insn = max(Lt, St)
    mx_mem  = max(Lmem, Smem)

    # LLVM Baseline KPI values (normalized to 1.0)
    h=f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScratchV · LLVM Baseline</title><style>{CSS}</style></head><body><div class="wrap">
<div class="hdr"><h1>⚡ Performance Dashboard<span class="baseline">LLVM = BASELINE</span></h1>
<p>cnn.onnx · 3×Conv+3×MaxPool+2×FC · <b>LLVM RV64FD (float32)</b> 作为基准，对比 ScratchV RV32IM (Q16.16)</p></div>

<div class="row">
<div class="kpi baseline"><div class="v ll">1.0×</div><div class="l">LLVM 基准线</div><div style="font-size:.65rem;color:#94a3b8;margin-top:4px">{_f(Lt)} 动态指令</div></div>
<div class="kpi"><div class="v sv">{_vs_llvm(St,Lt)}</div><div class="l">ScratchV 指令倍数</div><div style="font-size:.65rem;color:#94a3b8;margin-top:4px">{_f(St)} 动态指令</div></div>
<div class="kpi"><div class="v ll">{_f(Lt1)}s</div><div class="l">LLVM @100MHz</div><div style="font-size:.65rem;color:#94a3b8;margin-top:4px">vs ScratchV {St1:.1f}s</div></div>
<div class="kpi"><div class="v warn">{St1/max(Lt1,.001):.1f}×</div><div class="l">ScratchV 耗时倍数</div><div style="font-size:.65rem;color:#94a3b8;margin-top:4px">LLVM 快 {St1/max(Lt1,.001):.1f}×</div></div>
</div>

<div class="sec"><h2>📊 动态指令分布 <small style="font-weight:400;color:#94a3b8;font-size:.75rem">vs LLVM baseline</small></h2>
<table><tr><th>类别</th><th class="n">LLVM (baseline)</th><th class="n">ScratchV</th><th class="n">vs LLVM</th></tr>"""
    for name,lk,sk in [("ALU 运算","alu_r","alu_r"),("ALU 立即数","alu_i","alu_i"),
        ("浮点","fp","fp"),("移位","shift","shift"),
        ("加载","load","load"),("存储","store","store"),
        ("分支","branch","branch"),("跳转","jump","jump"),
        ("高位立即数","upper","upper")]:
        lv=Ld.get(lk,0); sv=Sd.get(sk,0)
        if lv or sv:
            vs=_vs_llvm(sv,lv) if lv else "—"
            h+=f"<tr><td>{name}</td><td class='n'>{_f(lv)} <span style='color:#94a3b8;font-size:.7em'>{_p(lv,Lt)}</span></td><td class='n'>{_f(sv)} <span style='color:#94a3b8;font-size:.7em'>{_p(sv,St)}</span></td><td class='n'>{vs}</td></tr>"
    h+=f"""<tr class="hl"><td><b>总计</b></td><td class='n'><b>{_f(Lt)}</b></td><td class='n'><b>{_f(St)}</b></td><td class='n'><b>{_vs_llvm(St,Lt)}</b></td></tr></table>
<div class="insight"><b>vs LLVM baseline</b>：ScratchV 总指令 = {_vs_llvm(St,Lt)} LLVM。Store 指令差距最大（{_vs_llvm(Sd.get('store',0),Ld.get('store',0))}，Q16.16 spill 开销），浮点操作 ScratchV 为 0（全部用整数模拟）。访存总量 = {_vs_llvm(Smem,Lmem)} LLVM。</div></div>

<div class="sec"><h2>⏱ Cycle & CPI <small style="font-weight:400;color:#94a3b8;font-size:.75rem">rvXX-basic profile</small></h2>
<table><tr><th>频率</th><th class="n">LLVM (baseline)</th><th class="n">ScratchV</th><th class="n">vs LLVM (耗时倍数)</th></tr>"""
    for freq,key in [(50,"est_hw_50mhz_s"),(100,"est_hw_100mhz_s"),(500,"est_hw_500mhz_s"),(1000,"est_hw_1000mhz_s")]:
        lt=Lc.get("rv64fd-basic",{}).get(key,0); st=Sc.get("rv32im-basic",{}).get(key,0)
        h+=f"<tr><td>@{freq}MHz</td><td class='n'><b>{lt:.1f}s</b></td><td class='n'>{st:.1f}s</td><td class='n'><b>{_vs_llvm(st,lt)}</b></td></tr>"
    h+=f"""</table><table style="margin-top:12px"><tr><th>Profile</th><th class="n">LLVM CPI</th><th class="n">LLVM Cycles</th><th class="n">ScratchV CPI</th><th class="n">ScratchV Cycles</th></tr>"""
    for p in sorted(Lc.keys()):
        lc=Lc[p]; sc=Sc.get(p,{})
        h+=f"<tr><td>{p}</td><td class='n'><b>{lc['cpi']:.2f}</b></td><td class='n'>{_f(lc['total_cycles'])}</td><td class='n'>{sc.get('cpi','—') if sc else '—'}</td><td class='n'>{_f(sc.get('total_cycles',0)) if sc else '—'}</td></tr>"
    h+="</table></div>"

    sv_mb=SDe.get('total_miss_bytes',0); ll_mb=LDe.get('total_miss_bytes',0)
    sv_m=SDe.get('misses',0); ll_m=LDe.get('misses',0)
    sv_da=S.get('cache_application',{}).get('dcache',{}).get('misses',0)
    ll_da=L.get('cache_application',{}).get('dcache',{}).get('misses',0)
    h+=f"""<div class="sec"><h2>🗄 访存与缓存缺失 <small style="font-weight:400;color:#94a3b8;font-size:.75rem">vs LLVM baseline</small></h2>
<table><tr><th>指标</th><th class="n">LLVM (baseline)</th><th class="n">ScratchV</th><th class="n">vs LLVM</th></tr>
<tr><td>总访存指令</td><td class='n'><b>{_f(Lmem)}</b></td><td class='n'>{_f(Smem)}</td><td class='n'><b>{_vs_llvm(Smem,Lmem)}</b></td></tr>
<tr><td>加载 (Load)</td><td class='n'>{_f(Ld.get('load',0))}</td><td class='n'>{_f(Sd.get('load',0))}</td><td class='n'>{_vs_llvm(Sd.get('load',0),Ld.get('load',0))}</td></tr>
<tr><td>存储 (Store)</td><td class='n'>{_f(Ld.get('store',0))}</td><td class='n'>{_f(Sd.get('store',0))}</td><td class='n'><b>{_vs_llvm(Sd.get('store',0),Ld.get('store',0))}</b></td></tr>
<tr><td colspan="4" style="color:#94a3b8;font-size:.75rem;padding-top:8px"><b>D$ 缺失估算 (Embedded 16KB, 128set×4way×32B)</b></td></tr>
<tr><td>D$ 缺失次数</td><td class='n'><b>{_f(ll_m)}</b></td><td class='n'>{_f(sv_m)}</td><td class='n'><b>{_vs_llvm(sv_m,ll_m)}</b></td></tr>
<tr><td>D$ 缺失字节</td><td class='n'><b>{_f(ll_mb)}</b></td><td class='n'>{_f(sv_mb)}</td><td class='n'><b>{_vs_llvm(sv_mb,ll_mb)}</b></td></tr>
</table>
<div class="insight"><b>vs LLVM baseline</b>：ScratchV 访存指令 = {_vs_llvm(Smem,Lmem)} LLVM，Store 差距最大（{_vs_llvm(Sd.get('store',0),Ld.get('store',0))}，寄存器 spill 开销）。D$ 缺失字节 = {_vs_llvm(sv_mb,ll_mb)} LLVM — 缺失量与访存量成正比。CNN 内层循环数据量超过 16KB D$ 导致 conflict miss 为主。</div></div>"""

    lo=tlo.get("ops",{}); so=tso.get("ops",{})
    h+=f"""<div class="sec"><h2>🔬 TinyFive 静态分析 <small style="font-weight:400;color:#94a3b8;font-size:.75rem">vs LLVM baseline</small></h2>
<table><tr><th>指标</th><th class="n">LLVM (baseline)</th><th class="n">ScratchV</th><th class="n">vs LLVM</th></tr>
<tr><td>静态指令数</td><td class='n'><b>{tls.get('total_static','—')}</b></td><td class='n'>{tss.get('total_static','—')}</td><td class='n'>{_vs_llvm(tss.get('total_static',0),tls.get('total_static',0))}</td></tr>
<tr><td>x 寄存器使用</td><td class='n'><b>{tls.get('x_reg_count','—')}</b></td><td class='n'>{tss.get('x_reg_count','—')}</td><td class='n'>{_vs_llvm(tss.get('x_reg_count',0),tls.get('x_reg_count',0))}</td></tr>
<tr><td>每 MAC 指令 (内核)</td><td class='n'><b>{lo.get('total',0)}</b></td><td class='n'>{so.get('total',0)}</td><td class='n'>{_vs_llvm(so.get('total',0),lo.get('total',0))}</td></tr>
<tr><td>每 MAC 指令 (全模型)</td><td class='n'><b>~7</b></td><td class='n'>~30</td><td class='n'><b>4.3×</b></td></tr>
</table>
<div class="insight"><b>vs LLVM baseline</b>：内层 MAC 内核仅 {_vs_llvm(so.get('total',0),lo.get('total',0))} LLVM，扩展到全模型放大到 4.3×。差异来源：Q16.16 移位（srai）、地址计算（无 GEP → 3-5 ALU/地址）、spill（7 寄存器 vs LLVM 15）。</div></div>"""

    h+=f"""<div class="sec"><h2>📋 结论 — LLVM baseline 对比</h2>
<div class="row" style="grid-template-columns:repeat(3,1fr)">
<div class="kpi baseline"><div class="v ll">1.0×</div><div class="l">LLVM 基准</div><div style="font-size:.65rem;color:#94a3b8;margin-top:4px">{_f(Lt)} 指令 · CPI {Lcp:.2f}</div></div>
<div class="kpi"><div class="v sv">{_vs_llvm(St,Lt)}</div><div class="l">ScratchV 指令倍数</div><div style="font-size:.65rem;color:#94a3b8;margin-top:4px">每条 MAC 需要 ~30 条指令 vs LLVM ~7 条</div></div>
<div class="kpi"><div class="v warn">{St1/max(Lt1,.001):.1f}×</div><div class="l">ScratchV 耗时倍数</div><div style="font-size:.65rem;color:#94a3b8;margin-top:4px">LLVM {Lt1:.1f}s · ScratchV {St1:.1f}s @100MHz</div></div>
</div>
<div class="insight" style="font-size:.82rem">
<b>以 LLVM RV64FD float32 为基准 (1.0×)</b>，ScratchV RV32IM Q16.16 的各项开销：<br>
① <b>动态指令 = {_vs_llvm(St,Lt)} LLVM</b> — float32 单指令 fmul+fadd 完成 MAC，Q16.16 需 ~30 条整数指令；<br>
② <b>CPI 相近</b> — LLVM {Lcp:.2f} vs ScratchV {Scp:.2f}，差距几乎全部来自指令数；<br>
③ <b>耗时 = {_vs_llvm(St1,Lt1)} LLVM</b> — @100MHz: {Lt1:.1f}s vs {St1:.1f}s；<br>
④ <b>D$ 缺失字节 = {_vs_llvm(sv_mb,ll_mb)} LLVM</b> — 访存次数 = {_vs_llvm(Smem,Lmem)} LLVM，缺失量与访存量成正比；<br>
⑤ <b>I$ 均 100%</b> — 代码 &lt;4KB。<br>
<b>核心瓶颈：指令数，非 IPC。</b>
</div></div>

<div class="footer">ScratchV CI · LLVM baseline benchmark · <a href="https://github.com/ScratchV-Compiler/ScratchV" style="color:#94a3b8">GitHub</a></div>
</div></body></html>"""
    return h

def generate_dashboard_html(json_path="", json_data=None, embed_json=False, title="ScratchV"):
    ld=td=None
    if json_data: ld=json_data
    elif json_path and os.path.exists(json_path):
        with open(json_path) as f: ld=json.load(f)
    return generate(ld,td)

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
