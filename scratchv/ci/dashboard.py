"""Static HTML dashboard — LLVM vs ScratchV complete comparison.

Generates a clean, self-contained HTML page with ALL data:
  - Dynamic instruction distribution (full breakdown)
  - Cycle estimates across 5 microarchitecture profiles
  - Cache performance (embedded + application)
  - TinyFive static analysis + inner loop kernel
  - Summary cards + key insights

Pure HTML+CSS, zero JavaScript dependencies.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent

# ═══════════════════════════════════════════════════════════════
# Data collection
# ═══════════════════════════════════════════════════════════════

def _run_tool(script: str, args: list[str]) -> dict:
    """Run a standalone script that writes JSON to a file, return parsed dict."""
    tmp = tempfile.mktemp(suffix=".json")
    cmd = [sys.executable, str(PROJ / script)] + args
    subprocess.run(cmd, capture_output=True, cwd=str(PROJ), timeout=60)
    if os.path.exists(tmp):
        with open(tmp) as f:
            return json.load(f)
    # Try stdout if file not written
    r = subprocess.run(cmd + ["--json"], capture_output=True, text=True, cwd=str(PROJ), timeout=60)
    if r.stdout.strip():
        return json.loads(r.stdout)
    return {}


def collect_all() -> tuple[dict, dict]:
    """Collect llvm_cache_compare + tinyfive_compare data."""
    llvm = _run_tool("scratchv/standalone/llvm_cache_compare.py", ["--json-output", "/tmp/_llvm_sv.json"])
    tf = _run_tool("scratchv/standalone/tinyfive_compare.py", ["--json", "/tmp/_tinyfive.json"])
    return llvm, tf


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _f(n, d=0):
    if n is None or n == 0:
        return "—"
    return f"{n:,.{d}f}" if isinstance(n, float) and d else f"{n:,}"

def _p(part, total):
    if not total: return "—"
    return f"{part/total*100:.1f}%"

def _r(a, b):
    if not b: return "—"
    return f"{a/b:.1f}x"


# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f7fafc;color:#2d3748;line-height:1.6}
.container{max-width:1100px;margin:0 auto;padding:20px}
.header{background:linear-gradient(135deg,#1a202c,#2d3748);color:#f7fafc;padding:28px 36px;margin-bottom:20px;border-radius:10px}
.header h1{font-size:1.5rem;margin-bottom:4px}
.header .sub{color:#a0aec0;font-size:.82rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:20px}
.card{background:#fff;border-radius:8px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,.07);border-left:4px solid #4299e1}
.card.ll{border-left-color:#48bb78}
.card.ratio{border-left-color:#ed8936}
.card .label{font-size:.7rem;text-transform:uppercase;letter-spacing:.5px;color:#718096;margin-bottom:2px}
.card .value{font-size:1.4rem;font-weight:700}
.card .detail{font-size:.75rem;color:#a0aec0;margin-top:2px}
.section{background:#fff;border-radius:8px;padding:22px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.07)}
.section h2{font-size:1.05rem;font-weight:700;margin-bottom:14px;padding-bottom:8px;border-bottom:2px solid #e2e8f0}
.section h3{font-size:.85rem;color:#718096;margin:14px 0 8px}
table{width:100%;border-collapse:collapse;font-size:.85rem}
th{background:#edf2f7;padding:8px 10px;text-align:left;font-weight:600;color:#4a5568;font-size:.75rem;text-transform:uppercase;letter-spacing:.3px}
td{padding:7px 10px;border-bottom:1px solid #edf2f7}
tr:hover td{background:#f7fafc}
.n{text-align:right;font-variant-numeric:tabular-nums}
.hl{font-weight:700;background:#fffff0}
.svc{color:#4299e1;font-weight:600}
.llc{color:#48bb78;font-weight:600}
.insight{background:#ebf8ff;border-radius:6px;padding:14px 18px;margin-top:14px}
.insight h4{font-size:.85rem;color:#2b6cb0;margin-bottom:5px}
.insight ul{margin-left:18px;font-size:.82rem}
.insight li{margin-bottom:2px}
.footer{text-align:center;color:#a0aec0;font-size:.75rem;padding:18px}
@media(prefers-color-scheme:dark){
body{background:#1a202c;color:#e2e8f0}
.card,.section{background:#2d3748;box-shadow:0 1px 3px rgba(0,0,0,.3)}
th{background:#4a5568;color:#cbd5e0}
td{border-color:#4a5568}
tr:hover td{background:#3a4556}
.hl{background:#744210}
.insight{background:#2a4365}
.insight h4{color:#90cdf4}
.section h2{border-color:#4a5568}
}
"""


# ═══════════════════════════════════════════════════════════════
# HTML generator
# ═══════════════════════════════════════════════════════════════

def generate(ld: dict | None = None, td: dict | None = None) -> str:
    if ld is None or td is None:
        ld, td = collect_all()

    L = ld["llvm"]; S = ld["scratchv"]
    Ld = L["dynamic_instructions"]; Sd = S["dynamic_instructions"]
    Lc = L["cycles"]; Sc = S["cycles"]
    LIe = L.get("cache_embedded",{}).get("icache",{})
    LDe = L.get("cache_embedded",{}).get("dcache",{})
    SIe = S.get("cache_embedded",{}).get("icache",{})
    SDe = S.get("cache_embedded",{}).get("dcache",{})
    LIa = L.get("cache_application",{}).get("icache",{})
    LDa = L.get("cache_application",{}).get("dcache",{})
    SIa = S.get("cache_application",{}).get("icache",{})
    SDa = S.get("cache_application",{}).get("dcache",{})

    tls = td.get("llvm_static",{}); tss = td.get("scratchv_static",{})
    tlo = td.get("llvm_tinyfive",{}); tso = td.get("scratchv_tinyfive",{})

    L_total = Ld["total"]; S_total = Sd["total"]
    L_cpi = Lc.get("rv64fd-basic",{}).get("cpi",0)
    S_cpi = Sc.get("rv32im-basic",{}).get("cpi",0)
    L_t100 = Lc.get("rv64fd-basic",{}).get("est_hw_100mhz_s",0)
    S_t100 = Sc.get("rv32im-basic",{}).get("est_hw_100mhz_s",0)
    sp = S_t100/max(L_t100,.001)

    L_mem = Ld["load"]+Ld["store"]; S_mem = Sd["load"]+Sd["store"]

    h = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScratchV Performance Dashboard</title><style>{CSS}</style></head><body><div class="container">
<div class="header"><h1>📊 LLVM vs ScratchV Performance Dashboard</h1>
<div class="sub">cnn.onnx · 3×Conv+3×MaxPool+2×FC · LLVM RV64FD float32 vs ScratchV RV32IM Q16.16</div></div>

<div class="cards">
<div class="card"><div class="label">ScratchV Static</div><div class="value">785 <small style="font-size:.55em">insns</small></div><div class="detail">3,140 B · Q16.16 RV32IM</div></div>
<div class="card ll"><div class="label">LLVM Static</div><div class="value">956 <small style="font-size:.55em">insns</small></div><div class="detail">3,824 B · float32 RV64FD</div></div>
<div class="card" style="border-left-color:#9f7aea"><div class="label">Dynamic Ratio</div><div class="value">{_r(S_total,L_total)}</div><div class="detail">ScratchV {S_total/1e9:.2f}B vs LLVM {L_total/1e9:.2f}B</div></div>
<div class="card ratio"><div class="label">Speedup @100MHz</div><div class="value">{sp:.1f}×</div><div class="detail">LLVM {L_t100:.1f}s · ScratchV {S_t100:.1f}s</div></div>
</div>

<div class="section"><h2>1. Dynamic Instruction Distribution</h2><table>
<tr><th>Category</th><th class="n">LLVM RV64FD</th><th class="n">%</th><th class="n">ScratchV RV32IM</th><th class="n">%</th><th class="n">Ratio</th></tr>"""

    for name,lk,sk in [("ALU R-type","alu_r","alu_r"),("ALU I-type","alu_i","alu_i"),
        ("Shift","shift","shift"),("Load","load","load"),("Store","store","store"),
        ("Branch","branch","branch"),("Jump","jump","jump"),("Upper imm","upper","upper"),
        ("Float ops","fp","fp")]:
        lv=Ld.get(lk,0); sv=Sd.get(sk,0)
        if lv or sk in Sd:
            h+=f"<tr><td>{name}</td><td class='n'>{_f(lv)}</td><td class='n'>{_p(lv,L_total)}</td><td class='n'>{_f(sv)}</td><td class='n'>{_p(sv,S_total)}</td><td class='n'>{_r(sv,lv) if lv else '—'}</td></tr>"

    h+=f"""<tr class="hl"><td><b>TOTAL</b></td><td class='n'><b>{_f(L_total)}</b></td><td class='n'><b>100%</b></td><td class='n'><b>{_f(S_total)}</b></td><td class='n'><b>100%</b></td><td class='n'><b>{_r(S_total,L_total)}</b></td></tr></table>
<div class="insight"><h4>Memory Access</h4><table style="margin-top:6px"><tr><th></th><th class="n">LLVM</th><th class="n">ScratchV</th><th class="n">Ratio</th></tr>
<tr><td>Loads</td><td class='n'>{_f(Ld["load"])}</td><td class='n'>{_f(Sd["load"])}</td><td class='n'>{_r(Sd["load"],Ld["load"])}</td></tr>
<tr><td>Stores</td><td class='n'>{_f(Ld["store"])}</td><td class='n'>{_f(Sd["store"])}</td><td class='n'>{_r(Sd["store"],Ld["store"])}</td></tr>
<tr><td>Total Memory Ops</td><td class='n'><b>{_f(L_mem)}</b></td><td class='n'><b>{_f(S_mem)}</b></td><td class='n'><b>{_r(S_mem,L_mem)}</b></td></tr>
</table></div></div>

<div class="section"><h2>2. Cycle Estimates by Microarchitecture Profile</h2>
<p style="font-size:.8rem;color:#718096;margin-bottom:10px">基于指令分布估算总 cycle 数。不同 profile 代表不同 CPU 核的指令延迟模型。</p><table>
<tr><th>Profile</th><th class="n">LLVM CPI</th><th class="n">LLVM Cycles</th><th class="n">@100MHz</th><th class="n">ScratchV CPI</th><th class="n">ScratchV Cycles</th><th class="n">@100MHz</th><th class="n">Speedup</th></tr>"""

    for p in sorted(Lc.keys()):
        lc=Lc[p]; sc=Sc.get(p,{})
        lt=lc.get("est_hw_100mhz_s",0); st=sc.get("est_hw_100mhz_s",0) if sc else 0
        su=st/max(lt,.001)
        h+=f"<tr><td>{p}</td><td class='n'>{lc['cpi']:.2f}</td><td class='n'>{_f(lc['total_cycles'])}</td><td class='n'>{lt:.1f}s</td><td class='n'>{sc.get('cpi','—') if sc else '—'}</td><td class='n'>{_f(sc.get('total_cycles',0)) if sc else '—'}</td><td class='n'>{st:.1f}s</td><td class='n'>{su:.1f}x</td></tr>"

    h+="""</table>
<div class="insight"><h4>Profile 说明</h4><ul>
<li><b>single-cycle</b>: 每条指令 1 cycle（纯理想模型）</li>
<li><b>rv32im-basic</b>: mul=4cyc, load=2cyc, branch=2+1cyc — 典型嵌入式 RV32IM</li>
<li><b>rv64fd-basic</b>: mul=3cyc, fp=1cyc(流水线), load=2cyc — 带 FPU 的 RV64FD</li>
<li><b>rv64fd-fast</b>: 全部 1cyc — 近似高性能乱序核</li>
<li><b>rv64fd-slow</b>: mul=5cyc, fp=2cyc, load=3cyc — 保守估算</li>
</ul></div></div>

<div class="section"><h2>3. Cache Performance</h2>
<h3>Embedded — I$=4KB (64×2×32B), D$=16KB (128×4×32B)</h3><table>
<tr><th></th><th class="n">LLVM</th><th class="n">ScratchV</th><th class="n">Ratio</th></tr>
<tr><td>I$ Hit Rate</td><td class='n'><span class="llc">{LIe.get('hit_rate_pct','—')}%</span></td><td class='n'><span class="svc">{SIe.get('hit_rate_pct','—')}%</span></td><td class='n'>~1x</td></tr>
<tr><td>D$ Hit Rate</td><td class='n'><span class="llc">{LDe.get('hit_rate_pct','—')}%</span></td><td class='n'><span class="svc">{SDe.get('hit_rate_pct','—')}%</span></td><td class='n'>~1x</td></tr>
<tr><td>D$ Misses</td><td class='n'>{_f(LDe.get('misses',0))}</td><td class='n'><b>{_f(SDe.get('misses',0))}</b></td><td class='n'><b>{_r(SDe.get('misses',0),LDe.get('misses',0))}</b></td></tr>
<tr><td>D$ Miss Bytes</td><td class='n'>{_f(LDe.get('total_miss_bytes',0))}</td><td class='n'><b>{_f(SDe.get('total_miss_bytes',0))}</b></td><td class='n'><b>{_r(SDe.get('total_miss_bytes',0),LDe.get('total_miss_bytes',0))}</b></td></tr>
</table>
<h3>Application — I$=32KB (128×4×64B), D$=128KB (256×8×64B)</h3><table>
<tr><th></th><th class="n">LLVM</th><th class="n">ScratchV</th><th class="n">Ratio</th></tr>
<tr><td>D$ Hit Rate</td><td class='n'><span class="llc">{LDa.get('hit_rate_pct','—')}%</span></td><td class='n'><span class="svc">{SDa.get('hit_rate_pct','—')}%</span></td><td class='n'>~1x</td></tr>
<tr><td>D$ Misses</td><td class='n'>{_f(LDa.get('misses',0))}</td><td class='n'><b>{_f(SDa.get('misses',0))}</b></td><td class='n'><b>{_r(SDa.get('misses',0),LDa.get('misses',0))}</b></td></tr>
</table>
<div class="insight"><h4>Cache 分析</h4><ul>
<li>I$ 两者 ~100% — 代码 &lt;4KB，完全适合最小 I$</li>
<li>D$ 命中率相近但 ScratchV 访存多 3.9x → 缺失体积 3.9x</li>
<li>128KB D$ 可将缺失率从 ~11% 降至 ~0.6%</li>
<li>ScratchV 内存带宽压力约为 LLVM 的 4x</li>
</ul></div></div>

<div class="section"><h2>4. TinyFive Static Code Analysis</h2><table>
<tr><th>Metric</th><th class="n">LLVM RV64FD</th><th class="n">ScratchV RV32IM</th><th class="n">Ratio</th></tr>
<tr><td>Static Instructions</td><td class='n'>{tls.get('total_static','—')}</td><td class='n'>{tss.get('total_static','—')}</td><td class='n'>{_r(tls.get('total_static',0),tss.get('total_static',0))}</td></tr>
<tr><td>Code Bytes</td><td class='n'>{_f(tls.get('code_bytes',0))}</td><td class='n'>{_f(tss.get('code_bytes',0))}</td><td class='n'>{_r(tls.get('code_bytes',0),tss.get('code_bytes',0))}</td></tr>
<tr><td>x Registers Used</td><td class='n'>{tls.get('x_reg_count','—')}</td><td class='n'>{tss.get('x_reg_count','—')}</td><td class='n'>{_r(tls.get('x_reg_count',0),tss.get('x_reg_count',0))}</td></tr>
<tr><td>f Registers Used</td><td class='n'>{tls.get('f_reg_count','—')}</td><td class='n'>{tss.get('f_reg_count','—')}</td><td class='n'>—</td></tr>
</table>
<h3>Static Op Distribution</h3><table>
<tr><th>Op Type</th><th class="n">LLVM</th><th class="n">%</th><th class="n">ScratchV</th><th class="n">%</th></tr>"""

    for k,n in [("load","Load"),("store","Store"),("mul","Mul"),("add","Add/ALU"),
                 ("madd","Mul-Add(FP)"),("branch","Branch")]:
        lv=tls.get(k,0); sv=tss.get(k,0)
        h+=f"<tr><td>{n}</td><td class='n'>{_f(lv)}</td><td class='n'>{_p(lv,tls.get('total_static',1))}</td><td class='n'>{_f(sv)}</td><td class='n'>{_p(sv,tss.get('total_static',1))}</td></tr>"
    h+="</table></div>"

    # ── 5. TinyFive Inner Loop ──
    lo=tlo.get("ops",{}); so=tso.get("ops",{})
    h+="""<div class="section"><h2>5. TinyFive Inner Loop Kernel (per-MAC)</h2>
<p style="font-size:.8rem;color:#718096;margin-bottom:10px">100 MAC 迭代，RV32IM 等价内核。TinyFive ops counters 输出。</p><table>
<tr><th>Op Counter</th><th class="n">LLVM Kernel</th><th class="n">ScratchV Kernel</th><th class="n">Ratio</th></tr>"""
    for o in ["load","store","mul","add","madd","branch","total"]:
        lv=lo.get(o,0); sv=so.get(o,0)
        h+=f"<tr><td>{o}</td><td class='n'>{lv}</td><td class='n'>{sv}</td><td class='n'>{_r(sv,lv) if lv else '—'}</td></tr>"
    h+=f"""</table><table style="margin-top:12px"><tr><th></th><th class="n">LLVM</th><th class="n">ScratchV</th></tr>
<tr><td>Insns/MAC (kernel)</td><td class='n'>{lo.get('total',0)/100:.1f}</td><td class='n'>{so.get('total',0)/100:.1f}</td></tr>
<tr><td>Insns/MAC (conv, full)</td><td class='n'>~7</td><td class='n'>~30</td></tr>
<tr><td>Insns/MAC (FC, full)</td><td class='n'>~5</td><td class='n'>~15</td></tr>
<tr><td>x Regs Used</td><td class='n'>{tlo.get('x_regs_used_count','—')}</td><td class='n'>{tso.get('x_regs_used_count','—')}</td></tr>
</table></div>"""

    # ── 6. Summary ──
    h+=f"""<div class="section"><h2>6. Full Summary</h2><table>
<tr><th>Metric</th><th class="n">LLVM RV64FD</th><th class="n">ScratchV RV32IM</th><th class="n">Ratio</th></tr>
<tr><td>Static Insns</td><td class='n'>956</td><td class='n'>785</td><td class='n'>1.2x</td></tr>
<tr><td>Dynamic Insns</td><td class='n'>{_f(L_total)}</td><td class='n'>{_f(S_total)}</td><td class='n'><b>{_r(S_total,L_total)}</b></td></tr>
<tr><td>CPI (basic)</td><td class='n'>{L_cpi:.2f}</td><td class='n'>{S_cpi:.2f}</td><td class='n'>~1x</td></tr>
<tr><td>Cycles (basic)</td><td class='n'>{_f(Lc.get('rv64fd-basic',{}).get('total_cycles',0))}</td><td class='n'>{_f(Sc.get('rv32im-basic',{}).get('total_cycles',0))}</td><td class='n'><b>{_r(Sc.get('rv32im-basic',{}).get('total_cycles',0),Lc.get('rv64fd-basic',{}).get('total_cycles',0))}</b></td></tr>
<tr><td>Time @50MHz</td><td class='n'>{Lc.get('rv64fd-basic',{}).get('est_hw_50mhz_s',0):.1f}s</td><td class='n'>{Sc.get('rv32im-basic',{}).get('est_hw_50mhz_s',0):.1f}s</td><td class='n'>{sp:.1f}x</td></tr>
<tr><td>Time @100MHz</td><td class='n'><b>{L_t100:.1f}s</b></td><td class='n'><b>{S_t100:.1f}s</b></td><td class='n'><b>{sp:.1f}x</b></td></tr>
<tr><td>Time @500MHz</td><td class='n'>{Lc.get('rv64fd-basic',{}).get('est_hw_500mhz_s',0):.1f}s</td><td class='n'>{Sc.get('rv32im-basic',{}).get('est_hw_500mhz_s',0):.1f}s</td><td class='n'>{sp:.1f}x</td></tr>
<tr><td>D$ Misses (16KB)</td><td class='n'>{_f(LDe.get('misses',0))}</td><td class='n'>{_f(SDe.get('misses',0))}</td><td class='n'><b>{_r(SDe.get('misses',0),LDe.get('misses',0))}</b></td></tr>
<tr><td>D$ Miss Bytes</td><td class='n'>{_f(LDe.get('total_miss_bytes',0))}</td><td class='n'>{_f(SDe.get('total_miss_bytes',0))}</td><td class='n'><b>{_r(SDe.get('total_miss_bytes',0),LDe.get('total_miss_bytes',0))}</b></td></tr>
<tr><td>Mem Ops</td><td class='n'>{_f(L_mem)}</td><td class='n'>{_f(S_mem)}</td><td class='n'><b>{_r(S_mem,L_mem)}</b></td></tr>
</table>
<div class="insight"><h4>关键结论</h4><ul>
<li><b>动态指令 4.2x</b> 是性能差距主因 — float32 单指令 MAC vs Q16.16 ~30 条整数指令</li>
<li><b>CPI 相近 (~1.3)</b> — 单指令效率差异小，差距在指令数</li>
<li><b>LLVM 快 {sp:.1f}x @100MHz</b> — {L_t100:.1f}s vs {S_t100:.1f}s</li>
<li><b>D$ 缺失体积 {_r(SDe.get('total_miss_bytes',0),LDe.get('total_miss_bytes',0))}</b> — 访存多 → 带宽压力大</li>
<li><b>I$ 两者 100%</b> — 代码极小(&lt;4KB)，适合任何缓存</li>
</ul></div></div>

<div class="footer">ScratchV CI Benchmark · Generated by scratchv.ci.dashboard · <a href="https://github.com/ScratchV-Compiler/ScratchV" style="color:#a0aec0">GitHub</a></div>
</div></body></html>"""
    return h


def generate_dashboard_html(
    json_path: str = "",
    json_data: dict | None = None,
    embed_json: bool = False,
    title: str = "ScratchV Performance Dashboard",
) -> str:
    """Backward-compatible wrapper called by ci_benchmark.py."""
    ld = None
    td = None
    if json_data:
        ld = json_data
    elif json_path and os.path.exists(json_path):
        with open(json_path) as f:
            ld = json.load(f)
    return generate(ld, td)


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--llvm-json", help="Pre-computed llvm_cache_compare JSON")
    p.add_argument("--tinyfive-json", help="Pre-computed tinyfive_compare JSON")
    p.add_argument("-o","--output", default="benchmark_reports/dashboard.html")
    p.add_argument("--run", action="store_true", help="Collect fresh data by running tools")
    a = p.parse_args()

    ld = None; td = None
    if a.llvm_json:
        with open(a.llvm_json) as f: ld = json.load(f)
    if a.tinyfive_json:
        with open(a.tinyfive_json) as f: td = json.load(f)

    if a.run or (ld is None and td is None):
        print("Collecting fresh data...", file=sys.stderr)
        ld, td = collect_all()

    html = generate(ld, td)
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w") as f: f.write(html)
    print(f"Dashboard: {a.output} ({len(html):,} bytes)", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
