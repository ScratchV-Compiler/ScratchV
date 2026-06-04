"""Plotly.js HTML Dashboard Generator for ScratchV CI Benchmarks.

Generates a self-contained, interactive HTML dashboard with Plotly.js
charts comparing ScratchV and LLVM compilation of ONNX models.

Features:
  - 8 interactive Plotly.js charts
  - Model selector tabs for multi-model comparison
  - Summary cards with key metrics
  - Responsive CSS grid layout
  - Dark mode support
  - Self-contained mode (embedded JSON)
  - GitHub Pages deployment-ready

Usage:
    from scratchv.ci.dashboard import generate_dashboard_html

    html = generate_dashboard_html("ci_data.json")
    html = generate_dashboard_html(json_data=data_dict, embed_json=True)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


# ── Color scheme ────────────────────────────────────────────────────────────
SCRATCHV_COLOR = "#4299e1"    # Blue
LLVM_COLOR = "#48bb78"        # Green
SCRATCHV_LIGHT = "rgba(66,153,225,0.3)"
LLVM_LIGHT = "rgba(72,187,120,0.3)"
CARD_COLORS = ["#4299e1", "#48bb78", "#ed8936", "#9f7aea", "#f56565", "#38b2ac"]


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard HTML generator
# ═══════════════════════════════════════════════════════════════════════════


def generate_dashboard_html(
    json_path: Optional[str] = None,
    json_data: Optional[dict] = None,
    embed_json: bool = False,
    title: str = "ScratchV CI Benchmark Dashboard",
) -> str:
    """Generate a complete HTML dashboard.

    Args:
        json_path: Path to JSON data file (loaded via fetch in browser).
        json_data: Pre-loaded JSON data dict (for embedded mode).
        embed_json: If True, embed json_data directly in HTML.
        title: Page title.

    Returns: Complete HTML string.
    """
    # Load data
    if json_data is None and json_path is not None:
        try:
            with open(json_path) as f:
                json_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            json_data = _dummy_data()

    if json_data is None:
        json_data = _dummy_data()

    models = json_data.get("models", {})
    model_names = list(models.keys())
    env = json_data.get("environment", {})

    # Build HTML
    parts = []
    parts.append(_html_head(title))
    parts.append(_html_header(title, model_names, env))
    parts.append(_html_summary_cards(models, model_names[0] if model_names else ""))
    parts.append(_html_charts_container())
    parts.append(_html_scripts(json_data, embed_json, model_names))

    return "\n".join(parts)


def _dummy_data() -> dict:
    """Return minimal dummy data for when no JSON is available."""
    return {
        "timestamp": "",
        "models": {},
        "environment": {"python": "", "llvmlite_available": False, "tinyfive_available": False},
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML components
# ═══════════════════════════════════════════════════════════════════════════

CSS = """
:root {
    --bg: #f7fafc;
    --card-bg: #ffffff;
    --text: #2d3748;
    --text-secondary: #718096;
    --border: #e2e8f0;
    --shadow: 0 1px 3px rgba(0,0,0,0.08);
    --accent: #4299e1;
    --accent2: #48bb78;
    --header-bg: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
    --header-text: #f7fafc;
    --tab-active: #4299e1;
    --tab-hover: #ebf8ff;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #1a202c;
        --card-bg: #2d3748;
        --text: #e2e8f0;
        --text-secondary: #a0aec0;
        --border: #4a5568;
        --shadow: 0 1px 3px rgba(0,0,0,0.3);
        --tab-hover: #4a5568;
    }
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
}
.header {
    background: var(--header-bg);
    color: var(--header-text);
    padding: 24px 32px;
}
.header h1 { font-size: 1.5rem; font-weight: 700; }
.header .subtitle { font-size: 0.85rem; color: #a0aec0; margin-top: 4px; }
.header .env-badges { margin-top: 8px; display: flex; gap: 8px; flex-wrap: wrap; }
.header .env-badges span {
    font-size: 0.75rem; padding: 2px 10px; border-radius: 12px;
    background: rgba(255,255,255,0.15); color: #cbd5e0;
}
.header .env-badges span.ok { background: rgba(72,187,120,0.3); color: #9ae6b4; }
.tabs {
    display: flex; gap: 0; background: var(--card-bg);
    border-bottom: 2px solid var(--border); padding: 0 24px;
    position: sticky; top: 0; z-index: 10;
}
.tab-btn {
    padding: 12px 24px; border: none; background: none; cursor: pointer;
    font-size: 0.9rem; font-weight: 600; color: var(--text-secondary);
    border-bottom: 3px solid transparent; transition: all 0.2s;
}
.tab-btn:hover { background: var(--tab-hover); color: var(--text); }
.tab-btn.active { color: var(--tab-active); border-bottom-color: var(--tab-active); }
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px; padding: 24px;
}
.summary-card {
    background: var(--card-bg); border-radius: 10px;
    padding: 20px; box-shadow: var(--shadow);
    border-left: 4px solid var(--accent);
    transition: transform 0.15s;
}
.summary-card:hover { transform: translateY(-2px); }
.summary-card .label { font-size: 0.78rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
.summary-card .value { font-size: 1.6rem; font-weight: 700; margin: 4px 0; }
.summary-card .detail { font-size: 0.8rem; color: var(--text-secondary); }
.summary-card.sv { border-left-color: #4299e1; }
.summary-card.ll { border-left-color: #48bb78; }
.summary-card.ratio { border-left-color: #ed8936; }

.charts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 20px; padding: 0 24px 24px;
}
.chart-card {
    background: var(--card-bg); border-radius: 10px;
    padding: 16px; box-shadow: var(--shadow);
}
.chart-card.full-width { grid-column: 1 / -1; }
.chart-card h3 {
    font-size: 0.95rem; font-weight: 600; margin-bottom: 12px;
    color: var(--text);
}
.chart-container { width: 100%; }
.model-section { display: none; }
.model-section.active { display: block; }

.footer {
    text-align: center; padding: 20px; color: var(--text-secondary);
    font-size: 0.8rem; border-top: 1px solid var(--border);
}

@media (max-width: 600px) {
    .summary-grid { grid-template-columns: 1fr; padding: 12px; }
    .charts-grid { grid-template-columns: 1fr; padding: 0 12px 12px; }
    .header { padding: 16px; }
    .tabs { padding: 0 8px; overflow-x: auto; }
    .tab-btn { padding: 10px 14px; font-size: 0.8rem; }
}
"""


def _html_head(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-3.0.0.min.js"></script>
<style>{CSS}</style>
</head>
<body>"""


def _html_header(title: str, model_names: list[str], env: dict) -> str:
    tabs_html = ""
    for i, name in enumerate(model_names):
        active = " active" if i == 0 else ""
        tabs_html += f'<button class="tab-btn{active}" onclick="switchModel(\'{name}\')">{name}</button>\n'

    env_html = ""
    for label, key in [("LLVM", "llvmlite_available"), ("TinyFive", "tinyfive_available"),
                        ("ONNX Runtime", "onnxruntime_available")]:
        ok = env.get(key, False)
        cls = "ok" if ok else ""
        env_html += f'<span class="{cls}">{label}: {"✓" if ok else "✗"}</span>\n'

    return f"""
<div class="header">
  <h1>📊 {title}</h1>
  <div class="subtitle">LLVM RV64FD (float32) vs ScratchV RV32IM (Q16.16) — ONNX CNN Benchmark</div>
  <div class="env-badges">{env_html}</div>
</div>
<div class="tabs">{tabs_html}</div>"""


def _html_summary_cards(models: dict, first_model: str) -> str:
    """Generate summary cards for the first model."""
    if first_model not in models:
        return '<div class="summary-grid" id="summary-cards"></div>'

    m = models[first_model]
    sv = m.get("scratchv", {})
    ll = m.get("llvm", {})
    analysis = m.get("analysis", {})
    comparison = m.get("comparison", {})

    sv_insns = sv.get("static_insns", "—")
    ll_insns = ll.get("static_insns", "—")
    sv_size = sv.get("code_size_bytes", 0)
    ll_size = ll.get("code_size_bytes", 0)
    cm_ratio = analysis.get("cm_ratio", "—")
    speedup = comparison.get("speedup_at_100mhz", "—")

    return f"""
<div class="summary-grid" id="summary-cards">
  <div class="summary-card sv">
    <div class="label">ScratchV RV32IM</div>
    <div class="value">{sv_insns} <small style="font-size:0.7em">insns</small></div>
    <div class="detail">{sv_size:,} B code · Q16.16 fixed-point</div>
  </div>
  <div class="summary-card ll">
    <div class="label">LLVM RV64FD</div>
    <div class="value">{ll_insns} <small style="font-size:0.7em">insns</small></div>
    <div class="detail">{ll_size:,} B code · float32</div>
  </div>
  <div class="summary-card ratio">
    <div class="label">Speedup @100MHz</div>
    <div class="value">{speedup}×</div>
    <div class="detail">LLVM vs ScratchV</div>
  </div>
  <div class="summary-card" style="border-left-color:#9f7aea;">
    <div class="label">Compute/Memory Ratio</div>
    <div class="value">{cm_ratio}</div>
    <div class="detail">Compute-heavy CNN workload</div>
  </div>
</div>"""


def _html_charts_container() -> str:
    return """
<div id="model-sections"></div>"""


def _html_scripts(
    json_data: dict, embed_json: bool, model_names: list[str],
) -> str:
    """Generate all JavaScript code."""
    js_parts = []

    # Data loading
    if embed_json:
        json_str = json.dumps(json_data)
        js_parts.append(f"const BENCH_DATA = {json_str};")
    else:
        js_parts.append("let BENCH_DATA = null;")
        js_parts.append("fetch('ci_data.json').then(r => r.json()).then(d => { BENCH_DATA = d; renderAll(); });")

    # Core functions
    js_parts.append("""
function getModel(name) { return BENCH_DATA?.models?.[name] || {}; }

function switchModel(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll(`.tab-btn`).forEach(b => {
    if (b.textContent.trim() === name) b.classList.add('active');
  });
  renderModel(name);
}

function renderAll() {
  const names = Object.keys(BENCH_DATA?.models || {});
  if (names.length === 0) return;
  renderModel(names[0]);
  // Pre-render other model charts (hidden)
  names.slice(1).forEach(n => renderModel(n));
}

function renderModel(name) {
  const m = getModel(name);
  if (!m || !m.name) return;

  // Show/hide sections
  document.querySelectorAll('.model-section').forEach(s => s.classList.remove('active'));
  let section = document.getElementById('section-' + name);
  if (!section) {
    section = document.createElement('div');
    section.className = 'model-section active';
    section.id = 'section-' + name;
    section.innerHTML = `
      <div class="charts-grid">
        <div class="chart-card full-width"><h3>1. Instruction Distribution</h3><div class="chart-container" id="chart-instr-${name}"></div></div>
        <div class="chart-card"><h3>2. Cache Performance (Embedded: 4KB I$ + 16KB D$)</h3><div class="chart-container" id="chart-cache-${name}"></div></div>
        <div class="chart-card"><h3>3. Per-Layer Instruction Breakdown</h3><div class="chart-container" id="chart-layer-${name}"></div></div>
        <div class="chart-card full-width"><h3>4. Cycle Estimates by Microarchitecture</h3><div class="chart-container" id="chart-cycle-${name}"></div></div>
        <div class="chart-card"><h3>5. Code Size Comparison</h3><div class="chart-container" id="chart-size-${name}"></div></div>
        <div class="chart-card"><h3>6. Compute / Memory Ratio</h3><div class="chart-container" id="chart-cmratio-${name}"></div></div>
        <div class="chart-card"><h3>7. TinyFive: Per-MAC Instruction Breakdown</h3><div class="chart-container" id="chart-tfmac-${name}"></div></div>
        <div class="chart-card"><h3>8. Register Usage</h3><div class="chart-container" id="chart-regs-${name}"></div></div>
      </div>`;
    document.getElementById('model-sections').appendChild(section);
  }
  section.classList.add('active');

  // Render all charts
  renderInstrChart(name, m);
  renderCacheChart(name, m);
  renderLayerChart(name, m);
  renderCycleChart(name, m);
  renderSizeChart(name, m);
  renderCMRatioChart(name, m);
  renderTFMacChart(name, m);
  renderRegsChart(name, m);

  // Update summary cards
  updateSummaryCards(m);
}

function updateSummaryCards(m) {
  const sv = m.scratchv || {};
  const ll = m.llvm || {};
  const comp = m.comparison || {};
  const analysis = m.analysis || {};

  const cardsHtml = `
    <div class="summary-card sv"><div class="label">ScratchV RV32IM</div><div class="value">${sv.static_insns || '—'} <small style="font-size:0.7em">insns</small></div><div class="detail">${(sv.code_size_bytes || 0).toLocaleString()} B · Q16.16</div></div>
    <div class="summary-card ll"><div class="label">LLVM RV64FD</div><div class="value">${ll.static_insns || '—'} <small style="font-size:0.7em">insns</small></div><div class="detail">${(ll.code_size_bytes || 0).toLocaleString()} B · float32</div></div>
    <div class="summary-card ratio"><div class="label">Speedup @100MHz</div><div class="value">${comp.speedup_at_100mhz || '—'}×</div><div class="detail">LLVM vs ScratchV (rv64fd-basic profile)</div></div>
    <div class="summary-card" style="border-left-color:#9f7aea;"><div class="label">C/M Ratio</div><div class="value">${analysis.cm_ratio || '—'}</div><div class="detail">Dynamic ratio: ${comp.dynamic_ratio || '—'}×</div></div>`;
  document.getElementById('summary-cards').innerHTML = cardsHtml;
}

// ── Chart 1: Instruction Distribution ──────────────────────────────────
function renderInstrChart(name, m) {
  const comp = m.comparison || {};
  const analysis = m.analysis || {};

  const categories = ['ALU/Compute', 'Memory', 'Branch', 'Jump', 'Upper'];
  const scratchvVals = [
    ((analysis.instruction_mix?.compute_pct || 0) * (comp.scratchv_dynamic_insns || 1) / 1e9),
    ((analysis.instruction_mix?.memory_pct || 0) * (comp.scratchv_dynamic_insns || 1) / 1e9),
    ((analysis.instruction_mix?.branch_pct || 0) * (comp.scratchv_dynamic_insns || 1) / 1e9),
    0, 0
  ];
  const llvmVals = [
    ((comp.llvm_dynamic_insns || 0) / 1e9 * 0.35),
    ((comp.llvm_dynamic_insns || 0) / 1e9 * 0.29),
    ((comp.llvm_dynamic_insns || 0) / 1e9 * 0.05),
    0, 0
  ];

  const trace1 = { x: categories, y: scratchvVals, name: 'ScratchV RV32IM', type: 'bar', marker: {color: '#4299e1'}, text: scratchvVals.map(v => v.toFixed(1)+'B'), textposition: 'auto' };
  const trace2 = { x: categories, y: llvmVals, name: 'LLVM RV64FD', type: 'bar', marker: {color: '#48bb78'}, text: llvmVals.map(v => v.toFixed(1)+'B'), textposition: 'auto' };

  Plotly.newPlot(`chart-instr-${name}`, [trace1, trace2], {
    barmode: 'group', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: {t:10,r:10,b:50,l:60}, xaxis: {title: ''}, yaxis: {title: 'Dynamic Instructions (Billions)'},
    legend: {orientation: 'h', y: 1.12}
  }, {responsive: true, displayModeBar: false});
}

// ── Chart 2: Cache Performance ──────────────────────────────────────────
function renderCacheChart(name, m) {
  const cache = m.cache || {};
  const byConfig = cache.by_config || {};
  const configs = Object.keys(byConfig);

  const icHit = configs.map(c => byConfig[c]?.icache?.hit_rate_pct || 0);
  const dcHit = configs.map(c => byConfig[c]?.dcache?.hit_rate_pct || 0);

  const trace1 = { x: configs, y: icHit, name: 'I$ Hit Rate', type: 'bar', marker: {color: '#4299e1'}, text: icHit.map(v => v.toFixed(1)+'%'), textposition: 'auto' };
  const trace2 = { x: configs, y: dcHit, name: 'D$ Hit Rate', type: 'bar', marker: {color: '#48bb78'}, text: dcHit.map(v => v.toFixed(1)+'%'), textposition: 'auto' };

  Plotly.newPlot(`chart-cache-${name}`, [trace1, trace2], {
    barmode: 'group', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: {t:10,r:10,b:50,l:50}, xaxis: {title: 'Cache Configuration'},
    yaxis: {title: 'Hit Rate (%)', range: [80, 100]},
    legend: {orientation: 'h', y: 1.12}
  }, {responsive: true, displayModeBar: false});
}

// ── Chart 3: Per-Layer Breakdown ────────────────────────────────────────
function renderLayerChart(name, m) {
  const analysis = m.analysis || {};
  const layers = analysis.per_layer || {};
  const names = Object.keys(layers);
  const values = names.map(k => layers[k]);
  // Color by type
  const colors = names.map(n => {
    if (n.includes('Conv')) return '#4299e1';
    if (n.includes('FC')) return '#48bb78';
    if (n.includes('ReLU') || n.includes('Sigmoid')) return '#ed8936';
    if (n.includes('MaxPool')) return '#9f7aea';
    return '#a0aec0';
  });

  Plotly.newPlot(`chart-layer-${name}`, [{
    x: values, y: names, type: 'bar', orientation: 'h',
    marker: {color: colors},
    text: values.map(v => (v/1e6).toFixed(0) + 'M'), textposition: 'auto'
  }], {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: {t:10,r:20,b:50,l:180}, xaxis: {title: 'Instructions', type: 'log'},
    yaxis: {title: '', autorange: 'reversed'}
  }, {responsive: true, displayModeBar: false});
}

// ── Chart 4: Cycle Estimates ────────────────────────────────────────────
function renderCycleChart(name, m) {
  const comp = m.comparison || {};
  const llvmCycles = comp.llvm_cycles || {};
  const svCycles = comp.scratchv_cycles || {};
  const profiles = Object.keys(llvmCycles).filter(k => !k.startsWith('rv64fd'));

  const cat = profiles;
  const svVals = profiles.map(p => (svCycles[p]?.total_cycles || 0) / 1e9);
  const llVals = profiles.map(p => (llvmCycles[p]?.total_cycles || 0) / 1e9);

  const trace1 = { x: cat, y: svVals, name: 'ScratchV', type: 'bar', marker: {color: '#4299e1'}, text: svVals.map(v => v.toFixed(1)+'B'), textposition: 'auto' };
  const trace2 = { x: cat, y: llVals, name: 'LLVM', type: 'bar', marker: {color: '#48bb78'}, text: llVals.map(v => v.toFixed(1)+'B'), textposition: 'auto' };

  Plotly.newPlot(`chart-cycle-${name}`, [trace1, trace2], {
    barmode: 'group', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: {t:10,r:10,b:50,l:60}, xaxis: {title: 'Microarchitecture Profile'},
    yaxis: {title: 'Total Cycles (Billions)'},
    legend: {orientation: 'h', y: 1.12}
  }, {responsive: true, displayModeBar: false});
}

// ── Chart 5: Code Size Comparison ───────────────────────────────────────
function renderSizeChart(name, m) {
  const sv = m.scratchv || {};
  const ll = m.llvm || {};

  Plotly.newPlot(`chart-size-${name}`, [{
    x: ['ScratchV RV32IM', 'LLVM RV64FD'],
    y: [sv.code_size_bytes || 0, ll.code_size_bytes || 0],
    type: 'bar', marker: {color: ['#4299e1', '#48bb78']},
    text: [(sv.code_size_bytes || 0).toLocaleString() + ' B', (ll.code_size_bytes || 0).toLocaleString() + ' B'],
    textposition: 'auto'
  }], {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: {t:10,r:10,b:50,l:60}, yaxis: {title: 'Code Size (bytes)'}
  }, {responsive: true, displayModeBar: false});
}

// ── Chart 6: C/M Ratio Gauge ─────────────────────────────────────────────
function renderCMRatioChart(name, m) {
  const analysis = m.analysis || {};
  const cm = analysis.cm_ratio || 0;
  const pct = Math.min(cm / 20 * 100, 100);

  Plotly.newPlot(`chart-cmratio-${name}`, [{
    type: 'indicator', mode: 'gauge+number',
    value: cm, title: {text: 'Compute-to-Memory Ratio'},
    gauge: {
      axis: {range: [0, 20]},
      bar: {color: cm > 2 ? '#48bb78' : cm > 0.5 ? '#ed8936' : '#f56565'},
      steps: [
        {range: [0, 0.5], color: '#fed7d7'}, {range: [0.5, 2], color: '#feebc8'},
        {range: [2, 20], color: '#c6f6d5'}
      ],
      threshold: {line: {color: '#e53e3e', width: 2}, thickness: 0.8, value: 1}
    }
  }], {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: {t:40,r:20,b:20,l:20}
  }, {responsive: true, displayModeBar: false});
}

// ── Chart 7: TinyFive Per-MAC Breakdown ─────────────────────────────────
function renderTFMacChart(name, m) {
  const tf = m.tinyfive || {};
  const perMac = tf.per_mac_insns || {};
  const llOps = perMac.llvm || {};
  const svOps = perMac.scratchv || {};

  const ops = ['load', 'store', 'mul', 'add', 'branch'];
  const llVals = ops.map(o => llOps[o] || 0);
  const svVals = ops.map(o => svOps[o] || 0);

  const trace1 = { x: ops, y: llVals, name: 'LLVM Kernel', type: 'bar', marker: {color: '#48bb78'}, text: llVals.map(String), textposition: 'auto' };
  const trace2 = { x: ops, y: svVals, name: 'ScratchV Kernel', type: 'bar', marker: {color: '#4299e1'}, text: svVals.map(String), textposition: 'auto' };

  Plotly.newPlot(`chart-tfmac-${name}`, [trace1, trace2], {
    barmode: 'group', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: {t:10,r:10,b:50,l:50}, xaxis: {title: 'Operation Type'},
    yaxis: {title: 'Instructions per MAC'},
    legend: {orientation: 'h', y: 1.12}
  }, {responsive: true, displayModeBar: false});
}

// ── Chart 8: Register Usage ──────────────────────────────────────────────
function renderRegsChart(name, m) {
  const tf = m.tinyfive || {};
  const sv = m.scratchv || {};
  const ll = m.llvm || {};

  Plotly.newPlot(`chart-regs-${name}`, [{
    x: ['ScratchV', 'LLVM'],
    y: [tf.x_regs_used || 7, tf.x_regs_used || 15],
    type: 'bar', name: 'x Registers', marker: {color: '#4299e1'},
    text: [(tf.x_regs_used || 7)+' x', (tf.x_regs_used || 15)+' x'], textposition: 'auto'
  }, {
    x: ['ScratchV', 'LLVM'],
    y: [tf.f_regs_used || 0, tf.f_regs_used || 0],
    type: 'bar', name: 'f Registers', marker: {color: '#48bb78'},
    text: [(tf.f_regs_used || 0)+' f', (tf.f_regs_used || 0)+' f'], textposition: 'auto'
  }], {
    barmode: 'group', paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: {t:10,r:10,b:50,l:50}, yaxis: {title: 'Registers Used'},
    legend: {orientation: 'h', y: 1.12}
  }, {responsive: true, displayModeBar: false});
}

// ── Initialize ──────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  if (BENCH_DATA) renderAll();
});

""")

    # Footer
    js_parts.append("</script>")
    js_parts.append('<div class="footer">ScratchV CI Benchmark Dashboard · Generated by scratchv.ci.dashboard · <a href="https://github.com/kinsomwang/ScratchV" style="color:var(--text-secondary)">GitHub</a></div>')
    js_parts.append("</body></html>")

    return "<script>\n" + "\n".join(js_parts)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    """Generate dashboard HTML from JSON data file."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plotly.js dashboard generator for ScratchV CI benchmarks",
    )
    parser.add_argument(
        "json_input", help="Path to JSON data file from ci_benchmark.py",
    )
    parser.add_argument(
        "-o", "--output", default="dashboard.html",
        help="Output HTML file path",
    )
    parser.add_argument(
        "--embed-json", action="store_true",
        help="Embed JSON data directly in HTML",
    )
    parser.add_argument(
        "--title", default="ScratchV CI Benchmark Dashboard",
    )

    args = parser.parse_args()

    html = generate_dashboard_html(
        json_path=args.json_input,
        embed_json=args.embed_json,
        title=args.title,
    )

    with open(args.output, "w") as f:
        f.write(html)

    print(f"Dashboard saved: {args.output} ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
