"""Generate unit test visualization page from JUnit XML or JSON data."""

from __future__ import annotations
import json, os, sys, xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

PROJ = Path(__file__).resolve().parent.parent.parent
HIST_FILE = PROJ / "benchmark_reports" / "test_history.json"

CSS = """*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;line-height:1.6}
.wrap{max-width:900px;margin:0 auto;padding:24px 20px}
.header{background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #334155;border-radius:12px;padding:20px 28px;margin-bottom:20px}
.header h1{font-size:1.3rem;color:#f8fafc}
.header .sub{font-size:.72rem;color:#64748b;margin-top:6px}
.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:20px}
.summary .kpi{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:16px;text-align:center}
.summary .kpi .v{font-size:1.8rem;font-weight:800}
.summary .kpi .v.g{color:#22c55e}.summary .kpi .v.r{color:#ef4444}
.summary .kpi .l{font-size:.62rem;color:#64748b;text-transform:uppercase;letter-spacing:.4px;margin-top:3px}
.module{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px 16px;margin-bottom:6px;display:flex;align-items:center;gap:12px}
.module:hover{border-color:#475569}
.module .icon{width:32px;height:32px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:.9rem;font-weight:700;flex-shrink:0}
.module .icon.g{background:#052e16;color:#22c55e}.module .icon.r{background:#450a0a;color:#ef4444}
.module .info{flex:1;min-width:0}
.module .name{font-size:.8rem;font-weight:600;color:#e2e8f0}
.module .desc{font-size:.65rem;color:#64748b;margin-top:1px}
.module .bar-wrap{width:120px;height:6px;background:#334155;border-radius:3px;overflow:hidden;flex-shrink:0}
.module .bar-fill{height:100%;border-radius:3px;transition:width .3s}
.module .bar-fill.g{background:#22c55e}.module .bar-fill.r{background:#ef4444}
.module .count{font-size:.72rem;color:#94a3b8;text-align:right;flex-shrink:0;min-width:50px}
.module .time{font-size:.65rem;color:#475569;flex-shrink:0;min-width:45px;text-align:right}
.grid2{display:grid;grid-template-columns:repeat(2,1fr);gap:6px}
@media(max-width:700px){.grid2{grid-template-columns:1fr}.summary{grid-template-columns:repeat(2,1fr)}.module .bar-wrap{display:none}}
.ft{text-align:center;color:#475569;font-size:.62rem;padding:16px;margin-top:8px}
.ft a{color:#64748b}
"""

# ── Test module metadata ──────────────────────────────────────────────
MODULES = {
    "test_asm_beautifier.py":     ("ASM Beautifier",    "汇编美化器"),
    "test_asm_peephole.py":       ("ASM Peephole",      "汇编窥孔优化"),
    "test_backend.py":            ("Backend",            "LLVM IR 后端"),
    "test_bench_runner.py":       ("Bench Runner",       "基准运行器"),
    "test_cfg_builder.py":        ("CFG Builder",        "控制流图构建"),
    "test_cnn_pipeline.py":       ("CNN Pipeline",       "CNN ONNX 编译管线"),
    "test_const_merge.py":        ("Const Merge",        "常量合并优化"),
    "test_dsl_errors.py":         ("DSL Errors",         "DSL 错误处理"),
    "test_dsl_extended.py":       ("DSL Extended",       "DSL 扩展功能"),
    "test_inst_counter.py":       ("Inst Counter",       "指令计数器"),
    "test_inst_scheduler.py":     ("Inst Scheduler",     "指令调度器"),
    "test_inst_select_ext.py":    ("Inst Select Ext",    "扩展指令选择"),
    "test_ir.py":                 ("IR",                 "中间表示"),
    "test_ir_verifier.py":        ("IR Verifier",        "IR 验证器"),
    "test_llvm_codegen.py":       ("LLVM Codegen",       "LLVM 代码生成"),
    "test_logger.py":             ("Logger",             "日志系统"),
    "test_optimizer.py":          ("Optimizer",          "优化器基础"),
    "test_optimizer_advanced.py": ("Optimizer Adv",      "优化器高级"),
    "test_parser.py":             ("Parser",             "DSL 解析器"),
    "test_regalloc_linear.py":    ("RegAlloc Linear",    "线性寄存器分配"),
    "test_simulator.py":          ("Simulator",          "RISC-V 仿真器"),
    "test_verification.py":       ("Verification",       "验证模块"),
}


def parse_junit(xml_path: str) -> list[dict]:
    """Parse pytest JUnit XML into structured test results."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    results = []
    for suite in root.iter("testsuite"):
        for case in suite.iter("testcase"):
            name = case.get("classname", "").split(".")[-1] + "::" + case.get("name", "")
            passed = True
            error_msg = ""
            for child in case:
                if child.tag in ("failure", "error"):
                    passed = False
                    error_msg = child.get("message", "")[:120]
            results.append({
                "name": name,
                "passed": passed,
                "time": float(case.get("time", 0)),
                "error": error_msg,
                "file": case.get("classname", "").split(".")[0],
            })
    return results


def aggregate(results: list[dict]) -> dict:
    """Aggregate individual test results into per-module summary."""
    modules = {}
    for r in results:
        # Extract the test file name from the test name
        fname = r.get("file", "")
        if not fname:
            # Fallback: derive from test name
            fname = r["name"].split("::")[0] + ".py"
        if fname not in modules:
            modules[fname] = {"passed": 0, "failed": 0, "time": 0.0, "tests": []}
        m = modules[fname]
        m["time"] += r["time"]
        if r["passed"]:
            m["passed"] += 1
        else:
            m["failed"] += 1
        m["tests"].append(r)
    return modules


def generate(modules: dict | None = None, junit_path: str = "") -> str:
    """Generate the test visualization HTML page."""
    if junit_path and os.path.exists(junit_path):
        results = parse_junit(junit_path)
        modules = aggregate(results)

    if modules is None:
        modules = {}

    total_pass = sum(m["passed"] for m in modules.values())
    total_fail = sum(m["failed"] for m in modules.values())
    total_tests = total_pass + total_fail
    pass_rate = (total_pass / total_tests * 100) if total_tests > 0 else 100

    # Save history
    _save_history(total_pass, total_fail, len(modules), pass_rate)

    # Sort: failures first, then by name
    sorted_mods = sorted(modules.items(),
                         key=lambda x: (-x[1]["failed"], x[0]))

    h = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScratchV · Unit Tests</title><style>{CSS}</style></head><body><div class="wrap">
<div class="header">
<h1>ScratchV 单元测试</h1>
<div class="sub">{len(modules)} 个模块 · {total_tests} 个测试用例 · 更新于 {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</div>

<div class="summary">
<div class="kpi"><div class="v g">{total_pass}</div><div class="l">通过</div></div>
<div class="kpi"><div class="v{' r' if total_fail>0 else ' g'}">{total_fail}</div><div class="l">失败</div></div>
<div class="kpi"><div class="v{' g' if pass_rate>=90 else ' r'}">{pass_rate:.0f}%</div><div class="l">通过率</div></div>
</div>"""

    # ── History trend ──
    hist = _load_history()
    if len(hist) >= 2:
        h += """<div style="margin-bottom:16px;font-size:.68rem;color:#64748b">历史: """
        for entry in hist[-8:]:
            dot = "#22c55e" if entry["fail"] == 0 else "#ef4444"
            h += f"""<span style="color:{dot};margin:0 2px" title="{entry['ts'][:10]}: {entry['pass']}/{entry['pass']+entry['fail']}">●</span>"""
        h += "</div>"

    # ── Module grid ──
    h += '<div class="grid2">'
    for fname, m in sorted_mods:
        meta = MODULES.get(fname, (fname.replace("test_", "").replace(".py", "").title(), ""))
        name_en, name_cn = meta
        total = m["passed"] + m["failed"]
        pct = m["passed"] / total * 100 if total > 0 else 100
        ok = m["failed"] == 0
        h += f"""<div class="module">
<div class="icon {'g' if ok else 'r'}">{'✓' if ok else '✗'}</div>
<div class="info">
<div class="name">{name_en} <span style="color:#475569;font-size:.65rem">{name_cn}</span></div>
<div class="desc">{m['passed']}/{total} 通过 · {m['time']:.2f}s</div>
</div>
<div class="bar-wrap"><div class="bar-fill {'g' if ok else 'r'}" style="width:{pct}%"></div></div>
<div class="count">{pct:.0f}%</div>
</div>"""
    h += "</div>"

    h += f"""<div class="ft">
ScratchV CI · <a href="https://github.com/ScratchV-Compiler/ScratchV">GitHub</a>
&nbsp;·&nbsp; <a href="test_results.xml">JUnit XML</a>
&nbsp;·&nbsp; Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}
</div></div></body></html>"""
    return h


def _save_history(passed: int, failed: int, modules: int, rate: float):
    try:
        hist = json.loads(HIST_FILE.read_text()) if HIST_FILE.exists() else []
    except Exception:
        hist = []
    hist.append({
        "ts": datetime.now().isoformat(),
        "pass": passed, "fail": failed,
        "modules": modules, "rate": round(rate, 1),
    })
    hist = hist[-50:]
    HIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    HIST_FILE.write_text(json.dumps(hist, indent=2))


def _load_history() -> list[dict]:
    try:
        return json.loads(HIST_FILE.read_text()) if HIST_FILE.exists() else []
    except Exception:
        return []


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate test visualization page")
    p.add_argument("--junit", default="", help="pytest JUnit XML path")
    p.add_argument("-o", "--output", default=str(PROJ / "benchmark_reports" / "tests.html"),
                   help="Output HTML path")
    args = p.parse_args()

    modules = None
    if args.junit and os.path.exists(args.junit):
        results = parse_junit(args.junit)
        modules = aggregate(results)

    html = generate(modules)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"→ {args.output} ({len(html):,}B)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
