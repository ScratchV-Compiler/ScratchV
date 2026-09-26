#!/usr/bin/env python3
"""Build HTML course site from ScratchV markdown docs.

Generates a multi-level course-directory website at docs/topics/html/
with sidebar navigation, search, dark/light theme, progress tracking,
and responsive design.  All pages are pure static HTML — zero runtime
dependencies, deployable directly to GitHub Pages.

Usage:
    python scripts/build_docs_html.py
    python scripts/build_docs_html.py --output-dir docs/topics/html
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

# ── requires: pip install markdown ──────────────────────────────────────────
try:
    import markdown as md_lib
except ImportError:
    print("ERROR: pip install markdown", file=sys.stderr)
    sys.exit(1)

# ── Project root ─────────────────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parent.parent
DOCS = PROJ / "docs"
TOPICS = DOCS / "topics"
OUT = TOPICS / "html"

# ═══════════════════════════════════════════════════════════════════════════════
# Course data — defines the site structure
# ═══════════════════════════════════════════════════════════════════════════════

COURSE = {
    "title": "ScratchV 编译器课程",
    "subtitle": "从零开始，亲手搭建一个 AI→RISC-V 编译器",
    "sections": [
        {
            "id": "foundation",
            "title": "基础入门",
            "icon": "📘",
            "desc": "环境搭建、编译器概念、快速上手、指标解读、故障排除",
            "docs": [
                {"file": "docs/guide/00-环境搭建指南.md", "num": "00", "title": "环境搭建指南", "time": "15 min"},
                {"file": "docs/guide/01-编译器概念入门.md", "num": "01", "title": "编译器概念入门", "time": "15 min"},
                {"file": "docs/guide/02-快速上手教程.md", "num": "02", "title": "快速上手教程", "time": "30 min"},
                {"file": "docs/guide/03-指标解读指南.md", "num": "03", "title": "指标解读指南", "time": "15 min"},
                {"file": "docs/guide/04-故障排除FAQ.md", "num": "04", "title": "故障排除 FAQ", "time": "10 min"},
            ],
        },
        {
            "id": "beginner-build",
            "title": "入门模块 — 动手做",
            "icon": "🔰",
            "desc": "从零开始实现编译器的基础工具：美化器、日志、CFG、窥孔、调度",
            "docs": [
                {"file": "docs/topics/05-汇编代码美化器/05-汇编代码美化器.md", "num": "05", "title": "汇编代码美化器"},
                {"file": "docs/topics/07-编译器日志增强器/07-编译器日志增强器.md", "num": "07", "title": "编译器日志增强器"},
                {"file": "docs/topics/09-DSL错误提示美化器/09-DSL错误提示美化器.md", "num": "09", "title": "DSL 错误提示美化器"},
                {"file": "docs/topics/11-控制流图生成器/11-控制流图生成器.md", "num": "11", "title": "控制流图 (CFG) 生成器"},
                {"file": "docs/topics/13-窥孔优化器/13-窥孔优化器.md", "num": "13", "title": "窥孔优化器"},
                {"file": "docs/topics/18-指令调度器/18-指令调度器.md", "num": "18", "title": "指令调度器"},
                {"file": "docs/topics/20-代码规范/20-代码规范.md", "num": "20", "title": "代码规范与格式化"},
            ],
        },
        {
            "id": "beginner-observe",
            "title": "入门模块 — 学会看",
            "icon": "📊",
            "desc": "学习使用性能分析与指令计数工具",
            "docs": [
                {"file": "docs/topics/06-性能基准套件/06-性能基准套件.md", "num": "06", "title": "性能基准套件"},
                {"file": "docs/topics/12-指令计数统计器/12-指令计数统计器.md", "num": "12", "title": "指令计数统计器"},
            ],
        },
        {
            "id": "intermediate",
            "title": "中级模块",
            "icon": "📗",
            "desc": "核心编译器管线：DSL 增强、常量加载合并、寄存器分配",
            "docs": [
                {"file": "docs/topics/01-DSL前端增强器/01-DSL前端增强器.md", "num": "01", "title": "DSL 前端增强器"},
                {"file": "docs/topics/14-常量加载合并/14-常量加载合并.md", "num": "14", "title": "常量加载合并优化"},
                {"file": "docs/topics/17-寄存器分配/17-寄存器分配.md", "num": "17", "title": "寄存器分配"},
            ],
        },
        {
            "id": "advanced",
            "title": "高级模块",
            "icon": "📕",
            "desc": "深入实现：IR 验证、扩展指令选择",
            "docs": [
                {"file": "docs/topics/21-IR验证器/21-IR验证器.md", "num": "21", "title": "IR 验证器"},
                {"file": "docs/topics/28-扩展指令选择/28-扩展指令选择.md", "num": "28", "title": "扩展指令选择 (F/D/abs/sqrt)"},
            ],
        },
    ],
}

# ── Flatten to ordered list for prev/next ────────────────────────────────────
ALL_DOCS = []
for sec in COURSE["sections"]:
    for d in sec["docs"]:
        d["section_id"] = sec["id"]
        ALL_DOCS.append(d)

# ═══════════════════════════════════════════════════════════════════════════════
# HTML template engine
# ═══════════════════════════════════════════════════════════════════════════════

def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_page(*, title: str, body_class: str = "", content: str = "",
                breadcrumbs: list[tuple[str, str]] | None = None,
                current_section: str = "", current_doc: str = "",
                prev_doc: dict | None = None, next_doc: dict | None = None) -> str:
    """Render a full HTML page from the shared template."""

    bc_html = ""
    if breadcrumbs:
        parts = []
        for label, href in breadcrumbs:
            parts.append(f'<a href="{_escape(href)}">{_escape(label)}</a>')
        bc_html = ' <span class="bc-sep">›</span> '.join(parts)

    sidebar = _build_sidebar(current_section, current_doc)

    nav_prev = ""
    nav_next = ""
    if prev_doc:
        nav_prev = f"""<a class="page-nav prev" href="{_prev_next_href(prev_doc)}">
            <span class="pn-label">← 上一节</span>
            <span class="pn-title">{_escape(prev_doc['title'])}</span>
        </a>"""
    if next_doc:
        nav_next = f"""<a class="page-nav next" href="{_prev_next_href(next_doc)}">
            <span class="pn-label">下一节 →</span>
            <span class="pn-title">{_escape(next_doc['title'])}</span>
        </a>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_escape(title)} — ScratchV 编译器课程</title>
<meta name="description" content="ScratchV 零基础编译器课程 — 从 ONNX 到 RISC-V 的全流程实战">
{_css()}
</head>
<body class="{body_class}">
<div id="app">
  <button class="sidebar-toggle" id="sidebarToggle" aria-label="菜单">☰</button>
  <div class="sidebar-overlay" id="sidebarOverlay"></div>

  <aside class="sidebar" id="sidebar">
    <div class="sidebar-header">
      <a href="index.html" class="sidebar-logo">
        <span class="logo-icon">⚡</span>
        <span class="logo-text">ScratchV</span>
      </a>
      <p class="sidebar-subtitle">编译器课程</p>
    </div>
    <div class="sidebar-search">
      <input type="text" id="searchInput" placeholder="搜索课程内容..." autocomplete="off">
      <div id="searchResults" class="search-results"></div>
    </div>
    <nav class="sidebar-nav">
      {sidebar}
    </nav>
    <div class="sidebar-footer">
      <button class="theme-toggle" id="themeToggle" aria-label="切换主题">🌓 主题</button>
      <a href="../dashboard.html" class="gh-link" title="性能仪表盘">📊</a>
      <a href="../history.html" class="gh-link" title="优化历史">📈</a>
      <a href="https://github.com/ScratchV-Compiler/ScratchV" class="gh-link" target="_blank" rel="noopener">GitHub</a>
    </div>
  </aside>

  <main class="main">
    <div class="topbar">
      <div class="breadcrumbs">{bc_html}</div>
      <div class="topbar-actions">
        <button class="btn-progress" id="btnProgress" title="学习进度">📊</button>
      </div>
    </div>

    <article class="content">
      {_build_progress_bar()}
      {content}
    </article>

    <footer class="content-footer">
      <nav class="page-nav-row">
        {nav_prev}
        {nav_next}
      </nav>
      <div class="footer-info">
        <p>ScratchV — AI 编译器实战课程 · <a href="https://github.com/ScratchV-Compiler/ScratchV">GitHub</a></p>
      </div>
    </footer>
  </main>
</div>

<div class="progress-toast" id="progressToast"></div>
{_js()}
</body>
</html>"""


def _prev_next_href(doc: dict) -> str:
    title = doc["title"]
    num = doc["num"]
    safe = title.replace(" ", "-").replace("(", "").replace(")", "").replace("/", "-")
    return f"{num}-{safe}.html"


# ── Build MD path → HTML filename mapping ──────────────────────────────────
_MD_TO_HTML: dict[str, str] = {}
for _d in ALL_DOCS:
    rel = _d["file"]
    if rel.startswith("docs/"):
        rel = rel[5:]
    html_name = _prev_next_href(_d)
    _MD_TO_HTML[rel] = html_name
    _MD_TO_HTML[_d["file"]] = html_name


def _md_link_to_html(md_path: str) -> str:
    """Convert a .md link to the correct .html filename using COURSE mapping."""
    cleaned = md_path
    while cleaned.startswith("../"):
        cleaned = cleaned[3:]
    cleaned = cleaned.lstrip("./")

    # Deliberately no bare-basename fallback: several docs legitimately share a
    # basename (e.g. INDEX.md), and a basename match would silently point a link
    # at the wrong page.  An unmatched link degrades to an un-rewritten .md path,
    # which check_docs_links.py catches, instead of mis-resolving.
    for key, val in _MD_TO_HTML.items():
        if key.endswith(cleaned) or cleaned.endswith(key) or key == cleaned:
            return val

    if cleaned.endswith(".md"):
        filename = cleaned.split("/")[-1]
        if filename in ("ARCHITECTURE.md", "developer_guide.md",
                        "optimization_guide.md", "verification.md",
                        "CODING_STANDARDS.md", "CONTRIBUTING.md",
                        "ScratchV.md", "INDEX.md"):
            return f"https://github.com/ScratchV-Compiler/ScratchV/blob/main/docs/{cleaned}"
        if "topics/INDEX" in cleaned or cleaned == "INDEX.md":
            return f"https://github.com/ScratchV-Compiler/ScratchV/blob/main/docs/topics/INDEX.md"

    return md_path


def _build_sidebar(current_section: str, current_doc: str) -> str:
    """Build sidebar navigation HTML."""
    parts = []
    parts.append('<ul class="sidenav">')
    parts.append(
        f'<li><a href="index.html" class="sidenav-home">🏠 课程首页</a></li>'
    )

    for sec in COURSE["sections"]:
        is_active_section = sec["id"] == current_section
        cls = "active" if is_active_section else ""
        parts.append(
            f'<li class="sidenav-section {cls}">'
            f'<button class="sidenav-section-btn" aria-expanded="{str(is_active_section).lower()}">'
            f'<span>{sec["icon"]} {_escape(sec["title"])}</span>'
            f'<span class="sidenav-count">{len(sec["docs"])}</span>'
            f'</button>'
            f'<ul class="sidenav-sub">'
        )

        for d in sec["docs"]:
            href = _prev_next_href(d)
            active = "active" if d["title"] == current_doc else ""
            num = d.get("num", "")
            parts.append(
                f'<li><a href="{href}" class="{active}" data-doc="{_escape(d["title"])}">'
                f'<span class="sidenav-num">{num}</span>'
                f'<span class="sidenav-doc-title">{_escape(d["title"])}</span>'
                f'<span class="sidenav-check" data-check="{_escape(d["title"])}"></span>'
                f'</a></li>'
            )

        parts.append('</ul></li>')

    parts.append('</ul>')
    return "\n".join(parts)


def _build_progress_bar() -> str:
    return """<div class="progress-bar-wrap" id="progressBar">
      <div class="progress-bar-fill" id="progressBarFill"></div>
      <span class="progress-bar-text" id="progressBarText"></span>
    </div>"""


def _css() -> str:
    return """<style>
/* ═══════════════════════════════════════════════════════════════════════════
   ScratchV Course — Design System
   ═══════════════════════════════════════════════════════════════════════════ */

:root {
  --bg: #f8f9fb;
  --bg-card: #ffffff;
  --bg-sidebar: #1a1d28;
  --text: #2c3e50;
  --text-secondary: #6b7280;
  --text-sidebar: #c8cdd8;
  --text-sidebar-active: #ffffff;
  --accent: #2563eb;
  --accent-light: #dbeafe;
  --accent-glow: rgba(37, 99, 235, 0.15);
  --border: #e5e7eb;
  --code-bg: #1e293b;
  --code-text: #e2e8f0;
  --badge-green: #059669;
  --badge-green-bg: #d1fae5;
  --badge-yellow: #d97706;
  --badge-yellow-bg: #fef3c7;
  --badge-red: #dc2626;
  --badge-red-bg: #fee2e2;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
  --shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
  --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
  --radius: 12px;
  --radius-sm: 8px;
  --sidebar-width: 300px;
  --topbar-height: 56px;
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
  --font-mono: "JetBrains Mono", "Fira Code", "Cascadia Code", "SF Mono", Consolas, monospace;
  --transition: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

[data-theme="dark"] {
  --bg: #0f1119;
  --bg-card: #1a1d28;
  --bg-sidebar: #11131c;
  --text: #e2e8f0;
  --text-secondary: #94a3b8;
  --text-sidebar: #94a3b8;
  --accent: #3b82f6;
  --accent-light: #1e3a5f;
  --border: #2d3148;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
  --shadow: 0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3);
  --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.4);
  --code-bg: #0d1117;
  --code-text: #c9d1d9;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: var(--font);
  background: var(--bg);
  color: var(--text);
  line-height: 1.75;
  -webkit-font-smoothing: antialiased;
}

#app { display: flex; min-height: 100vh; }

/* ── Sidebar ──────────────────────────────────────────────────────────── */

.sidebar {
  position: fixed; top: 0; left: 0; bottom: 0;
  width: var(--sidebar-width);
  background: var(--bg-sidebar);
  color: var(--text-sidebar);
  display: flex; flex-direction: column;
  z-index: 100;
  overflow-y: auto;
  border-right: 1px solid rgba(255,255,255,0.05);
  transition: transform var(--transition);
}

.sidebar-header { padding: 28px 24px 16px; border-bottom: 1px solid rgba(255,255,255,0.06); }
.sidebar-logo { text-decoration: none; display: flex; align-items: center; gap: 10px; }
.logo-icon { font-size: 28px; }
.logo-text { font-size: 22px; font-weight: 700; color: #fff; letter-spacing: -0.5px; }
.sidebar-subtitle { font-size: 12px; color: #6b7280; margin-top: 4px; }

.sidebar-search { padding: 16px 20px; position: relative; }
.sidebar-search input {
  width: 100%; padding: 10px 14px;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: var(--radius-sm);
  background: rgba(255,255,255,0.04);
  color: #fff; font-size: 13px; outline: none;
  transition: all var(--transition);
}
.sidebar-search input:focus { border-color: var(--accent); background: rgba(255,255,255,0.08); }
.sidebar-search input::placeholder { color: #6b7280; }

.search-results {
  display: none; position: absolute; top: 100%; left: 20px; right: 20px;
  background: #252836; border: 1px solid rgba(255,255,255,0.08);
  border-radius: var(--radius-sm); max-height: 300px; overflow-y: auto; z-index: 200;
  box-shadow: var(--shadow-lg);
}
.search-results.show { display: block; }
.search-results a { display: block; padding: 8px 14px; color: #c8cdd8; text-decoration: none; font-size: 13px; border-bottom: 1px solid rgba(255,255,255,0.04); }
.search-results a:hover, .search-results a.active { background: var(--accent); color: #fff; }
.search-results .sr-section { font-size: 10px; color: #6b7280; padding: 6px 14px 2px; text-transform: uppercase; letter-spacing: 1px; }

.sidebar-nav { flex: 1; overflow-y: auto; padding: 12px 0; }

.sidenav { list-style: none; }
.sidenav li { margin: 0; }
.sidenav-home {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 24px; color: #c8cdd8; text-decoration: none;
  font-size: 14px; font-weight: 500; transition: all var(--transition);
}
.sidenav-home:hover { color: #fff; background: rgba(255,255,255,0.04); }

.sidenav-section { margin: 4px 0; }
.sidenav-section-btn {
  display: flex; align-items: center; justify-content: space-between;
  width: 100%; padding: 10px 24px;
  background: none; border: none;
  color: #9ca3af; font-size: 13px; font-weight: 600;
  cursor: pointer; transition: all var(--transition);
  font-family: var(--font); text-align: left;
}
.sidenav-section-btn:hover { color: #fff; }
.sidenav-section.active .sidenav-section-btn { color: #fff; }
.sidenav-count { font-size: 11px; color: #6b7280; background: rgba(255,255,255,0.06); padding: 2px 8px; border-radius: 10px; }

.sidenav-sub { list-style: none; display: none; padding: 0 0 4px; }
.sidenav-section.active .sidenav-sub { display: block; }

.sidenav-sub a {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 24px 8px 40px;
  color: #9ca3af; text-decoration: none; font-size: 13px;
  transition: all var(--transition); border-left: 2px solid transparent;
}
.sidenav-sub a:hover { color: #e2e8f0; background: rgba(255,255,255,0.03); }
.sidenav-sub a.active { color: #fff; background: rgba(37,99,235,0.15); border-left-color: var(--accent); }
.sidenav-num { font-size: 11px; color: #6b7280; min-width: 22px; }
.sidenav-check { margin-left: auto; font-size: 11px; opacity: 0; transition: opacity var(--transition); }
.sidenav-check.done { opacity: 1; }
.sidenav-check::before { content: "✅"; }

.sidebar-footer {
  padding: 16px 24px; border-top: 1px solid rgba(255,255,255,0.06);
  display: flex; gap: 12px;
}
.theme-toggle, .gh-link {
  padding: 8px 14px; border: 1px solid rgba(255,255,255,0.08);
  border-radius: var(--radius-sm); background: none;
  color: #9ca3af; font-size: 12px; cursor: pointer; text-decoration: none;
  transition: all var(--transition); font-family: var(--font);
}
.theme-toggle:hover, .gh-link:hover { color: #fff; border-color: rgba(255,255,255,0.2); }

.sidebar-toggle { display: none; position: fixed; top: 12px; left: 12px; z-index: 200; width: 40px; height: 40px; border: none; background: var(--bg-card); border-radius: var(--radius-sm); font-size: 20px; cursor: pointer; box-shadow: var(--shadow); color: var(--text); }
.sidebar-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 99; }

/* ── Main ─────────────────────────────────────────────────────────────── */

.main { margin-left: var(--sidebar-width); flex: 1; min-width: 0; }

.topbar {
  position: sticky; top: 0; z-index: 50;
  height: var(--topbar-height);
  background: var(--bg); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 32px; backdrop-filter: blur(12px);
}
.breadcrumbs { font-size: 13px; color: var(--text-secondary); }
.breadcrumbs a { color: var(--text-secondary); text-decoration: none; }
.breadcrumbs a:hover { color: var(--accent); }
.bc-sep { margin: 0 8px; color: var(--border); }
.topbar-actions { display: flex; gap: 8px; }
.btn-progress { padding: 6px 12px; border: 1px solid var(--border); border-radius: var(--radius-sm); background: none; cursor: pointer; font-size: 16px; }

.content { max-width: 860px; margin: 0 auto; padding: 40px 32px 32px; }

/* ── Progress bar ─────────────────────────────────────────────────────── */

.progress-bar-wrap {
  margin-bottom: 32px; padding: 12px 16px;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-sm); display: flex; align-items: center; gap: 12px;
}
.progress-bar-fill {
  height: 6px; background: var(--accent); border-radius: 3px;
  transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  min-width: 0;
}
.progress-bar-text { font-size: 12px; color: var(--text-secondary); white-space: nowrap; }

.progress-toast {
  position: fixed; bottom: 24px; right: 24px; z-index: 999;
  padding: 12px 20px; border-radius: var(--radius-sm);
  background: var(--accent); color: #fff; font-size: 14px;
  box-shadow: var(--shadow-lg); opacity: 0; transform: translateY(16px);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); pointer-events: none;
}
.progress-toast.show { opacity: 1; transform: translateY(0); }

/* ── Typography ───────────────────────────────────────────────────────── */

.content h1 { font-size: 2rem; font-weight: 800; margin: 0 0 8px; letter-spacing: -0.5px; color: var(--text); }
.content h2 { font-size: 1.35rem; font-weight: 700; margin: 48px 0 16px; padding-bottom: 8px; border-bottom: 2px solid var(--accent-light); color: var(--text); }
.content h3 { font-size: 1.1rem; font-weight: 600; margin: 32px 0 12px; color: var(--text); }
.content h4 { font-size: 1rem; font-weight: 600; margin: 24px 0 8px; }
.content p { margin: 12px 0; }
.content a { color: var(--accent); text-decoration: none; font-weight: 500; }
.content a:hover { text-decoration: underline; }
.content strong { font-weight: 600; color: var(--text); }
.content blockquote {
  margin: 16px 0; padding: 12px 20px;
  border-left: 4px solid var(--accent);
  background: var(--accent-light);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  color: var(--text);
}
.content blockquote p { margin: 4px 0; }
.content hr { margin: 32px 0; border: none; border-top: 1px solid var(--border); }

.content code {
  font-family: var(--font-mono); font-size: 0.88em;
  padding: 2px 6px; border-radius: 4px;
  background: rgba(0,0,0,0.06); color: #d63384;
}
[data-theme="dark"] .content code { background: rgba(255,255,255,0.08); color: #f0a0c0; }

.content pre {
  margin: 16px 0; padding: 20px 24px;
  background: var(--code-bg); color: var(--code-text);
  border-radius: var(--radius-sm);
  overflow-x: auto; font-size: 13.5px; line-height: 1.65;
  font-family: var(--font-mono);
  border: 1px solid var(--border);
}
.content pre code { background: none; padding: 0; color: inherit; font-size: inherit; }

.content table {
  width: 100%; border-collapse: collapse; margin: 16px 0;
  font-size: 14px; border-radius: var(--radius-sm); overflow: hidden;
  border: 1px solid var(--border);
}
.content th {
  background: var(--accent-light); font-weight: 600; text-align: left;
  padding: 12px 16px; border-bottom: 2px solid var(--border);
}
.content td { padding: 10px 16px; border-bottom: 1px solid var(--border); }
.content tr:last-child td { border-bottom: none; }
.content tr:hover td { background: rgba(0,0,0,0.02); }
[data-theme="dark"] .content tr:hover td { background: rgba(255,255,255,0.02); }

.content ul, .content ol { margin: 12px 0; padding-left: 24px; }
.content li { margin: 6px 0; }

/* ── Cards (index pages) ──────────────────────────────────────────────── */

.course-hero {
  text-align: center; padding: 48px 24px 40px;
  background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 50%, #7c3aed 100%);
  color: #fff; border-radius: var(--radius); margin-bottom: 40px;
}
.course-hero h1 { font-size: 2.5rem; color: #fff; margin-bottom: 8px; }
.course-hero p { font-size: 1.1rem; opacity: 0.9; max-width: 600px; margin: 0 auto; }
.course-hero .hero-stats { display: flex; justify-content: center; gap: 32px; margin-top: 24px; }
.hero-stat { text-align: center; }
.hero-stat-val { font-size: 2rem; font-weight: 800; }
.hero-stat-label { font-size: 12px; opacity: 0.7; text-transform: uppercase; letter-spacing: 1px; }

.section-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 20px; margin: 24px 0; }

.section-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 28px; transition: all var(--transition);
  text-decoration: none; color: var(--text); display: block;
  box-shadow: var(--shadow-sm);
}
.section-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-lg); border-color: var(--accent); }
.section-card-icon { font-size: 2rem; margin-bottom: 12px; }
.section-card h3 { font-size: 1.15rem; font-weight: 700; margin: 0 0 8px; }
.section-card p { font-size: 13px; color: var(--text-secondary); margin: 0 0 16px; line-height: 1.5; }
.section-card-meta { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text-secondary); }
.section-card-meta .count { background: var(--accent-light); color: var(--accent); padding: 3px 10px; border-radius: 12px; font-weight: 600; }

.topic-list { margin: 24px 0; }
.topic-item {
  display: flex; align-items: center; gap: 16px;
  padding: 16px 20px; margin: 4px 0;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-sm); text-decoration: none; color: var(--text);
  transition: all var(--transition);
}
.topic-item:hover { border-color: var(--accent); box-shadow: var(--shadow-sm); transform: translateX(4px); }
.topic-num { font-size: 12px; font-weight: 700; color: var(--accent); min-width: 28px; }
.topic-item-content { flex: 1; }
.topic-item-title { font-weight: 600; font-size: 14px; }
.topic-item-desc { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }
.topic-badge {
  font-size: 11px; padding: 4px 10px; border-radius: 12px; font-weight: 600;
}
.topic-badge.beginner { background: var(--badge-green-bg); color: var(--badge-green); }
.topic-badge.intermediate { background: var(--badge-yellow-bg); color: var(--badge-yellow); }
.topic-badge.advanced { background: var(--badge-red-bg); color: var(--badge-red); }

/* ── Page nav ─────────────────────────────────────────────────────────── */

.content-footer { max-width: 860px; margin: 0 auto; padding: 0 32px 48px; }
.page-nav-row { display: flex; justify-content: space-between; gap: 16px; margin-bottom: 32px; }
.page-nav {
  flex: 1; padding: 16px 20px; border: 1px solid var(--border);
  border-radius: var(--radius-sm); text-decoration: none; color: var(--text);
  transition: all var(--transition); background: var(--bg-card);
}
.page-nav:hover { border-color: var(--accent); box-shadow: var(--shadow-sm); }
.page-nav.prev { text-align: left; }
.page-nav.next { text-align: right; margin-left: auto; }
.pn-label { font-size: 11px; color: var(--text-secondary); display: block; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px; }
.pn-title { font-weight: 600; font-size: 14px; }
.footer-info { text-align: center; font-size: 12px; color: var(--text-secondary); padding-top: 24px; border-top: 1px solid var(--border); }
.footer-info a { color: var(--accent); }

/* ── Responsive ───────────────────────────────────────────────────────── */

@media (max-width: 900px) {
  .sidebar { transform: translateX(-100%); }
  .sidebar.open { transform: translateX(0); }
  .sidebar-overlay.show { display: block; }
  .sidebar-toggle { display: flex; align-items: center; justify-content: center; }
  .main { margin-left: 0; }
  .content { padding: 24px 20px; }
  .content-footer { padding: 0 20px 40px; }
  .topbar { padding: 0 20px; }
  .section-cards { grid-template-columns: 1fr; }
  .course-hero h1 { font-size: 1.8rem; }
  .hero-stats { gap: 16px; }
}
@media (max-width: 600px) {
  .content h1 { font-size: 1.5rem; }
  .page-nav-row { flex-direction: column; }
  .topic-item { flex-wrap: wrap; }
}
</style>"""


def _js() -> str:
    return """<script>
(function() {
  // ── Theme ──────────────────────────────────────────────────────────
  const themeToggle = document.getElementById('themeToggle');
  const html = document.documentElement;
  const saved = localStorage.getItem('scratchv-theme');
  if (saved) html.setAttribute('data-theme', saved);
  else if (window.matchMedia('(prefers-color-scheme: dark)').matches) html.setAttribute('data-theme', 'dark');
  themeToggle.addEventListener('click', () => {
    const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('scratchv-theme', next);
  });

  // ── Sidebar ────────────────────────────────────────────────────────
  const sidebar = document.getElementById('sidebar');
  const toggle = document.getElementById('sidebarToggle');
  const overlay = document.getElementById('sidebarOverlay');
  toggle.addEventListener('click', () => { sidebar.classList.toggle('open'); overlay.classList.toggle('show'); });
  overlay.addEventListener('click', () => { sidebar.classList.remove('open'); overlay.classList.remove('show'); });

  // Expand active section
  document.querySelectorAll('.sidenav-section-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const section = btn.parentElement;
      const wasActive = section.classList.contains('active');
      // Collapse all
      document.querySelectorAll('.sidenav-section').forEach(s => s.classList.remove('active'));
      if (!wasActive) section.classList.add('active');
    });
  });

  // ── Search ─────────────────────────────────────────────────────────
  const searchInput = document.getElementById('searchInput');
  const searchResults = document.getElementById('searchResults');
  const allLinks = [];
  document.querySelectorAll('.sidenav-sub a, .sidenav-home').forEach(a => {
    allLinks.push({ title: a.textContent.trim().replace(/^\\d+\\s*/, ''), href: a.getAttribute('href'), el: a });
  });

  searchInput.addEventListener('input', () => {
    const q = searchInput.value.trim().toLowerCase();
    if (!q) { searchResults.classList.remove('show'); return; }
    const matches = allLinks.filter(l => l.title.toLowerCase().includes(q));
    if (!matches.length) { searchResults.classList.remove('show'); return; }
    searchResults.innerHTML = matches.map((m, i) =>
      `<a href="${m.href}" class="${i===0?'active':''}">${m.title}</a>`
    ).join('');
    searchResults.classList.add('show');
  });
  document.addEventListener('click', e => { if (!e.target.closest('.sidebar-search')) searchResults.classList.remove('show'); });

  // ── Progress tracking ──────────────────────────────────────────────
  const docTitle = document.querySelector('.content h1')?.textContent?.trim() || document.title;
  const progressKey = 'scratchv-progress';
  let progress = JSON.parse(localStorage.getItem(progressKey) || '{}');

  // Mark current page as in-progress
  if (docTitle && !progress[docTitle]) {
    progress[docTitle] = 'in_progress';
    localStorage.setItem(progressKey, JSON.stringify(progress));
  }

  // Mark as done after 30s on page
  setTimeout(() => {
    if (docTitle) { progress[docTitle] = 'done'; localStorage.setItem(progressKey, JSON.stringify(progress)); updateChecks(); updateProgressBar(); }
  }, 30000);

  function updateChecks() {
    const data = JSON.parse(localStorage.getItem(progressKey) || '{}');
    document.querySelectorAll('.sidenav-check').forEach(el => {
      const key = el.getAttribute('data-check');
      if (data[key] === 'done') el.classList.add('done');
    });
  }

  function updateProgressBar() {
    const bar = document.getElementById('progressBarFill');
    const text = document.getElementById('progressBarText');
    if (!bar || !text) return;
    const data = JSON.parse(localStorage.getItem(progressKey) || '{}');
    const total = 32, done = Object.values(data).filter(v => v === 'done').length;
    const pct = Math.round(done / total * 100);
    bar.style.width = pct + '%';
    text.textContent = done + '/' + total + ' 已完成 · ' + pct + '%';
  }

  updateChecks();
  updateProgressBar();

  // ── Mark as done button ────────────────────────────────────────────
  const btn = document.getElementById('btnProgress');
  const toast = document.getElementById('progressToast');
  btn?.addEventListener('click', () => {
    if (!docTitle) return;
    const data = JSON.parse(localStorage.getItem(progressKey) || '{}');
    data[docTitle] = 'done';
    localStorage.setItem(progressKey, JSON.stringify(data));
    updateChecks(); updateProgressBar();
    toast.textContent = '✅ 已标记完成: ' + docTitle;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2000);
  });
})();
</script>"""


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown → HTML conversion
# ═══════════════════════════════════════════════════════════════════════════════

def md_to_html(md_text: str) -> str:
    """Convert markdown text to HTML with extensions for tables, code, etc."""
    # Preprocess: fix Chinese headers (add space after # if missing)
    md_text = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', md_text, flags=re.MULTILINE)

    html = md_lib.markdown(
        md_text,
        extensions=[
            "fenced_code",
            "tables",
            "codehilite",
        ],
    )

    # Post-process: add target="_blank" to external links
    html = re.sub(r'<a href="(https?://[^"]+)"', r'<a href="\1" target="_blank" rel="noopener"', html)

    # Post-process: add stable heading ids matching the hand-written TOC format
    # TOC links look like: [1. 你需要什么](#1-你需要什么)
    # So we generate:     <h2 id="1-你需要什么">1. 你需要什么</h2>
    def _add_heading_id(m: re.Match) -> str:
        tag = m.group(1)
        text = m.group(2)
        stripped = text.strip()
        # Check if heading has a numeric prefix like "2. "
        num_match = re.match(r'^(\d+)\.\s+(.+)', stripped)
        if num_match:
            num = num_match.group(1)
            title_part = num_match.group(2)
            # Slug: keep Chinese chars, ASCII, hyphens; lowercase English; remove punctuation
            slug = re.sub(r'[（）()""《》：【】、。，？！\':：]', '', title_part)
            slug = re.sub(r'\s+', '-', slug.strip())
            slug = slug.lower()  # case-insensitive matching with TOC links
            return f'<{tag} id="{num}-{slug}">{text}'
        else:
            # No number — use position counter
            return f'<{tag}>{text}'

    html = re.sub(r'<(h[234])>(.*?)</\1>', _add_heading_id, html)

    # Fix internal .md links → correct .html using the COURSE mapping
    # Also clean up the link text (remove .md extension from display text)
    def _replace_md_link(m: re.Match) -> str:
        md_path = m.group(1)
        target = _md_link_to_html(md_path)
        return f'href="{target}"'

    html = re.sub(r'href="([^"]+\.md)"', _replace_md_link, html)
    html = re.sub(r'href="([^"]+\.md#[^"]+)"', lambda m: f'href="{_md_link_to_html(m.group(1))}"', html)

    # Clean up link text: remove trailing .md from display text
    html = re.sub(r'>([^<]+)\.md</a>', r'>\1</a>', html)

    # Fix relative links from topics/ to docs/
    html = html.replace('href="../index.html"', 'href="index.html"')

    return html


# ═══════════════════════════════════════════════════════════════════════════════
# Build functions
# ═══════════════════════════════════════════════════════════════════════════════

def build_index():
    """Build course home page."""
    total = sum(len(s["docs"]) for s in COURSE["sections"])
    cards = []
    for sec in COURSE["sections"]:
        done = 0
        cards.append(f"""
        <a class="section-card" href="{sec['id']}.html">
          <div class="section-card-icon">{sec['icon']}</div>
          <h3>{_escape(sec['title'])}</h3>
          <p>{_escape(sec['desc'])}</p>
          <div class="section-card-meta">
            <span class="count">{len(sec['docs'])} 个模块</span>
          </div>
        </a>""")

    content = f"""
    <div class="course-hero">
      <h1>{COURSE['title']}</h1>
      <p>{COURSE['subtitle']}</p>
      <div class="hero-stats">
        <div class="hero-stat"><div class="hero-stat-val">35</div><div class="hero-stat-label">课程文档</div></div>
        <div class="hero-stat"><div class="hero-stat-val">5</div><div class="hero-stat-label">难度层级</div></div>
        <div class="hero-stat"><div class="hero-stat-val">30</div><div class="hero-stat-label">专题模块</div></div>
      </div>
    </div>
    <h2>📚 课程目录</h2>
    <div class="section-cards">{''.join(cards)}</div>
    """

    return render_page(
        title="课程首页",
        body_class="page-index",
        content=content,
        breadcrumbs=[("首页", "index.html")],
    )


def build_section_page(sec: dict):
    """Build a section overview page (e.g., foundation.html)."""
    items = []
    for d in sec["docs"]:
        href = _prev_next_href(d)
        badge_class = sec["id"]
        if badge_class == "foundation":
            badge_class = "beginner"
        elif badge_class in ("beginner-build", "beginner-observe"):
            badge_class = "beginner"
        elif badge_class == "planned":
            badge_class = "intermediate"  # use existing style for planned
        time_str = ""
        if d.get("time"):
            time_str = f'<span style="font-size:11px;color:var(--text-secondary)">⏱ {d["time"]}</span>'
        items.append(f"""
        <a class="topic-item" href="{href}">
          <span class="topic-num">{d['num']}</span>
          <div class="topic-item-content">
            <div class="topic-item-title">{_escape(d['title'])}</div>
          </div>
          {time_str}
          <span class="topic-badge {badge_class}">{sec['title']}</span>
        </a>""")

    content = f"""
    <h1>{sec['icon']} {_escape(sec['title'])}</h1>
    <p style="color:var(--text-secondary);font-size:15px;margin-bottom:24px">{_escape(sec['desc'])}</p>
    <div class="topic-list">{''.join(items)}</div>
    """

    return render_page(
        title=f"{sec['title']} — ScratchV 编译器课程",
        body_class=f"page-section page-{sec['id']}",
        content=content,
        breadcrumbs=[("首页", "index.html"), (sec["title"], f"{sec['id']}.html")],
        current_section=sec["id"],
    )


# Docs listed in COURSE whose markdown is missing on disk.  A miss produces a
# sidebar entry pointing at an HTML file that was never written, so main()
# turns a non-empty list into a non-zero exit code.
MISSES: list[str] = []


def build_doc_page(doc: dict):
    """Build a single topic page."""
    md_path = PROJ / doc["file"]
    if not md_path.exists():
        print(f"  WARNING: {doc['file']} not found, skipping")
        MISSES.append(doc["file"])
        return None

    md_text = md_path.read_text(encoding="utf-8")
    body_html = md_to_html(md_text)

    # Find section
    sec = next(s for s in COURSE["sections"] if s["id"] == doc["section_id"])

    # Find prev/next
    idx = ALL_DOCS.index(doc)
    prev_doc = ALL_DOCS[idx - 1] if idx > 0 else None
    next_doc = ALL_DOCS[idx + 1] if idx < len(ALL_DOCS) - 1 else None

    return render_page(
        title=f"{doc['title']} — ScratchV 编译器课程",
        body_class=f"page-doc page-{sec['id']}",
        content=body_html,
        breadcrumbs=[
            ("首页", "index.html"),
            (sec["title"], f"{sec['id']}.html"),
            (doc["title"], _prev_next_href(doc)),
        ],
        current_section=sec["id"],
        current_doc=doc["title"],
        prev_doc=prev_doc,
        next_doc=next_doc,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(output_dir: str = "docs/topics/html"):
    out = PROJ / output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Wipe stale pages: the build only overwrites files it regenerates, so
    # anything renamed or dropped from COURSE would otherwise linger and be
    # mistaken for current output.
    for stale in out.glob("*.html"):
        stale.unlink()

    built = 0

    # Course home
    print("Building course home...")
    (out / "index.html").write_text(build_index(), encoding="utf-8")
    built += 1

    # Section overview pages
    for sec in COURSE["sections"]:
        print(f"Building section: {sec['title']}...")
        (out / f"{sec['id']}.html").write_text(build_section_page(sec), encoding="utf-8")
        built += 1

    # Individual doc pages
    for doc in ALL_DOCS:
        print(f"Building doc: {doc['title']}...")
        page = build_doc_page(doc)
        if page:
            href = _prev_next_href(doc)
            (out / href).write_text(page, encoding="utf-8")
            built += 1

    # Copy CSS/JS if needed (currently inline, nothing to copy)

    expected = 1 + len(COURSE["sections"]) + len(ALL_DOCS)
    print(f"\n✅ Built {built} HTML pages → {out}")
    print(f"   Open: {out}/index.html")

    if MISSES:
        print(f"\n❌ {len(MISSES)} COURSE doc(s) missing on disk:", file=sys.stderr)
        for m in MISSES:
            print(f"     {m}", file=sys.stderr)
        sys.exit(1)
    if built != expected:
        print(f"\n❌ built {built} pages, expected {expected}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build ScratchV course HTML site")
    p.add_argument("--output-dir", default="docs/topics/html", help="Output directory")
    args = p.parse_args()
    main(args.output_dir)
