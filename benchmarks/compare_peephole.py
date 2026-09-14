# flake8: noqa
"""Compare assembly size and optimizer activity with peephole disabled/enabled.

The comparison deliberately keeps the input assembly identical in both modes.
It reports effective instruction counts, static savings, per-rule matches and
reference optimizer timings in a machine-readable JSON document, then renders
a self-contained HTML report styled like the existing ScratchV benchmark page.

Usage:
    python benchmarks/compare_peephole.py
    python benchmarks/compare_peephole.py --repeats 20 --output-dir benchmark_reports
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

# Direct script execution omits the repository root from sys.path.
# Put this worktree first so the benchmark measures the checked-out code.
if __package__ is None:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from scratchv.backend.asm_peephole import AsmPeepholeOptimizer
from scratchv.standalone.bench_report import HTML_CSS

from benchmarks.bench_asm_peephole import (
    PR39_RULES,
    BenchmarkCase,
    count_instructions,
    default_cases,
)


def _validate_repeats(repeats: int) -> int:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    return repeats


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _optimize_with_timing(
    assembly: str,
    repeats: int,
) -> tuple[str, int, dict[str, int], list[float]]:
    timings: list[float] = []
    output = assembly
    changes = 0
    rule_matches: dict[str, int] = {}

    for _ in range(_validate_repeats(repeats)):
        optimizer = AsmPeepholeOptimizer()
        started = time.perf_counter()
        output, changes = optimizer.optimize(assembly)
        timings.append((time.perf_counter() - started) * 1000.0)
        rule_matches = optimizer.total_matches

    return output, changes, rule_matches, timings


def compare_cases(
    cases: Optional[Sequence[BenchmarkCase]] = None,
    repeats: int = 5,
) -> dict:
    """Compare the same assembly cases with peephole off and on."""

    repeats = _validate_repeats(repeats)
    selected = list(cases if cases is not None else default_cases())
    results: list[dict] = []

    for case in selected:
        before = count_instructions(case.assembly)
        output, changes, rule_matches, timings = _optimize_with_timing(
            case.assembly,
            repeats,
        )
        after = count_instructions(output)
        reduced = before - after
        reduction_percent = 100.0 * reduced / before if before else 0.0
        all_rule_matches = {name: rule_matches.get(name, 0) for name in PR39_RULES}
        expected_hit = (
            case.expected_rule is not None
            and all_rule_matches.get(case.expected_rule, 0) > 0
        )

        results.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "description": case.description,
                "expected_rule": case.expected_rule,
                "expected_rule_hit": expected_hit,
                "input_sha256": _sha256(case.assembly),
                "peephole_off": {
                    "enabled": False,
                    "instructions": before,
                    "changes": 0,
                    "elapsed_ms_median": 0.0,
                },
                "peephole_on": {
                    "enabled": True,
                    "instructions": after,
                    "changes": changes,
                    "rule_matches": all_rule_matches,
                    "elapsed_ms_median": round(statistics.median(timings), 6),
                    "elapsed_ms_min": round(min(timings), 6),
                    "elapsed_ms_max": round(max(timings), 6),
                },
                "before_instructions": before,
                "after_instructions": after,
                "reduced_instructions": reduced,
                "reduction_percent": round(reduction_percent, 3),
                "changes": changes,
                "rule_matches": all_rule_matches,
                "elapsed_ms_median": round(statistics.median(timings), 6),
                "repeats": repeats,
            }
        )

    before_total = sum(item["before_instructions"] for item in results)
    after_total = sum(item["after_instructions"] for item in results)
    reduced_total = before_total - after_total
    rule_matches = {name: 0 for name in PR39_RULES}
    for item in results:
        for name, count in item["rule_matches"].items():
            rule_matches[name] += count

    positive_count = sum(item["category"] != "negative" for item in results)
    negative_count = sum(item["category"] == "negative" for item in results)
    unchanged_count = sum(item["reduced_instructions"] == 0 for item in results)

    return {
        "schema_version": 1,
        "benchmark": "ScratchV assembly peephole on/off comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "repeats": repeats,
            "comparison": "same input assembly, optimizer disabled vs enabled",
        },
        "summary": {
            "case_count": len(results),
            "positive_cases": positive_count,
            "negative_cases": negative_count,
            "unchanged_cases": unchanged_count,
            "before_instructions": before_total,
            "after_instructions": after_total,
            "reduced_instructions": reduced_total,
            "reduction_percent": round(
                100.0 * reduced_total / before_total if before_total else 0.0,
                3,
            ),
            "changes": sum(item["changes"] for item in results),
            "rule_matches": rule_matches,
            "optimizer_elapsed_ms_median_sum": round(
                sum(item["elapsed_ms_median"] for item in results),
                6,
            ),
        },
        "cases": results,
    }


def save_comparison(
    report: dict,
    json_path: str | Path,
    html_path: str | Path,
) -> None:
    """Write JSON data and a self-contained HTML report."""

    json_output = Path(json_path)
    html_output = Path(html_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    html_output.write_text(generate_html_report(report), encoding="utf-8")


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _bar_row(
    label: str,
    value: float,
    maximum: float,
    color: str = "compute",
    suffix: str = "",
) -> str:
    if maximum <= 0:
        width = 0.5 if value > 0 else 0.0
    else:
        width = max(
            0.5 if value > 0 else 0.0,
            min(value / maximum * 100.0, 100.0),
        )
    return (
        "<tr>"
        f"<td>{_escape(label)}</td>"
        '<td class="bar-cell"><div class="bar-bg">'
        f'<div class="bar-fill {color}" style="width:{width:.1f}%"></div>'
        "</div></td>"
        f'<td class="bar-value">{value:,.3f}{_escape(suffix)}</td>'
        "</tr>"
    )


def _metric_card(label: str, value: str, color: str = "") -> str:
    color_class = f" {color}" if color else ""
    return (
        f'<div class="card{color_class}"><div class="label">{_escape(label)}</div>'
        f'<div class="value">{_escape(value)}</div></div>'
    )


def generate_html_report(report: dict) -> str:
    """Render a comparison report using the existing ScratchV card/bar style."""

    summary = report.get("summary", {})
    metadata = report.get("metadata", {})
    cases = report.get("cases", [])
    before = int(summary.get("before_instructions", 0))
    after = int(summary.get("after_instructions", 0))
    saved = int(summary.get("reduced_instructions", 0))
    reduction = float(summary.get("reduction_percent", 0.0))
    generated_at = _escape(report.get("generated_at", ""))
    commit = _escape(metadata.get("git_commit") or "工作树")
    repeats = _escape(metadata.get("repeats", ""))

    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN"><head><meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>ScratchV 窥孔优化器 Benchmark</title>",
        HTML_CSS,
        """<style>
          .comparison-note { color:#718096; margin:0 0 16px; line-height:1.6; }
          .bar-value { width:120px; text-align:right; font-weight:600; white-space:nowrap; }
          .delta { color:#2f855a; font-weight:700; }
          .muted { color:#718096; }
          .case-table td, .case-table th { vertical-align:top; }
          .case-table td:nth-child(n+4) { text-align:right; white-space:nowrap; }
          code { color:#2b6cb0; }
        </style>""",
        "</head><body>",
        "<h1>ScratchV 窥孔优化器 Benchmark</h1>",
        (
            f'<div class="subtitle">Commit: <code>{commit}</code> | '
            f"样例: {len(cases)} | 重复次数: {repeats} | 生成时间: {generated_at}</div>"
        ),
        '<div class="cards">',
        _metric_card("优化前指令", f"{before:,}", "blue"),
        _metric_card("优化后指令", f"{after:,}", "green"),
        _metric_card("静态节省", f"{saved:,} ({reduction:.1f}%)", "orange"),
        _metric_card("规则命中", f"{int(summary.get('changes', 0)):,}", "purple"),
        _metric_card("未变化样例", f"{int(summary.get('unchanged_cases', 0)):,}", "red"),
        "</div>",
    ]

    parts.extend(
        [
            "<section><h2>peephole 开关对比</h2>",
            (
                '<p class="comparison-note">所有样例使用同一份输入汇编；'
                "关闭表示跳过汇编窥孔优化，开启表示执行 PR39 默认规则。"
                "</p>"
            ),
            "<table><tr><th>指标</th><th>对比</th><th>结果</th></tr>",
            _bar_row("优化前指令", before, max(before, after, 1), "compute", " 条"),
            _bar_row("优化后指令", after, max(before, after, 1), "branch", " 条"),
            _bar_row("静态节省", saved, max(before, 1), "memory", " 条"),
            "</table></section>",
            "<section><h2>规则命中与节省</h2>",
            "<table><tr><th>规则</th><th>命中次数</th><th>结果</th></tr>",
        ]
    )
    rule_matches = summary.get("rule_matches", {})
    maximum_matches = max(
        [int(rule_matches.get(name, 0)) for name in PR39_RULES] or [1]
    )
    for index, name in enumerate(PR39_RULES):
        color = ("compute", "memory", "branch", "upper", "shift", "neutral")[
            index % 6
        ]
        parts.append(
            _bar_row(
                name,
                int(rule_matches.get(name, 0)),
                maximum_matches,
                color,
                " 次",
            )
        )
    parts.append("</table></section>")

    parts.extend(
        [
            "<section><h2>样例明细</h2>",
            '<table class="case-table"><tr><th>样例</th><th>类别</th>'
            "<th>关闭</th><th>开启</th><th>节省</th><th>命中</th><th>输入摘要</th></tr>",
        ]
    )
    for item in cases:
        expected = item.get("expected_rule")
        hit = "是" if item.get("expected_rule_hit") else "否"
        hit_class = "ok" if item.get("expected_rule_hit") else "muted"
        category = item.get("category", "")
        digest = str(item.get("input_sha256", ""))
        digest_short = digest[:12] if digest else "-"
        parts.append(
            "<tr>"
            f"<td>{_escape(item.get('case_id', ''))}<br>"
            f'<span class="muted">{_escape(item.get("description", ""))}</span></td>'
            f"<td>{_escape(category)}</td>"
            f"<td>{int(item.get('peephole_off', {}).get('instructions', 0)):,}</td>"
            f"<td>{int(item.get('peephole_on', {}).get('instructions', 0)):,}</td>"
            f'<td class="delta">{int(item.get("reduced_instructions", 0)):,} '
            f'({float(item.get("reduction_percent", 0.0)):.1f}%)</td>'
            f'<td class="{hit_class}">{_escape(hit)}'
            f'<br><span class="muted">{_escape(expected or "-")}</span></td>'
            f"<td><code>{_escape(digest_short)}</code></td>"
            "</tr>"
        )
    parts.append("</table></section>")

    elapsed = float(summary.get("optimizer_elapsed_ms_median_sum", 0.0))
    parts.extend(
        [
            "<section><h2>环境与结论</h2>",
            "<table>",
            f"<tr><td>Python</td><td>{_escape(metadata.get('python', ''))}</td></tr>",
            f"<tr><td>优化器参考耗时（样例中位数之和）</td><td>{elapsed:.3f} ms</td></tr>",
            f"<tr><td>正向样例 / 负向样例</td><td>{int(summary.get('positive_cases', 0))} / {int(summary.get('negative_cases', 0))}</td></tr>",
            f"<tr><td>结论</td><td>{'观察到静态指令减少' if saved > 0 else '未观察到静态指令减少'}</td></tr>",
            "</table></section>",
            f'<div class="footer">ScratchV 窥孔优化器报告 · {generated_at}</div>',
            "</body></html>",
        ]
    )
    return "\n".join(parts)


def _print_summary(report: dict, json_path: Path, html_path: Path) -> None:
    summary = report["summary"]
    print("=" * 88)
    print("ScratchV Peephole On/Off Comparison")
    print("=" * 88)
    print(
        f"Cases: {summary['case_count']} | "
        f"Instructions: {summary['before_instructions']} -> "
        f"{summary['after_instructions']} | "
        f"Saved: {summary['reduced_instructions']} "
        f"({summary['reduction_percent']:.1f}%)"
    )
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare ScratchV peephole optimizer disabled/enabled",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of optimizer timing repetitions per case (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_reports"),
        help="Directory for JSON and HTML reports (default: benchmark_reports)",
    )
    args = parser.parse_args(argv)

    report = compare_cases(repeats=args.repeats)
    json_path = args.output_dir / "peephole_compare.json"
    html_path = args.output_dir / "peephole_compare.html"
    save_comparison(report, json_path, html_path)
    _print_summary(report, json_path, html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
