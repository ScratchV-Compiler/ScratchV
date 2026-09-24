"""Compare assembly size and optimizer activity with peephole disabled/enabled.

The comparison deliberately keeps the input assembly identical in both modes.
It reports effective instruction counts, static savings, per-rule matches and
reference optimizer timings in a machine-readable JSON document, then renders
a self-contained HTML report styled like the existing ScratchV benchmark page.

Usage:
    python benchmarks/compare_peephole_html.py
    python benchmarks/compare_peephole_html.py --repeats 20 --output-dir benchmark_reports
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
from typing import Any, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Direct script execution omits the repository root from sys.path.
# Put this worktree first so the benchmark measures the checked-out code.
if __package__ is None:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from benchmarks import bench_asm_peephole as peephole_bench  # noqa: E402
from scratchv.backend.asm_peephole import AsmPeepholeOptimizer  # noqa: E402
from scratchv.standalone.bench_report import HTML_CSS  # noqa: E402


def _validate_repeats(repeats: int) -> int:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    return repeats


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        git_args = ["git", "-c", f"safe.directory={_REPO_ROOT}"]
        completed = subprocess.run(
            [*git_args, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_ROOT,
        )
        status = subprocess.run(
            [*git_args, "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    commit = completed.stdout.strip()
    return f"{commit}-dirty" if status.stdout.strip() else commit


def _optimize_with_timing(
    assembly: str,
    repeats: int,
) -> tuple[str, int, dict[str, int], list[float]]:
    timings: list[float] = []
    output = assembly
    changes = 0
    rule_matches: dict[str, int] = {}
    probe = AsmPeepholeOptimizer()
    peephole_bench.validate_default_rules(rule.name for rule in probe.rules)

    for _ in range(_validate_repeats(repeats)):
        optimizer = AsmPeepholeOptimizer()
        started = time.perf_counter()
        output, changes = optimizer.optimize(assembly)
        timings.append((time.perf_counter() - started) * 1000.0)
        rule_matches = optimizer.total_matches

    return output, changes, rule_matches, timings


def compare_cases(
    cases: Optional[Sequence[peephole_bench.BenchmarkCase]] = None,
    repeats: int = 5,
) -> dict:
    """Compare the same assembly cases with peephole off and on."""

    repeats = _validate_repeats(repeats)
    selected = list(cases if cases is not None else peephole_bench.default_cases())
    results: list[dict] = []

    for case in selected:
        before = peephole_bench.count_instructions(case.assembly)
        output, changes, rule_matches, timings = _optimize_with_timing(
            case.assembly,
            repeats,
        )
        after = peephole_bench.count_instructions(output)
        reduced = before - after
        reduction_percent = 100.0 * reduced / before if before else 0.0
        all_rule_matches = {
            name: rule_matches.get(name, 0) for name in peephole_bench.PR39_RULES
        }
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
                    "elapsed_ms_median": None,
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
    rule_matches = {name: 0 for name in peephole_bench.PR39_RULES}
    for item in results:
        for name, count in item["rule_matches"].items():
            rule_matches[name] += count

    positive_count = sum(item["category"] != "negative" for item in results)
    negative_count = sum(item["category"] == "negative" for item in results)
    unchanged_count = sum(item["changes"] == 0 for item in results)

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
        f'<td class="bar-value">{int(value):,}{_escape(suffix)}</td>'
        "</tr>"
    )


def _metric_card(label: str, value: str, color: str = "") -> str:
    color_class = f" {color}" if color else ""
    return (
        f'<div class="card{color_class}"><div class="label">{_escape(label)}</div>'
        f'<div class="value">{_escape(value)}</div></div>'
    )


def _dsl_report_payload(dsl_report: Any) -> dict:
    """Adapt ``run_dsl_suite()`` output to the shared HTML report schema."""
    rule_matches = {name: 0 for name in peephole_bench.PR39_RULES}
    cases = []
    for case in dsl_report.cases:
        case_rule_matches = {name: int(count) for name, count in case.rule_hits.items()}
        for name, count in case_rule_matches.items():
            rule_matches[name] = rule_matches.get(name, 0) + count
        cases.append(
            {
                "case_id": case.name,
                "description": "",
                "peephole_off": {"instructions": case.before_total},
                "peephole_on": {"instructions": case.after_total},
                "reduced_instructions": case.saved,
                "reduction_percent": case.saved_pct,
                "changes": case.peephole_changes,
                "rule_matches": case_rule_matches,
            }
        )

    before = int(dsl_report.total_before)
    saved = int(dsl_report.total_saved)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "python": platform.python_version(),
            "comparison": "DSL suite, optimizer disabled vs enabled",
        },
        "summary": {
            "case_count": len(cases),
            "unchanged_cases": sum(
                case.peephole_changes == 0 for case in dsl_report.cases
            ),
            "before_instructions": before,
            "after_instructions": int(dsl_report.total_after),
            "reduced_instructions": saved,
            "reduction_percent": round(100.0 * saved / before, 3) if before else 0.0,
            "changes": sum(case.peephole_changes for case in dsl_report.cases),
            "rule_matches": rule_matches,
        },
        "cases": cases,
    }


def generate_html_report(
    report: dict,
    *,
    title: str = "ScratchV 窥孔优化器 Benchmark",
    comparison_note: str = (
        "每个样例的关闭/开启结果均基于同一份输入汇编；"
        "关闭表示跳过汇编窥孔优化，开启表示执行窥孔优化规则。"
    ),
    show_case_polarity: bool = True,
) -> str:
    """Render a comparison report using the existing ScratchV card/bar style."""

    summary = report.get("summary", {})
    metadata = report.get("metadata", {})
    cases = report.get("cases", [])
    before = int(summary.get("before_instructions", 0))
    after = int(summary.get("after_instructions", 0))
    saved = int(summary.get("reduced_instructions", 0))
    reduction = float(summary.get("reduction_percent", 0.0))
    generated_at = _escape(report.get("generated_at", ""))
    subtitle = f"样例: {len(cases)}"
    if "repeats" in metadata:
        subtitle += f" | 重复次数: {_escape(metadata['repeats'])}"
    subtitle += f" | 生成时间: {generated_at}"

    parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN"><head><meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_escape(title)}</title>",
        HTML_CSS,
        """<style>
          .comparison-note { color:#718096; margin:0 0 16px; line-height:1.6; }
          .bar-value { width:120px; text-align:right; font-weight:600; white-space:nowrap; }
          .delta { color:#2f855a; font-weight:700; }
          .muted { color:#718096; }
          .case-table { table-layout:fixed; }
          .case-table td, .case-table th { vertical-align:top; }
          .case-table th:first-child, .case-table td:first-child {
            width:32%; text-align:left; white-space:normal; overflow-wrap:anywhere;
          }
          .case-table th:nth-child(2), .case-table td:nth-child(2),
          .case-table th:nth-child(3), .case-table td:nth-child(3),
          .case-table th:nth-child(4), .case-table td:nth-child(4) {
            width:12%; text-align:right; white-space:nowrap;
          }
          .case-table th:last-child, .case-table td:last-child {
            width:32%; text-align:left; white-space:normal; overflow-wrap:anywhere;
          }
          code { color:#2b6cb0; }
        </style>""",
        "</head><body>",
        f"<h1>{_escape(title)}</h1>",
        f'<div class="subtitle">{subtitle}</div>',
        '<div class="cards">',
        _metric_card("优化前指令", f"{before:,}", "blue"),
        _metric_card("优化后指令", f"{after:,}", "green"),
        _metric_card("静态节省", f"{saved:,} ({reduction:.1f}%)", "orange"),
        _metric_card("规则应用次数", f"{int(summary.get('changes', 0)):,}", "purple"),
        _metric_card(
            "未变化样例", f"{int(summary.get('unchanged_cases', 0)):,}", "red"
        ),
        "</div>",
    ]

    parts.extend(
        [
            "<section><h2>peephole 开关对比</h2>",
            f'<p class="comparison-note">{_escape(comparison_note)}</p>',
            "<table><tr><th>指标</th><th>对比</th><th>结果</th></tr>",
            _bar_row("优化前指令", before, max(before, after, 1), "compute", " 条"),
            _bar_row("优化后指令", after, max(before, after, 1), "branch", " 条"),
            _bar_row("静态节省", saved, max(before, 1), "memory", " 条"),
            "</table></section>",
            "<section><h2>各规则应用次数</h2>",
            "<table><tr><th>规则</th><th>应用次数</th><th>结果</th></tr>",
        ]
    )
    rule_matches = summary.get("rule_matches", {})
    maximum_matches = max(
        int(rule_matches.get(name, 0)) for name in peephole_bench.PR39_RULES
    )
    for index, name in enumerate(peephole_bench.PR39_RULES):
        color = ("compute", "memory", "branch", "upper", "shift", "neutral")[index % 6]
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
            '<table class="case-table"><tr><th>样例</th>'
            "<th>关闭</th><th>开启</th><th>节省</th><th>应用规则</th></tr>",
        ]
    )
    for item in cases:
        applied_rules = [
            name
            for name, count in item.get("rule_matches", {}).items()
            if int(count) > 0
        ]
        applied_rules_html = "<br>".join(_escape(name) for name in applied_rules) or "—"
        parts.append(
            "<tr>"
            f"<td>{_escape(item.get('case_id', ''))}<br>"
            f'<span class="muted">{_escape(item.get("description", ""))}</span></td>'
            f"<td>{int(item.get('peephole_off', {}).get('instructions', 0)):,}</td>"
            f"<td>{int(item.get('peephole_on', {}).get('instructions', 0)):,}</td>"
            f'<td class="delta">{int(item.get("reduced_instructions", 0)):,} '
            f'({float(item.get("reduction_percent", 0.0)):.1f}%)</td>'
            f'<td class="rule-list">{applied_rules_html}</td>'
            "</tr>"
        )
    parts.append("</table></section>")

    environment_rows = [
        f"<tr><td>Python</td><td>{_escape(metadata.get('python', ''))}</td></tr>",
    ]
    if "optimizer_elapsed_ms_median_sum" in summary:
        elapsed = float(summary["optimizer_elapsed_ms_median_sum"])
        environment_rows.append(
            f"<tr><td>优化器参考耗时（样例中位数之和）</td><td>{elapsed:.3f} ms</td></tr>"
        )
    if show_case_polarity:
        environment_rows.append(
            f"<tr><td>正向样例 / 负向样例</td><td>"
            f"{int(summary.get('positive_cases', 0))} / "
            f"{int(summary.get('negative_cases', 0))}</td></tr>"
        )
    parts.extend(
        [
            "<section><h2>环境与结论</h2>",
            "<table>",
            *environment_rows,
            f"<tr><td>结论</td><td>{'观察到静态指令减少' if saved > 0 else '未观察到静态指令减少'}</td></tr>",
            "</table></section>",
            f'<div class="footer">ScratchV 窥孔优化器报告 · {generated_at}</div>',
            "</body></html>",
        ]
    )
    return "\n".join(parts)


def generate_dsl_html_report(dsl_report: Any) -> str:
    """Render the 23-case DSL suite with the shared HTML renderer."""
    return generate_html_report(
        _dsl_report_payload(dsl_report),
        title="ScratchV 窥孔优化器 DSL Benchmark",
        comparison_note=(
            "每个 DSL 案例的关闭/开启结果均基于同一份输入汇编；"
            "关闭表示跳过汇编窥孔优化，开启表示执行窥孔优化规则。"
        ),
        show_case_polarity=False,
    )


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
    json_path = args.output_dir / "peephole_compare_html.json"
    html_path = args.output_dir / "peephole_compare.html"
    save_comparison(report, json_path, html_path)
    _print_summary(report, json_path, html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
