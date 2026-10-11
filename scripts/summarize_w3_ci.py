"""Publish bounded W3 numerical reports to GitHub Actions without changing gates."""
from __future__ import annotations

import argparse
from html import escape
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.numeric_summary import compact_metric_table
from probes.w3_summary import evidence_index_views, numerical_summary

GATES = (
    ("medium", "w3-medium-host", "6-layer random Qwen3 / host IR"),
    ("subgraphs", "w3_qwen3_subgraphs", "Pretrained layer-0 subgraphs / host IR"),
    ("attention", "unit:attention-backend", "Small Attention / RISC-V QEMU"),
    ("full-preflight", "preflight:w3-full-assets", "Full assets / static checks only"),
    ("layer-diff", "w3_layer_diff", "Saved medium-model checkpoints only"),
)


def cell(value, limit=180):
    text = " ".join(str(value).split())
    if len(text) > limit:
        text = text[:limit] + "…"
    return escape(text).replace("|", "&#124;")


def load_report(path, gate):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "MISSING REPORT"
    except (OSError, ValueError):
        return None, "INVALID REPORT"
    if not isinstance(report, dict) or report.get("gate") != gate:
        return None, "INVALID REPORT (gate identity)"
    # Fail closed on malformed structures before calling the shared renderer.
    for key in ("cases", "gates", "invariants"):
        rows = report.get(key, [])
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            return None, "INVALID REPORT (rows)"
    return report, ""


def status(report):
    if report.get("status") == "PARTIAL":
        return "PARTIAL (not acceptance)"
    if report.get("status") == "PASS" and report.get("passed") is True:
        if report.get("gate") == "numeric:ir-full-qwen3" and report.get("coverage_complete") is not True:
            return "INCOMPLETE (not acceptance)"
        return "PASS"
    return "FAIL" if report.get("passed") is False else "UNCONFIRMED"


def seconds(report):
    value = report.get("elapsed_seconds", report.get("seconds"))
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return f"{value:.3f} s"
    return "not recorded"


def coverage(report):
    gate = report.get("gate")
    cases = report.get("cases", [])
    if gate == "w3_layer_diff":
        values = report.get("coverage", {})
        return f"{values.get('compared_checkpoints', '?')}/{values.get('total_checkpoints', '?')} checkpoints"
    if gate == "preflight:w3-full-assets":
        return "no model execution"
    if gate == "unit:attention-backend":
        count = sum(len(row.get("executions", [])) for row in cases)
        return f"{len(cases)} cases / {count} target executions recorded"
    required = report.get("required_cases")
    expected = len(required) if isinstance(required, list) else {"w3-medium-host": 7, "w3_qwen3_subgraphs": 14}.get(gate, "?")
    return f"{len(cases)}/{expected} cases recorded"


def details(report, title):
    """Reuse bounded semantic selectors, never dump arbitrary report dictionaries."""
    description, comparisons, notes = numerical_summary(report)
    index, _ = evidence_index_views(report)
    text = f"<details><summary>{cell(title)}</summary>\n\n{cell(description, 700)}\n\n"
    if comparisons:
        text += compact_metric_table(comparisons) + "\n\n"
    if report.get("error"):
        text += "Issue: " + cell(report["error"], 400) + "\n\n"
    text += "\n\n".join(cell(note, 500) for note in notes) + "\n\n" + index + "</details>\n\n"
    return text


def render(kind, directory, outcome):
    title = "W3 preparation" if kind == "preparation" else "W3 complete IR numeric"
    artifact = "w3-preparation-reports" if kind == "preparation" else "w3-full-numeric-reports"
    lines = [f"## {title}\n\n"]
    if outcome in ("skipped", "", "cancelled"):
        label = "CANCELLED" if outcome == "cancelled" else "NOT RUN"
        lines.append(f"**{label}** — probe step did not complete; no numerical acceptance claimed.\n\n")
        lines.append("Existing files, if any, are not used as results for this step.\n")
        return "".join(lines)
    identity = "preparation:w3" if kind == "preparation" else "numeric:ir-full-qwen3"
    report, problem = load_report(directory / "report.json", identity)
    if report is None:
        lines.append(f"**{problem}** — step outcome: **{cell(outcome)}**. No acceptance claimed; inspect step logs.\n")
        return "".join(lines)
    conclusion = status(report)
    if outcome != "success" and conclusion == "PASS":
        conclusion = "FAIL (step did not succeed)"
    lines.append(f"**{conclusion}** · step: {cell(outcome)} · pipeline: {seconds(report)}\n\n")
    comparisons, folded = [], []
    try:
        if kind == "preparation":
            lines.append("Preparation only; full 28-layer IR numeric and full-model QEMU are **NOT RUN by this job**.\n\n")
            lines.append("| Check / scope | Result | Recorded coverage | Pipeline time |\n| --- | --- | --- | --- |\n")
            for name, identity, scope in GATES:
                rows = [row for row in report.get("gates", []) if row.get("name") == name]
                row = rows[0] if len(rows) == 1 else None
                child, problem = load_report(directory / name / "report.json", identity) if row else (None, "NOT RUN" if not rows else "INVALID REPORT (duplicate gate)")
                child_status = problem or status(child)
                if row and status(row) != "PASS" and child_status == "PASS":
                    child_status = "FAIL (parent validation)"
                if child_status != "PASS" and conclusion == "PASS":
                    lines[1] = f"**INCOMPLETE REPORTS (not acceptance)** · step: {cell(outcome)} · pipeline: {seconds(report)}\n\n"
                lines.append("| " + " | ".join(cell(x) for x in (scope, child_status, coverage(child) if child else "not recorded", seconds(row) if row else "not recorded")) + " |\n")
                if child:
                    _, numeric, _ = numerical_summary(child)
                    selected = numeric[1:] if name in ("medium", "subgraphs") else numeric[1:2] if name == "attention" else []
                    records = [item for _, items in selected for item in items]
                    if selected:
                        comparisons.append((scope + " vs ORT", records))
                    folded.append(details(child, scope + " — details"))
            folded.insert(0, details(report, "Preparation execution / failures"))
        else:
            lines.append("28-layer pretrained Qwen3, FP32 / L=256 / no KV cache; IR vs ORT. **No full-model QEMU result in this job.**\n\n")
            lines.append(f"Coverage: **{cell(coverage(report))}** · FP32 mode: **{cell(report.get('fp32_mode', 'not recorded'))}** · maximum absolute error threshold: **{cell(report.get('atol', 'not recorded'))}** (strict &lt;, rtol=0).\n\n")
            display = dict(report)
            recorded = {row.get("name") for row in report.get("cases", [])}
            display["cases"] = list(report.get("cases", [])) + [
                {"name": name, "status": "NOT RUN"} for name in report.get("required_cases", [])
                if name not in recorded]
            _, comparisons, _ = numerical_summary(display)
            folded.append(details(display, "Cases, diagnostics, invariants and failures"))
        if comparisons:
            lines.append("\n" + compact_metric_table(comparisons) + "\n\n")
            lines.append("Metrics summarize final outputs only (worst MaxAbs / relative L2, lowest cosine across recorded comparisons); checkpoints still affect the original gate. Missing metrics remain explicit.\n\n")
        if report.get("error"):
            lines.append("Issue: " + cell(report["error"], 300) + "\n\n")
        lines.extend(folded)
    except (TypeError, KeyError, ValueError, AttributeError):
        # Unexpected report schemas must not make an apparently passing page.
        return f"## {title}\n\n**INVALID REPORT** — could not render numerical evidence; step: {cell(outcome)}. No acceptance claimed.\n"
    lines.append(f"Artifacts: **{artifact}** (`report.json`, `report.md`, `report.html`). Raw arrays are separate where retained. This summary does not replace independent reproduction or team acceptance.\n")
    return "".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("preparation", "full"), required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--outcome", default=os.environ.get("PROBE_OUTCOME", ""))
    parser.add_argument("--summary", type=Path, default=os.environ.get("GITHUB_STEP_SUMMARY"))
    args = parser.parse_args(argv)
    text = render(args.kind, args.report_dir, args.outcome)
    if args.summary:
        with args.summary.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    else:
        print(text, end="")
    return 0  # Reporting never substitutes for or changes the probe's exit code.


if __name__ == "__main__":
    raise SystemExit(main())
