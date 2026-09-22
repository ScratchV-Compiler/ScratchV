"""Generate Topic 06 reports from test-run metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import run_topic06_benchmarks as benchmark
except ModuleNotFoundError:
    from scripts import run_topic06_benchmarks as benchmark


def configure_output_dir(output_dir: Path) -> None:
    benchmark.REPORT_DIR = output_dir
    benchmark.REPORT_FILE = output_dir / "report.md"
    benchmark.JSON_REPORT_FILE = output_dir / "report.json"
    benchmark.HTML_REPORT_FILE = output_dir / "report.html"
    benchmark.CHART_FILE = output_dir / "course_report_instructions.png"
    benchmark.CASE_REPORT_DIR = output_dir / "cases"


def load_test_metadata(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "topic06_test_metadata":
        raise ValueError(f"unsupported Topic 06 metadata: {path}")
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported Topic 06 metadata schema: {payload.get('schema_version')}")
    if not isinstance(payload.get("results"), list):
        raise ValueError(f"metadata results must be a list: {path}")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate Topic 06 reports from benchmark metadata.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=benchmark.METADATA_FILE,
        help="Test metadata produced by run_topic06_benchmarks.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=benchmark.REPORT_DIR,
        help="Directory for generated report files.",
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Also generate the optional HTML report and PNG chart.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        payload = load_test_metadata(args.metadata)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Cannot generate Topic 06 report: {exc}")
        return 2

    configure_output_dir(args.output_dir)
    summary = payload.get("summary") or {}
    selection = payload.get("selection") or {}
    results = payload["results"]
    passed = summary.get("passed", sum(r.get("status") == "PASS" for r in results))
    failed = summary.get("failed", len(results) - passed)

    benchmark.write_report(
        results,
        passed,
        failed,
        regression_threshold_pct=payload.get(
            "regression_threshold_pct",
            benchmark.REGRESSION_THRESHOLD_PCT,
        ),
        selection_category=selection.get("category"),
        selection_filter=selection.get("filter"),
        full_report=args.full_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
