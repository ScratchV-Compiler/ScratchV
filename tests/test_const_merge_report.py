"""Integration checks for the one-case constant-merge CI report."""

import pytest

from benchmarks.run_const_merge_case import DEFAULT_CASE, _markdown, run_case


def test_const_merge_case_uses_feature_and_real_tinyfive():
    pytest.importorskip("tinyfive")

    report = run_case(DEFAULT_CASE)

    assert report["status"] == "passed"
    assert report["benchmark_type"] == "feature-case"
    assert report["feature"]["compiler_config_const_merge"] is True
    assert report["feature"]["pipeline_matches_public_pass"] is True
    assert report["feature"]["used"] is True
    assert report["optimization"]["merged_pairs"] == 1
    assert report["optimization"]["redundant_lui_removed"] == 1
    assert report["simulation"]["backend"] == "tinyfive"
    assert report["simulation"]["fallback"] is False
    assert report["simulation"]["output_equal"] is True
    assert report["simulation"]["before"]["registers"] == {
        "x5": 4098,
        "x6": 8195,
        "x7": 7,
    }
    assert (
        report["simulation"]["before"]["registers"]
        == report["simulation"]["after"]["registers"]
    )

    markdown = _markdown(report)
    assert markdown.count("<details>") == 2
    assert markdown.count("</details>") == 2
    assert "<summary>Assembly before (click to expand)</summary>" in markdown
    assert "<summary>Assembly after (click to expand)</summary>" in markdown
    assert report["assembly"]["before"].rstrip() in markdown
    assert report["assembly"]["after"].rstrip() in markdown
