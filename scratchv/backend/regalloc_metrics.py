"""Shared, explicitly named metrics for linear-scan register allocation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def peak_live_intervals(intervals: Iterable[Any]) -> int:
    """Return the exact maximum number of overlapping half-open intervals."""

    interval_list = list(intervals)
    if not interval_list:
        return 0
    starts = {interval.start for interval in interval_list}
    return max(
        sum(
            interval.start <= position < interval.end
            for interval in interval_list
        )
        for position in starts
    )


def count_spill_reload_sites(assembly: str) -> tuple[int, int]:
    """Count allocator-inserted static spill stores and reload loads.

    Counts are based on allocator-only tags, so a user comment containing
    words such as ``spill`` or ``reload`` cannot alter benchmark results.
    Dynamic execution counts are intentionally a separate metric.
    """

    spill_stores = 0
    reload_loads = 0
    for line in assembly.splitlines():
        content = line.strip()
        if content.startswith("sw ") and "[regalloc:spill]" in content:
            spill_stores += 1
        if content.startswith("lw ") and "[regalloc:reload]" in content:
            reload_loads += 1
    return spill_stores, reload_loads
