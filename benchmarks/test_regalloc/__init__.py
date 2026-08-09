# flake8: noqa
"""
Register Allocation Benchmarks for ScratchV.

This package contains three benchmark suites for the linear scan
register allocator (``LinearScanAllocator``):

1. **bench1_simple**   — Simple arithmetic (3-5 vregs, no spills)
2. **bench2_dense**    — Dense computation (20+ vregs, triggers spilling)
3. **bench3_cnn**      — CNN model integration (operations from ``models/graph/cnn.onnx``)

Each suite measures allocation time, spill count, peak register pressure,
and validates output correctness.

Usage::

    python -m benchmarks.test_regalloc.run_all
"""

from __future__ import annotations
