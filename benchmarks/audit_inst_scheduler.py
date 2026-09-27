"""Compatibility entry point for the CNN scheduling audit.

The maintained benchmark, LLVM evaluator and report schema are defined in
benchmarks.bench_cnn_schedule. Synthetic stress tests remain in bench_inst_scheduler.
"""

from benchmarks.bench_cnn_schedule import main


if __name__ == "__main__":
    raise SystemExit(main())
