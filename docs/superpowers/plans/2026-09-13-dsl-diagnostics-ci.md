# DSL Diagnostics CI Implementation Plan

**Goal:** Add independently visible test and benchmark checks for Topic 9 to PR #38.

**Architecture:** A dedicated GitHub Actions workflow runs the existing DSL tests and a new diagnostics benchmark on hosted Linux runners. The benchmark uses the same current-branch case corpus and measurement script in separate Python processes for the baseline and current checkouts. Functional failures block CI; the documented 1.5x parsing target is reported separately, with optional enforcement.

**Tech Stack:** Python standard library, pytest, GitHub Actions, JSON/Markdown/standalone HTML.

The implementation follows the design already discussed with the user. It does not rebase or alter compiler behavior, the constant-merge feature, or the existing model benchmark workflow.

- [x] Add subprocess regression tests for real diagnostic reports, empty/malformed input corpora, baseline isolation, and report failure propagation.
- [x] Run the new tests and confirm the missing benchmark fails them.
- [x] Implement `benchmarks/bench_dsl_diagnostics.py`: fixed diagnostic expectations, current/base parser measurements, IR fingerprints, environment/revision metadata, and three report formats. Defaults: 5 warmups, 10 groups, 100 corpus iterations per group.
- [x] Add `.github/workflows/dsl-diagnostics.yml` with `test` and `benchmark` jobs, exact event/base checkout, Python 3.12, explicit pytest installation, always-uploaded reports, and benchmark Job Summary.
- [x] Document local commands, baseline selection, timing boundaries, and the distinction between functional checks and the performance target.
- [x] Run targeted tests, full repository tests, the benchmark against the pre-diagnostics revision, YAML validation, and `git diff --check`. Attempt the mandated L2 command and record a missing harness honestly.
- [x] Self-review the implementation and record a reusable memory entry.

Finalization: stage only task files, commit in English, and push the current PR branch. Validation evidence and known pre-existing failures are recorded in `docs/topics/09-DSL诊断-CI与Benchmark.md`.
