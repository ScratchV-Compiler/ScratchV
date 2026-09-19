# Topic 01 CI Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit Topic 01 tests and a semantic control-flow benchmark to the existing CI pipeline, with detailed logs collapsed by default.

**Architecture:** A standalone standard-library report CLI compiles the three Topic 01 examples through the real parser, LLVM backend, JIT, instruction selector, register allocator, and assembly emitter. It writes JSON, Markdown, and HTML; Markdown and HTML keep the summary visible and wrap each case's source/LLVM/RISC-V logs in closed `<details>` elements. Existing `test` and `benchmark` jobs call the test and report CLI and reuse current artifacts and Job Summary.

**Tech Stack:** Python 3.12, pytest, llvmlite, GitHub Actions, JSON/Markdown/HTML.

---

### Task 1: Lock the report contract with tests

**Files:**
- Create: `tests/test_dsl_control_flow_benchmark.py`

- [ ] Run the report CLI through a subprocess with small repeat counts.
- [ ] Assert JSON records three passing examples and their expected results.
- [ ] Assert Markdown and HTML contain one closed `<details>` section per example, no `<details open>`, and visible summary tables.
- [ ] Run `python -m pytest tests/test_dsl_control_flow_benchmark.py -q` and confirm it fails because the module does not exist.

### Task 2: Implement the Topic 01 semantic benchmark

**Files:**
- Create: `benchmarks/bench_dsl_control_flow.py`

- [ ] Parse and compile `if_else.dsl`, `while_sum.dsl`, and `nested_loop.dsl` through LLVM and RISC-V paths.
- [ ] Verify LLVM IR, execute `main` with MCJIT, compare the result with 7, 15, and 6, and record compile timing and structural metrics.
- [ ] Write JSON, Markdown, and HTML reports; keep case logs inside closed `<details>` sections.
- [ ] Return a nonzero status if any semantic or backend check fails.
- [ ] Run the new benchmark tests and confirm they pass.

### Task 3: Integrate the existing CI jobs

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] Add `Topic 01 DSL frontend regressions` to the existing `test` job.
- [ ] Add `Topic 01 DSL frontend benchmark` to the existing `benchmark` job.
- [ ] Write the Markdown report into the existing Job Summary and leave the report files under `benchmark_reports/` for the existing artifact upload.
- [ ] Do not add a workflow, job, or Pages deployment path.

### Task 4: Document and verify

**Files:**
- Modify: `docs/topics/01-DSL前端增强器-开发文档.md`
- Modify: `memory/memory.md`

- [ ] Replace the preimplementation status with the delivered files, commands, report contract, and CI placement.
- [ ] Run the Topic 01 tests, benchmark tests, report CLI, `git diff --check`, and the relevant full regression suite.
- [ ] Self-review the diff, stage only task files, commit, push, and confirm both GitHub CI checks.
