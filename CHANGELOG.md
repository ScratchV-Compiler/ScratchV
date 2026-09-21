# Changelog

## Unreleased — Topic 18 review iteration (2026-09-22)

- Keep instruction scheduling disabled by default at the assembly-text V1 entry point. Model cycles and llvm-mca estimates are not hardware speedups or dynamic-instruction-count improvements.
- Use LLVM MCA / SiFive E76 directly for scheduling acceptance and reports. Record the actual LLVM version; the checked-in benchmark was measured with 18.1.3. Remove the custom timing model, parameter table and extractor. Retain raw LLVM timelines; report critical paths as unavailable for this in-order target.
- Restore linear-allocator branch operands and labels; classify branch/store operands as reads and emit canonical integer zero-register sources.
- Report coverage, unsupported opcodes and rejected candidates; restore section-stack state and distinguish quoted text from layout directives.
- Validate the tracked CNN through standalone scheduling A/B, LLVM timing and full QEMU execution. Run scheduling and standalone regressions in the existing test CI job; retain tool-version checks and performance reports in the benchmark job.
- Integrate the current register allocator: preserve symbolic targets during renaming, keep DAG virtual names separate from physical registers, and lower nested-loop bounds before allocation.
- Preserve the unified CFG control targets and register-map statistics. Keep CI on `ubuntu-latest` and validate LLVM analysis capabilities without requiring a specific release.
- API migration: `SchedInst` is immutable and recomputes def/use; `build_dag` rejects multiple regions; `machine_instrs_from_scheduled` rejects lossy conversions. Use `schedule_assembly` for complete assembly, and read the compiler report from `stats["schedule"]["report"]` instead of `warnings`.
- Structured post-RA/pre-emission scheduling and Fast/BURR strategy selection remain future work; see `docs/topic-18/README.md`.

## [0.3.0] — 2026-05-18

### Added
- `scratchv_dag/`: standalone LLVM-style SelectionDAG infrastructure package
  - `sdnode.py`: SDNode, SDValue, MVT, SelectionDAG container
  - `selection_dag.py`: DAGBuilder, DAGCombiner, DAGScheduler pipeline
  - `cache.py`: 4 MB L1 cache simulator (set-associative, LRU, write-back)
  - `allocator.py`: Buddy-system memory allocator with cache-line alignment and scratchpad
- `docs/developer_guide.md`: guide for extending ScratchV with new ops and passes
- `Makefile`: standard dev targets (install, test, clean, lint, docs)
- `CHANGELOG.md`, `CONTRIBUTING.md`: project metadata files
- `pyproject.toml`: classifiers, readme field, license field

### Changed
- Consolidated `scratchv/codegen/` and `scratchv/memory/` into re-export shims over `scratchv_dag/`
- Python requirement lowered to 3.8 with full compatibility fixes
- `pyproject.toml` version bumped to 0.3.0
- `.gitignore` extended for `.ll` files and `.claude/`

## [0.2.0] — 2026-05-15

### Added
- LLVM IR backend (`llvm_codegen.py`)
- Advanced optimizations: peephole, muladd fusion, LICM
- Verification framework: ONNX Runtime comparison, numpy reference, DSL interpreter
- TinyFive adapter for assembly verification and profiling
- CLI options: `--backend`, `--optimize`, `--verify`, `--reg-alloc`
- Documentation: optimization guide, verification guide

### Changed
- Instruction selector supports all major ops (add, sub, mul, div, neg, exp,
  relu, gelu, softmax, maxpool, matmul, dot)
- Register allocator: greedy mode (LRU-based) added alongside naive

## [0.1.0] — 2026-05-01

### Added
- Initial IR: types (Value, Instruction, BasicBlock, Function, Program)
- ONNX parser: Add, Mul, Sub, Div, MatMul, ReLU, GELU, Softmax, MaxPool
- DSL parser for fast iteration without ONNX dependency
- IR builder with chainable API
- IR printer for debugging
- Instruction selector: IR → RISC-V pseudo-instructions
- Register allocator: naive (spill-all) mode
- Assembly emitter: GAS-syntax output
- Constant folding and dead code elimination passes
- CLI entry point with `-o`, `--dump-ir` flags
