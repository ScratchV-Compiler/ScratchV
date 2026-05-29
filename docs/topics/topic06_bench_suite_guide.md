# Compiler Benchmark Suite - User Guide

## Overview

The `BenchmarkRunner` in `benchmarks/bench_runner.py` automates regression testing and performance benchmarking of the ScratchV compiler. It runs DSL test cases through the compiler pipeline, compares outputs against expectations, and generates comprehensive reports.

## Test Case Format

Each test case is a group of files in `benchmarks/cases/`:

```
benchmarks/cases/
   001_simple_add.dsl       # DSL source input
   001_simple_add.expected  # Expected output text
   001_simple_add.desc     # Short description (one line)
```

- `.dsl` is required
- `.expected` is optional (test passes if the DSL compiles without error)
- `.desc` is optional (provides a human-readable name in reports)

## Usage

### Command Line

```bash
# Run all cases
python benchmarks/bench_runner.py

# Run with specific directory
python benchmarks/bench_runner.py benchmarks/cases

# Generate reports
python benchmarks/bench_runner.py --output-json report.json
python benchmarks/bench_runner.py --output-html report.html
python benchmarks/bench_runner.py --output-md report.md

# Benchmark mode (multiple repetitions for averaging)
python benchmarks/bench_runner.py --repeat 3

# Quiet mode
python benchmarks/bench_runner.py --quiet
```

### Python API

```python
from benchmarks.bench_runner import BenchmarkRunner

# Create runner
runner = BenchmarkRunner(
    test_dir="benchmarks/cases",
    timeout=30.0,
    verbose=True,
)

# Discover test cases
cases = runner.discover_cases()
print(f"Found {len(cases)} test cases")

# Run all
report = runner.run_all()
report.print_summary()

# Generate reports
report.save_json("results.json")
report.save_html("results.html")
report.save_markdown("results.md")

# Benchmark mode (repeat for better timing)
report = runner.run_benchmark(repeat=3)
```

### Run a Single Case

```python
case = {
    "name": "001_simple_add",
    "dsl_path": "benchmarks/cases/001_simple_add.dsl",
    "expected_path": "benchmarks/cases/001_simple_add.expected",
    "desc_path": "benchmarks/cases/001_simple_add.desc",
}
result = runner.run_case(case)
print(f"Passed: {result.passed}, Time: {result.total_time_s:.4f}s")
```

## Report Formats

### Terminal Summary

```
================================================================================
BENCHMARK REPORT
================================================================================
Total cases: 23 | Passed: 20 | Failed: 3
Pass rate: 87.0% | Total time: 2.345s
--------------------------------------------------------------------------------
Name                    Status   Parse(s)   Compile(s)   Sim(s)    Inst     Description
--------------------------------------------------------------------------------
001_simple_add          PASS     0.0001     0.0002        0.0010    2        Basic addition
002_simple_mul          PASS     0.0001     0.0001        0.0010    2        Element-wise multiply
...
016_if_simple           FAIL     0.0005     0.0003        0.0000    0        If-else branch
    ERROR: DSLParseError: Cannot parse line: if (a > b):
...
```

### JSON Report

Machine-readable format suitable for CI dashboards and trend tracking:

```json
{
  "timestamp": "2025-01-15T10:30:00",
  "total_time_s": 2.345,
  "pass_count": 20,
  "fail_count": 3,
  "pass_rate": 87.0,
  "results": [...]
}
```

### Markdown/HTML Report

Human-readable format with tables and statistics, suitable for documentation and code review.

## Metrics Collected

For each test case:
- **Status**: PASS / FAIL
- **Parse time**: Time spent parsing DSL to IR
- **Compile time**: Total compile time (parse + codegen)
- **Simulation time**: Time spent in the interpreter
- **Instruction count**: Number of IR instructions generated
- **Error**: Error message if the test failed

## Test Case Coverage

The built-in test suite covers:

| Category     | Cases | Examples                      |
|--------------|-------|-------------------------------|
| Arithmetic   | 5     | add, mul, sub, div, chained   |
| NN Ops       | 6     | relu, gelu, softmax, matmul, dot, maxpool |
| Control Flow | 7     | for-loop, if/else, while, nested |
| Complex      | 3     | NN pipeline, large chain      |
| Constants    | 1     | constant propagation          |
| **Total**    | **23** |                              |

## Adding New Test Cases

1. Create `{name}.dsl` in `benchmarks/cases/`
2. Optionally create `{name}.expected` with expected output
3. Optionally create `{name}.desc` with a description
4. Run the benchmark suite to verify

## Regression Testing

For CI integration, save a baseline and compare:

```bash
# Generate baseline
python benchmarks/bench_runner.py --output-json baseline.json

# Later, compare against baseline
python benchmarks/bench_runner.py --output-json current.json
python -c "
import json
baseline = json.load(open('baseline.json'))
current = json.load(open('current.json'))
if current['pass_rate'] < baseline['pass_rate']:
    print('REGRESSION DETECTED')
    exit(1)
"
```

## See Also

- `benchmarks/run_benchmark.py` - ONNX model benchmark runner
- `benchmarks/generate_models.py` - ONNX model generator
- `scratchv/verification/verifier.py` - Reference interpreter used for output comparison
