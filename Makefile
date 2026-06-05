# ScratchV developer makefile
.POSIX:

.PHONY: install test bench bench-cnn clean lint

# ── Installation ──────────────────────────────────────────────────────────

install:
	pip install -e .
	pip install -e ".[all]" 2>/dev/null || pip install -e .

# ── 课题功能测试 ──────────────────────────────────────────────────────────

test:
	python3 -m pytest tests/ -v --tb=short

# ── 模型性能基准 ──────────────────────────────────────────────────────────

bench:
	python3 -m pytest benchmarks/test_benchmark.py -v --tb=short
	python3 benchmarks/bench_runner.py benchmarks/cases \
		--output-json benchmark_reports/dsl_bench.json \
		--output-html benchmark_reports/dsl_bench.html

# ── CNN RISC-V 编译 + 估算 ────────────────────────────────────────────────

bench-cnn:
	python3 scratchv/standalone/onnx_to_riscv_standalone.py models/graph/cnn.onnx \
		-o /tmp/cnn_riscv.bin --estimate --report
	@echo "Reports: benchmark_reports/"

# ── CNN RISC-V 编译 + TinyFive 仿真验证 ───────────────────────────────────

bench-tinyfive:
	python3 scratchv/standalone/onnx_to_riscv_standalone.py models/graph/cnn.onnx \
		-o /tmp/cnn_riscv.bin --estimate --tinyfive --tinyfive-max-instr 200000
	@echo "TinyFive simulation complete."

# ── CI Benchmark (LLVM + TinyFive + 可视化仪表盘) ──────────────────────────

bench-ci:
	@mkdir -p benchmark_reports
	python3 scratchv/ci/ci_benchmark.py \
		--model-registry ci_models.json \
		--output-dir benchmark_reports/ \
		--html dashboard.html \
		--json-out ci_data.json \
		--md github_summary.md \
		--embed-json \
		--skip-cache
	@echo ""
	@echo "Dashboard: benchmark_reports/dashboard.html"
	@echo "JSON data: benchmark_reports/ci_data.json"
	@echo "GitHub summary: benchmark_reports/github_summary.md"

# ── Clean ─────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache scratchv.egg-info scratchv_dag.egg-info dist build
	rm -f output.s output.ll
	rm -rf benchmark_reports

# ── Lint (development only) ───────────────────────────────────────────────

lint:
	-python3 -m flake8 scratchv/ scratchv_dag/ tests/ 2>/dev/null || echo "pip install flake8"
	-python3 -m mypy scratchv/ scratchv_dag/ --ignore-missing-imports 2>/dev/null || echo "pip install mypy"
