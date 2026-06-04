# ScratchV — LLVM 后端 CNN 编译器

## 项目概述

将 ONNX CNN 模型编译为 RISC-V 汇编代码，支持两条路径:
- **ScratchV 原生**: RV32IM Q16.16 定点 (轻量、无依赖)
- **LLVM**: RV64FD float32 (通过 llvmlite, 性能更好)

## 关键命令

```bash
# 安装
pip install -e ".[all]"

# 测试
make test                    # pytest tests/
make bench                   # ONNX + DSL 基准测试

# CNN 编译 + 分析
make bench-cnn               # ScratchV 编译 cnn.onnx → RISC-V + 估算
make bench-ci                # LLVM+TinyFive 全量对比 → benchmark_reports/dashboard.html

# 单文件工具
python scratchv/standalone/onnx_to_riscv_standalone.py models/graph/cnn.onnx -o output.bin --estimate --report
python scratchv/standalone/llvm_cache_compare.py        # LLVM vs ScratchV 缓存对比
python scratchv/standalone/tinyfive_compare.py           # TinyFive 静态分析
python scratchv/ci/dashboard.py --run -o dashboard.html  # 生成对比仪表盘
```

## 项目结构

```
scratchv/
  backend/          LLVM IR 代码生成
  ci/               CI 基准测试编排器 + 仪表盘生成器
  standalone/       ONNX→RISC-V编译器、仿真器、分析工具
  simulator/        TinyFive 适配器
  runtime/          RISC-V 运行时库
  verification/     ONNX Runtime 验证

benchmarks/         DSL 用例 + ONNX 管线测试
models/graph/       ONNX 模型文件
output/             分析报告 (spike_bench, llvm_vs_scratchv, tinyfive)
benchmark_reports/  CI 产物目录
```

## 依赖

- **必需**: onnx, numpy, protobuf
- **可选**: tinyfive (RV32IM 仿真), llvmlite (LLVM 编译), onnxruntime (验证)
- CI benchmark 工具是纯分析 (不需要 GPU)

## 代码风格

- Python 3.12+, type hints
- argparse CLI 约定: `--json` (bool, stdout), `--json-output` (path), `--markdown` (path)
- 报告工具: 生成 HTML/JSON/MD 三种格式
- 零外部可视化依赖 (HTML 纯内联 CSS, 不用 matplotlib/plotly)
