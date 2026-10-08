# ONNX 算子补齐 benchmark：Linux 复跑入口

正式交付与复现使用 **Ubuntu 24.04 x86_64 / Bash / Python 3.12**。本页是入口，**不是新测量报告**。环境、Git 历史和 LFS 准备见 [算子补齐与单测报告](算子补齐与单测报告.txt)，通用交付约定见 [Linux 复现约定](../LINUX_REPRODUCTION.md)。

在仓库根目录激活 `.venv-operators` 后执行：

```bash
set -euo pipefail
source .venv-operators/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -X utf8 -B -m benchmarks.bench_onnx_operators \
  --baseline-ref 20b105e80ba3dbe13cb01a3d4ca18c32b6c31ed6 \
  --output-dir benchmark_reports/onnx_operators-linux-01 \
  --warmup 1 --repeats 5
```

输出目录每次换新名字，生成本次 JSON、Markdown、HTML 及原始样本。仅前后均 PASS 的用例比较耗时；缺失、不支持或失败不等于零耗时。结果用于支持/正确性对比，IR 解释器计时不等于 QEMU 或目标硬件性能。

原始个人测量保持原平台身份和数据，分别保存于固定提交：

- [原始 MD](https://github.com/ScratchV-Compiler/ScratchV/blob/2d07ccc42936dcfde19d552bca02a324cd33821c/docs/llm-deploy-v1.0/W1/算子补齐与单测-benchmark.md)
- [原始 JSON](https://github.com/ScratchV-Compiler/ScratchV/blob/2d07ccc42936dcfde19d552bca02a324cd33821c/docs/llm-deploy-v1.0/W1/算子补齐与单测-benchmark.json)
- [原始 HTML](https://github.com/ScratchV-Compiler/ScratchV/blob/2d07ccc42936dcfde19d552bca02a324cd33821c/docs/llm-deploy-v1.0/W1/算子补齐与单测-benchmark.html)

当前同名 JSON 的 `document_kind=historical_benchmark_index`，只保存归档链接，不含 PASS、环境测量或计时字段，不能作为 benchmark 结果输入。新 Linux 测量必须从 CLI 输出目录读取。
