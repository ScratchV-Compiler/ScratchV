# W2 Linux 独立复现与验收

正式交付、CI 和第二人复现统一使用 **Ubuntu 24.04 x86_64 / Bash / Python 3.12**。完整环境准备见 [Linux 复现约定](../LINUX_REPRODUCTION.md)。Linux 为宿主环境；QEMU 内的 RV64GC/LP64D 程序仍使用当前裸机 ABI。

W2 的范围是完整前端解析、小模型 IR 数值、基础后端和 Host 单步接口。完整 28 层模型的 IR 数值前向属于 W3；完整模型 QEMU 与生成循环属于后续阶段。W1 的人工接口决议继续见 [W1 记录](../W1/README.md) 和 [Issue #92](https://github.com/ScratchV-Compiler/ScratchV/issues/92)。

## 独立复现版本

[PR #94](https://github.com/ScratchV-Compiler/ScratchV/pull/94) 的既有 Linux 记录对应 `2d07ccc42936dcfde19d552bca02a324cd33821c`。后续文档或源码更新不自动继承该提交的执行结果。开始前记录约定 SHA、`git rev-parse HEAD` 和 `git status --short`；有未提交增量则保留完整 diff 和新增文件，不能只报告 HEAD。

发布前个人工作树的详细测量按原始身份保留于 [固定提交归档](https://github.com/ScratchV-Compiler/ScratchV/blob/2d07ccc42936dcfde19d552bca02a324cd33821c/docs/llm-deploy-v1.0/W2/README.md)。当前页面只提供 Linux 交付入口与可关联到具体提交的 Linux CI 记录，不把历史个人测量改名为 Linux 结果。

## 七项统一验收

先完成环境准备，激活 `.venv-linux`，并设置 `SCRATCHV_PYTHON`、`SCRATCHV_CC`、`SCRATCHV_QEMU`。模型及官方 Tokenizer 目录必须包含固定 manifest 对应的真实资产。在仓库根目录执行：

```bash
set -euo pipefail
source .venv-linux/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONIOENCODING=utf-8
export SCRATCHV_PYTHON="$(pwd)/.venv-linux/bin/python"
export SCRATCHV_CC="$(python -c 'from pathlib import Path; import ziglang; print(Path(ziglang.__file__).parent / "zig")')"
export SCRATCHV_QEMU="$(command -v qemu-system-riscv64)"
W2_TOKENIZER=/absolute/path/to/fixed-qwen3-tokenizer
W2_MODEL=/absolute/path/to/fixed-qwen3-onnx

git rev-parse HEAD
git status --short
"$SCRATCHV_PYTHON" -B -X utf8 scripts/run_w2_acceptance.py \
  --python "$SCRATCHV_PYTHON" \
  --output-dir output/w2-acceptance-linux-01 \
  --tokenizer-dir "$W2_TOKENIZER" \
  --model-dir "$W2_MODEL" \
  --cc "$SCRATCHV_CC" \
  --qemu "$SCRATCHV_QEMU"
```

把两项资产路径改成 Linux 实际目录。默认使用 `verify` 离线校验；无资产时可显式增加 `--tokenizer-mode download` 与 `--model-mode download`。后者下载约 1.24 GB 的固定完整发布包，解包及解析需要额外磁盘与数 GiB 空闲内存。入口不安装依赖。`--output-dir` 必须是本次全新的路径，连已存在的空目录也会被拒绝。

| 门禁 | 实际执行与判断 |
|---|---|
| `frontend` | RMSNorm、RoPE、SwiGLU、GQA 四种基础算子分解图；独立公式与反例 |
| `backend` | 七类基础算子的 16 个小图 × none/all，共 32 次 QEMU；误差 `<1e-5` |
| `runtime` | 官方 Tokenizer 的 20 组语料、480 次对照，以及输入、掩码、greedy、停止边界 |
| `runtime-model` | 完整词表两层随机小配置；2 文本、8 次 Torch/ORT/IR 调用及单步 greedy/解码 |
| `small-ir` | 官方 Qwen3 类两层小配置，7 输入、29 检查点、none/basic/all，误差 `<1e-5` |
| `small-qemu` | 同一本轮普通/诊断图 × none/all × 7 输入，共 28 次真实 QEMU，误差 `<1e-5` |
| `full-parse` | 固定完整 28 层 ONNX 的结构、逐节点转换、真实权重绑定与 IR verifier 审计 |

所有数值门禁同时检查 shape、dtype、有限值；上述阈值使用最大绝对误差、`rtol=0`，包括 padding 输出。Attention 使用基础算子组合，不要求新增融合 opcode。Tokenizer 的“往返 100%”指与官方 NFC/特殊 Token 规则一致，不是任意字节可逆。

仅七项全部执行并通过才会退出 **0 / PASS**；`--gates` 选择部分门禁即使成功，也只会退出 **2 / PARTIAL**。失败退出 1，中断按实际退出状态记录。`runtime-model` 自动带上 `runtime`，`small-qemu` 自动带上 `small-ir`。缺失、跳过、非有限误差、复用旧输出或执行中源码变化均不能冒充完整通过。

## 报告与独立复现记录

总报告位于本次 `--output-dir` 下的 `report.json`、`report.md`、`report.html`；子目录保存各门禁原始报告、命令、日志、数组和模型/ELF 身份。源码指纹核对的是实际执行文件，不能把别处安装的模块误当当前 checkout。

- 保存 checkout SHA、dirty/diff、Linux 发行版、CPU、Python/NumPy/ORT/Torch、Zig/QEMU 版本。
- 保存每项实际退出码、PASS/FAIL/PARTIAL/未执行、原始误差、首次失败阶段及报告位置。
- QEMU 时长是进程墙钟，包含启动、guest 执行、UART 和退出；不是目标硬件性能或纯 forward 时间。
- 重新生成自己的模型、参考与输出，不能下载第一人的数组后称为独立执行。
- 独立复现和团队接口/出口确认分开记录，作者再次执行或 CI 自动运行不替代第二人。

## 已有 Linux CI 证据

以下全部归属历史 SHA `2d07ccc42936dcfde19d552bca02a324cd33821c`，不作为这次文档更新的新执行结果：

| 工作流 | 已记录结果 |
|---|---|
| [通用 CI](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/37277915411) | 成功；2927 passed / 7 skipped。Linux `test_w2_acceptance.py` 55/55、无跳过 |
| [LLM Deploy](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/37277915420) | 成功；32/32 基础 QEMU、28/28 两层 QEMU、3/3 显式产物、20/20 超时清理；两层诊断最大误差 `1.7285346984863281e-6` |
| [Topic 06](https://github.com/ScratchV-Compiler/ScratchV/actions/runs/37277915428) | 成功；范围以该 run 的实际步骤与 artifact 为准 |

PR 中按条件未运行的完整 ONNX ORT 重型任务仍是未执行，不因上述绿色记录而算作完整 ORT 重验。新的提交需要新的 Checks；已有 Linux CI 不替代本次七项统一入口的第二人独立复现。

## 分项入口与后续边界

- [完整前端解析](../../../probes/w2_qwen3_parse/README.md)：无须 Torch 或 QEMU，可单独准备固定解析环境。报告含 `nodes.json`、`bindings.json`、`ir.txt`。
- [前端模式与基础后端](../../../probes/w2_backend_ops/README.md)：NumPy 独立公式、ORT、IR 和 QEMU 四路检查。
- [Tokenizer 与输入](../../../probes/w2_runtime/README.md) 和 [单步模型接口](../../../probes/w2_runtime_model/README.md)：模型词表 151936 与实际可解码 ID 集合分开处理，不取模重映射。
- [两层数值探测](../../../probes/w2_qwen3_small/README.md)：普通图、诊断图、产物绑定、仿真计时与精度边界。
- [发布范围与修复](W2-分支验收与发布说明.md)：共享后端、日志和 Linux 取消清理的交付归属。

完整解析通过不能替代完整 IR 数值、完整权重独立装载、完整模型 QEMU 前向或真实文本生成。阶段安排见 [开发计划](../开发计划.md)。
